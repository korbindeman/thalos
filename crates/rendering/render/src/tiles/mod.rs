//! NTR-X1 — the probe-extracted tile terrain renderer (keystone:
//! ADR-20260723T142945Z, plan `ntr §6`): terrain as ordinary `Mesh` +
//! `StandardMaterial` entities streamed by a camera-driven, 2:1-balanced
//! cube-sphere quadtree, entirely on Bevy's standard render path.
//!
//! Ported from the standalone probe (`thalos-terrain-probe`, M0–M4) with the
//! Thalos adaptations: per-body radius, tiles placed on the big_space **root**
//! grid in f64 every frame from the body's pose (NOT parented to the body's
//! rotating grid — that is a decimetre of ground jitter, see
//! [`TileBodyOrigin`]), content from the body's canonical
//! `Arc<dyn SurfaceQuery>` (real albedo/roughness per sample), and the selection
//! eye supplied by the game from `ViewAnchor` (body-fixed camera position — the
//! one per-frame answer to "where is the view?").
//!
//! **This is the default ground renderer.** The legacy udlod path
//! (`crates/rendering/udlod`, reached through [`crate::ground`]) still streams
//! bodies the tile driver has not installed on, and can be forced for the whole
//! process with `THALOS_TILE_RENDERER=0` as an A/B baseline. Known limits,
//! tracked in the backlog: heights-only displacement of `sample_d`, and a
//! flatten handle read per tile *bake*, so pads installed over resident tiles
//! need those tiles dropped.
//!
//! **Authored ground outranks the distance rule.** Structure pads publish a
//! [`RefinementSite`] floor that selection honours at any camera distance, so
//! the mesh under a base always resolves the flat footprint stamped into it
//! (INC-20260725T184654Z). Everything else — ruggedness, the residency brake —
//! may only ever take detail *away* from the distance rule.
//!
//! Ground consumers (scatter, colliders, camera floor, HUD altitude) read the
//! meshed heights back through [`height_mirror`], the tile-path analogue of
//! udlod's GPU-atlas mirror — see that module for why sampling the analytic
//! surface directly is not equivalent.
//!
//! **Residency is budgeted in bytes** ([`TILE_MESH_BYTES`],
//! `THALOS_TILE_BUDGET_MB`). The budget brakes by *coarsening selection*, never
//! by evicting resident tiles — the despawn rule is hole-free, so eviction would
//! punch holes in the ground. Any new per-tile GPU resource must join
//! `TILE_MESH_BYTES` or the budget silently under-counts VRAM
//! (INC-20260725T012104Z-tile-residency-had-no-budget).
//!
//! The budget is **machine-wide, not per process**: it is divided by the number
//! of live Thalos renderers ([`vram_share`]), because the card is shared and two
//! instances each reading the full figure is how the second `DeviceLost`
//! happened after the first budget landed.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::math::{DQuat, DVec3};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::tasks::{Task, block_on, poll_once};
use big_space::prelude::*;
use thalos_terrain::SurfaceQuery;

pub mod cache;
pub mod height_mirror;
pub mod material;
pub mod vram_share;

pub use cache::{CachedTileProvider, SurfaceTileCache, TileNamespaceFn};
pub use height_mirror::{TileHeightMirror, TileHeightMirrorHandle};

/// Vertices per tile side (core grid, excluding halo).
pub const TILE_RES: usize = 65;
/// Halo rings included in every sampled grid (edge-exact normals).
pub const TILE_HALO: usize = 1;
/// Coarsest selected level (8×8 tiles per face).
pub const MIN_LEVEL: u8 = 3;
/// Split while camera distance < factor × tile arc — the *floor* of the
/// ruggedness-scaled rule below.
///
/// This factor alone sets the *geometric* resolution of any framing: a
/// resident tile at distance `d` has arc ≈ `d / SPLIT_FACTOR`, so its sample
/// spacing is `d / (SPLIT_FACTOR × (TILE_RES − 1))`. At 3.0 the NTR-X4
/// showcase framings (god view 22 km out) meshed the ground at ~100–200 m
/// while the diffusion source carries a 90 m band with ~90 m of RMS detail
/// below the 180 m scale — the mountains came out as soft domes because the
/// *mesh*, not the data, was the limit. 6.0 puts the near/mid field at
/// ~50–100 m so the source band is actually resolved; it costs ~4× the
/// resident tiles at a given distance.
const SPLIT_FACTOR: f64 = 6.0;
/// Split cap for the most rugged terrain (ntr §7's relief-aware rule).
///
/// A single distance factor buys resolution *everywhere* at 4× the tiles per
/// doubling, which is why 6.0 is as far as a uniform rule can go. But the
/// resolution a frame actually needs is not uniform: from altitude the ocean
/// and the plains are already converged at 6.0 while mountain ridges are
/// still the mesh's fault, not the data's — a distant massif reads as a soft
/// dome at 300 m/vertex however sharp the source is. So the cap is raised to
/// this value and [`tile_ruggedness_weight`] *removes* the extra everywhere
/// the terrain is smooth, which is the shape ntr §7 asks for (relief may only
/// take detail away from the distance rule, never add it) with the base rule
/// re-based from "everywhere" to "rugged". Cost lands only on the terrain the
/// player is looking at when they say "mountains".
const SPLIT_FACTOR_RUGGED: f64 = 18.0;
/// Ruggedness (tile relief ÷ tile arc — a mean-slope proxy) at and below which
/// a tile refines on the plain [`SPLIT_FACTOR`] rule, and at or above which it
/// gets the full [`SPLIT_FACTOR_RUGGED`] cap. Measured on Thalos's diffusion
/// terrain: ocean and coastal plain sit under 0.01, rolling upland ~0.02–0.04,
/// the NE massif's flanks 0.06+.
const RUGGED_LO: f32 = 0.012;
const RUGGED_HI: f32 = 0.055;
/// Sample spacing (m/vertex) below which the ruggedness boost stops buying
/// anything and switches off, leaving the plain [`SPLIT_FACTOR`] rule.
///
/// The boost exists to make *distant* relief legible. Refining past the
/// source's own finest band just re-meshes a signal that is no longer there:
/// Thalos's diffusion window carries a 90 m band, so ~45 m/vertex already
/// resolves it at Nyquist and the mesh stops being the limit. Measured
/// (`artifacts/visual/runs/altitude-detail/`): without this floor the 22 km
/// god view paid 2.7× the tiles (2,874 → 7,752) and 2.7× the cold stream for
/// a frame the eye cannot tell from the baseline, because that framing's near
/// field was already under 45 m. The floor gates only the boost — the base
/// rule and `max_level` are untouched, so near-field mesh, colliders and
/// scatter keep the resolution they have.
const RUGGED_SPACING_FLOOR_M: f64 = 45.0;
/// Motion brake: minimum time (s) a newly split tile should remain useful at
/// the eye's current speed. Selection refuses splits whose children would be
/// crossed faster than this (`child_arc < speed × MOTION_CROSS_MIN_S`), so a
/// fast-moving view streams a coarse-but-**stable** ground instead of fine
/// tiles that land mid-frame and read as pop-in — and a settling view
/// (speed → 0) refines back to the full distance rule in place. The brake is
/// speed-proportional and per-level: walking (~2 m/s) never trips it (the
/// finest Thalos tile arc is ~19 m), a 100 m/s freecam stops one or two
/// levels short, and only genuinely fast travel — which happens far from the
/// surface — coarsens further. Authored [`RefinementSite`] floors outrank it,
/// exactly as they outrank the distance rule.
const MOTION_CROSS_MIN_S: f64 = 1.0;
const MAX_IN_FLIGHT: usize = 24;

// --- residency budget --------------------------------------------------------
//
// INC-20260725T012104Z-tile-residency-had-no-budget: this renderer could
// allocate VRAM without limit until a `DeviceLost` stopped it.

/// Skirt ring vertices appended by [`build_tile_mesh`] (one loop around the
/// core grid: `4·(res−1)`, i.e. each side minus its shared corner).
const TILE_SKIRT_VERTS: usize = 4 * TILE_RES - 4;
/// Vertices in one tile mesh — core grid plus the skirt ring.
const TILE_VERTS: usize = TILE_RES * TILE_RES + TILE_SKIRT_VERTS;
/// Bytes per vertex across the attribute set [`build_tile_mesh`] writes:
/// POSITION 12 + NORMAL 12 + COLOR 16 + UV_0 8 + UV_1 8. Kept adjacent to the
/// mesher so adding an attribute updates the budget's denominator in the same
/// edit instead of silently under-counting VRAM.
///
/// COLOR and UV_1 carry the NTR-X4 spare-channel contract `tile_terrain.wgsl`
/// reads. They are also why this mesh can never enter the raytracing scene —
/// see [`RT_TILE_MESH_BYTES`] and [`crate::rt`].
const TILE_VERTEX_BYTES: usize = 12 + 12 + 16 + 8 + 8;
/// Indices in one tile mesh: two triangles per core quad plus two per skirt
/// quad.
const TILE_INDICES: usize = ((TILE_RES - 1) * (TILE_RES - 1) * 2 + TILE_SKIRT_VERTS * 2) * 3;
/// GPU bytes one resident tile costs — **347 KiB** at `TILE_RES = 65`.
///
/// This is the number the residency budget is denominated in, and the reason it
/// has to be: a *count* cap looks harmless and silently means gigabytes (the
/// same lesson `rendering::tile_cache`'s payload budget already records).
pub const TILE_MESH_BYTES: usize = TILE_VERTS * TILE_VERTEX_BYTES + TILE_INDICES * 4;

/// Extra GPU bytes a tile costs **on top of** [`TILE_MESH_BYTES`] when it also
/// carries an RT twin — **312 KiB**, i.e. an RT-covered tile costs **1.90×** a
/// plain one.
///
/// Solari's attribute gate is an equality, so the RT mesh is a second copy of
/// the geometry rather than a re-use of the visible one ([`crate::rt`] explains
/// why in full). That makes RT terrain coverage a VRAM decision before it is a
/// tracing-cost decision: covering the whole 4 GiB resident set would want
/// **another ~3.6 GiB**, on a budget that is already machine-wide and shared
/// between concurrent instances (INC-20260725T012104Z). That is the number
/// behind NTR-RT3 scoping RT proxies to a radius around `ViewAnchor` rather
/// than to residency — near-only coverage is a necessity here, not a tuning
/// choice.
pub const RT_TILE_MESH_BYTES: usize = TILE_VERTS * crate::rt::RT_VERTEX_BYTES + TILE_INDICES * 4;

/// Soft VRAM target for tile meshes **across every Thalos renderer on this
/// machine**, overridable with `THALOS_TILE_BUDGET_MB` (0 disables the budget
/// entirely). One process's share is this divided by the live instance count —
/// see [`residency_budget_bytes`] and [`vram_share`].
///
/// 4 GiB ≈ 12,080 tiles. Chosen to sit **above every working set we have
/// actually measured**, so the brake cannot silently regress a framing that was
/// already capture-verified: the launch-pad first-coverage is ~3,100 tiles
/// (1.0 GiB), the NTR-X4 showcase framings 3,427–4,419 (1.2–1.5 GiB), and the
/// hungriest measured case — NTR-X12's 30 m Mira descent under the ruggedness
/// rule — 10,900 (3.6 GiB). That leaves ~11 % margin over the worst known
/// framing while still braking hard on the churn that reached a `DeviceLost`.
///
/// It is therefore a **runaway brake, not a quality cap**, and deliberately
/// generous: the residency gauge is what tells us where it can be tightened, and
/// a smaller card wants `THALOS_TILE_BUDGET_MB` set down rather than this
/// guessed lower.
const DEFAULT_RESIDENCY_BUDGET_BYTES: usize = 4096 * 1024 * 1024;

/// Pressure fraction below which the split scale is allowed to recover. The
/// deadband between this and 1.0 is what stops the controller oscillating (a
/// scale that pumped every frame would read as visible LOD breathing).
const BUDGET_RECOVER_FRACTION: f64 = 0.8;
/// Multiplicative step down when over budget, and up when recovering. Down is
/// faster than up: overshoot costs VRAM we may not have, undershoot costs only
/// a few frames of softer distant relief.
const SPLIT_SCALE_DOWN: f64 = 0.9;
const SPLIT_SCALE_UP: f64 = 1.02;
/// Floor on the split scale — `SPLIT_FACTOR × 0.333 ≈ 2.0`, coarse but a
/// complete, walkable surface. The budget may take detail away; it may never
/// take the ground away.
const MIN_SPLIT_SCALE: f64 = 1.0 / 3.0;
/// Seconds between residency gauge lines.
const GAUGE_INTERVAL_S: f32 = 5.0;

/// The machine-wide tile VRAM target, read once from `THALOS_TILE_BUDGET_MB`.
fn machine_budget_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        let Ok(raw) = std::env::var("THALOS_TILE_BUDGET_MB") else {
            return DEFAULT_RESIDENCY_BUDGET_BYTES;
        };
        match raw.trim().parse::<usize>() {
            // 0 = no budget: the pre-budget behaviour, for an A/B that shows
            // what the brake is actually holding back.
            Ok(0) => {
                warn!("THALOS_TILE_BUDGET_MB=0 — tile residency budget DISABLED");
                usize::MAX
            }
            Ok(mb) => mb * 1024 * 1024,
            Err(_) => {
                warn!("THALOS_TILE_BUDGET_MB={raw:?} is not MiB; using the default");
                DEFAULT_RESIDENCY_BUDGET_BYTES
            }
        }
    })
}

/// This process's share of the tile VRAM budget: the machine-wide target split
/// evenly across the renderer instances currently alive.
///
/// A single instance therefore behaves **exactly** as before this split existed,
/// which is what keeps every capture-verified framing unregressed. A second
/// instance halves both — the case that reached a `DeviceLost` on 2026-07-25 at
/// 20:08 UTC with two 4 GiB-entitled processes on one 12 GB card, neither of
/// them ever braking (INC-20260725T012104Z-tile-residency-had-no-budget).
///
/// Re-read every frame rather than cached at boot: instances start and die at
/// arbitrary times, so a share fixed at startup would be wrong for whichever
/// process was already running — the one that would keep its full entitlement
/// precisely when a peer appears.
///
/// Public because it is the denominator of
/// [`TileTerrainRoot::resident_bytes`]: any readout of tile residency that
/// omits it cannot tell "512 MiB, plenty of room" from "512 MiB, about to
/// brake". `usize::MAX` means the budget is disabled
/// (`THALOS_TILE_BUDGET_MB=0`).
pub fn residency_budget_bytes() -> usize {
    let machine = machine_budget_bytes();
    if machine == usize::MAX {
        return usize::MAX;
    }
    machine / vram_share::live_instances().max(1)
}
/// Despawn laxness: a despawn-ready tile lingers until BOTH gates pass —
/// wall-clock (covers upload stalls) and frame count (covers the render
/// pipeline, which is per-frame not per-second). The lingering tile keeps
/// drawing *alongside* its replacement, so the overlap has to be resolved in
/// favour of the finer surface — see [`LEVEL_RENDER_LIFT_M`].
const DESPAWN_GRACE_S: f32 = 0.5;
const DESPAWN_GRACE_FRAMES: u16 = 20;
/// Per-level **geometric** lift (metres of extra radius) baked into a tile's
/// rendered vertices: level `L` draws at `radius + h + L ×` this.
///
/// The overlap this resolves is structural, not incidental. On a merge the
/// coarse ancestor lands *before* its fine children retire (the despawn rule is
/// hole-free by construction), so for `DESPAWN_GRACE_S` the two surfaces are
/// both drawn and interpenetrate wherever they disagree — and the coarse one
/// wins any pixel where it happens to sit nearer, which is what buried
/// structures draped on the ground (paving sits only ~0.12 m proud).
///
/// This **used to be** `StandardMaterial::depth_bias`, on the belief that it was
/// a hardware `DepthBiasState.constant`. It is not: Bevy folds `depth_bias` into
/// the render phase's *sort distance* only (`core_3d`: `rangefinder.distance() +
/// depth_bias`), and sort order decides nothing among opaque geometry — the
/// depth test does. So the "finer detail always wins" invariant the tile
/// renderer documented was never actually implemented, and coarse-over-fine
/// overdraw flickered on every camera move (INC-20260725T191500Z).
///
/// A radial lift is the honest version of the same intent: it is real geometry,
/// so it holds at any distance and any depth precision, and it is
/// view-independent, so it can never pop. It buys **coplanar** cases — the
/// interior of a structure pad, flat plains — where the two levels sample the
/// same surface and only the mesh differs. It does not, and cannot, order two
/// meshes that genuinely disagree by metres over rugged ground; nothing short of
/// not overlapping does that, and not overlapping trades flicker for holes.
///
/// The step is bounded by what may hide under it: the runway asphalt sits
/// 0.12 m over the pad, and `max_level` is 18, so the deepest tile draws 36 mm
/// high — invisible on terrain, and 3× clear of the thinnest thing draped on it.
/// The height mirror publishes the *provider's* heights, not these, so
/// colliders, the camera floor and HUD altitude are untouched by the lift.
pub const LEVEL_RENDER_LIFT_M: f64 = 0.002;

// --- cube-sphere addressing (probe tiles.rs, verbatim) -----------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileKey {
    /// Canonical cube face, 0..6 (+X, −X, +Y, −Y, +Z, −Z).
    pub face: u8,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl TileKey {
    pub fn tiles_per_side(self) -> u32 {
        1u32 << self.level
    }

    pub fn uv_rect(self) -> (f64, f64, f64, f64) {
        let n = self.tiles_per_side() as f64;
        let w = 2.0 / n;
        let u0 = -1.0 + w * self.x as f64;
        let v0 = -1.0 + w * self.y as f64;
        (u0, v0, u0 + w, v0 + w)
    }

    pub fn containing(face: u8, level: u8, u: f64, v: f64) -> Self {
        let n = 1u32 << level;
        let t = |c: f64| (((c + 1.0) * 0.5 * n as f64) as i64).clamp(0, n as i64 - 1) as u32;
        Self {
            face,
            level,
            x: t(u),
            y: t(v),
        }
    }

    pub fn containing_dir(dir: DVec3, level: u8) -> Self {
        let (face, u, v) = face_uv_of_dir(dir);
        Self::containing(face, level, u, v)
    }

    pub fn parent(self) -> Option<Self> {
        (self.level > 0).then(|| Self {
            face: self.face,
            level: self.level - 1,
            x: self.x / 2,
            y: self.y / 2,
        })
    }

    pub fn children(self) -> [Self; 4] {
        let (f, l, x, y) = (self.face, self.level + 1, self.x * 2, self.y * 2);
        [
            Self {
                face: f,
                level: l,
                x,
                y,
            },
            Self {
                face: f,
                level: l,
                x: x + 1,
                y,
            },
            Self {
                face: f,
                level: l,
                x,
                y: y + 1,
            },
            Self {
                face: f,
                level: l,
                x: x + 1,
                y: y + 1,
            },
        ]
    }

    /// Unit direction at fractional in-tile coords (halo addresses extend
    /// past [0, 1] — gnomonic extension stays valid across face edges).
    pub fn dir_at(self, s: f64, t: f64) -> DVec3 {
        let (u0, v0, u1, v1) = self.uv_rect();
        face_dir(self.face, u0 + (u1 - u0) * s, v0 + (v1 - v0) * t)
    }

    pub fn center_dir(self) -> DVec3 {
        self.dir_at(0.5, 0.5)
    }

    pub fn sample_spacing_m(self, radius_m: f64) -> f64 {
        let face_arc = radius_m * core::f64::consts::FRAC_PI_2;
        face_arc / self.tiles_per_side() as f64 / (TILE_RES - 1) as f64
    }
}

/// Cube-face → unit direction; ∂u × ∂v points outward on every face (the
/// mesher's winding self-test enforces the consequence — probe M0 postmortem).
pub fn face_dir(face: u8, u: f64, v: f64) -> DVec3 {
    let d = match face {
        0 => DVec3::new(1.0, v, -u),
        1 => DVec3::new(-1.0, v, u),
        2 => DVec3::new(u, 1.0, -v),
        3 => DVec3::new(u, -1.0, v),
        4 => DVec3::new(u, v, 1.0),
        5 => DVec3::new(-u, v, -1.0),
        _ => unreachable!("six faces"),
    };
    d.normalize()
}

/// Exact inverse of [`face_dir`] — the cross-face adjacency transform.
pub fn face_uv_of_dir(dir: DVec3) -> (u8, f64, f64) {
    let a = dir.abs();
    if a.x >= a.y && a.x >= a.z {
        if dir.x >= 0.0 {
            (0, -dir.z / dir.x, dir.y / dir.x)
        } else {
            (1, dir.z / -dir.x, dir.y / -dir.x)
        }
    } else if a.y >= a.z {
        if dir.y >= 0.0 {
            (2, dir.x / dir.y, -dir.z / dir.y)
        } else {
            (3, dir.x / -dir.y, dir.z / -dir.y)
        }
    } else if dir.z >= 0.0 {
        (4, dir.x / dir.z, dir.y / dir.z)
    } else {
        (5, dir.x / dir.z, -dir.y / dir.z)
    }
}

// --- tile payload + provider ---------------------------------------------------

/// One sampled tile: heights + linear albedo + canonical material-selection
/// inputs (`[eco_altitude_m, forest]`, see `thalos_terrain::MaterialBands`)
/// on the halo grid.
pub struct SurfaceTile {
    pub key: TileKey,
    pub sample_spacing_m: f64,
    pub heights_m: Vec<f32>,
    pub albedo_linear: Vec<[f32; 3]>,
    pub bands: Vec<[f32; 2]>,
}

impl SurfaceTile {
    pub fn grid_side() -> usize {
        TILE_RES + 2 * TILE_HALO
    }
}

/// The provider seam (ADR-20260722T105147Z part 1). Slice 1 wraps the body's
/// canonical `SurfaceQuery`; package/neural producers slot in behind the same
/// boundary.
pub trait TerrainTileProvider: Send + Sync {
    fn request(&self, key: TileKey, radius_m: f64) -> SurfaceTile;

    /// Peak relief above the reference sphere, metres — the allowance the
    /// horizon test lifts every tile by so terrain that legitimately pokes over
    /// the limb is never culled. Conservative by default (no provider metadata
    /// means "assume anything", which only costs refinement work).
    fn height_range_m(&self) -> f32 {
        f32::INFINITY
    }
}

/// `SurfaceQuery`-backed provider — samples the body's one height/albedo
/// authority per tile vertex (f64 directions; see `SurfaceQuery::sample_d`).
pub struct SurfaceQueryProvider {
    pub surface: Arc<dyn SurfaceQuery>,
}

impl TerrainTileProvider for SurfaceQueryProvider {
    fn height_range_m(&self) -> f32 {
        self.surface.height_range_m()
    }

    fn request(&self, key: TileKey, radius_m: f64) -> SurfaceTile {
        let spacing = key.sample_spacing_m(radius_m);
        let side = SurfaceTile::grid_side();
        let step = 1.0 / (TILE_RES - 1) as f64;
        // Package/point-query sampling is the expensive half (the
        // raster->point->raster tax, ADR-20260722T105147Z) — spread rows
        // across the shared *bounded* eval pool, exactly like udlod's
        // `compute_tile_pixels` (never rayon's implicit global pool; see
        // `ground::tile_synthesis_pool`).
        //
        // Evaluating whole tiles on a widened outer pool instead was tried and
        // reverted; see `TILE_SYNTHESIS_THREADS` for the measurement that killed
        // it. Cheap tiles come from the cache tier above this provider, not from
        // a different thread shape.
        let rows: Vec<(Vec<f32>, Vec<[f32; 3]>, Vec<[f32; 2]>)> =
            crate::ground::tile_synthesis_pool::tile_eval_pool().install(|| {
                use rayon::prelude::*;
                (0..side)
                    .into_par_iter()
                    .map(|j| {
                        let mut h = Vec::with_capacity(side);
                        let mut a = Vec::with_capacity(side);
                        let mut b = Vec::with_capacity(side);
                        let t = (j as f64 - TILE_HALO as f64) * step;
                        for i in 0..side {
                            let s = (i as f64 - TILE_HALO as f64) * step;
                            let (sample, bands) = self
                                .surface
                                .sample_bands_d(key.dir_at(s, t), spacing as f32);
                            h.push(sample.height_m);
                            a.push([
                                sample.albedo_linear.x,
                                sample.albedo_linear.y,
                                sample.albedo_linear.z,
                            ]);
                            b.push([bands.eco_altitude_m, bands.canopy]);
                        }
                        (h, a, b)
                    })
                    .collect()
            });
        let mut heights = Vec::with_capacity(side * side);
        let mut albedo = Vec::with_capacity(side * side);
        let mut bands = Vec::with_capacity(side * side);
        for (h, a, b) in rows {
            heights.extend(h);
            albedo.extend(a);
            bands.extend(b);
        }
        SurfaceTile {
            key,
            sample_spacing_m: spacing,
            heights_m: heights,
            albedo_linear: albedo,
            bands,
        }
    }
}

// --- mesher (probe mesher.rs, albedo-driven) ------------------------------------

/// Wrap period (m) of the body-fixed position the mesh carries in its spare
/// UV channels for the material shader (NTR-X4 layers). Vertices store
/// `p_body − anchor` where `anchor` is the tile origin snapped down to a
/// multiple of this, so values are continuous within a tile and agree across
/// tiles mod the period — every texture wavelength in `tile_terrain.wgsl`
/// must divide this exactly (same discipline as udlod's wrapped detail
/// noise). Mirrored as `TILE_WRAP_M` in the shader.
pub const TILE_WRAP_M: f64 = 8192.0;

/// Fallback skirt depth for a provider with **no relief metadata**
/// (`height_range_m() == INFINITY`): chord sag + band-gate allowance (the
/// original probe formula).
///
/// Every real provider has a finite envelope, and those tiles skirt down to
/// the body-wide floor sphere instead (see the curtain construction in
/// [`build_tile_mesh`]). This formula's 150 m clamp is exactly what made
/// inter-level cracks visible: its own allowance model (`spacing × 0.06`)
/// wants ~550 m at Thalos's `MIN_LEVEL` spacing, and a *transient* junction —
/// a fast-orbiting eye juxtaposing freshly-revealed coarse tiles with
/// lingering fine ones, levels apart — disagrees by far more than any
/// per-own-spacing drop can cover (`skirt_tests::junction_cracks_exceed_the_old_skirt_clamp`).
pub fn skirt_drop_m(sample_spacing_m: f64, radius_m: f64) -> f32 {
    let sag = sample_spacing_m * sample_spacing_m / (8.0 * radius_m);
    (sag * 4.0 + sample_spacing_m * 0.06).clamp(0.5, 150.0) as f32
}

struct BuiltTile {
    mesh: Mesh,
    /// Body-fixed f64 position of the mesh origin (displaced tile center).
    origin: DVec3,
    /// Radial deviation range (m) of the built interior vertices from the
    /// reference sphere — the mesh-side counterpart of the provider height
    /// range, so "heights sampled" vs "heights in the mesh" separate in the
    /// telemetry.
    mesh_h: (f32, f32),
}

/// Debug relief exaggeration (`THALOS_TILE_HEIGHT_SCALE`, default 1.0) —
/// makes "is displacement reaching the rendered mesh at all" undeniable.
fn debug_height_scale() -> f64 {
    static SCALE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *SCALE.get_or_init(|| {
        std::env::var("THALOS_TILE_HEIGHT_SCALE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(1.0)
    })
}

fn build_tile_mesh(tile: &SurfaceTile, radius_m: f64, relief_m: f64) -> BuiltTile {
    let key = tile.key;
    let halo = TILE_HALO;
    let side = SurfaceTile::grid_side();
    let res = TILE_RES;
    let step = 1.0 / (res - 1) as f64;
    let h_scale = debug_height_scale();

    // Finer levels render fractionally further out so a refined tile owns the
    // pixels wherever it overlaps a lingering coarser one (see
    // `LEVEL_RENDER_LIFT_M`). Rendering only — `tile.heights_m` is what the
    // height mirror publishes to colliders and the camera floor.
    let lift = key.level as f64 * LEVEL_RENDER_LIFT_M;
    let mut pos_grid: Vec<DVec3> = Vec::with_capacity(side * side);
    for j in 0..side {
        for i in 0..side {
            let s = (i as f64 - halo as f64) * step;
            let t = (j as f64 - halo as f64) * step;
            let h = tile.heights_m[j * side + i] as f64 * h_scale;
            pos_grid.push(key.dir_at(s, t) * (radius_m + h + lift));
        }
    }
    let origin =
        key.center_dir() * (radius_m + tile.heights_m[(side / 2) * side + side / 2] as f64);
    // Material-shader position anchor: the origin snapped down to the wrap
    // period, so `p_body − anchor` is continuous within the tile and agrees
    // with neighbouring tiles mod TILE_WRAP_M (see the const's docs).
    let wrap_anchor = DVec3::new(
        (origin.x / TILE_WRAP_M).floor() * TILE_WRAP_M,
        (origin.y / TILE_WRAP_M).floor() * TILE_WRAP_M,
        (origin.z / TILE_WRAP_M).floor() * TILE_WRAP_M,
    );

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(res * res);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(res * res);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(res * res);
    let mut uv0: Vec<[f32; 2]> = Vec::with_capacity(res * res);
    let mut uv1: Vec<[f32; 2]> = Vec::with_capacity(res * res);
    let mut indices: Vec<u32> = Vec::with_capacity((res - 1) * (res - 1) * 6);

    let mut mesh_h = (f32::INFINITY, f32::NEG_INFINITY);
    for j in 0..res {
        for i in 0..res {
            let (gi, gj) = (i + halo, j + halo);
            let p = pos_grid[gj * side + gi];
            // Report the terrain height, not the render lift folded into it, so
            // the diagnostic stays comparable with the provider's own range.
            let dev = (p.length() - radius_m - lift) as f32;
            mesh_h.0 = mesh_h.0.min(dev);
            mesh_h.1 = mesh_h.1.max(dev);
            let rel = p - origin;
            positions.push([rel.x as f32, rel.y as f32, rel.z as f32]);
            let du = pos_grid[gj * side + gi + 1] - pos_grid[gj * side + gi - 1];
            let dv = pos_grid[(gj + 1) * side + gi] - pos_grid[(gj - 1) * side + gi];
            let mut n = du.cross(dv).normalize();
            let outward = key.dir_at(i as f64 * step, j as f64 * step);
            if n.dot(outward) < 0.0 {
                n = -n;
            }
            normals.push([n.x as f32, n.y as f32, n.z as f32]);
            let a = tile.albedo_linear[gj * side + gi];
            // Spare-channel contract with `tile_terrain.wgsl` (NTR-X4):
            // uv0 + uv1.x = wrapped body-fixed position, uv1.y = canonical
            // ecological altitude (m), color.a = canonical forest band weight.
            let b = tile.bands[gj * side + gi];
            colors.push([a[0], a[1], a[2], b[1]]);
            let wrapped = p - wrap_anchor;
            uv0.push([wrapped.x as f32, wrapped.y as f32]);
            uv1.push([wrapped.z as f32, b[0]]);
        }
    }
    for j in 0..res - 1 {
        for i in 0..res - 1 {
            let a = (j * res + i) as u32;
            let b = a + 1;
            let c = a + res as u32;
            let d = c + 1;
            // CCW in (s, t) → outward front faces.
            indices.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }

    // Winding self-test — a probe M0 contract obligation. Skipped under the
    // debug height exaggeration: extreme scales legitimately invert steep
    // triangles, which is the exaggeration doing its job, not a winding bug.
    if (h_scale - 1.0).abs() < 1e-9 {
        let tri = [
            indices[0] as usize,
            indices[1] as usize,
            indices[2] as usize,
        ];
        let (p0, p1, p2) = (
            DVec3::from(positions[tri[0]].map(f64::from)),
            DVec3::from(positions[tri[1]].map(f64::from)),
            DVec3::from(positions[tri[2]].map(f64::from)),
        );
        let n = (p1 - p0).cross(p2 - p0);
        debug_assert!(
            n.dot(key.center_dir()) > 0.0,
            "tile {key:?}: first triangle winds inward"
        );
    }

    // Skirt: every border vertex hangs a radial curtain down to a body-wide
    // floor sphere below the deepest terrain (`radius − relief`, the provider's
    // conservative absolute-height envelope). Depth here is what decides
    // whether an inter-level junction reads as ground or as a see-through
    // crack: the two edges disagree by the coarser side's chord error plus the
    // LOD-gated detail difference, and while streaming settles after a fast
    // camera move the resident mosaic transiently juxtaposes tiles *levels*
    // apart — freshly-landed coarse ground against lingering fine tiles —
    // where that disagreement reaches hundreds of metres. A curtain to the
    // floor sphere covers ANY resident partner by construction: no partner
    // surface (or chord between its samples) can sit below the body's own
    // minimum terrain. Settled junctions bury the walls under the neighbour's
    // surface, so the extra depth costs no vertices, no VRAM, and no visible
    // geometry — only a deeper mesh AABB.
    //
    // A provider without relief metadata (INFINITY) falls back to the original
    // spacing-scaled drop; feeding an unbounded figure through the curtain
    // maths would put the floor at −∞, not at "deep enough".
    let legacy_drop_m = skirt_drop_m(tile.sample_spacing_m, radius_m) as f64;
    let floor_radius_m = radius_m - relief_m.max(0.0);
    let border: Vec<u32> = {
        let mut b = Vec::new();
        for i in 0..res {
            b.push(i as u32);
        }
        for j in 1..res {
            b.push((j * res + res - 1) as u32);
        }
        for i in (0..res - 1).rev() {
            b.push(((res - 1) * res + i) as u32);
        }
        for j in (1..res - 1).rev() {
            b.push((j * res) as u32);
        }
        b
    };
    let down = -key.center_dir();
    let base = positions.len() as u32;
    for &idx in &border {
        let p = positions[idx as usize];
        let abs = origin + DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
        let bottom = if relief_m.is_finite() {
            abs.normalize() * floor_radius_m
        } else {
            abs + down * legacy_drop_m
        } - origin;
        positions.push([bottom.x as f32, bottom.y as f32, bottom.z as f32]);
        normals.push(normals[idx as usize]);
        colors.push(colors[idx as usize]);
        uv0.push(uv0[idx as usize]);
        uv1.push(uv1[idx as usize]);
    }
    let n_border = border.len() as u32;
    for k in 0..n_border {
        let k2 = (k + 1) % n_border;
        let (top_a, top_b) = (border[k as usize], border[k2 as usize]);
        let (bot_a, bot_b) = (base + k, base + k2);
        indices.extend_from_slice(&[top_a, bot_a, bot_b, top_a, bot_b, top_b]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv0);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    mesh.insert_indices(Indices::U32(indices));
    BuiltTile {
        mesh,
        origin,
        mesh_h,
    }
}

// --- streaming (probe streaming.rs, per-root) ------------------------------------

/// The selection eye, written each frame by the game (from `ViewAnchor`):
/// body-fixed camera position for the root entity's body.
#[derive(Resource, Default)]
pub struct TileEye {
    pub target: Option<TileEyeTarget>,
}

pub struct TileEyeTarget {
    /// The entity carrying [`TileTerrainRoot`] (the body's big_space grid).
    pub root: Entity,
    /// Camera position in the body-fixed frame, meters.
    pub cam_body: DVec3,
    /// Smoothed body-fixed eye speed (m/s) — drives the motion brake
    /// ([`MOTION_CROSS_MIN_S`]). Zero disables it (a settled or headless eye).
    pub speed_m_s: f64,
    /// Body centre in world space, **f64**.
    pub body_position: DVec3,
    /// Body-fixed → world surface orientation, **f64** — the same authority
    /// every other surface consumer shares
    /// (`transforms::surface_orientation_authored`). Precision here is
    /// load-bearing, not incidental: see [`TileBodyOrigin`].
    pub body_orientation: DQuat,
}

/// A resident tile's origin in the **body-fixed** frame — the point its mesh
/// vertices are stored relative to — kept so [`stream_tile_terrain`] can
/// re-place the tile in world space from the body's f64 pose every frame.
///
/// **Why tiles are not children of the body's rotating grid.** That is the
/// natural-looking parenting, and it is a trap this repo has already paid for
/// twice. A tile's body-fixed origin has magnitude ≈ the planet radius, and
/// big_space would rotate that multi-Mm offset into world space with the grid
/// entity's `Transform.rotation` — an **f32** quaternion. f32 quaternion ULP at
/// 3,186 km is a *decimetre*: measured over the spin cycle, mean 0.055 m,
/// p95 0.165 m, worst 0.256 m, re-rolling every frame as the body turns. The
/// runway's asphalt is lifted 0.12 m over the pad, so the ground crossed above
/// and below the paving frame to frame and the whole space center flickered in
/// and out of the terrain (INC-20260725T195500Z).
///
/// So tiles are root-grid children placed in f64 here, exactly as
/// `update_runway_transform` places the runway and for the identical reason,
/// leaving the f32 `Transform.rotation` acting only on in-tile vertex offsets
/// (≤ 0.04 m of ULP even for a `MIN_LEVEL` tile, and micrometres near the
/// surface). The legacy udlod ground dodges the same trap via
/// `PreciseRotation`; this is the tile path's equivalent.
#[derive(Component, Debug, Clone, Copy)]
pub struct TileBodyOrigin(pub DVec3);

/// Set containing [`stream_tile_terrain`]. The driver that writes [`TileEye`]
/// must run **before** this: the streamer places tiles from the target's f64
/// body pose, and a frame-stale pose would slide the ground metres against
/// everything built on it (232 m/s of surface speed at Thalos's equator).
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileStreamSet;

/// World placement of a body-fixed point, in f64 up to the final grid split.
fn place_body_point(origin_body: DVec3, target: &TileEyeTarget, grid: &Grid) -> (CellCoord, Vec3) {
    grid.translation_to_grid(target.body_position + target.body_orientation * origin_body)
}

struct StreamedTile {
    key: TileKey,
    built: BuiltTile,
    gen_micros: u64,
    /// Sampled height range (m) — diagnostic for flat-terrain regressions.
    h_range: (f32, f32),
    /// The provider's halo height grid, handed to the CPU height mirror so
    /// scatter / colliders read the heights this mesh was built from (see
    /// [`height_mirror`]).
    heights_m: Arc<Vec<f32>>,
}

/// One streaming tile terrain on a body grid entity. Insert on the body's
/// `RealSpaceBody` grid; tiles spawn as its co-rotating children.
#[derive(Component)]
pub struct TileTerrainRoot {
    pub radius_m: f64,
    pub provider: Arc<dyn TerrainTileProvider>,
    /// The one material every tile shares. This used to be a per-level `Vec`
    /// differing only in `depth_bias`; that bias never did anything (see
    /// [`LEVEL_RENDER_LIFT_M`]), so the levels are now genuinely identical and
    /// the set collapsed to a single handle.
    pub material: Handle<material::TileTerrainMaterial>,
    pub max_level: u8,
    /// CPU mirror of the resident tiles' heights — the ground authority every
    /// surface consumer (scatter, colliders, camera floor, HUD altitude) reads
    /// while this renderer owns the body. Cloned into the game's rendered-ground
    /// registry when the root is installed.
    pub height_mirror: TileHeightMirrorHandle,
    /// Render layers stamped on every spawned tile. Explicit (not defaulted)
    /// because the tiles are real-scale ground: a host with more than one
    /// camera over the same scene — Thalos draws the 1:1 ship view and the
    /// 1e-6-scaled map view from the same world — will otherwise draw a
    /// planet-sized landscape into the far-scale view — which is exactly what
    /// the default layer 0 did (the udlod ground it replaces has always been
    /// `SHIP_LAYER`-only).
    pub render_layers: RenderLayers,
    /// When set, every streamed tile also spawns a **shadow-caster child**: the
    /// same mesh handle drawn with this material on these layers (the game's
    /// sun-shadow cascade layer). The child is what lets terrain cast into the
    /// cascade rig — ridges shadow valleys, hills shadow plains — without the
    /// cascade cameras paying the real tile material's layer-stack fragment
    /// cost over their whole 4096² targets: the caster material is a bare
    /// unlit `StandardMaterial`, so the ortho passes rasterize depth almost
    /// for free. `None` (default) keeps terrain a pure receiver.
    pub caster: Option<(Handle<StandardMaterial>, RenderLayers)>,
    desired: HashSet<TileKey>,
    resident: HashMap<TileKey, Entity>,
    pending: HashMap<TileKey, Task<StreamedTile>>,
    /// Tiles whose despawn condition holds, counting down the lax grace
    /// gates (seconds, frames remaining). If coverage regresses meanwhile,
    /// the entry is dropped and the tile stays.
    retiring: HashMap<TileKey, (f32, u16)>,
    /// Seconds until the next coverage-invariant audit (see the check in
    /// `stream_tile_terrain`).
    coverage_check_countdown: f32,
    /// Forensics ring: recent despawns as (key, elapsed-seconds stamp,
    /// merge-case?) so a failing audit can name the ancestor that dropped and
    /// under which certificate. Capped at 512.
    recent_despawns: Vec<(TileKey, f64, bool)>,
    /// Measured ruggedness (relief ÷ arc) of every tile this session has ever
    /// generated, feeding the relief-aware split rule via
    /// [`Self::ruggedness_at`]. Deliberately **not** pruned on despawn: the
    /// split factor of a key must not change when its tile leaves residency,
    /// or refinement would undo itself the moment it succeeded. Monotone
    /// knowledge (each key measured once) is also what keeps the selection
    /// from oscillating — see the descent simulation.
    ruggedness: HashMap<TileKey, f32>,
    /// Authored ground the selection must resolve however far the eye is (see
    /// [`RefinementSite`]). Republished each frame by the driver that owns the
    /// body's structure pads, so a pad placed at runtime is honoured on the next
    /// selection instead of at the next root install.
    refinement_sites: Vec<RefinementSite>,
    /// Peak relief above the reference sphere (m), from the provider — the
    /// allowance [`above_horizon`] lifts tiles by. Cached at construction
    /// because it is a per-body constant.
    relief_m: f64,
    gen_stats: GenStats,
    /// Latched true the first time the desired set is fully resident. After
    /// that, the hole-free despawn rule keeps coverage complete, so the
    /// impostor↔terrain swap can trust it without flickering back.
    covered_once: bool,
    /// Budget controller state: multiplier on the split factors, 1.0 = the
    /// unconstrained rule. Driven by [`Self::update_split_scale`].
    split_scale: f64,
    /// Seconds until the next residency gauge line.
    gauge_countdown: f32,
}

/// Rolling per-tile generation-time telemetry; records every 200 landings so
/// the raster->point->raster tax stays measured (the tile-native package
/// provider is the planned fix). The runtime routes this target to JSONL.
#[derive(Default)]
struct GenStats {
    samples: Vec<u64>,
    total_landed: u64,
    h_min: f32,
    h_max: f32,
    mesh_min: f32,
    mesh_max: f32,
}

impl GenStats {
    fn record(&mut self, micros: u64, h_range: (f32, f32), mesh_h: (f32, f32)) {
        self.samples.push(micros);
        self.total_landed += 1;
        self.h_min = self.h_min.min(h_range.0);
        self.h_max = self.h_max.max(h_range.1);
        self.mesh_min = self.mesh_min.min(mesh_h.0);
        self.mesh_max = self.mesh_max.max(mesh_h.1);
        if self.samples.len() >= 200 {
            self.samples.sort_unstable();
            let mean = self.samples.iter().sum::<u64>() / self.samples.len() as u64;
            let p95 = self.samples[self.samples.len() * 95 / 100];
            info!(
                target: "thalos::diagnostic::tile_terrain",
                event = "generation_batch",
                total_landed = self.total_landed,
                sample_count = self.samples.len(),
                mean_ms = mean as f64 / 1000.0,
                p95_ms = p95 as f64 / 1000.0,
                provider_height_min_m = self.h_min,
                provider_height_max_m = self.h_max,
                mesh_height_min_m = self.mesh_min,
                mesh_height_max_m = self.mesh_max,
                "tile generation batch"
            );
            self.samples.clear();
        }
    }
}

/// Deepest quadtree level for a body of `radius_m` — targets ~9 m sample
/// spacing (probe budget). Public so drivers can size per-level resources
/// (e.g. the depth-biased material set) before constructing the root.
pub fn max_level_for(radius_m: f64) -> u8 {
    let face_arc = radius_m * core::f64::consts::FRAC_PI_2;
    ((face_arc / ((TILE_RES - 1) as f64 * 9.0)).log2().ceil() as u8).clamp(MIN_LEVEL + 1, 18)
}

impl TileTerrainRoot {
    pub fn new(
        radius_m: f64,
        provider: Arc<dyn TerrainTileProvider>,
        material: Handle<material::TileTerrainMaterial>,
        render_layers: RenderLayers,
    ) -> Self {
        let max_level = max_level_for(radius_m);
        let relief_m = provider.height_range_m() as f64;
        Self {
            height_mirror: Arc::new(RwLock::new(TileHeightMirror::new(radius_m, max_level))),
            radius_m,
            provider,
            material,
            max_level,
            render_layers,
            caster: None,
            refinement_sites: Vec::new(),
            desired: HashSet::new(),
            resident: HashMap::new(),
            pending: HashMap::new(),
            retiring: HashMap::new(),
            coverage_check_countdown: 2.0,
            recent_despawns: Vec::new(),
            ruggedness: HashMap::new(),
            relief_m,
            gen_stats: GenStats::default(),
            covered_once: false,
            split_scale: 1.0,
            gauge_countdown: 0.0,
        }
    }

    /// Release everything this root owns, returning the tile entities the caller
    /// must despawn.
    ///
    /// Used when the root hands off to another body: tiles are children of the
    /// **big_space root**, not of this component's entity (see
    /// [`TileBodyOrigin`]), so removing the component would orphan every
    /// resident tile into the scene rather than despawn it.
    ///
    /// `ruggedness` is cleared too, unlike on despawn where it is deliberately
    /// kept: it is a measurement of *this body's* terrain, and a `TileKey` is
    /// only unique within a body. Carrying it across a handoff would apply one
    /// body's mountains to another's coordinates.
    pub fn release_all(&mut self) -> Vec<Entity> {
        let entities: Vec<Entity> = self.resident.values().copied().collect();
        self.resident.clear();
        // Dropping a pending task aborts it (the streamer relies on the same
        // behaviour when it cancels un-wanted tiles).
        self.pending.clear();
        self.desired.clear();
        self.retiring.clear();
        self.recent_despawns.clear();
        self.ruggedness.clear();
        self.refinement_sites.clear();
        self.covered_once = false;
        if let Ok(mut mirror) = self.height_mirror.write() {
            mirror.clear();
        }
        entities
    }

    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// Tiles the selector currently *wants*, before the brake. Paired with
    /// [`Self::resident_count`] this is what separates "the framing is settled"
    /// from "the framing is being held back": a desire far above residency
    /// while [`Self::split_scale`] is under 1.0 is the brake biting, and a
    /// capture receipt records both so the reader does not have to guess.
    pub fn desired_count(&self) -> usize {
        self.desired.len()
    }

    /// Replace the authored refinement floors (see [`RefinementSite`]).
    /// Idempotent and cheap — the driver calls it every frame from the body's
    /// flatten handle, and a no-op change costs one slice comparison.
    pub fn set_refinement_sites(&mut self, sites: Vec<RefinementSite>) {
        if self.refinement_sites != sites {
            self.refinement_sites = sites;
        }
    }

    /// VRAM the landed tile meshes occupy right now.
    pub fn resident_bytes(&self) -> usize {
        self.resident.len() * TILE_MESH_BYTES
    }

    /// VRAM already committed: landed tiles plus the in-flight ones that are
    /// going to land. The budget controller's input — using resident alone
    /// would let 24 more tiles arrive after the brake was already needed.
    pub fn committed_bytes(&self) -> usize {
        (self.resident.len() + self.pending.len()) * TILE_MESH_BYTES
    }

    /// Current split-factor multiplier: 1.0 while inside the residency budget,
    /// lower while the brake is holding detail back.
    pub fn split_scale(&self) -> f64 {
        self.split_scale
    }

    /// Steer [`Self::split_scale`] from committed VRAM.
    ///
    /// **Why selection and not eviction.** The despawn rule is hole-free by
    /// construction — a tile may only leave once its footprint is served by
    /// other resident tiles — so a budget that evicted resident tiles would
    /// punch holes in the ground. Coarsening *desire* instead makes the coarse
    /// ancestor desired again; once it lands, the fine tiles it covers retire
    /// through the normal merge certificate and the budget is recovered with no
    /// hole at any instant.
    ///
    /// The consequence to keep in mind: satisfying the brake costs a transient
    /// *increase* (the ancestor lands before its children leave, ≤ ~1/3 pyramid
    /// overhead), which is why the budget must sit below real VRAM with headroom
    /// rather than at it.
    fn update_split_scale(&mut self, budget_bytes: usize) {
        if budget_bytes == usize::MAX {
            self.split_scale = 1.0;
            return;
        }
        let committed = self.committed_bytes() as f64;
        let budget = budget_bytes as f64;
        if committed > budget {
            self.split_scale = (self.split_scale * SPLIT_SCALE_DOWN).max(MIN_SPLIT_SCALE);
        } else if committed < budget * BUDGET_RECOVER_FRACTION && self.split_scale < 1.0 {
            self.split_scale = (self.split_scale * SPLIT_SCALE_UP).min(1.0);
        }
    }

    /// Ruggedness under `key` for the split rule: its own measurement if this
    /// session has ever generated it, else the nearest measured ancestor's
    /// (ruggedness is scale-invariant, so it inherits unscaled). `None` above
    /// the first measurement — a cold selection is the plain distance rule.
    fn ruggedness_at(&self, key: TileKey) -> Option<f32> {
        let mut k = key;
        loop {
            if let Some(&r) = self.ruggedness.get(&k) {
                return Some(r);
            }
            match k.parent() {
                Some(p) if p.level >= MIN_LEVEL => k = p,
                _ => return None,
            }
        }
    }

    /// True once the streamed terrain has (ever) fully covered the desired
    /// selection — the impostor↔terrain handoff criterion (the analogue of
    /// udlod's `pinned_tiles_ready`).
    pub fn coverage_ready(&self) -> bool {
        self.covered_once
    }

    pub fn settled(&self) -> bool {
        self.pending.is_empty() && self.desired.iter().all(|k| self.resident.contains_key(k))
    }

    /// Sample spacing (m/vertex) of the finest resident tile containing
    /// `dir_body`, or `None` while nothing covers that direction yet. The
    /// tile-path analogue of udlod's `renderer_tile_lod_m_at` — used by the
    /// surface-settle gate and the capture readiness hold to answer "how
    /// refined is the streamed ground at this exact spot".
    pub fn resident_spacing_m_at(&self, dir_body: DVec3) -> Option<f32> {
        for level in (MIN_LEVEL..=self.max_level).rev() {
            let key = TileKey::containing_dir(dir_body, level);
            if self.resident.contains_key(&key) {
                return Some(key.sample_spacing_m(self.radius_m) as f32);
            }
        }
        None
    }
}

fn tile_arc_m(level: u8, radius_m: f64) -> f64 {
    radius_m * core::f64::consts::FRAC_PI_2 / (1u64 << level) as f64
}

/// How much of the [`SPLIT_FACTOR`] → [`SPLIT_FACTOR_RUGGED`] range a tile of
/// this ruggedness earns: 0 on water and plains, 1 on mountain flanks.
///
/// Ruggedness is `relief ÷ arc`, which is *scale-invariant* for fractal
/// terrain — a mountainside reads the same at level 6 and level 12. That is
/// what lets an unmeasured tile inherit its nearest measured ancestor's value
/// unscaled (see [`TileTerrainRoot::ruggedness_at`]); a relief figure could
/// not be inherited without a Hurst assumption that compounds every level and
/// ratchets the selection deeper each descent.
fn tile_ruggedness_weight(rugged: f32) -> f64 {
    let t = ((rugged - RUGGED_LO) / (RUGGED_HI - RUGGED_LO)).clamp(0.0, 1.0) as f64;
    // Smoothstep: no hard line in the terrain where the tile density jumps.
    t * t * (3.0 - 2.0 * t)
}

/// Is any part of `key` above the body's own horizon from `cam_body`?
///
/// Exact sphere-tangent test: a point `p` on a sphere of radius `r` centred at
/// the origin is above the horizon seen from `c` iff `p · c ≥ r²`. Every corner
/// (and the centre) is lifted to `r + relief_m` first, so a mountain that
/// legitimately pokes over the limb is never culled — with Thalos's ±9.8 km and
/// Mira's ±13 km that allowance is what keeps the test honest rather than
/// merely cheap.
///
/// Used to gate *refinement only*, never coverage: a tile that fails still
/// enters the leaf set at whatever level it reached, so the sphere stays fully
/// tiled and nothing downstream (the coverage invariant, the height mirror's
/// fallback, the impostor handoff) sees a hole. Cutting them out entirely would
/// save more (46–60 % vs 28–40 % measured) but trades a guarantee for it.
///
/// This is view-*independent* — it depends on the eye's position, not where it
/// looks — so unlike frustum culling it can never pop when the camera turns.
fn above_horizon(key: TileKey, cam_body: DVec3, radius_m: f64, relief_m: f64) -> bool {
    // No relief bound (a provider without metadata) means "assume anything is
    // up there", i.e. refine as the distance rule alone would. Must be an
    // explicit escape: feeding a huge finite allowance through the maths below
    // does *not* degrade to "always visible", it degrades to nonsense.
    if !relief_m.is_finite() {
        return true;
    }
    let cam_len = cam_body.length();
    if cam_len <= radius_m {
        return true;
    }
    let top = radius_m + relief_m;
    // Widest angle from the sub-camera point at which ground lifted to `top`
    // still clears the tangent plane: `p · c ≥ r²` with `|p| = top` becomes
    // `cos θ ≥ r² / (top · |c|)`.
    let cos_max = radius_m * radius_m / (top * cam_len);
    if cos_max <= -1.0 {
        return true;
    }
    let theta_max = cos_max.min(1.0).acos();

    // Bound the tile by its cone, NOT by point samples. A tile at MIN_LEVEL
    // spans 589 km on Thalos, so from 756 m up its corners and centre are all
    // far below the horizon while the ground directly beneath the camera —
    // inside the same tile — is in plain view. Sampling points culled every
    // coarse tile and refinement stopped dead at the MIN_LEVEL shell; the cone
    // is conservative for any tile size.
    let centre = key.center_dir();
    let cam_dir = cam_body / cam_len;
    let theta_c = centre.dot(cam_dir).clamp(-1.0, 1.0).acos();
    let theta_r = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        .iter()
        .map(|&(s, t)| centre.dot(key.dir_at(s, t)).clamp(-1.0, 1.0).acos())
        .fold(0.0f64, f64::max);
    theta_c - theta_r <= theta_max
}

/// A body-fixed patch whose ground the selection must resolve to at least
/// `spacing_m`, **whatever the camera distance rule says**.
///
/// The distance/ruggedness rule answers "how much detail does this framing
/// deserve?", which is the right question for natural terrain and the wrong one
/// for authored ground. A structure pad is a flat plane stamped into the
/// heightfield with a hard edge and a blend ramp; the mesh either resolves that
/// footprint or it reverts to the natural terrain the pad cut away — at the
/// spaceport, 83 m of it (`terrain 537..692 m` levelled to 609 m). Everything
/// draped on the pad is proud of it by centimetres, so an under-resolved tile
/// does not degrade the base gracefully, it swallows it whole.
///
/// So the pad publishes its own resolution requirement and selection honours it
/// as a **floor**. This is deliberately the mirror image of
/// [`tile_ruggedness_weight`], which may only ever take detail *away* from the
/// distance rule: authored ground is the one input allowed to add it back.
///
/// The floor still yields to [`above_horizon`] — a base on the far side of the
/// body earns nothing — which is what keeps it from pinning fine tiles at the
/// antipode for the whole session.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefinementSite {
    /// Unit body-fixed direction to the patch centre.
    pub center_dir: DVec3,
    /// Angular radius (radians) the guarantee covers — the authored footprint
    /// plus whatever blend surrounds it.
    pub angular_radius: f64,
    /// Sample spacing (m/vertex) the ground under the patch must reach.
    pub spacing_m: f64,
}

impl RefinementSite {
    /// Does `key`'s cone overlap this site's cone? Cone-vs-cone, not a point
    /// test, for the same reason [`above_horizon`] bounds tiles by their cone: a
    /// coarse tile spans hundreds of km, so a centre-direction test would miss
    /// every ancestor that actually covers the site — which is exactly the set
    /// the floor exists to catch.
    fn overlaps(&self, key: TileKey) -> bool {
        let centre = key.center_dir();
        let theta_c = centre.dot(self.center_dir).clamp(-1.0, 1.0).acos();
        let theta_r = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
            .iter()
            .map(|&(s, t)| centre.dot(key.dir_at(s, t)).clamp(-1.0, 1.0).acos())
            .fold(0.0f64, f64::max);
        theta_c - theta_r <= self.angular_radius
    }
}

/// Must `key` refine to honour an authored [`RefinementSite`]?
fn below_site_floor(key: TileKey, radius_m: f64, sites: &[RefinementSite]) -> bool {
    let spacing = key.sample_spacing_m(radius_m);
    sites
        .iter()
        .any(|site| spacing > site.spacing_m && site.overlaps(key))
}

/// Pure selection: distance-split descent + cross-face 2:1 balance via
/// direction probes (probe M2/M3).
///
/// `ruggedness` answers "how rough is the terrain under this key" (see
/// [`tile_ruggedness_weight`]); `None` — nothing measured there yet — means
/// the tile refines on the plain [`SPLIT_FACTOR`] rule, so a cold start
/// behaves exactly as the distance-only selection did and sharpens as
/// measurements land.
pub fn select_leaves_with_relief(
    cam_body: DVec3,
    radius_m: f64,
    max_level: u8,
    relief_m: f64,
    ruggedness: &dyn Fn(TileKey) -> Option<f32>,
) -> HashSet<TileKey> {
    select_leaves_scaled(
        cam_body,
        radius_m,
        max_level,
        relief_m,
        ruggedness,
        1.0,
        0.0,
        &[],
    )
}

/// [`select_leaves_with_relief`] with the residency budget's multiplier applied
/// to both split factors.
///
/// `split_scale` ≤ 1.0 shrinks the distance at which every level refines, so the
/// whole selection coarsens proportionally instead of a single band collapsing.
/// Leaf count goes roughly as the square of the factor, so 0.9 is ~20% fewer
/// tiles per step — the brake reaches a 2.5 GiB target from a runaway in a
/// handful of frames. `max_level` and the horizon test are untouched: the budget
/// only moves *where* detail stops, never removes the surface.
///
/// `motion_arc_m` is the motion brake's floor on **child** tile arc
/// (`eye speed × `[`MOTION_CROSS_MIN_S`]): a split producing tiles the eye
/// would cross faster than that is refused. `0.0` disables it.
///
/// `sites` are authored [`RefinementSite`] floors. They outrank both the
/// distance rule and `split_scale` — the brake may make distant mountains softer
/// (nobody can tell), but it may not let the ground swallow a base.
#[allow(clippy::too_many_arguments)]
pub fn select_leaves_scaled(
    cam_body: DVec3,
    radius_m: f64,
    max_level: u8,
    relief_m: f64,
    ruggedness: &dyn Fn(TileKey) -> Option<f32>,
    split_scale: f64,
    motion_arc_m: f64,
    sites: &[RefinementSite],
) -> HashSet<TileKey> {
    let split_scale = split_scale.clamp(MIN_SPLIT_SCALE, 1.0);
    let want_split = |key: TileKey| -> bool {
        if key.level >= max_level {
            return false;
        }
        // Authored floor first: it is unconditional apart from the horizon, so
        // there is nothing the distance rule below could add to the answer.
        if below_site_floor(key, radius_m, sites) {
            return above_horizon(key, cam_body, radius_m, relief_m);
        }
        let arc = tile_arc_m(key.level, radius_m);
        // Motion brake: don't create tiles the moving eye would out-run (see
        // MOTION_CROSS_MIN_S). Placed after the site floor so authored ground
        // resolves regardless of speed, and before the distance rule because
        // it is cheaper than the ruggedness lookup it short-circuits.
        if arc * 0.5 < motion_arc_m {
            return false;
        }
        let d = ((cam_body - key.center_dir() * radius_m).length() - arc * 0.75).max(1.0);
        // The boost only applies while a split still lands above the source's
        // detail floor (see `RUGGED_SPACING_FLOOR_M`).
        let child_spacing = key.sample_spacing_m(radius_m) * 0.5;
        let w = if child_spacing >= RUGGED_SPACING_FLOOR_M {
            ruggedness(key).map_or(0.0, tile_ruggedness_weight)
        } else {
            0.0
        };
        let factor = (SPLIT_FACTOR + (SPLIT_FACTOR_RUGGED - SPLIT_FACTOR) * w) * split_scale;
        if d >= factor * arc {
            return false;
        }
        // Cheapest test last: only worth paying once distance has already said
        // yes. Ground the eye cannot possibly see does not earn refinement.
        above_horizon(key, cam_body, radius_m, relief_m)
    };

    let mut leaves: HashSet<TileKey> = HashSet::new();
    let n = 1u32 << MIN_LEVEL;
    let mut stack: Vec<TileKey> = Vec::with_capacity(1024);
    for face in 0..6u8 {
        for y in 0..n {
            for x in 0..n {
                stack.push(TileKey {
                    face,
                    level: MIN_LEVEL,
                    x,
                    y,
                });
            }
        }
    }
    while let Some(key) = stack.pop() {
        if want_split(key) {
            stack.extend(key.children());
        } else {
            leaves.insert(key);
        }
    }

    const PROBES: [(f64, f64); 4] = [(-0.5, 0.5), (1.5, 0.5), (0.5, -0.5), (0.5, 1.5)];
    for _ in 0..16 {
        let mut to_split: HashSet<TileKey> = HashSet::new();
        for &leaf in &leaves {
            if leaf.level <= MIN_LEVEL + 1 {
                continue;
            }
            let coarse = leaf.level - 2;
            for (s, t) in PROBES {
                let mut probe = TileKey::containing_dir(leaf.dir_at(s, t), leaf.level);
                if probe == leaf {
                    continue;
                }
                loop {
                    if leaves.contains(&probe) {
                        if probe.level <= coarse {
                            to_split.insert(probe);
                        }
                        break;
                    }
                    match probe.parent() {
                        Some(p) if p.level >= MIN_LEVEL => probe = p,
                        _ => break,
                    }
                }
            }
        }
        if to_split.is_empty() {
            break;
        }
        for key in to_split {
            leaves.remove(&key);
            for child in key.children() {
                leaves.insert(child);
            }
        }
    }
    leaves
}

/// [`select_leaves_with_relief`] with no relief knowledge and no horizon
/// allowance bound — the plain distance rule. Kept as the pure form the
/// selection tests drive.
pub fn select_leaves(cam_body: DVec3, radius_m: f64, max_level: u8) -> HashSet<TileKey> {
    select_leaves_with_relief(cam_body, radius_m, max_level, f64::INFINITY, &|_| None)
}

/// Is `key`'s whole footprint covered by resident tiles (itself, or recursively
/// by its children at any mix of levels)? Short-circuits at each resident node,
/// so the traversal is bounded by the resident set, not by 4^depth.
fn covered_by_resident(key: TileKey, resident: &HashMap<TileKey, Entity>, max_level: u8) -> bool {
    if resident.contains_key(&key) {
        return true;
    }
    if key.level >= max_level {
        return false;
    }
    key.children()
        .into_iter()
        .all(|child| covered_by_resident(child, resident, max_level))
}

/// Bridge requests: children of stale resident ancestors whose replacement
/// gap spans more than one level. Only desired tiles are generated otherwise,
/// so a fast approach (selection skipping levels) left a coarse parent
/// waiting for its ENTIRE deep leaf set — up to 4^gap tiles — while its
/// geometry poked through the refined terrain. Bridges make replacement
/// cascade level-by-level: each step releases on just 4 landings, and the
/// visible overlap never spans more than one level (a divergence small
/// enough for the per-level depth bias to hide). Pyramid overhead ≤ ~1/3.
fn bridge_requests(
    desired: &HashSet<TileKey>,
    resident: &HashMap<TileKey, Entity>,
) -> HashSet<TileKey> {
    let mut bridges = HashSet::new();
    for k in desired {
        // Nearest resident strict ancestor decides the current gap.
        let mut probe = *k;
        while let Some(p) = probe.parent() {
            if p.level < MIN_LEVEL {
                break;
            }
            probe = p;
            if resident.contains_key(&probe) {
                if k.level > probe.level + 1 {
                    for child in probe.children() {
                        if !resident.contains_key(&child) && !desired.contains(&child) {
                            bridges.insert(child);
                        }
                    }
                }
                break;
            }
        }
    }
    bridges
}

/// The despawn decision + retirement bookkeeping, extracted pure so the
/// descent simulation test can drive the exact production logic. Returns the
/// keys whose grace gates expired this tick (the caller despawns them).
fn despawn_ready(
    desired: &HashSet<TileKey>,
    resident: &HashMap<TileKey, Entity>,
    retiring: &mut HashMap<TileKey, (f32, u16)>,
    max_level: u8,
    dt: f32,
) -> Vec<TileKey> {
    let removable: HashSet<TileKey> = resident
        .keys()
        .filter(|k| !desired.contains(k))
        .filter(|k| {
            let mut probe = **k;
            loop {
                match probe.parent() {
                    Some(p) if p.level >= MIN_LEVEL => {
                        probe = p;
                        if desired.contains(&probe) {
                            // Merge case: coverage passes to the ancestor.
                            return resident.contains_key(&probe);
                        }
                    }
                    _ => break,
                }
            }
            // Split case: coverage passes to resident DESCENDANTS. Descend
            // from the children — `covered_by_resident(k)` would see k
            // itself (still in the resident map while we decide its removal)
            // and trivially certify the tile with ITSELF. That self-
            // certificate was the black-tile bug: any stale tile above the
            // desired level "covered" itself, retired, and despawned after
            // grace, abandoning its still-pending children (reproduced by
            // `streaming_tests::descent_keeps_every_desired_tile_covered`).
            if k.level >= max_level {
                return false;
            }
            k.children()
                .into_iter()
                .all(|child| covered_by_resident(child, resident, max_level))
        })
        .copied()
        .collect();
    retiring.retain(|key, _| removable.contains(key));
    let mut expired = Vec::new();
    for key in removable {
        let (secs, frames) = retiring
            .entry(key)
            .or_insert((DESPAWN_GRACE_S, DESPAWN_GRACE_FRAMES));
        if *secs > 0.0 || *frames > 0 {
            *secs -= dt;
            *frames = frames.saturating_sub(1);
            continue;
        }
        retiring.remove(&key);
        expired.push(key);
    }
    expired
}

/// Desired tiles whose footprint is unserved by the resident set (no resident
/// self, ancestor, or descendant cover) — the coverage invariant. Shared by
/// the runtime audit and the descent simulation test.
fn uncovered_desired(
    desired: &HashSet<TileKey>,
    resident: &HashMap<TileKey, Entity>,
    max_level: u8,
) -> Vec<TileKey> {
    desired
        .iter()
        .filter(|k| {
            if resident.contains_key(*k) {
                return false;
            }
            let mut probe = **k;
            loop {
                match probe.parent() {
                    Some(p) if p.level >= MIN_LEVEL => {
                        probe = p;
                        if resident.contains_key(&probe) {
                            return false;
                        }
                    }
                    _ => break,
                }
            }
            !covered_by_resident(**k, resident, max_level)
        })
        .copied()
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn stream_tile_terrain(
    eye: Res<TileEye>,
    time: Res<Time>,
    mut roots: Query<&mut TileTerrainRoot>,
    big_space: Query<(Entity, &Grid), With<BigSpace>>,
    mut placed: Query<(&TileBodyOrigin, &mut CellCoord, &mut Transform)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut commands: Commands,
) {
    let Some(target) = &eye.target else {
        return;
    };
    // Tiles hang off the big_space ROOT, not the body's rotating grid — see
    // [`TileBodyOrigin`] for why that distinction is a decimetre of ground.
    let Ok((root_entity, grid)) = big_space.single() else {
        return;
    };
    let body_rotation = target.body_orientation.as_quat();
    let Ok(mut root) = roots.get_mut(target.root) else {
        return;
    };
    let cam = target.cam_body;
    let radius = root.radius_m;
    let max_level = root.max_level;
    let root_ref = &mut *root;

    // Residency budget: steer the split scale from the VRAM already committed
    // *before* selecting, so a frame that is over budget selects a coarser set
    // rather than committing more first (INC-20260725T012104Z-tile-residency-had-no-budget).
    let budget_bytes = residency_budget_bytes();
    root_ref.update_split_scale(budget_bytes);

    // Selection reads the measured-ruggedness memo, so mountains hold their
    // mesh out to `SPLIT_FACTOR_RUGGED` while water and plains stay on the
    // plain `SPLIT_FACTOR` rule, and skips refinement below the body's own
    // horizon. Split the borrow: `ruggedness_at` walks `&self` while `desired`
    // is being written.
    let motion_arc_m = target.speed_m_s.max(0.0) * MOTION_CROSS_MIN_S;
    let desired_now = {
        let known: &TileTerrainRoot = root_ref;
        select_leaves_scaled(
            cam,
            radius,
            max_level,
            known.relief_m,
            &|key| known.ruggedness_at(key),
            known.split_scale,
            motion_arc_m,
            &known.refinement_sites,
        )
    };
    root_ref.desired = desired_now;

    // Bridge tiles keep multi-level replacement progressive (see
    // `bridge_requests`).
    let bridges = bridge_requests(&root_ref.desired, &root_ref.resident);

    // Cancel pending tiles nobody wants (task drop aborts).
    let desired = &root_ref.desired;
    root_ref
        .pending
        .retain(|key, _| desired.contains(key) || bridges.contains(key));

    // Admit missing (desired + bridges), screen-size-priority (distance /
    // tile size — absolute nearest-first starves coarse merge-targets; probe
    // M3 finding).
    let mut missing: Vec<TileKey> = root_ref
        .desired
        .iter()
        .chain(bridges.iter())
        .filter(|k| !root_ref.resident.contains_key(k) && !root_ref.pending.contains_key(k))
        .copied()
        .collect();
    missing.sort_by(|a, b| {
        let pa = (cam - a.center_dir() * radius).length() / tile_arc_m(a.level, radius);
        let pb = (cam - b.center_dir() * radius).length() / tile_arc_m(b.level, radius);
        pa.total_cmp(&pb)
    });
    let budget = MAX_IN_FLIGHT.saturating_sub(root_ref.pending.len());
    // Dedicated pool: routing this through AsyncComputeTaskPool starves
    // Avian's collider-tree optimisation and hitches the main thread (the
    // documented reason `ground::tile_synthesis_pool` exists).
    let pool = crate::ground::tile_synthesis_pool::tile_synthesis_pool();
    for key in missing.into_iter().take(budget) {
        let provider = root_ref.provider.clone();
        // The skirt curtain's floor — per-body constant, cached at root
        // construction from the provider's height envelope.
        let relief_m = root_ref.relief_m;
        let task = pool.spawn(async move {
            let started = std::time::Instant::now();
            let tile = provider.request(key, radius);
            let h_range = tile
                .heights_m
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &h| {
                    (lo.min(h), hi.max(h))
                });
            let built = build_tile_mesh(&tile, radius, relief_m);
            StreamedTile {
                key,
                built,
                gen_micros: started.elapsed().as_micros() as u64,
                h_range,
                heights_m: Arc::new(tile.heights_m),
            }
        });
        root_ref.pending.insert(key, task);
    }

    // Land finished tiles as co-rotating children of the body grid.
    let mut landed: Vec<StreamedTile> = Vec::new();
    root_ref.pending.retain(|_, task| {
        if let Some(done) = block_on(poll_once(task)) {
            landed.push(done);
            false
        } else {
            true
        }
    });
    for done in landed {
        root_ref
            .gen_stats
            .record(done.gen_micros, done.h_range, done.built.mesh_h);
        // Feed the relief-aware split rule. `h_range` is the provider's own
        // sampled spread over this tile, so the measurement costs nothing
        // beyond the generation that already happened.
        if done.h_range.0.is_finite() && done.h_range.1.is_finite() {
            let relief = done.h_range.1 - done.h_range.0;
            let arc = tile_arc_m(done.key.level, radius) as f32;
            root_ref.ruggedness.insert(done.key, relief / arc);
        }
        let (cell, local) = place_body_point(done.built.origin, target, grid);
        let mesh_handle = meshes.add(done.built.mesh);
        let mut tile = commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(root_ref.material.clone()),
            // Rotation carries the body's spin, so it acts only on the
            // in-tile vertex offsets; the multi-Mm part of the placement is
            // already resolved in f64 above.
            Transform::from_translation(local).with_rotation(body_rotation),
            cell,
            TileBodyOrigin(done.built.origin),
            root_ref.render_layers.clone(),
            ChildOf(root_entity),
        ));
        // Shadow-caster child (see `TileTerrainRoot::caster`): same mesh, cheap
        // material, cascade layer only. Identity transform — it inherits the
        // tile's placement — and it despawns with the tile (recursive despawn).
        if let Some((caster_mat, caster_layers)) = &root_ref.caster {
            tile.with_child((
                Mesh3d(mesh_handle),
                MeshMaterial3d(caster_mat.clone()),
                Transform::IDENTITY,
                caster_layers.clone(),
            ));
        }
        let entity = tile.id();
        root_ref.resident.insert(done.key, entity);
        // Publish the heights this tile was meshed from, so every ground
        // consumer seats on the geometry that is actually drawn.
        if let Ok(mut mirror) = root_ref.height_mirror.write() {
            mirror.insert(done.key, done.heights_m);
        }
    }

    // Re-place every resident tile from this frame's f64 body pose. The body
    // spins, so this is not optional bookkeeping — it is what keeps the ground
    // welded to everything built on it. Doing it here rather than in a system of
    // its own means one pose value serves both the tiles that landed this frame
    // and the ones already up, so the two can never be a frame apart.
    for (origin, mut cell, mut transform) in &mut placed {
        let (next_cell, local) = place_body_point(origin.0, target, grid);
        if *cell != next_cell {
            *cell = next_cell;
        }
        transform.translation = local;
        transform.rotation = body_rotation;
    }

    if !root_ref.covered_once
        && !root_ref.desired.is_empty()
        && root_ref
            .desired
            .iter()
            .all(|k| root_ref.resident.contains_key(k))
    {
        root_ref.covered_once = true;
        info!(
            "tile terrain: first full coverage ({} tiles) — impostor handoff ready",
            root_ref.resident.len()
        );
    }

    // Hole-free despawn (probe M2 rule), with PROGRESSIVE replacement on
    // split: a no-longer-desired tile is removable the moment every point it
    // covers is served by SOME resident tile —
    //  - merge case (a desired ancestor exists): that ancestor must be
    //    resident;
    //  - split case: each child subtree must be covered by resident tiles at
    //    ANY level, not only the final desired leaves. Waiting for the full
    //    leaf set (up to 4^n descendants) kept coarse blocky geometry poking
    //    through refined crater walls long after its children had landed
    //    (user finding 2026-07-24); releasing per covered level makes
    //    refinement read as gradual sharpening instead of a late swap.
    // Lax retirement: a removable tile only despawns after its condition has
    // held continuously through BOTH grace gates. Overlap while lingering is
    // rendered correctly by the per-level render lift (finer wins — see
    // `LEVEL_RENDER_LIFT_M`), so there is no pressure to despawn promptly.
    // Decision logic lives in `despawn_ready` (pure — exercised by the descent
    // simulation test).
    let dt = time.delta_secs();
    let now = time.elapsed_secs_f64();
    let expired = despawn_ready(
        &root_ref.desired,
        &root_ref.resident,
        &mut root_ref.retiring,
        root_ref.max_level,
        dt,
    );
    for key in expired {
        if let Some(entity) = root_ref.resident.remove(&key) {
            commands.entity(entity).despawn();
            if let Ok(mut mirror) = root_ref.height_mirror.write() {
                mirror.remove(key);
            }
            // Forensics: record the despawn + which certificate held at this
            // final revalidation frame (merge = desired ancestor resident).
            let merge_case = {
                let mut probe = key;
                let mut merge = false;
                loop {
                    match probe.parent() {
                        Some(p) if p.level >= MIN_LEVEL => {
                            probe = p;
                            if root_ref.desired.contains(&probe) {
                                merge = true;
                                break;
                            }
                        }
                        _ => break,
                    }
                }
                merge
            };
            if root_ref.recent_despawns.len() >= 512 {
                root_ref.recent_despawns.remove(0);
            }
            root_ref.recent_despawns.push((key, now, merge_case));
        }
    }

    // Coverage invariant check (cheap, every ~2 s): every desired tile's
    // footprint must be served by the resident set — itself, a resident
    // ancestor, or resident descendants. If this ever fires, holes are
    // LOGICAL (a selection/despawn bug); if the screen shows holes while
    // this stays silent, the problem is presentation-side (upload, culling,
    // material) — the discriminator for the black-tile reports.
    // Residency gauge. The generation-stats line next to it counts *cumulative*
    // landings, which cannot answer "how much VRAM is the ground holding now" —
    // the question the OOM in INC-20260725T012104Z had no data for, because a
    // 19,200-landing log is equally consistent with a 1 GiB working set and a
    // 6 GiB one. This is periodic rather than per-landing so a settled scene
    // still reports its footprint.
    root_ref.gauge_countdown -= dt;
    if root_ref.gauge_countdown <= 0.0 {
        root_ref.gauge_countdown = GAUGE_INTERVAL_S;
        let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
        let budget_note = if budget_bytes == usize::MAX {
            "off".to_string()
        } else {
            format!("{:.0} MiB", mib(budget_bytes))
        };
        // `instances` is load-bearing when reading this line back: a budget that
        // suddenly halves is a peer starting, not the brake misbehaving.
        info!(
            target: "thalos::diagnostic::tile_terrain",
            event = "residency_gauge",
            resident = root_ref.resident.len(),
            resident_mib = mib(root_ref.resident_bytes()),
            pending = root_ref.pending.len(),
            desired = root_ref.desired.len(),
            retiring = root_ref.retiring.len(),
            budget = %budget_note,
            instances = vram_share::live_instances(),
            split_scale = root_ref.split_scale,
            eye_speed_m_s = target.speed_m_s,
            motion_arc_m,
            "tile residency gauge"
        );
    }

    root_ref.coverage_check_countdown -= dt;
    if root_ref.coverage_check_countdown <= 0.0 {
        root_ref.coverage_check_countdown = 2.0;
        let uncovered =
            uncovered_desired(&root_ref.desired, &root_ref.resident, root_ref.max_level);
        if !uncovered.is_empty() && root_ref.covered_once {
            warn!(
                "tile terrain: {} desired tiles LOGICALLY uncovered (of {}) — despawn/selection bug, not presentation",
                uncovered.len(),
                root_ref.desired.len()
            );
            // Forensics for the first few: the full ancestor-chain state, and
            // any recent despawn among the chain (with age + certificate).
            for key in uncovered.iter().take(3) {
                let mut chain = format!("{key:?}: self[");
                chain.push_str(if root_ref.pending.contains_key(key) {
                    "pending"
                } else {
                    "absent"
                });
                chain.push(']');
                let mut probe = *key;
                while let Some(p) = probe.parent() {
                    if p.level < MIN_LEVEL {
                        break;
                    }
                    probe = p;
                    let state = if root_ref.resident.contains_key(&probe) {
                        "R".to_string()
                    } else if root_ref.pending.contains_key(&probe) {
                        "pend".to_string()
                    } else if let Some((_, stamp, merge)) = root_ref
                        .recent_despawns
                        .iter()
                        .rev()
                        .find(|(k, _, _)| *k == probe)
                    {
                        format!(
                            "despawned {:.2}s ago ({})",
                            now - stamp,
                            if *merge { "merge" } else { "split" }
                        )
                    } else {
                        "-".to_string()
                    };
                    chain.push_str(&format!(" L{}[{}]", probe.level, state));
                }
                warn!("tile terrain: uncovered chain {chain}");
            }
        }
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    const R: f64 = 3_186_000.0;
    const RELIEF: f64 = 9_797.6;

    fn flat_tile(key: TileKey) -> SurfaceTile {
        let n = SurfaceTile::grid_side().pow(2);
        SurfaceTile {
            key,
            sample_spacing_m: key.sample_spacing_m(R),
            heights_m: vec![0.0; n],
            albedo_linear: vec![[0.5, 0.4, 0.3]; n],
            bands: vec![[0.0, 0.0]; n],
        }
    }

    /// The budget's denominator must equal what the mesher actually uploads.
    /// A new vertex attribute (NTR-RT1 wants TANGENT) silently under-counts
    /// VRAM otherwise, and an under-counting budget is worse than none: it
    /// reports headroom that does not exist.
    #[test]
    fn tile_mesh_bytes_matches_the_built_mesh() {
        let built = build_tile_mesh(
            &flat_tile(TileKey {
                face: 0,
                level: 6,
                x: 3,
                y: 5,
            }),
            R,
            RELIEF,
        );
        let vertex_bytes = built.mesh.get_vertex_buffer_size();
        let index_bytes = built
            .mesh
            .get_index_buffer_bytes()
            .expect("tile meshes are indexed")
            .len();
        assert_eq!(
            vertex_bytes + index_bytes,
            TILE_MESH_BYTES,
            "TILE_MESH_BYTES ({TILE_MESH_BYTES}) disagrees with the mesh \
             ({vertex_bytes} vertex + {index_bytes} index) — an attribute changed"
        );
    }

    /// A real tile mesh — not a toy grid — converts into something the
    /// raytracing scene will actually accept, and costs what the budget says.
    ///
    /// This is the test that would have caught the whole premise being wrong:
    /// the visible tile mesh carries COLOR and UV_1, so it can never pass
    /// Solari's gate itself, and a proxy "sharing the mesh handle" would have
    /// been silently skipped with no ground in any reflection.
    #[test]
    fn a_real_tile_converts_into_an_rt_twin() {
        let built = build_tile_mesh(
            &flat_tile(TileKey {
                face: 0,
                level: 6,
                x: 3,
                y: 5,
            }),
            R,
            RELIEF,
        );
        assert!(
            !crate::rt::is_raytracing_eligible(&built.mesh),
            "the visible mesh must NOT be RT-eligible — if it becomes so, the \
             separate RT twin is dead weight and this design should change"
        );

        let twin = crate::rt::rt_twin_of_tile(&built.mesh).expect("tile mesh converts");
        assert!(
            crate::rt::is_raytracing_eligible(&twin),
            "attributes {:?}",
            crate::rt::attribute_names(&twin)
        );

        let vertex_bytes = twin.get_vertex_buffer_size();
        let index_bytes = twin
            .get_index_buffer_bytes()
            .expect("RT twins are indexed")
            .len();
        assert_eq!(
            vertex_bytes + index_bytes,
            RT_TILE_MESH_BYTES,
            "RT_TILE_MESH_BYTES ({RT_TILE_MESH_BYTES}) disagrees with the built \
             twin ({vertex_bytes} vertex + {index_bytes} index)"
        );
        // The twin is geometry-identical to the raster mesh: a reflection can
        // never disagree with the ground under it about where the ground is.
        assert_eq!(twin.count_vertices(), built.mesh.count_vertices());

        // Pin the figures the RT_TILE_MESH_BYTES docs quote and NTR-RT1's
        // budget reasons from, so prose and code cannot drift apart.
        assert_eq!(RT_TILE_MESH_BYTES / 1024, 312, "documented as 312 KiB");
        let ratio = (TILE_MESH_BYTES + RT_TILE_MESH_BYTES) as f64 / TILE_MESH_BYTES as f64;
        assert!(
            (ratio - 1.90).abs() < 0.005,
            "documented as 1.90x, computed {ratio:.3}"
        );
    }

    /// Coarsening is proportional and bounded: a lower scale always costs fewer
    /// tiles, and even the floor keeps a complete surface (never fewer leaves
    /// than the MIN_LEVEL shell, which is the whole sphere).
    #[test]
    fn split_scale_trades_tiles_for_coverage_not_ground() {
        let max_level = max_level_for(R);
        let cam = DVec3::new(0.31, 0.72, 0.62).normalize() * (R + 2_000.0);
        let shell = 6 * (1usize << MIN_LEVEL).pow(2);
        let mut previous = usize::MAX;
        for scale in [1.0, 0.8, 0.6, MIN_SPLIT_SCALE] {
            let leaves =
                select_leaves_scaled(cam, R, max_level, RELIEF, &|_| None, scale, 0.0, &[]);
            assert!(
                leaves.len() < previous,
                "scale {scale} did not reduce the working set ({} vs {previous})",
                leaves.len()
            );
            assert!(
                leaves.len() >= shell,
                "scale {scale} left the sphere uncovered ({} leaves < {shell})",
                leaves.len()
            );
            previous = leaves.len();
        }
    }

    /// The motion brake coarsens proportionally with speed, never uncovers the
    /// sphere, never creates a tile finer than the eye could keep for
    /// [`MOTION_CROSS_MIN_S`] — and an authored site floor still resolves at
    /// any speed (a base must not be swallowed because the camera flew over).
    #[test]
    fn motion_brake_coarsens_but_never_beats_a_site_floor() {
        let max_level = max_level_for(R);
        let cam = DVec3::new(0.31, 0.72, 0.62).normalize() * (R + 2_000.0);
        let shell = 6 * (1usize << MIN_LEVEL).pow(2);
        let mut previous =
            select_leaves_scaled(cam, R, max_level, RELIEF, &|_| None, 1.0, 0.0, &[]).len() + 1;
        for speed in [50.0, 200.0, 1_000.0] {
            let motion_arc = speed * MOTION_CROSS_MIN_S;
            let leaves =
                select_leaves_scaled(cam, R, max_level, RELIEF, &|_| None, 1.0, motion_arc, &[]);
            assert!(
                leaves.len() <= previous,
                "speed {speed} m/s grew the working set ({} vs {previous})",
                leaves.len()
            );
            assert!(
                leaves.len() >= shell,
                "speed {speed} m/s left the sphere uncovered ({} leaves < {shell})",
                leaves.len()
            );
            let min_arc = leaves
                .iter()
                .map(|k| tile_arc_m(k.level, R))
                .fold(f64::INFINITY, f64::min);
            assert!(
                min_arc >= motion_arc * 0.999,
                "speed {speed} m/s still produced a {min_arc:.0} m tile the eye \
                 crosses in under {MOTION_CROSS_MIN_S} s"
            );
            previous = leaves.len();
        }

        let site = RefinementSite {
            center_dir: cam.normalize(),
            angular_radius: 500.0 / R,
            spacing_m: 30.0,
        };
        let braked =
            select_leaves_scaled(cam, R, max_level, RELIEF, &|_| None, 1.0, 5_000.0, &[site]);
        let finest_at_site = braked
            .iter()
            .filter(|k| site.overlaps(**k))
            .map(|k| k.sample_spacing_m(R))
            .fold(f64::INFINITY, f64::min);
        assert!(
            finest_at_site <= site.spacing_m,
            "a 5 km/s motion brake coarsened the site floor to {finest_at_site:.0} m/vertex"
        );
    }

    /// Placing a tile through an **f32** body rotation misses by more than the
    /// paving is lifted; placing it in f64 does not.
    ///
    /// This is the arithmetic behind [`TileBodyOrigin`], kept executable because
    /// the failure it guards is invisible in a still and unremarkable in code
    /// review: "parent the tiles to the body's grid" reads like the obviously
    /// right thing and costs a decimetre of ground (INC-20260725T195500Z).
    #[test]
    fn f32_body_rotation_cannot_place_ground_under_a_runway() {
        // A point on the surface, and the body turning through its day.
        let p_body = DVec3::new(0.31, 0.13, 0.94).normalize() * R;
        let axis = DVec3::Y;
        const PAVING_LIFT_M: f64 = 0.12;

        let mut worst_f32 = 0.0f64;
        let mut worst_f64 = 0.0f64;
        for step in 0..2_000 {
            let angle = step as f64 * core::f64::consts::TAU / 2_000.0;
            let exact = DQuat::from_axis_angle(axis, angle);
            // What a grid-parented tile gets: the same rotation, quantised to
            // the f32 `Transform.rotation` big_space propagates through.
            let via_f32 = exact.as_quat().normalize().as_dquat();
            worst_f32 = worst_f32.max((via_f32 * p_body - exact * p_body).length());
            // What this renderer does instead: f64 all the way to the grid
            // split, with f32 acting only on in-tile vertex offsets.
            let offset = DVec3::new(120.0, -40.0, 300.0); // a near-surface vertex
            let f64_path = exact * p_body + (exact.as_quat() * offset.as_vec3()).as_dvec3();
            worst_f64 = worst_f64.max((f64_path - exact * (p_body + offset)).length());
        }
        assert!(
            worst_f32 > PAVING_LIFT_M,
            "f32 body rotation only strayed {worst_f32:.3} m — if this stops \
             exceeding the {PAVING_LIFT_M} m paving lift the tile placement could \
             be simplified back to grid parenting"
        );
        assert!(
            worst_f64 < 0.001,
            "f64 placement strayed {worst_f64:.4} m — the precision this renderer \
             depends on is gone"
        );
    }

    /// The meshed ground over a structure pad lands on the pad's plane, to a
    /// tolerance far under the 0.12 m the paving is lifted by.
    ///
    /// Runs the real chain — `FlattenedSurface` → `SurfaceQueryProvider` →
    /// [`build_tile_mesh`] — over the runway basin's actual geometry, at every
    /// level the pad floor admits, and measures each vertex against the tangent
    /// plane analytically. This is the falsifier that separates "the ground is
    /// in the wrong place" from "the thing drawn on the ground is": if paving
    /// z-fights and this passes, the fault is not on the terrain side.
    #[test]
    fn meshed_ground_lands_on_the_pad_plane() {
        use thalos_terrain::{FlattenRegion, FlattenedSurface, TerrainFlatten, flatten_handle};

        /// Rolling synthetic terrain — the pad has to cut real relief for the
        /// test to mean anything (the spaceport basin removes 83 m).
        struct RollingSurface;
        impl SurfaceQuery for RollingSurface {
            fn sample(&self, dir: Vec3, _lod_m: f32) -> thalos_terrain::query::SurfaceSample {
                let d = dir.as_dvec3();
                let h = 600.0 + 90.0 * (d.x * 900.0).sin() * (d.z * 700.0).cos();
                thalos_terrain::query::SurfaceSample {
                    height_m: h as f32,
                    albedo_linear: Vec3::splat(0.4),
                    roughness: 0.9,
                    moisture: 0.0,
                }
            }
            fn radius_m(&self) -> f32 {
                R as f32
            }
            fn height_range_m(&self) -> f32 {
                RELIEF as f32
            }
        }

        // The runway basin, as `finish_runway_spawn` installs it.
        let center_dir = DVec3::new(0.31, 0.13, 0.94).normalize();
        let along = center_dir.cross(DVec3::Y).normalize();
        let across = center_dir.cross(along).normalize();
        const ELEVATION_M: f64 = 609.0;
        let pad = TerrainFlatten::new(
            center_dir,
            along,
            across,
            3_500.0,
            2_500.0,
            500.0,
            ELEVATION_M,
            R,
        );
        let handle = flatten_handle();
        handle.write().expect("fresh handle").push(FlattenRegion {
            id: 1,
            flatten: pad,
        });

        let provider = SurfaceQueryProvider {
            surface: Arc::new(FlattenedSurface::new(Arc::new(RollingSurface), handle)),
        };

        // Every level from the pad floor (10 on Thalos) up to the deepest.
        for level in 10..=max_level_for(R) {
            let key = TileKey::containing_dir(center_dir, level);
            let built = build_tile_mesh(&provider.request(key, R), R, RELIEF);
            let lift = level as f64 * LEVEL_RENDER_LIFT_M;
            let positions = built
                .mesh
                .attribute(Mesh::ATTRIBUTE_POSITION)
                .and_then(|a| a.as_float3())
                .expect("tile meshes carry positions")
                .to_vec();
            let mut worst = 0.0f64;
            // Core grid only — the skirt ring that follows it is *supposed* to
            // hang below the surface (it exists to hide inter-level seams).
            for p in positions.into_iter().take(TILE_RES * TILE_RES) {
                let world = built.origin + DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
                let dir = world.normalize();
                // Only the levelled interior is under test; the ramp is
                // *supposed* to leave the plane.
                if pad.weight(dir) < 1.0 {
                    continue;
                }
                // Distance from the tangent plane the pad levels to, which is
                // `p · center_dir = R + elevation` by construction.
                let off = (world.dot(center_dir) - (R + ELEVATION_M + lift)).abs();
                worst = worst.max(off);
            }
            assert!(
                worst < 0.01,
                "level {level}: meshed ground strays {worst:.4} m from the pad plane \
                 (paving sits 0.12 m over it)"
            );
        }
    }

    /// A structure pad's ground holds its resolution at **every** camera
    /// distance and under the residency brake at its floor.
    ///
    /// This is the regression probe for INC-20260725T191500Z: the spaceport
    /// paving flickered in and out as the craft moved, because selection
    /// coarsened the basin until the mesh cut straight across the levelled
    /// footprint and the natural terrain the pad removed (83 m of it) drew back
    /// over paving standing 0.12 m proud. The distance sweep is the point — the
    /// bug never appeared at one framing, only across a move.
    #[test]
    fn pad_sites_hold_their_resolution_at_every_distance() {
        let max_level = max_level_for(R);
        let pad_dir = DVec3::new(0.31, 0.13, 0.94).normalize();
        // The runway basin: ~4.3 km half-diagonal + a 500 m ramp, resolved four
        // samples across that ramp (the game driver's `PAD_RAMP_SAMPLES`).
        let site = RefinementSite {
            center_dir: pad_dir,
            angular_radius: (4_800.0f64 / R).atan(),
            spacing_m: 125.0,
        };
        // Straight up over the pad, so it is above the horizon throughout —
        // from a low pass to well out of the atmosphere.
        for altitude in [500.0, 2_000.0, 8_000.0, 30_000.0, 120_000.0, 600_000.0] {
            let cam = pad_dir * (R + altitude);
            for scale in [1.0, MIN_SPLIT_SCALE] {
                let leaves =
                    select_leaves_scaled(cam, R, max_level, RELIEF, &|_| None, scale, 0.0, &[site]);
                let worst = leaves
                    .iter()
                    .filter(|k| site.overlaps(**k))
                    .map(|k| k.sample_spacing_m(R))
                    .fold(0.0f64, f64::max);
                assert!(
                    worst <= site.spacing_m,
                    "at {altitude} m (scale {scale}) the pad was meshed at {worst:.0} m/vertex, \
                     coarser than the {} m floor its ramp needs",
                    site.spacing_m
                );
            }
        }
    }

    /// The floor is a floor, not a licence: ground with no pad over it must
    /// select exactly as it did before sites existed.
    #[test]
    fn pad_sites_leave_the_rest_of_the_body_alone() {
        let max_level = max_level_for(R);
        let pad_dir = DVec3::new(0.31, 0.13, 0.94).normalize();
        let site = RefinementSite {
            center_dir: pad_dir,
            angular_radius: (4_800.0f64 / R).atan(),
            spacing_m: 125.0,
        };
        // And it stays cheap. Measured: the floor costs *nothing* below ~30 km
        // (the distance rule is already finer than 125 m there), +33 leaves from
        // a 200 km orbit and +90 from 2,000 km — the pad's own tiles plus the
        // 2:1 balance cascade down to them. At 347 KiB each that peak is 31 MiB,
        // under 1 % of the residency budget. If this bound ever fails, the floor
        // has started buying detail the distance rule should be buying.
        for altitude in [200_000.0, 2_000_000.0] {
            let cam = pad_dir * (R + altitude);
            let plain = select_leaves_scaled(cam, R, max_level, RELIEF, &|_| None, 1.0, 0.0, &[]);
            let with_site =
                select_leaves_scaled(cam, R, max_level, RELIEF, &|_| None, 1.0, 0.0, &[site]);
            let extra = with_site.len().saturating_sub(plain.len());
            assert!(
                extra <= 128,
                "pad floor cost {extra} extra leaves at {altitude} m \
                 ({} → {})",
                plain.len(),
                with_site.len()
            );
        }
        let cam = pad_dir * (R + 30_000.0);
        let plain = select_leaves_scaled(cam, R, max_level, RELIEF, &|_| None, 1.0, 0.0, &[]);
        let with_site =
            select_leaves_scaled(cam, R, max_level, RELIEF, &|_| None, 1.0, 0.0, &[site]);
        // Every leaf the plain rule chose is still a leaf unless the site (or
        // the 2:1 balance around it) refined it — i.e. unless its descendants
        // are present instead.
        for key in plain.difference(&with_site) {
            assert!(
                with_site.iter().any(|k| k.level > key.level && {
                    let mut probe = *k;
                    loop {
                        match probe.parent() {
                            Some(p) if p.level >= key.level => {
                                probe = p;
                                if probe == *key {
                                    break true;
                                }
                            }
                            _ => break false,
                        }
                    }
                }),
                "{key:?} vanished from the selection without being refined"
            );
        }
    }

    /// The controller brakes while over budget and recovers once under it —
    /// and the deadband means it does neither in between (no LOD pumping).
    #[test]
    fn budget_controller_brakes_then_recovers() {
        let budget = 64 * TILE_MESH_BYTES;
        let mut root = TileTerrainRoot::new(
            R,
            Arc::new(NullProvider),
            Handle::default(),
            RenderLayers::default(),
        );

        // Over budget: brake, all the way down to the floor but no further.
        root.resident = (0..200)
            .map(|i| {
                (
                    TileKey {
                        face: 0,
                        level: 6,
                        x: i,
                        y: 0,
                    },
                    Entity::PLACEHOLDER,
                )
            })
            .collect();
        for _ in 0..200 {
            root.update_split_scale(budget);
        }
        assert!(
            (root.split_scale() - MIN_SPLIT_SCALE).abs() < 1e-9,
            "brake stopped at {} instead of the floor",
            root.split_scale()
        );

        // Inside the deadband (between the recover fraction and the budget):
        // hold, so the scale does not oscillate frame to frame.
        root.resident = (0..58)
            .map(|i| {
                (
                    TileKey {
                        face: 0,
                        level: 6,
                        x: i,
                        y: 0,
                    },
                    Entity::PLACEHOLDER,
                )
            })
            .collect();
        let held = root.split_scale();
        for _ in 0..50 {
            root.update_split_scale(budget);
        }
        assert_eq!(root.split_scale(), held, "scale moved inside the deadband");

        // Well under budget: recover to unconstrained and stop there.
        root.resident.clear();
        for _ in 0..500 {
            root.update_split_scale(budget);
        }
        assert_eq!(root.split_scale(), 1.0, "scale did not recover");
    }

    /// A disabled budget is exactly the pre-budget renderer.
    #[test]
    fn disabled_budget_never_brakes() {
        let mut root = TileTerrainRoot::new(
            R,
            Arc::new(NullProvider),
            Handle::default(),
            RenderLayers::default(),
        );
        root.resident = (0..100_000)
            .map(|i| {
                (
                    TileKey {
                        face: 0,
                        level: 9,
                        x: i,
                        y: 0,
                    },
                    Entity::PLACEHOLDER,
                )
            })
            .collect();
        root.update_split_scale(usize::MAX);
        assert_eq!(root.split_scale(), 1.0);
    }

    struct NullProvider;

    impl TerrainTileProvider for NullProvider {
        fn request(&self, key: TileKey, _radius_m: f64) -> SurfaceTile {
            flat_tile(key)
        }

        fn height_range_m(&self) -> f32 {
            RELIEF as f32
        }
    }
}

#[cfg(test)]
mod horizon_tests {
    use super::*;

    const R: f64 = 3_186_000.0;
    const RELIEF: f64 = 9_797.6;

    /// The ground under your feet must always be refinable. A tile at
    /// MIN_LEVEL spans 589 km on Thalos, so from a few hundred metres up its
    /// corners and centre all sit far below the horizon even though the patch
    /// directly beneath the camera does not — the bug that froze refinement at
    /// the 384-tile MIN_LEVEL shell and made every framing render as a smooth
    /// sphere.
    #[test]
    fn horizon_gate_keeps_the_ground_underfoot() {
        for alt in [30.0, 756.0, 22_000.0, 200_000.0] {
            let cam = DVec3::new(0.31, 0.72, 0.62).normalize() * (R + alt);
            for level in MIN_LEVEL..=12 {
                let key = TileKey::containing_dir(cam.normalize(), level);
                assert!(
                    above_horizon(key, cam, R, RELIEF),
                    "nadir tile culled at alt {alt} m, level {level}"
                );
            }
        }
    }

    /// ...and the antipode never is, at any altitude below synchronous silly.
    #[test]
    fn horizon_gate_culls_the_far_side() {
        for alt in [30.0, 22_000.0, 200_000.0] {
            let up = DVec3::new(0.31, 0.72, 0.62).normalize();
            let cam = up * (R + alt);
            let key = TileKey::containing_dir(-up, 8);
            assert!(
                !above_horizon(key, cam, R, RELIEF),
                "antipodal tile survived at alt {alt} m"
            );
        }
    }

    /// An unbounded relief allowance must degrade to "refine as distance says",
    /// never to a nonsense comparison.
    #[test]
    fn unbounded_relief_never_culls() {
        let up = DVec3::new(0.31, 0.72, 0.62).normalize();
        let cam = up * (R + 22_000.0);
        let key = TileKey::containing_dir(-up, 8);
        assert!(above_horizon(key, cam, R, f64::INFINITY));
    }

    /// End-to-end: the gate trims the working set without stalling refinement.
    #[test]
    fn horizon_gate_trims_without_stalling() {
        let max_level = max_level_for(R);
        for alt in [756.0, 22_000.0, 160_000.0] {
            let cam = DVec3::new(0.31, 0.72, 0.62).normalize() * (R + alt);
            let open = select_leaves(cam, R, max_level);
            let gated = select_leaves_with_relief(cam, R, max_level, RELIEF, &|_| None);
            let shell = 6 * (1usize << MIN_LEVEL).pow(2);
            assert!(
                gated.len() > shell * 2,
                "alt {alt} m: refinement stalled near the MIN_LEVEL shell ({} leaves)",
                gated.len()
            );
            assert!(
                gated.len() < open.len(),
                "alt {alt} m: gate saved nothing ({} vs {})",
                gated.len(),
                open.len()
            );
        }
    }
}

#[cfg(test)]
mod skirt_tests {
    use super::*;

    const R: f64 = 3_186_000.0;
    /// Test surface: macro relief only — 2 km amplitude at ~29 km wavelength, a
    /// rugged massif. Present identically at every LOD, so the inter-level edge
    /// disagreement measured below is PURE sampling geometry (coarse chords
    /// cutting across relief the fine edge resolves); the real providers'
    /// LOD-gated detail bands only widen it.
    const AMP_M: f64 = 2_000.0;

    fn h(dir: DVec3) -> f64 {
        AMP_M * (dir.x * 700.0).sin() * (dir.z * 640.0).cos()
    }

    /// The falsifier for the transient-seam mechanism (fast orbit at altitude:
    /// tile junctions visibly crack open until streaming settles). A junction
    /// three levels deep — which the 2:1 balance forbids *settled* but the
    /// resident mosaic produces *transiently* (freshly-landed coarse ground
    /// beside lingering fine tiles) — disagrees across the shared edge by more
    /// than the old per-own-spacing skirt drops could curtain, on either side.
    /// If terrain ever becomes tame enough that this stops failing the old
    /// drops, the floor curtain could be retired for the cheaper formula.
    #[test]
    fn junction_cracks_exceed_the_old_skirt_clamp() {
        let a = TileKey {
            face: 0,
            level: 3,
            x: 3,
            y: 5,
        };
        let coarse_spacing = a.sample_spacing_m(R);
        let fine_spacing = coarse_spacing / 8.0;
        // A's right-edge polyline: 3D vertices at its own sampling.
        let coarse: Vec<DVec3> = (0..TILE_RES)
            .map(|j| {
                let t = j as f64 / (TILE_RES - 1) as f64;
                let dir = a.dir_at(1.0, t);
                dir * (R + h(dir))
            })
            .collect();

        let mut worst_above = 0.0f64; // fine edge above the coarse chord
        let mut worst_below = 0.0f64; // fine edge below it
        // Every level-6 tile bordering that edge from the other side.
        for y6 in (a.y * 8)..((a.y + 1) * 8) {
            let f = TileKey {
                face: 0,
                level: 6,
                x: (a.x + 1) * 8,
                y: y6,
            };
            for k in 0..TILE_RES {
                let t_f = k as f64 / (TILE_RES - 1) as f64;
                let dir = f.dir_at(0.0, t_f);
                let p_f = dir * (R + h(dir));
                // The enclosing coarse mesh segment, by shared-edge parameter.
                let seg = ((y6 - a.y * 8) as f64 + t_f) / 8.0 * (TILE_RES - 1) as f64;
                let i = (seg as usize).min(TILE_RES - 2);
                let chord = coarse[i].lerp(coarse[i + 1], seg - i as f64);
                let gap = (p_f - chord).dot(dir);
                worst_above = worst_above.max(gap);
                worst_below = worst_below.max(-gap);
            }
        }
        let coarse_drop = skirt_drop_m(coarse_spacing, R) as f64;
        let fine_drop = skirt_drop_m(fine_spacing, R) as f64;
        assert!(
            worst_above > fine_drop || worst_below > coarse_drop,
            "the transient junction never out-ran the old skirts (fine edge rises \
             {worst_above:.0} m over the coarse chord vs a {fine_drop:.0} m fine drop; \
             dips {worst_below:.0} m under it vs a {coarse_drop:.0} m coarse drop)"
        );
    }

    /// The invariant that replaces the old 150 m clamp: every skirt vertex
    /// lands on the body-wide floor sphere (`radius − relief`), which no
    /// partner tile's surface — nor any chord between its samples — can
    /// undercut, since heights are bounded by the same envelope. A junction
    /// crack of ANY transient level gap therefore backs onto terrain-coloured
    /// curtain, never onto see-through.
    #[test]
    fn skirt_curtains_reach_the_body_floor() {
        const RELIEF: f64 = 2_500.0; // envelope over the 2 km test amplitude
        let key = TileKey {
            face: 0,
            level: 6,
            x: 25,
            y: 42,
        };
        let side = SurfaceTile::grid_side();
        let step = 1.0 / (TILE_RES - 1) as f64;
        let mut heights = Vec::with_capacity(side * side);
        for j in 0..side {
            for i in 0..side {
                let s = (i as f64 - TILE_HALO as f64) * step;
                let t = (j as f64 - TILE_HALO as f64) * step;
                heights.push(h(key.dir_at(s, t)) as f32);
            }
        }
        let tile = SurfaceTile {
            key,
            sample_spacing_m: key.sample_spacing_m(R),
            heights_m: heights,
            albedo_linear: vec![[0.4, 0.4, 0.4]; side * side],
            bands: vec![[0.0, 0.0]; side * side],
        };
        let built = build_tile_mesh(&tile, R, RELIEF);
        let positions = built
            .mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|a| a.as_float3())
            .expect("tile meshes carry positions");
        let floor = R - RELIEF;
        for (n, p) in positions.iter().enumerate() {
            let r = (built.origin + DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64)).length();
            if n < TILE_RES * TILE_RES {
                assert!(
                    r > floor + 100.0,
                    "core vertex {n} sits at the floor — the curtain ate the surface"
                );
            } else {
                assert!(
                    r <= floor + 0.01,
                    "skirt vertex {n} floats {:.1} m above the floor sphere",
                    r - floor
                );
            }
        }
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    /// Simulate the production streaming loop through a continuous descent —
    /// selection, cancellation, capped admission, landing latency, lax
    /// retirement, despawn — and assert the coverage invariant every tick.
    /// This is the falsifiable repro for the in-game "black tiles while
    /// descending into craters" reports: if the despawn/selection logic can
    /// drop the last cover of a desired tile, this fails with the tick and
    /// chain state.
    ///
    /// `rugged_field` supplies each tile's true ruggedness, measured at landing
    /// and memoised exactly as `stream_tile_terrain` does, so the relief-aware
    /// split rule is driven through the same feedback loop it has in
    /// production: selection deepens as measurements arrive. `None` runs the
    /// plain distance rule.
    /// `sim_end_s` must outlast the streamer, not just the retirement gates —
    /// see the settle comment in the loop. It is sized from the measured drain
    /// rate, so it grows with the working set the rule selects.
    fn simulate_descent(rugged_field: Option<&dyn Fn(TileKey) -> f32>, sim_end_s: f32) {
        const RADIUS_M: f64 = 869_000.0;
        // Mira's real height range, so the rugged twin drives the horizon
        // refinement gate too: it moves with the eye, which is exactly the
        // thing that could strand a resident tile's cover.
        const RELIEF_M: f64 = 13_019.0;
        const DT: f32 = 1.0 / 60.0;
        const GEN_LATENCY_S: f32 = 0.032;
        // Production parallelism: the synthesis pool runs 4 workers; the
        // in-flight cap only bounds the queue. Landing rate ≈ 125 tiles/s,
        // matching live telemetry.
        const WORKERS: usize = 4;
        let max_level = max_level_for(RADIUS_M);

        // The harshest path we ship: plunge from 200 km to 30 m, then sweep
        // laterally at aircraft speed just above the ground (the user's
        // "close to the ground, in the craters" flight), then climb out and
        // dive again elsewhere.
        let site = DVec3::new(-0.71, 0.18, -0.68).normalize();
        let drift = site.cross(DVec3::Y).normalize();
        let cam_at = |t: f32| -> DVec3 {
            let t = t as f64;
            if t < 20.0 {
                // Plunge.
                let k = t / 20.0;
                let alt = 200_000.0 * (1.0 - k).powi(3) + 30.0;
                site * (RADIUS_M + alt)
            } else if t < 50.0 {
                // Low lateral sweep: 250 m/s at 30 m — the selection window
                // races across the surface far faster than tiles can land.
                let s = (t - 20.0) * 250.0 / RADIUS_M;
                let dir = (site + drift * s).normalize();
                dir * (RADIUS_M + 30.0)
            } else {
                // Climb out and dive onto a new spot.
                let k = ((t - 50.0) / 10.0).min(1.0);
                let s = 30.0 * 250.0 / RADIUS_M;
                let dir = (site + drift * (s + k * 0.01)).normalize();
                let alt = 30.0 + 20_000.0 * (0.5 - (k - 0.5).abs()).max(0.0) * 2.0;
                dir * (RADIUS_M + alt)
            }
        };

        let mut resident: HashMap<TileKey, Entity> = HashMap::new();
        let mut retiring: HashMap<TileKey, (f32, u16)> = HashMap::new();
        // The production ruggedness memo: measured once per key, never pruned.
        let mut rugged_memo: HashMap<TileKey, f32> = HashMap::new();
        let select = |cam: DVec3, memo: &HashMap<TileKey, f32>| -> HashSet<TileKey> {
            if rugged_field.is_none() {
                return select_leaves(cam, RADIUS_M, max_level);
            }
            select_leaves_with_relief(cam, RADIUS_M, max_level, RELIEF_M, &|key| {
                let mut k = key;
                loop {
                    if let Some(&r) = memo.get(&k) {
                        return Some(r);
                    }
                    match k.parent() {
                        Some(p) if p.level >= MIN_LEVEL => k = p,
                        _ => return None,
                    }
                }
            })
        };
        // (key, seconds-until-landed); at most MAX_IN_FLIGHT actively count
        // down, the rest queue — mirrors the bounded pool.
        let mut pending: Vec<(TileKey, f32)> = Vec::new();
        let mut covered_once = false;
        // Event log for post-mortem on failure: (t, what, key).
        let mut events: Vec<(f32, &'static str, TileKey)> = Vec::new();
        let related = |a: TileKey, b: TileKey| -> bool {
            if a.face != b.face {
                return false;
            }
            let (hi, lo) = if a.level <= b.level { (a, b) } else { (b, a) };
            let shift = lo.level - hi.level;
            (lo.x >> shift) == hi.x && (lo.y >> shift) == hi.y
        };

        let mut desired: HashSet<TileKey> = HashSet::new();
        let mut last_cam = DVec3::ZERO;
        let mut last_memo_len = usize::MAX;

        // 62 s of flight, then hold still until residency settles.
        //
        // The hold has to outlast the *streamer*, not just the retirement
        // gates: the low sweep ends with ~6,450 desired tiles and a shortfall
        // of ~3,850, which drains at the simulated 120 tiles/s — ~32 s. At the
        // original 13 s hold the coverage latch could never close, so the
        // convergence half of this gate was asserting on a state the harness
        // never reached (the per-tick invariant above was still live and
        // silent, which is why no hole hid behind it). The relief rule wants
        // more still, since ruggedness deepens the selection.
        let mut t = 0.0_f32;
        while t < sim_end_s {
            let cam = cam_at(t);
            // Reuse the previous set when neither the eye nor the ruggedness
            // memo moved — production recomputes per frame and would get the
            // identical set, and the stationary settle is most of the ticks.
            let memo_len = rugged_memo.len();
            if cam != last_cam || memo_len != last_memo_len {
                desired = select(cam, &rugged_memo);
                last_cam = cam;
                last_memo_len = memo_len;
            }
            let bridges = bridge_requests(&desired, &resident);

            // Cancel pendings nobody wants (production: Task drop).
            let (keep, cancelled): (Vec<_>, Vec<_>) = pending
                .into_iter()
                .partition(|(key, _)| desired.contains(key) || bridges.contains(key));
            pending = keep;
            for (key, _) in cancelled {
                events.push((t, "cancel", key));
            }

            // Admit missing (desired + bridges) by screen-size priority up to
            // the in-flight cap.
            let mut missing: Vec<TileKey> = desired
                .iter()
                .chain(bridges.iter())
                .filter(|k| !resident.contains_key(k) && !pending.iter().any(|(p, _)| p == *k))
                .copied()
                .collect();
            missing.sort_by(|a, b| {
                let pa = (cam - a.center_dir() * RADIUS_M).length() / tile_arc_m(a.level, RADIUS_M);
                let pb = (cam - b.center_dir() * RADIUS_M).length() / tile_arc_m(b.level, RADIUS_M);
                pa.total_cmp(&pb)
            });
            let budget = MAX_IN_FLIGHT.saturating_sub(pending.len());
            for key in missing.into_iter().take(budget) {
                pending.push((key, GEN_LATENCY_S));
            }

            // Advance the workers (only WORKERS progress at once) and land.
            for slot in pending.iter_mut().take(WORKERS) {
                slot.1 -= DT;
            }
            let mut landed: Vec<TileKey> = Vec::new();
            pending.retain(|(key, remaining)| {
                if *remaining <= 0.0 {
                    landed.push(*key);
                    false
                } else {
                    true
                }
            });
            for key in landed {
                events.push((t, "land", key));
                resident.insert(key, Entity::PLACEHOLDER);
                if let Some(field) = rugged_field {
                    rugged_memo.insert(key, field(key));
                }
            }

            if !covered_once
                && !desired.is_empty()
                && desired.iter().all(|k| resident.contains_key(k))
            {
                covered_once = true;
            }

            for key in despawn_ready(&desired, &resident, &mut retiring, max_level, DT) {
                events.push((t, "despawn", key));
                resident.remove(&key);
            }

            if covered_once {
                let uncovered = uncovered_desired(&desired, &resident, max_level);
                if let Some(first) = uncovered.first().copied() {
                    eprintln!("--- column history for uncovered {first:?} ---");
                    for (et, what, key) in events.iter().filter(|(_, _, k)| related(*k, first)) {
                        eprintln!("  t={et:.2} {what} {key:?}");
                    }
                    eprintln!("--- column current state ---");
                    for (k, _) in resident.iter().filter(|(k, _)| related(**k, first)) {
                        eprintln!("  resident {k:?} (retiring: {:?})", retiring.get(k));
                    }
                    for (k, _) in pending.iter().filter(|(k, _)| related(*k, first)) {
                        eprintln!("  pending {k:?}");
                    }
                    for k in desired.iter().filter(|k| related(**k, first)) {
                        eprintln!("  desired {k:?}");
                    }
                    panic!(
                        "t={t:.2}s alt={:.0}m: {} desired tiles uncovered; first={first:?}; resident={} pending={} retiring={}",
                        cam.length() - RADIUS_M,
                        uncovered.len(),
                        resident.len(),
                        pending.len(),
                        retiring.len(),
                    );
                }
            }

            // Shortfall trace during the stationary settle: distinguishes "the
            // streamer just needs longer" from "selection and residency never
            // agree" when the coverage latch fails to close.
            if !covered_once && t > 62.0 && (t * 60.0).round() as i64 % 60 == 0 {
                let missing = desired.iter().filter(|k| !resident.contains_key(k)).count();
                eprintln!(
                    "  t={t:.1} settle: desired={} missing={} resident={} pending={} bridges={}",
                    desired.len(),
                    missing,
                    resident.len(),
                    pending.len(),
                    bridges.len(),
                );
            }

            t += DT;
        }
        assert!(covered_once, "descent never reached full coverage");

        // After 13 s stationary, every stale tile (bridges included) must
        // have retired: residents converge exactly onto the desired set. With
        // the relief rule live this also catches a ratcheting oracle — a
        // ruggedness estimate that keeps deepening the selection would leave
        // the resident set permanently chasing it.
        let desired = select(cam_at(sim_end_s), &rugged_memo);
        let stale: Vec<TileKey> = resident
            .keys()
            .filter(|k| !desired.contains(k))
            .copied()
            .collect();
        assert!(
            stale.is_empty(),
            "{} stale residents never retired after settling; first={:?}",
            stale.len(),
            stale.first()
        );
    }

    #[test]
    fn descent_keeps_every_desired_tile_covered() {
        simulate_descent(None, 125.0);
    }

    /// The same descent with the relief-aware split rule live. A belt of rugged
    /// terrain crosses the descent site, so the boost engages mid-stream and
    /// the desired set deepens *underneath* tiles that are already resident —
    /// the failure mode the plain sim cannot reach.
    #[test]
    fn rugged_descent_keeps_every_desired_tile_covered() {
        let belt = |key: TileKey| -> f32 {
            let d = key.center_dir();
            // Smooth basins between two crossing ridges: spans RUGGED_LO
            // (0.012) through past RUGGED_HI (0.055), so both ends of the
            // ruggedness ramp are exercised.
            let s = ((d.x * 9.0).sin() * (d.z * 7.0).cos()).abs();
            (0.004 + 0.06 * s) as f32
        };
        simulate_descent(Some(&belt), 185.0);
    }
}

/// Registers the tile material + streaming system; inert until a
/// [`TileTerrainRoot`] exists and the game writes [`TileEye`].
pub struct TileTerrainPlugin;

impl Plugin for TileTerrainPlugin {
    fn build(&self, app: &mut App) {
        // The tile shader imports `thalos::shadow` / `thalos::lighting`; make
        // sure the library is registered (no-op if already added).
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
        }
        app.add_plugins(bevy::pbr::MaterialPlugin::<material::TileTerrainMaterial>::default());
        app.init_resource::<TileEye>()
            .add_systems(Update, stream_tile_terrain.in_set(TileStreamSet))
            // `Last`, not `PostUpdate`: the game's cloud driver resolves the
            // cascade frame in `PostUpdate`, and the compute pass marches THAT
            // frame the same frame. Fanning the receiver block from an
            // unordered `PostUpdate` system would hand materials the previous
            // frame's placement half the time, so a moving camera would sample
            // the map through a frame it was not marched in — a shadow that
            // slides against the ground it belongs to.
            .add_systems(Last, apply_cloud_shadow);
    }
}

/// Fan the live cloud sun-transmittance cascade onto every tile material —
/// the same shape as `craft::apply_craft_shadow` fanning the sun cascade, and
/// for the same reason: the field is one global authority, not per-material
/// state, so materials must never be given their own copy to drift with.
///
/// No-op without [`CloudShadowMap`] (a binary with no cloud renderer), which
/// leaves the block zeroed and the shader fully lit.
fn apply_cloud_shadow(
    map: Option<Res<crate::clouds::CloudShadowMap>>,
    mut materials: ResMut<Assets<material::TileTerrainMaterial>>,
) {
    let Some(map) = map else {
        return;
    };
    let block = map.block();
    for (_, mat) in materials.iter_mut() {
        mat.extension.cloud_shadow = block;
        mat.extension.cloud_shadow_map = map.handle.clone();
    }
}
