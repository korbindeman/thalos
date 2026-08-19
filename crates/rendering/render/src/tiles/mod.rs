//! NTR-X1 — the probe-extracted tile terrain renderer (keystone:
//! ADR-20260723T142945Z, plan `ntr §6`): terrain as ordinary `Mesh` +
//! `StandardMaterial` entities streamed by a camera-driven, 2:1-balanced
//! cube-sphere quadtree, entirely on Bevy's standard render path. Resident
//! entities share three patch meshes; exact f64-built positions and surface
//! channels live in a GPU array atlas selected through `MeshTag`.
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
//! (`crates/rendering/udlod`, reached through [`crate::ground`]) is absent from
//! default builds and can be selected only by an opt-in `legacy-udlod` binary
//! with `THALOS_TILE_RENDERER=0`. Known limits,
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
//! punch holes in the ground. The finite atlas adds a second hard ceiling and
//! reserves replacement layers so bridge-before-retire cannot deadlock. Any
//! new per-tile GPU resource must join
//! `TILE_MESH_BYTES` or the budget silently under-counts VRAM
//! (INC-20260725T012104Z-tile-residency-had-no-budget).
//!
//! The budget is **machine-wide, not per process**: it is divided by the number
//! of live Thalos renderers ([`vram_share`]), because the card is shared and two
//! instances each reading the full figure is how the second `DeviceLost`
//! happened after the first budget landed.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::Instant;

#[cfg(test)]
use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::Aabb;
use bevy::camera::visibility::NoAutoAabb;
use bevy::camera::visibility::RenderLayers;
use bevy::math::{DQuat, DVec3};
use bevy::mesh::MeshTag;
#[cfg(test)]
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::tasks::{Task, block_on, poll_once};
use big_space::prelude::*;
use thalos_terrain::{SurfacePatch, SurfaceQuery};

pub mod cache;
pub mod gpu;
pub mod height_mirror;
pub mod material;
pub mod vram_share;

pub use cache::{CachedTileProvider, SurfaceTileCache, TileNamespaceFn};
pub use gpu::TileGpuStore;
pub use height_mirror::{TileHeightMirror, TileHeightMirrorHandle};

/// Vertices per tile side (core grid, excluding halo).
pub const TILE_RES: usize = 129;
/// Halo rings included in every sampled grid (edge-exact normals).
pub const TILE_HALO: usize = 1;
/// Coarsest selected level (4×4 tiles per face).
pub const MIN_LEVEL: u8 = 2;
/// Split while camera distance < factor × tile arc — the *floor* of the
/// ruggedness-scaled rule below.
///
/// This factor alone sets the *geometric* resolution of any framing: a
/// resident tile at distance `d` has arc ≈ `d / SPLIT_FACTOR`, so its sample
/// spacing is `d / (SPLIT_FACTOR × (TILE_RES − 1))`.
///
/// Keep the product, not the bare factor, stable when tile resolution changes.
/// NTR-X12 moved 65² tiles at level L to 129² tiles at L−1: each new tile
/// covers four old footprints with the same 384 samples across the split
/// distance, cutting entities and draw submissions toward one quarter without
/// changing ground sample density.
const BASE_SPLIT_SAMPLES: f64 = 6.0 * 64.0;
const SPLIT_FACTOR: f64 = BASE_SPLIT_SAMPLES / (TILE_RES - 1) as f64;
/// Split cap for the most rugged terrain (ntr §7's relief-aware rule).
///
/// A single density rule buys resolution *everywhere* at 4× the tiles per
/// doubling, which is why [`BASE_SPLIT_SAMPLES`] is as far as a uniform rule
/// can go. But the resolution a frame actually needs is not uniform: from
/// altitude the ocean and plains are converged at the base density while
/// mountain ridges are still the mesh's fault, not the data's. The rugged cap
/// raises that density and [`tile_ruggedness_weight`] removes the extra
/// everywhere terrain is smooth. Relief may only take detail away from the
/// distance cap, never add it. Cost lands only on the terrain the player is
/// looking at when they say "mountains".
const RUGGED_SPLIT_SAMPLES: f64 = 18.0 * 64.0;
const SPLIT_FACTOR_RUGGED: f64 = RUGGED_SPLIT_SAMPLES / (TILE_RES - 1) as f64;
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
/// Maximum package-declared refinement displacement allowed on screen before
/// the ruggedness boost must split. One pixel is below the stable silhouette
/// and shading footprint of the standard TAA path; the base distance rule is
/// never gated by this value.
const GEOMETRIC_ERROR_THRESHOLD_PX: f64 = 1.0;

/// Runtime A/B for matched captures. Default-on; `0`, `false`, or `off`
/// restores the exact pre-package-error selector without a recompile.
fn package_screen_error_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("THALOS_PACKAGE_SCREEN_ERROR")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off"
                )
            })
    })
}
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
/// Split-distance scale for the initial resident cover, relative to the final
/// selection. One quarter moves the distance-driven refinement boundary about
/// two quadtree levels inward while the ground directly under the camera (and
/// authored sites) retains its required level. The result is a smaller valid,
/// non-overlapping selection that remains locally usable; the ordinary bridge
/// path then refines the surroundings to the unchanged desired set.
const BOOTSTRAP_SPLIT_SCALE: f64 = 0.25;

// --- residency budget --------------------------------------------------------
//
// INC-20260725T012104Z-tile-residency-had-no-budget: this renderer could
// allocate VRAM without limit until a `DeviceLost` stopped it.

/// Skirt ring vertices appended by each shared patch (one loop around the
/// core grid: `4·(res−1)`, i.e. each side minus its shared corner).
const TILE_SKIRT_VERTS: usize = 4 * TILE_RES - 4;
/// Vertices in one tile mesh — core grid plus the skirt ring.
const TILE_VERTS: usize = TILE_RES * TILE_RES + TILE_SKIRT_VERTS;
/// Vertex width of the retired per-tile mesh, retained only by the test oracle
/// that proves the GPU payload reconstructs the same surface.
#[cfg(test)]
const TILE_VERTEX_BYTES: usize = 12 + 12 + 16 + 8 + 8;
/// Triangles in one tile mesh: two per core quad plus two per skirt quad.
const TILE_TRIANGLES: usize = (TILE_RES - 1) * (TILE_RES - 1) * 2 + TILE_SKIRT_VERTS * 2;
/// Restart-delimited `U16` indices in the visible mesh. Each grid row is one
/// strip, and each skirt quad is its own four-index strip so it preserves the
/// exact diagonal and winding of the original triangle list.
#[cfg(test)]
const TILE_STRIPS: usize = (TILE_RES - 1) + TILE_SKIRT_VERTS;
#[cfg(test)]
const TILE_STRIP_INDICES: usize =
    2 * TILE_RES * (TILE_RES - 1) + 4 * TILE_SKIRT_VERTS + (TILE_STRIPS - 1);
const _: () = assert!(TILE_VERTS < u16::MAX as usize);
/// GPU payload bytes one resident tile occupies in the displacement atlas.
/// Kept under the established name because diagnostics and RT accounting use
/// it as the residency denominator; it no longer describes a unique `Mesh`.
pub const TILE_MESH_BYTES: usize = gpu::TILE_GPU_SLOT_BYTES;

/// Grid resolution of the shared patch used by broad shadow cascades.
/// 33² samples retain every fourth atlas sample, reducing surface triangles
/// 16× while preserving tile boundaries and the skirt.
const FAR_CASTER_RES: usize = 33;
#[cfg(test)]
const FAR_CASTER_STEP: usize = (TILE_RES - 1) / (FAR_CASTER_RES - 1);
const FAR_CASTER_SKIRT_VERTS: usize = 4 * FAR_CASTER_RES - 4;
const FAR_CASTER_VERTS: usize = FAR_CASTER_RES * FAR_CASTER_RES + FAR_CASTER_SKIRT_VERTS;
#[cfg(test)]
const FAR_CASTER_STRIPS: usize = (FAR_CASTER_RES - 1) + FAR_CASTER_SKIRT_VERTS;
#[cfg(test)]
const FAR_CASTER_STRIP_INDICES: usize = 2 * FAR_CASTER_RES * (FAR_CASTER_RES - 1)
    + 4 * FAR_CASTER_SKIRT_VERTS
    + (FAR_CASTER_STRIPS - 1);
/// Broad shadows reuse the one shared 33² patch mesh, so their per-tile
/// geometry residency is zero. Their atlas payload is already counted above.
pub const FAR_CASTER_MESH_BYTES: usize = 0;
const _: () = assert!((TILE_RES - 1).is_multiple_of(FAR_CASTER_RES - 1));
const _: () = assert!(FAR_CASTER_VERTS < u16::MAX as usize);

const TILE_INDEX_PROBE_ENV: &str = "THALOS_TILE_INDEX_PROBE";
const TILE_CULL_PROBE_ENV: &str = "THALOS_TILE_CULL_PROBE";

/// Extra GPU bytes a tile costs **on top of** [`TILE_MESH_BYTES`] when it also
/// carries an RT twin — **1,200 KiB**, i.e. an RT-covered tile costs about
/// **4.50×** a plain atlas slot.
///
/// Solari cannot execute the raster material's vertex displacement or resolve
/// its per-instance atlas slot, so RT geometry remains a second copy
/// ([`crate::rt`] explains why in full). Near-radius-only coverage remains a
/// necessity rather than a tuning choice.
pub const RT_TILE_MESH_BYTES: usize =
    TILE_VERTS * crate::rt::RT_VERTEX_BYTES + TILE_TRIANGLES * 3 * 4;

/// Soft VRAM target for occupied tile payloads **across every Thalos renderer
/// on this machine**, overridable with `THALOS_TILE_BUDGET_MB`. One process's
/// share is this divided by the live instance count; the atlas's finite usable
/// capacity still applies when the environment budget is disabled.
///
/// The 2,048-layer atlas is ~687 MiB allocated; 1,792 layers are admissible and
/// the remainder are replacement headroom. On current hardware that capacity,
/// rather than this legacy 4 GiB default, is normally the effective brake.
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
/// Floor on the split scale — `SPLIT_FACTOR × 0.333 ≈ 1.0`, coarse but a
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

    /// Maximum displacement one child-level refinement can reveal in `key`.
    /// `None` retains the selector's existing distance/ruggedness rule.
    fn refinement_error_m(&self, _key: TileKey, _radius_m: f64) -> Option<f32> {
        None
    }

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

    fn refinement_error_m(&self, key: TileKey, radius_m: f64) -> Option<f32> {
        let side = key.tiles_per_side();
        self.surface.refinement_error_m(
            SurfacePatch {
                face: key.face,
                level: key.level,
                x: key.x,
                // Renderer face coordinates grow upward; package cubemap rows
                // grow downward. The face order and x axis are identical.
                y: side - 1 - key.y,
            },
            (key.sample_spacing_m(radius_m) * 0.5) as f32,
        )
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

#[cfg(test)]
struct BuiltTile {
    mesh: Mesh,
    /// Opt-in compact visible twin for the warmed geometry-density probe.
    index_probe_mesh: Option<Mesh>,
    /// Opt-in full/skirt and tight/surface bounds for the warmed culling probe.
    culling_probe_bounds: Option<TileCullingProbeBounds>,
    /// Coarse position+normal twin for the broad shadow cascades. Absent for a
    /// root that does not request terrain casting.
    far_caster_mesh: Option<Mesh>,
    /// Body-fixed f64 position of the mesh origin (displaced tile center).
    origin: DVec3,
    /// Radial deviation range (m) of the built interior vertices from the
    /// reference sphere — the mesh-side counterpart of the provider height
    /// range, so "heights sampled" vs "heights in the mesh" separate in the
    /// telemetry.
    mesh_h: (f32, f32),
    /// Culling box over the surface band only, excluding the skirt curtain —
    /// see where it is built in [`build_tile_mesh`].
    surface_aabb: Aabb,
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

/// Build the compact raster topology for a tile.
///
/// Core rows run right-to-left so strip parity produces the same `a-b-d,
/// a-d-c` diagonal and outward winding as the former triangle list. Skirt
/// quads remain separate strips: joining them would flip their diagonal on
/// alternating segments, making the optimisation subtly change the curtain
/// geometry it is meant only to encode more cheaply.
fn build_tile_strip_indices(res: usize, border: &[u16], skirt_base: u16) -> Vec<u16> {
    let strip_count = (res - 1) + border.len();
    let index_count = 2 * res * (res - 1) + 4 * border.len() + strip_count - 1;
    let mut indices = Vec::with_capacity(index_count);

    let start_strip = |indices: &mut Vec<u16>| {
        if !indices.is_empty() {
            indices.push(u16::MAX);
        }
    };

    for j in 0..res - 1 {
        start_strip(&mut indices);
        for i in (0..res).rev() {
            indices.push((j * res + i) as u16);
            indices.push(((j + 1) * res + i) as u16);
        }
    }

    for k in 0..border.len() {
        start_strip(&mut indices);
        let k2 = (k + 1) % border.len();
        let (top_a, top_b) = (border[k], border[k2]);
        let (bot_a, bot_b) = (skirt_base + k as u16, skirt_base + k2 as u16);
        // Expands to top_a-bot_a-bot_b, top_a-bot_b-top_b.
        indices.extend_from_slice(&[bot_a, bot_b, top_a, top_b]);
    }

    debug_assert_eq!(indices.len(), index_count);
    indices
}

/// Number of indices in the headless tile-density probe for one sampling step.
///
/// This is deliberately a diagnostics-only seam. Production tiles always use
/// step 1; the warmed probe swaps in a compact mesh built from every fourth
/// source sample to distinguish geometry traversal from pixel fill without
/// changing tile entities, material work, or boundary coverage.
#[doc(hidden)]
pub fn tile_index_count_for_benchmark(step: usize) -> Option<usize> {
    if step == 0 || !(TILE_RES - 1).is_multiple_of(step) {
        return None;
    }
    let res = (TILE_RES - 1) / step + 1;
    let border = TILE_SKIRT_VERTS / step;
    let strips = (res - 1) + border;
    Some(2 * res * (res - 1) + 4 * border + strips - 1)
}

fn tile_index_probe_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(TILE_INDEX_PROBE_ENV)
            .ok()
            .is_some_and(|raw| matches!(raw.trim(), "1" | "true" | "on" | "yes"))
    })
}

fn tile_cull_probe_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(TILE_CULL_PROBE_ENV)
            .ok()
            .is_some_and(|raw| matches!(raw.trim(), "1" | "true" | "on" | "yes"))
    })
}

#[cfg(test)]
fn build_tile_index_probe_mesh(
    source_positions: &[[f32; 3]],
    source_normals: &[[f32; 3]],
    source_colors: &[[f32; 4]],
    source_uv0: &[[f32; 2]],
    source_uv1: &[[f32; 2]],
) -> Mesh {
    let mut sources = Vec::with_capacity(FAR_CASTER_VERTS);
    for j in 0..FAR_CASTER_RES {
        for i in 0..FAR_CASTER_RES {
            sources.push((j * FAR_CASTER_STEP) * TILE_RES + i * FAR_CASTER_STEP);
        }
    }
    sources.extend(
        (0..TILE_SKIRT_VERTS)
            .step_by(FAR_CASTER_STEP)
            .map(|offset| TILE_RES * TILE_RES + offset),
    );
    debug_assert_eq!(sources.len(), FAR_CASTER_VERTS);

    let positions = sources
        .iter()
        .map(|&source| source_positions[source])
        .collect::<Vec<_>>();
    let normals = sources
        .iter()
        .map(|&source| source_normals[source])
        .collect::<Vec<_>>();
    let colors = sources
        .iter()
        .map(|&source| source_colors[source])
        .collect::<Vec<_>>();
    let uv0 = sources
        .iter()
        .map(|&source| source_uv0[source])
        .collect::<Vec<_>>();
    let uv1 = sources
        .iter()
        .map(|&source| source_uv1[source])
        .collect::<Vec<_>>();

    let mut border = Vec::with_capacity(FAR_CASTER_SKIRT_VERTS);
    for i in 0..FAR_CASTER_RES {
        border.push(i as u16);
    }
    for j in 1..FAR_CASTER_RES {
        border.push((j * FAR_CASTER_RES + FAR_CASTER_RES - 1) as u16);
    }
    for i in (0..FAR_CASTER_RES - 1).rev() {
        border.push(((FAR_CASTER_RES - 1) * FAR_CASTER_RES + i) as u16);
    }
    for j in (1..FAR_CASTER_RES - 1).rev() {
        border.push((j * FAR_CASTER_RES) as u16);
    }
    let indices = build_tile_strip_indices(
        FAR_CASTER_RES,
        &border,
        (FAR_CASTER_RES * FAR_CASTER_RES) as u16,
    );

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleStrip,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv0);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    mesh.insert_indices(Indices::U16(indices));
    mesh
}

#[cfg(test)]
fn build_far_caster_mesh(
    surface_positions: &[[f32; 3]],
    surface_normals: &[[f32; 3]],
    origin: DVec3,
    floor_radius_m: f64,
    relief_is_bounded: bool,
    legacy_drop_m: f64,
    down: DVec3,
) -> Mesh {
    let mut positions = Vec::with_capacity(FAR_CASTER_VERTS);
    let mut normals = Vec::with_capacity(FAR_CASTER_VERTS);
    for j in 0..FAR_CASTER_RES {
        for i in 0..FAR_CASTER_RES {
            let source = (j * FAR_CASTER_STEP) * TILE_RES + i * FAR_CASTER_STEP;
            positions.push(surface_positions[source]);
            normals.push(surface_normals[source]);
        }
    }

    let mut border = Vec::with_capacity(FAR_CASTER_SKIRT_VERTS);
    for i in 0..FAR_CASTER_RES {
        border.push(i as u16);
    }
    for j in 1..FAR_CASTER_RES {
        border.push((j * FAR_CASTER_RES + FAR_CASTER_RES - 1) as u16);
    }
    for i in (0..FAR_CASTER_RES - 1).rev() {
        border.push(((FAR_CASTER_RES - 1) * FAR_CASTER_RES + i) as u16);
    }
    for j in (1..FAR_CASTER_RES - 1).rev() {
        border.push((j * FAR_CASTER_RES) as u16);
    }

    let skirt_base = positions.len() as u16;
    for &idx in &border {
        let p = positions[idx as usize];
        let abs = origin + DVec3::from(p.map(f64::from));
        let bottom = if relief_is_bounded {
            abs.normalize() * floor_radius_m
        } else {
            abs + down * legacy_drop_m
        } - origin;
        positions.push(bottom.as_vec3().to_array());
        normals.push(normals[idx as usize]);
    }
    let indices = build_tile_strip_indices(FAR_CASTER_RES, &border, skirt_base);

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleStrip,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U16(indices));
    mesh
}

#[cfg(test)]
fn build_tile_mesh(
    tile: &SurfaceTile,
    radius_m: f64,
    relief_m: f64,
    build_far_caster: bool,
) -> BuiltTile {
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
    let border: Vec<u16> = {
        let mut b = Vec::new();
        for i in 0..res {
            b.push(i as u16);
        }
        for j in 1..res {
            b.push((j * res + res - 1) as u16);
        }
        for i in (0..res - 1).rev() {
            b.push(((res - 1) * res + i) as u16);
        }
        for j in (1..res - 1).rev() {
            b.push((j * res) as u16);
        }
        b
    };
    let down = -key.center_dir();
    let base = positions.len() as u16;
    // The SURFACE band, measured before the skirt curtain is appended below.
    //
    // Bevy derives a mesh's culling `Aabb` from all of its positions, and the
    // curtain hangs every border vertex down to the body-wide floor sphere —
    // `relief_m` below the datum, ~10 km on Thalos. So the automatic box for a
    // 300 m tile is ~10 km TALL: a 35:1 slab that intersects almost any frustum
    // pointed anywhere near the body, which makes frustum culling nearly
    // inoperative for tiles. In `forest-stand`, the same dense mesh with a
    // surface-only box reduced view-visible tiles from 388 to 274; the shadow
    // rig previously saw the same failure as 122 terrain tiles inside a
    // cascade covering 64 m of ground. The curtain still rasterizes whenever
    // the surface is in frame, which is the only time it is anything but
    // buried (BL-20260731T202656Z).
    let surface_aabb = positions[..base as usize].iter().fold(
        ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
        |(mut lo, mut hi), p| {
            for axis in 0..3 {
                lo[axis] = lo[axis].min(p[axis]);
                hi[axis] = hi[axis].max(p[axis]);
            }
            (lo, hi)
        },
    );
    let far_caster_mesh = build_far_caster.then(|| {
        build_far_caster_mesh(
            &positions,
            &normals,
            origin,
            floor_radius_m,
            relief_m.is_finite(),
            legacy_drop_m,
            down,
        )
    });
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
    let indices = build_tile_strip_indices(res, &border, base);

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

    let surface_aabb = Aabb::from_min_max(
        Vec3::from_array(surface_aabb.0),
        Vec3::from_array(surface_aabb.1),
    );
    let culling_probe_bounds = tile_cull_probe_enabled().then(|| {
        let (lo, hi) = positions.iter().fold(
            ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
            |(mut lo, mut hi), position| {
                for axis in 0..3 {
                    lo[axis] = lo[axis].min(position[axis]);
                    hi[axis] = hi[axis].max(position[axis]);
                }
                (lo, hi)
            },
        );
        TileCullingProbeBounds {
            full: Aabb::from_min_max(Vec3::from_array(lo), Vec3::from_array(hi)),
            surface: surface_aabb,
        }
    });
    let index_probe_mesh = tile_index_probe_enabled()
        .then(|| build_tile_index_probe_mesh(&positions, &normals, &colors, &uv0, &uv1));
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleStrip,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv0);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    mesh.insert_indices(Indices::U16(indices));
    BuiltTile {
        mesh,
        index_probe_mesh,
        culling_probe_bounds,
        far_caster_mesh,
        origin,
        mesh_h,
        surface_aabb,
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
    /// Perspective focal length along the viewport's vertical axis, in
    /// physical pixels. `None` retains the heuristic selector (startup before
    /// the camera has a viewport, or a non-perspective view).
    pub vertical_focal_length_px: Option<f64>,
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

/// Screen-space topology choice for a visible tile or its near shadow twin.
/// The atlas payload remains full resolution; changing this component only
/// changes which exact subset of its samples the shared patch visits.
#[derive(Component, Clone, Copy)]
struct TilePatchLod {
    key: TileKey,
    gpu_slot: u32,
    resolution: gpu::PatchResolution,
    /// 0 = this topology collapsed onto the next-coarser lattice, 1 = exact
    /// samples at this topology. Packed into `MeshTag` for the vertex shader.
    morph: f32,
}

#[derive(Resource, Default)]
struct TilePatchLodGauge {
    r33: usize,
    r65: usize,
    r129: usize,
}

const PATCH_R33_MAX_PX: f64 = 128.0;
const PATCH_R65_MAX_PX: f64 = 256.0;
const PATCH_HYSTERESIS: f64 = 0.125;
const PATCH_MORPH_PER_S: f32 = 5.0;
const TILE_GPU_SLOT_BITS: u32 = gpu::TILE_GPU_ATLAS_SLOTS.trailing_zeros();
const TILE_GPU_SLOT_MASK: u32 = gpu::TILE_GPU_ATLAS_SLOTS - 1;
const _: () = assert!(gpu::TILE_GPU_ATLAS_SLOTS.is_power_of_two());

fn tile_mesh_tag(slot: u32, morph: f32, previous_morph: f32) -> MeshTag {
    let morph_u8 = (morph.clamp(0.0, 1.0) * 255.0).round() as u32;
    let previous_u8 = (previous_morph.clamp(0.0, 1.0) * 255.0).round() as u32;
    MeshTag(
        (slot & TILE_GPU_SLOT_MASK)
            | (morph_u8 << TILE_GPU_SLOT_BITS)
            | (previous_u8 << (TILE_GPU_SLOT_BITS + 8)),
    )
}

fn patch_rank(resolution: gpu::PatchResolution) -> u8 {
    match resolution {
        gpu::PatchResolution::R33 => 0,
        gpu::PatchResolution::R65 => 1,
        gpu::PatchResolution::R129 => 2,
    }
}

fn adjacent_patch(
    current: gpu::PatchResolution,
    toward: gpu::PatchResolution,
) -> gpu::PatchResolution {
    match (current, patch_rank(toward).cmp(&patch_rank(current))) {
        (gpu::PatchResolution::R33, std::cmp::Ordering::Greater) => gpu::PatchResolution::R65,
        (gpu::PatchResolution::R65, std::cmp::Ordering::Greater) => gpu::PatchResolution::R129,
        (gpu::PatchResolution::R129, std::cmp::Ordering::Less) => gpu::PatchResolution::R65,
        (gpu::PatchResolution::R65, std::cmp::Ordering::Less) => gpu::PatchResolution::R33,
        _ => current,
    }
}

fn projected_tile_span_px(
    key: TileKey,
    radius_m: f64,
    cam_body: DVec3,
    focal_length_px: f64,
) -> f64 {
    let center = key.center_dir() * radius_m;
    focal_length_px * tile_arc_m(key.level, radius_m) / (cam_body - center).length().max(1.0)
}

fn initial_patch_resolution(span_px: f64) -> gpu::PatchResolution {
    if span_px <= PATCH_R33_MAX_PX {
        gpu::PatchResolution::R33
    } else if span_px <= PATCH_R65_MAX_PX {
        gpu::PatchResolution::R65
    } else {
        gpu::PatchResolution::R129
    }
}

fn patch_resolution_for_tile(
    key: TileKey,
    radius_m: f64,
    cam_body: DVec3,
    focal_length_px: Option<f64>,
    current: gpu::PatchResolution,
) -> gpu::PatchResolution {
    let Some(focal_length_px) = focal_length_px.filter(|value| value.is_finite() && *value > 0.0)
    else {
        return current;
    };
    let span = projected_tile_span_px(key, radius_m, cam_body, focal_length_px);
    match current {
        gpu::PatchResolution::R33 => {
            if span > PATCH_R65_MAX_PX * (1.0 + PATCH_HYSTERESIS) {
                gpu::PatchResolution::R129
            } else if span > PATCH_R33_MAX_PX * (1.0 + PATCH_HYSTERESIS) {
                gpu::PatchResolution::R65
            } else {
                current
            }
        }
        gpu::PatchResolution::R65 => {
            if span > PATCH_R65_MAX_PX * (1.0 + PATCH_HYSTERESIS) {
                gpu::PatchResolution::R129
            } else if span < PATCH_R33_MAX_PX * (1.0 - PATCH_HYSTERESIS) {
                gpu::PatchResolution::R33
            } else {
                current
            }
        }
        gpu::PatchResolution::R129 => {
            if span < PATCH_R33_MAX_PX * (1.0 - PATCH_HYSTERESIS) {
                gpu::PatchResolution::R33
            } else if span < PATCH_R65_MAX_PX * (1.0 - PATCH_HYSTERESIS) {
                gpu::PatchResolution::R65
            } else {
                current
            }
        }
    }
}

fn update_patch_lod(
    eye: Res<TileEye>,
    time: Res<Time>,
    roots: Query<&TileTerrainRoot>,
    patch_meshes: Res<gpu::TilePatchMeshes>,
    mut gauge: ResMut<TilePatchLodGauge>,
    mut patches: Query<(
        &mut Mesh3d,
        &mut MeshTag,
        &mut TilePatchLod,
        Option<&TileBodyOrigin>,
    )>,
) {
    // The warmed density matrix owns mesh selection while its probe is active.
    if tile_index_probe_enabled() {
        return;
    }
    let Some(target) = &eye.target else {
        return;
    };
    let Ok(root) = roots.get(target.root) else {
        return;
    };
    *gauge = default();
    let morph_step = PATCH_MORPH_PER_S * time.delta_secs();
    for (mut mesh, mut tag, mut patch, visible_tile) in &mut patches {
        let mut previous_morph = patch.morph;
        let desired = patch_resolution_for_tile(
            patch.key,
            root.radius_m,
            target.cam_body,
            target.vertical_focal_length_px,
            patch.resolution,
        );
        match patch_rank(desired).cmp(&patch_rank(patch.resolution)) {
            std::cmp::Ordering::Greater => {
                // A finer mesh at morph 0 is exactly the current coarse mesh,
                // so the handle swap itself cannot pop. It then reveals the
                // additional samples over subsequent frames.
                let finer = adjacent_patch(patch.resolution, desired);
                mesh.0 = patch_meshes.handle(finer);
                patch.resolution = finer;
                patch.morph = 0.0;
                previous_morph = 0.0;
            }
            std::cmp::Ordering::Less => {
                patch.morph = (patch.morph - morph_step).max(0.0);
                if patch.morph == 0.0 {
                    let coarser = adjacent_patch(patch.resolution, desired);
                    mesh.0 = patch_meshes.handle(coarser);
                    patch.resolution = coarser;
                    // The coarser mesh's exact surface is the finer mesh's
                    // fully-collapsed surface.
                    patch.morph = 1.0;
                    previous_morph = 1.0;
                }
            }
            std::cmp::Ordering::Equal => {
                patch.morph = (patch.morph + morph_step).min(1.0);
            }
        }
        tag.set_if_neq(tile_mesh_tag(patch.gpu_slot, patch.morph, previous_morph));
        if visible_tile.is_some() {
            match patch.resolution {
                gpu::PatchResolution::R33 => gauge.r33 += 1,
                gpu::PatchResolution::R65 => gauge.r65 += 1,
                gpu::PatchResolution::R129 => gauge.r129 += 1,
            }
        }
    }
}

/// Dense and compact handles for the opt-in warmed geometry-density probe.
/// Production processes never build or attach this component.
#[derive(Component)]
#[doc(hidden)]
pub struct TileIndexProbeMeshes {
    pub dense: Handle<Mesh>,
    pub coarse: Handle<Mesh>,
}

/// Full skirt-inflated and tight surface-only bounds for the opt-in warmed
/// culling probe. Production processes never build or attach this component.
#[derive(Component, Clone, Copy)]
#[doc(hidden)]
pub struct TileCullingProbeBounds {
    pub full: Aabb,
    pub surface: Aabb,
}

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
    built: gpu::TileGpuPayload,
    gpu_slot: u32,
    gen_micros: u64,
    /// Sampled height range (m) — diagnostic for flat-terrain regressions.
    h_range: (f32, f32),
    /// The provider's halo height grid, handed to the CPU height mirror so
    /// scatter / colliders read the heights this mesh was built from (see
    /// [`height_mirror`]).
    heights_m: Arc<Vec<f32>>,
}

struct PendingTile {
    task: Task<StreamedTile>,
    gpu_slot: u32,
}

/// Reusable storage for the streaming driver's per-frame work.
///
/// Surface views carry several thousand tiles. Reallocating the selection,
/// balancing, admission, landing, and retirement buffers every frame made the
/// allocator part of the steady-state render loop even when their capacity was
/// identical from one frame to the next. None of these buffers carry state;
/// keeping their allocations beside the root only avoids that churn.
#[derive(Default)]
struct StreamScratch {
    selection_stack: Vec<TileKey>,
    balance_splits: HashSet<TileKey>,
    bridges: HashSet<TileKey>,
    missing: Vec<(TileKey, f64)>,
    landed: Vec<StreamedTile>,
    removable: HashSet<TileKey>,
    expired: Vec<TileKey>,
}

impl StreamScratch {
    fn clear(&mut self) {
        self.selection_stack.clear();
        self.balance_splits.clear();
        self.bridges.clear();
        self.missing.clear();
        self.landed.clear();
        self.removable.clear();
        self.expired.clear();
    }
}

#[derive(Clone, Copy, PartialEq)]
struct SelectionInput {
    cam_body: DVec3,
    motion_arc_m: f64,
    split_scale: f64,
    quality_split_scale: f64,
    vertical_focal_length_px: Option<f64>,
}

/// Marks the shadow-caster twin spawned beside every resident tile (see
/// [`TileTerrainRoot::caster`]).
///
/// Exists purely so the shadow rig's `stability_gauge` can split its per-cascade
/// mesh counts into "terrain caster twins" and "everything else". Those two
/// classes want opposite remedies — twins are few, huge meshes (fewer triangles
/// each is the lever), props and scatter are many, tiny meshes (fewer draws is
/// the lever) — and a single post-cull count cannot tell them apart, so the
/// choice between them was being made by argument instead of measurement
/// (BL-20260731T202656Z).
#[derive(Component, Clone, Copy)]
pub struct TileShadowCaster;

/// Render policy for terrain caster twins. Near cascades reuse the visible
/// mesh; broad cascades receive a separate coarse mesh. Separate layers keep
/// that LOD choice per-view without affecting non-terrain casters.
#[derive(Clone)]
pub struct TileShadowCasterConfig {
    pub material: Handle<material::TileCasterMaterial>,
    pub near_layers: RenderLayers,
    pub far_layers: RenderLayers,
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
    /// When set, every streamed tile also spawns two shadow-only children: the
    /// visible patch LOD on the near-cascade layer and the shared 33² patch on
    /// the broad-cascade layer. Both displace from the tile's atlas slot. This
    /// lets ridges shadow valleys without paying the full 129² surface through
    /// every 4096² view. `None` (default) keeps terrain a pure receiver.
    pub caster: Option<TileShadowCasterConfig>,
    desired: HashSet<TileKey>,
    /// Frozen cold-start selection. Keeping this on the root, rather than in
    /// per-frame scratch storage, prevents camera motion or newly measured
    /// ruggedness from moving the handoff target while it is loading.
    bootstrap: HashSet<TileKey>,
    /// Body-fixed eye used to compute [`Self::bootstrap`]. A teleport that
    /// dwarfs ordinary per-frame motion (parking-orbit → pad) invalidates the
    /// frozen set so loading does not finish a whole-planet cover for a view
    /// the player will never see.
    bootstrap_cam: Option<DVec3>,
    resident: HashMap<TileKey, Entity>,
    pending: HashMap<TileKey, PendingTile>,
    /// Atlas layer paired with every resident entity. Kept separate from the
    /// coverage map so the pure selection/despawn helpers remain concerned
    /// only with topology.
    resident_gpu_slots: HashMap<TileKey, u32>,
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
    recent_despawns: VecDeque<(TileKey, f64, bool)>,
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
    /// Latched true the first time the startup selection is fully resident.
    /// After that, the hole-free despawn rule keeps coverage complete while the
    /// streamer refines to the final desired set, so the impostor↔terrain
    /// swap can trust it without flickering back.
    covered_once: bool,
    /// Budget controller state: multiplier on the split factors, 1.0 = the
    /// unconstrained rule. Driven by [`Self::update_split_scale`].
    split_scale: f64,
    /// Quality-profile multiplier on the same split distances. 1.0 is Showcase;
    /// Laptop writes 0.5. The VRAM brake still multiplies on top.
    quality_split_scale: f64,
    /// Exact inputs used for the current desired set. Body placement is absent
    /// because LOD selection is body-fixed; planet spin must move meshes, not
    /// rebuild the same quadtree.
    selection_input: Option<SelectionInput>,
    /// Set when newly-landed ruggedness or authored refinement changes the
    /// selector without moving the eye.
    selection_dirty: bool,
    /// Body pose used to place all resident tiles. An exactly unchanged pose
    /// means the broad mutable placement query is a no-op.
    placement_pose: Option<(DVec3, DQuat)>,
    scratch: StreamScratch,
    /// Seconds until the next residency gauge line.
    gauge_countdown: f32,
    /// Wall-clock start of this root's first stream, used by the permanent
    /// first-coverage diagnostic. Wall time is intentional: loading latency is
    /// player time, including scheduling and mesh admission stalls.
    installed_at: Instant,
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
            bootstrap: HashSet::new(),
            bootstrap_cam: None,
            resident: HashMap::new(),
            pending: HashMap::new(),
            resident_gpu_slots: HashMap::new(),
            retiring: HashMap::new(),
            coverage_check_countdown: 2.0,
            recent_despawns: VecDeque::new(),
            ruggedness: HashMap::new(),
            relief_m,
            gen_stats: GenStats::default(),
            covered_once: false,
            split_scale: 1.0,
            quality_split_scale: 1.0,
            selection_input: None,
            selection_dirty: true,
            placement_pose: None,
            scratch: StreamScratch::default(),
            gauge_countdown: 0.0,
            installed_at: Instant::now(),
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
    pub fn release_all(&mut self, gpu_store: &mut TileGpuStore) -> Vec<Entity> {
        let entities: Vec<Entity> = self.resident.values().copied().collect();
        self.resident.clear();
        for (_, slot) in self.resident_gpu_slots.drain() {
            gpu_store.release(slot);
        }
        // Dropping a pending task aborts it. Its reserved atlas slot must be
        // returned in the same operation or body handoff leaks capacity.
        for (_, pending) in self.pending.drain() {
            gpu_store.release(pending.gpu_slot);
        }
        self.desired.clear();
        self.bootstrap.clear();
        self.bootstrap_cam = None;
        self.retiring.clear();
        self.recent_despawns.clear();
        self.ruggedness.clear();
        self.refinement_sites.clear();
        self.covered_once = false;
        self.selection_input = None;
        self.selection_dirty = true;
        self.placement_pose = None;
        self.scratch.clear();
        self.installed_at = Instant::now();
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
    pub fn set_refinement_sites(&mut self, sites: &[RefinementSite]) {
        if self.refinement_sites != sites {
            self.refinement_sites.clear();
            self.refinement_sites.extend_from_slice(sites);
            self.selection_dirty = true;
        }
    }

    /// VRAM the landed tile meshes occupy right now.
    pub fn resident_bytes(&self) -> usize {
        self.resident.len() * self.bytes_per_tile()
    }

    /// VRAM already committed: landed tiles plus the in-flight ones that are
    /// going to land. The budget controller's input — using resident alone
    /// would let 24 more tiles arrive after the brake was already needed.
    pub fn committed_bytes(&self) -> usize {
        (self.resident.len() + self.pending.len()) * self.bytes_per_tile()
    }

    fn bytes_per_tile(&self) -> usize {
        TILE_MESH_BYTES
            + if self.caster.is_some() {
                FAR_CASTER_MESH_BYTES
            } else {
                0
            }
    }

    /// Current split-factor multiplier: 1.0 while inside the residency budget,
    /// lower while the brake is holding detail back.
    pub fn split_scale(&self) -> f64 {
        self.split_scale
    }

    /// Quality-profile coarsening, independent of the VRAM brake.
    pub fn quality_split_scale(&self) -> f64 {
        self.quality_split_scale
    }

    /// Combined split multiplier actually used for selection.
    pub fn effective_split_scale(&self) -> f64 {
        (self.split_scale * self.quality_split_scale).clamp(MIN_SPLIT_SCALE, 1.0)
    }

    pub fn set_quality_split_scale(&mut self, scale: f64) {
        let scale = if scale.is_finite() {
            scale.clamp(MIN_SPLIT_SCALE, 1.0)
        } else {
            1.0
        };
        if (self.quality_split_scale - scale).abs() > 1.0e-6 {
            self.quality_split_scale = scale;
            self.selection_dirty = true;
        }
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
        let next = if budget_bytes == usize::MAX {
            1.0
        } else {
            let committed = self.committed_bytes() as f64;
            let budget = budget_bytes as f64;
            if committed > budget {
                (self.split_scale * SPLIT_SCALE_DOWN).max(MIN_SPLIT_SCALE)
            } else if committed < budget * BUDGET_RECOVER_FRACTION && self.split_scale < 1.0 {
                (self.split_scale * SPLIT_SCALE_UP).min(1.0)
            } else {
                self.split_scale
            }
        };
        if next != self.split_scale {
            self.split_scale = next;
            self.selection_dirty = true;
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

    /// True once resident terrain has (ever) fully covered the desired
    /// footprints. The first cover may be the bounded coarse bootstrap; normal
    /// streaming continues to the exact desired leaves after handoff.
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

/// True when the selection eye jumped far enough that a frozen cold-start
/// bootstrap computed at `from` is the wrong planet-scale cover for `to`.
///
/// Deferred placement drops the camera from a ~200 km parking orbit onto the
/// pad. Ordinary per-frame motion is metres to maybe a kilometre; this
/// threshold sits between those. Floor is 20 km so a small body still trips
/// a pad drop; 2 % of radius covers Thalos (~64 km) without chasing LEO
/// drift during a short load.
fn bootstrap_eye_teleported(from: DVec3, to: DVec3, radius_m: f64) -> bool {
    (from - to).length() > (radius_m * 0.02).max(20_000.0)
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
#[derive(Clone, Copy)]
enum HorizonContext {
    Unbounded,
    Limited { cam_dir: DVec3, theta_max: f64 },
}

fn horizon_context(cam_body: DVec3, radius_m: f64, relief_m: f64) -> HorizonContext {
    // No relief bound (a provider without metadata) means "assume anything is
    // up there", i.e. refine as the distance rule alone would. Must be an
    // explicit escape: feeding a huge finite allowance through the maths below
    // does *not* degrade to "always visible", it degrades to nonsense.
    if !relief_m.is_finite() {
        return HorizonContext::Unbounded;
    }
    let cam_len = cam_body.length();
    if cam_len <= radius_m {
        return HorizonContext::Unbounded;
    }
    let top = radius_m + relief_m;
    // Widest angle from the sub-camera point at which ground lifted to `top`
    // still clears the tangent plane: `p · c ≥ r²` with `|p| = top` becomes
    // `cos θ ≥ r² / (top · |c|)`.
    let cos_max = radius_m * radius_m / (top * cam_len);
    if cos_max <= -1.0 {
        return HorizonContext::Unbounded;
    }
    HorizonContext::Limited {
        cam_dir: cam_body / cam_len,
        theta_max: cos_max.min(1.0).acos(),
    }
}

fn above_horizon_in(key: TileKey, horizon: HorizonContext) -> bool {
    let HorizonContext::Limited { cam_dir, theta_max } = horizon else {
        return true;
    };

    // Bound the tile by its cone, NOT by point samples. A tile at MIN_LEVEL
    // spans 589 km on Thalos, so from 756 m up its corners and centre are all
    // far below the horizon while the ground directly beneath the camera —
    // inside the same tile — is in plain view. Sampling points culled every
    // coarse tile and refinement stopped dead at the MIN_LEVEL shell; the cone
    // is conservative for any tile size.
    let centre = key.center_dir();
    let theta_c = centre.dot(cam_dir).clamp(-1.0, 1.0).acos();
    let theta_r = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        .iter()
        .map(|&(s, t)| centre.dot(key.dir_at(s, t)).clamp(-1.0, 1.0).acos())
        .fold(0.0f64, f64::max);
    theta_c - theta_r <= theta_max
}

#[cfg(test)]
fn above_horizon(key: TileKey, cam_body: DVec3, radius_m: f64, relief_m: f64) -> bool {
    above_horizon_in(key, horizon_context(cam_body, radius_m, relief_m))
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
    let mut leaves = HashSet::new();
    let mut stack = Vec::with_capacity(1024);
    let mut balance_splits = HashSet::new();
    select_leaves_scaled_into(
        cam_body,
        radius_m,
        max_level,
        relief_m,
        ruggedness,
        split_scale,
        motion_arc_m,
        sites,
        &mut leaves,
        &mut stack,
        &mut balance_splits,
    );
    leaves
}

#[allow(clippy::too_many_arguments)]
fn select_leaves_scaled_into(
    cam_body: DVec3,
    radius_m: f64,
    max_level: u8,
    relief_m: f64,
    ruggedness: &dyn Fn(TileKey) -> Option<f32>,
    split_scale: f64,
    motion_arc_m: f64,
    sites: &[RefinementSite],
    leaves: &mut HashSet<TileKey>,
    stack: &mut Vec<TileKey>,
    balance_splits: &mut HashSet<TileKey>,
) {
    select_leaves_scaled_with_error_into(
        cam_body,
        radius_m,
        max_level,
        relief_m,
        ruggedness,
        &|_| None,
        None,
        split_scale,
        motion_arc_m,
        sites,
        leaves,
        stack,
        balance_splits,
    );
}

#[allow(clippy::too_many_arguments)]
fn select_leaves_scaled_with_error_into(
    cam_body: DVec3,
    radius_m: f64,
    max_level: u8,
    relief_m: f64,
    ruggedness: &dyn Fn(TileKey) -> Option<f32>,
    refinement_error_m: &dyn Fn(TileKey) -> Option<f32>,
    vertical_focal_length_px: Option<f64>,
    split_scale: f64,
    motion_arc_m: f64,
    sites: &[RefinementSite],
    leaves: &mut HashSet<TileKey>,
    stack: &mut Vec<TileKey>,
    balance_splits: &mut HashSet<TileKey>,
) {
    // Runtime residency policy never falls below MIN_SPLIT_SCALE, but startup
    // deliberately asks this pure selector for a still-coarser, complete cover.
    // Zero remains safe: the MIN_LEVEL shell is always emitted.
    let split_scale = split_scale.clamp(0.0, 1.0);
    let horizon = horizon_context(cam_body, radius_m, relief_m);
    let want_split = |key: TileKey| -> bool {
        if key.level >= max_level {
            return false;
        }
        // Authored floor first: it is unconditional apart from the horizon, so
        // there is nothing the distance rule below could add to the answer.
        if below_site_floor(key, radius_m, sites) {
            return above_horizon_in(key, horizon);
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
        // Package error only gates the ruggedness *boost*. The base distance
        // rule remains the fidelity floor, so metadata can never make terrain
        // coarser than the path that preceded relief-aware selection. Unknown
        // metadata likewise retains the exact old answer.
        if d >= SPLIT_FACTOR * split_scale * arc
            && let Some(focal_length_px) = vertical_focal_length_px
            && let Some(error_m) = refinement_error_m(key)
            // Screen projection uses a conservative nearest-point distance,
            // not the selector's historical centre-distance approximation.
            // `arc` bounds the tile's centre-to-corner chord (pinned below),
            // and relief covers radial displacement above the reference sphere.
            && projected_error_px(
                f64::from(error_m),
                ((cam_body - key.center_dir() * radius_m).length()
                    - arc
                    - relief_m.max(0.0))
                    .max(1.0),
                focal_length_px,
            )
                <= GEOMETRIC_ERROR_THRESHOLD_PX
        {
            return false;
        }
        // Cheapest test last: only worth paying once distance has already said
        // yes. Ground the eye cannot possibly see does not earn refinement.
        above_horizon_in(key, horizon)
    };

    leaves.clear();
    let n = 1u32 << MIN_LEVEL;
    stack.clear();
    stack.reserve(1024usize.saturating_sub(stack.capacity()));
    balance_splits.clear();
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
        for &leaf in leaves.iter() {
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
                            balance_splits.insert(probe);
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
        if balance_splits.is_empty() {
            break;
        }
        for key in balance_splits.drain() {
            leaves.remove(&key);
            for child in key.children() {
                leaves.insert(child);
            }
        }
    }
}

fn projected_error_px(error_m: f64, distance_m: f64, focal_length_px: f64) -> f64 {
    if !error_m.is_finite()
        || error_m < 0.0
        || !distance_m.is_finite()
        || distance_m <= 0.0
        || !focal_length_px.is_finite()
        || focal_length_px <= 0.0
    {
        return f64::INFINITY;
    }
    error_m * focal_length_px / distance_m
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

/// Whether `key` is represented by itself or one resident ancestor.
///
/// Kept separate from recursive descendant coverage because the cold bootstrap
/// calls this while residency is sparse. Recursing an empty level-2 subtree to
/// `max_level` would explore billions of absent keys instead of doing the
/// bounded ancestor walk the startup path actually needs.
fn resident_ancestor_or_self(key: TileKey, resident: &HashMap<TileKey, Entity>) -> bool {
    if resident.contains_key(&key) {
        return true;
    }
    let mut probe = key;
    while let Some(parent) = probe.parent() {
        if parent.level < MIN_LEVEL {
            break;
        }
        probe = parent;
        if resident.contains_key(&probe) {
            return true;
        }
    }
    false
}

/// Whether `key`'s footprint has any complete resident representation: itself,
/// one resident ancestor, or a recursively complete descendant mosaic.
fn footprint_covered_by_resident(
    key: TileKey,
    resident: &HashMap<TileKey, Entity>,
    max_level: u8,
) -> bool {
    resident_ancestor_or_self(key, resident) || covered_by_resident(key, resident, max_level)
}

/// Bridge requests: children of stale resident ancestors whose replacement
/// gap spans more than one level. Only desired tiles are generated otherwise,
/// so a fast approach (selection skipping levels) left a coarse parent
/// waiting for its ENTIRE deep leaf set — up to 4^gap tiles — while its
/// geometry poked through the refined terrain. Bridges make replacement
/// cascade level-by-level: each step releases on just 4 landings, and the
/// visible overlap never spans more than one level (a divergence small
/// enough for the per-level depth bias to hide). Pyramid overhead ≤ ~1/3.
#[cfg(test)]
fn bridge_requests(
    desired: &HashSet<TileKey>,
    resident: &HashMap<TileKey, Entity>,
) -> HashSet<TileKey> {
    let mut bridges = HashSet::new();
    bridge_requests_into(desired, resident, &mut bridges);
    bridges
}

fn bridge_requests_into(
    desired: &HashSet<TileKey>,
    resident: &HashMap<TileKey, Entity>,
    bridges: &mut HashSet<TileKey>,
) {
    bridges.clear();
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
}

/// The despawn decision + retirement bookkeeping, extracted pure so the
/// descent simulation test can drive the exact production logic. Returns the
/// keys whose grace gates expired this tick (the caller despawns them).
#[cfg(test)]
fn despawn_ready(
    desired: &HashSet<TileKey>,
    resident: &HashMap<TileKey, Entity>,
    retiring: &mut HashMap<TileKey, (f32, u16)>,
    max_level: u8,
    dt: f32,
) -> Vec<TileKey> {
    let mut removable = HashSet::new();
    let mut expired = Vec::new();
    despawn_ready_into(
        desired,
        resident,
        retiring,
        max_level,
        dt,
        &mut removable,
        &mut expired,
    );
    expired
}

#[allow(clippy::too_many_arguments)]
fn despawn_ready_into(
    desired: &HashSet<TileKey>,
    resident: &HashMap<TileKey, Entity>,
    retiring: &mut HashMap<TileKey, (f32, u16)>,
    max_level: u8,
    dt: f32,
    removable: &mut HashSet<TileKey>,
    expired: &mut Vec<TileKey>,
) {
    removable.clear();
    removable.extend(
        resident
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
            .copied(),
    );
    retiring.retain(|key, _| removable.contains(key));
    expired.clear();
    for key in removable.drain() {
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
        .filter(|k| !footprint_covered_by_resident(**k, resident, max_level))
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
    patch_meshes: Res<gpu::TilePatchMeshes>,
    mut gpu_store: ResMut<TileGpuStore>,
    patch_lod_gauge: Res<TilePatchLodGauge>,
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
    let mut scratch = std::mem::take(&mut root_ref.scratch);

    // Residency budget: steer the split scale from the VRAM already committed
    // *before* selecting, so a frame that is over budget selects a coarser set
    // rather than committing more first (INC-20260725T012104Z-tile-residency-had-no-budget).
    // The array allocation is finite even when the environment disables the
    // machine-wide byte budget. Its usable occupancy is therefore an
    // unconditional second ceiling, with the final layers reserved for the
    // bridge-before-retire replacement rule.
    let budget_bytes = residency_budget_bytes().min(gpu_store.usable_budget_bytes());
    root_ref.update_split_scale(budget_bytes);

    // Selection reads the measured-ruggedness memo, so mountains hold their
    // mesh out to `SPLIT_FACTOR_RUGGED` while water and plains stay on the
    // plain `SPLIT_FACTOR` rule, and skips refinement below the body's own
    // horizon. Split the borrow: `ruggedness_at` walks `&self` while `desired`
    // is being written.
    let motion_arc_m = target.speed_m_s.max(0.0) * MOTION_CROSS_MIN_S;
    let selection_input = SelectionInput {
        cam_body: cam,
        motion_arc_m,
        split_scale: root_ref.split_scale,
        quality_split_scale: root_ref.quality_split_scale,
        vertical_focal_length_px: package_screen_error_enabled()
            .then_some(target.vertical_focal_length_px)
            .flatten(),
    };
    if root_ref.selection_dirty || root_ref.selection_input != Some(selection_input) {
        let mut desired_now = std::mem::take(&mut root_ref.desired);
        {
            let known: &TileTerrainRoot = root_ref;
            select_leaves_scaled_with_error_into(
                cam,
                radius,
                max_level,
                known.relief_m,
                &|key| known.ruggedness_at(key),
                &|key| known.provider.refinement_error_m(key, radius),
                selection_input.vertical_focal_length_px,
                known.effective_split_scale(),
                motion_arc_m,
                &known.refinement_sites,
                &mut desired_now,
                &mut scratch.selection_stack,
                &mut scratch.balance_splits,
            );
        }
        root_ref.desired = desired_now;
        root_ref.selection_input = Some(selection_input);
        root_ref.selection_dirty = false;
    }

    // Before first handoff, admit only a complete reduced-distance cover. The
    // previous cold path admitted hundreds of final leaves and held the loading
    // screen until every one landed even though a usable ground mosaic needs a
    // fraction of that work. After handoff, bridge tiles keep replacement
    // progressive all the way to the unchanged final selection.
    let desired_fully_resident = root_ref.pending.is_empty()
        && root_ref
            .desired
            .iter()
            .all(|key| root_ref.resident.contains_key(key));
    if !root_ref.covered_once {
        if root_ref
            .bootstrap_cam
            .is_some_and(|from| bootstrap_eye_teleported(from, cam, radius))
        {
            // The frozen cover was for a view the player will never see
            // (parking-orbit placeholder → pad). Drop it and start again at
            // the real eye; leftover meshes would only spend VRAM.
            for (_, pending) in root_ref.pending.drain() {
                gpu_store.release(pending.gpu_slot);
            }
            for entity in root_ref.resident.values().copied() {
                commands.entity(entity).despawn();
            }
            root_ref.resident.clear();
            for (_, slot) in root_ref.resident_gpu_slots.drain() {
                gpu_store.release(slot);
            }
            if let Ok(mut mirror) = root_ref.height_mirror.write() {
                mirror.clear();
            }
            root_ref.bootstrap.clear();
            root_ref.bootstrap_cam = None;
            root_ref.retiring.clear();
            info!("tile terrain: cold-start bootstrap discarded after an eye teleport");
        }
        if root_ref.bootstrap.is_empty() {
            let mut bootstrap = std::mem::take(&mut root_ref.bootstrap);
            let known: &TileTerrainRoot = root_ref;
            select_leaves_scaled_with_error_into(
                cam,
                radius,
                max_level,
                known.relief_m,
                &|key| known.ruggedness_at(key),
                &|key| known.provider.refinement_error_m(key, radius),
                selection_input.vertical_focal_length_px,
                known.effective_split_scale() * BOOTSTRAP_SPLIT_SCALE,
                motion_arc_m,
                &known.refinement_sites,
                &mut bootstrap,
                &mut scratch.selection_stack,
                &mut scratch.balance_splits,
            );
            root_ref.bootstrap = bootstrap;
            root_ref.bootstrap_cam = Some(cam);
        }
        scratch.bridges.clear();
    } else if desired_fully_resident {
        root_ref.bootstrap.clear();
        scratch.bridges.clear();
    } else {
        root_ref.bootstrap.clear();
        bridge_requests_into(&root_ref.desired, &root_ref.resident, &mut scratch.bridges);
    }

    // Cancel pending tiles nobody wants (task drop aborts).
    let desired = &root_ref.desired;
    if root_ref.covered_once {
        root_ref.pending.retain(|key, pending| {
            let keep = desired.contains(key) || scratch.bridges.contains(key);
            if !keep {
                gpu_store.release(pending.gpu_slot);
            }
            keep
        });
    } else {
        root_ref.pending.retain(|key, pending| {
            let keep = root_ref.bootstrap.contains(key);
            if !keep {
                gpu_store.release(pending.gpu_slot);
            }
            keep
        });
    }

    // Admit missing (desired + bridges), screen-size-priority (distance /
    // tile size — absolute nearest-first starves coarse merge-targets; probe
    // M3 finding).
    scratch.missing.clear();
    if !desired_fully_resident || !root_ref.covered_once {
        if root_ref.covered_once {
            scratch.missing.extend(
                root_ref
                    .desired
                    .iter()
                    .chain(scratch.bridges.iter())
                    .filter(|k| {
                        !root_ref.resident.contains_key(k) && !root_ref.pending.contains_key(k)
                    })
                    .map(|&key| {
                        let priority = (cam - key.center_dir() * radius).length()
                            / tile_arc_m(key.level, radius);
                        (key, priority)
                    }),
            );
        } else {
            scratch.missing.extend(
                root_ref
                    .bootstrap
                    .iter()
                    .filter(|k| {
                        !root_ref.resident.contains_key(k) && !root_ref.pending.contains_key(k)
                    })
                    .map(|&key| {
                        let priority = (cam - key.center_dir() * radius).length()
                            / tile_arc_m(key.level, radius);
                        (key, priority)
                    }),
            );
        }
        scratch.missing.sort_by(|a, b| a.1.total_cmp(&b.1));
    }
    let budget = MAX_IN_FLIGHT.saturating_sub(root_ref.pending.len());
    // Dedicated pool: routing this through AsyncComputeTaskPool starves
    // Avian's collider-tree optimisation and hitches the main thread (the
    // documented reason `ground::tile_synthesis_pool` exists).
    let pool = crate::ground::tile_synthesis_pool::tile_synthesis_pool();
    let admit = budget
        .min(scratch.missing.len())
        .min(gpu_store.free_slot_count());
    for (key, _) in scratch.missing.drain(..admit) {
        let gpu_slot = gpu_store
            .allocate()
            .expect("admission count was capped to available terrain atlas slots");
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
            let built = gpu::build_tile_payload(&tile, radius, relief_m);
            StreamedTile {
                key,
                built,
                gpu_slot,
                gen_micros: started.elapsed().as_micros() as u64,
                h_range,
                heights_m: Arc::new(tile.heights_m),
            }
        });
        root_ref.pending.insert(key, PendingTile { task, gpu_slot });
    }
    scratch.missing.clear();

    // Land finished tiles as co-rotating children of the body grid.
    scratch.landed.clear();
    root_ref.pending.retain(|_, pending| {
        if let Some(done) = block_on(poll_once(&mut pending.task)) {
            debug_assert_eq!(done.gpu_slot, pending.gpu_slot);
            scratch.landed.push(done);
            false
        } else {
            true
        }
    });
    for done in scratch.landed.drain(..) {
        root_ref
            .gen_stats
            .record(done.gen_micros, done.h_range, done.built.mesh_h);
        // Feed the relief-aware split rule. `h_range` is the provider's own
        // sampled spread over this tile, so the measurement costs nothing
        // beyond the generation that already happened.
        if done.h_range.0.is_finite() && done.h_range.1.is_finite() {
            let relief = done.h_range.1 - done.h_range.0;
            let arc = tile_arc_m(done.key.level, radius) as f32;
            let ruggedness = relief / arc;
            if root_ref.ruggedness.insert(done.key, ruggedness) != Some(ruggedness) {
                root_ref.selection_dirty = true;
            }
        }
        let origin = done.built.origin;
        let surface_aabb = done.built.surface_aabb;
        let full_aabb = done.built.full_aabb;
        let (cell, local) = place_body_point(origin, target, grid);
        let initial_resolution = if tile_index_probe_enabled() {
            gpu::PatchResolution::R129
        } else {
            target
                .vertical_focal_length_px
                .filter(|value| value.is_finite() && *value > 0.0)
                .map(|focal| {
                    initial_patch_resolution(projected_tile_span_px(done.key, radius, cam, focal))
                })
                .unwrap_or(gpu::PatchResolution::R129)
        };
        let mesh_handle = patch_meshes.handle(initial_resolution);
        let mut tile = commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(root_ref.material.clone()),
            tile_mesh_tag(done.gpu_slot, 1.0, 1.0),
            // Rotation carries the body's spin, so it acts only on the
            // in-tile vertex offsets; the multi-Mm part of the placement is
            // already resolved in f64 above.
            Transform::from_translation(local).with_rotation(body_rotation),
            cell,
            TileBodyOrigin(origin),
            TilePatchLod {
                key: done.key,
                gpu_slot: done.gpu_slot,
                resolution: initial_resolution,
                morph: 1.0,
            },
            root_ref.render_layers.clone(),
            ChildOf(root_entity),
            // The skirt is a crack curtain below this exact surface band; it
            // is visible only when the surface itself is in frame. Letting
            // Bevy derive bounds from the kilometre-deep curtain submitted
            // 114 extra dense tiles in `forest-stand` to both prepass and
            // main. Shadow twins already use this same tight box.
            surface_aabb,
            NoAutoAabb,
        ));
        if tile_index_probe_enabled() {
            tile.insert(TileIndexProbeMeshes {
                dense: patch_meshes.handle(gpu::PatchResolution::R129),
                coarse: patch_meshes.handle(gpu::PatchResolution::R33),
            });
        }
        if tile_cull_probe_enabled() {
            let bounds = TileCullingProbeBounds {
                full: full_aabb,
                surface: surface_aabb,
            };
            tile.insert((bounds, bounds.full));
        }
        // Shadow-caster children (see `TileTerrainRoot::caster`). The near
        // cascades track the visible shared-patch LOD; broad cascades use the
        // fixed shared 33² patch. Both read the same atlas slot, inherit tile
        // placement, and despawn recursively.
        if let Some(caster) = &root_ref.caster {
            tile.with_child((
                Mesh3d(mesh_handle),
                MeshMaterial3d(caster.material.clone()),
                tile_mesh_tag(done.gpu_slot, 1.0, 1.0),
                Transform::IDENTITY,
                caster.near_layers.clone(),
                TileShadowCaster,
                TilePatchLod {
                    key: done.key,
                    gpu_slot: done.gpu_slot,
                    resolution: initial_resolution,
                    morph: 1.0,
                },
                // Explicit tight box, overriding the ~10 km-tall one Bevy would
                // compute from the skirt curtain (see `surface_aabb`). This is
                // the whole of "terrain stops flooding the near cascades".
                surface_aabb,
                // REQUIRED, and the reason the first attempt at this changed
                // nothing: `calculate_bounds` does not merely fill in a missing
                // `Aabb`, it also has an `update_aabb` query that OVERWRITES an
                // existing one from `mesh.compute_aabb()` whenever `Mesh3d` is
                // `Changed` — which is true on the frame the component is
                // inserted. So a hand-authored box is silently replaced by the
                // derived one on the first visibility pass after spawn, and the
                // post-cull counts do not move by a single entity. `NoAutoAabb`
                // excludes the entity from both queries.
                NoAutoAabb,
            ));
            tile.with_child((
                Mesh3d(patch_meshes.handle(gpu::PatchResolution::R33)),
                MeshMaterial3d(caster.material.clone()),
                tile_mesh_tag(done.gpu_slot, 1.0, 1.0),
                Transform::IDENTITY,
                caster.far_layers.clone(),
                TileShadowCaster,
                surface_aabb,
                NoAutoAabb,
            ));
        }
        let entity = tile.id();
        root_ref.resident.insert(done.key, entity);
        root_ref.resident_gpu_slots.insert(done.key, done.gpu_slot);
        gpu_store.upload(done.gpu_slot, done.built);
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
    let placement_pose = (target.body_position, target.body_orientation);
    if root_ref.placement_pose != Some(placement_pose) {
        for (origin, mut cell, mut transform) in &mut placed {
            let (next_cell, local) = place_body_point(origin.0, target, grid);
            cell.set_if_neq(next_cell);
            let scale = transform.scale;
            transform.set_if_neq(
                Transform::from_translation(local)
                    .with_rotation(body_rotation)
                    .with_scale(scale),
            );
        }
        root_ref.placement_pose = Some(placement_pose);
    }

    if !root_ref.covered_once
        && !root_ref.bootstrap.is_empty()
        && root_ref.pending.is_empty()
        && root_ref
            .bootstrap
            .iter()
            .all(|key| root_ref.resident.contains_key(key))
    {
        root_ref.covered_once = true;
        let elapsed_ms = root_ref.installed_at.elapsed().as_secs_f64() * 1_000.0;
        info!(
            target: "thalos::diagnostic::tile_terrain",
            event = "first_coverage",
            elapsed_ms,
            desired = root_ref.desired.len(),
            resident = root_ref.resident.len(),
            pending = root_ref.pending.len(),
            bootstrap = root_ref.bootstrap.len(),
            bootstrap_split_scale = BOOTSTRAP_SPLIT_SCALE,
            "initial tile coverage ready"
        );
        info!(
            "tile terrain: initial coverage ready in {:.2} s ({} tiles); refining in place",
            elapsed_ms / 1_000.0,
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
    despawn_ready_into(
        &root_ref.desired,
        &root_ref.resident,
        &mut root_ref.retiring,
        root_ref.max_level,
        dt,
        &mut scratch.removable,
        &mut scratch.expired,
    );
    for key in scratch.expired.drain(..) {
        if let Some(entity) = root_ref.resident.remove(&key) {
            commands.entity(entity).despawn();
            let slot = root_ref
                .resident_gpu_slots
                .remove(&key)
                .expect("resident terrain tile must own an atlas slot");
            gpu_store.release(slot);
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
                root_ref.recent_despawns.pop_front();
            }
            root_ref.recent_despawns.push_back((key, now, merge_case));
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
            atlas_allocated_mib = mib(gpu::TILE_GPU_ALLOCATED_BYTES),
            atlas_free_slots = gpu_store.free_slot_count(),
            atlas_slots = gpu::TILE_GPU_ATLAS_SLOTS,
            atlas_usable_slots = gpu::TILE_GPU_USABLE_SLOTS,
            patch_lod_33 = patch_lod_gauge.r33,
            patch_lod_65 = patch_lod_gauge.r65,
            patch_lod_129 = patch_lod_gauge.r129,
            pending = root_ref.pending.len(),
            desired = root_ref.desired.len(),
            retiring = root_ref.retiring.len(),
            budget = %budget_note,
            instances = vram_share::live_instances(),
            split_scale = root_ref.split_scale,
            quality_split_scale = root_ref.quality_split_scale,
            effective_split_scale = root_ref.effective_split_scale(),
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
    scratch.clear();
    root_ref.scratch = scratch;
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    const R: f64 = 3_186_000.0;
    const RELIEF: f64 = 9_797.6;

    /// NTR-X12 changes topology, not geometric fidelity: one 129² tile at
    /// L−1 replaces four 65² tiles at L. These are the coupled constants that
    /// make that true. If one moves without the others, the change either
    /// softens terrain or adds vertex work without reducing submissions.
    #[test]
    fn larger_tiles_preserve_the_65_squared_sample_density() {
        const PREVIOUS_TILE_RES: usize = 65;
        const PREVIOUS_MIN_LEVEL: u8 = 3;
        const PREVIOUS_SPLIT_FACTOR: f64 = 6.0;
        const PREVIOUS_RUGGED_SPLIT_FACTOR: f64 = 18.0;

        assert_eq!(TILE_RES, 129);
        assert_eq!(MIN_LEVEL + 1, PREVIOUS_MIN_LEVEL);
        assert_eq!(
            SPLIT_FACTOR * (TILE_RES - 1) as f64,
            PREVIOUS_SPLIT_FACTOR * (PREVIOUS_TILE_RES - 1) as f64
        );
        assert_eq!(
            SPLIT_FACTOR_RUGGED * (TILE_RES - 1) as f64,
            PREVIOUS_RUGGED_SPLIT_FACTOR * (PREVIOUS_TILE_RES - 1) as f64
        );

        let new_level = 7u8;
        let previous_level = new_level + 1;
        let new_key = TileKey {
            face: 0,
            level: new_level,
            x: 0,
            y: 0,
        };
        let face_arc = R * core::f64::consts::FRAC_PI_2;
        let previous_spacing =
            face_arc / (1u32 << previous_level) as f64 / (PREVIOUS_TILE_RES - 1) as f64;
        assert_eq!(new_key.sample_spacing_m(R), previous_spacing);
        assert_eq!(
            4 * (1usize << new_level).pow(2),
            (1usize << previous_level).pow(2),
            "the new level must need one quarter as many tile footprints"
        );

        let previous_max_level = ((face_arc / ((PREVIOUS_TILE_RES - 1) as f64 * 9.0))
            .log2()
            .ceil() as u8)
            .clamp(PREVIOUS_MIN_LEVEL + 1, 18);
        assert_eq!(max_level_for(R) + 1, previous_max_level);
        assert!(TILE_VERTS < u16::MAX as usize);
    }

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

    /// GPU displacement must reproduce the retired CPU mesh exactly. This is
    /// the precision contract: the atlas carries f64-built local positions;
    /// the vertex shader does not reconstruct a multi-megametre sphere in f32.
    #[test]
    fn tile_gpu_payload_matches_the_reference_mesh() {
        let source = flat_tile(TileKey {
            face: 0,
            level: 6,
            x: 3,
            y: 5,
        });
        let built = build_tile_mesh(&source, R, RELIEF, false);
        let payload = gpu::build_tile_payload(&source, R, RELIEF);
        assert_eq!(payload.position_bytes.len(), gpu::TILE_GPU_POSITION_BYTES);
        assert_eq!(payload.surface_bytes.len(), gpu::TILE_GPU_SURFACE_BYTES);
        assert_eq!(TILE_MESH_BYTES, gpu::TILE_GPU_SLOT_BYTES);

        let atlas_positions: &[[f32; 4]] = bytemuck::cast_slice(&payload.position_bytes);
        let mesh_positions = built
            .mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|values| values.as_float3())
            .expect("reference mesh positions");
        let side = SurfaceTile::grid_side();
        for j in 0..TILE_RES {
            for i in 0..TILE_RES {
                let mesh = mesh_positions[j * TILE_RES + i];
                let atlas = atlas_positions[(j + TILE_HALO) * side + i + TILE_HALO];
                assert_eq!(mesh, atlas[..3]);
            }
        }
        for skirt in 0..TILE_SKIRT_VERTS {
            assert_eq!(
                mesh_positions[TILE_RES * TILE_RES + skirt],
                atlas_positions[side * side + skirt][..3],
            );
        }
        assert_eq!(payload.origin, built.origin);
        assert_eq!(payload.surface_aabb, built.surface_aabb);
    }

    #[test]
    fn index_density_probe_builds_a_compact_full_attribute_twin() {
        let positions = (0..TILE_VERTS)
            .map(|index| [index as f32, 0.0, 0.0])
            .collect::<Vec<_>>();
        let normals = vec![[0.0, 1.0, 0.0]; TILE_VERTS];
        let colors = vec![[0.5, 0.4, 0.3, 0.2]; TILE_VERTS];
        let uv0 = vec![[0.1, 0.2]; TILE_VERTS];
        let uv1 = vec![[0.3, 0.4]; TILE_VERTS];
        let probe = build_tile_index_probe_mesh(&positions, &normals, &colors, &uv0, &uv1);

        assert_eq!(probe.count_vertices(), FAR_CASTER_VERTS);
        assert_eq!(
            probe.get_vertex_buffer_size(),
            FAR_CASTER_VERTS * TILE_VERTEX_BYTES
        );
        assert!(
            matches!(probe.indices(), Some(Indices::U16(indices)) if indices.len() == FAR_CASTER_STRIP_INDICES)
        );
        let compact_positions = probe
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|attribute| attribute.as_float3())
            .expect("probe positions");
        for j in 0..FAR_CASTER_RES {
            for i in 0..FAR_CASTER_RES {
                let compact = j * FAR_CASTER_RES + i;
                let dense = (j * FAR_CASTER_STEP) * TILE_RES + i * FAR_CASTER_STEP;
                assert_eq!(compact_positions[compact], positions[dense]);
            }
        }
        assert_eq!(
            compact_positions[FAR_CASTER_RES * FAR_CASTER_RES],
            positions[TILE_RES * TILE_RES]
        );
        assert!(probe.attribute(Mesh::ATTRIBUTE_COLOR).is_some());
        assert!(probe.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
        assert!(probe.attribute(Mesh::ATTRIBUTE_UV_1).is_some());
    }

    #[test]
    fn far_caster_mesh_is_budgeted_and_samples_visible_tile_boundaries() {
        let caster = gpu::build_patch_mesh(gpu::PatchResolution::R33);
        assert_eq!(FAR_CASTER_MESH_BYTES, 0);
        assert_eq!(
            caster.primitive_topology(),
            PrimitiveTopology::TriangleStrip
        );
        assert!(
            matches!(caster.indices(), Some(Indices::U16(indices)) if indices.len() == FAR_CASTER_STRIP_INDICES)
        );

        let coarse = caster
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|attribute| attribute.as_float3())
            .expect("far caster has atlas addresses");
        for j in 0..FAR_CASTER_RES {
            for i in 0..FAR_CASTER_RES {
                let coarse_index = j * FAR_CASTER_RES + i;
                assert_eq!(
                    coarse[coarse_index][0] as usize,
                    TILE_HALO + i * FAR_CASTER_STEP
                );
                assert_eq!(
                    coarse[coarse_index][1] as usize,
                    TILE_HALO + j * FAR_CASTER_STEP
                );
            }
        }
    }

    /// The compact strip is only an encoding change: it covers every core and
    /// skirt triangle from the former list with the same diagonal and winding.
    /// Canonicalising by cyclic rotation preserves winding while making draw
    /// order irrelevant.
    #[test]
    fn tile_strip_matches_the_original_triangle_list() {
        fn canonical([a, b, c]: [u32; 3]) -> [u32; 3] {
            if a <= b && a <= c {
                [a, b, c]
            } else if b <= c {
                [b, c, a]
            } else {
                [c, a, b]
            }
        }

        let built = build_tile_mesh(
            &flat_tile(TileKey {
                face: 0,
                level: 6,
                x: 3,
                y: 5,
            }),
            R,
            RELIEF,
            false,
        );
        let twin = crate::rt::rt_twin_of_tile(&built.mesh).expect("strip expands");
        let Indices::U32(actual_indices) = twin.indices().expect("RT twin is indexed") else {
            panic!("RT twin must use U32 indices");
        };
        let mut actual: Vec<_> = actual_indices
            .chunks_exact(3)
            .map(|triangle| canonical([triangle[0], triangle[1], triangle[2]]))
            .collect();

        let res = TILE_RES;
        let mut expected = Vec::with_capacity(TILE_TRIANGLES);
        for j in 0..res - 1 {
            for i in 0..res - 1 {
                let a = (j * res + i) as u32;
                let b = a + 1;
                let c = a + res as u32;
                let d = c + 1;
                expected.push(canonical([a, b, d]));
                expected.push(canonical([a, d, c]));
            }
        }
        let mut border = Vec::with_capacity(TILE_SKIRT_VERTS);
        for i in 0..res {
            border.push(i as u32);
        }
        for j in 1..res {
            border.push((j * res + res - 1) as u32);
        }
        for i in (0..res - 1).rev() {
            border.push(((res - 1) * res + i) as u32);
        }
        for j in (1..res - 1).rev() {
            border.push((j * res) as u32);
        }
        let base = (res * res) as u32;
        for k in 0..border.len() {
            let k2 = (k + 1) % border.len();
            let (top_a, top_b) = (border[k], border[k2]);
            let (bot_a, bot_b) = (base + k as u32, base + k2 as u32);
            expected.push(canonical([top_a, bot_a, bot_b]));
            expected.push(canonical([top_a, bot_b, top_b]));
        }

        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual.len(), TILE_TRIANGLES);
        assert_eq!(actual, expected);
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
            false,
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
        assert_eq!(RT_TILE_MESH_BYTES / 1024, 1_200, "documented as 1,200 KiB");
        assert_eq!(TILE_MESH_BYTES, 351_604, "documented atlas slot payload");
        let ratio = (TILE_MESH_BYTES + RT_TILE_MESH_BYTES) as f64 / TILE_MESH_BYTES as f64;
        assert!(
            (ratio - 4.50).abs() < 0.01,
            "documented as about 4.50x, computed {ratio:.3}"
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

    #[test]
    fn reusable_selection_storage_preserves_the_exact_leaf_set() {
        let max_level = max_level_for(R);
        let cam = DVec3::new(0.31, 0.72, 0.62).normalize() * (R + 2_000.0);
        let expected = select_leaves_scaled(cam, R, max_level, RELIEF, &|_| None, 1.0, 0.0, &[]);

        let mut reused = HashSet::new();
        let mut stack = Vec::new();
        let mut balance_splits = HashSet::new();
        for _ in 0..2 {
            select_leaves_scaled_into(
                cam,
                R,
                max_level,
                RELIEF,
                &|_| None,
                1.0,
                0.0,
                &[],
                &mut reused,
                &mut stack,
                &mut balance_splits,
            );
            assert_eq!(reused, expected);
        }
    }

    fn select_with_package_error(
        cam: DVec3,
        error: &dyn Fn(TileKey) -> Option<f32>,
        focal_length_px: Option<f64>,
        sites: &[RefinementSite],
    ) -> HashSet<TileKey> {
        let mut leaves = HashSet::new();
        let mut stack = Vec::new();
        let mut balance_splits = HashSet::new();
        select_leaves_scaled_with_error_into(
            cam,
            R,
            max_level_for(R),
            RELIEF,
            &|_| Some(RUGGED_HI),
            error,
            focal_length_px,
            1.0,
            0.0,
            sites,
            &mut leaves,
            &mut stack,
            &mut balance_splits,
        );
        leaves
    }

    #[test]
    fn missing_package_error_preserves_the_exact_heuristic_selection() {
        let cam = DVec3::new(0.31, 0.72, 0.62).normalize() * (R + 22_000.0);
        let old = select_leaves_scaled(
            cam,
            R,
            max_level_for(R),
            RELIEF,
            &|_| Some(RUGGED_HI),
            1.0,
            0.0,
            &[],
        );

        assert_eq!(
            old,
            select_with_package_error(cam, &|_| None, Some(1_300.0), &[])
        );
    }

    #[test]
    fn subpixel_package_error_removes_only_the_ruggedness_boost() {
        let cam = DVec3::new(0.31, 0.72, 0.62).normalize() * (R + 22_000.0);
        let boosted = select_with_package_error(cam, &|_| None, Some(1_300.0), &[]);
        let trimmed = select_with_package_error(cam, &|_| Some(0.0), Some(1_300.0), &[]);
        let base = select_leaves_scaled(cam, R, max_level_for(R), RELIEF, &|_| None, 1.0, 0.0, &[]);

        assert!(trimmed.len() < boosted.len());
        assert_eq!(
            trimmed, base,
            "package metadata crossed the base-detail floor"
        );
    }

    #[test]
    fn visible_package_error_and_authored_sites_keep_their_detail() {
        let cam = DVec3::new(0.31, 0.72, 0.62).normalize() * (R + 160_000.0);
        let heuristic = select_with_package_error(cam, &|_| None, Some(1_300.0), &[]);
        let visible = select_with_package_error(cam, &|_| Some(1.0e6), Some(1_300.0), &[]);
        assert_eq!(visible, heuristic);

        let site = RefinementSite {
            center_dir: cam.normalize(),
            angular_radius: 500.0 / R,
            spacing_m: 30.0,
        };
        let with_site = select_with_package_error(cam, &|_| Some(0.0), Some(1_300.0), &[site]);
        let finest_at_site = with_site
            .iter()
            .filter(|key| site.overlaps(**key))
            .map(|key| key.sample_spacing_m(R))
            .fold(f64::INFINITY, f64::min);
        assert!(finest_at_site <= site.spacing_m);
    }

    #[test]
    fn screen_space_error_responds_to_distance_and_physical_resolution() {
        assert_eq!(projected_error_px(10.0, 1_000.0, 1_000.0), 10.0);
        assert!(
            projected_error_px(10.0, 2_000.0, 1_000.0) < projected_error_px(10.0, 1_000.0, 1_000.0)
        );
        assert!(
            projected_error_px(10.0, 1_000.0, 2_000.0) > projected_error_px(10.0, 1_000.0, 1_000.0)
        );
        assert!(projected_error_px(f64::NAN, 1_000.0, 1_000.0).is_infinite());
    }

    #[test]
    fn tile_arc_bounds_the_screen_error_distance_footprint() {
        for level in MIN_LEVEL..=18 {
            let side = 1u32 << level;
            let coordinates = [0, side / 2, side - 1];
            for face in 0..6 {
                for x in coordinates {
                    for y in coordinates {
                        let key = TileKey { face, level, x, y };
                        let center = key.center_dir() * R;
                        let corner_radius = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
                            .into_iter()
                            .map(|(s, t)| (key.dir_at(s, t) * R - center).length())
                            .fold(0.0f64, f64::max);
                        assert!(
                            corner_radius <= tile_arc_m(level, R),
                            "face {face} L{level} ({x},{y}) corner {corner_radius:.3} m exceeds arc"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn package_error_reduces_mira_rugged_selection_without_crossing_the_base_rule() {
        use std::path::Path;
        use thalos_terrain::{
            DynamicSurfaceState, PackageSurface, PlanetSurface, load_static_package_artifact,
        };

        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/terrain_packages/Mira.bin");
        let loaded = load_static_package_artifact(&path, "Mira").unwrap();
        let radius = f64::from(loaded.manifest.body_radius_m);
        let surface = Arc::new(PackageSurface::new(
            loaded.manifest,
            PlanetSurface {
                static_surface: loaded.static_surface,
                dynamic_layers: Default::default(),
                tectonics: None,
            },
            DynamicSurfaceState::default(),
        ));
        let provider = SurfaceQueryProvider { surface };
        let cam = DVec3::new(0.31, 0.72, 0.62).normalize() * (radius + 46_000.0);
        let relief = f64::from(provider.height_range_m());
        let max_level = max_level_for(radius);
        let old = select_leaves_scaled(
            cam,
            radius,
            max_level,
            relief,
            &|_| Some(RUGGED_HI),
            1.0,
            0.0,
            &[],
        );
        let base = select_leaves_scaled(cam, radius, max_level, relief, &|_| None, 1.0, 0.0, &[]);
        let mut bounded = HashSet::new();
        let mut stack = Vec::new();
        let mut balance_splits = HashSet::new();
        select_leaves_scaled_with_error_into(
            cam,
            radius,
            max_level,
            relief,
            &|_| Some(RUGGED_HI),
            &|key| provider.refinement_error_m(key, radius),
            Some(1_303.0),
            1.0,
            0.0,
            &[],
            &mut bounded,
            &mut stack,
            &mut balance_splits,
        );

        assert!(bounded.len() < old.len());
        assert!(bounded.len() >= base.len());
    }

    #[test]
    fn surface_provider_maps_renderer_rows_to_package_rows() {
        struct RowSurface;
        impl SurfaceQuery for RowSurface {
            fn sample(&self, _dir: Vec3, _lod_m: f32) -> thalos_terrain::query::SurfaceSample {
                thalos_terrain::query::SurfaceSample {
                    height_m: 0.0,
                    albedo_linear: Vec3::ZERO,
                    roughness: 1.0,
                    moisture: 0.0,
                }
            }

            fn radius_m(&self) -> f32 {
                R as f32
            }

            fn height_range_m(&self) -> f32 {
                RELIEF as f32
            }

            fn refinement_error_m(
                &self,
                patch: SurfacePatch,
                _refined_spacing_m: f32,
            ) -> Option<f32> {
                Some(patch.y as f32)
            }
        }

        let provider = SurfaceQueryProvider {
            surface: Arc::new(RowSurface),
        };
        let key = TileKey {
            face: 0,
            level: 3,
            x: 4,
            y: 2,
        };
        assert_eq!(provider.refinement_error_m(key, R), Some(5.0));
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
            let built = build_tile_mesh(&provider.request(key, R), R, RELIEF, false);
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
        // And it stays cheap. The 128-leaf ceiling includes the pad's own tiles
        // plus the 2:1 balance cascade down to them. At 1,008 KiB each it is
        // under 4 % of the residency budget. If this bound ever fails, the floor
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
    /// MIN_LEVEL spans more than 1,000 km on Thalos, so from a few hundred metres up its
    /// corners and centre all sit far below the horizon even though the patch
    /// directly beneath the camera does not — the bug that froze refinement at
    /// the coarsest MIN_LEVEL shell and made every framing render as a smooth
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
        let built = build_tile_mesh(&tile, R, RELIEF, false);
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

    /// The culling box handed to the shadow-caster twin must describe the
    /// SURFACE, not the curtain. Bevy's own `calculate_bounds` derives an `Aabb`
    /// from every position in the mesh, so with the floor-sphere skirt that box
    /// is `relief` deep — kilometres — for a tile a few hundred metres across,
    /// and a slab that shape intersects nearly any frustum aimed at the body.
    /// This asserts the two boxes actually differ, so a regression that lets the
    /// skirt back into the caster's bounds fails here rather than silently
    /// reinstating the flood (BL-20260731T202656Z).
    /// Deliberately a FINE tile. The curtain is a fixed `relief`-deep drop while
    /// a tile's own width halves every level, so the slab only becomes
    /// pathological at depth: at level 6 the tile is ~78 km across and a 2.5 km
    /// curtain is noise (the two boxes agree to 0.4 %), while at level 14 it is
    /// ~300 m across and the curtain is an order of magnitude deeper than the
    /// tile is wide. The near cascades are packed with exactly these fine tiles,
    /// which is why the flood showed up there first.
    #[test]
    fn visible_and_caster_bounds_describe_the_surface_not_the_curtain() {
        const RELIEF: f64 = 2_500.0;
        let key = TileKey {
            face: 0,
            level: 14,
            x: 6_400,
            y: 10_752,
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
        use bevy::camera::primitives::MeshAabb;
        let built = build_tile_mesh(&tile, R, RELIEF, false);
        let whole = built
            .mesh
            .compute_aabb()
            .expect("a freshly built tile still carries CPU positions");

        // Deepest axis of each box — the curtain runs radially, and the tile's
        // own orientation on the cube face decides which axis that lands on.
        let surface_depth = built.surface_aabb.half_extents.max_element();
        let whole_depth = whole.half_extents.max_element();
        assert!(
            surface_depth * 4.0 < whole_depth,
            "the tile box is not meaningfully tighter than the full mesh box \
             (surface {surface_depth:.1} m vs whole {whole_depth:.1} m half-extent) — \
             either the skirt is inside `surface_aabb` or the curtain stopped hanging"
        );
        // And it must still CONTAIN the surface: a box that culls correctly but
        // clips its own tile is the other way to fail this.
        let positions = built
            .mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|a| a.as_float3())
            .expect("tile meshes carry positions");
        let lo = built.surface_aabb.min();
        let hi = built.surface_aabb.max();
        for (n, p) in positions.iter().take(TILE_RES * TILE_RES).enumerate() {
            for axis in 0..3 {
                assert!(
                    p[axis] >= lo[axis] - 1.0e-3 && p[axis] <= hi[axis] + 1.0e-3,
                    "surface vertex {n} falls outside the tile box on axis {axis}"
                );
            }
        }
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    #[test]
    fn bootstrap_cuts_real_thalos_cold_admission_by_at_least_half() {
        const RADIUS_M: f64 = 3_186_000.0;
        const RELIEF_M: f64 = 9_797.6;
        let cam = DVec3::new(0.31, 0.72, 0.62).normalize() * (RADIUS_M + 1_000.0);
        // Laptop is the macOS startup default. A cold root has no ruggedness
        // measurements yet, matching the first production selection exactly.
        let desired = select_leaves_scaled(
            cam,
            RADIUS_M,
            max_level_for(RADIUS_M),
            RELIEF_M,
            &|_| None,
            0.5,
            0.0,
            &[],
        );
        let bootstrap = select_leaves_scaled(
            cam,
            RADIUS_M,
            max_level_for(RADIUS_M),
            RELIEF_M,
            &|_| None,
            0.5 * BOOTSTRAP_SPLIT_SCALE,
            0.0,
            &[],
        );
        let resident: HashMap<_, _> = bootstrap
            .iter()
            .copied()
            .map(|key| (key, Entity::PLACEHOLDER))
            .collect();
        let cam_dir = cam.normalize();
        let final_nadir = desired
            .iter()
            .find(|key| TileKey::containing_dir(cam_dir, key.level) == **key)
            .expect("final selection covers the camera direction");
        let bootstrap_nadir = bootstrap
            .iter()
            .find(|key| TileKey::containing_dir(cam_dir, key.level) == **key)
            .expect("bootstrap selection covers the camera direction");

        eprintln!(
            "Thalos cold admission: {} L{} final leaves -> {} L{} bootstrap tiles",
            desired.len(),
            final_nadir.level,
            bootstrap.len(),
            bootstrap_nadir.level,
        );
        assert!(
            bootstrap.len() * 2 <= desired.len(),
            "bootstrap no longer removes at least half of initial mesh admissions"
        );
        assert_eq!(
            bootstrap_nadir.level, final_nadir.level,
            "bootstrap reduced the ground detail directly under the camera"
        );
        assert!(
            desired
                .iter()
                .all(|&key| resident_ancestor_or_self(key, &resident)),
            "bootstrap is not a complete ancestor cover of the final selection"
        );
    }

    #[test]
    fn parking_orbit_to_pad_invalidates_frozen_bootstrap() {
        const RADIUS_M: f64 = 3_186_000.0;
        let pad = DVec3::Y * (RADIUS_M + 5.0);
        let parking = DVec3::Y * (RADIUS_M + 200_000.0);
        assert!(
            bootstrap_eye_teleported(parking, pad, RADIUS_M),
            "a 200 km pad drop must discard the orbital bootstrap"
        );
        let drift = DVec3::Y * (RADIUS_M + 200_000.0) + DVec3::X * 1_000.0;
        assert!(
            !bootstrap_eye_teleported(parking, drift, RADIUS_M),
            "a kilometre of ordinary motion must not restart the cold cover"
        );
    }

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
        let select = |cam: DVec3, memo: &HashMap<TileKey, f32>, scale: f64| -> HashSet<TileKey> {
            select_leaves_scaled(
                cam,
                RADIUS_M,
                max_level,
                RELIEF_M,
                &|key| {
                    rugged_field?;
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
                },
                scale,
                0.0,
                &[],
            )
        };
        // (key, seconds-until-landed); at most MAX_IN_FLIGHT actively count
        // down, the rest queue — mirrors the bounded pool.
        let mut pending: Vec<(TileKey, f32)> = Vec::new();
        let mut bootstrap: HashSet<TileKey> = HashSet::new();
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
                desired = select(cam, &rugged_memo, 1.0);
                last_cam = cam;
                last_memo_len = memo_len;
            }
            let bridges = if covered_once {
                bridge_requests(&desired, &resident)
            } else {
                if bootstrap.is_empty() {
                    bootstrap = select(cam, &rugged_memo, BOOTSTRAP_SPLIT_SCALE);
                }
                HashSet::new()
            };

            // Cancel pendings nobody wants (production: Task drop).
            let (keep, cancelled): (Vec<_>, Vec<_>) = pending.into_iter().partition(|(key, _)| {
                if covered_once {
                    desired.contains(key) || bridges.contains(key)
                } else {
                    bootstrap.contains(key)
                }
            });
            pending = keep;
            for (key, _) in cancelled {
                events.push((t, "cancel", key));
            }

            // Admit missing (desired + bridges) by screen-size priority up to
            // the in-flight cap.
            let requests: Vec<TileKey> = if covered_once {
                desired.iter().chain(bridges.iter()).copied().collect()
            } else {
                bootstrap.iter().copied().collect()
            };
            let mut missing: Vec<TileKey> = requests
                .iter()
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
                && !bootstrap.is_empty()
                && pending.is_empty()
                && bootstrap.iter().all(|key| resident.contains_key(key))
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
        let desired = select(cam_at(sim_end_s), &rugged_memo, 1.0);
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
        app.add_plugins((
            gpu::TileGpuPlugin,
            bevy::pbr::MaterialPlugin::<material::TileTerrainMaterial>::default(),
            bevy::pbr::MaterialPlugin::<material::TileCasterMaterial>::default(),
        ));
        app.init_resource::<TileEye>()
            .init_resource::<TilePatchLodGauge>()
            .add_systems(
                Update,
                (stream_tile_terrain, update_patch_lod)
                    .chain()
                    .in_set(TileStreamSet),
            )
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
