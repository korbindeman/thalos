//! Grass-blade decoration layer for vegetated bodies.
//!
//! A camera-local grid of body-fixed **grass tiles** (~25 m tangent squares on
//! the shared cube-sphere [`TileLattice`](crate::ground::tile_lattice)). Each
//! tile is one batched [`Mesh`] of a few thousand tapered blade strips, built
//! off-thread by sampling the body's [`HeightSource`] — the same source the
//! terrain collider uses, so blades sit on the rendered ground by construction.
//! Blade placement runs through the shared
//! [`placement_gate`](crate::ground::scatter::placement_gate) (the slope /
//! curvature material-mask gate the terrain shader bakes from), plus
//! above-sea-level and altitude-band gates, so grass appears where the ground
//! *looks* like grassland.
//!
//! This module is pure geometry + material types; the per-frame tile lifecycle
//! (which tiles exist, big_space anchoring, wind/lighting updates) is driven by
//! the game crate (`thalos_game::rendering::grass`).

use std::sync::Arc;

use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::math::{DVec3, Vec4};
use bevy::mesh::{Indices, Mesh, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;
use thalos_terrain::TerrainFlatten;

use crate::ground::body_material::ShadowCascadeBlock;
use crate::ground::height_source::HeightSource;
use crate::ground::rendered_height::TerrainPatchBasis;
use crate::ground::scatter::placement_gate;
use crate::ground::tile_lattice::{self, TileLattice, cube_dir};

/// Nominal metric side of one grass tile at a cube-face centre. Toward face
/// corners the cube projection shrinks tiles laterally (down to ~1/2 side);
/// the builder compensates by computing blade count from the tile's actual
/// metric area.
pub const GRASS_TILE_SIZE_M: f64 = 25.0;

/// `tile_lod_m` hint for height sampling — small enough to engage the full
/// near-field procedural cascade, matching what the player stands on.
const GRASS_SAMPLE_LOD_M: f32 = 0.5;

/// Hard ceiling on placement *candidates* per tile, against pathological
/// density configs. Final blade count is this × `blades_per_clump`.
const MAX_BLADES_PER_TILE: usize = 8192;

/// Sink blade roots slightly below the sampled surface so bilinear
/// height-mirror error can't leave a row of floating root quads.
const ROOT_SINK_M: f64 = 0.03;

// Altitude gate for grass density: fades out approaching the treeline (where
// the terrain shader paints dry alpine scree).
const GRASS_FADE_LO_M: f32 = 2400.0;
const GRASS_FADE_HI_M: f32 = 3100.0;

// Fallback flat blade tint for the HeightSource-free standalone clump/field
// preview assets. The in-game tiles colour per-clump from the shared
// [`crate::ground::landcover`] field instead (so blades match the terrain's
// large-scale palette); this anchor is only for the previews.
const C_GRASS: Vec3 = Vec3::new(0.078, 0.112, 0.052);

// ---------------------------------------------------------------------------
// Lattice wrappers
// ---------------------------------------------------------------------------
//
// The cube-sphere lattice math now lives in `tile_lattice`, shared with the
// shrub/tree scatter system. These thin wrappers preserve the grass driver's
// existing `grass_*` call sites and the `GrassTileKey` name.

/// One grass tile on the body's cube-sphere lattice (see
/// [`tile_lattice::TileKey`]).
pub use crate::ground::tile_lattice::TileKey as GrassTileKey;

/// Tiles along one cube-face edge for a body, sized so a tile at the face
/// centre is ~`tile_size_m` across.
pub fn grass_tiles_per_side(radius_m: f64, tile_size_m: f64) -> i64 {
    tile_lattice::tiles_per_side(radius_m, tile_size_m)
}

/// Tile containing a body-fixed unit direction.
pub fn grass_tile_key(dir: DVec3, tiles_per_side: i64) -> GrassTileKey {
    TileLattice { tiles_per_side }.key_of(dir)
}

/// Centre direction + tangent basis of a tile. Returns `None` for keys outside
/// the face grid (callers enumerate raw neighbour offsets).
pub fn grass_tile_frame(
    key: GrassTileKey,
    tiles_per_side: i64,
) -> Option<(DVec3, TerrainPatchBasis)> {
    TileLattice { tiles_per_side }.frame(key)
}

// ---------------------------------------------------------------------------
// Tile mesh builder
// ---------------------------------------------------------------------------

/// Blade geometry level of detail. Near rings use the full curved 7-vertex
/// blade; far rings use a cheap 3-vertex flat blade, widened into a clump so
/// ground coverage holds as per-blade density drops (the constant-coverage
/// rule — see `docs/vegetation.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrassBladeLod {
    /// 7 vertices, curved + tapered. Near ring.
    Full,
    /// 3 vertices, flat triangle. Far rings, widened via `width_scale`.
    Wide,
}

/// Everything [`build_grass_tile_mesh`] needs, snapshotted by the game-side
/// driver so the build can run on the async compute pool without touching ECS.
pub struct GrassTileBuildInput {
    pub key: GrassTileKey,
    pub tiles_per_side: i64,
    pub height_source: Arc<dyn HeightSource>,
    pub radius_m: f64,
    /// Sea level (m above reference radius); blades require
    /// `height > sea_level + 1 m`. Pass `f32::MIN` for bodies without oceans.
    pub sea_level_m: f32,
    pub blades_per_m2: f32,
    /// Blades emitted per accepted placement point — a fanned tuft. Multiplies
    /// ground coverage *without* re-paying the placement gate (the expensive
    /// part), so a sparse scatter reads as a thick field. Far rings use 1.
    pub blades_per_clump: u32,
    /// Blade geometry LOD for this clipmap ring.
    pub blade_lod: GrassBladeLod,
    /// Width multiplier — coarser rings widen blades to hold ground coverage
    /// as density drops.
    pub width_scale: f32,
    /// Height multiplier — far blades a touch taller so the field reads at
    /// grazing angles.
    pub height_scale: f32,
    pub seed: u64,
    /// Active terrain-flatten pad (e.g. the runway). Blades are skipped where
    /// the pad has meaningful weight so they never poke through the paving.
    pub flatten_exclusion: Option<TerrainFlatten>,
    /// Far-ring forest cull strength in `[0, 1]`. Distant grass under a tree
    /// canopy is occluded by the trees in front of it, so the far clipmap rings
    /// thin grass by `forest_cull × forest_coverage(dir)` — full grass on open
    /// plains, ~none deep inside a grove. `0` (near rings) keeps all grass.
    pub forest_cull: f32,
}

/// A finished grass tile: one batched mesh whose vertex positions are small
/// f32 offsets from `center_surface_body_m` (body-fixed, metres), so the
/// game-side anchor can re-pose the tile in f64 every frame and the f32
/// `Transform.rotation` only ever acts on ≤ ~20 m offsets.
pub struct GrassTileMesh {
    pub mesh: Mesh,
    pub center_surface_body_m: DVec3,
    pub blade_count: u32,
    /// `HeightSource::revision()` at build time, for staleness checks.
    pub built_revision: u64,
    /// Sampled terrain height at the tile centre, for cheap rebuild gating.
    pub center_height_m: f32,
}

/// Build one grass tile. Pure (deterministic for a given input + source
/// state); intended to run on `AsyncComputeTaskPool`. Returns `None` when no
/// blade passes the placement gates (open water, rock, alpine, flattened pad).
pub fn build_grass_tile_mesh(input: &GrassTileBuildInput) -> Option<GrassTileMesh> {
    let lattice = TileLattice {
        tiles_per_side: input.tiles_per_side,
    };
    let (center_dir, basis) = lattice.frame(input.key)?;
    let source = input.height_source.as_ref();
    let built_revision = source.revision();

    let center_height_m = source.sample_height_m(center_dir.as_vec3(), GRASS_SAMPLE_LOD_M)?;
    let center_surface_body_m = center_dir * (input.radius_m + center_height_m as f64);

    // Metric extents from the actual uv span (cube distortion shrinks tiles
    // toward face corners), so blade density stays uniform per square metre.
    let (u_lo, u_hi, v_lo, v_hi) = lattice.uv_span(input.key);
    let (ext_u_m, ext_v_m) = lattice.tile_extents_m(input.key, input.radius_m);
    let area_m2 = (ext_u_m * ext_v_m).max(0.0);
    let candidate_count =
        ((area_m2 * input.blades_per_m2 as f64).round() as usize).min(MAX_BLADES_PER_TILE);
    if candidate_count == 0 {
        return None;
    }

    let mut buf = GrassMeshBuf::default();
    let mut blade_count = 0u32;

    for blade in 0..candidate_count {
        let rng = |salt: u64| blade_hash(input.seed, input.key, blade as u64, salt);

        let u = u_lo + rng(0) * (u_hi - u_lo);
        let v = v_lo + rng(1) * (v_hi - v_lo);
        let dir = cube_dir(input.key.face, u, v);

        if let Some(flatten) = &input.flatten_exclusion
            && flatten.weight(dir) > 0.05
        {
            continue;
        }

        // Shared placement gate: height, grass material-mask weight, slope, and
        // the body-fixed terrain normal — the same stencil the tile baker
        // writes into the material attachment's grass channel.
        let Some(sample) = placement_gate(source, &basis, dir, input.radius_m) else {
            continue;
        };
        if sample.height_m <= input.sea_level_m + 1.0 {
            continue;
        }
        if sample.slope > 0.45 {
            continue;
        }
        let h = sample.height_m;
        let grass_w = sample.grass_w;
        let normal_body = sample.normal_body;

        // Acceptance: keep (near-)all candidates on real grassland — the old
        // `smoothstep(0.45, 0.8)` halved density even where the ground *is*
        // grass, which is the main reason the field read sparse. We still reject
        // rock/soil (grass_w low) and fade out toward the treeline.
        let mut accept = smoothstep(0.20, 0.50, grass_w)
            * (1.0 - smoothstep(GRASS_FADE_LO_M, GRASS_FADE_HI_M, h));
        // Far rings cull grass under canopy — the trees occlude it, so the
        // distant blades are pure overdraw. Near rings pass `forest_cull = 0`.
        if input.forest_cull > 0.0 {
            accept *= 1.0 - input.forest_cull * crate::ground::scatter::forest_coverage(dir);
        }
        if (rng(2) as f32) >= accept {
            continue;
        }

        // Shared landcover field — the SAME moisture→colour model the terrain
        // albedo paints from, so grass picks up the terrain's large-scale palette
        // variation (lush green ↔ dry tan ↔ forest) AND thins on the same
        // dry/bare patches the ground shows. Sampled only past the cheap gates
        // above; `coverage` is a keep-probability (research: coverage = a hash vs
        // the field), so plains stay full and the carpet breaks up where it dries.
        let landcover =
            crate::ground::landcover::sample_landcover(dir * (input.radius_m + h as f64), h);
        if (rng(14) as f32) >= landcover.coverage {
            continue;
        }

        // Per accepted point: emit one fountain tussock (the shared
        // [`emit_grass_clump`] — same shape as the standalone clump/field and
        // the preview), rooted on the surface in the tile's local tangent frame.
        // Per-blade hue/value drift, the rounded normals, and the dark-AO→bright
        // tip gradient all live in the emitter, so near grass matches the preview.
        let clump_root_body = dir * (input.radius_m + h as f64 - ROOT_SINK_M);
        let origin = (clump_root_body - center_surface_body_m).as_vec3();
        // Tuft spread grows gently with blade width so far (wide) clumps stay
        // cohesive tufts rather than isolated cards.
        let spread = (0.14 * input.width_scale as f64).clamp(0.10, 0.50) as f32;

        // Per-clump base colour from the landcover field (large-scale palette
        // variation that matches the terrain ground); the per-blade root→tip
        // gradient + hue drift in `push_grass_blade` sit on top.
        let band = landcover.veg_color;

        // Distinct per-clump RNG stream off the tile key + candidate index (the
        // emitter's internal blade key is fixed, so all per-tuft variation must
        // come from the seed).
        let clump_seed = input.seed
            ^ (input.key.face as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (input.key.x as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ (input.key.y as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
            ^ (blade as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);

        blade_count += emit_grass_clump(
            &mut buf,
            &ClumpSpec {
                origin,
                right: basis.tangent_x.as_vec3(),
                up: normal_body.as_vec3(),
                fwd: basis.tangent_z.as_vec3(),
                color: band,
                seed: clump_seed,
                blade_count: input.blades_per_clump.max(1),
                radius_m: spread,
                // Nominal height ~0.40 m; the emitter jitters around it. Folds
                // the ring's `height_scale` (far rings taller to read at grazing
                // angles).
                height_m: 0.40 * input.height_scale,
                width_scale: input.width_scale,
                lod: input.blade_lod,
                // No baked lean — the wind shader animates near grass.
                lean: Vec3::ZERO,
            },
        );
    }

    if blade_count == 0 {
        return None;
    }

    let mesh = buf.into_mesh();

    Some(GrassTileMesh {
        mesh,
        center_surface_body_m,
        blade_count,
        built_revision,
        center_height_m,
    })
}

// ---------------------------------------------------------------------------
// Standalone clump (impostor asset)
// ---------------------------------------------------------------------------

/// Parameters for a single dense grass *tuft*, built on its own rather than as
/// part of a tile.
///
/// Unlike [`build_grass_tile_mesh`] this is `HeightSource`-free: blades sit on a
/// flat XZ plane with `+Y` up, centred on the origin, so the tuft can be rendered
/// from a hemisphere of view directions into an octahedral impostor atlas (the
/// "billboard a clump, not a blade" path — see `docs/vegetation.md` §5/§7) and
/// previewed on its own with [`GrassMaterial`]. The blade geometry mirrors
/// [`GrassBladeLod::Full`] so the impostor and the near blades read as the same
/// grass.
#[derive(Clone, Copy)]
pub struct GrassClumpParams {
    /// Footprint radius of the tuft (m); blades scatter within this disc.
    pub radius_m: f32,
    /// Number of blades fanned in the tuft.
    pub blade_count: u32,
    /// Nominal blade height (m); a per-blade factor jitters around it.
    pub height_m: f32,
    /// Base blade colour (linear); a gentle per-blade hue/value drift applies
    /// around it, as in the tile builder. Use the terrain `C_GRASS` band so the
    /// clump matches the ground.
    pub color: Vec3,
    pub seed: u64,
}

impl Default for GrassClumpParams {
    fn default() -> Self {
        Self {
            // A small but real footprint so blades root at distinct spots
            // instead of all stacking at one point.
            radius_m: 0.13,
            // Dense but SEPARATED: an even sunflower spread of near-upright blades
            // that touch without interpenetrating (vs the old 140-in-a-point mess
            // where everything stabbed through the centre).
            blade_count: 55,
            height_m: 0.36,
            color: C_GRASS,
            seed: 0x6_8A55,
        }
    }
}

/// Build a single dense grass tuft, origin-centred on flat ground (`+Y` up), as a
/// vertex-coloured blade mesh. Pure + deterministic for a given `params`. The
/// asset baked into the clump octahedral impostor and previewed on its own; the
/// blade geometry mirrors the [`GrassBladeLod::Full`] arm of
/// [`build_grass_tile_mesh`] (a [`emit_grass_clump`] fountain) so the impostor
/// matches the near field by construction.
pub fn build_grass_clump_mesh(params: &GrassClumpParams) -> Mesh {
    let mut buf = GrassMeshBuf::default();
    emit_grass_clump(
        &mut buf,
        &ClumpSpec {
            origin: Vec3::ZERO,
            right: Vec3::X,
            up: Vec3::Y,
            fwd: Vec3::Z,
            color: params.color,
            seed: params.seed,
            blade_count: params.blade_count,
            radius_m: params.radius_m,
            height_m: params.height_m,
            width_scale: 1.0,
            lod: GrassBladeLod::Full,
            lean: Vec3::ZERO,
        },
    );
    buf.into_mesh()
}

// ---------------------------------------------------------------------------
// Shared fluffy-clump emission
// ---------------------------------------------------------------------------
//
// A grass clump is a *fountain tussock*: blades root in a small disc and arch
// radially outward, drooping more toward the rim, so the clump closes into a
// rounded fluffy mass rather than a sparse spray of spikes. The droop is what
// makes the clump read both at eye level (a full dome) and from directly above
// (the arcing outer blades present face area straight down, where bolt-upright
// blades would show nothing). The same emitter builds the standalone clump
// asset and every clump in a field — and is the shape the in-game tile builder
// mirrors — so the look is defined in exactly one place.

/// Accumulating vertex buffers for a batched grass mesh.
#[derive(Default)]
struct GrassMeshBuf {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl GrassMeshBuf {
    fn into_mesh(self) -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, self.colors);
        mesh.insert_indices(Indices::U32(self.indices));
        mesh
    }
}

/// One fountain tussock to emit. `right`/`up`/`fwd` is the local tangent frame
/// (`up` = surface normal — blades carry it so they light like the ground via
/// the shared `shade_foliage`). `lean` is an optional uniform horizontal comb
/// added to every blade tip (a steady "wind" for the dry-prairie look); `ZERO`
/// leaves upright fountains, which the per-blade wind shader then animates.
struct ClumpSpec {
    origin: Vec3,
    right: Vec3,
    up: Vec3,
    fwd: Vec3,
    color: Vec3,
    seed: u64,
    blade_count: u32,
    radius_m: f32,
    height_m: f32,
    /// Blade width multiplier (far LOD rings widen blades to hold coverage as
    /// per-blade density drops). `1.0` for the hero clump/field.
    width_scale: f32,
    lod: GrassBladeLod,
    lean: Vec3,
}

/// Push a fountain tussock of blades around `spec.origin` into `buf`. Returns
/// the number of blades emitted (always `spec.blade_count.max(1)`).
fn emit_grass_clump(buf: &mut GrassMeshBuf, spec: &ClumpSpec) -> u32 {
    // Synthetic tile key — placement-free, so any fixed key gives a stable
    // per-blade RNG stream off `spec.seed`.
    let key = GrassTileKey {
        face: 0,
        x: 0,
        y: 0,
    };
    let n = spec.blade_count.max(1);
    let r = spec.radius_m.max(1.0e-3);
    let (right, up, fwd) = (spec.right, spec.up, spec.fwd);

    // Sunflower (phyllotaxis) placement: blades root on an even, well-separated
    // spiral filling the disc — no central cluster, so each blade rises from its
    // own spot instead of all stabbing through the middle (the "blades go into
    // each other" failure of random centre-biased placement). A little jitter
    // breaks the perfect spiral.
    const GOLDEN_ANGLE: f32 = 2.399_963_4;
    for blade in 0..n {
        let jit = |salt: u64| blade_hash(spec.seed, key, blade as u64, salt) as f32;
        let t = (blade as f32 + 0.5) / n as f32;
        let rad = r * t.sqrt() * (0.86 + jit(12) * 0.26);
        let ang = blade as f32 * GOLDEN_ANGLE + (jit(13) - 0.5) * 0.7;
        let root = spec.origin + (right * ang.cos() + fwd * ang.sin()) * rad;
        push_grass_blade(
            buf,
            root,
            right,
            up,
            fwd,
            spec.color,
            spec.seed,
            blade as u64,
            spec.height_m,
            spec.width_scale,
            spec.lod,
            spec.lean,
        );
    }
    n
}

/// Emit one mostly-upright grass blade rooted at `root` in the (`right`, `up`,
/// `fwd`) tangent frame (`up` = surface normal). The STATIC blade stays orderly
/// — a small random lean + a gentle tip bow — so densely-packed neighbours touch
/// without stabbing through each other; the in-game wind shader supplies the
/// motion, so the resting geometry can stay separated. `seed` + `idx` drive the
/// per-blade variation; `lean_comb` is an optional steady wind comb (metres of
/// tip lean per unit height) for the dry-prairie look.
#[allow(clippy::too_many_arguments)]
fn push_grass_blade(
    buf: &mut GrassMeshBuf,
    root: Vec3,
    right: Vec3,
    up: Vec3,
    fwd: Vec3,
    color: Vec3,
    seed: u64,
    idx: u64,
    height_m: f32,
    width_scale: f32,
    lod: GrassBladeLod,
    lean_comb: Vec3,
) {
    let key = GrassTileKey {
        face: 0,
        x: 0,
        y: 0,
    };
    let rng = |salt: u64| blade_hash(seed, key, idx, salt) as f32;

    // Lean azimuth is random, but the lean ANGLE is small, so blades stay
    // near-upright and roughly parallel and don't cross through one another.
    let az = rng(2) * std::f32::consts::TAU;
    let arch = (right * az.cos() + fwd * az.sin()).normalize_or(right);

    let h = height_m * (0.80 + rng(3) * 0.40);
    let lean = 0.04 + rng(4) * 0.12;
    let bow = 0.10 + rng(10) * 0.16;

    // Quadratic-Bézier centreline: rises nearly straight up, the tip easing out
    // by `lean` and bowing forward by `bow`, staying tall so blades read upright.
    let tip_out = h * (lean + bow * 0.5);
    let tip_up = h * (0.99 - 0.12 * lean);
    let ctrl_up = h * 0.60;
    let ctrl_out = h * (lean * 0.30);
    let p0 = root;
    let p1 = root + up * ctrl_up + arch * ctrl_out + lean_comb * (h * 0.35);
    let p2 = root + up * tip_up + arch * tip_out + lean_comb * (h * 1.0);

    // Width axis: horizontal, perpendicular to the arch, twisted slightly toward
    // up so neighbouring blades present different faces.
    let perp = up.cross(arch).normalize_or(right);
    let twist = (rng(5) - 0.5) * 0.7;
    let side = (perp + up * (twist * 0.30)).normalize_or(perp);

    let width_m = (0.018 + rng(6) * 0.016) * width_scale;

    // Tint: base palette + gentle per-blade hue/value drift.
    let hue = (rng(11) - 0.5) * 0.10;
    let tinted = Vec3::new(color.x * (1.0 + hue), color.y, color.z * (1.0 - hue));
    let tint = tinted * (0.86 + 0.24 * rng(7));
    let phase = rng(9);

    push_curved_blade(buf, p0, p1, p2, side, up, width_m, h, tint, phase, lod);
}

/// Loft one curved, tapered grass blade along the quadratic-Bézier centreline
/// `p0`→`p1`→`p2`, with a width that tapers to a point at the tip. `normal` is
/// the surface up (blades light like the ground), `side` the in-plane width
/// axis. `blade_h` rides in `color.a` for the shader's clipmap scale-fade,
/// `phase` in `uv.y` for per-blade wind; `uv.x` is the height fraction (0 root
/// → 1 tip) the wind sway weights by. A dark-AO base brightens toward the tip.
#[allow(clippy::too_many_arguments)]
fn push_curved_blade(
    buf: &mut GrassMeshBuf,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    side: Vec3,
    normal: Vec3,
    width_m: f32,
    blade_h: f32,
    tint: Vec3,
    phase: f32,
    lod: GrassBladeLod,
) {
    let bezier = |t: f32| {
        let u = 1.0 - t;
        p0 * (u * u) + p1 * (2.0 * u * t) + p2 * (t * t)
    };
    let hw = width_m * 0.5;
    let base = buf.positions.len() as u32;
    let color_at =
        |lighten: f32| [tint.x * lighten, tint.y * lighten, tint.z * lighten, blade_h];

    // Rounded blade normals (the key "fluffy" shading trick — Ghost of Tsushima
    // / AMD GPUOpen): a flat strip on the terrain-up normal lights like a spike.
    // Instead the two edge vertices' normals fan *outward* across the width, so
    // the blade shades like a rounded cylinder. Kept **up-dominant** (the
    // surface `normal` is weighted ≫ the fan) so blades still match the ground
    // brightness and the double-sided back faces don't go dark (the shader does
    // not flip on back faces). A small forward `face` tilt adds blade-to-blade
    // light variation.
    let axis = (p2 - p0).normalize_or(normal);
    let mut face = side.cross(axis).normalize_or(normal);
    if face.dot(normal) < 0.0 {
        face = -face;
    }
    let edge_n = |sgn: f32| (normal + side * (sgn * 0.5) + face * 0.16).normalize_or(normal);
    let n_l = edge_n(-1.0);
    let n_r = edge_n(1.0);
    let n_tip = (normal + face * 0.12).normalize_or(normal);

    let mut push = |pos: Vec3, sway: f32, lighten: f32, nrm: Vec3| {
        buf.positions.push(pos.to_array());
        buf.normals.push(nrm.to_array());
        buf.uvs.push([sway, phase]);
        buf.colors.push(color_at(lighten));
    };
    match lod {
        GrassBladeLod::Full => {
            // 4 cross-sections (root, 40 %, 72 %, tip), tapering to a point.
            // Value ramp: the TIP sits at 1.0 = the unmodified `tint` (which is
            // the exact terrain ground colour), so the visible canopy matches the
            // bare ground it fades into — no brightness seam. The base darkens as
            // ambient occlusion (grass roots are shaded); the sunlit-tip glow then
            // comes from the *lighting*, not a brighter albedo.
            let c1 = bezier(0.40);
            let c2 = bezier(0.72);
            let tip = bezier(1.0);
            push(p0 - side * hw, 0.0, 0.62, n_l);
            push(p0 + side * hw, 0.0, 0.62, n_r);
            push(c1 - side * (hw * 0.80), 0.40, 0.80, n_l);
            push(c1 + side * (hw * 0.80), 0.40, 0.80, n_r);
            push(c2 - side * (hw * 0.46), 0.72, 0.93, n_l);
            push(c2 + side * (hw * 0.46), 0.72, 0.93, n_r);
            push(tip, 1.0, 1.0, n_tip);
            buf.indices.extend_from_slice(&[
                base,
                base + 1,
                base + 2,
                base + 1,
                base + 3,
                base + 2,
                base + 2,
                base + 3,
                base + 4,
                base + 3,
                base + 5,
                base + 4,
                base + 4,
                base + 5,
                base + 6,
            ]);
        }
        GrassBladeLod::Wide => {
            // 3 cross-sections (root, mid, tip) — a soft tuft card when widened.
            // Tip = 1.0 = the ground colour (see the Full arm), so far clumps
            // fade into the matching terrain albedo with no brightness seam.
            let mid = bezier(0.55);
            let tip = bezier(1.0);
            push(p0 - side * hw, 0.0, 0.66, n_l);
            push(p0 + side * hw, 0.0, 0.66, n_r);
            push(mid - side * (hw * 0.62), 0.55, 0.86, n_l);
            push(mid + side * (hw * 0.62), 0.55, 0.86, n_r);
            push(tip, 1.0, 1.0, n_tip);
            buf.indices.extend_from_slice(&[
                base,
                base + 1,
                base + 2,
                base + 1,
                base + 3,
                base + 2,
                base + 2,
                base + 3,
                base + 4,
            ]);
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone field (preview asset)
// ---------------------------------------------------------------------------

/// Parameters for a flat patch of fluffy grass — a field of fountain clumps on
/// a flat XZ plane (`+Y` up), centred on the origin. `HeightSource`-free like
/// [`build_grass_clump_mesh`], for previewing the aggregate look (especially
/// from above) without a running game.
#[derive(Clone, Copy)]
pub struct GrassFieldParams {
    /// Side length of the (square) patch, metres.
    pub size_m: f32,
    /// Clump (placement-point) density per m².
    pub clumps_per_m2: f32,
    /// Blades fanned per clump.
    pub blades_per_clump: u32,
    /// Footprint radius of each clump, metres.
    pub clump_radius_m: f32,
    /// Nominal blade height, metres.
    pub height_m: f32,
    /// Base blade colour (linear).
    pub color: Vec3,
    /// Steady "windswept" comb toward `+X`, as a fraction of blade height the
    /// tips lean. `0` = upright fountains (the in-game wind shader animates
    /// those); a positive value bakes the dry-prairie wind lean into the mesh.
    pub wind_lean: f32,
    pub seed: u64,
}

impl Default for GrassFieldParams {
    fn default() -> Self {
        Self {
            // Distinct tufts, spaced so they just touch (≈0.25 m apart, ≈0.22 m
            // wide) — a dense field of SEPARATED clumps, not a merged mass. Each
            // tuft's blades are sunflower-spread so they don't interpenetrate.
            size_m: 5.0,
            clumps_per_m2: 16.0,
            blades_per_clump: 14,
            clump_radius_m: 0.11,
            height_m: 0.34,
            color: C_GRASS,
            wind_lean: 0.0,
            seed: 0x47_1E_1D,
        }
    }
}

/// Build a flat field of fountain clumps (origin-centred, `+Y` up). Pure +
/// deterministic for a given `params`. Each clump is a [`emit_grass_clump`]
/// fountain, so the field reads exactly as the in-game grass aggregated — the
/// tool for judging fluffiness and the from-above look headlessly.
pub fn build_grass_field_mesh(params: &GrassFieldParams) -> Mesh {
    let key = GrassTileKey {
        face: 0,
        x: 0,
        y: 0,
    };
    let area = params.size_m * params.size_m;
    let n_clumps = ((area * params.clumps_per_m2).round() as i64).max(1) as u32;
    let lean = Vec3::X * params.wind_lean;

    let mut buf = GrassMeshBuf::default();
    for c in 0..n_clumps {
        let rng = |salt: u64| blade_hash(params.seed ^ 0x0091_513D, key, c as u64, salt) as f32;
        let x = (rng(0) - 0.5) * params.size_m;
        let z = (rng(1) - 0.5) * params.size_m;
        emit_grass_clump(
            &mut buf,
            &ClumpSpec {
                origin: Vec3::new(x, 0.0, z),
                right: Vec3::X,
                up: Vec3::Y,
                fwd: Vec3::Z,
                color: params.color,
                // Distinct per-clump RNG stream so neighbouring clumps differ.
                seed: params.seed ^ (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                blade_count: params.blades_per_clump,
                radius_m: params.clump_radius_m,
                height_m: params.height_m,
                width_scale: 1.0,
                lod: GrassBladeLod::Full,
                lean,
            },
        );
    }
    buf.into_mesh()
}

/// Integer-mix hash → `[0, 1)`. Deterministic per (seed, tile, blade, salt);
/// no trig, stable at any coordinate (same family as the cloud weather hash).
fn blade_hash(seed: u64, key: GrassTileKey, blade: u64, salt: u64) -> f64 {
    let mut h = seed
        ^ (key.face as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (key.x as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (key.y as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
        ^ blade.wrapping_mul(0xD6E8_FEB8_6659_FD93)
        ^ salt.wrapping_mul(0xA24B_AED4_963E_E407);
    h ^= h >> 31;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 32;
    (h & 0x000F_FFFF_FFFF_FFFF) as f64 / (1u64 << 52) as f64
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(f32::EPSILON)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/// Per-frame grass shading parameters.
///
/// Field order is load-bearing — the WGSL `GrassParams` mirror in `grass.wgsl`
/// must match. The lighting fields (`sun_dir.w`, `sky_up`, `sky_tau`) feed the
/// shared `thalos::lighting` sky model so blades light identically to the
/// ground; see `crate::rendering::grass::update_grass_material` for the values.
#[derive(Clone, Copy, ShaderType, Default)]
pub struct GrassParams {
    /// xyz = unit direction toward the star, world render space. w = sun flux
    /// (lux × exposure gain — the same value the terrain `SceneLighting` star
    /// carries), fed to the shared `compute_surface_sky`.
    pub sun_dir: Vec4,
    /// xyz = wind direction (world render space, roughly tangent at the
    /// camera), w = sway amplitude at the blade tip, metres.
    pub wind: Vec4,
    /// x = animation time (seconds), y = near-edge fade distance (m),
    /// z = far-edge fade distance (m), w = fade band half-width (m). A ring's
    /// blades fade in around its near edge and out around its far edge, so
    /// adjacent clipmap rings cross-fade through their shared boundary.
    pub time_fade: Vec4,
    /// xyz = local radial up (world render space) for the sky hemisphere split,
    /// w unused.
    pub sky_up: Vec4,
    /// xyz = Rayleigh vertical optical depth τ_v (= β_R · H_R, the authored
    /// per-channel zenith optical depth, scale-independent), w = artistic
    /// atmosphere strength. Drives the blue sky tint + sunset reddening.
    pub sky_tau: Vec4,
    /// xyz = vegetation focus OFFSET = (player craft − camera), render space.
    /// The shader rebuilds the LOD/fade reference as `view.world_position +
    /// offset` — the craft in the *current* frame's render origin. Passing an
    /// offset (not an absolute world point) keeps the fade craft-anchored
    /// (zoom/orbit-independent) while staying robust to big_space floating-origin
    /// recentres: an absolute anchor is read one frame stale and jumps a whole
    /// cell on recentre, popping fade-band instances in/out while the craft
    /// moves. w = 1.0 when valid; 0.0 → offset 0 = fade around the camera.
    pub anchor: Vec4,
}

/// Batched grass-blade material: vertex wind sway + wrap-diffuse shading that
/// mirrors the vegetated terrain BRDF's constants, so blades match the ground
/// brightness by construction. Dithered discard handles the distance fade in
/// the opaque pass (no sorting).
///
/// Blades **receive** the cascaded sun-shadows trees cast (the same maps the
/// ground + tree materials bind), sampled **per-vertex** at the blade — cheap on
/// this overdraw-heavy material, and a blade is small enough that a per-vertex
/// (interpolated) factor is visually indistinguishable from per-fragment. The
/// depth-map bindings carry `vertex` visibility for that VS sample;
/// `shadow.gate.x == 0` (preview/inactive) skips the sample entirely.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct GrassMaterial {
    #[uniform(0)]
    pub params: GrassParams,
    /// Cascaded sun-shadow transforms + strength (see [`ShadowCascadeBlock`]).
    #[uniform(1)]
    pub shadow: ShadowCascadeBlock,
    /// Per-cascade sun-shadow depth maps (near→far) — the same handles the
    /// ground + tree materials bind. Each a plain `texture_depth_2d`; always
    /// valid (see [`fallback_shadow_map`](crate::ground::fallback_shadow_map)).
    /// `vertex` visibility so the blade can sample them in the vertex shader.
    #[texture(2, sample_type = "depth", visibility(vertex, fragment))]
    pub sun_shadow_map_0: Handle<Image>,
    #[texture(3, sample_type = "depth", visibility(vertex, fragment))]
    pub sun_shadow_map_1: Handle<Image>,
    #[texture(4, sample_type = "depth", visibility(vertex, fragment))]
    pub sun_shadow_map_2: Handle<Image>,
}

impl Material for GrassMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/grass.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/grass.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Blades are single-card strips seen from both sides.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

pub(crate) fn embed_grass_shader(app: &mut App) {
    embedded_asset!(app, "grass.wgsl");
}

/// Standalone plugin that registers [`GrassMaterial`] and embeds its shader, for
/// consumers that render grass without the full [`ThalosTerrainPlugin`](crate::ground::ThalosTerrainPlugin)
/// (e.g. the headless object preview). The terrain plugin registers the same
/// material directly, so don't add both.
pub struct GrassMaterialPlugin;

impl Plugin for GrassMaterialPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
        }
        app.add_plugins(MaterialPlugin::<GrassMaterial>::default());
        embed_grass_shader(app);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ground::height_source::ConstantHeightSource;

    #[test]
    fn tile_key_roundtrip() {
        let tiles_per_side = grass_tiles_per_side(3_186_000.0, GRASS_TILE_SIZE_M);
        for &dir in &[
            DVec3::new(0.3, 0.8, -0.5).normalize(),
            DVec3::X,
            DVec3::new(-0.6, 0.1, 0.79).normalize(),
        ] {
            let key = grass_tile_key(dir, tiles_per_side);
            let (center, _) = grass_tile_frame(key, tiles_per_side).unwrap();
            // The centre of the tile containing `dir` must map back to the
            // same tile, and must be within one tile's angular size of `dir`.
            assert_eq!(grass_tile_key(center, tiles_per_side), key);
            let max_angle = 2.0 * GRASS_TILE_SIZE_M / 3_186_000.0;
            assert!(center.angle_between(dir) < max_angle);
        }
    }

    #[test]
    fn builder_is_deterministic_and_flat_ground_grows_grass() {
        let radius_m = 3_186_000.0;
        let input = GrassTileBuildInput {
            key: grass_tile_key(DVec3::new(0.2, 0.3, 0.93).normalize(), 255_000),
            tiles_per_side: 255_000,
            height_source: Arc::new(ConstantHeightSource::new(2000.0)),
            radius_m,
            sea_level_m: 0.0,
            blades_per_m2: 4.0,
            blades_per_clump: 4,
            blade_lod: GrassBladeLod::Full,
            width_scale: 1.0,
            height_scale: 1.0,
            seed: 7,
            flatten_exclusion: None,
            forest_cull: 0.0,
        };
        let a = build_grass_tile_mesh(&input).expect("flat land grows grass");
        let b = build_grass_tile_mesh(&input).expect("flat land grows grass");
        assert!(a.blade_count > 0);
        assert_eq!(a.blade_count, b.blade_count);
        assert_eq!(a.center_surface_body_m, b.center_surface_body_m);
        // Flat ground at 2000 m: grass mask is fully dominant, altitude fade
        // ≈ 1 → nearly every candidate should survive.
        assert!(a.blade_count >= 100);
    }

    #[test]
    fn no_grass_below_sea_level() {
        let input = GrassTileBuildInput {
            key: GrassTileKey {
                face: 4,
                x: 100,
                y: 100,
            },
            tiles_per_side: 255_000,
            height_source: Arc::new(ConstantHeightSource::new(-50.0)),
            radius_m: 3_186_000.0,
            sea_level_m: 0.0,
            blades_per_m2: 4.0,
            blades_per_clump: 4,
            blade_lod: GrassBladeLod::Full,
            width_scale: 1.0,
            height_scale: 1.0,
            seed: 7,
            flatten_exclusion: None,
            forest_cull: 0.0,
        };
        assert!(build_grass_tile_mesh(&input).is_none());
    }
}
