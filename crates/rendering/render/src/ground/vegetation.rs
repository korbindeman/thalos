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
//! the game crate (`thalos_runtime::rendering::grass`).

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

use crate::ground::body_material::ShadowCascadeBlock;
use crate::ground::height_source::HeightSource;
use crate::ground::rendered_height::TerrainPatchBasis;
use crate::ground::scatter::{ScatterClass, ScatterRegion, classify_scatter, placement_gate};
use crate::ground::tile_lattice::{self, TileLattice, cube_dir};
use crate::ground::tree_atlas::GRASS_CARD_VARIANTS;

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

/// Raised candidate ceiling for a **lawn** tile only. A lawn force-accepts every
/// candidate (no gate thinning), so its placement spacing is what reads — a
/// denser grid closes the gaps that look patchy. Scoped to lawn tiles (a small
/// area near a base) so wild/far grass and their cost are untouched by the
/// higher density.
const MAX_LAWN_BLADES_PER_TILE: usize = 14336;

/// Sink blade roots slightly below the sampled surface so bilinear
/// height-mirror error can't leave a row of floating root quads.
const ROOT_SINK_M: f64 = 0.03;

// Altitude gate for grass density: fades out approaching the treeline (where
// the terrain shader paints grey alpine scree). Tracks the temperate treeline
// (TREELINE_LO/HI ~2400–3000 m) so blades are gone before the scree takes over.
const GRASS_FADE_LO_M: f32 = 2000.0;
const GRASS_FADE_HI_M: f32 = 2700.0;

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
/// rule — see `docs/world/vegetation.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrassBladeLod {
    /// 7 vertices, curved + tapered. Near ring.
    Full,
    /// 3 vertices, flat triangle. Far rings, widened via `width_scale`.
    Wide,
    /// **Clump card** — two crossed vertical quads (8 verts) for the *whole* clump
    /// instead of a fountain of blades (~100 verts). The far/mid bands are
    /// **vertex-bound** (the cost is blade vertex throughput, not fragment
    /// overdraw — a depth prepass didn't help), so one billboard tuft per clump is
    /// the lever there. The grass shader samples the **baked card atlas**
    /// ([`thalos_texgen::grass_card_atlas`], bound as
    /// [`GrassMaterial::card_atlas`]) across the card — a painted cluster of
    /// layered blades, the classic grass-card technique — discarding the gaps
    /// and modulating the per-clump tint, so it reads as real grass at distance;
    /// the card reuses the blade displacement (it sways + fade-shrinks like a
    /// tall blade). See `docs/world/vegetation.md` §6.
    Card,
}

/// The tunable *shape* of a grass clump — its **type**. Bundles the levers that
/// decide whether grass reads short-and-fluffy or tall-and-wispy, so the in-game
/// distribution can blend between named types per clump (long grass in some
/// clumps, thick short grass in others) and the previews can dial them in.
///
/// A clump is a fountain tussock: blades root in a small disc and arch radially
/// outward, drooping more toward the rim, so the tuft closes into a rounded
/// fluffy dome. `droop`/`dome` shape that dome; `width_m`/`height_m` set the
/// blade thickness and length; `blades_per_clump` sets how densely the tuft is
/// packed (the primary fluffiness lever, since it multiplies coverage without
/// re-paying the placement gate).
#[derive(Clone, Copy, Debug)]
pub struct GrassProfile {
    /// Nominal blade length (m); a per-blade factor jitters around it. The
    /// **length** lever.
    pub height_m: f32,
    /// Base blade width at the root (m); tapers to a point at the tip. The
    /// **thickness** lever — fat blades read soft/fluffy, thin blades wispy.
    pub width_m: f32,
    /// Blades fanned per clump — the primary **fluffiness** lever. More blades
    /// per accepted placement point thicken the tuft without re-paying the
    /// (expensive) placement gate.
    pub blades_per_clump: u32,
    /// Footprint radius of the tuft (m); blade roots scatter within this disc.
    pub radius_m: f32,
    /// Outward fountain droop in `[0, 1]`: `0` = bolt-upright spikes, `1` = blades
    /// arch strongly outward and bow over toward the rim — the rounded fluffy
    /// dome. Graded by each blade's root radius, so rim blades droop most.
    pub droop: f32,
    /// Dome height taper in `[0, 1]`: how much shorter rim blades are than the
    /// centre, rounding the tuft's top (`0` = flat-topped, `1` = strong dome).
    pub dome: f32,
}

impl GrassProfile {
    /// Manicured **lawn** clump — short, thick, tidy. The managed-ground cover
    /// for spaceport/base terrain (see [`ScatterTreatment::Lawn`](crate::ground::scatter::ScatterTreatment)):
    /// low and dense with many short, near-upright blades and only a gentle
    /// dome, so a carpet of these tufts reads as kept grass rather than a wild
    /// drooping meadow. Shorter than [`fluffy_short`](Self::fluffy_short) and far
    /// less droopy.
    pub const fn lawn() -> Self {
        Self {
            // A touch longer + fluffier than the first manicured cut: more blades
            // arching off a wider footprint so neighbouring tufts overlap into a
            // continuous soft carpet instead of reading as separated patches.
            height_m: 0.16,
            width_m: 0.027,
            blades_per_clump: 30,
            radius_m: 0.13,
            droop: 0.42,
            dome: 0.6,
        }
    }

    /// Short, thick, very fluffy lawn/meadow clump — the soft dense reference
    /// look. Many wide blades, a strong droopy dome. Bumped to a lawn-like blade
    /// count + a wider footprint so wild tufts overlap into a continuous lush
    /// carpet (the near ring is small, so the extra blades are affordable).
    pub const fn fluffy_short() -> Self {
        Self {
            height_m: 0.26,
            width_m: 0.036,
            blades_per_clump: 28,
            radius_m: 0.18,
            droop: 0.80,
            dome: 0.60,
        }
    }

    /// Taller, thinner, more upright wispy/prairie clump. Narrow arching blades —
    /// denser + a wider footprint now so the prairie reads full, not spiky/bald.
    pub const fn wispy_tall() -> Self {
        Self {
            height_m: 0.58,
            width_m: 0.016,
            blades_per_clump: 20,
            radius_m: 0.15,
            droop: 0.32,
            dome: 0.25,
        }
    }

    /// Linear blend between two types — the per-clump distribution mixes the
    /// body's `dry`/`lush` profiles by a `[0, 1]` factor. `blades_per_clump`
    /// rounds to the nearest whole blade.
    pub fn lerp(a: GrassProfile, b: GrassProfile, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let mix = |x: f32, y: f32| x + (y - x) * t;
        Self {
            height_m: mix(a.height_m, b.height_m),
            width_m: mix(a.width_m, b.width_m),
            blades_per_clump: mix(a.blades_per_clump as f32, b.blades_per_clump as f32).round()
                as u32,
            radius_m: mix(a.radius_m, b.radius_m),
            droop: mix(a.droop, b.droop),
            dome: mix(a.dome, b.dome),
        }
    }

    /// Apply a clipmap ring's LOD scalars (constant-coverage rule): far rings
    /// widen + lengthen blades and thin the tuft as per-blade density drops. The
    /// footprint widens gently with the blades so a thinned wide tuft stays a
    /// cohesive clump rather than isolated cards.
    pub fn scaled(self, width_scale: f32, height_scale: f32, clump_scale: f32) -> Self {
        Self {
            height_m: self.height_m * height_scale,
            width_m: self.width_m * width_scale,
            blades_per_clump: ((self.blades_per_clump as f32 * clump_scale).round() as u32).max(1),
            radius_m: self.radius_m * (0.7 + 0.3 * width_scale),
            droop: self.droop,
            dome: self.dome,
        }
    }
}

impl Default for GrassProfile {
    /// A lush fluffy meadow clump — the new baseline (thicker, denser, and
    /// droopier than the old spiky upright blades).
    fn default() -> Self {
        Self {
            height_m: 0.34,
            width_m: 0.024,
            blades_per_clump: 14,
            radius_m: 0.13,
            droop: 0.55,
            dome: 0.45,
        }
    }
}

/// Everything [`build_grass_tile_mesh`] needs, snapshotted by the game-side
/// driver so the build can run on the async compute pool without touching ECS.
pub struct GrassTileBuildInput {
    pub key: GrassTileKey,
    pub tiles_per_side: i64,
    pub height_source: Arc<dyn HeightSource>,
    pub radius_m: f64,
    /// Sea level (m above reference radius); blades require
    /// `height > sea_level + VEG_BEACH_CLEAR_M`. Pass `f32::MIN` for bodies without oceans.
    pub sea_level_m: f32,
    pub blades_per_m2: f32,
    /// The two grass *types* this body's meadow blends between, per clump, by the
    /// landcover moisture field: `profile_dry` on the drier ground, `profile_lush`
    /// on the wetter. Both clipmap rings carry the same pair so the meadow reads
    /// consistent across LOD; only the LOD scalars below differ per ring. This is
    /// the "long grass in some clumps, thick short grass in others" lever.
    pub profile_dry: GrassProfile,
    pub profile_lush: GrassProfile,
    /// Blade geometry LOD for this clipmap ring.
    pub blade_lod: GrassBladeLod,
    /// Blade-width LOD multiplier — coarser rings widen blades to hold ground
    /// coverage as per-blade density drops (the constant-coverage rule).
    pub width_scale: f32,
    /// Blade-height LOD multiplier — far blades a touch taller so the field reads
    /// at grazing angles.
    pub height_scale: f32,
    /// Blades-per-clump LOD multiplier in `(0, 1]` — far rings thin each tuft
    /// (the profile's blade count × this), trading per-clump density for the
    /// wider blades above so total coverage stays roughly constant.
    pub clump_scale: f32,
    pub seed: u64,
    /// Building-terrain scatter footprints near this tile (the spaceport basin,
    /// runway strip, launchpads, buildings, tanks). Resolved per candidate by
    /// [`classify_scatter`]: blades are skipped under a clearing (paving), and
    /// forced to a tidy [`lawn_profile`](Self::lawn_profile) cover (ignoring the
    /// natural moisture/coverage gates) inside a lawn footprint. Empty off-base.
    /// Cheaply shared (`Arc`) across every in-flight tile build in a frame.
    pub scatter_regions: Arc<Vec<ScatterRegion>>,
    /// The grass *type* forced inside a lawn footprint — short, thick, managed
    /// (typically [`GrassProfile::lawn`]). Ring LOD scalars still apply.
    pub lawn_profile: GrassProfile,
    /// Placement-point density (per m²) inside a lawn, used in place of
    /// `blades_per_m2` when the tile sits in a lawn footprint. Higher than the
    /// wild density (the lawn force-accepts every point, so the spacing is what
    /// shows), and capped by [`MAX_LAWN_BLADES_PER_TILE`] rather than the wild
    /// ceiling. `0` falls back to `blades_per_m2`.
    pub lawn_density_per_m2: f32,
    /// Far-ring forest cull strength in `[0, 1]`. Distant grass under a tree
    /// canopy is occluded by the trees in front of it, so the far clipmap rings
    /// thin grass by `forest_cull × canopy_coverage(dir)` — full grass on open
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

    // Hoisted once per tile — see the per-candidate use in the acceptance gate.
    let canopy_climate = source.canopy_climate(center_dir, GRASS_SAMPLE_LOD_M);

    let center_height_m = source.sample_height_m(center_dir.as_vec3(), GRASS_SAMPLE_LOD_M)?;
    let center_surface_body_m = center_dir * (input.radius_m + center_height_m as f64);

    // Metric extents from the actual uv span (cube distortion shrinks tiles
    // toward face corners), so blade density stays uniform per square metre.
    let (u_lo, u_hi, v_lo, v_hi) = lattice.uv_span(input.key);
    let (ext_u_m, ext_v_m) = lattice.tile_extents_m(input.key, input.radius_m);
    let area_m2 = (ext_u_m * ext_v_m).max(0.0);
    // A lawn tile places at its (higher) lawn density under a raised ceiling, so
    // the forced-full cover reads as a continuous carpet rather than a sparse
    // grid of tufts. Decided from the tile centre; per-candidate classification
    // below still clears any paving and grades the lawn edge.
    let lawn_tile = classify_scatter(&input.scatter_regions, center_dir) == ScatterClass::Lawn;
    let (density, cap) = if lawn_tile {
        (
            input.lawn_density_per_m2.max(input.blades_per_m2),
            MAX_LAWN_BLADES_PER_TILE,
        )
    } else {
        (input.blades_per_m2, MAX_BLADES_PER_TILE)
    };
    let candidate_count = ((area_m2 * density as f64).round() as usize).min(cap);
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

        // Building-terrain treatment: a clearing (paving / footprint) removes
        // the blade outright; a lawn forces a tidy managed cover, bypassing the
        // natural grassland/coverage gates below; off-base it's a no-op.
        let lawn = match classify_scatter(&input.scatter_regions, dir) {
            ScatterClass::Clear => continue,
            ScatterClass::Lawn => true,
            ScatterClass::Natural => false,
        };

        // Shared placement gate: height, grass material-mask weight, slope, and
        // the body-fixed terrain normal — the same stencil the tile baker
        // writes into the material attachment's grass channel.
        let Some(sample) = placement_gate(source, &basis, dir, input.radius_m) else {
            continue;
        };
        if sample.height_m <= input.sea_level_m + crate::ground::scatter::VEG_BEACH_CLEAR_M {
            continue;
        }
        if sample.slope > 0.45 {
            continue;
        }
        let h = sample.height_m;
        let grass_w = sample.grass_w;
        let normal_body = sample.normal_body;

        // Natural grassland gates — skipped inside a lawn, which is groomed grass
        // by construction regardless of the underlying material/moisture field.
        // Climate-shifted (ecological) altitude: the treeline fade descends
        // with latitude, so polar tundra thins out at low ground exactly like
        // the terrain paint (docs/world/terrain_macro.md Phase 2).
        let eco_h = h + thalos_terrain::climate_cold_lift_m(dir.y.abs()) as f32;
        if !lawn {
            // Acceptance: keep (near-)all candidates on real grassland — the old
            // `smoothstep(0.45, 0.8)` halved density even where the ground *is*
            // grass, which is the main reason the field read sparse. We still
            // reject rock/soil (grass_w low) and fade out toward the treeline.
            let mut accept = smoothstep(0.20, 0.50, grass_w)
                * (1.0 - smoothstep(GRASS_FADE_LO_M, GRASS_FADE_HI_M, eco_h));
            // Far rings cull grass under canopy — the trees occlude it, so the
            // distant blades are pure overdraw. Near rings pass `forest_cull = 0`.
            if input.forest_cull > 0.0 {
                // Same canopy authority the trees are placed from, so grass is
                // culled exactly where canopy actually closes over it and returns
                // in the glades (`thalos_terrain::canopy`). Climate is hoisted per
                // tile — this runs once per blade candidate, thousands per tile.
                accept *= 1.0 - input.forest_cull * canopy_climate.coverage(dir, h);
            }
            if (rng(2) as f32) >= accept {
                continue;
            }
        }

        // Shared landcover field — the SAME moisture→colour model the terrain
        // albedo paints from, so grass picks up the terrain's large-scale palette
        // variation (lush green ↔ dry tan ↔ forest) AND thins on the same
        // dry/bare patches the ground shows. Sampled only past the cheap gates
        // above; `coverage` is a keep-probability (research: coverage = a hash vs
        // the field), so plains stay full and the carpet breaks up where it dries.
        // A lawn samples it only for the ground-matched colour — its coverage is
        // forced full so the kept grass reads as a continuous manicured carpet.
        let landcover = crate::ground::landcover::sample_landcover(
            dir * (input.radius_m + h as f64),
            h,
            input.height_source.landcover_moisture(dir),
            dir.y.abs() as f32,
        );
        // Floor the keep-probability so lush grassland stays a continuous carpet —
        // the raw coverage dipped enough to read as bald patches in the fields. Only
        // the genuinely dry/bare areas (coverage well below the floor) still thin.
        if !lawn && (rng(14) as f32) >= landcover.coverage.max(0.85) {
            continue;
        }

        // Per accepted point: emit one fountain tussock of the clump's resolved
        // grass *type* (the shared [`emit_grass_clump`] — same shape as the
        // standalone clump/field and the preview), rooted on the surface in the
        // tile's local tangent frame. Per-blade hue/value drift, the rounded
        // normals, and the dark-AO→bright tip gradient live in the emitter, so
        // near grass matches the preview.
        let clump_root_body = dir * (input.radius_m + h as f64 - ROOT_SINK_M);
        let origin = (clump_root_body - center_surface_body_m).as_vec3();

        // Per-clump base colour from the landcover field (large-scale palette
        // variation that matches the terrain ground); the per-blade root→tip
        // gradient + hue drift in `push_grass_blade` sit on top.
        let band = landcover.veg_color;

        // Grass *type* per clump. On a lawn every tuft is the one managed
        // (short, thick) profile so the cover reads uniform; off-lawn, blend the
        // body's two profiles by moisture so the meadow carries thick short grass
        // on drier patches and longer wispy grass on wetter ones. Either way this
        // ring's LOD scalars (width / height / clump density) then apply.
        let base_profile = if lawn {
            input.lawn_profile
        } else {
            let type_mix = smoothstep(-0.55, 0.55, landcover.moisture);
            GrassProfile::lerp(input.profile_dry, input.profile_lush, type_mix)
        };
        let profile = base_profile.scaled(input.width_scale, input.height_scale, input.clump_scale);

        // Distinct per-clump RNG stream off the tile key + candidate index (the
        // emitter's internal blade key is fixed, so all per-tuft variation must
        // come from the seed).
        let clump_seed = input.seed
            ^ (input.key.face as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (input.key.x as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ (input.key.y as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
            ^ (blade as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);

        let spec = ClumpSpec {
            origin,
            right: basis.tangent_x.as_vec3(),
            up: normal_body.as_vec3(),
            fwd: basis.tangent_z.as_vec3(),
            color: band,
            seed: clump_seed,
            profile,
            lod: input.blade_lod,
            // No baked lean — the wind shader animates near grass.
            lean: Vec3::ZERO,
        };
        // Far/mid rings draw one cheap billboard tuft per clump (vertex-bound band);
        // near rings draw the full fountain of blades.
        blade_count += if input.blade_lod == GrassBladeLod::Card {
            emit_grass_card(&mut buf, &spec)
        } else {
            emit_grass_clump(&mut buf, &spec)
        };
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
/// "billboard a clump, not a blade" path — see `docs/world/vegetation.md` §5/§7) and
/// previewed on its own with [`GrassMaterial`]. The blade geometry mirrors
/// [`GrassBladeLod::Full`] so the impostor and the near blades read as the same
/// grass.
#[derive(Clone, Copy)]
pub struct GrassClumpParams {
    /// The grass *type* — thickness, length, fluffiness, droop (see
    /// [`GrassProfile`]).
    pub profile: GrassProfile,
    /// Base blade colour (linear); a gentle per-blade hue/value drift applies
    /// around it, as in the tile builder. Use the terrain `C_GRASS` band so the
    /// clump matches the ground.
    pub color: Vec3,
    pub seed: u64,
}

impl Default for GrassClumpParams {
    fn default() -> Self {
        Self {
            profile: GrassProfile::default(),
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
            profile: params.profile,
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
    /// Per-vertex BLADE ROOT (tile-local position), shared by all verts of a
    /// blade. Carried in the standard `ATTRIBUTE_TANGENT` slot (xyz; w unused) so
    /// the shader's clipmap height-fade can shrink each blade *uniformly toward
    /// its own root* instead of collapsing only the vertical component — which
    /// flattened the blade's baked horizontal lean onto the ground (the "flat
    /// triangle shards in the LOD cross-fade bands" artifact). Reusing a standard
    /// attribute keeps Bevy's auto-generated vertex layout intact (a custom
    /// attribute would force a layout override that breaks the prepass — see the
    /// note in `tree_material::specialize`).
    roots: Vec<[f32; 4]>,
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
        mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, self.roots);
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
    /// The resolved grass type for this clump (already LOD-scaled by the caller
    /// via [`GrassProfile::scaled`] where it's part of a clipmap ring).
    profile: GrassProfile,
    lod: GrassBladeLod,
    lean: Vec3,
}

/// Push a fountain tussock of blades around `spec.origin` into `buf`. Returns
/// the number of blades emitted (`spec.profile.blades_per_clump.max(1)`).
fn emit_grass_clump(buf: &mut GrassMeshBuf, spec: &ClumpSpec) -> u32 {
    // Synthetic tile key — placement-free, so any fixed key gives a stable
    // per-blade RNG stream off `spec.seed`.
    let key = GrassTileKey {
        face: 0,
        x: 0,
        y: 0,
    };
    let p = spec.profile;
    let n = p.blades_per_clump.max(1);
    let r = p.radius_m.max(1.0e-3);
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
        // `rf` = normalized root radius (0 centre → 1 rim); drives the fountain
        // droop/dome so the tuft closes into a rounded mass.
        let rf = t.sqrt();
        let rad = r * rf * (0.86 + jit(12) * 0.26);
        let ang = blade as f32 * GOLDEN_ANGLE + (jit(13) - 0.5) * 0.7;
        // Outward (clump-radial) direction this blade arches along.
        let outward = (right * ang.cos() + fwd * ang.sin()).normalize_or(right);
        let root = spec.origin + outward * rad;
        push_grass_blade(
            buf,
            root,
            BladeFrame {
                right,
                up,
                fwd,
                outward,
                rf,
            },
            spec.color,
            spec.seed,
            blade as u64,
            &p,
            spec.lod,
            spec.lean,
        );
    }
    n
}

/// Emit one grass **clump card**: two crossed vertical quads (8 verts, 12 indices)
/// standing for the whole clump, instead of a fountain of blades (~100 verts) — the
/// vertex-count cut the far/mid bands need. The card reuses the blade displacement:
/// `uv.x` is the height fraction (so it sways at the top and fade-shrinks toward its
/// base like a tall blade), `uv.y` is the across-card fraction, and `root.w = 1 +
/// variant` marks it a CARD so the shader samples the baked card atlas
/// ([`GrassMaterial::card_atlas`], variant column `root.w − 1`) across it and
/// discards the gaps. Lit with the terrain-up normal like the blades, so it
/// matches the ground. Returns 1.
fn emit_grass_card(buf: &mut GrassMeshBuf, spec: &ClumpSpec) -> u32 {
    let key = GrassTileKey {
        face: 0,
        x: 0,
        y: 0,
    };
    let rng = |salt: u64| blade_hash(spec.seed, key, 0, salt) as f32;
    let p = spec.profile;
    // Per-card size jitter — uniform cards read as a mowed hedge; the painted
    // tips only vary *within* a card, so the card quads themselves must vary too.
    let hw = (p.radius_m * 2.4 * (0.88 + rng(22) * 0.30)).max(0.08); // a slice of meadow
    // Tall enough that the atlas's ragged blade tips read (the painted blades
    // top out well below the cell edge, so the *visual* height is ~0.8 of this).
    let h = (p.height_m * 1.15 * (0.84 + rng(23) * 0.36)).max(0.06);
    let up = spec.up;
    let base = spec.origin;
    let tint = spec.color;
    // Root = the card base; the fade shrinks the card toward it. w = 1 + variant
    // → CARD sampling atlas column `variant` (a per-clump hash, so neighbouring
    // cards differ).
    let variant = (blade_hash(spec.seed, key, 0, 21) * GRASS_CARD_VARIANTS as f64) as u32
        % GRASS_CARD_VARIANTS;
    let root_attr = [base.x, base.y, base.z, 1.0 + variant as f32];
    // Two crossed quads, spun by a per-clump azimuth — every clump sharing the
    // tile's tangent axes lines the crosses up into visible rows.
    let az = rng(24) * std::f32::consts::TAU;
    let (sin_az, cos_az) = az.sin_cos();
    let spun_right = (spec.right * cos_az + spec.fwd * sin_az).normalize_or(spec.right);
    let spun_fwd = (spec.fwd * cos_az - spec.right * sin_az).normalize_or(spec.fwd);
    for across in [spun_right, spun_fwd] {
        let start = buf.positions.len() as u32;
        // BL, BR, TR, TL — uv = (height_frac, across_frac).
        let corners = [
            (base - across * hw, 0.0, 0.0),
            (base + across * hw, 0.0, 1.0),
            (base + across * hw + up * h, 1.0, 1.0),
            (base - across * hw + up * h, 1.0, 0.0),
        ];
        for (pos, vfrac, ufrac) in corners {
            buf.positions.push(pos.to_array());
            buf.normals.push(up.to_array());
            buf.uvs.push([vfrac, ufrac]);
            buf.colors.push([tint.x, tint.y, tint.z, 1.0]);
            buf.roots.push(root_attr);
        }
        buf.indices
            .extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
    }
    1
}

/// The tangent frame + fountain placement of one blade within its clump.
struct BladeFrame {
    right: Vec3,
    up: Vec3,
    fwd: Vec3,
    /// Outward clump-radial direction this blade arches along.
    outward: Vec3,
    /// Normalized root radius within the clump: 0 centre → 1 rim.
    rf: f32,
}

/// Emit one grass blade rooted at `root`, arching along its clump-radial
/// direction so the tuft reads as a **fountain**: centre blades stand near
/// upright (the dome's spire), rim blades arch outward and bow over (the soft
/// fluffy edge), with their length tapered toward the rim so the top rounds off.
/// The resting geometry stays separated (no interpenetration); the in-game wind
/// shader supplies the motion. `seed` + `idx` drive the per-blade variation;
/// `lean_comb` is an optional steady wind comb (metres of tip lean per unit
/// height) for the dry-prairie look.
#[allow(clippy::too_many_arguments)]
fn push_grass_blade(
    buf: &mut GrassMeshBuf,
    root: Vec3,
    frame: BladeFrame,
    color: Vec3,
    seed: u64,
    idx: u64,
    profile: &GrassProfile,
    lod: GrassBladeLod,
    lean_comb: Vec3,
) {
    let key = GrassTileKey {
        face: 0,
        x: 0,
        y: 0,
    };
    let rng = |salt: u64| blade_hash(seed, key, idx, salt) as f32;
    let (right, up, fwd) = (frame.right, frame.up, frame.fwd);
    let rf = frame.rf;

    // Arch direction = the blade's OUTWARD radial direction (so the tuft opens
    // like a fountain), with a little azimuth jitter so the fan isn't a perfect
    // star. The very centre blades (rf≈0) have no meaningful outward dir, so
    // they take a random azimuth and stay near-upright — the spire of the dome.
    let arch = if rf > 0.06 {
        let jaz = (rng(2) - 0.5) * 1.1;
        (Quat::from_axis_angle(up, jaz) * frame.outward).normalize_or(frame.outward)
    } else {
        let az = rng(2) * std::f32::consts::TAU;
        (right * az.cos() + fwd * az.sin()).normalize_or(right)
    };

    // Length: per-blade jitter, then the dome taper (rim blades shorter → the
    // tuft's top rounds off instead of a flat brush).
    let h = profile.height_m * (0.82 + rng(3) * 0.36) * (1.0 - profile.dome * rf * 0.45);

    // Droop grows toward the rim: centre upright, rim arches out and bows over.
    let d = profile.droop * (0.20 + 0.80 * rf);

    // Quadratic-Bézier centreline. The control point lifts up and a touch out;
    // the tip eases outward by `tip_out` and drops from vertical by `d` (the
    // "over" of the fountain) — together a smooth arch filling the clump volume.
    let tip_out = h * (0.10 + d * 0.95);
    let tip_up = h * (1.0 - d * 0.60);
    let ctrl_up = h * (0.62 - d * 0.12);
    let ctrl_out = h * (d * 0.40);
    let p0 = root;
    let p1 = root + up * ctrl_up + arch * ctrl_out + lean_comb * (h * 0.35);
    let p2 = root + up * tip_up + arch * tip_out + lean_comb * (h * 1.0);

    // Width axis: horizontal, perpendicular to the arch, twisted slightly toward
    // up so neighbouring blades present different faces.
    let perp = up.cross(arch).normalize_or(right);
    let twist = (rng(5) - 0.5) * 0.7;
    let side = (perp + up * (twist * 0.30)).normalize_or(perp);

    let width_m = profile.width_m * (0.78 + rng(6) * 0.50);

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
    let color_at = |lighten: f32| {
        [
            tint.x * lighten,
            tint.y * lighten,
            tint.z * lighten,
            blade_h,
        ]
    };

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

    let root_attr = [p0.x, p0.y, p0.z, 0.0];
    let mut push = |pos: Vec3, sway: f32, lighten: f32, nrm: Vec3| {
        buf.positions.push(pos.to_array());
        buf.normals.push(nrm.to_array());
        buf.uvs.push([sway, phase]);
        buf.colors.push(color_at(lighten));
        buf.roots.push(root_attr);
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
        // `Card` clumps don't reach here (they route through `emit_grass_card`);
        // fold into `Wide` defensively so the match stays exhaustive.
        GrassBladeLod::Wide | GrassBladeLod::Card => {
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
    /// The grass *type* every clump in the field is built from (see
    /// [`GrassProfile`]).
    pub profile: GrassProfile,
    /// Base blade colour (linear).
    pub color: Vec3,
    /// Steady "windswept" comb toward `+X`, as a fraction of blade height the
    /// tips lean. `0` = upright fountains (the in-game wind shader animates
    /// those); a positive value bakes the dry-prairie wind lean into the mesh.
    pub wind_lean: f32,
    pub seed: u64,
    /// Blade geometry LOD: `Full`/`Wide` build a fountain of blades per clump;
    /// `Card` builds one crossed-quad billboard tuft per clump (the far-band
    /// representation) — set this to preview the clump-card look.
    pub lod: GrassBladeLod,
}

impl Default for GrassFieldParams {
    fn default() -> Self {
        Self {
            // Dense fluffy meadow: overlapping fountain tufts whose droopy domes
            // close into a continuous soft mass (the reference look), not a spray
            // of separated spikes.
            size_m: 5.0,
            clumps_per_m2: 22.0,
            profile: GrassProfile::default(),
            color: C_GRASS,
            wind_lean: 0.0,
            seed: 0x47_1E_1D,
            lod: GrassBladeLod::Full,
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
        let spec = ClumpSpec {
            origin: Vec3::new(x, 0.0, z),
            right: Vec3::X,
            up: Vec3::Y,
            fwd: Vec3::Z,
            color: params.color,
            // Distinct per-clump RNG stream so neighbouring clumps differ.
            seed: params.seed ^ (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            profile: params.profile,
            lod: params.lod,
            lean,
        };
        if params.lod == GrassBladeLod::Card {
            emit_grass_card(&mut buf, &spec);
        } else {
            emit_grass_clump(&mut buf, &spec);
        }
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

// WGSL-parity smoothstep, safe for descending edges. Never guard the
// denominator with `.max(EPSILON)` — that inverts descending-edge calls into
// a hard step (INC-0005).
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = edge1 - edge0;
    if denom.abs() < f32::EPSILON {
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / denom).clamp(0.0, 1.0);
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
    /// Baked grass clump-card atlas (see
    /// [`build_grass_card_atlas`](crate::ground::build_grass_card_atlas)) the
    /// far-band CARD quads sample: A = coverage (discard), RGB = a modulation
    /// over the per-clump tint. Like the shadow maps, the binding needs a valid
    /// image at creation — every construction site must set it.
    #[texture(5)]
    #[sampler(6)]
    pub card_atlas: Handle<Image>,
}

impl Material for GrassMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/grass.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/grass.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        // Opaque with a fragment `discard` on the far clump cards (KSP-style): the
        // main opaque pass writes depth where the tuft alpha passes, so cards occlude
        // correctly without a transparent sort. Blades have no discard.
        AlphaMode::Opaque
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Blades / cards are single strips seen from both sides.
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
            profile_dry: GrassProfile::fluffy_short(),
            profile_lush: GrassProfile::wispy_tall(),
            blade_lod: GrassBladeLod::Full,
            width_scale: 1.0,
            height_scale: 1.0,
            clump_scale: 1.0,
            seed: 7,
            scatter_regions: Arc::new(Vec::new()),
            lawn_profile: GrassProfile::lawn(),
            lawn_density_per_m2: 22.0,
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
            profile_dry: GrassProfile::fluffy_short(),
            profile_lush: GrassProfile::wispy_tall(),
            blade_lod: GrassBladeLod::Full,
            width_scale: 1.0,
            height_scale: 1.0,
            clump_scale: 1.0,
            seed: 7,
            scatter_regions: Arc::new(Vec::new()),
            lawn_profile: GrassProfile::lawn(),
            lawn_density_per_m2: 22.0,
            forest_cull: 0.0,
        };
        assert!(build_grass_tile_mesh(&input).is_none());
    }

    #[test]
    fn clearing_suppresses_grass_lawn_forces_it() {
        use crate::ground::scatter::{ScatterRegion, ScatterTreatment};
        use thalos_terrain::TerrainFlatten;

        let radius_m = 3_186_000.0;
        let tiles_per_side = 255_000;
        let key = grass_tile_key(DVec3::new(0.2, 0.3, 0.93).normalize(), tiles_per_side);
        let lattice = TileLattice { tiles_per_side };
        let (center_dir, basis) = lattice.frame(key).unwrap();

        let input = |regions: Vec<ScatterRegion>| GrassTileBuildInput {
            key,
            tiles_per_side,
            height_source: Arc::new(ConstantHeightSource::new(2000.0)),
            radius_m,
            sea_level_m: 0.0,
            blades_per_m2: 4.0,
            profile_dry: GrassProfile::fluffy_short(),
            profile_lush: GrassProfile::wispy_tall(),
            blade_lod: GrassBladeLod::Full,
            width_scale: 1.0,
            height_scale: 1.0,
            clump_scale: 1.0,
            seed: 7,
            scatter_regions: Arc::new(regions),
            lawn_profile: GrassProfile::lawn(),
            lawn_density_per_m2: 22.0,
            forest_cull: 0.0,
        };

        // A footprint comfortably larger than one tile, centred on it, so every
        // candidate falls well inside it.
        let region = |treatment| ScatterRegion {
            footprint: TerrainFlatten::new(
                center_dir,
                basis.tangent_x,
                basis.tangent_z,
                500.0,
                500.0,
                50.0,
                2000.0,
                radius_m,
            ),
            treatment,
        };

        // A clearing over the whole tile removes every blade (paving/footprint).
        assert!(build_grass_tile_mesh(&input(vec![region(ScatterTreatment::Clear)])).is_none());

        // A lawn over the same tile still grows a full carpet (forced cover).
        let lawn = build_grass_tile_mesh(&input(vec![region(ScatterTreatment::Lawn)]))
            .expect("lawn grows grass");
        assert!(lawn.blade_count > 0);

        // A clearing wins over a co-located lawn (a building on the lawn clears
        // the grass under it), regardless of region order.
        assert!(
            build_grass_tile_mesh(&input(vec![
                region(ScatterTreatment::Lawn),
                region(ScatterTreatment::Clear),
            ]))
            .is_none()
        );
    }
}
