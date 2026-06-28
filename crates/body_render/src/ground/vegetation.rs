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

// Altitude gates, mirroring `body_terrain.wgsl`'s ecological bands
// (`LUSH_*_M`, `TREELINE_*_M`): grass density fades out approaching the
// treeline and is gone where the shader paints dry alpine scree.
const GRASS_FADE_LO_M: f32 = 2400.0;
const GRASS_FADE_HI_M: f32 = 3100.0;
const LUSH_LO_M: f32 = 1800.0;
const LUSH_HI_M: f32 = 2900.0;

// Linear-space blade tints, anchored to the terrain shader's palette
// (`C_FOREST` / `C_GRASS` / `C_DRYGRASS` in `body_terrain.wgsl`) so blades
// read as the ground they grow from, not a separate asset.
const C_FOREST: Vec3 = Vec3::new(0.040, 0.066, 0.030);
const C_GRASS: Vec3 = Vec3::new(0.078, 0.112, 0.052);
const C_DRYGRASS: Vec3 = Vec3::new(0.130, 0.132, 0.074);

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

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
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

        // Per-clump (shared by every blade in the tuft): the surface root, the
        // local tangent frame the tuft fans out in, and the altitude colour band.
        let clump_root_body = dir * (input.radius_m + h as f64 - ROOT_SINK_M);
        let clump_root = (clump_root_body - center_surface_body_m).as_vec3();
        let tan_x = basis.tangent_x.as_vec3();
        let tan_z = basis.tangent_z.as_vec3();
        let normal = normal_body.as_vec3();
        // Tuft spread grows gently with blade width so far (wide) clumps stay
        // cohesive tufts rather than isolated cards.
        let spread = (0.14 * input.width_scale as f64).clamp(0.10, 0.50) as f32;

        let lush = 1.0 - smoothstep(LUSH_LO_M, LUSH_HI_M, h);
        let dry = smoothstep(GRASS_FADE_LO_M, GRASS_FADE_HI_M, h);
        let band = C_GRASS.lerp(C_FOREST, lush * 0.6).lerp(C_DRYGRASS, dry);

        // Fan a small cluster of blades per accepted point. Coverage multiplies
        // without re-paying the placement gate, turning a sparse scatter into a
        // thick field (the single biggest lush-vs-sparse lever).
        for clump in 0..input.blades_per_clump.max(1) {
            // Distinct per (candidate, blade-in-tuft); offset salts off the
            // per-clump `rng` so the tuft doesn't correlate with placement.
            let brng = |salt: u64| {
                blade_hash(input.seed, input.key, blade as u64 * 131 + clump as u64, salt ^ 0x55)
            };

            // Sub-offset within the clump, in the tangent plane.
            let ox = (brng(0) as f32 - 0.5) * spread;
            let oz = (brng(1) as f32 - 0.5) * spread;
            let root = clump_root + tan_x * ox + tan_z * oz;

            // Blade frame: yaw in the tangent plane + a small forward lean.
            let yaw = brng(3) * std::f64::consts::TAU;
            let lean = 0.10 + brng(4) * 0.28;
            let side_body =
                (basis.tangent_x * yaw.cos() + basis.tangent_z * yaw.sin()).normalize();
            let lean_dir = normal_body.cross(side_body).normalize();
            let up_body = (normal_body + lean_dir * lean).normalize();
            let side = side_body.as_vec3();
            let up = up_body.as_vec3();
            let bend_dir = lean_dir.as_vec3();

            let height_m = (0.22 + brng(5) * 0.32) * input.height_scale as f64; // 0.22–0.54 m
            let width_m = (0.050 + brng(6) * 0.040) * input.width_scale as f64;
            // Pronounced forward arch — blades flop over rather than spike up.
            let bend = (0.32 + brng(10) as f32 * 0.45) * height_m as f32;

            // Tint: ground-palette band + a *gentle* per-blade hue drift and a
            // small value jitter, kept CLOSE to the ground colour so the field
            // reads as one continuous green mass instead of pale spikes.
            let hue = (brng(11) as f32 - 0.5) * 0.12;
            let tinted = Vec3::new(band.x * (1.0 + hue), band.y, band.z * (1.0 - hue));
            let tint = tinted * (0.85 + 0.28 * brng(7) as f32);
            // color.a carries the blade height (m) for the shader's scale-fade.
            let blade_h = height_m as f32;
            // Root darker (melts into the ground) → tip a touch brighter; far
            // less lift than before, so no washed-out look.
            let color_at =
                |lighten: f32| [tint.x * lighten, tint.y * lighten, tint.z * lighten, blade_h];
            let phase = brng(9) as f32;

            let hw = width_m as f32 * 0.5;
            let h_f = height_m as f32;
            let base_index = positions.len() as u32;
            let mut push_vert = |pos: Vec3, sway: f32, lighten: f32| {
                positions.push(pos.to_array());
                normals.push(normal.to_array());
                uvs.push([sway, phase]);
                colors.push(color_at(lighten));
            };
            match input.blade_lod {
                GrassBladeLod::Full => {
                    // Curved tapered blade: a centreline arcing forward along
                    // `bend_dir`, four cross-sections (root → 40 % → 72 % → tip)
                    // tapering to a point. 7 verts, 5 triangles. uv.x (0 root → 1
                    // tip) drives the wind sway.
                    let arc = |t: f32| root + up * (h_f * t) + bend_dir * (bend * t * t);
                    let c1 = arc(0.40);
                    let c2 = arc(0.72);
                    let tip = arc(1.0);
                    push_vert(root - side * hw, 0.0, 0.60);
                    push_vert(root + side * hw, 0.0, 0.60);
                    push_vert(c1 - side * (hw * 0.74), 0.18, 0.84);
                    push_vert(c1 + side * (hw * 0.74), 0.18, 0.84);
                    push_vert(c2 - side * (hw * 0.42), 0.55, 1.02);
                    push_vert(c2 + side * (hw * 0.42), 0.55, 1.02);
                    push_vert(tip, 1.0, 1.16);
                    indices.extend_from_slice(&[
                        base_index,
                        base_index + 1,
                        base_index + 2,
                        base_index + 1,
                        base_index + 3,
                        base_index + 2,
                        base_index + 2,
                        base_index + 3,
                        base_index + 4,
                        base_index + 3,
                        base_index + 5,
                        base_index + 4,
                        base_index + 4,
                        base_index + 5,
                        base_index + 6,
                    ]);
                }
                GrassBladeLod::Wide => {
                    // Tapered 5-vertex blade (root pair → mid pair → tip), bent
                    // forward — a soft tuft card when widened, NOT a single flat
                    // shard. Holds coverage at low density without the billboard
                    // look the old single triangle had.
                    let mid = root + up * (h_f * 0.55) + bend_dir * (bend * 0.32);
                    let tip = root + up * h_f + bend_dir * (bend * 0.6);
                    push_vert(root - side * hw, 0.0, 0.62);
                    push_vert(root + side * hw, 0.0, 0.62);
                    push_vert(mid - side * (hw * 0.60), 0.5, 0.92);
                    push_vert(mid + side * (hw * 0.60), 0.5, 0.92);
                    push_vert(tip, 1.0, 1.12);
                    indices.extend_from_slice(&[
                        base_index,
                        base_index + 1,
                        base_index + 2,
                        base_index + 1,
                        base_index + 3,
                        base_index + 2,
                        base_index + 2,
                        base_index + 3,
                        base_index + 4,
                    ]);
                }
            }
            blade_count += 1;
        }
    }

    if blade_count == 0 {
        return None;
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));

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
            radius_m: 0.16,
            blade_count: 22,
            height_m: 0.45,
            color: C_GRASS,
            seed: 0x6_8A55,
        }
    }
}

/// Build a single dense grass tuft, origin-centred on flat ground (`+Y` up), as a
/// vertex-coloured blade mesh. Pure + deterministic for a given `params`. The
/// asset baked into the clump octahedral impostor and previewed on its own; the
/// blade geometry deliberately mirrors the [`GrassBladeLod::Full`] arm of
/// [`build_grass_tile_mesh`] so the impostor matches the near field by
/// construction.
pub fn build_grass_clump_mesh(params: &GrassClumpParams) -> Mesh {
    // Synthetic tile key — the clump is placement-free, so any fixed key gives a
    // stable per-blade RNG stream off `params.seed`.
    let key = GrassTileKey {
        face: 0,
        x: 0,
        y: 0,
    };
    let n = params.blade_count.max(1);
    let r = params.radius_m.max(1.0e-3);
    let up = Vec3::Y;

    // Clump-level prevailing direction: every blade arches roughly this way
    // (a tuft combed one direction), with only a moderate fan of variation —
    // not a chaotic every-which-way ball. Per-clump variety in the field comes
    // from the per-instance billboard yaw, not from scrambling each blade here.
    let phi0 = blade_hash(params.seed, key, 0xC1, 0) as f32 * std::f32::consts::TAU;
    // Total fan spread of blade arch azimuth around `phi0` (radians).
    const FAN: f32 = 0.85;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for blade in 0..n {
        let rng = |salt: u64| blade_hash(params.seed, key, blade as u64, salt) as f32;

        // Uniform-density disc placement (sqrt radius), in the tangent plane.
        let ang = rng(0) * std::f32::consts::TAU;
        let rad = r * rng(1).sqrt();
        let root = Vec3::new(rad * ang.cos(), 0.0, rad * ang.sin());

        // Arch azimuth clustered around the clump's prevailing direction `phi0`
        // (± FAN/2): blades mostly arch the same way, with some variation.
        let az = phi0 + (rng(3) - 0.5) * FAN;
        let lean_dir = Vec3::new(az.cos(), 0.0, az.sin());
        // The blade's flat-face (width) axis gets an INDEPENDENT yaw twist, so
        // blades that share an arch direction still cross at angles instead of
        // stacking as identical coplanar sheets ("all inside each other").
        let fy = az + std::f32::consts::FRAC_PI_2 + (rng(8) - 0.5) * 1.3;
        let side = Vec3::new(fy.cos(), 0.0, fy.sin());
        // Mostly upright with a slight, shared lean — not flopped over.
        let lean = 0.07 + rng(4) * 0.18;
        let blade_up = (up + lean_dir * lean).normalize();
        let bend_dir = lean_dir;

        let height_m = params.height_m * (0.70 + rng(5) * 0.6); // ~0.70–1.30×
        let width_m = 0.030 + rng(6) * 0.022; // thin blades
        // Gentle forward arch — a soft bend, not a curl-over.
        let bend = (0.24 + rng(10) * 0.28) * height_m;

        // Tint: ground-palette band + gentle per-blade hue/value drift (kept
        // close to the base so the tuft reads as one green mass).
        let hue = (rng(11) - 0.5) * 0.12;
        let tinted = Vec3::new(
            params.color.x * (1.0 + hue),
            params.color.y,
            params.color.z * (1.0 - hue),
        );
        let tint = tinted * (0.85 + 0.28 * rng(7));
        let blade_h = height_m;
        let phase = rng(9);
        let color_at =
            |lighten: f32| [tint.x * lighten, tint.y * lighten, tint.z * lighten, blade_h];

        let hw = width_m * 0.5;
        let base_index = positions.len() as u32;
        let mut push_vert = |pos: Vec3, sway: f32, lighten: f32| {
            positions.push(pos.to_array());
            normals.push(up.to_array());
            uvs.push([sway, phase]);
            colors.push(color_at(lighten));
        };

        // Curved tapered blade (root → 40 % → 72 % → tip), arching forward along
        // `bend_dir` — the `GrassBladeLod::Full` shape.
        let arc = |t: f32| root + blade_up * (height_m * t) + bend_dir * (bend * t * t);
        let c1 = arc(0.40);
        let c2 = arc(0.72);
        let tip = arc(1.0);
        // Soft root→tip value gradient (gentle, so blades don't read as
        // high-contrast toony shards).
        push_vert(root - side * hw, 0.0, 0.72);
        push_vert(root + side * hw, 0.0, 0.72);
        push_vert(c1 - side * (hw * 0.74), 0.18, 0.86);
        push_vert(c1 + side * (hw * 0.74), 0.18, 0.86);
        push_vert(c2 - side * (hw * 0.42), 0.55, 0.97);
        push_vert(c2 + side * (hw * 0.42), 0.55, 0.97);
        push_vert(tip, 1.0, 1.03);
        indices.extend_from_slice(&[
            base_index,
            base_index + 1,
            base_index + 2,
            base_index + 1,
            base_index + 3,
            base_index + 2,
            base_index + 2,
            base_index + 3,
            base_index + 4,
            base_index + 3,
            base_index + 5,
            base_index + 4,
            base_index + 4,
            base_index + 5,
            base_index + 6,
        ]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
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
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct GrassMaterial {
    #[uniform(0)]
    pub params: GrassParams,
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
