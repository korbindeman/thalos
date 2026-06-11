//! Grass-blade decoration layer for vegetated bodies.
//!
//! A camera-local grid of body-fixed **grass tiles** (~25 m tangent squares on
//! a cube-sphere lattice). Each tile is one batched [`Mesh`] of a few thousand
//! tapered blade strips, built off-thread by sampling the body's
//! [`HeightSource`] — the same source the terrain collider uses, so blades sit
//! on the rendered ground by construction. Blade placement reuses the exact
//! slope/curvature mask logic the terrain shader's grass channel is baked
//! from ([`material_masks_from_heights`]), plus above-sea-level and
//! altitude-band gates, so grass appears where the ground *looks* like
//! grassland.
//!
//! This module is pure geometry + material types; the per-frame tile
//! lifecycle (which tiles exist, big_space anchoring, wind/lighting updates)
//! is driven by the game crate (`thalos_game::rendering::grass`).

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
use crate::ground::pipeline::material_masks_from_heights;
use crate::ground::rendered_height::TerrainPatchBasis;

/// Nominal metric side of one grass tile at a cube-face centre. Toward face
/// corners the cube projection shrinks tiles laterally (down to ~1/2 side);
/// the builder compensates by computing blade count from the tile's actual
/// metric area.
pub const GRASS_TILE_SIZE_M: f64 = 25.0;

/// `tile_lod_m` hint for height sampling — small enough to engage the full
/// near-field procedural cascade, matching what the player stands on.
const GRASS_SAMPLE_LOD_M: f32 = 0.5;

/// Finite-difference step for the per-blade slope/curvature probe. Matches
/// the lower clamp of the tile baker's mask step (`step_m.clamp(2.0, 250.0)`)
/// so the CPU gate reproduces the shader's near-field grass mask.
const MASK_STEP_M: f64 = 2.0;

/// Hard ceiling on blades per tile, against pathological density configs.
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
// Cube-sphere tile lattice
// ---------------------------------------------------------------------------

/// One grass tile on the body's cube-sphere lattice. Faces are
/// `0..6 = +X, -X, +Y, -Y, +Z, -Z`; `x, y` index the face's tile grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GrassTileKey {
    pub face: u8,
    pub x: i64,
    pub y: i64,
}

/// Tiles along one cube-face edge for a body, sized so a tile at the face
/// centre is ~`tile_size_m` across (`u = tan θ`, metric `≈ R·du` at centre).
pub fn grass_tiles_per_side(radius_m: f64, tile_size_m: f64) -> i64 {
    ((2.0 * radius_m) / tile_size_m.max(1.0)).ceil().max(1.0) as i64
}

/// Cube face + face uv (each in `[-1, 1]`) of a body-fixed unit direction.
fn cube_face_uv(dir: DVec3) -> (u8, f64, f64) {
    let a = dir.abs();
    if a.x >= a.y && a.x >= a.z {
        if dir.x >= 0.0 {
            (0, -dir.z / a.x, dir.y / a.x)
        } else {
            (1, dir.z / a.x, dir.y / a.x)
        }
    } else if a.y >= a.x && a.y >= a.z {
        if dir.y >= 0.0 {
            (2, dir.x / a.y, -dir.z / a.y)
        } else {
            (3, dir.x / a.y, dir.z / a.y)
        }
    } else if dir.z >= 0.0 {
        (4, dir.x / a.z, dir.y / a.z)
    } else {
        (5, -dir.x / a.z, dir.y / a.z)
    }
}

/// Inverse of [`cube_face_uv`]: unit direction of face uv coordinates.
fn cube_dir(face: u8, u: f64, v: f64) -> DVec3 {
    let d = match face {
        0 => DVec3::new(1.0, v, -u),
        1 => DVec3::new(-1.0, v, u),
        2 => DVec3::new(u, 1.0, -v),
        3 => DVec3::new(u, -1.0, v),
        4 => DVec3::new(u, v, 1.0),
        _ => DVec3::new(-u, v, -1.0),
    };
    d.normalize()
}

/// Tile containing a body-fixed unit direction.
pub fn grass_tile_key(dir: DVec3, tiles_per_side: i64) -> GrassTileKey {
    let (face, u, v) = cube_face_uv(dir.normalize());
    let n = tiles_per_side as f64;
    let to_index = |c: f64| (((c + 1.0) * 0.5 * n).floor() as i64).clamp(0, tiles_per_side - 1);
    GrassTileKey {
        face,
        x: to_index(u),
        y: to_index(v),
    }
}

/// Centre direction + tangent basis of a tile. Returns `None` for keys
/// outside the face grid (callers enumerate raw neighbour offsets).
pub fn grass_tile_frame(key: GrassTileKey, tiles_per_side: i64) -> Option<(DVec3, TerrainPatchBasis)> {
    if key.face > 5 || key.x < 0 || key.y < 0 || key.x >= tiles_per_side || key.y >= tiles_per_side
    {
        return None;
    }
    let n = tiles_per_side as f64;
    let u = -1.0 + (key.x as f64 + 0.5) * 2.0 / n;
    let v = -1.0 + (key.y as f64 + 0.5) * 2.0 / n;
    let center = cube_dir(key.face, u, v);
    Some((center, TerrainPatchBasis::from_normal(center)))
}

/// Face-uv corner span of a tile (`u_lo..u_hi`, `v_lo..v_hi`).
fn tile_uv_span(key: GrassTileKey, tiles_per_side: i64) -> (f64, f64, f64, f64) {
    let n = tiles_per_side as f64;
    let u_lo = -1.0 + key.x as f64 * 2.0 / n;
    let v_lo = -1.0 + key.y as f64 * 2.0 / n;
    (u_lo, u_lo + 2.0 / n, v_lo, v_lo + 2.0 / n)
}

// ---------------------------------------------------------------------------
// Tile mesh builder
// ---------------------------------------------------------------------------

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
    pub seed: u64,
    /// Active terrain-flatten pad (e.g. the runway). Blades are skipped where
    /// the pad has meaningful weight so they never poke through the paving.
    pub flatten_exclusion: Option<TerrainFlatten>,
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
    let (center_dir, basis) = grass_tile_frame(input.key, input.tiles_per_side)?;
    let source = input.height_source.as_ref();
    let built_revision = source.revision();

    let center_height_m = source.sample_height_m(center_dir.as_vec3(), GRASS_SAMPLE_LOD_M)?;
    let center_surface_body_m = center_dir * (input.radius_m + center_height_m as f64);

    // Metric extents from the actual uv span (cube distortion shrinks tiles
    // toward face corners), so blade density stays uniform per square metre.
    let (u_lo, u_hi, v_lo, v_hi) = tile_uv_span(input.key, input.tiles_per_side);
    let u_mid = (u_lo + u_hi) * 0.5;
    let v_mid = (v_lo + v_hi) * 0.5;
    let ext_u_m = (cube_dir(input.key.face, u_hi, v_mid) - cube_dir(input.key.face, u_lo, v_mid))
        .length()
        * input.radius_m;
    let ext_v_m = (cube_dir(input.key.face, u_mid, v_hi) - cube_dir(input.key.face, u_mid, v_lo))
        .length()
        * input.radius_m;
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

        let Some(h) = source.sample_height_m(dir.as_vec3(), GRASS_SAMPLE_LOD_M) else {
            continue;
        };
        if h <= input.sea_level_m + 1.0 {
            continue;
        }

        // Slope/curvature probe — the same stencil + mask math the tile baker
        // writes into the material attachment's grass channel.
        let p = dir * input.radius_m;
        let probe = |offset: DVec3| {
            source.sample_height_m(((p + offset).normalize()).as_vec3(), GRASS_SAMPLE_LOD_M)
        };
        let (Some(h_l), Some(h_r), Some(h_d), Some(h_u)) = (
            probe(basis.tangent_x * -MASK_STEP_M),
            probe(basis.tangent_x * MASK_STEP_M),
            probe(basis.tangent_z * -MASK_STEP_M),
            probe(basis.tangent_z * MASK_STEP_M),
        ) else {
            continue;
        };
        let masks = material_masks_from_heights(h, h_l, h_r, h_d, h_u, MASK_STEP_M as f32);
        let grass_w = masks[0] as f32 / 255.0;

        let grad_x = (h_r - h_l) / (2.0 * MASK_STEP_M as f32);
        let grad_z = (h_u - h_d) / (2.0 * MASK_STEP_M as f32);
        let slope = (grad_x * grad_x + grad_z * grad_z).sqrt();
        if slope > 0.45 {
            continue;
        }

        let accept = smoothstep(0.45, 0.8, grass_w) * (1.0 - smoothstep(GRASS_FADE_LO_M, GRASS_FADE_HI_M, h));
        if (rng(2) as f32) >= accept {
            continue;
        }

        // Terrain normal from the same gradients, in the tile tangent frame
        // (X = tangent_x, Y = up, Z = tangent_z) → body-fixed.
        let normal_body = basis
            .local_to_body_vec(DVec3::new(-grad_x as f64, 1.0, -grad_z as f64))
            .normalize();

        // Blade frame: yaw in the tangent plane + a small random lean.
        let yaw = rng(3) * std::f64::consts::TAU;
        let lean = rng(4) * 0.25;
        let side_body =
            (basis.tangent_x * yaw.cos() + basis.tangent_z * yaw.sin()).normalize();
        let lean_dir = normal_body.cross(side_body).normalize();
        let up_body = (normal_body + lean_dir * lean).normalize();

        let height_m = 0.20 + rng(5) * 0.25;
        let width_m = 0.025 + rng(6) * 0.020;

        let root_body = dir * (input.radius_m + h as f64 - ROOT_SINK_M);
        let root = (root_body - center_surface_body_m).as_vec3();
        let side = side_body.as_vec3();
        let up = up_body.as_vec3();
        let normal = normal_body.as_vec3().to_array();

        // Tint: terrain palette band by altitude, ± value jitter, lit tips.
        let lush = 1.0 - smoothstep(LUSH_LO_M, LUSH_HI_M, h);
        let dry = smoothstep(GRASS_FADE_LO_M, GRASS_FADE_HI_M, h);
        // Lifted above the ground palette (×~1.3) so blades read against the
        // terrain they colour-match; the tip lightening adds the sun-through-
        // blade glow that sells grass at grazing angles.
        let base = C_GRASS.lerp(C_FOREST, lush * 0.6).lerp(C_DRYGRASS, dry);
        let tint = base * (1.10 + 0.40 * rng(7) as f32);
        let dither = rng(8) as f32;
        let color_at = |lighten: f32| [tint.x * lighten, tint.y * lighten, tint.z * lighten, dither];
        let phase = rng(9) as f32;

        // Tapered two-segment strip: 5 verts, 3 triangles.
        //   4        tip
        //  2 3       mid (55 % height, 55 % width)
        // 0   1      root
        let hw = width_m as f32 * 0.5;
        let h_f = height_m as f32;
        let mid = up * (h_f * 0.55);
        let verts = [
            (root - side * hw, 0.0, 0.9),
            (root + side * hw, 0.0, 0.9),
            (root - side * (hw * 0.55) + mid, 0.30, 1.2),
            (root + side * (hw * 0.55) + mid, 0.30, 1.2),
            (root + up * h_f, 1.0, 1.5),
        ];
        let base_index = positions.len() as u32;
        for (pos, sway, lighten) in verts {
            positions.push(pos.to_array());
            normals.push(normal);
            uvs.push([sway, phase]);
            colors.push(color_at(lighten));
        }
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

    let blade_count = (positions.len() / 5) as u32;
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
#[derive(Clone, Copy, ShaderType, Default)]
pub struct GrassParams {
    /// xyz = unit direction toward the star, world render space. w unused.
    pub sun_dir: Vec4,
    /// xyz = wind direction (world render space, roughly tangent at the
    /// camera), w = sway amplitude at the blade tip, metres.
    pub wind: Vec4,
    /// x = animation time (seconds, wall clock), y = fade start distance (m),
    /// z = fade end distance (m), w unused.
    pub time_fade: Vec4,
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
            seed: 7,
            flatten_exclusion: None,
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
            key: GrassTileKey { face: 4, x: 100, y: 100 },
            tiles_per_side: 255_000,
            height_source: Arc::new(ConstantHeightSource::new(-50.0)),
            radius_m: 3_186_000.0,
            sea_level_m: 0.0,
            blades_per_m2: 4.0,
            seed: 7,
            flatten_exclusion: None,
        };
        assert!(build_grass_tile_mesh(&input).is_none());
    }
}
