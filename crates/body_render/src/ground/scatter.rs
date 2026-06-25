//! Vegetation scatter system: the shared placement gate plus per-tile instance
//! scatter for shrubs and trees.
//!
//! Grass ([`crate::ground::vegetation`]) bakes thousands of blades into one
//! mesh per tile; shrubs and trees instead resolve to *discrete instances*
//! that the game-side driver realizes through an LOD cascade (mesh LODs →
//! octahedral impostor → terrain albedo). Both share:
//!
//! - the cube-sphere [`TileLattice`](crate::ground::tile_lattice) and
//!   deterministic hashed placement, and
//! - [`placement_gate`], the single definition of *where vegetation can grow*
//!   (the slope/curvature material-mask gate the terrain shader bakes from,
//!   plus the surface normal), so the layers never disagree.
//!
//! This module is pure data + math (no ECS); the per-tile lifecycle, anchoring,
//! and LOD selection live in `thalos_game::rendering::vegetation`.

use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::math::{DVec3, Quat, Vec3};
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use thalos_terrain::TerrainFlatten;

use crate::ground::height_source::HeightSource;
use crate::ground::pipeline::material_masks_from_heights;
use crate::ground::rendered_height::TerrainPatchBasis;
use crate::ground::tile_lattice::{TileKey, TileLattice, cube_dir};
use crate::ground::tree_mesh::TreeMeshData;

/// LOD sample hint for placement height queries — small enough to engage the
/// full near-field cascade, matching what the player stands on.
const PLACEMENT_LOD_M: f32 = 0.5;

/// Finite-difference step for the per-candidate slope/curvature probe. Matches
/// the lower clamp of the tile baker's mask step so the CPU gate reproduces the
/// shader's near-field grass mask.
const MASK_STEP_M: f64 = 2.0;

/// Hard ceiling on candidates per species per tile, against pathological
/// density configs at coarse (large-area) LOD rings.
const MAX_CANDIDATES_PER_TILE: usize = 16_384;

/// How far beyond a flatten pad's ramp the forest stays cleared (metres). The
/// tree/shrub density fades from 0 at the pad edge to full over this margin, so
/// the airfield sits in an open clearing instead of a wall of trees.
const VEG_CLEARING_MARGIN_M: f32 = 320.0;

// ---------------------------------------------------------------------------
// Shared placement gate
// ---------------------------------------------------------------------------

/// What the shared placement gate resolves at one candidate direction. Callers
/// apply per-species thresholds (slope limit, altitude band, accept
/// probability) on top of this.
#[derive(Debug, Clone, Copy)]
pub struct PlacementSample {
    /// Terrain height, metres above the body's reference radius.
    pub height_m: f32,
    /// Grass material-mask weight in `[0, 1]` (the terrain shader's grass
    /// channel — high on grassland, low on rock/soil).
    pub grass_w: f32,
    /// Slope magnitude `|∇h|` (rise over run).
    pub slope: f32,
    /// Body-fixed terrain normal at the sample.
    pub normal_body: DVec3,
}

/// Evaluate the shared vegetation placement gate at body-fixed unit direction
/// `dir`. Returns `None` when any height probe is missing (off the resident
/// atlas) — callers skip the candidate.
///
/// This is the exact slope/curvature/mask/normal block grass placement uses,
/// factored out so grass, shrubs, and trees share one definition.
pub fn placement_gate(
    source: &dyn HeightSource,
    basis: &TerrainPatchBasis,
    dir: DVec3,
    radius_m: f64,
) -> Option<PlacementSample> {
    let height_m = source.sample_height_m(dir.as_vec3(), PLACEMENT_LOD_M)?;

    let p = dir * radius_m;
    let probe =
        |offset: DVec3| source.sample_height_m((p + offset).normalize().as_vec3(), PLACEMENT_LOD_M);
    let (Some(h_l), Some(h_r), Some(h_d), Some(h_u)) = (
        probe(basis.tangent_x * -MASK_STEP_M),
        probe(basis.tangent_x * MASK_STEP_M),
        probe(basis.tangent_z * -MASK_STEP_M),
        probe(basis.tangent_z * MASK_STEP_M),
    ) else {
        return None;
    };

    let masks = material_masks_from_heights(height_m, h_l, h_r, h_d, h_u, MASK_STEP_M as f32);
    let grass_w = masks[0] as f32 / 255.0;

    let grad_x = (h_r - h_l) / (2.0 * MASK_STEP_M as f32);
    let grad_z = (h_u - h_d) / (2.0 * MASK_STEP_M as f32);
    let slope = (grad_x * grad_x + grad_z * grad_z).sqrt();
    let normal_body = basis
        .local_to_body_vec(DVec3::new(-grad_x as f64, 1.0, -grad_z as f64))
        .normalize();

    Some(PlacementSample {
        height_m,
        grass_w,
        slope,
        normal_body,
    })
}

// ---------------------------------------------------------------------------
// Species + instances
// ---------------------------------------------------------------------------

/// Which vegetation layer a species belongs to — selects the payload and the
/// far end of its LOD cascade (grass → terrain albedo, shrub → fade out, tree
/// → impostor → fold).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VegLayer {
    GroundCover,
    Shrub,
    Tree,
}

/// Placement parameters for one species. Asset-handle-free, so it can be
/// snapshotted into the async scatter build; the game-side `SpeciesLibrary`
/// pairs each entry (by index) with its mesh-LOD chain, material, and impostor.
#[derive(Debug, Clone, Copy)]
pub struct VegSpeciesPlacement {
    pub layer: VegLayer,
    /// Candidate density per square metre before gates thin it out.
    pub density_per_m2: f32,
    /// Uniform-scale range `(min, max)` applied per instance.
    pub scale_range: (f32, f32),
    /// Reject candidates on terrain steeper than this (rise/run).
    pub slope_limit: f32,
    /// Altitude bands `(lush_lo, lush_hi, fade_lo, fade_hi)` in metres: full
    /// density up to `fade_lo`, smoothstep out to `fade_hi`.
    pub altitude_band: (f32, f32, f32, f32),
    /// Clumping affinity: 0 = uniform scatter, 1 = tight groves.
    pub clump_affinity: f32,
    /// Require the grass material-mask weight to exceed this (trees tolerate a
    /// little soil; grass wants near-pure grassland).
    pub min_grass_w: f32,
}

/// One placed plant, in the tile's local frame. Appearance variation beyond
/// these spatial fields (tint, wind phase) is derived in-shader from the
/// world-space root, so instances of one species stay auto-batchable.
#[derive(Debug, Clone, Copy)]
pub struct VegInstance {
    pub species: u16,
    /// Root position, body-fixed metres, relative to the tile's surface centre.
    pub root_offset_body_m: Vec3,
    /// Local up (body-fixed terrain normal) for orienting the plant.
    pub up_body: Vec3,
    pub yaw: f32,
    pub scale: f32,
    /// Small lean off local-up (radians), for natural variation.
    pub tilt: f32,
}

/// A finished scatter tile: the instances plus the f64 anchor + staleness
/// metadata, mirroring [`GrassTileMesh`](crate::ground::vegetation::GrassTileMesh).
pub struct VegScatterTile {
    pub center_surface_body_m: DVec3,
    pub instances: Vec<VegInstance>,
    /// `HeightSource::revision()` at build time.
    pub built_revision: u64,
    /// Sampled terrain height at the tile centre, for cheap rebuild gating.
    pub center_height_m: f32,
}

/// Everything [`build_scatter_tile`] needs, snapshotted by the driver so the
/// build runs on the async compute pool without touching ECS.
pub struct VegScatterInput {
    pub key: TileKey,
    pub lattice: TileLattice,
    pub radius_m: f64,
    pub height_source: Arc<dyn HeightSource>,
    /// Placement params for every species this layer scatters.
    pub species: Arc<[VegSpeciesPlacement]>,
    pub seed: u64,
    /// Sea level (m above reference radius); plants require
    /// `height > sea_level + 1 m`. Pass `f32::MIN` for bodies without oceans.
    pub sea_level_m: f32,
    /// Active terrain-flatten pad (e.g. the runway); plants are skipped where
    /// the pad has meaningful weight.
    pub flatten_exclusion: Option<TerrainFlatten>,
}

/// Build one scatter tile: deterministic jittered-grid placement per species,
/// gated by [`placement_gate`] + per-species slope/altitude/clump, with
/// per-instance variation hashed from the candidate. Pure and deterministic for
/// a given input + source state; intended for `AsyncComputeTaskPool`. Returns
/// `None` when no instance passes the gates.
pub fn build_scatter_tile(input: &VegScatterInput) -> Option<VegScatterTile> {
    let (center_dir, basis) = input.lattice.frame(input.key)?;
    let source = input.height_source.as_ref();
    let built_revision = source.revision();

    let center_height_m = source.sample_height_m(center_dir.as_vec3(), PLACEMENT_LOD_M)?;
    let center_surface_body_m = center_dir * (input.radius_m + center_height_m as f64);

    let (u_lo, u_hi, v_lo, v_hi) = input.lattice.uv_span(input.key);
    let (ext_u_m, ext_v_m) = input.lattice.tile_extents_m(input.key, input.radius_m);
    let area_m2 = (ext_u_m * ext_v_m).max(0.0);

    let mut instances = Vec::new();
    for (sp_idx, sp) in input.species.iter().enumerate() {
        let count = ((area_m2 * sp.density_per_m2 as f64).round() as usize)
            .min(MAX_CANDIDATES_PER_TILE);
        for cand in 0..count {
            let rng = |salt: u64| veg_hash(input.seed, input.key, sp_idx as u64, cand as u64, salt);

            let u = u_lo + rng(0) * (u_hi - u_lo);
            let v = v_lo + rng(1) * (v_hi - v_lo);
            let dir = cube_dir(input.key.face, u, v);

            if let Some(flatten) = &input.flatten_exclusion
                && flatten.weight(dir) > 0.05
            {
                continue;
            }

            let Some(sample) = placement_gate(source, &basis, dir, input.radius_m) else {
                continue;
            };
            if sample.height_m <= input.sea_level_m + 1.0 {
                continue;
            }
            if sample.slope > sp.slope_limit || sample.grass_w < sp.min_grass_w {
                continue;
            }

            let alt = altitude_fade(sample.height_m, sp.altitude_band);
            let clump = clump_field(dir, sp.layer, sp.clump_affinity);
            let mut accept = sample.grass_w * alt * clump;
            // Clearing around a flatten pad (e.g. the runway): the forest fades
            // out approaching the airfield over a margin beyond the pad ramp, so
            // trees don't crowd right up to the strip.
            if let Some(flatten) = &input.flatten_exclusion {
                let cos = dir.dot(flatten.center_dir).clamp(-1.0, 1.0);
                let d = (cos.acos() * input.radius_m) as f32;
                let pad_reach = (flatten.half_along_m.hypot(flatten.half_across_m) + flatten.ramp_m)
                    as f32;
                accept *= smoothstep(pad_reach, pad_reach + VEG_CLEARING_MARGIN_M, d);
            }
            if (rng(2) as f32) >= accept {
                continue;
            }

            let root_body = dir * (input.radius_m + sample.height_m as f64);
            instances.push(VegInstance {
                species: sp_idx as u16,
                root_offset_body_m: (root_body - center_surface_body_m).as_vec3(),
                up_body: sample.normal_body.as_vec3(),
                yaw: (rng(3) * std::f64::consts::TAU) as f32,
                scale: lerp(sp.scale_range.0, sp.scale_range.1, rng(4) as f32),
                tilt: (rng(5) * 0.12) as f32,
            });
        }
    }

    (!instances.is_empty()).then_some(VegScatterTile {
        center_surface_body_m,
        instances,
        built_revision,
        center_height_m,
    })
}

// ---------------------------------------------------------------------------
// Per-tile mesh combine (one batched mesh per tile, grass-style)
// ---------------------------------------------------------------------------

/// Bake all of a tile's instances into ONE mesh — the same one-mesh-per-tile
/// batching the grass uses, so there's no per-tree ECS entity and forests scale
/// to dense/far. Each instance's species mesh `species_lod[inst.species]`
/// (`None` skips it — e.g. shrubs outside the near band) is transformed by the
/// instance (orient to terrain normal, yaw, lean, scale) and appended.
///
/// Vertices are relative to the tile's surface centre (like the grass tile
/// mesh, for f64 anchoring). The tree's **base** (its root offset from the tile
/// centre) is baked into `UV_0.xy` + `UV_1.x` so the shader can scale-fade each
/// tree about its own root and hash a stable per-tree wind/tint seed from it.
/// `COLOR` carries the species tint (`rgb`) + per-vertex wind weight (`a`).
///
/// Returns `None` if nothing was emitted.
pub fn combine_tree_tile_mesh(
    instances: &[VegInstance],
    species_lod: &[Option<Arc<TreeMeshData>>],
) -> Option<Mesh> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut uv0: Vec<[f32; 2]> = Vec::new();
    let mut uv1: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for inst in instances {
        let Some(Some(data)) = species_lod.get(inst.species as usize) else {
            continue;
        };
        let rot = Quat::from_rotation_arc(Vec3::Y, inst.up_body.normalize_or(Vec3::Y))
            * Quat::from_rotation_y(inst.yaw)
            * Quat::from_rotation_x(inst.tilt);
        let base = inst.root_offset_body_m; // tree base, tile-centre-relative
        let base_uv0 = [base.x, base.y];
        let base_uv1 = [base.z, 0.0];
        let start = positions.len() as u32;
        for i in 0..data.positions.len() {
            let p = Vec3::from_array(data.positions[i]);
            let n = Vec3::from_array(data.normals[i]);
            let wp = base + rot * (p * inst.scale);
            positions.push(wp.to_array());
            normals.push((rot * n).normalize_or_zero().to_array());
            colors.push(data.colors[i]);
            uv0.push(base_uv0);
            uv1.push(base_uv1);
        }
        indices.extend(data.indices.iter().map(|idx| start + idx));
    }

    if positions.is_empty() {
        return None;
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
    Some(mesh)
}

/// Bake a tile's **tree** instances into ONE billboard-quad mesh for the
/// octahedral impostor far band: 4 verts + 2 triangles per tree, no canopy
/// geometry. Per vertex the tree base goes in `POSITION` (all four corners
/// share it — degenerate for the standard prepass, expanded into a camera-facing
/// card by `TreeImpostorMaterial`'s vertex shader), the terrain up in `NORMAL`,
/// the card corner in `UV_0`, `(instance scale, atlas species index)` in `UV_1`,
/// and `(tint.rgb, yaw / TAU)` in `COLOR`.
///
/// `atlas_species[species]` maps a placement species index to its atlas layer,
/// or `None` for species without an impostor (shrubs) — those instances are
/// skipped. Returns `None` if nothing was emitted.
pub fn combine_impostor_tile_mesh(
    instances: &[VegInstance],
    atlas_species: &[Option<u32>],
) -> Option<Mesh> {
    const CORNERS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut uv0: Vec<[f32; 2]> = Vec::new();
    let mut uv1: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for inst in instances {
        let Some(Some(layer)) = atlas_species.get(inst.species as usize) else {
            continue;
        };
        let base = inst.root_offset_body_m.to_array();
        let up = inst.up_body.normalize_or(Vec3::Y).to_array();
        let yaw01 = (inst.yaw / std::f32::consts::TAU).rem_euclid(1.0);
        let start = positions.len() as u32;
        for corner in CORNERS {
            positions.push(base);
            normals.push(up);
            colors.push([1.0, 1.0, 1.0, yaw01]);
            uv0.push(corner);
            uv1.push([inst.scale, *layer as f32]);
        }
        indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
    }

    if positions.is_empty() {
        return None;
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
    Some(mesh)
}

// ---------------------------------------------------------------------------
// Clumping field
// ---------------------------------------------------------------------------

/// Angular frequency of the forest-patch mask. At planet radius this sets the
/// patch scale: `~ radius / FREQ` metres per lattice cell (≈ a few hundred m on
/// a ~3000 km body), so groves read as patches the player can walk between — not
/// the near-constant ~100 km blanket a low frequency gives.
const FOREST_PATCH_FREQ: f64 = 8000.0;

/// Forest-patch clumping in `[0, 1]`. A domain-warped value-noise fBM at
/// patch scale ([`FOREST_PATCH_FREQ`]), **contrasted** (smoothstep) into real
/// groves and real clearings rather than a smooth everywhere-gradient — noise is
/// legitimate here because this is placement *breakup*, not visible terrain
/// height/albedo (CLAUDE.md's process-first rule). Shrubs hug the grove edges;
/// ground cover is full in clearings and sparser on the forest floor.
pub fn clump_field(dir: DVec3, layer: VegLayer, affinity: f32) -> f32 {
    // Warp then sample so patch edges aren't grid-aligned; contrast into patches.
    let warp = fbm(dir * (FOREST_PATCH_FREQ * 0.45), 2) as f64;
    let mask = fbm(dir * FOREST_PATCH_FREQ + DVec3::splat(warp * 5.0), 4);
    let forest = smoothstep(0.40, 0.60, mask);
    match layer {
        // affinity 0 = uniform; 1 = trees only in patches (clearings near zero).
        VegLayer::Tree => lerp(1.0, forest, affinity),
        VegLayer::Shrub => {
            // Peak in the grove edge band (undergrowth margins) + a few lone bushes.
            let edge = 1.0 - (forest - 0.5).abs() * 2.0;
            let lone = fbm(dir * (FOREST_PATCH_FREQ * 2.5), 2);
            lerp(lone * 0.35, (edge * 0.7 + forest * 0.4).clamp(0.0, 1.0), affinity)
        }
        VegLayer::GroundCover => 1.0 - 0.5 * forest,
    }
}

/// Value-noise fBM over a direction (treated as a 3D position). Cheap,
/// deterministic, no trig; for placement masks only.
fn fbm(p: DVec3, octaves: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 0.5f32;
    let mut freq = 1.0f64;
    let mut norm = 0.0f32;
    for _ in 0..octaves.max(1) {
        sum += amp * value_noise(p * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm.max(f32::EPSILON)
}

/// Trilinearly-interpolated value noise in `[0, 1]`.
fn value_noise(p: DVec3) -> f32 {
    let pf = p.floor();
    let (ix, iy, iz) = (pf.x as i64, pf.y as i64, pf.z as i64);
    let f = p - pf;
    let s = DVec3::new(smooth01(f.x), smooth01(f.y), smooth01(f.z));
    let c = |dx: i64, dy: i64, dz: i64| lattice_value(ix + dx, iy + dy, iz + dz);
    let x00 = lerp(c(0, 0, 0), c(1, 0, 0), s.x as f32);
    let x10 = lerp(c(0, 1, 0), c(1, 1, 0), s.x as f32);
    let x01 = lerp(c(0, 0, 1), c(1, 0, 1), s.x as f32);
    let x11 = lerp(c(0, 1, 1), c(1, 1, 1), s.x as f32);
    let y0 = lerp(x00, x10, s.y as f32);
    let y1 = lerp(x01, x11, s.y as f32);
    lerp(y0, y1, s.z as f32)
}

fn lattice_value(ix: i64, iy: i64, iz: i64) -> f32 {
    // u32 integer hash, kept **WGSL-portable** (no u64): the terrain shader
    // replicates this exactly so the far-ground forest/grass-patch tint lines up
    // with where the trees are placed. Mirror in `body_terrain.wgsl`:
    //   var h = bitcast<u32>(cx)*374761393u + bitcast<u32>(cy)*668265263u
    //         + bitcast<u32>(cz)*2246822519u;
    //   h = (h ^ (h >> 13u)) * 1274126177u; h = h ^ (h >> 16u);
    //   return f32(h) * (1.0 / 4294967296.0);
    let x = (ix as i32) as u32;
    let y = (iy as i32) as u32;
    let z = (iz as i32) as u32;
    let mut h = x
        .wrapping_mul(374_761_393)
        .wrapping_add(y.wrapping_mul(668_265_263))
        .wrapping_add(z.wrapping_mul(2_246_822_519));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    h as f32 / 4_294_967_296.0
}

fn smooth01(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Altitude density factor: full below `fade_lo`, smoothstep to zero by
/// `fade_hi`. The lush band `(lush_lo, lush_hi)` is consumed by appearance
/// (tint) code, not density, so only the fade band gates count here.
fn altitude_fade(h: f32, band: (f32, f32, f32, f32)) -> f32 {
    let (_lush_lo, _lush_hi, fade_lo, fade_hi) = band;
    1.0 - smoothstep(fade_lo, fade_hi, h)
}

/// Integer-mix hash → `[0, 1)`. Deterministic per (seed, tile, species,
/// candidate, salt); same family as the grass blade hash.
fn veg_hash(seed: u64, key: TileKey, species: u64, cand: u64, salt: u64) -> f64 {
    let mut h = seed
        ^ (key.face as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (key.x as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (key.y as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
        ^ species.wrapping_mul(0x2545_F491_4F6C_DD1D)
        ^ cand.wrapping_mul(0xD6E8_FEB8_6659_FD93)
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

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ground::height_source::ConstantHeightSource;

    fn test_species() -> Arc<[VegSpeciesPlacement]> {
        Arc::from(vec![VegSpeciesPlacement {
            layer: VegLayer::Tree,
            density_per_m2: 0.02,
            scale_range: (0.8, 1.4),
            slope_limit: 0.4,
            altitude_band: (1800.0, 2900.0, 2400.0, 3100.0),
            clump_affinity: 0.0, // uniform, so flat ground reliably scatters
            min_grass_w: 0.3,
        }])
    }

    #[test]
    fn scatter_is_deterministic_and_flat_ground_grows_trees() {
        let lattice = TileLattice::for_body(3_186_000.0, 250.0);
        let input = VegScatterInput {
            key: lattice.key_of(DVec3::new(0.2, 0.3, 0.93).normalize()),
            lattice,
            radius_m: 3_186_000.0,
            height_source: Arc::new(ConstantHeightSource::new(2000.0)),
            species: test_species(),
            seed: 7,
            sea_level_m: 0.0,
            flatten_exclusion: None,
        };
        let a = build_scatter_tile(&input).expect("flat land scatters trees");
        let b = build_scatter_tile(&input).expect("flat land scatters trees");
        assert!(!a.instances.is_empty());
        assert_eq!(a.instances.len(), b.instances.len());
        assert_eq!(
            a.instances[0].root_offset_body_m,
            b.instances[0].root_offset_body_m
        );
    }

    #[test]
    fn no_trees_below_sea_level() {
        let lattice = TileLattice::for_body(3_186_000.0, 250.0);
        let input = VegScatterInput {
            key: TileKey {
                face: 4,
                x: 100,
                y: 100,
            },
            lattice,
            radius_m: 3_186_000.0,
            height_source: Arc::new(ConstantHeightSource::new(-50.0)),
            species: test_species(),
            seed: 7,
            sea_level_m: 0.0,
            flatten_exclusion: None,
        };
        assert!(build_scatter_tile(&input).is_none());
    }
}
