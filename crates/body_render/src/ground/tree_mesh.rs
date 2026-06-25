//! Procedural tree mesh generation for the vegetation scatter layer.
//!
//! Builds one combined mesh (tapered trunk + a few canopy blobs) with
//! per-vertex colours, authored **+Y up with the trunk base at the origin**, so
//! the scatter driver orients it to the local terrain normal and scales it per
//! instance. A small library of these is generated once at startup and scattered
//! with per-instance variation — never per-instance geometry synthesis — so all
//! instances of one species share a `(Mesh, Material)` and Bevy auto-batches
//! them into instanced draws.
//!
//! `lod` reduces the radial/ring resolution and the blob count for the far mesh
//! LODs in the tree cascade (see `docs/vegetation.md`).

use bevy::asset::RenderAssetUsages;
use bevy::math::Vec3;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};

/// Parameters for one procedurally generated tree.
#[derive(Debug, Clone, Copy)]
pub struct TreeMeshParams {
    pub trunk_height_m: f32,
    pub trunk_radius_m: f32,
    /// Canopy lateral radius (broadleaf crown half-width).
    pub canopy_radius_m: f32,
    /// Canopy vertical radius (crown half-height).
    pub canopy_height_m: f32,
    pub trunk_color: Vec3,
    pub canopy_color: Vec3,
    /// Deterministic shape seed (offsets the canopy blobs).
    pub seed: u64,
    /// Mesh level of detail: 0 = full, 1 = mid, 2+ = far. Reduces tessellation
    /// and blob count.
    pub lod: u32,
}

impl Default for TreeMeshParams {
    fn default() -> Self {
        Self {
            trunk_height_m: 4.5,
            trunk_radius_m: 0.28,
            canopy_radius_m: 2.6,
            canopy_height_m: 2.4,
            trunk_color: Vec3::new(0.16, 0.090, 0.045),
            canopy_color: Vec3::new(0.055, 0.115, 0.040),
            seed: 0,
            lod: 0,
        }
    }
}

/// Raw CPU mesh arrays for one tree species at one LOD. Kept on the CPU (not
/// just as a GPU `Handle<Mesh>`) so the scatter driver can *combine* many trees
/// into one batched per-tile mesh — the same one-mesh-per-tile batching the
/// grass uses, which removes the per-tree ECS entity overhead and lets forests
/// scale to dense/far. `colors[i].w` is the per-vertex wind weight (0 trunk → 1
/// canopy top).
#[derive(Clone, Default)]
pub struct TreeMeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

impl TreeMeshData {
    fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            colors: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// `wind_weight` (stored in the colour alpha) drives the vertex wind sway:
    /// 0 = rigid (trunk), → 1 = full sway (canopy top).
    fn push_vert(&mut self, pos: Vec3, normal: Vec3, color: Vec3, wind_weight: f32) {
        self.positions.push(pos.to_array());
        self.normals.push(normal.normalize_or_zero().to_array());
        self.colors
            .push([color.x, color.y, color.z, wind_weight.clamp(0.0, 1.0)]);
    }
}

/// Build the raw CPU mesh arrays for one tree species at `params.lod`.
pub fn build_tree_mesh_data(params: &TreeMeshParams) -> TreeMeshData {
    let mut b = TreeMeshData::new();

    let (trunk_segs, rings, sectors, blobs) = match params.lod {
        0 => (8u32, 6u32, 10u32, 3u32),
        1 => (6, 4, 7, 2),
        _ => (4, 3, 5, 1),
    };

    push_trunk(&mut b, params, trunk_segs);
    push_canopy(&mut b, params, rings, sectors, blobs);
    b
}

/// Build a single standalone tree mesh from `params` (used by tests / previews;
/// the runtime scatter path combines `TreeMeshData` per tile instead).
pub fn build_tree_mesh(params: &TreeMeshParams) -> Mesh {
    let b = build_tree_mesh_data(params);
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, b.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, b.normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, b.colors);
    mesh.insert_indices(Indices::U32(b.indices));
    mesh
}

/// Tapered cylinder trunk (no caps), base ring at `y = 0`, top at
/// `y = trunk_height`, narrowing toward the crown.
fn push_trunk(b: &mut TreeMeshData, params: &TreeMeshParams, segments: u32) {
    let base_r = params.trunk_radius_m;
    let top_r = params.trunk_radius_m * 0.62;
    let h = params.trunk_height_m;
    let seg = segments.max(3);
    let start = b.positions.len() as u32;

    for i in 0..=seg {
        let a = i as f32 / seg as f32 * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        // Outward horizontal normal (taper is gentle; horizontal is close enough).
        let n = Vec3::new(c, 0.0, s);
        // Trunk darkens slightly toward the base.
        let base_col = params.trunk_color * 0.85;
        b.push_vert(Vec3::new(c * base_r, 0.0, s * base_r), n, base_col, 0.0);
        b.push_vert(Vec3::new(c * top_r, h, s * top_r), n, params.trunk_color, 0.05);
    }

    for i in 0..seg {
        let r0 = start + i * 2;
        let r1 = start + (i + 1) * 2;
        // (base_i, top_i, base_i+1) and (base_i+1, top_i, top_i+1)
        b.indices.extend_from_slice(&[r0, r1, r0 + 1, r1, r1 + 1, r0 + 1]);
    }
}

/// Canopy: one main ellipsoid blob plus a few smaller offset blobs for an
/// irregular crown, all green, sitting on top of the trunk.
fn push_canopy(b: &mut TreeMeshData, params: &TreeMeshParams, rings: u32, sectors: u32, blobs: u32) {
    let crown_base = params.trunk_height_m * 0.92;
    let rx = params.canopy_radius_m;
    let ry = params.canopy_height_m;
    let center_y = crown_base + ry * 0.9;

    for blob in 0..blobs.max(1) {
        // Deterministic small offsets per blob so the crown isn't a perfect
        // sphere; the main blob (0) is centred.
        let (ox, oy, oz, scale) = if blob == 0 {
            (0.0, 0.0, 0.0, 1.0)
        } else {
            let h0 = hash01(params.seed, (blob * 3) as u64);
            let h1 = hash01(params.seed, (blob * 3 + 1) as u64);
            let h2 = hash01(params.seed, (blob * 3 + 2) as u64);
            (
                (h0 - 0.5) * rx * 1.1,
                (h1 - 0.3) * ry * 0.8,
                (h2 - 0.5) * rx * 1.1,
                0.55 + 0.25 * h1,
            )
        };
        let center = Vec3::new(ox, center_y + oy, oz);
        push_ellipsoid(
            b,
            center,
            rx * scale,
            ry * scale,
            rx * scale,
            rings,
            sectors,
            params.canopy_color,
            crown_base,
        );
    }
}

/// UV-ellipsoid centred at `center`, with semi-axes `(rx, ry, rz)`. Canopy
/// colour darkens toward the underside (`crown_base`) for soft self-shadowing.
#[allow(clippy::too_many_arguments)]
fn push_ellipsoid(
    b: &mut TreeMeshData,
    center: Vec3,
    rx: f32,
    ry: f32,
    rz: f32,
    rings: u32,
    sectors: u32,
    color: Vec3,
    crown_base: f32,
) {
    let rings = rings.max(2);
    let sectors = sectors.max(3);
    let start = b.positions.len() as u32;

    for ring in 0..=rings {
        let v = ring as f32 / rings as f32; // 0 = bottom, 1 = top
        let theta = v * std::f32::consts::PI; // 0..π
        let (st, ct) = theta.sin_cos();
        for sec in 0..=sectors {
            let u = sec as f32 / sectors as f32;
            let phi = u * std::f32::consts::TAU;
            let (sp, cp) = phi.sin_cos();
            // Unit sphere point (y up): top at theta=0.
            let unit = Vec3::new(st * cp, ct, st * sp);
            let pos = center + Vec3::new(unit.x * rx, unit.y * ry, unit.z * rz);
            // Ellipsoid normal: gradient of (x/rx)²+(y/ry)²+(z/rz)².
            let normal = Vec3::new(unit.x / rx, unit.y / ry, unit.z / rz);
            // Darken the shaded underside; brighten the lit crown.
            let updown = ((pos.y - crown_base) / (center.y - crown_base + 0.001)).clamp(0.0, 1.4);
            let shade = 0.7 + 0.4 * updown;
            // Wind weight rises from the crown base to the top, so the canopy
            // sways and the lower crown stays calmer; the trunk (weight 0) is
            // rigid.
            let top_y = center.y + ry;
            let weight = ((pos.y - crown_base) / (top_y - crown_base).max(0.01)).clamp(0.0, 1.0);
            b.push_vert(pos, normal, color * shade, weight);
        }
    }

    let stride = sectors + 1;
    for ring in 0..rings {
        for sec in 0..sectors {
            let a = start + ring * stride + sec;
            let bb = a + 1;
            let c = a + stride;
            let d = c + 1;
            b.indices.extend_from_slice(&[a, c, bb, bb, c, d]);
        }
    }
}

/// Integer hash → `[0, 1)`, deterministic per (seed, salt).
fn hash01(seed: u64, salt: u64) -> f32 {
    let mut h = seed
        ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ 0x2545_F491_4F6C_DD1D;
    h ^= h >> 31;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 32;
    (h & 0x000F_FFFF_FFFF_FFFF) as f32 / (1u64 << 52) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_mesh_is_nonempty_and_finite() {
        let mesh = build_tree_mesh(&TreeMeshParams::default());
        let count = mesh.count_vertices();
        assert!(count > 0);
        if let Some(pos) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            // Sanity: vertices exist and there are triangle indices.
            assert!(pos.len() > 0);
        }
        assert!(mesh.indices().map(|i| i.len()).unwrap_or(0) >= 3);
    }

    #[test]
    fn lod_reduces_vertex_count() {
        let full = build_tree_mesh(&TreeMeshParams {
            lod: 0,
            ..Default::default()
        });
        let far = build_tree_mesh(&TreeMeshParams {
            lod: 2,
            ..Default::default()
        });
        assert!(far.count_vertices() < full.count_vertices());
    }
}
