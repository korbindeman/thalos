//! Shared terrain-height query and local-patch contract.
//!
//! This module is intentionally renderer- and physics-backend-agnostic. The
//! renderer supplies GPU-atlas-backed implementations, while local physics,
//! gameplay, and offline tools consume the same interface.

use glam::{DMat3, DQuat, DVec3, Vec3};
use rayon::prelude::*;

/// Canonical per-body height query used by rendering and near-surface systems.
pub trait HeightSource: Send + Sync {
    /// Height in metres above the body's reference radius at a body-fixed unit
    /// direction. `tile_lod_m` is a scale hint for procedural sources.
    fn sample_height_m(&self, dir: Vec3, tile_lod_m: f32) -> Option<f32>;

    /// Monotonic revision for consumers that cache geometry derived from this
    /// source. Static sources keep the default revision of zero.
    fn revision(&self) -> u64 {
        0
    }

    /// Macro landcover moisture in `[-1, 1]` at a body-fixed unit direction.
    fn landcover_moisture(&self, _dir: DVec3) -> f32 {
        0.0
    }

    /// Build a collider patch from native resident geometry when available.
    /// Sources without such geometry return `None` and callers resample the
    /// height contract onto a tangent grid.
    fn build_collider_patch(
        &self,
        center_dir: Vec3,
        max_resolution: u32,
    ) -> Option<TerrainPatchMesh> {
        let _ = (center_dir, max_resolution);
        None
    }
}

/// Tangent basis for a local terrain patch in body-fixed coordinates.
/// Local axes are `+X = tangent_x`, `+Y = normal/up`, `+Z = tangent_z`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainPatchBasis {
    pub tangent_x: DVec3,
    pub normal: DVec3,
    pub tangent_z: DVec3,
}

impl TerrainPatchBasis {
    pub fn from_normal(normal: DVec3) -> Self {
        let normal = normal.normalize();
        let seed = if normal.y.abs() < 0.9 {
            DVec3::Y
        } else {
            DVec3::X
        };
        let tangent_x = seed.cross(normal).normalize();
        let tangent_z = tangent_x.cross(normal).normalize();
        Self {
            tangent_x,
            normal,
            tangent_z,
        }
    }

    pub fn local_to_body_matrix(self) -> DMat3 {
        DMat3::from_cols(self.tangent_x, self.normal, self.tangent_z)
    }

    pub fn local_to_body_rotation(self) -> DQuat {
        DQuat::from_mat3(&self.local_to_body_matrix())
    }

    pub fn local_to_body_vec(self, local: DVec3) -> DVec3 {
        self.local_to_body_matrix() * local
    }

    pub fn body_to_local_vec(self, body: DVec3) -> DVec3 {
        DVec3::new(
            body.dot(self.tangent_x),
            body.dot(self.normal),
            body.dot(self.tangent_z),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainPatchConfig {
    pub half_extent_m: f64,
    pub resolution: u32,
}

impl Default for TerrainPatchConfig {
    fn default() -> Self {
        Self {
            half_extent_m: 4096.0,
            resolution: 129,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TerrainPatchMesh {
    pub vertices_body_m: Vec<DVec3>,
    pub indices: Vec<[u32; 3]>,
    pub center_surface_body_m: DVec3,
    pub basis: TerrainPatchBasis,
    pub half_extent_m: f64,
}

/// Resample any height source onto a body-fixed tangent-grid patch.
pub fn build_terrain_patch_from_source(
    height_source: &dyn HeightSource,
    body_radius_m: f64,
    center_dir: DVec3,
    basis: TerrainPatchBasis,
    config: TerrainPatchConfig,
) -> TerrainPatchMesh {
    let resolution = config.resolution.max(2);
    let center_dir = center_dir.normalize();
    let step = (config.half_extent_m * 2.0) / (resolution - 1) as f64;
    let tile_lod_m = step.max(1.0) as f32;

    let center_height = height_source
        .sample_height_m(center_dir.as_vec3(), tile_lod_m)
        .unwrap_or(0.0) as f64;
    let center_surface_body_m = center_dir * (body_radius_m + center_height);

    let row_count = resolution as usize;
    let mut vertices_body_m = vec![DVec3::ZERO; row_count * row_count];
    vertices_body_m
        .par_chunks_mut(row_count)
        .enumerate()
        .for_each(|(z, row)| {
            let local_z = -config.half_extent_m + z as f64 * step;
            for (x, slot) in row.iter_mut().enumerate() {
                let local_x = -config.half_extent_m + x as f64 * step;
                let tangent_point =
                    center_surface_body_m + basis.tangent_x * local_x + basis.tangent_z * local_z;
                let dir = tangent_point.normalize();
                let height = height_source
                    .sample_height_m(dir.as_vec3(), tile_lod_m)
                    .unwrap_or(0.0) as f64;
                *slot = dir * (body_radius_m + height);
            }
        });

    let mut indices = Vec::with_capacity(((resolution - 1) * (resolution - 1) * 2) as usize);
    for z in 0..(resolution - 1) {
        for x in 0..(resolution - 1) {
            let i0 = z * resolution + x;
            let i1 = i0 + 1;
            let i2 = i0 + resolution;
            let i3 = i2 + 1;
            indices.push([i0, i2, i1]);
            indices.push([i1, i2, i3]);
        }
    }

    TerrainPatchMesh {
        vertices_body_m,
        indices,
        center_surface_body_m,
        basis,
        half_extent_m: config.half_extent_m,
    }
}
