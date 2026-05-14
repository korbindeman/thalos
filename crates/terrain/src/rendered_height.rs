//! Rendered-height helpers shared by ground LOD and local terrain colliders.
//!
//! These utilities intentionally read the baked R16 height cubemap directly.
//! That matches [`PipelineTileProvider`](crate::PipelineTileProvider)'s
//! current UDLOD path and avoids the fuller `sample_static_surface` detail
//! stack, which the rendered terrain does not yet include.

use bevy::math::{DMat3, DQuat, DVec3, Vec3};
use thalos_terrain_gen::{Cubemap, StaticSurfaceData, cubemap::dir_to_face_uv};

/// Tangent basis for a local terrain patch in body-fixed coordinates.
///
/// Local patch axes are `+X = tangent_x`, `+Y = normal/up`, `+Z = tangent_z`.
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
}

/// Decode one rendered-height R16 texel into metres.
#[inline]
pub fn decode_rendered_height_m(texel: u16, height_range_m: f32) -> f32 {
    ((texel as f32 / u16::MAX as f32) * 2.0 - 1.0) * height_range_m
}

/// Height in metres from the same baked R16 cubemap source used by UDLOD.
pub fn rendered_height_m(surface: &StaticSurfaceData, dir: Vec3) -> f32 {
    let texel = cubemap_texel_nearest(&surface.height_cubemap, dir.normalize_or_zero());
    decode_rendered_height_m(texel, surface.height_range)
}

/// Build a body-fixed tangent-plane patch from rendered-height samples.
pub fn build_rendered_terrain_patch(
    surface: &StaticSurfaceData,
    body_radius_m: f64,
    center_dir: DVec3,
    basis: TerrainPatchBasis,
    config: TerrainPatchConfig,
) -> TerrainPatchMesh {
    let resolution = config.resolution.max(2);
    let center_dir = center_dir.normalize();
    let center_height = rendered_height_m(surface, center_dir.as_vec3()) as f64;
    let center_surface_body_m = center_dir * (body_radius_m + center_height);

    let step = (config.half_extent_m * 2.0) / (resolution - 1) as f64;
    let mut vertices_body_m = Vec::with_capacity((resolution * resolution) as usize);
    for z in 0..resolution {
        for x in 0..resolution {
            let local_x = -config.half_extent_m + x as f64 * step;
            let local_z = -config.half_extent_m + z as f64 * step;
            let tangent_point =
                center_surface_body_m + basis.tangent_x * local_x + basis.tangent_z * local_z;
            let dir = tangent_point.normalize();
            let height = rendered_height_m(surface, dir.as_vec3()) as f64;
            vertices_body_m.push(dir * (body_radius_m + height));
        }
    }

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
    }
}

fn cubemap_texel_nearest<T>(cube: &Cubemap<T>, dir: Vec3) -> T
where
    T: Copy + Default,
{
    let (face, u, v) = dir_to_face_uv(dir);
    let res = cube.resolution();
    let x = ((u * res as f32) as u32).min(res - 1);
    let y = ((v * res as f32) as u32).min(res - 1);
    cube.get(face, x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use thalos_terrain_gen::{
        CubemapFace, DetailNoiseParams, IcoBuckets, Material, surface_color::WaterAppearance,
        surface_field::BiomeMixTexel,
    };

    fn synthetic_surface() -> StaticSurfaceData {
        let mut height = Cubemap::<u16>::new(2);
        height.set(CubemapFace::PosX, 1, 1, u16::MAX);
        height.set(CubemapFace::NegX, 0, 0, 0);
        StaticSurfaceData {
            radius_m: 1000.0,
            cubemap_bake_threshold_m: 0.0,
            height_cubemap: height,
            height_range: 200.0,
            albedo_cubemap: Cubemap::new(2),
            material_cubemap: Cubemap::new(2),
            biome_weights_cubemap: Cubemap::<BiomeMixTexel>::new(2),
            roughness_cubemap: Cubemap::new(2),
            normal_cubemap: Cubemap::new(2),
            craters: Vec::new(),
            volcanoes: Vec::new(),
            channels: Vec::new(),
            feature_index: IcoBuckets::empty(),
            detail_params: DetailNoiseParams::default(),
            materials: vec![Material {
                albedo: [0.5, 0.5, 0.5],
                roughness: 0.5,
            }],
            mean_albedo: [0.5, 0.5, 0.5],
            sea_level_m: None,
            water_appearance: None::<WaterAppearance>,
        }
    }

    #[test]
    fn decodes_rendered_height_from_r16_cubemap() {
        let surface = synthetic_surface();
        let dir = Vec3::new(1.0, -0.5, -0.5).normalize();
        assert!((rendered_height_m(&surface, dir) - 200.0).abs() < 0.01);

        let dir = Vec3::new(-1.0, -0.5, -0.5).normalize();
        assert!((rendered_height_m(&surface, dir) + 200.0).abs() < 0.01);
    }

    #[test]
    fn terrain_patch_uses_requested_resolution() {
        let surface = synthetic_surface();
        let basis = TerrainPatchBasis::from_normal(DVec3::X);
        let patch = build_rendered_terrain_patch(
            &surface,
            1000.0,
            DVec3::X,
            basis,
            TerrainPatchConfig {
                half_extent_m: 16.0,
                resolution: 5,
            },
        );
        assert_eq!(patch.vertices_body_m.len(), 25);
        assert_eq!(patch.indices.len(), 32);
    }
}
