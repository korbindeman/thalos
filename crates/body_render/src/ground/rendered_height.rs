//! Body-fixed terrain-patch builder.
//!
//! The patch sources every vertex height from [`pipeline::rendered_height_m`],
//! the canonical query for "what does UDLOD draw at this direction?". By
//! routing through the same function the atlas tile baker uses, the collider
//! mesh tracks the rendered surface across all three pipeline stages —
//! nearest-cubemap base, dynamic layers (ice, dunes), and procedural noise +
//! erosion detail. The patch evaluates detail at its own vertex spacing so
//! the LOD plan matches the resolution the mesh can faithfully represent.

use bevy::math::{DMat3, DQuat, DVec3};
use rayon::prelude::*;
use thalos_terrain::{DynamicSurfaceState, PlanetSurface};

use crate::ground::height_source::HeightSource;
use crate::ground::pipeline::rendered_height_m;

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
    /// Metric lateral half-extent of the patch, for window-relative collider
    /// rebuild scheduling. For the tangent-grid builder this is the config
    /// half-extent; the tile-based builder reports its texel-window extent.
    pub half_extent_m: f64,
}

struct BorrowedRenderedHeightSource<'a> {
    surface: &'a PlanetSurface,
    dynamic_state: &'a DynamicSurfaceState,
}

impl HeightSource for BorrowedRenderedHeightSource<'_> {
    fn sample_height_m(&self, dir: bevy::math::Vec3, tile_lod_m: f32) -> Option<f32> {
        Some(rendered_height_m(
            &thalos_terrain::SurfaceRef {
                surface: self.surface,
                dynamic_state: self.dynamic_state,
            },
            dir,
            tile_lod_m,
        ))
    }
}

/// Build a body-fixed tangent-plane patch from canonical rendered-height
/// samples.
///
/// Each vertex direction is sampled via [`rendered_height_m`] at
/// `tile_lod_m = vertex_spacing`, so the patch represents the same height
/// field the UDLOD ground would show at that resolution. Vertex evaluation
/// is parallelised with rayon — at the default 129² resolution this is ~16K
/// noise + erosion evaluations per rebuild, which would otherwise dominate
/// the main thread.
pub fn build_rendered_terrain_patch(
    surface: &PlanetSurface,
    dynamic_state: &DynamicSurfaceState,
    body_radius_m: f64,
    center_dir: DVec3,
    basis: TerrainPatchBasis,
    config: TerrainPatchConfig,
) -> TerrainPatchMesh {
    let source = BorrowedRenderedHeightSource {
        surface,
        dynamic_state,
    };
    build_rendered_terrain_patch_from_source(&source, body_radius_m, center_dir, basis, config)
}

pub fn build_rendered_terrain_patch_from_source(
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

#[cfg(test)]
mod tests {
    use super::*;
    use thalos_terrain::{
        Cubemap, CubemapFace, DetailNoiseParams, DynamicSurfaceLayers, IcoBuckets, Material,
        StaticSurfaceData, surface_color::WaterAppearance, surface_field::BiomeMixTexel,
    };

    fn synthetic_static() -> StaticSurfaceData {
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
            runtime_detail: Default::default(),
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

    fn synthetic_surface() -> PlanetSurface {
        PlanetSurface {
            static_surface: synthetic_static(),
            dynamic_layers: DynamicSurfaceLayers::default(),
            tectonics: None,
        }
    }

    #[test]
    fn terrain_patch_uses_requested_resolution() {
        let surface = synthetic_surface();
        let state = DynamicSurfaceState::for_layers(&surface.dynamic_layers);
        let basis = TerrainPatchBasis::from_normal(DVec3::X);
        let patch = build_rendered_terrain_patch(
            &surface,
            &state,
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
