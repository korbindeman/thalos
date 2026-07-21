//! Body-fixed terrain-patch builder.
//!
//! The patch sources every vertex height from [`pipeline::rendered_height_m`],
//! the canonical query for "what does UDLOD draw at this direction?". By
//! routing through the same function the atlas tile baker uses, the collider
//! mesh tracks the rendered surface across all three pipeline stages —
//! nearest-cubemap base, dynamic layers (ice, dunes), and procedural noise +
//! erosion detail. The patch evaluates detail at its own vertex spacing so
//! the LOD plan matches the resolution the mesh can faithfully represent.

use crate::ground::pipeline::rendered_height_m;
use bevy::math::DVec3;
use thalos_terrain::{
    DynamicSurfaceState, HeightSource, PlanetSurface, build_terrain_patch_from_source,
};
pub use thalos_terrain::{TerrainPatchBasis, TerrainPatchConfig, TerrainPatchMesh};

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
    build_terrain_patch_from_source(&source, body_radius_m, center_dir, basis, config)
}

pub fn build_rendered_terrain_patch_from_source(
    height_source: &dyn HeightSource,
    body_radius_m: f64,
    center_dir: DVec3,
    basis: TerrainPatchBasis,
    config: TerrainPatchConfig,
) -> TerrainPatchMesh {
    build_terrain_patch_from_source(height_source, body_radius_m, center_dir, basis, config)
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
