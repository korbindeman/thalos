pub mod bake;
mod film_grain;
mod gas_giant;
mod map_ocean;
mod material;
pub mod post_stack;
mod proc_impostor;
mod reference_clouds;
mod rings;
pub mod shader_types;
mod solid_planet;
mod texture;

pub use bake::{
    PreparedPlanetBake, bake_from_planet_surface, blank_cloud_cover_image,
    equirect_to_cloud_cover_image, prepare_planet_bake, upload_prepared_bake,
};
pub use film_grain::FilmGrain;
pub use gas_giant::{
    GasGiantLayers, GasGiantMaterial, GasGiantMaterialHandle, GasGiantParams, MAX_PALETTE_STOPS,
};
pub use map_ocean::{MapOceanMaterial, MapOceanParams};
pub use proc_impostor::{bake_impostor_albedo_cube, blank_impostor_cube};
pub use material::{
    PlanetCoastlineParams, PlanetDetailParams, PlanetHaloMaterial, PlanetHaloMaterialHandle,
    PlanetMaterial, PlanetMaterialHandle, PlanetParams, PlanetWaterParams,
};
pub use post_stack::space_camera_post_stack;
pub use reference_clouds::{
    ReferenceClouds, cloud_cover_image_for_body, convert_reference_clouds_when_ready,
    load_reference_cloud_sources, reference_cloud_path,
};
pub use rings::{
    MAX_RING_STOPS, RingLayers, RingMaterial, RingMaterialHandle, RingParams, build_ring_mesh,
    ring_plane_normal,
};
pub use shader_types::{GpuCellRange, GpuCrater, GpuDuneSea, GpuIceCap, GpuRadialFeature};
pub use solid_planet::{SolidPlanetHaloMaterial, SolidPlanetMaterial, SolidPlanetParams};
pub use texture::PlanetTextures;

use bevy::prelude::*;

/// Bevy plugin for planet impostor rendering.
///
/// Add this plugin to any Bevy app that needs to render planets (game, editor, etc.).
/// It registers the `PlanetMaterial` asset type — callers are responsible for
/// spawning entities with the material and updating uniforms per frame.
pub struct PlanetRenderingPlugin;

impl Plugin for PlanetRenderingPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
        }
        bevy::shader::load_shader_library!(app, "shaders/noise.wgsl");
        app.add_plugins(bevy_erosion_filter::ErosionFilterPlugin);
        app.add_plugins((
            MaterialPlugin::<GasGiantMaterial>::default(),
            MaterialPlugin::<RingMaterial>::default(),
            MaterialPlugin::<SolidPlanetMaterial>::default(),
            MaterialPlugin::<SolidPlanetHaloMaterial>::default(),
            MaterialPlugin::<MapOceanMaterial>::default(),
            film_grain::FilmGrainPlugin,
        ));
    }
}
