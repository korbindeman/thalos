pub mod bake;
mod film_grain;
mod gas_giant;
pub mod lighting;
mod material;
pub mod post_stack;
mod rings;
pub mod shader_types;
mod solid_planet;
mod texture;

pub use bake::{
    bake_cloud_cover_image, bake_from_body_data, blank_cloud_cover_image,
    equirect_to_cloud_cover_image,
};
pub use film_grain::FilmGrain;
pub use gas_giant::{
    GasGiantLayers, GasGiantMaterial, GasGiantMaterialHandle, GasGiantParams, MAX_PALETTE_STOPS,
};
pub use lighting::{MAX_STARS, SceneLighting, StarLight};
pub use material::{
    AtmosphereBlock, CLOUD_BAND_COUNT, MAX_ECLIPSE_OCCLUDERS, PlanetDetailParams,
    PlanetHaloMaterial, PlanetHaloMaterialHandle, PlanetMaterial, PlanetMaterialHandle,
    PlanetParams,
};
pub use post_stack::space_camera_post_stack;
pub use rings::{
    MAX_RING_STOPS, RingLayers, RingMaterial, RingMaterialHandle, RingParams, build_ring_mesh,
    ring_plane_normal,
};
pub use shader_types::{GpuCellRange, GpuCrater, GpuMaterial};
pub use solid_planet::{SolidPlanetMaterial, SolidPlanetParams};
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
        bevy::shader::load_shader_library!(app, "shaders/lighting.wgsl");
        bevy::shader::load_shader_library!(app, "shaders/atmosphere.wgsl");
        bevy::shader::load_shader_library!(app, "shaders/noise.wgsl");
        app.add_plugins((
            MaterialPlugin::<PlanetMaterial>::default(),
            MaterialPlugin::<PlanetHaloMaterial>::default(),
            MaterialPlugin::<GasGiantMaterial>::default(),
            MaterialPlugin::<RingMaterial>::default(),
            MaterialPlugin::<SolidPlanetMaterial>::default(),
            film_grain::FilmGrainPlugin,
        ));
    }
}
