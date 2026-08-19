//! Planetary cloud composition over the shared [`thalos_clouds`] mechanism.
//!
//! The compute marcher, weather cubes, and sun-transmittance cascade live in
//! `thalos_clouds`. This module keeps the BodySky-coupled fullscreen composite
//! and the plugin that installs both halves for the Thalos game.

mod composite;

use bevy::prelude::*;

pub use composite::CloudCompositeMaterial;
pub use thalos_clouds::{
    CLOUD_SHADOW_SIZE, CameraMatrices, CellStyle, CloudDistanceTexture, CloudFillCalibration,
    CloudRenderTexture, CloudShadowBlock, CloudShadowFrame, CloudShadowMap, CloudSurfaceDensityMap,
    CloudTargetMemory, CloudWeatherMap, CloudsConfig, CloudsImage, FILL_LUT_VERSION,
    FillCalibrationInput, RENDER_HEIGHT, RENDER_WIDTH, WEATHER_FACE_SIZE, WEATHER_MIP_LEVELS,
    cell_style, cloud_target_memory, cloud_target_memory_for, cloud_weather_image,
    derive_fill_calibration,
};

/// Installs the shared cloud compute mechanism plus the planetary composite.
pub struct CloudsPlugin;

impl Plugin for CloudsPlugin {
    fn build(&self, app: &mut App) {
        // The compositor clips its transparent cloud layer against the shared
        // analytic-ocean hit. Keep the shader library available when this
        // plugin is used without the full ground-render stack.
        if !app.is_plugin_added::<thalos_ocean::OceanMechanismPlugin>() {
            app.add_plugins(thalos_ocean::OceanMechanismPlugin);
        }
        if !app.is_plugin_added::<thalos_clouds::CloudsPlugin>() {
            app.add_plugins(thalos_clouds::CloudsPlugin);
        }
        composite::embed_cloud_composite_shader(app);
        app.add_plugins(bevy::pbr::MaterialPlugin::<CloudCompositeMaterial>::default());
    }
}
