//! Planetary rendering adapter.
//!
//! This adapter owns near-body spherical composition: cube-sphere terrain,
//! analytic atmosphere/ocean, body-fixed volumetric clouds, and the sealed
//! legacy UDLOD fallback. Application state projection remains in
//! `thalos_runtime`; distant orbital impostors remain in the far-body adapter.

use bevy::prelude::*;

/// Installs the concrete planetary adapter used by the Thalos application.
pub struct PlanetaryRenderPlugin;

impl Plugin for PlanetaryRenderPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
        }
        app.add_plugins(crate::clouds::CloudsPlugin);
        app.add_plugins(crate::ground::GroundAppearancePlugin);
        #[cfg(feature = "legacy-udlod")]
        app.add_plugins(crate::ground::LegacyUdlodPlugin);
        app.add_plugins(crate::tiles::TileTerrainPlugin);
    }
}
