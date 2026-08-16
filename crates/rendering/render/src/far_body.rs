//! Far-body rendering adapter.
//!
//! This adapter owns distant body projection: solid and gas-giant impostors,
//! rings, map-scale ocean projection, and the payloads they consume. It does
//! not install planetary terrain, near-body atmosphere/ocean composites, or
//! application drivers.

use bevy::prelude::*;

/// Installs the concrete far-body projection used by the Thalos application.
///
/// The shared lighting libraries are added defensively so the adapter remains
/// independently composable during the `thalos_body_render` facade migration.
pub struct FarBodyRenderPlugin;

impl Plugin for FarBodyRenderPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
        }
        app.add_plugins(crate::impostor::PlanetRenderingPlugin);
    }
}
