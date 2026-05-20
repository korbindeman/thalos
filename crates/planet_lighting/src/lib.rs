//! Shared planet lighting types and shader libraries.
//!
//! Holds the data structures and WGSL libraries every planet-surface
//! material reads from: scene lighting (stars, eclipse occluders,
//! planetshine), per-body atmosphere parameters, and the shared Hapke
//! BRDF helper. Lives in its own crate so [`thalos_terrain_render`] and
//! [`thalos_planet_rendering`] can both depend on a single source of
//! truth without one having to depend on the other.
//!
//! ## Plugin
//!
//! [`PlanetLightingPlugin`] registers the two shader libraries
//! ([`thalos::lighting`] and [`thalos::atmosphere`]). Bring it in once
//! per app — `thalos_planet_rendering` and `thalos_terrain_render` both rely
//! on the libraries being available globally.

mod atmosphere;
mod lighting;
mod multi_scatter;

pub use atmosphere::{AtmosphereBlock, CLOUD_BAND_COUNT};
pub use lighting::{MAX_ECLIPSE_OCCLUDERS, MAX_STARS, SceneLighting, StarLight};
pub use multi_scatter::{
    MULTI_SCATTER_LUT_HEIGHT, MULTI_SCATTER_LUT_WIDTH, bake_multi_scatter_lut,
};

use bevy::prelude::*;

/// Sun irradiance at 1 AU in shader units (W/m² scaled).
pub const LIGHT_AT_1AU: f32 = 10.0;

/// Meters per astronomical unit.
pub const AU_M: f64 = 1.496e11;

/// Bevy plugin that registers the shared `thalos::lighting` and
/// `thalos::atmosphere` WGSL shader libraries. Must be added before
/// any material that imports either module.
pub struct PlanetLightingPlugin;

impl Plugin for PlanetLightingPlugin {
    fn build(&self, app: &mut App) {
        bevy::shader::load_shader_library!(app, "shaders/lighting.wgsl");
        bevy::shader::load_shader_library!(app, "shaders/atmosphere.wgsl");
    }
}
