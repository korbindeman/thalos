//! Shared celestial-body shading types and shader libraries.
//!
//! The single source of truth for how a body's surface looks, consumed by
//! both render backends in this crate: the [`crate::impostor`] billboard and
//! the [`crate::ground`] udlod terrain LOD. Holds the data structures and WGSL
//! libraries every body-surface material reads from: scene lighting (stars,
//! eclipse occluders, planetshine), per-body atmosphere parameters, and the
//! shared Hapke BRDF helper. A backend chooses its geometry but never its own
//! lighting/atmosphere/cloud math — see `docs/architecture.md`.
//!
//! ## Plugin
//!
//! [`PlanetLightingPlugin`] registers the two shader libraries
//! ([`thalos::lighting`] and [`thalos::atmosphere`]). `BodyRenderPlugin` adds
//! it once; the impostor and ground sub-plugins also add it defensively so
//! either works standalone.

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
        bevy::shader::load_shader_library!(app, "shaders/landcover.wgsl");
        // Shared foliage MATERIAL model (the albedo analogue of `shade_foliage`),
        // imported by both the near mesh trees and the impostor bake so the two
        // cannot drift. See `shaders/foliage.wgsl`.
        bevy::shader::load_shader_library!(app, "shaders/foliage.wgsl");
    }
}
