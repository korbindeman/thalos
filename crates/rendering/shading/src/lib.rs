//! Shared celestial-body shading types and shader libraries.
//!
//! The single source of truth for how a body's surface looks, consumed by
//! both render backends in `thalos_body_render`: the `impostor` billboard and
//! the `ground` udlod terrain LOD. Holds the data structures and WGSL
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
mod sky_view;

pub use atmosphere::{AtmosphereBlock, CLOUD_BAND_COUNT};
pub use lighting::{
    MAX_ECLIPSE_OCCLUDERS, MAX_STARS, SCENE_FLUX_SCALE, SURFACE_DIRECT_SCALE, SceneLighting,
    StarLight, spine_parity_exposure,
};
pub use multi_scatter::{
    MULTI_SCATTER_LUT_HEIGHT, MULTI_SCATTER_LUT_WIDTH, MultiScatterLut, bake_multi_scatter_lut,
};
pub use sky_view::{SKY_VIEW_LUT_HEIGHT, SKY_VIEW_LUT_WIDTH, SkyViewLut};

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
        // One scattering model for every participating medium — clouds, vapour
        // cones, fog, dust. Owns the radiance terms, never the march.
        // See `shaders/volumetrics.wgsl` and ADR-20260730T034500Z.
        bevy::shader::load_shader_library!(app, "shaders/volumetrics.wgsl");
        bevy::shader::load_shader_library!(app, "shaders/atmosphere.wgsl");
        bevy::shader::load_shader_library!(app, "shaders/landcover.wgsl");
        // Shared analytic-ocean shading (wave normals + GGX water BRDF +
        // depth-graded subsurface), imported by the `BodySky` fullscreen pass to
        // ray-trace the planet ocean as a smooth math sphere. See
        // `shaders/water.wgsl`.
        bevy::shader::load_shader_library!(app, "shaders/water.wgsl");
        // Shared cascaded sun-shadow sampler, imported by the terrain, tree,
        // grass, and ground-patch materials (one copy instead of four). See
        // `shaders/shadow.wgsl`.
        bevy::shader::load_shader_library!(app, "shaders/shadow.wgsl");
        // Shared cloud sun-transmittance sampler — the receiving half of the
        // one cloud-occlusion field (CLOUD-5 / W2). Imported by every surface
        // material so terrain, foliage, rock, and hull cannot end up under
        // different weather. See `shaders/cloud_shadow.wgsl`.
        bevy::shader::load_shader_library!(app, "shaders/cloud_shadow.wgsl");
        // Shared foliage MATERIAL model (the albedo analogue of `shade_foliage`),
        // imported by both the near mesh trees and the impostor bake so the two
        // cannot drift. See `shaders/foliage.wgsl`.
        bevy::shader::load_shader_library!(app, "shaders/foliage.wgsl");
        // Shared grass-blade vertex displacement, imported by BOTH the main grass
        // shader and its depth-prepass so their depths match (early-Z correctness).
        // See `shaders/grass_displace.wgsl`.
        bevy::shader::load_shader_library!(app, "shaders/grass_displace.wgsl");
    }
}

/// Bevy 0.19's `check_dir_light_mesh_visibility` reuses one
/// `Local<Parallel<Vec<Vec<Entity>>>>` thread-queue across frames *and* across
/// lights, resizing each participating worker's slot to the current light's
/// cascade count. If two shadow-casting directional lights disagree on that
/// count, a worker truncated by the smaller-cascade light and then skipped by
/// the larger-cascade light's `par_iter` gets over-indexed at collection —
/// panicking with `index out of bounds` (observed as a 2-vs-4 mismatch between
/// `SunLight` and the shipyard key light). Keeping one count everywhere makes
/// the thread-queue slots uniform so the over-index can never happen.
pub const SHADOW_CASCADE_COUNT: usize = 2;
