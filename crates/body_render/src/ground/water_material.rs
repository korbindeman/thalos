//! Ground-LOD water surface for ocean bodies.
//!
//! Renders an icosphere at `body_radius + sea_level_m + ε` as an opaque
//! depth-writing pass. The water sits at the same iso-radius the BodySky
//! pass clips at analytically (`body_sky.wgsl::c_planet`), so aerial
//! perspective composites correctly on water pixels without the water
//! shader having to sample scene depth.
//!
//! Calibration of the GGX water BRDF (`α = 0.10`, `F0 = 0.02`,
//! `brdf_scale = 0.5`) matches `planet_impostor.wgsl::shade_water` so the
//! ground-LOD ↔ impostor handoff reads continuously.

use crate::shading::SceneLighting;
use bevy::asset::embedded_asset;
use bevy::math::Vec4;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

/// Per-body water parameters.
#[derive(Clone, Copy, ShaderType, Default)]
pub struct BodyWaterParams {
    /// xyz = deep-water linear-RGB tint, w = minimum optical-depth (metres).
    /// Mirrors the impostor's `water_color_depth` so the bake's
    /// `WaterAppearance` drives both rendering paths.
    pub color_depth: Vec4,
    /// xyz = planet centre in render-space metres, w = water-surface radius
    /// (`body.radius_m + sea_level_m + ε`).
    pub planet_center_radius: Vec4,
    /// xyz unused, w = monotonic animation time in seconds. Drives the
    /// wave-normal noise scroll.
    pub time: Vec4,
}

#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct BodyWaterMaterial {
    #[uniform(0)]
    pub scene: SceneLighting,
    #[uniform(1)]
    pub params: BodyWaterParams,
}

impl Material for BodyWaterMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/body_water.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}

pub(crate) fn embed_body_water_shader(app: &mut App) {
    embedded_asset!(app, "body_water.wgsl");
}
