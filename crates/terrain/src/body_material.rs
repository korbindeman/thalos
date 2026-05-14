//! Ground-LOD terrain material for procedural bodies.
//!
//! Reads the height + albedo + roughness tile attachments produced by
//! [`crate::pipeline::PipelineTileProvider`] and shades them with the
//! shared Hapke BRDF helper (`thalos::lighting::shade_hapke_surface`).
//! Atmospheric scattering for this surface is composited downstream by
//! the `BodySky` fullscreen pass when the camera sits inside the
//! atmosphere shell — this material's atmosphere block is bound so the
//! material stays self-contained at upload time and so future inline
//! transmittance work doesn't need a fresh binding contract.

use bevy::asset::embedded_asset;
use bevy::math::Vec4;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use thalos_planet_lighting::{AtmosphereBlock, SceneLighting};

/// Per-frame dynamic data for `BodySkyMaterial`.
///
/// The fullscreen sky pass needs the explicit sun direction + flux to
/// drive `integrate_atmosphere`, whereas the terrain reads its sun from
/// the bound [`SceneLighting`] uniform.
#[derive(Clone, Copy, ShaderType, Default)]
pub struct BodySkyExtra {
    /// Normalized sun direction in render space (xyz), sun irradiance (w).
    pub sun_dir_flux: Vec4,
    /// Planet center in render space (xyz), planet solid radius in render
    /// units (w).
    pub planet_center_radius: Vec4,
}

#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct BodyTerrainMaterial {
    /// Static Rayleigh + Mie atmosphere parameters. Set once at spawn from
    /// `TerrestrialAtmosphere`; zero for airless bodies (vacuum early-out).
    /// Bound so the material stays self-contained — `body_terrain.wgsl`
    /// doesn't currently read it, but the binding is in place for a
    /// future inline transmittance path.
    #[uniform(0)]
    pub atmosphere: AtmosphereBlock,
    /// Per-frame scene lighting: primary star direction + flux, eclipse
    /// occluders, planetshine parent, ambient floor. The terrain shader
    /// reads `scene.stars[0]` for the sun direction (and shares it with
    /// the impostor's shading path) so both render paths stay aligned.
    #[uniform(1)]
    pub scene: SceneLighting,
}

impl Material for BodyTerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_terrain/body_terrain.wgsl".into()
    }
}

pub(crate) fn embed_body_terrain_shader(app: &mut App) {
    embedded_asset!(app, "body_terrain.wgsl");
}
