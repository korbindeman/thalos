//! Ground-LOD terrain material for procedural bodies.
//!
//! Reads the height + albedo + roughness tile attachments produced by
//! [`crate::pipeline::PipelineTileProvider`] and shades them with the
//! shared Hapke BRDF helper (`thalos::lighting::shade_hapke_surface`).
//! Atmospheric scattering for this surface is composited downstream by
//! the `BodySky` fullscreen pass while ground LOD terrain is active —
//! this material's atmosphere block is bound so the material stays
//! self-contained at upload time and so future inline transmittance work
//! doesn't need a fresh binding contract.

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
#[derive(Clone, Copy, ShaderType)]
pub struct BodySkyExtra {
    /// Normalized sun direction in render space (xyz), sun irradiance (w).
    pub sun_dir_flux: Vec4,
    /// Planet center in render space (xyz), planet solid radius in render
    /// units (w).
    pub planet_center_radius: Vec4,
    /// Quaternion rotating render-space directions into the body-local frame
    /// used by terrain/cloud cubemaps.
    pub world_to_body_orientation: Vec4,
}

impl Default for BodySkyExtra {
    fn default() -> Self {
        Self {
            sun_dir_flux: Vec4::ZERO,
            planet_center_radius: Vec4::ZERO,
            world_to_body_orientation: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }
}

/// Local craft-shadow proxy consumed by `body_terrain.wgsl`.
///
/// This is intentionally analytic rather than Bevy CSM state: the terrain
/// pass is a custom UDLOD pipeline and the stock cascades are camera-sized,
/// which makes tiny near-field craft shadows slide and vanish with zoom.
#[derive(Clone, Copy, ShaderType)]
pub struct BodyTerrainShadow {
    /// xyz = craft proxy center in render-space metres, w = capsule radius.
    pub caster_pos_radius: Vec4,
    /// xyz = craft long axis, w = capsule half-length.
    pub caster_axis_half_len: Vec4,
    /// x = strength, y = penumbra width in metres, z = max receiver distance,
    /// w = enabled flag.
    pub params: Vec4,
}

impl Default for BodyTerrainShadow {
    fn default() -> Self {
        Self {
            caster_pos_radius: Vec4::ZERO,
            caster_axis_half_len: Vec4::new(0.0, 1.0, 0.0, 0.0),
            params: Vec4::ZERO,
        }
    }
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
    /// Analytic local craft shadow, evaluated per terrain fragment.
    #[uniform(2)]
    pub craft_shadow: BodyTerrainShadow,
}

impl Material for BodyTerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_terrain/body_terrain.wgsl".into()
    }
}

pub(crate) fn embed_body_terrain_shader(app: &mut App) {
    embedded_asset!(app, "body_terrain.wgsl");
}
