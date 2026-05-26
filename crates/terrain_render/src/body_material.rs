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

/// Maximum number of procedural craft parts projected onto terrain.
///
/// The Apollo starter stack has far fewer parts than this, and excess parts
/// are ignored rather than growing the terrain material's uniform every frame.
pub const MAX_TERRAIN_SHADOW_CASTERS: usize = 16;

/// Local player-vessel shadow proxy consumed by `body_terrain.wgsl`.
///
/// This is intentionally analytic rather than Bevy CSM state: the terrain
/// pass is a custom UDLOD pipeline and the stock cascades are camera-sized,
/// which makes tiny near-field craft shadows slide and vanish with zoom.
#[derive(Clone, Copy, ShaderType)]
pub struct BodyTerrainShadow {
    /// x = strength, y = minimum penumbra width in metres,
    /// z = max receiver distance, w = valid caster count.
    pub params: Vec4,
    /// xyz = part top/near endpoint in render-space metres, w = endpoint radius.
    pub caster_a_radius: [Vec4; MAX_TERRAIN_SHADOW_CASTERS],
    /// xyz = part bottom/far endpoint in render-space metres, w = endpoint radius.
    pub caster_b_radius: [Vec4; MAX_TERRAIN_SHADOW_CASTERS],
}

impl Default for BodyTerrainShadow {
    fn default() -> Self {
        Self {
            params: Vec4::ZERO,
            caster_a_radius: [Vec4::ZERO; MAX_TERRAIN_SHADOW_CASTERS],
            caster_b_radius: [Vec4::ZERO; MAX_TERRAIN_SHADOW_CASTERS],
        }
    }
}

/// Body-fixed phase/debug parameters consumed by `body_terrain.wgsl`.
///
/// Production terrain uses these fields to anchor shader-synthesized albedo
/// breakup and micro-normal detail in body-fixed metres, so the visible surface
/// remains static under time warp and floating-origin shifts. The optional
/// debug mode additionally renders a 3D anti-aliased checkerboard for flat-mode
/// debug terrain. Both paths are evaluated per fragment via small-magnitude
/// inputs, kept well clear of the body-radius f32 noise floor:
///
/// - `view_phase.xyz`: the camera's body-fixed position taken modulo the
///   terrain-detail repeat period per axis, recomputed each frame on the CPU in
///   f64 before downcasting. This is the only term whose source value carries
///   body-scale magnitude, and the modulo happens before the cast.
/// - The shader recovers the fragment's offset from the camera as
///   `info.world_position − view.world_position` (vertex-interpolated,
///   so the rasterizer takes care of smoothness across the triangle).
///   With UDLOD's vertex high-precision branch active — see
///   `TERRAIN_PRECISION_THRESHOLD_M` in `ground_terrain.rs` — that
///   delta is computed via the Taylor relative-position path and stays
///   sub-mm precise near the camera.
/// - `world_to_body_rot` rotates that delta from render space into the
///   body-fixed frame the cell grid lives in, so the pattern doesn't
///   drag under the player's feet as the body spins.
///
/// `params.x`: mode flag — `0.0` disables the checkerboard overlay, `>= 0.5`
///             enables it. Body-fixed detail anchoring is always active.
/// `params.y`: unused (kept for `vec4` alignment).
/// `params.z`: checker cell size in metres.
/// `params.w`: unused.
#[derive(Clone, Copy, ShaderType)]
pub struct BodyTerrainDebug {
    pub params: Vec4,
    /// Body-fixed camera position taken modulo the terrain-detail repeat
    /// period per axis. Updated each frame; w is unused but kept for `vec4`
    /// alignment.
    pub view_phase: Vec4,
    /// Render-space → body-fixed rotation as a quaternion `(x, y, z, w)`.
    /// Equal to the inverse of the body grid's render-space rotation.
    pub world_to_body_rot: Vec4,
}

impl Default for BodyTerrainDebug {
    fn default() -> Self {
        Self {
            params: Vec4::ZERO,
            view_phase: Vec4::ZERO,
            world_to_body_rot: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }
}

#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct BodyTerrainMaterial {
    /// Static Rayleigh + Mie atmosphere parameters. Set once at spawn from
    /// `TerrestrialAtmosphere`; zero for airless bodies (vacuum early-out).
    /// Bound so the material can add atmosphere-driven sky fill on nearby
    /// terrain; `BodySky` still owns camera-path transmittance and haze.
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
    /// Debug overlay parameters. Zeroed by default so production paths
    /// pay nothing; the spawn code enables the checkerboard for flat-mode
    /// debug terrains.
    #[uniform(3)]
    pub debug: BodyTerrainDebug,
    /// Inspection flags for editor/debug views.
    /// x = fullbright albedo output, yzw reserved.
    #[uniform(4)]
    pub inspection: Vec4,
}

impl Material for BodyTerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_terrain_render/body_terrain.wgsl".into()
    }
}

pub(crate) fn embed_body_terrain_shader(app: &mut App) {
    embedded_asset!(app, "body_terrain.wgsl");
}
