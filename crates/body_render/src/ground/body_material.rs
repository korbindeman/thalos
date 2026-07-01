//! Ground-LOD terrain material for procedural bodies.
//!
//! Reads the height + albedo + roughness tile attachments produced by
//! [`crate::pipeline::PipelineTileProvider`] and shades them by a per-body
//! [`TerrainShadingStyle`] in `body_terrain.wgsl`: a rough-dielectric BRDF
//! (Oren–Nayar diffuse + Cook–Torrance GGX specular) + ecological albedo bands
//! for wet, vegetated terrestrial bodies (Thalos), or the impostor's Hapke
//! regolith model over the baked gray albedo for airless bodies (Mira) so the
//! two render paths reconverge across the impostor↔ground LOD swap.
//! Atmospheric scattering for this surface is composited downstream by
//! the `BodySky` fullscreen pass while ground LOD terrain is active —
//! this material's atmosphere block is bound so the material stays
//! self-contained at upload time and so future inline transmittance work
//! doesn't need a fresh binding contract.

use crate::shading::{AtmosphereBlock, SceneLighting};
use bevy::asset::embedded_asset;
use bevy::math::Vec4;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

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
    /// Volumetric cloud band radii in render units: x = base (planet_radius +
    /// base altitude), y = top. Used by `body_sky.wgsl` to suppress the
    /// composited `cloud_layer` where opaque geometry (ship hull, terrain) sits
    /// in front of the cloud band. z = aerial-perspective airlight ratio.
    /// **w = cloud-composite-enable flag**: 1.0 only on the body whose live
    /// cloud texture is bound; 0.0 everywhere else (and when clouds are disabled
    /// in graphics settings). The shader skips the cloud composite when w < 0.5,
    /// because the 1×1 blank fallback bound on inactive bodies is read out of
    /// bounds by the screen-space `textureLoad` (→ opaque black sky) if
    /// composited. Zero on bodies with no active cloud layer.
    pub cloud_band_radii: Vec4,
    /// Analytic ocean parameters for `body_sky.wgsl`. The sky pass ray-traces a
    /// math sphere at `ocean.x` and shades it as water wherever that hit sits in
    /// front of the opaque seabed/terrain in the scene-depth buffer — smooth at
    /// every scale, with no mesh. x = ocean sphere radius in render units
    /// (`planet_radius + sea_level_m`; render space is 1 unit = 1 m on
    /// SHIP_LAYER, so this is just metres). y = enable flag (1 on ocean bodies,
    /// 0 disables the whole branch). zw reserved.
    pub ocean: Vec4,
    /// Deep-water linear-RGB tint (xyz) + minimum optical-depth scale (w),
    /// matching the impostor water BRDF fallback so the ground↔impostor handoff
    /// stays consistent. Only read when `ocean.y >= 0.5`.
    pub ocean_color_depth: Vec4,
}

impl Default for BodySkyExtra {
    fn default() -> Self {
        Self {
            sun_dir_flux: Vec4::ZERO,
            planet_center_radius: Vec4::ZERO,
            world_to_body_orientation: Vec4::new(0.0, 0.0, 0.0, 1.0),
            cloud_band_radii: Vec4::ZERO,
            ocean: Vec4::ZERO,
            ocean_color_depth: Vec4::ZERO,
        }
    }
}

/// Maximum number of procedural craft parts projected onto terrain.
///
/// Sized for an airliner-class craft: a fuselage loft contributes three
/// tapered segments (nose / barrel / tail) and each podded nacelle one.
/// Excess parts are ignored rather than growing the terrain material's
/// uniform every frame. Must match `MAX_TERRAIN_SHADOW_CASTERS` (and the
/// two array lengths) in `body_terrain.wgsl`.
pub const MAX_TERRAIN_SHADOW_CASTERS: usize = 24;

/// Maximum number of thin planform-quad casters (lifting surfaces). Must
/// match `MAX_TERRAIN_SHADOW_QUADS` (and the four array lengths) in
/// `body_terrain.wgsl`. The Meridian uses 5 (two wings, two tailplanes,
/// one fin).
pub const MAX_TERRAIN_SHADOW_QUADS: usize = 8;

/// Local player-vessel shadow proxy consumed by `body_terrain.wgsl`.
///
/// This is intentionally analytic rather than Bevy CSM state: the terrain
/// pass is a custom UDLOD pipeline and the stock cascades are camera-sized,
/// which makes tiny near-field craft shadows slide and vanish with zoom.
///
/// Two caster primitives: tapered capsule segments for bodies of revolution
/// (tanks, fuselage barrels, nacelles), and thin planform quads for lifting
/// surfaces — a wing modelled as a capsule reads chord-*thick* from the
/// side, throwing an enormous slab at low sun, while the quad projects the
/// true trapezoid at any sun angle and vanishes edge-on.
#[derive(Clone, Copy, ShaderType)]
pub struct BodyTerrainShadow {
    /// x = strength, y = minimum penumbra width in metres,
    /// z = max receiver distance, w = valid capsule caster count.
    pub params: Vec4,
    /// x = valid quad caster count, yzw reserved.
    pub quad_params: Vec4,
    /// xyz = part top/near endpoint in render-space metres, w = endpoint radius.
    pub caster_a_radius: [Vec4; MAX_TERRAIN_SHADOW_CASTERS],
    /// xyz = part bottom/far endpoint in render-space metres, w = endpoint radius.
    pub caster_b_radius: [Vec4; MAX_TERRAIN_SHADOW_CASTERS],
    /// Planform quad corners in render-space metres (w unused), wound
    /// root-leading → tip-leading → tip-trailing → root-trailing so
    /// consecutive corners trace the outline.
    pub quad_a: [Vec4; MAX_TERRAIN_SHADOW_QUADS],
    pub quad_b: [Vec4; MAX_TERRAIN_SHADOW_QUADS],
    pub quad_c: [Vec4; MAX_TERRAIN_SHADOW_QUADS],
    pub quad_d: [Vec4; MAX_TERRAIN_SHADOW_QUADS],
}

impl Default for BodyTerrainShadow {
    fn default() -> Self {
        Self {
            params: Vec4::ZERO,
            quad_params: Vec4::ZERO,
            caster_a_radius: [Vec4::ZERO; MAX_TERRAIN_SHADOW_CASTERS],
            caster_b_radius: [Vec4::ZERO; MAX_TERRAIN_SHADOW_CASTERS],
            quad_a: [Vec4::ZERO; MAX_TERRAIN_SHADOW_QUADS],
            quad_b: [Vec4::ZERO; MAX_TERRAIN_SHADOW_QUADS],
            quad_c: [Vec4::ZERO; MAX_TERRAIN_SHADOW_QUADS],
            quad_d: [Vec4::ZERO; MAX_TERRAIN_SHADOW_QUADS],
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

/// Surface shading style for the ground LOD.
///
/// Selects which shading path `body_terrain.wgsl` takes per body. The ground
/// LOD historically hard-coded the wet, vegetated terrestrial path (Thalos);
/// airless regolith bodies (Mira) want their orbital impostor's gray Hapke
/// look instead, so the two render paths reconverge at the impostor↔ground LOD
/// swap.
///
/// Encoded into [`BodyTerrainExtras::inspection`]`.y` (no extra uniform
/// binding — see that field's slot-budget rationale).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerrainShadingStyle {
    /// Wet, vegetated terrestrial body: Oren–Nayar + GGX dielectric BRDF,
    /// ecological albedo bands (grass/soil/rock/snow), atmospheric sky fill.
    #[default]
    Vegetated,
    /// Airless particulate regolith: Hapke radiative-transfer BRDF over the
    /// baked gray albedo, no vegetation/snow bands, no atmospheric sky fill.
    Regolith,
}

impl TerrainShadingStyle {
    /// Shader flag value stored in `inspection.y`.
    pub fn shader_flag(self) -> f32 {
        match self {
            Self::Vegetated => 0.0,
            Self::Regolith => 1.0,
        }
    }
}

/// Number of sun-shadow cascades (near→far). Shared between the shadow camera
/// rig (`thalos_game::rendering::sun_shadow`), the terrain + tree materials, and
/// their shaders. Each cascade has its OWN `texture_depth_2d` (a known-good
/// binding) — **not** a depth array, which broke terrain rendering. Keep in sync
/// with the per-cascade texture bindings + unrolled sampling in the WGSL.
pub const CASCADE_COUNT: usize = 3;

/// Sun-shadow cascade transforms + per-cascade compare params (the UNIFORM half
/// of the cascade binding; the depth maps are separate `texture_depth_2d`
/// bindings per cascade, sampled unrolled in the shader). Mirrored in
/// `body_terrain.wgsl` / `tree.wgsl`. `config.x == 0` ⇒ shader skips sampling.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct ShadowCascadeBlock {
    /// Render-space → cascade clip (Bevy reverse-z orthographic), one per cascade.
    pub view_proj: [Mat4; CASCADE_COUNT],
    /// Per cascade: x = depth bias (clip units), yzw reserved.
    pub params: [Vec4; CASCADE_COUNT],
    /// x = strength (0 ⇒ skip), y = active cascade count, zw reserved. Named
    /// `gate` (not `config`) because `body_terrain.wgsl` `#import`s a udlod
    /// global called `config`, and a matching field name collides in naga_oil.
    pub gate: Vec4,
}

impl Default for ShadowCascadeBlock {
    fn default() -> Self {
        Self {
            view_proj: [Mat4::IDENTITY; CASCADE_COUNT],
            params: [Vec4::ZERO; CASCADE_COUNT],
            gate: Vec4::ZERO,
        }
    }
}

/// Packed bag of terrain-specific per-frame uniforms.
///
/// Exists so the material lands a single uniform binding instead of three.
/// Bevy 0.18's `AsBindGroup` derive hardcodes `VERTEX | FRAGMENT | COMPUTE`
/// visibility for `#[uniform(N)]` (the `visibility(...)` annotation is
/// silently ignored for that attribute), and the Metal backend caps a
/// pipeline's vertex stage at `MAX_VERTEX_BUFFERS = 16` buffer slots
/// (wgpu-hal). Each extra `#[uniform]` adds one buffer to *both* stages, so
/// the previous five-uniform layout pushed `terrain_pipeline` to 17 vertex
/// buffers and failed validation. Atmosphere and scene stay separate
/// because they are shared `planet_lighting` types reused elsewhere; the
/// three terrain-only knobs collapse here.
#[derive(Clone, Copy, ShaderType)]
pub struct BodyTerrainExtras {
    pub craft_shadow: BodyTerrainShadow,
    pub debug: BodyTerrainDebug,
    /// x = fullbright albedo output, y = surface shading style
    /// ([`TerrainShadingStyle::shader_flag`]: 0 = vegetated, 1 = regolith),
    /// zw reserved.
    pub inspection: Vec4,
    /// Cascaded sun-shadow transforms + per-cascade compare params (see
    /// [`ShadowCascadeBlock`]). Packed here rather than as its own `#[uniform]`
    /// to avoid a second vertex buffer (Metal 16-slot cap).
    pub shadow: ShadowCascadeBlock,
}

impl Default for BodyTerrainExtras {
    fn default() -> Self {
        Self {
            craft_shadow: BodyTerrainShadow::default(),
            debug: BodyTerrainDebug::default(),
            inspection: Vec4::ZERO,
            shadow: ShadowCascadeBlock::default(),
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
    /// Terrain-specific per-frame state (craft shadow, debug overlay,
    /// inspection flags, cascaded sun-shadow transforms). Packed into one uniform
    /// — see [`BodyTerrainExtras`] for the slot-budget rationale.
    #[uniform(2)]
    pub extras: BodyTerrainExtras,
    /// Per-cascade sun-shadow depth maps (near→far), rendered by the game's
    /// `rendering::sun_shadow` rig. Each is a plain `texture_depth_2d` (the
    /// known-good single-map binding, replicated per cascade — no depth array).
    /// Sampled via `textureLoad` and projected through `extras.shadow.view_proj[c]`;
    /// `extras.shadow.config.x == 0` skips them. Bound on every instance so the
    /// depth `sample_type` always has a valid texture.
    #[texture(3, sample_type = "depth")]
    pub sun_shadow_map_0: Handle<Image>,
    #[texture(4, sample_type = "depth")]
    pub sun_shadow_map_1: Handle<Image>,
    #[texture(5, sample_type = "depth")]
    pub sun_shadow_map_2: Handle<Image>,
}

impl Material for BodyTerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/body_terrain.wgsl".into()
    }
}

pub(crate) fn embed_body_terrain_shader(app: &mut App) {
    embedded_asset!(app, "body_terrain.wgsl");
}
