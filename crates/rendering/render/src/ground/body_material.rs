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
use thalos_terrain::TerrainFlatten;

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
    /// base altitude), y = top. `CloudCompositeMaterial` uses these to clip and
    /// blend its near/orbital projections. z = aerial-perspective airlight
    /// ratio shared with the legacy atmosphere A/B. w = cloud-composite-enable
    /// flag: 1.0 only for the active body with live cloud textures; 0.0 on
    /// inactive/non-cloud bodies and when clouds are disabled.
    pub cloud_band_radii: Vec4,
    /// Analytic ocean parameters for `body_sky.wgsl`. The sky pass ray-traces a
    /// math sphere at `ocean.x` and shades it as water wherever that hit sits in
    /// front of the opaque seabed/terrain in the scene-depth buffer — smooth at
    /// every scale, with no mesh. x = ocean sphere radius in render units
    /// (`planet_radius + sea_level_m`; render space is 1 unit = 1 m on
    /// SHIP_LAYER, so this is just metres). y = enable flag (1 on ocean bodies,
    /// 0 disables the whole branch). z = shore-breaker time reduced to its
    /// exact repeat period in f64; w = the camera's f64-computed height above
    /// sea level in metres.
    pub ocean: Vec4,
    /// Deep-water linear-RGB tint (xyz) + minimum optical-depth scale (w),
    /// matching the impostor water BRDF fallback so the ground↔impostor handoff
    /// stays consistent. Only read when `ocean.y >= 0.5`.
    pub ocean_color_depth: Vec4,
    /// xy = camera's body-fixed wind/crosswind coordinates modulo the periodic
    /// wave domain, computed in f64 before upload. The shader adds its small
    /// camera-relative hit offset. This keeps metre-scale waves stable at
    /// planet radius and across floating-origin rebases.
    pub ocean_camera_phase: Vec4,
    /// Low/high frequency packet phase in texture cycles for each physical
    /// cascade, reduced from canonical simulation time in f64 on the CPU.
    pub ocean_low_phase: Vec4,
    pub ocean_high_phase: Vec4,
    /// Resolved slope amplitude for the 8192/1024/128/16 m cascades.
    pub ocean_slope_amplitudes: Vec4,
    /// x = independent swell angle from wind in the local tangent plane,
    /// y = swell energy 0..1, z = open-water foam slope onset,
    /// w = slope-field diagnostic view enable.
    pub ocean_spectrum: Vec4,
    /// Body-local wind/crosswind basis used by the directional wave table.
    /// xyz are unit vectors; w is unused. Kept explicit so the shader's
    /// camera-relative evaluation uses the same directions as the CPU phase.
    pub ocean_wind_basis: Vec4,
    pub ocean_crosswind_basis: Vec4,
    /// Resident-height-tile lookup parameters
    /// (ADR-20260720T185958Z-water-projects-one-signed-sea-field). The ocean branch
    /// samples signed sea height straight from the udlod height atlas bound at
    /// bindings 7–10 — the exact texels the visible terrain mesh is displaced
    /// from — with the coast atlas as the coarse tail.
    /// x = enable (≥ 0.5 only when the terrain's tile tree + atlas are bound),
    /// y = `lod_count`, z = `tree_size`, w = attachment-0 center size (texels
    /// per tile edge, borders excluded).
    pub tile_lookup: Vec4,
    /// x = attachment-0 atlas-UV scale (center/texture size), y = atlas-UV
    /// offset (border/texture size), z = height-encoding `min_height` (m),
    /// w = `max_height` (m) — the UNORM16 decode range, mirroring
    /// `thalos_udlod::attachments::decode_height_m`.
    pub tile_atlas_uv: Vec4,
    /// Near-volume march-reach contract for the cloud composite's partition of
    /// unity: x = configured view raymarch step count (f32), so the composite
    /// can reproduce the marcher's per-ray reach analytically and hand every
    /// texel beyond it to the weather-column orbital estimator. y/z/w spare.
    pub cloud_march: Vec4,
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
            ocean_camera_phase: Vec4::ZERO,
            ocean_low_phase: Vec4::ZERO,
            ocean_high_phase: Vec4::ZERO,
            ocean_slope_amplitudes: Vec4::ZERO,
            ocean_spectrum: Vec4::ZERO,
            ocean_wind_basis: Vec4::ZERO,
            ocean_crosswind_basis: Vec4::ZERO,
            tile_lookup: Vec4::ZERO,
            tile_atlas_uv: Vec4::ZERO,
            cloud_march: Vec4::ZERO,
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
/// `params.y`: unused.
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
/// rig (`thalos_runtime::rendering::sun_shadow`), the terrain + tree materials, and
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
    /// Per cascade: x = clip units per metre of light-space depth
    /// (`1 / (far − near)`), y = shadow-map texel size in world metres. The
    /// sampler derives its capped texel-proportional bias + receiver
    /// normal-offset from these (stable-CSM W6 — see the bias model note in
    /// `shadow.wgsl`). zw reserved.
    pub params: [Vec4; CASCADE_COUNT],
    /// x = strength (0 ⇒ skip), y = active cascade count, zw reserved. Named
    /// `gate` (not `config`) because `body_terrain.wgsl` `#import`s a udlod
    /// global called `config`, and a matching field name collides in naga_oil.
    pub gate: Vec4,
    /// xyz = normalized render-space direction toward the sun (drives the
    /// slope-scaled bias in `sun_shadow_factor_nrm`), w reserved.
    pub sun_dir: Vec4,
}

impl Default for ShadowCascadeBlock {
    fn default() -> Self {
        Self {
            view_proj: [Mat4::IDENTITY; CASCADE_COUNT],
            params: [Vec4::ZERO; CASCADE_COUNT],
            gate: Vec4::ZERO,
            sun_dir: Vec4::ZERO,
        }
    }
}

/// Max flatten pads mirrored to the GPU per body terrain. Keep in sync with
/// the `array<TerrainFlattenRegion, 4>` in `body_terrain.wgsl`. When a body
/// carries more regions than this, the material driver uploads the pads
/// nearest the camera — only pads near the view can produce visible error.
pub const MAX_FLATTEN_REGIONS: usize = 4;

/// GPU mirror of one [`thalos_terrain::TerrainFlatten`] pad, consumed by the
/// analytic vertex-stage flatten in `body_terrain.wgsl` (`flattened_height`).
///
/// **Why this exists (the structural invariant):** structures are built
/// against the flatten plane, but the terrain the player sees is the tile
/// atlas — whose vertex-stage LOD blend/morph mixes in coarse ancestor tiles
/// with kilometre-scale texels that average natural terrain into the pad
/// (decimetres of error, more than the few-cm paving lift), and which can
/// hold tiles baked before a runtime flatten existed at all. The shader
/// therefore re-applies the flatten *analytically per vertex*: inside a pad
/// rectangle the rendered ground is pinned to the exact tangent-plane
/// elevation, at every LOD, every morph state, and every bake state. Rendered
/// ground under structures can no longer depend on tile streaming/bake
/// timing, so structures always draw above it.
#[derive(Clone, Copy, Default, ShaderType)]
pub struct FlattenRegionGpu {
    /// xyz = unit body-fixed direction to the pad centre, w = plane elevation
    /// at the centre (m above the reference radius).
    pub center_elev: Vec4,
    /// xyz = unit body-fixed tangent along the pad, w = rect half-length (m).
    pub along: Vec4,
    /// xyz = unit body-fixed tangent across the pad, w = rect half-width (m).
    pub across: Vec4,
    /// x = rect centre offset along (m), y = rect centre offset across (m),
    /// z = cos of the angular reject radius, w = interior feather width (m).
    pub rect: Vec4,
}

impl FlattenRegionGpu {
    /// Pack one CPU flatten pad for the shader.
    pub fn from_flatten(f: &TerrainFlatten) -> Self {
        let radius = f.radius_m.max(1.0);
        // Angular reject over the *rectangle* only (the ramp is not part of
        // the vertex override), plus generous slack: near cosθ = 1 an f32
        // dot-product compare is noisy at the equivalent of ~100–200 m of
        // lateral reach, so a tight bound could cull valid pad-corner
        // vertices. 512 m of slack still rejects virtually the whole planet.
        let reach_along = f.offset_along_m.abs() + f.half_along_m;
        let reach_across = f.offset_across_m.abs() + f.half_across_m;
        let reach = (reach_along * reach_along + reach_across * reach_across).sqrt() + 512.0;
        let cos_max = (reach / radius).atan().cos();
        // Feather the override in across a band just inside the rectangle
        // edge so the (analytically exact) interior meets the (baked, ramped)
        // exterior without a crease. Structures sit well inside the pad rect,
        // so the feather band never runs under them; scale it down for small
        // pads so it can't eat a pad whole.
        let feather_m = (0.25 * f.half_along_m.min(f.half_across_m)).clamp(1.0, 30.0);
        Self {
            center_elev: Vec4::new(
                f.center_dir.x as f32,
                f.center_dir.y as f32,
                f.center_dir.z as f32,
                f.elevation_m as f32,
            ),
            along: Vec4::new(
                f.tangent_along.x as f32,
                f.tangent_along.y as f32,
                f.tangent_along.z as f32,
                f.half_along_m as f32,
            ),
            across: Vec4::new(
                f.tangent_across.x as f32,
                f.tangent_across.y as f32,
                f.tangent_across.z as f32,
                f.half_across_m as f32,
            ),
            rect: Vec4::new(
                f.offset_along_m as f32,
                f.offset_across_m as f32,
                cos_max as f32,
                feather_m as f32,
            ),
        }
    }
}

/// The per-body flatten-pad set bound to the terrain vertex stage. Zero
/// `meta.x` (the [`Default`]) disables the override entirely — the map
/// terrain and airless bodies never pay for it.
#[derive(Clone, Copy, Default, ShaderType)]
pub struct FlattenBlock {
    /// x = active region count, y = body reference radius (m), zw reserved.
    pub meta: Vec4,
    pub regions: [FlattenRegionGpu; MAX_FLATTEN_REGIONS],
}

impl FlattenBlock {
    /// Pack up to [`MAX_FLATTEN_REGIONS`] pads. All pads must belong to the
    /// same body (they share `meta.y`); extras beyond the cap are dropped, so
    /// callers with more should pre-sort by relevance (nearest the camera).
    pub fn pack<'a>(flattens: impl IntoIterator<Item = &'a TerrainFlatten>) -> Self {
        let mut block = Self::default();
        let mut count = 0usize;
        for f in flattens {
            if count >= MAX_FLATTEN_REGIONS {
                break;
            }
            block.meta.y = f.radius_m as f32;
            block.regions[count] = FlattenRegionGpu::from_flatten(f);
            count += 1;
        }
        block.meta.x = count as f32;
        block
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
    pub debug: BodyTerrainDebug,
    /// x = fullbright albedo output, y = surface shading style
    /// ([`TerrainShadingStyle::shader_flag`]: 0 = vegetated, 1 = regolith),
    /// z = distant-schematic flag (1 = orbital map terrain: matte specular, no
    /// valley AO), w = SSAO enable (1 = sample+apply the `ao` map — graphics F5).
    pub inspection: Vec4,
    /// Cascaded sun-shadow transforms + per-cascade compare params (see
    /// [`ShadowCascadeBlock`]). Packed here rather than as its own `#[uniform]`
    /// to avoid a second vertex buffer (Metal 16-slot cap).
    pub shadow: ShadowCascadeBlock,
    /// Analytic vertex-stage flatten pads (see [`FlattenBlock`] /
    /// [`FlattenRegionGpu`]). Mirrored per frame from the body's
    /// `TerrainFlattenRegistry` handle by the game's terrain material driver;
    /// the WGSL field is named `pads`. Packed here for the same slot-budget
    /// reason as `shadow`.
    pub flatten: FlattenBlock,
}

impl Default for BodyTerrainExtras {
    fn default() -> Self {
        Self {
            debug: BodyTerrainDebug::default(),
            inspection: Vec4::ZERO,
            shadow: ShadowCascadeBlock::default(),
            flatten: FlattenBlock::default(),
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
    /// Half-res screen-space AO (`rendering::ssao`'s `AoImage`, R8Unorm; 1 =
    /// unoccluded), multiplied into the ambient occlusion term only — graphics F5.
    /// Default handle binds the white fallback (no AO); the game patches the live
    /// AO image on the surface terrain and gates sampling via
    /// `extras.inspection.w` (0 skips it — orbital map terrain, or before the pass
    /// is valid). Sampled bilinear at the fragment's screen UV (1-frame latency).
    #[texture(6)]
    #[sampler(7)]
    pub ao: Handle<Image>,
}

impl Material for BodyTerrainMaterial {
    /// Custom vertex stage: the udlod default plus the analytic pad flatten
    /// (see [`FlattenRegionGpu`]). Both terrain paths (surface + orbital map)
    /// render through thalos_udlod's `TerrainRenderPipeline`, which reads this
    /// — there is no Bevy mesh-pipeline consumer of this material, so the
    /// override cannot leak into a standard prepass.
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/body_terrain.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/body_terrain.wgsl".into()
    }
}

pub(crate) fn embed_body_terrain_shader(app: &mut App) {
    embedded_asset!(app, "body_terrain.wgsl");
}
