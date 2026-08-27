//! NTR-X1 tile-terrain surface material.
//!
//! `ExtendedMaterial<StandardMaterial, _>` — the keystone shape: Bevy's PBR
//! pipeline keeps driving lighting units, exposure, tonemapping, and the
//! prepass; the extension adds only what the standard path can't express:
//!
//! - the shared `thalos::shadow` cascade *receive* (one shadow world — same
//!   bindings/discipline as `ShadowReceiveExtension` / the hull), and
//! - an optional Hapke regolith BRDF branch for airless bodies, driven by
//!   Bevy's own directional light + ambient (no `SceneLighting` coupling), so
//!   tile ground reconverges with the impostor's Hapke look across the swap.
//!
//! The shadow maps are fanned in per-frame by the game's
//! `sun_shadow::sync_shadow_receivers` (the SOLE tile shadow writer),
//! exactly like the hull and `ShadowedStandardMaterial`.

use bevy::math::Vec4;
use bevy::pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

use super::gpu::TileGpuImages;
use crate::ShadowCascadeBlock;
use crate::clouds::CloudShadowBlock;

pub type TileTerrainMaterial = ExtendedMaterial<StandardMaterial, TileShadingExtension>;
pub type TileCasterMaterial = ExtendedMaterial<StandardMaterial, TileCasterExtension>;

const TILE_DISPLACEMENT_SHADER: &str = "shaders/tile_displacement.wgsl";

/// **Load-bearing, and not about transparency at all** — this is what keeps the
/// tile atlases bound while the displacement vertex shader runs in the depth
/// prepass and in Bevy's shadow pass. Leave it on both extensions.
///
/// Bevy 0.19 skips the material bind group for a *depth-only opaque* pass: the
/// prepass pipeline gets `empty_layout` at group 3 and the phase is drawn with
/// `PrepassOpaqueDepthOnlyDrawFunction` / `ShadowsDepthOnlyDrawFunction`, which
/// never bind it (`bevy_pbr::prepass::is_depth_only_opaque_prepass`,
/// `render/light.rs`). Our prepass/shadow vertex stage is
/// [`TILE_DISPLACEMENT_SHADER`], which *reads* group 3 (bindings 111/112) to
/// find the vertex it is meant to place — so that pass builds a pipeline whose
/// vertex shader references a binding the layout doesn't have, and wgpu kills
/// the process at pipeline creation:
///
/// ```text
/// In Device::create_render_pipeline, label = 'pbr_prepass_pipeline'
///   Shader global ResourceBinding { group: 3, binding: 111 } is not available
///   in the pipeline layout
/// ```
///
/// The escape hatch Bevy checks for is `MeshPipelineKey::MAY_DISCARD` (and
/// `PREPASS_READS_MATERIAL`, which 0.19.0 defines and reads but never sets, so
/// it is unreachable from a `Material` impl). `MAY_DISCARD` is reachable, and
/// only through the alpha mode: `AlphaMode::Mask` puts the material in the
/// alpha-mask phase, whose draw function binds group 3 and whose pipeline gets
/// the real material layout in both the prepass and the shadow pass.
///
/// Nothing is actually masked. This overrides only the *pipeline-level* alpha
/// mode; the base `StandardMaterial` stays `AlphaMode::Opaque`, so the GPU-side
/// material flags still say opaque and `alpha_discard` forces alpha to 1.0 in
/// both `tile_terrain.wgsl` and Bevy's prepass fragment. The main-pass pipeline
/// is unchanged apart from a `MAY_DISCARD` shader def (blend, depth write and
/// depth compare are all picked from the blend bits, which stay opaque).
///
/// Cost: terrain moves from `Opaque3d` to `AlphaMask3d` (same node, drawn
/// straight after opaque), and the depth prepass now compiles a fragment stage
/// for it — Bevy's void `prepass_alpha_discard` entry — which forfeits early-Z
/// depth writes on the prepass draw. If that ever shows up in the perf lane,
/// the fix is a no-op prepass fragment shader of our own, not removing this.
///
/// INC-20260826T124420Z-tile-prepass-material-bind-group.
const DISPLACED_PREPASS_ALPHA_MODE: AlphaMode = AlphaMode::Mask(0.5);

/// Mirror of the WGSL `TileShadingParams` (declaration order is the contract).
#[derive(Clone, Copy, ShaderType, Debug)]
pub struct TileShadingParams {
    /// 1 = Hapke regolith (airless bodies), 0 = stock PBR.
    pub style: u32,
    /// Capture-only inspection mode, mirroring udlod's
    /// `THALOS_TERRAIN_INSPECTION` so one compare axis reads both renderers:
    /// 0 = lit, 1 = fullbright (emit the layer stack's albedo, no lighting),
    /// 2 = geometric normal (drop the detail-normal offsets), 4 = baked base
    /// colour before the procedural layer stack (performance attribution).
    ///
    /// Fullbright is the one that separates "the paint is wrong" from "the
    /// light is wrong" — the question a rendered frame alone cannot answer,
    /// and the one the tile path had no way to ask.
    pub inspect: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    /// Body→world rotation as a unit quaternion (xyzw). Written per frame by
    /// the game's tile driver so the NTR-X4 material layers can classify
    /// slope / build detail normals in the body-fixed frame (stable under
    /// planet spin and floating-origin moves), then rotate back to world.
    pub orient: Vec4,
    /// Radial "up" at the view anchor, body-fixed (xyz; w unused). Up varies
    /// ~1° per 50 km, so one uniform serves the whole frame.
    pub up_body: Vec4,
    /// xyz = body centre in **world render space**, written per frame by the
    /// game's tile driver. Gives every fragment its own radial up
    /// (`normalize(world_pos - centre)`) — which [`up_body`] cannot, since one
    /// anchor up only holds for the near field, not for a whole globe seen
    /// from orbit.
    ///
    /// w = the day/night gate already folded into `GlobalAmbientLight` at the
    /// craft (`lighting::SunDaylight`, clamped away from zero). The shader
    /// divides it out so the near field — where the fragment's gate equals the
    /// craft's — comes out exactly unchanged instead of ramping twice.
    ///
    /// [`up_body`]: Self::up_body
    pub center_ws: Vec4,
    /// xyz = unit direction toward the star in world render space.
    ///
    /// w = the fraction of `GlobalAmbientLight` that is the **night floor**
    /// (starlight / planetshine), i.e. `AMBIENT_NIGHT_BRIGHTNESS /
    /// ambient.brightness`. The shader gates the ambient fill from `1` in
    /// daylight down to this floor across the terminator, so the sky/space
    /// fill stops lighting the night hemisphere.
    ///
    /// Why a fraction and not a second colour: `GlobalAmbientLight` is one
    /// per-camera value derived at the *craft*, so from orbit its daylight
    /// gate smears the day-side fill over the whole visible globe. The CPU
    /// keeps sole authority over the ambient's magnitude; this only tells the
    /// shader how to redistribute it spatially. When the craft is already at
    /// night the ambient IS the floor, the fraction is 1, and the gate is
    /// identity — which is why the surface case is untouched.
    ///
    /// Default `1.0` (identity gate) so an unwritten uniform can't darken
    /// anything.
    pub sun_night: Vec4,
}

impl TileShadingParams {
    pub fn pbr() -> Self {
        Self {
            style: 0,
            inspect: 0,
            _pad1: 0,
            _pad2: 0,
            orient: Vec4::new(0.0, 0.0, 0.0, 1.0),
            up_body: Vec4::new(0.0, 1.0, 0.0, 0.0),
            center_ws: Vec4::new(0.0, 0.0, 0.0, 1.0),
            sun_night: Vec4::new(0.0, 1.0, 0.0, 1.0),
        }
    }

    pub fn hapke() -> Self {
        Self {
            style: 1,
            ..Self::pbr()
        }
    }
}

impl Default for TileShadingParams {
    fn default() -> Self {
        Self::pbr()
    }
}

/// Bindings mirror `ShadowReceiveExtension` (uniform 100, depth maps 101–103)
/// plus the tile params at 104, the cloud sun-transmittance cascade at
/// 105–107, and the same-frame contact-shadow term at 108–109.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct TileShadingExtension {
    #[uniform(100)]
    pub shadow: ShadowCascadeBlock,
    /// Cascade 0 — the ±64 m near box added 2026-07-31. Bound at the END of
    /// this material's range rather than renumbered into the near→far slot:
    /// only the ARGUMENT order at the `thalos::shadow` call site is
    /// ordering-significant, so shifting live binding indices would be risk
    /// with no payoff. Field name still equals the cascade index.
    #[texture(110, sample_type = "depth")]
    pub sun_shadow_map_0: Handle<Image>,
    #[texture(101, sample_type = "depth")]
    pub sun_shadow_map_1: Handle<Image>,
    #[texture(102, sample_type = "depth")]
    pub sun_shadow_map_2: Handle<Image>,
    #[texture(103, sample_type = "depth")]
    pub sun_shadow_map_3: Handle<Image>,
    #[uniform(104)]
    pub params: TileShadingParams,
    /// Cloud sun-transmittance cascade (CLOUD-5 / W2): `r` = the fraction of
    /// the sun beam that survives the deck at this point. Fanned in per frame
    /// by [`apply_cloud_shadow`](crate::tiles::apply_cloud_shadow) alongside
    /// the block below, exactly as `sync_shadow_receivers` fans the cascade
    /// maps.
    #[texture(105)]
    #[sampler(106)]
    pub cloud_shadow_map: Handle<Image>,
    /// Placement of that cascade — the frame this material projects into. Its
    /// own uniform rather than a field of [`TileShadingParams`] because the two
    /// have different writers (the game's tile driver owns the params; the
    /// cloud driver owns this), and a shared struct would let one clobber the
    /// other depending on system order.
    #[uniform(107)]
    pub cloud_shadow: CloudShadowBlock,
    /// Full-resolution contact shadow produced after the depth prepass and
    /// before opaque shading. One field supplies both texture and sampler,
    /// matching the cloud map convention above.
    #[texture(108)]
    #[sampler(109)]
    pub contact_shadow_map: Handle<Image>,
    /// Exact body-local positions + ecological altitude, one array layer per
    /// resident tile. Rgba32Float is intentionally unfilterable: every patch
    /// resolution visits an exact subset of the authoritative 129² samples.
    #[texture(111, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub tile_position_atlas: Handle<Image>,
    /// Linear macro albedo + canopy coverage, quantized to Rgba8Unorm.
    #[texture(112, dimension = "2d_array")]
    pub tile_surface_atlas: Handle<Image>,
}

impl MaterialExtension for TileShadingExtension {
    fn alpha_mode() -> Option<AlphaMode> {
        Some(DISPLACED_PREPASS_ALPHA_MODE)
    }

    fn vertex_shader() -> ShaderRef {
        TILE_DISPLACEMENT_SHADER.into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        TILE_DISPLACEMENT_SHADER.into()
    }

    fn deferred_vertex_shader() -> ShaderRef {
        TILE_DISPLACEMENT_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/tile_terrain.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        "shaders/tile_terrain.wgsl".into()
    }
}

/// Bare shadow material extension. It binds the same atlases and uses the
/// exact same vertex entry point as the visible material, while retaining
/// `StandardMaterial`'s cheap depth-only/ordinary fragment behavior.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct TileCasterExtension {
    #[texture(111, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub tile_position_atlas: Handle<Image>,
    #[texture(112, dimension = "2d_array")]
    pub tile_surface_atlas: Handle<Image>,
}

impl MaterialExtension for TileCasterExtension {
    fn alpha_mode() -> Option<AlphaMode> {
        Some(DISPLACED_PREPASS_ALPHA_MODE)
    }

    fn vertex_shader() -> ShaderRef {
        TILE_DISPLACEMENT_SHADER.into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        TILE_DISPLACEMENT_SHADER.into()
    }

    fn deferred_vertex_shader() -> ShaderRef {
        TILE_DISPLACEMENT_SHADER.into()
    }
}

/// Wrap a base `StandardMaterial` into the tile material with the given
/// shading style and default (fallback) shadow state — the game's
/// `sun_shadow::sync_shadow_receivers` patches the live cascade in each frame.
pub fn tile_material(
    base: StandardMaterial,
    params: TileShadingParams,
    atlases: &TileGpuImages,
) -> TileTerrainMaterial {
    TileTerrainMaterial {
        base,
        extension: TileShadingExtension {
            params,
            tile_position_atlas: atlases.position.clone(),
            tile_surface_atlas: atlases.surface.clone(),
            ..Default::default()
        },
    }
}

pub fn tile_caster_material(base: StandardMaterial, atlases: &TileGpuImages) -> TileCasterMaterial {
    TileCasterMaterial {
        base,
        extension: TileCasterExtension {
            tile_position_atlas: atlases.position.clone(),
            tile_surface_atlas: atlases.surface.clone(),
        },
    }
}
