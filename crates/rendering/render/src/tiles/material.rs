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
//! The shadow maps are fanned in per-frame by `craft::apply_craft_shadow`,
//! exactly like the hull and `ShadowedStandardMaterial`.

use bevy::math::Vec4;
use bevy::pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

use crate::ShadowCascadeBlock;
use crate::clouds::CloudShadowBlock;

pub type TileTerrainMaterial = ExtendedMaterial<StandardMaterial, TileShadingExtension>;

/// Mirror of the WGSL `TileShadingParams` (declaration order is the contract).
#[derive(Clone, Copy, ShaderType, Debug)]
pub struct TileShadingParams {
    /// 1 = Hapke regolith (airless bodies), 0 = stock PBR.
    pub style: u32,
    /// Capture-only inspection mode, mirroring udlod's
    /// `THALOS_TERRAIN_INSPECTION` so one compare axis reads both renderers:
    /// 0 = lit, 1 = fullbright (emit the layer stack's albedo, no lighting),
    /// 2 = geometric normal (drop the detail-normal offsets).
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
/// plus the tile params at 104 and the cloud sun-transmittance cascade at
/// 105–107.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct TileShadingExtension {
    #[uniform(100)]
    pub shadow: ShadowCascadeBlock,
    #[texture(101, sample_type = "depth")]
    pub sun_shadow_map_0: Handle<Image>,
    #[texture(102, sample_type = "depth")]
    pub sun_shadow_map_1: Handle<Image>,
    #[texture(103, sample_type = "depth")]
    pub sun_shadow_map_2: Handle<Image>,
    #[uniform(104)]
    pub params: TileShadingParams,
    /// Cloud sun-transmittance cascade (CLOUD-5 / W2): `r` = the fraction of
    /// the sun beam that survives the deck at this point. Fanned in per frame
    /// by [`apply_cloud_shadow`](crate::tiles::apply_cloud_shadow) alongside
    /// the block below, exactly as `apply_craft_shadow` fans the cascade maps.
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
}

impl MaterialExtension for TileShadingExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/tile_terrain.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        "shaders/tile_terrain.wgsl".into()
    }
}

/// Wrap a base `StandardMaterial` into the tile material with the given
/// shading style and default (fallback) shadow state — `apply_craft_shadow`
/// patches the live cascade in each frame.
pub fn tile_material(base: StandardMaterial, params: TileShadingParams) -> TileTerrainMaterial {
    TileTerrainMaterial {
        base,
        extension: TileShadingExtension {
            params,
            ..Default::default()
        },
    }
}
