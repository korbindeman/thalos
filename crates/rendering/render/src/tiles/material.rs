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

pub type TileTerrainMaterial = ExtendedMaterial<StandardMaterial, TileShadingExtension>;

/// Mirror of the WGSL `TileShadingParams` (declaration order is the contract).
#[derive(Clone, Copy, ShaderType, Debug)]
pub struct TileShadingParams {
    /// 1 = Hapke regolith (airless bodies), 0 = stock PBR.
    pub style: u32,
    pub _pad0: u32,
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
}

impl TileShadingParams {
    pub fn pbr() -> Self {
        Self {
            style: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            orient: Vec4::new(0.0, 0.0, 0.0, 1.0),
            up_body: Vec4::new(0.0, 1.0, 0.0, 0.0),
        }
    }

    pub fn hapke() -> Self {
        Self { style: 1, ..Self::pbr() }
    }
}

impl Default for TileShadingParams {
    fn default() -> Self {
        Self::pbr()
    }
}

/// Bindings mirror `ShadowReceiveExtension` (uniform 100, depth maps 101–103)
/// plus the tile params at 104.
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
        extension: TileShadingExtension { params, ..Default::default() },
    }
}
