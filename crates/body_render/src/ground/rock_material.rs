//! Scattered pebble / rock instanced material.
//!
//! Lights a vertex-coloured stone through the SAME shared `thalos::lighting`
//! rough-dielectric surface model the in-game ground LOD and the diorama ground
//! patch use, and receives the SAME cascaded sun-shadows the trees cast — so a
//! pebble in the meadow lights exactly like the ground it rests on. One instance
//! is shared by every rock on a body; per-rock variation (scale, rotation) is
//! baked into the per-tile combined mesh, and the per-rock base in `UV_0`/`UV_1`
//! drives the clipmap scale-grow fade.
//!
//! Reuses [`GrassParams`] as its uniform (same field layout), and the
//! [`ShadowCascadeBlock`] + per-cascade depth-map binding layout of
//! [`GroundPatchMaterial`](crate::ground::GroundPatchMaterial), so one sun-shadow
//! pass feeds the trees, the ground, and the rocks alike.

use bevy::asset::embedded_asset;
use bevy::pbr::Material;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

use crate::ground::body_material::ShadowCascadeBlock;
use crate::ground::vegetation::GrassParams;

/// Material for scattered pebbles and rocks. Vertex colours carry the stone
/// albedo × baked cavity-AO / top-bleach; `UV_0`/`UV_1` carry the per-rock base
/// for the scale-grow fade.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct RockMaterial {
    /// Sky/sun lighting + the clipmap fade band (see [`GrassParams`]).
    #[uniform(0)]
    pub params: GrassParams,
    /// Cascaded sun-shadow transforms + strength (see [`ShadowCascadeBlock`]).
    #[uniform(1)]
    pub shadow: ShadowCascadeBlock,
    /// Per-cascade sun-shadow depth maps (near→far) — the same handles the tree
    /// and ground materials bind. Each a plain `texture_depth_2d`; always valid
    /// (see [`fallback_shadow_map`](crate::ground::fallback_shadow_map)).
    /// `shadow.gate.x` gates whether they're sampled.
    #[texture(2, sample_type = "depth")]
    pub sun_shadow_map_0: Handle<Image>,
    #[texture(3, sample_type = "depth")]
    pub sun_shadow_map_1: Handle<Image>,
    #[texture(4, sample_type = "depth")]
    pub sun_shadow_map_2: Handle<Image>,
}

impl Material for RockMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/rock.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/rock.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }

    /// Opt out of the depth prepass: the vertex shader scale-grows each stone
    /// across the near-band fade, and there's no matching prepass shader, so a
    /// standard rest-pose prepass would mismatch the main color pass's depth and
    /// flicker. Same rationale as `TreeMaterial` (see `ground/mod.rs`).
    fn enable_prepass() -> bool {
        false
    }
}

pub(crate) fn embed_rock_shader(app: &mut App) {
    embedded_asset!(app, "rock.wgsl");
}

/// Lean plugin to render [`RockMaterial`] in a standalone / headless app (the
/// object-preview tool) without the full UDLOD terrain stack: registers the
/// material, embeds its shader, and ensures the `thalos::lighting` library is
/// present. The game gets all this from `ThalosTerrainPlugin`.
pub struct RockMaterialPlugin;

impl Plugin for RockMaterialPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
        }
        app.add_plugins(bevy::pbr::MaterialPlugin::<RockMaterial>::default());
        embed_rock_shader(app);
    }
}
