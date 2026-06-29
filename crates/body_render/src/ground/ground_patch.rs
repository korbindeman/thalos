//! Flat, sky-model-lit ground patch material — a non-UDLOD ground for previews
//! and dioramas (the object preview today; composed scenes later).
//!
//! Lights a flat plane through the SAME shared `thalos::lighting` surface model
//! the in-game ground LOD uses (rough-dielectric BRDF + analytic hemisphere sky
//! fill) and receives the SAME cascaded sun-shadows scattered trees cast, so a
//! previewed plant sits on ground that reads like the terrain it grows from,
//! with its own shadow falling across it. It is intentionally NOT part of the
//! UDLOD terrain stack ([`crate::ground::BodyTerrainMaterial`]); it's a
//! deliberately simple flat-ground analogue for tooling, sharing the cascade
//! binding layout with [`TreeMaterial`](crate::ground::TreeMaterial) so one
//! shadow rig feeds both.

use bevy::asset::embedded_asset;
use bevy::pbr::Material;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

use crate::ground::body_material::ShadowCascadeBlock;
use crate::ground::vegetation::GrassParams;

/// A flat ground patch that shades through `thalos::lighting` and receives
/// cascaded sun-shadows. Binds the same shadow cascade block + per-cascade depth
/// maps as [`TreeMaterial`](crate::ground::TreeMaterial), so a single sun-shadow
/// pass shadows both the trees and the ground beneath them.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct GroundPatchMaterial {
    /// Sky/sun lighting parameters — only the lighting fields of [`GrassParams`]
    /// (`sun_dir` w=flux, `sky_up`, `sky_tau`) are read; the rest are ignored,
    /// reused so the driver writes one shared params type for ground + grass.
    #[uniform(0)]
    pub params: GrassParams,
    /// Cascaded sun-shadow transforms + strength (see [`ShadowCascadeBlock`]).
    #[uniform(1)]
    pub shadow: ShadowCascadeBlock,
    /// Per-cascade sun-shadow depth maps (near→far) — the same handles the tree
    /// material binds. Each a plain `texture_depth_2d`; always valid (see
    /// [`fallback_shadow_map`](crate::ground::fallback_shadow_map)).
    /// `shadow.gate.x` gates whether they're sampled.
    #[texture(2, sample_type = "depth")]
    pub sun_shadow_map_0: Handle<Image>,
    #[texture(3, sample_type = "depth")]
    pub sun_shadow_map_1: Handle<Image>,
    #[texture(4, sample_type = "depth")]
    pub sun_shadow_map_2: Handle<Image>,
}

impl Material for GroundPatchMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/ground_patch.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/ground_patch.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}

pub(crate) fn embed_ground_patch_shader(app: &mut App) {
    embedded_asset!(app, "ground_patch.wgsl");
}

/// Lean plugin to render [`GroundPatchMaterial`] in a standalone / headless app
/// (the object-preview tool, future diorama scenes) without the full UDLOD
/// terrain stack: registers the material, embeds its shader, and ensures the
/// `thalos::lighting` library is present.
pub struct GroundPatchMaterialPlugin;

impl Plugin for GroundPatchMaterialPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
        }
        app.add_plugins(bevy::pbr::MaterialPlugin::<GroundPatchMaterial>::default());
        embed_ground_patch_shader(app);
    }
}
