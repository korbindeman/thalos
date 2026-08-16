//! Octahedral tree impostors — the far band of the tree LOD cascade.
//!
//! A far tree is drawn as a single camera-facing quad that samples a pre-baked
//! **hemisphere octahedral atlas** of the species: the tree captured from an
//! `N×N` grid of view directions (equator = ground-level views, pole = top-down
//! aerial views, since the player flies). The fragment maps the camera→tree view
//! direction to atlas coords, bilinearly blends the surrounding captured views,
//! alpha-tests, and lights the blended object-frame normal through the *same*
//! `thalos::lighting` sky model the mesh trees and ground use — so the
//! mesh→impostor handoff reads as one continuous forest.
//!
//! This module is the planetary material adapter: it adds cloud and cascaded
//! sun-shadow bindings to the shared `thalos_vegetation` atlas and geometry
//! mechanism. Atlas capture, layout, and four-vertex batches belong to that
//! shared crate so Kòrsou and the spherical game cannot drift.

use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

use crate::ground::vegetation::GrassParams;
pub use thalos_vegetation::{
    BakeParams, FoliageImpostorBakePlugin, IMPOSTOR_MAX_SPECIES, ImpostorAtlasLayout,
    ImpostorParams, TreeBakeMaterial, hemioct_decode, impostor_bake_rotation, make_impostor_atlas,
    recenter_tree_mesh, tree_bounding_sphere,
};

/// Extension for far tree impostors: billboards one quad per tree (baked by
/// [`combine_impostor_tile_mesh`](crate::ground::scatter::combine_impostor_tile_mesh))
/// and octahedral-samples the atlas. Reuses [`GrassParams`] for the
/// craft-anchor scale-fade, so the mesh→impostor handoff and the far cull are
/// seamless and zoom-independent, exactly like the mesh `TreeMaterial`.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct TreeImpostorExtension {
    #[uniform(100)]
    pub params: GrassParams,
    #[uniform(101)]
    pub impostor: ImpostorParams,
    #[texture(102)]
    #[sampler(103)]
    pub albedo: Handle<Image>,
    #[texture(104)]
    #[sampler(105)]
    pub normal: Handle<Image>,
    /// Cloud sun-transmittance cascade — the same map/block the tile ground and
    /// the mesh trees sample, fanned in per frame by the game's tree driver.
    /// NEW versus the spine impostor, which never dimmed under the deck.
    #[texture(106)]
    #[sampler(107)]
    pub cloud_shadow_map: Handle<Image>,
    #[uniform(108)]
    pub cloud_shadow: crate::clouds::CloudShadowBlock,
    /// Cascaded sun-shadow receive — impostor rings 0–1 CAST into cascades 1–2
    /// (they sit on `SHADOW_CASTER_LAYER` from 1.2 km out, and the cascades are
    /// pinned to cover exactly that band), so they must SAMPLE them too or a
    /// tree casts a shadow it cannot itself receive: bright trees on dark
    /// ground across a shadowed valley, and a shadowed→lit pop at the 1.2 km
    /// mesh↔impostor swap (reviews/20260730T011353Z §8). Past cascade 2's
    /// reach the sampler fades out and the W12 horizon term owns the far
    /// field, as before.
    #[uniform(109)]
    pub shadow: crate::ShadowCascadeBlock,
    /// Cascade 0 — the ±64 m near box added 2026-07-31. Bound at the END of
    /// this material's range rather than renumbered into the near→far slot:
    /// only the ARGUMENT order at the `thalos::shadow` call site is
    /// ordering-significant, so shifting live binding indices would be risk
    /// with no payoff. Field name still equals the cascade index.
    #[texture(113, sample_type = "depth")]
    pub sun_shadow_map_0: Handle<Image>,
    #[texture(110, sample_type = "depth")]
    pub sun_shadow_map_1: Handle<Image>,
    #[texture(111, sample_type = "depth")]
    pub sun_shadow_map_2: Handle<Image>,
    #[texture(112, sample_type = "depth")]
    pub sun_shadow_map_3: Handle<Image>,
}

/// Far-band impostor card on the standard path — `ExtendedMaterial` like the
/// mesh trees ([`tree_base_material`](crate::ground::tree_material::tree_base_material)
/// supplies the shared foliage base: double-sided, diffuse-transmitting), so
/// the mesh→impostor handoff stays inside the one lighting universe.
pub type TreeImpostorMaterial =
    bevy::pbr::ExtendedMaterial<bevy::pbr::StandardMaterial, TreeImpostorExtension>;

/// Wrap a [`TreeImpostorExtension`] into the full impostor material.
pub fn tree_impostor_material(extension: TreeImpostorExtension) -> TreeImpostorMaterial {
    TreeImpostorMaterial {
        base: crate::ground::tree_material::tree_base_material(),
        extension,
    }
}

impl bevy::pbr::MaterialExtension for TreeImpostorExtension {
    fn vertex_shader() -> ShaderRef {
        "shaders/tree_impostor_standard.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/tree_impostor_standard.wgsl".into()
    }

    // Opaque (depth-written) with a manual coverage `discard` in the fragment.
    // Opaque keeps the prepass binding-agnostic: the standard prepass uses
    // POSITION (all four quad corners share the tree base → degenerate, draws
    // nothing), so impostors never touch the prepass bind group.
    fn alpha_mode() -> Option<AlphaMode> {
        Some(AlphaMode::Opaque)
    }
}

/// Lean plugin to render [`TreeImpostorMaterial`] (and run the [`TreeBakeMaterial`]
/// off-screen bake) in a standalone / headless app — the object preview, examples
/// — **without** the full UDLOD terrain stack. Mirrors
/// [`TreeMaterialPlugin`](crate::ground::TreeMaterialPlugin): registers both
/// materials, embeds their shaders, and ensures the shared shader libraries
/// (`thalos::lighting` / `thalos::foliage`) are present. The game gets all this
/// from `GroundAppearancePlugin`; this is the minimal entry point for tools that want
/// to verify the mesh↔impostor handoff in isolation.
pub struct TreeImpostorMaterialPlugin;

impl Plugin for TreeImpostorMaterialPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
        }
        app.add_plugins(bevy::pbr::MaterialPlugin::<TreeImpostorMaterial>::default());
        if !app.is_plugin_added::<FoliageImpostorBakePlugin>() {
            app.add_plugins(FoliageImpostorBakePlugin);
        }
    }
}
