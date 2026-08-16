//! Tree / shrub instanced material — `ExtendedMaterial<StandardMaterial, _>`,
//! the keystone shape (one lighting universe = Bevy's): Bevy's PBR pipeline
//! owns lighting units, the airmass-reddened sun, the ambient fill, exposure,
//! and tonemapping, exactly as it does for the tile ground and the hull. The
//! extension carries only what the standard path can't express — the batched
//! tile mesh's wind/grow vertex stage, the foliage atlas + shared
//! `thalos::foliage` albedo model, the shared `thalos::shadow` cascade
//! *receive*, and the cloud sun-transmittance gate (which the spine tree never
//! sampled: forests under a cloud deck stayed lit while the ground dimmed).
//!
//! The spine predecessor lit these through `thalos::lighting`
//! (`compute_surface_sky` + `shade_foliage`), which made vegetation the last
//! large surface class in the other lighting universe — the root of the
//! "trees look pasted on the terrain" reports. See `tree_standard.wgsl` for
//! what was ported vs deliberately dropped.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
    StandardMaterial,
};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, RenderPipelineDescriptor, SpecializedMeshPipelineError,
    TextureDimension, TextureFormat, TextureUsages,
};
use bevy::shader::ShaderRef;

use crate::clouds::CloudShadowBlock;
use crate::ground::body_material::ShadowCascadeBlock;
use crate::ground::vegetation::GrassParams;

/// A 1×1 depth texture to bind as a tree material's per-cascade `sun_shadow_map`
/// when there is no real sun-shadow pass yet (the standalone preview, or before
/// the game's shadow rig publishes one). The `texture_depth_2d` binding has no
/// usable fallback image (Bevy's `FallbackImage` is colour-only), so the sample
/// type needs *some* real depth texture even when `shadow.config.x == 0` keeps
/// the shader from ever reading it.
pub fn fallback_shadow_map() -> Image {
    let mut img = Image::new_uninit(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Depth32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    img
}

/// Material for scattered trees and shrubs on the standard path. One instance is
/// shared by every plant on a body; the mesh's vertex colours carry the
/// landcover tint × AO and the per-vertex wind weight, and `UV_1.y` carries the
/// foliage-atlas leaf code the fragment samples for leaf shape + alpha.
pub type TreeMaterial = ExtendedMaterial<StandardMaterial, TreeShadingExtension>;

/// The tree extension's bindings (100+, clear of the base material's).
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct TreeShadingExtension {
    /// Wind + clipmap fade band + anchor (lighting fields are layout-parity
    /// leftovers from the shared [`GrassParams`]; the Bevy sun lights the tree).
    #[uniform(100)]
    pub params: GrassParams,
    /// Procedural foliage atlas (leaf clusters + shell + bark), built once at
    /// startup by [`crate::ground::build_foliage_atlas`].
    #[texture(101)]
    #[sampler(102)]
    pub atlas: Handle<Image>,
    /// Companion **material atlas** (bark normal + roughness), built by
    /// [`crate::ground::build_foliage_material_atlas`]. Linear data; shares the
    /// albedo atlas's sampler.
    #[texture(103)]
    pub material_atlas: Handle<Image>,
    /// Cascaded sun-shadow transforms + strength (see [`ShadowCascadeBlock`]).
    #[uniform(104)]
    pub shadow: ShadowCascadeBlock,
    /// Per-cascade sun-shadow depth maps (near→far) — the same handles the
    /// terrain binds. Must always be valid depth textures (see
    /// [`fallback_shadow_map`]); `shadow.config.x` gates sampling.
    /// Cascade 0 — the ±64 m near box added 2026-07-31. Bound at the END of
    /// this material's range rather than renumbered into the near→far slot:
    /// only the ARGUMENT order at the `thalos::shadow` call site is
    /// ordering-significant, so shifting live binding indices would be risk
    /// with no payoff. Field name still equals the cascade index.
    #[texture(111, sample_type = "depth")]
    pub sun_shadow_map_0: Handle<Image>,
    #[texture(105, sample_type = "depth")]
    pub sun_shadow_map_1: Handle<Image>,
    #[texture(106, sample_type = "depth")]
    pub sun_shadow_map_2: Handle<Image>,
    #[texture(107, sample_type = "depth")]
    pub sun_shadow_map_3: Handle<Image>,
    /// Cloud sun-transmittance cascade (CLOUD-5 / W2) — the same map the tile
    /// ground samples, fanned in per frame by the game's cloud-shadow driver so
    /// a forest under the deck dims exactly with the ground it stands on.
    #[texture(108)]
    #[sampler(109)]
    pub cloud_shadow_map: Handle<Image>,
    /// Placement of that cascade (separate writer from `params`, same reason as
    /// the tile material's split).
    #[uniform(110)]
    pub cloud_shadow: CloudShadowBlock,
}

/// The base `StandardMaterial` every tree material wraps. `diffuse_transmission`
/// must be non-zero HERE (not only per-fragment) — it is what compiles the
/// transmission branch into the pipeline; the fragment then scales it by the
/// leaf flag so bark stays opaque.
pub fn tree_base_material() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.95,
        metallic: 0.0,
        diffuse_transmission: 1.0,
        // Thin canopy silhouettes keep their back faces. `prepare_world_normal`
        // flips `pbr_input.world_normal` on back-facing fragments — and the
        // fragment shader must derive its shading normal from THAT field, not
        // the raw `in.world_normal` varying, or the flip is silently lost
        // (tree_standard.wgsl, reviews/20260730T011353Z §1).
        double_sided: true,
        cull_mode: None,
        ..Default::default()
    }
}

/// Wrap a [`TreeShadingExtension`] into the full tree material.
pub fn tree_material(extension: TreeShadingExtension) -> TreeMaterial {
    TreeMaterial {
        base: tree_base_material(),
        extension,
    }
}

impl MaterialExtension for TreeShadingExtension {
    fn vertex_shader() -> ShaderRef {
        "shaders/tree_standard.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/tree_standard.wgsl".into()
    }

    // Stay OUT of the depth prepass. The camera runs a `DepthPrepass` (grass
    // early-Z), but the broadleaf canopy is dozens of overlapping double-sided
    // translucent leaf cards at near-identical depths, and pre-populating depth
    // for them washes the canopy out pale — **empirically confirmed** on the
    // spine version with both an Equal and a forced GreaterEqual depth test.
    fn enable_prepass() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Anti-aliased foliage under MSAA: the leaf cards are a 1-bit alpha
        // cutout, and plain MSAA super-samples geometry coverage, not the alpha
        // test. With MSAA on, the shader writes a sharpened fractional coverage
        // (`TREE_ALPHA_TO_COVERAGE` branch) and hardware alpha-to-coverage
        // turns it into a per-sample mask. The MSAA-off path is byte-identical
        // to a plain opaque draw.
        if descriptor.multisample.count > 1 {
            descriptor.multisample.alpha_to_coverage_enabled = true;
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("TREE_ALPHA_TO_COVERAGE".into());
            }
        }
        Ok(())
    }
}

/// Lean plugin to render [`TreeMaterial`] in a standalone / headless app (the
/// object-preview tool, examples) **without** the full terrain stack: registers
/// the material and ensures the shared shader libraries are present. The game
/// gets all this from `GroundAppearancePlugin`.
pub struct TreeMaterialPlugin;

impl Plugin for TreeMaterialPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
        }
        app.add_plugins(bevy::pbr::MaterialPlugin::<TreeMaterial>::default());
    }
}
