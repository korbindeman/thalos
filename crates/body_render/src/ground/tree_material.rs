//! Tree / shrub instanced material: vertex wind sway + the shared
//! `thalos::lighting` sky model, so scattered plants move in the wind and light
//! exactly like the grass and ground they grow from.
//!
//! Reuses [`GrassParams`] as its uniform (same field layout): `sun_dir`
//! (w = flux), `wind` (w = canopy sway amplitude), `time_fade.x` (= time), and
//! the `sky_up` / `sky_tau` hemisphere inputs. The per-vertex **wind weight**
//! rides the mesh's vertex-colour alpha (0 at the trunk, → 1 at the canopy top),
//! and per-instance phase + tint jitter are hashed in-shader from the instance's
//! world position, so all instances of a species still share one
//! `(Mesh, Material)` and auto-batch.

use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, RenderPipelineDescriptor, SpecializedMeshPipelineError,
    TextureDimension, TextureFormat, TextureUsages,
};
use bevy::shader::ShaderRef;

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

/// Material for scattered trees and shrubs. One instance is shared by every
/// plant on a body; the mesh's vertex colours carry the trunk/canopy tint × AO
/// and the per-vertex wind weight, and `UV_1.y` carries the foliage-atlas leaf
/// code the fragment samples for leaf shape + alpha.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct TreeMaterial {
    #[uniform(0)]
    pub params: GrassParams,
    /// Procedural foliage atlas (leaf clusters + shell + bark), built once at
    /// startup by [`crate::ground::build_foliage_atlas`].
    #[texture(1)]
    #[sampler(2)]
    pub atlas: Handle<Image>,
    /// Companion **material atlas** (bark normal + roughness), built by
    /// [`crate::ground::build_foliage_material_atlas`]. Linear data; shares the
    /// albedo atlas's sampler (binding 2). Only bark fragments sample it.
    #[texture(7)]
    pub material_atlas: Handle<Image>,
    /// Cascaded sun-shadow transforms + strength (see [`ShadowCascadeBlock`]).
    #[uniform(3)]
    pub shadow: ShadowCascadeBlock,
    /// Per-cascade sun-shadow depth maps (near→far) — the same handles the
    /// terrain binds. Each a plain `texture_depth_2d` (no depth array). Must
    /// always be valid depth textures (see [`fallback_shadow_map`]);
    /// `shadow.config.x` gates whether they're sampled.
    #[texture(4, sample_type = "depth")]
    pub sun_shadow_map_0: Handle<Image>,
    #[texture(5, sample_type = "depth")]
    pub sun_shadow_map_1: Handle<Image>,
    #[texture(6, sample_type = "depth")]
    pub sun_shadow_map_2: Handle<Image>,
}

impl Material for TreeMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/tree.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/tree.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Two-sided: thin canopy silhouettes shouldn't drop their back faces.
        // Do NOT override `descriptor.vertex.buffers` — Bevy's mesh pipeline
        // auto-includes every attribute the mesh has (POSITION/NORMAL/UV_0/UV_1/
        // COLOR at their standard locations) for both the main and prepass
        // pipelines. Overriding it truncated the layout the standard prepass
        // vertex shader needs (a location-7 input) and failed pipeline creation.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

pub(crate) fn embed_tree_shader(app: &mut App) {
    embedded_asset!(app, "tree.wgsl");
}

/// Lean plugin to render [`TreeMaterial`] in a standalone / headless app (the
/// object-preview tool, examples) **without** the full UDLOD terrain stack:
/// registers the material, embeds its shader, and ensures the `thalos::lighting`
/// library is present. The game gets all this from `ThalosTerrainPlugin`; this is
/// the minimal entry point for everything that only needs to draw plants.
pub struct TreeMaterialPlugin;

impl Plugin for TreeMaterialPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
        }
        app.add_plugins(bevy::pbr::MaterialPlugin::<TreeMaterial>::default());
        embed_tree_shader(app);
    }
}
