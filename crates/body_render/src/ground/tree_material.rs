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

use bevy::asset::embedded_asset;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::ground::vegetation::GrassParams;

/// Material for scattered trees and shrubs. One instance is shared by every
/// plant on a body; the mesh's vertex colours carry the trunk/canopy palette and
/// the per-vertex wind weight.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct TreeMaterial {
    #[uniform(0)]
    pub params: GrassParams,
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
