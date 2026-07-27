//! Canonical fullscreen cloud composite.
//!
//! The cloud compute pass is atmosphere-agnostic and writes one premultiplied
//! cloud layer plus hit depth. This material is the sole screen compositor for
//! that layer: it runs in Bevy's transparent main pass after the canonical
//! `BodySky` atmosphere and keeps cloud ownership independent of the sky
//! material.

use bevy::asset::embedded_asset;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, CompareFunction, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::ground::BodySkyExtra;
use crate::shading::AtmosphereBlock;

#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct CloudCompositeMaterial {
    /// Authored atmosphere block. The orbital cloud projection reuses its
    /// cloud albedo/shape contract; the near volume is already lit upstream.
    #[uniform(0)]
    pub atmosphere: AtmosphereBlock,
    /// Per-frame planet, sun, orientation, and cloud-band geometry.
    #[uniform(1)]
    pub params: BodySkyExtra,
    /// Opaque main-pass depth copied between opaque and transparent rendering.
    #[texture(2, sample_type = "depth")]
    pub scene_depth: Handle<Image>,
    /// Canonical per-body weather cubemap used by the orbital projection.
    #[texture(3, dimension = "cube")]
    #[sampler(4)]
    pub weather: Handle<Image>,
    /// Near-volume premultiplied cloud radiance + transmittance.
    #[texture(5, sample_type = "float", filterable = false)]
    pub cloud_layer: Handle<Image>,
    /// Near-volume first-hit distance, metres from the ship camera.
    #[texture(6, sample_type = "float", filterable = false)]
    pub cloud_distance: Handle<Image>,
    /// Canonical four-stratum surface-space broad density. Reuses the weather
    /// sampler at binding 4 so both projections select identical mip footprints.
    #[texture(7, dimension = "cube")]
    pub surface_density: Handle<Image>,
}

impl Material for CloudCompositeMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_body_render/clouds/shaders/cloud_composite.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_body_render/clouds/shaders/cloud_composite.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Premultiplied
    }

    fn depth_bias(&self) -> f32 {
        crate::composite_order::CLOUDS
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = Some(false);
            depth.depth_compare = Some(CompareFunction::Always);
        }
        Ok(())
    }
}

pub(super) fn embed_cloud_composite_shader(app: &mut App) {
    embedded_asset!(app, "shaders/cloud_composite.wgsl");
}
