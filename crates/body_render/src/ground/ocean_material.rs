//! Dedicated analytic-ocean fullscreen projection.
//!
//! The ocean and legacy atmosphere share one per-body optical/resource
//! contract, but they are distinct render owners. `BodySkyMaterial` compiles
//! `body_sky.wgsl` as atmosphere-only; this material compiles the same source
//! as ocean-only. Sharing the shader keeps the signed-field lookup, spectral
//! slopes, and foreground-air integration canonical while allowing Bevy's
//! atmosphere to replace the legacy sky without deleting water.

use bevy::ecs::system::SystemParamItem;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, AsBindGroupError, BindGroupLayout, BindGroupLayoutEntry, CompareFunction,
    RenderPipelineDescriptor, SpecializedMeshPipelineError, UnpreparedBindGroup,
};
use bevy::render::renderer::RenderDevice;
use bevy::shader::ShaderRef;

use super::BodySkyMaterial;

/// Ocean-only projection using the exact resource contract of the legacy sky.
///
/// The wrapper deliberately delegates `AsBindGroup` to `BodySkyMaterial` so
/// there is one binding implementation for scene depth, the signed-height
/// cascade, and the shared spectral slope field.
#[derive(Asset, TypePath, Clone, Default)]
pub struct BodyOceanMaterial {
    pub optical: BodySkyMaterial,
}

impl AsBindGroup for BodyOceanMaterial {
    type Data = <BodySkyMaterial as AsBindGroup>::Data;
    type Param = <BodySkyMaterial as AsBindGroup>::Param;

    fn label() -> &'static str {
        "body_ocean_material"
    }

    fn bind_group_data(&self) -> Self::Data {
        <BodySkyMaterial as AsBindGroup>::bind_group_data(&self.optical)
    }

    fn unprepared_bind_group(
        &self,
        layout: &BindGroupLayout,
        render_device: &RenderDevice,
        param: &mut SystemParamItem<'_, '_, Self::Param>,
        force_no_bindless: bool,
    ) -> Result<UnpreparedBindGroup, AsBindGroupError> {
        <BodySkyMaterial as AsBindGroup>::unprepared_bind_group(
            &self.optical,
            layout,
            render_device,
            param,
            force_no_bindless,
        )
    }

    fn bind_group_layout_entries(
        render_device: &RenderDevice,
        force_no_bindless: bool,
    ) -> Vec<BindGroupLayoutEntry> {
        <BodySkyMaterial as AsBindGroup>::bind_group_layout_entries(
            render_device,
            force_no_bindless,
        )
    }
}

impl Material for BodyOceanMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/body_sky.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/body_sky.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Premultiplied
    }

    fn depth_bias(&self) -> f32 {
        // Stable transparent order for body-centred fullscreen siblings:
        // legacy BodySky atmosphere (0) < ocean (500) < clouds (1000).
        500.0
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader_defs.push("OCEAN_ONLY".into());
        }
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = Some(false);
            depth.depth_compare = Some(CompareFunction::Always);
        }
        Ok(())
    }
}
