//! Solid-color planet placeholder material.
//!
//! Used for bodies that don't have a terrain pipeline configured yet:
//! same camera-facing billboard / ray-traced sphere as the impostor and
//! gas-giant materials, so close approaches don't clip against the
//! camera near plane. The fragment shader skips all cubemap and SSBO
//! sampling — a single linear-RGB albedo drives the surface read.

use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::lighting::SceneLighting;

#[derive(Clone, ShaderType)]
pub struct SolidPlanetParams {
    /// Sphere radius in render units.
    pub radius: f32,
    /// Linear-RGB surface albedo (`color × albedo` from the body's RON,
    /// converted from sRGB to linear at spawn). xyz used; w reserved.
    pub albedo: Vec4,
    /// Stars, eclipse occluders, ambient, planetshine parent.
    pub scene: SceneLighting,
}

impl Default for SolidPlanetParams {
    fn default() -> Self {
        Self {
            radius: 1.0,
            albedo: Vec4::splat(0.5),
            scene: SceneLighting::default(),
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct SolidPlanetMaterial {
    #[uniform(0)]
    pub params: SolidPlanetParams,
}

impl Material for SolidPlanetMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/solid_planet.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/solid_planet.wgsl".into()
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
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}
