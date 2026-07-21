//! Analytic ocean sphere for the orbital map view.
//!
//! A fullscreen ray-traced sphere at sea level — the same billboard trick as
//! [`super::SolidPlanetMaterial`] — that shades water through
//! `thalos::water::shade_ocean` and writes true `@builtin(frag_depth)`. Because
//! it is an ordinary opaque material writing real per-pixel depth, it depth-tests
//! against the map terrain: land occludes it, the seabed sits behind it, and the
//! waterline is an exact analytic curve. No mesh, no facets, no z-fighting, and
//! no scene-depth copy — the hardware depth buffer does the land/sea sorting.
//! Map-scale projection of the same analytic ocean authority used in ship view.
//!
//! At MAP_SCALE (1 render unit = 1000 km) the body is a ~few-unit sphere, so the
//! naive ray-sphere quadratic is f32-stable here (unlike the ship path, which is
//! at planet-radius magnitude and needs the CPU-altitude reformulation).

use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::shading::SceneLighting;

#[derive(Clone, ShaderType)]
pub struct MapOceanParams {
    /// Ocean sphere radius in render units (`body_radius + sea_level`, MAP_SCALE).
    pub radius: f32,
    /// Deep-water linear-RGB tint (xyz) + minimum optical depth (w). Matches the
    /// ship ocean's fallback so the two views agree.
    pub color_depth: Vec4,
    /// Sun direction + flux (`scene.stars[0]`), ambient, planetshine, eclipse —
    /// the lighting the water BRDF reads.
    pub scene: SceneLighting,
}

impl Default for MapOceanParams {
    fn default() -> Self {
        Self {
            radius: 1.0,
            color_depth: Vec4::new(0.012, 0.040, 0.090, 120.0),
            scene: SceneLighting::default(),
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone, Default)]
pub struct MapOceanMaterial {
    #[uniform(0)]
    pub params: MapOceanParams,
}

impl Material for MapOceanMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/map_ocean.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/map_ocean.wgsl".into()
    }

    // Opaque: writes depth and depth-tests against the map terrain so land/sea
    // sorting is done by the hardware depth buffer.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Fullscreen quad — no culling.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}
