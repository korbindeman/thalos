//! Fullscreen-quad atmospheric sky for one body.
//!
//! Renders the in-scatter integral of `thalos::atmosphere` for every view
//! ray, premultiplied with opacity. Blended over the celestial background so
//! stars dim where the daytime sky is bright (the in-scatter alpha boost
//! crushes them) and re-emerge only as the sky darkens toward night/twilight.
//!
//! Reuses the same atmosphere uniforms as [`crate::BodyTerrainMaterial`] so a
//! single per-frame update system writes both.

use bevy::asset::embedded_asset;
use bevy::image::Image;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, CompareFunction, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use thalos_planet_lighting::AtmosphereBlock;

use crate::body_material::BodySkyExtra;

#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct BodySkyMaterial {
    #[uniform(0)]
    pub atmosphere: AtmosphereBlock,
    #[uniform(1)]
    pub atmosphere_extra: BodySkyExtra,
    /// Scene-depth texture written by the game crate's `CopySceneDepthNode`
    /// each frame between `Node3d::MainOpaquePass` and
    /// `Node3d::MainTransparentPass`. Sampled with `textureLoad` in the
    /// fragment shader to clip the atmosphere raymarch at opaque geometry,
    /// which is what produces aerial perspective on terrain pixels.
    #[texture(2, sample_type = "depth")]
    pub scene_depth: Handle<Image>,
    /// Reference cloud-cover cubemap shared with the impostor material.
    /// Bodies without a registered overlay bind the same blank cube fallback.
    #[texture(3, dimension = "cube")]
    #[sampler(4)]
    pub cloud_cover: Handle<Image>,
    /// Per-body multi-scatter LUT (32×32 `Rgba16Float`), baked once at spawn by
    /// `thalos_planet_lighting::bake_multi_scatter_lut` and never updated — the
    /// atmosphere parameters are static. `body_sky.wgsl` samples it at every
    /// view step via `integrate_atmosphere_multiscatter` to add the
    /// second-order in-scatter that single scattering omits. That term is what
    /// gives the daytime dome its blue luminance and lifts the in-scatter into
    /// the range where the sky-luminance alpha boost washes out stars at noon;
    /// without it the midday sky is physically dim and the celestial backdrop
    /// bleeds through. Indexed by `(u = (sun·zenith + 1) / 2, v = h / atmos_top)`.
    #[texture(5)]
    #[sampler(6)]
    pub multi_scatter_lut: Handle<Image>,
}

impl Material for BodySkyMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_terrain_render/body_sky.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_terrain_render/body_sky.wgsl".into()
    }

    // Premultiplied: `rgb = in_scatter` is additive over the background, and
    // `(1 − alpha)` = mean atmospheric transmittance dims whatever was behind
    // (stars, galaxies, terrain). The fullscreen pass now overdraws onto
    // terrain too (depth_compare = Always) so its in-scatter and
    // transmittance composite uniformly across the frame.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Premultiplied
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Fullscreen quad — no culling.
        descriptor.primitive.cull_mode = None;
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            // The terrain atmosphere pass renders on every pixel and clips
            // the raymarch with sampled scene depth instead of via
            // depth-compare. Disable both depth write and depth test.
            depth.depth_write_enabled = false;
            depth.depth_compare = CompareFunction::Always;
        }
        Ok(())
    }
}

pub(crate) fn embed_body_sky_shader(app: &mut App) {
    embedded_asset!(app, "body_sky.wgsl");
}
