//! Fullscreen-quad atmospheric sky for one body.
//!
//! Renders the in-scatter integral of `thalos::atmosphere` for every view
//! ray, premultiplied with opacity. Blended over the celestial background so
//! stars dim where the atmosphere is thick (near horizon, near sun) and
//! remain visible at the zenith where the atmosphere is thin.
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
    /// CPU-baked multi-scatter LUT for this body's atmosphere. Indexed by
    /// `(μ_s = cos(sun_zenith), h_norm = h / atmos_top)` and sampled once
    /// per view step in `integrate_atmosphere` to add the second-order
    /// in-scatter term that pure single-scattering can't produce — the
    /// physical source of the daytime sky's blue cast and the
    /// luminance-headroom over star brightness. Baked once at body spawn
    /// (`thalos_planet_lighting::bake_multi_scatter_lut`); never reuploaded
    /// at runtime.
    #[texture(3, sample_type = "float", dimension = "2d")]
    #[sampler(4)]
    pub multi_scatter_lut: Handle<Image>,
}

impl Material for BodySkyMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_terrain/sky_dome.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_terrain/sky_dome.wgsl".into()
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
            // The unified atmosphere pass renders on every pixel (terrain
            // + impostor body + sky) and clips the raymarch with the
            // sampled scene depth instead of via depth-compare. Disable
            // both depth write and depth test.
            depth.depth_write_enabled = false;
            depth.depth_compare = CompareFunction::Always;
        }
        Ok(())
    }
}

pub(crate) fn embed_sky_dome_shader(app: &mut App) {
    embedded_asset!(app, "sky_dome.wgsl");
}
