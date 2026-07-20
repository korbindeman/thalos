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

use crate::shading::{AtmosphereBlock, SceneLighting};

#[derive(Clone, ShaderType)]
pub struct SolidPlanetParams {
    /// Sphere radius in render units.
    pub radius: f32,
    /// Linear-RGB surface albedo (`color × albedo` from the body's RON,
    /// converted from sRGB to linear at spawn). xyz = flat colour; **w = use the
    /// baked `albedo_cube` instead** (≥ 0.5 → sample the cube by the body-fixed
    /// normal; < 0.5 → use the flat xyz colour). Procedural bodies set it to 1.
    pub albedo: Vec4,
    /// Quaternion (xyzw) rotating a render-space direction into the body-fixed
    /// frame the `albedo_cube` is baked in, so the texture co-rotates with the
    /// planet. Identity when the flat colour is used.
    pub orientation: Vec4,
    /// Stars, eclipse occluders, ambient, planetshine parent.
    pub scene: SceneLighting,
    /// Single-scattering atmosphere optics for the rim halo + on-disc aerial
    /// perspective. `AtmosphereBlock::default()` is a vacuum (every gate scalar
    /// zero), so airless solid bodies render bit-identically to the pre-
    /// atmosphere placeholder — the shader early-outs on the `strength == 0`
    /// gate. Authored in the same render units as `radius`.
    pub atmosphere: AtmosphereBlock,
}

impl Default for SolidPlanetParams {
    fn default() -> Self {
        Self {
            radius: 1.0,
            albedo: Vec4::splat(0.5),
            orientation: Vec4::new(0.0, 0.0, 0.0, 1.0),
            scene: SceneLighting::default(),
            atmosphere: AtmosphereBlock::default(),
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct SolidPlanetMaterial {
    #[uniform(0)]
    pub params: SolidPlanetParams,
    /// Baked low-frequency impostor albedo (continents + oceans), sampled by the
    /// body-fixed normal when `params.albedo.w >= 0.5`. Procedural bodies bind a
    /// [`crate::bake_impostor_albedo_cube`] result; solid-colour bodies bind a
    /// 1×1 [`crate::blank_impostor_cube`] and never sample it. Only the body pass
    /// declares this binding (gated `#ifndef HALO_PASS`), so
    /// [`SolidPlanetHaloMaterial`] does not carry it.
    #[texture(1, dimension = "cube")]
    #[sampler(2)]
    pub albedo_cube: Handle<Image>,
    /// Per-body multi-scatter LUT (32×32 `Rgba16Float`) — the **same** texture
    /// [`crate::BodySkyMaterial`] binds. Only the body pass samples it (via
    /// `integrate_atmosphere_multiscatter`, gated `#ifndef HALO_PASS`), to add the
    /// diffuse second-order blue fill single scattering omits so the distant disc
    /// reads as a real atmosphere-veiled planet rather than a hairline rim. The
    /// halo material carries no LUT. Airless bodies bind a 1×1 blank — the
    /// shader's `atmosphere_scattering_active` gate skips the sample there.
    #[texture(3)]
    #[sampler(4)]
    pub multi_scatter_lut: Handle<Image>,
    /// Canonical RGBA8 cloud-weather cubemap. The first orbital projection uses
    /// its coverage channel; later cloud LODs consume type/base/top as well.
    /// Clear bodies bind a shared zero cube.
    #[texture(5, dimension = "cube")]
    #[sampler(6)]
    pub cloud_weather: Handle<Image>,
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

/// Rim-glow companion to [`SolidPlanetMaterial`].
///
/// Same fullscreen ray-traced billboard and the same [`SolidPlanetParams`]
/// uniform, compiled with the `HALO_PASS` shader-def so the fragment shader
/// keeps only the atmospheric in-scatter on rays that *miss* the solid disc —
/// the blue limb shell outside the silhouette. This is the orbital-map /
/// distant-impostor atmosphere: cheap, depth-decoupled (no scene-depth copy),
/// and driven by the same `AtmosphereBlock` the in-context `BodySky` pass uses.
///
/// Premultiplied-alpha and *no depth write* (mirroring the impostor's
/// `PlanetHaloMaterial`): the rim must depth-test against opaque foreground
/// bodies but must not occlude the stars/galaxies/orbit lines drawn behind it.
/// Spawn one as a sibling of the body's `SolidPlanetMaterial` billboard and
/// keep its `params` in lockstep (radius + scene lighting) each frame.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct SolidPlanetHaloMaterial {
    #[uniform(0)]
    pub params: SolidPlanetParams,
}

impl From<&SolidPlanetMaterial> for SolidPlanetHaloMaterial {
    fn from(material: &SolidPlanetMaterial) -> Self {
        Self {
            params: material.params.clone(),
        }
    }
}

impl Material for SolidPlanetHaloMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/solid_planet.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/solid_planet.wgsl".into()
    }

    // The halo returns premultiplied atmospheric in-scatter over whatever
    // passed the depth test behind the rim.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Premultiplied
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader_defs.push("HALO_PASS".into());
        }
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            // Depth-test against opaque foreground, but never write: stars and
            // galaxies sit at the reverse-Z far plane and must stay visible
            // behind the rim glow.
            depth.depth_write_enabled = Some(false);
        }
        Ok(())
    }
}
