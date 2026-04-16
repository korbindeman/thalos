//! Saturn-style ring material.
//!
//! Renders a flat annulus as a Bevy material with a per-fragment radial
//! lookup against an authored palette. The ring mesh lives as a child
//! of the gas-giant body entity and inherits its translation; a
//! per-ring transform carries the body's axial tilt so the ring plane
//! lines up with the body's equatorial plane.
//!
//! ## Features
//!
//! - Radial palette: up to 8 stops in [0, 1], each with RGB + opacity.
//! - Procedural multi-octave ringlet noise that breaks the smooth
//!   palette into thousands of bright/dark ringlets without needing a
//!   texture asset.
//! - Planet shadow: the shader ray-casts from every ring point toward
//!   the sun and tests against the planet sphere — points behind the
//!   planet go dark, producing the classic shadow stripe across the
//!   lit face of the rings.
//! - Forward-scatter / back-scatter lighting: rings pick up Mie-like
//!   forward scattering when viewed against the sun (dust glow on the
//!   unlit side) plus Lambert diffuse on the lit side.

use bevy::asset::RenderAssetUsages;
use bevy::math::{Vec3, Vec4};
use bevy::mesh::{Indices, Mesh, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use thalos_atmosphere_gen::RingSystem;

use crate::lighting::SceneLighting;

/// Maximum number of radial palette stops the ring shader supports.
/// Matches the WGSL `MAX_RING_STOPS` constant.
pub const MAX_RING_STOPS: usize = 8;

/// Per-frame uniform: planet shadow geometry + shared scene lighting.
#[derive(Clone, ShaderType)]
pub struct RingParams {
    /// Planet center in world space (for shadow ray test).
    /// w = planet render radius (used as ray-sphere radius).
    pub planet_center_radius: Vec4,
    /// Inner radius in render units. Mirrors the per-body authoring
    /// so the shader can recover the normalised radial coordinate.
    pub inner_radius: f32,
    /// Outer radius in render units.
    pub outer_radius: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    /// Stars, ambient, eclipse occluders. Rings leave planetshine zeroed.
    pub scene: SceneLighting,
}

impl Default for RingParams {
    fn default() -> Self {
        Self {
            planet_center_radius: Vec4::ZERO,
            inner_radius: 1.0,
            outer_radius: 2.0,
            _pad0: 0.0,
            _pad1: 0.0,
            scene: SceneLighting::default(),
        }
    }
}

/// Per-body authoring uniform: radial palette, ringlet noise, seed.
#[derive(Clone, ShaderType)]
pub struct RingLayers {
    /// Radial palette stops. xyz = linear RGB, w = normalised radial
    /// position in [0, 1]. Only the first `stop_count` entries are read.
    pub palette_color: [Vec4; MAX_RING_STOPS],
    /// Per-stop opacity + unused padding. x = opacity in [0, 1].
    pub palette_opacity: [Vec4; MAX_RING_STOPS],
    /// Number of valid stops.
    pub stop_count: u32,
    /// Overall opacity multiplier.
    pub opacity: f32,
    /// Ringlet-noise amplitude.
    pub ringlet_noise: f32,
    /// Ringlet-noise octave count (clamped in-shader to [1, 8]).
    pub ringlet_octaves: u32,
    /// Noise seed low dword.
    pub seed_lo: u32,
    /// Noise seed high dword.
    pub seed_hi: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl Default for RingLayers {
    fn default() -> Self {
        Self {
            palette_color: [Vec4::ZERO; MAX_RING_STOPS],
            palette_opacity: [Vec4::ZERO; MAX_RING_STOPS],
            stop_count: 0,
            opacity: 1.0,
            ringlet_noise: 0.0,
            ringlet_octaves: 6,
            seed_lo: 0,
            seed_hi: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

impl RingLayers {
    /// Pack a `RingSystem` into a GPU-friendly layer set.
    pub fn from_system(rings: &RingSystem) -> Self {
        let mut layers = Self::default();
        let n = rings.palette.len().min(MAX_RING_STOPS);
        for (i, stop) in rings.palette.iter().take(n).enumerate() {
            layers.palette_color[i] = Vec4::new(
                stop.color[0],
                stop.color[1],
                stop.color[2],
                stop.r.clamp(0.0, 1.0),
            );
            layers.palette_opacity[i] =
                Vec4::new(stop.opacity.clamp(0.0, 1.0), 0.0, 0.0, 0.0);
        }
        layers.stop_count = n as u32;
        layers.opacity = rings.opacity.clamp(0.0, 1.0);
        layers.ringlet_noise = rings.ringlet_noise.clamp(0.0, 1.0);
        layers.ringlet_octaves = rings.ringlet_octaves.clamp(1, 8);
        layers.seed_lo = rings.seed as u32;
        layers.seed_hi = (rings.seed >> 32) as u32;
        layers
    }
}

/// Bevy `Material` for the ring annulus.
///
/// Binding layout (group 3):
///
/// | Binding | Kind    | WGSL type  | Source   |
/// |---------|---------|------------|----------|
/// | 0       | uniform | RingParams | `params` |
/// | 1       | uniform | RingLayers | `layers` |
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct RingMaterial {
    #[uniform(0)]
    pub params: RingParams,
    #[uniform(1)]
    pub layers: RingLayers,
}

impl Material for RingMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/ring.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/ring.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Rings are double-sided — viewer may look from either face.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Handle component so the per-frame update system can find and mutate
/// the ring material from its owning body entity.
#[derive(Component)]
pub struct RingMaterialHandle(pub Handle<RingMaterial>);

/// Build a flat annulus mesh in the XZ plane (body-local equatorial).
///
/// `segments` controls the angular tessellation. Each angular slice
/// contributes two triangles connecting the inner and outer rims.
/// Vertex attribute layout:
/// - POSITION: 3 floats in render units
/// - NORMAL: 3 floats (+Y for top face, -Y for bottom face)
/// - UV_0: 2 floats where `u` is the normalised radial position in
///   [0, 1] (0 = inner rim, 1 = outer rim) and `v` is the angular
///   position in [0, 1] (wraps around the ring).
pub fn build_ring_mesh(inner_radius: f32, outer_radius: f32, segments: u32) -> Mesh {
    let segments = segments.max(64);
    let step = std::f32::consts::TAU / segments as f32;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((segments as usize + 1) * 2);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity((segments as usize + 1) * 2);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity((segments as usize + 1) * 2);
    let mut indices: Vec<u32> = Vec::with_capacity(segments as usize * 6);

    for i in 0..=segments {
        let angle = i as f32 * step;
        let cs = angle.cos();
        let sn = angle.sin();
        let v = i as f32 / segments as f32;

        // Inner vertex
        positions.push([inner_radius * cs, 0.0, inner_radius * sn]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([0.0, v]);

        // Outer vertex
        positions.push([outer_radius * cs, 0.0, outer_radius * sn]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([1.0, v]);
    }

    for i in 0..segments {
        let base = i * 2;
        // Quad made of two triangles: (inner_i, outer_i, inner_i+1),
        // (outer_i, outer_i+1, inner_i+1).
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Compute the ring plane normal in world space from a body's axial
/// tilt quaternion. Returns a unit vector.
pub fn ring_plane_normal(axial_tilt: Quat) -> Vec3 {
    axial_tilt * Vec3::Y
}
