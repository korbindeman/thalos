use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

/// Per-planet uniform data sent to the impostor shader.
///
/// Field layout is explicit to match WGSL std140 alignment:
///   offset 0  -- radius          (f32, 4 bytes)
///   offset 4  -- rotation_phase  (f32, 4 bytes)
///   offset 8  -- light_intensity (f32, 4 bytes)
///   offset 12 -- ambient_intensity (f32, 4 bytes)
///   offset 16 -- light_dir       (vec4, 16 bytes)
///   total: 32 bytes
#[derive(Clone, ShaderType)]
pub struct PlanetParams {
    /// Sphere radius in render units.
    pub radius: f32,
    /// Rotation offset (radians / 2pi) added to the sample direction for planet spin.
    pub rotation_phase: f32,
    /// Illuminance of the directional sun light in lux (matches DirectionalLight.illuminance).
    pub light_intensity: f32,
    /// Ambient illuminance in lux (matches GlobalAmbientLight.brightness).
    pub ambient_intensity: f32,
    /// Normalised direction FROM the planet's surface TOWARD the star, in world space.
    /// xyz = direction, w = surface roughness (0.0 = smooth gas giant, 1.0 = very rough).
    pub light_dir: Vec4,
}

impl Default for PlanetParams {
    fn default() -> Self {
        Self {
            radius: 1.0,
            rotation_phase: 0.0,
            light_intensity: 80.0,
            ambient_intensity: 2.0,
            light_dir: Vec4::new(0.0, 1.0, 0.0, 0.0),
        }
    }
}

/// Per-planet **detail** uniform consumed by the shader's per-fragment
/// small-crater synthesis pass.
#[derive(Clone, ShaderType)]
pub struct PlanetDetailParams {
    pub body_radius_m: f32,
    pub d_min_m: f32,
    pub d_max_m: f32,
    pub sfd_alpha: f32,
    pub global_k_per_km2: f32,
    pub d_sc_m: f32,
    pub body_age_gyr: f32,
    pub _pad: f32,
    pub seed_lo: u32,
    pub seed_hi: u32,
}

impl Default for PlanetDetailParams {
    fn default() -> Self {
        Self {
            body_radius_m: 1.0,
            d_min_m: 0.0,
            d_max_m: 0.0,
            sfd_alpha: 2.0,
            global_k_per_km2: 0.0,
            d_sc_m: 1.0,
            body_age_gyr: 4.5,
            _pad: 0.0,
            seed_lo: 0,
            seed_hi: 0,
        }
    }
}

impl PlanetDetailParams {
    /// Build from terrain_gen's DetailNoiseParams.
    pub fn from_detail_noise(params: &thalos_terrain_gen::DetailNoiseParams) -> Self {
        Self {
            body_radius_m: params.body_radius_m,
            d_min_m: params.d_min_m,
            d_max_m: params.d_max_m,
            sfd_alpha: params.sfd_alpha,
            global_k_per_km2: params.global_k_per_km2,
            d_sc_m: params.d_sc_m,
            body_age_gyr: params.body_age_gyr,
            _pad: 0.0,
            seed_lo: params.seed as u32,
            seed_hi: (params.seed >> 32) as u32,
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct PlanetMaterial {
    #[uniform(0)]
    pub params: PlanetParams,
    #[texture(1, dimension = "cube")]
    #[sampler(2)]
    pub albedo: Handle<Image>,
    #[texture(3, dimension = "cube")]
    #[sampler(4)]
    pub height: Handle<Image>,
    #[uniform(5)]
    pub detail: PlanetDetailParams,
}

impl Material for PlanetMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/planet_impostor.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/planet_impostor.wgsl".into()
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

/// Stored on the parent `CelestialBody` entity so the per-frame update system
/// can find and mutate the material without traversing children.
#[derive(Component)]
pub struct PlanetMaterialHandle(pub Handle<PlanetMaterial>);
