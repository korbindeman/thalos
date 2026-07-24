use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::{AsBindGroup, ShaderType, UniformBuffer},
    },
};

use super::images::{RENDER_HEIGHT, RENDER_WIDTH};

#[derive(Clone, Resource, ExtractResource, Reflect, ShaderType)]
#[reflect(Resource, Default)]
pub(crate) struct CloudsUniform {
    pub clouds_base_shape_scale_m: f32,
    pub clouds_raymarch_steps_count: u32,
    pub clouds_bottom_height: f32,
    pub clouds_top_height: f32,
    pub clouds_coverage: f32,
    pub clouds_density: f32,
    pub clouds_detail_scale_m: f32,
    pub surface_density_coupling: f32,
    pub clouds_detail_strength: f32,
    pub clouds_base_edge_softness: f32,
    pub clouds_bottom_softness: f32,
    pub clouds_shadow_raymarch_steps_count: u32,
    pub clouds_shadow_raymarch_step_size: f32,
    pub clouds_shadow_raymarch_step_multiply: f32,
    pub clouds_ambient_color_top: Vec4,
    pub clouds_ambient_color_bottom: Vec4,
    pub clouds_min_transmittance: f32,
    pub planet_radius: f32,
    pub forward_scattering_g: f32,
    pub backward_scattering_g: f32,
    pub scattering_lerp: f32,
    pub sun_dir: Vec4,
    pub sun_color: Vec4,
    pub cloud_albedo: Vec4,
    pub camera_translation: Vec3,
    pub time: f32,
    pub reprojection_strength: f32,
    pub render_resolution: Vec2,
    pub frame_index: u32,
    pub history_epoch: u32,
    pub sparse_march: u32,
    pub inverse_camera_view: Mat4,
    pub inverse_camera_projection: Mat4,
    pub wind_displacement: Vec3,
    /// Derived formation-threshold curve (8 piecewise-linear nodes over
    /// strata density, packed x0..w0, x1..w1). See `fill_lut`.
    pub fill_threshold0: Vec4,
    pub fill_threshold1: Vec4,
    /// Derived homogenized-field response `E[shaped | env]` (16 nodes) for
    /// the banded march's coarse LOD. See `fill_lut`.
    pub shape_response0: Vec4,
    pub shape_response1: Vec4,
    pub shape_response2: Vec4,
    pub shape_response3: Vec4,
}

impl Default for CloudsUniform {
    fn default() -> Self {
        Self {
            clouds_raymarch_steps_count: 0,
            clouds_shadow_raymarch_steps_count: 0,
            planet_radius: 0.0,
            clouds_bottom_height: 0.,
            clouds_top_height: 0.,
            clouds_coverage: 0.0,
            clouds_detail_strength: 0.0,
            clouds_base_edge_softness: 0.0,
            clouds_bottom_softness: 0.0,
            clouds_density: 0.0,
            clouds_shadow_raymarch_step_size: 0.0,
            clouds_shadow_raymarch_step_multiply: 0.0,
            forward_scattering_g: 0.0,
            backward_scattering_g: 0.0,
            scattering_lerp: 0.0,
            clouds_ambient_color_top: Vec4::ZERO,
            clouds_ambient_color_bottom: Vec4::ZERO,
            clouds_min_transmittance: 0.0,
            clouds_base_shape_scale_m: 0.0,
            clouds_detail_scale_m: 0.0,
            surface_density_coupling: 1.0,
            sun_dir: Vec4::ZERO,
            sun_color: Vec4::ZERO,
            cloud_albedo: Vec4::ONE,
            camera_translation: Vec3::ZERO,
            time: 0.0,
            reprojection_strength: 0.95,
            render_resolution: Vec2::new(RENDER_WIDTH as f32, RENDER_HEIGHT as f32),
            frame_index: 0,
            history_epoch: 1,
            sparse_march: 1,
            inverse_camera_view: Mat4::IDENTITY,
            inverse_camera_projection: Mat4::IDENTITY,
            wind_displacement: Vec3::new(-11.0, 0.0, 23.0),
            fill_threshold0: Vec4::new(0.81, 0.757, 0.704, 0.651),
            fill_threshold1: Vec4::new(0.599, 0.546, 0.493, 0.44),
            shape_response0: Vec4::new(0.0, 0.03, 0.07, 0.10),
            shape_response1: Vec4::new(0.13, 0.17, 0.20, 0.23),
            shape_response2: Vec4::new(0.27, 0.30, 0.33, 0.37),
            shape_response3: Vec4::new(0.40, 0.43, 0.47, 0.50),
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct CloudsUniformBuffer {
    pub buffer: UniformBuffer<CloudsUniform>,
}

/// Compute-pass image bindings. Public so the game can REBIND the weather /
/// strata cube handles to a body's spawn-uploaded images (handle swap only —
/// runtime re-upload of cube data scrambles face/mip layout on the GPU, see
/// BL-20260723T214730Z). The texture bind group is rebuilt from this resource
/// every frame, so a swap takes effect immediately.
#[derive(Resource, Clone, ExtractResource, AsBindGroup)]
pub struct CloudsImage {
    #[storage_texture(0, image_format = Rgba32Float, access = ReadWrite)]
    pub cloud_render_image: Handle<Image>,

    #[storage_texture(1, image_format = Rgba32Float, access = ReadWrite, dimension = "3d")]
    pub cloud_worley_image: Handle<Image>,

    /// Nearest cloud-hit distance per pixel (metres from the camera; ≥ 1e8
    /// sentinel = no cloud on this ray). The game samples it as a regular
    /// texture in the `body_sky` composite; the raymarch's own history reads
    /// go through `history_distance_image`.
    #[storage_texture(2, image_format = R32Float, access = WriteOnly)]
    pub cloud_distance_image: Handle<Image>,

    /// Planet-fixed cubemap weather field, sampled by body-fixed direction in
    /// the raymarch. Visibility
    /// must be `compute` explicitly — the AsBindGroup default for sampled
    /// textures is vertex|fragment, which fails pipeline validation against
    /// this compute-only pipeline (storage textures default to compute).
    #[texture(3, visibility(compute), dimension = "cube")]
    #[sampler(4, visibility(compute))]
    pub weather_image: Handle<Image>,

    /// Previous frame's render texture, snapshotted by the render node after
    /// each `update` dispatch. Sole source for temporal-history reads (and
    /// the saved camera rows) so the raymarch never races its own writes.
    #[texture(5, visibility(compute), sample_type = "float", filterable = false)]
    pub history_image: Handle<Image>,

    /// Previous frame's nearest cloud-hit distance — the disocclusion test
    /// for motion reprojection.
    #[texture(6, visibility(compute), sample_type = "float", filterable = false)]
    pub history_distance_image: Handle<Image>,

    /// Four body-fixed broad-density strata generated by the canonical weather
    /// authority. The weather sampler at binding 4 is shared deliberately so
    /// both cubes use the same filtered footprint contract.
    #[texture(7, visibility(compute), dimension = "cube")]
    pub surface_density_image: Handle<Image>,
}
