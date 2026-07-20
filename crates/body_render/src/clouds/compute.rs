use bevy::{
    asset::load_embedded_asset,
    ecs::system::ResMut,
    prelude::*,
    render::{
        Extract,
        Render,
        RenderApp,
        RenderSystems,
        diagnostic::{RecordDiagnostics, begin_diagnostics_frame},
        extract_resource::ExtractResourcePlugin,
        render_asset::RenderAssets,
        render_resource::{
            AsBindGroup, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor,
            BindGroupLayoutEntries, CachedComputePipelineId, CachedPipelineState,
            ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages,
            binding_types::uniform_buffer,
        },
        // Bevy 0.19 replaced the node-based render graph with systems scheduled
        // in the root `RenderGraph` schedule (see `renderer::RenderGraphSystems`).
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
        texture::GpuImage,
    },
};
/// Controls the compute shader which renders the volumetric clouds.
use std::borrow::Cow;

use super::config::CloudsConfig;

use super::{
    images::{IMAGE_SIZE, RENDER_HEIGHT, RENDER_WIDTH},
    uniforms::{CloudsImage, CloudsUniform, CloudsUniformBuffer},
};

const WORKGROUP_SIZE: u32 = 8;
/// Camera basis fed to the cloud raymarch, in the **body-fixed frame** of the
/// active cloud body: `translation` is the camera position relative to the
/// planet centre, rotated into body-fixed coordinates (so it co-rotates with
/// the surface), and `inverse_camera_view` is `body_from_world ×
/// world_from_view`, so view rays emerge directly in body-fixed space. Thalos
/// drives this each frame from the `ShipCamera`, rather than the upstream
/// `Single<Camera>` system — see `CloudsPlugin` docs.
#[derive(Resource, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub struct CameraMatrices {
    pub translation: Vec3,
    pub inverse_camera_view: Mat4,
    pub inverse_camera_projection: Mat4,
}

#[derive(Resource)]
struct CloudsUniformBindGroup(BindGroup);

#[derive(Resource)]
struct CloudsImageBindGroup(BindGroup);

#[expect(clippy::too_many_arguments)]
fn prepare_uniforms_bind_group(
    mut commands: Commands,
    pipeline: Res<CloudsPipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_queue: Res<RenderQueue>,
    mut clouds_uniform_buffer: ResMut<CloudsUniformBuffer>,
    camera: ResMut<CameraMatrices>,
    clouds_config: Res<CloudsConfig>,
    render_device: Res<RenderDevice>,
    time: Res<Time>,
) {
    let buffer = clouds_uniform_buffer.buffer.get_mut();

    buffer.clouds_raymarch_steps_count = clouds_config.clouds_raymarch_steps_count;
    buffer.planet_radius = clouds_config.planet_radius;
    buffer.clouds_bottom_height = clouds_config.clouds_bottom_height;
    buffer.clouds_top_height = clouds_config.clouds_top_height;
    buffer.clouds_coverage = clouds_config.clouds_coverage;
    buffer.clouds_detail_strength = clouds_config.clouds_detail_strength;
    buffer.clouds_base_edge_softness = clouds_config.clouds_base_edge_softness;
    buffer.clouds_bottom_softness = clouds_config.clouds_bottom_softness;
    buffer.clouds_density = clouds_config.clouds_density;
    buffer.clouds_shadow_raymarch_steps_count = clouds_config.clouds_shadow_raymarch_steps_count;
    buffer.clouds_shadow_raymarch_step_size = clouds_config.clouds_shadow_raymarch_step_size;
    buffer.clouds_shadow_raymarch_step_multiply =
        clouds_config.clouds_shadow_raymarch_step_multiply;
    buffer.forward_scattering_g = clouds_config.forward_scattering_g;
    buffer.backward_scattering_g = clouds_config.backward_scattering_g;
    buffer.scattering_lerp = clouds_config.scattering_lerp;
    buffer.clouds_ambient_color_top = clouds_config.clouds_ambient_color_top;
    buffer.clouds_ambient_color_bottom = clouds_config.clouds_ambient_color_bottom;
    buffer.clouds_min_transmittance = clouds_config.clouds_min_transmittance;
    buffer.clouds_base_scale = clouds_config.clouds_base_scale;
    buffer.clouds_detail_scale = clouds_config.clouds_detail_scale;
    buffer.sun_dir = clouds_config.sun_dir;
    buffer.sun_color = clouds_config.sun_color;
    buffer.camera_translation = camera.translation;
    buffer.time = time.elapsed_secs_wrapped();
    buffer.reprojection_strength = clouds_config.reprojection_strength;
    buffer.inverse_camera_view = camera.inverse_camera_view;
    buffer.inverse_camera_projection = camera.inverse_camera_projection;
    buffer.wind_displacement += time.delta_secs() * clouds_config.wind_velocity;

    clouds_uniform_buffer
        .buffer
        .write_buffer(&render_device, &render_queue);

    let bind_group_uniforms = render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipeline.uniform_bind_group_layout),
        &BindGroupEntries::single(clouds_uniform_buffer.buffer.binding().unwrap().clone()),
    );
    commands.insert_resource(CloudsUniformBindGroup(bind_group_uniforms));
}

fn prepare_textures_bind_group(
    mut commands: Commands,
    pipeline: Res<CloudsPipeline>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    clouds_image: Res<CloudsImage>,
    render_device: Res<RenderDevice>,
) {
    let cloud_render_view = gpu_images.get(&clouds_image.cloud_render_image).unwrap();
    let cloud_atlas_view = gpu_images.get(&clouds_image.cloud_atlas_image).unwrap();
    let cloud_worley_view = gpu_images.get(&clouds_image.cloud_worley_image).unwrap();
    let cloud_distance_view = gpu_images.get(&clouds_image.cloud_distance_image).unwrap();
    let weather_view = gpu_images.get(&clouds_image.weather_image).unwrap();
    let history_view = gpu_images.get(&clouds_image.history_image).unwrap();
    let history_distance_view = gpu_images
        .get(&clouds_image.history_distance_image)
        .unwrap();

    let bind_group = render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipeline.texture_bind_group_layout),
        &BindGroupEntries::sequential((
            &cloud_render_view.texture_view,
            &cloud_atlas_view.texture_view,
            &cloud_worley_view.texture_view,
            &cloud_distance_view.texture_view,
            &weather_view.texture_view,
            &weather_view.sampler,
            &history_view.texture_view,
            &history_distance_view.texture_view,
        )),
    );
    commands.insert_resource(CloudsImageBindGroup(bind_group));
}

/// The compute shading pipeline
///
/// Note that the compute shader is loaded in [`CloudsShaderPlugin`] so this resource depends on
/// that plugin.
#[derive(Resource)]
struct CloudsPipeline {
    texture_bind_group_layout: BindGroupLayoutDescriptor,
    uniform_bind_group_layout: BindGroupLayoutDescriptor,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

impl FromWorld for CloudsPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let texture_bind_group_layout = CloudsImage::bind_group_layout_descriptor(render_device);
        let shader = load_embedded_asset!(world, "shaders/clouds_compute.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();

        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (uniform_buffer::<CloudsUniform>(false),),
        );

        let uniform_bind_group_layout =
            BindGroupLayoutDescriptor::new("uniform_bind_group_layout", &entries);

        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            zero_initialize_workgroup_memory: false,
            label: None,
            layout: vec![
                uniform_bind_group_layout.clone(),
                texture_bind_group_layout.clone(),
            ],
            // 0.19 replaced `push_constant_ranges` with `immediate_size: u32`.
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("init")),
        });
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            zero_initialize_workgroup_memory: false,
            label: None,
            layout: vec![
                uniform_bind_group_layout.clone(),
                texture_bind_group_layout.clone(),
            ],
            // 0.19 replaced `push_constant_ranges` with `immediate_size: u32`.
            immediate_size: 0,
            shader,
            shader_defs: vec![],
            entry_point: Some(Cow::from("update")),
        });

        CloudsPipeline {
            texture_bind_group_layout,
            uniform_bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

#[derive(Default, Clone, Copy)]
enum CloudsState {
    #[default]
    Loading,
    Init,
    Update,
}

/// The cloud compute dispatch, ported from the former `CloudsNode`
/// (`render_graph::Node`) to a Bevy 0.19 render-graph **system**.
///
/// Scheduled in `RenderGraphSystems::Begin`, which is chained before
/// `RenderGraphSystems::Render` (where `camera_driver` records the per-view
/// passes that sample the cloud textures) — preserving the old
/// `add_node_edge(CloudsLabel, CameraDriverLabel)` ordering so the sky pass
/// reads this frame's clouds. The node's two phases collapse into one system:
/// the `Local<CloudsState>` carries the Loading→Init→Update transition that
/// used to live in `Node::update`, and `RenderContext` (a SystemParam in 0.19)
/// supplies the command encoder.
#[expect(clippy::too_many_arguments)]
fn run_clouds_compute(
    mut ctx: RenderContext,
    mut state: Local<CloudsState>,
    pipeline: Res<CloudsPipeline>,
    pipeline_cache: Res<PipelineCache>,
    texture_bind_group: Option<Res<CloudsImageBindGroup>>,
    uniform_bind_group: Option<Res<CloudsUniformBindGroup>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    clouds_image: Option<Res<CloudsImage>>,
) {
    // Advance the pipeline-load state machine (was `Node::update`).
    match *state {
        CloudsState::Loading => {
            if let CachedPipelineState::Ok(_) =
                pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline)
            {
                *state = CloudsState::Init;
            }
        }
        CloudsState::Init => {
            if let CachedPipelineState::Ok(_) =
                pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
            {
                *state = CloudsState::Update;
            }
        }
        CloudsState::Update => {}
    }

    // The bind groups are inserted by the prepare systems each frame; bail if
    // they (or the cloud images) aren't ready yet.
    let (Some(texture_bind_group), Some(uniform_bind_group), Some(clouds_image)) =
        (texture_bind_group, uniform_bind_group, clouds_image)
    else {
        return;
    };

    {
        let diagnostics = ctx.diagnostic_recorder();
        let diagnostics = diagnostics.as_deref();
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        let pass_name = match *state {
            CloudsState::Init => "volumetric_clouds_init",
            CloudsState::Update => "volumetric_clouds",
            CloudsState::Loading => "volumetric_clouds_loading",
        };
        let pass_span = diagnostics.pass_span(&mut pass, pass_name);

        pass.set_bind_group(0, &uniform_bind_group.0, &[]);
        pass.set_bind_group(1, &texture_bind_group.0, &[]);

        match *state {
            CloudsState::Loading => {}
            CloudsState::Init => {
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(
                    IMAGE_SIZE / WORKGROUP_SIZE,
                    IMAGE_SIZE / WORKGROUP_SIZE,
                    1,
                );
            }
            CloudsState::Update => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(
                    RENDER_WIDTH / WORKGROUP_SIZE,
                    RENDER_HEIGHT / WORKGROUP_SIZE,
                    1,
                );
            }
        }
        pass_span.end(&mut pass);
    }

    // Snapshot this frame's output into the history textures the next
    // frame reads (same-pixel accumulation, motion reprojection, the
    // saved camera rows). A separate history copy keeps the raymarch from
    // ever reading the texture it is writing — in-pass history reads race
    // across workgroups and showed as coherent streak artifacts.
    if matches!(*state, CloudsState::Update) {
        let pairs = [
            (
                &clouds_image.cloud_render_image,
                &clouds_image.history_image,
            ),
            (
                &clouds_image.cloud_distance_image,
                &clouds_image.history_distance_image,
            ),
        ];
        for (src, dst) in pairs {
            let (Some(src), Some(dst)) = (gpu_images.get(src), gpu_images.get(dst)) else {
                continue;
            };
            ctx.command_encoder().copy_texture_to_texture(
                src.texture.as_image_copy(),
                dst.texture.as_image_copy(),
                src.texture.size(),
            );
        }
    }
}

/// A plugin for the compute shader which renders clouds.
pub(crate) struct CloudsComputePlugin;

impl Plugin for CloudsComputePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<CloudsImage>::default());
        app.add_plugins(ExtractResourcePlugin::<CloudsUniform>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_textures_bind_group.in_set(RenderSystems::PrepareResources),
        );
        render_app.add_systems(
            Render,
            prepare_uniforms_bind_group.in_set(RenderSystems::PrepareResources),
        );

        // Bevy 0.19: the former `CloudsNode` is now a system in the root
        // `RenderGraph` schedule. `Begin` is chained before `Render` (where
        // `camera_driver` runs), so the dispatch records before the per-view
        // sky pass that samples the cloud textures — matching the old
        // `add_node_edge(CloudsLabel, CameraDriverLabel)`.
        render_app.add_systems(
            RenderGraph,
            run_clouds_compute
                .in_set(RenderGraphSystems::Begin)
                .after(begin_diagnostics_frame),
        );

        render_app.add_systems(
            ExtractSchedule,
            (extract_clouds_config, extract_time, extract_camera_matrices),
        );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<CloudsPipeline>();
        render_app.init_resource::<CloudsUniformBuffer>();
    }
}

fn extract_clouds_config(mut commands: Commands, config: Extract<Res<CloudsConfig>>) {
    commands.insert_resource(**config);
}

fn extract_time(mut commands: Commands, time: Extract<Res<Time>>) {
    commands.insert_resource(**time);
}

fn extract_camera_matrices(mut commands: Commands, camera: Extract<Res<CameraMatrices>>) {
    commands.insert_resource(**camera);
}
