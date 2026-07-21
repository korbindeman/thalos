//! Screen-space ambient occlusion (graphics-fidelity F5).
//!
//! A half-resolution hemisphere SSAO pass computed from [`SceneDepthImage`]
//! (`rendering::scene_depth`) — the depth copy that, unlike Bevy's depth prepass,
//! **sees the forked-udlod terrain**. The result ([`AoImage`]) is sampled one
//! frame later by the terrain material and multiplied into its *ambient*
//! occlusion only, so a ship parked in a valley, grass at a tree base, or a
//! building corner darkens where geometry crowds the sky — one-world invariant #4.
//!
//! Why a custom pass rather than Bevy's `ScreenSpaceAmbientOcclusion`: the fork
//! doesn't queue terrain into `Opaque3dPrepass`, so Bevy's GTAO (which reads the
//! prepass) would be terrain-blind. `SceneDepthImage` is the only depth that sees
//! the dominant surface, and this pass mirrors the `CopySceneDepthNode` pattern
//! that already consumes it.
//!
//! The whole pass runs in **view space** (camera-relative render metres), so it
//! is f32-safe under big_space's floating origin. The AO texture is written after
//! the opaque pass (once the depth copy is populated) and sampled by the terrain
//! the next frame — a 1-frame latency invisible at planet-cam speeds.

use crate::camera::ShipCamera;
use crate::rendering::scene_depth::SceneDepthImage;
use bevy::asset::{Assets, Handle, RenderAssetUsages};
use bevy::camera::Camera;
use bevy::core_pipeline::core_3d::main_transparent_pass_3d;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::ecs::prelude::*;
use bevy::image::Image;
use bevy::math::{Mat4, Vec4};
use bevy::prelude::*;
use bevy::render::{
    RenderApp, RenderStartup,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::{binding_types::*, encase, *},
    renderer::{RenderContext, RenderDevice, ViewQuery},
    texture::GpuImage,
    view::ExtractedView,
};

/// AO tuning, edited live-ish (extracted to the render world each frame). Defaults
/// are the calibration anchor; the whole set wants a `just game runway` / `landing`
/// screenshot — SSAO is verified by eye (no haloing, no acne, right contact darkness).
#[derive(Resource, Clone, Copy, ExtractResource)]
pub struct SsaoConfig {
    pub enabled: bool,
    /// Diagnostic: paint the raw AO value on the terrain instead of shading
    /// (`THALOS_SSAO=show`). Splits "the AO pass produces artifacts" from "the
    /// terrain samples/applies it wrong" in one screenshot.
    pub debug_show: bool,
    /// Sample radius in view/render units (≈ metres near the camera). Small, per
    /// the "small world radius" rule — contact AO, not large-scale GI.
    pub radius: f32,
    /// Depth bias (view units) to avoid self-occlusion acne on flat ground.
    pub bias: f32,
    /// Occlusion strength before the contrast power.
    pub intensity: f32,
    /// Contrast power on the final visibility (higher = punchier creases).
    pub power: f32,
}

impl Default for SsaoConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            debug_show: false,
            radius: 1.0,
            bias: 0.025,
            intensity: 1.2,
            power: 1.5,
        }
    }
}

impl SsaoConfig {
    /// Read the `THALOS_SSAO` session override: `off`/`0`/`false` disables the
    /// pass entirely (terrain skips sampling, node skips rendering), `show`
    /// paints raw AO on the terrain (diagnostic), anything else / unset = on.
    fn from_env() -> Self {
        let mode = std::env::var("THALOS_SSAO")
            .unwrap_or_default()
            .to_ascii_lowercase();
        match mode.as_str() {
            "off" | "0" | "false" => Self {
                enabled: false,
                ..Default::default()
            },
            "show" | "debug" => Self {
                debug_show: true,
                ..Default::default()
            },
            _ => Self::default(),
        }
    }

    /// The terrain-material gate value carried in `inspection.w`:
    /// 0 = skip AO, 1 = apply AO, 2 = paint raw AO (debug).
    pub fn terrain_flag(&self) -> f32 {
        if !self.enabled {
            0.0
        } else if self.debug_show {
            2.0
        } else {
            1.0
        }
    }
}

/// Half-resolution **resolved** AO target (**R16Float**, 1 = unoccluded): the
/// depth-aware-blurred output of [`AoRawImage`], and the image the terrain
/// samples. White until the first pass runs, so it never over-darkens.
///
/// R16Float, not R8Unorm: 8-bit posterizes the smooth AO gradient across a grazing
/// ground plane into visible horizontal bands (the receding floor's AO varies
/// smoothly with screen-Y). f16 has the headroom to keep it smooth.
#[derive(Resource, Clone, ExtractResource)]
pub struct AoImage {
    pub handle: Handle<Image>,
}

/// Half-resolution **raw** AO target (RG16Float — R = noisy visibility from the
/// IGN-dithered kernel, G = view-space distance for the blur's depth-similarity
/// weights). Internal: written by the SSAO pass, consumed only by the blur.
#[derive(Resource, Clone, ExtractResource)]
pub struct AoRawImage {
    pub handle: Handle<Image>,
}

/// Resolved-AO texture format. R16Float is renderable + filterable with no extra
/// wgpu features, and has the precision R8Unorm lacks (see [`AoImage`]).
const AO_FORMAT: TextureFormat = TextureFormat::R16Float;
/// Raw-AO format: visibility + view distance for the depth-aware blur.
const AO_RAW_FORMAT: TextureFormat = TextureFormat::Rg16Float;

/// Std140 uniform for `ssao.wgsl`. Field order + widths must match the shader.
#[derive(ShaderType)]
struct AoUniform {
    view_from_clip: Mat4,
    clip_from_view: Mat4,
    /// AO target size in px (xy); zw padding.
    target_res: Vec4,
    /// x = radius, y = bias, z = intensity, w = power.
    params: Vec4,
}

pub struct SsaoPlugin;

impl Plugin for SsaoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SsaoConfig::from_env())
            .add_plugins(ExtractResourcePlugin::<SsaoConfig>::default())
            .add_plugins(ExtractResourcePlugin::<AoImage>::default())
            .add_plugins(ExtractResourcePlugin::<AoRawImage>::default())
            .add_systems(Startup, setup_ao_image)
            .add_systems(Update, resize_ao_image);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_systems(RenderStartup, init_ssao_pipeline)
                .add_systems(
                    Core3d,
                    // After the transparent pass → strictly after `copy_scene_depth`
                    // (which runs before it), so `SceneDepthImage` is populated. The
                    // blur resolves the raw pass's dither noise before the terrain
                    // samples the result next frame.
                    (compute_ssao, blur_ssao)
                        .chain()
                        .in_set(Core3dSystems::MainPass)
                        .after(main_transparent_pass_3d),
                );
        }
    }
}

/// 1×1 white placeholders; `resize_ao_image` grows both to half the viewport
/// once the camera reports a size.
fn setup_ao_image(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut build = |data: Vec<u8>, format: TextureFormat| {
        let mut image = Image::new(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data,
            format,
            RenderAssetUsages::RENDER_WORLD,
        );
        image.texture_descriptor.usage =
            TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
        images.add(image)
    };
    // f16 1.0 (LE) = white = fully unoccluded; raw G (distance) = 0.
    let resolved = build(vec![0x00, 0x3C], AO_FORMAT);
    let raw = build(vec![0x00, 0x3C, 0x00, 0x00], AO_RAW_FORMAT);
    commands.insert_resource(AoImage { handle: resolved });
    commands.insert_resource(AoRawImage { handle: raw });
}

/// Keep both AO targets at half the ship camera's viewport (rounded up, min 1).
fn resize_ao_image(
    ao_image: Option<Res<AoImage>>,
    ao_raw: Option<Res<AoRawImage>>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<&Camera, With<ShipCamera>>,
) {
    let (Some(ao_image), Some(ao_raw)) = (ao_image, ao_raw) else {
        return;
    };
    let Ok(camera) = cameras.single() else {
        return;
    };
    let Some(viewport) = camera.physical_viewport_size() else {
        return;
    };
    if viewport.x == 0 || viewport.y == 0 {
        return;
    }
    let half = Extent3d {
        width: viewport.x.div_ceil(2).max(1),
        height: viewport.y.div_ceil(2).max(1),
        depth_or_array_layers: 1,
    };
    for handle in [&ao_image.handle, &ao_raw.handle] {
        let Some(mut image) = images.get_mut(handle) else {
            continue;
        };
        if image.texture_descriptor.size != half {
            image.resize(half);
        }
    }
}

/// Render-world pipelines for the SSAO + blur fullscreen passes.
#[derive(Resource)]
struct SsaoPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline_id: CachedRenderPipelineId,
    blur_layout: BindGroupLayoutDescriptor,
    blur_pipeline_id: CachedRenderPipelineId,
}

fn init_ssao_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let fullscreen = |shader: Handle<Shader>,
                      label: &'static str,
                      layout: BindGroupLayoutDescriptor,
                      format: TextureFormat| {
        RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout],
            immediate_size: 0,
            vertex: VertexState {
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some("vertex".into()),
                buffers: vec![],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader,
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            zero_initialize_workgroup_memory: false,
        }
    };

    let layout = BindGroupLayoutDescriptor::new(
        "ssao_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (texture_depth_2d(), uniform_buffer::<AoUniform>(false)),
        ),
    );
    let pipeline_id = pipeline_cache.queue_render_pipeline(fullscreen(
        asset_server.load("shaders/ssao.wgsl"),
        "ssao_pipeline",
        layout.clone(),
        AO_RAW_FORMAT,
    ));

    let blur_layout = BindGroupLayoutDescriptor::new(
        "ssao_blur_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::FRAGMENT,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),
    );
    let blur_pipeline_id = pipeline_cache.queue_render_pipeline(fullscreen(
        asset_server.load("shaders/ssao_blur.wgsl"),
        "ssao_blur_pipeline",
        blur_layout.clone(),
        AO_FORMAT,
    ));

    commands.insert_resource(SsaoPipeline {
        layout,
        pipeline_id,
        blur_layout,
        blur_pipeline_id,
    });
}

/// Compute raw AO into [`AoRawImage`] from the ship view's projection +
/// [`SceneDepthImage`]. Runs once for the ship-camera view (the `ViewQuery`
/// filters via the extracted [`ShipCamera`] marker), after the depth copy is
/// populated; [`blur_ssao`] then resolves it into [`AoImage`].
fn compute_ssao(
    view: ViewQuery<(&'static ExtractedView, &'static ShipCamera)>,
    config: Res<SsaoConfig>,
    ao_raw: Option<Res<AoRawImage>>,
    scene_depth: Option<Res<SceneDepthImage>>,
    pipeline: Option<Res<SsaoPipeline>>,
    render_assets: Res<RenderAssets<GpuImage>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    if !config.enabled {
        return;
    }
    let (extracted, _ship) = view.into_inner();
    let (Some(ao_raw), Some(scene_depth), Some(pipeline)) = (ao_raw, scene_depth, pipeline) else {
        return;
    };
    let (Some(dest), Some(depth)) = (
        render_assets.get(&ao_raw.handle),
        render_assets.get(&scene_depth.handle),
    ) else {
        return;
    };
    let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipeline.pipeline_id) else {
        return;
    };

    let clip_from_view = extracted.clip_from_view;
    let size = dest.texture.size();
    let uniform = AoUniform {
        view_from_clip: clip_from_view.inverse(),
        clip_from_view,
        target_res: Vec4::new(size.width as f32, size.height as f32, 0.0, 0.0),
        params: Vec4::new(config.radius, config.bias, config.intensity, config.power),
    };
    let mut scratch = encase::UniformBuffer::new(Vec::<u8>::new());
    // Infallible for an in-memory Vec sink.
    let _ = scratch.write(&uniform);
    let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("ssao_uniform"),
        usage: BufferUsages::UNIFORM,
        contents: scratch.as_ref(),
    });

    let bind_group = render_device.create_bind_group(
        Some("ssao_bind_group"),
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((&depth.texture_view, buffer.as_entire_binding())),
    );

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("ssao_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &dest.texture_view,
            depth_slice: None,
            resolve_target: None,
            // Fullscreen triangle overwrites every pixel; default (load) is fine.
            ops: Operations::default(),
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_render_pipeline(render_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);
}

/// Depth-aware blur: resolve [`AoRawImage`]'s dither noise into [`AoImage`]
/// (the image the terrain samples). See `ssao_blur.wgsl`.
fn blur_ssao(
    view: ViewQuery<&'static ShipCamera>,
    config: Res<SsaoConfig>,
    ao_raw: Option<Res<AoRawImage>>,
    ao_image: Option<Res<AoImage>>,
    pipeline: Option<Res<SsaoPipeline>>,
    render_assets: Res<RenderAssets<GpuImage>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    if !config.enabled {
        return;
    }
    let _ship = view.into_inner();
    let (Some(ao_raw), Some(ao_image), Some(pipeline)) = (ao_raw, ao_image, pipeline) else {
        return;
    };
    let (Some(src), Some(dest)) = (
        render_assets.get(&ao_raw.handle),
        render_assets.get(&ao_image.handle),
    ) else {
        return;
    };
    let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipeline.blur_pipeline_id)
    else {
        return;
    };
    // Resize is async across two images: skip frames where they disagree.
    if src.texture.size() != dest.texture.size() {
        return;
    }

    let bind_group = render_device.create_bind_group(
        Some("ssao_blur_bind_group"),
        &pipeline_cache.get_bind_group_layout(&pipeline.blur_layout),
        &BindGroupEntries::single(&src.texture_view),
    );

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("ssao_blur_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &dest.texture_view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations::default(),
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_render_pipeline(render_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);
}
