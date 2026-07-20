//! Frosted-glass panel rendering.
//!
//! Two halves:
//!
//! - [`GlassMaterial`] — a `UiMaterial` drawing a rounded, hairline-stroked
//!   glass slab. When a backdrop texture is bound it samples a blurred copy of
//!   the scene under the node (true frosted glass); without one it falls back
//!   to a translucent tint, so the same panels work in worlds with no 3D scene
//!   (or in tools that skip the backdrop pass).
//! - [`UiBackdropPlugin`] — the scene-colour source for the frost. A render
//!   system in `Core3d`, ordered after post-processing and before the UI pass,
//!   blits the view target of the camera marked [`UiBackdropSource`] into a
//!   half-resolution [`UiBackdrop`] image. The glass shader then does a
//!   16-tap jittered spiral blur over that image per fragment. Mirrors the
//!   `scene_depth`/`ssao` render-pass-system pattern in the game crate.
//!
//! Only *panels* are glass. Buttons, rows, and fills inside a panel are plain
//! translucent `BackgroundColor` nodes layered on top — one blur per surface,
//! not per control.

use bevy::asset::{embedded_asset, RenderAssetUsages};
use bevy::camera::Camera;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::ecs::prelude::*;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::{
    RenderApp, RenderStartup,
};
use bevy::shader::ShaderRef;
use bevy::render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderContext, RenderDevice, ViewQuery},
    texture::GpuImage,
    view::ViewTarget,
};
use bevy::ui_render::prelude::{UiMaterial, UiMaterialPlugin};
use bevy::ui_render::ui_pass;

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/// Frosted-glass UI material. Attach via
/// `MaterialNode<GlassMaterial>` (usually through
/// [`UiTheme::glass_regular`](crate::UiTheme::glass_regular) — panels share
/// one material asset per style). Corner radius flows in from the node's
/// `border_radius`, so any node shape works.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct GlassMaterial {
    /// Dark tint multiplied over the blurred backdrop (linear RGB; `w` is the
    /// tint opacity over the blur, not the final panel alpha).
    #[uniform(0)]
    pub tint: Vec4,
    /// Hairline edge stroke (linear RGBA).
    #[uniform(0)]
    pub stroke: Vec4,
    /// `x` blur radius (px), `y` frost grain amount, `z` top-sheen amount,
    /// `w` backdrop enable (1 = frosted, 0 = translucent fallback).
    #[uniform(0)]
    pub params: Vec4,
    /// The blurred-scene source ([`UiBackdrop::handle`]); `None` (or
    /// `params.w = 0`) renders the translucent fallback.
    #[texture(1)]
    #[sampler(2)]
    pub backdrop: Option<Handle<Image>>,
}

impl UiMaterial for GlassMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_ui/glass.wgsl".into()
    }
}

impl GlassMaterial {
    /// Build a glass material from a tint token, wired to the backdrop if one
    /// exists.
    pub fn new(tint: Vec4, backdrop: Option<Handle<Image>>) -> Self {
        let stroke = LinearRgba::from(crate::tokens::STROKE);
        let enabled = if backdrop.is_some() { 1.0 } else { 0.0 };
        Self {
            tint,
            stroke: Vec4::new(stroke.red, stroke.green, stroke.blue, stroke.alpha),
            params: Vec4::new(34.0, 0.012, 1.0, enabled),
            backdrop,
        }
    }
}

// ---------------------------------------------------------------------------
// Backdrop source
// ---------------------------------------------------------------------------

/// Marker for the camera whose output feeds the frost. Add it to whichever
/// camera renders the scene the UI floats over (the game marks the ship
/// camera and the shipyard editor camera; inactive cameras are ignored, so
/// marking several is fine as long as one renders at a time).
#[derive(Component, Clone, Copy, Default, ExtractComponent)]
pub struct UiBackdropSource;

/// Handle to the half-resolution scene-colour copy the glass shader blurs.
/// Created at startup (1×1 until a marked camera reports a viewport).
///
/// **Sole writer:** [`copy_ui_backdrop`] (render world);
/// resized by [`resize_ui_backdrop`].
#[derive(Resource, Clone, ExtractResource)]
pub struct UiBackdrop {
    pub handle: Handle<Image>,
}

pub struct UiBackdropPlugin;

impl Plugin for UiBackdropPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "backdrop_blit.wgsl");
        app.add_plugins((
            ExtractResourcePlugin::<UiBackdrop>::default(),
            ExtractComponentPlugin::<UiBackdropSource>::default(),
        ))
        .add_systems(PreStartup, setup_ui_backdrop)
        .add_systems(Update, resize_ui_backdrop);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_systems(RenderStartup, init_backdrop_pipeline)
                .add_systems(
                    Core3d,
                    // After post-processing (so the copy is the tonemapped
                    // scene), before the UI pass reads it through the glass.
                    copy_ui_backdrop
                        .after(Core3dSystems::PostProcess)
                        .before(ui_pass),
                );
        }
    }
}

/// Create the placeholder backdrop image. `PreStartup` so `Startup` theme
/// setup (which binds the handle into the shared glass materials) sees it.
fn setup_ui_backdrop(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[8, 10, 12, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
    let handle = images.add(image);
    commands.insert_resource(UiBackdrop { handle });
}

/// Track the marked camera's target at half resolution.
fn resize_ui_backdrop(
    backdrop: Option<Res<UiBackdrop>>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<&Camera, With<UiBackdropSource>>,
) {
    let Some(backdrop) = backdrop else {
        return;
    };
    let Some(size) = cameras
        .iter()
        .find(|c| c.is_active)
        .and_then(|c| c.physical_target_size())
    else {
        return;
    };
    let half = UVec2::new((size.x / 2).max(1), (size.y / 2).max(1));
    if half.x == 0 || half.y == 0 {
        return;
    }
    let Some(mut image) = images.get_mut(&backdrop.handle) else {
        return;
    };
    if image.size() != half {
        image.resize(Extent3d {
            width: half.x,
            height: half.y,
            depth_or_array_layers: 1,
        });
    }
}

// ---------------------------------------------------------------------------
// Render-world blit
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct BackdropBlitPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline_id: CachedRenderPipelineId,
    sampler: Sampler,
}

fn init_backdrop_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "ui_backdrop_blit_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );
    let shader: Handle<Shader> = asset_server.load("embedded://thalos_ui/backdrop_blit.wgsl");
    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("ui_backdrop_blit_pipeline".into()),
        layout: vec![layout.clone()],
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
                format: TextureFormat::Rgba8UnormSrgb,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    });
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("ui_backdrop_blit_sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..Default::default()
    });
    commands.insert_resource(BackdropBlitPipeline {
        layout,
        pipeline_id,
        sampler,
    });
}

/// Blit the marked view's post-processed colour into [`UiBackdrop`].
/// `ViewTarget::main_texture_view` is always the single-sample ping-pong
/// texture (MSAA resolves into it), so no resolve special-casing is needed.
fn copy_ui_backdrop(
    view: ViewQuery<(&'static ViewTarget, &'static UiBackdropSource)>,
    backdrop: Option<Res<UiBackdrop>>,
    render_assets: Res<RenderAssets<GpuImage>>,
    pipeline: Option<Res<BackdropBlitPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (target, _marker) = view.into_inner();
    let (Some(backdrop), Some(pipeline)) = (backdrop, pipeline) else {
        return;
    };
    let Some(dest) = render_assets.get(&backdrop.handle) else {
        return;
    };
    let Some(blit) = pipeline_cache.get_render_pipeline(pipeline.pipeline_id) else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    let bind_group = render_device.create_bind_group(
        Some("ui_backdrop_blit_bind_group"),
        &layout,
        &BindGroupEntries::sequential((target.main_texture_view(), &pipeline.sampler)),
    );
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("ui_backdrop_blit_pass"),
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
    pass.set_render_pipeline(blit);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);
}

// ---------------------------------------------------------------------------
// Material plugin glue
// ---------------------------------------------------------------------------

pub struct GlassPlugin;

impl Plugin for GlassPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "glass.wgsl");
        app.add_plugins((UiMaterialPlugin::<GlassMaterial>::default(), UiBackdropPlugin));
    }
}
