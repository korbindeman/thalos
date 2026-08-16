//! Shared Bevy GPU-pass infrastructure for Thalos render adapters.
//!
//! This crate owns render resources and pass ordering that do not know about a
//! planet, a gameplay camera, or application state. Applications select the
//! participating view with [`SceneDepthView`] and consume the resulting
//! [`SceneDepthImage`].

use bevy::asset::{Assets, Handle, RenderAssetUsages, embedded_asset};
use bevy::camera::Camera;
use bevy::core_pipeline::core_3d::{main_opaque_pass_3d, main_transparent_pass_3d};
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::{
    RenderApp, RenderStartup,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderContext, RenderDevice, ViewQuery},
    texture::GpuImage,
    view::ViewDepthTexture,
};

/// Marks the one 3D view whose opaque depth is made sampleable for downstream
/// render passes.
///
/// The application chooses this view. The foundation neither imports nor
/// infers an application-specific camera type.
#[derive(Component, Clone, Copy, Debug, Default, ExtractComponent)]
pub struct SceneDepthView;

/// Texture usages required on the selected view's [`Camera3d`] depth target.
///
/// `COPY_SRC` supports the single-sample fast path. `TEXTURE_BINDING` supports
/// the MSAA resolve pass, which samples the multisampled depth attachment.
pub fn scene_depth_view_texture_usages() -> TextureUsages {
    TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING
}

/// Sampleable mirror of the selected view's opaque depth attachment.
#[derive(Resource, Clone, ExtractResource)]
pub struct SceneDepthImage {
    pub handle: Handle<Image>,
}

/// Installs the scene-depth image lifecycle and the opaque-to-transparent copy
/// pass for the view carrying [`SceneDepthView`].
pub struct SceneDepthPlugin;

impl Plugin for SceneDepthPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "msaa_depth_resolve.wgsl");
        app.add_plugins(ExtractComponentPlugin::<SceneDepthView>::default())
            .add_systems(Startup, setup_scene_depth_image)
            .add_systems(Update, resize_scene_depth_image);

        if app.get_sub_app(RenderApp).is_some() {
            app.add_plugins(ExtractResourcePlugin::<SceneDepthImage>::default());
        }

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_systems(RenderStartup, init_msaa_depth_resolve)
                .add_systems(
                    Core3d,
                    copy_scene_depth
                        .in_set(Core3dSystems::MainPass)
                        .after(main_opaque_pass_3d)
                        .before(main_transparent_pass_3d),
                );
        }
    }
}

/// Copy the selected view's main-pass depth into [`SceneDepthImage`].
fn copy_scene_depth(
    view: ViewQuery<(&'static ViewDepthTexture, &'static SceneDepthView)>,
    scene_depth: Option<Res<SceneDepthImage>>,
    render_assets: Res<RenderAssets<GpuImage>>,
    msaa_resolve: Option<Res<MsaaDepthResolve>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (depth, _selected_view) = view.into_inner();

    let Some(scene_depth) = scene_depth else {
        return;
    };
    let Some(dest) = render_assets.get(&scene_depth.handle) else {
        return;
    };

    // Resize crosses the main/render-world seam asynchronously. Skip the
    // mismatched frame; the main-world resize system closes the gap.
    let src_size = depth.texture.size();
    let dst_size = dest.texture.size();
    if src_size.width != dst_size.width || src_size.height != dst_size.height {
        return;
    }

    if depth.texture.sample_count() > 1 {
        let Some(resolve) = msaa_resolve else {
            return;
        };
        let Some(pipeline) = pipeline_cache.get_render_pipeline(resolve.pipeline_id) else {
            return;
        };
        let layout = pipeline_cache.get_bind_group_layout(&resolve.layout);
        let bind_group = render_device.create_bind_group(
            Some("msaa_depth_resolve_bind_group"),
            &layout,
            &BindGroupEntries::single(depth.view()),
        );

        let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("msaa_depth_resolve_pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &dest.texture_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
        return;
    }

    if depth.texture.sample_count() != dest.texture.sample_count() {
        return;
    }
    ctx.command_encoder().copy_texture_to_texture(
        depth.texture.as_image_copy(),
        dest.texture.as_image_copy(),
        src_size,
    );
}

/// Render-world pipeline for resolving a multisampled depth attachment into
/// the single-sample [`SceneDepthImage`].
#[derive(Resource)]
struct MsaaDepthResolve {
    layout: BindGroupLayoutDescriptor,
    pipeline_id: CachedRenderPipelineId,
}

fn init_msaa_depth_resolve(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "msaa_depth_resolve_layout",
        &BindGroupLayoutEntries::single(ShaderStages::FRAGMENT, texture_depth_2d_multisampled()),
    );
    let shader = asset_server.load("embedded://thalos_render_foundation/msaa_depth_resolve.wgsl");
    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("msaa_depth_resolve_pipeline".into()),
        layout: vec![layout.clone()],
        immediate_size: 0,
        vertex: VertexState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("vertex".into()),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Always),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            shader,
            shader_defs: vec![],
            entry_point: Some("fragment".into()),
            targets: vec![],
        }),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(MsaaDepthResolve {
        layout,
        pipeline_id,
    });
}

fn setup_scene_depth_image(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_uninit(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Depth32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
    let handle = images.add(image);
    commands.insert_resource(SceneDepthImage { handle });
}

fn resize_scene_depth_image(
    scene_depth: Option<Res<SceneDepthImage>>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<&Camera, With<SceneDepthView>>,
) {
    let Some(scene_depth) = scene_depth else {
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
    let Some(mut image) = images.get_mut(&scene_depth.handle) else {
        return;
    };
    if image.size() != viewport {
        image.resize(Extent3d {
            width: viewport.x,
            height: viewport.y,
            depth_or_array_layers: 1,
        });
    }
}
