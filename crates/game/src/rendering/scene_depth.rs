//! Scene-depth copy for the unified atmosphere pass.
//!
//! The atmosphere fullscreen pass needs to sample the depth of opaque
//! geometry (terrain, impostor body, ship hull) so the in-scatter raymarch
//! can clip at the surface and produce aerial-perspective continuity from
//! sky to ground. WebGPU disallows sampling the live depth attachment from
//! a fragment shader, and our forked `thalos_udlod` does not queue into
//! `Opaque3dPrepass`, so the standard prepass-depth texture is terrain-blind.
//!
//! Workaround: insert a render-graph node between [`Node3d::MainOpaquePass`]
//! and [`Node3d::MainTransparentPass`] that `copy_texture_to_texture`s the
//! main pass's `ViewDepthTexture` into an [`Image`] we own. Materials bind
//! that `Image` via `AsBindGroup` and sample it as `texture_depth_2d` from
//! WGSL. One extra full-screen depth copy per frame; trivial cost.
//!
//! When the ship camera runs MSAA the main-pass depth is multisampled and a
//! single-sample `copy_texture_to_texture` is illegal, so this node instead
//! runs a tiny depth-only resolve pass ([`MsaaDepthResolve`]) that writes
//! sample 0 of the multisampled depth straight into the destination Image. The
//! atmosphere pass keeps sampling it as `texture_depth_2d` either way. The
//! MSAA-off path stays the cheap direct copy.

use crate::camera::ShipCamera;
use bevy::asset::{Assets, Handle, RenderAssetUsages};
use bevy::camera::Camera;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::prelude::*;
use bevy::ecs::query::QueryItem;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::{
    RenderApp, RenderStartup,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_graph::{
        NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
    },
    render_resource::{binding_types::*, *},
    renderer::RenderContext,
    texture::GpuImage,
    view::ViewDepthTexture,
};

// `ShipCamera` is extracted to the render world by
// `ExtractComponentPlugin::<ShipCamera>` (added in `CameraPlugin::build`)
// so the `ViewQuery` below can filter to that view.

/// Handle to the Image that mirrors the main pass's depth attachment. Bound
/// on materials that need to sample scene depth. Updated each frame by
/// [`CopySceneDepthNode`] and resized by [`resize_scene_depth_image`].
#[derive(Resource, Clone, ExtractResource)]
pub struct SceneDepthImage {
    pub handle: Handle<Image>,
}

#[derive(RenderLabel, Hash, PartialEq, Eq, Debug, Clone)]
struct CopySceneDepth;

#[derive(Default)]
struct CopySceneDepthNode;

impl ViewNode for CopySceneDepthNode {
    // Filter to the ship-camera view via the extracted `ShipCamera`
    // marker. Other render-world views (map camera, future light /
    // shadow views, picking sub-cameras) lack the marker and the node
    // doesn't fire for them — important because they may have different
    // depth formats / sample counts / usage flags.
    type ViewQuery = (&'static ViewDepthTexture, &'static ShipCamera);

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (depth, _ship): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(scene_depth) = world.get_resource::<SceneDepthImage>() else {
            return Ok(());
        };
        let render_assets = world.resource::<RenderAssets<GpuImage>>();
        let Some(dest) = render_assets.get(&scene_depth.handle) else {
            return Ok(());
        };

        // Resize is async: skip frames where source and destination disagree
        // on size. The next frame's `resize_scene_depth_image` call (or the
        // following one) will close the gap.
        let src_size = depth.texture.size();
        let dst_size = dest.texture.size();
        if src_size.width != dst_size.width || src_size.height != dst_size.height {
            return Ok(());
        }

        // MSAA path: the source depth is multisampled, so a single-sample copy
        // is illegal. Resolve sample 0 into the destination with a depth-only
        // fullscreen pass instead. Skipped silently until the resolve pipeline
        // has compiled (the next frame closes the gap).
        if depth.texture.sample_count() > 1 {
            let Some(resolve) = world.get_resource::<MsaaDepthResolve>() else {
                return Ok(());
            };
            let pipeline_cache = world.resource::<PipelineCache>();
            let Some(pipeline) = pipeline_cache.get_render_pipeline(resolve.pipeline_id) else {
                return Ok(());
            };
            let layout = pipeline_cache.get_bind_group_layout(&resolve.layout);
            let bind_group = render_context.render_device().create_bind_group(
                Some("msaa_depth_resolve_bind_group"),
                &layout,
                &BindGroupEntries::single(depth.view()),
            );

            let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("msaa_depth_resolve_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &dest.texture_view,
                    // The fullscreen triangle writes every pixel with
                    // `depth_compare: Always`, so the clear value is overwritten
                    // everywhere — only `store` matters.
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
            return Ok(());
        }

        // MSAA-off path: both single-sample, so a straight copy is cheapest.
        // (Other render-world views — shadow cascades, light views — never
        // reach here; the node is filtered to the ship camera.)
        if depth.texture.sample_count() != dest.texture.sample_count() {
            return Ok(());
        }
        render_context.command_encoder().copy_texture_to_texture(
            depth.texture.as_image_copy(),
            dest.texture.as_image_copy(),
            src_size,
        );
        Ok(())
    }
}

pub struct SceneDepthPlugin;

impl Plugin for SceneDepthPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<SceneDepthImage>::default())
            .add_systems(Startup, setup_scene_depth_image)
            .add_systems(Update, resize_scene_depth_image);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_systems(RenderStartup, init_msaa_depth_resolve)
                .add_render_graph_node::<ViewNodeRunner<CopySceneDepthNode>>(Core3d, CopySceneDepth)
                .add_render_graph_edges(
                    Core3d,
                    (
                        Node3d::MainOpaquePass,
                        CopySceneDepth,
                        Node3d::MainTransparentPass,
                    ),
                );
        }
    }
}

/// Render-world pipeline for the MSAA scene-depth resolve pass. Reads the
/// multisampled main-pass depth as a texture and writes sample 0 into the
/// single-sample [`SceneDepthImage`] via a depth-only fullscreen draw, so the
/// atmosphere pass can sample scene depth under MSAA exactly as it does without.
#[derive(Resource)]
struct MsaaDepthResolve {
    layout: BindGroupLayoutDescriptor,
    pipeline_id: CachedRenderPipelineId,
}

/// Build the resolve pipeline. Runs in [`RenderStartup`], matching the rest of
/// the 0.18 render-pipeline init style (e.g. `udlod`'s terrain pipeline).
fn init_msaa_depth_resolve(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "msaa_depth_resolve_layout",
        &BindGroupLayoutEntries::single(ShaderStages::FRAGMENT, texture_depth_2d_multisampled()),
    );

    let shader = asset_server.load("shaders/msaa_depth_resolve.wgsl");

    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("msaa_depth_resolve_pipeline".into()),
        layout: vec![layout.clone()],
        push_constant_ranges: vec![],
        vertex: VertexState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("vertex".into()),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        // Depth-only: no color targets, always write the resolved sample.
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::Always,
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

/// Create a 1×1 placeholder Image at startup. `resize_scene_depth_image`
/// grows it to viewport size on the first frame the camera reports one.
///
/// `new_uninit` skips the initial-data write, which `Queue::write_texture`
/// forbids for depth formats. The GPU texture is allocated empty in the
/// render-asset extract pass and overwritten each frame by
/// [`CopySceneDepthNode`].
pub(crate) fn setup_scene_depth_image(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
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
    // COPY_DST for the MSAA-off direct copy; TEXTURE_BINDING so the atmosphere
    // pass can sample it; RENDER_ATTACHMENT so the MSAA depth-resolve pass can
    // write the resolved sample 0 straight into it.
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
    let handle = images.add(image);
    commands.insert_resource(SceneDepthImage { handle });
}

/// Keep the SceneDepthImage in lockstep with the camera's physical viewport.
/// Runs every frame; the early-out for matching sizes is the common path.
fn resize_scene_depth_image(
    scene_depth: Option<Res<SceneDepthImage>>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<&Camera, With<ShipCamera>>,
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
    let Some(image) = images.get_mut(&scene_depth.handle) else {
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
