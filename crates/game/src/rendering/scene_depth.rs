//! Scene-depth copy for the unified atmosphere pass.
//!
//! The atmosphere fullscreen pass needs to sample the depth of opaque
//! geometry (terrain, impostor body, ship hull) so the in-scatter raymarch
//! can clip at the surface and produce aerial-perspective continuity from
//! sky to ground. WebGPU disallows sampling the live depth attachment from
//! a fragment shader, and our forked `bevy_terrain` does not queue into
//! `Opaque3dPrepass`, so the standard prepass-depth texture is terrain-blind.
//!
//! Workaround: insert a render-graph node between [`Node3d::MainOpaquePass`]
//! and [`Node3d::MainTransparentPass`] that `copy_texture_to_texture`s the
//! main pass's `ViewDepthTexture` into an [`Image`] we own. Materials bind
//! that `Image` via `AsBindGroup` and sample it as `texture_depth_2d` from
//! WGSL. One extra full-screen depth copy per frame; trivial cost.
//!
//! Assumes MSAA = 1 (the default for `ShipCamera`). With MSAA the source
//! is multisampled and the copy would need a resolve pass — fix when MSAA
//! lands, not before.

use crate::camera::ShipCamera;
use bevy::asset::{Assets, Handle, RenderAssetUsages};
use bevy::camera::Camera;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::prelude::*;
use bevy::ecs::query::QueryItem;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::{
    RenderApp,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_graph::{
        NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
    },
    render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
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
        // The ViewNode runs for every view in the render world. Shadow
        // cascades, light views, and any future multisampled cameras may
        // have a higher sample count than our single-sample destination
        // Image. `copy_texture_to_texture` requires matching sample counts;
        // skip the copy when they don't match. Atmosphere is only sampled
        // from the ship camera's view anyway.
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
    image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
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
