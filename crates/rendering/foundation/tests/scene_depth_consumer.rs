use bevy::asset::{AssetPlugin, Assets};
use bevy::camera::Viewport;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use thalos_render_foundation::{
    SceneDepthImage, SceneDepthPlugin, SceneDepthView, scene_depth_view_texture_usages,
};

#[test]
fn standalone_consumer_gets_a_view_sized_sampleable_depth_image() {
    let mut app = App::new();
    app.add_plugins(AssetPlugin::default())
        .insert_resource(Assets::<Image>::default())
        .add_plugins(SceneDepthPlugin);
    app.world_mut().spawn((
        Camera {
            viewport: Some(Viewport {
                physical_size: UVec2::new(640, 360),
                ..default()
            }),
            ..default()
        },
        SceneDepthView,
    ));

    app.update();

    let scene_depth = app.world().resource::<SceneDepthImage>();
    let images = app.world().resource::<Assets<Image>>();
    let image = images
        .get(&scene_depth.handle)
        .expect("scene-depth image should be owned by the foundation");
    assert_eq!(image.size(), UVec2::new(640, 360));
    assert_eq!(image.texture_descriptor.format, TextureFormat::Depth32Float);
    assert_eq!(
        image.texture_descriptor.usage,
        TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT
    );
    assert_eq!(
        scene_depth_view_texture_usages(),
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING
    );
}
