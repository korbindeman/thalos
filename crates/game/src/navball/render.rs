//! Render-to-texture pipeline for the navball.
//!
//! Spawns:
//! - A unit sphere on a dedicated render layer with [`NavballMaterial`].
//! - An orthographic camera on the same layer that renders the sphere to
//!   an off-screen [`Image`]; the image handle is exposed via
//!   [`NavballRenderTarget`] for the UI overlay to display.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{ClearColorConfig, ImageRenderTarget, RenderTarget, ScalingMode};
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};

use crate::navball::material::{NavballMaterial, NavballParams};
use crate::navball::texture::{DEFAULT_HEIGHT, DEFAULT_WIDTH, generate_navball_rgba8};

/// Dedicated render layer for the navball's off-screen scene. No other
/// game entity lives on this layer.
pub const NAVBALL_LAYER: usize = 3;

/// Off-screen render target size (square).
const RENDER_TARGET_SIZE: u32 = 512;

/// Holds the off-screen image the navball renders into. The UI module
/// pulls this handle and displays it as an `ImageNode`.
#[derive(Resource, Clone)]
pub struct NavballRenderTarget {
    pub image: Handle<Image>,
}

#[derive(Component)]
pub struct NavballSphere;

pub fn setup_navball_render(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<NavballMaterial>>,
) {
    // ---- Baked equirect texture ---------------------------------------
    let pixels = generate_navball_rgba8(DEFAULT_WIDTH, DEFAULT_HEIGHT);
    let mut nav_tex = Image::new(
        Extent3d {
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    nav_tex.sampler = ImageSampler::linear();
    let nav_tex_handle = images.add(nav_tex);

    // ---- Off-screen render target -------------------------------------
    let mut target = Image::new_fill(
        Extent3d {
            width: RENDER_TARGET_SIZE,
            height: RENDER_TARGET_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    target.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    let target_handle = images.add(target);
    commands.insert_resource(NavballRenderTarget {
        image: target_handle.clone(),
    });

    // ---- Sphere mesh + material ---------------------------------------
    let mesh = meshes.add(Sphere::new(1.0).mesh().uv(64, 32));
    let material = materials.add(NavballMaterial {
        params: NavballParams::default(),
        texture: nav_tex_handle,
    });

    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::IDENTITY,
        RenderLayers::layer(NAVBALL_LAYER),
        NavballSphere,
        Name::new("NavballSphere"),
    ));

    // ---- Dedicated off-screen camera ----------------------------------
    //
    // Orthographic, looking at the sphere from +Z at distance 2.5. The
    // 2.4 scale gives ~0.2 unit margin around a unit sphere.
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: -1,
            clear_color: ClearColorConfig::Custom(Color::NONE),
            ..default()
        },
        RenderTarget::Image(ImageRenderTarget::from(target_handle)),
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: 2.4,
                height: 2.4,
            },
            near: 0.1,
            far: 10.0,
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.0, 0.0, 2.5).looking_at(Vec3::ZERO, Vec3::Y),
        RenderLayers::layer(NAVBALL_LAYER),
        Name::new("NavballCamera"),
    ));
}
