//! Thalos fork of `bevy-volumetric-clouds` (MIT, evroon — see `LICENSE` and
//! `README.md` for upstream attribution).
//!
//! Upstream renders Horizon-Zero-Dawn-style raymarched volumetric clouds and
//! composites them onto a Y-up, single-camera skybox. Thalos is spherical, runs
//! under `big_space` floating origin, and has two cameras (ship + map), so this
//! fork keeps the valuable, geometry-agnostic half — the noise generation
//! (multi-channel 3-D Perlin-Worley volume, built by the `init` compute pass)
//! and the HZD density+lighting raymarch (`update` pass) — and reworks the
//! geometry around a real spherical planet:
//!
//!   * removed the 6-plane skybox composite, the built-in blue `get_sky_color`,
//!     and the `setup_daylight` directional light;
//!   * removed the `Single<Camera>` `update_camera_matrices` system (it panics
//!     with our two cameras). [`CameraMatrices`] is now a plain public resource
//!     that the game writes each frame in the **body-fixed frame** of the
//!     active cloud body: the camera's true planet-centred position plus a
//!     `body_from_world`-rotated view basis. The raymarch is a true ray-sphere
//!     shell march from that position, and all noise fields are sampled at
//!     body-fixed positions, so clouds stay glued to the ground, co-rotate
//!     with the planet, and the horizon is correct at any altitude and at the
//!     limb;
//!   * large-scale weather comes from a planet-fixed cubemap
//!     ([`CloudWeatherMap`]), sampled by body-fixed direction. RGBA carries
//!     coverage, cloud type, normalized base, and normalized top from the
//!     canonical per-body runtime field;
//!   * the compute shader stores the *clean* cloud layer (rgb = premultiplied
//!     in-scatter, a = transmittance) to [`CloudRenderTexture`], plus the
//!     per-pixel nearest cloud-hit distance to [`CloudDistanceTexture`], for
//!     the game to composite over its own scene with true depth occlusion.
//!
//! Remaining work (evolving/advected weather, storms, half-res + temporal
//! upscale) is tracked in `docs/atmosphere.md`.

mod compute;
/// Controls the compute shader which renders the volumetric clouds.
pub mod config;
mod images;
mod uniforms;

use bevy::asset::embedded_asset;
use bevy::prelude::*;
use bevy::shader::load_shader_library;

pub use self::compute::CameraMatrices;
pub use self::config::CloudsConfig;
pub use self::images::{
    CloudTargetMemory, RENDER_HEIGHT, RENDER_WIDTH, WEATHER_FACE_SIZE, cloud_target_memory,
    cloud_target_memory_for, cloud_weather_image,
};

use self::compute::CloudsComputePlugin;
use self::images::build_images;
use self::uniforms::CloudsImage;

/// Handle to the final cloud render texture (RGBA32F: `rgb` = premultiplied
/// in-scatter, `a` = transmittance). The game binds this in a fullscreen
/// premultiplied composite pass: `out.rgb = rgb`, `out.a = 1 - transmittance`.
#[derive(Resource, Clone)]
pub struct CloudRenderTexture {
    pub handle: Handle<Image>,
}

/// Handle to the per-pixel nearest cloud-hit distance texture (R32F, metres
/// from the camera; ≥ 1e8 sentinel where the ray hit no cloud). Paired with
/// [`CloudRenderTexture`] so the composite can occlude clouds against opaque
/// geometry by true depth instead of a shell-band approximation.
#[derive(Resource, Clone)]
pub struct CloudDistanceTexture {
    pub handle: Handle<Image>,
}

/// Handle to the active body's planet-fixed RGBA8 cubemap weather field:
/// coverage, cloud type, normalized base, normalized top. The game copies the
/// canonical per-body field here when active body or weather version changes.
#[derive(Resource, Clone)]
pub struct CloudWeatherMap {
    pub handle: Handle<Image>,
}

/// Renders volumetric clouds into [`CloudRenderTexture`] each frame.
///
/// The configuration is the [`CloudsConfig`] resource; the camera basis is the
/// [`CameraMatrices`] resource, which the consumer must write each frame (this
/// fork does not auto-derive it from a camera entity — see the module docs).
pub struct CloudsPlugin;

impl Plugin for CloudsPlugin {
    fn build(&self, app: &mut App) {
        // `common.wgsl` is a shader library imported by `clouds_compute.wgsl`
        // as `bevy_open_world::common` (the import path is declared inside the
        // file, so the crate rename is irrelevant). The compute pipeline loads
        // `clouds_compute.wgsl` via `load_embedded_asset!` in `compute.rs`.
        load_shader_library!(app, "shaders/common.wgsl");
        embedded_asset!(app, "shaders/clouds_compute.wgsl");

        app.insert_resource(CloudsConfig::default())
            .add_plugins(CloudsComputePlugin)
            .add_systems(Startup, clouds_setup)
            .add_systems(Update, resize_cloud_targets);
    }
}

/// Resize all view/history targets together. Handles stay stable, so every
/// material and extracted bind group follows the new viewport-relative images
/// without a parallel rebind path.
fn resize_cloud_targets(
    mut config: ResMut<CloudsConfig>,
    cloud_images: Option<Res<CloudsImage>>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(cloud_images) = cloud_images else {
        return;
    };
    let width = config.render_resolution.x.max(8.0).round() as u32;
    let height = config.render_resolution.y.max(8.0).round() as u32;
    let extent = bevy::render::render_resource::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let handles = [
        &cloud_images.cloud_render_image,
        &cloud_images.cloud_distance_image,
        &cloud_images.history_image,
        &cloud_images.history_distance_image,
    ];
    let needs_resize = handles.iter().any(|handle| {
        images
            .get(*handle)
            .is_some_and(|image| image.size() != UVec2::new(width, height))
    });
    if !needs_resize {
        return;
    }
    for handle in handles {
        if let Some(mut image) = images.get_mut(handle) {
            image.resize(extent);
        }
    }
    config.history_epoch = config.history_epoch.wrapping_add(1).max(1);
}

fn clouds_setup(mut commands: Commands, images: ResMut<Assets<Image>>) {
    let built = build_images(images);

    commands.insert_resource(CloudRenderTexture {
        handle: built.cloud_render_image.clone(),
    });
    commands.insert_resource(CloudDistanceTexture {
        handle: built.cloud_distance_image.clone(),
    });
    commands.insert_resource(CloudWeatherMap {
        handle: built.weather_image.clone(),
    });
    commands.insert_resource(CloudsImage {
        cloud_render_image: built.cloud_render_image,
        cloud_worley_image: built.cloud_worley_image,
        cloud_distance_image: built.cloud_distance_image,
        weather_image: built.weather_image,
        history_image: built.history_image,
        history_distance_image: built.history_distance_image,
    });
    commands.insert_resource(CameraMatrices {
        translation: Vec3::ZERO,
        inverse_camera_projection: Mat4::IDENTITY,
        inverse_camera_view: Mat4::IDENTITY,
    });
}
