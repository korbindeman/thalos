//! Thalos fork of `bevy-volumetric-clouds` (MIT, evroon — see `LICENSE` and
//! `README.md` for upstream attribution).
//!
//! Upstream renders Horizon-Zero-Dawn-style raymarched volumetric clouds and
//! composites them onto a Y-up, single-camera skybox. Thalos is spherical, runs
//! under `big_space` floating origin, and has two cameras (ship + map), so this
//! fork keeps the valuable, geometry-agnostic half — the noise generation
//! (Perlin-Worley atlas + 3-D Worley volume, built by the `init` compute pass)
//! and the HZD density+lighting raymarch (`update` pass) — and strips the half
//! that fights us:
//!
//!   * removed the 6-plane skybox composite, the built-in blue `get_sky_color`,
//!     and the `setup_daylight` directional light;
//!   * removed the `Single<Camera>` `update_camera_matrices` system (it panics
//!     with our two cameras). [`CameraMatrices`] is now a plain public resource
//!     that the game writes each frame in a **planet-local tangent frame**
//!     (local "up" → +Y, altitude in `translation.y`), which makes upstream's
//!     Y-up raymarch render correctly as a local tangent-plane approximation
//!     without touching the raymarch geometry;
//!   * the compute shader stores the *clean* cloud layer (rgb = premultiplied
//!     in-scatter, a = transmittance) to [`CloudRenderTexture`], for the game to
//!     composite over its own scene in a separate fullscreen pass.
//!
//! This is the **NOW / "quick clouds"** path. The tangent-plane approximation
//! degrades at high altitude and at the limb; the game fades clouds out there.
//! The proper spherical-shell raymarch + weather-driven coverage + storms plan
//! lives in `docs/atmosphere.md`.

mod compute;
/// Controls the compute shader which renders the volumetric clouds.
pub mod config;
mod images;
mod uniforms;

use bevy::asset::embedded_asset;
use bevy::prelude::*;
use bevy::shader::load_shader_library;

pub use crate::compute::CameraMatrices;
pub use crate::config::CloudsConfig;

use crate::compute::CloudsComputePlugin;
use crate::images::build_images;
use crate::uniforms::CloudsImage;

/// Handle to the final cloud render texture (RGBA32F: `rgb` = premultiplied
/// in-scatter, `a` = transmittance). The game binds this in a fullscreen
/// premultiplied composite pass: `out.rgb = rgb`, `out.a = 1 - transmittance`.
#[derive(Resource, Clone)]
pub struct CloudRenderTexture {
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
            .add_systems(Startup, clouds_setup);
    }
}

fn clouds_setup(mut commands: Commands, images: ResMut<Assets<Image>>) {
    let (cloud_render_image, cloud_atlas_image, cloud_worley_image, sky_image) =
        build_images(images);

    commands.insert_resource(CloudRenderTexture {
        handle: cloud_render_image.clone(),
    });
    commands.insert_resource(CloudsImage {
        cloud_render_image,
        cloud_atlas_image,
        cloud_worley_image,
        sky_image,
    });
    commands.insert_resource(CameraMatrices {
        translation: Vec3::ZERO,
        inverse_camera_projection: Mat4::IDENTITY,
        inverse_camera_view: Mat4::IDENTITY,
    });
}
