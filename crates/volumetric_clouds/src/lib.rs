//! Thalos fork of `bevy-volumetric-clouds` (MIT, evroon — see `LICENSE` and
//! `README.md` for upstream attribution).
//!
//! Upstream renders Horizon-Zero-Dawn-style raymarched volumetric clouds and
//! composites them onto a Y-up, single-camera skybox. Thalos is spherical, runs
//! under `big_space` floating origin, and has two cameras (ship + map), so this
//! fork keeps the valuable, geometry-agnostic half — the noise generation
//! (Perlin-Worley atlas + 3-D Worley volume, built by the `init` compute pass)
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
//!   * large-scale coverage comes from a planet-fixed equirect **weather map**
//!     ([`CloudCoverageMap`]), sampled by body-fixed direction and scaled by
//!     the scalar `clouds_coverage` knob. The map defaults to full coverage
//!     (scalar knob alone), and the game regenerates it from per-body
//!     environment state — the hook for the future weather system;
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

pub use crate::compute::CameraMatrices;
pub use crate::config::CloudsConfig;
pub use crate::images::{
    COVERAGE_HEIGHT, COVERAGE_WIDTH, CloudTargetMemory, RENDER_HEIGHT, RENDER_WIDTH,
    cloud_target_memory,
};

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

/// Handle to the per-pixel nearest cloud-hit distance texture (R32F, metres
/// from the camera; ≥ 1e8 sentinel where the ray hit no cloud). Paired with
/// [`CloudRenderTexture`] so the composite can occlude clouds against opaque
/// geometry by true depth instead of a shell-band approximation.
#[derive(Resource, Clone)]
pub struct CloudDistanceTexture {
    pub handle: Handle<Image>,
}

/// Handle to the planet-fixed equirect coverage (weather) map (R8Unorm,
/// [`COVERAGE_WIDTH`]×[`COVERAGE_HEIGHT`]): u = longitude (`atan2(z, x)`,
/// wrapping), v = colatitude (`acos(y)`, clamped), value = local overcast
/// fraction in [0, 1]. The raymarch multiplies it with the scalar
/// `clouds_coverage` knob. Defaults to all-1, so until a consumer writes a
/// real field the scalar knob alone reproduces uniform coverage. The game owns
/// the contents: rewrite the `Image` asset at this handle (it retains a
/// MAIN_WORLD copy) to install a weather field.
#[derive(Resource, Clone)]
pub struct CloudCoverageMap {
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
    let built = build_images(images);

    commands.insert_resource(CloudRenderTexture {
        handle: built.cloud_render_image.clone(),
    });
    commands.insert_resource(CloudDistanceTexture {
        handle: built.cloud_distance_image.clone(),
    });
    commands.insert_resource(CloudCoverageMap {
        handle: built.coverage_image.clone(),
    });
    commands.insert_resource(CloudsImage {
        cloud_render_image: built.cloud_render_image,
        cloud_atlas_image: built.cloud_atlas_image,
        cloud_worley_image: built.cloud_worley_image,
        cloud_distance_image: built.cloud_distance_image,
        coverage_image: built.coverage_image,
        history_image: built.history_image,
        history_distance_image: built.history_distance_image,
    });
    commands.insert_resource(CameraMatrices {
        translation: Vec3::ZERO,
        inverse_camera_projection: Mat4::IDENTITY,
        inverse_camera_view: Mat4::IDENTITY,
    });
}
