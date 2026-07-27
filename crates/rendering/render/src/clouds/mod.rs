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
//! upscale) is tracked in `docs/rendering/atmosphere.md`.

/// CPU mirror of the shared per-place cell morphology field
/// (`thalos::atmosphere`'s `cloud_cell_*`).
pub mod cell_field;
mod composite;
mod compute;
/// Controls the compute shader which renders the volumetric clouds.
pub mod config;
/// CPU-derived shared near/far fill-opacity response (BL-20260723T214730Z).
pub mod fill_lut;
mod images;
/// Placement of the cloud sun-transmittance cascade (CLOUD-5 / W2).
pub mod shadow_frame;
mod uniforms;

use bevy::asset::embedded_asset;
use bevy::prelude::*;
use bevy::shader::load_shader_library;

pub use self::cell_field::{CellStyle, cell_style};
pub use self::composite::CloudCompositeMaterial;
pub use self::compute::CameraMatrices;
pub use self::config::CloudsConfig;
pub use self::fill_lut::{
    CloudFillCalibration, FILL_LUT_VERSION, FillCalibrationInput, derive_fill_calibration,
};
pub use self::images::{
    CLOUD_SHADOW_SIZE, CloudTargetMemory, RENDER_HEIGHT, RENDER_WIDTH, WEATHER_FACE_SIZE,
    WEATHER_MIP_LEVELS, cloud_target_memory, cloud_target_memory_for, cloud_weather_image,
};
pub use self::shadow_frame::{CloudShadowBlock, CloudShadowFrame};

use self::compute::CloudsComputePlugin;
use self::images::build_images;
pub use self::uniforms::CloudsImage;

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

/// Handle to the active body's canonical four-stratum surface-density cube.
/// It is generated and versioned with [`CloudWeatherMap`], not a second weather
/// authority.
#[derive(Resource, Clone)]
pub struct CloudSurfaceDensityMap {
    pub handle: Handle<Image>,
}

/// Handle to the cloud sun-transmittance cascade (RGBA16F, `r` = surviving
/// fraction of the sun beam; 1 = unshadowed) together with the frame it was
/// marched in. This is the **one** cloud-occlusion field: terrain, foliage,
/// rock and craft receivers all sample this handle through the block
/// [`CloudShadowFrame::block`] packs for them, so a second screen-space
/// approximation that could disagree with the visible volume never exists
/// (`docs/rendering/clouds.md` §2, principle 4).
///
/// **Sole writer:** the game's cloud driver (`rendering::clouds::drive_clouds`).
#[derive(Resource, Clone)]
pub struct CloudShadowMap {
    pub handle: Handle<Image>,
    /// Where the cascade was marched, body-fixed.
    pub frame: CloudShadowFrame,
    /// World render space → the marcher's body-fixed frame. This is
    /// `ActiveCloudFrame`'s rotation, never a second derivation of it.
    pub world_to_body: Quat,
    /// Active cloud body's centre in world render space.
    pub body_center_ws: Vec3,
    /// Body-fixed unit direction toward the sun.
    pub sun_body: Vec3,
    /// Artistic scale on the whole term (1 = physical extinction, 0 = stand the
    /// term down without stopping the march).
    pub strength: f32,
    /// Diagnostic: receivers paint the raw transmittance instead of shading it
    /// in (`THALOS_CLOUD_SHADOW=show`). Splits "the cascade is wrong" from "the
    /// receiver projects into it wrong" in one capture — the split that matters
    /// most here, because producer and receiver reach the same lookup frame by
    /// different routes.
    pub debug_show: bool,
}

impl CloudShadowMap {
    /// The uniform block a receiving material embeds. Mirrors
    /// `thalos::cloud_shadow`'s `CloudShadowBlock` field for field.
    pub fn block(&self) -> CloudShadowBlock {
        if !self.frame.active || self.strength <= 0.0 {
            return CloudShadowBlock::default();
        }
        let q = self.world_to_body;
        CloudShadowBlock {
            world_to_body: Vec4::new(q.x, q.y, q.z, q.w),
            body_center_ws: self.body_center_ws.extend(self.strength.clamp(0.0, 1.0)),
            center: self.frame.center.extend(self.frame.half_extent_m),
            axis_u: self
                .frame
                .axis_u
                .extend(self.frame.texel_m(CLOUD_SHADOW_SIZE)),
            // Mode, not a bool: 0 = skip (handled above), 1 = apply, 2 = paint
            // raw — the same convention `ShadowCascadeBlock::gate.z` uses for
            // the contact tier, so a receiver reads one lane for both.
            axis_v: self
                .frame
                .axis_v
                .extend(if self.debug_show { 2.0 } else { 1.0 }),
            up_sun: self.frame.up.extend(self.frame.sun_elevation_cos),
            sun_body: self.sun_body.normalize_or_zero().extend(0.0),
        }
    }
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
        composite::embed_cloud_composite_shader(app);

        app.insert_resource(CloudsConfig::default())
            .add_plugins(bevy::pbr::MaterialPlugin::<CloudCompositeMaterial>::default())
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
    commands.insert_resource(CloudSurfaceDensityMap {
        handle: built.surface_density_image.clone(),
    });
    commands.insert_resource(CloudShadowMap {
        handle: built.cloud_shadow_image.clone(),
        frame: CloudShadowFrame::default(),
        world_to_body: Quat::IDENTITY,
        body_center_ws: Vec3::ZERO,
        sun_body: Vec3::Y,
        strength: 1.0,
        debug_show: false,
    });
    commands.insert_resource(CloudsImage {
        cloud_render_image: built.cloud_render_image,
        cloud_worley_image: built.cloud_worley_image,
        cloud_distance_image: built.cloud_distance_image,
        weather_image: built.weather_image,
        surface_density_image: built.surface_density_image,
        history_image: built.history_image,
        history_distance_image: built.history_distance_image,
        cloud_shadow_image: built.cloud_shadow_image,
    });
    commands.insert_resource(CameraMatrices {
        translation: Vec3::ZERO,
        inverse_camera_projection: Mat4::IDENTITY,
        inverse_camera_view: Mat4::IDENTITY,
    });
}
