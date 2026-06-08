//! Volumetric clouds — integration of the vendored [`thalos_volumetric_clouds`]
//! crate (HZD-style raymarch, MIT, evroon fork).
//!
//! The vendored crate raymarches a planet-relative cloud layer into a texture.
//! This module makes it work on Thalos's sphere via the **tangent-frame trick**:
//! [`drive_clouds`] reads the `ShipCamera`, the homeworld's render-space centre,
//! and the sun, then feeds the crate a rotated `inverse_camera_view` so the
//! player's local "up" (radial from the planet centre) maps to the shader's +Y,
//! with altitude in `camera_translation.y`.
//!
//! **Compositing.** The cloud texture is *not* drawn by a separate quad — a
//! fullscreen transparent quad sorts unreliably against the body's fullscreen
//! `BodySky` atmosphere pass under big_space, so the atmosphere would draw over
//! the clouds at some camera angles. Instead the texture is bound onto
//! [`thalos_body_render::BodySkyMaterial`] (`cloud_layer`) and composited as the
//! final step of `body_sky.wgsl`, which lands the clouds deterministically on
//! top of the sky. The bind happens per-frame for the body the player is at, in
//! `super::ground_terrain::update_body_terrain_atmosphere`. High-altitude
//! fade-out is handled for free by `BodySky`'s LOD visibility (hidden once the
//! camera leaves the atmosphere shell).
//!
//! Known limitations (see `docs/atmosphere.md` for the proper plan): the layer
//! is a tangent-plane approximation that degrades at the limb / high altitude,
//! and the coverage field is 2-D (extruded vertically), so it follows the
//! camera rather than staying glued to the ground.

use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::transform::TransformSystems;

use thalos_body_render::{AU_M, LIGHT_AT_1AU};
use thalos_volumetric_clouds::{CameraMatrices, CloudsConfig, CloudsPlugin};

use crate::camera::ShipCamera;

use super::types::{CameraExposure, RealSpaceBody, SimulationState, SolarSystemState};

// ── Tunable appearance ───────────────────────────────────────────────────────
/// Cloud-base altitude above the body's reference radius, metres.
const BASE_ALTITUDE_M: f32 = 2000.0;
/// Cloud-layer thickness, metres (top = base + thickness). A thin deck (vs the
/// 3 km starting slab) reads as broken cumulus rather than a fuzzy wall.
const THICKNESS_M: f32 = 1300.0;
/// Overcast fraction knob (0 = clear, 1 = solid). Tuned down toward broken /
/// scattered cloud rather than full overcast (BRP-tunable at runtime).
const COVERAGE: f32 = 0.38;
/// Extinction density multiplier. Some core contrast, but not so high that the
/// flat deck base reads as a hard sliced edge.
const DENSITY: f32 = 0.09;
/// Fraction of the band over which density fades in from the base. Wide, so the
/// (single, flat) deck base reads as a soft underside rather than a hard cutoff.
const BOTTOM_SOFTNESS: f32 = 0.5;
/// Overall cloud feature scale (upstream default 1.5). Larger = smaller, more
/// frequent cells (a shorter atlas repeat period in world space) → a busier,
/// more varied sky rather than a few big blobs.
const BASE_SCALE: f32 = 2.4;
/// Detail-erosion noise scale. Lower than upstream's 42 so the high-frequency
/// Worley erosion has a coarser period (~hundreds of m, not ~tens) and doesn't
/// alias into salt-and-pepper noise at the raymarch's constant ~280 m step.
const DETAIL_SCALE: f32 = 16.0;
/// Detail-erosion strength (upstream 0.27). Some structure, but eased back from
/// 0.35 because strong erosion accentuated the vertical-column "elongation" of
/// the 2-D coverage field.
const DETAIL_STRENGTH: f32 = 0.26;
/// Base-shape edge softness.
const EDGE_SOFTNESS: f32 = 0.11;
/// `SCENE_FLUX_SCALE` mirror from `thalos_body_render::shading` — the cloud sun
/// radiance is scaled to the same exposure range as terrain/atmosphere.
const SUN_FLUX_SCALE: f32 = 0.5;
const AMBIENT_TOP_SCALE: f32 = 0.10;
const AMBIENT_BOTTOM_SCALE: f32 = 0.05;

pub struct CloudsRenderPlugin;

impl Plugin for CloudsRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(CloudsPlugin)
            .register_type::<CloudsConfig>()
            .register_type::<CameraMatrices>()
            .add_systems(bevy::app::PostStartup, init_cloud_appearance)
            .add_systems(
                bevy::app::PostUpdate,
                drive_clouds.after(TransformSystems::Propagate),
            );
    }
}

/// One-time static appearance setup. Kept out of the per-frame [`drive_clouds`]
/// (which writes only the dynamic fields — sun, planet radius, camera) so
/// runtime BRP edits to coverage/density/scale/heights persist instead of being
/// overwritten every frame. Runs in `PostStartup`, after `CloudsPlugin` inserts
/// the default config at `Startup`.
fn init_cloud_appearance(mut config: ResMut<CloudsConfig>) {
    config.clouds_bottom_height = BASE_ALTITUDE_M;
    config.clouds_top_height = BASE_ALTITUDE_M + THICKNESS_M;
    config.clouds_coverage = COVERAGE;
    config.clouds_density = DENSITY;
    config.clouds_base_scale = BASE_SCALE;
    config.clouds_detail_scale = DETAIL_SCALE;
    config.clouds_detail_strength = DETAIL_STRENGTH;
    config.clouds_base_edge_softness = EDGE_SOFTNESS;
    config.clouds_bottom_softness = BOTTOM_SOFTNESS;
}

/// Per-frame: build the planet-local tangent frame from the ship camera and the
/// homeworld, and feed it (plus the sun) to the cloud crate.
///
/// Runs in `PostUpdate` after `TransformSystems::Propagate` so body
/// `GlobalTransform`s (and the camera's) are the recentred big_space values —
/// same ordering reason as `update_body_terrain_atmosphere`.
fn drive_clouds(
    ship_cam_q: Query<(&GlobalTransform, &Camera), With<ShipCamera>>,
    body_q: Query<(&RealSpaceBody, &GlobalTransform)>,
    sim: Res<SimulationState>,
    cache: Res<SolarSystemState>,
    exposure: Res<CameraExposure>,
    mut cam_mat: ResMut<CameraMatrices>,
    mut config: ResMut<CloudsConfig>,
) {
    let homeworld_id = sim.system.homeworld_id;
    let radius = sim.system.bodies[homeworld_id].radius_m as f32;

    // Homeworld centre in render space (big_space SHIP_LAYER, 1 unit = 1 m).
    let mut planet_center = None;
    for (rsb, gt) in &body_q {
        if rsb.body_id == homeworld_id {
            planet_center = Some(gt.translation());
            break;
        }
    }
    let Some(planet_center) = planet_center else {
        return;
    };
    let Ok((cam_gt, camera)) = ship_cam_q.single() else {
        return;
    };

    let cam_pos = cam_gt.translation();
    let to_cam = cam_pos - planet_center;
    let dist = to_cam.length();
    if dist < 1.0 {
        return;
    }
    let up = to_cam / dist;
    let altitude = dist - radius;

    // Orthonormal planet-local tangent basis: any stable horizontal axes work
    // (they just orient the noise field). `tangent_from_world` maps a world
    // vector v → (v·east, v·up, v·north), i.e. local up → +Y.
    let world_ref = if up.dot(Vec3::Y).abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let east = world_ref.cross(up).normalize();
    let north = up.cross(east);
    let tangent_from_world = Mat3::from_cols(east, up, north).transpose();

    // Feed a rotated world_from_view so the raymarch's world-space rays emerge
    // in the tangent frame. Only the rotation matters (rays use w = 0); the
    // translation component is ignored by the shader.
    let world_from_view = cam_gt.to_matrix();
    cam_mat.inverse_camera_view = Mat4::from_mat3(tangent_from_world) * world_from_view;
    cam_mat.inverse_camera_projection = camera.computed.clip_from_view.inverse();
    cam_mat.translation = Vec3::new(0.0, altitude, 0.0);

    // Sun direction (toward the star) and scene-matched flux.
    let Some(states) = cache.states.as_ref() else {
        return;
    };
    let star_pos = states.first().map(|s| s.position).unwrap_or(DVec3::ZERO);
    let hw_pos = states
        .get(homeworld_id)
        .map(|s| s.position)
        .unwrap_or(DVec3::ZERO);
    let star_off = star_pos - hw_pos;
    let d_star = star_off.length();
    let sun_world = if d_star > 0.0 {
        (star_off / d_star).as_vec3()
    } else {
        Vec3::Y
    };
    let sun_tangent = tangent_from_world * sun_world;
    let au_over_d = (AU_M / d_star.max(1.0)) as f32;
    let scene_flux = LIGHT_AT_1AU * au_over_d * au_over_d * exposure.gain;

    // Drive only the dynamic config (static appearance is set once in
    // `init_cloud_appearance`, leaving it BRP-tunable at runtime).
    config.planet_radius = radius;
    config.sun_dir = Vec4::new(sun_tangent.x, sun_tangent.y, sun_tangent.z, 0.0);
    let sun_rgb = Vec3::new(1.0, 0.97, 0.92) * scene_flux * SUN_FLUX_SCALE;
    config.sun_color = Vec4::new(sun_rgb.x, sun_rgb.y, sun_rgb.z, 1.0);
    config.clouds_ambient_color_top =
        Vec4::new(0.55, 0.66, 0.86, 0.0) * scene_flux * AMBIENT_TOP_SCALE;
    config.clouds_ambient_color_bottom =
        Vec4::new(0.36, 0.43, 0.55, 0.0) * scene_flux * AMBIENT_BOTTOM_SCALE;
}
