//! Volumetric-cloud orchestration for [`thalos_body_render::clouds`] (an
//! HZD-style raymarch retaining the upstream MIT/evroon attribution).
//!
//! The body-render module raymarches a planet-relative cloud layer into a texture.
//! [`drive_clouds`] runs it in the **body-fixed frame** of the active cloud
//! body (the nearest body with an authored terrestrial cloud climate): it
//! feeds the mechanism the
//! camera's true planet-centred position rotated into body-fixed coordinates
//! plus a `body_from_world`-rotated view basis, so the raymarch is a real
//! spherical-shell march and every noise field is sampled planet-fixed —
//! clouds stay glued to the ground, co-rotate with the planet, and the horizon
//! is correct at any altitude and at the limb.
//!
//! **Weather.** Large-scale structure comes from a planet-fixed RGBA cubemap
//! ([`thalos_body_render::CloudWeatherMap`]).
//! [`sync_cloud_weather_map`] projects the per-body
//! [`CloudWeatherField`](crate::solar_system_state::CloudWeatherField)
//! (owned by `SolarSystemState`, like the other per-body environment state)
//! into that texture. RGBA carries coverage, type, normalized base, and
//! normalized top. The future weather system evolves the field and bumps its
//! `version`; every render projection then re-uploads the same authority.
//!
//! **Compositing.** The cloud texture is *not* drawn by a separate quad — a
//! fullscreen transparent quad sorts unreliably against the body's fullscreen
//! `BodySky` atmosphere pass under big_space, so the atmosphere would draw over
//! the clouds at some camera angles. Instead the texture (and the per-pixel
//! nearest cloud-hit distance the raymarch exports alongside it) is bound onto
//! [`thalos_body_render::BodySkyMaterial`] and composited as the final step of
//! `body_sky.wgsl`, which lands the clouds deterministically on top of the sky
//! and occludes them against geometry by true depth. The bind happens
//! per-frame for the [`ActiveCloudBody`], in
//! `super::ground_terrain::update_body_terrain_atmosphere`. High-altitude
//! fade-out is handled for free by `BodySky`'s LOD visibility (hidden once the
//! camera leaves the atmosphere shell).

use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::transform::TransformSystems;

use thalos_body_render::{
    AU_M, CameraMatrices, CloudWeatherMap, CloudsConfig, LIGHT_AT_1AU, WEATHER_FACE_SIZE,
};
use thalos_world::BodyId;

use crate::camera::ShipCamera;
use crate::graphics_settings::GraphicsSettings;

use super::types::{CameraExposure, RealSpaceBody, SimulationState, SolarSystemState};

// ── Tunable appearance ───────────────────────────────────────────────────────
/// Global scale on the planet-fixed weather coverage map (which carries the
/// local overcast fraction, mean ≈ `CloudWeatherField::coverage_mean`).
/// 1.0 = trust the weather field; global trim.
const COVERAGE_SCALE: f32 = 1.0;
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

/// Which body the volumetric cloud raymarch is currently rendered for — the
/// authored cloudy body the camera is closest to, or `None` when no such body
/// exists. `ground_terrain::update_body_terrain_atmosphere` binds
/// the live cloud textures onto this body's `BodySkyMaterial` (every other
/// body keeps the blank fallback). **Sole writer:** [`drive_clouds`].
#[derive(Resource, Default, Clone, Copy)]
pub struct ActiveCloudBody(pub Option<BodyId>);

pub struct CloudsRenderPlugin;

impl Plugin for CloudsRenderPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<CloudsConfig>()
            .register_type::<CameraMatrices>()
            .init_resource::<ActiveCloudBody>()
            .add_systems(bevy::app::PostStartup, init_cloud_appearance)
            .add_systems(Update, sync_cloud_weather_map)
            .add_systems(
                bevy::app::PostUpdate,
                drive_clouds.after(TransformSystems::Propagate),
            );
    }
}

/// One-time projection-quality setup. Authored climate plus camera/light state
/// are projected per frame; sampling and reconstruction controls remain
/// independently editable. Runs in `PostStartup`, after `CloudsPlugin` inserts
/// the default config at `Startup`.
fn init_cloud_appearance(mut config: ResMut<CloudsConfig>) {
    config.clouds_coverage = COVERAGE_SCALE;
    config.clouds_density = DENSITY;
    config.clouds_base_scale = BASE_SCALE;
    config.clouds_detail_scale = DETAIL_SCALE;
    config.clouds_detail_strength = DETAIL_STRENGTH;
    config.clouds_base_edge_softness = EDGE_SOFTNESS;
    config.clouds_bottom_softness = BOTTOM_SOFTNESS;
    // 4 self-shadow steps (upstream 6): each view sample pays this many extra
    // density evaluations, and the adaptive view step already raised the
    // sample count through the band — the lighting difference is subtle, the
    // cost is not. Raise it back up if shadow gradients look flat.
    config.clouds_shadow_raymarch_steps_count = 4;
}

/// Per-frame: pick the active cloud body, build its body-fixed frame from the
/// ship camera, and feed it (plus the sun) to the cloud crate.
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
    graphics: Res<GraphicsSettings>,
    mut active: ResMut<ActiveCloudBody>,
    mut cam_mat: ResMut<CameraMatrices>,
    mut config: ResMut<CloudsConfig>,
    time: Res<Time>,
    mut wind_angle: Local<f32>,
) {
    // Clouds disabled in graphics settings: park the raymarch camera far
    // outside any shell (same as the no-cloud-body case below) so every ray
    // misses and `update_body_terrain_atmosphere` binds the blank fallback,
    // leaving the sky clear at near-zero GPU cost.
    if !graphics.clouds {
        active.0 = None;
        cam_mat.translation = Vec3::new(0.0, config.planet_radius * 1.0e3 + 1.0e9, 0.0);
        return;
    }

    let Ok((cam_gt, camera)) = ship_cam_q.single() else {
        return;
    };
    let cam_pos = cam_gt.translation();

    // Active cloud body: the terrestrial-atmosphere body the camera is
    // closest to (by altitude above its surface).
    let mut best: Option<(BodyId, Vec3, Quat, f32, f32)> = None;
    for (rsb, gt) in &body_q {
        let Some(body) = sim.system.bodies.get(rsb.body_id) else {
            continue;
        };
        if body
            .terrestrial_atmosphere
            .as_ref()
            .and_then(|atmosphere| atmosphere.clouds.as_ref())
            .is_none()
        {
            continue;
        }
        let center = gt.translation();
        let radius = body.radius_m as f32;
        let alt = (cam_pos - center).length() - radius;
        if best.is_none_or(|(_, _, _, _, best_alt)| alt < best_alt) {
            best = Some((
                rsb.body_id,
                center,
                gt.compute_transform().rotation,
                radius,
                alt,
            ));
        }
    }
    let Some((body_id, planet_center, body_rot, radius, _alt)) = best else {
        // No cloud-capable body: park the raymarch camera far outside any
        // shell so every ray misses and the pass is near-free.
        active.0 = None;
        cam_mat.translation = Vec3::new(0.0, config.planet_radius * 1.0e3 + 1.0e9, 0.0);
        return;
    };
    active.0 = Some(body_id);
    let climate = sim.system.bodies[body_id]
        .terrestrial_atmosphere
        .as_ref()
        .and_then(|atmosphere| atmosphere.clouds.as_ref())
        .expect("active cloud body was filtered by authored climate");

    let to_cam = cam_pos - planet_center;
    if to_cam.length() < 1.0 {
        return;
    }

    // Body-fixed frame: rotate the camera basis and position by the inverse
    // of the body's render-space orientation, so the raymarch's rays and
    // sample positions co-rotate with the surface (same convention as
    // `BodySkyExtra::world_to_body_orientation`). Zonal wind is folded in as
    // a slow extra rotation about the spin axis — the shader then samples an
    // already-advected field with zero per-sample wind math (~1e-8 rad/frame,
    // far below the reprojection change threshold).
    *wind_angle = (*wind_angle + climate.wind_m_s[0] * time.delta_secs() / radius.max(1.0))
        .rem_euclid(std::f32::consts::TAU);
    let q_bw = (Quat::from_rotation_y(*wind_angle) * body_rot.inverse()).normalize();
    let cam_body = q_bw * to_cam;
    cam_mat.translation = cam_body;
    // Rays use only the rotation part of this matrix (w = 0), but the shader's
    // temporal-reprojection change detection compares whole columns — so put
    // the *body-fixed planet-centred* camera position in the translation
    // column. The raw render-space translation drifts with the body's orbital
    // motion every frame, which would permanently disable reprojection even
    // for a parked, surface-static camera. Scaled down so the ~0.25 m f32
    // rounding jitter of `q_bw * to_cam` at planet radius stays below the
    // shader's `CAM_EPSILON` (1e-4) change threshold while real motion
    // (≳ metres/frame) still trips it. The shader recovers the position with
    // `CAM_POS_COLUMN_SCALE` (1e4) for motion reprojection — keep them inverse.
    let mut view_mat = Mat4::from_quat(q_bw) * cam_gt.to_matrix();
    view_mat.w_axis = (cam_body * 1.0e-4).extend(1.0);
    cam_mat.inverse_camera_view = view_mat;
    cam_mat.inverse_camera_projection = camera.computed.clip_from_view.inverse();

    // Sun direction (toward the star, body-fixed) and scene-matched flux.
    let Some(states) = cache.states.as_ref() else {
        return;
    };
    let star_pos = states.first().map(|s| s.position).unwrap_or(DVec3::ZERO);
    let body_pos = states
        .get(body_id)
        .map(|s| s.position)
        .unwrap_or(DVec3::ZERO);
    let star_off = star_pos - body_pos;
    let d_star = star_off.length();
    let sun_world = if d_star > 0.0 {
        (star_off / d_star).as_vec3()
    } else {
        Vec3::Y
    };
    let sun_body = q_bw * sun_world;
    let au_over_d = (AU_M / d_star.max(1.0)) as f32;
    let scene_flux = LIGHT_AT_1AU * au_over_d * au_over_d * exposure.gain;

    // Project authored, quality-neutral climate plus dynamic view/light state.
    // Sampling/reconstruction quality remains independently editable.
    config.planet_radius = radius;
    config.clouds_bottom_height = climate.base_altitude_m.max(0.0);
    config.clouds_top_height =
        (climate.base_altitude_m + climate.thickness_m).max(config.clouds_bottom_height + 1.0);
    config.clouds_density = DENSITY * climate.density.max(0.0);
    config.wind_velocity = Vec3::new(climate.wind_m_s[0], climate.wind_m_s[1], 0.0);
    config.sun_dir = Vec4::new(sun_body.x, sun_body.y, sun_body.z, 0.0);
    let cloud_albedo = Vec3::from_array(climate.albedo).max(Vec3::ZERO);
    let sun_rgb = Vec3::new(1.0, 0.97, 0.92) * cloud_albedo * scene_flux * SUN_FLUX_SCALE;
    config.sun_color = Vec4::new(sun_rgb.x, sun_rgb.y, sun_rgb.z, 1.0);
    config.clouds_ambient_color_top =
        Vec4::new(0.55, 0.66, 0.86, 0.0) * scene_flux * AMBIENT_TOP_SCALE;
    config.clouds_ambient_color_bottom =
        Vec4::new(0.36, 0.43, 0.55, 0.0) * scene_flux * AMBIENT_BOTTOM_SCALE;
}

/// Upload the active body's canonical cubemap field. No default is generated
/// here: authored `CloudClimate::None` is authoritative.
fn sync_cloud_weather_map(
    active: Res<ActiveCloudBody>,
    cache: Res<SolarSystemState>,
    weather: Option<Res<CloudWeatherMap>>,
    mut images: ResMut<Assets<Image>>,
    mut last: Local<Option<(BodyId, u32)>>,
) {
    let Some(body_id) = active.0 else {
        return;
    };
    let Some(weather) = weather else {
        return;
    };
    let Some(field) = cache
        .environment
        .get(body_id)
        .and_then(|env| env.cloud_weather.as_ref())
    else {
        return;
    };
    if field.face_size != WEATHER_FACE_SIZE {
        error!(
            target: "thalos::clouds",
            "weather field face size {} does not match renderer {}",
            field.face_size,
            WEATHER_FACE_SIZE,
        );
        return;
    }
    if *last == Some((body_id, field.version)) {
        return;
    }
    let Some(mut image) = images.get_mut(&weather.handle) else {
        return;
    };
    image.data = Some(field.rgba8_bytes());
    *last = Some((body_id, field.version));
}
