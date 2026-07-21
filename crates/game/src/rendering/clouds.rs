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
const COVERAGE_SCALE: f32 = 1.25;
/// Extinction density multiplier. Some core contrast, but not so high that the
/// flat deck base reads as a hard sliced edge.
const DENSITY: f32 = 0.0026;
/// Fraction of the local typed layer over which density fades in from its base.
/// The weather field varies the base altitude, so this no longer needs to hide
/// a single sliced deck with an excessively broad fog ramp.
const BOTTOM_SOFTNESS: f32 = 0.16;
/// Fine 3-D boundary erosion. Solid cores remain intact in the shader; this
/// only cuts the cauliflower-scale silhouette where density is already low.
const DETAIL_STRENGTH: f32 = 0.16;
/// Base-shape edge softness.
const EDGE_SOFTNESS: f32 = 0.055;
/// `SCENE_FLUX_SCALE` mirror from `thalos_body_render::shading` — the cloud sun
/// radiance is scaled to the same exposure range as terrain/atmosphere.
/// CLOUD-4 keeps this below the pre-coupling 0.48 so stacked volume samples
/// and the orbital disc no longer clip to pure white before air-mass tinting.
const SUN_FLUX_SCALE: f32 = 0.36;
const AMBIENT_TOP_SCALE: f32 = 0.038;
const AMBIENT_BOTTOM_SCALE: f32 = 0.014;

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
    config.clouds_detail_strength = DETAIL_STRENGTH;
    config.clouds_base_edge_softness = EDGE_SOFTNESS;
    config.clouds_bottom_softness = BOTTOM_SOFTNESS;
    // One lobe-scale directional probe gives a stable lee-side cue. Three
    // exponentially spaced taps exposed the typed vertical profile as nested
    // horizontal bands; a resolved light volume belongs to CLOUD-4.
    config.clouds_shadow_raymarch_steps_count = 1;
    config.clouds_shadow_raymarch_step_size = 900.0;
    config.clouds_shadow_raymarch_step_multiply = 2.5;
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
    mut history_continuity: Local<Option<(BodyId, u32, f64)>>,
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
    if let Some(viewport) = camera.physical_viewport_size() {
        config.set_viewport_resolution(viewport);
    }
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
    let weather_version = cache
        .environment
        .get(body_id)
        .and_then(|environment| environment.cloud_weather.as_ref())
        .map_or(0, |weather| weather.version);
    let sim_time = sim.simulation.sim_time();
    let continuity_changed =
        history_continuity.is_none_or(|(last_body, last_version, last_time)| {
            last_body != body_id
                || last_version != weather_version
                || (sim_time - last_time).abs() > 0.5
        });
    if continuity_changed {
        config.history_epoch = config.history_epoch.wrapping_add(1).max(1);
    }
    *history_continuity = Some((body_id, weather_version, sim_time));
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
    config.clouds_base_shape_scale_m = climate.base_shape_scale_m.max(500.0);
    config.clouds_detail_scale_m = climate.detail_scale_m.max(50.0);
    config.wind_velocity = Vec3::new(climate.wind_m_s[0], climate.wind_m_s[1], 0.0);
    config.sun_dir = Vec4::new(sun_body.x, sun_body.y, sun_body.z, 0.0);
    let cloud_albedo = Vec3::from_array(climate.albedo).max(Vec3::ZERO);

    // CLOUD-4 first slice: camera-local day factor still sets the baseline
    // sun/ambient chromaticity; the volume shader then multiplies a per-sample
    // air-mass transmittance so low-sun and low-altitude samples redden further.
    // Finished CLOUD-4 swaps this CPU guess for shared atmosphere LUT samples.
    let local_up = cam_body.normalize_or_zero();
    let sun_mu = local_up.dot(sun_body);
    let day_t = ((sun_mu + 0.04) / 0.28).clamp(0.0, 1.0);
    let day_blend = day_t * day_t * (3.0 - 2.0 * day_t);
    // Slightly cooler noon white and a warmer horizon than the pre-CLOUD-4
    // path — matches the analytic β_R in the volume shader.
    let sun_chromaticity = Vec3::new(1.0, 0.38, 0.10).lerp(Vec3::new(1.0, 0.96, 0.90), day_blend);
    let horizon_transmittance = 0.55 + 0.45 * day_blend;
    // Albedo is applied once here; the volume does not re-multiply climate
    // albedo, so keep it a touch under 1.0 to leave headroom for phase peaks.
    let albedo = cloud_albedo * Vec3::new(0.94, 0.96, 0.99);
    let sun_rgb = sun_chromaticity * albedo * scene_flux * SUN_FLUX_SCALE * horizon_transmittance;
    config.sun_color = Vec4::new(sun_rgb.x, sun_rgb.y, sun_rgb.z, 1.0);
    // Sky-like ambient: more blue at the top, greyer fill under overcast cores.
    let horizon_ambient = 0.28 + 0.72 * day_blend;
    config.clouds_ambient_color_top =
        Vec4::new(0.42, 0.58, 0.88, 0.0) * scene_flux * AMBIENT_TOP_SCALE * horizon_ambient;
    config.clouds_ambient_color_bottom =
        Vec4::new(0.30, 0.36, 0.48, 0.0) * scene_flux * AMBIENT_BOTTOM_SCALE * horizon_ambient;
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
