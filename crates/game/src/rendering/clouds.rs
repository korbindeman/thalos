//! Volumetric clouds — integration of the vendored [`thalos_volumetric_clouds`]
//! crate (HZD-style raymarch, MIT, evroon fork).
//!
//! The vendored crate raymarches a planet-relative cloud layer into a texture.
//! [`drive_clouds`] runs it in the **body-fixed frame** of the active cloud
//! body (the nearest terrestrial-atmosphere body): it feeds the crate the
//! camera's true planet-centred position rotated into body-fixed coordinates
//! plus a `body_from_world`-rotated view basis, so the raymarch is a real
//! spherical-shell march and every noise field is sampled planet-fixed —
//! clouds stay glued to the ground, co-rotate with the planet, and the horizon
//! is correct at any altitude and at the limb.
//!
//! **Weather.** Large-scale coverage comes from a planet-fixed equirect
//! coverage map ([`thalos_volumetric_clouds::CloudCoverageMap`]).
//! [`sync_cloud_weather_map`] projects the per-body
//! [`CloudWeatherState`](crate::solar_system_state::CloudWeatherState)
//! (owned by `SolarSystemState`, like the other per-body environment state)
//! into that texture: latitude bands (ITCZ, subtropical dry belts, storm
//! tracks) plus seeded low-frequency variation. The future weather system
//! evolves `CloudWeatherState` and bumps its `version`; the map re-uploads on
//! change.
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

use thalos_body_render::{AU_M, LIGHT_AT_1AU};
use thalos_volumetric_clouds::{
    COVERAGE_HEIGHT, COVERAGE_WIDTH, CameraMatrices, CloudCoverageMap, CloudsConfig, CloudsPlugin,
};
use thalos_world::BodyId;

use crate::camera::ShipCamera;
use crate::graphics_settings::GraphicsSettings;
use crate::solar_system_state::CloudWeatherState;

use super::types::{CameraExposure, RealSpaceBody, SimulationState, SolarSystemState};

// ── Tunable appearance ───────────────────────────────────────────────────────
/// Cloud-base altitude above the body's reference radius, metres.
const BASE_ALTITUDE_M: f32 = 2000.0;
/// Cloud-layer thickness, metres (top = base + thickness). A thin deck (vs the
/// 3 km starting slab) reads as broken cumulus rather than a fuzzy wall.
const THICKNESS_M: f32 = 1300.0;
/// Global scale on the planet-fixed weather coverage map (which carries the
/// local overcast fraction, mean ≈ `CloudWeatherState::coverage_mean`).
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
/// Frequency of the weather map's large-scale variation noise over the unit
/// sphere (≈ feature size of planet circumference / frequency).
const WEATHER_NOISE_FREQ: f32 = 2.5;

/// Which body the volumetric cloud raymarch is currently rendered for — the
/// terrestrial-atmosphere body the camera is closest to, or `None` when no
/// such body exists. `ground_terrain::update_body_terrain_atmosphere` binds
/// the live cloud textures onto this body's `BodySkyMaterial` (every other
/// body keeps the blank fallback). **Sole writer:** [`drive_clouds`].
#[derive(Resource, Default, Clone, Copy)]
pub struct ActiveCloudBody(pub Option<BodyId>);

pub struct CloudsRenderPlugin;

impl Plugin for CloudsRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(CloudsPlugin)
            .register_type::<CloudsConfig>()
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

/// One-time static appearance setup. Kept out of the per-frame [`drive_clouds`]
/// (which writes only the dynamic fields — sun, planet radius, camera) so
/// runtime edits to coverage/density/scale/heights persist instead of being
/// overwritten every frame. Runs in `PostStartup`, after `CloudsPlugin` inserts
/// the default config at `Startup`.
fn init_cloud_appearance(mut config: ResMut<CloudsConfig>) {
    config.clouds_bottom_height = BASE_ALTITUDE_M;
    config.clouds_top_height = BASE_ALTITUDE_M + THICKNESS_M;
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
        if body.terrestrial_atmosphere.is_none() {
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
    *wind_angle = (*wind_angle + config.wind_velocity.x * time.delta_secs() / radius.max(1.0))
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

    // Drive only the dynamic config (static appearance is set once in
    // `init_cloud_appearance`, leaving it editable at runtime).
    config.planet_radius = radius;
    config.sun_dir = Vec4::new(sun_body.x, sun_body.y, sun_body.z, 0.0);
    let sun_rgb = Vec3::new(1.0, 0.97, 0.92) * scene_flux * SUN_FLUX_SCALE;
    config.sun_color = Vec4::new(sun_rgb.x, sun_rgb.y, sun_rgb.z, 1.0);
    config.clouds_ambient_color_top =
        Vec4::new(0.55, 0.66, 0.86, 0.0) * scene_flux * AMBIENT_TOP_SCALE;
    config.clouds_ambient_color_bottom =
        Vec4::new(0.36, 0.43, 0.55, 0.0) * scene_flux * AMBIENT_BOTTOM_SCALE;
}

/// Project the active body's [`CloudWeatherState`] into the planet-fixed
/// equirect coverage map. Installs a default weather state for a body on
/// first contact (seeded per body), and regenerates the texture only when the
/// `(body, version)` pair changes — the weather system's re-upload hook.
fn sync_cloud_weather_map(
    active: Res<ActiveCloudBody>,
    mut cache: ResMut<SolarSystemState>,
    coverage: Option<Res<CloudCoverageMap>>,
    mut images: ResMut<Assets<Image>>,
    mut last: Local<Option<(BodyId, u32)>>,
) {
    let Some(body_id) = active.0 else {
        return;
    };
    let Some(coverage) = coverage else {
        return;
    };
    let state = match cache
        .environment
        .get(body_id)
        .and_then(|env| env.cloud_weather)
    {
        Some(state) => state,
        None => {
            let state = CloudWeatherState {
                seed: CloudWeatherState::default().seed ^ (body_id as u64).wrapping_mul(0x9E37),
                ..CloudWeatherState::default()
            };
            cache.install_cloud_weather(body_id, state);
            state
        }
    };
    if *last == Some((body_id, state.version)) {
        return;
    }
    let Some(image) = images.get_mut(&coverage.handle) else {
        return;
    };
    image.data = Some(generate_coverage_map(&state));
    *last = Some((body_id, state.version));
}

/// Bake the weather state into the R8 equirect coverage grid (u = longitude,
/// v = colatitude — must match `clouds_compute.wgsl::sample_coverage`).
fn generate_coverage_map(state: &CloudWeatherState) -> Vec<u8> {
    let w = COVERAGE_WIDTH as usize;
    let h = COVERAGE_HEIGHT as usize;
    let mut data = vec![0u8; w * h];
    for row in 0..h {
        let colat = std::f32::consts::PI * (row as f32 + 0.5) / h as f32;
        let lat = std::f32::consts::FRAC_PI_2 - colat;
        let band = latitude_band_profile(lat);
        let (sin_c, cos_c) = colat.sin_cos();
        for col in 0..w {
            let lon = std::f32::consts::TAU * ((col as f32 + 0.5) / w as f32 - 0.5);
            let dir = Vec3::new(sin_c * lon.cos(), cos_c, sin_c * lon.sin());
            let n = fbm3(dir * WEATHER_NOISE_FREQ, state.seed, 4);
            let c = state.coverage_mean
                + state.band_strength * band
                + state.variation * (n - 0.5);
            data[row * w + col] = (c.clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    data
}

/// Centered (≈ [-1, 1]) latitude modulation of coverage: wetter at the ITCZ
/// and the mid-latitude storm tracks, drier in the subtropical belts and at
/// the poles. Latitudes in radians.
fn latitude_band_profile(lat: f32) -> f32 {
    let gauss = |x: f32, c: f32, wd: f32| (-((x - c) / wd) * ((x - c) / wd)).exp();
    let a = lat.abs();
    gauss(a, 0.0, 0.10) + 0.7 * gauss(a, 0.96, 0.24)
        - 0.8 * gauss(a, 0.44, 0.15)
        - 0.4 * gauss(a, std::f32::consts::FRAC_PI_2, 0.25)
}

/// Integer-mix hash → [0, 1) (no trig — stable at any coordinate).
fn hash3(p: IVec3, seed: u64) -> f32 {
    let mut h = (p.x as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (p.y as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (p.z as i64 as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
        ^ seed;
    h ^= h >> 31;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 32;
    (h & 0x00FF_FFFF) as f32 / 16_777_216.0
}

/// Trilinearly-interpolated value noise in [0, 1].
fn value_noise3(p: Vec3, seed: u64) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let i = i.as_ivec3();
    let corner = |dx: i32, dy: i32, dz: i32| hash3(i + IVec3::new(dx, dy, dz), seed);
    let x00 = corner(0, 0, 0) + (corner(1, 0, 0) - corner(0, 0, 0)) * u.x;
    let x10 = corner(0, 1, 0) + (corner(1, 1, 0) - corner(0, 1, 0)) * u.x;
    let x01 = corner(0, 0, 1) + (corner(1, 0, 1) - corner(0, 0, 1)) * u.x;
    let x11 = corner(0, 1, 1) + (corner(1, 1, 1) - corner(0, 1, 1)) * u.x;
    let y0 = x00 + (x10 - x00) * u.y;
    let y1 = x01 + (x11 - x01) * u.y;
    y0 + (y1 - y0) * u.z
}

/// Normalized fractal value noise in [0, 1].
fn fbm3(p: Vec3, seed: u64, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut norm = 0.0;
    let mut q = p;
    for _ in 0..octaves {
        sum += amp * value_noise3(q, seed);
        norm += amp;
        amp *= 0.5;
        q *= 2.17;
    }
    sum / norm
}
