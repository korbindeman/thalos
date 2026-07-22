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
//! **Compositing.** One dedicated [`thalos_body_render::CloudCompositeMaterial`]
//! owns both the near-volume layer and the weather-derived orbital projection.
//! It renders after the canonical `BodySky` atmosphere without acquiring a
//! second cloud path. [`sync_cloud_composite_materials`] mirrors the active body's
//! planet/light state and binds the live textures each frame.

use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::transform::TransformSystems;

use thalos_body_render::{
    AU_M, CameraMatrices, CloudCompositeMaterial, CloudDistanceTexture, CloudRenderTexture,
    CloudSurfaceDensityMap, CloudWeatherMap, CloudsConfig, LIGHT_AT_1AU, WEATHER_FACE_SIZE,
};
use thalos_world::BodyId;

use crate::camera::ShipCamera;
use crate::graphics_settings::GraphicsSettings;

use super::types::{CameraExposure, RealSpaceBody, SimulationState, SolarSystemState};

/// Per-body fullscreen cloud projection. Separate from `BodySky` so cloud
/// ownership and visibility remain independent of the atmosphere material.
#[derive(Component, Debug)]
pub(super) struct BodyClouds {
    pub(super) body_id: BodyId,
}

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
/// Sky-ambient fill. Raised for the CLOUD-4 lighting completion: with the
/// former ~10:1 sun:ambient ratio, anything the (now multi-octave) shadow
/// term attenuated fell to near-black and shaded cores read charcoal instead
/// of soft blue-grey. These also feed the marcher's airlight veil estimate.
const AMBIENT_TOP_SCALE: f32 = 0.085;
const AMBIENT_BOTTOM_SCALE: f32 = 0.042;

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
            .add_systems(
                Update,
                (
                    sync_cloud_weather_map,
                    sync_cloud_composite_visibility
                        .after(super::ground_terrain::sync_body_render_lod),
                ),
            )
            .add_systems(
                bevy::app::PostUpdate,
                (
                    drive_clouds.after(TransformSystems::Propagate),
                    sync_cloud_composite_materials
                        .after(drive_clouds)
                        .after(super::ground_terrain::update_body_terrain_atmosphere),
                ),
            );
    }
}

/// Match the cloud compositor's lifecycle to the resident terrain projection,
/// independently of `BodySky`.
fn sync_cloud_composite_visibility(
    terrains: Query<(&super::ground_terrain::BodyTerrain, &Visibility), Without<BodyClouds>>,
    mut composites: Query<
        (&BodyClouds, &mut Visibility),
        Without<super::ground_terrain::BodyTerrain>,
    >,
) {
    let visible_bodies: std::collections::HashSet<BodyId> = terrains
        .iter()
        .filter_map(|(terrain, visibility)| {
            (*visibility != Visibility::Hidden).then_some(terrain.body_id)
        })
        .collect();
    for (clouds, mut visibility) in &mut composites {
        let want = if visible_bodies.contains(&clouds.body_id) {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *visibility != want {
            *visibility = want;
        }
    }
}

/// Mirror the active body's already-resolved planet/sun/orientation state from
/// its custom sky material, but bind the cloud textures to the dedicated
/// compositor. This is the one cloud-composition writer; `BodySky` retains the
/// atmosphere projection.
fn sync_cloud_composite_materials(
    active: Res<ActiveCloudBody>,
    cloud_layer: Option<Res<CloudRenderTexture>>,
    cloud_distance: Option<Res<CloudDistanceTexture>>,
    blanks: Option<Res<super::spawn::BlankCloudTextures>>,
    skies: Query<(
        &super::ground_terrain::BodySky,
        &MeshMaterial3d<thalos_body_render::BodySkyMaterial>,
    )>,
    composites: Query<(&BodyClouds, &MeshMaterial3d<CloudCompositeMaterial>)>,
    sky_materials: Res<Assets<thalos_body_render::BodySkyMaterial>>,
    mut cloud_materials: ResMut<Assets<CloudCompositeMaterial>>,
) {
    let sky_state: std::collections::HashMap<BodyId, _> = skies
        .iter()
        .filter_map(|(sky, handle)| {
            sky_materials.get(handle).map(|material| {
                (
                    sky.body_id,
                    (material.atmosphere, material.atmosphere_extra),
                )
            })
        })
        .collect();

    for (clouds, handle) in &composites {
        let Some((atmosphere, params)) = sky_state.get(&clouds.body_id).copied() else {
            continue;
        };
        let Some(mut material) = cloud_materials.get_mut(handle) else {
            continue;
        };
        material.atmosphere = atmosphere;
        material.params = params;
        if active.0 == Some(clouds.body_id) {
            if let Some(layer) = cloud_layer.as_deref() {
                material.cloud_layer = layer.handle.clone();
            }
            if let Some(distance) = cloud_distance.as_deref() {
                material.cloud_distance = distance.handle.clone();
            }
        } else if let Some(blanks) = blanks.as_deref() {
            material.cloud_layer = blanks.layer.clone();
            material.cloud_distance = blanks.distance.clone();
        }
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
    // Filtered multi-tap sun depth (CLOUD-4). The old single unfiltered probe
    // keyed the whole direct term on ~55 m erosion noise (cellular charcoal
    // mottling); the historical banding of fixed multi-tap ladders is handled
    // in-shader by sampling only the smooth broad mass and jittering the tap
    // ladder per pixel (`volumetric_sun_depth`).
    config.clouds_shadow_raymarch_steps_count = 3;
    config.clouds_shadow_raymarch_step_size = 700.0;
    config.clouds_shadow_raymarch_step_multiply = 2.0;
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
    let atmosphere = sim.system.bodies[body_id]
        .terrestrial_atmosphere
        .as_ref()
        .expect("active cloud body was filtered by terrestrial atmosphere");
    let climate = atmosphere
        .clouds
        .as_ref()
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
    config.cloud_albedo = cloud_albedo.extend(1.0);

    // CLOUD-4 first slice: camera-local day factor still sets the baseline
    // sun/ambient chromaticity; the volume shader then multiplies a per-sample
    // air-mass transmittance so low-sun and low-altitude samples redden further.
    // Finished CLOUD-4 swaps this CPU guess for shared atmosphere LUT samples.
    let local_up = cam_body.normalize_or_zero();
    let sun_mu = local_up.dot(sun_body);
    let day_t = ((sun_mu + 0.04) / 0.28).clamp(0.0, 1.0);
    let day_blend = day_t * day_t * (3.0 - 2.0 * day_t);
    // Low-sun reddening is owned by the shader's per-sample air-mass
    // transmittance (`atmosphere_sun_transmittance`); this CPU chromaticity
    // only nudges the baseline. The former deep-amber floor stacked ON TOP of
    // that per-sample term, double-reddening every low-sun cloud into mud.
    let sun_chromaticity = Vec3::new(1.0, 0.84, 0.72).lerp(Vec3::new(1.0, 0.97, 0.93), day_blend);
    let horizon_transmittance = 0.85 + 0.15 * day_blend;
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
    surface_density: Option<Res<CloudSurfaceDensityMap>>,
    mut images: ResMut<Assets<Image>>,
    mut last: Local<Option<(BodyId, u32)>>,
) {
    let Some(body_id) = active.0 else {
        return;
    };
    let Some(weather) = weather else {
        return;
    };
    let Some(surface_density) = surface_density else {
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
    {
        let Some(mut image) = images.get_mut(&weather.handle) else {
            return;
        };
        image.data = Some(field.rgba8_mip_chain());
    }
    {
        let Some(mut image) = images.get_mut(&surface_density.handle) else {
            return;
        };
        image.data = Some(field.surface_density_rgba8_mip_chain());
    }
    *last = Some((body_id, field.version));
}
