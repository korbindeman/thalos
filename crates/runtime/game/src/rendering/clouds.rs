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
//! [`sync_cloud_weather_binding`] projects the per-body
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
    AU_M, CameraMatrices, CloudCompositeMaterial, CloudDistanceTexture, CloudFillCalibration,
    CloudRenderTexture, CloudShadowFrame, CloudShadowMap, CloudSurfaceDensityMap, CloudWeatherMap,
    CloudsConfig, FillCalibrationInput, LIGHT_AT_1AU, WEATHER_FACE_SIZE, derive_fill_calibration,
};
use thalos_world::{BodyId, CloudClimate};

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
///
/// This is the ONE definition. The CPU strata producer
/// (`solar_system_state::cloud_surface_density_cpu`) and the fill-calibration
/// input both read it from here — it used to be hand-copied into four places,
/// so trimming coverage moved the near tier and the far tier by different
/// amounts and the two stopped agreeing on how cloudy the planet was.
///
/// Held at 1.0: the producer's own occupancy already means areal cloud
/// fraction, so boosting it here just re-inflated coverage after the
/// distribution was fixed (2026-07-25).
pub(crate) const COVERAGE_SCALE: f32 = 1.0;
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
/// Sky-ambient fill — since the SkyAmbient binding (BL-20260724T003705Z,
/// whiteness track) these analytic constants are only the space/no-sky-LUT
/// STAND-IN the physical sky irradiance fades in over (`drive_clouds`); on a
/// surface-adjacent camera the ambient is `E_sky / π` from the F3/F4 LUT.
/// They also feed the marcher's airlight veil estimate through the same blend.
const AMBIENT_TOP_SCALE: f32 = 0.085;
const AMBIENT_BOTTOM_SCALE: f32 = 0.042;

/// Whether the cloud deck shadows the ground, and how — the receiving half of
/// CLOUD-5 / W2. Mirrors [`ContactShadowConfig`](super::contact_shadow) in
/// shape so the `cloud-shadow` capture axis behaves like the `shadow` one:
/// per-shot on the persistent host, not a boot-time `OnceLock`.
///
/// `THALOS_CLOUD_SHADOW`: `off`/`0`/`false` stands the term down (the cascade
/// still marches, so the only factor under test is whether receivers apply it);
/// `show` paints the raw transmittance on receivers; anything else applies it.
///
/// `THALOS_CLOUD_GODRAY`: `off`/`0`/`false` stands down only the atmosphere
/// march's crepuscular-shaft term (the CLOUD-5 §3.5 sky receiver) while every
/// surface receiver keeps its shadow — the isolating lever for the `godray`
/// compare axis. Anything else (or unset) applies it.
#[derive(Resource, Clone, Copy)]
pub struct CloudShadowConfig {
    pub enabled: bool,
    pub debug_show: bool,
    /// Whether the atmosphere raymarch applies the cascade to its per-sample
    /// sun term (godrays). Independent of `enabled` so one capture axis can
    /// isolate the sky term from the surface receivers, but a disabled term
    /// (`enabled = false`) zeroes the cascade strength and takes the shafts
    /// down with it — one field, one authority.
    pub shafts: bool,
}

impl Default for CloudShadowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            debug_show: false,
            shafts: true,
        }
    }
}

impl CloudShadowConfig {
    fn from_env() -> Self {
        let mut config = Self::default();
        config.apply_capture_mode(std::env::var("THALOS_CLOUD_SHADOW").ok().as_deref());
        config.apply_godray_mode(std::env::var("THALOS_CLOUD_GODRAY").ok().as_deref());
        config
    }

    pub(crate) fn apply_capture_mode(&mut self, mode: Option<&str>) {
        let shafts = self.shafts;
        *self = match mode
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "off" | "0" | "false" | "no" => Self {
                enabled: false,
                debug_show: false,
                shafts,
            },
            "show" | "raw" | "debug" => Self {
                enabled: true,
                debug_show: true,
                shafts,
            },
            _ => Self {
                shafts,
                ..Self::default()
            },
        };
    }

    pub(crate) fn apply_godray_mode(&mut self, mode: Option<&str>) {
        self.shafts = !matches!(
            mode.unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "off" | "0" | "false" | "no"
        );
    }
}

/// Which body the volumetric cloud raymarch is currently rendered for — the
/// authored cloudy body the camera is closest to, or `None` when no such body
/// exists. `ground_terrain::update_body_terrain_atmosphere` binds
/// the live cloud textures onto this body's `BodySkyMaterial` (every other
/// body keeps the blank fallback). **Sole writer:** [`drive_clouds`].
#[derive(Resource, Default, Clone, Copy)]
pub struct ActiveCloudBody(pub Option<BodyId>);

/// The exact world→body-fixed frame the near-volume marcher samples the
/// weather/strata cubes in (wind advection included). The cloud composite's
/// far tier MUST sample through this same quat: deriving a second world→body
/// rotation in `ground_terrain` misregistered the two tiers' weather fields on
/// the sphere — the far tier rendered the deck displaced, which read as "the
/// impostor shows clear sky over a solid deck" on ascent (probe-diagnosed
/// 2026-07-23). **Sole writer:** [`drive_clouds`].
#[derive(Resource, Default, Clone, Copy)]
pub struct ActiveCloudFrame(pub Quat);

/// Per-body derived near/far fill calibration (BL-20260723T214730Z): the near
/// tier's formation-threshold curve plus the far tier's opacity-response LUT,
/// derived together at body spawn by a CPU Monte-Carlo mirror of the marcher
/// over the body's actual weather cube (`fill_lut::derive_fill_calibration`).
/// `drive_clouds` feeds the threshold to the compute pass;
/// `ground_terrain::update_body_terrain_atmosphere` feeds the LUT to the
/// composite. **Sole writer:** `rendering::spawn` (at world spawn).
#[derive(Resource, Default)]
pub struct BodyCloudFill(pub std::collections::HashMap<BodyId, CloudFillCalibration>);

/// Derive the shared fill calibration for one body from its runtime weather
/// field and authored climate. Lives here so the production appearance
/// constants above stay the single source the calibration mirrors.
pub(crate) fn derive_body_fill_calibration(
    field: &crate::solar_system_state::CloudWeatherField,
    climate: &CloudClimate,
    planet_radius_m: f32,
) -> CloudFillCalibration {
    let bottom_height_m = climate.base_altitude_m.max(0.0);
    let input = FillCalibrationInput {
        weather_texels: &field.texels,
        strata_texels: &field.surface_density_texels,
        face_size: field.face_size,
        coverage_scale: COVERAGE_SCALE,
        density: DENSITY * climate.density.max(0.0),
        detail_strength: DETAIL_STRENGTH,
        base_edge_softness: EDGE_SOFTNESS,
        bottom_softness: BOTTOM_SOFTNESS,
        base_shape_scale_m: climate.base_shape_scale_m.max(500.0),
        detail_scale_m: climate.detail_scale_m.max(50.0),
        bottom_height_m,
        top_height_m: (climate.base_altitude_m + climate.thickness_m).max(bottom_height_m + 1.0),
        planet_radius_m,
        seed: field.seed,
    };
    // The Monte-Carlo derivation is a pure function of `input` (~4 s of boot
    // on the dev box, BL-20260724T153620Z), so it disk-caches keyed by a hash
    // of every input plus `FILL_LUT_VERSION` — algorithm changes must bump
    // that constant in `fill_lut.rs` or a cached run calibrates yesterday's
    // renderer.
    let key = fill_lut_cache::key(&input);
    if let Some(calibration) = fill_lut_cache::load(&key) {
        info!(
            target: "thalos::clouds",
            threshold_nodes = ?calibration.threshold_nodes,
            "cloud fill calibration loaded from cache ({key}.json)"
        );
        return calibration;
    }
    let calibration = derive_fill_calibration(&input);
    info!(
        target: "thalos::diagnostic::clouds",
        event = "fill_calibration_derived",
        threshold_nodes = ?calibration.threshold_nodes,
        far_response = ?calibration.far_response,
        "cloud fill calibration derived"
    );
    fill_lut_cache::store(&key, &calibration);
    calibration
}

/// Tiny disk cache for [`CloudFillCalibration`] (three fixed f32 arrays as
/// JSON), mirroring the tile cache's location split: project-local `user/`
/// in debug, OS app-data in release. `THALOS_CLOUD_LUT_CACHE=0` disables it
/// while iterating on `fill_lut.rs` itself.
mod fill_lut_cache {
    use super::CloudFillCalibration;
    use bevy::log::warn;
    use serde::{Deserialize, Serialize};
    use std::hash::{Hash, Hasher};
    use std::path::PathBuf;

    #[derive(Serialize, Deserialize)]
    struct CachedCalibration {
        threshold_nodes: Vec<f32>,
        far_response: Vec<f32>,
        far_cell_edge: Vec<f32>,
        far_cell_solid: Vec<f32>,
    }

    pub(super) fn key(input: &thalos_body_render::FillCalibrationInput<'_>) -> String {
        let mut hasher = std::hash::DefaultHasher::new();
        thalos_body_render::FILL_LUT_VERSION.hash(&mut hasher);
        input.weather_texels.hash(&mut hasher);
        input.strata_texels.hash(&mut hasher);
        input.face_size.hash(&mut hasher);
        for scalar in [
            input.coverage_scale,
            input.density,
            input.detail_strength,
            input.base_edge_softness,
            input.bottom_softness,
            input.base_shape_scale_m,
            input.detail_scale_m,
            input.bottom_height_m,
            input.top_height_m,
            input.planet_radius_m,
        ] {
            scalar.to_bits().hash(&mut hasher);
        }
        input.seed.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    fn enabled() -> bool {
        !std::env::var("THALOS_CLOUD_LUT_CACHE").is_ok_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
    }

    fn cache_path(key: &str) -> PathBuf {
        #[cfg(debug_assertions)]
        let root = PathBuf::from("user/cloudfill");
        #[cfg(not(debug_assertions))]
        let root = bevy::platform::dirs::preferences_dir()
            .map(|dir| dir.join("thalos").join("cloudfill"))
            .unwrap_or_else(|| PathBuf::from("user/cloudfill"));
        root.join(format!("{key}.json"))
    }

    pub(super) fn load(key: &str) -> Option<CloudFillCalibration> {
        if !enabled() {
            return None;
        }
        let cached: CachedCalibration =
            serde_json::from_str(&std::fs::read_to_string(cache_path(key)).ok()?).ok()?;
        Some(CloudFillCalibration {
            threshold_nodes: cached.threshold_nodes.try_into().ok()?,
            far_response: cached.far_response.try_into().ok()?,
            far_cell_edge: cached.far_cell_edge.try_into().ok()?,
            far_cell_solid: cached.far_cell_solid.try_into().ok()?,
        })
    }

    pub(super) fn store(key: &str, calibration: &CloudFillCalibration) {
        if !enabled() {
            return;
        }
        let path = cache_path(key);
        let cached = CachedCalibration {
            threshold_nodes: calibration.threshold_nodes.to_vec(),
            far_response: calibration.far_response.to_vec(),
            far_cell_edge: calibration.far_cell_edge.to_vec(),
            far_cell_solid: calibration.far_cell_solid.to_vec(),
        };
        let write = || -> std::io::Result<()> {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, serde_json::to_string(&cached)?.as_bytes())
        };
        if let Err(error) = write() {
            warn!(target: "thalos::clouds", "could not store fill-calibration cache: {error}");
        }
    }
}

/// Per-body spawn-uploaded weather/strata cube handles (weather, strata).
/// EVERY consumer — the near-volume compute pass included — must sample these
/// initial-upload cubes: the runtime re-upload path (both `image.data`
/// mutation and wholesale asset replacement) scrambles cube face/mip layout
/// on the GPU, so a runtime-refreshed copy silently diverges from the field
/// every other consumer reads (BL-20260723T214730Z — the near volumetrics
/// flew through a corrupted field while the impostor showed the correct one).
/// Populated at body spawn; consumed by [`sync_cloud_weather_binding`].
#[derive(Resource, Default)]
pub struct BodyCloudCubes(pub std::collections::HashMap<BodyId, (Handle<Image>, Handle<Image>)>);

pub struct CloudsRenderPlugin;

impl Plugin for CloudsRenderPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<CloudsConfig>()
            .register_type::<CameraMatrices>()
            .insert_resource(CloudShadowConfig::from_env())
            .init_resource::<ActiveCloudBody>()
            .init_resource::<ActiveCloudFrame>()
            .init_resource::<BodyCloudCubes>()
            .init_resource::<BodyCloudFill>()
            .add_systems(bevy::app::PostStartup, init_cloud_appearance)
            .add_systems(
                Update,
                (
                    sync_cloud_weather_binding,
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

/// Match the cloud compositor's lifecycle to the body's near-surface
/// projection, by mirroring its `BodySky` visibility.
///
/// `BodySky` — not the udlod `BodyTerrain` entity — is the right authority:
/// `sync_body_render_lod` owns the single surface-LOD handoff (is ground
/// resident, and is the camera inside the swap radius) and already answers it
/// for *both* renderers, udlod atlas and NTR-X1 tiles. Keying the composite off
/// `BodyTerrain` instead meant that on the tile path — where udlod stands down
/// and no `BodyTerrain` entity exists — the composite never left `Hidden` and
/// the body had no clouds at all, while its sky and ocean rendered normally.
/// Every body that spawns a cloud composite also spawns a `BodySky` (both live
/// in the has-atmosphere branch of `spawn`), so the mirror is total.
fn sync_cloud_composite_visibility(
    skies: Query<(&super::ground_terrain::BodySky, &Visibility), Without<BodyClouds>>,
    mut composites: Query<(&BodyClouds, &mut Visibility), Without<super::ground_terrain::BodySky>>,
) {
    let visible_bodies: std::collections::HashSet<BodyId> = skies
        .iter()
        .filter_map(|(sky, visibility)| (*visibility != Visibility::Hidden).then_some(sky.body_id))
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
    frame: Res<ActiveCloudFrame>,
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
            // Registration by construction: the far tier samples the weather
            // sphere through the marcher's exact frame, not ground_terrain's
            // independently derived rotation (see `ActiveCloudFrame`).
            let q = frame.0;
            bevy::log::info_once!(
                target: "thalos::diagnostic::clouds",
                event = "composite_frame_override",
                marcher_orientation = ?q,
                ground_orientation = ?material.params.world_to_body_orientation,
                "cloud composite frame override"
            );
            material.params.world_to_body_orientation = Vec4::new(q.x, q.y, q.z, q.w);
            if let Some(layer) = cloud_layer.as_deref() {
                material.cloud_layer = layer.handle.clone();
            }
            if let Some(distance) = cloud_distance.as_deref() {
                material.cloud_distance = distance.handle.clone();
            }
            // The composite keeps the spawn-time cube bindings it was built
            // with — the same spawn-uploaded cubes `sync_cloud_weather_binding`
            // rebinds the compute pass onto, so every consumer reads the one
            // correctly-uploaded field (INC-20260723T221126Z: runtime cube
            // re-upload scrambles face/mip layout on the GPU).
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
pub(super) fn drive_clouds(
    ship_cam_q: Query<(&GlobalTransform, &Camera), With<ShipCamera>>,
    body_q: Query<(&RealSpaceBody, &GlobalTransform)>,
    sim: Res<SimulationState>,
    cache: Res<SolarSystemState>,
    exposure: Res<CameraExposure>,
    graphics: Res<GraphicsSettings>,
    fill: Res<BodyCloudFill>,
    sky_ambient: Res<crate::reflection_probe::SkyAmbient>,
    mut active: ResMut<ActiveCloudBody>,
    mut frame: ResMut<ActiveCloudFrame>,
    mut cam_mat: ResMut<CameraMatrices>,
    mut config: ResMut<CloudsConfig>,
    time: Res<Time>,
    mut wind_angle: Local<f32>,
    mut history_continuity: Local<Option<(BodyId, u32, f64)>>,
    // Bundled: `drive_clouds` sits at Bevy's 16-param ceiling.
    // .0 = the published cloud sun-transmittance cascade (CLOUD-5 / W2),
    // .1 = whether receivers apply it (the `cloud-shadow` capture axis).
    mut cloud_shadow_io: (ResMut<CloudShadowMap>, Res<CloudShadowConfig>),
) {
    let (ref mut cloud_shadow, ref shadow_config) = cloud_shadow_io;
    // Stand the sun-transmittance cascade down FIRST, and let the success path
    // below raise it again. Every early-out here (clouds off, no camera, no
    // cloud body, no ephemeris) means the map is about to go stale, and a stale
    // cloud shadow is worse than none: it would keep darkening ground under a
    // deck that is no longer being rendered.
    config.shadow_frame = CloudShadowFrame::default();
    cloud_shadow.frame = CloudShadowFrame::default();

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
    // Cell-scale cloud evolution phase. SIM time, so captures stay
    // reproducible and time warp accelerates the weather like everything else.
    //
    // Deliberately NOT wrapped. A modulo was tried and removed: the cell field
    // is hash-keyed lattice value noise and therefore aperiodic, so no wrap
    // period returns it to itself and every candidate one is a visible jump in
    // the sky. Raw sim time is safe here because Thalos scenarios boot at
    // time-of-day epochs (~1e4–1e6 s), where f32 still resolves ~0.1 s;
    // `evolution_phase_resolves_at_sim_epochs` pins that headroom.
    config.cell_evolution_s = sim_time as f32;
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
    // Publish the marcher's frame so the composite's far tier samples the
    // weather sphere through the IDENTICAL rotation (see `ActiveCloudFrame`).
    frame.0 = q_bw;
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
    // Derived formation-threshold curve (see `BodyCloudFill`). Missing
    // calibration on a cloudy body means the spawn path skipped derivation —
    // the near fill would silently fall back to stale constants, so shout.
    if let Some(calibration) = fill.0.get(&body_id) {
        config.fill_threshold_nodes = calibration.threshold_vec4s();
    } else {
        bevy::log::warn_once!(
            target: "thalos::clouds",
            "active cloud body {body_id} has no derived fill calibration; \
             using default threshold curve"
        );
    }
    config.wind_velocity = Vec3::new(climate.wind_m_s[0], climate.wind_m_s[1], 0.0);
    config.sun_dir = Vec4::new(sun_body.x, sun_body.y, sun_body.z, 0.0);

    // Sun-transmittance cascade (CLOUD-5 / W2): resolve it from the SAME
    // body-fixed camera position, sun direction, and radius the marcher above
    // was just handed, so the shadow field and the visible deck are two
    // projections of one state rather than two derivations of it.
    let shadow_frame = CloudShadowFrame::resolve(cam_body, sun_body, radius);
    config.shadow_frame = shadow_frame;
    cloud_shadow.frame = shadow_frame;
    cloud_shadow.world_to_body = q_bw;
    cloud_shadow.body_center_ws = planet_center;
    cloud_shadow.sun_body = sun_body;
    cloud_shadow.strength = f32::from(u8::from(shadow_config.enabled));
    cloud_shadow.debug_show = shadow_config.debug_show;
    if shadow_config.debug_show {
        bevy::log::info_once!(
            target: "thalos::diagnostic::clouds",
            event = "shadow_cascade",
            active = shadow_frame.active,
            half_extent_m = shadow_frame.half_extent_m,
            texel_m = shadow_frame.texel_m(thalos_body_render::CLOUD_SHADOW_SIZE),
            sun_elevation_cos = shadow_frame.sun_elevation_cos,
            altitude_m = cam_body.length() - radius,
            center_radius_m = shadow_frame.center.length(),
            body_center_world_radius_m = planet_center.length(),
            "cloud shadow cascade"
        );
    }
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
    // albedo. It carries CHROMA only — the reflectance magnitude that makes a
    // lit cloud white is `CLOUD_MS_ALBEDO` in the marcher, and peak headroom
    // is the marcher's (achromatic) Reinhard white point. The former extra
    // `(0.94, 0.96, 0.99)` headroom factor stacked on Thalos's authored
    // (0.94, 0.96, 1.0) and biased sunlit cloud ~12% blue.
    let sun_rgb =
        sun_chromaticity * cloud_albedo * scene_flux * SUN_FLUX_SCALE * horizon_transmittance;
    config.sun_color = Vec4::new(sun_rgb.x, sun_rgb.y, sun_rgb.z, 1.0);
    // Ambient in-scatter. The PHYSICAL source is the F3/F4 sky-view LUT's
    // hemispherical irradiance (`SkyAmbient`, scene-flux units — the same
    // authority the surface `GlobalAmbientLight` and env cubemap read):
    // mean incident sky radiance L ≈ E_sky / π, seen fully by cloud tops and
    // partially (view factor ~0.45) by shaded undersides. The user-flagged
    // "uniformly gray clouds" came in large part from the old hand-tuned
    // flat ambient pair sitting ~2× below the physical sky radiance (and
    // being far less blue); the reference systems' interiors read luminous
    // for exactly this reason (Blackrack raymarches ambient toward the sky).
    // The analytic pair remains only as the space/no-LUT stand-in, faded by
    // the same altitude blend every other F4 consumer uses.
    let horizon_ambient = 0.28 + 0.72 * day_blend;
    let analytic_top =
        Vec3::new(0.42, 0.58, 0.88) * scene_flux * AMBIENT_TOP_SCALE * horizon_ambient;
    let analytic_bottom =
        Vec3::new(0.30, 0.36, 0.48) * scene_flux * AMBIENT_BOTTOM_SCALE * horizon_ambient;
    let sky_radiance = sky_ambient.surface_irradiance / std::f32::consts::PI;
    let sky_blend = sky_ambient.surface_blend.clamp(0.0, 1.0);
    let ambient_top = analytic_top.lerp(sky_radiance, sky_blend);
    let ambient_bottom = analytic_bottom.lerp(sky_radiance * 0.45, sky_blend);
    config.clouds_ambient_color_top = ambient_top.extend(0.0);
    config.clouds_ambient_color_bottom = ambient_bottom.extend(0.0);
}

/// Point the near-volume compute pass at the ACTIVE body's spawn-uploaded
/// weather/strata cubes. This is a pure HANDLE REBIND — no image data is ever
/// mutated at runtime, because the re-upload path scrambles cube face/mip
/// layout on the GPU (BL-20260723T214730Z): the previous in-place
/// `image.data` refresh fed the marcher a corrupted field, so the volumetrics
/// could never line up with the impostor's correct spawn-time field. The
/// compute bind group is rebuilt from `CloudsImage` every frame, so the swap
/// takes effect on the next frame. Future live weather (CLOUD-7 advection)
/// must create a NEW image asset per version rather than mutating in place.
fn sync_cloud_weather_binding(
    active: Res<ActiveCloudBody>,
    cubes: Res<BodyCloudCubes>,
    cache: Res<SolarSystemState>,
    clouds_image: Option<ResMut<thalos_body_render::CloudsImage>>,
    weather: Option<ResMut<CloudWeatherMap>>,
    surface_density: Option<ResMut<CloudSurfaceDensityMap>>,
    mut last: Local<Option<BodyId>>,
) {
    let Some(body_id) = active.0 else {
        return;
    };
    if *last == Some(body_id) {
        return;
    }
    let Some((weather_handle, strata_handle)) = cubes.0.get(&body_id) else {
        return;
    };
    if let Some(field) = cache
        .environment
        .get(body_id)
        .and_then(|env| env.cloud_weather.as_ref())
        && field.face_size != WEATHER_FACE_SIZE
    {
        error!(
            target: "thalos::clouds",
            "weather field face size {} does not match renderer {}",
            field.face_size,
            WEATHER_FACE_SIZE,
        );
        return;
    }
    let Some(mut clouds_image) = clouds_image else {
        return;
    };
    clouds_image.weather_image = weather_handle.clone();
    clouds_image.surface_density_image = strata_handle.clone();
    if let Some(mut weather) = weather {
        weather.handle = weather_handle.clone();
    }
    if let Some(mut surface_density) = surface_density {
        surface_density.handle = strata_handle.clone();
    }
    *last = Some(body_id);
}
