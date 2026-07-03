//! CPU-authored environment map for ship reflections.
//!
//! Maintains a small cubemap painted from CPU code — sun, planet,
//! stars — and feeds it to the main camera via
//! [`GeneratedEnvironmentMapLight`]. Bevy's realtime filter pipeline
//! prefilters it into diffuse + specular mips every time the image
//! asset is marked changed, so metallic ship parts get IBL reflections
//! that respond to orbital state.
//!
//! See `docs/reflection_probe.md` for the full design note — why this
//! is CPU-painted rather than a 6-camera render of the actual scene,
//! the Bevy 0.18 `camera_system` trap that blocks the "correct" path,
//! upstream status (PR #13840), and the migration plan when it lands.
//!
//! # Why CPU-authored rather than rendering the real scene?
//!
//! Rendering the actual scene into a 6-face cubemap in Bevy 0.18
//! requires crossing the main-world / render-world boundary to manage
//! per-face `TextureView`s, which is fragile and ate more time than it
//! was worth. A CPU-painted env map uses only stable Bevy APIs and
//! gives us mirror reflections of the key orbital features (sun,
//! planet disc, star background) that look right from low-orbit.
//! Upgrading to real-scene capture later is a drop-in replacement:
//! same `Image` handle, different writer.
//!
//! # Update cadence
//!
//! The cubemap is rewritten every [`REFRESH_INTERVAL`] game-time
//! seconds. Orbital angular rates near Thalos are on the order of
//! 1e-3 rad/s, so a 0.25 s refresh is well under the threshold at
//! which the eye can pick up staleness in a reflection.

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::light::GeneratedEnvironmentMapLight;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};

use crate::camera::OrbitCamera;
use crate::rendering::{CameraExposure, SimulationState};
use thalos_body_render::{
    AU_M, AtmosphereBlock, LIGHT_AT_1AU, MULTI_SCATTER_LUT_HEIGHT, MULTI_SCATTER_LUT_WIDTH,
    MultiScatterLut, SkyViewLut,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Cubemap face resolution. 256 balances reflection sharpness against
/// CPU write cost (each refresh touches 6 × 256² = ~400k texels).
const PROBE_SIZE: u32 = 256;

/// **Real** seconds between considering a cubemap refresh. This bounds how
/// stale the reflected sky + the published [`SkyAmbient`] can be in *real* time —
/// under high warp a single interval can span sim-hours of sun motion, so the
/// painted time-of-day lags by up to one interval after a warp change. 2 s keeps
/// that window short; the repaint itself (LUT bake + 256²×6 CPU paint) is a few
/// ms, and the change-gate below still skips repaints when nothing moved.
const REFRESH_INTERVAL: f32 = 2.0;
/// Refresh interval while warping fast (sim time racing): tighter, so the
/// painted time-of-day tracks the fast-moving sun with less visible desync.
/// The change-gate still suppresses repaints that wouldn't change anything.
const REFRESH_INTERVAL_WARP: f32 = 0.5;
/// Warp factor above which [`REFRESH_INTERVAL_WARP`] applies.
const WARP_FAST_THRESHOLD: f64 = 50.0;

/// Sim-seconds of clock drift that force a repaint even when the direction
/// gates don't fire. Catches time-of-day creep the four direction thresholds
/// can miss (and any state change they don't encode) — the sun moves ~0.5°/min
/// of sim time on a 12 h day, so 120 sim-s keeps the painted sky within a
/// degree of the real one at any warp factor.
const REPAINT_SIM_DRIFT_S: f64 = 120.0;

/// Below these thresholds the env hasn't moved enough for a re-paint to
/// produce a visibly different cubemap, so we skip the `images.get_mut`
/// (which would mark the asset changed and re-trigger the IBL prefilter).
/// 0.9999 ≈ 0.81° angular drift on a unit direction; 1e-4 on `planet_cos`
/// ≈ a fraction of a percent change in the planet's angular radius.
const ENV_DIR_DOT_MIN: f32 = 0.9999;
const ENV_COS_EPS: f32 = 1.0e-4;

/// Environment map intensity multiplier handed to
/// [`GeneratedEnvironmentMapLight`]. 1.0 matches scene luminance;
/// bump if reflections read too dark on polished metals.
const PROBE_INTENSITY: f32 = 1.0;

/// Bright HDR gain for the reflected sun disc, in scene-flux units. `flux ≈ 10`
/// at the homeworld → ~30 (matching the old flat sun_color), but now scaling
/// with heliocentric distance + exposure so a far/dim sun reflects dimmer.
const SUN_DISC_GAIN: f32 = 3.0;

// ── Physical surface sky (graphics-fidelity F3) ───────────────────────────────
// The reflection cubemap is CPU-painted (the GPU cubemap-render path is blocked —
// see docs/atmosphere.md), so the sky it reflects is evaluated on the CPU. The
// *sky* upper hemisphere is now the physical `SkyViewLut` (a raymarch of the same
// single+multi-scatter model the terrain shades through), replacing the former
// hand-kept `cpu_surface_sky` analytic mirror of the WGSL `compute_surface_sky` —
// so the metallic hull reflects the SAME atmosphere-derived sky the terrain is
// lit by (one atmosphere, one environment), with no CPU/WGSL drift hazard to keep
// in lockstep. The two terms the sky raymarch does NOT provide — a warm terrain
// ground-bounce (lower hemisphere) and the direct-beam sun-disc reddening — stay
// analytic below (`surface_ground_sun`).
const SCENE_FLUX_SCALE: f32 = 0.5;
const SURFACE_GROUND_ALBEDO: Vec3 = Vec3::new(0.10, 0.085, 0.055);
const SURFACE_GROUND_SCALE: f32 = 0.10;
const SURFACE_NIGHT_AMBIENT: Vec3 = Vec3::new(0.008, 0.010, 0.014);

/// Sky-view LUT resolution baked for the reflection probe. Smaller than the GPU
/// reference (Hillaire 192×108) because Bevy prefilters the cubemap into diffuse
/// SH + rough specular, which blurs away fine sky detail anyway; keeps the
/// per-repaint CPU raymarch cost bounded (~48×64 view rays every refresh).
const PROBE_SKY_LUT_W: u32 = 48;
const PROBE_SKY_LUT_H: u32 = 64;

/// Calibration gain from the physically-raymarched `SkyViewLut` radiance (already
/// in scene-flux units) into the reflected sky. `1.0` is the physical baseline;
/// nudge from a `just game runway` / `landing` screenshot if the reflected sky
/// reads too bright/dim against the terrain it is meant to match.
const PHYSICAL_SKY_SCALE: f32 = 1.0;

fn vec3_exp(v: Vec3) -> Vec3 {
    Vec3::new(v.x.exp(), v.y.exp(), v.z.exp())
}

/// The two surface-sky terms the `SkyViewLut` raymarch does not produce: the warm
/// terrain ground-bounce filling the lower hemisphere, and the reddened
/// direct-beam sun tint for the sun disc.
struct SurfaceGroundSun {
    ground_radiance: Vec3,
    sun_color: Vec3,
}

/// Analytic warm ground-bounce (lower hemisphere) + reddened direct-beam sun
/// tint, from the vertical Rayleigh optical depth `tau_zenith`, artistic
/// `strength`, sun elevation, and per-fragment flux. The *sky* itself is the
/// physical `SkyViewLut`; these are the terrain-bounce and direct-disc terms it
/// does not model.
fn surface_ground_sun(tau_zenith: Vec3, strength: f32, sun_elev: f32, flux: f32) -> SurfaceGroundSun {
    let scene_radiance = flux.max(0.0) * SCENE_FLUX_SCALE;
    let sun_up = sun_elev.clamp(0.0, 1.0);
    let tau_eff = tau_zenith.max(Vec3::ZERO) * strength.max(0.0);
    let airmass = (1.0 / (sun_up + 0.10)).clamp(1.0, 8.0);
    let sun_color = vec3_exp(-tau_eff * (airmass - 1.0));
    let ground_radiance = SURFACE_GROUND_ALBEDO * (scene_radiance * SURFACE_GROUND_SCALE * sun_up)
        + SURFACE_NIGHT_AMBIENT;
    SurfaceGroundSun {
        ground_radiance,
        sun_color,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Physical sky-fill ambient published from the F3 sky-view LUT, consumed by
/// `rendering::lighting::update_sun_light` to drive the surface-regime
/// `GlobalAmbientLight` (graphics-fidelity **F4** — replacing the hand-tuned flat
/// day-ambient constant). `surface_irradiance` is the cosine-weighted
/// hemispherical sky irradiance in scene-flux units (the SH DC term); it already
/// encodes time-of-day + sun elevation + atmosphere. `surface_blend` is the same
/// altitude ramp the env cubemap uses (1 = on the surface, 0 = space), so the
/// consumer can fade physical sky ambient against the unchanged space stand-in.
///
/// **Sole writer:** [`refresh_cubemap`] (this module).
#[derive(Resource, Default, Clone, Copy)]
pub struct SkyAmbient {
    pub surface_irradiance: Vec3,
    pub surface_blend: f32,
}

pub struct ReflectionProbePlugin;

impl Plugin for ReflectionProbePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProbeRefreshTimer>()
            .init_resource::<SkyAmbient>()
            .add_systems(Startup, setup_probe)
            .add_systems(Update, (attach_env_map_to_main_camera, refresh_cubemap));
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

#[derive(Resource, Clone)]
struct ReflectionProbe {
    cubemap: Handle<Image>,
}

#[derive(Resource)]
struct ProbeRefreshTimer {
    elapsed: f32,
    first_fill_done: bool,
    /// Last `EnvParams` the cubemap was painted from. Used by the
    /// change-detection gate to skip painting when the env hasn't
    /// shifted enough to produce a visibly different cubemap.
    last_painted: Option<EnvParams>,
    /// Sim time (s) of the last paint — repaints are forced once the sim clock
    /// drifts by [`REPAINT_SIM_DRIFT_S`] so warp can't leave the painted
    /// time-of-day hours behind the world's.
    last_paint_sim_time: f64,
    /// Cached multi-scatter LUT keyed by dominant body id. The LUT depends only
    /// on the (static) atmosphere, so it is baked once per body and reused across
    /// sun/altitude changes; only the view-dependent `SkyViewLut` rebakes.
    ms_cache: Option<(usize, MultiScatterLut)>,
}

impl Default for ProbeRefreshTimer {
    fn default() -> Self {
        Self {
            elapsed: REFRESH_INTERVAL, // force an update on frame 1
            first_fill_done: false,
            last_painted: None,
            last_paint_sim_time: f64::NEG_INFINITY,
            ms_cache: None,
        }
    }
}

fn setup_probe(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    // Six-layer 2D image with a cube view descriptor is how Bevy's PBR
    // stack expects an environment-map source. `Rgba16Float` keeps
    // HDR headroom for the sun without banding on the planet gradient.
    let mut image = Image::new_fill(
        Extent3d {
            width: PROBE_SIZE,
            height: PROBE_SIZE,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        &[0u8; 8], // one black `Rgba16Float` texel, broadcast
        TextureFormat::Rgba16Float,
        RenderAssetUsages::all(),
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        label: Some("reflection_probe_cube_view"),
        dimension: Some(TextureViewDimension::Cube),
        ..Default::default()
    });

    let cubemap = images.add(image);
    commands.insert_resource(ReflectionProbe { cubemap });
}

/// Install the env-map component on the main camera once it exists.
/// Idempotent via the `Without` filter — after the first successful
/// match, the query iterates nothing.
fn attach_env_map_to_main_camera(
    mut commands: Commands,
    probe: Option<Res<ReflectionProbe>>,
    cam: Query<Entity, (With<OrbitCamera>, Without<GeneratedEnvironmentMapLight>)>,
) {
    let Some(probe) = probe else { return };
    for e in cam.iter() {
        commands.entity(e).insert(GeneratedEnvironmentMapLight {
            environment_map: probe.cubemap.clone(),
            intensity: PROBE_INTENSITY,
            rotation: Quat::IDENTITY,
            affects_lightmapped_mesh_diffuse: true,
        });
    }
}

/// Rewrite the cubemap pixels from the current ship-relative sun /
/// planet directions. Marks the asset changed so Bevy re-uploads and
/// `GeneratedEnvironmentMapLight` re-filters.
///
/// Two gates suppress the work:
/// 1. Time-based: only consider painting every `REFRESH_INTERVAL` seconds.
/// 2. Change-detection: even when the timer fires, skip the actual
///    `images.get_mut` if the env hasn't moved beyond
///    `ENV_DIR_DOT_MIN` / `ENV_COS_EPS` since the last paint. The
///    `get_mut` is the asset-changed trigger; not calling it skips the
///    downstream IBL prefilter pass entirely.
fn refresh_cubemap(
    time: Res<Time<Real>>,
    mut timer: ResMut<ProbeRefreshTimer>,
    probe: Option<Res<ReflectionProbe>>,
    mut images: ResMut<Assets<Image>>,
    sim: Option<Res<SimulationState>>,
    exposure: Option<Res<CameraExposure>>,
    mut sky_ambient: ResMut<SkyAmbient>,
) {
    let Some(probe) = probe else { return };

    // Tighter cadence while warping fast, so the painted time-of-day tracks the
    // racing sun instead of visibly desyncing until warp stops.
    let interval = if sim
        .as_deref()
        .map(|s| s.simulation.warp.speed() > WARP_FAST_THRESHOLD)
        .unwrap_or(false)
    {
        REFRESH_INTERVAL_WARP
    } else {
        REFRESH_INTERVAL
    };
    timer.elapsed += time.delta_secs();
    if timer.first_fill_done && timer.elapsed < interval {
        return;
    }
    timer.elapsed = 0.0;
    timer.first_fill_done = true;

    // Derive scene directions from the current sim state. When sim
    // isn't available yet (early frames) fall back to sensible
    // defaults so we still paint *something* — a static gradient is
    // better than an all-black cubemap that would read as "no IBL".
    //
    // `gain == 0` means `CameraExposure` hasn't had its first update yet (this
    // system can run before `update_camera_exposure` during boot). A paint at
    // gain 0 scales flux — and with it the whole sky LUT, sun disc, and the
    // published `SkyAmbient` — to zero, and because the change-gate below only
    // watches *directions* (which are correct in that black paint), nothing
    // would ever trigger a repaint: the black environment sticks. Defer instead.
    let gain = exposure.as_deref().map(|e| e.gain).unwrap_or(1.0);
    if gain <= 0.0 {
        return;
    }
    let (env, sky_inputs) = sim
        .as_deref()
        .map(|s| derive_environment(s, gain))
        .unwrap_or_else(|| (default_environment(), None));

    // Repaint when the env geometry moved, OR when the sim clock drifted (warp:
    // the sun can cross hours of sky between real-time ticks with the ship's
    // direction gates barely moving — see `REPAINT_SIM_DRIFT_S`).
    let sim_time = sim
        .as_deref()
        .map(|s| s.simulation.sim_time())
        .unwrap_or(f64::NEG_INFINITY);
    let sim_drifted = (sim_time - timer.last_paint_sim_time).abs() > REPAINT_SIM_DRIFT_S;
    if let Some(last) = timer.last_painted
        && !env_changed_meaningfully(&last, &env)
        && !sim_drifted
    {
        return;
    }

    // Bake the physical sky-view LUT for the current sun/altitude (surface
    // regime only). The multi-scatter LUT it needs is static per body, so cache
    // it and rebake only the view-dependent sky LUT.
    let sky_inputs_dbg = sky_inputs; // Copy, kept for the diagnostics log below.
    let sky_lut = sky_inputs.map(|si| {
        let ms_hit = matches!(&timer.ms_cache, Some((id, _)) if *id == si.body_id);
        if !ms_hit {
            let ms = MultiScatterLut::bake(
                &si.atmos,
                si.planet_radius_m,
                MULTI_SCATTER_LUT_WIDTH,
                MULTI_SCATTER_LUT_HEIGHT,
            );
            timer.ms_cache = Some((si.body_id, ms));
        }
        let ms = &timer.ms_cache.as_ref().expect("ms cache just populated").1;
        SkyViewLut::bake(
            &si.atmos,
            si.planet_radius_m,
            si.altitude_m,
            si.sun_dir,
            si.up,
            si.flux,
            ms,
            PROBE_SKY_LUT_W,
            PROBE_SKY_LUT_H,
        )
    });

    // Publish the physical sky-fill ambient (F4): the sky-view LUT's hemispherical
    // irradiance + the altitude blend, for `update_sun_light` to drive the
    // surface `GlobalAmbientLight`. Zero irradiance out in space (no sky).
    *sky_ambient = SkyAmbient {
        surface_irradiance: sky_lut
            .as_ref()
            .map(|l| l.ambient_sky_irradiance())
            .unwrap_or(Vec3::ZERO),
        surface_blend: env.surface_blend,
    };
    // Calibration signal for the F3/F4 physical-sky path (repaint cadence only,
    // ~every 5 s): the raw irradiance driving the surface ambient. Grep
    // `thalos::sky` in the console when judging hull/structure brightness.
    let irr = sky_ambient.surface_irradiance;
    info!(
        target: "thalos::sky",
        "sky irradiance ({:.3}, {:.3}, {:.3}) lum {:.3} | surface_blend {:.2} | sun_disc ({:.1}, {:.1}, {:.1})",
        irr.x, irr.y, irr.z,
        0.2126 * irr.x + 0.7152 * irr.y + 0.0722 * irr.z,
        env.surface_blend,
        env.sun_disc_radiance.x, env.sun_disc_radiance.y, env.sun_disc_radiance.z,
    );
    // The radiances actually painted into the reflection cubemap (what the hull's
    // IBL diffuse + specular see): the F3 physical sky at the zenith and at the
    // horizon, plus the analytic ground bounce. Differential for "hull too dark":
    // the pre-F3 analytic sky sat around 0.2–0.4 at this scene flux. The `inputs`
    // tail is the raw geometry the probe evaluated — compare its `sun_elev`
    // against the lighting system's log line to catch the two systems diverging.
    if let (Some(lut), Some(si)) = (sky_lut.as_ref(), sky_inputs_dbg) {
        let zen = lut.sample(env.up) * PHYSICAL_SKY_SCALE;
        let hor = lut.sample(env.up.any_orthonormal_vector()) * PHYSICAL_SKY_SCALE;
        info!(
            target: "thalos::sky",
            "env paint: sky zenith ({:.3}, {:.3}, {:.3}) horizon ({:.3}, {:.3}, {:.3}) ground ({:.3}, {:.3}, {:.3}) | inputs: sun_elev {:.3} alt {:.0} m flux {:.2} body {} t {:.0} s",
            zen.x, zen.y, zen.z, hor.x, hor.y, hor.z,
            env.ground_radiance.x, env.ground_radiance.y, env.ground_radiance.z,
            si.up.dot(si.sun_dir), si.altitude_m, si.flux, si.body_id,
            sim.as_deref().map(|s| s.simulation.sim_time()).unwrap_or(f64::NAN),
        );
    }

    // 0.19: `Assets::get_mut` returns `AssetMut` (DerefMut); `&mut image`
    // deref-coerces to the `&mut Image` `paint_cubemap` expects.
    let Some(mut image) = images.get_mut(&probe.cubemap) else {
        return;
    };

    paint_cubemap(&mut image, &env, sky_lut.as_ref());
    timer.last_painted = Some(env);
    timer.last_paint_sim_time = sim_time;
}

/// `true` when at least one of the env params has shifted enough that
/// re-painting will produce a visibly different cubemap. Used as the
/// asset-changed gate in [`refresh_cubemap`].
fn env_changed_meaningfully(last: &EnvParams, new: &EnvParams) -> bool {
    // Sun/up drift also gates the sky-view LUT rebake (the LUT is a function of
    // sun direction + local zenith), so this same threshold keeps the reflected
    // sky in step without a separate gate.
    last.sun_dir.dot(new.sun_dir) < ENV_DIR_DOT_MIN
        || last.planet_dir.dot(new.planet_dir) < ENV_DIR_DOT_MIN
        || (last.planet_cos - new.planet_cos).abs() > ENV_COS_EPS
        || last.up.dot(new.up) < ENV_DIR_DOT_MIN
        || (last.surface_blend - new.surface_blend).abs() > ENV_COS_EPS
        // Sun-disc radiance is the one field that encodes flux × day × reddening,
        // so brightness-only changes (exposure gain, terminator crossing) repaint
        // even when every direction is static (a parked craft).
        || (last.sun_disc_radiance - new.sun_disc_radiance).length()
            > 0.02 * last.sun_disc_radiance.length().max(0.1)
}

/// Inputs for baking the physical [`SkyViewLut`] (surface regime only). `atmos`
/// is built in **meters** (`meters_per_render_unit = 1`); the integral is
/// scale-invariant, so meters is the natural unit for the sim-side quantities.
#[derive(Clone, Copy)]
struct SkyLutInputs {
    atmos: AtmosphereBlock,
    /// Dominant body id, the key for the cached multi-scatter LUT.
    body_id: usize,
    planet_radius_m: f32,
    altitude_m: f32,
    /// World-space unit vector toward the sun.
    sun_dir: Vec3,
    /// World-space local radial up (away from the body centre).
    up: Vec3,
    /// Scene-flux sun irradiance (`LIGHT_AT_1AU·(AU/d)²·gain`); sets the LUT's
    /// radiance units so it shares the scene exposure.
    flux: f32,
}

#[derive(Clone, Copy)]
struct EnvParams {
    /// Unit vector from ship toward sun, in world space.
    sun_dir: Vec3,
    /// Cosine half-angle of the sun disc. Sun is drawn where
    /// `dot(view, sun_dir) > sun_cos`.
    sun_cos: f32,
    /// HDR sun-disc radiance (reddened + day-gated near the surface, white in
    /// space, blended by `surface_blend`). Drives the hull's specular highlight.
    sun_disc_radiance: Vec3,
    // ── Orbital / space model (used where `surface_blend < 1`) ──
    /// Unit vector from ship toward the dominant body centre.
    planet_dir: Vec3,
    /// Cosine half-angle of the planet disc from the ship. Planet is
    /// drawn where `dot(view, planet_dir) > planet_cos`.
    planet_cos: f32,
    /// Lit-side planet colour.
    planet_color: Vec3,
    /// Dim ambient fill for the starfield (below the sun, behind the
    /// planet's horizon ring).
    starfield_tint: Vec3,
    // ── Surface-sky model (used where `surface_blend > 0`) ──
    /// Local radial up (away from the dominant body centre).
    up: Vec3,
    /// Warm ground-bounce radiance filling the lower hemisphere. The *upper*
    /// hemisphere (sky) comes from the physical `SkyViewLut` passed alongside.
    ground_radiance: Vec3,
    /// 0 = pure space (planet disc + stars), 1 = pure surface (sky-dome +
    /// ground). Ramps across the Kármán line with altitude.
    surface_blend: f32,
}

fn default_environment() -> EnvParams {
    EnvParams {
        sun_dir: Vec3::X,
        sun_cos: (1.0_f32 - 1.0e-4).max(0.999),
        sun_disc_radiance: Vec3::splat(30.0),
        planet_dir: Vec3::NEG_Y,
        planet_cos: 0.5, // ~60° half-angle — low orbit fills a lot of sky
        planet_color: Vec3::new(0.25, 0.35, 0.55),
        starfield_tint: Vec3::splat(0.02),
        up: Vec3::Y,
        ground_radiance: Vec3::ZERO,
        surface_blend: 0.0,
    }
}

/// Derive the reflection environment from the current sim state + camera
/// exposure gain. Under a terrestrial-atmosphere body it paints the *surface
/// sky* — blue sky-dome above the local horizon, warm ground-bounce below, a
/// reddened sun disc — the same environment the terrain is lit by; out in space
/// it fades (by altitude across the Kármán line) into the orbital model (a lit
/// planet disc over a dim starfield). `gain` matches the scene's `CameraExposure`
/// so the reflected radiances share its exposure and dim with distance.
///
/// Returns the painted `EnvParams` plus, in the surface regime, the
/// [`SkyLutInputs`] the caller uses to bake the physical [`SkyViewLut`] for the
/// upper hemisphere (`None` out in space).
///
/// (This is the atmosphere-derived env-map keystone of graphics-fidelity F3/F4:
/// the metallic hull now reflects the world it is actually in, and dielectric
/// structures pick up the real sky as ambient. The sky is now the physical
/// sky-view LUT (F3); the eventual upgrade is a GPU cubemap render of the actual
/// scene; see `docs/atmosphere.md`.)
fn derive_environment(sim: &SimulationState, gain: f32) -> (EnvParams, Option<SkyLutInputs>) {
    let ship_pos = sim.simulation.ship_state().position;
    let epoch = thalos_physics_canonical::canonical::Epoch(sim.simulation.sim_time());

    // Sun (star index 0): direction + heliocentric flux in the SAME units the
    // spine gives every surface (`build_scene_lighting`: LIGHT_AT_1AU·(AU/d)²·gain).
    let sun_state = sim.ephemeris.state(0, epoch);
    let to_sun = (sun_state.position - ship_pos).as_vec3();
    let sun_dir = to_sun.try_normalize().unwrap_or(Vec3::X);
    let helio_d_m = (ship_pos - sun_state.position).length().max(1.0);
    let au_over_d = (AU_M / helio_d_m) as f32;
    let flux = LIGHT_AT_1AU * au_over_d * au_over_d * gain;

    // Body the ship is bound to. In the star's SOI there is none → pure space.
    let dominant = sim.simulation.dominant_body();
    let planet_id = if dominant != 0 {
        dominant
    } else {
        sim.system.name_to_id.get("Thalos").copied().unwrap_or(0)
    };

    // Orbital planet-disc model (space fallback). Angular radius asin(r / d);
    // the cubemap paints the disc where `dot(view, planet_dir) > cos(radius)`.
    let planet_state = sim.ephemeris.state(planet_id, epoch);
    let to_planet = (planet_state.position - ship_pos).as_vec3();
    let planet_radius_m = sim
        .system
        .bodies
        .get(planet_id)
        .map(|b| b.radius_m as f32)
        .unwrap_or(1.0);
    let planet_dist_m = to_planet.length().max(planet_radius_m * 1.0001);
    let planet_ang = (planet_radius_m / planet_dist_m).clamp(0.0, 0.999).asin();
    let planet_cos = planet_ang.cos();
    let planet_dir = to_planet.try_normalize().unwrap_or(Vec3::NEG_Y);

    // Surface-sky model — only meaningful under a terrestrial-atmosphere body.
    let mut up = -planet_dir;
    let mut ground_radiance = Vec3::ZERO;
    let mut sun_disc_radiance = Vec3::splat(flux * SUN_DISC_GAIN);
    let mut surface_blend = 0.0_f32;
    let mut sky_inputs = None;

    if dominant != 0
        && let Some(body) = sim.system.bodies.get(dominant)
        && let Some(atmos) = body.terrestrial_atmosphere.as_ref()
    {
        let center = sim.ephemeris.state(dominant, epoch).position;
        let radial = (ship_pos - center).as_vec3();
        let dist = radial.length();
        if dist > 1.0 {
            up = radial / dist;
        }
        let altitude = (dist - body.radius_m as f32).max(0.0);
        let karman = atmos.karman_line_m.max(1.0);
        // Full surface sky below the Kármán line, fading to pure space a few
        // Kármán heights up.
        surface_blend = 1.0 - smoothstep_f32(karman, karman * 4.0, altitude);
        if surface_blend > 0.0 {
            let (tau, strength) = atmos
                .scattering
                .as_ref()
                .map(|sc| (Vec3::from_array(sc.vertical_optical_depth), sc.strength))
                .unwrap_or((Vec3::ZERO, 0.0));
            let sun_elev = up.dot(sun_dir);
            // Analytic ground-bounce (lower hemisphere) + reddened direct sun.
            let gs = surface_ground_sun(tau, strength, sun_elev, flux);
            ground_radiance = gs.ground_radiance;
            // Reddened, day-gated sun disc for the surface regime, blended toward
            // the white orbital disc by altitude.
            let day = smoothstep_f32(-0.15, 0.12, sun_elev);
            let surface_sun = gs.sun_color * (flux * SUN_DISC_GAIN * day);
            sun_disc_radiance = sun_disc_radiance.lerp(surface_sun, surface_blend);

            // Physical sky (upper hemisphere): bake the sky-view LUT in meters.
            if strength > 0.0 {
                sky_inputs = Some(SkyLutInputs {
                    atmos: AtmosphereBlock::from_terrestrial(atmos, 1.0),
                    body_id: dominant,
                    planet_radius_m: body.radius_m as f32,
                    altitude_m: altitude,
                    sun_dir,
                    up,
                    flux,
                });
            }
        }
    }

    let env = EnvParams {
        sun_dir,
        sun_cos: 0.9995_f32.max(0.999),
        sun_disc_radiance,
        planet_dir,
        planet_cos,
        planet_color: Vec3::new(0.25, 0.35, 0.55),
        starfield_tint: Vec3::splat(0.015),
        up,
        ground_radiance,
        surface_blend,
    };
    (env, sky_inputs)
}

/// Write Rgba16Float pixels into the cubemap. Layer order matches
/// WGPU / D3D: +X, -X, +Y, -Y, +Z, -Z. `sky_lut` is the physical upper-hemisphere
/// sky sampled per direction in the surface regime (`None` out in space).
fn paint_cubemap(image: &mut Image, env: &EnvParams, sky_lut: Option<&SkyViewLut>) {
    let size = PROBE_SIZE as i32;
    let inv_size = 1.0 / size as f32;
    const FACE_COUNT: usize = 6;
    let face_bytes = (PROBE_SIZE * PROBE_SIZE * 8) as usize; // 4 × 2B
    let Some(data) = image.data.as_mut() else {
        return;
    };
    if data.len() != face_bytes * FACE_COUNT {
        data.resize(face_bytes * FACE_COUNT, 0);
    }

    for face in 0..FACE_COUNT {
        let offset = face_bytes * face;
        let face_data = &mut data[offset..offset + face_bytes];
        for y in 0..size {
            for x in 0..size {
                // Convert (face, x, y) → unit direction in world space
                // using the WGPU cubemap convention. u,v in [-1, +1].
                let u = (x as f32 + 0.5) * inv_size * 2.0 - 1.0;
                let v = (y as f32 + 0.5) * inv_size * 2.0 - 1.0;
                let dir = face_dir(face, u, v);

                let color = sample_environment(env, dir, sky_lut);

                let texel_off = ((y * size + x) * 4) as usize * 2;
                write_rgba16f(&mut face_data[texel_off..texel_off + 8], color);
            }
        }
    }

    image.asset_usage = RenderAssetUsages::all();
    // Force Bevy to re-upload the image this frame.
    // The asset change is detected by `Assets<Image>::get_mut` marking the handle dirty.
}

fn face_dir(face: usize, u: f32, v: f32) -> Vec3 {
    // WGPU / D3D / Vulkan cube face convention, left-handed.
    // Matches the ordering used by the PBR IBL sampler: 0=+X, 1=-X,
    // 2=+Y, 3=-Y, 4=+Z, 5=-Z.
    let raw = match face {
        0 => Vec3::new(1.0, -v, -u),
        1 => Vec3::new(-1.0, -v, u),
        2 => Vec3::new(u, 1.0, v),
        3 => Vec3::new(u, -1.0, -v),
        4 => Vec3::new(u, -v, 1.0),
        5 => Vec3::new(-u, -v, -1.0),
        _ => Vec3::Z,
    };
    raw.normalize()
}

fn sample_environment(env: &EnvParams, dir: Vec3, sky_lut: Option<&SkyViewLut>) -> Vec3 {
    // Surface hemisphere: warm ground bounce below the local horizon, the
    // physical sky-view LUT above (with a small night-ambient floor so the sky
    // doesn't go pure black once the sun sets and the LUT returns ~0).
    let w_up = (0.5 + 0.5 * dir.dot(env.up)).clamp(0.0, 1.0);
    let sky_radiance = sky_lut
        .map(|l| l.sample(dir) * PHYSICAL_SKY_SCALE + SURFACE_NIGHT_AMBIENT)
        .unwrap_or(SURFACE_NIGHT_AMBIENT);
    let surface_col = env.ground_radiance.lerp(sky_radiance, w_up);

    // Space: lit planet disc over a dim starfield.
    let orbital_col = orbital_sample(env, dir);

    // Blend the base environment by altitude (surface ↔ space).
    let mut col = orbital_col.lerp(surface_col, env.surface_blend);

    // Sun disc (both regimes): the HDR hot spot that gives polished metal its
    // specular highlight. Sits on top of whatever was below.
    if dir.dot(env.sun_dir) > env.sun_cos {
        col = env.sun_disc_radiance;
    }

    col
}

/// Orbital (space) base radiance for a direction: a sun-lit planet disc with a
/// soft terminator over a dim starfield. Used where `surface_blend < 1`.
fn orbital_sample(env: &EnvParams, dir: Vec3) -> Vec3 {
    let mut col = env.starfield_tint;
    let planet_dot = dir.dot(env.planet_dir);
    if planet_dot > env.planet_cos {
        let point_on_planet = dir - env.planet_dir * planet_dot;
        let normal = -(env.planet_dir - point_on_planet * (1.0 - env.planet_cos))
            .normalize_or(-env.planet_dir);
        let lit = env.sun_dir.dot(normal).max(0.0);
        // Soft limb gradient + Lambert term. The 0.15 floor keeps the night
        // side visible as a slightly-bluish disc rather than a hole.
        let limb = smoothstep_f32(env.planet_cos, env.planet_cos + 0.02, planet_dot);
        col = env.planet_color * (lit * 0.85 + 0.15) * limb + col * (1.0 - limb);
    }
    col
}

fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Write a `[f32; 4]` as four little-endian `f16` texels into `out`
/// (8 bytes). Uses `half::f16`-compatible bit layout via `f32_to_f16`.
fn write_rgba16f(out: &mut [u8], color: Vec3) {
    fn f32_to_f16_bits(f: f32) -> u16 {
        // Minimal IEEE-754 single → half conversion. Flushes denormals
        // to zero and saturates overflow to +/- inf.
        let bits = f.to_bits();
        let sign = ((bits >> 31) & 0x1) as u16;
        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mantissa = bits & 0x7f_ffff;
        if exponent <= 0 {
            sign << 15
        } else if exponent >= 31 {
            (sign << 15) | 0x7c00
        } else {
            (sign << 15) | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
        }
    }

    let channels = [color.x, color.y, color.z, 1.0_f32];
    for (i, &c) in channels.iter().enumerate() {
        let h = f32_to_f16_bits(c);
        out[i * 2] = (h & 0xff) as u8;
        out[i * 2 + 1] = ((h >> 8) & 0xff) as u8;
    }
}
