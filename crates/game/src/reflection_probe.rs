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
use thalos_body_render::{AU_M, LIGHT_AT_1AU};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Cubemap face resolution. 256 balances reflection sharpness against
/// CPU write cost (each refresh touches 6 × 256² = ~400k texels).
const PROBE_SIZE: u32 = 256;

/// Seconds between cubemap refreshes. The painted content is "vaguely
/// lit hemisphere with a sun disc" — geometric, not detailed — so
/// reflection content changes slowly even at 1× warp. Low rates here
/// matter because each refresh marks the cubemap asset changed, which
/// re-triggers Bevy's diffuse+specular IBL prefilter convolution
/// (`prepare_generated_environment_map_bind_groups`) — that prefilter
/// is the dominant cost downstream of the CPU paint.
const REFRESH_INTERVAL: f32 = 5.0;

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

// ── CPU mirror of the spine surface-sky (graphics-fidelity F3/F4) ─────────────
// The reflection cubemap is CPU-painted (the GPU cubemap-render path is blocked —
// see docs/atmosphere.md), so the surface sky it reflects must be evaluated on the
// CPU. These constants + `cpu_surface_sky` are a hand-kept mirror of the spine's
// WGSL `compute_surface_sky` (`shading/shaders/lighting.wgsl`), so the metallic
// hull reflects the SAME blue sky-dome / warm ground the terrain is lit by, and
// dielectric structures get that sky as ambient — one atmosphere-derived
// environment. Keep in lockstep when the spine's `SURFACE_*` constants change.
const SCENE_FLUX_SCALE: f32 = 0.5;
const SURFACE_SKY_SCALE: f32 = 0.15;
const SURFACE_SKY_CHROMA_GAIN: f32 = 8.0;
const SURFACE_GROUND_ALBEDO: Vec3 = Vec3::new(0.10, 0.085, 0.055);
const SURFACE_GROUND_SCALE: f32 = 0.10;
const SURFACE_NIGHT_AMBIENT: Vec3 = Vec3::new(0.008, 0.010, 0.014);

fn vec3_exp(v: Vec3) -> Vec3 {
    Vec3::new(v.x.exp(), v.y.exp(), v.z.exp())
}

/// Resolved surface-sky radiances (scene-flux units) — mirror of the spine's
/// `SurfaceSky`. `sun_color` is the reddened direct-beam tint.
struct SurfaceSkyCpu {
    sky_radiance: Vec3,
    ground_radiance: Vec3,
    sun_color: Vec3,
}

/// CPU mirror of `compute_surface_sky`: blue sky-dome (up) + warm ground-bounce
/// (down) + the reddened direct-beam tint, from the vertical Rayleigh optical
/// depth `tau_zenith`, artistic `strength`, sun elevation, and per-fragment flux.
fn cpu_surface_sky(tau_zenith: Vec3, strength: f32, sun_elev: f32, flux: f32) -> SurfaceSkyCpu {
    let scene_radiance = flux.max(0.0) * SCENE_FLUX_SCALE;
    let day = smoothstep_f32(-0.15, 0.12, sun_elev);
    let sun_up = sun_elev.clamp(0.0, 1.0);
    let tau_eff = tau_zenith.max(Vec3::ZERO) * strength.max(0.0);
    let airmass = (1.0 / (sun_up + 0.10)).clamp(1.0, 8.0);
    let sun_color = vec3_exp(-tau_eff * (airmass - 1.0));
    let sky_chroma = Vec3::ONE - vec3_exp(-tau_eff * SURFACE_SKY_CHROMA_GAIN);
    let sky_strength = scene_radiance * SURFACE_SKY_SCALE * day * (0.35 + 0.65 * sun_up);
    let sky_radiance = sky_chroma * sky_strength + SURFACE_NIGHT_AMBIENT;
    let ground_radiance =
        SURFACE_GROUND_ALBEDO * (scene_radiance * SURFACE_GROUND_SCALE * sun_up) + SURFACE_NIGHT_AMBIENT;
    SurfaceSkyCpu {
        sky_radiance,
        ground_radiance,
        sun_color,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct ReflectionProbePlugin;

impl Plugin for ReflectionProbePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProbeRefreshTimer>()
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
}

impl Default for ProbeRefreshTimer {
    fn default() -> Self {
        Self {
            elapsed: REFRESH_INTERVAL, // force an update on frame 1
            first_fill_done: false,
            last_painted: None,
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
) {
    let Some(probe) = probe else { return };

    timer.elapsed += time.delta_secs();
    if timer.first_fill_done && timer.elapsed < REFRESH_INTERVAL {
        return;
    }
    timer.elapsed = 0.0;
    timer.first_fill_done = true;

    // Derive scene directions from the current sim state. When sim
    // isn't available yet (early frames) fall back to sensible
    // defaults so we still paint *something* — a static gradient is
    // better than an all-black cubemap that would read as "no IBL".
    let gain = exposure.as_deref().map(|e| e.gain).unwrap_or(1.0);
    let env = sim
        .as_deref()
        .map(|s| derive_environment(s, gain))
        .unwrap_or_else(default_environment);

    if let Some(last) = timer.last_painted
        && !env_changed_meaningfully(&last, &env)
    {
        return;
    }

    // 0.19: `Assets::get_mut` returns `AssetMut` (DerefMut); `&mut image`
    // deref-coerces to the `&mut Image` `paint_cubemap` expects.
    let Some(mut image) = images.get_mut(&probe.cubemap) else {
        return;
    };

    paint_cubemap(&mut image, &env);
    timer.last_painted = Some(env);
}

/// `true` when at least one of the env params has shifted enough that
/// re-painting will produce a visibly different cubemap. Used as the
/// asset-changed gate in [`refresh_cubemap`].
fn env_changed_meaningfully(last: &EnvParams, new: &EnvParams) -> bool {
    last.sun_dir.dot(new.sun_dir) < ENV_DIR_DOT_MIN
        || last.planet_dir.dot(new.planet_dir) < ENV_DIR_DOT_MIN
        || (last.planet_cos - new.planet_cos).abs() > ENV_COS_EPS
        || last.up.dot(new.up) < ENV_DIR_DOT_MIN
        || (last.surface_blend - new.surface_blend).abs() > ENV_COS_EPS
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
    /// Blue sky-dome radiance filling the upper hemisphere (scene-flux units).
    sky_radiance: Vec3,
    /// Warm ground-bounce radiance filling the lower hemisphere.
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
        sky_radiance: Vec3::ZERO,
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
/// (This is the atmosphere-derived env-map keystone of graphics-fidelity F3/F4:
/// the metallic hull now reflects the world it is actually in, and dielectric
/// structures pick up the real sky as ambient — replacing the old fake orbital-
/// only paint. The eventual upgrade is a GPU cubemap render of the actual scene;
/// see `docs/atmosphere.md`.)
fn derive_environment(sim: &SimulationState, gain: f32) -> EnvParams {
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
    let mut sky_radiance = Vec3::ZERO;
    let mut ground_radiance = Vec3::ZERO;
    let mut sun_disc_radiance = Vec3::splat(flux * SUN_DISC_GAIN);
    let mut surface_blend = 0.0_f32;

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
            let sky = cpu_surface_sky(tau, strength, sun_elev, flux);
            sky_radiance = sky.sky_radiance;
            ground_radiance = sky.ground_radiance;
            // Reddened, day-gated sun disc for the surface regime, blended toward
            // the white orbital disc by altitude.
            let day = smoothstep_f32(-0.15, 0.12, sun_elev);
            let surface_sun = sky.sun_color * (flux * SUN_DISC_GAIN * day);
            sun_disc_radiance = sun_disc_radiance.lerp(surface_sun, surface_blend);
        }
    }

    EnvParams {
        sun_dir,
        sun_cos: 0.9995_f32.max(0.999),
        sun_disc_radiance,
        planet_dir,
        planet_cos,
        planet_color: Vec3::new(0.25, 0.35, 0.55),
        starfield_tint: Vec3::splat(0.015),
        up,
        sky_radiance,
        ground_radiance,
        surface_blend,
    }
}

/// Write Rgba16Float pixels into the cubemap. Layer order matches
/// WGPU / D3D: +X, -X, +Y, -Y, +Z, -Z.
fn paint_cubemap(image: &mut Image, env: &EnvParams) {
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

                let color = sample_environment(env, dir);

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

fn sample_environment(env: &EnvParams, dir: Vec3) -> Vec3 {
    // Surface hemisphere: warm ground bounce below the local horizon, blue
    // sky-dome above — the same split the terrain's sky ambient uses.
    let w_up = (0.5 + 0.5 * dir.dot(env.up)).clamp(0.0, 1.0);
    let surface_col = env.ground_radiance.lerp(env.sky_radiance, w_up);

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
