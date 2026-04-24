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
use crate::rendering::SimulationState;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Cubemap face resolution. 256 balances reflection sharpness against
/// CPU write cost (each refresh touches 6 × 256² = ~400k texels).
const PROBE_SIZE: u32 = 256;

/// Seconds between cubemap refreshes. Set low enough that orbital
/// angular motion is invisible between updates, high enough that the
/// CPU cost is negligible.
const REFRESH_INTERVAL: f32 = 0.25;

/// Environment map intensity multiplier handed to
/// [`GeneratedEnvironmentMapLight`]. 1.0 matches scene luminance;
/// bump if reflections read too dark on polished metals.
const PROBE_INTENSITY: f32 = 1.0;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct ReflectionProbePlugin;

impl Plugin for ReflectionProbePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProbeRefreshTimer>()
            .add_systems(Startup, setup_probe)
            .add_systems(
                Update,
                (attach_env_map_to_main_camera, refresh_cubemap),
            );
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
}

impl Default for ProbeRefreshTimer {
    fn default() -> Self {
        Self {
            elapsed: REFRESH_INTERVAL, // force an update on frame 1
            first_fill_done: false,
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
fn refresh_cubemap(
    time: Res<Time>,
    mut timer: ResMut<ProbeRefreshTimer>,
    probe: Option<Res<ReflectionProbe>>,
    mut images: ResMut<Assets<Image>>,
    sim: Option<Res<SimulationState>>,
) {
    let Some(probe) = probe else { return };

    timer.elapsed += time.delta_secs();
    if timer.first_fill_done && timer.elapsed < REFRESH_INTERVAL {
        return;
    }
    timer.elapsed = 0.0;
    timer.first_fill_done = true;

    let Some(image) = images.get_mut(&probe.cubemap) else {
        return;
    };

    // Derive scene directions from the current sim state. When sim
    // isn't available yet (early frames) fall back to sensible
    // defaults so we still paint *something* — a static gradient is
    // better than an all-black cubemap that would read as "no IBL".
    let env = sim
        .as_deref()
        .map(derive_environment)
        .unwrap_or_else(default_environment);

    paint_cubemap(image, &env);
}

struct EnvParams {
    /// Unit vector from ship toward sun, in world space.
    sun_dir: Vec3,
    /// Cosine half-angle of the sun disc. Sun is drawn where
    /// `dot(view, sun_dir) > sun_cos`.
    sun_cos: f32,
    /// Solar disc luminance.
    sun_color: Vec3,
    /// Unit vector from ship toward planet centre.
    planet_dir: Vec3,
    /// Cosine half-angle of the planet disc from the ship. Planet is
    /// drawn where `dot(view, planet_dir) > planet_cos`.
    planet_cos: f32,
    /// Lit-side planet colour.
    planet_color: Vec3,
    /// Dim ambient fill for the starfield (below the sun, behind the
    /// planet's horizon ring).
    starfield_tint: Vec3,
}

fn default_environment() -> EnvParams {
    EnvParams {
        sun_dir: Vec3::X,
        sun_cos: (1.0_f32 - 1.0e-4).max(0.999),
        sun_color: Vec3::splat(30.0),
        planet_dir: Vec3::NEG_Y,
        planet_cos: 0.5, // ~60° half-angle — low orbit fills a lot of sky
        planet_color: Vec3::new(0.25, 0.35, 0.55),
        starfield_tint: Vec3::splat(0.02),
    }
}

/// Pull sun/planet directions from the current simulation state. For
/// now: sun = origin of the system, planet = focus body (Thalos).
/// When ships get coupled in, this should key off the ship's world
/// position rather than the sim homeworld.
fn derive_environment(sim: &SimulationState) -> EnvParams {
    // For this first pass, use heliocentric state at t=0. The visible
    // effect of orbital motion on reflections is tiny over typical
    // session time scales; this keeps the first-cut code minimal.
    // Once ships are rendered, replace `ship_pos` with the real ship
    // world position.
    let ship_pos = sim.simulation.ship_state().position;
    let home_id = sim
        .system
        .name_to_id
        .get("Thalos")
        .copied()
        .unwrap_or_else(|| {
            sim.system
                .bodies
                .iter()
                .find(|b| b.parent.is_some())
                .map(|b| b.id)
                .unwrap_or(0)
        });

    let sim_time = sim.simulation.sim_time();
    let planet_state = sim.ephemeris.query_body(home_id, sim_time);
    let sun_state = sim.ephemeris.query_body(0, sim_time);

    let to_planet = (planet_state.position - ship_pos).as_vec3();
    let to_sun = (sun_state.position - ship_pos).as_vec3();

    // Planet angular radius from the ship: asin(r / d). The cubemap
    // paints the disc as `dot(view, planet_dir) > cos(angular_radius)`.
    let planet_radius_m = sim.system.bodies[home_id].radius_m as f32;
    let planet_dist_m = to_planet.length().max(planet_radius_m * 1.0001);
    let planet_ang = (planet_radius_m / planet_dist_m).clamp(0.0, 0.999).asin();
    let planet_cos = planet_ang.cos();

    EnvParams {
        sun_dir: to_sun.try_normalize().unwrap_or(Vec3::X),
        sun_cos: (0.9995_f32).max(0.999),
        sun_color: Vec3::splat(30.0),
        planet_dir: to_planet.try_normalize().unwrap_or(Vec3::NEG_Y),
        planet_cos,
        planet_color: Vec3::new(0.25, 0.35, 0.55),
        starfield_tint: Vec3::splat(0.015),
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
    // Start with a very dim starfield so reflections never read as
    // pure black. A real star catalog bake would go here later.
    let mut col = env.starfield_tint;

    // Planet disc: lit by the sun, simple Lambert falloff along the
    // terminator so the reflection reads as a lit hemisphere rather
    // than a flat circle.
    let planet_dot = dir.dot(env.planet_dir);
    if planet_dot > env.planet_cos {
        let point_on_planet = dir - env.planet_dir * planet_dot;
        let normal = -(env.planet_dir - point_on_planet * (1.0 - env.planet_cos)).normalize_or(
            -env.planet_dir,
        );
        let lit = env.sun_dir.dot(normal).max(0.0);
        // Soft limb gradient + Lambert term. The 0.15 floor keeps the
        // night side visible as a slightly-bluish disc rather than a
        // hole.
        let limb = smoothstep_f32(env.planet_cos, env.planet_cos + 0.02, planet_dot);
        col = env.planet_color * (lit * 0.85 + 0.15) * limb + col * (1.0 - limb);
    }

    // Sun disc: HDR hot spot. Sits on top of whatever was below.
    let sun_dot = dir.dot(env.sun_dir);
    if sun_dot > env.sun_cos {
        col = env.sun_color;
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

// Extension trait: glam's `Vec3::normalize_or` is `normalize_or_zero`
// only; define a version that returns a fallback for when length≈0.
trait NormalizeOr {
    fn normalize_or(self, fallback: Vec3) -> Vec3;
}

impl NormalizeOr for Vec3 {
    fn normalize_or(self, fallback: Vec3) -> Vec3 {
        self.try_normalize().unwrap_or(fallback)
    }
}
