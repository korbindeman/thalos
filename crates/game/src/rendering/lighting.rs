//! Per-frame lighting state: camera exposure, scene lighting struct
//! population (one star + eclipse occluders + planetshine), planet/solid
//! material light updates, sun-light direction.

use bevy::prelude::*;
use thalos_body_render::{
    AU_M, FilmGrain, LIGHT_AT_1AU, MAX_ECLIPSE_OCCLUDERS, PlanetHaloMaterial, PlanetMaterial,
    SceneLighting, SolidPlanetMaterial, StarLight,
};
use thalos_physics_canonical::types::BodyStates;

use super::types::{
    CameraExposure, CelestialBody, PlanetMaterials, PlanetshineTints, SimulationState,
    SolarSystemState, SolidPlanetMaterials, SunLight,
};
use crate::camera::{CameraFocus, CameraFocusTarget};
use crate::coords::{MAP_SCALE, RenderOrigin, SHIP_SCALE};
use crate::view::ViewMode;

/// Ambient floor. Vacuum has no fill light — night sides are black.
const PLANET_AMBIENT: f32 = 0.0;

/// Exposure exponent. 2.0 = full compensation (distant bodies look identical
/// to focused Thalos — destroys distance cue). 0.0 = no compensation (Nyx is
/// black). 1.0 = linear-in-distance compensation: display flux at focus is
/// `LIGHT_AT_1AU / focus_d_AU`, so Nyx focus lands at ~0.24 — visibly dim,
/// leaves shadows dark, and doesn't collide with Bevy `AutoExposure` pulling
/// the scene up independently in the post stack.
const EXPOSURE_ALPHA: f64 = 1.0;

/// Maximum positive EV used to drive grain. Beyond this, grain saturates.
/// log2(42^1.0) ≈ 5.4 — Nyx is roughly here.
const EXPOSURE_EV_GRAIN_MAX: f32 = 6.0;

/// Update the `CameraExposure` resource from the current focus body. This is
/// the single source of truth for how much gain the "camera" applies to the
/// raw inverse-square solar flux each body sees. Runs once per frame after
/// `sync_solar_system_state`, before any consumer reads `CameraExposure`.
pub(super) fn update_camera_exposure(
    cache: Res<SolarSystemState>,
    focus: Res<CameraFocus>,
    bodies: Query<&CelestialBody>,
    sim: Res<SimulationState>,
    player: Option<Res<crate::player_controller::PlayerControllerState>>,
    mut exposure: ResMut<CameraExposure>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let star_pos = states.first().map(|s| s.position).unwrap_or_default();

    let focus_dist_m = match focus.target {
        CameraFocusTarget::Body(body_id) => bodies
            .iter()
            .find(|body| body.body_id == body_id && !body.is_star)
            .and_then(|body| states.get(body.body_id))
            .map(|s| (s.position - star_pos).length()),
        CameraFocusTarget::Ship => Some((sim.simulation.ship_state().position - star_pos).length()),
        CameraFocusTarget::PlayerController => player
            .as_deref()
            .and_then(|state| state.active_position_m())
            .map(|position| (position - star_pos).length())
            .or_else(|| Some((sim.simulation.ship_state().position - star_pos).length())),
        CameraFocusTarget::Ghost(ghost_focus) => states
            .get(ghost_focus.body_id)
            .map(|s| (s.position - star_pos).length()),
        CameraFocusTarget::None => None,
    }
    .unwrap_or(AU_M);

    let focus_d_au = (focus_dist_m / AU_M).max(1.0e-3);
    let gain = focus_d_au.powf(EXPOSURE_ALPHA) as f32;

    exposure.focus_dist_m = focus_dist_m;
    exposure.gain = gain;
    exposure.ev = gain.max(1.0e-6).log2();
}

/// Drive per-camera film grain strength from the current exposure push. When
/// the exposure system is lifting a dark outer-system scene by several EV,
/// that's equivalent to running a real sensor at high ISO: the visible result
/// is more grain. We add grain proportional to the positive EV push so Nyx
/// reads as "dim, sensor-limited" rather than "just another 1 AU body in
/// weird light."
pub(super) fn sync_film_grain_to_exposure(
    exposure: Res<CameraExposure>,
    mut grains: Query<&mut FilmGrain>,
) {
    // Only positive EV adds grain. Pulling bright scenes down (inner-system
    // focus) doesn't add noise in a real sensor.
    let push_ev = exposure.ev.max(0.0);
    let normalized = (push_ev / EXPOSURE_EV_GRAIN_MAX).clamp(0.0, 1.0);
    const BASE_INTENSITY: f32 = 0.006;
    const MAX_EXTRA: f32 = 0.010;
    let target = BASE_INTENSITY + normalized * MAX_EXTRA;
    for mut grain in &mut grains {
        grain.intensity = target;
    }
}

/// Build a `SceneLighting` snapshot for one body: one star (index 0),
/// eclipse occluders drawn from every other non-trivial body, shared
/// exposure gain, ambient floor. Planetshine is filled separately by
/// the caller because only terrestrial moons need it.
pub(super) fn build_scene_lighting(
    body_id: usize,
    states: &BodyStates,
    occluders: &[(usize, Vec3, f32)],
    gain: f32,
) -> SceneLighting {
    let mut scene = SceneLighting {
        ambient_intensity: PLANET_AMBIENT,
        ..Default::default()
    };

    let star_pos = states.first().map(|s| s.position).unwrap_or_default();
    let body_pos = states.get(body_id).map(|s| s.position).unwrap_or_default();
    let offset = star_pos - body_pos;
    let distance_m = offset.length();
    let to_star = if distance_m > 0.0 {
        (offset / distance_m).as_vec3()
    } else {
        Vec3::Y
    };
    let au_over_d = AU_M / distance_m.max(1.0);
    let flux = LIGHT_AT_1AU * (au_over_d * au_over_d) as f32 * gain;

    scene.star_count = 1;
    scene.stars[0] = StarLight {
        dir_flux: Vec4::new(to_star.x, to_star.y, to_star.z, flux),
        color: Vec4::new(1.0, 1.0, 1.0, 0.0),
    };

    let mut count = 0usize;
    for (other_id, pos, radius) in occluders {
        if *other_id == body_id {
            continue;
        }
        if count >= MAX_ECLIPSE_OCCLUDERS {
            break;
        }
        scene.occluders[count] = Vec4::new(pos.x, pos.y, pos.z, *radius);
        count += 1;
    }
    scene.occluder_count = count as u32;

    scene
}

/// Collect eclipse-occluder candidates from every visible non-star body
/// at the given metres → render-units scale. Used twice per frame: once
/// at [`MAP_SCALE`] for the map-layer impostor materials, once at
/// [`SHIP_SCALE`] for the ship-layer ones.
///
/// Caller owns the buffer (typically a `Local`) so the per-frame
/// allocation is paid once at startup and reused thereafter.
pub(super) fn collect_occluders<'a>(
    out: &mut Vec<(usize, Vec3, f32)>,
    states: &BodyStates,
    origin: &RenderOrigin,
    scale: f64,
    bodies: impl IntoIterator<Item = &'a CelestialBody>,
) {
    out.clear();
    for body in bodies {
        if body.is_star || body.radius_m < 1.0 {
            continue;
        }
        let Some(state) = states.get(body.body_id) else {
            continue;
        };
        let render_pos = ((state.position - origin.position) * scale).as_vec3();
        let render_radius = ((body.radius_m * scale) as f32).max(0.005);
        out.push((body.body_id, render_pos, render_radius));
    }
}

/// Updates each planet material's `light_dir` uniform to point from the body
/// toward the star.  Must run after `sync_solar_system_state`.
pub(super) fn update_planet_light_dirs(
    query: Query<(&CelestialBody, &PlanetMaterials)>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
    mut halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
    cache: Res<SolarSystemState>,
    origin: Res<RenderOrigin>,
    sim: Res<SimulationState>,
    exposure: Res<CameraExposure>,
    view: Res<ViewMode>,
    planetshine: Res<PlanetshineTints>,
    mut map_occluders: Local<Vec<(usize, Vec3, f32)>>,
    mut ship_occluders: Local<Vec<(usize, Vec3, f32)>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let body_defs = sim.simulation.bodies();
    let gain = exposure.gain;

    // Each `materials.get_mut` below marks the asset changed and forces
    // a full re-prepare in the render world. The inactive view's
    // material isn't being rendered (`RenderLayers` excludes it), so
    // pushing fresh uniforms into it every frame is wasted work — gate
    // it on `view.is_changed()` so the inactive scale catches up exactly
    // once per view toggle and stays quiet otherwise.
    let force_both = view.is_changed();
    let do_map = force_both || matches!(*view, ViewMode::Map);
    let do_ship = force_both || matches!(*view, ViewMode::Ship);

    // Compute occluder lists at both scales once per frame. Buffers are
    // `Local`s, so we pay the allocation once and reuse on subsequent
    // frames.
    let body_iter = || query.iter().map(|(b, _)| b);
    collect_occluders(&mut map_occluders, states, &origin, MAP_SCALE, body_iter());
    collect_occluders(
        &mut ship_occluders,
        states,
        &origin,
        SHIP_SCALE,
        body_iter(),
    );

    // Legacy cloud time uniform: kept in sync from canonical cloud-band
    // state for materials/shaders that still read `cloud_dynamics.y`.
    let sim_time = sim.simulation.sim_time();

    let map_slice: &[(usize, Vec3, f32)] = &map_occluders;
    let ship_slice: &[(usize, Vec3, f32)] = &ship_occluders;
    for (body, mats) in &query {
        let body_def = &body_defs[body.body_id];
        let cloud_time = cache
            .environment
            .get(body.body_id)
            .and_then(|env| env.cloud_bands.as_ref())
            .map(|clouds| {
                let scroll = clouds.scroll_rate_rad_s.abs();
                let period = if scroll > 1.0e-9 {
                    std::f64::consts::TAU / scroll
                } else {
                    86_400.0
                };
                sim_time.rem_euclid(period) as f32
            })
            .unwrap_or(0.0);
        // Same scale-independent inputs feed both materials; only the
        // scale-dependent fields (radius, occluders, planetshine pos)
        // differ.
        for (handle, halo_handle, occluders, scale, want) in [
            (&mats.map, &mats.map_halo, map_slice, MAP_SCALE, do_map),
            (&mats.ship, &mats.ship_halo, ship_slice, SHIP_SCALE, do_ship),
        ] {
            if !want {
                continue;
            }
            let radius = (body.radius_m * scale) as f32;
            let mut scene = build_scene_lighting(body.body_id, states, occluders, gain);

            // Planetshine: pick the orbital parent, skipping the star.
            // The parent's mean albedo (from its baked surface, or its
            // cloud palette for gas giants) drives the tint. Bodies the
            // resource hasn't been populated for contribute no shine.
            if let Some(parent_id) = body_def.parent {
                let parent_def = &body_defs[parent_id];
                if !matches!(parent_def.kind, thalos_world::BodyKind::Star)
                    && let Some(parent_state) = states.get(parent_id)
                    && let Some(tint) = planetshine.by_body.get(&parent_id)
                {
                    let parent_render_pos =
                        ((parent_state.position - origin.position) * scale).as_vec3();
                    let parent_radius = (parent_def.radius_m * scale) as f32;
                    scene.planetshine_pos_radius = Vec4::new(
                        parent_render_pos.x,
                        parent_render_pos.y,
                        parent_render_pos.z,
                        parent_radius,
                    );
                    scene.planetshine_tint_flag = Vec4::new(tint[0], tint[1], tint[2], 1.0);
                }
            }

            // Peek before mutating: `get_mut` flags the asset changed and
            // triggers a full re-extract in the render world, so skip it
            // when the inputs are bit-identical (paused sim, no view drift).
            let primary_dirty = matches!(
                materials.get(handle),
                Some(mat) if mat.params.radius != radius
                    || mat.params.scene != scene
                    || mat.atmosphere.cloud_dynamics.y != cloud_time
            );
            if primary_dirty && let Some(mat) = materials.get_mut(handle) {
                mat.params.radius = radius;
                mat.params.scene = scene.clone();
                // Drive the cloud layer's time uniform. Bodies without a
                // cloud layer have `cloud_albedo_coverage.w = 0`, so the
                // shader skips the layer entirely and this value is ignored.
                // Scroll rate is scale-independent (rad/s on the unit
                // sphere), so the period is the same for both materials.
                mat.atmosphere.cloud_dynamics.y = cloud_time;
            }

            let halo_dirty = matches!(
                halo_materials.get(halo_handle),
                Some(mat) if mat.params.radius != radius
                    || mat.params.scene != scene
                    || mat.atmosphere.cloud_dynamics.y != cloud_time
            );
            if halo_dirty && let Some(mat) = halo_materials.get_mut(halo_handle) {
                mat.params.radius = radius;
                mat.params.scene = scene;
                mat.atmosphere.cloud_dynamics.y = cloud_time;
            }
        }
    }
}

/// Push lighting state into every [`SolidPlanetMaterial`] each frame.
///
/// Mirrors [`update_planet_light_dirs`] for placeholder bodies (no terrain
/// pipeline yet): same scene-lighting build, same planetshine logic for
/// moons. The placeholder has no orientation, atmosphere, or cloud state,
/// so the work stops at `params.scene`.
pub(super) fn update_solid_planet_params(
    query: Query<(&CelestialBody, &SolidPlanetMaterials)>,
    mut materials: ResMut<Assets<SolidPlanetMaterial>>,
    cache: Res<SolarSystemState>,
    origin: Res<RenderOrigin>,
    sim: Res<SimulationState>,
    exposure: Res<CameraExposure>,
    view: Res<ViewMode>,
    planetshine: Res<PlanetshineTints>,
    mut map_occluders: Local<Vec<(usize, Vec3, f32)>>,
    mut ship_occluders: Local<Vec<(usize, Vec3, f32)>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let body_defs = sim.simulation.bodies();
    let gain = exposure.gain;

    // See note on `update_planet_light_dirs` — only push uniforms into
    // the active view's material; the other one isn't being rendered.
    let force_both = view.is_changed();
    let do_map = force_both || matches!(*view, ViewMode::Map);
    let do_ship = force_both || matches!(*view, ViewMode::Ship);

    let body_iter = || query.iter().map(|(b, _)| b);
    collect_occluders(&mut map_occluders, states, &origin, MAP_SCALE, body_iter());
    collect_occluders(
        &mut ship_occluders,
        states,
        &origin,
        SHIP_SCALE,
        body_iter(),
    );
    let map_slice: &[(usize, Vec3, f32)] = &map_occluders;
    let ship_slice: &[(usize, Vec3, f32)] = &ship_occluders;

    for (body, mats) in &query {
        let body_def = &body_defs[body.body_id];
        for (handle, occluders, scale, want) in [
            (&mats.map, map_slice, MAP_SCALE, do_map),
            (&mats.ship, ship_slice, SHIP_SCALE, do_ship),
        ] {
            if !want {
                continue;
            }
            let radius = ((body.radius_m * scale) as f32).max(0.005);
            let mut scene = build_scene_lighting(body.body_id, states, occluders, gain);

            if let Some(parent_id) = body_def.parent {
                let parent_def = &body_defs[parent_id];
                if !matches!(parent_def.kind, thalos_world::BodyKind::Star)
                    && let Some(parent_state) = states.get(parent_id)
                    && let Some(tint) = planetshine.by_body.get(&parent_id)
                {
                    let parent_render_pos =
                        ((parent_state.position - origin.position) * scale).as_vec3();
                    let parent_radius = (parent_def.radius_m * scale) as f32;
                    scene.planetshine_pos_radius = Vec4::new(
                        parent_render_pos.x,
                        parent_render_pos.y,
                        parent_render_pos.z,
                        parent_radius,
                    );
                    scene.planetshine_tint_flag = Vec4::new(tint[0], tint[1], tint[2], 1.0);
                }
            }

            // Peek before mutating; `get_mut` re-uploads the uniform even
            // on identical writes.
            let dirty = matches!(
                materials.get(handle),
                Some(mat) if mat.params.radius != radius || mat.params.scene != scene
            );
            if dirty && let Some(mat) = materials.get_mut(handle) {
                mat.params.radius = radius;
                mat.params.scene = scene;
            }
        }
    }
}

/// Directional-light illuminance with the sun high over the local horizon.
/// Hand-tuned daytime value retained from the original constant light.
const SUN_DAY_ILLUMINANCE: f32 = 10_000.0;
/// Ambient (sky-fill) brightness in full daylight and on the deep night side.
/// The ratio mirrors the terrain shader's `day_fill = 0.15` / `night_fill =
/// 0.012` so the Bevy-PBR ship + runway fade to the same faint starlight floor
/// the ground does instead of staying lit when the sun is below the horizon.
const AMBIENT_DAY_BRIGHTNESS: f32 = 50.0;
const AMBIENT_NIGHT_BRIGHTNESS: f32 = 4.0;

/// `smoothstep`, matching the WGSL builtin the terrain shader uses so the
/// CPU-side day/night gate lines up with the ground's terminator exactly.
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Point the directional sun light from the star toward the camera's focus
/// body, and modulate its illuminance + the global ambient floor by the sun's
/// elevation over the craft's local horizon.
///
/// The ship hull and runway are Bevy `StandardMaterial`s lit by this one
/// directional light, while the terrain is lit by its own shader that fades
/// direct sun and sky-fill across the terminator (`body_terrain.wgsl`:
/// `daylight = smoothstep(-0.06, 0.12, dot(local_up, sun_dir))`). With a
/// constant-illuminance light the hull stayed brightly lit on the night side
/// while the ground around it went dark. We evaluate the *same* day/night
/// function here at the craft's position over its dominant (SOI) body and scale
/// the light + ambient by it, so all surfaces dim together as the sun sets or
/// the planet eclipses the craft. Deep space (dominant body is the star) keeps
/// full illuminance — no planet to block the sun there.
pub(super) fn update_sun_light(
    cache: Res<SolarSystemState>,
    focus: Res<CameraFocus>,
    sim: Res<SimulationState>,
    player: Option<Res<crate::player_controller::PlayerControllerState>>,
    mut ambient: ResMut<GlobalAmbientLight>,
    mut light_query: Query<(&mut Transform, &mut DirectionalLight), With<SunLight>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };

    // Find the focus body's physics-space position — or the ship's when
    // focus is on the player's ship (so sun direction tracks the ship in
    // ship view).
    let focus_pos = match focus.target {
        CameraFocusTarget::Body(body_id) => states.get(body_id).map(|s| s.position),
        CameraFocusTarget::Ship => Some(sim.simulation.ship_state().position),
        CameraFocusTarget::PlayerController => player
            .as_deref()
            .and_then(|state| state.active_position_m())
            .or_else(|| Some(sim.simulation.ship_state().position)),
        CameraFocusTarget::Ghost(ghost_focus) => {
            states.get(ghost_focus.body_id).map(|s| s.position)
        }
        CameraFocusTarget::None => None,
    }
    .unwrap_or(bevy::math::DVec3::ZERO);

    // Star is always at index 0.
    let star_pos = states
        .first()
        .map(|s| s.position)
        .unwrap_or(bevy::math::DVec3::ZERO);

    let offset = focus_pos - star_pos;
    if offset.length_squared() < 1.0e6 {
        return; // Focus is on (or very near) the star; direction undefined.
    }

    // Day/night gate, evaluated at the craft's position (not the camera focus,
    // which may be a remote body in map view) over the body it's gravitationally
    // bound to. `daylight` ∈ [0,1]: 1 with the sun high overhead, 0 on the deep
    // night side. Falls back to full daylight when the craft sits in the star's
    // SOI (interplanetary cruise) — there's no nearby horizon to occlude.
    let craft_pos = player
        .as_deref()
        .and_then(|state| state.active_position_m())
        .unwrap_or_else(|| sim.simulation.ship_state().position);
    let dominant = sim.simulation.dominant_body();
    let daylight = match states.get(dominant) {
        Some(body) if dominant != 0 => {
            let to_center = body.position - craft_pos;
            let r = to_center.length();
            let radius = sim.simulation.bodies()[dominant].radius_m;
            let up = (-to_center).normalize_or_zero();
            let to_sun = (star_pos - craft_pos).normalize_or_zero();
            let sun_elevation = up.dot(to_sun);
            // Slide the terminator to the geometric umbra entry for this
            // altitude. The sun grazes the planet's limb (shadow-cylinder
            // edge, sun treated as infinitely far) when
            //   sun_elevation == -sqrt(1 - (R/r)^2).
            // At the surface (r == R) that threshold is 0 and the whole
            // expression reduces to the terrain shader's
            // `smoothstep(-0.06, 0.12, sun_elevation)`, keeping the on-foot /
            // landed terminator identical to the ground. As the craft climbs,
            // the threshold slides toward -1 so only the true shadow cylinder
            // behind the planet — the high-orbit umbra — darkens the hull,
            // instead of the whole far hemisphere going dark.
            let ratio = if r > 0.0 { (radius / r).min(1.0) } else { 1.0 };
            let threshold = -(1.0 - ratio * ratio).max(0.0).sqrt();
            smoothstep(threshold - 0.06, threshold + 0.12, sun_elevation) as f32
        }
        _ => 1.0,
    };

    let dir_f32 = offset.normalize().as_vec3();
    let illuminance = SUN_DAY_ILLUMINANCE * daylight;
    for (mut transform, mut light) in &mut light_query {
        // DirectionalLight shines along its local -Z, so we look in the light's travel direction.
        transform.look_to(dir_f32, Vec3::Y);
        light.illuminance = illuminance;
    }

    ambient.brightness =
        AMBIENT_NIGHT_BRIGHTNESS + (AMBIENT_DAY_BRIGHTNESS - AMBIENT_NIGHT_BRIGHTNESS) * daylight;
}
