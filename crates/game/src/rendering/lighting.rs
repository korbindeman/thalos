//! Per-frame lighting state: camera exposure, scene lighting struct
//! population (one star + eclipse occluders + planetshine), planet/solid
//! material light updates, sun-light direction.

use bevy::prelude::*;
use thalos_body_render::{
    AU_M, FilmGrain, LIGHT_AT_1AU, MAX_ECLIPSE_OCCLUDERS, SceneLighting, SolidPlanetMaterial,
    StarLight,
};
use thalos_physics_canonical::types::BodyStates;

use super::types::{
    CameraExposure, CelestialBody, MoonLight, PlanetshineTints, SimulationState, SolarSystemState,
    SolidPlanetMaterials, SunLight,
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

/// Push lighting state into every [`SolidPlanetMaterial`] each frame.
///
/// Mirrors the terrestrial planet light update for placeholder bodies (no terrain
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
            if dirty && let Some(mut mat) = materials.get_mut(handle) {
                mat.params.radius = radius;
                mat.params.scene = scene;
            }
        }
    }
}

/// Calibration: Bevy lux per unit of shading-spine flux. The Bevy-lit
/// `StandardMaterial` surfaces (hull, gear, structures, runway) take their sun
/// illuminance from the SAME heliocentric flux the spine gives every terrain /
/// vegetation surface (`build_scene_lighting`: `LIGHT_AT_1AU·(AU/d)²·gain`),
/// converted to lux through this one constant. At the homeworld (spine flux ≈ 10,
/// focus gain ≈ 1) it lands the daytime hull at ~10 000 lux — the value the old
/// flat `SUN_DAY_ILLUMINANCE` pinned — but now scaling with heliocentric distance
/// and exposure, so a ship at a far body dims like the ground beneath it instead
/// of staying noon-bright. Tune from a `just game runway` noon screenshot until the
/// hull reads at the same brightness as the terrain beside it.
const LUX_PER_SPINE_FLUX: f32 = 1_000.0;
/// Ambient (sky-fill) brightness in full daylight and on the deep night side, in
/// Bevy lux, for the Bevy-PBR surfaces (hull, gear, structures, runway). The day
/// value approximates outdoor **sky fill**: at ~700 lux against the 10 000-lux sun
/// it gives a ~7:1 lit:shadow ratio, so a dielectric structure's shadowed faces
/// read as dim sky-lit grey instead of the near-black the old flat 50 lux produced
/// (200:1 — physically far too contrasty; that was the white-top/black-side look).
/// Flat (non-directional), so this is an INTERIM stand-in for the proper
/// hemispheric sky ambient — F4 of the graphics-fidelity foundation replaces this
/// whole `GlobalAmbientLight` with SH from the sky-view LUT (and gives the metallic
/// hull a real reflection, which diffuse ambient can't). See `docs/graphics_fidelity.md` §3.
const AMBIENT_DAY_BRIGHTNESS: f32 = 700.0;
const AMBIENT_NIGHT_BRIGHTNESS: f32 = 4.0;
/// Sky-blue tint for the daytime ambient fill, so shadowed faces read as
/// sky-lit (cool) rather than neutral grey — a coarse stand-in for the terrain's
/// blue sky-dome fill. Tune alongside [`AMBIENT_DAY_BRIGHTNESS`].
const AMBIENT_DAY_TINT: Color = Color::srgb(0.62, 0.72, 0.95);

/// `smoothstep`, matching the WGSL builtin the terrain shader uses so the
/// CPU-side day/night gate lines up with the ground's terminator exactly.
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The single day/night terminator for the Bevy-lit (`StandardMaterial`) surfaces
/// — hull, gear, structures, runway — so they fade across the terminator in
/// lockstep with the terrain shader AND with each other. One definition consumed
/// by both [`update_sun_light`] (the sun/ambient gate) and [`update_moon_light`]
/// (its night gate `= 1 - surface_daylight`), replacing the two hand-copied
/// smoothsteps that had drifted apart.
///
/// Mirrors `body_terrain.wgsl`'s `daylight = smoothstep(-0.06, 0.12, dot(up,
/// sun_dir))` at the surface (`altitude_ratio = R/r = 1` → threshold 0), and
/// slides the terminator to the geometric umbra entry as the craft climbs, so only
/// the true shadow cylinder behind the body darkens a high orbit (not the whole far
/// hemisphere). `sun_elevation = dot(local_up, to_sun)`, `altitude_ratio = R_body /
/// r_craft`. Returns 1 = full day, 0 = night.
fn surface_daylight(sun_elevation: f64, altitude_ratio: f64) -> f64 {
    let ratio = altitude_ratio.clamp(0.0, 1.0);
    let threshold = -((1.0 - ratio * ratio).max(0.0)).sqrt();
    smoothstep(threshold - 0.06, threshold + 0.12, sun_elevation)
}

/// Point the directional sun light from the star toward the camera's focus
/// body, and set its illuminance + the global ambient floor as a *projection of
/// the shading spine's lighting* (F1 of the graphics-fidelity unification —
/// `docs/graphics_fidelity.md` §3).
///
/// The ship hull, gear, structures, and runway are Bevy `StandardMaterial`s lit by
/// this one directional light + `GlobalAmbientLight`, while the terrain / vegetation
/// are lit by the `thalos::lighting` spine. To keep the two universes in lockstep
/// this system drives the Bevy lights from the *same two quantities* the spine uses:
/// (1) the heliocentric flux `LIGHT_AT_1AU·(AU/d)²·gain` (`build_scene_lighting`),
/// so the hull dims with distance + exposure instead of the old flat 10 000 lux that
/// lit a ship at a far body identically to one at Thalos; and (2) one shared
/// terminator ([`surface_daylight`], mirroring `body_terrain.wgsl`), so direct sun +
/// sky-fill fade across the same day/night band the ground does. All surfaces dim
/// together as the sun sets or the body eclipses the craft. Deep space (dominant
/// body is the star) keeps full daylight — no body to block the sun. **Sole writer**
/// of the sun `DirectionalLight` + `GlobalAmbientLight`.
pub(super) fn update_sun_light(
    cache: Res<SolarSystemState>,
    focus: Res<CameraFocus>,
    sim: Res<SimulationState>,
    exposure: Res<CameraExposure>,
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
            // One shared terminator (altitude-aware umbra entry) — see
            // `surface_daylight`. At the surface this reduces to the terrain
            // shader's `smoothstep(-0.06, 0.12, sun_elevation)`.
            let ratio = if r > 0.0 { radius / r } else { 1.0 };
            surface_daylight(up.dot(to_sun), ratio) as f32
        }
        _ => 1.0,
    };

    // Heliocentric flux the craft receives, in the SAME units the shading spine
    // gives every terrain/vegetation surface (`build_scene_lighting`):
    // LIGHT_AT_1AU · (AU/d)² · exposure_gain. Routing the Bevy sun through it ties
    // the StandardMaterial hull/structures to the spine's flux, so they dim with
    // heliocentric distance + exposure exactly like the ground.
    let helio_d_m = (craft_pos - star_pos).length();
    let au_over_d = (AU_M / helio_d_m.max(1.0)) as f32;
    let flux = LIGHT_AT_1AU * au_over_d * au_over_d * exposure.gain;

    let dir_f32 = offset.normalize().as_vec3();
    let illuminance = LUX_PER_SPINE_FLUX * flux * daylight;
    for (mut transform, mut light) in &mut light_query {
        // DirectionalLight shines along its local -Z, so we look in the light's travel direction.
        transform.look_to(dir_f32, Vec3::Y);
        light.illuminance = illuminance;
    }

    // Sky-fill ambient is scattered sunlight, so dim its day component by the same
    // flux (normalised to the homeworld nominal, capped at 1) — a distant dim sun
    // must not leave a fixed bright ambient out-shining it — while keeping the cool
    // night floor.
    let flux_norm = (flux / LIGHT_AT_1AU).clamp(0.0, 1.0);
    let ambient_day = AMBIENT_DAY_BRIGHTNESS * flux_norm;
    ambient.brightness =
        AMBIENT_NIGHT_BRIGHTNESS + (ambient_day - AMBIENT_NIGHT_BRIGHTNESS).max(0.0) * daylight;
    // Cool sky-blue tint so shadowed faces read as sky-lit, not neutral grey
    // (interim stand-in for the terrain's hemispheric blue sky fill).
    ambient.color = AMBIENT_DAY_TINT;
}

/// Full-moon illuminance for the `StandardMaterial` hull + structures, in Bevy
/// lux. The terrain has its own moonlight term (`body_terrain.wgsl`); this is the
/// matching directional light so the craft and surface buildings catch the same
/// moonlight. Soft and night-gated; tune from a night screenshot.
const MOON_FULL_LUX: f32 = 60.0;
/// Reference "bright moon" reflectance shape (albedo_luminance × (R/d)²) mapping
/// to full brightness — mirrors `compute_moonlight` in `ground_terrain.rs` so the
/// hull and the ground agree on which moon is brightest and how bright it is.
const MOON_REF_SHAPE: f64 = 3.0e-6;

/// Drive the [`MoonLight`] directional light from the brightest child moon of the
/// body the craft is on. Mirrors the terrain's `compute_moonlight`: Lambert phase
/// × per-moon size/albedo/distance, night-gated over the craft's local horizon,
/// and faded in as the moon rises. Off (illuminance 0) by day or with no lit moon
/// up — so it only ever adds the night fill, never competes with the sun.
pub(super) fn update_moon_light(
    cache: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    player: Option<Res<crate::player_controller::PlayerControllerState>>,
    mut light_query: Query<(&mut Transform, &mut DirectionalLight), With<MoonLight>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let Ok((mut transform, mut light)) = light_query.single_mut() else {
        return;
    };

    let star_pos = states
        .first()
        .map(|s| s.position)
        .unwrap_or(bevy::math::DVec3::ZERO);
    let craft_pos = player
        .as_deref()
        .and_then(|state| state.active_position_m())
        .unwrap_or_else(|| sim.simulation.ship_state().position);
    let dominant = sim.simulation.dominant_body();
    let dominant_pos = states.get(dominant).map(|b| b.position);

    // Night gate over the dominant body's horizon at the craft (mirrors
    // `update_sun_light`): moonlight fades in as the sun sets. Zero in the star's
    // own SOI (no nearby body to set night) — moonlight is a surface effect.
    let night = match dominant_pos {
        Some(dpos) if dominant != 0 => {
            // Same terminator the sun + ground use, so moonlight fades in exactly
            // as the sun's day-fill fades out (no overlap band, no gap).
            let to_center = dpos - craft_pos;
            let r = to_center.length();
            let radius = sim.simulation.bodies()[dominant].radius_m;
            let to_sun = (star_pos - craft_pos).normalize_or_zero();
            let up = (-to_center).normalize_or_zero();
            let ratio = if r > 0.0 { radius / r } else { 1.0 };
            1.0 - surface_daylight(up.dot(to_sun), ratio) as f32
        }
        _ => 0.0,
    };

    if night <= 0.0 || dominant_pos.is_none() {
        light.illuminance = 0.0;
        return;
    }
    let up = (craft_pos - dominant_pos.unwrap()).normalize_or_zero();

    let mut best_lux = 0.0f32;
    let mut best_dir = bevy::math::DVec3::Y;
    let mut best_color = Color::WHITE;
    for moon in sim.simulation.bodies() {
        if !matches!(moon.kind, thalos_world::BodyKind::Moon) || moon.parent != Some(dominant) {
            continue;
        }
        let Some(moon_state) = states.get(moon.id) else {
            continue;
        };
        let to_moon = moon_state.position - craft_pos;
        let d = to_moon.length();
        if d <= 0.0 {
            continue;
        }
        let moon_dir = to_moon / d;
        // Fade in as the moon clears the local horizon; skip when it's down.
        let horizon = smoothstep(0.0, 0.10, up.dot(moon_dir)) as f32;
        if horizon <= 0.0 {
            continue;
        }
        // Lambert phase as seen from the craft: angle AT the moon between the
        // star and the craft (full moon → 0 → phase 1).
        let to_star_from_moon = (star_pos - moon_state.position).normalize_or_zero();
        let to_craft_from_moon = (craft_pos - moon_state.position).normalize_or_zero();
        let cos_g = to_star_from_moon.dot(to_craft_from_moon).clamp(-1.0, 1.0);
        let g = cos_g.acos();
        let phase = ((g.sin() + (std::f64::consts::PI - g) * cos_g) / std::f64::consts::PI)
            .clamp(0.0, 1.0);

        let color_lin = Color::srgb(moon.color[0], moon.color[1], moon.color[2]).to_linear();
        let albedo_lum = (0.2126 * color_lin.red + 0.7152 * color_lin.green
            + 0.0722 * color_lin.blue) as f64;
        let ang = moon.radius_m / d;
        let shape = albedo_lum * ang * ang;
        let rel = (shape / MOON_REF_SHAPE).clamp(0.0, 1.5);

        let lux = MOON_FULL_LUX * (phase * rel) as f32 * night * horizon;
        if lux > best_lux {
            best_lux = lux;
            best_dir = moon_dir;
            best_color = Color::srgb(moon.color[0], moon.color[1], moon.color[2]);
        }
    }

    light.illuminance = best_lux;
    if best_lux > 0.0 {
        light.color = best_color;
        // DirectionalLight travels along its local -Z: aim it from the moon down
        // toward the surface, i.e. opposite the craft→moon direction.
        transform.look_to((-best_dir).as_vec3(), Vec3::Y);
    }
}
