//! Per-frame lighting state: camera exposure, scene lighting struct
//! population (one star + eclipse occluders + planetshine), planet/solid
//! material light updates, sun-light direction.

use bevy::prelude::*;
use thalos_physics::types::BodyStates;
use thalos_planet_rendering::{
    FilmGrain, MAX_ECLIPSE_OCCLUDERS, PlanetHaloMaterial, PlanetMaterial, SceneLighting,
    SolidPlanetMaterial, StarLight,
};

use super::types::{
    CameraExposure, CelestialBody, FrameBodyStates, PlanetMaterials, SimulationState,
    SolidPlanetMaterials, SunLight,
};
use crate::camera::{CameraFocus, CameraFocusTarget};
use crate::coords::{MAP_SCALE, RenderOrigin, SHIP_SCALE};
use crate::view::ViewMode;

/// Sun irradiance at 1 AU in shader units (W/m² scaled). Editor uses the same
/// value — keep them in sync. Per-body intensity is scaled by focus-relative
/// exposure (see `update_planet_light_dirs`) rather than by raw inverse-square
/// falloff, so distant bodies stay legible when the camera focuses on them.
const LIGHT_AT_1AU: f32 = 10.0;

/// Ambient floor. Vacuum has no fill light — night sides are black.
const PLANET_AMBIENT: f32 = 0.0;

const AU_M: f64 = 1.496e11;

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
/// `cache_body_states`, before any consumer reads `CameraExposure`.
pub(super) fn update_camera_exposure(
    cache: Res<FrameBodyStates>,
    focus: Res<CameraFocus>,
    bodies: Query<&CelestialBody>,
    sim: Res<SimulationState>,
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
    const BASE_INTENSITY: f32 = 0.020;
    const MAX_EXTRA: f32 = 0.030;
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
    let mut scene = SceneLighting::default();
    scene.ambient_intensity = PLANET_AMBIENT;

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
pub(super) fn collect_occluders<'a>(
    states: &BodyStates,
    origin: &RenderOrigin,
    scale: f64,
    bodies: impl IntoIterator<Item = &'a CelestialBody>,
) -> Vec<(usize, Vec3, f32)> {
    let mut occluders: Vec<(usize, Vec3, f32)> = Vec::new();
    for body in bodies {
        if body.is_star || body.radius_m < 1.0 {
            continue;
        }
        let Some(state) = states.get(body.body_id) else {
            continue;
        };
        let render_pos = ((state.position - origin.position) * scale).as_vec3();
        let render_radius = ((body.radius_m * scale) as f32).max(0.005);
        occluders.push((body.body_id, render_pos, render_radius));
    }
    occluders
}

/// Updates each planet material's `light_dir` uniform to point from the body
/// toward the star.  Must run after `cache_body_states`.
pub(super) fn update_planet_light_dirs(
    query: Query<(&CelestialBody, &PlanetMaterials)>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
    mut halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
    cache: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    sim: Res<SimulationState>,
    exposure: Res<CameraExposure>,
    view: Res<ViewMode>,
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

    // Compute occluder lists at both scales once per frame.
    let body_iter = || query.iter().map(|(b, _)| b);
    let map_occluders = collect_occluders(states, &origin, MAP_SCALE, body_iter());
    let ship_occluders = collect_occluders(states, &origin, SHIP_SCALE, body_iter());

    // Cloud layer drift: wrap sim time at the body's equatorial cloud
    // period (`TAU / scroll_rate`) so the equator rotates seamlessly
    // across the wrap. Falls back to one sim-day when `scroll_rate` is
    // zero (no drift). Polar latitudes with non-zero differential
    // rotation still seam at each wrap (`TAU * lat_factor` jump), but
    // at a slow multi-day cadence.
    let sim_time = sim.simulation.sim_time();

    for (body, mats) in &query {
        let body_def = &body_defs[body.body_id];
        // Same scale-independent inputs feed both materials; only the
        // scale-dependent fields (radius, occluders, planetshine pos)
        // differ.
        for (handle, halo_handle, occluders, scale, want) in [
            (&mats.map, &mats.map_halo, &map_occluders, MAP_SCALE, do_map),
            (
                &mats.ship,
                &mats.ship_halo,
                &ship_occluders,
                SHIP_SCALE,
                do_ship,
            ),
        ] {
            if !want {
                continue;
            }
            let radius = (body.radius_m * scale) as f32;
            let mut scene = build_scene_lighting(body.body_id, states, occluders, gain);

            // Planetshine: pick the orbital parent, skipping the star.
            // The parent's Bond albedo × color is the effective
            // reflected tint.
            if let Some(parent_id) = body_def.parent {
                let parent_def = &body_defs[parent_id];
                if !matches!(parent_def.kind, thalos_physics::types::BodyKind::Star)
                    && let Some(parent_state) = states.get(parent_id)
                {
                    let parent_render_pos =
                        ((parent_state.position - origin.position) * scale).as_vec3();
                    let parent_radius = (parent_def.radius_m * scale) as f32;
                    let tint = Vec3::new(
                        parent_def.color[0],
                        parent_def.color[1],
                        parent_def.color[2],
                    ) * parent_def.albedo;
                    scene.planetshine_pos_radius = Vec4::new(
                        parent_render_pos.x,
                        parent_render_pos.y,
                        parent_render_pos.z,
                        parent_radius,
                    );
                    scene.planetshine_tint_flag = Vec4::new(tint.x, tint.y, tint.z, 1.0);
                }
            }

            let scroll = materials
                .get(handle)
                .map(|mat| mat.atmosphere.cloud_dynamics.x.abs() as f64)
                .unwrap_or(0.0);
            let period = if scroll > 1e-9 {
                std::f64::consts::TAU / scroll
            } else {
                86_400.0
            };
            let cloud_time = (sim_time - (sim_time / period).floor() * period) as f32;

            if let Some(mat) = materials.get_mut(handle) {
                mat.params.radius = radius;
                mat.params.scene = scene.clone();
                // Drive the cloud layer's time uniform. Bodies without a
                // cloud layer have `cloud_albedo_coverage.w = 0`, so the
                // shader skips the layer entirely and this value is ignored.
                // Scroll rate is scale-independent (rad/s on the unit
                // sphere), so the period is the same for both materials.
                mat.atmosphere.cloud_dynamics.y = cloud_time;
            }

            if let Some(mat) = halo_materials.get_mut(halo_handle) {
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
    cache: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    sim: Res<SimulationState>,
    exposure: Res<CameraExposure>,
    view: Res<ViewMode>,
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
    let map_occluders = collect_occluders(states, &origin, MAP_SCALE, body_iter());
    let ship_occluders = collect_occluders(states, &origin, SHIP_SCALE, body_iter());

    for (body, mats) in &query {
        let body_def = &body_defs[body.body_id];
        for (handle, occluders, scale, want) in [
            (&mats.map, &map_occluders, MAP_SCALE, do_map),
            (&mats.ship, &ship_occluders, SHIP_SCALE, do_ship),
        ] {
            if !want {
                continue;
            }
            let Some(mat) = materials.get_mut(handle) else {
                continue;
            };
            mat.params.radius = ((body.radius_m * scale) as f32).max(0.005);
            let mut scene = build_scene_lighting(body.body_id, states, occluders, gain);

            if let Some(parent_id) = body_def.parent {
                let parent_def = &body_defs[parent_id];
                if !matches!(parent_def.kind, thalos_physics::types::BodyKind::Star)
                    && let Some(parent_state) = states.get(parent_id)
                {
                    let parent_render_pos =
                        ((parent_state.position - origin.position) * scale).as_vec3();
                    let parent_radius = (parent_def.radius_m * scale) as f32;
                    let tint = Vec3::new(
                        parent_def.color[0],
                        parent_def.color[1],
                        parent_def.color[2],
                    ) * parent_def.albedo;
                    scene.planetshine_pos_radius = Vec4::new(
                        parent_render_pos.x,
                        parent_render_pos.y,
                        parent_render_pos.z,
                        parent_radius,
                    );
                    scene.planetshine_tint_flag = Vec4::new(tint.x, tint.y, tint.z, 1.0);
                }
            }

            mat.params.scene = scene;
        }
    }
}

/// Point the directional sun light from the star toward the camera's focus body.
pub(super) fn update_sun_light(
    cache: Res<FrameBodyStates>,
    focus: Res<CameraFocus>,
    sim: Res<SimulationState>,
    mut light_query: Query<&mut Transform, With<SunLight>>,
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
        CameraFocusTarget::Ghost(ghost_focus) => {
            states.get(ghost_focus.body_id).map(|s| s.position)
        }
        CameraFocusTarget::None => None,
    }
    .unwrap_or(bevy::math::DVec3::ZERO);

    // Star is always at index 0.
    let star_pos = states
        .get(0)
        .map(|s| s.position)
        .unwrap_or(bevy::math::DVec3::ZERO);

    let offset = focus_pos - star_pos;
    if offset.length_squared() < 1.0e6 {
        return; // Focus is on (or very near) the star; direction undefined.
    }

    let dir_f32 = offset.normalize().as_vec3();
    for mut transform in &mut light_query {
        // DirectionalLight shines along its local -Z, so we look in the light's travel direction.
        transform.look_to(dir_f32, Vec3::Y);
    }
}
