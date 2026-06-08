//! Per-frame material parameter updates for paths that aren't covered by
//! `lighting` (which owns terrestrial planet light state). Includes:
//!
//! - Gas-giant material (rotation, occluders, exposure)
//! - Ring material (planet center, scene lighting)
//! - Cloud-band rotation phases (latitudinal decomposition advanced on
//!   the CPU and uploaded to the planet material each frame)

use bevy::prelude::*;
use thalos_body_render::{GasGiantMaterial, PlanetHaloMaterial, PlanetMaterial, RingMaterial};
use thalos_physics_canonical::types::BodyStates;

use super::lighting::{build_scene_lighting, collect_occluders};
use super::types::{
    CameraExposure, CelestialBody, GasGiantMaterials, MapRingMaterial, PlanetMaterials,
    ShipRingMaterial, SimulationState, SolarSystemState,
};
use crate::coords::{MAP_SCALE, RenderOrigin, SHIP_SCALE};
use crate::view::ViewMode;

/// Push camera/light/rotation state into every `GasGiantMaterial` each
/// frame. Mirrors `update_planet_light_dirs` for baked planets but
/// operates on the smaller `GasGiantParams` uniform.
///
/// Keeping this in its own system lets the scheduler parallelise with
/// the terrestrial path — the two queries are disjoint.
pub(super) fn update_gas_giant_params(
    query: Query<(&CelestialBody, &GasGiantMaterials)>,
    all_bodies: Query<&CelestialBody>,
    mut materials: ResMut<Assets<GasGiantMaterial>>,
    cache: Res<SolarSystemState>,
    origin: Res<RenderOrigin>,
    sim: Res<SimulationState>,
    exposure: Res<CameraExposure>,
    view: Res<ViewMode>,
    mut map_occluders: Local<Vec<(usize, Vec3, f32)>>,
    mut ship_occluders: Local<Vec<(usize, Vec3, f32)>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let body_defs = sim.simulation.bodies();
    let sim_time = sim.simulation.sim_time();
    let gain = exposure.gain;

    // See note on `update_planet_light_dirs` — gate inactive scale.
    let force_both = view.is_changed();
    let do_map = force_both || matches!(*view, ViewMode::Map);
    let do_ship = force_both || matches!(*view, ViewMode::Ship);

    collect_occluders(
        &mut map_occluders,
        states,
        &origin,
        MAP_SCALE,
        all_bodies.iter(),
    );
    collect_occluders(
        &mut ship_occluders,
        states,
        &origin,
        SHIP_SCALE,
        all_bodies.iter(),
    );
    let map_slice: &[(usize, Vec3, f32)] = &map_occluders;
    let ship_slice: &[(usize, Vec3, f32)] = &ship_occluders;

    // Raw sim seconds — the gas-giant shader uses this for differential
    // rotation scroll, edge-wave phase, and edge vortex chain epoch
    // hashing. Modulo a day-scale period so the f32 stays precise.
    let time_mod = (sim_time % 86_400.0) as f32;

    for (body, mats) in &query {
        let body_def = &body_defs[body.body_id];

        // Rotation phase: advance bands at the body's real rotation
        // rate. sim_time is seconds, rotation_period_s is seconds, so
        // the modulo drops the large integer part before conversion
        // to f32 and keeps precision high at long run times.
        let period = body_def.rotation_period_s.max(1.0);
        let phase = ((sim_time % period) / period) as f32 * std::f32::consts::TAU;

        // Orientation: axial tilt around the X axis. Gas giants aren't
        // tidally locked, so rotation is already folded into the band
        // phase above; the quaternion here only carries the tilt.
        let tilt = body_def.axial_tilt_rad as f32;
        let q = Quat::from_rotation_x(tilt);
        let orientation = Vec4::new(q.x, q.y, q.z, q.w);

        for (handle, occluders, scale, want) in [
            (&mats.map, map_slice, MAP_SCALE, do_map),
            (&mats.ship, ship_slice, SHIP_SCALE, do_ship),
        ] {
            if !want {
                continue;
            }
            let Some(mat) = materials.get_mut(handle) else {
                continue;
            };
            mat.params.radius = (body.radius_m * scale) as f32;
            mat.params.elapsed_time = time_mod;
            mat.params.scene = build_scene_lighting(body.body_id, states, occluders, gain);
            mat.params.rotation_phase = phase;
            mat.params.orientation = orientation;
        }
    }
}

/// Per-frame `RingParams` update.
///
/// Ring materials need two pieces of live state:
///
/// 1. **Sun direction** — the shader uses it for Lambert + forward
///    scatter and for the planet-shadow ray test.
/// 2. **Planet center in world space** — the ring mesh is a child of
///    the body entity, so the body moves, so the center used by the
///    shadow ray must move with it.
///
/// Light intensity is re-exposed each frame against the current
/// camera exposure gain, matching `update_gas_giant_params` so the
/// ring and disk stay photometrically consistent.
pub(super) fn update_ring_params(
    map_rings: Query<(&ChildOf, &MapRingMaterial)>,
    ship_rings: Query<(&ChildOf, &ShipRingMaterial)>,
    body_query: Query<&CelestialBody>,
    mut materials: ResMut<Assets<RingMaterial>>,
    origin: Res<RenderOrigin>,
    cache: Res<SolarSystemState>,
    exposure: Res<CameraExposure>,
    view: Res<ViewMode>,
    mut occluder_buf: Local<Vec<(usize, Vec3, f32)>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let gain = exposure.gain;

    // See note on `update_planet_light_dirs` — gate inactive scale.
    let force_both = view.is_changed();
    let do_map = force_both || matches!(*view, ViewMode::Map);
    let do_ship = force_both || matches!(*view, ViewMode::Ship);

    if do_map {
        collect_occluders(
            &mut occluder_buf,
            states,
            &origin,
            MAP_SCALE,
            body_query.iter(),
        );
        for (parent, handle) in &map_rings {
            write_ring_params(
                &body_query,
                states,
                &mut materials,
                handle.0.clone(),
                parent.0,
                &origin,
                MAP_SCALE,
                &occluder_buf,
                gain,
            );
        }
    }
    if do_ship {
        collect_occluders(
            &mut occluder_buf,
            states,
            &origin,
            SHIP_SCALE,
            body_query.iter(),
        );
        for (parent, handle) in &ship_rings {
            write_ring_params(
                &body_query,
                states,
                &mut materials,
                handle.0.clone(),
                parent.0,
                &origin,
                SHIP_SCALE,
                &occluder_buf,
                gain,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_ring_params(
    body_query: &Query<&CelestialBody>,
    states: &BodyStates,
    materials: &mut Assets<RingMaterial>,
    handle: Handle<RingMaterial>,
    parent: Entity,
    origin: &RenderOrigin,
    scale: f64,
    occluders: &[(usize, Vec3, f32)],
    gain: f32,
) {
    let Ok(body) = body_query.get(parent) else {
        return;
    };
    let Some(mat) = materials.get_mut(&handle) else {
        return;
    };

    let body_pos_m = states
        .get(body.body_id)
        .map(|s| s.position)
        .unwrap_or_default();

    // Planet center in the render frame — same transform the matching
    // body mesh uses, so the shadow ray tests the right sphere
    // regardless of the rolling render origin.
    let center_render = ((body_pos_m - origin.position) * scale).as_vec3();
    mat.params.planet_center_radius = Vec4::new(
        center_render.x,
        center_render.y,
        center_render.z,
        (body.radius_m * scale) as f32,
    );
    mat.params.scene = build_scene_lighting(body.body_id, states, occluders, gain);
}

// ---------------------------------------------------------------------------
// Cloud rotation bands (latitudinal decomposition)
//
// The impostor shader samples the cloud cube at two bands that bracket
// a fragment's latitude and blends by `sin²(lat)` position. Each band
// has its own rigid rotation speed `ω_i = scroll × (1 − diff × sin²(lat_i))`,
// and each band's phase is accumulated on the CPU mod TAU in f64 — so
// phase wraps cause no shader-visible discontinuity, differential
// rotation is preserved, and state is trivially persistable (16 × f64
// per cloudy body).
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
pub(super) struct LastCloudBandUpdate(Option<f64>);

pub(super) fn update_cloud_bands(
    mut last_time: ResMut<LastCloudBandUpdate>,
    sim: Res<SimulationState>,
    mut solar_system: ResMut<SolarSystemState>,
    query: Query<(&CelestialBody, &PlanetMaterials)>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
    mut halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
    view: Res<ViewMode>,
) {
    let now = sim.simulation.sim_time();
    let dt = last_time.0.map(|prev| now - prev).unwrap_or(0.0);
    last_time.0 = Some(now);
    if dt == 0.0 {
        return;
    }

    // See note on `update_planet_light_dirs` — gate inactive scale.
    let force_both = view.is_changed();
    let do_map = force_both || matches!(*view, ViewMode::Map);
    let do_ship = force_both || matches!(*view, ViewMode::Ship);

    for (body, mats) in &query {
        // Cloud drift is physical runtime state, not a render-material
        // side effect. Advance the canonical body environment once, then
        // mirror the packed phases to every projection material.
        let Some(clouds) = solar_system
            .environment_mut(body.body_id)
            .and_then(|env| env.cloud_bands.as_mut())
        else {
            continue;
        };
        clouds.advance(dt);
        let p = clouds.phases;

        let bands_a = Vec4::new(p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32);
        let bands_b = Vec4::new(p[4] as f32, p[5] as f32, p[6] as f32, p[7] as f32);
        let bands_c = Vec4::new(p[8] as f32, p[9] as f32, p[10] as f32, p[11] as f32);
        let bands_d = Vec4::new(p[12] as f32, p[13] as f32, p[14] as f32, p[15] as f32);
        for (handle, halo_handle, want) in [
            (&mats.map, &mats.map_halo, do_map),
            (&mats.ship, &mats.ship_halo, do_ship),
        ] {
            if !want {
                continue;
            }
            let Some(mat) = materials.get_mut(handle) else {
                if let Some(mat) = halo_materials.get_mut(halo_handle) {
                    mat.atmosphere.cloud_bands_a = bands_a;
                    mat.atmosphere.cloud_bands_b = bands_b;
                    mat.atmosphere.cloud_bands_c = bands_c;
                    mat.atmosphere.cloud_bands_d = bands_d;
                }
                continue;
            };
            mat.atmosphere.cloud_bands_a = bands_a;
            mat.atmosphere.cloud_bands_b = bands_b;
            mat.atmosphere.cloud_bands_c = bands_c;
            mat.atmosphere.cloud_bands_d = bands_d;
            if let Some(mat) = halo_materials.get_mut(halo_handle) {
                mat.atmosphere.cloud_bands_a = bands_a;
                mat.atmosphere.cloud_bands_b = bands_b;
                mat.atmosphere.cloud_bands_c = bands_c;
                mat.atmosphere.cloud_bands_d = bands_d;
            }
        }
    }
}
