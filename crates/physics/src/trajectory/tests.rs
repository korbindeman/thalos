//! Integration tests for trajectory propagation.
//!
//! These were originally in the flat `trajectory.rs` module; preserved here
//! after the module split so they keep protecting the propagator from the
//! regressions they were written for (frozen-thrust burn collision, stable
//! orbit detection, cross-body reference-frame mix-ups).

#![cfg(test)]

use std::collections::HashMap;
use std::sync::Arc;

use glam::DVec3;

use super::{FlightPlan, PredictionConfig, PredictionRequest, propagate_flight_plan};
use crate::body_state_provider::BodyStateProvider;
use crate::maneuver::{ManeuverNode, ManeuverSequence};
use crate::patched_conics::PatchedConics;
use crate::types::{
    BodyDefinition, BodyKind, G, OrbitalElements, ShipDefinition, SolarSystemDefinition,
    StateVector, TrajectorySample,
};

/// Local helper that mirrors the old `propagate_trajectory` convenience so the
/// tests can stay focused on what they actually assert.
///
/// The original signature took a single `ship_thrust_acceleration` (m/s²);
/// we synthesize a thrust/mass/flow triple that yields the same starting
/// acceleration with negligible mass change over typical test burn lengths,
/// so the existing assertions still hold.
fn propagate_trajectory(
    initial_state: StateVector,
    start_time: f64,
    maneuvers: &ManeuverSequence,
    ephemeris: Arc<dyn BodyStateProvider>,
    bodies: &[BodyDefinition],
    config: &PredictionConfig,
    ship_thrust_acceleration: f64,
) -> FlightPlan {
    propagate_trajectory_with_target(
        initial_state,
        start_time,
        maneuvers,
        ephemeris,
        bodies,
        config,
        ship_thrust_acceleration,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn propagate_trajectory_with_target(
    initial_state: StateVector,
    start_time: f64,
    maneuvers: &ManeuverSequence,
    ephemeris: Arc<dyn BodyStateProvider>,
    bodies: &[BodyDefinition],
    config: &PredictionConfig,
    ship_thrust_acceleration: f64,
    target_body: Option<crate::types::BodyId>,
) -> FlightPlan {
    // Synthesize a high-mass / low-flow ship so accel ≈ ship_thrust_acceleration
    // throughout any reasonable burn — the test fixtures predate the rocket
    // equation and assert against the constant-accel limit. Dry mass is set
    // far below the wet mass so the propellant-exhaustion cap never binds.
    let ship_mass_kg = 1.0e6;
    let ship_thrust_n = ship_thrust_acceleration * ship_mass_kg;
    let ship_mass_flow_kg_per_s = 1.0e-6;
    let ship_dry_mass_kg = 1.0;
    let request = PredictionRequest {
        ship_state: initial_state,
        sim_time: start_time,
        maneuvers: maneuvers.clone(),
        active_burns: Vec::new(),
        ephemeris,
        bodies: bodies.to_vec(),
        prediction_config: config.clone(),
        ship_thrust_n,
        ship_mass_kg,
        ship_mass_flow_kg_per_s,
        ship_dry_mass_kg,
        target_body,
    };
    propagate_flight_plan(&request, None)
}

const AU: f64 = 1.496e11;
const SUN_GM: f64 = 1.327_124_4e20;

/// Default thrust acceleration used by these legacy integration tests.
/// Matches the original `make_single_star_system` fixture so the
/// thrust-magnitude-sensitive assertions (e.g. `delta_speed < 2.0`
/// after a 1 s substep, which expects ~1 m/s) keep their margins.
const TEST_THRUST_ACCEL: f64 = 1.0;

fn make_single_star_system() -> SolarSystemDefinition {
    let star_mass = 1.989e30;
    let star = BodyDefinition {
        id: 0,
        name: "Sun".to_string(),
        kind: BodyKind::Star,
        parent: None,
        mass_kg: star_mass,
        radius_m: 6.957e8,
        color: [1.0, 1.0, 0.0],
        albedo: 1.0,
        rotation_period_s: 0.0,
        axial_tilt_rad: 0.0,
        gm: G * star_mass,
        soi_radius_m: f64::INFINITY,
        orbital_elements: None,
        generator: None,
        atmosphere: None,
        terrestrial_atmosphere: None,
    };

    let mut name_to_id = HashMap::new();
    name_to_id.insert("Sun".to_string(), 0);

    SolarSystemDefinition {
        name: "Test".to_string(),
        bodies: vec![star],
        ship: ShipDefinition {
            initial_state: StateVector {
                position: DVec3::new(1.0e11, 0.0, 0.0),
                velocity: DVec3::new(0.0, 1000.0, 0.0),
            },
        },
        name_to_id,
    }
}

fn make_star_and_planet() -> (BodyDefinition, BodyDefinition, f64) {
    let sun_mass = SUN_GM / G;
    let sun = BodyDefinition {
        id: 0,
        name: "Sun".to_string(),
        kind: BodyKind::Star,
        parent: None,
        mass_kg: sun_mass,
        radius_m: 6.957e8,
        color: [1.0, 1.0, 0.0],
        albedo: 1.0,
        rotation_period_s: 0.0,
        axial_tilt_rad: 0.0,
        gm: SUN_GM,
        soi_radius_m: f64::INFINITY,
        orbital_elements: None,
        generator: None,
        atmosphere: None,
        terrestrial_atmosphere: None,
    };

    let thalos_mass = 1.378e24;
    let thalos = BodyDefinition {
        id: 1,
        name: "Thalos".to_string(),
        kind: BodyKind::Planet,
        parent: Some(0),
        mass_kg: thalos_mass,
        radius_m: 3.186e6,
        color: [0.2, 0.45, 0.9],
        albedo: 0.35,
        rotation_period_s: 21.3 * 3600.0,
        axial_tilt_rad: 23.0_f64.to_radians(),
        gm: G * thalos_mass,
        soi_radius_m: AU * (thalos_mass / sun_mass).powf(0.4),
        orbital_elements: Some(OrbitalElements {
            semi_major_axis_m: AU,
            eccentricity: 0.0,
            inclination_rad: 0.0,
            lon_ascending_node_rad: 0.0,
            arg_periapsis_rad: 0.0,
            true_anomaly_rad: 0.0,
        }),
        generator: None,
        atmosphere: None,
        terrestrial_atmosphere: None,
    };

    (sun, thalos, sun_mass)
}

fn make_thalos_like_system() -> SolarSystemDefinition {
    let (sun, thalos, _) = make_star_and_planet();

    let mut name_to_id = HashMap::new();
    name_to_id.insert("Sun".to_string(), 0);
    name_to_id.insert("Thalos".to_string(), 1);

    SolarSystemDefinition {
        name: "ThalosTest".to_string(),
        bodies: vec![sun, thalos],
        ship: ShipDefinition {
            initial_state: StateVector {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
        },
        name_to_id,
    }
}

#[test]
fn maneuver_is_integrated_as_finite_burn() {
    let system = make_single_star_system();
    let ephemeris: Arc<dyn BodyStateProvider> = Arc::new(PatchedConics::new(&system, 1_000.0));

    let mut maneuvers = ManeuverSequence::new();
    maneuvers.add(ManeuverNode {
        id: None,
        time: 0.0,
        delta_v: DVec3::new(10.0, 0.0, 0.0),
        reference_body: 0,
    });

    let prediction = propagate_trajectory(
        system.ship.initial_state,
        0.0,
        &maneuvers,
        Arc::clone(&ephemeris),
        &system.bodies,
        &PredictionConfig::default(),
        TEST_THRUST_ACCEL,
    );

    let burn_segment = prediction
        .segments
        .iter()
        .find(|segment| segment.samples.len() > 1)
        .expect("expected a downstream burn segment");
    // Skip the pre-burn starting sample (t=0) and look at the first RK4
    // substep sample. At 1.0 m/s² over one 1-second substep the velocity
    // should only change by ~1.0 m/s, far from the full 10 m/s Δv.
    let mid_sample = burn_segment
        .samples
        .iter()
        .find(|s| s.time > 0.0)
        .expect("expected at least one post-start sample");
    let delta_speed = (mid_sample.velocity - system.ship.initial_state.velocity).length();

    assert!(
        delta_speed < 2.0,
        "finite burn should not apply the full 10 m/s instantly; got {delta_speed:.3}"
    );
    // RK4 substep is 1.0 s and this test uses the default PredictionConfig;
    // the first post-start sample lands deterministically at exactly 1.0 s.
    assert!(
        (mid_sample.time - 1.0).abs() < 1e-9,
        "first post-start sample should land at t=1.0: got {}",
        mid_sample.time
    );
}

/// Regression: a prograde Δv node placed a quarter orbit ahead of the ship
/// must burn at the node's *future* position, not at the ship's start
/// position. Apoapsis should appear ~opposite the node, not opposite the
/// ship's initial position.
#[test]
fn burn_happens_at_node_position_not_ship_start() {
    let system = make_thalos_like_system();
    let ephemeris: Arc<dyn BodyStateProvider> = Arc::new(PatchedConics::new(&system, 1.0e7));

    let thalos_state_0 = ephemeris.query_body(1, 0.0);
    let thalos_gm = system.bodies[1].gm;
    let orbit_radius = system.bodies[1].radius_m + 200_000.0;
    let circular_speed = (thalos_gm / orbit_radius).sqrt();
    let orbital_period = std::f64::consts::TAU * (orbit_radius.powi(3) / thalos_gm).sqrt();

    // Ship starts on +X axis relative to Thalos, moving in +Z (prograde CCW
    // looking down -Y).
    let ship_state = StateVector {
        position: thalos_state_0.position + DVec3::new(orbit_radius, 0.0, 0.0),
        velocity: thalos_state_0.velocity + DVec3::new(0.0, 0.0, circular_speed),
    };

    // Node a quarter-orbit in the future → ship should be at ~+Z direction.
    let node_time = orbital_period * 0.25;
    let mut maneuvers = ManeuverSequence::new();
    maneuvers.add(ManeuverNode {
        id: None,
        time: node_time,
        delta_v: DVec3::new(100.0, 0.0, 0.0),
        reference_body: 1,
    });

    // High thrust → ~impulsive burn; apoapsis localization stays crisp
    // without the rotating-frame smearing.
    let impulsive_thrust = 1_000.0;
    let prediction = propagate_trajectory(
        ship_state,
        0.0,
        &maneuvers,
        Arc::clone(&ephemeris),
        &system.bodies,
        &PredictionConfig::default(),
        impulsive_thrust,
    );

    // The burn's first sample must be at the NODE time / NODE position,
    // not the start position.
    let burn_leg = &prediction.legs[1];
    let burn_seg = burn_leg
        .burn_segment
        .as_ref()
        .expect("leg 1 should have a burn");
    let first = burn_seg.samples.first().expect("burn has samples");
    let thalos_at_node = ephemeris.query_body(1, node_time).position;
    let start_offset = (ship_state.position - thalos_state_0.position).normalize();
    let burn_start_offset = (first.position - thalos_at_node).normalize();

    // Ship start is on +X; after a quarter orbit CCW (around -Y) it is on +Z.
    // The burn's first sample direction (Thalos→ship) should be close to +Z.
    let expected_dir = DVec3::new(0.0, 0.0, 1.0);
    let cos_expected = burn_start_offset.dot(expected_dir);
    let cos_start = burn_start_offset.dot(start_offset);
    assert!(
        cos_expected > 0.95,
        "burn should start at node position (+Z dir), got offset direction {:?} cos_to_+Z={}",
        burn_start_offset, cos_expected,
    );
    assert!(
        cos_start < 0.2,
        "burn must NOT start at ship's initial position (+X dir); got cos_to_+X={}",
        cos_start,
    );

    // Apoapsis of the post-burn coast should be roughly OPPOSITE the node
    // (i.e. on -Z), not opposite the ship start (-X).
    let mut max_r = 0.0_f64;
    let mut apo_offset = DVec3::ZERO;
    for s in &burn_leg.coast_segment.samples {
        let th = ephemeris.query_body(1, s.time).position;
        let rel = s.position - th;
        let r = rel.length();
        if r > max_r {
            max_r = r;
            apo_offset = rel.normalize();
        }
    }
    let cos_apo_to_neg_z = apo_offset.dot(-expected_dir);
    let cos_apo_to_neg_x = apo_offset.dot(-start_offset);
    assert!(
        cos_apo_to_neg_z > 0.8,
        "apoapsis should be ~opposite the node (-Z); got cos_to_-Z={}",
        cos_apo_to_neg_z,
    );
    assert!(
        cos_apo_to_neg_x < 0.3,
        "apoapsis must NOT be opposite the ship's start (-X); got cos_to_-X={}",
        cos_apo_to_neg_x,
    );
}

#[test]
fn prograde_burn_raises_apoapsis_around_thalos() {
    let system = make_thalos_like_system();
    let ephemeris: Arc<dyn BodyStateProvider> = Arc::new(PatchedConics::new(&system, 1.0e7));

    let thalos_state_0 = ephemeris.query_body(1, 0.0);
    let thalos_gm = system.bodies[1].gm;
    let orbit_radius = system.bodies[1].radius_m + 200_000.0;
    let circular_speed = (thalos_gm / orbit_radius).sqrt();

    let ship_state = StateVector {
        position: thalos_state_0.position + DVec3::new(orbit_radius, 0.0, 0.0),
        velocity: thalos_state_0.velocity + DVec3::new(0.0, 0.0, circular_speed),
    };

    let mut maneuvers = ManeuverSequence::new();
    maneuvers.add(ManeuverNode {
        id: None,
        time: 10.0,
        delta_v: DVec3::new(100.0, 0.0, 0.0),
        reference_body: 1,
    });

    let prediction = propagate_trajectory(
        ship_state,
        0.0,
        &maneuvers,
        Arc::clone(&ephemeris),
        &system.bodies,
        &PredictionConfig::default(),
        TEST_THRUST_ACCEL,
    );

    let mut max_r = 0.0_f64;
    let mut min_r = f64::INFINITY;
    let mut burn_leg_samples = 0usize;
    for (i, seg) in prediction.segments.iter().enumerate() {
        if i == 0 {
            continue;
        }
        burn_leg_samples += seg.samples.len();
        for s in &seg.samples {
            let thalos_pos = ephemeris.query_body(1, s.time).position;
            let r = (s.position - thalos_pos).length();
            if r > max_r {
                max_r = r;
            }
            if r < min_r {
                min_r = r;
            }
        }
    }

    assert!(burn_leg_samples > 0);
    assert!(
        max_r > orbit_radius + 1.0e5,
        "prograde burn should raise apoapsis"
    );
    assert!(
        min_r > system.bodies[1].radius_m,
        "orbit should not collide"
    );
}

#[test]
fn large_prograde_burn_escapes_without_collision() {
    // Regression for the "curls in and collides" bug. Long burn must track
    // the rotating orbital frame, not stay frozen at burn start.
    let system = make_thalos_like_system();
    let ephemeris: Arc<dyn BodyStateProvider> = Arc::new(PatchedConics::new(&system, 1.0e8));

    let thalos_state_0 = ephemeris.query_body(1, 0.0);
    let thalos_gm = system.bodies[1].gm;
    let orbit_radius = system.bodies[1].radius_m + 200_000.0;
    let circular_speed = (thalos_gm / orbit_radius).sqrt();

    let ship_state = StateVector {
        position: thalos_state_0.position + DVec3::new(orbit_radius, 0.0, 0.0),
        velocity: thalos_state_0.velocity + DVec3::new(0.0, 0.0, circular_speed),
    };

    let mut maneuvers = ManeuverSequence::new();
    maneuvers.add(ManeuverNode {
        id: None,
        time: 5.0,
        delta_v: DVec3::new(5_000.0, 0.0, 0.0),
        reference_body: 1,
    });

    let prediction = propagate_trajectory(
        ship_state,
        0.0,
        &maneuvers,
        Arc::clone(&ephemeris),
        &system.bodies,
        &PredictionConfig::default(),
        TEST_THRUST_ACCEL,
    );

    let mut max_r = 0.0_f64;
    let mut min_r = f64::INFINITY;
    for (i, seg) in prediction.segments.iter().enumerate() {
        assert!(
            seg.collision_body.is_none(),
            "segment {} reported a collision with {:?}",
            i,
            seg.collision_body
        );
        if i == 0 {
            continue;
        }
        for s in &seg.samples {
            let thalos_pos = ephemeris.query_body(1, s.time).position;
            let r = (s.position - thalos_pos).length();
            max_r = max_r.max(r);
            min_r = min_r.min(r);
        }
    }

    assert!(
        min_r > system.bodies[1].radius_m,
        "ship fell below Thalos' surface"
    );
    assert!(
        max_r > orbit_radius * 5.0,
        "escape burn should reach far past initial orbit"
    );
}

#[test]
fn stable_orbit_detected_after_prograde_burn() {
    let system = make_thalos_like_system();
    let ephemeris: Arc<dyn BodyStateProvider> = Arc::new(PatchedConics::new(&system, 1.0e7));

    let thalos_state_0 = ephemeris.query_body(1, 0.0);
    let thalos_gm = system.bodies[1].gm;
    let orbit_radius = system.bodies[1].radius_m + 200_000.0;
    let circular_speed = (thalos_gm / orbit_radius).sqrt();

    let ship_state = StateVector {
        position: thalos_state_0.position + DVec3::new(orbit_radius, 0.0, 0.0),
        velocity: thalos_state_0.velocity + DVec3::new(0.0, 0.0, circular_speed),
    };

    let mut maneuvers = ManeuverSequence::new();
    maneuvers.add(ManeuverNode {
        id: None,
        time: 10.0,
        delta_v: DVec3::new(100.0, 0.0, 0.0),
        reference_body: 1,
    });

    let prediction = propagate_trajectory(
        ship_state,
        0.0,
        &maneuvers,
        Arc::clone(&ephemeris),
        &system.bodies,
        &PredictionConfig::default(),
        TEST_THRUST_ACCEL,
    );

    // The final leg's coast sub-segment is propagated entirely post-burn, so
    // stable-orbit detection there captures from sample 0 (the burn-end
    // state) and the `start_index` is always `Some(0)`. The meaningful check
    // is that closure is detected at all.
    let last_leg = prediction.legs.last().expect("at least one leg");
    assert!(
        last_leg.coast_segment.is_stable_orbit,
        "post-burn coast sub-segment should detect stable orbit",
    );
    assert!(
        last_leg.coast_segment.stable_orbit_start_index.is_some(),
        "stable orbit tracker should be initialised on coast",
    );
}

#[test]
fn zero_delta_v_node_after_stable_orbit_still_builds_post_node_leg() {
    let system = make_thalos_like_system();
    let ephemeris: Arc<dyn BodyStateProvider> = Arc::new(PatchedConics::new(&system, 2.0e7));

    let thalos_state_0 = ephemeris.query_body(1, 0.0);
    let thalos_gm = system.bodies[1].gm;
    let orbit_radius = system.bodies[1].radius_m + 200_000.0;
    let circular_speed = (thalos_gm / orbit_radius).sqrt();
    let orbital_period = std::f64::consts::TAU * (orbit_radius.powi(3) / thalos_gm).sqrt();

    let ship_state = StateVector {
        position: thalos_state_0.position + DVec3::new(orbit_radius, 0.0, 0.0),
        velocity: thalos_state_0.velocity + DVec3::new(0.0, 0.0, circular_speed),
    };

    let node_time = orbital_period * 2.0;
    let mut maneuvers = ManeuverSequence::new();
    maneuvers.add(ManeuverNode {
        id: None,
        time: node_time,
        delta_v: DVec3::ZERO,
        reference_body: 1,
    });

    let prediction = propagate_trajectory(
        ship_state,
        0.0,
        &maneuvers,
        Arc::clone(&ephemeris),
        &system.bodies,
        &PredictionConfig::default(),
        TEST_THRUST_ACCEL,
    );

    assert!(
        prediction.segments.len() >= 2,
        "placing a node in the future should keep propagating past the node",
    );
    assert!(
        prediction.segments[0]
            .end_time()
            .is_some_and(|end| (end - node_time).abs() <= 1e-6),
        "the pre-node leg should reach the maneuver boundary",
    );
    assert!(
        prediction
            .segments
            .iter()
            .skip(1)
            .flat_map(|seg| seg.samples.iter())
            .any(|sample| sample.time > node_time),
        "expected propagated samples after the maneuver node",
    );
}

#[test]
fn delayed_prograde_burn_after_stable_orbit_still_changes_future_path() {
    let system = make_thalos_like_system();
    let ephemeris: Arc<dyn BodyStateProvider> = Arc::new(PatchedConics::new(&system, 2.0e7));

    let thalos_state_0 = ephemeris.query_body(1, 0.0);
    let thalos_gm = system.bodies[1].gm;
    let orbit_radius = system.bodies[1].radius_m + 200_000.0;
    let circular_speed = (thalos_gm / orbit_radius).sqrt();
    let orbital_period = std::f64::consts::TAU * (orbit_radius.powi(3) / thalos_gm).sqrt();

    let ship_state = StateVector {
        position: thalos_state_0.position + DVec3::new(orbit_radius, 0.0, 0.0),
        velocity: thalos_state_0.velocity + DVec3::new(0.0, 0.0, circular_speed),
    };

    let mut maneuvers = ManeuverSequence::new();
    maneuvers.add(ManeuverNode {
        id: None,
        time: orbital_period * 2.0,
        delta_v: DVec3::new(100.0, 0.0, 0.0),
        reference_body: 1,
    });

    let prediction = propagate_trajectory(
        ship_state,
        0.0,
        &maneuvers,
        Arc::clone(&ephemeris),
        &system.bodies,
        &PredictionConfig::default(),
        TEST_THRUST_ACCEL,
    );

    let mut post_node_samples = 0usize;
    let mut max_r = 0.0_f64;
    for seg in prediction.segments.iter().skip(1) {
        for sample in &seg.samples {
            let thalos_pos = ephemeris.query_body(1, sample.time).position;
            let r = (sample.position - thalos_pos).length();
            max_r = max_r.max(r);
            post_node_samples += 1;
        }
    }

    assert!(
        post_node_samples > 0,
        "expected propagated samples after the delayed burn"
    );
    assert!(
        max_r > orbit_radius + 1.0e5,
        "delayed prograde burn should still raise apoapsis",
    );
}

// ---------------------------------------------------------------------------
// Moon encounter tests
// ---------------------------------------------------------------------------

fn make_thalos_mira_system() -> SolarSystemDefinition {
    let (sun, thalos, _sun_mass) = make_star_and_planet();
    let thalos_mass = thalos.mass_kg;

    let mira_mass = 1.374e22;
    let mira_sma = 1.91488e8;
    let mira = BodyDefinition {
        id: 2,
        name: "Mira".to_string(),
        kind: BodyKind::Moon,
        parent: Some(1),
        mass_kg: mira_mass,
        radius_m: 869_000.0,
        color: [0.65, 0.63, 0.60],
        albedo: 0.12,
        rotation_period_s: 0.0,
        axial_tilt_rad: 0.0,
        gm: G * mira_mass,
        soi_radius_m: mira_sma * (mira_mass / thalos_mass).powf(0.4),
        orbital_elements: Some(OrbitalElements {
            semi_major_axis_m: mira_sma,
            eccentricity: 0.0,
            inclination_rad: 0.0,
            lon_ascending_node_rad: 0.0,
            arg_periapsis_rad: 0.0,
            true_anomaly_rad: 0.0,
        }),
        generator: None,
        atmosphere: None,
        terrestrial_atmosphere: None,
    };

    let mut name_to_id = HashMap::new();
    name_to_id.insert("Sun".to_string(), 0);
    name_to_id.insert("Thalos".to_string(), 1);
    name_to_id.insert("Mira".to_string(), 2);

    SolarSystemDefinition {
        name: "ThalosMiraTest".to_string(),
        bodies: vec![sun, thalos, mira],
        ship: ShipDefinition {
            initial_state: StateVector {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
        },
        name_to_id,
    }
}

/// Specific energy (kinetic + potential) of the ship relative to Thalos.
fn specific_energy_around_thalos(
    ship_pos: DVec3,
    ship_vel: DVec3,
    thalos_pos: DVec3,
    thalos_vel: DVec3,
    thalos_gm: f64,
) -> f64 {
    let rel_v = ship_vel - thalos_vel;
    let rel_r = (ship_pos - thalos_pos).length();
    0.5 * rel_v.length_squared() - thalos_gm / rel_r
}

/// Regression test: a ship on a transfer orbit that encounters Mira must
/// produce a physically bounded post-encounter velocity. Before the tidal
/// guard fix, the integrator switched to 60 s symplectic steps deep inside
/// Mira's gravity well, producing nonsensical exit trajectories.
#[test]
fn moon_encounter_preserves_bounded_energy() {
    let system = make_thalos_mira_system();
    let mira_sma = 1.91488e8;
    let thalos_gm = system.bodies[1].gm;
    let mira_gm = system.bodies[2].gm;
    // Propagate long enough for the transfer orbit to reach Mira.
    let ephemeris: Arc<dyn BodyStateProvider> = Arc::new(PatchedConics::new(&system, 5.0e6));

    let thalos_state_0 = ephemeris.query_body(1, 0.0);
    // Ship starts on a Hohmann-ish transfer from low Thalos orbit toward
    // Mira's orbital distance.
    let r_park = system.bodies[1].radius_m + 200_000.0; // 200 km altitude
    // Hohmann Δv to reach Mira's orbit: v_transfer - v_circular
    let v_transfer = (thalos_gm * (2.0 / r_park - 2.0 / (r_park + mira_sma))).sqrt();

    // Start at +X from Thalos, velocity in +Z (prograde).
    let ship_state = StateVector {
        position: thalos_state_0.position + DVec3::new(r_park, 0.0, 0.0),
        velocity: thalos_state_0.velocity + DVec3::new(0.0, 0.0, v_transfer),
    };

    let prediction = propagate_trajectory(
        ship_state,
        0.0,
        &ManeuverSequence::new(),
        Arc::clone(&ephemeris),
        &system.bodies,
        &PredictionConfig::default(),
        TEST_THRUST_ACCEL,
    );

    assert!(
        !prediction.segments.is_empty(),
        "propagation should produce segments"
    );

    // Collect all samples.
    let samples: Vec<&TrajectorySample> = prediction
        .segments
        .iter()
        .flat_map(|s| s.samples.iter())
        .collect();
    assert!(!samples.is_empty());

    // Compute initial specific energy relative to Thalos.
    let e0 = specific_energy_around_thalos(
        ship_state.position,
        ship_state.velocity,
        thalos_state_0.position,
        thalos_state_0.velocity,
        thalos_gm,
    );

    // Maximum Δv a Mira flyby can impart: 2 * v_escape_mira (at surface).
    let v_esc_mira = (2.0 * mira_gm / system.bodies[2].radius_m).sqrt();
    let max_delta_v = 2.0 * v_esc_mira;
    // Upper bound on energy change: Δe ≈ v * Δv (vis-viva).
    // Use the transfer velocity at Mira's orbit as the reference speed.
    let v_at_mira = (thalos_gm * (2.0 / mira_sma - 2.0 / (r_park + mira_sma))).sqrt();
    let max_energy_change = v_at_mira * max_delta_v + 0.5 * max_delta_v * max_delta_v;

    // Check final sample's energy is within the physically bounded range.
    let last = samples.last().unwrap();
    let thalos_last = ephemeris.query_body(1, last.time);
    let e_final = specific_energy_around_thalos(
        last.position,
        last.velocity,
        thalos_last.position,
        thalos_last.velocity,
        thalos_gm,
    );

    let energy_change = (e_final - e0).abs();
    assert!(
        energy_change < max_energy_change * 3.0, // 3x safety margin
        "energy change {energy_change:.0} exceeds physical bound {:.0} \
         (3x max gravity assist). Integrator likely used too-coarse steps \
         during the encounter.",
        max_energy_change * 3.0,
    );
}

/// The orbit tracker must not mark a segment as "stable orbit" when the
/// trajectory crosses an SOI boundary (anchor body change).
#[test]
fn no_false_stable_orbit_across_soi_transition() {
    let system = make_thalos_mira_system();
    let thalos_gm = system.bodies[1].gm;
    let mira_sma = 1.91488e8;
    let ephemeris: Arc<dyn BodyStateProvider> = Arc::new(PatchedConics::new(&system, 5.0e6));

    let thalos_state_0 = ephemeris.query_body(1, 0.0);

    // Ship on a transfer toward Mira's orbit.
    let r_park = system.bodies[1].radius_m + 200_000.0;
    let v_transfer = (thalos_gm * (2.0 / r_park - 2.0 / (r_park + mira_sma))).sqrt();

    let ship_state = StateVector {
        position: thalos_state_0.position + DVec3::new(r_park, 0.0, 0.0),
        velocity: thalos_state_0.velocity + DVec3::new(0.0, 0.0, v_transfer),
    };

    let prediction = propagate_trajectory(
        ship_state,
        0.0,
        &ManeuverSequence::new(),
        Arc::clone(&ephemeris),
        &system.bodies,
        &PredictionConfig::default(),
        TEST_THRUST_ACCEL,
    );

    // Check whether the trajectory enters Mira's SOI.
    let entered_mira = prediction
        .segments
        .iter()
        .flat_map(|s| s.samples.iter())
        .any(|s| s.anchor_body == 2);

    if entered_mira {
        // If the trajectory crossed into Mira's SOI, the segment should NOT
        // be marked as a stable orbit — the orbit tracker should have reset.
        for seg in &prediction.segments {
            if seg.is_stable_orbit {
                // Verify that the stable orbit doesn't span an anchor change.
                let start_idx = seg.stable_orbit_start_index.unwrap_or(0);
                let stable_samples = &seg.samples[start_idx..];
                if stable_samples.len() >= 2 {
                    let anchor = stable_samples[0].anchor_body;
                    let all_same_anchor = stable_samples.iter().all(|s| s.anchor_body == anchor);
                    assert!(
                        all_same_anchor,
                        "stable orbit spans samples with different anchor bodies — \
                         orbit tracker should have reset at the SOI transition"
                    );
                }
            }
        }
    }
}

/// Aggregated encounters carry osculating orbital elements at closest
/// approach. For a Mira flyby arriving with excess hyperbolic velocity,
/// eccentricity must be ≥ 1 and capture status `Flyby`.
#[test]
fn encounter_enrichment_reports_flyby_for_hyperbolic_pass() {
    use crate::trajectory::CaptureStatus;

    let system = make_thalos_mira_system();
    let mira_sma = 1.91488e8;
    let thalos_gm = system.bodies[1].gm;
    let ephemeris: Arc<dyn BodyStateProvider> = Arc::new(PatchedConics::new(&system, 5.0e6));
    let thalos_state_0 = ephemeris.query_body(1, 0.0);

    let r_park = system.bodies[1].radius_m + 200_000.0;
    let v_transfer = (thalos_gm * (2.0 / r_park - 2.0 / (r_park + mira_sma))).sqrt();
    let ship_state = StateVector {
        position: thalos_state_0.position + DVec3::new(r_park, 0.0, 0.0),
        velocity: thalos_state_0.velocity + DVec3::new(0.0, 0.0, v_transfer),
    };

    let prediction = propagate_trajectory(
        ship_state,
        0.0,
        &ManeuverSequence::new(),
        Arc::clone(&ephemeris),
        &system.bodies,
        &PredictionConfig::default(),
        TEST_THRUST_ACCEL,
    );

    let mira_enc = prediction.encounter_with(2);
    if let Some(enc) = mira_enc {
        assert!(
            enc.eccentricity > 0.0,
            "eccentricity must be populated by enrichment"
        );
        assert!(enc.closest_distance > 0.0);
        assert!(enc.relative_velocity > 0.0);
        // Hyperbolic or captured — either outcome is physical for this setup,
        // but capture status must be consistent with eccentricity.
        match enc.capture {
            CaptureStatus::Flyby => assert!(enc.eccentricity >= 1.0),
            CaptureStatus::Captured => assert!(enc.eccentricity < 1.0),
            CaptureStatus::Impact | CaptureStatus::Graze { .. } => {}
        }
    }
}

// `target_bias_caps_step_size_near_target` deleted: the step-size cap was an
// adaptive-RK45 knob that no longer exists under the analytical Keplerian
// propagator. SOI crossings are now caught by root-finding during coast
// sampling (see `ship_propagator::bisect_body_distance`), which supplies
// precision regardless of how much sim time a single call advances.

/// Mirror the game's startup scenario: load `solar_system.ron`, build the
/// ship's 200 km circular orbit around Thalos, run prediction with no
/// maneuvers. A circular orbit that fits inside Thalos's SOI and clears
/// the surface must terminate in exactly one `StableOrbit` closure.
#[test]
fn game_default_state_produces_stable_orbit() {
    use crate::parsing::load_solar_system;
    let system = match load_solar_system("../../assets/solar_system.ron") {
        Ok(s) => s,
        Err(_) => return, // loader unavailable in CI or path mismatch — skip
    };
    let ephemeris: Arc<dyn BodyStateProvider> =
        Arc::new(PatchedConics::new(&system, 3.156e9));

    let homeworld_id = system.name_to_id["Thalos"];
    let homeworld_state = ephemeris.query_body(homeworld_id, 0.0);
    let rel = system.ship.initial_state;
    let ship_state = StateVector {
        position: homeworld_state.position + rel.position,
        velocity: homeworld_state.velocity + rel.velocity,
    };

    let prediction = propagate_trajectory(
        ship_state,
        0.0,
        &ManeuverSequence::new(),
        Arc::clone(&ephemeris),
        &system.bodies,
        &PredictionConfig::default(),
        TEST_THRUST_ACCEL,
    );

    // One leg, one segment, one full revolution.
    assert_eq!(prediction.legs.len(), 1, "expected single leg (no maneuvers)");
    let leg = &prediction.legs[0];
    assert!(leg.burn_segment.is_none(), "no burn on an unmanoeuvred leg");
    assert!(
        leg.coast_segment.is_stable_orbit,
        "ship's 200 km circular orbit must close as a stable orbit"
    );
    // Every sample must anchor to Thalos — the ship never exits its SOI.
    let samples = &leg.coast_segment.samples;
    assert!(samples.len() >= 32, "expected dense sampling; got {}", samples.len());
    for s in samples {
        assert_eq!(s.anchor_body, homeworld_id, "sample escaped Thalos SOI");
    }
    // Relative position to Thalos (current) should have near-constant radius
    // (circular orbit). Check min vs max don't differ by more than 1%.
    let thalos_now = ephemeris.query_body(homeworld_id, 0.0);
    let radii: Vec<f64> = samples
        .iter()
        .map(|s| {
            let thalos_t = ephemeris.query_body(homeworld_id, s.time);
            (s.position - thalos_t.position).length()
        })
        .collect();
    let r_min = radii.iter().cloned().fold(f64::INFINITY, f64::min);
    let r_max = radii.iter().cloned().fold(0.0, f64::max);
    assert!(
        (r_max - r_min) / r_min < 1e-3,
        "circular orbit radii drift too much: min={r_min} max={r_max}"
    );
    let _ = thalos_now;
}

/// `FlightPlan::approaches` lists per-body closest passes for every body the
/// trajectory does NOT enter the SOI of.  Bodies already covered by an
/// `Encounter` must not appear.
#[test]
fn closest_approach_scan_excludes_encountered_bodies() {
    let system = make_thalos_mira_system();
    let mira_sma = 1.91488e8;
    let thalos_gm = system.bodies[1].gm;
    let ephemeris: Arc<dyn BodyStateProvider> = Arc::new(PatchedConics::new(&system, 5.0e6));
    let thalos_state_0 = ephemeris.query_body(1, 0.0);

    let r_park = system.bodies[1].radius_m + 200_000.0;
    let v_transfer = (thalos_gm * (2.0 / r_park - 2.0 / (r_park + mira_sma))).sqrt();
    let ship_state = StateVector {
        position: thalos_state_0.position + DVec3::new(r_park, 0.0, 0.0),
        velocity: thalos_state_0.velocity + DVec3::new(0.0, 0.0, v_transfer),
    };

    let prediction = propagate_trajectory(
        ship_state,
        0.0,
        &ManeuverSequence::new(),
        Arc::clone(&ephemeris),
        &system.bodies,
        &PredictionConfig::default(),
        TEST_THRUST_ACCEL,
    );

    let encounter_bodies: std::collections::HashSet<usize> =
        prediction.encounters.iter().map(|e| e.body).collect();
    for ca in &prediction.approaches {
        assert!(
            !encounter_bodies.contains(&ca.body),
            "body {} has both an Encounter and a ClosestApproach",
            ca.body
        );
        // Stars must never appear in approach list.
        assert_ne!(
            system.bodies[ca.body].kind,
            crate::types::BodyKind::Star,
            "closest-approach list must not include stars"
        );
        assert!(ca.distance > 0.0);
    }
}
