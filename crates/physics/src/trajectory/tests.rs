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

use super::{FlightPlan, PredictionConfig, PredictionRequest, cone_width, propagate_flight_plan};
use crate::body_state_provider::BodyStateProvider;
use crate::integrator::IntegratorConfig;
use crate::maneuver::{ManeuverNode, ManeuverSequence};
use crate::patched_conics::PatchedConics;
use crate::types::{
    BodyDefinition, BodyKind, G, OrbitalElements, ShipDefinition, SolarSystemDefinition,
    StateVector, TrajectorySample,
};

/// Local helper that mirrors the old `propagate_trajectory` convenience so the
/// tests can stay focused on what they actually assert.
fn propagate_trajectory(
    initial_state: StateVector,
    start_time: f64,
    maneuvers: &ManeuverSequence,
    ephemeris: Arc<dyn BodyStateProvider>,
    bodies: &[BodyDefinition],
    config: &PredictionConfig,
    integrator_config: IntegratorConfig,
    ship_thrust_acceleration: f64,
) -> FlightPlan {
    let request = PredictionRequest {
        epoch: 0,
        ship_state: initial_state,
        sim_time: start_time,
        maneuvers: maneuvers.clone(),
        active_burns: Vec::new(),
        ephemeris,
        bodies: bodies.to_vec(),
        prediction_config: config.clone(),
        integrator_config,
        ship_thrust_acceleration,
    };
    propagate_flight_plan(&request, None)
}

const AU: f64 = 1.496e11;
const SUN_GM: f64 = 1.327_124_4e20;

fn sample_with(perturbation_ratio: f64, step_size: f64) -> TrajectorySample {
    TrajectorySample {
        time: 0.0,
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        dominant_body: 0,
        perturbation_ratio,
        step_size,
        anchor_body: 0,
        anchor_body_pos: DVec3::ZERO,
    }
}

#[test]
fn cone_width_zero_when_unperturbed() {
    assert_eq!(cone_width(&sample_with(0.0, 60.0)), 0.0);
}

#[test]
fn cone_width_grows_when_step_size_shrinks() {
    let wide = cone_width(&sample_with(0.1, 1.0));
    let narrow = cone_width(&sample_with(0.1, 100.0));
    assert!(wide > narrow);
}

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
            thrust_acceleration: 1.0,
        },
        name_to_id,
    }
}

fn make_thalos_like_system() -> SolarSystemDefinition {
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
    };

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
            thrust_acceleration: 0.5,
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
        &PredictionConfig {
            max_steps_per_segment: 32,
            min_orbit_samples: usize::MAX,
            ..PredictionConfig::default()
        },
        IntegratorConfig {
            symplectic_dt: 1.0,
            rk_initial_dt: 1.0,
            ..IntegratorConfig::default()
        },
        system.ship.thrust_acceleration,
    );

    let burn_segment = prediction
        .segments
        .iter()
        .find(|segment| !segment.samples.is_empty())
        .expect("expected a downstream burn segment");
    let first_sample = burn_segment.samples.first().unwrap();
    let delta_speed = (first_sample.velocity - system.ship.initial_state.velocity).length();

    assert!(
        delta_speed < 2.0,
        "finite burn should not apply the full 10 m/s instantly; got {delta_speed:.3}"
    );
    assert!(
        (first_sample.time - 1.0).abs() < 1e-9,
        "prediction should step exactly to the first capped sample"
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
        &PredictionConfig {
            max_steps_per_segment: 20_000,
            cone_fade_threshold: f64::INFINITY,
            min_orbit_samples: usize::MAX,
            ..PredictionConfig::default()
        },
        IntegratorConfig {
            symplectic_dt: 1.0,
            rk_initial_dt: 1.0,
            ..IntegratorConfig::default()
        },
        system.ship.thrust_acceleration,
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
        &PredictionConfig {
            max_steps_per_segment: 200_000,
            cone_fade_threshold: f64::INFINITY,
            min_orbit_samples: usize::MAX,
            ..PredictionConfig::default()
        },
        IntegratorConfig {
            symplectic_dt: 60.0,
            rk_initial_dt: 60.0,
            ..IntegratorConfig::default()
        },
        system.ship.thrust_acceleration,
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
        IntegratorConfig {
            symplectic_dt: 60.0,
            rk_initial_dt: 60.0,
            ..IntegratorConfig::default()
        },
        system.ship.thrust_acceleration,
    );

    let post_burn = &prediction.segments[prediction.segments.len() - 1];
    assert!(
        post_burn.is_stable_orbit,
        "post-burn segment should detect stable orbit",
    );
    assert!(
        post_burn
            .stable_orbit_start_index
            .is_some_and(|idx| idx > 0),
        "stable orbit should start after finite-burn lead-in",
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
        &PredictionConfig {
            max_steps_per_segment: 40_000,
            cone_fade_threshold: f64::INFINITY,
            ..PredictionConfig::default()
        },
        IntegratorConfig {
            symplectic_dt: 60.0,
            rk_initial_dt: 60.0,
            ..IntegratorConfig::default()
        },
        system.ship.thrust_acceleration,
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
        &PredictionConfig {
            max_steps_per_segment: 40_000,
            cone_fade_threshold: f64::INFINITY,
            ..PredictionConfig::default()
        },
        IntegratorConfig {
            symplectic_dt: 60.0,
            rk_initial_dt: 60.0,
            ..IntegratorConfig::default()
        },
        system.ship.thrust_acceleration,
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
