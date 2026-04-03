//! Trajectory prediction: propagate a ship's state through a maneuver sequence.
//!
//! The predicted path is split into [`TrajectorySegment`]s — one per leg
//! between maneuver nodes (plus the leg before the first node and after the
//! last). Each segment is integrated independently with its own
//! [`ForceRegistry`] so that thrust is only active during the correct window.
//!
//! # Termination conditions (any one stops the current segment)
//!
//! - **Cone width** exceeds `config.cone_fade_threshold` — the prediction is
//!   too uncertain to be useful beyond this point.
//! - **Stable orbit** — the ship returns within `orbit_position_threshold` and
//!   `orbit_velocity_threshold` of the segment's starting state after at least
//!   `min_orbit_samples` steps.
//! - **Collision** — the ship is inside a body's radius.
//! - **Max steps** — `config.max_steps_per_segment` steps have been taken.

use glam::DVec3;

use crate::ephemeris::Ephemeris;
use crate::forces::{ForceRegistry, GravityForce, ThrustForce};
use crate::integrator::{Integrator, IntegratorConfig};
use crate::maneuver::ManeuverSequence;
use crate::types::{BodyDefinition, BodyId, StateVector, TrajectorySample};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One leg of the predicted trajectory.
#[derive(Debug, Clone)]
pub struct TrajectorySegment {
    pub samples: Vec<TrajectorySample>,
    /// True if a stable closed orbit was detected during this segment.
    pub is_stable_orbit: bool,
    /// Set to the body ID if the ship collided with a body.
    pub collision_body: Option<BodyId>,
}

/// The full predicted path from the ship's current state through all maneuvers.
#[derive(Debug, Clone)]
pub struct TrajectoryPrediction {
    pub segments: Vec<TrajectorySegment>,
}

/// Tuning parameters for trajectory prediction.
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    /// Maximum integration steps per segment. Default: 100_000.
    pub max_steps_per_segment: usize,
    /// Cone width (meters) at which prediction is no longer useful. Default: 1e6.
    pub cone_fade_threshold: f64,
    /// Position proximity (meters) for stable-orbit detection. Default: 1e4.
    pub orbit_position_threshold: f64,
    /// Velocity proximity (m/s) for stable-orbit detection. Default: 1.0.
    pub orbit_velocity_threshold: f64,
    /// Minimum samples before stable-orbit detection is attempted. Default: 100.
    pub min_orbit_samples: usize,
    /// Scale factor applied to `perturbation_ratio * step_size` for cone width.
    /// Default: 1.0.
    pub cone_width_scale: f64,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            max_steps_per_segment: 100_000,
            cone_fade_threshold: 1e6,
            orbit_position_threshold: 1e4,
            orbit_velocity_threshold: 1.0,
            min_orbit_samples: 100,
            cone_width_scale: 1.0,
        }
    }
}

/// Limits the number of integrator steps across a [`propagate_trajectory_budgeted`]
/// call. Useful for progressive refinement: run a coarse pass first, then
/// resume with more budget each frame.
#[derive(Debug, Clone, Copy)]
pub struct PropagationBudget {
    /// Total steps allowed for this call.
    pub max_steps: usize,
}

impl PropagationBudget {
    pub fn new(max_steps: usize) -> Self {
        Self { max_steps }
    }
}

// ---------------------------------------------------------------------------
// Cone width
// ---------------------------------------------------------------------------

/// Proxy for trajectory cone width at a sample point (meters).
///
/// Uses `perturbation_ratio * step_size` as the raw uncertainty signal.  This
/// naturally widens near gravitational boundaries and during burns where the
/// adaptive integrator takes small steps with high perturbation ratios.
pub fn cone_width(sample: &TrajectorySample) -> f64 {
    sample.perturbation_ratio * sample.step_size
}

fn cone_width_scaled(sample: &TrajectorySample, config: &PredictionConfig) -> f64 {
    cone_width(sample) * config.cone_width_scale
}

// ---------------------------------------------------------------------------
// Delta-v frame conversion
// ---------------------------------------------------------------------------

/// Convert a delta-v vector from the local prograde/normal/radial frame of
/// `reference_body` into the heliocentric world frame.
///
/// Frame definition (right-hand, standard mission planning convention):
/// - **prograde** (`delta_v.x`) — along the ship's velocity relative to the
///   reference body.
/// - **normal**   (`delta_v.y`) — `cross(pos_rel, vel_rel)` normalised; out of
///   the orbit plane, pointing "north".
/// - **radial**   (`delta_v.z`) — `cross(vel_rel, normal)` normalised; points
///   away from the reference body.
///
/// If the relative velocity or the orbit-plane normal is degenerate (zero),
/// the delta-v is returned unchanged (treated as already in world frame).
fn delta_v_to_world(
    delta_v_local: DVec3,
    ship_state: &StateVector,
    reference_body_position: DVec3,
    reference_body_velocity: DVec3,
) -> DVec3 {
    let pos_rel = ship_state.position - reference_body_position;
    let vel_rel = ship_state.velocity - reference_body_velocity;

    let prograde = if vel_rel.length_squared() > 0.0 {
        vel_rel.normalize()
    } else {
        return delta_v_local;
    };

    let normal_unnorm = pos_rel.cross(vel_rel);
    let normal = if normal_unnorm.length_squared() > 0.0 {
        normal_unnorm.normalize()
    } else {
        return delta_v_local;
    };

    let radial_unnorm = vel_rel.cross(normal_unnorm);
    let radial = if radial_unnorm.length_squared() > 0.0 {
        radial_unnorm.normalize()
    } else {
        return delta_v_local;
    };

    delta_v_local.x * prograde + delta_v_local.y * normal + delta_v_local.z * radial
}

// ---------------------------------------------------------------------------
// Stable-orbit detection
// ---------------------------------------------------------------------------

fn is_stable_orbit(
    current: &StateVector,
    start: &StateVector,
    config: &PredictionConfig,
    sample_count: usize,
) -> bool {
    if sample_count < config.min_orbit_samples {
        return false;
    }
    let pos_err = (current.position - start.position).length();
    let vel_err = (current.velocity - start.velocity).length();
    pos_err < config.orbit_position_threshold && vel_err < config.orbit_velocity_threshold
}

// ---------------------------------------------------------------------------
// Segment propagation
// ---------------------------------------------------------------------------

/// Propagate one trajectory segment from `initial_state` at `start_time` until
/// `end_time` or a termination condition.
///
/// `thrust` — optional continuous burn active from `start_time`:
///   `(world-frame direction, acceleration m/s², duration s)`.
fn propagate_segment(
    initial_state: StateVector,
    start_time: f64,
    end_time: f64,
    thrust: Option<(DVec3, f64, f64)>,
    ephemeris: &Ephemeris,
    bodies: &[BodyDefinition],
    config: &PredictionConfig,
    integrator_config: IntegratorConfig,
    remaining_budget: &mut Option<usize>,
) -> TrajectorySegment {
    let mut forces = ForceRegistry::new();
    forces.add(Box::new(GravityForce));

    if let Some((direction, accel, duration)) = thrust {
        forces.add(Box::new(ThrustForce::new(direction, accel, start_time, duration)));
    }

    let mut integrator = Integrator::new(integrator_config);
    let mut state = initial_state;
    let mut time = start_time;
    let mut samples: Vec<TrajectorySample> = Vec::new();

    loop {
        if samples.len() >= config.max_steps_per_segment {
            break;
        }
        if let Some(rem) = remaining_budget {
            if *rem == 0 {
                break;
            }
        }
        if time >= end_time {
            break;
        }

        let (new_state, sample) = integrator.step(state, time, &forces, ephemeris);

        // Collision check against all body surfaces at the new position/time.
        // Uses per-body queries to avoid allocating a full BodyStates Vec each step.
        let mut collision_id: Option<BodyId> = None;
        for body_def in bodies.iter() {
            if body_def.id >= ephemeris.body_count() {
                break;
            }
            let body_state = ephemeris.query_body(body_def.id, sample.time);
            let dist = (sample.position - body_state.position).length();
            if dist < body_def.radius_m {
                collision_id = Some(body_def.id);
                break;
            }
        }
        if let Some(cid) = collision_id {
            samples.push(sample);
            if let Some(rem) = remaining_budget.as_mut() {
                *rem = rem.saturating_sub(1);
            }
            return TrajectorySegment {
                samples,
                is_stable_orbit: false,
                collision_body: Some(cid),
            };
        }

        samples.push(sample);
        if let Some(rem) = remaining_budget.as_mut() {
            *rem = rem.saturating_sub(1);
        }

        // Cone width fade.
        if cone_width_scaled(&sample, config) > config.cone_fade_threshold {
            break;
        }

        // Stable orbit: check after minimum samples have accumulated.
        if is_stable_orbit(&new_state, &initial_state, config, samples.len()) {
            return TrajectorySegment {
                samples,
                is_stable_orbit: true,
                collision_body: None,
            };
        }

        state = new_state;
        time = sample.time;
    }

    TrajectorySegment {
        samples,
        is_stable_orbit: false,
        collision_body: None,
    }
}

// ---------------------------------------------------------------------------
// Public propagation API
// ---------------------------------------------------------------------------

/// Propagate a trajectory through the given maneuver sequence.
///
/// Returns a [`TrajectoryPrediction`] with one [`TrajectorySegment`] per leg.
/// Legs run from start → first node, node → node, …, last node → end of
/// ephemeris.  Early termination (collision, cone fade, stable orbit) stops
/// further processing.
pub fn propagate_trajectory(
    initial_state: StateVector,
    start_time: f64,
    maneuvers: &ManeuverSequence,
    ephemeris: &Ephemeris,
    bodies: &[BodyDefinition],
    config: &PredictionConfig,
    integrator_config: IntegratorConfig,
) -> TrajectoryPrediction {
    propagate_trajectory_budgeted(
        initial_state,
        start_time,
        maneuvers,
        ephemeris,
        bodies,
        config,
        integrator_config,
        None,
    )
}

/// Like [`propagate_trajectory`] but respects a [`PropagationBudget`] for
/// progressive refinement.  Pass `Some(budget)` to cap the total integrator
/// steps across this call.  The caller can track how far the prediction got by
/// inspecting the last sample in the last segment.
pub fn propagate_trajectory_budgeted(
    initial_state: StateVector,
    start_time: f64,
    maneuvers: &ManeuverSequence,
    ephemeris: &Ephemeris,
    bodies: &[BodyDefinition],
    config: &PredictionConfig,
    integrator_config: IntegratorConfig,
    budget: Option<PropagationBudget>,
) -> TrajectoryPrediction {
    let mut segments: Vec<TrajectorySegment> = Vec::new();
    let mut state = initial_state;
    let mut time = start_time;
    let mut remaining_budget: Option<usize> = budget.map(|b| b.max_steps);

    // Build a sorted index into the nodes vec to avoid cloning.
    let mut node_order: Vec<usize> = (0..maneuvers.nodes.len()).collect();
    node_order.sort_by(|&a, &b| {
        maneuvers.nodes[a]
            .time
            .partial_cmp(&maneuvers.nodes[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let ephemeris_end = start_time + ephemeris.time_span();
    let leg_count = node_order.len() + 1;

    for leg_idx in 0..leg_count {
        let leg_end = if leg_idx < node_order.len() {
            maneuvers.nodes[node_order[leg_idx]].time
        } else {
            ephemeris_end
        };

        // For MVP, maneuver delta-v is an instantaneous kick applied at the
        // node boundary, so no ThrustForce is needed inside the segment.
        // To support finite burns later, pass the thrust parameters here.
        let segment = propagate_segment(
            state,
            time,
            leg_end,
            None,
            ephemeris,
            bodies,
            config,
            integrator_config.clone(),
            &mut remaining_budget,
        );

        // Did we terminate before reaching the end of this leg?
        let early_exit = segment.is_stable_orbit
            || segment.collision_body.is_some()
            || segment
                .samples
                .last()
                .map(|s| cone_width_scaled(s, config) > config.cone_fade_threshold)
                .unwrap_or(false)
            || remaining_budget == Some(0);

        // Carry the last known state forward so the next leg starts correctly.
        if let Some(last) = segment.samples.last() {
            state = StateVector {
                position: last.position,
                velocity: last.velocity,
            };
            time = last.time;
        }

        segments.push(segment);

        if early_exit {
            break;
        }

        // Apply the instantaneous delta-v kick at the maneuver node.
        if leg_idx < node_order.len() {
            let node = &maneuvers.nodes[node_order[leg_idx]];
            time = node.time;

            let body_states = ephemeris.query(node.time);
            let (ref_pos, ref_vel) = if node.reference_body < body_states.len() {
                let b = &body_states[node.reference_body];
                (b.position, b.velocity)
            } else {
                (DVec3::ZERO, DVec3::ZERO)
            };

            let dv_world = delta_v_to_world(node.delta_v, &state, ref_pos, ref_vel);
            state.velocity += dv_world;
        }
    }

    TrajectoryPrediction { segments }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_with(perturbation_ratio: f64, step_size: f64) -> TrajectorySample {
        TrajectorySample {
            time: 0.0,
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            dominant_body: 0,
            perturbation_ratio,
            step_size,
        }
    }

    #[test]
    fn cone_width_zero_when_unperturbed() {
        assert_eq!(cone_width(&sample_with(0.0, 60.0)), 0.0);
    }

    #[test]
    fn cone_width_product_of_ratio_and_step() {
        let w = cone_width(&sample_with(0.1, 100.0));
        assert!((w - 10.0).abs() < 1e-10, "expected 10.0, got {w}");
    }

    #[test]
    fn stable_orbit_not_triggered_below_min_samples() {
        let sv = StateVector {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        };
        let config = PredictionConfig::default();
        assert!(!is_stable_orbit(&sv, &sv, &config, 50));
    }

    #[test]
    fn stable_orbit_triggered_when_close_enough() {
        let sv = StateVector {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        };
        let config = PredictionConfig::default();
        assert!(is_stable_orbit(&sv, &sv, &config, 200));
    }

    #[test]
    fn delta_v_prograde_aligns_with_velocity() {
        // Ship moving in +Y at 1000 m/s relative to a stationary body at origin.
        let ship = StateVector {
            position: DVec3::new(1e7, 0.0, 0.0),
            velocity: DVec3::new(0.0, 1000.0, 0.0),
        };
        // A delta-v of 1 m/s prograde (x=1 in local frame) should add ~1 m/s in +Y.
        let dv_world = delta_v_to_world(DVec3::new(1.0, 0.0, 0.0), &ship, DVec3::ZERO, DVec3::ZERO);
        assert!(dv_world.y > 0.99, "prograde should be ~+Y, got {dv_world}");
        assert!(dv_world.x.abs() < 1e-10);
        assert!(dv_world.z.abs() < 1e-10);
    }

    #[test]
    fn delta_v_degenerate_returns_passthrough() {
        // Zero relative velocity — conversion should return the local vector unchanged.
        let ship = StateVector {
            position: DVec3::new(1e7, 0.0, 0.0),
            velocity: DVec3::ZERO, // same as reference body
        };
        let dv_local = DVec3::new(1.0, 2.0, 3.0);
        let dv_world = delta_v_to_world(dv_local, &ship, DVec3::ZERO, DVec3::ZERO);
        assert_eq!(dv_world, dv_local);
    }
}
