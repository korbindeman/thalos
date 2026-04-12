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
//! - **Stable orbit** — the ship completes one full revolution (2π radians)
//!   around the dominant body after at least `min_orbit_samples` steps.
//! - **Collision** — the ship is inside a body's radius.
//! - **Max steps** — `config.max_steps_per_segment` steps have been taken.

use glam::DVec3;

use crate::body_state_provider::BodyStateProvider;
use crate::forces::{ForceRegistry, GravityForce, ManeuverThrustForce};
use crate::integrator::{Integrator, IntegratorConfig};
use crate::maneuver::{ManeuverSequence, burn_duration};
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
    /// First sample that belongs to the closed-loop portion of a stable orbit.
    ///
    /// Segments that include a finite burn can have leading samples before the
    /// orbit settles; the renderer should only close the loop from this index.
    pub stable_orbit_start_index: Option<usize>,
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
    /// Cone signal value at which prediction is no longer useful. Default: 1e6.
    pub cone_fade_threshold: f64,
    /// Minimum samples before stable-orbit detection is attempted. Default: 100.
    pub min_orbit_samples: usize,
    /// Scale factor applied to the raw cone-width proxy. Default: 1.0.
    pub cone_width_scale: f64,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            max_steps_per_segment: 100_000,
            cone_fade_threshold: 1e6,
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
/// Uses perturbation ratio and *inverse* step size as the uncertainty signal.
/// Smaller accepted steps mean the integrator found the region numerically
/// sensitive, so the cone should widen as `step_size` shrinks.
pub fn cone_width(sample: &TrajectorySample) -> f64 {
    const NOMINAL_STEP_SIZE: f64 = 60.0;
    const NOMINAL_CONE_WIDTH_M: f64 = 1.0e6;

    let step_factor = NOMINAL_STEP_SIZE / sample.step_size.max(1e-3);
    sample.perturbation_ratio * step_factor * NOMINAL_CONE_WIDTH_M
}

fn cone_width_scaled(sample: &TrajectorySample, config: &PredictionConfig) -> f64 {
    cone_width(sample) * config.cone_width_scale
}

// ---------------------------------------------------------------------------
// Stable-orbit detection
// ---------------------------------------------------------------------------

/// Tracks angular progress around the dominant body to detect orbit completion.
#[derive(Debug, Clone)]
struct OrbitTracker {
    /// Dominant body ID at reference time.
    dominant_body: BodyId,
    /// Previous body-relative position (for incremental angle tracking).
    prev_rel_pos: DVec3,
    /// Orbital plane normal, estimated from the reference state.
    normal: DVec3,
    /// Cumulative angle swept since the reference point (radians).
    cumulative_angle: f64,
}

impl OrbitTracker {
    fn new(rel_position: DVec3, rel_velocity: DVec3, dominant_body: BodyId) -> Self {
        // Orbital plane normal = r × v (points along angular momentum).
        let normal = rel_position.cross(rel_velocity);
        let normal = if normal.length_squared() > 1e-20 {
            normal.normalize()
        } else {
            DVec3::Y // fallback for degenerate cases
        };
        Self {
            dominant_body,
            prev_rel_pos: rel_position,
            normal,
            cumulative_angle: 0.0,
        }
    }

    /// Update with a new sample position. Returns true if one full revolution
    /// (≥ 2π radians) has been completed.
    fn update(&mut self, rel_position: DVec3, min_samples: usize, sample_count: usize) -> bool {
        if sample_count < min_samples {
            self.prev_rel_pos = rel_position;
            return false;
        }

        // Signed angle between prev and current position vectors, projected
        // onto the orbital plane.
        let cross = self.prev_rel_pos.cross(rel_position);
        let dot = self.prev_rel_pos.dot(rel_position);
        let sin_angle = cross.dot(self.normal);
        let angle = sin_angle.atan2(dot);
        self.cumulative_angle += angle;
        self.prev_rel_pos = rel_position;

        self.cumulative_angle.abs() >= std::f64::consts::TAU
    }
}

// ---------------------------------------------------------------------------
// Propagation context
// ---------------------------------------------------------------------------

/// Bundles the shared context needed by propagation functions.
pub struct PropagationContext<'a> {
    pub ephemeris: &'a dyn BodyStateProvider,
    pub bodies: &'a [BodyDefinition],
    pub prediction_config: &'a PredictionConfig,
    pub integrator_config: IntegratorConfig,
    pub ship_thrust_acceleration: f64,
}

#[derive(Debug, Clone, Copy)]
struct ScheduledBurn {
    /// Δv in the local prograde/normal/radial frame (m/s). The world-frame
    /// direction is recomputed each integrator substep from the live ship
    /// state, so long burns stay aligned with the rotating orbital frame.
    delta_v_local: DVec3,
    reference_body: BodyId,
    acceleration: f64,
    start_time: f64,
    duration: f64,
}

// ---------------------------------------------------------------------------
// Segment propagation
// ---------------------------------------------------------------------------

/// Propagate one trajectory segment from `initial_state` at `start_time` until
/// `end_time` or a termination condition.
fn propagate_segment(
    initial_state: StateVector,
    start_time: f64,
    end_time: f64,
    burns: &[ScheduledBurn],
    ctx: &PropagationContext,
    remaining_budget: &mut Option<usize>,
) -> TrajectorySegment {
    let mut forces = ForceRegistry::new();
    forces.add(Box::new(GravityForce));

    for burn in burns {
        forces.add(Box::new(ManeuverThrustForce::new(
            burn.delta_v_local,
            burn.reference_body,
            burn.acceleration,
            burn.start_time,
            burn.duration,
        )));
    }

    let mut integrator = Integrator::new(ctx.integrator_config.clone());
    let mut state = initial_state;
    let mut time = start_time;
    let mut samples: Vec<TrajectorySample> =
        Vec::with_capacity(ctx.prediction_config.max_steps_per_segment.min(8192));
    let mut body_states_buf = Vec::new();

    // Stable orbit detection needs a reference state from AFTER all burns end.
    // For coast-only segments (no burns) the reference is the initial state.
    // For segments with burns, we capture the state once all burns have finished.
    let last_burn_end = burns
        .iter()
        .map(|b| b.start_time + b.duration)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut orbit_tracker: Option<OrbitTracker> = None;
    let mut stable_orbit_start_index: Option<usize> = None;
    // For coast-only segments, we set the reference after the first sample so
    // we have body state info from the ephemeris query.
    let needs_initial_reference = burns.is_empty();
    let mut samples_since_reference: usize = 0;

    #[derive(Debug)]
    enum TerminationReason {
        MaxSteps,
        BudgetExhausted,
        EndTime,
        ConeFade,
        Normal,
    }
    let mut termination = TerminationReason::Normal;

    loop {
        if samples.len() >= ctx.prediction_config.max_steps_per_segment {
            termination = TerminationReason::MaxSteps;
            break;
        }
        if let Some(rem) = remaining_budget
            && *rem == 0
        {
            termination = TerminationReason::BudgetExhausted;
            break;
        }
        if time >= end_time {
            termination = TerminationReason::EndTime;
            break;
        }

        let remaining_time = end_time - time;
        if remaining_time <= 0.0 {
            break;
        }

        let (new_state, mut sample) =
            integrator.step_capped(state, time, &forces, ctx.ephemeris, remaining_time);

        ctx.ephemeris.query_into(sample.time, &mut body_states_buf);

        // Cache the dominant body's position at sample time so the renderer
        // can derive body-relative offsets without querying the ephemeris.
        if sample.dominant_body < body_states_buf.len() {
            sample.dominant_body_pos = body_states_buf[sample.dominant_body].position;
        }

        // Collision check against all body surfaces at the new position/time.
        let mut collision_id: Option<BodyId> = None;
        for body_def in ctx.bodies.iter() {
            if body_def.id >= body_states_buf.len() {
                break;
            }
            let body_state = body_states_buf[body_def.id];
            let dist_sq = (sample.position - body_state.position).length_squared();
            if dist_sq < body_def.radius_m * body_def.radius_m {
                collision_id = Some(body_def.id);
                break;
            }
        }
        if let Some(cid) = collision_id {
            samples.push(sample);
            if let Some(rem) = remaining_budget.as_mut() {
                *rem = rem.saturating_sub(1);
            }
            eprintln!("[trajectory] segment terminated: Collision, samples={}", samples.len());
            return TrajectorySegment {
                samples,
                is_stable_orbit: false,
                stable_orbit_start_index: None,
                collision_body: Some(cid),
            };
        }

        samples.push(sample);
        if let Some(rem) = remaining_budget.as_mut() {
            *rem = rem.saturating_sub(1);
        }

        // Cone width fade — prediction too uncertain to be useful.
        if cone_width_scaled(&sample, ctx.prediction_config)
            > ctx.prediction_config.cone_fade_threshold
        {
            termination = TerminationReason::ConeFade;
            break;
        }

        // Capture the orbit reference state once all burns have ended (or
        // after the first sample for coast-only segments).
        let should_capture = if needs_initial_reference {
            orbit_tracker.is_none()
        } else {
            orbit_tracker.is_none() && sample.time >= last_burn_end
        };
        if should_capture {
            let dom = sample.dominant_body;
            let body_pos = if dom < body_states_buf.len() {
                body_states_buf[dom].position
            } else {
                DVec3::ZERO
            };
            let body_vel = if dom < body_states_buf.len() {
                body_states_buf[dom].velocity
            } else {
                DVec3::ZERO
            };
            orbit_tracker = Some(OrbitTracker::new(
                new_state.position - body_pos,
                new_state.velocity - body_vel,
                dom,
            ));
            samples_since_reference = 0;
            stable_orbit_start_index = samples.len().checked_sub(1);
        } else if orbit_tracker.is_some() {
            samples_since_reference += 1;
        }

        // Stable orbit: check if we've completed one full revolution.
        if let Some(ref mut tracker) = orbit_tracker {
            let dom = tracker.dominant_body;
            let body_pos = if dom < body_states_buf.len() {
                body_states_buf[dom].position
            } else {
                DVec3::ZERO
            };
            let rel_pos = new_state.position - body_pos;
            if tracker.update(
                rel_pos,
                ctx.prediction_config.min_orbit_samples,
                samples_since_reference,
            ) {
                eprintln!("[trajectory] segment terminated: StableOrbit, samples={}", samples.len());
                return TrajectorySegment {
                    samples,
                    is_stable_orbit: true,
                    stable_orbit_start_index,
                    collision_body: None,
                };
            }
        }

        state = new_state;
        time = sample.time;
    }

    eprintln!("[trajectory] segment terminated: {:?}, samples={}, time={:.1}s", termination, samples.len(), time);
    TrajectorySegment {
        samples,
        is_stable_orbit: false,
        stable_orbit_start_index: None,
        collision_body: None,
    }
}

// ---------------------------------------------------------------------------
// Public propagation API
// ---------------------------------------------------------------------------

/// Propagate a trajectory through the given maneuver sequence.
///
/// Returns a [`TrajectoryPrediction`] with one [`TrajectorySegment`] per leg.
/// Legs run from start -> first node, node -> node, ..., last node -> end of
/// ephemeris. Early termination (collision, stable orbit) stops further
/// processing.
pub fn propagate_trajectory(
    initial_state: StateVector,
    start_time: f64,
    maneuvers: &ManeuverSequence,
    ephemeris: &dyn BodyStateProvider,
    bodies: &[BodyDefinition],
    config: &PredictionConfig,
    integrator_config: IntegratorConfig,
    ship_thrust_acceleration: f64,
) -> TrajectoryPrediction {
    propagate_trajectory_budgeted(
        initial_state,
        start_time,
        maneuvers,
        ephemeris,
        bodies,
        config,
        integrator_config,
        ship_thrust_acceleration,
        None,
    )
}

/// Like [`propagate_trajectory`] but respects a [`PropagationBudget`] for
/// progressive refinement.  Pass `Some(budget)` to cap the total integrator
/// steps across this call.  The caller can track how far the prediction got by
/// inspecting the last sample in the last segment.
#[allow(clippy::too_many_arguments)]
pub fn propagate_trajectory_budgeted(
    initial_state: StateVector,
    start_time: f64,
    maneuvers: &ManeuverSequence,
    ephemeris: &dyn BodyStateProvider,
    bodies: &[BodyDefinition],
    config: &PredictionConfig,
    integrator_config: IntegratorConfig,
    ship_thrust_acceleration: f64,
    budget: Option<PropagationBudget>,
) -> TrajectoryPrediction {
    let ctx = PropagationContext {
        ephemeris,
        bodies,
        prediction_config: config,
        integrator_config,
        ship_thrust_acceleration,
    };

    let mut segments: Vec<TrajectorySegment> = Vec::new();
    let mut state = initial_state;
    let mut time = start_time;
    let mut remaining_budget: Option<usize> = budget.map(|b| b.max_steps);
    let mut active_burns: Vec<ScheduledBurn> = Vec::new();

    // Build a sorted index into the nodes vec to avoid cloning.
    let mut node_order: Vec<usize> = (0..maneuvers.nodes.len()).collect();
    node_order.sort_by(|&a, &b| {
        maneuvers.nodes[a]
            .time
            .partial_cmp(&maneuvers.nodes[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let ephemeris_end = start_time + ctx.ephemeris.time_span();
    let leg_count = node_order.len() + 1;

    for leg_idx in 0..leg_count {
        let leg_end = if leg_idx < node_order.len() {
            maneuvers.nodes[node_order[leg_idx]].time
        } else {
            ephemeris_end
        };

        active_burns.retain(|burn| burn.start_time + burn.duration > time);
        let segment = propagate_segment(
            state,
            time,
            leg_end,
            &active_burns,
            &ctx,
            &mut remaining_budget,
        );

        let early_exit = segment.is_stable_orbit
            || segment.collision_body.is_some()
            || segment
                .samples
                .last()
                .map(|s| {
                    cone_width_scaled(s, ctx.prediction_config)
                        > ctx.prediction_config.cone_fade_threshold
                })
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

        // Plan the maneuver burn at the node boundary. The local-frame Δv is
        // stored on the burn and converted to a world direction each integrator
        // substep, so the thrust tracks the ship's instantaneous prograde /
        // normal / radial frame rather than freezing it at burn start.
        if leg_idx < node_order.len() {
            let node = &maneuvers.nodes[node_order[leg_idx]];
            time = node.time;

            let duration = burn_duration(node.delta_v.length(), ctx.ship_thrust_acceleration);
            if duration > 0.0 && node.delta_v.length_squared() > 0.0 {
                active_burns.push(ScheduledBurn {
                    delta_v_local: node.delta_v,
                    reference_body: node.reference_body,
                    acceleration: ctx.ship_thrust_acceleration,
                    start_time: node.time,
                    duration,
                });
            }
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
    use crate::integrator::IntegratorConfig;
    use crate::maneuver::{ManeuverNode, delta_v_to_world};
    use crate::patched_conics::PatchedConics;
    use crate::types::{BodyDefinition, BodyKind, G, OrbitalElements, ShipDefinition, SolarSystemDefinition};
    use std::collections::HashMap;

    fn sample_with(perturbation_ratio: f64, step_size: f64) -> TrajectorySample {
        TrajectorySample {
            time: 0.0,
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            dominant_body: 0,
            perturbation_ratio,
            step_size,
            dominant_body_pos: DVec3::ZERO,
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
        assert!(
            wide > narrow,
            "smaller adaptive steps should widen the cone"
        );
    }

    #[test]
    fn orbit_tracker_not_triggered_below_min_samples() {
        let rel_pos = DVec3::new(1e7, 0.0, 0.0);
        let rel_vel = DVec3::new(0.0, 1000.0, 0.0);
        let mut tracker = OrbitTracker::new(rel_pos, rel_vel, 0);
        // Feed the same position — should not trigger below min_samples.
        assert!(!tracker.update(rel_pos, 100, 50));
    }

    #[test]
    fn orbit_tracker_fires_after_full_revolution() {
        let radius = 1e7;
        let speed = 1000.0;
        let rel_pos = DVec3::new(radius, 0.0, 0.0);
        let rel_vel = DVec3::new(0.0, 0.0, speed);
        let mut tracker = OrbitTracker::new(rel_pos, rel_vel, 0);

        // Simulate positions around a circular orbit in 36 steps (10° each).
        // min_samples causes angle accumulation to start after the minimum,
        // so the trigger fires at ~(min_samples + 36) steps.
        let min_samples = 10;
        let mut triggered_at = None;
        for i in 1..=80 {
            let angle = (i as f64) * std::f64::consts::TAU / 36.0;
            let pos = DVec3::new(radius * angle.cos(), 0.0, radius * angle.sin());
            if tracker.update(pos, min_samples, i) {
                triggered_at = Some(i);
                break;
            }
        }
        let step = triggered_at.expect("orbit tracker should fire after one revolution");
        // Angle accumulation starts after min_samples, then needs ~36 more
        // steps (one revolution) to reach 2π.
        assert!(
            step >= min_samples + 35 && step <= min_samples + 38,
            "expected trigger near step {}, got {step}",
            min_samples + 36,
        );
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
            orbital_elements: None,
            procedural: None,
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

    #[test]
    fn delta_v_prograde_aligns_with_velocity() {
        let dv_world = delta_v_to_world(
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1000.0, 0.0),
            DVec3::new(1e7, 0.0, 0.0),
            DVec3::ZERO,
            DVec3::ZERO,
        );
        assert!(dv_world.y > 0.99, "prograde should be ~+Y, got {dv_world}");
        assert!(dv_world.x.abs() < 1e-10);
        assert!(dv_world.z.abs() < 1e-10);
    }

    #[test]
    fn delta_v_degenerate_uses_fallback_frame() {
        let dv_local = DVec3::new(1.0, 2.0, 3.0);
        let dv_world = delta_v_to_world(
            dv_local,
            DVec3::ZERO,
            DVec3::new(1e7, 0.0, 0.0),
            DVec3::ZERO,
            DVec3::ZERO,
        );
        assert!(dv_world.length() > 0.0, "should produce a non-zero result");
    }

    #[test]
    fn maneuver_is_integrated_as_finite_burn() {
        let system = make_single_star_system();
        let ephemeris = PatchedConics::new(&system, 1_000.0);

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
            &ephemeris,
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
            orbital_elements: None,
            procedural: None,
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
            orbital_elements: Some(OrbitalElements {
                semi_major_axis_m: AU,
                eccentricity: 0.0,
                inclination_rad: 0.0,
                lon_ascending_node_rad: 0.0,
                arg_periapsis_rad: 0.0,
                true_anomaly_rad: 0.0,
            }),
            procedural: None,
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

    const AU: f64 = 1.496e11;
    const SUN_GM: f64 = 1.327_124_4e20;

    #[test]
    fn prograde_burn_raises_apoapsis_around_thalos() {
        let system = make_thalos_like_system();
        let ephemeris = PatchedConics::new(&system, 1.0e7);

        let thalos_state_0 = ephemeris.query_body(1, 0.0);
        let thalos_gm = system.bodies[1].gm;
        let orbit_radius = system.bodies[1].radius_m + 200_000.0;
        let circular_speed = (thalos_gm / orbit_radius).sqrt();

        let ship_state = StateVector {
            position: thalos_state_0.position + DVec3::new(orbit_radius, 0.0, 0.0),
            velocity: thalos_state_0.velocity + DVec3::new(0.0, 0.0, circular_speed),
        };

        let mut maneuvers = ManeuverSequence::new();
        // 100 m/s prograde relative to Thalos — should raise apoapsis.
        maneuvers.add(ManeuverNode {
            time: 10.0,
            delta_v: DVec3::new(100.0, 0.0, 0.0),
            reference_body: 1,
        });

        let prediction = propagate_trajectory(
            ship_state,
            0.0,
            &maneuvers,
            &ephemeris,
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
                continue; // pre-burn leg
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

        assert!(
            burn_leg_samples > 0,
            "expected samples after the burn; got none"
        );
        assert!(
            max_r > orbit_radius + 1.0e5,
            "prograde burn should raise apoapsis above {:.0} m; got max_r={:.0} m",
            orbit_radius + 1.0e5,
            max_r
        );
        assert!(
            min_r > system.bodies[1].radius_m,
            "orbit should not collide with Thalos surface (r_min={:.0} m, surface={:.0} m)",
            min_r,
            system.bodies[1].radius_m
        );
    }

    #[test]
    fn prograde_burn_with_sun_reference_lowers_thalos_apoapsis() {
        // Regression: if reference_body == Sun for a ship orbiting Thalos,
        // "prograde" is along the ship's heliocentric velocity (~Thalos orbital
        // velocity direction), NOT along the Thalos-relative velocity. The
        // resulting burn deforms the Thalos-relative orbit into a crash or a
        // wildly wrong shape — matching the visual "orbit curls in" bug.
        let system = make_thalos_like_system();
        let ephemeris = PatchedConics::new(&system, 1.0e7);

        let thalos_state_0 = ephemeris.query_body(1, 0.0);
        let thalos_gm = system.bodies[1].gm;
        let orbit_radius = system.bodies[1].radius_m + 200_000.0;
        let circular_speed = (thalos_gm / orbit_radius).sqrt();

        let ship_state = StateVector {
            position: thalos_state_0.position + DVec3::new(orbit_radius, 0.0, 0.0),
            velocity: thalos_state_0.velocity + DVec3::new(0.0, 0.0, circular_speed),
        };

        let mut maneuvers = ManeuverSequence::new();
        // User *thinks* they are adding prograde relative to Thalos, but the
        // game captured reference_body as the Sun (body 0).
        maneuvers.add(ManeuverNode {
            time: 10.0,
            delta_v: DVec3::new(100.0, 0.0, 0.0),
            reference_body: 0,
        });

        let prediction = propagate_trajectory(
            ship_state,
            0.0,
            &maneuvers,
            &ephemeris,
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

        let mut min_r = f64::INFINITY;
        for (i, seg) in prediction.segments.iter().enumerate() {
            if i == 0 {
                continue;
            }
            for s in &seg.samples {
                let thalos_pos = ephemeris.query_body(1, s.time).position;
                let r = (s.position - thalos_pos).length();
                if r < min_r {
                    min_r = r;
                }
            }
        }
        eprintln!(
            "with Sun reference, min_r = {:.0} m (surface {:.0} m)",
            min_r, system.bodies[1].radius_m
        );
    }

    #[test]
    fn large_prograde_burn_escapes_without_collision() {
        // Regression for the "curls in and collides" bug. With frozen-direction
        // thrust, a 5000 m/s prograde burn from a 200 km Thalos orbit crashes
        // into the planet because the 10 000 s burn spans ~2.5 orbital periods
        // and the thrust direction stays fixed while the ship rotates. The
        // velocity-tracking maneuver thrust keeps the burn aligned with the
        // instantaneous prograde frame so the ship genuinely escapes.
        let system = make_thalos_like_system();
        let ephemeris = PatchedConics::new(&system, 1.0e8);

        let thalos_state_0 = ephemeris.query_body(1, 0.0);
        let thalos_gm = system.bodies[1].gm;
        let orbit_radius = system.bodies[1].radius_m + 200_000.0;
        let circular_speed = (thalos_gm / orbit_radius).sqrt();

        let ship_state = StateVector {
            position: thalos_state_0.position + DVec3::new(orbit_radius, 0.0, 0.0),
            velocity: thalos_state_0.velocity + DVec3::new(0.0, 0.0, circular_speed),
        };

        let mut maneuvers = ManeuverSequence::new();
        // Strong escape burn — a second of UI drag easily reaches this.
        maneuvers.add(ManeuverNode {
            time: 5.0,
            delta_v: DVec3::new(5_000.0, 0.0, 0.0),
            reference_body: 1,
        });

        let prediction = propagate_trajectory(
            ship_state,
            0.0,
            &maneuvers,
            &ephemeris,
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
            "ship fell below Thalos' surface (min_r = {:.0} m, surface = {:.0} m)",
            min_r,
            system.bodies[1].radius_m
        );
        assert!(
            max_r > orbit_radius * 5.0,
            "escape burn should reach far past the initial orbit; max_r = {:.0} m",
            max_r
        );
    }

    #[test]
    fn stable_orbit_detected_after_prograde_burn() {
        // After a small prograde burn the post-burn orbit should be detected as
        // stable within ~1 revolution, not keep drawing dozens of orbits.
        let system = make_thalos_like_system();
        let ephemeris = PatchedConics::new(&system, 1.0e7);

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
            &ephemeris,
            &system.bodies,
            &PredictionConfig::default(),
            IntegratorConfig {
                symplectic_dt: 60.0,
                rk_initial_dt: 60.0,
                ..IntegratorConfig::default()
            },
            system.ship.thrust_acceleration,
        );

        // The post-burn segment should be marked as a stable orbit.
        let post_burn = &prediction.segments[prediction.segments.len() - 1];
        assert!(
            post_burn.is_stable_orbit,
            "post-burn segment should detect stable orbit; got {} samples, is_stable={}",
            post_burn.samples.len(),
            post_burn.is_stable_orbit,
        );
        assert!(
            post_burn.stable_orbit_start_index.is_some_and(|idx| idx > 0),
            "stable orbit should start after the finite-burn lead-in; got {:?}",
            post_burn.stable_orbit_start_index,
        );

        // The post-burn orbit is elliptical (100 m/s prograde on a ~5200 m/s
        // circular orbit), so the actual period is longer than the circular
        // estimate. Use the vis-viva equation: after the burn the speed is
        // v_circ + Δv ≈ 5300 m/s at periapsis r = orbit_radius.
        // a = 1 / (2/r - v²/μ)
        let v_post = circular_speed + 100.0;
        let semi_major = 1.0 / (2.0 / orbit_radius - v_post * v_post / thalos_gm);
        let elliptical_period =
            2.0 * std::f64::consts::PI * (semi_major.powi(3) / thalos_gm).sqrt();
        let prediction_time = post_burn
            .samples
            .last()
            .map(|s| s.time - post_burn.samples.first().unwrap().time)
            .unwrap_or(0.0);

        // Should terminate within ~3 orbital periods (1 full revolution + burn
        // duration + margin for the detection to fire after min_orbit_samples).
        assert!(
            prediction_time < elliptical_period * 3.0,
            "prediction covered {:.0}s but orbital period is ~{:.0}s — too many orbits",
            prediction_time,
            elliptical_period,
        );
    }

    #[test]
    fn orbit_tracker_fires_after_full_retrograde_revolution() {
        let radius = 1e7;
        let speed = 1000.0;
        let rel_pos = DVec3::new(radius, 0.0, 0.0);
        let rel_vel = DVec3::new(0.0, 0.0, -speed);
        let mut tracker = OrbitTracker::new(rel_pos, rel_vel, 0);

        let min_samples = 10;
        let mut triggered_at = None;
        for i in 1..=80 {
            let angle = -(i as f64) * std::f64::consts::TAU / 36.0;
            let pos = DVec3::new(radius * angle.cos(), 0.0, radius * angle.sin());
            if tracker.update(pos, min_samples, i) {
                triggered_at = Some(i);
                break;
            }
        }

        let step = triggered_at.expect("retrograde orbit tracker should fire after one revolution");
        assert!(
            step >= min_samples + 35 && step <= min_samples + 38,
            "expected trigger near step {}, got {step}",
            min_samples + 36,
        );
    }
}
