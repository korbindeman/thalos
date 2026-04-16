//! Integrator-driven propagation that produces [`NumericSegment`]s.
//!
//! A "segment" is one propagation run from a start state at `start_time` until
//! a termination condition trips:
//!
//! - **Cone width** exceeds `config.cone_fade_threshold` — prediction too
//!   uncertain.
//! - **Stable orbit** — the ship swept ≥2π around the dominant body after at
//!   least `min_orbit_samples` steps.
//! - **Collision** — ship dropped below a body's radius.
//! - **Max steps** or **budget exhausted**.
//! - **End time** — leg boundary reached.
//!
//! [`FlightPlan`](super::FlightPlan) wraps repeated calls to
//! [`propagate_segment`] around a sequence of maneuver nodes.

use glam::DVec3;

use super::numeric::{NumericSegment, cone_width};
use crate::body_state_provider::BodyStateProvider;
use crate::effects::EffectRegistry;
use crate::integrator::{Integrator, IntegratorConfig};
use crate::types::{BodyDefinition, BodyId, StateVector, TrajectorySample};

/// Tuning for trajectory prediction.
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    /// Maximum integration steps per segment. Default: 100_000.
    pub max_steps_per_segment: usize,
    /// Cone signal at which prediction stops being useful. Default: 1e6 m.
    pub cone_fade_threshold: f64,
    /// Minimum samples before stable-orbit detection starts. Default: 100.
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

/// Caps the integrator-step budget across a [`propagate_flight_plan`]
/// call.  Useful for progressive refinement (coarse first pass, fine final
/// pass).
#[derive(Debug, Clone, Copy)]
pub struct PropagationBudget {
    pub max_steps: usize,
}

impl PropagationBudget {
    pub fn new(max_steps: usize) -> Self {
        Self { max_steps }
    }
}

pub(super) fn cone_width_scaled(sample: &TrajectorySample, config: &PredictionConfig) -> f64 {
    cone_width(sample) * config.cone_width_scale
}

/// Tracks angular progress around the anchor body to detect orbit closure.
#[derive(Debug, Clone)]
pub(super) struct OrbitTracker {
    pub anchor_body: BodyId,
    prev_rel_pos: DVec3,
    normal: DVec3,
    cumulative_angle: f64,
}

impl OrbitTracker {
    pub(super) fn new(rel_position: DVec3, rel_velocity: DVec3, anchor_body: BodyId) -> Self {
        let normal = rel_position.cross(rel_velocity);
        let normal = if normal.length_squared() > 1e-20 {
            normal.normalize()
        } else {
            DVec3::Y
        };
        Self {
            anchor_body,
            prev_rel_pos: rel_position,
            normal,
            cumulative_angle: 0.0,
        }
    }

    pub(super) fn update(
        &mut self,
        rel_position: DVec3,
        min_samples: usize,
        sample_count: usize,
    ) -> bool {
        if sample_count < min_samples {
            self.prev_rel_pos = rel_position;
            return false;
        }
        let cross = self.prev_rel_pos.cross(rel_position);
        let dot = self.prev_rel_pos.dot(rel_position);
        let sin_angle = cross.dot(self.normal);
        let angle = sin_angle.atan2(dot);
        self.cumulative_angle += angle;
        self.prev_rel_pos = rel_position;
        self.cumulative_angle.abs() >= std::f64::consts::TAU
    }
}

/// Shared context for one propagation call.
pub(super) struct PropagationContext<'a> {
    pub ephemeris: &'a dyn BodyStateProvider,
    pub bodies: &'a [BodyDefinition],
    pub prediction_config: &'a PredictionConfig,
    pub integrator_config: IntegratorConfig,
}

/// Definition of a finite-duration maneuver burn that can be turned into a
/// `ManeuverThrustEffect` at flight-plan construction time.
#[derive(Debug, Clone, Copy)]
pub struct ScheduledBurn {
    /// Δv in the local prograde/normal/radial frame (m/s).  The world-frame
    /// direction is recomputed each integrator substep so long burns track
    /// the rotating orbital frame.
    pub delta_v_local: DVec3,
    pub reference_body: BodyId,
    pub acceleration: f64,
    pub start_time: f64,
    pub duration: f64,
}

/// Propagate one segment from `initial_state` at `start_time` until `end_time`
/// or a termination condition trips. Effects active during this segment must
/// be baked into `registry` by the caller (thrust is not threaded as a
/// separate concept here — it's just another effect).
///
/// Stable-orbit detection fires only when `stop_on_stable_orbit` is true and
/// the registry has no active thrust (a burn interval can't close a stable
/// orbit by definition). The caller enforces the latter by splitting each
/// leg into burn + coast sub-segments.
pub(super) fn propagate_segment(
    initial_state: StateVector,
    start_time: f64,
    end_time: f64,
    registry: &EffectRegistry,
    ctx: &PropagationContext,
    stop_on_stable_orbit: bool,
    remaining_budget: &mut Option<usize>,
) -> NumericSegment {
    let mut integrator = Integrator::new(ctx.integrator_config.clone());
    let mut state = initial_state;
    let mut time = start_time;
    let mut samples: Vec<TrajectorySample> =
        Vec::with_capacity(ctx.prediction_config.max_steps_per_segment.min(8192));
    let mut body_states_buf = Vec::with_capacity(ctx.bodies.len());

    let mut orbit_tracker: Option<OrbitTracker> = None;
    let mut stable_orbit_start_index: Option<usize> = None;
    let mut samples_since_reference: usize = 0;

    loop {
        if samples.len() >= ctx.prediction_config.max_steps_per_segment {
            break;
        }
        if let Some(rem) = remaining_budget
            && *rem == 0
        {
            break;
        }
        if time >= end_time {
            break;
        }

        let remaining_time = end_time - time;
        if remaining_time <= 0.0 {
            break;
        }

        let (new_state, mut sample) =
            integrator.step_capped(state, time, registry, ctx.ephemeris, remaining_time);

        ctx.ephemeris.query_into(sample.time, &mut body_states_buf);

        // Anchor body: smallest SOI that contains the ship.  Geometric
        // containment is stable across steps, so the renderer can frame every
        // sample to its anchor without per-sample frame jumps.  Falls back to
        // the root body via INFINITY-SOI when no smaller SOI applies.
        // Surface collision check is fused into the same loop to avoid a
        // second O(N) pass over bodies per sample.
        let mut anchor_id = sample.dominant_body;
        let mut anchor_soi = f64::INFINITY;
        let mut collision_id: Option<BodyId> = None;
        for body_def in ctx.bodies.iter() {
            if body_def.id >= body_states_buf.len() {
                break;
            }
            let dist_sq =
                (sample.position - body_states_buf[body_def.id].position).length_squared();
            let soi = body_def.soi_radius_m;
            if dist_sq <= soi * soi && soi <= anchor_soi {
                anchor_id = body_def.id;
                anchor_soi = soi;
            }
            if collision_id.is_none() && dist_sq < body_def.radius_m * body_def.radius_m {
                collision_id = Some(body_def.id);
            }
        }
        sample.anchor_body = anchor_id;
        if anchor_id < body_states_buf.len() {
            sample.anchor_body_pos = body_states_buf[anchor_id].position;
        }
        if let Some(cid) = collision_id {
            samples.push(sample);
            if let Some(rem) = remaining_budget.as_mut() {
                *rem = rem.saturating_sub(1);
            }
            return NumericSegment {
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

        // Cone fade: prediction too uncertain.
        if cone_width_scaled(&sample, ctx.prediction_config)
            > ctx.prediction_config.cone_fade_threshold
        {
            break;
        }

        // Capture the orbit reference state on the first sample. Callers
        // only enable `stop_on_stable_orbit` for coast sub-segments, so no
        // burn-end synchronisation is needed here.
        if orbit_tracker.is_none() {
            let anchor = sample.anchor_body;
            let body_pos = if anchor < body_states_buf.len() {
                body_states_buf[anchor].position
            } else {
                DVec3::ZERO
            };
            let body_vel = if anchor < body_states_buf.len() {
                body_states_buf[anchor].velocity
            } else {
                DVec3::ZERO
            };
            orbit_tracker = Some(OrbitTracker::new(
                new_state.position - body_pos,
                new_state.velocity - body_vel,
                anchor,
            ));
            samples_since_reference = 0;
            stable_orbit_start_index = samples.len().checked_sub(1);
        } else {
            samples_since_reference += 1;
        }

        if let Some(ref mut tracker) = orbit_tracker {
            let anchor = tracker.anchor_body;
            let body_pos = if anchor < body_states_buf.len() {
                body_states_buf[anchor].position
            } else {
                DVec3::ZERO
            };
            let rel_pos = new_state.position - body_pos;
            let closed_orbit = tracker.update(
                rel_pos,
                ctx.prediction_config.min_orbit_samples,
                samples_since_reference,
            );
            if stop_on_stable_orbit && closed_orbit {
                return NumericSegment {
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

    NumericSegment {
        samples,
        is_stable_orbit: false,
        stable_orbit_start_index: None,
        collision_body: None,
    }
}
