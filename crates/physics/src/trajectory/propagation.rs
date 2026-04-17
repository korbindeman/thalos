//! Integrator-driven propagation that produces [`NumericSegment`]s.
//!
//! A "segment" is one propagation run from a start state at `start_time` until
//! a termination condition trips:
//!
//! - **Stable orbit** — the ship swept ≥2π around the dominant body after at
//!   least `min_orbit_samples` steps.
//! - **Collision** — ship dropped below a body's radius.
//! - **Max steps** or **budget exhausted**.
//! - **End time** — leg boundary reached.
//!
//! [`FlightPlan`](super::FlightPlan) wraps repeated calls to
//! [`propagate_segment`] around a sequence of maneuver nodes.

use glam::DVec3;

use super::numeric::NumericSegment;
use crate::body_state_provider::BodyStateProvider;
use crate::effects::EffectRegistry;
use crate::integrator::{Integrator, IntegratorConfig};
use crate::types::{BodyDefinition, BodyId, StateVector, TrajectorySample};

/// Tuning for trajectory prediction.
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    /// Maximum integration steps per segment. Default: 100_000.
    pub max_steps_per_segment: usize,
    /// Minimum samples before stable-orbit detection starts. Default: 100.
    pub min_orbit_samples: usize,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            max_steps_per_segment: 100_000,
            min_orbit_samples: 100,
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
    /// Optional target body for biased step-size capping near the target.
    pub target_body: Option<BodyId>,
}

/// Maximum step size (seconds) the propagator is allowed to take when the
/// craft is within [`TARGET_DT_CAP_FACTOR`] × target SOI of the target body.
/// Chosen so a 1-minute cap catches grazing flybys that a larger adaptive
/// step could otherwise skip over.
const TARGET_DT_CAP_SECS: f64 = 60.0;

/// Ship-to-target distance, expressed as a multiple of the target's SOI
/// radius, inside which [`TARGET_DT_CAP_SECS`] applies.
const TARGET_DT_CAP_FACTOR: f64 = 3.0;

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
    // Tracks the SOI-level anchor (smallest containing SOI) for orbit
    // tracker reset detection — separate from the rendering anchor which
    // is stepped up to the parent planet for moons.
    let mut prev_soi_anchor: Option<BodyId> = None;
    // Leg-locked rendering anchor: the body whose frame the entire leg is
    // drawn in. Set from the first sample's dominant body and held fixed
    // for every subsequent sample so orbits that transiently cross a
    // Hill-sphere boundary don't deform into the outer body's frame.
    let mut leg_anchor: Option<BodyId> = None;

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

        let mut remaining_time = end_time - time;
        if remaining_time <= 0.0 {
            break;
        }

        // Target-biased step cap: when a target body is set and the craft is
        // within a few SOI radii, shrink the allowed step so a coarse
        // integrator step can't skip across the Hill sphere and miss the
        // encounter entirely.
        if let Some(target_id) = ctx.target_body
            && let Some(target_def) = ctx.bodies.iter().find(|b| b.id == target_id)
            && target_def.soi_radius_m.is_finite()
        {
            let target_state = ctx.ephemeris.query_body(target_id, time);
            let dist = (state.position - target_state.position).length();
            if dist < target_def.soi_radius_m * TARGET_DT_CAP_FACTOR {
                remaining_time = remaining_time.min(TARGET_DT_CAP_SECS);
            }
        }

        let (new_state, mut sample) =
            integrator.step_capped(state, time, registry, ctx.ephemeris, remaining_time);

        ctx.ephemeris.query_into(sample.time, &mut body_states_buf);

        // SOI containment: find the smallest sphere-of-influence that
        // contains the ship.  This is the physical SOI anchor used for
        // orbit-tracker reset detection.  Surface collision check is fused
        // into the same O(N) pass.
        let mut soi_anchor_id = sample.dominant_body;
        let mut soi_anchor_soi = f64::INFINITY;
        let mut collision_id: Option<BodyId> = None;
        for body_def in ctx.bodies.iter() {
            if body_def.id >= body_states_buf.len() {
                break;
            }
            let dist_sq =
                (sample.position - body_states_buf[body_def.id].position).length_squared();
            let soi = body_def.soi_radius_m;
            if dist_sq <= soi * soi && soi <= soi_anchor_soi {
                soi_anchor_id = body_def.id;
                soi_anchor_soi = soi;
            }
            if collision_id.is_none() && dist_sq < body_def.radius_m * body_def.radius_m {
                collision_id = Some(body_def.id);
            }
        }

        // Rendering anchor = physics SOI anchor.  The trajectory leg inside
        // a moon's SOI renders in that moon's frame, so a capture orbit around
        // Mira shows as a proper ellipse rather than as epicycles in Thalos's
        // frame.  SOI crossings become run boundaries in the renderer, joined
        // by faint bridges.
        sample.soi_body = soi_anchor_id;

        // Leg-locked anchor: captured from the first sample's dominant body
        // and held fixed for the rest of the leg. Every sample renders in
        // this body's frame, so the full leg reads as a stable orbit / arc
        // relative to one reference — no deformation when the trajectory
        // crosses the anchor's Hill-sphere boundary (§7.2 caveat).
        let anchor = *leg_anchor.get_or_insert(sample.dominant_body);
        sample.anchor_body = anchor;
        if anchor < body_states_buf.len() {
            sample.ref_pos = body_states_buf[anchor].position;
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

        // Reset orbit tracker on SOI transitions (entering/exiting a
        // moon's SOI) so stable-orbit detection doesn't combine partial
        // orbits from before and after an encounter.  Uses the SOI-level
        // anchor, not the rendering anchor (which is always the planet).
        if let Some(prev) = prev_soi_anchor {
            if soi_anchor_id != prev {
                orbit_tracker = None;
            }
        }
        prev_soi_anchor = Some(soi_anchor_id);

        // Capture the orbit reference state on the first sample. Callers
        // only enable `stop_on_stable_orbit` for coast sub-segments, so no
        // burn-end synchronisation is needed here. Tracker uses the SOI
        // anchor (not the leg-locked render anchor) so stable-orbit
        // detection still works when a ship captures into a body whose
        // SOI it transitioned into within the leg.
        if orbit_tracker.is_none() {
            let tracker_anchor = soi_anchor_id;
            let body_pos = if tracker_anchor < body_states_buf.len() {
                body_states_buf[tracker_anchor].position
            } else {
                DVec3::ZERO
            };
            let body_vel = if tracker_anchor < body_states_buf.len() {
                body_states_buf[tracker_anchor].velocity
            } else {
                DVec3::ZERO
            };
            orbit_tracker = Some(OrbitTracker::new(
                new_state.position - body_pos,
                new_state.velocity - body_vel,
                tracker_anchor,
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
