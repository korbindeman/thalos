//! [`FlightPlan`]: a leg-based propagation through a maneuver sequence.
//!
//! A flight plan is a chain of [`Leg`]s. Each leg is the trajectory that
//! starts from the ship's state at some epoch, optionally applies a burn for
//! a finite duration, and then coasts until the next leg's start time (the
//! next maneuver) or the horizon.
//!
//! Burn and coast are modelled as separate `NumericSegment`s so they can be
//! propagated under different effect registries:
//!
//! - **`burn_segment`** — propagated with the leg's `ManeuverThrustEffect`
//!   active in the registry. Time span `[burn_start, burn_end)`.
//! - **`coast_segment`** — propagated with only the baseline registry
//!   (gravity + ambient effects). Time span `[burn_end, leg_end)`.
//!
//! The flat `segments` list is a back-compat mirror that flattens all burn +
//! coast segments in leg order; consumers that don't care about leg structure
//! (renderer, event detection) can iterate it unchanged.

use std::sync::Arc;

use super::Trajectory;
use super::events::{
    ClosestApproach, Encounter, EncounterId, TrajectoryEvent, aggregate_encounters,
    detect_segment_events, scan_closest_approaches,
};
use super::numeric::NumericSegment;
pub use super::propagation::ScheduledBurn;
use super::propagation::{
    PredictionConfig, PropagationBudget, PropagationContext, propagate_segment,
};
use crate::body_state_provider::BodyStateProvider;
use crate::effects::{Effect, EffectRegistry, ManeuverThrustEffect};
use crate::integrator::IntegratorConfig;
use crate::maneuver::{ManeuverSequence, burn_duration};
use crate::types::{BodyDefinition, BodyId, StateVector};
use glam::DVec3;

/// Immutable inputs for trajectory prediction.
#[derive(Clone)]
pub struct PredictionRequest {
    pub ship_state: StateVector,
    pub sim_time: f64,
    pub maneuvers: ManeuverSequence,
    pub active_burns: Vec<ScheduledBurn>,
    pub ephemeris: Arc<dyn BodyStateProvider>,
    pub bodies: Vec<BodyDefinition>,
    pub prediction_config: PredictionConfig,
    pub integrator_config: IntegratorConfig,
    pub ship_thrust_acceleration: f64,
    /// Optional target body for biased adaptive-step sizing.  When set, the
    /// propagator caps its maximum step size as the craft approaches this
    /// body so a coarse dt can't skip over a grazing flyby.
    pub target_body: Option<BodyId>,
}

/// One propagated leg of a flight plan.
///
/// Leg 0 has no burn (its `applied_delta_v` is `None` and `burn_segment` is
/// empty). Subsequent legs start with an optional burn sub-segment driven by
/// the leg's maneuver node, followed by a coast sub-segment.
#[derive(Debug, Clone)]
pub struct Leg {
    /// State vector at the leg's start — i.e. at the moment the burn begins
    /// (or at `initial_time` for leg 0).
    pub start_state: StateVector,
    pub start_time: f64,
    /// Δv in the local prograde/normal/radial frame, if this leg starts with
    /// a burn. `None` for leg 0.
    pub applied_delta_v: Option<DVec3>,
    /// Burn sub-segment, if the leg has a non-zero-duration burn. Spans
    /// `[start_time, start_time + burn_duration)`.
    pub burn_segment: Option<NumericSegment>,
    /// Coast sub-segment — always present, spans from the end of the burn
    /// (or from `start_time` if there is no burn) up to either the next
    /// leg's `start_time` or the horizon.
    pub coast_segment: NumericSegment,
}

impl Leg {
    /// Earliest propagated time covered by this leg.
    pub fn leg_start_time(&self) -> f64 {
        self.start_time
    }

    /// Latest propagated time covered by this leg (end of coast, or end of
    /// burn if coast is empty).
    pub fn leg_end_time(&self) -> Option<f64> {
        self.coast_segment
            .end_time()
            .or_else(|| self.burn_segment.as_ref().and_then(|s| s.end_time()))
    }

    /// Iterate burn (if present) then coast.
    pub fn segments(&self) -> impl Iterator<Item = &NumericSegment> {
        self.burn_segment.iter().chain(std::iter::once(&self.coast_segment))
    }

    /// Last state covered by the leg, or None if both sub-segments are empty.
    pub fn last_state(&self) -> Option<StateVector> {
        self.coast_segment
            .last_state()
            .or_else(|| self.burn_segment.as_ref().and_then(|s| s.last_state()))
    }

    /// Query the leg for a state vector at `time`.
    pub fn state_at(&self, time: f64) -> Option<StateVector> {
        if let Some(burn) = &self.burn_segment {
            let (start, end) = burn.epoch_range();
            if time >= start - 1e-9 && time <= end + 1e-9 {
                if let Some(s) = burn.state_at(time) {
                    return Some(s);
                }
            }
        }
        let (start, end) = self.coast_segment.epoch_range();
        if time >= start - 1e-9 && time <= end + 1e-9 {
            return self.coast_segment.state_at(time);
        }
        None
    }

    /// Whether the coast sub-segment terminated in a collision.
    pub fn has_collision(&self) -> bool {
        self.coast_segment.collision_body.is_some()
            || self
                .burn_segment
                .as_ref()
                .map(|s| s.collision_body.is_some())
                .unwrap_or(false)
    }
}

/// A propagated trajectory through a maneuver sequence.
///
/// `legs` is the canonical structure; `segments` is a flattened mirror kept
/// for consumers that don't care about leg boundaries.  `events` holds
/// low-level detections (SOI crossings, apsides, impacts); `encounters` are
/// richer SOI-window aggregates; `approaches` are geometric close-pass
/// records for bodies the trajectory doesn't enter.
#[derive(Debug, Clone)]
pub struct FlightPlan {
    pub initial_state: StateVector,
    pub initial_time: f64,
    pub legs: Vec<Leg>,
    pub segments: Vec<NumericSegment>,
    pub events: Vec<TrajectoryEvent>,
    pub encounters: Vec<Encounter>,
    pub approaches: Vec<ClosestApproach>,
    /// Full coast from initial state with no maneuvers applied.
    /// Present only when maneuver nodes exist, so the renderer can show
    /// the original orbit alongside the planned trajectory.
    pub baseline: Option<NumericSegment>,
}

impl FlightPlan {
    pub fn empty(initial_state: StateVector, initial_time: f64) -> Self {
        Self {
            initial_state,
            initial_time,
            legs: Vec::new(),
            segments: Vec::new(),
            events: Vec::new(),
            encounters: Vec::new(),
            approaches: Vec::new(),
            baseline: None,
        }
    }

    pub fn legs(&self) -> &[Leg] {
        &self.legs
    }

    pub fn segments(&self) -> &[NumericSegment] {
        &self.segments
    }

    pub fn events(&self) -> &[TrajectoryEvent] {
        &self.events
    }

    pub fn encounters(&self) -> &[Encounter] {
        &self.encounters
    }

    pub fn approaches(&self) -> &[ClosestApproach] {
        &self.approaches
    }

    /// Filter low-level events by kind.
    pub fn events_of<'a>(
        &'a self,
        kind: super::events::TrajectoryEventKind,
    ) -> impl Iterator<Item = &'a TrajectoryEvent> + 'a {
        self.events.iter().filter(move |e| e.kind == kind)
    }

    /// Which leg index contains the given time?
    pub fn leg_at_time(&self, time: f64) -> Option<usize> {
        for (i, leg) in self.legs.iter().enumerate() {
            let start = leg.leg_start_time();
            let end = leg.leg_end_time().unwrap_or(f64::MAX);
            if time >= start - 1e-6 && time <= end + 1e-6 {
                return Some(i);
            }
        }
        self.legs.len().checked_sub(1)
    }

    /// Ergonomic closest-approach search against a target body's ephemeris,
    /// using interpolation between stored samples for tighter precision than
    /// the bulk [`Self::approaches`] scan.
    pub fn closest_approach_to(
        &self,
        target: BodyId,
        ephemeris: &dyn BodyStateProvider,
    ) -> Option<ClosestApproach> {
        super::events::closest_approach(self, target, ephemeris)
    }

    /// Aggregated encounter for `body`, if one was detected.
    pub fn encounter_with(&self, body: BodyId) -> Option<&Encounter> {
        self.encounters.iter().find(|e| e.body == body)
    }
}

impl Trajectory for FlightPlan {
    fn state_at(&self, time: f64) -> Option<StateVector> {
        for leg in &self.legs {
            if let Some(s) = leg.state_at(time) {
                return Some(s);
            }
        }
        // Fall back to the flat segment list (legs are authoritative, but
        // a consumer may have mutated `segments` directly in legacy paths).
        for seg in &self.segments {
            let (start, end) = seg.epoch_range();
            if time >= start - 1e-9 && time <= end + 1e-9 {
                return seg.state_at(time);
            }
        }
        None
    }

    fn epoch_range(&self) -> (f64, f64) {
        let start = self
            .segments
            .iter()
            .find_map(|s| s.start_time())
            .unwrap_or(self.initial_time);
        let end = self
            .segments
            .iter()
            .rev()
            .find_map(|s| s.end_time())
            .unwrap_or(self.initial_time);
        (start, end)
    }

    fn anchor_body_at(&self, time: f64) -> Option<BodyId> {
        for seg in &self.segments {
            let (start, end) = seg.epoch_range();
            if time >= start && time <= end {
                return seg.anchor_body_at(time);
            }
        }
        None
    }
}

/// Build a flight plan by propagating through the maneuver sequence.
///
/// For each leg:
/// 1. If the leg starts with a burn of non-zero duration, propagate a burn
///    sub-segment with a `ManeuverThrustEffect` active in the registry.
/// 2. Propagate a coast sub-segment from the burn end (or leg start) to the
///    next leg's start time or the ephemeris horizon.
///
/// Stable-orbit detection fires only on coast sub-segments of the final leg.
/// Collision, cone fade, and budget exhaustion end propagation early.
pub fn propagate_flight_plan(
    request: &PredictionRequest,
    budget: Option<PropagationBudget>,
) -> FlightPlan {
    let PredictionRequest {
        ship_state: initial_state,
        sim_time: start_time,
        maneuvers,
        active_burns,
        ephemeris,
        bodies,
        prediction_config,
        integrator_config,
        ship_thrust_acceleration,
        target_body,
        ..
    } = request;
    let initial_state = *initial_state;
    let start_time = *start_time;
    let ship_thrust_acceleration = *ship_thrust_acceleration;
    let target_body = *target_body;

    let _span = tracing::info_span!(
        "propagate_flight_plan",
        budget = budget.map(|b| b.max_steps).unwrap_or(0),
        nodes = maneuvers.nodes.len(),
    )
    .entered();

    let ctx = PropagationContext {
        ephemeris: ephemeris.as_ref(),
        bodies,
        prediction_config,
        integrator_config: integrator_config.clone(),
        target_body,
    };

    let coast_registry = EffectRegistry::newtonian();

    let mut legs: Vec<Leg> = Vec::new();
    let mut segments: Vec<NumericSegment> = Vec::new();
    let mut events: Vec<TrajectoryEvent> = Vec::new();
    let mut id_counter: EncounterId = 0;
    let mut remaining_budget: Option<usize> = budget.map(|b| b.max_steps);

    // Sort node indices by time so out-of-order UI edits still propagate
    // correctly.
    let mut node_order: Vec<usize> = (0..maneuvers.nodes.len()).collect();
    node_order.sort_by(|&a, &b| {
        maneuvers.nodes[a]
            .time
            .partial_cmp(&maneuvers.nodes[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let ephemeris_end = start_time + ctx.ephemeris.time_span();
    let leg_count = node_order.len() + 1;

    // Baseline: full coast from initial state with no maneuvers.
    // Propagated with its own budget so it doesn't starve the main plan.
    let mut baseline = if !maneuvers.nodes.is_empty() {
        Some(propagate_segment(
            initial_state,
            start_time,
            ephemeris_end,
            &coast_registry,
            &ctx,
            true,
            &mut None,
        ))
    } else {
        None
    };

    // Walking state: where the next leg begins.
    let mut state = initial_state;
    let mut time = start_time;

    // Carry-over `active_burns` from the live sim (a burn that's already in
    // progress when prediction starts). These belong to leg 0.
    let carried_burns: Vec<ScheduledBurn> = active_burns.clone();

    for leg_idx in 0..leg_count {
        // Leg boundary: next node time or horizon.
        let mut leg_end = if leg_idx < node_order.len() {
            maneuvers.nodes[node_order[leg_idx]].time
        } else {
            ephemeris_end
        };

        // Build the burn for this leg.
        //
        // Leg 0: use any `carried_burns` that are still active at `time`
        //   (the live-sim in-progress burn).
        // Leg i>0: synthesize a new burn from the node that starts leg i.
        let burn_spec: Option<ManeuverThrustEffect> = if leg_idx == 0 {
            carried_burns
                .iter()
                .find(|b| b.start_time <= time && b.start_time + b.duration > time)
                .map(|b| {
                    // If a queued node falls inside this active burn, extend
                    // the leg so the burn completes before the next leg starts.
                    // Without this, the burn would be silently truncated and
                    // the predicted trajectory would under-apply delta-v.
                    let burn_end = b.start_time + b.duration;
                    if burn_end > leg_end {
                        leg_end = burn_end;
                    }
                    ManeuverThrustEffect::new(
                        b.delta_v_local,
                        b.reference_body,
                        b.acceleration,
                        b.start_time,
                        b.duration,
                    )
                })
        } else {
            let prev_node = &maneuvers.nodes[node_order[leg_idx - 1]];
            let duration = burn_duration(prev_node.delta_v.length(), ship_thrust_acceleration);
            if duration > 0.0 && prev_node.delta_v.length_squared() > 0.0 {
                Some(ManeuverThrustEffect::new(
                    prev_node.delta_v,
                    prev_node.reference_body,
                    ship_thrust_acceleration,
                    prev_node.time,
                    duration,
                ))
            } else {
                None
            }
        };

        let applied_delta_v: Option<DVec3> = if leg_idx == 0 {
            burn_spec.as_ref().map(|e| e.delta_v_local)
        } else {
            Some(maneuvers.nodes[node_order[leg_idx - 1]].delta_v)
        };

        let leg_start_state = state;
        let leg_start_time = time;

        // 1. Burn sub-segment.
        let burn_segment = if let Some(spec) = burn_spec {
            let burn_end = spec.end_time;
            let burn_stop = burn_end.min(leg_end);
            let burn_effect: Arc<dyn Effect> = Arc::new(spec);
            let burn_registry = EffectRegistry::with_effects(vec![burn_effect]);
            let seg = propagate_segment(
                state,
                time,
                burn_stop,
                &burn_registry,
                &ctx,
                false,
                &mut remaining_budget,
            );
            if let Some(last) = seg.last_state() {
                state = last;
            }
            if let Some(t) = seg.end_time() {
                time = t;
            }
            Some(seg)
        } else {
            None
        };

        // If the burn ate the whole leg (unusual) or ran into a collision,
        // short-circuit with an empty coast.
        let burn_collided = burn_segment
            .as_ref()
            .map(|s| s.collision_body.is_some())
            .unwrap_or(false);

        // 2. Coast sub-segment.
        let stop_on_stable_orbit = leg_idx + 1 == leg_count && !burn_collided;
        let coast_segment = if burn_collided {
            NumericSegment {
                samples: Vec::new(),
                is_stable_orbit: false,
                stable_orbit_start_index: None,
                collision_body: None,
            }
        } else {
            propagate_segment(
                state,
                time,
                leg_end,
                &coast_registry,
                &ctx,
                stop_on_stable_orbit,
                &mut remaining_budget,
            )
        };

        // Event detection runs over each sub-segment individually.
        if let Some(seg) = &burn_segment {
            events.extend(detect_segment_events(
                seg,
                bodies,
                ephemeris.as_ref(),
                &mut id_counter,
                leg_idx,
            ));
        }
        events.extend(detect_segment_events(
            &coast_segment,
            bodies,
            ephemeris.as_ref(),
            &mut id_counter,
            leg_idx,
        ));

        // Mirror into flat segments list (burn first if present, then coast).
        if let Some(seg) = &burn_segment {
            segments.push(seg.clone());
        }
        segments.push(coast_segment.clone());

        // Carry ship state forward for the next leg.
        let coast_collided = coast_segment.collision_body.is_some();
        let early_exit = burn_collided
            || coast_collided
            || remaining_budget == Some(0)
            || (stop_on_stable_orbit && coast_segment.is_stable_orbit);

        legs.push(Leg {
            start_state: leg_start_state,
            start_time: leg_start_time,
            applied_delta_v,
            burn_segment,
            coast_segment,
        });

        if early_exit {
            break;
        }

        // Derive the next leg's initial state by interpolating the
        // completed leg at the exact node time. This structurally
        // guarantees each leg starts where the previous one ends —
        // no drift from floating-point accumulation in (state, time).
        if leg_idx < node_order.len() {
            let node_time = maneuvers.nodes[node_order[leg_idx]].time;
            let built_leg = legs.last().unwrap();
            state = built_leg
                .state_at(node_time)
                .or_else(|| built_leg.last_state())
                .unwrap_or(state);
            time = node_time;
        }
    }

    // Lock the whole prediction to a single rendering anchor (the body
    // dominating at t=0). Per-leg anchors create frame discontinuities at
    // node boundaries — e.g. a capture-planning node near a flyby would
    // flip leg N+1 into the encounter body's frame, so the ghost and
    // adjacent legs could not all coincide. For intra-system trajectories
    // this collapses to "draw everything in the starting body's frame",
    // which matches the player's mental model of departing from home.
    let prediction_anchor = segments
        .iter()
        .flat_map(|s| s.samples.first())
        .next()
        .map(|s| s.anchor_body);
    if let Some(anchor) = prediction_anchor {
        for seg in segments.iter_mut() {
            for sample in seg.samples.iter_mut() {
                sample.anchor_body = anchor;
                sample.ref_pos = ephemeris.query_body(anchor, sample.time).position;
            }
        }
        for leg in legs.iter_mut() {
            if let Some(burn) = leg.burn_segment.as_mut() {
                for sample in burn.samples.iter_mut() {
                    sample.anchor_body = anchor;
                    sample.ref_pos = ephemeris.query_body(anchor, sample.time).position;
                }
            }
            for sample in leg.coast_segment.samples.iter_mut() {
                sample.anchor_body = anchor;
                sample.ref_pos = ephemeris.query_body(anchor, sample.time).position;
            }
        }
        if let Some(base) = baseline.as_mut() {
            for sample in base.samples.iter_mut() {
                sample.anchor_body = anchor;
                sample.ref_pos = ephemeris.query_body(anchor, sample.time).position;
            }
        }
    }

    // Aggregate SOI-window encounters from the low-level events.
    let encounters = aggregate_encounters(
        &events,
        &segments,
        bodies,
        ephemeris.as_ref(),
        &mut id_counter,
    );

    // Scan every segment sample for closest approaches to bodies the
    // trajectory does *not* enter (SOI entries already become encounters).
    let encounter_bodies: std::collections::HashSet<BodyId> =
        encounters.iter().map(|e| e.body).collect();
    let approaches =
        scan_closest_approaches(&segments, bodies, ephemeris.as_ref(), &encounter_bodies);

    FlightPlan {
        initial_state,
        initial_time: start_time,
        legs,
        segments,
        events,
        encounters,
        approaches,
        baseline,
    }
}

