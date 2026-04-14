//! [`FlightPlan`]: the chained result of propagating through a maneuver sequence.
//!
//! A flight plan holds one [`NumericSegment`] per leg (pre-first-node, between
//! nodes, post-last-node) and a flat list of detected [`Encounter`]s. It
//! implements [`Trajectory`] by dispatching `state_at` / `anchor_body_at`
//! to the segment containing the query time.
//!
//! Build a flight plan by calling [`propagate_flight_plan`]; recompute from
//! an edited node with [`FlightPlan::recompute_from`] (the latter is currently
//! equivalent to a full rebuild — retained as the abstraction boundary for a
//! future incremental recompute).

use std::sync::Arc;

use super::Trajectory;
use super::events::{Encounter, EncounterId, detect_segment_events};
use super::numeric::NumericSegment;
pub use super::propagation::ScheduledBurn;
use super::propagation::{
    PredictionConfig, PropagationBudget, PropagationContext, cone_width_scaled, propagate_segment,
};
use crate::body_state_provider::BodyStateProvider;
use crate::integrator::IntegratorConfig;
use crate::maneuver::{ManeuverSequence, burn_duration};
use crate::types::{BodyDefinition, BodyId, StateVector};

/// Immutable inputs for trajectory prediction.
///
/// This can be sent to a worker thread so prediction can run without blocking
/// the main simulation/update loop. Lives next to [`propagate_flight_plan`]
/// which consumes it.
#[derive(Clone)]
pub struct PredictionRequest {
    pub epoch: u64,
    pub ship_state: StateVector,
    pub sim_time: f64,
    pub maneuvers: ManeuverSequence,
    pub active_burns: Vec<ScheduledBurn>,
    pub ephemeris: Arc<dyn BodyStateProvider>,
    pub bodies: Vec<BodyDefinition>,
    pub prediction_config: PredictionConfig,
    pub integrator_config: IntegratorConfig,
    pub ship_thrust_acceleration: f64,
}

/// A propagated trajectory through a maneuver sequence.
#[derive(Debug, Clone)]
pub struct FlightPlan {
    pub initial_state: StateVector,
    pub initial_time: f64,
    pub segments: Vec<NumericSegment>,
    pub encounters: Vec<Encounter>,
}

impl FlightPlan {
    pub fn empty(initial_state: StateVector, initial_time: f64) -> Self {
        Self {
            initial_state,
            initial_time,
            segments: Vec::new(),
            encounters: Vec::new(),
        }
    }

    pub fn segments(&self) -> &[NumericSegment] {
        &self.segments
    }

    pub fn encounters(&self) -> &[Encounter] {
        &self.encounters
    }

    /// Filter encounters by kind.
    pub fn encounters_of<'a>(
        &'a self,
        kind: super::events::EncounterKind,
    ) -> impl Iterator<Item = &'a Encounter> + 'a {
        self.encounters.iter().filter(move |e| e.kind == kind)
    }

    /// Ergonomic closest-approach search against a target body's ephemeris.
    pub fn closest_approach_to(
        &self,
        target: BodyId,
        ephemeris: &dyn BodyStateProvider,
    ) -> Option<Encounter> {
        super::events::closest_approach(self, target, ephemeris)
    }

    /// Recompute the flight plan from `node_index` onward.
    ///
    /// For now this rebuilds the full plan from the stored initial state — it
    /// is kept as the abstraction boundary so a future implementation can
    /// reuse leading segments without rebuilding the whole thing.
    pub fn recompute_from(
        &mut self,
        _node_index: usize,
        request: &PredictionRequest,
        budget: Option<PropagationBudget>,
    ) {
        // Rebuild from the plan's own starting state; `request` supplies
        // everything else (bodies, ephemeris, integrator config, maneuvers).
        let mut local_request = request.clone();
        local_request.ship_state = self.initial_state;
        local_request.sim_time = self.initial_time;
        local_request.active_burns = Vec::new();
        let rebuilt = propagate_flight_plan(&local_request, budget);
        self.segments = rebuilt.segments;
        self.encounters = rebuilt.encounters;
    }
}

impl Trajectory for FlightPlan {
    fn state_at(&self, time: f64) -> Option<StateVector> {
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
/// One segment per leg: `start_time → node_0`, `node_0 → node_1`, …,
/// `node_last → end_of_horizon`. Propagation stops early on collision, stable
/// orbit, cone fade, or budget exhaustion.
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
        ..
    } = request;
    let initial_state = *initial_state;
    let start_time = *start_time;
    let ship_thrust_acceleration = *ship_thrust_acceleration;

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
    };

    let mut segments: Vec<NumericSegment> = Vec::new();
    let mut encounters: Vec<Encounter> = Vec::new();
    let mut encounter_counter: EncounterId = 0;

    let mut state = initial_state;
    let mut time = start_time;
    let mut remaining_budget: Option<usize> = budget.map(|b| b.max_steps);
    let mut active_burns: Vec<ScheduledBurn> = active_burns.clone();

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

    for leg_idx in 0..leg_count {
        let leg_end = if leg_idx < node_order.len() {
            maneuvers.nodes[node_order[leg_idx]].time
        } else {
            ephemeris_end
        };
        let stop_on_stable_orbit = leg_idx + 1 == leg_count;

        active_burns.retain(|burn| burn.start_time + burn.duration > time);
        let segment = propagate_segment(
            state,
            time,
            leg_end,
            &active_burns,
            &ctx,
            stop_on_stable_orbit,
            &mut remaining_budget,
        );

        // Event detection on the fresh segment.
        encounters.extend(detect_segment_events(
            &segment,
            bodies,
            ephemeris.as_ref(),
            &mut encounter_counter,
        ));

        let early_exit = segment.collision_body.is_some()
            || segment
                .samples
                .last()
                .map(|s| {
                    cone_width_scaled(s, ctx.prediction_config)
                        > ctx.prediction_config.cone_fade_threshold
                })
                .unwrap_or(false)
            || remaining_budget == Some(0)
            || (stop_on_stable_orbit && segment.is_stable_orbit);

        // Carry the last known state forward so the next leg starts correctly.
        if let Some(last) = segment.last_state() {
            state = last;
        }
        if let Some(t) = segment.end_time() {
            time = t;
        }

        segments.push(segment);

        if early_exit {
            break;
        }

        // Schedule the burn at the node boundary.  The local-frame Δv is
        // stored on the burn and converted to world each integrator substep,
        // so long burns track the rotating orbital frame.
        if leg_idx < node_order.len() {
            let node = &maneuvers.nodes[node_order[leg_idx]];
            time = node.time;

            let duration = burn_duration(node.delta_v.length(), ship_thrust_acceleration);
            if duration > 0.0 && node.delta_v.length_squared() > 0.0 {
                active_burns.push(ScheduledBurn {
                    delta_v_local: node.delta_v,
                    reference_body: node.reference_body,
                    acceleration: ship_thrust_acceleration,
                    start_time: node.time,
                    duration,
                });
            }
        }
    }

    FlightPlan {
        initial_state,
        initial_time: start_time,
        segments,
        encounters,
    }
}
