//! [`FlightPlan`]: a leg-based propagation through a maneuver sequence.
//!
//! A flight plan is a chain of [`Leg`]s. Each leg starts at a known ship
//! state and epoch, optionally applies a finite burn, and then coasts until
//! the next leg's start time (the next maneuver) or the ephemeris horizon.
//!
//! Burn and coast are modelled as separate `NumericSegment`s so they can be
//! propagated independently:
//!
//! - **`burn_segment`** — propagated with RK4 substeps under SOI-body gravity
//!   plus constant-acceleration thrust in the ship's local frame. Time span
//!   `[burn_start, burn_end)`.
//! - **`coast_segment`** — propagated analytically via Kepler in the SOI
//!   body's frame, SOI transitions detected and refined by root-finding.
//!   Time span `[burn_end, leg_end)`.
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
use crate::maneuver::{ManeuverSequence, burn_duration};
use crate::orbital_math::propagate_kepler;
use crate::ship_propagator::ShipPropagator;
use crate::types::{BodyDefinition, BodyId, StateVector, TrajectorySample};
use glam::DVec3;

/// Immutable inputs for trajectory prediction.
#[derive(Clone)]
pub struct PredictionRequest {
    pub ship_state: StateVector,
    pub sim_time: f64,
    pub maneuvers: ManeuverSequence,
    pub ephemeris: Arc<dyn BodyStateProvider>,
    /// Same propagator the live simulation is stepping with. Cloned (`Arc`)
    /// from [`crate::simulation::Simulation`] when the request is built so
    /// prediction and live ship motion route through one instance.
    pub propagator: Arc<dyn ShipPropagator>,
    pub bodies: Vec<BodyDefinition>,
    pub prediction_config: PredictionConfig,
    /// Total engine thrust in newtons.
    pub ship_thrust_n: f64,
    /// Ship mass at `sim_time`. Prediction tracks how this evolves through
    /// each scheduled burn so the rocket equation stays honored across
    /// long flight plans.
    pub ship_mass_kg: f64,
    /// Engine mass flow at full throttle, kg/s.
    pub ship_mass_flow_kg_per_s: f64,
    /// Dry mass — the floor at which propellant exhausts and thrust cuts
    /// off. Predictions cap each burn at the time it would drain to this
    /// floor.
    pub ship_dry_mass_kg: f64,
    /// Currently selected target body. Informational only — analytical
    /// propagation needs no step-size bias because Kepler has no step size.
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
        self.burn_segment
            .iter()
            .chain(std::iter::once(&self.coast_segment))
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

    /// State at `time` on the trajectory the ship would follow if the burn
    /// scheduled at `time` were not applied.
    ///
    /// Burns are centered on `node.time` (`[t − d/2, t + d/2]`), so for a
    /// finite burn the burn segment owns `t` and `Trajectory::state_at`
    /// returns a partially-thrusted state — wrong for any consumer that
    /// wants the unperturbed orbit (maneuver-marker rendering, sensitivity
    /// scaling, drag-frame computation). This walks the legs to find the
    /// pre-burn coast and analytically extends it to `time` under the SOI
    /// body's gravity. Impulsive burns are a no-op (coast already covers
    /// `t`).
    ///
    /// Returns `None` if the gap between the previous coast's end and
    /// `time` exceeds the burn's own duration — the symptom of an SOI
    /// transition cutting the coast short, where the anchor body of the
    /// last sample may no longer be the SOI for `time` and analytical
    /// extrapolation would be wrong.
    pub fn pre_burn_state_at(
        &self,
        time: f64,
        ephemeris: &dyn BodyStateProvider,
        bodies: &[BodyDefinition],
    ) -> Option<TrajectorySample> {
        // Coast directly contains `time`: impulsive-burn case, or `time`
        // is between bursts. Sample directly.
        for leg in &self.legs {
            let coast = &leg.coast_segment;
            let (Some(s), Some(e)) = (coast.start_time(), coast.end_time()) else {
                continue;
            };
            if time >= s - 1e-6 && time <= e + 1e-6 {
                let state = coast.state_at(time)?;
                let anchor = coast.samples.last()?.anchor_body;
                let body_state = ephemeris.query_body(anchor, time);
                return Some(TrajectorySample {
                    time,
                    position: state.position,
                    velocity: state.velocity,
                    anchor_body: anchor,
                    ref_pos: body_state.position,
                });
            }
        }

        // `time` lives inside a burn — Kepler-extrapolate from the
        // previous leg's coast end under that leg's SOI body.
        for (i, leg) in self.legs.iter().enumerate() {
            let Some(burn) = &leg.burn_segment else {
                continue;
            };
            let (bs, be) = burn.epoch_range();
            if time < bs - 1e-6 || time > be + 1e-6 {
                continue;
            }

            let pre_burn_leg = i.checked_sub(1).and_then(|p| self.legs.get(p))?;
            let last_sample = pre_burn_leg.coast_segment.samples.last()?;

            let dt = time - last_sample.time;
            let burn_dur = (be - bs).max(0.0);
            if dt < -1e-6 || dt > burn_dur + 1e-3 {
                return None;
            }

            let anchor = last_sample.anchor_body;
            let body = bodies.get(anchor)?;
            if body.gm <= 0.0 {
                return None;
            }

            let body_state_last = ephemeris.query_body(anchor, last_sample.time);
            let body_state_now = ephemeris.query_body(anchor, time);

            let rel = StateVector {
                position: last_sample.position - body_state_last.position,
                velocity: last_sample.velocity - body_state_last.velocity,
            };
            let advanced = propagate_kepler(rel, body.gm, dt);

            return Some(TrajectorySample {
                time,
                position: advanced.position + body_state_now.position,
                velocity: advanced.velocity + body_state_now.velocity,
                anchor_body: anchor,
                ref_pos: body_state_now.position,
            });
        }

        None
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
    _budget: Option<PropagationBudget>,
) -> FlightPlan {
    let PredictionRequest {
        ship_state: initial_state,
        sim_time: start_time,
        maneuvers,
        ephemeris,
        propagator,
        bodies,
        prediction_config,
        ship_thrust_n,
        ship_mass_kg,
        ship_mass_flow_kg_per_s,
        ship_dry_mass_kg,
        target_body,
        ..
    } = request;
    let initial_state = *initial_state;
    let start_time = *start_time;
    let ship_thrust_n = *ship_thrust_n;
    let ship_mass_flow_kg_per_s = *ship_mass_flow_kg_per_s;
    let ship_dry_mass_kg = *ship_dry_mass_kg;
    let mut running_mass_kg = *ship_mass_kg;
    let target_body = *target_body;

    let _span =
        tracing::info_span!("propagate_flight_plan", nodes = maneuvers.nodes.len(),).entered();

    let _ = target_body; // retained on PredictionRequest for UI; unused here
    let ctx = PropagationContext {
        ephemeris: ephemeris.as_ref(),
        bodies,
        prediction_config,
        propagator: propagator.as_ref(),
    };

    let mut legs: Vec<Leg> = Vec::new();
    let mut segments: Vec<NumericSegment> = Vec::new();
    let mut events: Vec<TrajectoryEvent> = Vec::new();
    let mut id_counter: EncounterId = 0;

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

    // Baseline: full coast from initial state with no maneuvers. Cheap in
    // analytical propagation, so we always produce it when maneuvers exist.
    let mut baseline = if !maneuvers.nodes.is_empty() {
        Some(propagate_segment(
            initial_state,
            start_time,
            ephemeris_end,
            None,
            &ctx,
            true,
        ))
    } else {
        None
    };

    // Walking state: where the next leg begins.
    let mut state = initial_state;
    let mut time = start_time;

    // Burns are centered on the node time: a node at `t` with computed
    // duration `d` runs across `[t − d/2, t + d/2]`. Two consequences for
    // the leg structure:
    //
    // * Leg `i` (for `i ≥ 1`) starts at `node[i−1].time − d_{i−1}/2`,
    //   integrates that burn, then coasts until the next leg's burn
    //   starts.
    // * Leg `i` therefore ends at `node[i].time − d_i/2`. Computing
    //   `d_i` requires the ship mass at the start of node `i`'s burn,
    //   which is the running mass after leg `i`'s burn has drained — so
    //   we pre-compute this leg's burn duration from the entry mass,
    //   subtract its drain to estimate the next-burn anchor mass, and
    //   use that to size `d_i`. Exact: the running coast doesn't change
    //   mass, so the post-this-burn mass is also the next-burn anchor.
    //
    // Clamp: when a node sits closer to `sim_time` than half its own
    // duration, `node.time − d/2 < time` and the unclamped formula
    // would integrate a burn whose mass anchor predates `time`. Clamp
    // `start_time` to `time` (and the leg ends to `time` too) so the
    // burn integrates forward from the current state and the rocket-
    // equation mass at integration time matches `running_mass_kg`. The
    // burn-execution autopilot applies the same clamp when deciding
    // when to open the throttle, so prediction and live execution
    // agree on the burn window.

    for leg_idx in 0..leg_count {
        // Duration of the burn that fires at the START of this leg
        // (i.e. the burn for `node[leg_idx − 1]`). Zero for leg 0.
        let this_leg_burn_duration: f64 = if leg_idx == 0 {
            0.0
        } else {
            let prev_node = &maneuvers.nodes[node_order[leg_idx - 1]];
            if prev_node.delta_v.length_squared() > 0.0 {
                burn_duration(
                    prev_node.delta_v.length(),
                    ship_thrust_n,
                    running_mass_kg,
                    ship_mass_flow_kg_per_s,
                    ship_dry_mass_kg,
                )
            } else {
                0.0
            }
        };

        // Mass after this leg's burn drain — the anchor for sizing the
        // NEXT leg's burn (no drain during the coast that follows).
        let mass_after_this_burn = (running_mass_kg
            - ship_mass_flow_kg_per_s * this_leg_burn_duration)
            .max(ship_dry_mass_kg);

        // Where this leg's coast ends = where the NEXT leg's burn
        // starts. For the last leg, run to the ephemeris horizon.
        let leg_end = if leg_idx < node_order.len() {
            let next_node = &maneuvers.nodes[node_order[leg_idx]];
            let next_duration = if next_node.delta_v.length_squared() > 0.0 {
                burn_duration(
                    next_node.delta_v.length(),
                    ship_thrust_n,
                    mass_after_this_burn,
                    ship_mass_flow_kg_per_s,
                    ship_dry_mass_kg,
                )
            } else {
                0.0
            };
            // Clamp at `time` so a near-now node that would otherwise
            // produce `leg_end < time` doesn't crash the propagator.
            (next_node.time - next_duration / 2.0).max(time)
        } else {
            ephemeris_end
        };

        let scheduled_burn: Option<ScheduledBurn> = if this_leg_burn_duration > 0.0 {
            let prev_node = &maneuvers.nodes[node_order[leg_idx - 1]];
            // Clamp burn start to the leg's entry time. With centered
            // burns this is the natural value (= `prev_node.time − d/2`)
            // unless the node sits inside the past half-window — in
            // which case the live autopilot would also start the burn
            // immediately, with its own delivered-Δv tracking handling
            // the asymmetric loss.
            let raw_start = prev_node.time - this_leg_burn_duration / 2.0;
            let clamped_start = raw_start.max(time);
            Some(ScheduledBurn {
                delta_v_local: prev_node.delta_v,
                reference_body: prev_node.reference_body,
                thrust_n: ship_thrust_n,
                initial_mass_kg: running_mass_kg,
                mass_flow_kg_per_s: ship_mass_flow_kg_per_s,
                dry_mass_kg: ship_dry_mass_kg,
                start_time: clamped_start,
                duration: this_leg_burn_duration,
            })
        } else {
            None
        };

        let applied_delta_v: Option<DVec3> = if leg_idx == 0 {
            None
        } else {
            Some(maneuvers.nodes[node_order[leg_idx - 1]].delta_v)
        };

        let leg_start_state = state;
        let leg_start_time = time;

        // 1. Burn sub-segment.
        let burn_segment = if let Some(b) = scheduled_burn {
            let burn_end = b.start_time + b.duration;
            let burn_stop = burn_end.min(leg_end);
            let seg = propagate_segment(state, time, burn_stop, Some(b), &ctx, false);
            if let Some(last) = seg.last_state() {
                state = last;
            }
            if let Some(t) = seg.end_time() {
                // The burn segment integrates thrust across
                // `[leg_start_time, t]` (after start-time clamping —
                // the burn always begins at the leg's entry time), so
                // `t − leg_start_time` is the active-burn duration.
                let burn_dt = (t - leg_start_time).max(0.0);
                running_mass_kg =
                    (running_mass_kg - ship_mass_flow_kg_per_s * burn_dt).max(ship_dry_mass_kg);
                time = t;
            }
            Some(seg)
        } else {
            None
        };

        // If the burn ran into a collision, short-circuit with an empty coast.
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
            propagate_segment(state, time, leg_end, None, &ctx, stop_on_stable_orbit)
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

        let coast_collided = coast_segment.collision_body.is_some();
        let early_exit = burn_collided
            || coast_collided
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

        // Derive the next leg's initial (state, time) pair from the
        // completed leg. They MUST stay aligned: the next leg's burn is
        // integrated from this state at this time, so any mismatch shows
        // up as a translated post-burn trajectory.
        //
        // For impulsive burns (d = 0) the previous coast covers the
        // node exactly; `state_at(node_time)` interpolates and we
        // advance time to the node.
        //
        // For finite burns (d > 0) the previous coast ends at
        // `node_time − d/2`; `state_at` returns None. We must then
        // anchor *both* state and time to the coast's actual end —
        // setting time to `node_time` while state is still at
        // `node_time − d/2` desyncs the integrator (it would treat the
        // pre-burn position as living at `node_time`, which displaces
        // every downstream sample by the body's heliocentric motion
        // across `d/2`). The clamp on the next leg's `burn.start_time`
        // then naturally produces a burn centered on the node time.
        if leg_idx < node_order.len() {
            let node_time = maneuvers.nodes[node_order[leg_idx]].time;
            let built_leg = legs.last().unwrap();
            if let Some(s) = built_leg.state_at(node_time) {
                state = s;
                time = node_time;
            } else if let (Some(s), Some(end_t)) =
                (built_leg.last_state(), built_leg.leg_end_time())
            {
                state = s;
                time = end_t;
            }
        }
    }

    // Per-leg anchor relock.
    //
    // The propagator emits samples whose `anchor_body` / `ref_pos` match the
    // SOI actually containing the ship at each sample. That is physically
    // correct but visually wrong: the renderer pins each anchor to its
    // *current* world position, so when a leg crosses an SOI boundary the
    // trajectory jumps by the anchor's intervening motion (Thalos moves a
    // lot over a year-long prediction — the inside-SOI and outside-SOI
    // segments no longer meet in the rendered frame).
    //
    // We therefore relock each leg to the SOI body of its first sample, so
    // the whole leg is drawn in one consistent frame. Distinct legs
    // (separated by maneuver nodes) can still differ — a capture-planning
    // node at an encounter ghost produces a downstream leg whose first
    // sample is in that moon's SOI, so that leg renders in the moon's
    // frame and pins to the ghost, which is exactly the encounter UX
    // described in §6.
    let relock_samples_to = |samples: &mut [crate::types::TrajectorySample], anchor: BodyId| {
        for sample in samples.iter_mut() {
            sample.anchor_body = anchor;
            sample.ref_pos = ephemeris.query_body(anchor, sample.time).position;
        }
    };
    let relock_samples_to_first = |samples: &mut [crate::types::TrajectorySample]| {
        let Some(first) = samples.first() else {
            return;
        };
        let anchor = first.anchor_body;
        relock_samples_to(samples, anchor);
    };
    for leg in legs.iter_mut() {
        // The leg's first sample is the burn's first when a burn exists,
        // otherwise the coast's first. Both burn and coast get relocked
        // to that single anchor so the whole leg renders in one frame —
        // matching the comment above. Per-segment relock would split a
        // burn-crosses-SOI leg across two frames at burn-end, which
        // shows up as a visible bridge gap in the rendered trajectory.
        let leg_anchor = leg
            .burn_segment
            .as_ref()
            .and_then(|s| s.samples.first())
            .or_else(|| leg.coast_segment.samples.first())
            .map(|s| s.anchor_body);
        if let Some(anchor) = leg_anchor {
            if let Some(burn) = leg.burn_segment.as_mut() {
                relock_samples_to(&mut burn.samples, anchor);
            }
            relock_samples_to(&mut leg.coast_segment.samples, anchor);
        }
    }
    // Build the flat segments cache from the post-relock legs in one pass.
    // Event detection (which needs the *pre-relock* per-sample anchors to
    // see SOI transitions) ran above against the leg sub-segments directly,
    // so this cache is consumed only by post-relock readers — renderer,
    // closest-approach scan, encounter aggregation.
    for leg in legs.iter() {
        if let Some(burn) = &leg.burn_segment {
            segments.push(burn.clone());
        }
        segments.push(leg.coast_segment.clone());
    }
    if let Some(base) = baseline.as_mut() {
        relock_samples_to_first(&mut base.samples);
    }

    let encounters = aggregate_encounters(
        &events,
        &segments,
        bodies,
        ephemeris.as_ref(),
        &mut id_counter,
    );

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
