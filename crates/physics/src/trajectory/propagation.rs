//! Propagation that produces [`NumericSegment`]s using whichever
//! [`ShipPropagator`] the caller supplies.
//!
//! A "segment" in the old N-body world was one integrator run terminating on
//! a handful of conditions. With analytical Kepler propagation, each call to
//! the propagator terminates on the first of: target time, SOI entry, SOI
//! exit, collision, stable-orbit closure, or burn end. This module loops the
//! propagator across SOI transitions inside a single leg so the caller
//! ([`super::flight_plan`]) still sees one [`NumericSegment`] per sub-leg
//! (burn vs. coast) regardless of how many SOI boundaries the ship crosses.
//!
//! The propagator is supplied by the caller via [`PropagationContext`] —
//! the same `Arc<dyn ShipPropagator>` the live [`crate::simulation::Simulation`]
//! is stepping with, so live and predicted motion cannot diverge.

use super::numeric::NumericSegment;
use crate::body_state_provider::BodyStateProvider;
use crate::ship_propagator::{BurnParams, BurnRequest, CoastRequest, SegmentTerminator, ShipPropagator};
use crate::types::{BodyDefinition, BodyId, StateVector, TrajectorySample};
use glam::DVec3;

/// Tuning for trajectory prediction.
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    /// Hint for samples per coast segment per SOI span. More samples = smoother
    /// rendered curve at some memory cost. Default 128.
    pub coast_samples_per_segment: usize,
    /// **Deprecated, no effect.** Burn substep now lives on the propagator
    /// itself ([`crate::ship_propagator::KeplerianPropagator::burn_substep_s`])
    /// since prediction borrows the live simulation's propagator instead of
    /// constructing its own. Slated for removal in a follow-up cleanup.
    pub burn_substep_s: f64,
    /// Hard cap on SOI transitions within a single propagate_segment call.
    /// Guards pathological inputs; in practice we see 0-3 crossings per leg.
    /// Default 16.
    pub max_soi_transitions: usize,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            coast_samples_per_segment: 128,
            burn_substep_s: 1.0,
            max_soi_transitions: 16,
        }
    }
}

/// Placeholder for back-compat with the integrator-era progressive-refinement
/// path. Analytical propagation is cheap enough that a coarse pass is no
/// longer needed — this type is kept only so downstream call sites compile.
#[derive(Debug, Clone, Copy)]
pub struct PropagationBudget {
    pub max_steps: usize,
}

impl PropagationBudget {
    pub fn new(max_steps: usize) -> Self {
        Self { max_steps }
    }
}

/// Shared inputs for one propagation call.
pub(super) struct PropagationContext<'a> {
    pub ephemeris: &'a dyn BodyStateProvider,
    pub bodies: &'a [BodyDefinition],
    pub prediction_config: &'a PredictionConfig,
    /// Same instance the live [`crate::simulation::Simulation`] is stepping
    /// with — borrowed here so prediction and live propagation cannot
    /// numerically diverge.
    pub propagator: &'a dyn ShipPropagator,
}

/// Finite-duration maneuver burn. Turned into a [`BurnParams`] at propagation
/// time so the engine can thread the thrust direction through each RK4 stage.
///
/// `initial_mass_kg` is the ship mass at `start_time` — captured once when
/// the burn is scheduled, not re-derived later. The propagator computes
/// mass at any point inside `[start_time, start_time + duration]` from
/// this anchor, so a mid-burn prediction rebuild produces the same
/// trajectory as if it had run uninterrupted. `dry_mass_kg` is the floor
/// at which thrust cuts off (propellant exhausted).
#[derive(Debug, Clone, Copy)]
pub struct ScheduledBurn {
    pub delta_v_local: DVec3,
    pub reference_body: BodyId,
    pub thrust_n: f64,
    pub initial_mass_kg: f64,
    pub mass_flow_kg_per_s: f64,
    pub dry_mass_kg: f64,
    pub start_time: f64,
    pub duration: f64,
}

impl ScheduledBurn {
    pub(crate) fn to_burn_params(self) -> BurnParams {
        BurnParams {
            delta_v_local: self.delta_v_local,
            reference_body: self.reference_body,
            thrust_n: self.thrust_n,
            initial_mass_kg: self.initial_mass_kg,
            mass_flow_kg_per_s: self.mass_flow_kg_per_s,
            dry_mass_kg: self.dry_mass_kg,
            start_time: self.start_time,
            end_time: self.start_time + self.duration,
        }
    }
}

/// Propagate one sub-leg (burn XOR coast) across whatever SOIs the ship
/// traverses in the interval `[start_time, end_time]`. The resulting
/// [`NumericSegment`] holds the flattened sample sequence; SOI transitions
/// manifest as anchor-body changes between consecutive samples.
pub(super) fn propagate_segment(
    initial_state: StateVector,
    start_time: f64,
    end_time: f64,
    burn: Option<ScheduledBurn>,
    ctx: &PropagationContext,
    stop_on_stable_orbit: bool,
) -> NumericSegment {
    let propagator = ctx.propagator;

    let mut samples: Vec<TrajectorySample> = Vec::new();
    let mut state = initial_state;
    let mut time = start_time;
    let mut collision_body: Option<BodyId> = None;
    let mut is_stable_orbit = false;
    let mut stable_orbit_start_index: Option<usize> = None;
    let mut transitions = 0usize;

    while time < end_time {
        if transitions > ctx.prediction_config.max_soi_transitions {
            break;
        }
        let soi_body = propagator.soi_body_of(state.position, time, ctx.ephemeris, ctx.bodies);

        // The burn (if any) is active on [burn.start, burn.end). A sub-call
        // to burn_segment advances at most to the burn window's end; after
        // that we fall through to coast.
        let burn_active = burn
            .map(|b| time >= b.start_time && time < b.start_time + b.duration)
            .unwrap_or(false);

        let result = if burn_active {
            let b = burn.unwrap();
            let burn_end = b.start_time + b.duration;
            let segment_target = end_time.min(burn_end);
            propagator.burn_segment(BurnRequest {
                state,
                time,
                soi_body,
                target_time: segment_target,
                burn: b.to_burn_params(),
                ephemeris: ctx.ephemeris,
                bodies: ctx.bodies,
            })
        } else {
            // If a burn is scheduled later in the window, clip coast at its start.
            let coast_target = burn
                .filter(|b| b.start_time > time && b.start_time < end_time)
                .map(|b| b.start_time)
                .unwrap_or(end_time);
            propagator.coast_segment(CoastRequest {
                state,
                time,
                soi_body,
                target_time: coast_target,
                stop_on_stable_orbit,
                sample_count_hint: ctx.prediction_config.coast_samples_per_segment,
                ephemeris: ctx.ephemeris,
                bodies: ctx.bodies,
            })
        };

        // Concatenate samples. Every call after the first overlaps the
        // previous terminator sample, so drop the leading duplicate.
        if samples.is_empty() {
            samples.extend(result.samples);
        } else {
            samples.extend(result.samples.into_iter().skip(1));
        }

        state = result.end_state;
        time = result.end_time;

        match result.terminator {
            SegmentTerminator::Horizon | SegmentTerminator::BurnEnd { .. } => {
                break;
            }
            SegmentTerminator::StableOrbit => {
                is_stable_orbit = true;
                stable_orbit_start_index = Some(0);
                break;
            }
            SegmentTerminator::Collision { body, .. } => {
                collision_body = Some(body);
                break;
            }
            SegmentTerminator::SoiEnter { .. } | SegmentTerminator::SoiExit { .. } => {
                transitions += 1;
                // Continue: next iteration picks up the new SOI body.
            }
        }
    }

    NumericSegment {
        samples,
        is_stable_orbit,
        stable_orbit_start_index,
        collision_body,
    }
}

