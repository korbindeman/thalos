//! Central simulation struct that owns all physics state and logic.
//!
//! The game crate becomes a thin consumer: it calls [`Simulation::step`] each
//! frame and reads back the ship state, prediction, and warp info. Live
//! stepping and trajectory prediction both route through the same
//! [`ShipPropagator`], so "where the ship is" and "where it is predicted to
//! be" cannot drift numerically.

use std::sync::Arc;

use glam::DVec3;

use crate::body_state_provider::BodyStateProvider;
use crate::maneuver::{ManeuverSequence, burn_duration};
use crate::ship_propagator::{
    BurnParams, BurnRequest, CoastRequest, KeplerianPropagator, SegmentTerminator, ShipPropagator,
};
use crate::trajectory::{
    FlightPlan, PredictionConfig, PredictionRequest, ScheduledBurn, propagate_flight_plan,
};
use crate::types::{BodyDefinition, BodyId, BodyKind, StateVector};

/// Forward-looking orbit trail for a single body, relative to its parent.
#[derive(Debug, Clone)]
pub struct BodyOrbitTrail {
    pub body_id: BodyId,
    pub parent_id: BodyId,
    /// Positions in metres, relative to the parent body.
    pub points: Vec<DVec3>,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub prediction_config: PredictionConfig,
    pub warp_levels: Vec<f64>,
    /// Hard cap on a single frame's wall-clock delta, seconds. Keeps a
    /// render stall from advancing sim-time by minutes on the next frame.
    pub max_real_delta: f64,
    /// Recompute the cached prediction when it becomes this stale, seconds
    /// of sim time. Keeps the drawn trail from drifting off the live ship.
    pub prediction_stale_after: f64,
    /// Hard cap on SOI transitions (including burn endpoints) processed in a
    /// single `step()` call. Infinite-loop guard; in practice even long
    /// warp ticks traverse at most a handful.
    pub max_transitions_per_frame: u32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            prediction_config: PredictionConfig::default(),
            warp_levels: vec![
                0.0,
                1.0,
                10.0,
                100.0,
                1_000.0,
                10_000.0,
                100_000.0,
                1_000_000.0,
                10_000_000.0,
            ],
            max_real_delta: 0.1,
            prediction_stale_after: 30.0,
            max_transitions_per_frame: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// Warp controller
// ---------------------------------------------------------------------------

pub struct WarpController {
    level_index: usize,
    levels: Vec<f64>,
    resume_index: Option<usize>,
}

impl WarpController {
    fn new(levels: Vec<f64>) -> Self {
        Self {
            level_index: 0,
            levels,
            resume_index: None,
        }
    }

    pub fn speed(&self) -> f64 {
        self.levels[self.level_index]
    }

    pub fn label(&self) -> String {
        let speed = self.speed();
        if speed == 0.0 {
            "PAUSED".to_string()
        } else if speed >= 1_000_000.0 {
            format!("{:.0}M\u{00d7}", speed / 1_000_000.0)
        } else if speed >= 1000.0 {
            format!("{:.0}k\u{00d7}", speed / 1000.0)
        } else {
            format!("{:.0}\u{00d7}", speed)
        }
    }

    pub fn increase(&mut self) {
        self.resume_index = None;
        self.level_index = (self.level_index + 1).min(self.levels.len() - 1);
    }

    pub fn decrease(&mut self) {
        self.resume_index = None;
        self.level_index = self.level_index.saturating_sub(1);
    }

    pub fn reset(&mut self) {
        self.resume_index = None;
        self.level_index = self.levels.iter().position(|&w| w == 1.0).unwrap_or(0);
    }

    pub fn toggle_pause(&mut self) {
        if self.level_index == 0 {
            let target = self
                .resume_index
                .take()
                .unwrap_or_else(|| self.levels.iter().position(|&w| w == 1.0).unwrap_or(0));
            self.level_index = target;
        } else {
            self.resume_index = Some(self.level_index);
            self.level_index = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Prediction state
// ---------------------------------------------------------------------------

pub struct PredictionState {
    config: PredictionConfig,
    prediction: Option<FlightPlan>,
    dirty: bool,
    stale_after: f64,
    last_recompute_time: Option<f64>,
    version: u64,
}

impl PredictionState {
    fn new(config: PredictionConfig, stale_after: f64) -> Self {
        Self {
            config,
            prediction: None,
            dirty: true,
            stale_after,
            last_recompute_time: None,
            version: 0,
        }
    }

    pub fn stale_after(&self) -> f64 {
        self.stale_after
    }

    pub fn config(&self) -> &PredictionConfig {
        &self.config
    }

    pub fn prediction(&self) -> Option<&FlightPlan> {
        self.prediction.as_ref()
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn needs_refresh(&self, sim_time: f64) -> bool {
        if self.dirty || self.prediction.is_none() {
            return true;
        }
        self.last_recompute_time
            .map(|t| (sim_time - t) >= self.stale_after)
            .unwrap_or(true)
    }

    fn install(&mut self, prediction: FlightPlan, at_sim_time: f64) {
        self.prediction = Some(prediction);
        self.last_recompute_time = Some(at_sim_time);
        self.dirty = false;
        self.version = self.version.wrapping_add(1);
    }
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

pub struct Simulation {
    ship_state: StateVector,
    sim_time: f64,
    propagator: Arc<dyn ShipPropagator>,

    max_real_delta: f64,
    max_transitions_per_frame: u32,

    ephemeris: Arc<dyn BodyStateProvider>,
    bodies: Vec<BodyDefinition>,
    ship_thrust_acceleration: f64,
    maneuvers: ManeuverSequence,
    active_burns: Vec<ScheduledBurn>,
    consumed_node_ids: Vec<u64>,
    target_body: Option<BodyId>,

    pub warp: WarpController,
    pub prediction_state: PredictionState,
}

impl Simulation {
    pub fn new(
        ship_state: StateVector,
        ship_thrust_acceleration: f64,
        ephemeris: Arc<dyn BodyStateProvider>,
        bodies: Vec<BodyDefinition>,
        config: SimulationConfig,
    ) -> Self {
        let propagator: Arc<dyn ShipPropagator> = Arc::new(KeplerianPropagator {
            burn_substep_s: config.prediction_config.burn_substep_s,
            ..KeplerianPropagator::default()
        });

        Self {
            ship_state,
            sim_time: 0.0,
            propagator,
            max_real_delta: config.max_real_delta,
            max_transitions_per_frame: config.max_transitions_per_frame,
            ephemeris,
            bodies,
            ship_thrust_acceleration,
            maneuvers: ManeuverSequence::new(),
            active_burns: Vec::new(),
            consumed_node_ids: Vec::new(),
            target_body: None,
            warp: WarpController::new(config.warp_levels),
            prediction_state: PredictionState::new(
                config.prediction_config,
                config.prediction_stale_after,
            ),
        }
    }

    /// Advance the simulation by `real_dt` seconds of wall-clock time.
    ///
    /// Stepping is event-driven: at each iteration we find the nearest
    /// upcoming boundary (burn start/end, frame target), propagate exactly
    /// to it with the appropriate coast or burn segment, and repeat. The
    /// analytical propagator has no substep cost for coast segments, so
    /// long warp ticks complete in a single call.
    pub fn step(&mut self, real_dt: f64) {
        let _span = tracing::info_span!("Simulation::step").entered();
        let real_delta = real_dt.min(self.max_real_delta);
        let warp_speed = self.warp.speed();
        let sim_delta = real_delta * warp_speed;
        if sim_delta <= 0.0 {
            return;
        }

        let target_time = self.sim_time + sim_delta;
        let mut transitions = 0u32;

        while self.sim_time < target_time {
            if transitions >= self.max_transitions_per_frame {
                break;
            }

            // Drain any nodes whose time has arrived into `active_burns`.
            self.drain_due_nodes();

            let soi_body = self.propagator.soi_body_of(
                self.ship_state.position,
                self.sim_time,
                self.ephemeris.as_ref(),
                &self.bodies,
            );

            // Is a burn active right now?
            let active_burn = self
                .active_burns
                .iter()
                .find(|b| {
                    b.start_time <= self.sim_time && b.start_time + b.duration > self.sim_time
                })
                .copied();

            // Segment boundary: the soonest of target, next scheduled burn,
            // or next scheduled node. Burns clip at their own end.
            let segment_target = if let Some(b) = active_burn {
                target_time.min(b.start_time + b.duration)
            } else {
                let next_node = self
                    .maneuvers
                    .nodes
                    .first()
                    .map(|n| n.time)
                    .unwrap_or(f64::INFINITY);
                let next_burn = self
                    .active_burns
                    .iter()
                    .filter(|b| b.start_time > self.sim_time)
                    .map(|b| b.start_time)
                    .fold(f64::INFINITY, f64::min);
                target_time.min(next_node).min(next_burn)
            };

            if segment_target <= self.sim_time {
                break;
            }

            let result = if let Some(b) = active_burn {
                self.propagator.burn_segment(BurnRequest {
                    state: self.ship_state,
                    time: self.sim_time,
                    soi_body,
                    target_time: segment_target,
                    burn: to_burn_params(b),
                    ephemeris: self.ephemeris.as_ref(),
                    bodies: &self.bodies,
                })
            } else {
                self.propagator.coast_segment(CoastRequest {
                    state: self.ship_state,
                    time: self.sim_time,
                    soi_body,
                    target_time: segment_target,
                    stop_on_stable_orbit: false,
                    // Enough samples to catch SOI crossings reliably at
                    // typical warp rates without paying for big allocations
                    // on every step. Samples are discarded immediately —
                    // only the end state matters for live stepping.
                    sample_count_hint: 32,
                    ephemeris: self.ephemeris.as_ref(),
                    bodies: &self.bodies,
                })
            };

            self.ship_state = result.end_state;
            self.sim_time = result.end_time;

            match result.terminator {
                SegmentTerminator::Collision { .. } => {
                    // Ship is wrecked — freeze the state here. A future
                    // pass can surface a `CollisionEvent` to the game.
                    break;
                }
                SegmentTerminator::SoiEnter { .. } | SegmentTerminator::SoiExit { .. } => {
                    transitions += 1;
                }
                SegmentTerminator::BurnEnd { .. } => {
                    self.retire_completed_burns();
                    transitions += 1;
                }
                SegmentTerminator::Horizon | SegmentTerminator::StableOrbit => {}
            }
        }

        self.retire_completed_burns();
    }

    /// Drain maneuver nodes whose start time has arrived into `active_burns`.
    /// Keeps `active_burns` sorted by start time is not required — callers
    /// iterate all of them each step.
    fn drain_due_nodes(&mut self) {
        while let Some(node) = self.maneuvers.nodes.first() {
            if node.time > self.sim_time {
                break;
            }
            let node = self.maneuvers.nodes.remove(0);
            if let Some(id) = node.id {
                self.consumed_node_ids.push(id);
            }
            self.prediction_state.mark_dirty();

            let duration = burn_duration(node.delta_v.length(), self.ship_thrust_acceleration);
            if duration > 0.0 && node.delta_v.length_squared() > 0.0 {
                self.active_burns.push(ScheduledBurn {
                    delta_v_local: node.delta_v,
                    reference_body: node.reference_body,
                    acceleration: self.ship_thrust_acceleration,
                    start_time: node.time,
                    duration,
                });
            }
        }
    }

    fn retire_completed_burns(&mut self) {
        let before = self.active_burns.len();
        self.active_burns
            .retain(|b| b.start_time + b.duration > self.sim_time);
        if self.active_burns.len() != before {
            self.prediction_state.mark_dirty();
        }
    }

    // -- Accessors ----------------------------------------------------------

    pub fn ship_state(&self) -> &StateVector {
        &self.ship_state
    }

    pub fn sim_time(&self) -> f64 {
        self.sim_time
    }

    // -- Maneuvers ----------------------------------------------------------

    pub fn maneuvers(&self) -> &ManeuverSequence {
        &self.maneuvers
    }

    pub fn maneuvers_mut(&mut self) -> &mut ManeuverSequence {
        self.prediction_state.mark_dirty();
        &mut self.maneuvers
    }

    pub fn drain_consumed_node_ids(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.consumed_node_ids)
    }

    // -- Target body --------------------------------------------------------

    pub fn target_body(&self) -> Option<BodyId> {
        self.target_body
    }

    pub fn set_target_body(&mut self, body: Option<BodyId>) {
        if self.target_body != body {
            self.target_body = body;
            self.prediction_state.mark_dirty();
        }
    }

    // -- Prediction ---------------------------------------------------------

    pub fn prediction_stale_after(&self) -> f64 {
        self.prediction_state.stale_after()
    }

    pub fn prediction_version(&self) -> u64 {
        self.prediction_state.version()
    }

    pub fn recompute_prediction(&mut self) {
        let req = PredictionRequest {
            ship_state: self.ship_state,
            sim_time: self.sim_time,
            maneuvers: self.maneuvers.clone(),
            active_burns: self.active_burns.clone(),
            ephemeris: Arc::clone(&self.ephemeris),
            bodies: self.bodies.clone(),
            prediction_config: self.prediction_state.config().clone(),
            ship_thrust_acceleration: self.ship_thrust_acceleration,
            target_body: self.target_body,
        };
        let prediction = propagate_flight_plan(&req, None);
        self.prediction_state.install(prediction, req.sim_time);
    }

    pub fn prediction_needs_refresh(&self) -> bool {
        self.prediction_state.needs_refresh(self.sim_time)
    }

    pub fn prediction(&self) -> Option<&FlightPlan> {
        self.prediction_state.prediction()
    }

    // -- Body orbit trails --------------------------------------------------

    pub fn body_orbit_trails(&self, num_samples: usize) -> Vec<Option<BodyOrbitTrail>> {
        self.bodies
            .iter()
            .map(|body| {
                let parent_id = body.parent?;
                if body.kind == BodyKind::Star {
                    return None;
                }
                let points =
                    self.ephemeris
                        .body_orbit_trail(body.id, parent_id, self.sim_time, num_samples);
                Some(BodyOrbitTrail {
                    body_id: body.id,
                    parent_id,
                    points,
                })
            })
            .collect()
    }

    pub fn bodies(&self) -> &[BodyDefinition] {
        &self.bodies
    }

    pub fn ephemeris(&self) -> &dyn BodyStateProvider {
        self.ephemeris.as_ref()
    }
}

fn to_burn_params(b: ScheduledBurn) -> BurnParams {
    BurnParams {
        delta_v_local: b.delta_v_local,
        reference_body: b.reference_body,
        acceleration: b.acceleration,
        start_time: b.start_time,
        end_time: b.start_time + b.duration,
    }
}
