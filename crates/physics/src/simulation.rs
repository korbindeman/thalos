//! Central simulation struct that owns all physics state and logic.
//!
//! The game crate becomes a thin consumer: it calls [`Simulation::step`] each
//! frame and reads back the ship state, prediction, and warp info. Both live
//! stepping and trajectory prediction use the **same** [`IntegratorConfig`],
//! eliminating numerical divergence between the two.

use std::sync::Arc;

use glam::DVec3;

use crate::body_state_provider::BodyStateProvider;
use crate::forces::{Forces, ManeuverThrustForce};
use crate::integrator::{Integrator, IntegratorConfig};
use crate::maneuver::{ManeuverSequence, burn_duration};
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

/// Configuration for constructing a [`Simulation`].
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub integrator_config: IntegratorConfig,
    pub prediction_config: PredictionConfig,
    pub warp_levels: Vec<f64>,
    pub max_steps_per_frame: u32,
    pub max_real_delta: f64,
    pub prediction_stale_after: f64,
    /// Warp speeds at or above this skip ship integration entirely and just
    /// advance `sim_time`. Analytical body positions still update, so planets
    /// keep moving for observation. Ship state freezes.
    pub observation_warp_threshold: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            integrator_config: IntegratorConfig {
                symplectic_dt: 1.0,
                rk_initial_dt: 1.0,
                ..IntegratorConfig::default()
            },
            prediction_config: PredictionConfig {
                max_steps_per_segment: 10_000,
                ..PredictionConfig::default()
            },
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
            max_steps_per_frame: 10_000,
            max_real_delta: 0.1,
            prediction_stale_after: 30.0,
            observation_warp_threshold: 5_000_000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Warp controller
// ---------------------------------------------------------------------------

/// Discrete warp speed controller with named levels.
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

    /// Toggle between paused (level 0) and the last non-zero level.
    /// If already at level 0 and nothing to resume, advances to 1x.
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

/// Tracks prediction lifecycle: dirty flag, epoch, staleness.
pub struct PredictionState {
    config: PredictionConfig,
    prediction: Option<FlightPlan>,
    dirty: bool,
    epoch: u64,
    stale_after: f64,
    last_prediction_time: Option<f64>,
}

impl PredictionState {
    fn new(config: PredictionConfig, stale_after: f64) -> Self {
        Self {
            config,
            prediction: None,
            dirty: true,
            epoch: 1,
            stale_after,
            last_prediction_time: None,
        }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
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

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
        self.epoch = self.epoch.wrapping_add(1);
    }

    pub fn needs_refresh(&self, sim_time: f64) -> bool {
        if self.dirty || self.prediction.is_none() {
            return true;
        }

        self.last_prediction_time
            .map(|t| (sim_time - t) >= self.stale_after)
            .unwrap_or(true)
    }

    /// Apply a computed prediction, replacing whatever was there.
    ///
    /// Previously this rejected any prediction whose epoch didn't match the
    /// current (latest) epoch. That blocked real-time feedback during a
    /// drag: each edit advances the epoch, so by the time the worker
    /// returned a result the "current" epoch had already moved on and the
    /// result was dropped. Showing a slightly stale prediction is strictly
    /// better than showing a very stale one. `dirty` is only cleared when
    /// the applied epoch actually matches.
    pub fn apply(
        &mut self,
        prediction: FlightPlan,
        predicted_at_sim_time: f64,
        epoch: u64,
    ) -> bool {
        self.prediction = Some(prediction);
        self.last_prediction_time = Some(predicted_at_sim_time);
        if epoch == self.epoch {
            self.dirty = false;
            true
        } else {
            false
        }
    }

    /// Unconditionally set the prediction (no epoch check).
    pub fn force_set(&mut self, prediction: FlightPlan) {
        self.prediction = Some(prediction);
    }

    /// Clear the dirty flag without applying a prediction.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }
}

fn build_forces(active_burns: &[ScheduledBurn]) -> Forces {
    Forces {
        thrusts: active_burns
            .iter()
            .map(|burn| {
                ManeuverThrustForce::new(
                    burn.delta_v_local,
                    burn.reference_body,
                    burn.acceleration,
                    burn.start_time,
                    burn.duration,
                )
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

/// Owns all simulation state: ship, integrator, forces, warp, and prediction.
///
/// A single [`IntegratorConfig`] is used for both live stepping and trajectory
/// prediction, ensuring they never diverge numerically.
pub struct Simulation {
    ship_state: StateVector,
    sim_time: f64,
    integrator: Integrator,
    forces: Forces,
    integrator_config: IntegratorConfig,

    // Accumulator
    accumulator: f64,
    max_steps_per_frame: u32,
    max_real_delta: f64,
    observation_warp_threshold: f64,

    // Shared state
    ephemeris: Arc<dyn BodyStateProvider>,
    bodies: Vec<BodyDefinition>,
    ship_thrust_acceleration: f64,
    maneuvers: ManeuverSequence,
    active_burns: Vec<ScheduledBurn>,

    // Composed subsystems
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
        let integrator = Integrator::new(config.integrator_config.clone());

        let forces = build_forces(&[]);

        Self {
            ship_state,
            sim_time: 0.0,
            integrator,
            forces,
            integrator_config: config.integrator_config,
            accumulator: 0.0,
            max_steps_per_frame: config.max_steps_per_frame,
            max_real_delta: config.max_real_delta,
            observation_warp_threshold: config.observation_warp_threshold,
            ephemeris,
            bodies,
            ship_thrust_acceleration,
            maneuvers: ManeuverSequence::new(),
            active_burns: Vec::new(),
            warp: WarpController::new(config.warp_levels),
            prediction_state: PredictionState::new(
                config.prediction_config,
                config.prediction_stale_after,
            ),
        }
    }

    /// Advance the simulation by `real_dt` seconds of wall-clock time.
    ///
    /// The real delta is capped, multiplied by warp speed, and accumulated.
    /// Sub-steps use the integrator's native step size. A step budget prevents
    /// spiral-of-death when the integrator can't keep up at extreme warps.
    pub fn step(&mut self, real_dt: f64) {
        let _span = tracing::info_span!("Simulation::step").entered();
        let real_delta = real_dt.min(self.max_real_delta);
        let warp_speed = self.warp.speed();

        // Observation mode: at extreme warps, skip ship integration entirely.
        // Just advance sim_time so analytical body positions keep updating.
        // Ship state freezes — any ongoing maneuver/trajectory is invalidated
        // until warp drops below the threshold.
        if warp_speed >= self.observation_warp_threshold {
            self.sim_time += real_delta * warp_speed;
            self.accumulator = 0.0;
            return;
        }

        self.accumulator += real_delta * warp_speed;

        if self.accumulator <= 0.0 {
            return;
        }

        let step_size = self.integrator_config.symplectic_dt;
        let max_budget = self.max_steps_per_frame as f64 * step_size;
        if self.accumulator > max_budget {
            self.accumulator = max_budget;
        }

        let mut steps = 0u32;
        while self.accumulator >= step_size && steps < self.max_steps_per_frame {
            if self.update_active_burns() {
                self.forces = build_forces(&self.active_burns);
            }

            let (new_state, sample) = self.integrator.step(
                self.ship_state,
                self.sim_time,
                &self.forces,
                self.ephemeris.as_ref(),
            );
            let step_taken = sample.time - self.sim_time;
            self.ship_state = new_state;
            self.sim_time = sample.time;
            self.accumulator -= step_taken;
            steps += 1;
        }

        if self.update_active_burns() {
            self.forces = build_forces(&self.active_burns);
        }
    }

    /// Drain maneuver nodes whose start time has passed into `active_burns`,
    /// and retire completed burns. Returns true if the set changed.
    fn update_active_burns(&mut self) -> bool {
        let mut changed = false;

        // Drain due nodes. `maneuvers.nodes` is kept time-sorted, so pop from
        // the front while the head's time is in the past.
        while let Some(node) = self.maneuvers.nodes.first() {
            if node.time > self.sim_time {
                break;
            }
            let node = self.maneuvers.nodes.remove(0);
            self.prediction_state.mark_dirty();
            changed = true;

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

        // Retire completed burns.
        let before = self.active_burns.len();
        self.active_burns
            .retain(|b| b.start_time + b.duration > self.sim_time);
        if self.active_burns.len() != before {
            self.prediction_state.mark_dirty();
            changed = true;
        }

        changed
    }

    // -- Accessors ----------------------------------------------------------

    pub fn ship_state(&self) -> &StateVector {
        &self.ship_state
    }

    pub fn sim_time(&self) -> f64 {
        self.sim_time
    }

    /// True when warp is at/above the observation threshold — ship integration
    /// and trajectory prediction are skipped to keep the frame cheap.
    pub fn is_observation_mode(&self) -> bool {
        self.warp.speed() >= self.observation_warp_threshold
    }

    // -- Maneuvers ----------------------------------------------------------

    pub fn maneuvers(&self) -> &ManeuverSequence {
        &self.maneuvers
    }

    pub fn maneuvers_mut(&mut self) -> &mut ManeuverSequence {
        self.prediction_state.mark_dirty();
        &mut self.maneuvers
    }

    // -- Prediction ---------------------------------------------------------

    pub fn prediction_epoch(&self) -> u64 {
        self.prediction_state.epoch()
    }

    pub fn prediction_stale_after(&self) -> f64 {
        self.prediction_state.stale_after()
    }

    pub fn prediction_request(&self) -> PredictionRequest {
        // Prediction uses a coarser time step than the live simulation —
        // orbit shape doesn't need 1 s resolution, and larger steps let us
        // cover multiple full orbits within the step budget.
        let mut prediction_integrator = self.integrator_config.clone();
        prediction_integrator.symplectic_dt = IntegratorConfig::default().symplectic_dt;
        prediction_integrator.rk_initial_dt = IntegratorConfig::default().rk_initial_dt;

        PredictionRequest {
            epoch: self.prediction_state.epoch(),
            ship_state: self.ship_state,
            sim_time: self.sim_time,
            maneuvers: self.maneuvers.clone(),
            active_burns: self.active_burns.clone(),
            ephemeris: Arc::clone(&self.ephemeris),
            bodies: self.bodies.clone(),
            prediction_config: self.prediction_state.config().clone(),
            integrator_config: prediction_integrator,
            ship_thrust_acceleration: self.ship_thrust_acceleration,
        }
    }

    /// Apply a computed prediction if it still matches the current epoch.
    pub fn apply_prediction(
        &mut self,
        prediction: FlightPlan,
        predicted_at_sim_time: f64,
        epoch: u64,
    ) -> bool {
        self.prediction_state
            .apply(prediction, predicted_at_sim_time, epoch)
    }

    /// Recompute trajectory prediction using the same integrator config as live
    /// stepping, ensuring numerical consistency.
    pub fn recompute_prediction(&mut self) {
        let req = self.prediction_request();
        let prediction = propagate_flight_plan(&req, None);
        let _ = self
            .prediction_state
            .apply(prediction, req.sim_time, req.epoch);
    }

    pub fn prediction_needs_refresh(&self) -> bool {
        self.prediction_state.needs_refresh(self.sim_time)
    }

    pub fn prediction(&self) -> Option<&FlightPlan> {
        self.prediction_state.prediction()
    }

    pub fn force_set_prediction(&mut self, prediction: FlightPlan) {
        self.prediction_state.force_set(prediction);
    }

    pub fn clear_prediction_dirty(&mut self) {
        self.prediction_state.clear_dirty();
    }

    // -- Body orbit trails --------------------------------------------------

    /// Compute forward-looking orbit trails for all non-star bodies.
    ///
    /// Each trail spans one orbital period forward from the current sim time,
    /// with positions relative to the body's parent (in metres). Returns one
    /// entry per body; stars get `None`.
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
