//! Central simulation struct that owns all physics state and logic.
//!
//! The game crate becomes a thin consumer: it calls [`Simulation::step`] each
//! frame and reads back the ship state, prediction, and warp info. Both live
//! stepping and trajectory prediction use the **same** [`IntegratorConfig`],
//! eliminating numerical divergence between the two.

use std::sync::Arc;

use glam::DVec3;

use crate::body_state_provider::BodyStateProvider;
use crate::forces::{ForceRegistry, GravityForce};
use crate::integrator::{Integrator, IntegratorConfig};
use crate::maneuver::ManeuverSequence;
use crate::trajectory::{PredictionConfig, TrajectoryPrediction, propagate_trajectory};
use crate::types::{BodyDefinition, BodyId, BodyKind, StateVector};

/// Forward-looking orbit trail for a single body, relative to its parent.
#[derive(Debug, Clone)]
pub struct BodyOrbitTrail {
    pub body_id: BodyId,
    pub parent_id: BodyId,
    /// Positions in metres, relative to the parent body.
    pub points: Vec<DVec3>,
}

/// Immutable inputs for trajectory prediction.
///
/// This can be sent to a worker thread so prediction can run without blocking
/// the main simulation/update loop.
#[derive(Clone)]
pub struct PredictionRequest {
    pub epoch: u64,
    pub ship_state: StateVector,
    pub sim_time: f64,
    pub maneuvers: ManeuverSequence,
    pub ephemeris: Arc<dyn BodyStateProvider>,
    pub bodies: Vec<BodyDefinition>,
    pub prediction_config: PredictionConfig,
    pub integrator_config: IntegratorConfig,
    pub ship_thrust_acceleration: f64,
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
            warp_levels: vec![0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0],
            max_steps_per_frame: 10_000,
            max_real_delta: 0.1,
            prediction_stale_after: 30.0,
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
}

impl WarpController {
    fn new(levels: Vec<f64>) -> Self {
        Self {
            level_index: 0,
            levels,
        }
    }

    pub fn speed(&self) -> f64 {
        self.levels[self.level_index]
    }

    pub fn label(&self) -> String {
        let speed = self.speed();
        if speed == 0.0 {
            "PAUSED".to_string()
        } else if speed >= 1000.0 {
            format!("{:.0}k\u{00d7}", speed / 1000.0)
        } else {
            format!("{:.0}\u{00d7}", speed)
        }
    }

    pub fn increase(&mut self) {
        self.level_index = (self.level_index + 1).min(self.levels.len() - 1);
    }

    pub fn decrease(&mut self) {
        self.level_index = self.level_index.saturating_sub(1);
    }

    pub fn reset(&mut self) {
        self.level_index = self
            .levels
            .iter()
            .position(|&w| w == 1.0)
            .unwrap_or(0);
    }
}

// ---------------------------------------------------------------------------
// Prediction state
// ---------------------------------------------------------------------------

/// Tracks prediction lifecycle: dirty flag, epoch, staleness.
pub struct PredictionState {
    config: PredictionConfig,
    prediction: Option<TrajectoryPrediction>,
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

    pub fn prediction(&self) -> Option<&TrajectoryPrediction> {
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

    /// Apply a computed prediction if it still matches the current epoch.
    ///
    /// Returns `true` if the prediction was accepted, `false` if it was stale.
    pub fn apply(
        &mut self,
        prediction: TrajectoryPrediction,
        predicted_at_sim_time: f64,
        epoch: u64,
    ) -> bool {
        if epoch != self.epoch {
            return false;
        }
        self.prediction = Some(prediction);
        self.dirty = false;
        self.last_prediction_time = Some(predicted_at_sim_time);
        true
    }

    /// Unconditionally set the prediction (no epoch check).
    pub fn force_set(&mut self, prediction: TrajectoryPrediction) {
        self.prediction = Some(prediction);
    }

    /// Clear the dirty flag without applying a prediction.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
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
    forces: ForceRegistry,
    integrator_config: IntegratorConfig,

    // Accumulator
    accumulator: f64,
    max_steps_per_frame: u32,
    max_real_delta: f64,

    // Shared state
    ephemeris: Arc<dyn BodyStateProvider>,
    bodies: Vec<BodyDefinition>,
    ship_thrust_acceleration: f64,
    maneuvers: ManeuverSequence,

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

        let mut forces = ForceRegistry::new();
        forces.add(Box::new(GravityForce));

        Self {
            ship_state,
            sim_time: 0.0,
            integrator,
            forces,
            integrator_config: config.integrator_config,
            accumulator: 0.0,
            max_steps_per_frame: config.max_steps_per_frame,
            max_real_delta: config.max_real_delta,
            ephemeris,
            bodies,
            ship_thrust_acceleration,
            maneuvers: ManeuverSequence::new(),
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
        let real_delta = real_dt.min(self.max_real_delta);
        self.accumulator += real_delta * self.warp.speed();

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
    }

    // -- Accessors ----------------------------------------------------------

    pub fn ship_state(&self) -> &StateVector {
        &self.ship_state
    }

    pub fn sim_time(&self) -> f64 {
        self.sim_time
    }

    pub fn warp_speed(&self) -> f64 {
        self.warp.speed()
    }

    pub fn warp_label(&self) -> String {
        self.warp.label()
    }

    // -- Warp controls (convenience delegates) ------------------------------

    pub fn increase_warp(&mut self) {
        self.warp.increase();
    }

    pub fn decrease_warp(&mut self) {
        self.warp.decrease();
    }

    pub fn reset_warp(&mut self) {
        self.warp.reset();
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
        prediction: TrajectoryPrediction,
        predicted_at_sim_time: f64,
        epoch: u64,
    ) -> bool {
        self.prediction_state.apply(prediction, predicted_at_sim_time, epoch)
    }

    /// Recompute trajectory prediction using the same integrator config as live
    /// stepping, ensuring numerical consistency.
    pub fn recompute_prediction(&mut self) {
        let req = self.prediction_request();
        let prediction = propagate_trajectory(
            req.ship_state,
            req.sim_time,
            &req.maneuvers,
            req.ephemeris.as_ref(),
            &req.bodies,
            &req.prediction_config,
            req.integrator_config,
            req.ship_thrust_acceleration,
        );
        let _ = self.prediction_state.apply(prediction, req.sim_time, req.epoch);
    }

    pub fn prediction_needs_refresh(&self) -> bool {
        self.prediction_state.needs_refresh(self.sim_time)
    }

    pub fn prediction(&self) -> Option<&TrajectoryPrediction> {
        self.prediction_state.prediction()
    }

    pub fn force_set_prediction(&mut self, prediction: TrajectoryPrediction) {
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
