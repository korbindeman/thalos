//! Central simulation struct that owns all physics state and logic.
//!
//! The game crate becomes a thin consumer: it calls [`Simulation::step`] each
//! frame and reads back the ship state, prediction, and warp info. Both live
//! stepping and trajectory prediction use the **same** [`IntegratorConfig`],
//! eliminating numerical divergence between the two.

use std::sync::Arc;

use crate::ephemeris::Ephemeris;
use crate::forces::{ForceRegistry, GravityForce};
use crate::integrator::{Integrator, IntegratorConfig};
use crate::maneuver::ManeuverSequence;
use crate::trajectory::{propagate_trajectory, PredictionConfig, TrajectoryPrediction};
use crate::types::{BodyDefinition, StateVector};

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
                max_steps_per_segment: 5_000,
                ..PredictionConfig::default()
            },
            warp_levels: vec![1.0, 10.0, 100.0, 1000.0, 10_000.0],
            max_steps_per_frame: 10_000,
            max_real_delta: 0.1,
        }
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

    // Warp
    warp_level_index: usize,
    warp_levels: Vec<f64>,

    // Accumulator
    accumulator: f64,
    max_steps_per_frame: u32,
    max_real_delta: f64,

    // Shared state
    ephemeris: Arc<Ephemeris>,
    bodies: Vec<BodyDefinition>,
    maneuvers: ManeuverSequence,

    // Prediction
    prediction_config: PredictionConfig,
    prediction: Option<TrajectoryPrediction>,
    prediction_dirty: bool,
}

impl Simulation {
    pub fn new(
        ship_state: StateVector,
        ephemeris: Arc<Ephemeris>,
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
            warp_level_index: 0,
            warp_levels: config.warp_levels,
            accumulator: 0.0,
            max_steps_per_frame: config.max_steps_per_frame,
            max_real_delta: config.max_real_delta,
            ephemeris,
            bodies,
            maneuvers: ManeuverSequence::new(),
            prediction_config: config.prediction_config,
            prediction: None,
            prediction_dirty: true,
        }
    }

    /// Advance the simulation by `real_dt` seconds of wall-clock time.
    ///
    /// The real delta is capped, multiplied by warp speed, and accumulated.
    /// Sub-steps use the integrator's native step size. A step budget prevents
    /// spiral-of-death when the integrator can't keep up at extreme warps.
    pub fn step(&mut self, real_dt: f64) {
        let real_delta = real_dt.min(self.max_real_delta);
        self.accumulator += real_delta * self.warp_speed();

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
                &self.ephemeris,
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
        self.warp_levels[self.warp_level_index]
    }

    pub fn warp_label(&self) -> String {
        let speed = self.warp_speed();
        if speed >= 1000.0 {
            format!("{:.0}k\u{00d7}", speed / 1000.0)
        } else {
            format!("{:.0}\u{00d7}", speed)
        }
    }

    // -- Warp controls ------------------------------------------------------

    pub fn increase_warp(&mut self) {
        self.warp_level_index = (self.warp_level_index + 1).min(self.warp_levels.len() - 1);
    }

    pub fn decrease_warp(&mut self) {
        self.warp_level_index = self.warp_level_index.saturating_sub(1);
    }

    pub fn reset_warp(&mut self) {
        self.warp_level_index = 0;
    }

    // -- Maneuvers ----------------------------------------------------------

    pub fn maneuvers(&self) -> &ManeuverSequence {
        &self.maneuvers
    }

    pub fn maneuvers_mut(&mut self) -> &mut ManeuverSequence {
        self.prediction_dirty = true;
        &mut self.maneuvers
    }

    // -- Prediction ---------------------------------------------------------

    /// Recompute trajectory prediction using the same integrator config as live
    /// stepping, ensuring numerical consistency.
    pub fn recompute_prediction(&mut self) {
        let prediction = propagate_trajectory(
            self.ship_state,
            self.sim_time,
            &self.maneuvers,
            &self.ephemeris,
            &self.bodies,
            &self.prediction_config,
            self.integrator_config.clone(),
        );
        self.prediction = Some(prediction);
        self.prediction_dirty = false;
    }

    pub fn prediction(&self) -> Option<&TrajectoryPrediction> {
        self.prediction.as_ref()
    }
}
