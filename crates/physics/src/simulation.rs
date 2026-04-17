//! Central simulation struct that owns all physics state and logic.
//!
//! The game crate becomes a thin consumer: it calls [`Simulation::step`] each
//! frame and reads back the ship state, prediction, and warp info. Both live
//! stepping and trajectory prediction use the **same** [`IntegratorConfig`],
//! eliminating numerical divergence between the two.

use std::sync::Arc;

use glam::DVec3;

use crate::body_state_provider::BodyStateProvider;
use crate::effects::{Effect, EffectRegistry, ManeuverThrustEffect};
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

    /// Drop to the highest level strictly below `max_speed`. Used to pull out
    /// of observation mode when an upcoming maneuver needs real integration.
    pub fn clamp_below(&mut self, max_speed: f64) {
        if self.levels[self.level_index] < max_speed {
            return;
        }
        self.level_index = self
            .levels
            .iter()
            .rposition(|&w| w < max_speed)
            .unwrap_or(0);
        self.resume_index = None;
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

/// Tracks the cached prediction: whether it's dirty, when it was last
/// recomputed, and the [`PredictionConfig`] used to build it.
///
/// Prediction is now recomputed synchronously on the main thread, so there
/// is no worker-thread epoch/stale-message machinery — a single `dirty` flag
/// plus a last-recompute timestamp are enough.
pub struct PredictionState {
    config: PredictionConfig,
    prediction: Option<FlightPlan>,
    dirty: bool,
    stale_after: f64,
    last_recompute_time: Option<f64>,
    /// Monotonic counter bumped every time a fresh prediction is installed.
    /// Consumers (ghost-body rebuild, etc.) key their caches on this.
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

    pub fn needs_refresh(&self, sim_time: f64, _warp_speed: f64) -> bool {
        if self.dirty || self.prediction.is_none() {
            return true;
        }
        // Refresh on the sim-time staleness threshold regardless of warp.
        // The old optimization skipped recomputes during warp under the
        // assumption that live stepping and prediction follow identical
        // trajectories — but the adaptive integrator accumulates tiny
        // per-step differences, so the ship visibly drifts off the drawn
        // trail over long warps.  Frequency is still bounded (one recompute
        // every `stale_after` sim-seconds) and is cheap.
        self.last_recompute_time
            .map(|t| (sim_time - t) >= self.stale_after)
            .unwrap_or(true)
    }

    /// Install a freshly computed prediction, clear the dirty flag, and
    /// bump the version counter so cache consumers rebuild.
    fn install(&mut self, prediction: FlightPlan, at_sim_time: f64) {
        self.prediction = Some(prediction);
        self.last_recompute_time = Some(at_sim_time);
        self.dirty = false;
        self.version = self.version.wrapping_add(1);
    }
}

fn build_registry(active_burns: &[ScheduledBurn]) -> EffectRegistry {
    EffectRegistry::with_effects(
        active_burns
            .iter()
            .map(|burn| {
                Arc::new(ManeuverThrustEffect::new(
                    burn.delta_v_local,
                    burn.reference_body,
                    burn.acceleration,
                    burn.start_time,
                    burn.duration,
                )) as Arc<dyn Effect>
            })
            .collect(),
    )
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
    registry: EffectRegistry,
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
    /// IDs of maneuver nodes drained into `active_burns` since the last
    /// `drain_consumed_node_ids()` call. Lets the bridge reconcile its UI
    /// state against what physics actually executed.
    consumed_node_ids: Vec<u64>,
    /// Currently selected target body.  Forwarded into `PredictionRequest`
    /// for biased adaptive step sizing near the target.
    target_body: Option<BodyId>,

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

        let registry = build_registry(&[]);

        Self {
            ship_state,
            sim_time: 0.0,
            integrator,
            registry,
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
            let sim_delta = real_delta * warp_speed;
            // If any pending maneuver falls within the jump window or an
            // active burn is in flight, drop out of observation mode so the
            // integrator can execute it. Don't advance this frame; next frame
            // runs normal integration.
            let threshold = self.observation_warp_threshold;
            let maneuver_due = self
                .maneuvers
                .nodes
                .first()
                .is_some_and(|n| n.time <= self.sim_time + sim_delta);
            if maneuver_due || !self.active_burns.is_empty() {
                self.warp.clamp_below(threshold);
                self.accumulator = 0.0;
                return;
            }
            self.sim_time += sim_delta;
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
                self.registry = build_registry(&self.active_burns);
            }

            let (new_state, sample) = self.integrator.step(
                self.ship_state,
                self.sim_time,
                &self.registry,
                self.ephemeris.as_ref(),
            );
            let step_taken = sample.time - self.sim_time;
            self.ship_state = new_state;
            self.sim_time = sample.time;
            self.accumulator -= step_taken;
            steps += 1;
        }

        if self.update_active_burns() {
            self.registry = build_registry(&self.active_burns);
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
            if let Some(id) = node.id {
                self.consumed_node_ids.push(id);
            }
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

    /// Drain the list of maneuver node IDs that physics has consumed since
    /// the last call. The caller (typically the bridge) uses this to remove
    /// the corresponding UI nodes — tying UI lifetime to actual execution
    /// rather than a wall-clock time comparison.
    pub fn drain_consumed_node_ids(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.consumed_node_ids)
    }

    // -- Target body --------------------------------------------------------

    pub fn target_body(&self) -> Option<BodyId> {
        self.target_body
    }

    /// Set (or clear) the currently selected target body.  Marks the
    /// prediction dirty so the next refresh uses biased adaptive step sizing
    /// near the new target.
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

    /// Monotonic counter bumped each time `recompute_prediction` installs a
    /// fresh result. Consumers use this to invalidate caches.
    pub fn prediction_version(&self) -> u64 {
        self.prediction_state.version()
    }

    /// Recompute trajectory prediction synchronously using the same
    /// integrator config as live stepping, ensuring numerical consistency.
    pub fn recompute_prediction(&mut self) {
        let req = PredictionRequest {
            ship_state: self.ship_state,
            sim_time: self.sim_time,
            maneuvers: self.maneuvers.clone(),
            active_burns: self.active_burns.clone(),
            ephemeris: Arc::clone(&self.ephemeris),
            bodies: self.bodies.clone(),
            prediction_config: self.prediction_state.config().clone(),
            integrator_config: self.integrator_config.clone(),
            ship_thrust_acceleration: self.ship_thrust_acceleration,
            target_body: self.target_body,
        };
        let prediction = propagate_flight_plan(&req, None);
        self.prediction_state.install(prediction, req.sim_time);
    }

    pub fn prediction_needs_refresh(&self) -> bool {
        self.prediction_state
            .needs_refresh(self.sim_time, self.warp.speed())
    }

    pub fn prediction(&self) -> Option<&FlightPlan> {
        self.prediction_state.prediction()
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
