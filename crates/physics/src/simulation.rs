//! Central simulation struct that owns all physics state and logic.
//!
//! The game crate becomes a thin consumer: it calls [`Simulation::step`] each
//! frame and reads back the ship state, prediction, and warp info. Live
//! stepping and trajectory prediction both route through the same
//! [`ShipPropagator`], so "where the ship is" and "where it is predicted to
//! be" cannot drift numerically.

use std::sync::Arc;

use glam::{DMat3, DQuat, DVec3};

use crate::body_fixed::evaluate_body_fixed_pose;
use crate::body_trajectory_provider::BodyTrajectoryProvider;
use crate::canonical::{
    AuthorityMode, CraftAuthorityBook, CraftState, Epoch, MassState, ResourceState,
    TranslationalState,
};
use crate::gravity_mode::GravityImpls;
use crate::maneuver::{ManeuverSequence, burn_duration};
use crate::ship_propagator::{CoastRequest, SegmentTerminator, ShipPropagator};
use crate::trajectory::{
    FlightPlan, PredictionConfig, PredictionRequest, TrajectoryBranchStack, propagate_branch_stack,
};
use crate::types::{
    AttitudeState, BodyDefinition, BodyId, BodyKind, ControlInput, ShipParameters, StateVector,
};

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

const WARP_TRANSITION_DURATION_S: f64 = 0.35;

pub struct WarpController {
    /// Target level index. The effective speed may be between this and the
    /// previous target while a transition is in flight.
    level_index: usize,
    levels: Vec<f64>,
    resume_index: Option<usize>,
    speed: f64,
    transition_from: f64,
    transition_elapsed: f64,
}

impl WarpController {
    fn new(levels: Vec<f64>) -> Self {
        let speed = levels.first().copied().unwrap_or(0.0);
        Self {
            level_index: 0,
            levels,
            resume_index: None,
            speed,
            transition_from: speed,
            transition_elapsed: WARP_TRANSITION_DURATION_S,
        }
    }

    /// Effective warp multiplier currently used to advance simulation time.
    /// This lerps toward [`Self::target_speed`] after most level changes.
    pub fn speed(&self) -> f64 {
        self.speed
    }

    pub fn levels(&self) -> &[f64] {
        &self.levels
    }

    /// Discrete multiplier for the currently selected target level.
    pub fn target_speed(&self) -> f64 {
        self.levels
            .get(self.level_index)
            .copied()
            .unwrap_or(self.speed)
    }

    pub fn level_index(&self) -> usize {
        self.level_index
    }

    pub fn latched_level_index(&self) -> Option<usize> {
        if self.level_index == 0 {
            self.resume_index
                .or_else(|| self.levels.iter().position(|&w| w == 1.0))
        } else {
            None
        }
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

    /// Advance the effective warp speed toward the selected target level.
    pub fn update(&mut self, real_dt: f64) {
        if real_dt <= 0.0 {
            return;
        }

        let target = self.target_speed();
        if (self.speed - target).abs() <= f64::EPSILON {
            self.speed = target;
            self.transition_from = target;
            self.transition_elapsed = WARP_TRANSITION_DURATION_S;
            return;
        }

        self.transition_elapsed += real_dt;
        let t = (self.transition_elapsed / WARP_TRANSITION_DURATION_S).clamp(0.0, 1.0);
        self.speed = self.transition_from + (target - self.transition_from) * t;

        if t >= 1.0 {
            self.speed = target;
            self.transition_from = target;
        }
    }

    pub fn increase(&mut self) {
        if self.levels.is_empty() {
            return;
        }
        self.resume_index = None;
        self.set_level_smooth((self.level_index + 1).min(self.levels.len() - 1));
    }

    pub fn decrease(&mut self) {
        self.resume_index = None;
        self.set_level_smooth(self.level_index.saturating_sub(1));
    }

    pub fn reset(&mut self) {
        self.resume_index = None;
        let index = self.levels.iter().position(|&w| w == 1.0).unwrap_or(0);
        self.set_level_capped(index);
    }

    /// Snap directly to 1x without the normal pause/resume transition.
    ///
    /// This is intentionally separate from [`Self::reset`], whose smooth
    /// upward transition is better for normal player input. Systems that must
    /// immediately hand live dynamics back to the player, such as debug launch
    /// clamps, use this so throttle gating sees exactly 1x on the next frame.
    pub fn reset_immediate(&mut self) {
        self.resume_index = None;
        let index = self.levels.iter().position(|&w| w == 1.0).unwrap_or(0);
        self.set_level_immediate(index);
    }

    pub fn toggle_pause(&mut self) {
        if self.level_index == 0 {
            let target = self
                .resume_index
                .take()
                .unwrap_or_else(|| self.levels.iter().position(|&w| w == 1.0).unwrap_or(0));
            self.set_level_smooth(target);
        } else {
            self.resume_index = Some(self.level_index);
            self.set_level_immediate(0);
        }
    }

    /// Select the highest configured speed that does not exceed
    /// `target_speed`. Used by the warp-to-event auto-warp to size
    /// each frame's advance against the time remaining to a target event,
    /// stepping the target up or down through the discrete levels in a single
    /// call. Upward changes lerp through effective speeds; downward changes
    /// clamp immediately so event-arrival and burn-prep windows cannot
    /// overshoot while decelerating.
    ///
    /// Falls back to the lowest level (typically pause) when no level
    /// satisfies the cap. Callers that always want forward motion should
    /// guard with a minimum themselves — the warp-to-event system does
    /// this with `target_speed.max(1.0)`.
    pub fn set_speed(&mut self, target_speed: f64) {
        self.resume_index = None;
        let index = self
            .levels
            .iter()
            .rposition(|&w| w <= target_speed)
            .unwrap_or(0);
        self.set_level_capped(index);
    }

    fn set_level_smooth(&mut self, target_index: usize) {
        let Some(target_speed) = self.levels.get(target_index).copied() else {
            return;
        };
        let previous_index = self.level_index;
        self.level_index = target_index;

        // Snap any transition that involves the paused (0) level. While
        // canonical authority is `LocalRigidBody`, `Simulation::step` is a
        // no-op for translation — the lerp would freeze the trajectory for
        // its duration and "lose" ~1.2 km of LEO motion per pause cycle as
        // sim-time advanced underneath. The smooth lerp is reserved for
        // transitions between non-zero levels (1x ↔ 10x, etc.) where
        // canonical advances under `OnRails`.
        if previous_index == 0 || target_index == 0 {
            self.snap_to_target();
            return;
        }

        if previous_index == target_index {
            return;
        }
        if (self.speed - target_speed).abs() <= f64::EPSILON {
            self.speed = target_speed;
            self.transition_from = target_speed;
            self.transition_elapsed = WARP_TRANSITION_DURATION_S;
            return;
        }

        self.transition_from = self.speed;
        self.transition_elapsed = 0.0;
    }

    fn set_level_capped(&mut self, target_index: usize) {
        let Some(target_speed) = self.levels.get(target_index).copied() else {
            return;
        };
        let previous_index = self.level_index;
        self.level_index = target_index;

        // Same reason as `set_level_smooth`: never lerp through the paused
        // level, the translation would be frozen for the duration.
        if previous_index == 0 || target_index == 0 {
            self.snap_to_target();
            return;
        }

        if target_speed <= self.speed {
            self.snap_to_target();
            return;
        }
        if previous_index != target_index {
            self.transition_from = self.speed;
            self.transition_elapsed = 0.0;
        }
    }

    fn set_level_immediate(&mut self, target_index: usize) {
        if self.levels.get(target_index).is_none() {
            return;
        }
        self.level_index = target_index;
        self.snap_to_target();
    }

    fn snap_to_target(&mut self) {
        let target = self.target_speed();
        self.speed = target;
        self.transition_from = target;
        self.transition_elapsed = WARP_TRANSITION_DURATION_S;
    }
}

// ---------------------------------------------------------------------------
// Prediction state
// ---------------------------------------------------------------------------

pub struct PredictionState {
    config: PredictionConfig,
    branches: Option<TrajectoryBranchStack>,
    dirty: bool,
    stale_after: f64,
    last_recompute_time: Option<f64>,
    version: u64,
}

impl PredictionState {
    fn new(config: PredictionConfig, stale_after: f64) -> Self {
        Self {
            config,
            branches: None,
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
        self.branches
            .as_ref()
            .map(TrajectoryBranchStack::active_plan)
    }

    pub fn branch_stack(&self) -> Option<&TrajectoryBranchStack> {
        self.branches.as_ref()
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn needs_refresh(&self, sim_time: f64) -> bool {
        if self.dirty || self.branches.is_none() {
            return true;
        }
        self.last_recompute_time
            .map(|t| (sim_time - t) >= self.stale_after)
            .unwrap_or(true)
    }

    fn install(&mut self, branches: TrajectoryBranchStack, at_sim_time: f64) {
        self.branches = Some(branches);
        self.last_recompute_time = Some(at_sim_time);
        self.dirty = false;
        self.version = self.version.wrapping_add(1);
    }

    /// Drop the cached branches and mark dirty so the next ballistic frame
    /// rebuilds. Bumping the version invalidates any downstream caches keyed
    /// off [`Self::version`].
    fn clear(&mut self) {
        if self.branches.is_some() {
            self.version = self.version.wrapping_add(1);
        }
        self.branches = None;
        self.last_recompute_time = None;
        self.dirty = true;
    }
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

pub struct Simulation {
    craft: CraftState,
    authority_book: CraftAuthorityBook,
    sim_time: f64,
    propagator: Arc<dyn ShipPropagator>,

    max_real_delta: f64,
    max_transitions_per_frame: u32,

    ephemeris: Arc<dyn BodyTrajectoryProvider>,
    bodies: Vec<BodyDefinition>,
    maneuvers: ManeuverSequence,
    consumed_node_ids: Vec<u64>,
    target_body: Option<BodyId>,

    /// Lifetime cumulative magnitude of Δv applied through
    /// [`Self::apply_external_mass_flow`], in m/s. Sums
    /// `|throttle · F/m · dt|` over every frame the engine fires (Avian
    /// owns the thrust impulse itself; the simulation only tracks the
    /// scalar). Read by the game-side autopilot (see
    /// `crates/game/src/autopilot.rs`) to detect when a finite burn has
    /// delivered its planned Δv — magnitude rather than projection
    /// because magnitude tracks the same scalar that fuel exhaustion
    /// clamps, so a propellant-starved burn stops accumulating without
    /// special-casing.
    delivered_dv: f64,

    ship_params: ShipParameters,
    control: ControlInput,

    pub warp: WarpController,
    pub prediction_state: PredictionState,
}

impl Simulation {
    /// Build a simulation with a placeholder ship: [`ShipParameters::default`]
    /// (no thrust, sentinel MOI) and matching sentinel mass. The real
    /// values are pushed in by the game crate at ship spawn via
    /// [`Self::set_ship_params`] and [`Self::set_ship_mass`] once the
    /// blueprint has been loaded.
    ///
    /// `impls` is produced by [`crate::gravity_mode::GravityMode::build`] —
    /// the construction site (today: `main.rs`; eventually: the savegame
    /// loader) picks the gravity model and hands the resulting trait
    /// objects in.
    pub fn new(
        ship_state: StateVector,
        impls: GravityImpls,
        bodies: Vec<BodyDefinition>,
        config: SimulationConfig,
    ) -> Self {
        let GravityImpls {
            body_trajectory: ephemeris,
            ship_propagator: propagator,
        } = impls;

        let ship_params = ShipParameters::default();
        let authority = AuthorityMode::OnRails { trajectory: 0 };
        Self {
            craft: CraftState {
                id: 0,
                epoch: Epoch::ZERO,
                translation: TranslationalState::from(ship_state),
                attitude: AttitudeState::default(),
                mass: MassState {
                    wet_mass_kg: ship_params.dry_mass_kg,
                    dry_mass_kg: ship_params.dry_mass_kg,
                    inertia_body_kg_m2: DMat3::IDENTITY,
                    center_of_mass_body_m: DVec3::ZERO,
                },
                resources: ResourceState,
                authority,
            },
            authority_book: CraftAuthorityBook::new(0, authority),
            sim_time: 0.0,
            propagator,
            max_real_delta: config.max_real_delta,
            max_transitions_per_frame: config.max_transitions_per_frame,
            ephemeris,
            bodies,
            maneuvers: ManeuverSequence::new(),
            consumed_node_ids: Vec::new(),
            target_body: None,
            delivered_dv: 0.0,
            ship_params,
            control: ControlInput::default(),
            warp: WarpController::new(config.warp_levels),
            prediction_state: PredictionState::new(
                config.prediction_config,
                config.prediction_stale_after,
            ),
        }
    }

    /// Advance the simulation by `real_dt` seconds of wall-clock time.
    ///
    /// Thrust and attitude no longer integrate here — they are owned by
    /// the Avian rigid body that the game crate spins up for every player
    /// craft, at every regime (deep space, orbit, surface). `step` keeps
    /// canonical state coherent under warp: `BodyFixed` evaluates pose at
    /// `sim_time`; `LocalRigidBody` is a no-op for translation (Avian
    /// reads back into canonical separately); `OnRails` /
    /// `WarpIntegrated` / `Docked` propagate the ship's coast across the
    /// warp-scaled time interval, breaking on SOI transitions until the
    /// cap is reached or the target time is hit.
    pub fn step(&mut self, real_dt: f64) {
        let _span = tracing::info_span!("Simulation::step").entered();
        let real_delta = real_dt.min(self.max_real_delta);
        self.warp.update(real_delta);

        let warp_speed = self.warp.speed();
        let sim_delta = real_delta * warp_speed;

        match self.craft.authority {
            AuthorityMode::BodyFixed { body, pose } => {
                if sim_delta > 0.0 {
                    self.sim_time += sim_delta;
                }
                self.craft.epoch = Epoch(self.sim_time);
                let body_state = self.ephemeris.state(body, Epoch(self.sim_time));
                let (translation, attitude) = evaluate_body_fixed_pose(&body_state, pose);
                self.craft.translation = translation;
                self.craft.attitude = attitude;
                return;
            }
            AuthorityMode::LocalRigidBody { .. } => {
                if sim_delta > 0.0 {
                    self.sim_time += sim_delta;
                }
                self.craft.epoch = Epoch(self.sim_time);
                return;
            }
            AuthorityMode::OnRails { .. }
            | AuthorityMode::WarpIntegrated { .. }
            | AuthorityMode::Docked { .. } => {}
        }

        if sim_delta <= 0.0 {
            return;
        }

        let target_time = self.sim_time + sim_delta;
        let mut transitions = 0u32;

        while self.sim_time < target_time {
            if transitions >= self.max_transitions_per_frame {
                break;
            }

            let soi_body = self.propagator.soi_body_of(
                self.craft.translation.position,
                self.sim_time,
                self.ephemeris.as_ref(),
                &self.bodies,
            );

            let result = self.propagator.coast_segment(CoastRequest {
                state: StateVector::from(self.craft.translation),
                time: self.sim_time,
                soi_body,
                target_time,
                stop_on_stable_orbit: false,
                // Enough samples to catch SOI crossings reliably at
                // typical warp rates without paying for big allocations
                // on every step. Samples are discarded immediately —
                // only the end state matters for live stepping.
                sample_count_hint: 32,
                ephemeris: self.ephemeris.as_ref(),
                bodies: &self.bodies,
            });

            self.craft.translation = TranslationalState::from(result.end_state);
            self.sim_time = result.end_time;
            self.craft.epoch = Epoch(self.sim_time);

            match result.terminator {
                SegmentTerminator::Collision { .. } => {
                    // Ship is wrecked — freeze the state here. A future
                    // pass can surface a `CollisionEvent` to the game.
                    break;
                }
                SegmentTerminator::SoiEnter { .. } | SegmentTerminator::SoiExit { .. } => {
                    transitions += 1;
                }
                SegmentTerminator::Horizon | SegmentTerminator::StableOrbit => {}
                SegmentTerminator::BurnEnd { .. } => {
                    // Coast segments don't fire BurnEnd; defensive only.
                    break;
                }
            }
        }
    }

    pub fn attitude(&self) -> &AttitudeState {
        &self.craft.attitude
    }

    pub fn set_attitude(&mut self, attitude: AttitudeState) {
        self.craft.attitude = attitude;
    }

    pub fn ship_params(&self) -> &ShipParameters {
        &self.ship_params
    }

    pub fn set_ship_params(&mut self, params: ShipParameters) {
        self.ship_params = params;
        self.craft.mass.dry_mass_kg = params.dry_mass_kg;
        self.craft.mass.inertia_body_kg_m2 = DMat3::from_diagonal(params.moment_of_inertia);
        // Re-floor mass at the new dry-mass invariant. Only raises mass
        // when the previous value was below `dry_mass_kg` (e.g. the
        // post-`new()` sentinel state); a partially-drained ship keeps
        // its current mass.
        if self.craft.mass.wet_mass_kg < self.ship_params.dry_mass_kg {
            self.craft.mass.wet_mass_kg = self.ship_params.dry_mass_kg;
        }
    }

    /// Current ship mass at `sim_time`, kg. Decreases as fuel burns,
    /// floored at `ship_params.dry_mass_kg`.
    pub fn ship_mass_kg(&self) -> f64 {
        self.craft.mass.wet_mass_kg
    }

    /// Push the ship's current mass — called by
    /// [`crate::simulation::Simulation::set_ship_mass`] each frame from
    /// `crates/game/src/fuel.rs` so the integrator runs on tank-derived
    /// truth, not its own internal estimate. Floored at `dry_mass_kg`.
    pub fn set_ship_mass(&mut self, mass_kg: f64) {
        self.craft.mass.wet_mass_kg = mass_kg.max(self.ship_params.dry_mass_kg);
    }

    /// Tsiolkovsky-aware estimate of the burn time required to deliver
    /// `delta_v_magnitude` of Δv from the ship's current state, seconds.
    /// Returns 0 when the ship has no thrust configured (the HUD treats
    /// that as "no engine") or when propellant is exhausted.
    pub fn estimated_burn_duration(&self, delta_v_magnitude: f64) -> f64 {
        burn_duration(
            delta_v_magnitude,
            self.ship_params.thrust_n,
            self.craft.mass.wet_mass_kg,
            self.ship_params.mass_flow_kg_per_s,
            self.ship_params.dry_mass_kg,
        )
    }

    pub fn set_control(&mut self, control: ControlInput) {
        self.control = control;
    }

    /// Update only the throttle field of the current [`ControlInput`].
    /// Attitude and throttle are produced by independent bridge
    /// systems; granular setters keep them from stomping each other.
    pub fn set_throttle(&mut self, throttle: f64) {
        self.control.throttle = throttle.clamp(0.0, 1.0);
    }

    pub fn control(&self) -> &ControlInput {
        &self.control
    }

    /// Drain propellant mass and accumulate delivered Δv for an external
    /// dynamics authority — i.e., the Avian rigid body that owns thrust in
    /// every regime. The game-side fuel reconciliation still pushes tank
    /// truth back into `Simulation`, so this only keeps the canonical craft
    /// mass coherent between those updates; Δv is also tracked here so the
    /// autopilot's burn-completion check (anchor + delta against
    /// [`Self::delivered_dv`]) keeps working without re-running thrust on
    /// the canonical side.
    pub fn apply_external_mass_flow(&mut self, throttle: f64, real_dt: f64) {
        if real_dt <= 0.0 || throttle <= 0.0 {
            return;
        }
        let throttle = throttle.clamp(0.0, 1.0);
        if self.ship_params.mass_flow_kg_per_s > 0.0 {
            let drained = self.ship_params.mass_flow_kg_per_s * throttle * real_dt;
            self.craft.mass.wet_mass_kg =
                (self.craft.mass.wet_mass_kg - drained).max(self.ship_params.dry_mass_kg);
        }
        if self.ship_params.thrust_n > 0.0 && self.craft.mass.wet_mass_kg > 0.0 {
            let accel_mag = self.ship_params.thrust_n / self.craft.mass.wet_mass_kg;
            self.delivered_dv += throttle * accel_mag * real_dt;
        }
    }

    // -- Accessors ----------------------------------------------------------

    pub fn ship_state(&self) -> StateVector {
        StateVector::from(self.craft.translation)
    }

    pub fn craft_state(&self) -> &CraftState {
        &self.craft
    }

    pub fn authority(&self) -> AuthorityMode {
        self.craft.authority
    }

    pub fn authority_log(&self) -> &[crate::canonical::AuthorityChanged] {
        &self.authority_book.log
    }

    pub fn transition_authority(&mut self, next: AuthorityMode) {
        self.authority_book
            .transition_to(Epoch(self.sim_time), next);
        self.craft.authority = self.authority_book.mode;
    }

    /// Replace the ship's state vector wholesale and invalidate the
    /// cached prediction. Intended for debug teleports — production code
    /// paths advance the ship through [`Self::step`] so that live and
    /// predicted trajectories stay numerically aligned.
    pub fn set_ship_state(&mut self, state: StateVector) {
        self.craft.translation = TranslationalState::from(state);
        self.craft.epoch = Epoch(self.sim_time);
        self.prediction_state.mark_dirty();
    }

    /// Install a state sampled from local physics. This is the canonical
    /// writeback boundary for `AuthorityMode::LocalRigidBody`.
    pub fn install_local_rigid_body_state(
        &mut self,
        translation: TranslationalState,
        attitude: AttitudeState,
    ) {
        self.craft.translation = translation;
        self.craft.attitude = attitude;
        self.craft.epoch = Epoch(self.sim_time);
        self.prediction_state.mark_dirty();
    }

    pub fn sim_time(&self) -> f64 {
        self.sim_time
    }

    /// SOI body the ship is currently inside — the innermost body whose
    /// sphere of influence contains the live ship position. The autopilot
    /// uses this as the reference frame for prograde/normal/radial modes
    /// so they always point relative to "what we're orbiting now."
    pub fn dominant_body(&self) -> BodyId {
        self.propagator.soi_body_of(
            self.craft.translation.position,
            self.sim_time,
            self.ephemeris.as_ref(),
            &self.bodies,
        )
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

    /// Remove the maneuver node with the given `id` from the schedule,
    /// queue its id for the bridge to retire on the UI side, and dirty
    /// prediction. Once a directive starts physically burning, it is no
    /// longer future plan input; prediction must rebuild from the live
    /// ship state plus any still-future nodes.
    ///
    /// Returns `true` if a node was found and removed.
    pub fn consume_maneuver_node(&mut self, id: u64) -> bool {
        let Some(idx) = self.maneuvers.nodes.iter().position(|n| n.id == Some(id)) else {
            return false;
        };
        self.maneuvers.nodes.remove(idx);
        self.consumed_node_ids.push(id);
        self.prediction_state.mark_dirty();
        true
    }

    /// Lifetime cumulative magnitude of Δv accumulated through
    /// [`Self::apply_external_mass_flow`], in m/s. The game-side autopilot
    /// reads this at burn-start (anchor) and each subsequent frame to
    /// compute "Δv delivered since burn start = current − anchor".
    pub fn delivered_dv(&self) -> f64 {
        self.delivered_dv
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
            ship_state: StateVector::from(self.craft.translation),
            sim_time: self.sim_time,
            maneuvers: self.maneuvers.clone(),
            ephemeris: Arc::clone(&self.ephemeris),
            propagator: Arc::clone(&self.propagator),
            bodies: self.bodies.clone(),
            prediction_config: self.prediction_state.config().clone(),
            ship_thrust_n: self.ship_params.thrust_n,
            ship_mass_kg: self.craft.mass.wet_mass_kg,
            ship_mass_flow_kg_per_s: self.ship_params.mass_flow_kg_per_s,
            ship_dry_mass_kg: self.ship_params.dry_mass_kg,
            target_body: self.target_body,
        };
        let branches = propagate_branch_stack(&req, None);
        self.prediction_state.install(branches, req.sim_time);
    }

    /// Drop the cached trajectory prediction and mark it dirty. Used when
    /// the ship is in a non-ballistic regime (landed, or in surface
    /// contact under Avian) — feeding a contact-affected velocity into
    /// Keplerian propagation produces a wobbling line that doesn't
    /// reflect what the ship will actually do, so the gate hides it.
    pub fn clear_prediction(&mut self) {
        self.prediction_state.clear();
    }

    pub fn prediction_needs_refresh(&self) -> bool {
        self.prediction_state.needs_refresh(self.sim_time)
    }

    pub fn prediction(&self) -> Option<&FlightPlan> {
        self.prediction_state.prediction()
    }

    pub fn trajectory_branches(&self) -> Option<&TrajectoryBranchStack> {
        self.prediction_state.branch_stack()
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

    pub fn ephemeris(&self) -> &dyn BodyTrajectoryProvider {
        self.ephemeris.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity_mode::GravityMode;
    use crate::types::{BodyDefinition, BodyKind, ShipDefinition, SolarSystemDefinition};
    use std::collections::HashMap;

    fn ctrl() -> WarpController {
        WarpController::new(vec![0.0, 1.0, 10.0, 100.0, 1_000.0])
    }

    fn minimal_simulation() -> Simulation {
        let bodies = vec![BodyDefinition {
            id: 0,
            name: "Pyros".to_string(),
            kind: BodyKind::Star,
            parent: None,
            mass_kg: 1.0e30,
            radius_m: 1.0e8,
            color: [1.0, 1.0, 1.0],
            rotation_period_s: 0.0,
            axial_tilt_rad: 0.0,
            gm: crate::types::G * 1.0e30,
            soi_radius_m: f64::INFINITY,
            orbital_elements: None,
            terrain: thalos_terrain_gen::TerrainConfig::None,
            tectonics: None,
            atmosphere: None,
            terrestrial_atmosphere: None,
            rings: None,
        }];
        let system = SolarSystemDefinition {
            name: "Test".to_string(),
            bodies: bodies.clone(),
            ship: ShipDefinition {
                initial_state: StateVector {
                    position: DVec3::X * 1.0e9,
                    velocity: DVec3::Z * 1000.0,
                },
            },
            name_to_id: HashMap::from([("Pyros".to_string(), 0)]),
        };
        let impls = GravityMode::PatchedConics.build(&system, 1.0e6);
        Simulation::new(
            system.ship.initial_state,
            impls,
            bodies,
            SimulationConfig::default(),
        )
    }

    #[test]
    fn set_speed_picks_highest_level_at_or_below_target() {
        let mut w = ctrl();
        w.set_speed(50.0);
        assert_eq!(w.target_speed(), 10.0);
        w.update(WARP_TRANSITION_DURATION_S);
        assert_eq!(w.speed(), 10.0);
        w.set_speed(99.999);
        assert_eq!(w.target_speed(), 10.0);
        assert_eq!(w.speed(), 10.0);
        w.set_speed(100.0);
        assert_eq!(w.target_speed(), 100.0);
        w.update(WARP_TRANSITION_DURATION_S);
        assert_eq!(w.speed(), 100.0);
        w.set_speed(1e9);
        assert_eq!(w.target_speed(), 1_000.0);
        w.update(WARP_TRANSITION_DURATION_S);
        assert_eq!(w.speed(), 1_000.0);
    }

    #[test]
    fn set_speed_below_lowest_falls_back_to_pause() {
        let mut w = ctrl();
        w.increase();
        w.increase();
        w.update(WARP_TRANSITION_DURATION_S);
        assert!(w.speed() > 0.0);
        w.set_speed(-1.0);
        assert_eq!(w.speed(), 0.0);
    }

    #[test]
    fn set_speed_clears_resume_index() {
        // Pause from 100× stashes that as the resume target. set_speed
        // should wipe it, so the next pause→resume cycle defaults to 1×
        // instead of resurrecting 100×.
        let mut w = ctrl();
        w.increase();
        w.increase();
        w.increase();
        w.update(WARP_TRANSITION_DURATION_S);
        assert_eq!(w.speed(), 100.0);
        w.toggle_pause();
        assert_eq!(w.speed(), 0.0);
        w.set_speed(0.0);
        w.toggle_pause();
        assert_eq!(w.target_speed(), 1.0);
        w.update(WARP_TRANSITION_DURATION_S);
        assert_eq!(w.speed(), 1.0);
    }

    #[test]
    fn reset_immediate_snaps_from_pause_to_one_x() {
        let mut w = ctrl();
        assert_eq!(w.speed(), 0.0);
        w.reset_immediate();
        assert_eq!(w.target_speed(), 1.0);
        assert_eq!(w.speed(), 1.0);
    }

    #[test]
    fn transitions_through_paused_level_snap() {
        // Lerping through the paused (0×) level would freeze translation
        // for the lerp duration while sim-time advances under the warp,
        // dropping orbital motion on the floor. 0↔non-zero must snap.
        let mut w = ctrl();
        w.set_speed(1.0);
        assert_eq!(w.speed(), 1.0); // 0 → 1 snaps, even though target>speed

        w.set_speed(0.0);
        assert_eq!(w.speed(), 0.0); // 1 → 0 snaps too

        w.increase();
        assert_eq!(w.speed(), 1.0); // increase() out of pause snaps
    }

    #[test]
    fn warp_increase_lerps_effective_speed_between_levels() {
        let mut w = ctrl();
        w.increase();
        w.update(WARP_TRANSITION_DURATION_S);
        assert_eq!(w.speed(), 1.0);

        w.increase();
        assert_eq!(w.target_speed(), 10.0);
        assert_eq!(w.speed(), 1.0);

        w.update(WARP_TRANSITION_DURATION_S * 0.5);
        assert!(w.speed() > 1.0 && w.speed() < 10.0);

        w.update(WARP_TRANSITION_DURATION_S);
        assert_eq!(w.speed(), 10.0);
    }

    #[test]
    fn warp_decrease_lerps_effective_speed_between_levels() {
        let mut w = ctrl();
        w.set_speed(100.0);
        w.update(WARP_TRANSITION_DURATION_S);
        assert_eq!(w.speed(), 100.0);

        w.decrease();
        assert_eq!(w.target_speed(), 10.0);
        assert_eq!(w.speed(), 100.0);

        w.update(WARP_TRANSITION_DURATION_S * 0.5);
        assert!(w.speed() > 10.0 && w.speed() < 100.0);

        w.update(WARP_TRANSITION_DURATION_S);
        assert_eq!(w.speed(), 10.0);
    }

    #[test]
    fn capped_speed_reduction_snaps_effective_speed_downward() {
        let mut w = ctrl();
        w.set_speed(1_000.0);
        w.update(WARP_TRANSITION_DURATION_S);
        assert_eq!(w.speed(), 1_000.0);

        w.set_speed(50.0);
        assert_eq!(w.target_speed(), 10.0);
        assert_eq!(w.speed(), 10.0);
    }

    #[test]
    fn simulation_exposes_one_canonical_craft_with_one_authority() {
        let mut sim = minimal_simulation();

        assert_eq!(sim.craft_state().id, 0);
        assert_eq!(sim.craft_state().authority, sim.authority());
        assert_eq!(sim.authority_log().len(), 0);

        sim.transition_authority(AuthorityMode::WarpIntegrated { integrator: 1 });

        assert_eq!(
            sim.craft_state().authority,
            AuthorityMode::WarpIntegrated { integrator: 1 }
        );
        assert_eq!(sim.authority_log().len(), 1);
    }

    #[test]
    fn authority_log_records_local_landing_handoff_path() {
        let mut sim = minimal_simulation();

        sim.transition_authority(AuthorityMode::LocalRigidBody {
            bubble: 7,
            root_entity: crate::canonical::EntityRef(11),
        });
        sim.transition_authority(AuthorityMode::BodyFixed {
            body: 0,
            pose: crate::canonical::BodyFixedPose {
                position_body_m: DVec3::Y * 1000.0,
                orientation_body: DQuat::IDENTITY,
            },
        });

        let log = sim.authority_log();
        assert_eq!(log.len(), 2);
        assert!(matches!(log[0].from, AuthorityMode::OnRails { .. }));
        assert!(matches!(log[0].to, AuthorityMode::LocalRigidBody { .. }));
        assert!(matches!(log[1].from, AuthorityMode::LocalRigidBody { .. }));
        assert!(matches!(log[1].to, AuthorityMode::BodyFixed { .. }));
    }

    #[test]
    fn consuming_maneuver_node_removes_future_input_and_dirties_prediction() {
        let mut sim = minimal_simulation();
        sim.maneuvers_mut()
            .nodes
            .push(crate::maneuver::ManeuverNode {
                id: Some(42),
                time: 120.0,
                delta_v: DVec3::X,
                reference_body: 0,
            });
        sim.recompute_prediction();
        assert!(!sim.prediction_needs_refresh());

        assert!(sim.consume_maneuver_node(42));

        assert!(sim.maneuvers().nodes.is_empty());
        assert_eq!(sim.drain_consumed_node_ids(), vec![42]);
        assert!(sim.prediction_needs_refresh());
    }
}
