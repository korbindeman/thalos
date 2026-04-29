//! Central simulation struct that owns all physics state and logic.
//!
//! The game crate becomes a thin consumer: it calls [`Simulation::step`] each
//! frame and reads back the ship state, prediction, and warp info. Live
//! stepping and trajectory prediction both route through the same
//! [`ShipPropagator`], so "where the ship is" and "where it is predicted to
//! be" cannot drift numerically.

use std::sync::Arc;

use glam::{DQuat, DVec3};

use crate::body_state_provider::BodyStateProvider;
use crate::gravity_mode::GravityImpls;
use crate::maneuver::{ManeuverSequence, burn_duration};
use crate::ship_propagator::{CoastRequest, SegmentTerminator, ShipPropagator};
use crate::trajectory::{FlightPlan, PredictionConfig, PredictionRequest, propagate_flight_plan};
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
    /// Current ship mass at `sim_time`. Decreases as fuel burns via
    /// per-frame impulse in [`Self::apply_live_thrust`]. Floored at
    /// `ship_params.dry_mass_kg` — once propellant exhausts, thrust
    /// gates off cleanly and mass stops decreasing.
    ship_mass_kg: f64,
    maneuvers: ManeuverSequence,
    consumed_node_ids: Vec<u64>,
    target_body: Option<BodyId>,

    /// Lifetime cumulative magnitude of Δv applied through
    /// [`Self::apply_live_thrust`], in m/s. Sums `|throttle · F/m · dt|`
    /// over every frame the engine fires. Read by the burn-execution
    /// autopilot (see `crates/game/src/autopilot.rs`) to detect when a
    /// finite burn has delivered its planned Δv — magnitude rather than
    /// projection because magnitude tracks the same scalar that fuel
    /// exhaustion clamps, so a propellant-starved burn stops
    /// accumulating without special-casing.
    delivered_dv: f64,

    attitude: AttitudeState,
    ship_params: ShipParameters,
    control: ControlInput,

    /// When `true`, the burn-execution autopilot in
    /// `crates/game/src/autopilot.rs` is allowed to drop warp, point
    /// the ship, and open the throttle to fly upcoming maneuver nodes.
    /// When `false`, nodes sit in `maneuvers` as planning aids and the
    /// player must execute them manually. The flag is read by the
    /// autopilot system; the simulation itself does not consume nodes.
    auto_maneuvers_enabled: bool,

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
            body_state: ephemeris,
            ship_propagator: propagator,
        } = impls;

        let ship_params = ShipParameters::default();
        Self {
            ship_state,
            sim_time: 0.0,
            propagator,
            max_real_delta: config.max_real_delta,
            max_transitions_per_frame: config.max_transitions_per_frame,
            ephemeris,
            bodies,
            ship_mass_kg: ship_params.dry_mass_kg,
            maneuvers: ManeuverSequence::new(),
            consumed_node_ids: Vec::new(),
            target_body: None,
            delivered_dv: 0.0,
            attitude: AttitudeState::default(),
            ship_params,
            control: ControlInput::default(),
            auto_maneuvers_enabled: true,
            warp: WarpController::new(config.warp_levels),
            prediction_state: PredictionState::new(
                config.prediction_config,
                config.prediction_stale_after,
            ),
        }
    }

    /// Advance the simulation by `real_dt` seconds of wall-clock time.
    ///
    /// Two parts in order: (1) at 1× warp only, advance attitude and
    /// apply any live engine thrust as a per-frame impulse on velocity;
    /// (2) propagate the ship's coast across the warp-scaled time
    /// interval, breaking on SOI transitions until the cap is reached
    /// or the target time is hit. Maneuver execution is owned by the
    /// burn-execution autopilot in the game crate — it points the
    /// ship, drives the throttle, and consumes nodes through
    /// [`Self::consume_maneuver_node`] — so this function never reads
    /// the maneuver list or schedules burns of its own.
    pub fn step(&mut self, real_dt: f64) {
        let _span = tracing::info_span!("Simulation::step").entered();
        let real_delta = real_dt.min(self.max_real_delta);

        // Attitude advances on real-time dt and only at 1× warp; under
        // any other warp (including pause) we zero ω so the ship doesn't
        // smear orientation across the warp gap.
        self.integrate_attitude(real_delta);

        // Live engine thrust. Treated as a single per-frame impulse on
        // velocity — at 1× warp the 16 ms frame window is small enough
        // that impulse-then-coast is numerically equivalent to RK4 with
        // body-frame thrust to well below noise. The autopilot keeps
        // warp at 1× while a burn is in flight, so this is the only
        // thrust path.
        self.apply_live_thrust(real_delta);

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

            let soi_body = self.propagator.soi_body_of(
                self.ship_state.position,
                self.sim_time,
                self.ephemeris.as_ref(),
                &self.bodies,
            );

            let result = self.propagator.coast_segment(CoastRequest {
                state: self.ship_state,
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
                SegmentTerminator::Horizon | SegmentTerminator::StableOrbit => {}
                SegmentTerminator::BurnEnd { .. } => {
                    // Coast segments don't fire BurnEnd; defensive only.
                    break;
                }
            }
        }
    }

    // -- Attitude -----------------------------------------------------------

    /// Integrate ship orientation forward by `real_dt` seconds.
    ///
    /// Skips entirely (and zeros ω) at any warp other than 1×, including
    /// pause. Player rotation has no meaning while time is warped, and
    /// allowing ω to persist would let a ship spin up at warp entry and
    /// keep tumbling out of warp.
    ///
    /// Uses Euler integration with quaternion renormalization. The
    /// gyroscopic term `ω × Iω` is omitted: at the angular rates a
    /// player-controlled ship reaches it is negligible, and ignoring it
    /// keeps the integrator a single line. Add Euler's equations if a
    /// future scenario demands tumbling-asymmetric-body fidelity.
    fn integrate_attitude(&mut self, real_dt: f64) {
        if (self.warp.speed() - 1.0).abs() > f64::EPSILON {
            self.attitude.angular_velocity = DVec3::ZERO;
            return;
        }
        if real_dt <= 0.0 {
            return;
        }

        let dt = real_dt;
        let i = self.ship_params.moment_of_inertia;
        let tau_max = self.ship_params.max_torque;

        let cmd = self
            .control
            .torque_command
            .clamp(DVec3::splat(-1.0), DVec3::splat(1.0));
        let player_torque = cmd * tau_max;

        let no_input = cmd.length_squared() < 1e-6;
        let torque = if self.control.sas_enabled && no_input {
            // Deadbeat rate damping: choose τ to zero ω in one step,
            // then clamp to per-axis max. Stable: when the unclamped
            // value exceeds the cap, the impulse `τ·dt = −Iω` stays
            // bounded, so ω still trends to zero in finite time.
            (-i * self.attitude.angular_velocity / dt).clamp(-tau_max, tau_max)
        } else {
            player_torque
        };

        // Per-axis 1/I, guarding the degenerate (no-mass) axis so that
        // ShipParameters::default() doesn't NaN the integrator.
        let inv_i = DVec3::new(
            if i.x > 0.0 { 1.0 / i.x } else { 0.0 },
            if i.y > 0.0 { 1.0 / i.y } else { 0.0 },
            if i.z > 0.0 { 1.0 / i.z } else { 0.0 },
        );
        self.attitude.angular_velocity += torque * inv_i * dt;

        let omega = self.attitude.angular_velocity;
        if omega.length_squared() > 0.0 {
            let delta = DQuat::from_scaled_axis(omega * dt);
            self.attitude.orientation = (self.attitude.orientation * delta).normalize();
        }
    }

    pub fn attitude(&self) -> &AttitudeState {
        &self.attitude
    }

    pub fn set_attitude(&mut self, attitude: AttitudeState) {
        self.attitude = attitude;
    }

    pub fn ship_params(&self) -> &ShipParameters {
        &self.ship_params
    }

    pub fn set_ship_params(&mut self, params: ShipParameters) {
        self.ship_params = params;
        // Re-floor mass at the new dry-mass invariant. Only raises mass
        // when the previous value was below `dry_mass_kg` (e.g. the
        // post-`new()` sentinel state); a partially-drained ship keeps
        // its current mass.
        if self.ship_mass_kg < self.ship_params.dry_mass_kg {
            self.ship_mass_kg = self.ship_params.dry_mass_kg;
        }
    }

    /// Current ship mass at `sim_time`, kg. Decreases as fuel burns,
    /// floored at `ship_params.dry_mass_kg`.
    pub fn ship_mass_kg(&self) -> f64 {
        self.ship_mass_kg
    }

    /// Push the ship's current mass — called by
    /// [`crate::simulation::Simulation::set_ship_mass`] each frame from
    /// `crates/game/src/fuel.rs` so the integrator runs on tank-derived
    /// truth, not its own internal estimate. Floored at `dry_mass_kg`.
    pub fn set_ship_mass(&mut self, mass_kg: f64) {
        self.ship_mass_kg = mass_kg.max(self.ship_params.dry_mass_kg);
    }

    /// Tsiolkovsky-aware estimate of the burn time required to deliver
    /// `delta_v_magnitude` of Δv from the ship's current state, seconds.
    /// Returns 0 when the ship has no thrust configured (the HUD treats
    /// that as "no engine") or when propellant is exhausted.
    pub fn estimated_burn_duration(&self, delta_v_magnitude: f64) -> f64 {
        burn_duration(
            delta_v_magnitude,
            self.ship_params.thrust_n,
            self.ship_mass_kg,
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

    pub fn auto_maneuvers_enabled(&self) -> bool {
        self.auto_maneuvers_enabled
    }

    pub fn set_auto_maneuvers_enabled(&mut self, enabled: bool) {
        if self.auto_maneuvers_enabled != enabled {
            self.auto_maneuvers_enabled = enabled;
            // The cached prediction was built assuming the prior auto
            // policy — flipping it changes which scheduled burns fire,
            // so the prediction must be rebuilt to stay coherent.
            self.prediction_state.mark_dirty();
        }
    }

    /// Apply engine thrust as a single impulse on velocity. Manual
    /// throttle and the burn-execution autopilot both feed in through
    /// the same path — both end up writing `control.throttle` and
    /// reading the ship's current attitude.
    ///
    /// Skipped at any warp other than 1× (mirrors attitude integration).
    /// The autopilot is responsible for ensuring 1× warp before opening
    /// the throttle on a scheduled burn.
    ///
    /// **Body-frame nose convention**: `+Y_body` points out the nose,
    /// shared with the autopilot in `crates/game/src/navigation.rs`.
    /// Flipping this convention requires updating both call sites.
    ///
    /// Accumulates `|throttle · F/m · dt|` into [`Self::delivered_dv`]
    /// every frame the engine fires, so the burn-execution autopilot can
    /// detect when a planned-Δv has been delivered and close the
    /// throttle. Magnitude (not projection onto a planned axis) means
    /// the same scalar drops to zero when the dry-mass clamp cuts
    /// thrust — fuel exhaustion stops the accumulator without any
    /// special case.
    ///
    /// Does *not* mark the prediction dirty — that would rebuild the
    /// flight plan every frame of a multi-second burn (expensive). The
    /// bridge detects the throttle-falling edge and dirties once on
    /// engine cut, letting the trail snap to reality without per-frame
    /// rebuild churn during the burn.
    fn apply_live_thrust(&mut self, real_dt: f64) {
        if (self.warp.speed() - 1.0).abs() > f64::EPSILON {
            return;
        }
        if real_dt <= 0.0 || self.control.throttle <= 0.0 {
            return;
        }
        if self.ship_params.thrust_n <= 0.0 {
            return;
        }
        // Out of fuel — engine cuts off, no thrust applied this frame.
        // Mass is capped at `dry_mass_kg` so this is the physical
        // "tanks empty" condition. The same frame's `fuel.rs` system
        // gates throttle to 0 in this case anyway, but checking here
        // keeps the simulation self-consistent.
        if self.ship_mass_kg <= self.ship_params.dry_mass_kg {
            return;
        }
        let throttle = self.control.throttle.clamp(0.0, 1.0);
        let nose_world = self.attitude.orientation * DVec3::Y;
        let accel_mag = self.ship_params.thrust_n / self.ship_mass_kg;
        let dv_mag = throttle * accel_mag * real_dt;
        self.ship_state.velocity += dv_mag * nose_world;
        self.delivered_dv += dv_mag;
        // Drain mass at the gated throttle the bridge has already applied
        // (which matches what `crates/game/src/fuel.rs` is draining from
        // the tanks this same frame, so sim mass and tank mass stay in
        // sync between fuel.rs's per-frame mass-push reconciliation).
        let drained = self.ship_params.mass_flow_kg_per_s * throttle * real_dt;
        self.ship_mass_kg = (self.ship_mass_kg - drained).max(self.ship_params.dry_mass_kg);
    }

    // -- Accessors ----------------------------------------------------------

    pub fn ship_state(&self) -> &StateVector {
        &self.ship_state
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
            self.ship_state.position,
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

    /// Remove the maneuver node with the given `id` from the schedule
    /// and queue its id for the bridge to retire on the UI side. Used
    /// by the burn-execution autopilot at burn start, before the live
    /// thrust on this node begins.
    ///
    /// Does *not* dirty the prediction. The cached flight plan was
    /// built showing this node firing as planned; if we dirtied here
    /// it would recompute mid-burn and show a "no upcoming burn"
    /// trajectory while the ship is mid-thrust. The throttle-falling-
    /// edge dirty in `crates/game/src/fuel.rs` handles the post-burn
    /// rebuild once the engine cuts.
    ///
    /// Returns `true` if a node was found and removed.
    pub fn consume_maneuver_node(&mut self, id: u64) -> bool {
        let Some(idx) = self.maneuvers.nodes.iter().position(|n| n.id == Some(id)) else {
            return false;
        };
        self.maneuvers.nodes.remove(idx);
        self.consumed_node_ids.push(id);
        true
    }

    /// Lifetime cumulative magnitude of Δv applied through
    /// [`Self::apply_live_thrust`], in m/s. The autopilot reads this
    /// at burn-start (anchor) and each subsequent frame to compute
    /// "Δv delivered since burn start = current − anchor".
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
            ship_state: self.ship_state,
            sim_time: self.sim_time,
            maneuvers: self.maneuvers.clone(),
            ephemeris: Arc::clone(&self.ephemeris),
            propagator: Arc::clone(&self.propagator),
            bodies: self.bodies.clone(),
            prediction_config: self.prediction_state.config().clone(),
            ship_thrust_n: self.ship_params.thrust_n,
            ship_mass_kg: self.ship_mass_kg,
            ship_mass_flow_kg_per_s: self.ship_params.mass_flow_kg_per_s,
            ship_dry_mass_kg: self.ship_params.dry_mass_kg,
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
