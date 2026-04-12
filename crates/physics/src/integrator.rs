//! Hybrid symplectic / adaptive RK45 integrator for N-body orbital mechanics.
//!
//! # Design
//!
//! Two integration strategies are maintained and switched based on the ratio of
//! perturbation accelerations to the dominant body's gravitational acceleration:
//!
//! - **Velocity-Verlet (symplectic leapfrog)** — energy-conserving, fixed step,
//!   used when one body dominates and perturbations are small.
//! - **Dormand-Prince RK45** — variable-step Runge-Kutta with PI step-size
//!   control, used when perturbations are significant.
//!
//! Switching uses hysteresis: the integrator upgrades to RK45 when the
//! perturbation ratio exceeds `switch_threshold`, and only downgrades back to
//! Verlet when the ratio falls below `switch_threshold * hysteresis_factor`.
//! This prevents rapid toggling at the boundary.
//!
//! Per-body gravitational accelerations are computed directly from the ephemeris
//! `BodyStates` at each step to identify the dominant body and the perturbation
//! ratio — both stored in the returned `TrajectorySample` metadata.

use glam::DVec3;

use crate::body_state_provider::BodyStateProvider;
use crate::forces::{ForceRegistry, GravityForce};
use crate::types::{BodyStates, StateVector, TrajectorySample};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tuning parameters for the hybrid integrator.
#[derive(Debug, Clone)]
pub struct IntegratorConfig {
    /// Fixed time-step for the symplectic integrator (seconds). Default: 60 s.
    pub symplectic_dt: f64,
    /// Initial adaptive time-step when entering RK45 mode (seconds). Default: 60 s.
    pub rk_initial_dt: f64,
    /// Minimum allowed adaptive step size (seconds). Default: 0.01 s.
    pub rk_min_dt: f64,
    /// Maximum allowed adaptive step size (seconds). Default: 3600 s.
    pub rk_max_dt: f64,
    /// Local error tolerance for RK45 step acceptance. Default: 1e-6.
    ///
    /// Tighter tolerances force smaller steps (more accurate, slower). 1e-6
    /// keeps per-step position error well below any renderable scale while
    /// letting the integrator take larger steps in benign regions — roughly
    /// halves the cost of a full prediction pass vs 1e-9 with no visible
    /// difference in the drawn trajectory.
    pub rk_tolerance: f64,
    /// Perturbation ratio above which the integrator switches to RK45. Default: 0.01.
    pub switch_threshold: f64,
    /// Multiplier for the switch-back threshold (hysteresis). The integrator
    /// returns to Verlet when ratio < `switch_threshold * hysteresis_factor`.
    /// Default: 0.5 — i.e., switch back at 0.005.
    pub hysteresis_factor: f64,
}

impl Default for IntegratorConfig {
    fn default() -> Self {
        Self {
            symplectic_dt: 60.0,
            rk_initial_dt: 60.0,
            rk_min_dt: 0.01,
            rk_max_dt: 3600.0,
            rk_tolerance: 1e-6,
            switch_threshold: 0.01,
            hysteresis_factor: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Integrator mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Symplectic,
    Adaptive,
}

// ---------------------------------------------------------------------------
// Dormand-Prince RK45 Butcher tableau
// ---------------------------------------------------------------------------
//
// Coefficients from:
//   Dormand, J.R.; Prince, P.J. (1980). "A family of embedded Runge-Kutta
//   formulae". Journal of Computational and Applied Mathematics. 6 (1): 19–26.
//
// The tableau is:
//   c2=1/5,   c3=3/10,  c4=4/5,   c5=8/9,   c6=1
//
//   (b) 5th-order solution weights:
//     35/384, 0, 500/1113, 125/192, −2187/6784, 11/84, 0
//   (b*) 4th-order weights (error estimation only):
//     5179/57600, 0, 7571/16695, 393/640, −92097/339200, 187/2100, 1/40
//   Error weights e = b − b*:
//     71/57600, 0, −71/16695, 71/1920, −17253/339200, 22/525, −1/40

struct Dp45;

impl Dp45 {
    // Node coefficients (c_i).
    const C2: f64 = 1.0 / 5.0;
    const C3: f64 = 3.0 / 10.0;
    const C4: f64 = 4.0 / 5.0;
    const C5: f64 = 8.0 / 9.0;
    // c6 = 1, c7 = 1 (used implicitly as t + h)

    // Stage 2 matrix entry.
    const A21: f64 = 1.0 / 5.0;

    // Stage 3 matrix entries.
    const A31: f64 = 3.0 / 40.0;
    const A32: f64 = 9.0 / 40.0;

    // Stage 4 matrix entries.
    const A41: f64 = 44.0 / 45.0;
    const A42: f64 = -56.0 / 15.0;
    const A43: f64 = 32.0 / 9.0;

    // Stage 5 matrix entries.
    const A51: f64 = 19372.0 / 6561.0;
    const A52: f64 = -25360.0 / 2187.0;
    const A53: f64 = 64448.0 / 6561.0;
    const A54: f64 = -212.0 / 729.0;

    // Stage 6 matrix entries.
    const A61: f64 = 9017.0 / 3168.0;
    const A62: f64 = -355.0 / 33.0;
    const A63: f64 = 46732.0 / 5247.0;
    const A64: f64 = 49.0 / 176.0;
    const A65: f64 = -5103.0 / 18656.0;

    // 5th-order solution weights (b_i).
    const B1: f64 = 35.0 / 384.0;
    // B2 = 0
    const B3: f64 = 500.0 / 1113.0;
    const B4: f64 = 125.0 / 192.0;
    const B5: f64 = -2187.0 / 6784.0;
    const B6: f64 = 11.0 / 84.0;
    // B7 = 0  (FSAL: k7 at step n is k1 at step n+1; not exploited here for
    //          clarity — saving 1 evaluation out of 7 is a minor optimisation)

    // Error coefficient weights e_i = b_i − b*_i.
    const E1: f64 = 71.0 / 57600.0;
    // E2 = 0
    const E3: f64 = -71.0 / 16695.0;
    const E4: f64 = 71.0 / 1920.0;
    const E5: f64 = -17253.0 / 339200.0;
    const E6: f64 = 22.0 / 525.0;
    const E7: f64 = -1.0 / 40.0;
}

// ---------------------------------------------------------------------------
// PI step-size controller
// ---------------------------------------------------------------------------

/// PI controller for adaptive step-size selection (Hairer et al., §IV.2).
///
/// Update rule:
///   h_new = h * safety * (tol/err)^alpha * (tol/err_prev)^beta
///
/// with alpha = 0.7/5, beta = 0.4/5 for order-5 methods.
struct PiController {
    err_prev: f64,
}

impl PiController {
    const ALPHA: f64 = 0.7 / 5.0;
    const BETA: f64 = 0.4 / 5.0;
    const SAFETY: f64 = 0.9;
    const MAX_FACTOR: f64 = 10.0;
    const MIN_FACTOR: f64 = 0.1;

    fn new() -> Self {
        // Initialise prev error to 1 so the first accepted step gets a neutral
        // integral contribution.
        Self { err_prev: 1.0 }
    }

    /// Compute the factor to multiply the current step size by.
    ///
    /// `err` is the normalised error (step accepted if err ≤ 1).
    /// The factor is clamped to `[MIN_FACTOR, MAX_FACTOR]`.
    fn factor(&mut self, err: f64) -> f64 {
        let safe_err = err.max(f64::EPSILON);
        let safe_prev = self.err_prev.max(f64::EPSILON);
        let factor = Self::SAFETY * safe_err.powf(-Self::ALPHA) * safe_prev.powf(Self::BETA);
        self.err_prev = err;
        factor.clamp(Self::MIN_FACTOR, Self::MAX_FACTOR)
    }
}

// ---------------------------------------------------------------------------
// Public integrator struct
// ---------------------------------------------------------------------------

/// Hybrid symplectic / adaptive RK45 N-body integrator.
///
/// Call [`Integrator::step`] repeatedly to advance a trajectory one step at a
/// time.  Each call returns both the new state and a [`TrajectorySample`] with
/// metadata about the step taken.
pub struct Integrator {
    config: IntegratorConfig,
    mode: Mode,
    /// Current adaptive step size.  Persists across mode transitions so a
    /// return to RK45 resumes from a sensible starting guess.
    rk_dt: f64,
    pi: PiController,
    /// Consecutive RK45 steps accepted unconditionally at `rk_min_dt`.
    min_step_streak: u32,
    body_states_buf: BodyStates,
    sample_body_states_buf: BodyStates,
}

impl Integrator {
    pub fn new(config: IntegratorConfig) -> Self {
        let rk_dt = config.rk_initial_dt;
        Self {
            config,
            mode: Mode::Symplectic,
            rk_dt,
            pi: PiController::new(),
            min_step_streak: 0,
            body_states_buf: Vec::new(),
            sample_body_states_buf: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Public step API
    // -----------------------------------------------------------------------

    /// Advance the propagated body by one integration step.
    ///
    /// # Arguments
    /// * `state`     — Current position and velocity.
    /// * `time`      — Current simulation time (seconds from epoch).
    /// * `forces`    — Active force registry (gravity, thrust, …).
    /// * `ephemeris` — Background body states at any time.
    ///
    /// # Returns
    /// `(new_state, sample)` where `sample` carries metadata for this step
    /// (time at *end* of step, mode used, step size, dominant body, ratio).
    pub fn step(
        &mut self,
        state: StateVector,
        time: f64,
        forces: &ForceRegistry,
        ephemeris: &dyn BodyStateProvider,
    ) -> (StateVector, TrajectorySample) {
        self.step_capped(state, time, forces, ephemeris, f64::INFINITY)
    }

    /// Like [`Self::step`] but clamps the accepted step so the sample never
    /// lands past `max_step`. This is required for prediction segments, where
    /// maneuver timing and leg boundaries must stay exact.
    pub fn step_capped(
        &mut self,
        state: StateVector,
        time: f64,
        forces: &ForceRegistry,
        ephemeris: &dyn BodyStateProvider,
        max_step: f64,
    ) -> (StateVector, TrajectorySample) {
        // Query body states at the current time.  Used for the switching
        // decision and, in Verlet mode, the first force evaluation.
        ephemeris.query_into(time, &mut self.body_states_buf);

        // Identify the dominant body and the perturbation ratio; this drives
        // the mode-switching logic.  Uses a single-pass gravity analysis that
        // also computes the acceleration (avoiding a redundant O(n) scan).
        let gravity_result =
            GravityForce::compute_with_analysis(state.position, &self.body_states_buf);

        // Update mode with hysteresis.
        self.update_mode(gravity_result.perturbation_ratio);

        // Perform the integration step in the selected mode.
        let (new_state, step_size) = match self.mode {
            Mode::Symplectic => {
                let dt = self.config.symplectic_dt.min(max_step);
                let a0 = forces.compute_acceleration(
                    state.position,
                    state.velocity,
                    time,
                    &self.body_states_buf,
                );
                let s = self.verlet_step(state, time, dt, forces, a0, ephemeris);
                self.min_step_streak = 0;
                (s, dt)
            }
            Mode::Adaptive => {
                let (s, dt) = self.rk45_step(state, time, forces, ephemeris, max_step);
                (s, dt)
            }
        };

        ephemeris.query_into(time + step_size, &mut self.sample_body_states_buf);
        let sample_gravity =
            GravityForce::compute_with_analysis(new_state.position, &self.sample_body_states_buf);

        let sample = TrajectorySample {
            time: time + step_size,
            position: new_state.position,
            velocity: new_state.velocity,
            dominant_body: sample_gravity.dominant_body,
            perturbation_ratio: sample_gravity.perturbation_ratio,
            step_size,
            // Populated by the caller (propagate_segment) which has the
            // ephemeris context to query the dominant body at sample.time.
            dominant_body_pos: DVec3::ZERO,
        };

        (new_state, sample)
    }

    // -----------------------------------------------------------------------
    // Mode switching
    // -----------------------------------------------------------------------

    fn update_mode(&mut self, perturbation_ratio: f64) {
        match self.mode {
            Mode::Symplectic => {
                if perturbation_ratio > self.config.switch_threshold {
                    self.mode = Mode::Adaptive;
                    // Clamp rk_dt in case it drifted out of range during a
                    // previous adaptive phase.
                    self.rk_dt = self
                        .rk_dt
                        .clamp(self.config.rk_min_dt, self.config.rk_max_dt);
                    // Reset the PI controller so stale error history from a
                    // previous adaptive phase doesn't bias the new one.
                    self.pi = PiController::new();
                }
            }
            Mode::Adaptive => {
                let hysteresis_threshold =
                    self.config.switch_threshold * self.config.hysteresis_factor;
                if perturbation_ratio < hysteresis_threshold {
                    self.mode = Mode::Symplectic;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Velocity-Verlet (symplectic leapfrog)
    // -----------------------------------------------------------------------
    //
    // Algorithm:
    //   x_{n+1} = x_n + v_n * h + ½ * a_n * h²
    //   a_{n+1} = f(x_{n+1}, v_n, t_{n+1})
    //   v_{n+1} = v_n + ½ * (a_n + a_{n+1}) * h
    //
    // For purely position-dependent forces this is exactly the symplectic
    // velocity-Verlet method, which is time-reversible and preserves the shadow
    // Hamiltonian over long integrations.  For velocity-dependent forces (e.g.
    // thrust, drag) it is 2nd-order accurate but no longer strictly symplectic.

    fn verlet_step(
        &mut self,
        state: StateVector,
        time: f64,
        h: f64,
        forces: &ForceRegistry,
        a0: DVec3,
        ephemeris: &dyn BodyStateProvider,
    ) -> StateVector {
        let StateVector {
            position: pos,
            velocity: vel,
        } = state;

        // Advance position.
        let pos1 = pos + vel * h + 0.5 * a0 * h * h;

        // Acceleration at the new position.  We use the current velocity as
        // the velocity argument because v_{n+1} is not yet known — this is the
        // standard velocity-Verlet approximation and is correct to second order.
        let t1 = time + h;
        ephemeris.query_into(t1, &mut self.sample_body_states_buf);
        let a1 = forces.compute_acceleration(pos1, vel, t1, &self.sample_body_states_buf);

        // Average acceleration update for velocity.
        let vel1 = vel + 0.5 * (a0 + a1) * h;

        StateVector {
            position: pos1,
            velocity: vel1,
        }
    }

    // -----------------------------------------------------------------------
    // Dormand-Prince RK45 with PI step-size control
    // -----------------------------------------------------------------------
    //
    // The 6-component state y = [position, velocity] ∈ ℝ⁶ satisfies
    //   dy/dt = [velocity, acceleration(position, velocity, time)].
    //
    // We take steps until one is accepted (normalised error ≤ 1), then return
    // the new state and the step size that was accepted.
    //
    // The inner loop will always terminate because:
    //  (a) On acceptance the loop returns immediately.
    //  (b) On rejection the step shrinks by at least MIN_FACTOR.
    //  (c) Once h == rk_min_dt we accept unconditionally to avoid infinite loops
    //      in extremely stiff regions (the caller can detect this: step_size
    //      in the returned sample will equal rk_min_dt).

    fn rk45_step(
        &mut self,
        state: StateVector,
        time: f64,
        forces: &ForceRegistry,
        ephemeris: &dyn BodyStateProvider,
        max_step: f64,
    ) -> (StateVector, f64) {
        loop {
            let h = self.rk_dt.min(max_step);
            let (new_state, err) = self.dp45_attempt(state, time, h, forces, ephemeris);

            let factor = self.pi.factor(err);

            if err <= 1.0 {
                // Accepted — set next step size and return.
                self.rk_dt = (h * factor).clamp(self.config.rk_min_dt, self.config.rk_max_dt);
                self.min_step_streak = 0;
                return (new_state, h);
            }

            // Rejected — shrink.
            let h_new = (h * factor).max(self.config.rk_min_dt);
            // Guard: if we are already at or below the minimum, accept
            // unconditionally to prevent an infinite loop.
            if h <= self.config.rk_min_dt {
                self.rk_dt = self.config.rk_min_dt;
                self.min_step_streak += 1;
                if self.min_step_streak > 100 {
                    eprintln!(
                        "WARNING: RK45 accepted {} consecutive steps at minimum dt ({:.4e} s). \
                         t={:.6e} s, pos=({:.6e}, {:.6e}, {:.6e}) m. \
                         Simulation may be inaccurate in this region.",
                        self.min_step_streak,
                        self.config.rk_min_dt,
                        time,
                        new_state.position.x,
                        new_state.position.y,
                        new_state.position.z,
                    );
                    self.min_step_streak = 0;
                }
                return (new_state, h);
            }
            self.rk_dt = h_new;
        }
    }

    /// Attempt a single Dormand-Prince RK45 step of size `h`.
    ///
    /// Returns `(new_state, normalised_error)`.  The step is acceptable when
    /// the normalised error ≤ 1.
    fn dp45_attempt(
        &mut self,
        state: StateVector,
        t: f64,
        h: f64,
        forces: &ForceRegistry,
        ephemeris: &dyn BodyStateProvider,
    ) -> (StateVector, f64) {
        let pos = state.position;
        let vel = state.velocity;

        // ---- Stage 1 (at t, pos, vel) ----
        ephemeris.query_into(t, &mut self.sample_body_states_buf);
        let k1_p = vel;
        let k1_v = forces.compute_acceleration(pos, vel, t, &self.sample_body_states_buf);

        // ---- Stage 2 (at t + c2*h) ----
        let t2 = t + Dp45::C2 * h;
        let p2 = pos + h * Dp45::A21 * k1_p;
        let v2 = vel + h * Dp45::A21 * k1_v;
        ephemeris.query_into(t2, &mut self.sample_body_states_buf);
        let k2_p = v2;
        let k2_v = forces.compute_acceleration(p2, v2, t2, &self.sample_body_states_buf);

        // ---- Stage 3 (at t + c3*h) ----
        let t3 = t + Dp45::C3 * h;
        let p3 = pos + h * (Dp45::A31 * k1_p + Dp45::A32 * k2_p);
        let v3 = vel + h * (Dp45::A31 * k1_v + Dp45::A32 * k2_v);
        ephemeris.query_into(t3, &mut self.sample_body_states_buf);
        let k3_p = v3;
        let k3_v = forces.compute_acceleration(p3, v3, t3, &self.sample_body_states_buf);

        // ---- Stage 4 (at t + c4*h) ----
        let t4 = t + Dp45::C4 * h;
        let p4 = pos + h * (Dp45::A41 * k1_p + Dp45::A42 * k2_p + Dp45::A43 * k3_p);
        let v4 = vel + h * (Dp45::A41 * k1_v + Dp45::A42 * k2_v + Dp45::A43 * k3_v);
        ephemeris.query_into(t4, &mut self.sample_body_states_buf);
        let k4_p = v4;
        let k4_v = forces.compute_acceleration(p4, v4, t4, &self.sample_body_states_buf);

        // ---- Stage 5 (at t + c5*h) ----
        let t5 = t + Dp45::C5 * h;
        let p5 =
            pos + h * (Dp45::A51 * k1_p + Dp45::A52 * k2_p + Dp45::A53 * k3_p + Dp45::A54 * k4_p);
        let v5 =
            vel + h * (Dp45::A51 * k1_v + Dp45::A52 * k2_v + Dp45::A53 * k3_v + Dp45::A54 * k4_v);
        ephemeris.query_into(t5, &mut self.sample_body_states_buf);
        let k5_p = v5;
        let k5_v = forces.compute_acceleration(p5, v5, t5, &self.sample_body_states_buf);

        // ---- Stage 6 (at t + h) ----
        let t6 = t + h;
        let p6 = pos
            + h * (Dp45::A61 * k1_p
                + Dp45::A62 * k2_p
                + Dp45::A63 * k3_p
                + Dp45::A64 * k4_p
                + Dp45::A65 * k5_p);
        let v6 = vel
            + h * (Dp45::A61 * k1_v
                + Dp45::A62 * k2_v
                + Dp45::A63 * k3_v
                + Dp45::A64 * k4_v
                + Dp45::A65 * k5_v);
        ephemeris.query_into(t6, &mut self.sample_body_states_buf);
        let k6_p = v6;
        let k6_v = forces.compute_acceleration(p6, v6, t6, &self.sample_body_states_buf);

        // ---- 5th-order solution (propagated state) ----
        let new_pos = pos
            + h * (Dp45::B1 * k1_p
                + Dp45::B3 * k3_p
                + Dp45::B4 * k4_p
                + Dp45::B5 * k5_p
                + Dp45::B6 * k6_p);
        let new_vel = vel
            + h * (Dp45::B1 * k1_v
                + Dp45::B3 * k3_v
                + Dp45::B4 * k4_v
                + Dp45::B5 * k5_v
                + Dp45::B6 * k6_v);

        // ---- Stage 7 — FSAL evaluation at the new state (for error estimate) ----
        // Body states buffer already contains t6 from stage 6; no need to re-query.
        let k7_p = new_vel;
        let k7_v = forces.compute_acceleration(new_pos, new_vel, t6, &self.sample_body_states_buf);

        // ---- Error estimate ----
        // e = h * sum_i (E_i * k_i),  a linear combination of stage derivatives.
        let err_p = h
            * (Dp45::E1 * k1_p
                + Dp45::E3 * k3_p
                + Dp45::E4 * k4_p
                + Dp45::E5 * k5_p
                + Dp45::E6 * k6_p
                + Dp45::E7 * k7_p);
        let err_v = h
            * (Dp45::E1 * k1_v
                + Dp45::E3 * k3_v
                + Dp45::E4 * k4_v
                + Dp45::E5 * k5_v
                + Dp45::E6 * k6_v
                + Dp45::E7 * k7_v);

        // Mixed absolute/relative RMS norm.
        let norm = mixed_rms_norm(err_p, err_v, new_pos, new_vel, self.config.rk_tolerance);

        let new_state = StateVector {
            position: new_pos,
            velocity: new_vel,
        };

        (new_state, norm)
    }
}

// ---------------------------------------------------------------------------
// Error norm
// ---------------------------------------------------------------------------

/// Mixed absolute/relative RMS error norm over the 6-component state vector.
///
/// For each component y_i, the scale is  sc_i = tol + |y_i| * tol,
/// and the norm is  sqrt( (1/6) * Σ (e_i / sc_i)² ).
///
/// A value ≤ 1 means the step is within tolerance.  Using the same `tol` for
/// both the absolute and relative parts is appropriate when all state
/// components are of similar relative magnitude (which they are after
/// nondimensionalisation by the orbital scale).
#[inline]
fn mixed_rms_norm(err_pos: DVec3, err_vel: DVec3, pos: DVec3, vel: DVec3, tol: f64) -> f64 {
    let sq = |e: f64, y: f64| -> f64 {
        let sc = tol + y.abs() * tol;
        let r = e / sc;
        r * r
    };

    let sum = sq(err_pos.x, pos.x)
        + sq(err_pos.y, pos.y)
        + sq(err_pos.z, pos.z)
        + sq(err_vel.x, vel.x)
        + sq(err_vel.y, vel.y)
        + sq(err_vel.z, vel.z);

    (sum / 6.0).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::{ForceRegistry, GravityForce};
    use crate::patched_conics::PatchedConics;
    use crate::types::{
        BodyDefinition, BodyKind, BodyState, G, OrbitalElements, ShipDefinition,
        SolarSystemDefinition, StateVector,
    };
    use glam::DVec3;
    use std::collections::HashMap;

    // -----------------------------------------------------------------------
    // Test fixture helpers
    // -----------------------------------------------------------------------

    const AU: f64 = 1.496e11;
    const SUN_GM: f64 = 1.327_124_4e20;

    /// A minimal two-body system (Sun + Earth) sufficient to exercise the
    /// integrator with a realistic gravity field.
    fn make_two_body_system() -> SolarSystemDefinition {
        let sun_mass = SUN_GM / G;
        let sun = BodyDefinition {
            id: 0,
            name: "Sun".to_string(),
            kind: BodyKind::Star,
            parent: None,
            mass_kg: sun_mass,
            radius_m: 6.957e8,
            color: [1.0, 1.0, 0.0],
            albedo: 1.0,
            rotation_period_s: 0.0,
            axial_tilt_rad: 0.0,
            gm: SUN_GM,
            orbital_elements: None,
            generator: None,
        };
        let earth_mass = 5.972e24;
        let earth = BodyDefinition {
            id: 1,
            name: "Earth".to_string(),
            kind: BodyKind::Planet,
            parent: Some(0),
            mass_kg: earth_mass,
            radius_m: 6.371e6,
            color: [0.0, 0.5, 1.0],
            albedo: 0.3,
            rotation_period_s: 86_400.0,
            axial_tilt_rad: 0.0,
            gm: G * earth_mass,
            orbital_elements: Some(OrbitalElements {
                semi_major_axis_m: AU,
                eccentricity: 0.0,
                inclination_rad: 0.0,
                lon_ascending_node_rad: 0.0,
                arg_periapsis_rad: 0.0,
                true_anomaly_rad: 0.0,
            }),
            generator: None,
        };

        let mut name_to_id = HashMap::new();
        name_to_id.insert("Sun".to_string(), 0);
        name_to_id.insert("Earth".to_string(), 1);

        SolarSystemDefinition {
            name: "Test".to_string(),
            bodies: vec![sun, earth],
            ship: ShipDefinition {
                initial_state: StateVector {
                    position: DVec3::ZERO,
                    velocity: DVec3::ZERO,
                },
                thrust_acceleration: 0.5,
            },
            name_to_id,
        }
    }

    fn test_ephemeris() -> PatchedConics {
        let system = make_two_body_system();
        PatchedConics::new(&system, 3.156e7)
    }

    /// ForceRegistry with only gravity enabled.
    fn gravity_forces() -> ForceRegistry {
        let mut reg = ForceRegistry::new();
        reg.add(Box::new(GravityForce));
        reg
    }

    // -----------------------------------------------------------------------
    // Gravity analysis
    // -----------------------------------------------------------------------

    #[test]
    fn dominant_body_is_closest() {
        // Sun at origin (very massive, far away); Earth close by.
        // Ship sits just outside Earth — Earth should dominate.
        let ship_r = 7.2e6_f64; // ~LEO
        let earth_dist = 7.0e6_f64;
        let body_states = vec![
            BodyState {
                position: DVec3::new(1e12, 0.0, 0.0), // Sun, very far
                velocity: DVec3::ZERO,
                mass_kg: SUN_GM / G,
            },
            BodyState {
                position: DVec3::new(earth_dist, 0.0, 0.0),
                velocity: DVec3::ZERO,
                mass_kg: 5.972e24,
            },
        ];
        let result = GravityForce::compute_with_analysis(DVec3::new(ship_r, 0.0, 0.0), &body_states);
        assert_eq!(result.dominant_body, 1, "Earth should dominate near LEO");
        assert!(
            result.perturbation_ratio < 1.0,
            "perturbation ratio must be < 1 when one body clearly dominates"
        );
    }

    #[test]
    fn single_body_perturbation_ratio_is_zero() {
        let body_states = vec![BodyState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            mass_kg: 5.972e24,
        }];
        let result = GravityForce::compute_with_analysis(DVec3::new(7e6, 0.0, 0.0), &body_states);
        assert_eq!(result.dominant_body, 0);
        assert_eq!(result.perturbation_ratio, 0.0);
    }

    #[test]
    fn equal_bodies_perturbation_ratio_is_one() {
        // Two identical bodies equidistant from the ship.
        let body_states = vec![
            BodyState {
                position: DVec3::new(1e9, 0.0, 0.0),
                velocity: DVec3::ZERO,
                mass_kg: 1e24,
            },
            BodyState {
                position: DVec3::new(-1e9, 0.0, 0.0),
                velocity: DVec3::ZERO,
                mass_kg: 1e24,
            },
        ];
        let result = GravityForce::compute_with_analysis(DVec3::ZERO, &body_states);
        // Both accelerations are equal so ratio == 1.
        assert!(
            (result.perturbation_ratio - 1.0).abs() < 1e-12,
            "equal bodies should give ratio 1, got {}",
            result.perturbation_ratio
        );
    }

    // -----------------------------------------------------------------------
    // Mode switching and hysteresis
    // -----------------------------------------------------------------------

    #[test]
    fn starts_in_symplectic_mode() {
        let integrator = Integrator::new(IntegratorConfig::default());
        assert_eq!(integrator.mode, Mode::Symplectic);
    }

    #[test]
    fn switches_to_adaptive_above_threshold() {
        let cfg = IntegratorConfig {
            switch_threshold: 0.01,
            hysteresis_factor: 0.5,
            ..Default::default()
        };
        let mut integrator = Integrator::new(cfg);
        integrator.update_mode(0.02);
        assert_eq!(integrator.mode, Mode::Adaptive);
    }

    #[test]
    fn does_not_switch_below_threshold_without_hysteresis() {
        let cfg = IntegratorConfig {
            switch_threshold: 0.01,
            hysteresis_factor: 0.5, // switch back at 0.005
            ..Default::default()
        };
        let mut integrator = Integrator::new(cfg);
        // Enter adaptive mode.
        integrator.update_mode(0.02);
        assert_eq!(integrator.mode, Mode::Adaptive);
        // Ratio below switch_threshold but above hysteresis band — stay adaptive.
        integrator.update_mode(0.008);
        assert_eq!(
            integrator.mode,
            Mode::Adaptive,
            "hysteresis should prevent premature switch-back"
        );
        // Now drop below the hysteresis band.
        integrator.update_mode(0.004);
        assert_eq!(integrator.mode, Mode::Symplectic);
    }

    #[test]
    fn no_spurious_switch_at_exact_threshold() {
        // Ratio exactly equal to switch_threshold: should NOT switch (threshold
        // is strictly greater-than).
        let cfg = IntegratorConfig {
            switch_threshold: 0.01,
            ..Default::default()
        };
        let mut integrator = Integrator::new(cfg);
        integrator.update_mode(0.01); // == threshold, not >
        assert_eq!(integrator.mode, Mode::Symplectic);
        integrator.update_mode(0.010_000_000_1); // just above
        assert_eq!(integrator.mode, Mode::Adaptive);
    }

    // -----------------------------------------------------------------------
    // PI controller
    // -----------------------------------------------------------------------

    #[test]
    fn pi_shrinks_step_on_large_error() {
        let mut pi = PiController::new();
        let factor = pi.factor(5.0);
        assert!(
            factor < 1.0,
            "large error should shrink the step, got {factor}"
        );
    }

    #[test]
    fn pi_grows_step_on_small_error() {
        let mut pi = PiController::new();
        let _ = pi.factor(1.0); // seed prev error
        let factor = pi.factor(0.01);
        assert!(
            factor > 1.0,
            "small error should grow the step, got {factor}"
        );
    }

    #[test]
    fn pi_clamps_to_bounds() {
        let mut pi = PiController::new();
        // Tiny error wants enormous factor.
        assert!(pi.factor(1e-15) <= PiController::MAX_FACTOR);
        // Huge error wants minuscule factor.
        assert!(pi.factor(1e15) >= PiController::MIN_FACTOR);
    }

    // -----------------------------------------------------------------------
    // Error norm
    // -----------------------------------------------------------------------

    #[test]
    fn mixed_norm_zero_error_is_zero() {
        let n = mixed_rms_norm(
            DVec3::ZERO,
            DVec3::ZERO,
            DVec3::new(1e9, 2e9, 3e9),
            DVec3::new(1e4, 2e4, 3e4),
            1e-9,
        );
        assert_eq!(n, 0.0);
    }

    #[test]
    fn mixed_norm_symmetric_in_sign() {
        let pos = DVec3::new(1e9, 2e9, 3e9);
        let vel = DVec3::new(1e4, 2e4, 3e4);
        let ep = DVec3::new(1e3, -2e3, 3e2);
        let ev = DVec3::new(-1.0, 2.0, -0.5);
        let n1 = mixed_rms_norm(ep, ev, pos, vel, 1e-9);
        let n2 = mixed_rms_norm(-ep, -ev, pos, vel, 1e-9);
        assert!(
            (n1 - n2).abs() < 1e-14 * n1,
            "norm should not depend on sign of error"
        );
    }

    // -----------------------------------------------------------------------
    // Integration smoke tests
    // -----------------------------------------------------------------------

    /// With zero forces, Verlet must reproduce exact linear motion.
    #[test]
    fn verlet_with_no_forces_is_linear() {
        let eph = test_ephemeris();
        let forces = ForceRegistry::new(); // empty — no forces
        let state = StateVector {
            position: DVec3::new(1.5e11, 0.0, 0.0),
            velocity: DVec3::new(0.0, 2.978e4, 0.0),
        };
        let mut integrator = Integrator::new(IntegratorConfig::default());
        let h = 60.0;
        let body_states = eph.query(0.0);
        let a0 = forces.compute_acceleration(state.position, state.velocity, 0.0, &body_states);
        let new_state = integrator.verlet_step(state, 0.0, h, &forces, a0, &eph);

        let expected_pos = state.position + state.velocity * h;
        // With no forces, position must advance exactly by v*dt.
        let pos_err = (new_state.position - expected_pos).length();
        assert!(pos_err < 1e-6, "position error with zero forces: {pos_err}");
        // Velocity should be unchanged.
        let vel_err = (new_state.velocity - state.velocity).length();
        assert!(
            vel_err < 1e-12,
            "velocity should not change with zero forces: {vel_err}"
        );
    }

    /// Sample time should equal t0 + symplectic_dt for a symplectic step.
    #[test]
    fn step_sample_time_is_correct() {
        let config = IntegratorConfig::default();
        let mut integrator = Integrator::new(config.clone());
        let eph = test_ephemeris();
        let forces = gravity_forces();
        let state = StateVector {
            position: DVec3::new(1.5e11, 0.0, 0.0),
            velocity: DVec3::new(0.0, 2.978e4, 0.0),
        };
        let t0 = 1000.0_f64;
        let (_new, sample) = integrator.step(state, t0, &forces, &eph);
        let expected = t0 + config.symplectic_dt;
        assert!(
            (sample.time - expected).abs() < 1e-9,
            "sample.time should be t0 + dt, got {}",
            sample.time
        );
    }

    #[test]
    fn capped_step_respects_segment_boundary() {
        let config = IntegratorConfig::default();
        let mut integrator = Integrator::new(config);
        let eph = test_ephemeris();
        let forces = gravity_forces();
        let state = StateVector {
            position: DVec3::new(1.5e11, 0.0, 0.0),
            velocity: DVec3::new(0.0, 2.978e4, 0.0),
        };
        let t0 = 1000.0_f64;
        let max_step = 5.0_f64;
        let (_new, sample) = integrator.step_capped(state, t0, &forces, &eph, max_step);
        assert!(
            (sample.time - (t0 + max_step)).abs() < 1e-9,
            "sample.time should stop exactly at the segment boundary"
        );
        assert!(
            (sample.step_size - max_step).abs() < 1e-9,
            "step size should be clamped to the remaining segment time"
        );
    }

    /// RK45 step with zero forces should closely match linear extrapolation.
    #[test]
    fn rk45_with_no_forces_is_linear() {
        let config = IntegratorConfig {
            rk_tolerance: 1e-9,
            rk_initial_dt: 60.0,
            ..Default::default()
        };
        let mut integrator = Integrator::new(config);
        integrator.mode = Mode::Adaptive;
        let eph = test_ephemeris();
        let forces = ForceRegistry::new(); // no forces
        let state = StateVector {
            position: DVec3::new(1.5e11, 0.0, 0.0),
            velocity: DVec3::new(0.0, 2.978e4, 0.0),
        };
        let (new_state, dt) = integrator.rk45_step(state, 0.0, &forces, &eph, f64::INFINITY);
        assert!(dt > 0.0, "step size must be positive");
        // With zero forces y'(t) = [v, 0], so the exact solution is linear.
        let expected_pos = state.position + state.velocity * dt;
        let pos_err = (new_state.position - expected_pos).length();
        // RK45 on a linear ODE should be exact to floating-point precision.
        assert!(
            pos_err < 1.0,
            "RK45 with zero forces should closely match linear extrapolation; error={pos_err}"
        );
    }

    /// A full step in symplectic mode with gravity should produce a physically
    /// plausible displacement for a ship in heliocentric orbit.
    #[test]
    fn step_with_gravity_moves_ship() {
        let mut integrator = Integrator::new(IntegratorConfig::default());
        let eph = test_ephemeris();
        let forces = gravity_forces();
        // Ship in a roughly Earth-like orbit.
        let state = StateVector {
            position: DVec3::new(AU, 0.0, 0.0),
            velocity: DVec3::new(0.0, 2.978e4, 0.0),
        };
        let (new_state, sample) = integrator.step(state, 0.0, &forces, &eph);
        // Position should have changed.
        assert!(
            (new_state.position - state.position).length() > 1.0,
            "position should change after a step"
        );
        // Step size should be positive.
        assert!(sample.step_size > 0.0);
        // dominant_body must be a valid index (0 = Sun, 1 = Earth).
        assert!(sample.dominant_body <= 1);
    }
}
