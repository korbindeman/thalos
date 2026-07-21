//! The attitude controller: a resolved [`AttitudeDemand`] → a normalized
//! body-frame torque command in `[-1, 1]` per axis.
//!
//! This is the single place attitude is turned into a control effort. It
//! replaces both the per-frame **deadbeat SAS damper** that used to live in
//! the game's `compute_angular_acceleration` (which annihilated all angular
//! velocity every frame and limit-cycled against continuous aero moments)
//! and the scattered PD in the old `navigation::autopilot_command`.
//!
//! Three control laws:
//! - [`AttitudeController::hold`] — full-quaternion PD to a captured target
//!   orientation (roll included). This is SAS / "centered stick = hold
//!   current attitude" for spaceships. Critically damped, so it settles
//!   instead of chattering.
//! - [`AttitudeController::point_nose`] — nose-direction PD that constrains
//!   only the `+Y` body axis and purely damps roll. Used for directional
//!   nav-mode holds and scheduled-burn pointing, where roll is free.
//! - The **flight-assist (fly-by-wire) law** — when the caller supplies a
//!   [`FlightState`] (SAS armed on a winged craft flying in atmosphere), the
//!   SAS hold becomes a pitch-attitude + bank-angle hold with sideslip
//!   damping and a slow pitch **auto-trim** integrator, and every pitch
//!   command (the hold *and* the deflected pilot stick) is clamped by the AoA
//!   envelope ([`crate::flight::pitch_command_envelope`]) so the craft cannot
//!   be pulled into a stall. A quaternion hold is wrong for a plane: holding
//!   heading in a banked turn fights the natural turn with skidding yaw, and
//!   holding attitude against the wing's restoring moment leaves a
//!   steady-state pitch sag the trim integrator exists to null.
//!
//! Body frame convention: `X` = pitch axis, `Y` = nose (roll axis),
//! `Z` = yaw axis. Output components map directly to
//! `ControlInput::torque_command` and to aero `pitch/roll/yaw`.

use glam::{DQuat, DVec3};
use thalos_physics_canonical::types::{AttitudeState, ShipParameters};

use crate::demand::AttitudeDemand;
use crate::flight::{
    ALPHA_PROTECT_LEAD_S, AssistStatus, FlightState, PlaneHoldTarget, pitch_command_envelope,
    wrap_angle,
};

/// Body-frame nose axis. `+Y` is the nose for Apollo-style stacks and the
/// shipyard's aircraft, matching the game's `SHIP_NOSE_BODY`.
pub const NOSE_BODY: DVec3 = DVec3::Y;

/// PD settling time (seconds). `ω_n = π / SETTLE_TIME_S`; the ship reaches a
/// small-angle target in ~`SETTLE_TIME_S` seconds, longer when the command
/// saturates against `max_torque`. The game's autopilot lead-time sizing
/// reads this same constant so engagement windows match the controller.
pub const SETTLE_TIME_S: f64 = 2.0;

/// Stateful attitude controller. The state is the captured hold target — the
/// orientation (spaceship) or pitch/bank pair (flight assist) SAS /
/// "centered stick" returns to — plus the flight-assist pitch trim. Targets
/// are captured on the first `Hold` frame and cleared whenever the pilot
/// deflects the stick (`Rate`) or a pointing demand takes over, so releasing
/// the stick recaptures and holds the *new* attitude. The trim survives stick
/// deflections (releasing mid-maneuver stays in trim) and resets when SAS
/// disengages (`Free`) or a pointing mode takes attitude.
#[derive(Debug, Default, Clone, Copy)]
pub struct AttitudeController {
    hold_target: Option<DQuat>,
    plane_target: Option<PlaneHoldTarget>,
    /// Flight-assist auto-trim: a slow integral on the held pitch error, in
    /// normalized command units. Nulls the steady-state attitude sag the
    /// pure PD leaves against the airframe's restoring moment.
    pitch_trim: f64,
    status: AssistStatus,
}

/// Magnitude below which a `Rate` command counts as "stick centered" — the
/// controller treats it as no deflection so SAS can recapture a hold target.
const RATE_DEADZONE: f64 = 1.0e-4;

// --- Flight-assist (plane FBW) constants --------------------------------------
/// Captured bank targets inside this of wings-level snap to exactly level, so
/// releasing the stick after a roughly-level maneuver flies level rather than
/// freezing a stray degree of bank.
const LEVEL_SNAP_RAD: f64 = 5.0 * std::f64::consts::PI / 180.0;
/// Captured bank targets are clamped to this: release the stick steeper and
/// the assist rolls back to a sustainable turn instead of holding a spiral.
const MAX_BANK_TARGET_RAD: f64 = 60.0 * std::f64::consts::PI / 180.0;
/// Yaw command per radian of sideslip — active turn coordination on top of
/// the airframe's own weathervane stability.
const BETA_DAMP_GAIN: f64 = 2.0;
/// Auto-trim integration rate: normalized pitch command per (rad of held
/// pitch error × second). Slow against the PD (settles the residual over a
/// few seconds) so it can never destabilize the loop.
const TRIM_RATE_PER_S: f64 = 1.0;
/// Anti-windup clamp on the trim command contribution.
const TRIM_AUTHORITY: f64 = 0.4;
/// Trim integration skips pathological frame gaps (loading hitches).
const MAX_TRIM_STEP_S: f64 = 0.25;

impl AttitudeController {
    pub fn new() -> Self {
        Self::default()
    }

    /// The currently held target orientation, if any. For diagnostics/HUD.
    pub fn hold_target(&self) -> Option<DQuat> {
        self.hold_target
    }

    /// The captured flight-assist pitch/bank hold target, if any.
    pub fn plane_hold_target(&self) -> Option<PlaneHoldTarget> {
        self.plane_target
    }

    /// What the flight assist did on the last `update`, for the HUD.
    pub fn assist_status(&self) -> AssistStatus {
        self.status
    }

    /// The flight-assist auto-trim contribution (normalized pitch command).
    pub fn pitch_trim(&self) -> f64 {
        self.pitch_trim
    }

    /// Drive the controller one frame and return the normalized body-frame
    /// torque command in `[-1, 1]`.
    ///
    /// `effector_authority` is the per-axis attitude authority (N·m) of every
    /// effector *other than the reaction wheels*: the aero control surfaces at
    /// the current dynamic pressure, plus a rocket's engine gimbal at the
    /// current throttle (both `0` in vacuum / at coast). The PD normalizes its
    /// desired torque by the **total** available authority
    /// (`max_torque + effector_authority`) so that, when the allocator drives
    /// every effector at the resulting fraction, the realized torque equals the
    /// PD's intended torque *exactly* — independent of how much of it the aero
    /// surfaces, the gimbal, or the wheels provide. Without this the command
    /// would either over-actuate (normalize by `max_torque`, drive all) or
    /// starve the other effectors (cap the total at `max_torque`).
    ///
    /// `flight` engages the flight assist: `Some` means SAS is armed on a
    /// winged craft flying in atmosphere, switching `Hold` to the
    /// pitch/bank law and envelope-protecting the pilot's `Rate`. `None`
    /// (spaceships, vacuum, SAS off) is the unchanged quaternion path.
    /// `dt_s` is the sim-time step the auto-trim integrates over (`0` while
    /// paused).
    pub fn update(
        &mut self,
        demand: AttitudeDemand,
        attitude: &AttitudeState,
        params: &ShipParameters,
        effector_authority: DVec3,
        flight: Option<&FlightState>,
        dt_s: f64,
    ) -> DVec3 {
        self.status = AssistStatus::default();
        match demand {
            AttitudeDemand::Free => {
                self.hold_target = None;
                self.plane_target = None;
                self.pitch_trim = 0.0;
                DVec3::ZERO
            }
            AttitudeDemand::Hold => {
                self.assisted_hold(attitude, params, effector_authority, flight, dt_s)
            }
            AttitudeDemand::PointNose(dir) => {
                // Pointing owns attitude; drop any captured hold so a later
                // release back to Hold recaptures the resulting orientation.
                self.hold_target = None;
                self.plane_target = None;
                self.pitch_trim = 0.0;
                point_nose(dir, attitude, params, effector_authority)
            }
            AttitudeDemand::Rate(cmd) => {
                if cmd.length_squared() <= RATE_DEADZONE * RATE_DEADZONE {
                    // Centered stick: behave as Hold (SAS recapture path).
                    self.assisted_hold(attitude, params, effector_authority, flight, dt_s)
                } else {
                    // Deflected: a direct full-authority deflection demand — the
                    // allocator drives both effectors at this fraction, so full
                    // stick = full surface deflection + full wheels. Forget the
                    // hold targets so the next centered frame captures the new
                    // attitude.
                    self.hold_target = None;
                    self.plane_target = None;
                    let mut c = cmd.clamp(DVec3::splat(-1.0), DVec3::splat(1.0));
                    if let Some(flight) = flight {
                        // Assisted manual flight: the stick rides the held
                        // trim (centered ≈ trimmed flight, not zero surface)
                        // and the AoA envelope caps the pull — full back
                        // stick buys the stall angle, never past it.
                        let raw = (c.x + self.pitch_trim).clamp(-1.0, 1.0);
                        let (lo, hi) = pitch_command_envelope(
                            predicted_alpha(flight, attitude),
                            flight.stall_alpha,
                        );
                        c.x = raw.clamp(lo, hi);
                        self.status = AssistStatus {
                            fbw_active: true,
                            protection_active: c.x != raw,
                        };
                    }
                    c
                }
            }
        }
    }

    /// The SAS hold, dispatched on regime: with a [`FlightState`] the craft is
    /// an assisted plane and holds pitch attitude + bank angle; without one it
    /// is a spaceship (or a plane that left the air) and holds the full
    /// quaternion. Each path clears the other's target so a regime transition
    /// recaptures cleanly from the current attitude.
    fn assisted_hold(
        &mut self,
        attitude: &AttitudeState,
        params: &ShipParameters,
        effector_authority: DVec3,
        flight: Option<&FlightState>,
        dt_s: f64,
    ) -> DVec3 {
        match flight {
            Some(flight) => {
                self.hold_target = None;
                self.plane_hold(flight, attitude, params, effector_authority, dt_s)
            }
            None => {
                self.plane_target = None;
                let target = *self.hold_target.get_or_insert(attitude.orientation);
                self.hold(target, attitude, params, effector_authority)
            }
        }
    }

    /// The flight-assist hold law: pitch-attitude + bank-angle PD (same
    /// deceleration-limited gains as the quaternion hold, applied per axis),
    /// sideslip-damping yaw, pitch auto-trim, and the AoA envelope over the
    /// summed pitch command. Heading is deliberately free — that is what
    /// makes a stick-released banked turn coordinated instead of skidding.
    fn plane_hold(
        &mut self,
        flight: &FlightState,
        attitude: &AttitudeState,
        params: &ShipParameters,
        effector_authority: DVec3,
        dt_s: f64,
    ) -> DVec3 {
        let pitch = flight.pitch();
        let bank = flight.bank();
        let target = *self.plane_target.get_or_insert_with(|| PlaneHoldTarget {
            pitch_rad: pitch,
            bank_rad: if bank.abs() < LEVEL_SNAP_RAD {
                0.0
            } else {
                bank.clamp(-MAX_BANK_TARGET_RAD, MAX_BANK_TARGET_RAD)
            },
        });

        let pitch_err = wrap_angle(target.pitch_rad - pitch);
        let bank_err = wrap_angle(target.bank_rad - bank);
        let omega = attitude.angular_velocity;
        let moi = params.moment_of_inertia;
        let authority = params.max_torque + effector_authority;
        let omega_n = std::f64::consts::PI / SETTLE_TIME_S;
        let kd = moi * (2.0 * omega_n);
        let rate_gain = omega_n * 0.5;

        let pd_pitch = slew_axis(pitch_err, omega.x, rate_gain, kd.x, authority.x, moi.x);
        let roll = slew_axis(bank_err, omega.y, rate_gain, kd.y, authority.y, moi.y);
        let yaw = (-BETA_DAMP_GAIN * flight.beta()).clamp(-1.0, 1.0);

        let (lo, hi) =
            pitch_command_envelope(predicted_alpha(flight, attitude), flight.stall_alpha);
        let raw = (pd_pitch + self.pitch_trim).clamp(-1.0, 1.0);
        let pitch_cmd = raw.clamp(lo, hi);
        let protected = pitch_cmd != raw;

        // Auto-trim: integrate the held pitch error while the loop is in its
        // linear region. Skipping saturated / envelope-clamped frames is the
        // anti-windup — trim must never wind up against a limit it cannot
        // overcome.
        if !protected && raw.abs() < 1.0 && dt_s > 0.0 && dt_s <= MAX_TRIM_STEP_S {
            self.pitch_trim = (self.pitch_trim + TRIM_RATE_PER_S * pitch_err * dt_s)
                .clamp(-TRIM_AUTHORITY, TRIM_AUTHORITY);
        }

        self.status = AssistStatus {
            fbw_active: true,
            protection_active: protected,
        };
        DVec3::new(pitch_cmd, roll, yaw)
    }

    /// Full-quaternion PD to `target`. Holds roll as well as pitch/yaw.
    pub fn hold(
        &self,
        target: DQuat,
        attitude: &AttitudeState,
        params: &ShipParameters,
        effector_authority: DVec3,
    ) -> DVec3 {
        let q = attitude.orientation;
        // Rotation that takes current → target, world frame. Take the
        // shortest path (w ≥ 0) so we never spin the long way around.
        let mut q_err = target * q.inverse();
        if q_err.w < 0.0 {
            q_err = DQuat::from_xyzw(-q_err.x, -q_err.y, -q_err.z, -q_err.w);
        }
        // Small-angle rotation vector ≈ 2·(x,y,z); exact axis·angle near 0,
        // and well-behaved up to π.
        let error_world = 2.0 * DVec3::new(q_err.x, q_err.y, q_err.z);
        let error_body = q.inverse() * error_world;
        // `attitude.angular_velocity` is already expressed in the body frame
        // (see `AttitudeState` docs); it is the ω the PD damps directly. Do
        // *not* rotate it by `q.inverse()` — that double-transform misaims the
        // damping torque by the ship's orientation, which manifests as SAS
        // failing to settle and pointing modes spinning up.
        let omega_body = attitude.angular_velocity;
        pd_to_normalized_torque(error_body, omega_body, params, effector_authority)
    }
}

/// The predictive AoA the envelope is evaluated at: current α led by the
/// body pitch rate (`α̇ ≈ ω_pitch` over the short period — the flight path
/// lags the nose). At zero rate this is exactly the static α; a fast pull
/// fades authority before α reaches the band, which is what keeps the
/// limiter from being blasted through by built-up pitch rate (flight test:
/// a static clamp let a full pull at high q overshoot the stall by ~10°).
fn predicted_alpha(flight: &FlightState, attitude: &AttitudeState) -> f64 {
    flight.alpha() + attitude.angular_velocity.x * ALPHA_PROTECT_LEAD_S
}

/// Nose-pointing PD: constrain `+Y` body to `target_nose_world`, purely damp
/// roll about the nose. Ported from the former `navigation::autopilot_command`.
pub fn point_nose(
    target_nose_world: DVec3,
    attitude: &AttitudeState,
    params: &ShipParameters,
    effector_authority: DVec3,
) -> DVec3 {
    let target_body = attitude.orientation.inverse() * target_nose_world;

    // `nose × target` gives axis·sin(angle); its Y component is always zero,
    // so torque.y (roll) comes purely from the −Kd·ω damping term. Near 180°
    // error sin(angle)→0 and the controller stalls, so inject a kick about
    // body X to break the symmetry.
    let error_axis = if target_body.y < -0.99 {
        DVec3::X
    } else {
        NOSE_BODY.cross(target_body)
    };
    // `attitude.angular_velocity` is already body-frame (see `AttitudeState`
    // docs) — the frame the PD damps in. The old `navigation::autopilot_command`
    // this was ported from used it directly; the `orientation.inverse() *` added
    // in the port was a frame bug that misaimed the roll/yaw damping.
    let omega_body = attitude.angular_velocity;
    pd_to_normalized_torque(error_axis, omega_body, params, effector_authority)
}

/// Fraction of the available angular deceleration the stopping-rate cap
/// budgets for braking. Below 1 it leaves headroom so the finite-gain rate
/// loop can track the shrinking stop-rate without coasting past it (a value
/// of 1 would brake exactly time-optimally on paper but leak a few degrees of
/// overshoot through tracking lag).
const DECEL_MARGIN: f64 = 0.9;

/// Critically-damped PD with a **deceleration-limited rate cap**, normalized
/// by the **total** available authority (`max_torque + effector_authority`) to
/// `[-1, 1]`. Normalizing by the total — not just `max_torque` — keeps the
/// realized closed-loop torque equal to the designed torque when the allocator
/// drives both the reaction wheels and the aero surfaces at this same fraction.
///
/// The bare PD `kp·e − kd·ω` is critically damped only in its *linear* region.
/// `kp` is sized for a small-angle `SETTLE_TIME_S` settle, so the command
/// saturates against the available authority at a fairly small error; a large
/// target change then runs open-loop at full torque, builds up rate, and
/// **overshoots** — it snaps to the target and bounces back. To ease in
/// instead, cap the PD's implied desired rate `ω_des = (kp/kd)·e = (ω_n/2)·e`
/// at the deceleration-limited *stopping rate* `ω_stop = √(2·α·|e|)`, the
/// fastest rate from which the available angular acceleration `α = authority/I`
/// can still null the error by the time it reaches zero. Far from the target
/// the stop-rate cap binds (near time-optimal — no overshoot); close in,
/// `ω_des` falls below the cap and the law *is* the original critically-damped
/// PD (no sqrt chatter at the target). So small slews and strong-wheel craft
/// are unchanged; only the saturating slews that used to overshoot are tamed.
fn pd_to_normalized_torque(
    error_body: DVec3,
    omega_body: DVec3,
    params: &ShipParameters,
    effector_authority: DVec3,
) -> DVec3 {
    let omega_n = std::f64::consts::PI / SETTLE_TIME_S;
    let moi = params.moment_of_inertia;
    let kd = moi * (2.0 * omega_n);
    let authority = params.max_torque + effector_authority;
    // (kp/kd) = ω_n/2 — the linear PD's implied desired rate per unit error.
    let rate_gain = omega_n * 0.5;
    DVec3::new(
        slew_axis(
            error_body.x,
            omega_body.x,
            rate_gain,
            kd.x,
            authority.x,
            moi.x,
        ),
        slew_axis(
            error_body.y,
            omega_body.y,
            rate_gain,
            kd.y,
            authority.y,
            moi.y,
        ),
        slew_axis(
            error_body.z,
            omega_body.z,
            rate_gain,
            kd.z,
            authority.z,
            moi.z,
        ),
    )
}

/// One axis of the deceleration-limited PD → normalized torque in `[-1, 1]`.
/// Reduces to the plain critically-damped PD (`kp·e − kd·ω`) whenever the
/// linear branch is taken (small error / ample authority).
fn slew_axis(error: f64, omega: f64, rate_gain: f64, kd: f64, authority: f64, moi: f64) -> f64 {
    let omega_des_linear = rate_gain * error;
    let alpha = if moi > 0.0 {
        DECEL_MARGIN * authority / moi
    } else {
        0.0
    };
    let omega_stop = (2.0 * alpha * error.abs()).sqrt().copysign(error);
    // Smaller magnitude wins: stop-rate far from target, linear PD near it.
    let omega_des = if omega_des_linear.abs() < omega_stop.abs() {
        omega_des_linear
    } else {
        omega_stop
    };
    // Rate loop closes on the (capped) desired rate. When the linear branch is
    // taken this is exactly kd·((kp/kd)·e − ω) = kp·e − kd·ω.
    normalize_axis(kd * (omega_des - omega), authority)
}

fn normalize_axis(desired: f64, max: f64) -> f64 {
    if max > 0.0 {
        (desired / max).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn params() -> ShipParameters {
        ShipParameters {
            moment_of_inertia: DVec3::splat(1000.0),
            center_of_mass: DVec3::ZERO,
            max_torque: DVec3::splat(500.0),
            gimbal_torque_full: DVec3::ZERO,
            thrust_n: 0.0,
            mass_flow_kg_per_s: 0.0,
            dry_mass_kg: 1000.0,
            impact_tolerance_m_s: f64::INFINITY,
            reference_area_m2: 0.0,
            drag_coefficient: 0.0,
        }
    }

    #[test]
    fn hold_captures_on_first_frame_and_zero_torque_when_settled() {
        let mut c = AttitudeController::new();
        let att = AttitudeState {
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
        };
        let t = c.update(
            AttitudeDemand::Hold,
            &att,
            &params(),
            DVec3::ZERO,
            None,
            0.0,
        );
        // At target with zero rate → zero command, no chatter.
        assert_abs_diff_eq!(t.x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t.y, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t.z, 0.0, epsilon = 1e-12);
        assert!(c.hold_target().is_some());
    }

    #[test]
    fn hold_pd_converges_without_limit_cycle() {
        // Closed-loop sim: a small initial error under continuous tiny
        // disturbance torque (mimicking aero). The deadbeat damper would
        // chatter; the PD should monotonically shrink |ω| toward 0 and the
        // error toward 0.
        let p = params();
        let mut c = AttitudeController::new();
        let target = DQuat::IDENTITY;
        c.hold_target = Some(target);

        let mut orientation = DQuat::from_axis_angle(DVec3::X, 0.2);
        let mut omega = DVec3::ZERO;
        let dt = 1.0 / 60.0;
        let disturbance = DVec3::new(2.0, 0.0, 0.0); // N·m, constant

        let mut last_err = f64::INFINITY;
        for step in 0..600 {
            let att = AttitudeState {
                orientation,
                angular_velocity: omega,
            };
            let cmd = c.hold(target, &att, &p, DVec3::ZERO);
            let torque = cmd * p.max_torque + disturbance;
            let ang_accel = torque / p.moment_of_inertia;
            omega += ang_accel * dt;
            orientation = (DQuat::from_scaled_axis(omega * dt) * orientation).normalize();

            let err = (target * orientation.inverse()).to_axis_angle().1;
            // After the initial transient, error must stay bounded small —
            // no growing oscillation.
            if step > 300 {
                assert!(err < 0.05, "step {step}: error {err} not settled");
            }
            last_err = err;
        }
        assert!(last_err < 0.05);
    }

    #[test]
    fn deflected_rate_clears_hold_then_recaptures_on_center() {
        let mut c = AttitudeController::new();
        let p = params();
        let att = AttitudeState {
            orientation: DQuat::from_axis_angle(DVec3::Z, 0.5),
            angular_velocity: DVec3::ZERO,
        };
        // Capture a hold first.
        c.update(AttitudeDemand::Hold, &att, &p, DVec3::ZERO, None, 0.0);
        assert!(c.hold_target().is_some());
        // Deflect → hold cleared, command passed through.
        let t = c.update(
            AttitudeDemand::Rate(DVec3::new(0.8, 0.0, 0.0)),
            &att,
            &p,
            DVec3::ZERO,
            None,
            0.0,
        );
        assert_eq!(t.x, 0.8);
        assert!(c.hold_target().is_none());
        // Center (Rate ~0) → recaptures the *current* attitude.
        c.update(
            AttitudeDemand::Rate(DVec3::ZERO),
            &att,
            &p,
            DVec3::ZERO,
            None,
            0.0,
        );
        assert_eq!(c.hold_target(), Some(att.orientation));
    }

    #[test]
    fn hold_damps_off_axis_omega_in_the_body_frame() {
        // Regression for the frame bug where `hold` rotated the (already
        // body-frame) `attitude.angular_velocity` by `orientation.inverse()`
        // before damping. With the craft *at* its hold target (zero
        // orientation error) and a pure body-X angular velocity, the only
        // term is `-kd·ω`, so the command must be a pure body-X brake.
        //
        // The orientation is a 90° roll about Z, chosen so the buggy
        // `q.inverse() * ω` would rotate body-X into body-(-Y): under the bug
        // the command would point along Y, not X. A non-axis-aligned
        // orientation is what the older on-axis tests never exercised.
        let p = params();
        let q = DQuat::from_axis_angle(DVec3::Z, std::f64::consts::FRAC_PI_2);
        let mut c = AttitudeController::new();
        c.hold_target = Some(q); // at target → zero positional error
        let att = AttitudeState {
            orientation: q,
            angular_velocity: DVec3::new(0.001, 0.0, 0.0), // body-frame, off the Z spin axis
        };
        let cmd = c.hold(q, &att, &p, DVec3::ZERO);
        // Pure body-X brake: oppose ω on X, nothing on Y/Z.
        assert!(
            cmd.x < 0.0,
            "expected a braking torque opposing +X ω, got {cmd:?}"
        );
        assert_abs_diff_eq!(cmd.y, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(cmd.z, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn point_nose_damps_off_axis_omega_in_the_body_frame() {
        // Same frame regression for the pointing law. Nose already on target
        // (so the cross-product error term vanishes) but spinning about body
        // X; the command must be a pure body-X brake, not a rotated one.
        let p = params();
        let q = DQuat::from_axis_angle(DVec3::Z, std::f64::consts::FRAC_PI_2);
        let nose_world = q * NOSE_BODY; // current nose dir → zero pointing error
        let att = AttitudeState {
            orientation: q,
            angular_velocity: DVec3::new(0.001, 0.0, 0.0),
        };
        let cmd = point_nose(nose_world, &att, &p, DVec3::ZERO);
        assert!(
            cmd.x < 0.0,
            "expected a braking torque opposing +X ω, got {cmd:?}"
        );
        assert_abs_diff_eq!(cmd.y, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(cmd.z, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn hold_large_slew_eases_in_without_overshoot() {
        // Weak reaction wheels relative to inertia (I/T = 10) — the saturating
        // regime. The plain critically-damped PD overshoots a 90° change by
        // ~0.43 rad and rings; the deceleration-limited cap must ease in with
        // only a sliver of overshoot. Single-axis (Z) closed-loop sim mirroring
        // the game: cmd·max_torque is applied as body-frame torque, integrated
        // as an isotropic rigid body.
        let p = ShipParameters {
            moment_of_inertia: DVec3::splat(10_000.0),
            max_torque: DVec3::splat(1_000.0),
            ..params()
        };
        let target_angle = std::f64::consts::FRAC_PI_2;
        let target = DQuat::from_axis_angle(DVec3::Z, target_angle);
        let mut c = AttitudeController::new();
        c.hold_target = Some(target);

        let mut orientation = DQuat::IDENTITY;
        let mut omega = DVec3::ZERO; // body frame
        let dt = 1.0 / 60.0;
        let mut max_overshoot = 0.0_f64;
        for _ in 0..1800 {
            let att = AttitudeState {
                orientation,
                angular_velocity: omega,
            };
            let cmd = c.hold(target, &att, &p, DVec3::ZERO);
            let ang_accel = (cmd * p.max_torque) / p.moment_of_inertia;
            omega += ang_accel * dt;
            orientation = (orientation * DQuat::from_scaled_axis(omega * dt)).normalize();
            let phi = orientation.to_scaled_axis().z; // pure-Z rotation angle
            max_overshoot = max_overshoot.max((phi - target_angle).max(0.0));
        }
        let final_err = (orientation.to_scaled_axis().z - target_angle).abs();
        assert!(final_err < 0.02, "did not settle: {final_err} rad");
        assert!(
            max_overshoot < 0.1,
            "overshoot too large: {max_overshoot} rad"
        );
    }

    #[test]
    fn point_nose_commands_toward_target() {
        let p = params();
        let att = AttitudeState {
            orientation: DQuat::IDENTITY, // nose along +Y
            angular_velocity: DVec3::ZERO,
        };
        // Ask to point nose along +X: needs a yaw/pitch torque, nonzero.
        let t = point_nose(DVec3::X, &att, &p, DVec3::ZERO);
        assert!(t.length() > 0.0);
    }

    // --- Flight assist (plane FBW) -----------------------------------------

    /// Aircraft-shaped parameters: no reaction wheels — all authority is aero.
    fn plane_params() -> ShipParameters {
        ShipParameters {
            moment_of_inertia: DVec3::splat(1.0e5),
            max_torque: DVec3::ZERO,
            ..params()
        }
    }

    const PLANE_AERO_AUTHORITY: DVec3 = DVec3::splat(5.0e4);

    /// A level-flight `FlightState` at the given pitch attitude / bank angle,
    /// with the relative wind on the nose (α = β = 0) so the envelope stays
    /// out of the way unless a test wants it.
    fn flight_at(pitch: f64, bank: f64) -> FlightState {
        // Aerospace Euler order, roll innermost: q = R_pitch(X) · R_bank(Y),
        // so both angles read back exactly from up_body.
        let q = glam::DQuat::from_axis_angle(DVec3::X, pitch)
            * glam::DQuat::from_axis_angle(DVec3::Y, bank);
        FlightState {
            up_body: q.inverse() * DVec3::Z,
            vel_body: DVec3::new(0.0, 100.0, 0.0),
            stall_alpha: 0.26,
        }
    }

    fn still_attitude() -> AttitudeState {
        AttitudeState {
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
        }
    }

    #[test]
    fn plane_hold_captures_pitch_and_bank() {
        let mut c = AttitudeController::new();
        let pitch = 0.12;
        let bank = 0.4;
        let f = flight_at(pitch, bank);
        c.update(
            AttitudeDemand::Hold,
            &still_attitude(),
            &plane_params(),
            PLANE_AERO_AUTHORITY,
            Some(&f),
            1.0 / 60.0,
        );
        let target = c.plane_hold_target().expect("plane hold captured");
        assert_abs_diff_eq!(target.pitch_rad, pitch, epsilon = 1e-9);
        assert_abs_diff_eq!(target.bank_rad, bank, epsilon = 1e-9);
        // The quaternion hold must not also engage — the plane law owns it.
        assert!(c.hold_target().is_none());
        assert!(c.assist_status().fbw_active);
        assert!(!c.assist_status().protection_active);
    }

    #[test]
    fn plane_hold_snaps_small_bank_level_and_clamps_steep_bank() {
        // 3° of residual bank on release → wings-level target.
        let mut c = AttitudeController::new();
        let f = flight_at(0.0, 3.0_f64.to_radians());
        c.update(
            AttitudeDemand::Hold,
            &still_attitude(),
            &plane_params(),
            PLANE_AERO_AUTHORITY,
            Some(&f),
            1.0 / 60.0,
        );
        assert_eq!(c.plane_hold_target().unwrap().bank_rad, 0.0);

        // Released at 80° of bank → the target clamps to the 60° sustainable
        // turn instead of holding a spiral.
        let mut c = AttitudeController::new();
        let f = flight_at(0.0, 80.0_f64.to_radians());
        c.update(
            AttitudeDemand::Hold,
            &still_attitude(),
            &plane_params(),
            PLANE_AERO_AUTHORITY,
            Some(&f),
            1.0 / 60.0,
        );
        assert_abs_diff_eq!(
            c.plane_hold_target().unwrap().bank_rad,
            60.0_f64.to_radians(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn plane_hold_commands_recover_toward_the_target() {
        // Capture straight-and-level, then disturb: nose below target must
        // command nose-up (+X), banked right of target must command roll-left
        // (−Y), and sideslip from the right must command coordinating yaw with
        // the same sign as the weathervane (negative for +β).
        let mut c = AttitudeController::new();
        let p = plane_params();
        c.update(
            AttitudeDemand::Hold,
            &still_attitude(),
            &p,
            PLANE_AERO_AUTHORITY,
            Some(&flight_at(0.0, 0.0)),
            1.0 / 60.0,
        );

        let disturbed = flight_at(-0.1, 0.3);
        let cmd = c.update(
            AttitudeDemand::Hold,
            &still_attitude(),
            &p,
            PLANE_AERO_AUTHORITY,
            Some(&disturbed),
            1.0 / 60.0,
        );
        assert!(cmd.x > 0.0, "nose low must pull up, got {}", cmd.x);
        assert!(cmd.y < 0.0, "banked right must roll left, got {}", cmd.y);

        let slipping = FlightState {
            vel_body: DVec3::new(10.0, 100.0, 0.0),
            ..flight_at(0.0, 0.0)
        };
        let cmd = c.update(
            AttitudeDemand::Hold,
            &still_attitude(),
            &p,
            PLANE_AERO_AUTHORITY,
            Some(&slipping),
            1.0 / 60.0,
        );
        assert!(
            cmd.z < 0.0,
            "+β must command coordinating yaw, got {}",
            cmd.z
        );
    }

    #[test]
    fn assisted_stick_cannot_pull_past_the_stall() {
        let mut c = AttitudeController::new();
        let p = plane_params();
        let full_pull = AttitudeDemand::Rate(DVec3::new(1.0, 0.0, 0.0));

        // At the stall angle: the envelope zeroes the pull.
        let stalling = FlightState {
            vel_body: DVec3::new(0.0, 100.0, -100.0 * 0.26_f64.tan()),
            ..flight_at(0.0, 0.0)
        };
        let cmd = c.update(
            full_pull,
            &still_attitude(),
            &p,
            PLANE_AERO_AUTHORITY,
            Some(&stalling),
            1.0 / 60.0,
        );
        assert!(
            cmd.x <= 1e-9,
            "full pull at stall AoA must be zeroed, got {}",
            cmd.x
        );
        assert!(c.assist_status().protection_active);

        // Past the stall: a firm nose-down override even against full pull.
        let deep = FlightState {
            vel_body: DVec3::new(0.0, 100.0, -100.0 * 0.32_f64.tan()),
            ..flight_at(0.0, 0.0)
        };
        let cmd = c.update(
            full_pull,
            &still_attitude(),
            &p,
            PLANE_AERO_AUTHORITY,
            Some(&deep),
            1.0 / 60.0,
        );
        assert!(cmd.x < 0.0, "past stall must push, got {}", cmd.x);

        // Same pull with no flight state (SAS off / spaceship): raw KSP
        // passthrough, untouched.
        let cmd = c.update(
            full_pull,
            &still_attitude(),
            &p,
            PLANE_AERO_AUTHORITY,
            None,
            1.0 / 60.0,
        );
        assert_eq!(cmd.x, 1.0);
        assert!(!c.assist_status().fbw_active);
    }

    #[test]
    fn protection_leads_with_pitch_rate() {
        // α still well below the band (5°), but pitching up hard at 20°/s:
        // the predictive limiter must already be cutting the pull, where a
        // static one would wait until α itself reached 12° and get blasted
        // through (the ~10° overshoot seen in flight test). At zero rate the
        // same α must remain unrestricted.
        let mut c = AttitudeController::new();
        let p = plane_params();
        let full_pull = AttitudeDemand::Rate(DVec3::new(1.0, 0.0, 0.0));
        let alpha = 5.0_f64.to_radians();
        let flight = FlightState {
            vel_body: DVec3::new(0.0, 100.0, -100.0 * alpha.tan()),
            ..flight_at(0.0, 0.0)
        };

        let pitching_up = AttitudeState {
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::new(20.0_f64.to_radians(), 0.0, 0.0),
        };
        let cmd = c.update(
            full_pull,
            &pitching_up,
            &p,
            PLANE_AERO_AUTHORITY,
            Some(&flight),
            1.0 / 60.0,
        );
        assert!(
            cmd.x < 1.0,
            "fast pull must fade early via the predictive α, got {}",
            cmd.x
        );
        assert!(c.assist_status().protection_active);

        let cmd = c.update(
            full_pull,
            &still_attitude(),
            &p,
            PLANE_AERO_AUTHORITY,
            Some(&flight),
            1.0 / 60.0,
        );
        assert_eq!(cmd.x, 1.0, "same α at zero rate must be unrestricted");
        assert!(!c.assist_status().protection_active);
    }

    #[test]
    fn plane_hold_protection_overrides_the_hold_itself() {
        // Holding an attitude while AoA decays past the stall (speed bleeding
        // off in a too-steep climb): the envelope must override the hold's
        // nose-up demand with a push, even though the pitch target is above.
        let mut c = AttitudeController::new();
        let p = plane_params();
        c.update(
            AttitudeDemand::Hold,
            &still_attitude(),
            &p,
            PLANE_AERO_AUTHORITY,
            Some(&flight_at(0.3, 0.0)),
            1.0 / 60.0,
        );
        let stalled = FlightState {
            vel_body: DVec3::new(0.0, 40.0, -40.0 * 0.30_f64.tan()),
            ..flight_at(0.1, 0.0) // nose below target → PD wants to pull
        };
        let cmd = c.update(
            AttitudeDemand::Hold,
            &still_attitude(),
            &p,
            PLANE_AERO_AUTHORITY,
            Some(&stalled),
            1.0 / 60.0,
        );
        assert!(
            cmd.x < 0.0,
            "stall protection must out-vote the hold, got {}",
            cmd.x
        );
        assert!(c.assist_status().protection_active);
    }

    #[test]
    fn auto_trim_nulls_steady_state_pitch_error() {
        // Closed-loop single-axis pitch sim against a constant nose-down
        // disturbance moment (the airframe's restoring moment held off-trim).
        // The bare PD parks at a steady-state sag; the trim integrator must
        // walk it out. Pitch kinematics only — attitude integrates the body
        // pitch rate; α stays on the nose so the envelope is quiet.
        let p = plane_params();
        let authority = PLANE_AERO_AUTHORITY;
        let disturbance = -0.2 * authority.x; // N·m, constant nose-down
        let target_pitch = 0.05;

        let mut c = AttitudeController::new();
        let dt = 1.0 / 60.0;
        let mut pitch = target_pitch; // start on target
        let mut omega_x = 0.0;
        // Capture the target at the initial attitude.
        let mut last_pitch_err: f64 = f64::INFINITY;
        for step in 0..(40 * 60) {
            let f = flight_at(pitch, 0.0);
            let att = AttitudeState {
                orientation: DQuat::IDENTITY,
                angular_velocity: DVec3::new(omega_x, 0.0, 0.0),
            };
            let cmd = c.update(AttitudeDemand::Hold, &att, &p, authority, Some(&f), dt);
            let torque = cmd.x * authority.x + disturbance;
            omega_x += torque / p.moment_of_inertia.x * dt;
            pitch += omega_x * dt;
            last_pitch_err = (pitch - target_pitch).abs();
            if step == 120 {
                // Two seconds in, before trim has integrated up: the PD alone
                // must be sagging visibly (this is the error trim exists for).
                assert!(
                    last_pitch_err > 0.01,
                    "expected an untrimmed PD sag, got {last_pitch_err}"
                );
            }
        }
        assert!(
            last_pitch_err < 0.005,
            "trim failed to null the hold error: {last_pitch_err} rad"
        );
        assert!(
            c.pitch_trim() > 0.0,
            "trim should hold nose-up, got {}",
            c.pitch_trim()
        );
    }

    #[test]
    fn trim_survives_stick_deflection_and_resets_on_free() {
        let mut c = AttitudeController::new();
        let p = plane_params();
        let f = flight_at(0.0, 0.0);
        // Build some trim by holding with a pitch error.
        c.update(
            AttitudeDemand::Hold,
            &still_attitude(),
            &p,
            PLANE_AERO_AUTHORITY,
            Some(&f),
            1.0 / 60.0,
        );
        let low = flight_at(-0.05, 0.0);
        for _ in 0..120 {
            c.update(
                AttitudeDemand::Hold,
                &still_attitude(),
                &p,
                PLANE_AERO_AUTHORITY,
                Some(&low),
                1.0 / 60.0,
            );
        }
        let trim = c.pitch_trim();
        assert!(trim > 0.0);

        // Deflect: the command rides the trim, and the trim is untouched.
        let cmd = c.update(
            AttitudeDemand::Rate(DVec3::new(0.1, 0.0, 0.0)),
            &still_attitude(),
            &p,
            PLANE_AERO_AUTHORITY,
            Some(&f),
            1.0 / 60.0,
        );
        assert_abs_diff_eq!(cmd.x, 0.1 + trim, epsilon = 1e-12);
        assert_eq!(c.pitch_trim(), trim);

        // SAS off → everything resets.
        c.update(
            AttitudeDemand::Free,
            &still_attitude(),
            &p,
            PLANE_AERO_AUTHORITY,
            None,
            1.0 / 60.0,
        );
        assert_eq!(c.pitch_trim(), 0.0);
        assert!(c.plane_hold_target().is_none());
    }

    #[test]
    fn leaving_the_air_hands_back_to_the_quaternion_hold() {
        let mut c = AttitudeController::new();
        let p = plane_params();
        c.update(
            AttitudeDemand::Hold,
            &still_attitude(),
            &p,
            PLANE_AERO_AUTHORITY,
            Some(&flight_at(0.2, 0.0)),
            1.0 / 60.0,
        );
        assert!(c.plane_hold_target().is_some());

        // Climbed out of the atmosphere: no flight state. The quaternion hold
        // captures the current orientation and the plane target clears.
        let att = AttitudeState {
            orientation: DQuat::from_axis_angle(DVec3::X, 0.3),
            angular_velocity: DVec3::ZERO,
        };
        c.update(
            AttitudeDemand::Hold,
            &att,
            &p,
            DVec3::ZERO,
            None,
            1.0 / 60.0,
        );
        assert!(c.plane_hold_target().is_none());
        assert_eq!(c.hold_target(), Some(att.orientation));
        assert!(!c.assist_status().fbw_active);
    }
}
