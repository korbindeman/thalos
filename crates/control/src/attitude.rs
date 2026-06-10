//! The attitude controller: a resolved [`AttitudeDemand`] → a normalized
//! body-frame torque command in `[-1, 1]` per axis.
//!
//! This is the single place attitude is turned into a control effort. It
//! replaces both the per-frame **deadbeat SAS damper** that used to live in
//! the game's `compute_angular_acceleration` (which annihilated all angular
//! velocity every frame and limit-cycled against continuous aero moments)
//! and the scattered PD in the old `navigation::autopilot_command`.
//!
//! Two PD laws:
//! - [`AttitudeController::hold`] — full-quaternion PD to a captured target
//!   orientation (roll included). This is SAS / "centered stick = hold
//!   current attitude". Critically damped, so it settles instead of
//!   chattering.
//! - [`AttitudeController::point_nose`] — nose-direction PD that constrains
//!   only the `+Y` body axis and purely damps roll. Used for directional
//!   nav-mode holds and scheduled-burn pointing, where roll is free.
//!
//! Body frame convention: `X` = pitch axis, `Y` = nose (roll axis),
//! `Z` = yaw axis. Output components map directly to
//! `ControlInput::torque_command` and to aero `pitch/roll/yaw`.

use glam::{DQuat, DVec3};
use thalos_physics_canonical::types::{AttitudeState, ShipParameters};

use crate::demand::AttitudeDemand;

/// Body-frame nose axis. `+Y` is the nose for Apollo-style stacks and the
/// shipyard's aircraft, matching the game's `SHIP_NOSE_BODY`.
pub const NOSE_BODY: DVec3 = DVec3::Y;

/// PD settling time (seconds). `ω_n = π / SETTLE_TIME_S`; the ship reaches a
/// small-angle target in ~`SETTLE_TIME_S` seconds, longer when the command
/// saturates against `max_torque`. The game's autopilot lead-time sizing
/// reads this same constant so engagement windows match the controller.
pub const SETTLE_TIME_S: f64 = 2.0;

/// Stateful attitude controller. The only state is the captured hold target
/// — the orientation SAS / "centered stick" returns to. It is captured on
/// the first `Hold` frame and cleared whenever the pilot deflects the stick
/// (`Rate`) or a pointing demand takes over, so releasing the stick recaptures
/// and holds the *new* attitude.
#[derive(Debug, Default, Clone, Copy)]
pub struct AttitudeController {
    hold_target: Option<DQuat>,
}

/// Magnitude below which a `Rate` command counts as "stick centered" — the
/// controller treats it as no deflection so SAS can recapture a hold target.
const RATE_DEADZONE: f64 = 1.0e-4;

impl AttitudeController {
    pub fn new() -> Self {
        Self::default()
    }

    /// The currently held target orientation, if any. For diagnostics/HUD.
    pub fn hold_target(&self) -> Option<DQuat> {
        self.hold_target
    }

    /// Drive the controller one frame and return the normalized body-frame
    /// torque command in `[-1, 1]`.
    ///
    /// `aero_authority` is the per-axis aero control-moment authority (N·m) at
    /// the current dynamic pressure (`0` in vacuum). The PD normalizes its
    /// desired torque by the **total** available authority
    /// (`max_torque + aero_authority`) so that, when both effectors are driven
    /// at full scale by the allocator, the realized torque equals the PD's
    /// intended torque *exactly* — independent of how much of it the aero
    /// surfaces vs. the wheels provide. Without this the command would either
    /// over-actuate (normalize by `max_torque`, drive both) or starve the aero
    /// surfaces (cap the total at `max_torque`).
    pub fn update(
        &mut self,
        demand: AttitudeDemand,
        attitude: &AttitudeState,
        params: &ShipParameters,
        aero_authority: DVec3,
    ) -> DVec3 {
        match demand {
            AttitudeDemand::Free => {
                self.hold_target = None;
                DVec3::ZERO
            }
            AttitudeDemand::Hold => {
                let target = *self.hold_target.get_or_insert(attitude.orientation);
                self.hold(target, attitude, params, aero_authority)
            }
            AttitudeDemand::PointNose(dir) => {
                // Pointing owns attitude; drop any captured hold so a later
                // release back to Hold recaptures the resulting orientation.
                self.hold_target = None;
                point_nose(dir, attitude, params, aero_authority)
            }
            AttitudeDemand::Rate(cmd) => {
                if cmd.length_squared() <= RATE_DEADZONE * RATE_DEADZONE {
                    // Centered stick: behave as Hold (SAS recapture path).
                    let target = *self.hold_target.get_or_insert(attitude.orientation);
                    self.hold(target, attitude, params, aero_authority)
                } else {
                    // Deflected: a direct full-authority deflection demand — the
                    // allocator drives both effectors at this fraction, so full
                    // stick = full surface deflection + full wheels. Forget the
                    // hold target so the next centered frame captures the new
                    // attitude.
                    self.hold_target = None;
                    cmd.clamp(DVec3::splat(-1.0), DVec3::splat(1.0))
                }
            }
        }
    }

    /// Full-quaternion PD to `target`. Holds roll as well as pitch/yaw.
    pub fn hold(
        &self,
        target: DQuat,
        attitude: &AttitudeState,
        params: &ShipParameters,
        aero_authority: DVec3,
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
        let omega_body = q.inverse() * attitude.angular_velocity;
        pd_to_normalized_torque(error_body, omega_body, params, aero_authority)
    }
}

/// Nose-pointing PD: constrain `+Y` body to `target_nose_world`, purely damp
/// roll about the nose. Ported from the former `navigation::autopilot_command`.
pub fn point_nose(
    target_nose_world: DVec3,
    attitude: &AttitudeState,
    params: &ShipParameters,
    aero_authority: DVec3,
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
    let omega_body = attitude.orientation.inverse() * attitude.angular_velocity;
    pd_to_normalized_torque(error_axis, omega_body, params, aero_authority)
}

/// Critically-damped PD: `kp·error − kd·ω`, gains derived per-axis from MOI
/// and `ω_n = π / SETTLE_TIME_S`, then normalized by the **total** available
/// authority (`max_torque + aero_authority`) to `[-1, 1]`. Normalizing by the
/// total — not just `max_torque` — is what keeps the realized closed-loop
/// torque equal to the designed PD torque when the allocator drives both the
/// reaction wheels and the aero surfaces at this same fraction.
fn pd_to_normalized_torque(
    error_body: DVec3,
    omega_body: DVec3,
    params: &ShipParameters,
    aero_authority: DVec3,
) -> DVec3 {
    let omega_n = std::f64::consts::PI / SETTLE_TIME_S;
    let kp = params.moment_of_inertia * (omega_n * omega_n);
    let kd = params.moment_of_inertia * (2.0 * omega_n);
    let desired = error_body * kp - omega_body * kd;
    let authority = params.max_torque + aero_authority;
    DVec3::new(
        normalize_axis(desired.x, authority.x),
        normalize_axis(desired.y, authority.y),
        normalize_axis(desired.z, authority.z),
    )
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
        let t = c.update(AttitudeDemand::Hold, &att, &params(), DVec3::ZERO);
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
        c.update(AttitudeDemand::Hold, &att, &p, DVec3::ZERO);
        assert!(c.hold_target().is_some());
        // Deflect → hold cleared, command passed through.
        let t = c.update(AttitudeDemand::Rate(DVec3::new(0.8, 0.0, 0.0)), &att, &p, DVec3::ZERO);
        assert_eq!(t.x, 0.8);
        assert!(c.hold_target().is_none());
        // Center (Rate ~0) → recaptures the *current* attitude.
        c.update(AttitudeDemand::Rate(DVec3::ZERO), &att, &p, DVec3::ZERO);
        assert_eq!(c.hold_target(), Some(att.orientation));
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
}
