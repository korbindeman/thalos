//! Flight-assist support for winged craft: the body-frame flight state the
//! plane control laws read, and the angle-of-attack envelope protection.
//!
//! When SAS is armed on a winged craft flying in atmosphere, the attitude
//! controller swaps its full-quaternion hold for a **fly-by-wire** law (see
//! [`crate::attitude::AttitudeController`]): centered stick holds pitch
//! attitude + bank angle (a quaternion hold in a banked turn fights the
//! natural heading change with skidding yaw), a slow auto-trim integrator
//! nulls the steady-state pitch sag, and every realized pitch command — the
//! hold law *and* the pilot's stick — passes through the AoA envelope here,
//! so an assisted craft cannot be pulled into a stall. SAS off is fully
//! manual (KSP behaviour); spaceships and vacuum never construct a
//! [`FlightState`] and keep the existing hold exactly.
//!
//! Everything is body-frame (X = right/pitch axis, Y = nose, Z = dorsal/yaw
//! axis): the game supplies local-up and the air-relative velocity rotated
//! into the body frame, and the angle math stays pure and testable here.

use glam::DVec3;

/// Fraction of the stall angle where envelope protection starts to bite:
/// above `ALPHA_PROTECT_START_FRAC · stall_alpha` the available nose-up
/// command fades linearly, reaching zero at the stall angle itself.
pub const ALPHA_PROTECT_START_FRAC: f64 = 0.8;

/// Lead time for the **predictive** AoA the envelope is evaluated at:
/// `α_pred = α + ω_pitch · ALPHA_PROTECT_LEAD_S`. A static limiter only
/// reacts once α is already in the band, so a fast full pull at high dynamic
/// pressure builds enough pitch rate to carry α ~10° past the stall before
/// the clamp can bite (observed in flight test). Leading with the pitch rate
/// fades authority while the pull is still *building*, the same trick real
/// α-limiters use; at zero rate it reduces to the static law exactly.
pub const ALPHA_PROTECT_LEAD_S: f64 = 0.5;

/// Body-frame snapshot of a winged craft's aerodynamic situation, built by
/// the game each frame the flight assist is engaged (SAS armed + winged +
/// in atmosphere above the assist airspeed floor) and `None` otherwise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlightState {
    /// Local radial up (unit), expressed in the body frame.
    pub up_body: DVec3,
    /// Air-relative velocity in the body frame (m/s).
    pub vel_body: DVec3,
    /// The craft's stall angle of attack (rad), from its aero config.
    pub stall_alpha: f64,
}

impl FlightState {
    /// Angle of attack (rad): flow from below the belly → positive. Matches
    /// the aero evaluator's convention exactly.
    pub fn alpha(&self) -> f64 {
        (-self.vel_body.z).atan2(self.vel_body.y)
    }

    /// Sideslip (rad): flow from the right → positive.
    pub fn beta(&self) -> f64 {
        self.vel_body.x.atan2(self.vel_body.y)
    }

    /// Pitch attitude (rad): nose above the local horizon → positive.
    pub fn pitch(&self) -> f64 {
        self.up_body.y.clamp(-1.0, 1.0).asin()
    }

    /// Bank angle (rad): right wing down → positive. `(−π, π]`, so inverted
    /// flight reads as ±(90°…180°) and the hold law recovers through the
    /// shorter roll.
    pub fn bank(&self) -> f64 {
        (-self.up_body.x).atan2(self.up_body.z)
    }
}

/// The captured flight-assist hold target: pitch attitude + bank angle.
/// Heading is deliberately free — holding it too is what made the quaternion
/// hold skid in turns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneHoldTarget {
    pub pitch_rad: f64,
    pub bank_rad: f64,
}

/// What the flight assist did this frame, for the HUD / diagnostics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AssistStatus {
    /// The fly-by-wire plane law (or its envelope on the pilot stick) is
    /// engaged this frame.
    pub fbw_active: bool,
    /// The AoA envelope is actively clamping the pitch command (stall
    /// protection biting).
    pub protection_active: bool,
}

/// The AoA envelope: the `(lo, hi)` bounds the normalized pitch command is
/// clamped to at angle of attack `alpha`.
///
/// Away from the stall both bounds are `±1` (no-op). As `alpha` climbs past
/// `ALPHA_PROTECT_START_FRAC · stall_alpha` the *upper* bound falls linearly,
/// hitting `0` at the stall angle — full back stick buys exactly the stall
/// AoA, never past it — and continuing below zero beyond it, a firm nose-down
/// override that grows to full push within one protection-band width past the
/// stall. The lower bound mirrors this for negative AoA (inverted stall /
/// push-over). Continuous and stateless, so it can clamp the hold law and the
/// raw pilot stick identically with no mode edges.
pub fn pitch_command_envelope(alpha: f64, stall_alpha: f64) -> (f64, f64) {
    // NaN-safe: a degenerate (zero, negative, or NaN) stall angle disables
    // the envelope entirely.
    if stall_alpha.is_nan() || stall_alpha <= 0.0 {
        return (-1.0, 1.0);
    }
    let band = (1.0 - ALPHA_PROTECT_START_FRAC) * stall_alpha;
    let hi = ((stall_alpha - alpha) / band).clamp(-1.0, 1.0);
    let lo = (-(stall_alpha + alpha) / band).clamp(-1.0, 1.0);
    (lo, hi)
}

/// Wrap an angle difference into `(−π, π]` so attitude errors take the short
/// way around.
pub(crate) fn wrap_angle(angle: f64) -> f64 {
    let wrapped = (angle + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
        - std::f64::consts::PI;
    if wrapped == -std::f64::consts::PI {
        std::f64::consts::PI
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn level_flight(stall_alpha: f64) -> FlightState {
        FlightState {
            up_body: DVec3::Z,
            vel_body: DVec3::new(0.0, 100.0, 0.0),
            stall_alpha,
        }
    }

    #[test]
    fn angles_read_correct_signs() {
        // 10° nose-up pitch: world up tilts aft in the body frame.
        let pitch = 10.0_f64.to_radians();
        let f = FlightState {
            up_body: DVec3::new(0.0, pitch.sin(), pitch.cos()),
            ..level_flight(0.26)
        };
        assert_abs_diff_eq!(f.pitch(), pitch, epsilon = 1e-12);
        assert_abs_diff_eq!(f.bank(), 0.0, epsilon = 1e-12);

        // 10° right bank: up tilts toward −X in the body frame.
        let bank = 10.0_f64.to_radians();
        let f = FlightState {
            up_body: DVec3::new(-bank.sin(), 0.0, bank.cos()),
            ..level_flight(0.26)
        };
        assert_abs_diff_eq!(f.bank(), bank, epsilon = 1e-12);
        assert_abs_diff_eq!(f.pitch(), 0.0, epsilon = 1e-12);

        // Flow from below (+α), flow from the right (+β) — the evaluator's
        // conventions.
        let f = FlightState {
            vel_body: DVec3::new(5.0, 100.0, -10.0),
            ..level_flight(0.26)
        };
        assert!(f.alpha() > 0.0);
        assert!(f.beta() > 0.0);
    }

    #[test]
    fn envelope_is_inactive_away_from_stall() {
        let (lo, hi) = pitch_command_envelope(0.0, 0.26);
        assert_eq!((lo, hi), (-1.0, 1.0));
        // Just below the protection threshold: still wide open.
        let (lo, hi) = pitch_command_envelope(0.79 * 0.26, 0.26);
        assert_eq!((lo, hi), (-1.0, 1.0));
    }

    #[test]
    fn envelope_fades_nose_up_to_zero_at_stall() {
        let stall = 0.26;
        // Mid-band: roughly half the nose-up authority left.
        let mid = 0.9 * stall;
        let (_, hi) = pitch_command_envelope(mid, stall);
        assert_abs_diff_eq!(hi, 0.5, epsilon = 1e-9);
        // At the stall angle: no nose-up at all.
        let (_, hi) = pitch_command_envelope(stall, stall);
        assert_abs_diff_eq!(hi, 0.0, epsilon = 1e-12);
        // One band past it: full nose-down override.
        let band = (1.0 - ALPHA_PROTECT_START_FRAC) * stall;
        let (_, hi) = pitch_command_envelope(stall + band, stall);
        assert_abs_diff_eq!(hi, -1.0, epsilon = 1e-12);
    }

    #[test]
    fn envelope_mirrors_for_negative_alpha() {
        let stall = 0.26;
        let (lo, _) = pitch_command_envelope(-stall, stall);
        assert_abs_diff_eq!(lo, 0.0, epsilon = 1e-12);
        let band = (1.0 - ALPHA_PROTECT_START_FRAC) * stall;
        let (lo, _) = pitch_command_envelope(-stall - band, stall);
        assert_abs_diff_eq!(lo, 1.0, epsilon = 1e-12);
        // Bounds never cross.
        for alpha in [-1.0, -0.3, 0.0, 0.3, 1.0] {
            let (lo, hi) = pitch_command_envelope(alpha, stall);
            assert!(lo <= hi, "envelope crossed at α={alpha}: ({lo}, {hi})");
        }
    }

    #[test]
    fn degenerate_stall_alpha_disables_the_envelope() {
        assert_eq!(pitch_command_envelope(0.5, 0.0), (-1.0, 1.0));
        assert_eq!(pitch_command_envelope(0.5, -1.0), (-1.0, 1.0));
        assert_eq!(pitch_command_envelope(0.5, f64::NAN), (-1.0, 1.0));
    }

    #[test]
    fn wrap_angle_takes_the_short_way() {
        assert_abs_diff_eq!(wrap_angle(0.1), 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(
            wrap_angle(350.0_f64.to_radians()),
            -10.0_f64.to_radians(),
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            wrap_angle(-350.0_f64.to_radians()),
            10.0_f64.to_radians(),
            epsilon = 1e-9
        );
    }
}
