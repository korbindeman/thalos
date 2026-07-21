//! Effector allocation: one normalized command → every attitude effector.
//!
//! Both the reaction wheels and the aero control surfaces are driven at the
//! *same* normalized fraction `c ∈ [-1, 1]`, each at its own full scale:
//! - reaction wheels realize `c · max_torque`,
//! - aero surfaces deflect to `c` (full deflection at `c = ±1`), realizing
//!   `c · aero_authority`.
//!
//! So at `c = 1` the craft commands its **full** combined authority
//! (`max_torque + aero_authority`) — full aileron deflection included, which is
//! what gives a plane real roll authority. The correctness of this split lives
//! in the *controller*: [`crate::attitude::AttitudeController`] normalizes its
//! PD output by the same `max_torque + aero_authority`, so the realized torque
//! equals the PD's intended torque exactly — no over-actuation — while a raw
//! pilot `Rate` deflection still maps straight through to full surface throw.
//!
//! Earlier cuts got this wrong in both directions: driving both effectors while
//! the PD normalized by `max_torque` alone over-actuated (yaw oscillation at
//! cruise); capping the realized torque at `max_torque` instead starved the
//! aero surfaces (full stick barely moved the ailerons). Normalizing by the
//! total authority and driving both at that fraction fixes both.

use glam::DVec3;
use thalos_physics_canonical::aero::ControlInputs;

/// The per-effector commands derived from one normalized command.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Allocation {
    /// Reaction-wheel command for `ControlInput::torque_command`
    /// (body frame, `[-1, 1]`, relative to `max_torque`).
    pub reaction_wheel: DVec3,
    /// Aero control-surface deflections (`[-1, 1]`) fed to `evaluate_aero`.
    pub aero: ControlInputs,
}

/// Drive every effector at the normalized `command` (body frame, `[-1, 1]`:
/// `x` pitch, `y` roll, `z` yaw). Reaction wheels and aero surfaces both run at
/// this fraction; an effector with no authority on an axis (no wheels, or no
/// aero / vacuum) simply contributes nothing there, so passing the command to
/// both is always safe.
pub fn allocate(command: DVec3) -> Allocation {
    let c = command.clamp(DVec3::splat(-1.0), DVec3::splat(1.0));
    Allocation {
        reaction_wheel: c,
        // Flap / spoiler deployment is a craft *configuration*, not an
        // attitude effector — the game overlays it downstream (see
        // `thalos_runtime::flight_config`), so the allocator leaves it default.
        aero: ControlInputs {
            pitch: c.x,
            roll: c.y,
            yaw: c.z,
            ..Default::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drives_both_effectors_at_the_command() {
        let a = allocate(DVec3::new(0.1, -0.2, 0.3));
        assert_eq!(a.reaction_wheel, DVec3::new(0.1, -0.2, 0.3));
        assert_eq!(a.aero.pitch, 0.1);
        assert_eq!(a.aero.roll, -0.2);
        assert_eq!(a.aero.yaw, 0.3);
    }

    #[test]
    fn full_command_is_full_deflection() {
        // Full roll command → full aileron throw (roll = 1), the authority the
        // capped allocator used to throw away.
        let a = allocate(DVec3::new(0.0, 1.0, 0.0));
        assert_eq!(a.aero.roll, 1.0);
        assert_eq!(a.reaction_wheel.y, 1.0);
    }

    #[test]
    fn command_is_clamped() {
        let a = allocate(DVec3::new(2.0, -3.0, 0.5));
        assert_eq!(a.reaction_wheel, DVec3::new(1.0, -1.0, 0.5));
        assert_eq!(a.aero.pitch, 1.0);
        assert_eq!(a.aero.roll, -1.0);
    }
}
