//! Ship-orientation mode widget and attitude autopilot.
//!
//! [`NavigationState`] holds the player's current orientation request.
//! The widget toggles modes via the side panel; [`nav_attitude_demand`]
//! turns the active mode into an [`AttitudeDemand`] for the fly-by-wire
//! control bus ([`crate::control_bus`]), which arbitrates it against the
//! pilot, the autopilot, and the SAS hold. Player stick input outranks the
//! nav mode at the arbiter.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_control::AttitudeDemand;
use thalos_physics_canonical::simulation::Simulation;
use thalos_physics_canonical::velocity_frame::{NavBasis, VelocityReferenceFrame, nav_basis};

use crate::maneuver::ManeuverPlan;
use crate::target::TargetBody;

pub use thalos_game_state::nav::{
    NavigationMode, NavigationState, SHIP_NOSE_BODY, maneuver_burn_direction,
    maneuver_node_burn_direction, safe_normalize,
};



pub struct NavigationPlugin;

impl Plugin for NavigationPlugin {
    fn build(&self, app: &mut App) {
        // The egui `navigation_panel` system has been replaced by the
        // bevy_ui `hud::nav_panel` cluster. We still need the
        // `NavigationState` resource so the autopilot/UI can communicate.
        app.init_resource::<NavigationState>();
    }
}

// ---------------------------------------------------------------------------
// Autopilot
// ---------------------------------------------------------------------------

/// PD controller settling time, seconds. ω_n = π/T gives a quarter-period
/// of T/2 — the ship reaches the target attitude in ~T seconds when it
/// starts within the linear-torque regime, longer when the controller
/// saturates against `max_torque`. Read by the scheduled-burn autopilot
/// in [`crate::autopilot`] to size its lead time before a maneuver.
pub(crate) const AUTOPILOT_SETTLE_S: f64 = thalos_control::SETTLE_TIME_S;


/// Resolve the active navigation mode into an [`AttitudeDemand`] for the
/// fly-by-wire control bus ([`crate::control_bus`]).
///
/// - `Stability` → [`AttitudeDemand::Hold`] (kill drift, hold the current
///   attitude — the controller captures and holds a target quaternion).
/// - A directional mode (Prograde/Retrograde/Normal/AntiNormal/Radial/
///   Target/AntiTarget/ManeuverNode) → [`AttitudeDemand::PointNose`] at the
///   resolved world direction.
/// - `None` mode, or a directional target that can't be resolved this frame
///   (e.g. zero relative velocity for prograde, missing prediction for a
///   maneuver node) → [`AttitudeDemand::Free`], yielding to the lower-priority
///   SAS hold rather than pointing at the wrong thing.
///
/// Priority against the pilot, the autopilot, and the SAS hold is resolved by
/// the bus arbiter, not here. This function only answers "what does the nav
/// mode want?".
pub fn nav_attitude_demand(
    nav_mode: Option<NavigationMode>,
    active: VelocityReferenceFrame,
    target: &TargetBody,
    plan: &ManeuverPlan,
    sim: &Simulation,
) -> AttitudeDemand {
    let Some(mode) = nav_mode else {
        return AttitudeDemand::Free;
    };
    if matches!(mode, NavigationMode::Stability) {
        return AttitudeDemand::Hold;
    }
    match compute_target_direction(mode, active, sim, target, plan) {
        Some(dir) => AttitudeDemand::PointNose(dir),
        None => AttitudeDemand::Free,
    }
}

/// World-frame unit vector the ship's nose should point at, given the
/// active mode and the current sim/target/plan state. Returns `None`
/// when the target can't be computed (e.g. target body not selected
/// for [`NavigationMode::Target`], no maneuver node for
/// [`NavigationMode::ManeuverNode`], or a degenerate frame). Stability
/// is handled by the caller — this fn never returns a target for it.
fn compute_target_direction(
    mode: NavigationMode,
    active: VelocityReferenceFrame,
    sim: &Simulation,
    target: &TargetBody,
    plan: &ManeuverPlan,
) -> Option<DVec3> {
    let ship = sim.ship_state();
    let time = sim.sim_time();

    match mode {
        NavigationMode::Stability => None,

        // Velocity-relative holds resolve through the active velocity frame,
        // so "hold prograde" follows the navball speed mode — surface
        // prograde while in Surface mode, target-relative prograde in Target
        // mode, and so on.
        NavigationMode::Prograde
        | NavigationMode::Retrograde
        | NavigationMode::Normal
        | NavigationMode::AntiNormal
        | NavigationMode::RadialIn
        | NavigationMode::RadialOut => {
            let basis = active_nav_basis(active, sim, target)?;
            match mode {
                NavigationMode::Prograde => basis.prograde,
                NavigationMode::Retrograde => basis.prograde.map(|d| -d),
                NavigationMode::Normal => basis.normal,
                NavigationMode::AntiNormal => basis.normal.map(|d| -d),
                NavigationMode::RadialOut => basis.radial,
                NavigationMode::RadialIn => basis.radial.map(|d| -d),
                _ => None,
            }
        }

        NavigationMode::Target | NavigationMode::AntiTarget => {
            let target_id = target.target?;
            let target_state = sim
                .ephemeris()
                .state(target_id, thalos_physics_canonical::canonical::Epoch(time));
            let to_target = safe_normalize(target_state.position - ship.position)?;
            Some(if mode == NavigationMode::Target {
                to_target
            } else {
                -to_target
            })
        }

        NavigationMode::ManeuverNode => maneuver_burn_direction(sim, plan),
    }
}

/// Build the navball [`NavBasis`] for the active velocity frame at the
/// ship's current state. Sources body/target states from the ephemeris:
/// the SAS control path runs in `SimStage::Physics`, before the per-frame
/// solar-system snapshot the navball/HUD path reads.
fn active_nav_basis(
    active: VelocityReferenceFrame,
    sim: &Simulation,
    target: &TargetBody,
) -> Option<NavBasis> {
    let ship = sim.ship_state();
    let time = sim.sim_time();
    let body_state = sim.ephemeris().state(
        sim.dominant_body(),
        thalos_physics_canonical::canonical::Epoch(time),
    );
    let target_state = target.target.map(|id| {
        sim.ephemeris()
            .state(id, thalos_physics_canonical::canonical::Epoch(time))
    });
    nav_basis(active, ship, &body_state, target_state.as_ref())
}

#[cfg(test)]
fn format_mission_time(seconds_until_event: f64) -> String {
    let rounded = seconds_until_event.round();
    let marker = if rounded >= 0.0 { '-' } else { '+' };
    format!("T{}{:.0}s", marker, rounded.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_mission_time_uses_countdown_sign_convention() {
        assert_eq!(format_mission_time(7.0), "T-7s");
        assert_eq!(format_mission_time(0.4), "T-0s");
        assert_eq!(format_mission_time(-7.0), "T+7s");
    }
}
