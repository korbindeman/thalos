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
use thalos_physics_canonical::maneuver::delta_v_to_world;
use thalos_physics_canonical::simulation::Simulation;
use thalos_physics_canonical::trajectory::Trajectory;
use thalos_physics_canonical::velocity_frame::{NavBasis, VelocityReferenceFrame, nav_basis};
use thalos_control::AttitudeDemand;

use crate::maneuver::{GameNode, ManeuverPlan};
use crate::target::TargetBody;

/// Discrete ship-orientation modes the player can request.
///
/// `None` in [`NavigationState::mode`] means free flight (no auto-orient).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationMode {
    /// Hold current attitude (kill rotation).
    Stability,
    /// Point along orbital velocity.
    Prograde,
    /// Point against orbital velocity.
    Retrograde,
    /// Point along the orbital plane normal.
    Normal,
    /// Point against the orbital plane normal.
    AntiNormal,
    /// Point toward the parent body.
    RadialIn,
    /// Point away from the parent body.
    RadialOut,
    /// Point toward the selected target.
    Target,
    /// Point away from the selected target.
    AntiTarget,
    /// Point along the next maneuver node's burn direction.
    ManeuverNode,
}

/// Currently requested orientation mode.
///
/// `mode` selects the autopilot's pointing target (`None` = free
/// flight). Scheduled burn execution is owned by
/// [`crate::autopilot::Autopilot`], not by the maneuver/navigation
/// UI state.
#[derive(Resource, Debug, Default)]
pub struct NavigationState {
    pub mode: Option<NavigationMode>,
}

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

/// Body-frame "nose" axis for ship pointing. Apollo-style stacks have
/// their long axis along body Y, with the command pod at +Y; flipping
/// this would also flip the autopilot's pointing convention.
pub(crate) const SHIP_NOSE_BODY: DVec3 = thalos_control::NOSE_BODY;

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

/// World-frame unit vector pointing along the next maneuver node's Δv.
/// Returns `None` when no node exists or the burn direction is
/// degenerate.
///
/// Pointing for a maneuver requires the ship and reference body states
/// *at burn time* — using "now" gets the wrong PRN frame for any
/// non-instant burn. Uses the cached prediction; when it's missing
/// (right after a node edit) falls back to *both* states at current
/// time so the PRN frame stays internally consistent rather than
/// mixing ship-now with body-future.
///
/// Shared by [`compute_target_direction`] (driving the
/// [`NavigationMode::ManeuverNode`] pointing target) and the burn-
/// directive publisher's direction calculation.
pub(crate) fn maneuver_burn_direction(sim: &Simulation, plan: &ManeuverPlan) -> Option<DVec3> {
    // Point at the next burn the autopilot would fly — a still-planned or
    // currently-executing node — never a spent one lingering for display.
    maneuver_node_burn_direction(sim, plan.nodes.iter().find(|n| n.drives_directive())?)
}

/// World-frame unit vector pointing along a maneuver node's Δv.
pub(crate) fn maneuver_node_burn_direction(sim: &Simulation, node: &GameNode) -> Option<DVec3> {
    let ship = sim.ship_state();
    let time = sim.sim_time();
    let prediction_state = sim
        .prediction()
        .and_then(|p| p.pre_burn_state_at(node.time, sim.ephemeris(), sim.bodies()))
        .map(|s| thalos_world::StateVector {
            position: s.position,
            velocity: s.velocity,
        })
        .or_else(|| sim.prediction().and_then(|p| p.state_at(node.time)));
    let (ship_pos, ship_vel, frame_time) = match prediction_state {
        Some(s) => (s.position, s.velocity, node.time),
        None => (ship.position, ship.velocity, time),
    };
    let body_state = sim.ephemeris().state(
        node.reference_body,
        thalos_physics_canonical::canonical::Epoch(frame_time),
    );
    let dv_world = delta_v_to_world(
        node.delta_v,
        ship_vel,
        ship_pos,
        body_state.position,
        body_state.velocity,
    );
    safe_normalize(dv_world)
}

fn safe_normalize(v: DVec3) -> Option<DVec3> {
    if v.length_squared() < 1e-20 {
        None
    } else {
        Some(v.normalize())
    }
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
