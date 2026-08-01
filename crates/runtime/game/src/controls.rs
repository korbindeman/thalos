//! Single source of truth for "what user controls are currently
//! disabled." Programmatic systems (today: the scheduled-burn
//! autopilot in [`crate::autopilot`]; tomorrow: docking autopilot,
//! kill-rot, photo-mode lockouts, …) push their requirements into one
//! [`ControlLocks`] resource each frame. Input handlers and UI panels
//! read from that resource — none of them know which subsystem
//! actually demanded the lock, only that they should treat input as
//! disallowed.
//!
//! Adding a new locker is a one-line change inside
//! [`update_control_locks`]; adding a new lockable surface is a new
//! field on [`ControlLocks`] plus checks at the (one) handler that
//! drives that surface. Without this resource the policy is smeared
//! across every input system, and extending the autopilot means
//! editing every one of them.

use bevy::prelude::*;

use crate::SimStage;
use crate::autopilot::{Autopilot, autopilot_system};
use crate::orbit_program::OrbitProgram;
use crate::route_autopilot::{LandAutopilot, update_land_autopilot};

pub use thalos_game_state::flight::ControlLocks;


pub struct ControlLocksPlugin;

impl Plugin for ControlLocksPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ControlLocks>().add_systems(
            Update,
            // Run after the autopilot so locks reflect this frame's
            // state-transition outcome, not last frame's. The natural
            // state→derived-state ordering — readers reason about it
            // more easily, even though the input-handler observable
            // behaviour is identical to the reverse order (one-frame
            // lag either way).
            update_control_locks
                .in_set(SimStage::Physics)
                .after(autopilot_system)
                .after(update_land_autopilot),
        );
    }
}

pub(crate) fn update_control_locks(
    autopilot: Res<Autopilot>,
    land: Res<LandAutopilot>,
    orbit: Res<OrbitProgram>,
    mut locks: ResMut<ControlLocks>,
) {
    // Each source answers for itself; the union is the policy. Nothing here
    // pattern-matches a mode enum, which is what makes it impossible to
    // reintroduce the defect this replaced: the old table read
    // `warp: maneuver || landing || orbiting`, where `orbiting` was true for
    // the *whole* ascent program — including the ballistic coast, which is
    // nothing but waiting. That killed warp-to-node for the minutes it was
    // most wanted, and `warp_to_maneuver_system` cancels itself on sight of
    // the flag, so the HUD's WARP button silently did nothing.
    //
    // A source knows whether it is time-critical this instant; a mode enum
    // cannot. Adding an executor now means adding a `required_locks` to it,
    // not editing a table that has to know about every executor.
    let required = autopilot
        .required_locks()
        .union(land.required_locks())
        .union(orbit.required_locks());

    *locks = ControlLocks {
        throttle: required.throttle,
        attitude: required.attitude,
        warp: required.warp,
        navigation_mode: required.navigation_mode,
        ground_steer: required.ground_steer,
        wheel_brake: required.wheel_brake,
    };
}
