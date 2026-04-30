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

/// Per-control-surface lockout flags. `true` = a programmatic system
/// is currently driving this surface and human input should be
/// dropped. Defaults are all `false` (everything free).
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct ControlLocks {
    /// Throttle setting — Z/X snap, Shift/Ctrl ramp, future throttle
    /// slider. Gated in [`crate::fuel::handle_throttle_input`].
    pub throttle: bool,
    /// Attitude torque commands — W/A/S/D/Q/E. Gated in
    /// [`crate::bridge::handle_attitude_controls`] (player_torque is
    /// zeroed; the autopilot's pointing target still runs through
    /// `compute_attitude_control`). The T (SAS toggle) key is not
    /// gated — it just flips a state bool that's irrelevant while the
    /// autopilot owns attitude.
    pub attitude: bool,
    /// Warp level changes — `.` / `,` / `\` keys, HUD `<` / `>` /
    /// `→ Next` buttons. Pause (Space, HUD pause if added) stays
    /// available unconditionally; that exemption lives in the warp
    /// handler itself rather than as a separate flag here.
    pub warp: bool,
    /// Navigation mode buttons in the side panel (Stability, Prograde,
    /// …, Maneuver). The autopilot checkbox at the top of the same
    /// panel is *not* gated — it's the only override path while the
    /// autopilot is engaged.
    pub navigation_mode: bool,
}

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
                .after(autopilot_system),
        );
    }
}

fn update_control_locks(autopilot: Res<Autopilot>, mut locks: ResMut<ControlLocks>) {
    let active = autopilot.is_active();
    *locks = ControlLocks {
        throttle: active,
        attitude: active,
        warp: active,
        navigation_mode: active,
    };
}
