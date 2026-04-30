//! Warp-to-next-maneuver auto-warp.
//!
//! Toggling [`WarpToManeuver::active`] (via the `G` keybind or the HUD
//! button in [`crate::hud`]) fast-forwards to the next scheduled
//! maneuver node. Each frame the system picks the highest discrete
//! warp level whose one-frame advance won't overshoot the autopilot's
//! lead window — a few seconds of 1× before the burn so the attitude
//! PD has time to settle — then drops to 1× and disengages, handing
//! warp to the scheduled-burn autopilot.
//!
//! Coexistence with the scheduled-burn autopilot. The autopilot's
//! [`crate::autopilot::lead_seconds_for`] sizes the safe window for
//! the attitude PD to settle plus the pre-burn half-duration. We use
//! the same formula here, so a free-running auto-warp ramps speed
//! down to land at exactly the moment the autopilot would otherwise
//! have to start clamping. Once the autopilot transitions into its
//! `Engaging` or `Burn` state and asserts [`ControlLocks::warp`], we
//! cancel auto-warp; the autopilot owns warp from there.

use bevy::prelude::*;

use thalos_physics::simulation::Simulation;

use crate::SimStage;
use crate::autopilot::{autopilot_system, lead_seconds_for};
use crate::bridge::{advance_simulation, handle_warp_controls};
use crate::controls::ControlLocks;
use crate::rendering::SimulationState;

/// Worst-case real-frame budget — must match
/// [`thalos_physics::simulation::SimulationConfig::max_real_delta`] so
/// a single frame at the chosen level can't advance past the safe
/// target.
const FRAME_DT_BUDGET_S: f64 = 0.1;

#[derive(Clone, Copy, Debug)]
pub struct ManeuverTarget {
    pub epoch: f64,
    /// Tsiolkovsky burn duration computed when the target was selected.
    /// Lets the safe-target calculation reuse the autopilot's lead
    /// formula without re-querying the simulation.
    pub duration_s: f64,
}

impl ManeuverTarget {
    /// Short HUD label such as "Maneuver in 1h 23m".
    pub fn label(&self, now: f64) -> String {
        let remaining = (self.epoch - now).max(0.0);
        format!("Maneuver in {}", format_duration(remaining))
    }
}

fn format_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.0}s", seconds)
    } else if seconds < 3600.0 {
        let m = (seconds / 60.0).floor();
        let s = seconds - m * 60.0;
        format!("{:.0}m {:02.0}s", m, s)
    } else if seconds < 86400.0 {
        let h = (seconds / 3600.0).floor();
        let m = ((seconds - h * 3600.0) / 60.0).floor();
        format!("{:.0}h {:02.0}m", h, m)
    } else {
        let d = (seconds / 86400.0).floor();
        let h = ((seconds - d * 86400.0) / 3600.0).floor();
        format!("{:.0}d {:02.0}h", d, h)
    }
}

#[derive(Resource, Default)]
pub struct WarpToManeuver {
    /// `true` while the auto-warp is engaged. Cleared on arrival, when
    /// no upcoming maneuver remains, when an active burn takes over, or
    /// when the player nudges warp manually (handled in
    /// [`crate::bridge::handle_warp_controls`]).
    pub active: bool,
    /// Latest target as of the most recent system tick — drives the
    /// HUD readout. `None` whenever auto-warp is off.
    pub current: Option<ManeuverTarget>,
}

impl WarpToManeuver {
    pub fn cancel(&mut self) {
        self.active = false;
        self.current = None;
    }
}

/// Soonest scheduled maneuver after `sim_time`. Skips past zero-Δv
/// nodes (placeholders that wouldn't fire). The strict `>` filter
/// rejects a node sitting at the current epoch — that node is already
/// mid-execution or stale.
pub fn find_next_maneuver(sim_time: f64, simulation: &Simulation) -> Option<ManeuverTarget> {
    for node in simulation.maneuvers().iter() {
        if node.time <= sim_time {
            continue;
        }
        let dv_mag = node.delta_v.length();
        if dv_mag <= 0.0 {
            continue;
        }
        return Some(ManeuverTarget {
            epoch: node.time,
            duration_s: simulation.estimated_burn_duration(dv_mag),
        });
    }
    None
}

pub(crate) fn warp_to_maneuver_system(
    mut state: ResMut<WarpToManeuver>,
    mut sim: ResMut<SimulationState>,
    locks: Res<ControlLocks>,
) {
    if !state.active {
        state.current = None;
        return;
    }

    if locks.warp {
        // The autopilot has entered its Engaging or Burn state and
        // asserted the warp lock. Hand off — it owns warp now.
        state.cancel();
        return;
    }

    let now = sim.simulation.sim_time();
    let Some(target) = find_next_maneuver(now, &sim.simulation) else {
        state.cancel();
        return;
    };
    let safe = target.epoch - lead_seconds_for(target.duration_s);
    let remaining = safe - now;

    if remaining <= 0.0 {
        sim.simulation.warp.reset();
        state.cancel();
        return;
    }

    let max_speed = (remaining / FRAME_DT_BUDGET_S).max(1.0);
    sim.simulation.warp.set_speed(max_speed);
    state.current = Some(target);
}

pub struct WarpToManeuverPlugin;

impl Plugin for WarpToManeuverPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WarpToManeuver>().add_systems(
            Update,
            warp_to_maneuver_system
                .in_set(SimStage::Physics)
                .after(handle_warp_controls)
                .before(autopilot_system)
                .before(advance_simulation),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_duration_units() {
        assert_eq!(format_duration(7.0), "7s");
        assert_eq!(format_duration(125.0), "2m 05s");
        assert_eq!(format_duration(3725.0), "1h 02m");
        assert_eq!(format_duration(90061.0), "1d 01h");
    }

    #[test]
    fn maneuver_target_label_formats_remaining_time() {
        let target = ManeuverTarget {
            epoch: 7325.0,
            duration_s: 30.0,
        };
        assert_eq!(target.label(0.0), "Maneuver in 2h 02m");
    }
}
