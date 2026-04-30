//! Bridge between the physics simulation and Bevy ECS.
//!
//! All simulation state lives in [`Simulation`] (physics crate). This module
//! is a thin adapter that:
//!
//! 1. Calls [`Simulation::step`] each frame to advance the ship.
//! 2. Recomputes trajectory prediction synchronously on the main thread
//!    whenever the maneuver plan is dirty or the cached result is stale.
//!    A single `propagate_flight_plan` pass terminates early (stable orbit,
//!    collision, cone fade), keeping the typical pass well under one frame.
//!    Running in-line means an edit on frame N produces the fresh trajectory
//!    on frame N — no worker-thread lag.
//! 3. Maps keyboard input to warp controls.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics::maneuver::ManeuverNode;

use crate::SimStage;
use crate::autopilot::Autopilot;
use crate::controls::ControlLocks;
use crate::fuel::ThrottleState;
use crate::maneuver::ManeuverPlan;
use crate::navigation::{NavigationState, compute_attitude_control};
use crate::rendering::SimulationState;
use crate::target::TargetBody;
use crate::warp_to_maneuver::{WarpToManeuver, find_next_maneuver};

pub fn advance_simulation(time: Res<Time>, mut sim: ResMut<SimulationState>) {
    let _span = tracing::info_span!("advance_simulation").entered();

    let pre_pos = sim.simulation.ship_state().position;
    let pre_t = sim.simulation.sim_time();

    sim.simulation.step(time.delta_secs_f64());

    // Diagnostic: log anything that looks physically impossible so we can
    // catch state corruption the instant it happens instead of noticing it
    // visually a few frames later.
    let post_pos = sim.simulation.ship_state().position;
    let post_t = sim.simulation.sim_time();
    let dt = post_t - pre_t;
    let dx = (post_pos - pre_pos).length();
    let warp = sim.simulation.warp.speed();
    // A ship in LEO around Thalos moves ~7 km/s relative to the body, but
    // the body itself drifts heliocentrically at ~30 km/s. Cap at
    // 100 km/s * dt as a rough sanity ceiling.
    let max_reasonable = 1.0e5 * dt.max(1.0);
    if dt > 0.0 && dx > max_reasonable {
        warn!(
            "ship jumped {:.3e} m in {:.2}s (warp={:.0}x, ratio={:.3}x reasonable); pre=({:.3e},{:.3e},{:.3e}) post=({:.3e},{:.3e},{:.3e})",
            dx,
            dt,
            warp,
            dx / max_reasonable,
            pre_pos.x,
            pre_pos.y,
            pre_pos.z,
            post_pos.x,
            post_pos.y,
            post_pos.z,
        );
    }
    if !post_pos.is_finite() {
        warn!(
            "ship position went non-finite: {:?} at sim_time {:.3} (warp={:.0}x)",
            post_pos, post_t, warp,
        );
    }
}

fn hold_prediction_for_scheduled_burn(autopilot: &Autopilot, throttle: &ThrottleState) -> bool {
    autopilot.is_burning() && throttle.effective > 0.0
}

fn update_prediction(
    autopilot: Res<Autopilot>,
    throttle: Res<ThrottleState>,
    mut sim: ResMut<SimulationState>,
) {
    let _span = tracing::info_span!("update_prediction").entered();

    if hold_prediction_for_scheduled_burn(&autopilot, &throttle) {
        return;
    }

    if !sim.simulation.prediction_needs_refresh() {
        return;
    }

    sim.simulation.recompute_prediction();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autopilot::{AutopilotDirectiveId, AutopilotState};

    #[test]
    fn scheduled_burn_with_effective_throttle_holds_prediction() {
        let mut autopilot = Autopilot::default();
        autopilot.state = AutopilotState::Burn {
            directive_id: AutopilotDirectiveId::new("test", 1),
            direction: DVec3::Y,
            planned_dv: 10.0,
            anchor_delivered_dv: 0.0,
        };
        let throttle = ThrottleState {
            commanded: 1.0,
            effective: 1.0,
        };

        assert!(hold_prediction_for_scheduled_burn(&autopilot, &throttle));
    }

    #[test]
    fn scheduled_burn_without_effective_throttle_can_refresh() {
        let mut autopilot = Autopilot::default();
        autopilot.state = AutopilotState::Burn {
            directive_id: AutopilotDirectiveId::new("test", 1),
            direction: DVec3::Y,
            planned_dv: 10.0,
            anchor_delivered_dv: 0.0,
        };
        let throttle = ThrottleState {
            commanded: 1.0,
            effective: 0.0,
        };

        assert!(!hold_prediction_for_scheduled_burn(&autopilot, &throttle));
    }
}

/// Sample player attitude input + active navigation mode and push the
/// resulting [`ControlInput`] into the simulation.
///
/// Player keys (W/S pitch, A/D yaw, Q/E roll) override any active
/// [`NavigationMode`] for the duration they're held; T toggles SAS for
/// free-flight rate damping. Mode-specific autopilot logic lives in
/// [`compute_attitude_control`] — this system just collects inputs.
///
/// Player torque is zeroed while [`ControlLocks::attitude`] is set so
/// whatever programmatic system holds the lock (today: the autopilot's
/// direct burn-pointing target) wins.
/// `compute_attitude_control` still runs — it's the path that drives
/// the autopilot's PD command from its direct target.
pub fn handle_attitude_controls(
    keys: Res<ButtonInput<KeyCode>>,
    nav: Res<NavigationState>,
    locks: Res<ControlLocks>,
    target: Res<TargetBody>,
    plan: Res<ManeuverPlan>,
    autopilot: Res<Autopilot>,
    mut sim: ResMut<SimulationState>,
    mut sas_enabled: Local<bool>,
) {
    if keys.just_pressed(KeyCode::KeyT) {
        *sas_enabled = !*sas_enabled;
    }

    let autopilot_target = autopilot.attitude_target();
    let mut player_torque = DVec3::ZERO;
    if !locks.attitude && autopilot_target.is_none() {
        if keys.pressed(KeyCode::KeyW) {
            player_torque.x += 1.0;
        }
        if keys.pressed(KeyCode::KeyS) {
            player_torque.x -= 1.0;
        }
        if keys.pressed(KeyCode::KeyD) {
            player_torque.z += 1.0;
        }
        if keys.pressed(KeyCode::KeyA) {
            player_torque.z -= 1.0;
        }
        if keys.pressed(KeyCode::KeyE) {
            player_torque.y += 1.0;
        }
        if keys.pressed(KeyCode::KeyQ) {
            player_torque.y -= 1.0;
        }
    }

    let control = compute_attitude_control(
        player_torque,
        nav.mode,
        autopilot_target,
        &target,
        &plan,
        &sim.simulation,
        *sas_enabled,
    );
    sim.simulation.set_control(control);
}

/// Handle keyboard input to adjust the warp multiplier.
///
/// - `.`      -- increase to next warp level
/// - `,`      -- decrease to previous warp level (0x = paused)
/// - `\`      -- reset to 1x
/// - `Space`  -- toggle pause (0x) / resume previous level
/// - `G`      -- toggle warp-to-next-maneuver auto-warp (see
///   [`crate::warp_to_maneuver`])
///
/// Warp-level changes are gated by [`ControlLocks::warp`] — when set,
/// some programmatic system (today: the scheduled-burn autopilot
/// during its lead-down) is driving warp and human nudges would just
/// get clobbered. Pause (Space) is always free; that exemption lives
/// here in the handler rather than as a separate lock flag, since the
/// throttle gate in
/// [`crate::fuel::gate_throttle_on_fuel_availability`] forces the
/// engine off at any non-1× warp anyway, so pausing mid-burn cleanly
/// suspends it and unpausing resumes. Pressing any manual warp key
/// cancels an in-progress auto-warp.
pub fn handle_warp_controls(
    keys: Res<ButtonInput<KeyCode>>,
    locks: Res<ControlLocks>,
    mut sim: ResMut<SimulationState>,
    mut warp_to: ResMut<WarpToManeuver>,
) {
    if keys.just_pressed(KeyCode::KeyG) {
        if warp_to.active {
            warp_to.cancel();
        } else if find_next_maneuver(sim.simulation.sim_time(), &sim.simulation).is_some() {
            warp_to.active = true;
        }
        return;
    }

    let manual_warp_key = keys.just_pressed(KeyCode::Space)
        || keys.just_pressed(KeyCode::Period)
        || keys.just_pressed(KeyCode::Comma)
        || keys.just_pressed(KeyCode::Backslash);
    if manual_warp_key && warp_to.active {
        warp_to.cancel();
    }

    if keys.just_pressed(KeyCode::Space) {
        sim.simulation.warp.toggle_pause();
        return;
    }

    if locks.warp {
        return;
    }

    if keys.just_pressed(KeyCode::Period) {
        sim.simulation.warp.increase();
    } else if keys.just_pressed(KeyCode::Comma) {
        sim.simulation.warp.decrease();
    } else if keys.just_pressed(KeyCode::Backslash) {
        sim.simulation.warp.reset();
    }
}

/// Sync the UI-side [`ManeuverPlan`] with the physics `ManeuverSequence`.
///
/// Lifecycle:
/// 1. Remove UI nodes that physics reports as consumed this frame. This is
///    the only way UI nodes retire — never by comparing time to `sim_time`,
///    which would silently drop nodes whose execution was skipped (e.g. at
///    observation warp). A UI node still sitting with `time <= sim_time`
///    means physics didn't burn it — a bug signal worth surfacing, not
///    hiding.
/// 2. When `plan.dirty` (user edit or consumption), push the current UI
///    list into physics, tagging each entry with its `NodeId` so the next
///    consumption cycle can round-trip.
fn sync_maneuver_plan(mut plan: ResMut<ManeuverPlan>, mut sim: ResMut<SimulationState>) {
    let consumed = sim.simulation.drain_consumed_node_ids();
    if !consumed.is_empty() {
        info!(
            "[bridge] physics consumed maneuver node ids: {:?} (sim_time={:.2})",
            consumed,
            sim.simulation.sim_time()
        );
        let before = plan.nodes.len();
        plan.nodes.retain(|n| !consumed.contains(&n.id.0));
        if plan.nodes.len() != before {
            plan.dirty = true;
        }
    }

    if !plan.dirty {
        return;
    }
    let _span = tracing::info_span!("sync_maneuver_plan").entered();
    plan.dirty = false;

    let seq = sim.simulation.maneuvers_mut();
    seq.nodes.clear();
    for node in &plan.nodes {
        seq.nodes.push(ManeuverNode {
            id: Some(node.id.0),
            time: node.time,
            delta_v: node.delta_v,
            reference_body: node.reference_body,
        });
    }
    seq.nodes.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct BridgePlugin;

impl Plugin for BridgePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                handle_warp_controls,
                handle_attitude_controls,
                advance_simulation,
                sync_maneuver_plan,
                update_prediction,
            )
                .chain()
                .in_set(SimStage::Physics),
        );
    }
}
