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

use bevy::prelude::*;
use thalos_physics::maneuver::ManeuverNode;

use crate::SimStage;
use crate::maneuver::ManeuverPlan;
use crate::rendering::SimulationState;

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
            dx, dt, warp, dx / max_reasonable,
            pre_pos.x, pre_pos.y, pre_pos.z,
            post_pos.x, post_pos.y, post_pos.z,
        );
    }
    if !post_pos.is_finite() {
        warn!(
            "ship position went non-finite: {:?} at sim_time {:.3} (warp={:.0}x)",
            post_pos, post_t, warp,
        );
    }
}

fn update_prediction(mut sim: ResMut<SimulationState>) {
    let _span = tracing::info_span!("update_prediction").entered();

    if !sim.simulation.prediction_needs_refresh() {
        return;
    }

    sim.simulation.recompute_prediction();
}

/// Handle keyboard input to adjust the warp multiplier.
///
/// - `.`      -- increase to next warp level
/// - `,`      -- decrease to previous warp level (0x = paused)
/// - `\`      -- reset to 1x
/// - `Space`  -- toggle pause (0x) / resume previous level
pub fn handle_warp_controls(keys: Res<ButtonInput<KeyCode>>, mut sim: ResMut<SimulationState>) {
    let prev = sim.simulation.warp.speed();

    if keys.just_pressed(KeyCode::Period) {
        sim.simulation.warp.increase();
    } else if keys.just_pressed(KeyCode::Comma) {
        sim.simulation.warp.decrease();
    } else if keys.just_pressed(KeyCode::Backslash) {
        sim.simulation.warp.reset();
    } else if keys.just_pressed(KeyCode::Space) {
        sim.simulation.warp.toggle_pause();
    }

    let new = sim.simulation.warp.speed();
    if (new - prev).abs() > 0.5 {
        info!("[bridge] warp speed: {}", sim.simulation.warp.label());
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
                advance_simulation,
                sync_maneuver_plan,
                update_prediction,
            )
                .chain()
                .in_set(SimStage::Physics),
        );
    }
}
