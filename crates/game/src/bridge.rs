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

/// Warn if a synchronous prediction pass eats more than this fraction of a
/// 60 Hz frame. Purely diagnostic — no behaviour change.
const SYNC_PREDICTION_WARN_MS: f64 = 8.0;

pub fn advance_simulation(time: Res<Time>, mut sim: ResMut<SimulationState>) {
    let _span = tracing::info_span!("advance_simulation").entered();
    sim.simulation.step(time.delta_secs_f64());
}

fn update_prediction(mut sim: ResMut<SimulationState>) {
    let _span = tracing::info_span!("update_prediction").entered();

    if sim.simulation.is_observation_mode() {
        return;
    }

    if !sim.simulation.prediction_needs_refresh() {
        return;
    }

    let t0 = std::time::Instant::now();
    sim.simulation.recompute_prediction();
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if elapsed_ms > SYNC_PREDICTION_WARN_MS {
        warn!(
            "[bridge] sync prediction pass took {:.1} ms (> {:.0} ms budget)",
            elapsed_ms, SYNC_PREDICTION_WARN_MS
        );
    }
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

/// Sync the UI-side [`ManeuverPlan`] to the physics `ManeuverSequence` when
/// dirty. `maneuvers_mut()` marks the prediction dirty; `update_prediction`
/// runs after this system and recomputes synchronously.
///
/// Also drops UI nodes whose execution time has passed — the physics side
/// auto-consumes them as burns, and leaving stale nodes in the UI would
/// re-inject them on the next edit.
fn sync_maneuver_plan(mut plan: ResMut<ManeuverPlan>, mut sim: ResMut<SimulationState>) {
    let sim_time = sim.simulation.sim_time();
    let before = plan.nodes.len();
    plan.nodes.retain(|n| n.time > sim_time);
    if plan.nodes.len() != before {
        plan.dirty = true;
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
