//! Bridge between the physics simulation and Bevy ECS.
//!
//! All simulation state lives in [`Simulation`] (physics crate). This module
//! is a thin adapter that:
//!
//! 1. Calls [`Simulation::step`] each frame to advance the ship.
//! 2. Periodically calls [`Simulation::recompute_prediction`] for the renderer.
//! 3. Maps keyboard input to warp controls.

use bevy::prelude::*;

use crate::rendering::SimulationState;
use crate::SimStage;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Re-run trajectory prediction every this many frames.
const PREDICTION_INTERVAL_FRAMES: u32 = 120;

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

pub fn advance_simulation(time: Res<Time>, mut sim: ResMut<SimulationState>) {
    sim.simulation.step(time.delta_secs_f64());
}

pub fn update_prediction(mut sim: ResMut<SimulationState>, mut frame_counter: Local<u32>) {
    *frame_counter += 1;
    if *frame_counter < PREDICTION_INTERVAL_FRAMES {
        return;
    }
    *frame_counter = 0;
    sim.simulation.recompute_prediction();
}

/// Handle keyboard input to adjust the warp multiplier.
///
/// - `=` / `+`  -- increase to next warp level
/// - `-`        -- decrease to previous warp level
/// - `\`        -- reset to 1x
pub fn handle_warp_controls(keys: Res<ButtonInput<KeyCode>>, mut sim: ResMut<SimulationState>) {
    let prev = sim.simulation.warp_speed();

    if keys.just_pressed(KeyCode::Equal) {
        sim.simulation.increase_warp();
    } else if keys.just_pressed(KeyCode::Minus) {
        sim.simulation.decrease_warp();
    } else if keys.just_pressed(KeyCode::Backslash) {
        sim.simulation.reset_warp();
    }

    let new = sim.simulation.warp_speed();
    if (new - prev).abs() > 0.5 {
        println!("[bridge] warp speed: {}", sim.simulation.warp_label());
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct BridgePlugin;

impl Plugin for BridgePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (handle_warp_controls, advance_simulation, update_prediction)
                .chain()
                .in_set(SimStage::Physics),
        );
    }
}
