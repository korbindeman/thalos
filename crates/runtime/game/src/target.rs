//! Target-body selection state.
//!
//! `Escape` clears the selected target. The selected target drives the
//! ghost-body projection in
//! [`crate::flight_plan_view`]: a translucent duplicate of the target body placed
//! at its future position at the flight plan's closest-approach epoch.

use bevy::prelude::*;

use crate::rendering::SimulationState;

pub use thalos_game_state::nav::TargetBody;

pub struct TargetPlugin;

impl Plugin for TargetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TargetBody>()
            .add_systems(Update, sync_target_to_simulation);
    }
}

/// Forward the current `TargetBody` resource into the physics `Simulation`
/// so trajectory prediction can bias its step size near the target.
fn sync_target_to_simulation(target: Res<TargetBody>, sim: Option<ResMut<SimulationState>>) {
    let Some(mut sim) = sim else { return };
    if sim.simulation.target_body() != target.target {
        sim.simulation.set_target_body(target.target);
    }
}
