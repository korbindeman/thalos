//! Target-body selection and cycling.
//!
//! The player picks a target body with `Tab` (forward) / `Shift+Tab` (reverse);
//! `Escape` clears. The selected target drives the ghost-body projection in
//! [`crate::ghost_bodies`]: a translucent duplicate of the target body placed
//! at its future position at the flight plan's closest-approach epoch.
//!
//! Only non-star bodies are cycled.

use bevy::prelude::*;

use crate::rendering::SimulationState;

/// Currently selected target body (world ID, not entity id).
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct TargetBody {
    pub target: Option<usize>,
}

pub struct TargetPlugin;

impl Plugin for TargetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TargetBody>()
            .add_systems(Update, cycle_target_input);
    }
}

fn cycle_target_input(
    keys: Res<ButtonInput<KeyCode>>,
    sim: Option<Res<SimulationState>>,
    mut target: ResMut<TargetBody>,
) {
    let Some(sim) = sim else { return };

    if keys.just_pressed(KeyCode::Escape) && target.target.is_some() {
        target.target = None;
        return;
    }

    if !keys.just_pressed(KeyCode::Tab) {
        return;
    }
    let reverse = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    // Candidate list: every non-star body.
    let candidates: Vec<usize> = sim
        .system
        .bodies
        .iter()
        .filter(|b| b.kind != thalos_physics::types::BodyKind::Star)
        .map(|b| b.id)
        .collect();
    if candidates.is_empty() {
        return;
    }

    let current_index = target
        .target
        .and_then(|id| candidates.iter().position(|c| *c == id));
    let next = match (current_index, reverse) {
        (None, false) => 0,
        (None, true) => candidates.len() - 1,
        (Some(i), false) => (i + 1) % candidates.len(),
        (Some(i), true) => (i + candidates.len() - 1) % candidates.len(),
    };
    target.target = Some(candidates[next]);
    info!(
        "[target] selected {}",
        sim.system.bodies[candidates[next]].name
    );
}
