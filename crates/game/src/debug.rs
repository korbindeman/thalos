//! Debug utilities. Hardcoded on for now; later this becomes an
//! in-game settings toggle.

use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use thalos_physics::{
    debug_orbits::debug_parking_orbit_state,
    types::{AttitudeState, BodyDefinition, BodyState, StateVector},
};

use crate::navigation::SHIP_NOSE_BODY;

#[derive(Resource, Debug, Clone, Copy)]
pub struct DebugMode {
    pub enabled: bool,
}

pub struct DebugPlugin;

impl Plugin for DebugPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DebugMode { enabled: true });
    }
}

/// Compute a near-circular low-orbit state vector around `body` at the given
/// `body_state` (the body's heliocentric state at the current sim_time).
///
/// Uses the same 200 km debug parking-orbit helper as initial ship spawn,
/// capped so small-body teleports stay inside the body's SOI.
///
/// Returns the heliocentric state plus a body→world attitude that points
/// the ship's nose along its prograde velocity.
pub fn low_orbit_state(
    body: &BodyDefinition,
    body_state: &BodyState,
) -> (StateVector, AttitudeState) {
    let state = debug_parking_orbit_state(body, body_state);
    let rel_vel = state.velocity - body_state.velocity;
    let attitude = AttitudeState {
        orientation: DQuat::from_rotation_arc(SHIP_NOSE_BODY, rel_vel.normalize()),
        angular_velocity: DVec3::ZERO,
    };
    (state, attitude)
}
