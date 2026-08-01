//! Debug-mode state shared across features (the F3 debug overlay mode and
//! the debug surface-teleport arming) plus the pure orbit helpers debug
//! teleports use. The debug systems stay with the runtime.

use bevy::prelude::*;
use thalos_physics_canonical::types::{AttitudeState, BodyState};
use bevy::math::{DQuat, DVec3};
use thalos_physics_canonical::debug_orbits::debug_parking_orbit_state;
use thalos_world::{BodyDefinition, BodyId, StateVector};

use crate::nav::SHIP_NOSE_BODY;

#[derive(Resource, Debug, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub struct DebugMode {
    pub enabled: bool,
    /// Draw physics hitboxes (craft colliders + gear contact + ground surface).
    /// Toggled by F3; see [`draw_debug_hitboxes`].
    pub show_hitboxes: bool,
    /// Debug hack: let air-breathing engines produce rated sea-level thrust
    /// regardless of atmosphere — fire in vacuum, no density lapse — so
    /// aircraft can taxi/fly on airless bodies for ground/wheel testing.
    /// Contradicts the atmosphere model; **off by default** now that Thalos
    /// has air (leaving it on pinned every jet at rated thrust and defeated
    /// the thrust lapse / transonic wall). Edit the default and rebuild to
    /// toggle it (Reflect-registered for a future debug UI).
    pub jets_in_vacuum: bool,
}

/// Explicit map-view debug teleport mode.
///
/// A body-tree `drop` button arms this resource. While armed, the map cursor
/// raycasts against that body's visible disc, draws a small cursor on the
/// corresponding terrain direction, and left-clicking mounts the craft at the
/// rendered height under the cursor.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct DebugSurfaceTeleport {
    pub armed_body: Option<BodyId>,
    pub hover: Option<DebugSurfaceTeleportHit>,
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

#[derive(Debug, Clone, Copy)]
pub struct DebugSurfaceTeleportHit {
    pub body_id: BodyId,
    pub dir_body: DVec3,
    pub surface_height_m: f64,
    pub render_pos: Vec3,
    pub normal_render: Vec3,
    pub used_rendered_surface: bool,
}

impl DebugSurfaceTeleport {
    pub fn arm(&mut self, body_id: BodyId) {
        self.armed_body = Some(body_id);
        self.hover = None;
    }

    pub fn cancel(&mut self) {
        *self = Self::default();
    }
}
