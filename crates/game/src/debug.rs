//! Debug utilities. Hardcoded on for now; later this becomes an
//! in-game settings toggle.

use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use thalos_physics::{
    body_fixed::body_fixed_surface_velocity,
    canonical::BodyFixedPose,
    debug_orbits::debug_parking_orbit_state,
    types::{AttitudeState, BodyDefinition, BodyId, BodyState, StateVector},
};

use crate::navigation::SHIP_NOSE_BODY;

#[derive(Resource, Debug, Clone, Copy)]
pub struct DebugMode {
    pub enabled: bool,
}

/// Temporary debug-only launch clamp used by command-shift body-tree surface
/// spawns. It keeps the craft in a stable body-fixed pose above terrain until
/// the player applies throttle, at which point game-side local physics releases
/// it. Remove this once real staging/launch clamps exist.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct DebugLaunchMount {
    pub active: Option<DebugLaunchMountState>,
}

#[derive(Debug, Clone, Copy)]
pub struct DebugLaunchMountState {
    pub body_id: BodyId,
    pub pose: BodyFixedPose,
}

pub struct DebugPlugin;

impl Plugin for DebugPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DebugMode { enabled: true })
            .init_resource::<DebugLaunchMount>();
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

/// Body-fixed unit direction to the equatorial subsolar meridian — the
/// point on the body's equator on the longitude line directly facing the
/// sun. Used by debug surface spawn so the ship lands in daylight on the
/// most-lit meridian.
///
/// Body-frame +Y is the spin axis (geographic north pole). The sun
/// direction is projected onto the equatorial plane to get the meridian.
/// Falls back to body-frame +X when the sun sits directly above a pole
/// (degenerate equatorial projection).
pub fn subsolar_equator_dir_body(body_state: &BodyState, sun_state: &BodyState) -> DVec3 {
    let to_sun_world = sun_state.position - body_state.position;
    let to_sun_body = body_state.orientation.inverse() * to_sun_world;
    let pole = DVec3::Y;
    let equatorial = to_sun_body - pole * to_sun_body.dot(pole);
    equatorial.try_normalize().unwrap_or(DVec3::X)
}

/// Compute a surface-aligned debug spawn state for `body`.
///
/// `dir_body` is a body-fixed unit direction, and `surface_height_m` is the
/// rendered-terrain height at that direction. The returned craft is stationary
/// relative to the rotating surface and upright for a rocket launch: ship
/// nose +Y points along local up, with roll chosen from the local east tangent
/// when the body has a spin axis.
pub fn surface_spawn_state(
    body: &BodyDefinition,
    body_state: &BodyState,
    dir_body: DVec3,
    surface_height_m: f64,
    clearance_m: f64,
) -> (StateVector, AttitudeState) {
    let up_body = dir_body.normalize();
    let position_body = up_body * (body.radius_m + surface_height_m + clearance_m);
    let state = StateVector {
        position: body_state.position + body_state.orientation * position_body,
        velocity: body_fixed_surface_velocity(body_state, position_body),
    };
    let attitude = AttitudeState {
        orientation: level_surface_attitude(body_state, up_body),
        angular_velocity: DVec3::ZERO,
    };
    (state, attitude)
}

fn level_surface_attitude(body_state: &BodyState, up_body: DVec3) -> DQuat {
    let nose_body = up_body.normalize();
    let spin_body = body_state.orientation.inverse() * body_state.angular_velocity;
    let mut dorsal_body = spin_body.cross(nose_body);
    if dorsal_body.length_squared() < 1.0e-18 {
        let reference = if nose_body.y.abs() < 0.9 {
            DVec3::Y
        } else {
            DVec3::X
        };
        dorsal_body = (reference - nose_body * reference.dot(nose_body)).normalize();
    } else {
        dorsal_body = dorsal_body.normalize();
    }
    let right_body = nose_body.cross(dorsal_body).normalize();
    let craft_to_body = DMat3::from_cols(right_body, nose_body, dorsal_body);
    (body_state.orientation * DQuat::from_mat3(&craft_to_body)).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use thalos_physics::canonical::Epoch;
    use thalos_physics::types::BodyKind;

    fn body_definition() -> BodyDefinition {
        BodyDefinition {
            id: 1,
            name: "Test".to_string(),
            kind: BodyKind::Planet,
            parent: None,
            mass_kg: 1.0e20,
            radius_m: 1000.0,
            color: [1.0, 1.0, 1.0],
            rotation_period_s: 86_400.0,
            axial_tilt_rad: 0.0,
            gm: 1.0,
            soi_radius_m: 100_000.0,
            orbital_elements: None,
            terrain: thalos_terrain_gen::TerrainConfig::None,
            tectonics: None,
            atmosphere: None,
            terrestrial_atmosphere: None,
            rings: None,
        }
    }

    fn body_state() -> BodyState {
        BodyState {
            id: 1,
            epoch: Epoch(0.0),
            position: DVec3::new(100.0, 20.0, -30.0),
            velocity: DVec3::new(5.0, 0.0, -2.0),
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::Y * 0.1,
            mass_kg: 1.0e20,
            gm: 1.0,
            radius_m: 1000.0,
        }
    }

    #[test]
    fn surface_spawn_matches_rotating_surface_velocity_and_up_attitude() {
        let body = body_definition();
        let body_state = body_state();
        let dir_body = DVec3::Z;
        let (state, attitude) = surface_spawn_state(&body, &body_state, dir_body, 12.0, 8.0);
        let position_body = dir_body * 1020.0;

        assert!((state.position - (body_state.position + position_body)).length() < 1.0e-9);
        assert!(
            (state.velocity - body_fixed_surface_velocity(&body_state, position_body)).length()
                < 1.0e-9
        );
        assert!(((attitude.orientation * DVec3::Y) - dir_body).length() < 1.0e-9);
        assert!((attitude.orientation * DVec3::Z).dot(DVec3::X) > 0.999);
    }
}
