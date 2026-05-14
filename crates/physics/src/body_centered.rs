//! Body-centered inertial frame helpers.
//!
//! The frame translates with the dominant body's center but its axes are the
//! parent inertial axes — they do not rotate with the body. This is the frame
//! the Avian rigid body lives in, so its integrator sees `dv/dt = −μr/r³`
//! with no fictitious forces.
//!
//! The frame is treated as inertial for local-physics purposes. The body's
//! own translational acceleration (orbital motion around its parent) is
//! ignored — for low-altitude work this is a tidal-only error and small
//! enough to defer.

use crate::canonical::TranslationalState;
use crate::types::{AttitudeState, BodyState};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyCenteredState {
    pub translation_bc: TranslationalState,
    pub attitude: AttitudeState,
}

pub fn inertial_to_body_centered(
    body: &BodyState,
    translation: TranslationalState,
    attitude: AttitudeState,
) -> BodyCenteredState {
    BodyCenteredState {
        translation_bc: TranslationalState {
            position: translation.position - body.position,
            velocity: translation.velocity - body.velocity,
        },
        attitude,
    }
}

pub fn body_centered_to_inertial(
    body: &BodyState,
    state: BodyCenteredState,
) -> (TranslationalState, AttitudeState) {
    (
        TranslationalState {
            position: state.translation_bc.position + body.position,
            velocity: state.translation_bc.velocity + body.velocity,
        },
        state.attitude,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::Epoch;
    use glam::{DQuat, DVec3};

    fn test_body() -> BodyState {
        BodyState {
            id: 0,
            epoch: Epoch(0.0),
            position: DVec3::new(100.0, 20.0, -50.0),
            velocity: DVec3::new(5.0, 0.0, -2.0),
            orientation: DQuat::from_rotation_z(0.3),
            angular_velocity: DVec3::Y * 0.25,
            mass_kg: 1.0e20,
            gm: 1.0,
            radius_m: 1000.0,
        }
    }

    #[test]
    fn round_trip_preserves_inertial_state() {
        let body = test_body();
        let translation = TranslationalState {
            position: DVec3::new(1500.0, 200.0, -110.0),
            velocity: DVec3::new(10.0, 3.0, -7.0),
        };
        let attitude = AttitudeState {
            orientation: DQuat::from_rotation_x(0.2) * DQuat::from_rotation_y(-0.1),
            angular_velocity: DVec3::new(0.01, 0.02, -0.03),
        };

        let state = inertial_to_body_centered(&body, translation, attitude);
        let (translation_rt, attitude_rt) = body_centered_to_inertial(&body, state);

        assert!((translation_rt.position - translation.position).length() < 1e-12);
        assert!((translation_rt.velocity - translation.velocity).length() < 1e-12);
        assert!(attitude_rt.orientation.angle_between(attitude.orientation) < 1e-12);
        assert!((attitude_rt.angular_velocity - attitude.angular_velocity).length() < 1e-12);
    }

    #[test]
    fn body_centered_position_subtracts_body_offset() {
        let body = test_body();
        let translation = TranslationalState {
            position: body.position + DVec3::X * 1000.0,
            velocity: body.velocity,
        };
        let state =
            inertial_to_body_centered(&body, translation, AttitudeState::default());
        assert!((state.translation_bc.position - DVec3::X * 1000.0).length() < 1e-12);
        assert!(state.translation_bc.velocity.length() < 1e-12);
    }

    #[test]
    fn body_centered_velocity_excludes_body_translational_velocity_only() {
        // A craft sitting at the body's surface, co-rotating with the body,
        // is NOT at rest in body-centered inertial — the rotational surface
        // velocity is still part of the craft's inertial velocity. Verify
        // that we subtract only the body's center-of-mass translation.
        let body = test_body();
        let surface_offset = DVec3::X * body.radius_m;
        let surface_world_vel = body.velocity + body.angular_velocity.cross(surface_offset);
        let translation = TranslationalState {
            position: body.position + surface_offset,
            velocity: surface_world_vel,
        };
        let state =
            inertial_to_body_centered(&body, translation, AttitudeState::default());
        let expected_vel = body.angular_velocity.cross(surface_offset);
        assert!((state.translation_bc.velocity - expected_vel).length() < 1e-12);
    }
}
