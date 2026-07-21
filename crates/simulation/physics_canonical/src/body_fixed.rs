//! Pure frame helpers for local rigidbody and landed body-fixed authority.

use glam::{DQuat, DVec3};

use crate::canonical::{BodyFixedPose, TranslationalState};
use crate::types::{AttitudeState, BodyState};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyFixedFrameState {
    pub translation_body: TranslationalState,
    /// Craft orientation relative to the body-fixed frame.
    pub orientation_body: DQuat,
    /// Craft angular velocity relative to the body-fixed frame, expressed in
    /// the craft body frame.
    pub angular_velocity_body: DVec3,
}

pub fn body_fixed_surface_velocity(body: &BodyState, position_body_m: DVec3) -> DVec3 {
    let offset_world = body.orientation * position_body_m;
    body.velocity + body.angular_velocity.cross(offset_world)
}

pub fn inertial_to_body_fixed(
    body: &BodyState,
    translation: TranslationalState,
    attitude: AttitudeState,
) -> BodyFixedFrameState {
    let offset_world = translation.position - body.position;
    let position_body_m = body.orientation.inverse() * offset_world;
    let surface_velocity = body_fixed_surface_velocity(body, position_body_m);
    let velocity_body = body.orientation.inverse() * (translation.velocity - surface_velocity);
    let orientation_body = (body.orientation.inverse() * attitude.orientation).normalize();
    let omega_world = attitude.orientation * attitude.angular_velocity;
    let angular_velocity_body =
        attitude.orientation.inverse() * (omega_world - body.angular_velocity);
    BodyFixedFrameState {
        translation_body: TranslationalState {
            position: position_body_m,
            velocity: velocity_body,
        },
        orientation_body,
        angular_velocity_body,
    }
}

pub fn body_fixed_to_inertial(
    body: &BodyState,
    frame: BodyFixedFrameState,
) -> (TranslationalState, AttitudeState) {
    let offset_world = body.orientation * frame.translation_body.position;
    let position = body.position + offset_world;
    let surface_velocity = body.velocity + body.angular_velocity.cross(offset_world);
    let velocity = surface_velocity + body.orientation * frame.translation_body.velocity;
    let orientation = (body.orientation * frame.orientation_body).normalize();
    let omega_world = body.angular_velocity + orientation * frame.angular_velocity_body;
    let angular_velocity = orientation.inverse() * omega_world;
    (
        TranslationalState { position, velocity },
        AttitudeState {
            orientation,
            angular_velocity,
        },
    )
}

pub fn body_fixed_pose_from_inertial(
    body: &BodyState,
    translation: TranslationalState,
    attitude: AttitudeState,
) -> BodyFixedPose {
    let frame = inertial_to_body_fixed(body, translation, attitude);
    BodyFixedPose {
        position_body_m: frame.translation_body.position,
        orientation_body: frame.orientation_body,
    }
}

pub fn evaluate_body_fixed_pose(
    body: &BodyState,
    pose: BodyFixedPose,
) -> (TranslationalState, AttitudeState) {
    body_fixed_to_inertial(
        body,
        BodyFixedFrameState {
            translation_body: TranslationalState {
                position: pose.position_body_m,
                velocity: DVec3::ZERO,
            },
            orientation_body: pose.orientation_body,
            angular_velocity_body: DVec3::ZERO,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::Epoch;

    fn test_body(orientation: DQuat) -> BodyState {
        BodyState {
            id: 0,
            epoch: Epoch(0.0),
            position: DVec3::new(100.0, 20.0, -50.0),
            velocity: DVec3::new(5.0, 0.0, -2.0),
            orientation,
            angular_velocity: DVec3::Y * 0.25,
            mass_kg: 1.0e20,
            gm: 1.0,
            radius_m: 1000.0,
        }
    }

    #[test]
    fn inertial_body_fixed_round_trip_includes_surface_rotation() {
        let body = test_body(DQuat::from_rotation_z(0.3));
        let frame = BodyFixedFrameState {
            translation_body: TranslationalState {
                position: DVec3::new(1000.0, 12.0, -40.0),
                velocity: DVec3::new(1.0, 2.0, 3.0),
            },
            orientation_body: DQuat::from_rotation_x(0.2),
            angular_velocity_body: DVec3::new(0.01, 0.02, 0.03),
        };

        let (translation, attitude) = body_fixed_to_inertial(&body, frame);
        let round_trip = inertial_to_body_fixed(&body, translation, attitude);

        assert!(
            (round_trip.translation_body.position - frame.translation_body.position).length()
                < 1e-9
        );
        assert!(
            (round_trip.translation_body.velocity - frame.translation_body.velocity).length()
                < 1e-9
        );
        assert!(
            round_trip
                .orientation_body
                .angle_between(frame.orientation_body)
                < 1e-9
        );
        assert!((round_trip.angular_velocity_body - frame.angular_velocity_body).length() < 1e-9);
    }

    #[test]
    fn body_fixed_pose_follows_body_rotation() {
        let pose = BodyFixedPose {
            position_body_m: DVec3::X * 1000.0,
            orientation_body: DQuat::IDENTITY,
        };
        let body_a = test_body(DQuat::IDENTITY);
        let body_b = test_body(DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2));

        let (a, _) = evaluate_body_fixed_pose(&body_a, pose);
        let (b, _) = evaluate_body_fixed_pose(&body_b, pose);

        assert!((a.position - (body_a.position + DVec3::X * 1000.0)).length() < 1e-9);
        assert!((b.position - (body_b.position - DVec3::Z * 1000.0)).length() < 1e-9);
        assert!(
            (a.velocity - body_fixed_surface_velocity(&body_a, pose.position_body_m)).length()
                < 1e-9
        );
        assert!(
            (b.velocity - body_fixed_surface_velocity(&body_b, pose.position_body_m)).length()
                < 1e-9
        );
    }
}
