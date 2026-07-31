//! Same-body spherical ingress guidance for a runway destination.
//!
//! Terminal approaches deliberately use a runway-centred gnomonic plane. That
//! representation is exact and convenient locally but diverges at the horizon,
//! so LAND reaches the terminal region with this spherical leg first.

use glam::DVec3;

/// Inputs that are fixed for one destination ingress.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DestinationParams {
    pub body_radius_m: f64,
    pub gravity_m_s2: f64,
    pub bank_limit_rad: f64,
    /// Conservative altitude used away from the terminal region.
    pub cruise_altitude_m: f64,
    /// Altitude at the arrival fix where the terminal plan takes over.
    pub arrival_altitude_m: f64,
    /// Distance over which the enroute profile descends to arrival altitude.
    pub descent_distance_m: f64,
    pub cruise_speed_m_s: f64,
    pub max_vertical_speed_m_s: f64,
}

/// Per-frame state required by [`compute_destination_guidance`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DestinationInput {
    pub position_body_fixed: DVec3,
    pub track_dir_body_fixed: DVec3,
    pub ground_speed_m_s: f64,
    pub altitude_m: f64,
}

/// Guidance for the spherical ingress leg.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DestinationGuidance {
    /// Great-circle distance remaining to the terminal arrival fix.
    pub distance_to_arrival_m: f64,
    /// Tangent direction to fly, in the body-fixed frame.
    pub desired_track_body_fixed: DVec3,
    /// Right-positive bank command.
    pub bank_command_rad: f64,
    pub target_altitude_m: f64,
    pub vertical_speed_command_m_s: f64,
    pub target_speed_m_s: f64,
}

const TRACK_GAIN_PER_S: f64 = 0.35;
const ALTITUDE_GAIN_PER_S: f64 = 0.04;

/// Great-circle angle between two body-fixed directions.
pub fn angular_distance_rad(a: DVec3, b: DVec3) -> f64 {
    let Some(a) = a.try_normalize() else {
        return 0.0;
    };
    let Some(b) = b.try_normalize() else {
        return 0.0;
    };
    a.dot(b).clamp(-1.0, 1.0).acos()
}

/// Unit great-circle tangent at `from` pointing toward `to`.
///
/// The exactly-antipodal case has infinitely many shortest great circles. Pick
/// one deterministically from the body axes so guidance remains finite and
/// reproducible rather than flipping with floating-point noise.
pub fn great_circle_tangent(from: DVec3, to: DVec3) -> Option<DVec3> {
    let from = from.try_normalize()?;
    let to = to.try_normalize()?;
    let tangent = to - from * to.dot(from);
    if let Some(tangent) = tangent.try_normalize() {
        return Some(tangent);
    }
    if from.dot(to) > 0.0 {
        return None;
    }
    let reference = if from.dot(DVec3::Y).abs() < 0.9 {
        DVec3::Y
    } else {
        DVec3::X
    };
    (reference - from * reference.dot(from)).try_normalize()
}

pub fn compute_destination_guidance(
    arrival_dir_body_fixed: DVec3,
    params: &DestinationParams,
    input: &DestinationInput,
) -> Option<DestinationGuidance> {
    let up = input.position_body_fixed.try_normalize()?;
    let arrival = arrival_dir_body_fixed.try_normalize()?;
    let angle = angular_distance_rad(up, arrival);
    let distance_to_arrival_m = angle * params.body_radius_m.max(1.0);
    let desired = great_circle_tangent(up, arrival)?;

    let track = {
        let horizontal = input.track_dir_body_fixed - up * input.track_dir_body_fixed.dot(up);
        horizontal.try_normalize().unwrap_or(desired)
    };
    // Positive signed angle about local-up is a left turn in the route math
    // convention. Bank is pilot-right-positive, hence the sign flip.
    let heading_error_left = up
        .dot(track.cross(desired))
        .atan2(track.dot(desired).clamp(-1.0, 1.0));
    let bank_command_rad = if params.gravity_m_s2 > 0.0 {
        let omega = TRACK_GAIN_PER_S * heading_error_left;
        let tan_bank = omega * input.ground_speed_m_s.max(1.0) / params.gravity_m_s2;
        (-tan_bank.atan()).clamp(-params.bank_limit_rad.abs(), params.bank_limit_rad.abs())
    } else {
        0.0
    };

    let descent_distance = params.descent_distance_m.max(1.0);
    let blend = (distance_to_arrival_m / descent_distance).clamp(0.0, 1.0);
    let target_altitude_m =
        params.arrival_altitude_m + (params.cruise_altitude_m - params.arrival_altitude_m) * blend;
    let vertical_speed_command_m_s = ((target_altitude_m - input.altitude_m) * ALTITUDE_GAIN_PER_S)
        .clamp(
            -params.max_vertical_speed_m_s.abs(),
            params.max_vertical_speed_m_s.abs(),
        );

    Some(DestinationGuidance {
        distance_to_arrival_m,
        desired_track_body_fixed: desired,
        bank_command_rad,
        target_altitude_m,
        vertical_speed_command_m_s,
        target_speed_m_s: params.cruise_speed_m_s.max(1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn params() -> DestinationParams {
        DestinationParams {
            body_radius_m: 6_000_000.0,
            gravity_m_s2: 9.0,
            bank_limit_rad: 25.0_f64.to_radians(),
            cruise_altitude_m: 12_000.0,
            arrival_altitude_m: 1_200.0,
            descent_distance_m: 120_000.0,
            cruise_speed_m_s: 180.0,
            max_vertical_speed_m_s: 15.0,
        }
    }

    #[test]
    fn tangent_points_along_the_shortest_great_circle() {
        let tangent = great_circle_tangent(DVec3::Z, DVec3::X).expect("tangent");
        assert_abs_diff_eq!(tangent.x, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(tangent.dot(DVec3::Z), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn antipodal_and_polar_routes_are_finite_and_deterministic() {
        let a = great_circle_tangent(DVec3::Y, -DVec3::Y).expect("antipode");
        let b = great_circle_tangent(DVec3::Y, -DVec3::Y).expect("antipode");
        assert_eq!(a, b);
        assert!(a.is_finite());
        assert_abs_diff_eq!(a.dot(DVec3::Y), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn arrival_profile_descends_and_hits_the_capture_altitude() {
        let p = params();
        let far_angle = 240_000.0 / p.body_radius_m;
        let far = compute_destination_guidance(
            DVec3::new(far_angle.sin(), 0.0, far_angle.cos()),
            &p,
            &DestinationInput {
                position_body_fixed: DVec3::Z * (p.body_radius_m + 12_000.0),
                track_dir_body_fixed: DVec3::X,
                ground_speed_m_s: 180.0,
                altitude_m: 12_000.0,
            },
        )
        .expect("guidance");
        assert_abs_diff_eq!(far.target_altitude_m, p.cruise_altitude_m, epsilon = 1.0);

        let near_angle = 10_000.0 / p.body_radius_m;
        let near = compute_destination_guidance(
            DVec3::new(near_angle.sin(), 0.0, near_angle.cos()),
            &p,
            &DestinationInput {
                position_body_fixed: DVec3::Z * (p.body_radius_m + 4_000.0),
                track_dir_body_fixed: DVec3::X,
                ground_speed_m_s: 150.0,
                altitude_m: 4_000.0,
            },
        )
        .expect("guidance");
        assert!(near.target_altitude_m < far.target_altitude_m);
        assert!(near.vertical_speed_command_m_s < 0.0);
    }

    #[test]
    fn bank_sign_is_right_positive() {
        let p = params();
        let input = DestinationInput {
            position_body_fixed: DVec3::Z * p.body_radius_m,
            track_dir_body_fixed: DVec3::Y,
            ground_speed_m_s: 150.0,
            altitude_m: p.cruise_altitude_m,
        };
        // At +Z, east is +X. From a northbound (+Y) track, +X is a right turn.
        let g = compute_destination_guidance(DVec3::X, &p, &input).expect("guidance");
        assert!(g.bank_command_rad > 0.0);
    }
}
