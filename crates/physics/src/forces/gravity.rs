use glam::DVec3;

use crate::types::*;

use super::ForceFunction;

/// Computes the gravitational acceleration on the ship from all bodies in
/// `BodyStates`.
///
/// For each body the contribution is:
///   a = -G * m * (ship_pos - body_pos) / |ship_pos - body_pos|³
pub struct GravityForce;

impl ForceFunction for GravityForce {
    fn compute(
        &self,
        position: DVec3,
        _velocity: DVec3,
        _time: f64,
        body_states: &BodyStates,
    ) -> DVec3 {
        let mut acceleration = DVec3::ZERO;

        for body in body_states {
            let r_vec = position - body.position; // points from body to ship
            let dist_sq = r_vec.length_squared();

            if dist_sq < MIN_DISTANCE_SQ {
                continue;
            }

            // a = -G * m / r² * r̂  =  -G * m * r_vec / r³
            let dist = dist_sq.sqrt();
            acceleration -= G * body.mass_kg * r_vec / (dist_sq * dist);
        }

        acceleration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single body at origin, ship at (r, 0, 0).
    /// Expected acceleration magnitude: G * M / r².
    #[test]
    fn test_single_body_magnitude() {
        let body_mass = 5.972e24; // Earth-like
        let r = 6_371_000.0 + 400_000.0; // 400 km altitude

        let body_states = vec![BodyState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            mass_kg: body_mass,
        }];

        let force = GravityForce;
        let accel = force.compute(DVec3::new(r, 0.0, 0.0), DVec3::ZERO, 0.0, &body_states);

        let expected = G * body_mass / (r * r);
        let actual = accel.length();
        let rel_error = (actual - expected).abs() / expected;

        assert!(rel_error < 1e-10, "Magnitude error: {rel_error}");
    }

    /// Acceleration must point from ship toward the body (i.e. negative X when
    /// ship is on the +X axis).
    #[test]
    fn test_direction_toward_body() {
        let body_states = vec![BodyState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            mass_kg: 1.0e24,
        }];

        let force = GravityForce;
        let accel = force.compute(DVec3::new(1.0e9, 0.0, 0.0), DVec3::ZERO, 0.0, &body_states);

        assert!(accel.x < 0.0, "Should point toward body");
        assert!(accel.y.abs() < 1e-30);
        assert!(accel.z.abs() < 1e-30);
    }

    /// Bodies closer than MIN_DISTANCE_M are skipped without panicking.
    #[test]
    fn test_singularity_guard() {
        let body_states = vec![BodyState {
            position: DVec3::new(50.0, 0.0, 0.0), // 50 m away — below threshold
            velocity: DVec3::ZERO,
            mass_kg: 1.0e30,
        }];

        let force = GravityForce;
        let accel = force.compute(DVec3::new(50.0, 0.0, 0.0), DVec3::ZERO, 0.0, &body_states);
        assert_eq!(accel, DVec3::ZERO);
    }
}
