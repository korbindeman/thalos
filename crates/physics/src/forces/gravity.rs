use glam::DVec3;

use crate::types::*;

/// Per-body gravity breakdown: total acceleration plus the dominant body and
/// perturbation ratio.  Computed in a single O(n) pass.
pub struct GravityResult {
    pub acceleration: DVec3,
    /// Index into `BodyStates` of the strongest gravitational source.
    pub dominant_body: BodyId,
    /// Ratio of the second-largest to the largest gravitational acceleration
    /// magnitude.  0.0 if there is only one body.
    pub perturbation_ratio: f64,
}

/// Computes the gravitational acceleration on the ship from all bodies in
/// `BodyStates`.
///
/// For each body the contribution is:
///   a = -G * m * (ship_pos - body_pos) / |ship_pos - body_pos|³
pub struct GravityForce;

impl GravityForce {
    /// Compute gravitational acceleration *and* dominant-body analysis in a
    /// single pass over `body_states`.
    pub fn compute_with_analysis(position: DVec3, body_states: &BodyStates) -> GravityResult {
        let mut acceleration = DVec3::ZERO;
        let mut best_id: BodyId = 0;
        let mut best_mag: f64 = 0.0;
        let mut second_mag: f64 = 0.0;

        for (id, body) in body_states.iter().enumerate() {
            let r_vec = position - body.position;
            let dist_sq = r_vec.length_squared();

            if dist_sq < MIN_DISTANCE_SQ {
                continue;
            }

            let dist = dist_sq.sqrt();
            let mag = G * body.mass_kg / dist_sq;
            acceleration -= mag * r_vec / dist;

            if mag > best_mag {
                second_mag = best_mag;
                best_mag = mag;
                best_id = id;
            } else if mag > second_mag {
                second_mag = mag;
            }
        }

        let perturbation_ratio = if best_mag > 0.0 {
            second_mag / best_mag
        } else {
            0.0
        };

        GravityResult {
            acceleration,
            dominant_body: best_id,
            perturbation_ratio,
        }
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

        let result =
            GravityForce::compute_with_analysis(DVec3::new(r, 0.0, 0.0), &body_states);

        let expected = G * body_mass / (r * r);
        let actual = result.acceleration.length();
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

        let result =
            GravityForce::compute_with_analysis(DVec3::new(1.0e9, 0.0, 0.0), &body_states);
        let accel = result.acceleration;

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

        let result =
            GravityForce::compute_with_analysis(DVec3::new(50.0, 0.0, 0.0), &body_states);
        assert_eq!(result.acceleration, DVec3::ZERO);
    }
}
