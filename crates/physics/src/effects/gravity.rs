use glam::DVec3;

use crate::types::{BodyId, BodyStates, G, MIN_DISTANCE_SQ};

/// Per-body gravity breakdown: total acceleration plus the dominant body and
/// perturbation ratio. Computed in a single O(n) pass.
pub struct GravityResult {
    pub acceleration: DVec3,
    /// Index into `BodyStates` of the strongest gravitational source.
    pub dominant_body: BodyId,
    /// Ratio of the second-largest to the largest gravitational acceleration
    /// magnitude. 0.0 if there is only one body.
    pub perturbation_ratio: f64,
}

/// Swappable gravity implementation. Distinguished from generic effects
/// because [`GravityResult`] metadata is consumed by the integrator mode
/// switch and renderer (cone width, dominant-body tint, anchor frame).
pub trait GravityModel: Send + Sync {
    fn compute(&self, position: DVec3, bodies: &BodyStates) -> GravityResult;
}

/// Direct summation over all bodies in `BodyStates`. For each body:
///   a = -G * m * (ship_pos - body_pos) / |ship_pos - body_pos|³
pub struct NewtonianGravity;

impl GravityModel for NewtonianGravity {
    #[inline]
    fn compute(&self, position: DVec3, bodies: &BodyStates) -> GravityResult {
        let mut acceleration = DVec3::ZERO;
        let mut best_id: BodyId = 0;
        let mut best_mag: f64 = 0.0;
        let mut second_mag: f64 = 0.0;

        for (id, body) in bodies.iter().enumerate() {
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
    use crate::types::BodyState;

    #[test]
    fn test_single_body_magnitude() {
        let body_mass = 5.972e24;
        let r = 6_371_000.0 + 400_000.0;

        let bodies = vec![BodyState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            mass_kg: body_mass,
        }];

        let result = NewtonianGravity.compute(DVec3::new(r, 0.0, 0.0), &bodies);

        let expected = G * body_mass / (r * r);
        let actual = result.acceleration.length();
        let rel_error = (actual - expected).abs() / expected;

        assert!(rel_error < 1e-10, "Magnitude error: {rel_error}");
    }

    #[test]
    fn test_direction_toward_body() {
        let bodies = vec![BodyState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            mass_kg: 1.0e24,
        }];

        let result = NewtonianGravity.compute(DVec3::new(1.0e9, 0.0, 0.0), &bodies);
        let accel = result.acceleration;

        assert!(accel.x < 0.0, "Should point toward body");
        assert!(accel.y.abs() < 1e-30);
        assert!(accel.z.abs() < 1e-30);
    }

    #[test]
    fn test_singularity_guard() {
        let bodies = vec![BodyState {
            position: DVec3::new(50.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
            mass_kg: 1.0e30,
        }];

        let result = NewtonianGravity.compute(DVec3::new(50.0, 0.0, 0.0), &bodies);
        assert_eq!(result.acceleration, DVec3::ZERO);
    }
}
