use glam::DVec3;

use crate::types::{BodyId, BodyStates, G, MIN_DISTANCE_SQ};

/// Capacity of the `body_weights` top-K list carried on each sample / result.
/// Four entries cover the typical hierarchy (star + planet + moon + next
/// strongest perturber) plus a tie-breaking slot. Stored on-stack; no heap
/// allocation per sample.
pub const BODY_WEIGHTS_CAP: usize = 4;

/// Per-body gravity breakdown computed in a single O(n) pass.
pub struct GravityResult {
    pub acceleration: DVec3,
    /// Index into `BodyStates` of the strongest gravitational source.
    pub dominant_body: BodyId,
    /// Ratio of the second-largest to the largest gravitational acceleration
    /// magnitude. 0.0 if there is only one body.
    pub perturbation_ratio: f64,
    /// Top-K bodies by acceleration magnitude, with weights `wᵢ = aᵢ / Σⱼ aⱼ`
    /// renormalised across the stored entries so they sum to 1.0. Unused
    /// slots have weight 0.0. Consumed by the renderer's gravity-weighted
    /// barycenter rule (see `docs/orbital_mechanics.md` §7.2).
    pub body_weights: [(BodyId, f32); BODY_WEIGHTS_CAP],
}

/// Empty `body_weights` sentinel: all slots zero.
pub const EMPTY_BODY_WEIGHTS: [(BodyId, f32); BODY_WEIGHTS_CAP] = [(0, 0.0); BODY_WEIGHTS_CAP];

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
        // Top-K (id, magnitude) kept in descending order; one sorted-insert per
        // body. K is small so the linear insertion cost is negligible.
        let mut top: [(BodyId, f64); BODY_WEIGHTS_CAP] = [(0, 0.0); BODY_WEIGHTS_CAP];

        for (id, body) in bodies.iter().enumerate() {
            let r_vec = position - body.position;
            let dist_sq = r_vec.length_squared();

            if dist_sq < MIN_DISTANCE_SQ {
                continue;
            }

            let dist = dist_sq.sqrt();
            let mag = G * body.mass_kg / dist_sq;
            acceleration -= mag * r_vec / dist;

            if mag > top[BODY_WEIGHTS_CAP - 1].1 {
                let mut i = BODY_WEIGHTS_CAP - 1;
                top[i] = (id, mag);
                while i > 0 && top[i].1 > top[i - 1].1 {
                    top.swap(i, i - 1);
                    i -= 1;
                }
            }
        }

        let best_mag = top[0].1;
        let second_mag = top[1].1;
        let perturbation_ratio = if best_mag > 0.0 {
            second_mag / best_mag
        } else {
            0.0
        };

        // Normalise kept weights so they sum to 1.0. Truncating to top-K loses
        // a tiny tail of the full Σⱼ aⱼ sum (<1% in realistic configurations);
        // renormalising across what we keep makes `Σ wᵢ · rᵢ` an exact
        // barycenter of the stored bodies, which is what the renderer needs.
        let kept_sum: f64 = top.iter().map(|(_, m)| *m).sum();
        let mut body_weights = EMPTY_BODY_WEIGHTS;
        if kept_sum > 0.0 {
            for (slot, (id, mag)) in body_weights.iter_mut().zip(top.iter()) {
                *slot = (*id, (mag / kept_sum) as f32);
            }
        }

        GravityResult {
            acceleration,
            dominant_body: top[0].0,
            perturbation_ratio,
            body_weights,
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
