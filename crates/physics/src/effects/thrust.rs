use glam::DVec3;

use super::{Effect, EffectContext};
use crate::maneuver::delta_v_to_world;
use crate::types::BodyId;

/// A maneuver thrust whose direction tracks the ship's instantaneous orbital
/// frame around `reference_body`.
///
/// The prograde/normal/radial frame is recomputed from the ship's live state
/// on every integrator substep — the only honest interpretation of a
/// local-frame Δv when the burn duration is a meaningful fraction of the
/// orbital period. Active over the half-open interval `[start_time, end_time)`.
pub struct ManeuverThrustEffect {
    /// Δv in the local (prograde, normal, radial) frame, m/s. Only the
    /// direction matters; magnitude sets the world-frame thrust direction.
    pub delta_v_local: DVec3,
    /// Body whose state defines the local orbital frame.
    pub reference_body: BodyId,
    /// Acceleration magnitude, m/s².
    pub acceleration: f64,
    pub start_time: f64,
    pub end_time: f64,
}

impl ManeuverThrustEffect {
    pub fn new(
        delta_v_local: DVec3,
        reference_body: BodyId,
        acceleration: f64,
        start_time: f64,
        duration: f64,
    ) -> Self {
        Self {
            delta_v_local,
            reference_body,
            acceleration,
            start_time,
            end_time: start_time + duration,
        }
    }

    #[inline]
    pub fn is_active(&self, time: f64) -> bool {
        time >= self.start_time && time < self.end_time
    }
}

impl Effect for ManeuverThrustEffect {
    #[inline]
    fn accelerate(&self, ctx: &EffectContext) -> DVec3 {
        if !self.is_active(ctx.time) {
            return DVec3::ZERO;
        }

        let (ref_pos, ref_vel) = if self.reference_body < ctx.bodies.len() {
            let b = &ctx.bodies[self.reference_body];
            (b.position, b.velocity)
        } else {
            (DVec3::ZERO, DVec3::ZERO)
        };

        let dv_world =
            delta_v_to_world(self.delta_v_local, ctx.velocity, ctx.position, ref_pos, ref_vel);
        if dv_world.length_squared() <= 0.0 {
            return DVec3::ZERO;
        }
        dv_world.normalize() * self.acceleration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_window() {
        let thrust = ManeuverThrustEffect::new(DVec3::X, 0, 10.0, 100.0, 50.0);

        assert!(!thrust.is_active(99.9));
        assert!(thrust.is_active(100.0));
        assert!(thrust.is_active(149.9));
        assert!(!thrust.is_active(150.0));
    }
}
