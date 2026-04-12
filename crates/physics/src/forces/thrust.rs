use glam::DVec3;

use crate::maneuver::delta_v_to_world;
use crate::types::{BodyId, BodyStates};

use super::ForceFunction;

/// A constant-direction, constant-magnitude thrust expressed as an
/// acceleration (m/s²).
///
/// Active during the half-open interval `[start_time, end_time)`.
pub struct ThrustForce {
    /// Unit vector in the direction of thrust.
    pub direction: DVec3,
    /// Acceleration magnitude in m/s².
    pub acceleration: f64,
    /// Simulation time (s) when thrust begins.
    pub start_time: f64,
    /// Simulation time (s) when thrust ends.
    pub end_time: f64,
}

impl ThrustForce {
    /// Create a thrust force from a direction, magnitude, start time, and
    /// duration.
    ///
    /// `direction` is normalised internally so the caller does not need to
    /// pre-normalise it (a zero vector will produce zero thrust).
    pub fn new(direction: DVec3, acceleration: f64, start_time: f64, duration: f64) -> Self {
        let normalised = if direction.length_squared() > 0.0 {
            direction.normalize()
        } else {
            DVec3::ZERO
        };

        Self {
            direction: normalised,
            acceleration,
            start_time,
            end_time: start_time + duration,
        }
    }
}

impl ForceFunction for ThrustForce {
    fn is_active(&self, time: f64) -> bool {
        time >= self.start_time && time < self.end_time
    }

    fn compute(
        &self,
        _position: DVec3,
        _velocity: DVec3,
        _time: f64,
        _body_states: &BodyStates,
    ) -> DVec3 {
        self.direction * self.acceleration
    }
}

/// A maneuver thrust whose direction tracks the ship's instantaneous orbital
/// frame around `reference_body`.
///
/// Unlike [`ThrustForce`], which freezes its world-space direction at burn
/// start, this force recomputes the prograde/normal/radial frame from the
/// ship's live state on every integrator substep. That is the only honest
/// interpretation of a local-frame Δv when the burn duration is a meaningful
/// fraction of the orbital period — otherwise the "prograde" component points
/// somewhere else entirely after a quarter orbit and the predicted path bends
/// into the central body.
pub struct ManeuverThrustForce {
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

impl ManeuverThrustForce {
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
}

impl ForceFunction for ManeuverThrustForce {
    fn is_active(&self, time: f64) -> bool {
        time >= self.start_time && time < self.end_time
    }

    fn compute(
        &self,
        position: DVec3,
        velocity: DVec3,
        _time: f64,
        body_states: &BodyStates,
    ) -> DVec3 {
        let (ref_pos, ref_vel) = if self.reference_body < body_states.len() {
            let b = &body_states[self.reference_body];
            (b.position, b.velocity)
        } else {
            (DVec3::ZERO, DVec3::ZERO)
        };

        let dv_world = delta_v_to_world(self.delta_v_local, velocity, position, ref_pos, ref_vel);
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
        let thrust = ThrustForce::new(DVec3::X, 10.0, 100.0, 50.0);

        assert!(!thrust.is_active(99.9));
        assert!(thrust.is_active(100.0));
        assert!(thrust.is_active(149.9));
        assert!(!thrust.is_active(150.0));
    }

    #[test]
    fn test_compute_returns_direction_times_magnitude() {
        let thrust = ThrustForce::new(DVec3::new(1.0, 0.0, 0.0), 5.0, 0.0, 100.0);
        let accel = thrust.compute(DVec3::ZERO, DVec3::ZERO, 50.0, &vec![]);

        assert!((accel.x - 5.0).abs() < 1e-15);
        assert_eq!(accel.y, 0.0);
        assert_eq!(accel.z, 0.0);
    }

    #[test]
    fn test_direction_is_normalised() {
        let thrust = ThrustForce::new(DVec3::new(3.0, 4.0, 0.0), 2.0, 0.0, 10.0);
        let accel = thrust.compute(DVec3::ZERO, DVec3::ZERO, 5.0, &vec![]);

        // magnitude should equal the acceleration parameter, not the input vector length
        let mag = accel.length();
        assert!((mag - 2.0).abs() < 1e-14, "magnitude was {mag}");
    }
}
