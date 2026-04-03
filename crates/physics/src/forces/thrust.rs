use glam::DVec3;

use crate::types::BodyStates;

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
