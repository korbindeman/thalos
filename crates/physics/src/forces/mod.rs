pub mod gravity;
pub mod thrust;

pub use gravity::{GravityForce, GravityResult};
pub use thrust::ManeuverThrustForce;

use glam::DVec3;

use crate::types::BodyStates;

/// Container for all non-gravity forces active on the ship.
///
/// Gravity is always-on and is computed directly (not stored here) because it
/// is needed by the mode-switching logic and comes packaged with body-level
/// analysis. Thrusts, by contrast, are scheduled time-windowed burns.
#[derive(Default)]
pub struct Forces {
    /// Currently scheduled maneuver burns. Each is active over
    /// `[start_time, end_time)`; callers check `is_active` per substep.
    pub thrusts: Vec<ManeuverThrustForce>,
}

impl Forces {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sum contributions from all active thrusts at the given state/time.
    #[inline]
    pub fn sum_thrusts(
        &self,
        position: DVec3,
        velocity: DVec3,
        time: f64,
        body_states: &BodyStates,
    ) -> DVec3 {
        let mut accel = DVec3::ZERO;
        for thrust in &self.thrusts {
            if thrust.is_active(time) {
                accel += thrust.compute(position, velocity, body_states);
            }
        }
        accel
    }
}

/// Compute the total acceleration on the ship at `(position, velocity, time)`
/// given pre-queried `body_states` and the thrust set.
///
/// Returns `(total_acceleration, gravity_result)` so callers that need the
/// perturbation analysis (e.g. the integrator's mode switch) can avoid a
/// redundant gravity pass.
#[inline]
pub fn compute_total_acceleration(
    position: DVec3,
    velocity: DVec3,
    time: f64,
    body_states: &BodyStates,
    forces: &Forces,
) -> (DVec3, GravityResult) {
    let gravity = GravityForce::compute_with_analysis(position, body_states);
    let thrust = forces.sum_thrusts(position, velocity, time, body_states);
    (gravity.acceleration + thrust, gravity)
}
