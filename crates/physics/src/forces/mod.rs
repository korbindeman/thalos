pub mod gravity;
pub mod thrust;

pub use gravity::GravityForce;
pub use thrust::ThrustForce;

use glam::DVec3;

use crate::types::BodyStates;

/// A force contribution expressed as an acceleration (m/s²).
///
/// Implementors are expected to be `Send + Sync` so the registry can be
/// shared across threads (e.g. passed into a rayon parallel integrator).
pub trait ForceFunction: Send + Sync {
    /// Return the acceleration contribution at the given state and time.
    fn compute(
        &self,
        position: DVec3,
        velocity: DVec3,
        time: f64,
        body_states: &BodyStates,
    ) -> DVec3;

    /// Whether this force is active at `time`. Defaults to always active.
    fn is_active(&self, _time: f64) -> bool {
        true
    }
}

/// Holds all registered forces and sums their contributions.
pub struct ForceRegistry {
    forces: Vec<Box<dyn ForceFunction>>,
}

impl ForceRegistry {
    pub fn new() -> Self {
        Self { forces: Vec::new() }
    }

    /// Register a new force.
    pub fn add(&mut self, force: Box<dyn ForceFunction>) {
        self.forces.push(force);
    }

    /// Sum accelerations from all active forces at the given state and time.
    pub fn compute_acceleration(
        &self,
        position: DVec3,
        velocity: DVec3,
        time: f64,
        body_states: &BodyStates,
    ) -> DVec3 {
        self.forces
            .iter()
            .filter(|f| f.is_active(time))
            .map(|f| f.compute(position, velocity, time, body_states))
            .fold(DVec3::ZERO, |acc, a| acc + a)
    }
}

impl Default for ForceRegistry {
    fn default() -> Self {
        Self::new()
    }
}
