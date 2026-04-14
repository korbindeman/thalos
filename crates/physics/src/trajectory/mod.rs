//! Trajectory prediction framework.
//!
//! Layers:
//!
//! - [`Trajectory`] trait — a queryable path: `state_at(t)`, `epoch_range`,
//!   `anchor_body_at`. Anything implementing this can be sampled at arbitrary
//!   time, not just at stored points.
//! - [`NumericSegment`] — one discrete-sample trajectory leg produced by the
//!   integrator. Implements `Trajectory` via cubic Hermite interpolation
//!   between samples (position + velocity → C¹ continuous).
//! - [`FlightPlan`] — ordered `NumericSegment`s joined by maneuver nodes, plus
//!   detected `Encounter`s. Implements `Trajectory` by dispatching to the
//!   segment containing the query time. This is the primary object the game
//!   crate consumes.
//! - [`events`] — encounter detection (SOI entry/exit, surface impact,
//!   periapsis/apoapsis) and closest-approach search over a `Trajectory`.
//! - [`propagation`] — the integrator-driven propagator that builds
//!   `NumericSegment`s from a ship state + maneuver sequence.

use crate::types::{BodyId, StateVector};

pub mod events;
mod flight_plan;
mod numeric;
mod propagation;

#[cfg(test)]
mod tests;

pub use events::{Encounter, EncounterKind, TrajectoryEvent, closest_approach};
pub use flight_plan::{FlightPlan, ScheduledBurn, propagate_flight_plan};
pub use numeric::{NumericSegment, cone_width};
pub use propagation::{PredictionConfig, PropagationBudget};

// Back-compat aliases so existing call sites and tests keep building.
pub use flight_plan::FlightPlan as TrajectoryPrediction;
pub use numeric::NumericSegment as TrajectorySegment;
pub use flight_plan::propagate_flight_plan as propagate_trajectory_budgeted;

/// A queryable path through space: states can be sampled at any time within
/// `epoch_range`, not only at pre-stored points.
pub trait Trajectory: Send + Sync {
    /// Interpolated state at `time`, if `time` lies within `epoch_range`.
    fn state_at(&self, time: f64) -> Option<StateVector>;

    /// Half-open time interval `[start, end]` over which the trajectory is
    /// defined.  Callers should clamp to this before querying.
    fn epoch_range(&self) -> (f64, f64);

    /// Best known anchor body at `time`.  Numerical segments return the
    /// anchor body of the nearest stored sample.
    fn anchor_body_at(&self, time: f64) -> Option<BodyId>;
}

/// Back-compat: simplified unbudgeted propagation entry point used by tests.
#[allow(clippy::too_many_arguments)]
pub fn propagate_trajectory(
    initial_state: StateVector,
    start_time: f64,
    maneuvers: &crate::maneuver::ManeuverSequence,
    ephemeris: &dyn crate::body_state_provider::BodyStateProvider,
    bodies: &[crate::types::BodyDefinition],
    config: &PredictionConfig,
    integrator_config: crate::integrator::IntegratorConfig,
    ship_thrust_acceleration: f64,
) -> FlightPlan {
    propagate_flight_plan(
        initial_state,
        start_time,
        maneuvers,
        Vec::new(),
        ephemeris,
        bodies,
        config,
        integrator_config,
        ship_thrust_acceleration,
        None,
    )
}
