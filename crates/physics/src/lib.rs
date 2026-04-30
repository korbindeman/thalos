//! Orbital-mechanics simulation: pure Rust, no Bevy.
//!
//! The crate sits behind two trait abstractions that draw the boundaries
//! between subsystems and let alternate implementations slot in without
//! touching call sites:
//!
//! - [`body_state_provider::BodyStateProvider`] — anything that can answer
//!   "where is body `i` at time `t`?". Today's only impl is
//!   [`patched_conics::PatchedConics`] (analytic Kepler chains); a baked
//!   ephemeris could replace it.
//! - [`ship_propagator::ShipPropagator`] — anything that can advance the
//!   ship's state across one segment of coast or burn. Today's only impl is
//!   [`ship_propagator::KeplerianPropagator`] (analytical Kepler coast +
//!   RK4 burn under a single SOI body).
//!
//! [`simulation::Simulation`] wires them together for live stepping, and
//! [`trajectory::propagate_flight_plan`] uses the same `ShipPropagator` to
//! build the predicted [`trajectory::FlightPlan`] — so "where the ship is"
//! and "where it will be" can never numerically diverge.

pub mod body_state_provider;
pub mod debug_orbits;
pub mod gravity_mode;
pub mod maneuver;
pub mod orbital_math;
pub mod parsing;
pub mod patched_conics;
pub mod ship_propagator;
pub mod simulation;
pub mod trajectory;
pub mod types;
