//! `thalos_control` — the unified fly-by-wire control layer.
//!
//! Every ship-control source (pilot stick, SAS/stability hold, nav-mode
//! pointing, the maneuver autopilot) speaks one language: it emits a
//! [`ControlDemand`] tagged with its [`DemandSource`]. Each frame the
//! [`arbitrate`] step picks the winning attitude and throttle demands by
//! priority, the [`AttitudeController`] turns the resolved attitude demand
//! into a normalized body-frame torque, and [`allocate`] distributes that
//! torque across every effector the craft has (reaction wheels + aero
//! control surfaces). No source ever drives an effector directly.
//!
//! Pure Rust, no Bevy. The game crate owns the resources and systems that
//! collect demands and apply the realized commands; this crate owns the
//! policy (arbitration, control law, allocation) as testable functions.
//!
//! Scope today: attitude + throttle for ships, over three attitude effectors —
//! reaction wheels, aero control surfaces, and **engine gimbal** (thrust
//! vectoring, folded into the controller's `effector_authority` alongside the
//! aero surfaces; see `docs/aerodynamics.md` *Thrust vectoring*). Warp
//! arbitration, EVA, and RCS are designed-in extension points — new
//! `DemandSource`s and effectors slot into the same arbitrate → control →
//! allocate pipeline — but are not yet wired. See `docs/control.md`.

pub mod allocator;
pub mod arbiter;
pub mod attitude;
pub mod demand;
pub mod flight;

pub use allocator::{Allocation, allocate};
pub use arbiter::{Arbitration, arbitrate};
pub use attitude::{AttitudeController, NOSE_BODY, SETTLE_TIME_S, point_nose};
pub use demand::{AttitudeDemand, ControlDemand, DemandSource};
pub use flight::{AssistStatus, FlightState, PlaneHoldTarget, pitch_command_envelope};
