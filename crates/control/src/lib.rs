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
//! Scope today: attitude + throttle for ships. Warp arbitration, EVA, and
//! RCS/gimbal effectors are designed-in extension points — new
//! `DemandSource`s and effectors slot into the same arbitrate → control →
//! allocate pipeline — but are not yet wired. See `docs/control.md`.

pub mod allocator;
pub mod arbiter;
pub mod attitude;
pub mod demand;

pub use allocator::{Allocation, allocate};
pub use arbiter::{Arbitration, arbitrate};
pub use attitude::{AttitudeController, NOSE_BODY, SETTLE_TIME_S, point_nose};
pub use demand::{AttitudeDemand, ControlDemand, DemandSource};
