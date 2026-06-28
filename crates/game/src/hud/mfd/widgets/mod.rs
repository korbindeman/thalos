//! MFD widgets. Each module exposes `build` (spawn its hidden root + children
//! under the slot's widget area), `relevance` (pure priority from
//! [`super::FlightContext`]), and — for widgets with live content — an
//! `update` system gated on being the active widget.

pub mod docking;
pub mod interplanetary;
pub mod nav_display;
pub mod trajectory;
