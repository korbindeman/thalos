// Bevy system signatures routinely exceed clippy's argument and type
// complexity budgets; same crate-level allowance as thalos_runtime.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

//! # `thalos_map` — the map/planning surface
//!
//! Peeled out of `thalos_runtime` (Phase 5b, ADR-20260731T024003Z): the
//! map-view snapshot boundary (sole writer of
//! `thalos_game_state::map::MapSnapshot`), orbit trails, the flight-plan
//! ghost/preview display, maneuver-node interaction, and the body tree
//! panel. Everything here reads the blackboard and canonical snapshots;
//! real-space entities are never touched (the map invariant).

pub mod body_tree_panel;
pub mod flight_plan_view;
pub mod maneuver;
pub mod map_view;
pub mod trails;
