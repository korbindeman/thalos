// Bevy system signatures routinely exceed clippy's argument and type
// complexity budgets; same crate-level allowance as thalos_runtime.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

//! # `thalos_structures` — terrain-anchored structure geometry
//!
//! Peeled out of `thalos_runtime` (Phase 5b, ADR-20260731T024003Z). This crate
//! owns *what a structure looks like*: the runway frame and its paving,
//! markings, designators, and posts; the taxiway/apron connection network and
//! the paved footprints scatter clears against.
//!
//! It deliberately does **not** own *where a structure goes or what happens
//! there*. Deferred placement, the Avian collider, the per-frame big_space
//! anchoring, spaceport orchestration, and the base-editor state machine stay
//! in `thalos_runtime`, because they read canonical craft state and mutate the
//! world. The structure vocabulary itself (`StructureKind`, `StructureSite`,
//! `StructureRegistry`) lives a layer down in `thalos_game_state::structures`,
//! so both halves name the same thing.

pub mod connection_geometry;
pub mod runway_geometry;
