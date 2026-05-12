//! Shared semantic input layer for Thalos Bevy binaries.
//!
//! This crate owns the `bevy_enhanced_input` action/context definitions and
//! the editable RON binding file. Runtime systems in each binary should read
//! the intent resources exported here instead of querying raw keyboard or
//! mouse button state.

pub mod game;
pub mod gating;
pub mod planet_editor;
pub mod settings;
pub mod shipyard;

pub use bevy_enhanced_input::prelude as enhanced;
