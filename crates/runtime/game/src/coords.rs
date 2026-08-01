//! Shared coordinate-system helpers — moved wholesale to
//! `thalos_game_state::coords` (Phase 5a, ADR-20260731T024003Z); re-exported
//! here so every existing `crate::coords::*` path keeps working.

pub use thalos_game_state::coords::*;
