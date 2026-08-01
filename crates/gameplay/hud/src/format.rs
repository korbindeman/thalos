//! HUD value formatters — moved to `thalos_game_state::units::format`
//! (Phase 5b, ADR-20260731T024003Z) so feature crates (shipyard editor, HUD)
//! share one set without depending on each other. Re-exported for path
//! stability.

pub use thalos_game_state::units::format::*;
