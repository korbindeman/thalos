//! Measurement-unit preference — see `thalos_game_state::units` for the
//! resource, the per-domain resolution model, and its tests (moved there in
//! Phase 5a, ADR-20260731T024003Z). This module keeps only the runtime
//! registration; the resource itself is inserted in `main()` from the unified
//! `settings.ron` and persisted by [`crate::settings`].

use bevy::prelude::*;

pub use thalos_game_state::units::{AviationUnits, UnitSystem, UnitsSettings};

pub struct UnitsSettingsPlugin;

impl Plugin for UnitsSettingsPlugin {
    fn build(&self, app: &mut App) {
        // The resource is inserted in `main()` from the unified `settings.ron`
        // and persisted by `crate::settings::AppSettingsPlugin`; this plugin
        // only registers the type for the reflection / debug-UI path.
        app.register_type::<UnitsSettings>();
    }
}
