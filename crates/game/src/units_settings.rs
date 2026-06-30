//! Measurement-unit preference (metric vs imperial).
//!
//! [`UnitsSettings`] is the unit-system preference. It is persisted (alongside
//! window + graphics) by [`crate::settings`] as the `units` section of the
//! unified `settings.ron`; this module owns only the resource + its `Reflect`
//! registration, not the file IO.
//!
//! SI is always the internal/simulation unit; this preference only affects how
//! the HUD *displays* distances, speeds, and masses. The settings menu's Units
//! tab is the sole writer; the HUD formatters in [`crate::hud::format`] read it
//! and dispatch on [`UnitSystem`].

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

// ── Unit system ─────────────────────────────────────────────────────────────────

/// Which measurement system the HUD formats values in.
///
/// `Imperial` is the aviation-flavoured set: feet for altitude, knots for speed,
/// feet-per-minute for vertical speed, nautical miles for long distances, and
/// pounds for mass.
#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitSystem {
    Metric,
    Imperial,
}

impl UnitSystem {
    pub const ALL: [UnitSystem; 2] = [UnitSystem::Metric, UnitSystem::Imperial];

    pub fn label(self) -> &'static str {
        match self {
            UnitSystem::Metric => "Metric (m, km, m/s)",
            UnitSystem::Imperial => "Imperial (ft, kn)",
        }
    }

    pub fn is_imperial(self) -> bool {
        matches!(self, UnitSystem::Imperial)
    }
}

// ── Resource ───────────────────────────────────────────────────────────────────

/// User measurement-unit preference, persisted to [`SETTINGS_PATH`].
///
/// Writer: the settings menu's Units tab. The HUD formatters read.
/// `Reflect`-registered (for a future in-game debug UI).
#[derive(Resource, Reflect, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[reflect(Resource)]
#[serde(default)]
pub struct UnitsSettings {
    /// Measurement system the HUD formats displayed values in.
    pub system: UnitSystem,
}

impl Default for UnitsSettings {
    fn default() -> Self {
        Self {
            system: UnitSystem::Metric,
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────────

pub struct UnitsSettingsPlugin;

impl Plugin for UnitsSettingsPlugin {
    fn build(&self, app: &mut App) {
        // The resource is inserted in `main()` from the unified `settings.ron`
        // and persisted by `crate::settings::AppSettingsPlugin`; this plugin
        // only registers the type for the reflection / debug-UI path.
        app.register_type::<UnitsSettings>();
    }
}
