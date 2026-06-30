//! Persisted measurement-unit preference (metric vs imperial).
//!
//! [`UnitsSettings`] is the file-backed unit-system preference, stored as RON at
//! [`SETTINGS_PATH`] (gitignored). It is loaded once when [`UnitsSettingsPlugin`]
//! builds and saved whenever its value actually changes ([`save_units_settings`]
//! value-compares against the last write, so an open settings tab doesn't churn
//! the file).
//!
//! SI is always the internal/simulation unit; this preference only affects how
//! the HUD *displays* distances, speeds, and masses. The settings menu's Units
//! tab is the sole writer; the HUD formatters in [`crate::hud::format`] read it
//! and dispatch on [`UnitSystem`].

use std::path::Path;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Where unit settings persist, relative to the working directory the game
/// already loads `assets/` from. Gitignored; recreated with defaults if missing
/// or unparseable.
pub const SETTINGS_PATH: &str = "user/units.ron";

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

// ── Load / save ─────────────────────────────────────────────────────────────────

/// Read persisted unit settings (defaults on first run or parse failure).
fn load() -> UnitsSettings {
    match std::fs::read_to_string(SETTINGS_PATH) {
        Ok(source) => ron::from_str::<UnitsSettings>(&source).unwrap_or_else(|err| {
            warn!("Failed to parse {SETTINGS_PATH}: {err}; using unit-settings defaults.");
            UnitsSettings::default()
        }),
        Err(_) => UnitsSettings::default(), // first run
    }
}

fn save(settings: &UnitsSettings) {
    let path = Path::new(SETTINGS_PATH);
    let result = (|| -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let body = ron::ser::to_string_pretty(settings, ron::ser::PrettyConfig::default())
            .map_err(std::io::Error::other)?;
        std::fs::write(path, body)
    })();
    if let Err(err) = result {
        warn!("Failed to save unit settings to {SETTINGS_PATH}: {err}");
    }
}

// ── Plugin / systems ────────────────────────────────────────────────────────────

pub struct UnitsSettingsPlugin;

impl Plugin for UnitsSettingsPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<UnitsSettings>()
            .insert_resource(load())
            .add_systems(Update, save_units_settings);
    }
}

/// Persist the settings whenever their value differs from the last write.
/// Value-compared (not change-detected) so the Units tab — which dereferences
/// the `ResMut` every frame it renders — doesn't rewrite the file each frame.
fn save_units_settings(settings: Res<UnitsSettings>, mut last_saved: Local<Option<UnitsSettings>>) {
    if last_saved.as_ref() != Some(&*settings) {
        save(&settings);
        *last_saved = Some(*settings);
    }
}
