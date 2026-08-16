//! Full-game settings — one `settings.ron` file for game-only sections.
//!
//! Application-wide window and anti-aliasing preferences live in
//! `thalos_preferences` so Kòrsou and the normal game share them. This module
//! persists only full-game rendering, units, and HUD workspace state. Foliage
//! is shared with Kòrsou and therefore persists with application preferences.
//!
//! **Storage location** (Bevy 0.19's [`bevy::platform::dirs::preferences_dir`]):
//! - **Debug builds** keep it project-local at `user/settings.ron` (gitignored)
//!   so it is trivial to inspect or reset during development.
//! - **Release builds** use the OS-standard application-data directory,
//!   `<preferences_dir>/thalos/settings.ron` (e.g. `%APPDATA%\thalos\` on
//!   Windows), falling back to the project-local path if the OS has no such dir.
//!
//! [`load`] runs before the app exists and migrates the legacy per-domain files
//! on first run. [`AppSettingsPlugin`] owns the single autosave.

use std::path::PathBuf;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::graphics_settings::GraphicsSettings;
use crate::units_settings::UnitsSettings;

/// Project-local settings path (the only location in debug; the fallback in
/// release). Relative to the working directory the game loads `assets/` from.
const DEV_SETTINGS_PATH: &str = "user/settings.ron";

/// The whole on-disk settings aggregate: one RON file with a section per domain.
/// `#[serde(default)]` everywhere means a missing or newly-added section loads
/// its defaults rather than failing the whole file.
#[derive(Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(default)]
pub struct AppSettings {
    pub graphics: GraphicsSettings,
    pub units: UnitsSettings,
    pub hud: thalos_hud::mfd::HudWorkspaceSettings,
}

/// Resolve the unified settings file path — project-local in debug, OS app-data
/// in release (see module docs).
pub fn settings_path() -> PathBuf {
    if cfg!(debug_assertions) {
        PathBuf::from(DEV_SETTINGS_PATH)
    } else {
        bevy::platform::dirs::preferences_dir()
            .map(|dir| dir.join("thalos").join("settings.ron"))
            .unwrap_or_else(|| PathBuf::from(DEV_SETTINGS_PATH))
    }
}

/// Read the unified settings (defaults on first run / parse failure), migrating
/// the legacy per-domain files when the unified file is absent. Called from
/// `main()` before the app is built, so any logging here is `eprintln`.
pub fn load() -> AppSettings {
    let path = settings_path();
    if let Ok(text) = std::fs::read_to_string(&path) {
        // The unified format names its sections (`graphics:` / `units:` /
        // `hud:`); a legacy window-only `settings.ron` (flat `(mode: …)`)
        // does not. Distinguish so the migration below isn't shadowed by a
        // lenient `AppSettings` parse that would silently drop the old window
        // prefs (the debug path collides with the legacy window file name).
        if is_unified(&text) {
            match ron::from_str::<AppSettings>(&text) {
                Ok(mut settings) => {
                    settings.graphics = settings.graphics.sanitized();
                    return settings;
                }
                Err(err) => {
                    eprintln!(
                        "Failed to parse {}: {err}; re-deriving from defaults/legacy.",
                        path.display()
                    );
                }
            }
        }
    }
    migrate_legacy()
}

/// Heuristic: does this file use the unified (sectioned) schema? The settings
/// structs have no string fields that could contain these section keys.
fn is_unified(text: &str) -> bool {
    text.contains("window:")
        || text.contains("graphics:")
        || text.contains("units:")
        || text.contains("hud:")
}

/// First-run migration: fold the legacy per-domain RON files into one
/// [`AppSettings`]. Best-effort — any file that is missing or unparseable just
/// leaves its section at the default. Shared preferences migrate their own
/// window and MSAA fields before this file is rewritten.
fn migrate_legacy() -> AppSettings {
    let mut settings = AppSettings::default();
    if let Ok(text) = std::fs::read_to_string("user/graphics.ron")
        && let Ok(graphics) = ron::from_str::<GraphicsSettings>(&text)
    {
        settings.graphics = graphics;
    }
    if let Ok(text) = std::fs::read_to_string("user/units.ron")
        && let Ok(units) = ron::from_str::<UnitsSettings>(&text)
    {
        settings.units = units;
    }
    settings
}

/// Write the unified settings file, creating the parent directory as needed.
pub fn save(settings: &AppSettings) {
    let path = settings_path();
    let result = (|| -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let body = ron::ser::to_string_pretty(settings, ron::ser::PrettyConfig::default())
            .map_err(std::io::Error::other)?;
        std::fs::write(&path, body)
    })();
    if let Err(err) = result {
        warn!(
            target: "thalos::settings",
            "Failed to save settings to {}: {err}",
            path.display()
        );
    }
}

/// Owns persistence for the three full-game resources inserted in `main()`.
pub struct AppSettingsPlugin;

impl Plugin for AppSettingsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, autosave_settings);
    }
}

/// Persist all three sections to the one file whenever any value changes.
/// Value-compared against the last write (the settings tabs deref their
/// `ResMut` every frame they render, so change-detection alone would churn the
/// file). The first frame seeds `last` from the live resources and writes once,
/// which also completes a legacy migration / creates the file on a fresh run.
fn autosave_settings(
    graphics: Res<GraphicsSettings>,
    units: Res<UnitsSettings>,
    hud: Res<thalos_hud::mfd::HudWorkspaceSettings>,
    quality_overrides: Res<thalos_preferences::QualityOverrides>,
    mut last: Local<Option<AppSettings>>,
) {
    if quality_overrides.preset.is_some() {
        return;
    }
    let current = AppSettings {
        graphics: graphics.clone(),
        units: *units,
        hud: hud.clone(),
    };
    if last.as_ref() != Some(&current) {
        save(&current);
        *last = Some(current);
    }
}
