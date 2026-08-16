//! Persistence and composition for preferences shared by every application.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::graphics::{
    GraphicsPreferenceCapabilities, GraphicsPreferences, MsaaSetting, QualityOverrides, apply_msaa,
    apply_render_scale, cap_frame_rate,
};
use crate::menu::{SettingsMenu, SettingsMenuPlugin, SettingsPage, register_settings_page};
use crate::window::{WindowSettings, WindowSettingsOverrides, WindowSettingsPlugin};

const DEV_PREFERENCES_PATH: &str = "user/preferences.ron";

/// Preferences whose meaning is identical in the game and lightweight apps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AppPreferences {
    pub window: WindowSettings,
    pub graphics: GraphicsPreferences,
}

impl AppPreferences {
    /// Stamp a named quality bundle onto the in-memory graphics knobs.
    /// Window mode and size stay with the Window page.
    pub fn apply_named_preset(&mut self, preset: crate::graphics::QualityPreset) {
        self.graphics.apply_preset(preset);
    }
}

/// Project-local in debug builds; OS-standard application preferences in release.
pub fn preferences_path() -> PathBuf {
    if cfg!(debug_assertions) {
        PathBuf::from(DEV_PREFERENCES_PATH)
    } else {
        bevy::platform::dirs::preferences_dir()
            .map(|dir| dir.join("thalos").join("preferences.ron"))
            .unwrap_or_else(|| PathBuf::from(DEV_PREFERENCES_PATH))
    }
}

/// Load shared preferences before constructing the primary window.
pub fn load() -> AppPreferences {
    let path = preferences_path();
    match load_from(&path) {
        Ok(Some(preferences)) => preferences,
        Ok(None) => {
            if path.with_file_name("settings.ron").is_file() {
                migrate_legacy(&path)
            } else {
                first_run_defaults()
            }
        }
        Err(err) => {
            eprintln!(
                "Failed to parse {}: {err}; using defaults/legacy settings.",
                path.display()
            );
            migrate_legacy(&path)
        }
    }
}

/// First-run defaults. macOS starts on the Laptop developer profile so the
/// machine is usable before anyone opens settings. The window stays the
/// default borderless fullscreen. Existing files are never rewritten by
/// this path.
pub fn first_run_defaults() -> AppPreferences {
    first_run_defaults_for(cfg!(target_os = "macos"))
}

pub fn first_run_defaults_for(macos: bool) -> AppPreferences {
    if macos {
        AppPreferences {
            window: WindowSettings::default(),
            graphics: GraphicsPreferences::laptop(),
        }
    } else {
        AppPreferences::default()
    }
}

fn load_from(path: &Path) -> Result<Option<AppPreferences>, ron::error::SpannedError> {
    let Ok(text) = fs::read_to_string(path) else {
        return Ok(None);
    };
    let mut preferences = ron::from_str::<AppPreferences>(&text)?;
    // Foliage moved from the game's settings.ron into this shared schema after
    // preferences.ron already existed in development builds. Preserve that
    // last game-side choice exactly once instead of silently resetting it.
    if !text.contains("foliage:")
        && let Some(foliage) = legacy_foliage(&path.with_file_name("settings.ron"))
    {
        preferences.graphics.foliage = foliage;
    }
    preferences.window = preferences.window.sanitized();
    preferences.graphics = preferences.graphics.sanitized();
    Ok(Some(preferences))
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct LegacySettings {
    window: WindowSettings,
    graphics: LegacyGraphicsSettings,
}

#[derive(Deserialize)]
#[serde(default)]
struct LegacyGraphicsSettings {
    msaa: MsaaSetting,
    foliage: bool,
}

impl Default for LegacyGraphicsSettings {
    fn default() -> Self {
        Self {
            msaa: MsaaSetting::default(),
            foliage: true,
        }
    }
}

fn migrate_legacy(preferences_path: &Path) -> AppPreferences {
    let Ok(text) = fs::read_to_string(preferences_path.with_file_name("settings.ron")) else {
        return AppPreferences::default();
    };

    migrate_legacy_text(&text).unwrap_or_default()
}

fn legacy_foliage(path: &Path) -> Option<bool> {
    let text = fs::read_to_string(path).ok()?;
    ron::from_str::<LegacySettings>(&text)
        .ok()
        .map(|settings| settings.graphics.foliage)
}

fn migrate_legacy_text(text: &str) -> Option<AppPreferences> {
    if let Ok(legacy) = ron::from_str::<LegacySettings>(text) {
        return Some(AppPreferences {
            window: legacy.window.sanitized(),
            graphics: GraphicsPreferences {
                msaa: legacy.graphics.msaa,
                foliage: legacy.graphics.foliage,
                ..GraphicsPreferences::showcase()
            }
            .sanitized(),
        });
    }

    // The oldest schema stored WindowSettings flat at the same path.
    if let Ok(window) = ron::from_str::<WindowSettings>(text) {
        return Some(AppPreferences {
            window: window.sanitized(),
            ..default()
        });
    }

    None
}

/// Atomically replace the shared preferences file with the current values.
pub fn save(preferences: &AppPreferences) {
    let path = preferences_path();
    if let Err(err) = save_to(&path, preferences) {
        warn!(
            target: "thalos::preferences",
            "Failed to save preferences to {}: {err}",
            path.display()
        );
    }
}

fn save_to(path: &Path, preferences: &AppPreferences) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let body = ron::ser::to_string_pretty(preferences, ron::ser::PrettyConfig::default())
        .map_err(io::Error::other)?;
    let temp_path = path.with_extension(format!("ron.{}.tmp", std::process::id()));
    let mut temp = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&temp_path)?;
    temp.write_all(body.as_bytes())?;
    temp.sync_all()?;
    drop(temp);

    if let Err(err) = fs::rename(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(err);
    }
    Ok(())
}

/// Installs shared preference resources and their projection into Bevy.
/// Interactive applications additionally receive the window driver, settings UI,
/// and autosave; headless applications only apply graphics preferences to cameras.
pub struct PreferencesPlugin {
    interactive: bool,
    foliage: bool,
}

impl PreferencesPlugin {
    pub const fn new(interactive: bool) -> Self {
        Self {
            interactive,
            foliage: false,
        }
    }

    pub const fn interactive() -> Self {
        Self::new(true)
    }

    pub const fn headless() -> Self {
        Self::new(false)
    }

    /// Expose and drive the shared foliage preference for applications whose
    /// composition includes a concrete foliage adapter.
    pub const fn with_foliage(mut self, enabled: bool) -> Self {
        self.foliage = enabled;
        self
    }
}

impl Plugin for PreferencesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WindowSettings>()
            .init_resource::<WindowSettingsOverrides>()
            .init_resource::<GraphicsPreferences>()
            .init_resource::<QualityOverrides>()
            .init_resource::<SettingsMenu>()
            .insert_resource(GraphicsPreferenceCapabilities {
                foliage: self.foliage,
            })
            .add_systems(Update, apply_msaa);

        if !self.interactive {
            return;
        }

        if !app.is_plugin_added::<thalos_ui::ThalosUiPlugin>() {
            app.add_plugins(thalos_ui::ThalosUiPlugin);
        }
        app.add_plugins((WindowSettingsPlugin, SettingsMenuPlugin));
        register_settings_page(
            app,
            SettingsPage {
                id: "window",
                label: "Window",
                order: 0,
            },
        );
        register_settings_page(
            app,
            SettingsPage {
                id: "graphics",
                label: "Graphics",
                order: 10,
            },
        );
        app.add_systems(Update, (apply_render_scale, cap_frame_rate, autosave));
    }
}

fn autosave(
    window: Res<WindowSettings>,
    graphics: Res<GraphicsPreferences>,
    overrides: Res<QualityOverrides>,
    mut last: Local<Option<AppPreferences>>,
) {
    if overrides.preset.is_some() {
        return;
    }
    let current = AppPreferences {
        window: window.clone(),
        graphics: *graphics,
    };
    if last.as_ref() != Some(&current) {
        save(&current);
        *last = Some(current);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::QualityPreset;

    #[test]
    fn round_trip_preserves_shared_sections() {
        let root = std::env::temp_dir().join(format!(
            "thalos-preferences-{}-{}",
            std::process::id(),
            line!()
        ));
        let path = root.join("preferences.ron");
        let expected = AppPreferences {
            window: WindowSettings {
                resolution: (1920, 1080),
                ui_scale: 1.25,
                ..default()
            },
            graphics: GraphicsPreferences {
                msaa: MsaaSetting::X4,
                foliage: false,
                ..GraphicsPreferences::showcase()
            }
            .sanitized(),
        };

        save_to(&path, &expected).expect("preferences should save");
        assert_eq!(
            load_from(&path).expect("preferences should parse"),
            Some(expected)
        );

        fs::remove_dir_all(root).expect("test preferences should be removable");
    }

    #[test]
    fn migration_extracts_shared_graphics_from_game_settings() {
        let legacy = r#"(
            window: (
                mode: Windowed,
                resolution: (1920, 1080),
                vsync: false,
                monitor: None,
                ui_scale: 1.25,
            ),
            graphics: (
                clouds: false,
                grass: false,
                foliage: false,
                msaa: X4,
            ),
            units: (system: Metric),
        )"#;
        let migrated = migrate_legacy_text(legacy).expect("legacy schema should migrate");

        assert_eq!(migrated.window.resolution, (1920, 1080));
        assert_eq!(migrated.window.ui_scale, 1.25);
        assert!(!migrated.window.vsync);
        assert_eq!(migrated.graphics.msaa, MsaaSetting::X4);
        assert!(!migrated.graphics.foliage);
        assert_eq!(migrated.graphics.preset, QualityPreset::Custom);
    }

    #[test]
    fn migration_defaults_missing_legacy_foliage_to_enabled() {
        let legacy = "(graphics: (msaa: Off))";
        let migrated = migrate_legacy_text(legacy).expect("legacy schema should migrate");

        assert!(migrated.graphics.foliage);
    }

    #[test]
    fn existing_preferences_import_the_old_foliage_choice_once() {
        let root = std::env::temp_dir().join(format!(
            "thalos-preferences-foliage-migration-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&root).unwrap();
        let preferences_path = root.join("preferences.ron");
        fs::write(
            &preferences_path,
            "(graphics: (msaa: X2), window: (resolution: (1600, 900)))",
        )
        .unwrap();
        fs::write(root.join("settings.ron"), "(graphics: (foliage: false))").unwrap();

        let loaded = load_from(&preferences_path)
            .expect("preferences should parse")
            .expect("preferences should exist");

        assert_eq!(loaded.graphics.msaa, MsaaSetting::X2);
        assert!(!loaded.graphics.foliage);
        assert_eq!(loaded.graphics.preset, QualityPreset::Custom);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn first_macos_run_defaults_to_laptop() {
        let prefs = first_run_defaults_for(true);
        assert_eq!(prefs.graphics, GraphicsPreferences::laptop());
        assert_eq!(prefs.window, WindowSettings::default());
        assert_eq!(
            prefs.window.mode,
            crate::window::WindowModeSetting::Borderless
        );
    }

    #[test]
    fn laptop_preset_leaves_the_window_alone() {
        let mut prefs = AppPreferences::default();
        assert_eq!(
            prefs.window.mode,
            crate::window::WindowModeSetting::Borderless
        );
        prefs.apply_named_preset(QualityPreset::Laptop);
        assert_eq!(prefs.graphics, GraphicsPreferences::laptop());
        assert_eq!(prefs.window, WindowSettings::default());
    }

    #[test]
    fn showcase_preset_leaves_the_window_alone() {
        let mut prefs = AppPreferences {
            window: WindowSettings {
                mode: crate::window::WindowModeSetting::Borderless,
                resolution: (1920, 1080),
                ..default()
            },
            graphics: GraphicsPreferences::laptop(),
        };
        prefs.apply_named_preset(QualityPreset::Showcase);
        assert_eq!(prefs.graphics, GraphicsPreferences::showcase());
        assert_eq!(
            prefs.window.mode,
            crate::window::WindowModeSetting::Borderless
        );
        assert_eq!(prefs.window.resolution, (1920, 1080));
    }

    #[test]
    fn first_non_macos_run_stays_on_showcase() {
        let prefs = first_run_defaults_for(false);
        assert_eq!(prefs, AppPreferences::default());
        assert_eq!(prefs.graphics.preset, QualityPreset::Showcase);
    }

    #[test]
    fn existing_preferences_keep_a_named_showcase_file() {
        let root = std::env::temp_dir().join(format!(
            "thalos-preferences-showcase-keep-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&root).unwrap();
        let preferences_path = root.join("preferences.ron");
        fs::write(
            &preferences_path,
            "(graphics: (msaa: Off, foliage: true), window: (resolution: (1600, 900)))",
        )
        .unwrap();

        let loaded = load_from(&preferences_path)
            .expect("preferences should parse")
            .expect("preferences should exist");

        assert_eq!(loaded.graphics.preset, QualityPreset::Showcase);
        assert_eq!(loaded.graphics.render_scale, 1.0);
        fs::remove_dir_all(root).unwrap();
    }
}
