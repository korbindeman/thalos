//! Shared application preferences: persistence, window behavior, MSAA, and UI.
//!
//! Applications insert [`AppPreferences`] before constructing `WindowPlugin`,
//! then add [`PreferencesPlugin`]. The interactive plugin owns the one common
//! settings modal. Application-specific pages append sections through
//! [`register_settings_page`] and [`SettingsPageBuild`] without becoming
//! dependencies of this crate.

mod graphics;
mod menu;
mod persistence;
mod render_scale;
mod window;

pub use graphics::{
    AntiAliasingFallback, FRAME_CAP_CHOICES, FRAME_CAP_OFF, GraphicsPreferences, MsaaSetting,
    PreferencesCamera, QualityOverrides, QualityPreset, RENDER_SCALE_MAX, RENDER_SCALE_MIN,
    effective_graphics,
};
pub use menu::{
    SettingsMenu, SettingsMenuPlugin, SettingsMenuSet, SettingsPage, SettingsPageBuild,
    register_settings_page,
};
pub use persistence::{
    AppPreferences, PreferencesPlugin, first_run_defaults_for, load, preferences_path, save,
};
pub use render_scale::{RenderScaleSet, RenderScaleState, scaled_physical_size};
pub use thalos_ui::UiBackdropSource;
pub use window::{
    RESOLUTION_PRESETS, UI_SCALE_MAX, UI_SCALE_MIN, WindowModeSetting, WindowSettings,
    WindowSettingsOverrides, initial_window, overrides_from_env,
};
