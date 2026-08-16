//! Shared window/display preferences and their live Bevy projection.

use bevy::prelude::*;
use bevy::window::{
    Monitor, MonitorSelection, PresentMode, PrimaryWindow, VideoModeSelection, WindowMode,
    WindowResolution,
};
use serde::{Deserialize, Serialize};

pub const UI_SCALE_MIN: f32 = 0.5;
pub const UI_SCALE_MAX: f32 = 2.0;

pub const RESOLUTION_PRESETS: &[(u32, u32)] = &[
    (1280, 720),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (3440, 1440),
    (3840, 2160),
];

#[derive(Resource, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowSettings {
    pub mode: WindowModeSetting,
    pub resolution: (u32, u32),
    pub vsync: bool,
    pub monitor: Option<String>,
    pub ui_scale: f32,
}

impl Default for WindowSettings {
    fn default() -> Self {
        Self {
            mode: WindowModeSetting::Borderless,
            resolution: (1600, 900),
            vsync: true,
            monitor: None,
            ui_scale: 1.0,
        }
    }
}

impl WindowSettings {
    pub fn sanitized(mut self) -> Self {
        self.ui_scale = if self.ui_scale.is_finite() {
            self.ui_scale.clamp(UI_SCALE_MIN, UI_SCALE_MAX)
        } else {
            1.0
        };
        self.resolution.0 = self.resolution.0.max(320);
        self.resolution.1 = self.resolution.1.max(240);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WindowModeSetting {
    Windowed,
    #[default]
    Borderless,
    Exclusive,
}

/// Session-only environment overrides. They shape the live window but are
/// never written back to preferences.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct WindowSettingsOverrides {
    pub mode: Option<WindowModeSetting>,
    pub resolution: Option<(u32, u32)>,
    pub vsync: Option<bool>,
}

pub fn overrides_from_env() -> WindowSettingsOverrides {
    let mode = std::env::var("THALOS_WINDOW_MODE").ok().map(|value| {
        match value.trim().to_ascii_lowercase().as_str() {
            "windowed" | "window" => WindowModeSetting::Windowed,
            "exclusive" | "fullscreen" | "true-fullscreen" | "true_fullscreen" => {
                WindowModeSetting::Exclusive
            }
            "borderless" | "borderless-fullscreen" | "borderless_fullscreen" | "" => {
                WindowModeSetting::Borderless
            }
            other => {
                eprintln!(
                    "Unknown THALOS_WINDOW_MODE={other:?}; using borderless fullscreen. \
                     Expected windowed, borderless, or fullscreen."
                );
                WindowModeSetting::Borderless
            }
        }
    });
    let resolution = std::env::var("THALOS_WINDOW_SIZE")
        .ok()
        .and_then(|value| parse_window_size(&value));
    let vsync = std::env::var("THALOS_VSYNC").ok().map(|value| {
        !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "off" | "0" | "false" | "no"
        )
    });
    WindowSettingsOverrides {
        mode,
        resolution,
        vsync,
    }
}

fn parse_window_size(value: &str) -> Option<(u32, u32)> {
    let (width, height) = value
        .trim()
        .split_once(['x', 'X', ','])
        .or_else(|| value.trim().split_once(' '))?;
    Some((width.trim().parse().ok()?, height.trim().parse().ok()?))
}

/// Build the initial primary window before Bevy creates monitor entities.
pub fn initial_window(
    title: impl Into<String>,
    settings: &WindowSettings,
    overrides: &WindowSettingsOverrides,
) -> Window {
    let mode = match overrides.mode.unwrap_or(settings.mode) {
        WindowModeSetting::Windowed => WindowMode::Windowed,
        WindowModeSetting::Borderless => {
            WindowMode::BorderlessFullscreen(MonitorSelection::Primary)
        }
        WindowModeSetting::Exclusive => {
            WindowMode::Fullscreen(MonitorSelection::Primary, VideoModeSelection::Current)
        }
    };

    let (width, height) = overrides.resolution.unwrap_or(settings.resolution);
    let mut resolution = WindowResolution::new(width, height);
    if let Ok(scale) = std::env::var("THALOS_SCALE")
        && let Ok(value) = scale.trim().parse::<f32>()
        && value > 0.0
    {
        resolution = resolution.with_scale_factor_override(value);
    }

    let present_mode = if overrides.vsync.unwrap_or(settings.vsync) {
        PresentMode::AutoVsync
    } else {
        PresentMode::AutoNoVsync
    };

    Window {
        title: title.into(),
        mode,
        resolution,
        present_mode,
        ..default()
    }
}

pub(crate) struct WindowSettingsPlugin;

impl Plugin for WindowSettingsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (apply_window_settings, apply_ui_scale));
    }
}

struct InFlightResolution {
    target: (u32, u32),
    frames_left: u8,
}

impl InFlightResolution {
    fn new(target: (u32, u32)) -> Self {
        Self {
            target,
            frames_left: 30,
        }
    }
}

fn apply_window_settings(
    mut settings: ResMut<WindowSettings>,
    overrides: Res<WindowSettingsOverrides>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    monitors: Query<(Entity, &Monitor)>,
    mut in_flight: Local<Option<InFlightResolution>>,
) {
    let Ok(mut window) = windows.single_mut() else {
        return;
    };

    let monitor_selection = settings
        .monitor
        .as_deref()
        .and_then(|wanted| {
            monitors
                .iter()
                .find(|(_, monitor)| monitor.name.as_deref() == Some(wanted))
                .map(|(entity, _)| MonitorSelection::Entity(entity))
        })
        .unwrap_or(MonitorSelection::Primary);

    let mode = overrides.mode.unwrap_or(settings.mode);
    let desired_mode = match mode {
        WindowModeSetting::Windowed => WindowMode::Windowed,
        WindowModeSetting::Borderless => WindowMode::BorderlessFullscreen(monitor_selection),
        WindowModeSetting::Exclusive => {
            WindowMode::Fullscreen(monitor_selection, VideoModeSelection::Current)
        }
    };
    if window.mode != desired_mode {
        info!("window mode {:?} → {:?}", window.mode, desired_mode);
        window.mode = desired_mode;
        *in_flight = Some(InFlightResolution::new(
            overrides.resolution.unwrap_or(settings.resolution),
        ));
    }

    let desired_present = if overrides.vsync.unwrap_or(settings.vsync) {
        PresentMode::AutoVsync
    } else {
        PresentMode::AutoNoVsync
    };
    if window.present_mode != desired_present {
        info!(
            "window present mode {:?} → {:?}",
            window.present_mode, desired_present
        );
        window.present_mode = desired_present;
    }

    if mode == WindowModeSetting::Windowed && overrides.resolution.is_none() {
        let current = (
            window.resolution.width().round() as u32,
            window.resolution.height().round() as u32,
        );
        if let Some(push) = in_flight.as_mut() {
            push.frames_left = push.frames_left.saturating_sub(1);
            if current == push.target || push.frames_left == 0 {
                *in_flight = None;
            }
        }
        if settings.is_changed() {
            if current != settings.resolution && current != (0, 0) {
                info!(
                    "window resolution {}×{} → {}×{}",
                    current.0, current.1, settings.resolution.0, settings.resolution.1
                );
                window
                    .resolution
                    .set(settings.resolution.0 as f32, settings.resolution.1 as f32);
                *in_flight = Some(InFlightResolution::new(settings.resolution));
            }
        } else if in_flight.is_none() && current != settings.resolution && current != (0, 0) {
            settings.resolution = current;
        }
    }
}

fn apply_ui_scale(settings: Res<WindowSettings>, mut ui_scale: ResMut<UiScale>) {
    let target = settings.ui_scale.clamp(UI_SCALE_MIN, UI_SCALE_MAX);
    if (ui_scale.0 - target).abs() > 1.0e-4 {
        ui_scale.0 = target;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitization_handles_invalid_persisted_values() {
        let settings = WindowSettings {
            resolution: (1, 2),
            ui_scale: f32::NAN,
            ..default()
        }
        .sanitized();
        assert_eq!(settings.resolution, (320, 240));
        assert_eq!(settings.ui_scale, 1.0);
    }

    #[test]
    fn window_size_parser_accepts_supported_separators() {
        assert_eq!(parse_window_size("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_window_size("1600, 900"), Some((1600, 900)));
        assert_eq!(parse_window_size("bad"), None);
    }
}
