//! Window / display settings.
//!
//! [`WindowSettings`] is the user preference set — window mode, windowed
//! resolution, vsync, fullscreen monitor, UI scale. It is persisted (alongside
//! graphics + units) by [`crate::settings`] as the `window` section of the
//! unified `settings.ron`; this module owns only the runtime behaviour, not the
//! file IO. The `THALOS_WINDOW_MODE` / `THALOS_WINDOW_SIZE` / `THALOS_VSYNC`
//! env vars become *session overrides* ([`WindowSettingsOverrides`], from
//! [`overrides_from_env`]): they win for the running session and disable their
//! control in the settings UI, but are never written back. (`THALOS_SCALE`
//! stays a pure env knob — it pins the window scale-factor override, a
//! diagnostic lever distinct from the user-facing UI scale here.)
//!
//! Writers of `WindowSettings`: the settings menu's Window tab (user edits, in
//! `crate::settings_menu`) and [`apply_window_settings`] (which writes back
//! OS-side window resizes in windowed mode so a drag-resized size sticks).
//! `apply_window_settings` pushes the effective settings onto the primary
//! `Window` each frame — value-compared, so untouched frames mark nothing
//! changed for `bevy_winit`. Persistence is the unified autosave's job
//! (`crate::settings::AppSettingsPlugin`).

use bevy::prelude::*;
use bevy::window::{
    Monitor, MonitorSelection, PresentMode, PrimaryWindow, VideoModeSelection, WindowMode,
    WindowResolution,
};
use serde::{Deserialize, Serialize};

pub(crate) const UI_SCALE_MIN: f32 = 0.5;
pub(crate) const UI_SCALE_MAX: f32 = 2.0;

pub(crate) const RESOLUTION_PRESETS: &[(u32, u32)] = &[
    (1280, 720),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (3440, 1440),
    (3840, 2160),
];

// ── Resources ─────────────────────────────────────────────────────────────────

/// User window/display preferences. Persisted as the `window` section of
/// [`crate::settings`]'s unified file.
///
/// Writers: the settings menu's Window tab and `apply_window_settings`'s
/// windowed drag-resize write-back. Everything else reads.
#[derive(Resource, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowSettings {
    pub mode: WindowModeSetting,
    /// Windowed-mode client size in physical pixels. Fullscreen modes use the
    /// monitor's size / current video mode instead.
    pub resolution: (u32, u32),
    pub vsync: bool,
    /// Fullscreen target monitor, matched against [`Monitor::name`];
    /// `None` = primary. Falls back to primary when the named monitor is
    /// absent (e.g. unplugged), without forgetting the preference.
    pub monitor: Option<String>,
    /// User UI-scale multiplier. Multiplies into the fractional-HiDPI
    /// compensation in `apply_ui_scale`; note that non-integer *effective*
    /// scales (window scale × UI scale) can soften Bevy UI text until the
    /// upstream fractional-scale text bug is fixed.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WindowModeSetting {
    Windowed,
    #[default]
    Borderless,
    Exclusive,
}

/// Session-only env-var overrides (`THALOS_WINDOW_MODE` / `THALOS_WINDOW_SIZE`
/// / `THALOS_VSYNC`). A `Some` field wins over the persisted setting for this
/// run and disables the corresponding settings-UI control; it is never saved.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct WindowSettingsOverrides {
    pub mode: Option<WindowModeSetting>,
    pub resolution: Option<(u32, u32)>,
    pub vsync: Option<bool>,
}

impl WindowSettings {
    /// Clamp loaded values into supported ranges (UI scale, minimum window
    /// size). Applied by [`crate::settings::load`] after reading the file.
    pub fn sanitized(mut self) -> Self {
        self.ui_scale = self.ui_scale.clamp(UI_SCALE_MIN, UI_SCALE_MAX);
        self.resolution.0 = self.resolution.0.max(320);
        self.resolution.1 = self.resolution.1.max(240);
        self
    }
}

// ── Env-var session overrides ───────────────────────────────────────────────────

/// Compute the env-var session overrides (`THALOS_WINDOW_MODE` /
/// `THALOS_WINDOW_SIZE` / `THALOS_VSYNC`). Called from `main()` before the app
/// is built so [`initial_window`] can honour them; never persisted.
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
    let width = width.trim().parse().ok()?;
    let height = height.trim().parse().ok()?;
    Some((width, height))
}

// ── Initial window ────────────────────────────────────────────────────────────

/// Primary-window descriptor for `WindowPlugin`, from persisted settings +
/// env overrides. The persisted monitor preference can't be resolved here —
/// [`Monitor`] entities don't exist before the app boots — so fullscreen
/// starts on the primary monitor and [`apply_window_settings`] re-targets on
/// the first frame.
pub fn initial_window(settings: &WindowSettings, overrides: &WindowSettingsOverrides) -> Window {
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
    // Dev/diagnostic: force a window scale factor (overrides the OS HiDPI
    // scale). `THALOS_SCALE=1.0` etc. Used to isolate fractional-scale text
    // rendering bugs.
    if let Ok(scale) = std::env::var("THALOS_SCALE")
        && let Ok(value) = scale.trim().parse::<f32>()
        && value > 0.0
    {
        resolution = resolution.with_scale_factor_override(value);
    }

    let present_mode = if overrides.vsync.unwrap_or(settings.vsync) {
        PresentMode::AutoVsync
    } else {
        // Uncapped framerate so frame-time deltas are observable when
        // profiling; wgpu falls back to a supported non-vsync present mode.
        PresentMode::AutoNoVsync
    };

    Window {
        title: "Thalos".into(),
        mode,
        resolution,
        present_mode,
        ..default()
    }
}

// ── Plugin / systems ──────────────────────────────────────────────────────────

pub struct WindowSettingsPlugin;

impl Plugin for WindowSettingsPlugin {
    fn build(&self, app: &mut App) {
        // `WindowSettings` + `WindowSettingsOverrides` are inserted in
        // `main()` (they shape the initial window before the app exists).
        app.add_systems(Update, (apply_window_settings, apply_ui_scale));
    }
}

/// A windowed-size push to winit that hasn't been observed back on the
/// `Window` component yet. While in flight, the drag-resize write-back is
/// suppressed so a half-applied transition can't overwrite the stored size;
/// expires after a grace period in case the OS clamps the request.
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
        // Logged because mode/present/resolution pushes recreate the swapchain,
        // which the flaky Windows surface-acquire path can turn into a crash —
        // these lines tie any such crash to the triggering write.
        info!("window mode {:?} → {:?}", window.mode, desired_mode);
        window.mode = desired_mode;
        // Mode switches resize the window asynchronously; hold off the
        // windowed write-back until the dust settles.
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

    // Windowed size: push UI edits to the window, pull OS drag-resizes back
    // into the settings so they persist. With an env-pinned size the window
    // is left alone entirely — the pin sized it at creation, and free
    // drag-resizing shouldn't fight a push or leak into the file.
    if mode == WindowModeSetting::Windowed && overrides.resolution.is_none() {
        let current = (
            window.resolution.physical_width(),
            window.resolution.physical_height(),
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
                    .set_physical_resolution(settings.resolution.0, settings.resolution.1);
                *in_flight = Some(InFlightResolution::new(settings.resolution));
            }
        } else if in_flight.is_none() && current != settings.resolution && current != (0, 0) {
            // The user resized the OS window — remember the dragged size. The
            // unified autosave (`crate::settings`) persists the change.
            settings.resolution = current;
        }
    }
}

/// Apply the user's [`WindowSettings::ui_scale`] preference. Bevy UI
/// rasterises at `window scale × UiScale`, so the OS HiDPI scale already
/// carries the display's own sizing — this only layers the preference on top.
///
/// **History.** Through Bevy 0.18 this system also *snapped* the effective
/// scale to the nearest integer, working around a cosmic-text bug that
/// rasterised glyphs at inconsistent sizes on fractional scale factors (text
/// looked non-uniform, "not monospace"). The snap cost real estate: on a 150 %
/// display it rounded 1.5 up to 2.0, inflating the whole UI by a third, which
/// is why the HUD swallowed a 4K screen. Bevy 0.19 replaced cosmic-text with
/// parley and the bug is gone — verified by rendering the UI kitchen sink at
/// `THALOS_UI_SCALE=1.5`, whose glyphs are as clean as the 1.0 capture — so
/// the snap is deleted and a 150 % display now gets a true 1.5.
///
/// Note for anyone tempted to reintroduce compensation: snap the *UI* scale,
/// never the window scale-factor override. `bevy_winit::changed_windows`
/// treats a scale-factor change as logical-size-preserving and physically
/// resizes the window — an earlier attempt grew the borderless-fullscreen
/// window to 4/3 of the monitor.
fn apply_ui_scale(settings: Res<WindowSettings>, mut ui_scale: ResMut<UiScale>) {
    let target = settings.ui_scale.clamp(UI_SCALE_MIN, UI_SCALE_MAX);
    if (ui_scale.0 - target).abs() > 1.0e-4 {
        ui_scale.0 = target;
    }
}

// ── Settings-menu helpers ───────────────────────────────────────────────────

/// A selectable fullscreen monitor, prepared by the settings menu from the
/// `Monitor` entity query (unnamed monitors are skipped — they can't be
/// persisted). Used to populate the Window tab's monitor picker.
pub struct MonitorChoice {
    /// [`Monitor::name`], the persisted key.
    pub name: String,
    /// Display label: name, size, primary marker.
    pub label: String,
}
