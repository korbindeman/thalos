use std::time::{Duration, Instant};

use bevy::anti_alias::smaa::{Smaa, SmaaPreset};
use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use serde::{Deserialize, Serialize};

/// Named quality bundle. Showcase is the canonical look; Laptop is the
/// developer profile that trades fidelity for frame time and battery.
///
/// Selecting Showcase or Laptop stamps the shared knobs to that bundle.
/// Editing a stamped knob moves the selector to [`QualityPreset::Custom`].
#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QualityPreset {
    #[default]
    Showcase,
    Laptop,
    Custom,
}

impl QualityPreset {
    pub const SELECTABLE: [Self; 2] = [Self::Showcase, Self::Laptop];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Showcase => "Showcase",
            Self::Laptop => "Laptop",
            Self::Custom => "Custom",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "showcase" | "high" | "default" => Some(Self::Showcase),
            "laptop" | "mac" | "low" => Some(Self::Laptop),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }
}

/// Multisample anti-aliasing level for an application's main 3D view.
#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MsaaSetting {
    #[default]
    Off,
    X2,
    X4,
    X8,
}

impl MsaaSetting {
    pub const ALL: [Self; 4] = [Self::Off, Self::X2, Self::X4, Self::X8];

    pub const fn samples(self) -> u32 {
        match self {
            Self::Off => 1,
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
        }
    }

    pub const fn to_msaa(self) -> Msaa {
        match self {
            Self::Off => Msaa::Off,
            Self::X2 => Msaa::Sample2,
            Self::X4 => Msaa::Sample4,
            Self::X8 => Msaa::Sample8,
        }
    }

    pub const fn is_multisampled(self) -> bool {
        self.samples() > 1
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "Off (post-process AA)",
            Self::X2 => "MSAA 2×",
            Self::X4 => "MSAA 4×",
            Self::X8 => "MSAA 8×",
        }
    }
}

/// Frame-rate ceiling. Zero means uncapped (VSync may still floor the rate).
pub const FRAME_CAP_OFF: u32 = 0;
pub const FRAME_CAP_CHOICES: [u32; 3] = [FRAME_CAP_OFF, 30, 60];

pub const RENDER_SCALE_MIN: f32 = 0.25;
pub const RENDER_SCALE_MAX: f32 = 1.0;

/// Graphics preferences with the same concrete meaning in every application.
#[derive(Resource, Reflect, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[reflect(Resource)]
#[serde(default)]
pub struct GraphicsPreferences {
    pub preset: QualityPreset,
    pub msaa: MsaaSetting,
    /// Render the application's woody foliage layer. Applications that do not
    /// supply a foliage adapter keep this value persisted but omit its control.
    pub foliage: bool,
    /// Reserved 3D resolution fraction. 1.0 is native window pixels.
    /// This must not change the window scale factor: UI stays at OS HiDPI.
    pub render_scale: f32,
    /// Frames per second ceiling. `0` leaves the rate uncapped.
    pub frame_cap_hz: u32,
}

impl Default for GraphicsPreferences {
    fn default() -> Self {
        Self::showcase()
    }
}

impl GraphicsPreferences {
    pub fn showcase() -> Self {
        Self {
            preset: QualityPreset::Showcase,
            msaa: MsaaSetting::Off,
            foliage: true,
            render_scale: 1.0,
            frame_cap_hz: FRAME_CAP_OFF,
        }
    }

    pub fn laptop() -> Self {
        Self {
            preset: QualityPreset::Laptop,
            msaa: MsaaSetting::Off,
            foliage: false,
            // Keep OS HiDPI. A scale-factor override here made the HUD 1× on
            // Retina. Laptop cuts cost with parked layers, not the window.
            render_scale: 1.0,
            frame_cap_hz: 30,
        }
    }

    pub fn for_preset(preset: QualityPreset) -> Option<Self> {
        match preset {
            QualityPreset::Showcase => Some(Self::showcase()),
            QualityPreset::Laptop => Some(Self::laptop()),
            QualityPreset::Custom => None,
        }
    }

    pub fn apply_preset(&mut self, preset: QualityPreset) {
        if let Some(stamped) = Self::for_preset(preset) {
            *self = stamped;
        }
    }

    pub fn sanitized(mut self) -> Self {
        self.render_scale = if self.render_scale.is_finite() {
            self.render_scale.clamp(RENDER_SCALE_MIN, RENDER_SCALE_MAX)
        } else {
            1.0
        };
        if !FRAME_CAP_CHOICES.contains(&self.frame_cap_hz) {
            self.frame_cap_hz = FRAME_CAP_OFF;
        }
        self.reconcile_preset();
        self
    }

    /// If a named preset's knobs no longer match the bundle, the selector
    /// becomes Custom. Knobs are never rewritten here.
    pub fn reconcile_preset(&mut self) {
        if self.preset == QualityPreset::Custom {
            return;
        }
        if let Some(expected) = Self::for_preset(self.preset)
            && !self.shared_knobs_match(&expected)
        {
            self.preset = QualityPreset::Custom;
        }
    }

    pub fn mark_custom_if_knobs_changed(&mut self) {
        if self.preset == QualityPreset::Custom {
            return;
        }
        if let Some(expected) = Self::for_preset(self.preset)
            && !self.shared_knobs_match(&expected)
        {
            self.preset = QualityPreset::Custom;
        }
    }

    fn shared_knobs_match(&self, other: &Self) -> bool {
        self.msaa == other.msaa
            && self.foliage == other.foliage
            && (self.render_scale - other.render_scale).abs() < 1.0e-3
            && self.frame_cap_hz == other.frame_cap_hz
    }
}

/// Session-only quality pin from `THALOS_QUALITY`. Wins over the persisted
/// preset without being written back.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QualityOverrides {
    pub preset: Option<QualityPreset>,
}

impl QualityOverrides {
    pub fn from_env() -> Self {
        let Some(raw) = std::env::var("THALOS_QUALITY").ok() else {
            return Self::default();
        };
        match QualityPreset::parse(&raw) {
            Some(QualityPreset::Custom) => {
                eprintln!(
                    "THALOS_QUALITY=custom is not a session pin; expected showcase or laptop. Ignoring."
                );
                Self::default()
            }
            Some(preset) => {
                eprintln!(
                    "THALOS_QUALITY={} — quality preset pinned for this session",
                    preset.label().to_ascii_lowercase()
                );
                Self {
                    preset: Some(preset),
                }
            }
            None => {
                eprintln!("Unknown THALOS_QUALITY={raw:?}; expected showcase or laptop. Ignoring.");
                Self::default()
            }
        }
    }
}

/// Effective shared graphics after applying a session pin.
pub fn effective_graphics(
    preferences: &GraphicsPreferences,
    overrides: &QualityOverrides,
) -> GraphicsPreferences {
    match overrides.preset {
        Some(preset) => GraphicsPreferences::for_preset(preset).unwrap_or(*preferences),
        None => *preferences,
    }
}

/// Which shared graphics preferences have a concrete consumer in this app.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub(crate) struct GraphicsPreferenceCapabilities {
    pub foliage: bool,
}

/// Post-process anti-aliasing restored when hardware MSAA is disabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntiAliasingFallback {
    Smaa,
    Taa,
    None,
}

/// Marks a 3D camera controlled by [`GraphicsPreferences::msaa`].
#[derive(Component, Debug, Clone, Copy)]
pub struct PreferencesCamera {
    pub fallback: AntiAliasingFallback,
}

impl PreferencesCamera {
    pub const fn smaa() -> Self {
        Self {
            fallback: AntiAliasingFallback::Smaa,
        }
    }

    pub const fn taa() -> Self {
        Self {
            fallback: AntiAliasingFallback::Taa,
        }
    }

    pub const fn without_fallback() -> Self {
        Self {
            fallback: AntiAliasingFallback::None,
        }
    }
}

pub(crate) fn apply_msaa(
    settings: Res<GraphicsPreferences>,
    overrides: Res<QualityOverrides>,
    cameras: Query<(Entity, &PreferencesCamera)>,
    mut commands: Commands,
    mut applied: Local<Option<MsaaSetting>>,
) {
    let effective = effective_graphics(&settings, &overrides);
    if *applied == Some(effective.msaa) {
        return;
    }

    let mut touched_any = false;
    for (entity, camera) in &cameras {
        let mut entity = commands.entity(entity);
        entity.insert(effective.msaa.to_msaa());
        if effective.msaa.is_multisampled() {
            entity.remove::<Smaa>();
            entity.remove::<TemporalAntiAliasing>();
        } else {
            match camera.fallback {
                AntiAliasingFallback::Smaa => {
                    entity.insert(Smaa {
                        preset: SmaaPreset::High,
                    });
                    entity.remove::<TemporalAntiAliasing>();
                }
                AntiAliasingFallback::Taa => {
                    entity.insert(TemporalAntiAliasing::default());
                    entity.remove::<Smaa>();
                }
                AntiAliasingFallback::None => {
                    entity.remove::<Smaa>();
                    entity.remove::<TemporalAntiAliasing>();
                }
            }
        }
        touched_any = true;
    }

    if touched_any {
        *applied = Some(effective.msaa);
    }
}

/// Keep the window on the OS HiDPI scale so UI stays at normal density.
///
/// `render_scale` used to write `scale_factor_override`, which also scaled
/// Bevy UI. Laptop does not change the window. `THALOS_SCALE` still pins
/// one session.
pub(crate) fn apply_render_scale(mut windows: Query<&mut Window, With<PrimaryWindow>>) {
    if std::env::var("THALOS_SCALE").is_ok() {
        return;
    }
    let Ok(mut window) = windows.single_mut() else {
        return;
    };
    if window.resolution.scale_factor_override().is_some() {
        window.resolution.set_scale_factor_override(None);
    }
}

pub(crate) fn cap_frame_rate(
    settings: Res<GraphicsPreferences>,
    overrides: Res<QualityOverrides>,
    mut last: Local<Option<Instant>>,
) {
    let cap = effective_graphics(&settings, &overrides).frame_cap_hz;
    if cap == 0 {
        *last = Some(Instant::now());
        return;
    }
    let min = Duration::from_secs_f64(1.0 / f64::from(cap));
    if let Some(prev) = *last {
        let elapsed = prev.elapsed();
        if elapsed < min {
            std::thread::sleep(min.saturating_sub(elapsed));
        }
    }
    *last = Some(Instant::now());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_counts_match_bevy_levels() {
        assert_eq!(MsaaSetting::Off.samples(), 1);
        assert_eq!(MsaaSetting::X2.samples(), 2);
        assert_eq!(MsaaSetting::X4.samples(), 4);
        assert_eq!(MsaaSetting::X8.samples(), 8);
    }

    #[test]
    fn foliage_is_enabled_by_default() {
        assert!(GraphicsPreferences::default().foliage);
    }

    #[test]
    fn default_is_the_showcase_bundle() {
        let defaults = GraphicsPreferences::default();
        assert_eq!(defaults, GraphicsPreferences::showcase());
        assert_eq!(defaults.preset, QualityPreset::Showcase);
        assert_eq!(defaults.render_scale, 1.0);
        assert_eq!(defaults.frame_cap_hz, 0);
    }

    #[test]
    fn laptop_bundle_is_cheaper_than_showcase() {
        let laptop = GraphicsPreferences::laptop();
        assert_eq!(laptop.preset, QualityPreset::Laptop);
        assert!(!laptop.foliage);
        assert_eq!(laptop.render_scale, 1.0);
        assert_eq!(laptop.frame_cap_hz, 30);
    }

    #[test]
    fn editing_a_stamped_knob_becomes_custom() {
        let mut prefs = GraphicsPreferences::laptop();
        prefs.foliage = true;
        prefs.mark_custom_if_knobs_changed();
        assert_eq!(prefs.preset, QualityPreset::Custom);
        assert!(prefs.foliage);
    }

    #[test]
    fn stale_named_preset_reconciles_to_custom() {
        let mut prefs = GraphicsPreferences::showcase();
        prefs.foliage = false;
        prefs.reconcile_preset();
        assert_eq!(prefs.preset, QualityPreset::Custom);
    }

    #[test]
    fn quality_parser_accepts_aliases() {
        assert_eq!(QualityPreset::parse("laptop"), Some(QualityPreset::Laptop));
        assert_eq!(QualityPreset::parse("MAC"), Some(QualityPreset::Laptop));
        assert_eq!(
            QualityPreset::parse("showcase"),
            Some(QualityPreset::Showcase)
        );
        assert_eq!(QualityPreset::parse("nope"), None);
    }

    #[test]
    fn session_pin_replaces_persisted_knobs() {
        let persisted = GraphicsPreferences::showcase();
        let overrides = QualityOverrides {
            preset: Some(QualityPreset::Laptop),
        };
        let effective = effective_graphics(&persisted, &overrides);
        assert_eq!(effective, GraphicsPreferences::laptop());
        assert_eq!(persisted, GraphicsPreferences::showcase());
    }
}
