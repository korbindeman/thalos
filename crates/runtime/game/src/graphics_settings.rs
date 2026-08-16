//! Graphics / rendering settings.
//!
//! [`GraphicsSettings`] is the graphics preference set. It is persisted
//! (alongside window + units) by [`crate::settings`] as the `graphics` section
//! of the unified `settings.ron`; this module owns only the resource + its
//! `Reflect` registration, not the file IO.
//!
//! The settings menu's Graphics tab is the interactive writer; the headless
//! capture runtime replaces the resource from a typed per-request profile.
//! Render systems only read.
//! Knobs: the volumetric-cloud toggle, consumed by
//! `rendering::clouds::drive_clouds` — when off it parks the cloud raymarch the
//! same way an absent cloud body does, so the sky composites with no cloud
//! layer at zero GPU cost; and the grass toggle, consumed by
//! `rendering::grass::drive_grass_tiles` — when off it parks the grass clipmap
//! (no tiles built, live tiles despawned). Anti-aliasing and foliage have the
//! same meaning in Kòrsou and live in `thalos_preferences::GraphicsPreferences`.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use thalos_capture_protocol::CaptureGraphicsOverrides;
use thalos_preferences::{
    GraphicsPreferences, QualityOverrides, QualityPreset, effective_graphics,
};

// ── Resource ───────────────────────────────────────────────────────────────────

/// User graphics/rendering preferences. Persisted as the `graphics` section of
/// [`crate::settings`]'s unified file.
///
/// Writers: the settings menu's Graphics tab in interactive play; the capture
/// runtime in headless mode. Everything else reads.
/// `Reflect`-registered (for a future in-game debug UI).
#[derive(Resource, Reflect, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[reflect(Resource)]
#[serde(default)]
pub struct GraphicsSettings {
    /// Render volumetric clouds. When off, the cloud raymarch is parked (no
    /// GPU work) and the body sky composites without a cloud layer.
    pub clouds: bool,
    /// Render the near-camera grass-blade decoration layer. When off, the grass
    /// clipmap is parked (no tiles built, live tiles despawned); the terrain
    /// albedo still carries the grass colour, so the ground reads green.
    pub grass: bool,
    /// Generate the near/mid grass blades on the GPU (the zero-persistent-memory
    /// vegetation path — `rendering::gpu_grass`, see `docs/world/vegetation.md` §13).
    /// When on, the CPU blade rings park and only the far card ring builds
    /// tiles; when off, the CPU clipmap covers the whole reach (the pre-rewrite
    /// behaviour, as a fallback). Draws nothing unless `grass` is also on.
    pub gpu_grass: bool,
    /// Multiplier on tile split distance. 1.0 is the unconstrained Showcase
    /// rule; 0.5 is the Laptop coarsening. The VRAM brake still multiplies on
    /// top of this.
    pub terrain_lod: f32,
    /// Live sun-shadow cascade count, 0..=4. `THALOS_SHADOW_CASCADES` still
    /// pins a measurement session.
    pub shadow_cascades: u8,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self::showcase()
    }
}

impl GraphicsSettings {
    pub const TERRAIN_LOD_MIN: f32 = 1.0 / 3.0;
    pub const TERRAIN_LOD_MAX: f32 = 1.0;
    pub const SHADOW_CASCADES_MAX: u8 = 4;

    pub fn showcase() -> Self {
        Self {
            clouds: true,
            grass: true,
            gpu_grass: true,
            terrain_lod: 1.0,
            shadow_cascades: Self::SHADOW_CASCADES_MAX,
        }
    }

    pub fn laptop() -> Self {
        Self {
            clouds: false,
            grass: false,
            gpu_grass: false,
            terrain_lod: 0.5,
            shadow_cascades: 2,
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

    pub fn matches_preset(&self, preset: QualityPreset) -> bool {
        Self::for_preset(preset).is_some_and(|expected| self == &expected)
    }

    pub fn sanitized(mut self) -> Self {
        self.terrain_lod = if self.terrain_lod.is_finite() {
            self.terrain_lod
                .clamp(Self::TERRAIN_LOD_MIN, Self::TERRAIN_LOD_MAX)
        } else {
            1.0
        };
        self.shadow_cascades = self.shadow_cascades.min(Self::SHADOW_CASCADES_MAX);
        self
    }

    /// Deterministic headless profile for one capture request.
    ///
    /// Captures never inherit the player's persisted preferences: every request
    /// starts from this type's defaults, then applies its typed patch. This also
    /// means a persistent host cannot leak one shot's settings into the next.
    pub fn for_capture(overrides: CaptureGraphicsOverrides) -> Self {
        let mut settings = Self::showcase();
        if let Some(clouds) = overrides.clouds {
            settings.clouds = clouds;
        }
        if let Some(grass) = overrides.grass {
            settings.grass = grass;
        }
        settings
    }
}

/// Session-only controls used to attribute interactive render cost.
///
/// These deliberately live beside, rather than inside, persisted preferences:
/// a diagnostic run must not silently change the player's normal graphics
/// setup. The effective value is recorded in every perf gauge.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub(crate) struct PerfRenderOverrides {
    foliage: Option<bool>,
}

impl PerfRenderOverrides {
    pub(crate) fn from_env() -> Self {
        let foliage = std::env::var("THALOS_PERF_FOLIAGE")
            .ok()
            .and_then(|raw| match parse_toggle(&raw) {
                Some(enabled) => {
                    eprintln!(
                        "THALOS_PERF_FOLIAGE={} — foliage pinned for this measurement session",
                        if enabled { "on" } else { "off" }
                    );
                    Some(enabled)
                }
                None => {
                    eprintln!(
                        "Unknown THALOS_PERF_FOLIAGE={raw:?}; expected on/off, true/false, or 1/0. Ignoring."
                    );
                    None
                }
            });
        Self { foliage }
    }

    pub(crate) fn foliage_enabled(
        &self,
        preferences: &thalos_preferences::GraphicsPreferences,
    ) -> bool {
        self.foliage.unwrap_or(preferences.foliage)
    }

    /// Change the session-only foliage gate during a controlled benchmark.
    ///
    /// This remains separate from persisted preferences: the offscreen matrix
    /// must never rewrite the player's graphics setup while it attributes a
    /// render cost.
    pub(crate) fn set_foliage(&mut self, enabled: bool) {
        self.foliage = Some(enabled);
    }
}

fn parse_toggle(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "on" | "1" | "true" | "yes" => Some(true),
        "off" | "0" | "false" | "no" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_legacy_atmosphere_setting_is_ignored() {
        let parsed: GraphicsSettings =
            ron::from_str("(clouds:true,grass:true,gpu_grass:true,msaa:Off,legacy_body_sky:true)")
                .expect("removed debug setting should not invalidate an existing settings file");

        assert_eq!(parsed, GraphicsSettings::default());
    }

    #[test]
    fn capture_settings_are_a_patch_over_deterministic_defaults() {
        let settings = GraphicsSettings::for_capture(CaptureGraphicsOverrides {
            clouds: Some(false),
            grass: None,
            foliage: Some(false),
        });
        assert!(!settings.clouds);
        assert!(settings.grass);
        assert!(settings.gpu_grass);
        assert_eq!(settings.terrain_lod, 1.0);
        assert_eq!(settings.shadow_cascades, 4);
    }

    #[test]
    fn laptop_game_bundle_parks_expensive_layers() {
        let laptop = GraphicsSettings::laptop();
        assert!(!laptop.clouds);
        assert!(!laptop.grass);
        assert_eq!(laptop.terrain_lod, 0.5);
        assert_eq!(laptop.shadow_cascades, 2);
        assert!(laptop.matches_preset(QualityPreset::Laptop));
        assert!(!laptop.matches_preset(QualityPreset::Showcase));
    }

    #[test]
    fn perf_foliage_toggle_is_typed() {
        assert_eq!(parse_toggle("off"), Some(false));
        assert_eq!(parse_toggle("YES"), Some(true));
        assert_eq!(parse_toggle("sometimes"), None);

        let preferences = thalos_preferences::GraphicsPreferences::default();
        let overrides = PerfRenderOverrides {
            foliage: Some(false),
        };
        assert!(preferences.foliage);
        assert!(!overrides.foliage_enabled(&preferences));
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────────

pub struct GraphicsSettingsPlugin;

impl Plugin for GraphicsSettingsPlugin {
    fn build(&self, app: &mut App) {
        // The resource is inserted in `main()` from the unified `settings.ron`
        // and persisted by `crate::settings::AppSettingsPlugin`; this plugin
        // registers the type and keeps game knobs stamped from the shared preset.
        app.register_type::<GraphicsSettings>().add_systems(
            Update,
            (
                stamp_game_graphics_from_preset,
                mark_custom_from_game_knobs,
                sync_shadow_cascade_budget,
            ),
        );
    }
}

fn stamp_game_graphics_from_preset(
    prefs: Res<GraphicsPreferences>,
    overrides: Res<QualityOverrides>,
    mut graphics: ResMut<GraphicsSettings>,
    mut last: Local<Option<QualityPreset>>,
) {
    let effective = effective_graphics(&prefs, &overrides);
    if last.is_none() {
        if effective.preset == QualityPreset::Laptop
            && !graphics.matches_preset(QualityPreset::Laptop)
        {
            graphics.apply_preset(QualityPreset::Laptop);
        }
        *last = Some(effective.preset);
        return;
    }
    if effective.preset != QualityPreset::Custom && Some(effective.preset) != *last {
        graphics.apply_preset(effective.preset);
    }
    *last = Some(effective.preset);
}

fn mark_custom_from_game_knobs(
    graphics: Res<GraphicsSettings>,
    overrides: Res<QualityOverrides>,
    mut prefs: ResMut<GraphicsPreferences>,
    mut menu: ResMut<thalos_preferences::SettingsMenu>,
    mut seen_startup: Local<bool>,
) {
    if overrides.preset.is_some() || prefs.preset == QualityPreset::Custom {
        return;
    }
    // The first observation is the persisted file, not a player edit. An
    // older settings.ron (grass already off, for example) must not flip a
    // named preset to Custom before anyone touches a knob.
    if !*seen_startup {
        *seen_startup = true;
        return;
    }
    if !graphics.is_changed() {
        return;
    }
    if !graphics.matches_preset(prefs.preset) {
        prefs.preset = QualityPreset::Custom;
        menu.dirty();
    }
}

fn sync_shadow_cascade_budget(
    graphics: Res<GraphicsSettings>,
    overrides: Res<QualityOverrides>,
    prefs: Res<GraphicsPreferences>,
) {
    if std::env::var("THALOS_SHADOW_CASCADES").is_ok() {
        return;
    }
    let cascades = if overrides.preset.is_some() {
        GraphicsSettings::for_preset(effective_graphics(&prefs, &overrides).preset)
            .map(|settings| settings.shadow_cascades)
            .unwrap_or(graphics.shadow_cascades)
    } else {
        graphics.shadow_cascades
    };
    crate::rendering::sun_shadow::set_cascade_budget(cascades as usize);
}
