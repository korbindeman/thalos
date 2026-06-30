//! Persisted graphics / rendering settings.
//!
//! [`GraphicsSettings`] is the file-backed graphics preference set, stored as
//! RON at [`SETTINGS_PATH`] (gitignored). It is loaded once when
//! [`GraphicsSettingsPlugin`] builds and saved whenever its value actually
//! changes ([`save_graphics_settings`] value-compares against the last write,
//! so an open settings tab doesn't churn the file).
//!
//! The settings menu's Graphics tab is the sole writer; render systems read.
//! Knobs: the volumetric-cloud toggle, consumed by
//! `rendering::clouds::drive_clouds` — when off it parks the cloud raymarch the
//! same way an absent cloud body does, so the sky composites with no cloud
//! layer at zero GPU cost; the grass toggle, consumed by
//! `rendering::grass::drive_grass_tiles` — when off it parks the grass clipmap
//! (no tiles built, live tiles despawned); and the MSAA level.

use std::path::Path;

use bevy::prelude::*;
use bevy::render::view::Msaa;
use serde::{Deserialize, Serialize};

/// Where graphics settings persist, relative to the working directory the game
/// already loads `assets/` from. Gitignored; recreated with defaults if missing
/// or unparseable.
pub const SETTINGS_PATH: &str = "user/graphics.ron";

// ── Resource ───────────────────────────────────────────────────────────────────

/// Multisample anti-aliasing level for the main 3D view.
///
/// `Off` keeps the post-process SMAA pass that [`space_camera_post_stack`] adds.
/// Any multisampled level **replaces** SMAA (MSAA covers geometry edges, and
/// running both just double-softens the image). Geometric specular AA in the
/// surface shaders is always on and independent of this knob.
///
/// [`space_camera_post_stack`]: thalos_body_render::space_camera_post_stack
#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MsaaSetting {
    Off,
    X2,
    X4,
    X8,
}

impl MsaaSetting {
    /// Sample count (1 = disabled).
    pub fn samples(self) -> u32 {
        match self {
            MsaaSetting::Off => 1,
            MsaaSetting::X2 => 2,
            MsaaSetting::X4 => 4,
            MsaaSetting::X8 => 8,
        }
    }

    /// Bevy per-camera [`Msaa`] component for this level.
    pub fn to_msaa(self) -> Msaa {
        match self {
            MsaaSetting::Off => Msaa::Off,
            MsaaSetting::X2 => Msaa::Sample2,
            MsaaSetting::X4 => Msaa::Sample4,
            MsaaSetting::X8 => Msaa::Sample8,
        }
    }

    /// Whether this level is multisampled (and therefore suppresses SMAA).
    pub fn is_multisampled(self) -> bool {
        self.samples() > 1
    }

    pub const ALL: [MsaaSetting; 4] = [
        MsaaSetting::Off,
        MsaaSetting::X2,
        MsaaSetting::X4,
        MsaaSetting::X8,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MsaaSetting::Off => "Off (SMAA)",
            MsaaSetting::X2 => "MSAA 2×",
            MsaaSetting::X4 => "MSAA 4×",
            MsaaSetting::X8 => "MSAA 8×",
        }
    }
}

/// User graphics/rendering preferences, persisted to [`SETTINGS_PATH`].
///
/// Writer: the settings menu's Graphics tab. Everything else reads.
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
    /// Multisample anti-aliasing level for the main 3D view. `Off` keeps SMAA;
    /// a multisampled level replaces it. See [`MsaaSetting`].
    pub msaa: MsaaSetting,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            clouds: true,
            grass: true,
            // Default off so the first run keeps the verified SMAA path; the
            // MSAA depth-resolve path is opt-in from the Graphics tab until it
            // has been runtime-verified.
            msaa: MsaaSetting::Off,
        }
    }
}

// ── Load / save ─────────────────────────────────────────────────────────────────

/// Read persisted graphics settings (defaults on first run or parse failure).
fn load() -> GraphicsSettings {
    match std::fs::read_to_string(SETTINGS_PATH) {
        Ok(source) => ron::from_str::<GraphicsSettings>(&source).unwrap_or_else(|err| {
            warn!("Failed to parse {SETTINGS_PATH}: {err}; using graphics-settings defaults.");
            GraphicsSettings::default()
        }),
        Err(_) => GraphicsSettings::default(), // first run
    }
}

fn save(settings: &GraphicsSettings) {
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
        warn!("Failed to save graphics settings to {SETTINGS_PATH}: {err}");
    }
}

// ── Plugin / systems ────────────────────────────────────────────────────────────

pub struct GraphicsSettingsPlugin;

impl Plugin for GraphicsSettingsPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<GraphicsSettings>()
            .insert_resource(load())
            .add_systems(Update, save_graphics_settings);
    }
}

/// Persist the settings whenever their value differs from the last write.
/// Value-compared (not change-detected) so the Graphics tab — which dereferences
/// the `ResMut` every frame it renders — doesn't rewrite the file each frame.
fn save_graphics_settings(
    settings: Res<GraphicsSettings>,
    mut last_saved: Local<Option<GraphicsSettings>>,
) {
    if last_saved.as_ref() != Some(&*settings) {
        save(&settings);
        *last_saved = Some(settings.clone());
    }
}
