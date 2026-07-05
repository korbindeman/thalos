//! Graphics / rendering settings.
//!
//! [`GraphicsSettings`] is the graphics preference set. It is persisted
//! (alongside window + units) by [`crate::settings`] as the `graphics` section
//! of the unified `settings.ron`; this module owns only the resource + its
//! `Reflect` registration, not the file IO.
//!
//! The settings menu's Graphics tab is the sole writer; render systems read.
//! Knobs: the volumetric-cloud toggle, consumed by
//! `rendering::clouds::drive_clouds` — when off it parks the cloud raymarch the
//! same way an absent cloud body does, so the sky composites with no cloud
//! layer at zero GPU cost; the grass toggle, consumed by
//! `rendering::grass::drive_grass_tiles` — when off it parks the grass clipmap
//! (no tiles built, live tiles despawned); and the MSAA level.

use bevy::prelude::*;
use bevy::render::view::Msaa;
use serde::{Deserialize, Serialize};

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

/// User graphics/rendering preferences. Persisted as the `graphics` section of
/// [`crate::settings`]'s unified file.
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
    /// Generate the near/mid grass blades on the GPU (the zero-persistent-memory
    /// vegetation path — `rendering::gpu_grass`, see `docs/vegetation.md` §13).
    /// When on, the CPU blade rings park and only the far card ring builds
    /// tiles; when off, the CPU clipmap covers the whole reach (the pre-rewrite
    /// behaviour, as a fallback). Draws nothing unless `grass` is also on.
    pub gpu_grass: bool,
    /// Multisample anti-aliasing level for the main 3D view. `Off` keeps SMAA;
    /// a multisampled level replaces it. See [`MsaaSetting`].
    pub msaa: MsaaSetting,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            clouds: true,
            grass: true,
            gpu_grass: true,
            // Default off so the first run keeps the verified SMAA path; the
            // MSAA depth-resolve path is opt-in from the Graphics tab until it
            // has been runtime-verified.
            msaa: MsaaSetting::Off,
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────────

pub struct GraphicsSettingsPlugin;

impl Plugin for GraphicsSettingsPlugin {
    fn build(&self, app: &mut App) {
        // The resource is inserted in `main()` from the unified `settings.ron`
        // and persisted by `crate::settings::AppSettingsPlugin`; this plugin
        // only registers the type for the reflection / debug-UI path.
        app.register_type::<GraphicsSettings>();
    }
}
