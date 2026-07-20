//! Screenshot capture — two independent paths.
//!
//! - **F2 window screenshot** ([`ScreenshotPlugin`]): interactive, saves the
//!   primary window to `~/Desktop/thalos`. Needs a real window.
//! - **Headless capture** ([`HeadlessScreenshotPlugin`]): no window, no winit —
//!   the whole game boots off-screen (driven by `ScheduleRunnerPlugin`, exactly
//!   like `just preview`), poses the ship camera at a scripted angle over a
//!   scene, renders to an off-screen image, writes a PNG, and exits. This is the
//!   agent-runnable path: build the game, run it with `THALOS_SCREENSHOT` set,
//!   and read the resulting PNG — the same self-inspection loop the procedural
//!   object previewer gives for assets, extended to a whole composed scene.
//!
//! The headless path is added by `main.rs` only when [`ScreenshotConfig::from_env`]
//! returns `Some` (i.e. `THALOS_SCREENSHOT` is set); `main.rs` also swaps the app
//! into no-window mode and forces the preset's spawn scenario so the world it
//! captures is fully built. The capture reuses the *real* [`ShipCamera`] (not a
//! fresh one) so the scene-depth copy, unified atmosphere pass, SSAO, and
//! sun-shadow rig all stay coupled to the view — a bespoke camera would render a
//! flat, sky-less, shadow-less scene.

use std::{
    env, fs,
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use bevy::{
    asset::RenderAssetUsages,
    camera::{ImageRenderTarget, RenderTarget},
    diagnostic::{DiagnosticPath, DiagnosticsStore},
    math::{DQuat, DVec3},
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
        view::screenshot::{Capturing, Screenshot, save_to_disk},
    },
    window::{CursorIcon, SystemCursorIcon},
};
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_body_render::HeightSource;
use thalos_body_render::renderer_tile_lod_m_at;
use thalos_body_render::udlod::prelude::{TerrainViewComponents, TileAtlas, TileTree};
use thalos_input::game::GameInputIntent;
use thalos_physics_local::HeightSourceRegistry;
use thalos_volumetric_clouds::{CloudsConfig, cloud_target_memory};
use thalos_world::BodyId;

use crate::camera::ShipCamera;
use crate::loading::AppState;
use crate::rendering::ground_terrain::BodyTerrain;
use crate::rendering::{SimulationState, SolarSystemState};
use crate::space_center::{HubContext, hub_context};
use crate::spawn::{Homeworld, SpawnSituation};
use crate::structures::StructureRegistry;

// ---------------------------------------------------------------------------
// F2 window screenshot
// ---------------------------------------------------------------------------

pub struct ScreenshotPlugin;

impl Plugin for ScreenshotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (screenshot_on_f2, screenshot_cursor));
    }
}

fn screenshot_on_f2(
    mut commands: Commands,
    input: Res<GameInputIntent>,
    active_captures: Query<Entity, With<Capturing>>,
) {
    if !input.screenshot || !active_captures.is_empty() {
        return;
    }

    let Some(dir) = screenshot_dir() else {
        warn!("could not resolve ~/Desktop/thalos for screenshot output");
        return;
    };

    if let Err(error) = fs::create_dir_all(&dir) {
        warn!(
            "could not create screenshot directory {}: {error}",
            dir.display()
        );
        return;
    }

    let path = dir.join(format!("thalos-{}.png", timestamp_millis()));
    info!("saving screenshot to {}", path.display());
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path));
}

fn screenshot_cursor(
    mut commands: Commands,
    active_captures: Query<Entity, With<Capturing>>,
    window: Single<Entity, With<Window>>,
) {
    if active_captures.is_empty() {
        commands.entity(*window).remove::<CursorIcon>();
    } else {
        commands
            .entity(*window)
            .insert(CursorIcon::from(SystemCursorIcon::Progress));
    }
}

fn screenshot_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(|home| PathBuf::from(home).join("Desktop").join("thalos"))
}

fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Headless capture — config
// ---------------------------------------------------------------------------

/// A named framing. Each preset knows which spawn scenario the world must boot
/// into (so `main.rs` can force it) and the default camera pose + output path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenshotPreset {
    /// A 3/4 aerial establishing shot of the surface spaceport (the runway,
    /// launchpads, tanks, hangars, and the parked aircraft). Boots the `runway`
    /// scenario, which builds the whole spaceport + settles the terrain behind
    /// the loading screen.
    SpaceportAerial,
    /// The space-center hub exactly as PLAY presents it: a clean start with the
    /// spaceport built but **no craft placed** — the canonical placeholder craft
    /// stays in orbit while the camera god-views the base. Boots the `hub`
    /// route (`just game hub`), i.e. the orbit scenario + `HubSpaceportBuild` +
    /// the hub opening on reveal. This is the regression probe for
    /// view-anchored surface detail: the camera is maximally decoupled from the
    /// craft, so anything anchored to the craft (scatter, shadows) goes missing
    /// here first.
    Hub,
    /// A low oblique survey over a **dry-belt desert** site — the verification
    /// probe for terrain-per-biome work (landcover palette, the scatter/biome
    /// gate). Boots the plain orbit scenario (no base), then searches the
    /// daylight hemisphere for the *driest* low-latitude dry-land direction and
    /// god-views the surface there, so the shot lands on genuine desert wherever
    /// the moisture field puts it (seed/rotation-independent). Trees/shrubs
    /// should be sparse-to-absent here and the ground tan; contrast with the
    /// green spaceport `spaceport-aerial` shot (equatorial wet belt).
    DryBelt,
    /// Low flight over the real runway, aimed through the lower sky so broken
    /// cumulus and its relationship to the ground are both visible.
    CloudRunway,
    /// Camera above the cloud deck at aircraft-cruise altitude, looking across
    /// the layer toward the sun.
    CloudCruise,
    /// Camera placed inside the current 2.0–3.3 km cloud shell.
    CloudInterior,
    /// Low-orbit tangent view of the cloud line inside the atmosphere limb.
    CloudLimb,
    /// Near-surface view toward a sun placed just above the local horizon.
    CloudSunset,
}

/// CLOUD-0 capture-only quality ladder. This intentionally controls only
/// knobs the current renderer already owns; CLOUD-2 replaces it with the real
/// viewport-relative quality ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloudCaptureQuality {
    Low,
    Baseline,
    High,
    Reference,
}

impl CloudCaptureQuality {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "low" => Some(Self::Low),
            "" | "baseline" | "current" | "default" => Some(Self::Baseline),
            "high" => Some(Self::High),
            "reference" | "ref" | "ultra" => Some(Self::Reference),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Baseline => "baseline",
            Self::High => "high",
            Self::Reference => "reference",
        }
    }

    fn view_steps(self) -> u32 {
        match self {
            Self::Low => 36,
            Self::Baseline => 60,
            Self::High => 72,
            Self::Reference => 96,
        }
    }

    fn shadow_steps(self) -> u32 {
        match self {
            Self::Low => 2,
            Self::Baseline => 4,
            Self::High => 6,
            Self::Reference => 8,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ScreenshotFraming {
    /// Original focus-orbit framing used by the base/hub regression shots.
    GodView,
    /// Camera at an exact AGL, looking relative to the local horizon. A
    /// `site_sun_elevation_deg` chooses a reproducible point on the globe whose
    /// local sun has that elevation; `None` keeps the real spaceport site.
    LocalCloud {
        camera_altitude_m: f64,
        look_elevation_deg: f32,
        site_sun_elevation_deg: Option<f32>,
        tangent_limb: bool,
    },
}

impl ScreenshotPreset {
    fn name(self) -> &'static str {
        match self {
            Self::SpaceportAerial => "spaceport-aerial",
            Self::Hub => "hub",
            Self::DryBelt => "dry-belt",
            Self::CloudRunway => "cloud-runway",
            Self::CloudCruise => "cloud-cruise",
            Self::CloudInterior => "cloud-interior",
            Self::CloudLimb => "cloud-limb",
            Self::CloudSunset => "cloud-sunset",
        }
    }

    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            // Truthy / unnamed → the default preset.
            "" | "1" | "true" | "yes" | "on" | "spaceport" | "spaceport-aerial" | "aerial"
            | "base" => Self::SpaceportAerial,
            "hub" | "space-center" | "spacecenter" | "play" => Self::Hub,
            "dry" | "dry-belt" | "drybelt" | "desert" | "biome" => Self::DryBelt,
            "cloud-runway" | "cloud_runway" | "clouds-runway" => Self::CloudRunway,
            "cloud-cruise" | "cloud_cruise" | "clouds-cruise" | "cloud-deck" => Self::CloudCruise,
            "cloud-interior" | "cloud_interior" | "inside-cloud" | "inside-clouds" => {
                Self::CloudInterior
            }
            "cloud-limb" | "cloud_limb" | "cloud-orbit" | "clouds-orbit" => Self::CloudLimb,
            "cloud-sunset" | "cloud_sunset" | "clouds-sunset" => Self::CloudSunset,
            other => {
                eprintln!("  Unknown THALOS_SCREENSHOT preset '{other}'; using spaceport-aerial.");
                Self::SpaceportAerial
            }
        }
    }

    /// The scenario the world must be booted into for this preset.
    pub fn spawn_situation(self) -> SpawnSituation {
        match self {
            Self::SpaceportAerial => SpawnSituation::Runway,
            // The hub is the PLAY path: the placeholder parking orbit plus the
            // spaceport build (armed by `main.rs` via `boots_hub`).
            Self::Hub => SpawnSituation::ShipOrbit,
            // Dry-belt frames wild terrain far from any base, so a plain orbit
            // scenario is enough; the driver poses the camera over the searched
            // desert site (the craft stays in orbit, irrelevant to the framing).
            Self::DryBelt => SpawnSituation::ShipOrbit,
            Self::CloudRunway => SpawnSituation::Runway,
            Self::CloudCruise | Self::CloudInterior | Self::CloudLimb | Self::CloudSunset => {
                SpawnSituation::ShipOrbit
            }
        }
    }

    /// Whether this preset boots the space-center hub route (spaceport built
    /// with no craft placed, hub opened on reveal) — `main.rs` arms
    /// `HubSpaceportBuild` + `InitialContext(Some(SpaceCenter))` for it, exactly
    /// like the start screen's PLAY.
    pub fn boots_hub(self) -> bool {
        matches!(self, Self::Hub)
    }

    fn defaults(self) -> ScreenshotConfig {
        match self {
            Self::SpaceportAerial => ScreenshotConfig {
                preset: self,
                out: PathBuf::from("tools/screenshots/spaceport_aerial.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 35.0,
                elevation_deg: 42.0,
                distance_m: 4200.0,
                warmup_frames: 180,
                tail_frames: 24,
                keep_hud: false,
                report: PathBuf::from("tools/screenshots/spaceport_aerial.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            // Matches the hub's establishing view (`BASE_ESTABLISHING_DISTANCE_M`,
            // the one god-view framing per base) so the capture shows what PLAY shows.
            Self::Hub => ScreenshotConfig {
                preset: self,
                out: PathBuf::from("tools/screenshots/hub.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 35.0,
                elevation_deg: 42.0,
                distance_m: crate::god_view::BASE_ESTABLISHING_DISTANCE_M as f64,
                warmup_frames: 240,
                tail_frames: 24,
                keep_hud: false,
                report: PathBuf::from("tools/screenshots/hub.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            // Low oblique, close in, so individual trees vs bare desert read
            // (like an eye-level survey across the ground). A long warmup: cold
            // tile streaming to a fresh wild site is slow (~15 s — the
            // cold-streaming floor), and nothing pre-built the terrain here.
            Self::DryBelt => ScreenshotConfig {
                preset: self,
                out: PathBuf::from("tools/screenshots/dry_belt.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 35.0,
                elevation_deg: 16.0,
                distance_m: 1400.0,
                warmup_frames: 600,
                tail_frames: 24,
                keep_hud: false,
                report: PathBuf::from("tools/screenshots/dry_belt.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::CloudRunway => ScreenshotConfig {
                preset: self,
                out: PathBuf::from("tools/screenshots/cloud_runway.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 30.0,
                elevation_deg: 8.0,
                distance_m: 4200.0,
                warmup_frames: 300,
                tail_frames: 24,
                keep_hud: false,
                report: PathBuf::from("tools/screenshots/cloud_runway.jsonl"),
                framing: ScreenshotFraming::LocalCloud {
                    camera_altitude_m: 850.0,
                    look_elevation_deg: -8.0,
                    site_sun_elevation_deg: None,
                    tangent_limb: false,
                },
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::CloudCruise => ScreenshotConfig {
                preset: self,
                out: PathBuf::from("tools/screenshots/cloud_cruise.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 20.0,
                elevation_deg: 8.0,
                distance_m: 4200.0,
                warmup_frames: 360,
                tail_frames: 24,
                keep_hud: false,
                report: PathBuf::from("tools/screenshots/cloud_cruise.jsonl"),
                framing: ScreenshotFraming::LocalCloud {
                    camera_altitude_m: 4_600.0,
                    look_elevation_deg: -3.0,
                    site_sun_elevation_deg: Some(35.0),
                    tangent_limb: false,
                },
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::CloudInterior => ScreenshotConfig {
                preset: self,
                out: PathBuf::from("tools/screenshots/cloud_interior.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 70.0,
                elevation_deg: 8.0,
                distance_m: 4200.0,
                warmup_frames: 360,
                tail_frames: 24,
                keep_hud: false,
                report: PathBuf::from("tools/screenshots/cloud_interior.jsonl"),
                framing: ScreenshotFraming::LocalCloud {
                    camera_altitude_m: 2_650.0,
                    look_elevation_deg: 0.0,
                    site_sun_elevation_deg: Some(35.0),
                    tangent_limb: false,
                },
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::CloudLimb => ScreenshotConfig {
                preset: self,
                out: PathBuf::from("tools/screenshots/cloud_limb.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 0.0,
                elevation_deg: 8.0,
                distance_m: 4200.0,
                warmup_frames: 360,
                tail_frames: 24,
                keep_hud: false,
                report: PathBuf::from("tools/screenshots/cloud_limb.jsonl"),
                framing: ScreenshotFraming::LocalCloud {
                    camera_altitude_m: 200_000.0,
                    look_elevation_deg: 0.35,
                    site_sun_elevation_deg: Some(3.0),
                    tangent_limb: true,
                },
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::CloudSunset => ScreenshotConfig {
                preset: self,
                out: PathBuf::from("tools/screenshots/cloud_sunset.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 0.0,
                elevation_deg: 8.0,
                distance_m: 4200.0,
                warmup_frames: 360,
                tail_frames: 24,
                keep_hud: false,
                report: PathBuf::from("tools/screenshots/cloud_sunset.jsonl"),
                framing: ScreenshotFraming::LocalCloud {
                    camera_altitude_m: 700.0,
                    look_elevation_deg: 1.5,
                    site_sun_elevation_deg: Some(1.0),
                    tangent_limb: false,
                },
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
        }
    }
}

/// A headless screenshot request, resolved from `THALOS_SCREENSHOT*` env vars.
///
/// The legacy spaceport/hub presets use a god-view around their resolved focus;
/// cloud presets use an exact AGL and local-horizon look direction. Every pose
/// is overridable so a diagnostic angle can be reproduced without recompiling.
#[derive(Resource, Clone, Debug)]
pub struct ScreenshotConfig {
    pub preset: ScreenshotPreset,
    /// Output PNG path (relative to the working dir). Its parent is created.
    pub out: PathBuf,
    pub width: u32,
    pub height: u32,
    /// Camera azimuth, degrees. God-view zero is local east; cloud-view zero
    /// faces the projected sun.
    pub azimuth_deg: f32,
    /// God-view camera elevation above the local horizon (90 = straight down).
    pub elevation_deg: f32,
    /// God-view boom distance from the focus, metres.
    pub distance_m: f64,
    /// Frames to render (posing the scripted camera) after reaching `Running`
    /// before the capture — lets pipelines compile and shadows / atmosphere /
    /// tiles converge to the new framing.
    pub warmup_frames: u32,
    /// Frames to keep running after the capture so the async GPU readback flushes
    /// to disk before the app exits.
    pub tail_frames: u32,
    /// Keep the flight HUD + overlays visible in the capture
    /// (`THALOS_SCREENSHOT_HUD=1`). Default hides them for clean scene shots;
    /// set it when iterating on the HUD itself.
    pub keep_hud: bool,
    /// Machine-readable CLOUD-0 timing/memory report. One JSON object is
    /// written per capture so reports can be concatenated directly.
    pub report: PathBuf,
    framing: ScreenshotFraming,
    pub cloud_quality: CloudCaptureQuality,
    /// Whether steady and moving cloud history are allowed. False produces a
    /// raw temporal-disabled diagnostic frame.
    pub cloud_temporal: bool,
    /// Optional global multiplier on the current weather coverage map.
    pub cloud_coverage_scale: Option<f32>,
}

impl ScreenshotConfig {
    /// Resolve a request from the environment. `None` unless `THALOS_SCREENSHOT`
    /// is set (that presence is what switches the whole binary into headless
    /// mode; see `main.rs`).
    ///
    /// - `THALOS_SCREENSHOT` — preset name (or a truthy value for the default).
    /// - `THALOS_SCREENSHOT_OUT` — output PNG path.
    /// - `THALOS_SCREENSHOT_SIZE` — `WIDTHxHEIGHT` (e.g. `2560x1440`).
    /// - `THALOS_SCREENSHOT_AZIMUTH` / `_ELEVATION` — camera angles, degrees.
    /// - `THALOS_SCREENSHOT_DISTANCE` — boom distance, metres.
    /// - `THALOS_SCREENSHOT_WARMUP` — warmup frames before the capture.
    /// - `THALOS_SCREENSHOT_CAMERA_ALTITUDE` / `_LOOK_ELEVATION` — local cloud
    ///   camera AGL and look angle above the horizon.
    /// - `THALOS_SCREENSHOT_SUN_ELEVATION` — select a globe site with this
    ///   local sun elevation (cloud presets only; moves away from the runway).
    /// - `THALOS_SCREENSHOT_CLOUD_QUALITY` — low, baseline, high, reference.
    /// - `THALOS_SCREENSHOT_CLOUD_TEMPORAL` — 0/off disables all history.
    /// - `THALOS_SCREENSHOT_CLOUD_COVERAGE` — optional global coverage scale.
    /// - `THALOS_SCREENSHOT_REPORT` — JSONL report path (defaults beside PNG).
    pub fn from_env() -> Option<Self> {
        let raw = env::var("THALOS_SCREENSHOT").ok()?;
        let mut cfg = ScreenshotPreset::parse(&raw).defaults();

        if let Some(out) = env::var_os("THALOS_SCREENSHOT_OUT") {
            cfg.out = PathBuf::from(out);
        }
        cfg.report = env::var_os("THALOS_SCREENSHOT_REPORT")
            .map(PathBuf::from)
            .unwrap_or_else(|| cfg.out.with_extension("jsonl"));
        if let Some((w, h)) = env::var("THALOS_SCREENSHOT_SIZE")
            .ok()
            .and_then(|s| parse_size(&s))
        {
            cfg.width = w;
            cfg.height = h;
        }
        if let Some(v) = env_parse::<f32>("THALOS_SCREENSHOT_AZIMUTH") {
            cfg.azimuth_deg = v;
        }
        if let Some(v) = env_parse::<f32>("THALOS_SCREENSHOT_ELEVATION") {
            cfg.elevation_deg = v.clamp(1.0, 90.0);
        }
        if let Some(v) = env_parse::<f64>("THALOS_SCREENSHOT_DISTANCE") {
            cfg.distance_m = v.max(1.0);
        }
        if let Some(v) = env_parse::<u32>("THALOS_SCREENSHOT_WARMUP") {
            cfg.warmup_frames = v;
        }
        if let Ok(v) = env::var("THALOS_SCREENSHOT_HUD") {
            cfg.keep_hud = matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }
        if let Some(v) = env_parse::<f64>("THALOS_SCREENSHOT_CAMERA_ALTITUDE")
            && let ScreenshotFraming::LocalCloud {
                camera_altitude_m, ..
            } = &mut cfg.framing
        {
            *camera_altitude_m = v.max(0.0);
        }
        if let Some(v) = env_parse::<f32>("THALOS_SCREENSHOT_LOOK_ELEVATION")
            && let ScreenshotFraming::LocalCloud {
                look_elevation_deg, ..
            } = &mut cfg.framing
        {
            *look_elevation_deg = v.clamp(-89.0, 89.0);
        }
        if let Some(v) = env_parse::<f32>("THALOS_SCREENSHOT_SUN_ELEVATION")
            && let ScreenshotFraming::LocalCloud {
                site_sun_elevation_deg,
                ..
            } = &mut cfg.framing
        {
            *site_sun_elevation_deg = Some(v.clamp(-10.0, 90.0));
        }
        if let Ok(raw) = env::var("THALOS_SCREENSHOT_CLOUD_QUALITY") {
            if let Some(quality) = CloudCaptureQuality::parse(&raw) {
                cfg.cloud_quality = quality;
            } else {
                eprintln!(
                    "  Unknown THALOS_SCREENSHOT_CLOUD_QUALITY={raw:?}; using {}.",
                    cfg.cloud_quality.name()
                );
            }
        }
        if let Ok(raw) = env::var("THALOS_SCREENSHOT_CLOUD_TEMPORAL") {
            match parse_bool(&raw) {
                Some(value) => cfg.cloud_temporal = value,
                None => eprintln!(
                    "  Unknown THALOS_SCREENSHOT_CLOUD_TEMPORAL={raw:?}; expected on/off."
                ),
            }
        }
        if let Some(v) = env_parse::<f32>("THALOS_SCREENSHOT_CLOUD_COVERAGE") {
            cfg.cloud_coverage_scale = Some(v.clamp(0.0, 4.0));
        }
        Some(cfg)
    }
}

fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    env::var(key).ok().and_then(|s| s.trim().parse::<T>().ok())
}

fn parse_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Parse a `WIDTHxHEIGHT` string (`x` or `*` separator).
fn parse_size(s: &str) -> Option<(u32, u32)> {
    let (w, h) = s.trim().split_once(['x', 'X', '*'])?;
    Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
}

// ---------------------------------------------------------------------------
// Headless capture — driver
// ---------------------------------------------------------------------------

/// Headless capture state machine. **Sole writer:** the headless systems below.
#[derive(Resource, Default)]
struct ScreenshotDriver {
    /// The off-screen render target the ship camera draws into and the capture
    /// reads back. `None` until [`setup_screenshot_target`] runs.
    target: Option<Handle<Image>>,
    /// Whether the ship camera has been retargeted onto [`Self::target`].
    retargeted: bool,
    /// Frames spent posing the camera in `Running` before the capture.
    running_frames: u32,
    /// The capture has been requested (screenshot entity spawned).
    captured: bool,
    /// Frames since the capture, for the readback-flush tail.
    tail: u32,
    /// Cached body-fixed direction of the searched dry-belt site (DryBelt preset
    /// only), resolved once so the framing stays fixed across warmup.
    dry_site_dir: Option<DVec3>,
}

pub struct HeadlessScreenshotPlugin;

impl Plugin for HeadlessScreenshotPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScreenshotDriver>()
            .add_systems(Startup, setup_screenshot_target)
            .add_systems(
                Update,
                (retarget_ship_camera, hide_overlays, configure_cloud_capture),
            )
            // The pose + capture driver runs *after* the flight camera so it wins
            // (last writer of the `ShipCamera` transform), and only once the world
            // is up so the spaceport it frames is fully built.
            .add_systems(
                Update,
                drive_headless_screenshot
                    .after(crate::SimStage::Camera)
                    .run_if(in_state(AppState::Running)),
            )
            // Diagnostic transect across the spaceport basin (headless runs
            // only): resident tile LOD + rendered height vs the basin plane.
            .add_systems(Update, probe_apron_lod.run_if(in_state(AppState::Running)));
    }
}

/// Apply capture-only cloud quality controls once, after the cloud plugin's
/// startup initialization. The normal game never sees these overrides because
/// this system only exists in [`HeadlessScreenshotPlugin`].
fn configure_cloud_capture(
    cfg: Res<ScreenshotConfig>,
    mut clouds: ResMut<CloudsConfig>,
    mut applied: Local<bool>,
) {
    if *applied {
        return;
    }
    clouds.clouds_raymarch_steps_count = cfg.cloud_quality.view_steps();
    clouds.clouds_shadow_raymarch_steps_count = cfg.cloud_quality.shadow_steps();
    clouds.reprojection_strength = if cfg.cloud_temporal { 0.95 } else { 0.0 };
    if let Some(coverage) = cfg.cloud_coverage_scale {
        clouds.clouds_coverage = coverage;
    }
    info!(
        target: "thalos::screenshot",
        "cloud probe: quality={} view_steps={} shadow_steps={} temporal={} coverage={:.2}",
        cfg.cloud_quality.name(),
        clouds.clouds_raymarch_steps_count,
        clouds.clouds_shadow_raymarch_steps_count,
        cfg.cloud_temporal,
        clouds.clouds_coverage,
    );
    *applied = true;
}

/// Diagnostic: for each probe offset across the basin (metres from the pad
/// centre along `center_dir × heading`), log the resident tile texel size and
/// the height-mirror sample relative to the basin elevation `E`, plus the tile
/// tree's view distance to the pad. Reads the same surfaces the renderer draws,
/// so a paving/terrain height fight shows up as numbers instead of guesswork —
/// this transect is what pinned the 2026-07 "dark serrated apron fringe" to the
/// basin flatten's plane being tangent at the offset rect centre instead of the
/// runway centre. Headless-only (the plugin is added only under
/// `THALOS_SCREENSHOT`), one line per ~4 s.
fn probe_apron_lod(
    mut frame: Local<u32>,
    sim: Res<SimulationState>,
    site: Option<Res<crate::runway::RunwaySite>>,
    tile_trees: Res<TerrainViewComponents<TileTree>>,
    terrains: Query<(Entity, &BodyTerrain, &TileAtlas)>,
    camera_q: Query<Entity, With<ShipCamera>>,
    height_sources: Res<HeightSourceRegistry>,
) {
    *frame += 1;
    if *frame % 240 != 0 {
        return;
    }
    let Some(site) = site else { return };
    let Some((terrain_entity, _, atlas)) =
        terrains.iter().find(|(_, t, _)| t.body_id == site.body_id)
    else {
        return;
    };
    let Some(camera) = camera_q.iter().next() else {
        return;
    };
    let Some(tree) = tile_trees.get(&(terrain_entity, camera)) else {
        return;
    };
    let r = sim.system.bodies[site.body_id].radius_m + site.elevation_m;
    let across = site.center_dir.cross(site.heading_tangent).normalize();
    let pad = site.center_dir * r;
    let view_dist_km = (tree.view_position() - pad).length() / 1000.0;
    let hs = height_sources.get(site.body_id);
    let offs = [
        -1200.0f64, -560.0, -520.0, -470.0, -350.0, 0.0, 350.0, 470.0, 520.0, 560.0, 1200.0,
    ];
    let lods: Vec<String> = offs
        .iter()
        .map(|off| {
            let dir = (site.center_dir * r + across * *off).normalize();
            let p = dir * r;
            let lod = match renderer_tile_lod_m_at(atlas, tree, p) {
                Some(m) => format!("{m:.1}"),
                None => "none".to_string(),
            };
            // Height relative to the basin plane elevation E, from the GPU-atlas
            // height mirror (the same surface the renderer draws).
            let dh = hs
                .as_ref()
                .map(|h| {
                    h.sample_height_m(dir.as_vec3(), 1.0)
                        .map(|hm| format!("{:+.2}", hm as f64 - site.elevation_m))
                        .unwrap_or_else(|| "?".into())
                })
                .unwrap_or_else(|| "?".into());
            format!("{off:+.0}m:[{lod}|{dh}]")
        })
        .collect();
    info!(
        target: "thalos::screenshot",
        "apron probe: view->pad {:.2} km | off:[texel_m|dh_m] {}",
        view_dist_km,
        lods.join(" ")
    );
}

/// Create the off-screen render target the ship camera will draw into.
fn setup_screenshot_target(
    mut driver: ResMut<ScreenshotDriver>,
    cfg: Res<ScreenshotConfig>,
    mut images: ResMut<Assets<Image>>,
) {
    let mut target = Image::new_fill(
        Extent3d {
            width: cfg.width,
            height: cfg.height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    target.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT;
    driver.target = Some(images.add(target));
}

/// Point the *real* ship camera at the off-screen target (once). Reusing the
/// ship camera keeps the scene-depth copy, atmosphere pass, SSAO, and sun-shadow
/// rig — all of which filter on the `ShipCamera` marker — coupled to the capture.
/// With no primary window, a camera without an image target renders nowhere.
fn retarget_ship_camera(
    mut commands: Commands,
    mut driver: ResMut<ScreenshotDriver>,
    cameras: Query<Entity, With<ShipCamera>>,
) {
    if driver.retargeted {
        return;
    }
    let Some(target) = driver.target.clone() else {
        return;
    };
    let Ok(camera) = cameras.single() else {
        return; // camera not spawned yet
    };
    commands
        .entity(camera)
        .insert(RenderTarget::Image(ImageRenderTarget::from(target)));
    driver.retargeted = true;
}

/// Hide the flight HUD and every photo-mode overlay so the capture shows only
/// the world. Photo mode also gates the gizmo draws (orbits, trajectory, etc.).
/// Skipped entirely under `THALOS_SCREENSHOT_HUD=1` — the HUD-iteration mode.
fn hide_overlays(
    cfg: Res<ScreenshotConfig>,
    mut photo: ResMut<crate::photo_mode::PhotoMode>,
    mut overlays: ParamSet<(
        Query<&mut Visibility, With<crate::hud::HudPanel>>,
        Query<&mut Visibility, With<crate::photo_mode::HideInPhotoMode>>,
    )>,
) {
    if cfg.keep_hud {
        return;
    }
    if !photo.active {
        photo.active = true;
    }
    for mut vis in overlays.p0().iter_mut() {
        if *vis != Visibility::Hidden {
            *vis = Visibility::Hidden;
        }
    }
    for mut vis in overlays.p1().iter_mut() {
        if *vis != Visibility::Hidden {
            *vis = Visibility::Hidden;
        }
    }
}

/// Pose the scripted camera every `Running` frame, then capture once warmed up and
/// exit after the readback tail.
#[allow(clippy::too_many_arguments)]
fn drive_headless_screenshot(
    cfg: Res<ScreenshotConfig>,
    mut driver: ResMut<ScreenshotDriver>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    height_sources: Res<HeightSourceRegistry>,
    registry: Res<StructureRegistry>,
    homeworld: Res<Homeworld>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut camera: Query<(&mut Transform, &mut CellCoord), With<ShipCamera>>,
    diagnostics: Res<DiagnosticsStore>,
    clouds: Res<CloudsConfig>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(target) = driver.target.clone() else {
        return;
    };
    if !driver.retargeted {
        return; // wait until the camera renders into our target
    }

    // Resolve the requested surface site. Dry-belt searches for a biome probe;
    // cloud shots may choose a reproducible point by local sun elevation; the
    // remaining shots retain the real spaceport focus.
    let ctx = if cfg.preset == ScreenshotPreset::DryBelt {
        dry_site_context(
            &sim,
            &solar,
            &height_sources,
            homeworld.0,
            &mut driver.dry_site_dir,
        )
    } else {
        match cfg.framing {
            ScreenshotFraming::GodView
            | ScreenshotFraming::LocalCloud {
                site_sun_elevation_deg: None,
                ..
            } => hub_context(&sim, &solar, &height_sources, &registry, homeworld.0),
            ScreenshotFraming::LocalCloud {
                site_sun_elevation_deg: Some(sun_elevation_deg),
                ..
            } => cloud_site_context(
                &sim,
                &solar,
                &height_sources,
                homeworld.0,
                sun_elevation_deg,
            ),
        }
    };
    let Some(ctx) = ctx else {
        return;
    };
    let Ok(root) = root_grid.single() else {
        return;
    };
    let Ok((mut transform, mut cell)) = camera.single_mut() else {
        return;
    };
    pose_camera(&cfg, &ctx, &solar, root, &mut transform, &mut cell);

    if driver.captured {
        driver.tail += 1;
        if driver.tail >= cfg.tail_frames {
            info!(target: "thalos::screenshot", "headless capture flushed — exiting");
            exit.write(AppExit::Success);
        }
        return;
    }

    driver.running_frames += 1;
    if driver.running_frames < cfg.warmup_frames {
        return;
    }

    if let Some(parent) = cfg.out.parent() {
        fs::create_dir_all(parent).ok();
    }
    info!(
        target: "thalos::screenshot",
        "capturing {} ({}x{}) preset={} quality={} temporal={}",
        cfg.out.display(),
        cfg.width,
        cfg.height,
        cfg.preset.name(),
        cfg.cloud_quality.name(),
        cfg.cloud_temporal,
    );
    if let Err(error) = write_cloud_probe_report(&cfg, &clouds, &diagnostics) {
        warn!(
            target: "thalos::screenshot",
            "could not write cloud probe report {}: {error}",
            cfg.report.display()
        );
    }
    commands
        .spawn(Screenshot::image(target))
        .observe(save_to_disk(cfg.out.clone()));
    driver.captured = true;
}

#[derive(Debug, Clone, Copy)]
struct ProbeStats {
    count: usize,
    min: f64,
    mean: f64,
    p50: f64,
    p95: f64,
    max: f64,
}

/// Render diagnostics inherit any enclosing recorder spans, so the exact path
/// may gain components as Bevy's render schedule evolves. Select by pass
/// component + terminal field instead of baking the current hierarchy into
/// the probe format.
fn cloud_probe_stats(
    diagnostics: &DiagnosticsStore,
    field: &str,
) -> (Option<String>, Option<ProbeStats>) {
    let diagnostic = diagnostics.iter().find(|diagnostic| {
        let path = diagnostic.path();
        path.components()
            .any(|component| component == "volumetric_clouds")
            && path.components().last() == Some(field)
    });
    let Some(diagnostic) = diagnostic else {
        return (None, None);
    };
    let path = diagnostic.path().clone();
    (
        Some(path.as_str().to_string()),
        probe_stats(diagnostics, &path),
    )
}

fn probe_stats(diagnostics: &DiagnosticsStore, path: &DiagnosticPath) -> Option<ProbeStats> {
    let mut values: Vec<f64> = diagnostics
        .get(path)?
        .values()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let count = values.len();
    let percentile = |p: f64| {
        let i = ((count as f64 * p).ceil() as usize)
            .saturating_sub(1)
            .min(count - 1);
        values[i]
    };
    Some(ProbeStats {
        count,
        min: values[0],
        mean: values.iter().sum::<f64>() / count as f64,
        p50: percentile(0.50),
        p95: percentile(0.95),
        max: values[count - 1],
    })
}

fn stats_json(stats: Option<ProbeStats>) -> String {
    match stats {
        Some(s) => format!(
            "{{\"samples\":{},\"min_ms\":{:.6},\"mean_ms\":{:.6},\"p50_ms\":{:.6},\"p95_ms\":{:.6},\"max_ms\":{:.6}}}",
            s.count, s.min, s.mean, s.p50, s.p95, s.max
        ),
        None => "null".to_string(),
    }
}

fn framing_json(cfg: &ScreenshotConfig) -> String {
    match cfg.framing {
        ScreenshotFraming::GodView => format!(
            "{{\"kind\":\"god_view\",\"azimuth_deg\":{:.4},\"elevation_deg\":{:.4},\"distance_m\":{:.3}}}",
            cfg.azimuth_deg, cfg.elevation_deg, cfg.distance_m
        ),
        ScreenshotFraming::LocalCloud {
            camera_altitude_m,
            look_elevation_deg,
            site_sun_elevation_deg,
            tangent_limb,
        } => {
            let sun_elevation = site_sun_elevation_deg
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "null".to_string());
            format!(
                "{{\"kind\":\"local_cloud\",\"azimuth_deg\":{:.4},\"camera_altitude_m\":{:.3},\"look_elevation_deg\":{:.4},\"site_sun_elevation_deg\":{},\"tangent_limb\":{}}}",
                cfg.azimuth_deg, camera_altitude_m, look_elevation_deg, sun_elevation, tangent_limb,
            )
        }
    }
}

/// Append one self-contained JSON object. Keeping it JSONL means repeated runs
/// and the five-preset suite can share a report path without a merge step.
fn write_cloud_probe_report(
    cfg: &ScreenshotConfig,
    clouds: &CloudsConfig,
    diagnostics: &DiagnosticsStore,
) -> std::io::Result<()> {
    if let Some(parent) = cfg.report.parent() {
        fs::create_dir_all(parent)?;
    }
    let memory = cloud_target_memory();
    let screenshot_target_bytes = cfg.width as u64 * cfg.height as u64 * 4;
    let framing = framing_json(cfg);
    let (gpu_path, gpu_stats) = cloud_probe_stats(diagnostics, "elapsed_gpu");
    let (cpu_path, cpu_stats) = cloud_probe_stats(diagnostics, "elapsed_cpu");
    let gpu = stats_json(gpu_stats);
    let cpu = stats_json(cpu_stats);
    let gpu_path = gpu_path
        .map(|path| format!("\"{path}\""))
        .unwrap_or_else(|| "null".to_string());
    let cpu_path = cpu_path
        .map(|path| format!("\"{path}\""))
        .unwrap_or_else(|| "null".to_string());
    let unix_ms = timestamp_millis();
    let line = format!(
        concat!(
            "{{\"schema\":\"thalos.cloud_probe.v1\",",
            "\"unix_ms\":{},\"preset\":\"{}\",",
            "\"viewport\":[{},{}],\"screenshot_target_bytes\":{},",
            "\"framing\":{},",
            "\"cloud_internal_resolution\":[{},{}],",
            "\"quality\":\"{}\",\"temporal\":{},",
            "\"view_steps\":{},\"shadow_steps\":{},\"coverage_scale\":{:.4},",
            "\"timing\":{{\"gpu\":{},\"cpu\":{}}},",
            "\"timing_paths\":{{\"gpu\":{},\"cpu\":{}}},",
            "\"memory\":{{\"render_bytes\":{},\"distance_bytes\":{},",
            "\"history_bytes\":{},\"history_distance_bytes\":{},",
            "\"base_atlas_bytes\":{},\"worley_bytes\":{},",
            "\"coverage_bytes\":{},\"total_bytes\":{}}}}}\n"
        ),
        unix_ms,
        cfg.preset.name(),
        cfg.width,
        cfg.height,
        screenshot_target_bytes,
        framing,
        thalos_volumetric_clouds::RENDER_WIDTH,
        thalos_volumetric_clouds::RENDER_HEIGHT,
        cfg.cloud_quality.name(),
        cfg.cloud_temporal,
        clouds.clouds_raymarch_steps_count,
        clouds.clouds_shadow_raymarch_steps_count,
        clouds.clouds_coverage,
        gpu,
        cpu,
        gpu_path,
        cpu_path,
        memory.render_bytes,
        memory.distance_bytes,
        memory.history_bytes,
        memory.history_distance_bytes,
        memory.base_atlas_bytes,
        memory.worley_bytes,
        memory.coverage_bytes,
        memory.total_bytes,
    );
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(&cfg.report)?
        .write_all(line.as_bytes())?;
    info!(
        target: "thalos::screenshot",
        "cloud probe report appended to {} (cloud targets {:.2} MiB)",
        cfg.report.display(),
        memory.total_bytes as f64 / (1024.0 * 1024.0),
    );
    Ok(())
}

/// Dispatch to the preset's god-view or local-horizon pose. Detail systems
/// (scatter, shadows) follow the camera via `rendering::view_anchor`.
fn pose_camera(
    cfg: &ScreenshotConfig,
    ctx: &HubContext,
    solar: &SolarSystemState,
    root: &Grid,
    transform: &mut Transform,
    cell: &mut CellCoord,
) {
    match cfg.framing {
        ScreenshotFraming::GodView => {
            pose_god_view_camera(cfg, ctx, root, transform, cell);
        }
        ScreenshotFraming::LocalCloud {
            camera_altitude_m,
            look_elevation_deg,
            tangent_limb,
            ..
        } => pose_local_cloud_camera(
            cfg,
            ctx,
            solar,
            root,
            transform,
            cell,
            camera_altitude_m,
            look_elevation_deg,
            tangent_limb,
        ),
    }
}

fn pose_god_view_camera(
    cfg: &ScreenshotConfig,
    ctx: &HubContext,
    root: &Grid,
    transform: &mut Transform,
    cell: &mut CellCoord,
) {
    let up = ctx.up_world;
    // Tangent basis on the local horizon (east / north), robust near the poles.
    let seed = if up.dot(DVec3::Y).abs() < 0.99 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let east = seed.cross(up).normalize();
    let north = up.cross(east).normalize();

    let az = (cfg.azimuth_deg as f64).to_radians();
    let elev = (cfg.elevation_deg as f64).to_radians();
    let horiz = east * az.cos() + north * az.sin();
    let offset_dir = horiz * elev.cos() + up * elev.sin();

    let focus = ctx.center_world;
    let camera_world = focus + offset_dir * cfg.distance_m;
    let to_focus = (focus - camera_world).normalize();
    // At (near) top-down the look direction is anti-parallel to `up`, which makes
    // `looking_to`'s roll reference degenerate — fall back to north.
    let look_up = if to_focus.dot(up).abs() > 0.99 {
        north
    } else {
        up
    };

    let (next_cell, local) = root.translation_to_grid(camera_world);
    *cell = next_cell;
    *transform =
        Transform::from_translation(local).looking_to(to_focus.as_vec3(), look_up.as_vec3());
}

/// Sample LOD hint (m) for the dry-site search's height / moisture probes — a
/// coarse focus query, not a placement gate, so a wide hint is fine.
const DRY_SITE_LOD_M: f32 = 8.0;
/// Minimum height above the reference radius (m) for a dry-site candidate, so
/// the search lands on real land, not the shoreline / seabed.
const DRY_SITE_MIN_HEIGHT_M: f32 = 3.0;
/// Keep the search below this `|sin(latitude)|` (~46°) so it returns a warm
/// subtropical desert, not a cold polar barren the treeline would clear anyway.
const DRY_SITE_MAX_ABS_LAT_SIN: f64 = 0.72;

/// Focus for the [`ScreenshotPreset::DryBelt`] biome probe: the driest sunlit
/// dry-land site on `body_id`, framed at the surface. Mirrors [`hub_context`]'s
/// output shape (world-space focus + local up), but instead of a base it
/// searches for desert via [`find_driest_site`] and caches the body-fixed
/// direction in `cached_dir` so the framing is stable across warmup. `None`
/// before body state is available.
fn dry_site_context(
    sim: &SimulationState,
    solar: &SolarSystemState,
    height_sources: &HeightSourceRegistry,
    body_id: BodyId,
    cached_dir: &mut Option<DVec3>,
) -> Option<HubContext> {
    let states = solar.states.as_deref()?;
    let body_state = states.get(body_id)?;
    let radius_m = sim.system.bodies.get(body_id)?.radius_m;
    let hs = height_sources.get(body_id)?;

    let dir_body = match *cached_dir {
        Some(d) => d,
        None => {
            // Sub-stellar direction in the body-fixed frame (Pyros sits at the
            // heliocentric origin, so `-body_position` is local noon) → the lit
            // hemisphere to search.
            let sun_inertial = (-body_state.position).normalize_or_zero();
            let sun_body = if sun_inertial == DVec3::ZERO {
                DVec3::Y
            } else {
                (body_state.orientation.inverse() * sun_inertial).normalize()
            };
            let d = find_driest_site(hs.as_ref(), sun_body);
            *cached_dir = Some(d);
            let moisture = hs.landcover_moisture(d);
            let lat_deg = d.y.clamp(-1.0, 1.0).asin().to_degrees();
            info!(
                target: "thalos::screenshot",
                "dry-belt site: lat {lat_deg:.0}°, macro moisture {moisture:+.2} (drier = more desert)"
            );
            d
        }
    };

    let up_world = (body_state.orientation * dir_body).normalize();
    let height_m = hs
        .sample_height_m(dir_body.as_vec3(), DRY_SITE_LOD_M)
        .unwrap_or(0.0)
        .max(0.0) as f64;
    let pad_r = radius_m + height_m;
    Some(HubContext {
        body_id,
        center_world: body_state.position + up_world * pad_r,
        up_world,
        pad_r,
    })
}

/// Spiral the daylight hemisphere around `sun_dir_body` and return the
/// body-fixed direction with the **lowest** macro landcover moisture among
/// dry-land, low-latitude candidates — the desert the scatter/biome gate should
/// render treeless. Falls back to the driest land seen, else the sub-stellar
/// point. Pure query over the [`HeightSource`] (analytic moisture + CPU-fallback
/// height), so it does not need resident tiles.
fn find_driest_site(hs: &dyn HeightSource, sun_dir_body: DVec3) -> DVec3 {
    let sun = sun_dir_body.try_normalize().unwrap_or(DVec3::Y);
    let t1 = {
        let seed = if sun.y.abs() < 0.9 { DVec3::Y } else { DVec3::X };
        (seed - sun * seed.dot(sun)).normalize()
    };
    let t2 = sun.cross(t1).normalize();

    const RINGS: usize = 22;
    // ~63° from local noon: comfortably lit, and wide enough to reach the
    // subtropical dry belt (~15–40° latitude) from an equatorial sub-stellar point.
    const MAX_ANGLE_RAD: f64 = 1.10;
    let mut best: Option<(f32, DVec3)> = None; // (moisture, dir) — minimise moisture
    for ring in 0..=RINGS {
        let theta = MAX_ANGLE_RAD * ring as f64 / RINGS as f64;
        let (st, ct) = theta.sin_cos();
        let spokes = ((st * 28.0).ceil() as usize).max(1);
        for spoke in 0..spokes {
            let phi = std::f64::consts::TAU * spoke as f64 / spokes as f64;
            let (sp, cp) = phi.sin_cos();
            let dir = (sun * ct + (t1 * cp + t2 * sp) * st)
                .try_normalize()
                .unwrap_or(sun);
            if dir.y.abs() > DRY_SITE_MAX_ABS_LAT_SIN {
                continue;
            }
            let Some(h) = hs.sample_height_m(dir.as_vec3(), DRY_SITE_LOD_M) else {
                continue;
            };
            if h <= DRY_SITE_MIN_HEIGHT_M {
                continue; // ocean / shoreline — want dry LAND desert
            }
            let moisture = hs.landcover_moisture(dir);
            if best.is_none_or(|(bm, _)| moisture < bm) {
                best = Some((moisture, dir));
            }
        }
    }
    best.map(|(_, d)| d).unwrap_or(sun)
}

/// Select a deterministic body-fixed surface point whose local solar
/// elevation matches the requested value. This lets the probe suite request a
/// sunset or broad daylight without changing the simulation epoch (and thus
/// without perturbing any other system that depends on canonical time).
fn cloud_site_context(
    sim: &SimulationState,
    solar: &SolarSystemState,
    height_sources: &HeightSourceRegistry,
    body_id: BodyId,
    sun_elevation_deg: f32,
) -> Option<HubContext> {
    let states = solar.states.as_deref()?;
    let body_state = states.get(body_id)?;
    let star_position = states.first()?.position;
    let sun_world = (star_position - body_state.position).normalize_or_zero();
    if sun_world == DVec3::ZERO {
        return None;
    }

    let sun_body = (body_state.orientation.inverse() * sun_world).normalize();
    let seed = if sun_body.dot(DVec3::Y).abs() < 0.99 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let across_terminator = (seed - sun_body * seed.dot(sun_body)).normalize();
    let elevation = (sun_elevation_deg as f64).to_radians();
    let up_body = (sun_body * elevation.sin() + across_terminator * elevation.cos()).normalize();
    let up_world = (body_state.orientation * up_body).normalize();

    let radius_m = sim.system.bodies.get(body_id)?.radius_m;
    let height_m = height_sources
        .get(body_id)
        .and_then(|source| source.sample_height_m(up_body.as_vec3(), 2_000.0))
        .unwrap_or(0.0) as f64;
    let pad_r = radius_m + height_m.max(0.0);
    Some(HubContext {
        body_id,
        center_world: body_state.position + up_world * pad_r,
        up_world,
        pad_r,
    })
}

/// Place a camera at an exact altitude above the probe site and aim relative
/// to the local horizon. Azimuth zero faces the projected sun; for the limb
/// preset the look angle is an offset above the geometric surface horizon.
#[allow(clippy::too_many_arguments)]
fn pose_local_cloud_camera(
    cfg: &ScreenshotConfig,
    ctx: &HubContext,
    solar: &SolarSystemState,
    root: &Grid,
    transform: &mut Transform,
    cell: &mut CellCoord,
    camera_altitude_m: f64,
    look_elevation_deg: f32,
    tangent_limb: bool,
) {
    let up = ctx.up_world;
    let sun_world = solar
        .states
        .as_deref()
        .and_then(|states| {
            let star = states.first()?;
            let body = states.get(ctx.body_id)?;
            Some((star.position - body.position).normalize_or_zero())
        })
        .unwrap_or(DVec3::Y);
    let sun_tangent = sun_world - up * sun_world.dot(up);

    let seed = if up.dot(DVec3::Y).abs() < 0.99 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let east = seed.cross(up).normalize();
    let base_heading = if sun_tangent.length_squared() > 1.0e-12 {
        sun_tangent.normalize()
    } else {
        east
    };
    let azimuth = (cfg.azimuth_deg as f64).to_radians();
    let heading = (DQuat::from_axis_angle(up, azimuth) * base_heading).normalize();

    let camera_radius = ctx.pad_r + camera_altitude_m;
    let elevation_deg = if tangent_limb {
        let horizon_dip = (ctx.pad_r / camera_radius.max(ctx.pad_r + 1.0))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        -horizon_dip + look_elevation_deg as f64
    } else {
        look_elevation_deg as f64
    };
    let elevation = elevation_deg.to_radians();
    let look_direction = (heading * elevation.cos() + up * elevation.sin()).normalize();
    let camera_world = ctx.center_world + up * camera_altitude_m;
    let look_up = if look_direction.dot(up).abs() > 0.99 {
        east
    } else {
        up
    };

    let (next_cell, local) = root.translation_to_grid(camera_world);
    *cell = next_cell;
    *transform =
        Transform::from_translation(local).looking_to(look_direction.as_vec3(), look_up.as_vec3());
}
