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
    collections::BTreeMap,
    env, fs,
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use bevy::{
    asset::{AssetEvent, RenderAssetUsages},
    camera::{ImageRenderTarget, RenderTarget},
    diagnostic::{DiagnosticPath, DiagnosticsStore},
    math::{DQuat, DVec3},
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
        view::screenshot::{Capturing, Screenshot, ScreenshotCaptured, save_to_disk},
    },
    shader::Shader,
    window::{CursorIcon, PrimaryWindow, SystemCursorIcon},
};
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_body_render::renderer_tile_lod_m_at;
use thalos_body_render::udlod::prelude::{TerrainViewComponents, TileAtlas, TileTree};
use thalos_body_render::{
    BodyTerrainMaterial, CloudsConfig, HeightSource, cloud_target_memory_for,
};
use thalos_capture_protocol::{
    CAPTURE_PROTOCOL_SCHEMA, CaptureAction, CaptureRequest, CaptureResponse, CaptureServerState,
};
use thalos_input::game::GameInputIntent;
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::camera::{ActiveCamera, ShipCamera};
use crate::loading::AppState;
use crate::rendering::ground_terrain::{BodyTerrain, OceanDebugSettings};
use crate::rendering::ssao::SsaoConfig;
use crate::rendering::{SimulationState, SolarSystemState};
use crate::space_center::{HubContext, hub_context};
use crate::spawn::{Homeworld, SpawnSituation};
use crate::structures::StructureRegistry;
use crate::terrain_registry::BodySurfaceRegistry;

const SAVED_PERSPECTIVE_SCHEMA: &str = "thalos.saved-perspective.v1";
const SAVED_PERSPECTIVE_FILENAME: &str = "latest_perspective.json";
const CAPTURE_REQUEST_FILENAME: &str = "visual_capture_request.json";
const CAPTURE_RESPONSE_FILENAME: &str = "visual_capture_response.json";
const CAPTURE_SERVER_STATE_FILENAME: &str = "visual_capture_server.json";

fn saved_perspective_path() -> PathBuf {
    crate::artifact_paths::default_diagnostic_path(SAVED_PERSPECTIVE_FILENAME)
}

// ---------------------------------------------------------------------------
// F2 window screenshot
// ---------------------------------------------------------------------------

pub struct ScreenshotPlugin;

impl Plugin for ScreenshotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (screenshot_on_f2, screenshot_cursor))
            // PostUpdate observes the final camera pose after flight, freecam,
            // hub, and editor drivers have all had their turn in Update.
            .add_systems(PostUpdate, save_perspective);
    }
}

fn screenshot_on_f2(
    mut commands: Commands,
    input: Res<GameInputIntent>,
    active_captures: Query<Entity, With<Capturing>>,
    photo_mode: Res<crate::photo_mode::PhotoMode>,
    theme: Res<thalos_ui::UiTheme>,
    toast_area: Query<Entity, With<thalos_ui::ToastArea>>,
) {
    if !input.screenshot || !active_captures.is_empty() {
        return;
    }

    let Some(dir) = screenshot_dir() else {
        warn!("could not resolve ~/Desktop/thalos for screenshot output");
        show_capture_toast(
            &mut commands,
            &photo_mode,
            &theme,
            &toast_area,
            "SCREENSHOT NOT SAVED · output folder unavailable",
            thalos_ui::ToastKind::Warn,
        );
        return;
    };

    if let Err(error) = fs::create_dir_all(&dir) {
        warn!(
            "could not create screenshot directory {}: {error}",
            dir.display()
        );
        show_capture_toast(
            &mut commands,
            &photo_mode,
            &theme,
            &toast_area,
            "SCREENSHOT NOT SAVED · could not prepare output folder",
            thalos_ui::ToastKind::Warn,
        );
        return;
    }

    let path = dir.join(format!("thalos-{}.png", timestamp_millis()));
    info!("saving screenshot to {}", path.display());
    commands.spawn(Screenshot::primary_window()).observe(
        move |screenshot_captured: On<ScreenshotCaptured>,
              mut commands: Commands,
              photo_mode: Res<crate::photo_mode::PhotoMode>,
              theme: Res<thalos_ui::UiTheme>,
              toast_area: Query<Entity, With<thalos_ui::ToastArea>>| {
            let result = screenshot_captured
                .image
                .clone()
                .try_into_dynamic()
                .map_err(|error| format!("could not encode the captured image: {error}"))
                .and_then(|dynamic_image| {
                    dynamic_image
                        .to_rgb8()
                        .save_with_format(&path, image::ImageFormat::Png)
                        .map_err(|error| format!("could not write {}: {error}", path.display()))
                });

            match result {
                Ok(()) => {
                    info!("saved screenshot to {}", path.display());
                    show_capture_toast(
                        &mut commands,
                        &photo_mode,
                        &theme,
                        &toast_area,
                        "SCREENSHOT SAVED · Desktop/thalos",
                        thalos_ui::ToastKind::Success,
                    );
                }
                Err(error) => {
                    warn!(target: "thalos::screenshot", "could not save screenshot: {error}");
                    show_capture_toast(
                        &mut commands,
                        &photo_mode,
                        &theme,
                        &toast_area,
                        format!("SCREENSHOT NOT SAVED · {error}"),
                        thalos_ui::ToastKind::Warn,
                    );
                }
            }
        },
    );
}

/// Emit capture feedback only outside F1 photo mode. The photo-mode plugin also
/// hides the shared toast area, while this prevents a toast saved during a
/// clean capture from appearing after the user leaves photo mode.
fn show_capture_toast(
    commands: &mut Commands,
    photo_mode: &crate::photo_mode::PhotoMode,
    theme: &thalos_ui::UiTheme,
    toast_area: &Query<Entity, With<thalos_ui::ToastArea>>,
    message: impl Into<String>,
    kind: thalos_ui::ToastKind,
) {
    if photo_mode.active {
        return;
    }
    let Ok(area) = toast_area.single() else {
        return;
    };
    thalos_ui::spawn_toast(commands, area, theme, message, kind);
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

/// Persist the active 3-D view for a later headless replay. The camera pose is
/// stored body-fixed, so floating-origin recentering and a fresh process's body
/// transform cannot move the saved geographic viewpoint.
#[allow(clippy::too_many_arguments)]
fn save_perspective(
    input: Res<GameInputIntent>,
    view_anchor: Res<crate::rendering::view_anchor::ViewAnchor>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    situation: Res<SpawnSituation>,
    space_center: Option<Res<crate::space_center::SpaceCenter>>,
    cameras: Query<(&CellCoord, &Transform, &Projection), (With<ShipCamera>, With<ActiveCamera>)>,
    windows: Query<&Window, With<PrimaryWindow>>,
    photo_mode: Res<crate::photo_mode::PhotoMode>,
    theme: Res<thalos_ui::UiTheme>,
    toast_area: Query<Entity, With<thalos_ui::ToastArea>>,
    mut commands: Commands,
) {
    if !input.save_perspective {
        return;
    }

    let result = (|| -> Result<SavedPerspective, String> {
        let anchor = view_anchor
            .resolved
            .ok_or_else(|| "the 3-D view is not anchored to a terrain body yet".to_string())?;
        let states = solar
            .states
            .as_deref()
            .ok_or_else(|| "the solar-system state is not ready yet".to_string())?;
        let body_state = states
            .get(anchor.body)
            .ok_or_else(|| "the view body's state is unavailable".to_string())?;
        let body = sim
            .system
            .bodies
            .get(anchor.body)
            .ok_or_else(|| "the view body is unavailable".to_string())?;
        let (cell, transform, projection) = cameras.single().map_err(|_| {
            "switch to the active 3-D ship/free camera before saving a perspective".to_string()
        })?;
        let Projection::Perspective(perspective) = projection else {
            return Err("the active 3-D camera is not perspective-projected".to_string());
        };

        let camera_world = DVec3::new(cell.x as f64, cell.y as f64, cell.z as f64)
            * crate::rendering::real_space::REAL_SPACE_CELL_SIZE_M as f64
            + transform.translation.as_dvec3();
        let camera_body = body_state.orientation.inverse() * (camera_world - body_state.position);
        let rotation_world = DQuat::from_xyzw(
            transform.rotation.x as f64,
            transform.rotation.y as f64,
            transform.rotation.z as f64,
            transform.rotation.w as f64,
        );
        let rotation_body = (body_state.orientation.inverse() * rotation_world).normalize();
        let viewport = windows
            .single()
            .map(|window| {
                [
                    window.resolution.physical_width().max(1),
                    window.resolution.physical_height().max(1),
                ]
            })
            .unwrap_or([1920, 1080]);

        Ok(SavedPerspective {
            schema: SAVED_PERSPECTIVE_SCHEMA.to_string(),
            saved_unix_ms: timestamp_millis(),
            body: body.name.clone(),
            spawn: SavedSpawn::from(*situation),
            boots_hub: space_center.as_deref().is_some_and(|hub| hub.open),
            sim_time_s: sim.simulation.sim_time(),
            camera_position_body_m: [camera_body.x, camera_body.y, camera_body.z],
            camera_rotation_body_xyzw: [
                rotation_body.x,
                rotation_body.y,
                rotation_body.z,
                rotation_body.w,
            ],
            vertical_fov_rad: perspective.fov,
            viewport,
        })
    })();

    let path = saved_perspective_path();
    match result.and_then(|saved| saved.write_to(path.clone())) {
        Ok(()) => {
            info!(
                target: "thalos::screenshot",
                "saved agent perspective to {}; replay with `just screenshot latest`",
                path.display(),
            );
            show_capture_toast(
                &mut commands,
                &photo_mode,
                &theme,
                &toast_area,
                "PERSPECTIVE SAVED · agents can run `just screenshot latest`",
                thalos_ui::ToastKind::Success,
            );
        }
        Err(error) => {
            warn!(target: "thalos::screenshot", "could not save perspective: {error}");
            show_capture_toast(
                &mut commands,
                &photo_mode,
                &theme,
                &toast_area,
                format!("PERSPECTIVE NOT SAVED · {error}"),
                thalos_ui::ToastKind::Warn,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Headless capture — config
// ---------------------------------------------------------------------------

/// The minimum boot scene needed to reconstruct the world behind a saved
/// camera. This deliberately snapshots no craft dynamics: the handoff owns a
/// perspective, while the named spawn remains the canonical scene builder.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum SavedSpawn {
    Orbit,
    Polar,
    Eva,
    Landing,
    Final,
    Runway,
    RunwayApproach,
    Cruise,
}

impl From<SpawnSituation> for SavedSpawn {
    fn from(value: SpawnSituation) -> Self {
        match value {
            SpawnSituation::ShipOrbit => Self::Orbit,
            SpawnSituation::PolarOrbit => Self::Polar,
            SpawnSituation::Eva => Self::Eva,
            SpawnSituation::Landing => Self::Landing,
            SpawnSituation::FinalApproach => Self::Final,
            SpawnSituation::Runway => Self::Runway,
            SpawnSituation::RunwayApproach => Self::RunwayApproach,
            SpawnSituation::Cruise => Self::Cruise,
        }
    }
}

impl From<SavedSpawn> for SpawnSituation {
    fn from(value: SavedSpawn) -> Self {
        match value {
            SavedSpawn::Orbit => Self::ShipOrbit,
            SavedSpawn::Polar => Self::PolarOrbit,
            SavedSpawn::Eva => Self::Eva,
            SavedSpawn::Landing => Self::Landing,
            SavedSpawn::Final => Self::FinalApproach,
            SavedSpawn::Runway => Self::Runway,
            SavedSpawn::RunwayApproach => Self::RunwayApproach,
            SavedSpawn::Cruise => Self::Cruise,
        }
    }
}

/// Versioned player-to-agent camera handoff. Position and orientation are in
/// the named body's fixed frame; headless replay projects them through that
/// body's fresh [`BodyState`](thalos_physics_canonical::types::BodyState).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SavedPerspective {
    schema: String,
    saved_unix_ms: u128,
    body: String,
    spawn: SavedSpawn,
    boots_hub: bool,
    /// Recorded for diagnosis/provenance. Replay intentionally uses the named
    /// spawn's canonical time so it never creates a half-restored craft state.
    sim_time_s: f64,
    camera_position_body_m: [f64; 3],
    camera_rotation_body_xyzw: [f64; 4],
    vertical_fov_rad: f32,
    viewport: [u32; 2],
}

impl SavedPerspective {
    fn load_from(path: PathBuf) -> Result<Self, String> {
        let bytes = fs::read(&path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let saved: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("could not parse {}: {error}", path.display()))?;
        saved.validate()?;
        Ok(saved)
    }

    fn write_to(&self, path: PathBuf) -> Result<(), String> {
        self.validate()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
        }
        let json = serde_json::to_vec_pretty(self)
            .map_err(|error| format!("could not encode perspective: {error}"))?;
        fs::write(&path, json)
            .map_err(|error| format!("could not write {}: {error}", path.display()))
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema != SAVED_PERSPECTIVE_SCHEMA {
            return Err(format!(
                "unsupported perspective schema {:?}; expected {SAVED_PERSPECTIVE_SCHEMA}",
                self.schema
            ));
        }
        if self.body.trim().is_empty() {
            return Err("saved perspective has no target body".to_string());
        }
        if !self.sim_time_s.is_finite()
            || !self
                .camera_position_body_m
                .iter()
                .all(|value| value.is_finite())
            || !self
                .camera_rotation_body_xyzw
                .iter()
                .all(|value| value.is_finite())
        {
            return Err("saved perspective contains a non-finite number".to_string());
        }
        let rotation_len_sq = self
            .camera_rotation_body_xyzw
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        if rotation_len_sq < 0.25 {
            return Err("saved perspective has an invalid camera rotation".to_string());
        }
        if !self.vertical_fov_rad.is_finite()
            || !(1.0f32.to_radians()..179.0f32.to_radians()).contains(&self.vertical_fov_rad)
        {
            return Err("saved perspective has an invalid vertical FOV".to_string());
        }
        if self.viewport[0] == 0 || self.viewport[1] == 0 {
            return Err("saved perspective has an empty viewport".to_string());
        }
        Ok(())
    }
}

/// A named framing. Each preset knows which spawn scenario the world must boot
/// into (so `main.rs` can force it) and the default camera pose + output path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenshotPreset {
    /// The active 3-D view most recently saved by the player with F8. Loads the
    /// body-fixed pose from [`saved_perspective_path`].
    LatestPerspective,
    /// A 3/4 aerial establishing shot of the surface spaceport (the runway,
    /// launchpads, tanks, hangars, and the parked aircraft). Boots the `runway`
    /// scenario, which builds the whole spaceport + settles the terrain behind
    /// the loading screen.
    SpaceportAerial,
    /// Low, near-horizontal view across the spaceport basin. This is the
    /// canonical inside-atmosphere regression probe: it exercises the surface
    /// sky, long slant-path haze, terrain recession, structures, and the real
    /// runway scenario through the same `ShipCamera` used in play.
    RunwayAtmosphere,
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
    /// A fixed ISS-like orbital view over Thalos's land near the spaceport.
    /// The 3:2 frame and high horizon mirror the Earth reference used to
    /// calibrate atmosphere thickness, aerial perspective, and exposure. This
    /// framing contains no rendered ocean. This is the orbital regression
    /// probe for the canonical custom atmosphere.
    EarthReference,
    /// Eye-level open-ocean validation shot. Selects deep water under a low sun
    /// and aligns the view near the specular path so wave scale, sun glitter,
    /// whitecaps, horizon energy, and atmospheric coupling are all visible.
    Ocean,
    /// Same framing as [`Self::Ocean`], but replaces the water BRDF with a
    /// false-colour resolved-slope / filtered-roughness diagnostic.
    OceanSlopes,
    /// Mira's cratered horizon from low orbit. Boots the canonical orbit
    /// scenario around Mira, then frames the daylight surface with enough boom
    /// distance for curvature and large impact structure to read.
    MiraOrbit,
    /// A close oblique survey of Mira regolith. This is the primary verification
    /// probe for package detail, terrain streaming, and the Hapke phase response.
    MiraSurface,
    /// Eye-level survey at the canonical `mira-eva` spawn site. Keeps live EVA
    /// terrain regressions reproducible without depending on the separate
    /// landmark-crater framing used by [`Self::MiraSurface`].
    MiraEva,
    /// Low flight over the real runway, aimed through the lower sky so broken
    /// cumulus and its relationship to the ground are both visible.
    CloudRunway,
    /// Same near-sun runway regime, but slews the camera through the final
    /// warm-up frames and captures while it is still moving. This is the
    /// deterministic temporal-reconstruction/disocclusion probe.
    CloudMotion,
    /// Camera above the cloud deck at aircraft-cruise altitude, looking across
    /// the layer toward the sun.
    CloudCruise,
    /// Camera placed inside the current 2.0–3.3 km cloud shell.
    CloudInterior,
    /// Low-orbit tangent view of the cloud line inside the atmosphere limb.
    CloudLimb,
    /// Full planetary disc from outside the terrain LOD swap — CLOUD-6 orbital
    /// impostor / weather-layer regression (SolidPlanet + atmosphere limb).
    CloudPlanet,
    /// Near-surface view toward a sun placed just above the local horizon.
    CloudSunset,
    /// Three-quarter hero shot of a firing rocket engine, for iterating on the
    /// plume look. Boots a plain orbit (space/planet backdrop), forces the engine
    /// to full throttle, and drives the plume's ambient pressure via
    /// [`PlumeDebugOverride`](crate::rendering::plume::PlumeDebugOverride) so the
    /// shock-diamond / sea-level look is reproducible regardless of the craft's
    /// real altitude. Scrub the pressure with `THALOS_PLUME_PRESSURE` (Pa).
    Plume,
}

/// Viewport-relative cloud quality ladder used by both verification captures
/// and the renderer's measured quality contract.
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
            Self::Low => 48,
            Self::Baseline => 80,
            Self::High => 96,
            Self::Reference => 128,
        }
    }

    fn shadow_steps(self) -> u32 {
        match self {
            Self::Low => 2,
            Self::Baseline => 3,
            Self::High => 4,
            Self::Reference => 6,
        }
    }

    fn resolution_scale(self) -> f32 {
        match self {
            Self::Low => 0.5,
            Self::Baseline => 2.0 / 3.0,
            // High spends its budget on range samples, not pixels. A 0.75
            // target measured 4.21 ms at 1440p on the development GPU; 2/3
            // keeps the playable tier inside the provisional 3.5 ms budget.
            Self::High => 2.0 / 3.0,
            Self::Reference => 1.0,
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
        /// Aim at the body centre for a full planetary disc (impostor range).
        /// Local-horizon AGL shots leave this false.
        look_at_body_center: bool,
    },
}

impl ScreenshotPreset {
    fn name(self) -> &'static str {
        match self {
            Self::LatestPerspective => "latest-perspective",
            Self::SpaceportAerial => "spaceport-aerial",
            Self::RunwayAtmosphere => "runway-atmosphere",
            Self::Hub => "hub",
            Self::DryBelt => "dry-belt",
            Self::EarthReference => "earth-reference",
            Self::Ocean => "ocean",
            Self::OceanSlopes => "ocean-slopes",
            Self::MiraOrbit => "mira-orbit",
            Self::MiraSurface => "mira-surface",
            Self::MiraEva => "mira-eva",
            Self::CloudRunway => "cloud-runway",
            Self::CloudMotion => "cloud-motion",
            Self::CloudCruise => "cloud-cruise",
            Self::CloudInterior => "cloud-interior",
            Self::CloudLimb => "cloud-limb",
            Self::CloudPlanet => "cloud-planet",
            Self::CloudSunset => "cloud-sunset",
            Self::Plume => "plume",
        }
    }

    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "latest" | "perspective" | "latest-perspective" | "latest_perspective" => {
                Self::LatestPerspective
            }
            // Truthy / unnamed → the default preset.
            "" | "1" | "true" | "yes" | "on" | "spaceport" | "spaceport-aerial" | "aerial"
            | "base" => Self::SpaceportAerial,
            "runway-atmosphere" | "runway_atmosphere" | "runway-sky" | "surface-atmosphere" => {
                Self::RunwayAtmosphere
            }
            "hub" | "space-center" | "spacecenter" | "play" => Self::Hub,
            "dry" | "dry-belt" | "drybelt" | "desert" | "biome" => Self::DryBelt,
            "earth-reference" | "earth_reference" | "earth-ref" | "atmosphere" | "atmo" => {
                Self::EarthReference
            }
            "ocean" | "open-ocean" | "open_ocean" | "sea" | "water" => Self::Ocean,
            "ocean-slopes" | "ocean_slopes" | "sea-slopes" | "slope-field" => Self::OceanSlopes,
            "mira" | "mira-orbit" | "mira_orbit" => Self::MiraOrbit,
            "mira-surface" | "mira_surface" | "regolith" => Self::MiraSurface,
            "mira-eva" | "mira_eva" | "regolith-eva" => Self::MiraEva,
            "cloud-runway" | "cloud_runway" | "clouds-runway" => Self::CloudRunway,
            "cloud-motion" | "cloud_motion" | "clouds-motion" => Self::CloudMotion,
            "cloud-cruise" | "cloud_cruise" | "clouds-cruise" | "cloud-deck" => Self::CloudCruise,
            "cloud-interior" | "cloud_interior" | "inside-cloud" | "inside-clouds" => {
                Self::CloudInterior
            }
            "cloud-limb" | "cloud_limb" | "cloud-orbit" | "clouds-orbit" => Self::CloudLimb,
            "cloud-planet" | "cloud_planet" | "cloud-globe" | "cloud_globe" | "cloud-disc"
            | "cloud_disc" | "full-planet" | "planet-disc" => Self::CloudPlanet,
            "cloud-sunset" | "cloud_sunset" | "clouds-sunset" => Self::CloudSunset,
            "plume" | "engine" | "exhaust" | "rocket" => Self::Plume,
            other => {
                eprintln!("  Unknown THALOS_SCREENSHOT preset '{other}'; using spaceport-aerial.");
                Self::SpaceportAerial
            }
        }
    }

    /// The scenario the world must be booted into for this preset.
    pub fn spawn_situation(self) -> SpawnSituation {
        match self {
            // The loaded handoff overrides this placeholder through
            // `ScreenshotConfig::spawn_situation`.
            Self::LatestPerspective => SpawnSituation::ShipOrbit,
            Self::SpaceportAerial
            | Self::RunwayAtmosphere
            | Self::CloudRunway
            | Self::CloudMotion => SpawnSituation::Runway,
            // The hub is the PLAY path: the placeholder parking orbit plus the
            // spaceport build (armed by `main.rs` via `boots_hub`).
            Self::Hub => SpawnSituation::ShipOrbit,
            // Dry-belt frames wild terrain far from any base, so a plain orbit
            // scenario is enough; the driver poses the camera over the searched
            // desert site (the craft stays in orbit, irrelevant to the framing).
            Self::DryBelt | Self::Ocean | Self::OceanSlopes => SpawnSituation::ShipOrbit,
            Self::EarthReference => SpawnSituation::Runway,
            Self::MiraOrbit | Self::MiraSurface => SpawnSituation::ShipOrbit,
            Self::MiraEva => SpawnSituation::Eva,
            Self::CloudCruise
            | Self::CloudInterior
            | Self::CloudLimb
            | Self::CloudPlanet
            | Self::CloudSunset => SpawnSituation::ShipOrbit,
            // Plain orbit: space/planet backdrop, engine forced to fire.
            Self::Plume => SpawnSituation::ShipOrbit,
        }
    }

    /// Body that owns the world and terrain framed by this preset.
    pub fn target_body_name(self) -> &'static str {
        match self {
            // The loaded handoff overrides this placeholder through
            // `ScreenshotConfig::target_body_name`.
            Self::LatestPerspective => "Thalos",
            Self::MiraOrbit | Self::MiraSurface | Self::MiraEva => "Mira",
            _ => "Thalos",
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
            Self::LatestPerspective => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/latest_perspective.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 0.0,
                elevation_deg: 45.0,
                distance_m: 1_000.0,
                // A handoff can point anywhere on a cold planet. Give terrain,
                // scatter, clouds, and render pipelines time to converge.
                warmup_frames: 600,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("latest_perspective.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::SpaceportAerial => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/spaceport_aerial.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 35.0,
                elevation_deg: 42.0,
                distance_m: 4200.0,
                warmup_frames: 180,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("spaceport_aerial.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::RunwayAtmosphere => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/runway_atmosphere.png"),
                width: 1920,
                height: 1080,
                // A 1.4 km boom at 3 degrees is ~73 m above the flattened
                // spaceport basin: low enough for the sky and long air column
                // to dominate, high enough to keep the camera above structures.
                azimuth_deg: 270.0,
                elevation_deg: 3.0,
                distance_m: 1_400.0,
                warmup_frames: 480,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("runway_atmosphere.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            // Matches the hub's establishing view (`BASE_ESTABLISHING_DISTANCE_M`,
            // the one god-view framing per base) so the capture shows what PLAY shows.
            Self::Hub => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/hub.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 35.0,
                elevation_deg: 42.0,
                distance_m: crate::god_view::BASE_ESTABLISHING_DISTANCE_M as f64,
                warmup_frames: 240,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("hub.jsonl"),
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
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/dry_belt.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 35.0,
                elevation_deg: 16.0,
                distance_m: 1400.0,
                warmup_frames: 600,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("dry_belt.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::CloudRunway => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/cloud_runway.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 30.0,
                elevation_deg: 8.0,
                distance_m: 4200.0,
                warmup_frames: 300,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("cloud_runway.jsonl"),
                framing: ScreenshotFraming::LocalCloud {
                    // Human-scale weather acceptance view: just above the
                    // runway, pitched into the cloud-bearing sky. The old
                    // 850 m / -8° pose was an aerial terrain survey despite
                    // the preset's name and hid the scene-scale impression.
                    camera_altitude_m: 35.0,
                    look_elevation_deg: 9.0,
                    site_sun_elevation_deg: None,
                    tangent_limb: false,
                    look_at_body_center: false,
                },
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::CloudMotion => {
                let mut cfg = Self::CloudRunway.defaults();
                cfg.preset = self;
                cfg.out = PathBuf::from("artifacts/visual/latest/cloud_motion.png");
                cfg.report = crate::artifact_paths::default_jsonl_path("cloud_motion.jsonl");
                // End directly toward the projected sun, where forward-scatter
                // contrast makes stale history easiest to see.
                cfg.azimuth_deg = 0.0;
                cfg.warmup_frames = 320;
                cfg
            }
            Self::CloudCruise => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/cloud_cruise.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 20.0,
                elevation_deg: 8.0,
                distance_m: 4200.0,
                warmup_frames: 360,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("cloud_cruise.jsonl"),
                framing: ScreenshotFraming::LocalCloud {
                    // Above the ordinary cumulus deck but below the tallest
                    // authored storm tops: the aviation-scale view across
                    // cloud tops and developing towers. Interior has its own
                    // intentionally dense probe below.
                    camera_altitude_m: 9_000.0,
                    look_elevation_deg: -5.0,
                    site_sun_elevation_deg: Some(35.0),
                    tangent_limb: false,
                    look_at_body_center: false,
                },
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::CloudInterior => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/cloud_interior.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 70.0,
                elevation_deg: 8.0,
                distance_m: 4200.0,
                warmup_frames: 360,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("cloud_interior.jsonl"),
                framing: ScreenshotFraming::LocalCloud {
                    camera_altitude_m: 2_650.0,
                    look_elevation_deg: 0.0,
                    site_sun_elevation_deg: Some(35.0),
                    tangent_limb: false,
                    look_at_body_center: false,
                },
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                // Force the local weather threshold into a dense cell so this
                // diagnostic reliably exercises traversal from inside cloud.
                cloud_coverage_scale: Some(1.60),
            },
            Self::CloudLimb => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/cloud_limb.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 0.0,
                elevation_deg: 8.0,
                distance_m: 4200.0,
                warmup_frames: 360,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("cloud_limb.jsonl"),
                framing: ScreenshotFraming::LocalCloud {
                    camera_altitude_m: 200_000.0,
                    look_elevation_deg: 0.35,
                    // Keep the low-orbit limb readable without letting a
                    // horizon-grazing ocean glint dominate the cloud probe.
                    site_sun_elevation_deg: Some(12.0),
                    tangent_limb: true,
                    look_at_body_center: false,
                },
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::CloudPlanet => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/cloud_planet.png"),
                width: 1920,
                height: 1080,
                // Small azimuth rotates the view around the sub-camera radial
                // for compositional variety without losing the full disc.
                azimuth_deg: 18.0,
                elevation_deg: 8.0,
                distance_m: 4200.0,
                // Impostor materials + weather cube only — no surface settle.
                warmup_frames: 240,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("cloud_planet.jsonl"),
                framing: ScreenshotFraming::LocalCloud {
                    // Outside the terrain LOD swap (4× radius ≈ 12.7 Mm for
                    // Thalos): SolidPlanet owns the disc so the CLOUD-6 orbital
                    // weather projection is what we see.
                    camera_altitude_m: 14_000_000.0,
                    look_elevation_deg: -90.0,
                    // Mid-afternoon phase so clouds cast readable shadows and
                    // the terminator is visible without a razor-thin crescent.
                    site_sun_elevation_deg: Some(42.0),
                    tangent_limb: false,
                    look_at_body_center: true,
                },
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::CloudSunset => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/cloud_sunset.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 0.0,
                elevation_deg: 8.0,
                distance_m: 4200.0,
                warmup_frames: 360,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("cloud_sunset.jsonl"),
                framing: ScreenshotFraming::LocalCloud {
                    camera_altitude_m: 700.0,
                    look_elevation_deg: 1.5,
                    site_sun_elevation_deg: Some(1.0),
                    tangent_limb: false,
                    look_at_body_center: false,
                },
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            // Three-quarter side view of the craft, framed on the engine + plume.
            // GodView around a craft-centered focus (see `craft_context`): up =
            // the ship's nose axis, so elevation is height above the "waist" and
            // azimuth swings around the stack.
            Self::Plume => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/plume.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 55.0,
                elevation_deg: 6.0,
                distance_m: 42.0,
                // Orbit scenario builds fast; the plume just needs the ship + a
                // few frames for the ignition transient to settle to full.
                warmup_frames: 90,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("plume.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::EarthReference => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/earth_reference.png"),
                // The Earth reference is 1000×667 (approximately 3:2). Keep
                // that composition instead of comparing a wider 16:9 crop.
                width: 1800,
                height: 1200,
                azimuth_deg: 270.0,
                elevation_deg: 34.0,
                distance_m: 500_000.0,
                warmup_frames: 480,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("earth_reference.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::Ocean => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/ocean.png"),
                width: 1920,
                height: 1080,
                // The ocean-site search chooses a point whose low sun lies
                // opposite local east; this small offset keeps the glitter
                // road off-centre so both lit and unlit slopes stay legible.
                azimuth_deg: 22.0,
                elevation_deg: 1.5,
                distance_m: 600.0,
                // A cold wild-ocean site still needs terrain tiles for the
                // signed sea-field mask and the cloud history needs to settle.
                warmup_frames: 600,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("ocean.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::OceanSlopes => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/ocean_slopes.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 22.0,
                elevation_deg: 1.5,
                distance_m: 600.0,
                warmup_frames: 600,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("ocean_slopes.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::MiraOrbit => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/mira_orbit.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 28.0,
                elevation_deg: 24.0,
                distance_m: 360_000.0,
                // A new package content key intentionally cold-misses the tile
                // cache. Leave enough time for orbital ancestors + detail to
                // populate on the first verification run, not only a warm run.
                warmup_frames: 720,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("mira_orbit.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::MiraSurface => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/mira_surface.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 32.0,
                elevation_deg: 34.0,
                distance_m: 46_000.0,
                // The close crater view is the heaviest cold package probe.
                // Disk-cache-disabled captures converged their final UDLOD
                // fallback boundaries at 1,200 frames (900 was still early).
                warmup_frames: 1_200,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("mira_surface.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::MiraEva => ScreenshotConfig {
                preset: self,
                saved: None,
                out: PathBuf::from("artifacts/visual/latest/mira_eva.png"),
                width: 2048,
                height: 1280,
                azimuth_deg: 32.0,
                // EVA uses tangent-look semantics rather than the aerial
                // presets' orbit boom: elevation is camera pitch and distance
                // is eye height over the exact surface focus.
                elevation_deg: 3.0,
                distance_m: 1.7,
                warmup_frames: 1_200,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("mira_eva.jsonl"),
                framing: ScreenshotFraming::GodView,
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
    /// Present only for the F8 player handoff. Scripted presets keep this
    /// `None` and continue through their typed framing paths.
    saved: Option<SavedPerspective>,
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
    ///   `latest` / `perspective` replays the F8 handoff at
    ///   `artifacts/diagnostics/latest_perspective.json`.
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
    /// - `THALOS_SCREENSHOT_CLOUD_RECONSTRUCTION` — raw, dense, or sparse;
    ///   capture-only diagnostic override used by the moving-cloud A/B.
    /// - `THALOS_SCREENSHOT_CLOUD_COVERAGE` — optional global coverage scale.
    /// - `THALOS_SCREENSHOT_REPORT` — JSONL report path (defaults under
    ///   `artifacts/diagnostics/`).
    /// - `THALOS_SCREENSHOT_OCEAN_TIME` — optional fixed canonical ocean time
    ///   in seconds (ocean diagnostics/phase comparisons only).
    pub fn from_env() -> Option<Self> {
        let raw = env::var("THALOS_SCREENSHOT").ok()?;
        let preset = ScreenshotPreset::parse(&raw);
        let mut cfg = Self::for_preset(preset);
        let overrides = CAPTURE_OVERRIDE_KEYS
            .iter()
            .filter_map(|key| env::var(key).ok().map(|value| ((*key).to_owned(), value)))
            .collect();
        cfg.apply_overrides(&overrides);
        Some(cfg)
    }

    fn for_preset(preset: ScreenshotPreset) -> Self {
        let mut cfg = preset.defaults();
        if preset == ScreenshotPreset::LatestPerspective {
            let saved = SavedPerspective::load_from(saved_perspective_path())
                .unwrap_or_else(|error| {
                    panic!(
                        "could not load the latest saved perspective: {error}. Press F8 in the game first."
                    )
                });
            cfg.width = saved.viewport[0];
            cfg.height = saved.viewport[1];
            cfg.saved = Some(saved);
        }
        cfg
    }

    fn apply_overrides(&mut self, overrides: &BTreeMap<String, String>) {
        if let Some(out) = overrides.get("THALOS_SCREENSHOT_OUT") {
            self.out = PathBuf::from(out);
        }
        if let Some(report) = overrides.get("THALOS_SCREENSHOT_REPORT") {
            self.report = PathBuf::from(report);
        }
        if let Some((w, h)) = overrides
            .get("THALOS_SCREENSHOT_SIZE")
            .and_then(|value| parse_size(value))
        {
            self.width = w;
            self.height = h;
        }
        if let Some(v) = override_parse::<f32>(overrides, "THALOS_SCREENSHOT_AZIMUTH") {
            self.azimuth_deg = v;
        }
        if let Some(v) = override_parse::<f32>(overrides, "THALOS_SCREENSHOT_ELEVATION") {
            self.elevation_deg = v.clamp(1.0, 90.0);
        }
        if let Some(v) = override_parse::<f64>(overrides, "THALOS_SCREENSHOT_DISTANCE") {
            self.distance_m = v.max(1.0);
        }
        if let Some(v) = override_parse::<u32>(overrides, "THALOS_SCREENSHOT_WARMUP") {
            self.warmup_frames = v;
        }
        if let Some(v) = overrides.get("THALOS_SCREENSHOT_HUD") {
            self.keep_hud = matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }
        if let Some(v) = override_parse::<f64>(overrides, "THALOS_SCREENSHOT_CAMERA_ALTITUDE") {
            if let ScreenshotFraming::LocalCloud {
                camera_altitude_m, ..
            } = &mut self.framing
            {
                *camera_altitude_m = v.max(0.0);
            }
        }
        if let Some(v) = override_parse::<f32>(overrides, "THALOS_SCREENSHOT_LOOK_ELEVATION") {
            if let ScreenshotFraming::LocalCloud {
                look_elevation_deg, ..
            } = &mut self.framing
            {
                *look_elevation_deg = v.clamp(-90.0, 90.0);
            }
        }
        if let Some(v) = override_parse::<f32>(overrides, "THALOS_SCREENSHOT_SUN_ELEVATION") {
            if let ScreenshotFraming::LocalCloud {
                site_sun_elevation_deg,
                ..
            } = &mut self.framing
            {
                *site_sun_elevation_deg = Some(v.clamp(-10.0, 90.0));
            }
        }
        if let Some(raw) = overrides.get("THALOS_SCREENSHOT_CLOUD_QUALITY") {
            if let Some(quality) = CloudCaptureQuality::parse(&raw) {
                self.cloud_quality = quality;
            } else {
                eprintln!(
                    "  Unknown THALOS_SCREENSHOT_CLOUD_QUALITY={raw:?}; using {}.",
                    self.cloud_quality.name()
                );
            }
        }
        if let Some(raw) = overrides.get("THALOS_SCREENSHOT_CLOUD_TEMPORAL") {
            match parse_bool(&raw) {
                Some(value) => self.cloud_temporal = value,
                None => eprintln!(
                    "  Unknown THALOS_SCREENSHOT_CLOUD_TEMPORAL={raw:?}; expected on/off."
                ),
            }
        }
        if let Some(v) = override_parse::<f32>(overrides, "THALOS_SCREENSHOT_CLOUD_COVERAGE") {
            self.cloud_coverage_scale = Some(v.clamp(0.0, 4.0));
        }
    }

    /// Body whose authored world must be loaded before the capture app boots.
    pub fn target_body_name(&self) -> &str {
        self.saved
            .as_ref()
            .map(|saved| saved.body.as_str())
            .unwrap_or_else(|| self.preset.target_body_name())
    }

    /// Canonical scene builder to run behind this capture.
    pub fn spawn_situation(&self) -> SpawnSituation {
        self.saved
            .as_ref()
            .map(|saved| SpawnSituation::from(saved.spawn))
            .unwrap_or_else(|| self.preset.spawn_situation())
    }

    /// Whether the boot should build the no-craft space-center hub route.
    pub fn boots_hub(&self) -> bool {
        self.saved
            .as_ref()
            .map(|saved| saved.boots_hub)
            .unwrap_or_else(|| self.preset.boots_hub())
    }
}

const CAPTURE_OVERRIDE_KEYS: &[&str] = &[
    "THALOS_SCREENSHOT_OUT",
    "THALOS_SCREENSHOT_REPORT",
    "THALOS_SCREENSHOT_SIZE",
    "THALOS_SCREENSHOT_AZIMUTH",
    "THALOS_SCREENSHOT_ELEVATION",
    "THALOS_SCREENSHOT_DISTANCE",
    "THALOS_SCREENSHOT_WARMUP",
    "THALOS_SCREENSHOT_HUD",
    "THALOS_SCREENSHOT_CAMERA_ALTITUDE",
    "THALOS_SCREENSHOT_LOOK_ELEVATION",
    "THALOS_SCREENSHOT_SUN_ELEVATION",
    "THALOS_SCREENSHOT_CLOUD_QUALITY",
    "THALOS_SCREENSHOT_CLOUD_TEMPORAL",
    "THALOS_SCREENSHOT_CLOUD_COVERAGE",
    "THALOS_SCREENSHOT_CLOUD_RECONSTRUCTION",
    "THALOS_SCREENSHOT_OCEAN_TIME",
    "THALOS_SSAO",
    "THALOS_TERRAIN_INSPECTION",
    "THALOS_TERRAIN_CULL",
];

#[derive(Resource, Clone, Debug, Default)]
struct CaptureRuntimeOverrides {
    values: BTreeMap<String, String>,
}

impl CaptureRuntimeOverrides {
    fn from_env() -> Self {
        Self {
            values: CAPTURE_OVERRIDE_KEYS
                .iter()
                .filter_map(|key| env::var(key).ok().map(|value| ((*key).to_owned(), value)))
                .collect(),
        }
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
}

fn override_parse<T: std::str::FromStr>(
    overrides: &BTreeMap<String, String>,
    key: &str,
) -> Option<T> {
    overrides
        .get(key)
        .and_then(|value| value.trim().parse::<T>().ok())
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

fn capture_request_path() -> PathBuf {
    crate::artifact_paths::default_diagnostic_path(CAPTURE_REQUEST_FILENAME)
}

fn capture_response_path() -> PathBuf {
    crate::artifact_paths::default_diagnostic_path(CAPTURE_RESPONSE_FILENAME)
}

fn capture_server_state_path() -> PathBuf {
    crate::artifact_paths::default_diagnostic_path(CAPTURE_SERVER_STATE_FILENAME)
}

#[derive(Resource, Debug, Default)]
struct PersistentCaptureServer {
    boot_preset: Option<ScreenshotPreset>,
    boot_width: u32,
    boot_height: u32,
    active_id: Option<String>,
    last_request_id: Option<String>,
    completed_captures: u64,
    code_reload_unix_ms: u128,
    shader_reload_unix_ms: u128,
    settle_frames: u32,
    heartbeat_frame: u32,
}

impl PersistentCaptureServer {
    fn publish(&self, ready: bool) {
        let Some(preset) = self.boot_preset else {
            return;
        };
        let state = CaptureServerState {
            schema_version: CAPTURE_PROTOCOL_SCHEMA,
            pid: std::process::id(),
            preset: preset.name().to_owned(),
            width: self.boot_width,
            height: self.boot_height,
            ready,
            busy: self.active_id.is_some(),
            completed_captures: self.completed_captures,
            code_reload_unix_ms: self.code_reload_unix_ms,
            shader_reload_unix_ms: self.shader_reload_unix_ms,
            heartbeat_unix_ms: timestamp_millis(),
        };
        if let Err(error) = write_json(capture_server_state_path(), &state) {
            warn!(target: "thalos::screenshot", "could not publish capture-server state: {error}");
        }
    }

    fn respond(&self, id: &str, ok: bool, message: impl Into<String>, output: Option<&PathBuf>) {
        let response = CaptureResponse {
            schema_version: CAPTURE_PROTOCOL_SCHEMA,
            id: id.to_owned(),
            ok,
            message: message.into(),
            output: output.map(|path| path.display().to_string()),
            completed_unix_ms: timestamp_millis(),
        };
        if let Err(error) = write_json(capture_response_path(), &response) {
            warn!(target: "thalos::screenshot", "could not publish capture response: {error}");
        }
    }
}

fn write_json(path: PathBuf, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    fs::write(&path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
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
    /// Cached deep-water direction for the low-sun open-ocean preset.
    ocean_site_dir: Option<DVec3>,
    /// Cached rugged, obliquely lit Mira site for both airless presets.
    airless_site_dir: Option<DVec3>,
}

pub struct HeadlessScreenshotPlugin {
    pub persistent: bool,
}

impl Plugin for HeadlessScreenshotPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScreenshotDriver>()
            .insert_resource(CaptureRuntimeOverrides::from_env())
            .add_systems(Startup, setup_screenshot_target)
            .add_systems(
                Update,
                (
                    retarget_ship_camera,
                    hide_overlays,
                    apply_live_capture_diagnostics,
                    configure_plume_capture,
                ),
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

        if self.persistent {
            app.init_resource::<PersistentCaptureServer>()
                .add_systems(Startup, initialize_capture_server)
                .add_systems(
                    Update,
                    (
                        poll_capture_requests.before(apply_live_capture_diagnostics),
                        publish_capture_server_state,
                    )
                        .chain()
                        .before(drive_headless_screenshot),
                )
                .add_systems(Update, record_shader_reloads);
            #[cfg(feature = "dev-iteration")]
            app.add_systems(Update, record_code_hotpatches);
        }
    }
}

fn record_shader_reloads(
    mut events: MessageReader<AssetEvent<Shader>>,
    mut server: ResMut<PersistentCaptureServer>,
) {
    if events.read().next().is_some() {
        server.shader_reload_unix_ms = timestamp_millis();
    }
}

#[cfg(feature = "dev-iteration")]
fn record_code_hotpatches(
    mut events: MessageReader<bevy::ecs::HotPatched>,
    mut server: ResMut<PersistentCaptureServer>,
) {
    if events.read().next().is_some() {
        server.code_reload_unix_ms = timestamp_millis();
        info!(target: "thalos::screenshot", "Rust hot patch applied");
    }
}

fn initialize_capture_server(
    cfg: Res<ScreenshotConfig>,
    mut server: ResMut<PersistentCaptureServer>,
) {
    server.boot_preset = Some(cfg.preset);
    server.boot_width = cfg.width;
    server.boot_height = cfg.height;
    server.settle_frames = env_parse("THALOS_CAPTURE_SETTLE_FRAMES").unwrap_or(60);
    server.publish(false);
}

fn publish_capture_server_state(
    driver: Res<ScreenshotDriver>,
    mut server: ResMut<PersistentCaptureServer>,
) {
    server.heartbeat_frame = server.heartbeat_frame.wrapping_add(1);
    if server.heartbeat_frame % 30 == 0 || server.heartbeat_frame == 1 {
        server.publish(driver.retargeted);
    }
}

fn poll_capture_requests(
    mut cfg: ResMut<ScreenshotConfig>,
    mut runtime: ResMut<CaptureRuntimeOverrides>,
    mut driver: ResMut<ScreenshotDriver>,
    mut server: ResMut<PersistentCaptureServer>,
    mut exit: MessageWriter<AppExit>,
) {
    let Ok(bytes) = fs::read(capture_request_path()) else {
        return;
    };
    let Ok(request) = serde_json::from_slice::<CaptureRequest>(&bytes) else {
        // The controller writes through an atomic rename, but tolerate a
        // partially-written request from a manually-authored client.
        return;
    };
    if server.last_request_id.as_deref() == Some(request.id.as_str()) {
        return;
    }
    server.last_request_id = Some(request.id.clone());

    if request.schema_version != CAPTURE_PROTOCOL_SCHEMA {
        server.respond(
            &request.id,
            false,
            format!(
                "capture protocol {} is unsupported; server expects {}",
                request.schema_version, CAPTURE_PROTOCOL_SCHEMA
            ),
            None,
        );
        return;
    }
    if request.action == CaptureAction::Shutdown {
        server.respond(&request.id, true, "capture server shutting down", None);
        exit.write(AppExit::Success);
        return;
    }
    if request.action != CaptureAction::Capture {
        server.respond(&request.id, false, "unknown capture action", None);
        return;
    }
    if server.active_id.is_some() {
        server.respond(&request.id, false, "capture server is already busy", None);
        return;
    }

    let requested_preset = ScreenshotPreset::parse(&request.preset);
    if Some(requested_preset) != server.boot_preset {
        server.respond(
            &request.id,
            false,
            format!(
                "server booted for {}; restart it for {}",
                server.boot_preset.map_or("unknown", ScreenshotPreset::name),
                requested_preset.name()
            ),
            None,
        );
        return;
    }

    let mut next = ScreenshotConfig::for_preset(requested_preset);
    next.apply_overrides(&request.overrides);
    if next.width != server.boot_width || next.height != server.boot_height {
        server.respond(
            &request.id,
            false,
            format!(
                "server target is {}x{}; restart it for {}x{}",
                server.boot_width, server.boot_height, next.width, next.height
            ),
            None,
        );
        return;
    }
    if server.completed_captures > 0 && !request.overrides.contains_key("THALOS_SCREENSHOT_WARMUP")
    {
        next.warmup_frames = server.settle_frames;
    }

    runtime.values = request.overrides;
    *cfg = next;
    driver.running_frames = 0;
    driver.captured = false;
    driver.tail = 0;
    server.active_id = Some(request.id);
    server.publish(driver.retargeted);
}

/// Apply capture-only cloud quality controls once, after the cloud plugin's
/// startup initialization. The normal game never sees these overrides because
/// this system only exists in [`HeadlessScreenshotPlugin`].
fn apply_live_capture_diagnostics(
    cfg: Res<ScreenshotConfig>,
    runtime: Res<CaptureRuntimeOverrides>,
    mut clouds: ResMut<CloudsConfig>,
    mut ocean_debug: ResMut<OceanDebugSettings>,
    mut ssao: ResMut<SsaoConfig>,
    mut terrain_materials: ResMut<Assets<BodyTerrainMaterial>>,
) {
    if !cfg.is_changed() && !runtime.is_changed() {
        return;
    }
    clouds.clouds_raymarch_steps_count = cfg.cloud_quality.view_steps();
    clouds.clouds_shadow_raymarch_steps_count = cfg.cloud_quality.shadow_steps();
    let reference = cfg.cloud_quality == CloudCaptureQuality::Reference;
    let default_mode = if cfg.cloud_temporal && !reference {
        "sparse"
    } else {
        "raw"
    };
    let reconstruction = runtime
        .get("THALOS_SCREENSHOT_CLOUD_RECONSTRUCTION")
        .map(|raw| raw.trim().to_ascii_lowercase())
        .unwrap_or_else(|| default_mode.to_owned());
    let (reprojection_strength, sparse_march) = match reconstruction.as_str() {
        "raw" | "off" | "none" => (0.0, false),
        "dense" | "dense-history" | "history" => (0.95, false),
        "sparse" | "sparse-history" | "default" => (0.95, true),
        other => {
            warn!(
                target: "thalos::screenshot",
                "unknown THALOS_SCREENSHOT_CLOUD_RECONSTRUCTION={other:?}; using {default_mode}"
            );
            if default_mode == "sparse" {
                (0.95, true)
            } else {
                (0.0, false)
            }
        }
    };
    clouds.reprojection_strength = reprojection_strength;
    clouds.resolution_scale = cfg.cloud_quality.resolution_scale();
    clouds.sparse_march = sparse_march;
    if let Some(coverage) = cfg.cloud_coverage_scale {
        clouds.clouds_coverage = coverage;
    }
    clouds.history_epoch = clouds.history_epoch.wrapping_add(1).max(1);

    ocean_debug.slope_view = matches!(cfg.preset, ScreenshotPreset::OceanSlopes);
    ocean_debug.phase_time_override_s = runtime
        .get("THALOS_SCREENSHOT_OCEAN_TIME")
        .and_then(|value| value.trim().parse().ok());

    ssao.apply_capture_mode(runtime.get("THALOS_SSAO"));
    let inspection = terrain_inspection_override(runtime.get("THALOS_TERRAIN_INSPECTION"));
    for (_, material) in terrain_materials.iter_mut() {
        material.extras.inspection.x = inspection;
    }
    info!(
        target: "thalos::screenshot",
        "cloud probe: quality={} view_steps={} shadow_steps={} reconstruction={} coverage={:.2}",
        cfg.cloud_quality.name(),
        clouds.clouds_raymarch_steps_count,
        clouds.clouds_shadow_raymarch_steps_count,
        reconstruction,
        clouds.clouds_coverage,
    );
}

fn terrain_inspection_override(raw: Option<&str>) -> f32 {
    match raw.unwrap_or_default().trim().to_ascii_lowercase().as_str() {
        "" | "lit" | "default" | "off" => 0.0,
        "fullbright" | "albedo" | "on" => 1.0,
        "geo-normal" | "geometric-normal" | "smooth-normal" => 2.0,
        "legacy-regolith" | "unfiltered-regolith" => 3.0,
        other => {
            warn!(target: "thalos::screenshot", "unknown terrain inspection {other:?}; using lit");
            0.0
        }
    }
}

/// Force the engine to fire and pin the plume's ambient pressure for the plume
/// preset (headless-only). Full throttle + a chosen back-pressure make the
/// shock-diamond look reproducible regardless of the craft's real orbit altitude
/// (which would otherwise give a vacuum plume). `THALOS_PLUME_PRESSURE` (Pa)
/// scrubs the pressure — 101325 (default) is sea level, lower values thin the
/// shocks toward the vacuum look.
fn configure_plume_capture(
    cfg: Res<ScreenshotConfig>,
    mut over: ResMut<crate::rendering::plume::PlumeDebugOverride>,
    mut applied: Local<bool>,
) {
    if *applied || cfg.preset != ScreenshotPreset::Plume {
        return;
    }
    let pressure = env_parse::<f32>("THALOS_PLUME_PRESSURE").unwrap_or(101_325.0);
    over.throttle = Some(1.0);
    over.ambient_pressure_pa = Some(pressure.max(0.0));
    over.ignition = Some(1.0);
    info!(
        target: "thalos::screenshot",
        "plume probe: throttle=1.0 ambient_pressure={pressure:.0} Pa"
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
    server: Option<Res<PersistentCaptureServer>>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    height_sources: Res<HeightSourceRegistry>,
    registry: Res<StructureRegistry>,
    surfaces: Res<BodySurfaceRegistry>,
    homeworld: Res<Homeworld>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut camera: Query<(&mut Transform, &mut CellCoord, &mut Projection), With<ShipCamera>>,
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
    let persistent = server.is_some();
    if server
        .as_ref()
        .is_some_and(|server| server.active_id.is_none())
    {
        return;
    }

    // Resolve the focus and pose the camera. If anything is not ready yet, hold
    // the frame counter so warmup only starts once the requested site is framed.
    let ctx = match cfg.preset {
        ScreenshotPreset::DryBelt => dry_site_context(
            &sim,
            &solar,
            &height_sources,
            homeworld.0,
            &mut driver.dry_site_dir,
        ),
        ScreenshotPreset::Ocean | ScreenshotPreset::OceanSlopes => ocean_site_context(
            &sim,
            &solar,
            &height_sources,
            homeworld.0,
            &mut driver.ocean_site_dir,
        ),
        ScreenshotPreset::MiraOrbit | ScreenshotPreset::MiraSurface => daylight_surface_context(
            &sim,
            &solar,
            &height_sources,
            homeworld.0,
            &surfaces,
            &mut driver.airless_site_dir,
        ),
        ScreenshotPreset::MiraEva => {
            eva_surface_context(&sim, &solar, &height_sources, homeworld.0)
        }
        // Frame the craft itself, not a surface site.
        ScreenshotPreset::Plume => craft_context(&sim),
        _ => match cfg.framing {
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
        },
    };
    let Some(ctx) = ctx else {
        return;
    };
    let Ok(root) = root_grid.single() else {
        return;
    };
    let Ok((mut transform, mut cell, mut projection)) = camera.single_mut() else {
        return;
    };
    let mut motion_cfg = None;
    if cfg.preset == ScreenshotPreset::CloudMotion {
        let mut posed = (*cfg).clone();
        posed.azimuth_deg += cloud_motion_azimuth_offset(driver.running_frames, cfg.warmup_frames);
        motion_cfg = Some(posed);
    }
    let pose_cfg = motion_cfg.as_ref().unwrap_or(&cfg);

    if let Some(saved) = pose_cfg.saved.as_ref() {
        if !pose_saved_perspective(
            saved,
            homeworld.0,
            &solar,
            root,
            &mut transform,
            &mut cell,
            &mut projection,
        ) {
            return;
        }
    } else if pose_cfg.preset == ScreenshotPreset::MiraEva {
        pose_eva_camera(pose_cfg, &ctx, root, &mut transform, &mut cell);
    } else {
        pose_camera(pose_cfg, &ctx, &solar, root, &mut transform, &mut cell);
    }

    if driver.captured {
        if persistent {
            return;
        }
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
    if let Some(request_id) = server
        .as_ref()
        .and_then(|server| server.active_id.as_ref())
        .cloned()
    {
        let output = cfg.out.clone();
        commands.spawn(Screenshot::image(target)).observe(
            move |captured: On<ScreenshotCaptured>, mut server: ResMut<PersistentCaptureServer>| {
                let result = captured
                    .image
                    .clone()
                    .try_into_dynamic()
                    .map_err(|error| format!("could not encode the captured image: {error}"))
                    .and_then(|dynamic_image| {
                        dynamic_image
                            .to_rgb8()
                            .save_with_format(&output, image::ImageFormat::Png)
                            .map_err(|error| {
                                format!("could not write {}: {error}", output.display())
                            })
                    });
                match result {
                    Ok(()) => {
                        info!(target: "thalos::screenshot", "saved {}", output.display());
                        server.completed_captures += 1;
                        server.respond(&request_id, true, "capture complete", Some(&output));
                    }
                    Err(error) => {
                        warn!(target: "thalos::screenshot", "{error}");
                        server.respond(&request_id, false, error, None);
                    }
                }
                server.active_id = None;
                server.publish(true);
            },
        );
    } else {
        commands
            .spawn(Screenshot::image(target))
            .observe(save_to_disk(cfg.out.clone()));
    }
    driver.captured = true;
}

const CLOUD_MOTION_SLEW_FRAMES: u32 = 36;
const CLOUD_MOTION_SLEW_DEG: f32 = 18.0;

/// Hold the initial pose until the scene is converged, then yaw at a constant
/// half degree per frame through the capture frame. The final frame therefore
/// contains history generated by continuous sub-cut camera motion instead of a
/// teleport (which correctly invalidates all history and cannot expose smear).
fn cloud_motion_azimuth_offset(running_frame: u32, warmup_frames: u32) -> f32 {
    let start = warmup_frames.saturating_sub(CLOUD_MOTION_SLEW_FRAMES);
    if running_frame < start {
        return -CLOUD_MOTION_SLEW_DEG;
    }
    let progressed = (running_frame - start + 1).min(CLOUD_MOTION_SLEW_FRAMES);
    -CLOUD_MOTION_SLEW_DEG * (1.0 - progressed as f32 / CLOUD_MOTION_SLEW_FRAMES as f32)
}

/// Resolve a stable deep-water focus whose low sun sits opposite local east.
/// The preset looks just beside that specular path, making the capture
/// sensitive to both resolved slopes and filtered horizon glitter. The focus
/// itself is the analytic sea sphere, not seabed.
fn ocean_site_context(
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
    let sun_world = (-body_state.position).normalize_or_zero();
    let sun_world = if sun_world == DVec3::ZERO {
        DVec3::Y
    } else {
        sun_world
    };

    let dir_body = match *cached_dir {
        Some(dir) => dir,
        None => {
            let dir = find_ocean_site(hs.as_ref(), body_state.orientation, sun_world);
            let up_world = (body_state.orientation * dir).normalize();
            let sun_elevation_deg = up_world.dot(sun_world).clamp(-1.0, 1.0).asin().to_degrees();
            let depth_m = -hs
                .sample_height_m(dir.as_vec3(), OCEAN_SITE_LOD_M)
                .unwrap_or(0.0);
            info!(
                target: "thalos::screenshot",
                "ocean site: depth {depth_m:.0} m, sun elevation {sun_elevation_deg:.1}°"
            );
            *cached_dir = Some(dir);
            dir
        }
    };

    let up_world = (body_state.orientation * dir_body).normalize();
    Some(HubContext {
        body_id,
        center_world: body_state.position + up_world * radius_m,
        up_world,
        pad_r: radius_m,
    })
}

const OCEAN_SITE_LOD_M: f32 = 256.0;

/// Search the globe for deep water at 10–32° sun elevation, preferring a
/// site where the sun's tangent direction is opposite local east. That makes
/// the fixed ocean framing deterministic and naturally backlights the waves.
fn find_ocean_site(source: &dyn HeightSource, body_to_world: DQuat, sun_world: DVec3) -> DVec3 {
    const CANDIDATES: usize = 2_048;
    const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;

    let mut best = DVec3::Y;
    let mut best_score = f64::NEG_INFINITY;
    for i in 0..CANDIDATES {
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / CANDIDATES as f64;
        let ring_r = (1.0 - y * y).sqrt();
        let theta = GOLDEN_ANGLE * i as f64;
        let dir_body = DVec3::new(ring_r * theta.cos(), y, ring_r * theta.sin());
        let height_m = source
            .sample_height_m(dir_body.as_vec3(), OCEAN_SITE_LOD_M)
            .unwrap_or(0.0) as f64;
        if height_m > -250.0 {
            continue;
        }

        let up_world = (body_to_world * dir_body).normalize();
        let sun_sine = up_world.dot(sun_world);
        if !(0.17..=0.53).contains(&sun_sine) {
            continue;
        }
        let seed = if up_world.dot(DVec3::Y).abs() < 0.99 {
            DVec3::Y
        } else {
            DVec3::X
        };
        let east = seed.cross(up_world).normalize();
        let sun_tangent = (sun_world - up_world * sun_sine).normalize_or_zero();
        let glint_alignment = sun_tangent.dot(-east);
        if glint_alignment < 0.75 {
            continue;
        }

        // Depth gives margin against remote islands entering the low horizon;
        // alignment dominates so the same framing keeps the glitter visible.
        let target_sun_sine = 0.32;
        let score =
            glint_alignment * 8_000.0 - (sun_sine - target_sun_sine).abs() * 2_000.0 - height_m;
        if score > best_score {
            best_score = score;
            best = dir_body;
        }
    }
    best
}

/// Stable daylight focus for an uninhabited airless body. The sub-stellar
/// direction keeps the first visual probe illuminated while azimuth/elevation
/// expose enough phase angle for Hapke backscatter and limb darkening to read.
fn daylight_surface_context(
    sim: &SimulationState,
    solar: &SolarSystemState,
    height_sources: &HeightSourceRegistry,
    body_id: BodyId,
    surfaces: &BodySurfaceRegistry,
    cached_dir: &mut Option<DVec3>,
) -> Option<HubContext> {
    let states = solar.states.as_deref()?;
    let body_state = states.get(body_id)?;
    let radius_m = sim.system.bodies.get(body_id)?.radius_m;
    let hs = height_sources.get(body_id)?;
    let sun_inertial = (-body_state.position).normalize_or_zero();
    let sun_world = if sun_inertial == DVec3::ZERO {
        DVec3::Y
    } else {
        sun_inertial
    };
    let sun_body = (body_state.orientation.inverse() * sun_world).normalize();
    let dir_body = match *cached_dir {
        Some(dir) => dir,
        None => {
            let dir = find_rugged_airless_site(
                hs.as_ref(),
                sun_body,
                surfaces.airless_landmarks(body_id),
            );
            let incidence_deg = dir.dot(sun_body).clamp(-1.0, 1.0).acos().to_degrees();
            info!(
                target: "thalos::screenshot",
                "airless survey site: solar incidence {incidence_deg:.0}°"
            );
            *cached_dir = Some(dir);
            dir
        }
    };
    let up_world = (body_state.orientation * dir_body).normalize();
    let height_m = hs
        .sample_height_m(dir_body.as_vec3(), DRY_SITE_LOD_M)
        .unwrap_or(0.0) as f64;
    let surface_r = radius_m + height_m;
    Some(HubContext {
        body_id,
        center_world: body_state.position + up_world * surface_r,
        up_world,
        pad_r: surface_r,
    })
}

/// Exact surface focus under the canonical EVA spawn.
///
/// EVA site selection deliberately moves away from the initial sub-stellar
/// seed to find a usable plain (or a relief site selected through
/// `THALOS_EVA_SITE`). Derive this probe from the canonical craft state after
/// that placement instead of duplicating the site search here. Height comes
/// from the live atlas-backed authority so the camera follows the exact LOD
/// surface currently rendered during warm-up, including provider A/Bs.
fn eva_surface_context(
    sim: &SimulationState,
    solar: &SolarSystemState,
    height_sources: &HeightSourceRegistry,
    body_id: BodyId,
) -> Option<HubContext> {
    let states = solar.states.as_deref()?;
    let body_state = states.get(body_id)?;
    let radius_m = sim.system.bodies.get(body_id)?.radius_m;
    let height_source = height_sources.get(body_id)?;

    let craft_world = sim.simulation.craft_state().translation.position;
    let up_world = (craft_world - body_state.position)
        .try_normalize()
        .unwrap_or_else(|| (-body_state.position).try_normalize().unwrap_or(DVec3::Y));
    let dir_body = (body_state.orientation.inverse() * up_world).normalize();
    let height_m = height_source
        .sample_height_m(dir_body.as_vec3(), 1.0)
        .unwrap_or(0.0) as f64;
    let surface_r = radius_m + height_m;

    Some(HubContext {
        body_id,
        center_world: body_state.position + up_world * surface_r,
        up_world,
        pad_r: surface_r,
    })
}

/// Focus context centered on the player craft, for the plume preset. Unlike the
/// surface presets this frames the vehicle: `center_world` sits a few metres down
/// the stack (toward the engine + plume) and `up_world` is the ship's nose axis,
/// so [`pose_god_view_camera`]'s azimuth/elevation orbit the rocket.
fn craft_context(sim: &SimulationState) -> Option<HubContext> {
    /// Shift the focus this far along the nose axis (negative = toward the
    /// engine) so the bell + plume land in frame rather than the pod.
    const FOCUS_OFFSET_M: f64 = -4.0;
    let orientation = sim.simulation.attitude().orientation;
    let nose = (orientation * DVec3::Y).try_normalize().unwrap_or(DVec3::Y);
    let craft = sim.simulation.ship_state().position;
    Some(HubContext {
        body_id: sim.simulation.dominant_body(),
        center_world: craft + nose * FOCUS_OFFSET_M,
        up_world: nose,
        pad_r: 0.0,
    })
}

/// Find a visibly structured airless site while keeping the sun oblique enough
/// for relief and Hapke backscatter to read. Candidates use a Fibonacci sphere;
/// each is scored by the elevation range of a ~25 km neighborhood.
fn find_rugged_airless_site(
    source: &dyn HeightSource,
    sun_dir: DVec3,
    landmarks: &[(DVec3, f32)],
) -> DVec3 {
    if let Some((dir, radius_m)) = landmarks
        .iter()
        .find(|(dir, _)| (0.30..=0.75).contains(&dir.dot(sun_dir)))
    {
        info!(
            target: "thalos::screenshot",
            "airless landmark crater: radius {:.1} km",
            radius_m / 1000.0
        );
        return *dir;
    }

    const CANDIDATES: usize = 768;
    const RING_SAMPLES: usize = 10;
    const RING_ANGLE_RAD: f64 = 0.03;
    const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;

    let mut best = sun_dir;
    let mut best_score = f32::NEG_INFINITY;
    for i in 0..CANDIDATES {
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / CANDIDATES as f64;
        let radius = (1.0 - y * y).sqrt();
        let theta = GOLDEN_ANGLE * i as f64;
        let dir = DVec3::new(radius * theta.cos(), y, radius * theta.sin());
        let light = dir.dot(sun_dir);
        // 41–72° incidence: clearly lit, but far enough from noon to reveal
        // crater walls and exercise Hapke's angular response.
        if !(0.30..=0.75).contains(&light) {
            continue;
        }

        let seed = if dir.dot(DVec3::Y).abs() < 0.95 {
            DVec3::Y
        } else {
            DVec3::X
        };
        let tangent_a = seed.cross(dir).normalize();
        let tangent_b = dir.cross(tangent_a).normalize();
        let mut min_h = f32::INFINITY;
        let mut max_h = f32::NEG_INFINITY;
        for ring_i in 0..RING_SAMPLES {
            let a = std::f64::consts::TAU * ring_i as f64 / RING_SAMPLES as f64;
            let ring = tangent_a * a.cos() + tangent_b * a.sin();
            let sample_dir = (dir * RING_ANGLE_RAD.cos() + ring * RING_ANGLE_RAD.sin()).normalize();
            let h = source
                .sample_height_m(sample_dir.as_vec3(), 128.0)
                .unwrap_or(0.0);
            min_h = min_h.min(h);
            max_h = max_h.max(h);
        }
        let score = max_h - min_h;
        if score > best_score {
            best_score = score;
            best = dir;
        }
    }
    best
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
    if let Some(saved) = cfg.saved.as_ref() {
        return serde_json::json!({
            "kind": "saved_perspective",
            "body": saved.body.as_str(),
            "position_body_m": saved.camera_position_body_m,
            "rotation_body_xyzw": saved.camera_rotation_body_xyzw,
            "vertical_fov_rad": saved.vertical_fov_rad,
            "recorded_sim_time_s": saved.sim_time_s,
        })
        .to_string();
    }
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
            look_at_body_center,
        } => {
            let sun_elevation = site_sun_elevation_deg
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "null".to_string());
            format!(
                "{{\"kind\":\"local_cloud\",\"azimuth_deg\":{:.4},\"camera_altitude_m\":{:.3},\"look_elevation_deg\":{:.4},\"site_sun_elevation_deg\":{},\"tangent_limb\":{},\"look_at_body_center\":{}}}",
                cfg.azimuth_deg,
                camera_altitude_m,
                look_elevation_deg,
                sun_elevation,
                tangent_limb,
                look_at_body_center,
            )
        }
    }
}

/// Re-project an F8 handoff through the freshly booted body's current state.
/// This preserves the exact geographic eye position, orientation, and lens
/// without restoring a partial simulation snapshot.
fn pose_saved_perspective(
    saved: &SavedPerspective,
    body_id: BodyId,
    solar: &SolarSystemState,
    root: &Grid,
    transform: &mut Transform,
    cell: &mut CellCoord,
    projection: &mut Projection,
) -> bool {
    let Some(body_state) = solar
        .states
        .as_deref()
        .and_then(|states| states.get(body_id))
    else {
        return false;
    };
    let camera_body = DVec3::from_array(saved.camera_position_body_m);
    let rotation_body = DQuat::from_xyzw(
        saved.camera_rotation_body_xyzw[0],
        saved.camera_rotation_body_xyzw[1],
        saved.camera_rotation_body_xyzw[2],
        saved.camera_rotation_body_xyzw[3],
    )
    .normalize();
    let camera_world = body_state.position + body_state.orientation * camera_body;
    let rotation_world = (body_state.orientation * rotation_body).normalize();
    let (next_cell, local) = root.translation_to_grid(camera_world);
    *cell = next_cell;
    transform.translation = local;
    transform.rotation = Quat::from_xyzw(
        rotation_world.x as f32,
        rotation_world.y as f32,
        rotation_world.z as f32,
        rotation_world.w as f32,
    )
    .normalize();
    if let Projection::Perspective(perspective) = projection {
        perspective.fov = saved.vertical_fov_rad;
    }
    true
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
    let memory = cloud_target_memory_for(
        clouds.render_resolution.x.round() as u32,
        clouds.render_resolution.y.round() as u32,
    );
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
    let temporal = clouds.reprojection_strength > 0.0;
    let reconstruction = if !temporal {
        "raw"
    } else if clouds.sparse_march {
        "sparse-history"
    } else {
        "dense-history"
    };
    let line = format!(
        concat!(
            "{{\"schema\":\"thalos.cloud_probe.v1\",",
            "\"unix_ms\":{},\"preset\":\"{}\",",
            "\"viewport\":[{},{}],\"screenshot_target_bytes\":{},",
            "\"framing\":{},",
            "\"cloud_internal_resolution\":[{},{}],",
            "\"quality\":\"{}\",\"temporal\":{},\"reconstruction\":\"{}\",",
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
        clouds.render_resolution.x.round() as u32,
        clouds.render_resolution.y.round() as u32,
        cfg.cloud_quality.name(),
        temporal,
        reconstruction,
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
            look_at_body_center,
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
            look_at_body_center,
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

/// Put the screenshot camera at EVA eye height and look along the local
/// horizon. An aerial orbit boom is invalid for this preset: at grazing angles
/// its kilometre-long horizontal offset can put the camera inside an unrelated
/// crater wall even though the spawn focus itself is safe.
fn pose_eva_camera(
    cfg: &ScreenshotConfig,
    ctx: &HubContext,
    root: &Grid,
    transform: &mut Transform,
    cell: &mut CellCoord,
) {
    let up = ctx.up_world;
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
    let look_dir = (horiz * elev.cos() + up * elev.sin()).normalize();
    let camera_world = ctx.center_world + up * cfg.distance_m;

    let (next_cell, local) = root.translation_to_grid(camera_world);
    *cell = next_cell;
    *transform = Transform::from_translation(local).looking_to(look_dir.as_vec3(), up.as_vec3());
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
        let seed = if sun.y.abs() < 0.9 {
            DVec3::Y
        } else {
            DVec3::X
        };
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
/// With `look_at_body_center` the camera sits on the site radial and aims at
/// the body centre (full planetary disc).
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
    look_at_body_center: bool,
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

    let body_center = ctx.center_world - up * ctx.pad_r;
    let camera_world = if look_at_body_center {
        // Slight heading offset so the disc isn't a pure sun-facing billboard
        // and weather systems read across the phase terminator.
        let radial = (up * 0.92 + heading * 0.08).normalize();
        body_center + radial * (ctx.pad_r + camera_altitude_m)
    } else {
        ctx.center_world + up * camera_altitude_m
    };

    let look_direction = if look_at_body_center {
        (body_center - camera_world).normalize()
    } else {
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
        (heading * elevation.cos() + up * elevation.sin()).normalize()
    };
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
