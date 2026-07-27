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

use serde::Serialize;

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
    window::{CursorIcon, SystemCursorIcon},
};
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_body_render::renderer_tile_lod_m_at;
use thalos_body_render::tiles::TileTerrainRoot;
use thalos_body_render::udlod::prelude::{TerrainViewComponents, TileAtlas, TileTree};
use thalos_body_render::{
    BodyTerrainMaterial, CloudsConfig, HeightSource, cloud_target_memory_for,
};
use thalos_capture_protocol::{
    CAPTURE_PRESETS, CAPTURE_PROTOCOL_SCHEMA, CaptureAction, CaptureRequest, CaptureResponse,
    CaptureServerState, Viewpoint,
};
use thalos_input::game::GameInputIntent;
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::camera::ShipCamera;
use crate::loading::AppState;
use crate::rendering::contact_shadow::ContactShadowConfig;
use crate::rendering::ground_terrain::{BodyTerrain, OceanDebugSettings};
use crate::rendering::ssao::SsaoConfig;
use crate::rendering::{SimulationState, SolarSystemState};
use crate::space_center::{HubContext, hub_context};
use crate::spawn::{Homeworld, SpawnSituation};
use crate::structures::StructureRegistry;
use crate::terrain_registry::{AirlessLandmark, BodySurfaceRegistry};

const CAPTURE_REQUEST_FILENAME: &str = "visual_capture_request.json";
const CAPTURE_RESPONSE_FILENAME: &str = "visual_capture_response.json";
const CAPTURE_SERVER_STATE_FILENAME: &str = "visual_capture_server.json";

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

pub(crate) fn timestamp_millis() -> u128 {
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
    /// Eye-level along the base's paved network — taxiway, apron and service
    /// road filling the near frame from ~13 m up.
    ///
    /// The regression probe for **scatter versus pavement**. Connections are a
    /// drape lifted 12 cm over the flattened ground, so anything that grows on
    /// the ground under them comes up through the tarmac as stubby tufts, and
    /// nothing in an aerial framing resolves a blade
    /// (INC-20260726T040431Z). It is the only preset that samples pavement
    /// from inside the grass layer's altitude band, so it is where a missing
    /// clear footprint shows first.
    PavedGround,
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
    /// A low oblique survey over a **forest stand** on the wet belt — the
    /// verification probe for the tree/grass/ground colour coupling (the
    /// shared landcover palette, the near-field understory recovery, the
    /// per-instance canopy tint). Mirrors [`Self::DryBelt`]'s searched-site
    /// shape: boots the plain orbit scenario, then searches the daylight
    /// hemisphere for the low-latitude land direction that maximises macro
    /// moisture *and* the scatter stand field, so the frame lands on real
    /// rendered trees standing on forest-painted ground wherever the fields
    /// put them (seed/rotation-independent).
    ForestStand,
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
    /// **Reference framing 1 of 3** — the whole body as a near-full disc from
    /// deep orbit, sun behind the camera.
    ///
    /// At this range every learned height band sits below one pixel, so the
    /// image is carried by **albedo province structure**: mare/highland
    /// contrast, fresh-crater ejecta, and ray systems. That makes this the
    /// verification probe for material-province work
    /// (ADR-20260722T084154Z), *not* for the height cascade — relief
    /// contributes only near the limb and terminator.
    MiraDisc,
    /// **Reference framing 2 of 3** — oblique orbital approach under grazing
    /// light, horizon in frame.
    ///
    /// Long rim shadows and basin rings make this the framing where the
    /// ~0.5–32 km bands of the learned cascade (S0–S2) actually read, coupled
    /// to the shadow and Hapke response. The regime where a bad macro
    /// heightfield is most obvious.
    MiraApproach,
    /// **Reference framing 3 of 3** — low oblique across a landmark crater's
    /// rim: terraced walls, central peak, and floor at ~10–100 m detail.
    ///
    /// The close-band probe for `Rclient` reconstruction and the S3/S4
    /// wavelengths. Sits nearer and far more obliquely than
    /// [`Self::MiraSurface`], which surveys regolith from a steeper survey
    /// altitude rather than reading rim structure against the sky.
    MiraRim,
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
    /// The plume seen from **below the engine, looking up past the craft** at a
    /// lit daytime sky, with the far ground still in the bottom of the frame.
    ///
    /// This is the ordering probe, not a hero shot. The body-centred fullscreen
    /// composites are pinned behind world transparency by
    /// [`thalos_body_render::composite_order`]; the failure that rule exists to
    /// prevent only appears when the camera is pitched **above** the local
    /// horizontal, because that is when the planet centre falls *behind* the
    /// camera and the atmosphere's geometric sort key stops being large and
    /// negative. A plume framed against sky from a level or downward view will
    /// not reproduce it. Any future capture meant to exercise composite order
    /// must keep this upward pitch.
    PlumeSkyline,
    /// **Showcase framing 1 of 3 (NTR-X4)** — aerial oblique establishing shot
    /// of the diffusion window's NE massif (5.8 km peaks ~54 km NNE of the
    /// spaceport), styled after the Alpine photogrammetry reference: the whole
    /// massif with its valley flanks in frame, morning cross-light. The
    /// primary judge-against-the-reference frame for mountain texturing.
    ///
    /// Boots the runway scenario purely for its fixed morning epoch (the sun
    /// geometry at the massif is then deterministic); the site itself is a
    /// fixed lat/lon derived from the 90 m detail raster, so these presets
    /// only frame the intended mountains under `THALOS_TERRAIN=diffusion`.
    MassifAerial,
    /// **Showcase framing 2 of 3 (NTR-X4)** — mid-altitude shot across the
    /// massif's summit ridge: close enough that per-material texture detail
    /// (rock strata, scree, snow line) carries the frame rather than macro
    /// shape.
    MassifRidge,
    /// **Showcase framing 3 of 3 (NTR-X4)** — from over the 1.8 km valley
    /// floor NW of the peak, looking up the valley at the 4 km face: the
    /// slope-driven rock/vegetation transition, talus aprons, and forest
    /// edges read against near-vertical relief.
    MassifValley,
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
        // Banded march budgets (thalos::atmosphere march contract): 176 steps
        // reach the full 300 km cap; Low's 112 reaches ~92 km — still past
        // the old 67 km everywhere-600 m reach.
        match self {
            Self::Low => 112,
            Self::Baseline => 176,
            Self::High => 192,
            Self::Reference => 224,
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

/// Which half of the frame the unlit terrain must fall on.
///
/// Sun-relative framings are otherwise mirror-ambiguous: rotating the boom by
/// `+θ` and `-θ` about the site's vertical produces the same lighting *character*
/// but puts the terminator on opposite sides. Naming the dark side resolves that
/// without hand-tuning an absolute azimuth per site.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FrameSide {
    Left,
    Right,
}

/// A resolved capture focus plus the feature scale a framing may want to key
/// off. Solar geometry is not carried here — pose functions derive the sun from
/// `SolarSystemState`, so there is one source for it.
struct CaptureFocus {
    hub: HubContext,
    /// Radius of the landmark crater the airless site search locked onto, when
    /// it found one. Lets a framing scale its boom to the actual feature rather
    /// than hard-coding a distance that only suits one crater size.
    landmark_radius_m: Option<f64>,
}

impl From<HubContext> for CaptureFocus {
    fn from(hub: HubContext) -> Self {
        Self {
            hub,
            landmark_radius_m: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ScreenshotFraming {
    /// Original focus-orbit framing used by the base/hub regression shots.
    GodView,
    /// `GodView` with the boom azimuth measured **from the sun** rather than
    /// from local east, and the terminator pinned to a chosen side of frame.
    ///
    /// Absolute azimuth is not reproducible lighting: the site search picks the
    /// site, so where the sun ends up in frame is incidental. Measuring from the
    /// sun's horizontal bearing makes the lighting character the authored
    /// quantity — which is the whole point of the Mira reference framings.
    SunRelativeGodView {
        /// Boom bearing measured from the sun's horizontal direction at the
        /// site. `180°` puts the sun behind the camera (flat, fully lit); `90°`
        /// is cross-lit; small angles look into the sun.
        sun_azimuth_deg: f32,
        dark_side: FrameSide,
        /// When set, the boom length becomes this multiple of the landmark
        /// crater's radius instead of `distance_m`.
        ///
        /// A framing that must contain a crater cannot use a fixed distance: the
        /// site search picks the feature, and a boom tuned for a 6 km crater
        /// frames open ground around a 30 km one. Falls back to `distance_m`
        /// when the search resolved no landmark.
        landmark_radii: Option<f32>,
    },
    /// The whole body as a disc, viewed at a chosen **phase angle** from the
    /// sun, rolled so the terminator runs vertically.
    ///
    /// Aims at the body centre rather than a surface point, so the disc is
    /// centred regardless of which site the search picked. `phase_deg = 0` is
    /// full; `90` is exactly half-lit.
    BodyDisc {
        phase_deg: f32,
        dark_side: FrameSide,
    },
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
    const ALL: &'static [Self] = &[
        Self::LatestPerspective,
        Self::SpaceportAerial,
        Self::RunwayAtmosphere,
        Self::PavedGround,
        Self::Hub,
        Self::DryBelt,
        Self::ForestStand,
        Self::EarthReference,
        Self::Ocean,
        Self::OceanSlopes,
        Self::MiraOrbit,
        Self::MiraSurface,
        Self::MiraEva,
        Self::MiraDisc,
        Self::MiraApproach,
        Self::MiraRim,
        Self::CloudRunway,
        Self::CloudMotion,
        Self::CloudCruise,
        Self::CloudInterior,
        Self::CloudLimb,
        Self::CloudPlanet,
        Self::CloudSunset,
        Self::Plume,
        Self::PlumeSkyline,
        Self::MassifAerial,
        Self::MassifRidge,
        Self::MassifValley,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::LatestPerspective => "latest-perspective",
            Self::SpaceportAerial => "spaceport-aerial",
            Self::RunwayAtmosphere => "runway-atmosphere",
            Self::PavedGround => "paved-ground",
            Self::Hub => "hub",
            Self::DryBelt => "dry-belt",
            Self::ForestStand => "forest-stand",
            Self::EarthReference => "earth-reference",
            Self::Ocean => "ocean",
            Self::OceanSlopes => "ocean-slopes",
            Self::MiraOrbit => "mira-orbit",
            Self::MiraSurface => "mira-surface",
            Self::MiraEva => "mira-eva",
            Self::MiraDisc => "mira-disc",
            Self::MiraApproach => "mira-approach",
            Self::MiraRim => "mira-rim",
            Self::CloudRunway => "cloud-runway",
            Self::CloudMotion => "cloud-motion",
            Self::CloudCruise => "cloud-cruise",
            Self::CloudInterior => "cloud-interior",
            Self::CloudLimb => "cloud-limb",
            Self::CloudPlanet => "cloud-planet",
            Self::CloudSunset => "cloud-sunset",
            Self::Plume => "plume",
            Self::PlumeSkyline => "plume-skyline",
            Self::MassifAerial => "massif-aerial",
            Self::MassifRidge => "massif-ridge",
            Self::MassifValley => "massif-valley",
        }
    }

    fn try_parse(raw: &str) -> Option<Self> {
        Some(match raw.trim().to_ascii_lowercase().as_str() {
            "latest" | "perspective" | "latest-perspective" | "latest_perspective" => {
                Self::LatestPerspective
            }
            // Truthy / unnamed → the default preset.
            "" | "1" | "true" | "yes" | "on" | "spaceport" | "spaceport-aerial" | "aerial"
            | "base" => Self::SpaceportAerial,
            "runway-atmosphere" | "runway_atmosphere" | "runway-sky" | "surface-atmosphere" => {
                Self::RunwayAtmosphere
            }
            "paved-ground" | "paved_ground" | "pavement" | "taxiway" | "apron" => {
                Self::PavedGround
            }
            "hub" | "space-center" | "spacecenter" | "play" => Self::Hub,
            "dry" | "dry-belt" | "drybelt" | "desert" | "biome" => Self::DryBelt,
            "forest" | "forest-stand" | "foreststand" | "trees" => Self::ForestStand,
            "earth-reference" | "earth_reference" | "earth-ref" | "atmosphere" | "atmo" => {
                Self::EarthReference
            }
            "ocean" | "open-ocean" | "open_ocean" | "sea" | "water" => Self::Ocean,
            "ocean-slopes" | "ocean_slopes" | "sea-slopes" | "slope-field" => Self::OceanSlopes,
            "mira" | "mira-orbit" | "mira_orbit" => Self::MiraOrbit,
            "mira-surface" | "mira_surface" | "regolith" => Self::MiraSurface,
            "mira-eva" | "mira_eva" | "regolith-eva" => Self::MiraEva,
            "mira-disc" | "mira_disc" | "mira-full" | "mira-globe" => Self::MiraDisc,
            "mira-approach" | "mira_approach" | "mira-oblique" | "mira-limb" => Self::MiraApproach,
            "mira-rim" | "mira_rim" | "mira-crater" | "crater-rim" => Self::MiraRim,
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
            "plume-skyline" | "plume_skyline" | "plume-sky" | "plume-ascent" => Self::PlumeSkyline,
            "massif-aerial" | "massif_aerial" | "massif" | "mountains" => Self::MassifAerial,
            "massif-ridge" | "massif_ridge" | "ridge" => Self::MassifRidge,
            "massif-valley" | "massif_valley" | "valley" => Self::MassifValley,
            _ => return None,
        })
    }

    /// The scenario the world must be booted into for this preset.
    pub fn spawn_situation(self) -> SpawnSituation {
        match self {
            // The loaded handoff overrides this placeholder through
            // `ScreenshotConfig::spawn_situation`.
            Self::LatestPerspective => SpawnSituation::ShipOrbit,
            Self::SpaceportAerial
            | Self::RunwayAtmosphere
            | Self::PavedGround
            | Self::CloudRunway
            | Self::CloudMotion => SpawnSituation::Runway,
            // The massif showcase presets ride the runway scenario for its
            // fixed morning epoch: the massif sits ~0.5° from the spaceport,
            // so the authored morning sun is the deterministic lighting there
            // too. The camera then leaves the base for the fixed massif site.
            Self::MassifAerial | Self::MassifRidge | Self::MassifValley => SpawnSituation::Runway,
            // The hub is the PLAY path: the placeholder parking orbit plus the
            // spaceport build (armed by `main.rs` via `boots_hub`).
            Self::Hub => SpawnSituation::ShipOrbit,
            // Dry-belt frames wild terrain far from any base, so a plain orbit
            // scenario is enough; the driver poses the camera over the searched
            // desert site (the craft stays in orbit, irrelevant to the framing).
            Self::DryBelt | Self::ForestStand | Self::Ocean | Self::OceanSlopes => {
                SpawnSituation::ShipOrbit
            }
            Self::EarthReference => SpawnSituation::Runway,
            Self::MiraOrbit
            | Self::MiraSurface
            | Self::MiraDisc
            | Self::MiraApproach
            | Self::MiraRim => SpawnSituation::ShipOrbit,
            Self::MiraEva => SpawnSituation::Eva,
            Self::CloudCruise
            | Self::CloudInterior
            | Self::CloudLimb
            | Self::CloudPlanet
            | Self::CloudSunset => SpawnSituation::ShipOrbit,
            // Plain orbit: space/planet backdrop, engine forced to fire.
            Self::Plume => SpawnSituation::ShipOrbit,
            // Final approach: the rocket airborne at ~1.5 km AGL over flat dry
            // land, nose retrograde (engine down). Gives a real atmospheric
            // plume, a lit sky, and ground still in frame — all three needed
            // for the composite-order probe.
            Self::PlumeSkyline => SpawnSituation::FinalApproach,
        }
    }

    /// Body that owns the world and terrain framed by this preset.
    pub fn target_body_name(self) -> &'static str {
        match self {
            // The loaded handoff overrides this placeholder through
            // `ScreenshotConfig::target_body_name`.
            Self::LatestPerspective => "Thalos",
            Self::MiraOrbit
            | Self::MiraSurface
            | Self::MiraEva
            | Self::MiraDisc
            | Self::MiraApproach
            | Self::MiraRim => "Mira",
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
                viewpoint: None,
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
                viewpoint: None,
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
                viewpoint: None,
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
            Self::PavedGround => ScreenshotConfig {
                preset: self,
                viewpoint: None,
                out: PathBuf::from("artifacts/visual/latest/paved_ground.png"),
                width: 1920,
                height: 1080,
                // 180 deg puts the camera on the airside pavement looking back
                // across the taxiway and apron; a 500 m boom at 1.5 deg is
                // ~13 m up — inside the grass layer's full-blade band
                // (`gpu_grass::HIDE_AGL_M` = 550), which is the whole point.
                azimuth_deg: 180.0,
                elevation_deg: 1.5,
                distance_m: 500.0,
                warmup_frames: 480,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("paved_ground.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            // Matches the hub's establishing view (`BASE_ESTABLISHING_DISTANCE_M`,
            // the one god-view framing per base) so the capture shows what PLAY shows.
            Self::Hub => ScreenshotConfig {
                preset: self,
                viewpoint: None,
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
                viewpoint: None,
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
            // Low oblique and close, mid-way into a stand: individual mesh
            // trees in the foreground, the impostor band and the far
            // forest-painted ground in one frame, so the whole tree/ground
            // colour handoff is visible at once. Same cold-streaming warmup
            // reasoning as `DryBelt`.
            Self::ForestStand => ScreenshotConfig {
                preset: self,
                viewpoint: None,
                out: PathBuf::from("artifacts/visual/latest/forest_stand.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 35.0,
                elevation_deg: 14.0,
                distance_m: 900.0,
                warmup_frames: 600,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("forest_stand.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                // Near-clear sky: this probe judges SUNLIT hue coupling between
                // canopy, blades, and ground — an overcast frame flattens all
                // three into the same grey-lit family and hides regressions,
                // and the wet-belt site the search picks is reliably under a
                // dense deck (0.35 still read fully overcast there).
                cloud_coverage_scale: Some(0.10),
            },
            Self::CloudRunway => ScreenshotConfig {
                preset: self,
                viewpoint: None,
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
                viewpoint: None,
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
                viewpoint: None,
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
                viewpoint: None,
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
                viewpoint: None,
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
                viewpoint: None,
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
                viewpoint: None,
                out: PathBuf::from("artifacts/visual/latest/plume.png"),
                width: 1920,
                height: 1080,
                // Framed against black sky, not the sunlit planet: the plume is
                // an *additive* emitter, so a bright backdrop washes out exactly
                // the colour and shock structure this probe exists to check.
                // Distance fits the whole sea-level column in frame *including
                // its tail*: the point of the probe is where the plume ends, so
                // a framing that crops the fade is useless (INC-20260724T235437Z-plume-ended-on-a-lit-rim).
                azimuth_deg: 235.0,
                elevation_deg: 8.0,
                distance_m: 130.0,
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
            // Same craft-centred GodView focus as `Plume` (up = the nose axis),
            // but posed *under* the waist so the camera looks up the stack.
            //
            // The negative elevation is the whole point and is not a taste
            // call: it is what puts the view above the local horizontal, which
            // is the only regime in which the fullscreen composites can sort
            // ahead of world transparency (see `PlumeSkyline`'s doc). 18° below
            // a craft 1.5 km up still leaves the far ground and horizon in the
            // bottom of the frame, so one image shows both backdrops — the
            // plume against sky and the plume against terrain — and the old
            // failure is a cut exactly at the skyline.
            Self::PlumeSkyline => ScreenshotConfig {
                preset: self,
                viewpoint: None,
                out: PathBuf::from("artifacts/visual/latest/plume_skyline.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 235.0,
                elevation_deg: -18.0,
                // Far enough back that the whole column plus a wide band of
                // sky and ground is in frame, not just the bell.
                distance_m: 260.0,
                // Descent scenario: deferred terrain-aware placement plus a
                // cold surface site, so this needs the long streaming warmup
                // rather than the orbital preset's 90 frames.
                warmup_frames: 600,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("plume_skyline.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            // NTR-X4 showcase framings. All three point at fixed sites in the
            // diffusion detail window (see `massif_site`), stream a cold wild
            // mountain site, and therefore need the long warmup.
            Self::MassifAerial => ScreenshotConfig {
                preset: self,
                viewpoint: None,
                out: PathBuf::from("artifacts/visual/latest/massif_aerial.png"),
                width: 1920,
                height: 1080,
                // Camera south of the peak looking north across the massif;
                // the fixed morning sun (local east) cross-lights from frame
                // right, the reference's relief-legible geometry. 22 km out /
                // 30° up frames the whole massif with its green valley flanks
                // (14 km filled the frame with the summit snow plateau alone).
                azimuth_deg: 270.0,
                elevation_deg: 30.0,
                distance_m: 22_000.0,
                warmup_frames: 720,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("massif_aerial.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::MassifRidge => ScreenshotConfig {
                preset: self,
                viewpoint: None,
                out: PathBuf::from("artifacts/visual/latest/massif_ridge.png"),
                width: 1920,
                height: 1080,
                // Closer boom from the south-southwest, low over the summit
                // ridge, so per-material texture (strata, scree, the snow
                // line) is the subject rather than the massif silhouette.
                // NOTE `pose_god_view_camera`'s tangent basis mirrors east
                // (its `east` = Y×up = −ENU east): for a camera-offset
                // bearing B (true, from north) the azimuth to author is
                // atan2(cos B, −sin B). B=210° (SSW) → 300°.
                azimuth_deg: 300.0,
                elevation_deg: 12.0,
                distance_m: 5_000.0,
                warmup_frames: 720,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("massif_ridge.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::MassifValley => ScreenshotConfig {
                preset: self,
                viewpoint: None,
                out: PathBuf::from("artifacts/visual/latest/massif_valley.png"),
                width: 1920,
                height: 1080,
                // Focus sits on the NW face; the boom runs down-valley along
                // the reciprocal of the measured valley→peak bearing (139°),
                // i.e. camera-offset bearing 319° (NW), putting the camera
                // over the 1.8 km valley floor looking up at the face. In
                // the mirrored GodView basis (see massif-ridge note):
                // atan2(cos 319°, −sin 319°) = 49°.
                azimuth_deg: 49.0,
                elevation_deg: 2.0,
                distance_m: 9_000.0,
                warmup_frames: 720,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("massif_valley.jsonl"),
                framing: ScreenshotFraming::GodView,
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::EarthReference => ScreenshotConfig {
                preset: self,
                viewpoint: None,
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
                viewpoint: None,
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
                viewpoint: None,
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
                viewpoint: None,
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
                viewpoint: None,
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
                viewpoint: None,
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
            Self::MiraDisc => ScreenshotConfig {
                preset: self,
                viewpoint: None,
                out: PathBuf::from("artifacts/visual/latest/mira_disc.png"),
                width: 1920,
                height: 1080,
                // Unused by `BodyDisc`, which derives its whole pose from the
                // sun and the body centre.
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
                // Geometry, not taste: measured from the surface, so the camera
                // sits at r = 869 km + 1,900 km = 2,769 km from centre. Angular
                // radius asin(869/2769) = 18.3° against the 22.5° half-FOV puts
                // the disc at ~81% of frame height. Also stays inside the
                // 4×radius impostor swap, so this frames the real ground LOD
                // rather than the billboard.
                distance_m: 1_900_000.0,
                // Coarse LOD at this range, but the cache still cold-misses on a
                // new package key.
                warmup_frames: 720,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("mira_disc.jsonl"),
                // Exactly half-lit with the terminator running vertically down
                // frame centre, unlit half to the left.
                framing: ScreenshotFraming::BodyDisc {
                    phase_deg: 90.0,
                    dark_side: FrameSide::Left,
                },
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::MiraApproach => ScreenshotConfig {
                preset: self,
                viewpoint: None,
                out: PathBuf::from("artifacts/visual/latest/mira_approach.png"),
                width: 1920,
                height: 1080,
                // Bearing is sun-relative (see `framing`), so this is unused.
                azimuth_deg: 0.0,
                // Shallow boom → camera ~183 km up and ~385 km downrange of the
                // focus. The look direction lands ~3° below the horizon, so the
                // limb crosses just above frame centre and the lit surface fills
                // the lower ~2/3 — the reference approach composition.
                elevation_deg: 16.0,
                distance_m: 400_000.0,
                warmup_frames: 900,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("mira_approach.jsonl"),
                // Cross-lit at 70° off the sun bearing: grazing enough for long
                // rim shadows, with the terrain falling into darkness on the
                // left of frame.
                framing: ScreenshotFraming::SunRelativeGodView {
                    sun_azimuth_deg: 70.0,
                    dark_side: FrameSide::Left,
                    landmark_radii: None,
                },
                cloud_quality: CloudCaptureQuality::Baseline,
                cloud_temporal: true,
                cloud_coverage_scale: None,
            },
            Self::MiraRim => ScreenshotConfig {
                preset: self,
                viewpoint: None,
                out: PathBuf::from("artifacts/visual/latest/mira_rim.png"),
                width: 1920,
                height: 1080,
                // Bearing is sun-relative (see `framing`), so this is unused.
                azimuth_deg: 0.0,
                // Looking *into* a crater needs enough depression to see the
                // floor and far wall: at 8° a 57 km-wide crater foreshortened
                // into a thin band on the horizon. 20° keeps the view clearly
                // oblique — the far rim still rises against the sky — while the
                // floor opens up. Distance is crater-scaled (see `framing`), so
                // this fallback only applies if no landmark resolved.
                elevation_deg: 20.0,
                distance_m: 60_000.0,
                // Closest of the three framings, so the heaviest cold package
                // probe — matches `MiraSurface`'s measured convergence.
                warmup_frames: 1_200,
                tail_frames: 24,
                keep_hud: false,
                report: crate::artifact_paths::default_jsonl_path("mira_rim.jsonl"),
                // Look *across* the crater with the sun low and off to one
                // side, so the near rim shadows the floor and the far terraces
                // catch the light — the reference crater-rim read.
                framing: ScreenshotFraming::SunRelativeGodView {
                    sun_azimuth_deg: 55.0,
                    dark_side: FrameSide::Left,
                    // Stand off ~2.6 crater radii so the far wall and floor both
                    // sit in frame whatever size the search lands on.
                    landmark_radii: Some(2.6),
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
    /// Present only for a catalog-authored viewpoint. Scripted presets keep
    /// this `None` and continue through their typed framing paths.
    viewpoint: Option<Viewpoint>,
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
    ///   A viewpoint id (or `viewpoint:<id>`) replays the shared authored
    ///   catalog at `assets/viewpoints.json`; `latest` selects its newest entry.
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
    /// - `THALOS_SCREENSHOT_CLOUD_DENSITY_COUPLING` — legacy-bias or
    ///   shared-envelope; capture-only near/far distribution contract A/B.
    /// - `THALOS_SCREENSHOT_CLOUD_TIER` — near-only, composite, or far-only;
    ///   capture-only estimator isolation diagnostic.
    /// - `THALOS_SCREENSHOT_CLOUD_FAR_FILTER` — chord-mip or pixel-footprint;
    ///   capture-only far projection footprint A/B.
    /// - `THALOS_SCREENSHOT_CLOUD_FAR_AGGREGATION` — stacked or
    ///   coverage-preserving; capture-only far opacity A/B.
    /// - `THALOS_SCREENSHOT_CLOUD_COVERAGE` — optional global coverage scale.
    /// - `THALOS_SCREENSHOT_REPORT` — JSONL report path (defaults under
    ///   `artifacts/diagnostics/`).
    /// - `THALOS_SCREENSHOT_OCEAN_TIME` — optional fixed canonical ocean time
    ///   in seconds (ocean diagnostics/phase comparisons only).
    pub fn from_env() -> Option<Self> {
        let raw = env::var("THALOS_SCREENSHOT").ok()?;
        let mut cfg = Self::for_scene(&raw)
            .unwrap_or_else(|error| panic!("could not resolve capture scene {raw:?}: {error}"));
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
            let viewpoint = crate::viewpoints::resolve_viewpoint("latest")
                .unwrap_or_else(|error| panic!("could not load the latest viewpoint: {error}"));
            cfg.apply_viewpoint(viewpoint);
            cfg.out = PathBuf::from("artifacts/visual/latest/latest_perspective.png");
            cfg.report = crate::artifact_paths::default_jsonl_path("latest_perspective.jsonl");
        }
        cfg
    }

    fn for_scene(raw: &str) -> Result<Self, String> {
        if let Some(preset) = ScreenshotPreset::try_parse(raw) {
            return Ok(Self::for_preset(preset));
        }
        let viewpoint = crate::viewpoints::resolve_viewpoint(raw)?;
        let mut cfg = ScreenshotPreset::LatestPerspective.defaults();
        cfg.apply_viewpoint(viewpoint);
        Ok(cfg)
    }

    fn apply_viewpoint(&mut self, viewpoint: Viewpoint) {
        self.width = viewpoint.viewport[0];
        self.height = viewpoint.viewport[1];
        self.out = PathBuf::from(format!(
            "artifacts/visual/latest/{}.png",
            viewpoint.id.replace('-', "_")
        ));
        self.report = crate::artifact_paths::default_jsonl_path(&format!(
            "viewpoint_{}.jsonl",
            viewpoint.id.replace('-', "_")
        ));
        self.viewpoint = Some(viewpoint);
    }

    fn scene_name(&self) -> String {
        self.viewpoint
            .as_ref()
            .map(crate::viewpoints::viewpoint_scene_name)
            .unwrap_or_else(|| self.preset.name().to_owned())
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
        self.viewpoint
            .as_ref()
            .map(|viewpoint| viewpoint.body.as_str())
            .unwrap_or_else(|| self.preset.target_body_name())
    }

    /// Canonical scene builder to run behind this capture.
    pub fn spawn_situation(&self) -> SpawnSituation {
        self.viewpoint
            .as_ref()
            .map(|viewpoint| SpawnSituation::from(viewpoint.spawn))
            .unwrap_or_else(|| self.preset.spawn_situation())
    }

    /// Whether the boot should build the no-craft space-center hub route.
    pub fn boots_hub(&self) -> bool {
        self.viewpoint
            .as_ref()
            .map(|viewpoint| viewpoint.boots_hub)
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
    "THALOS_SCREENSHOT_CLOUD_DENSITY_COUPLING",
    "THALOS_SCREENSHOT_CLOUD_TIER",
    "THALOS_SCREENSHOT_CLOUD_FAR_FILTER",
    "THALOS_SCREENSHOT_CLOUD_FAR_AGGREGATION",
    "THALOS_SCREENSHOT_OCEAN_TIME",
    "THALOS_PLUME_PRESSURE",
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
    boot_scene: Option<String>,
    boot_width: u32,
    boot_height: u32,
    compatible_presets: Vec<String>,
    active_id: Option<String>,
    last_request_id: Option<String>,
    completed_captures: u64,
    shader_reload_unix_ms: u128,
    settle_frames: u32,
    heartbeat_frame: u32,
}

impl PersistentCaptureServer {
    fn publish(&self, ready: bool) {
        let Some(scene) = self.boot_scene.as_deref() else {
            return;
        };
        let state = CaptureServerState {
            schema_version: CAPTURE_PROTOCOL_SCHEMA,
            pid: std::process::id(),
            preset: scene.to_owned(),
            compatible_presets: self.compatible_presets.clone(),
            width: self.boot_width,
            height: self.boot_height,
            ready,
            busy: self.active_id.is_some(),
            completed_captures: self.completed_captures,
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
    /// Cached body-fixed direction of the searched forest-stand site
    /// (ForestStand preset only), same latching as [`Self::dry_site_dir`].
    forest_site_dir: Option<DVec3>,
    /// Cached deep-water direction for the low-sun open-ocean preset.
    ocean_site_dir: Option<DVec3>,
    /// Cached Mira survey site for the airless presets: body-fixed direction
    /// plus the landmark crater radius when the search locked onto one.
    airless_site: Option<(DVec3, Option<f64>)>,
    /// Massif presets: latched once the streamed terrain at the focus is
    /// refined (or the stream hold timed out). Until it latches the warmup
    /// frame countdown is held — a cold wild-site fill is wall-clock bound,
    /// not frame bound, so counting frames alone captures the coarse mips
    /// (the 2026-07-24 baseline failure).
    terrain_ready: bool,
    /// Wall-clock seconds spent holding the warmup for terrain streaming.
    terrain_wait_s: f64,
    /// udlod-path plateau tracking: finest resident LOD (m/texel) seen at the
    /// view, and how long it has held without improving.
    terrain_best_lod_m: Option<f32>,
    terrain_lod_hold_s: f64,
}

/// Streamed-terrain readiness inputs for the massif warmup hold, plus the
/// in-flight capture query (bundled — `drive_headless_screenshot` sits at
/// Bevy's function-system param limit).
#[derive(bevy::ecs::system::SystemParam)]
struct TerrainReadiness<'w, 's> {
    active_captures: Query<'w, 's, (), With<Capturing>>,
    time: Res<'w, Time<Real>>,
    tile_roots: Query<
        'w,
        's,
        (
            &'static TileTerrainRoot,
            &'static crate::rendering::tile_terrain::TileTerrainBody,
        ),
    >,
    udlod_trees: Res<'w, TerrainViewComponents<TileTree>>,
    udlod_terrains: Query<
        'w,
        's,
        (
            Entity,
            &'static crate::rendering::ground_terrain::BodyTerrain,
            &'static TileAtlas,
        ),
    >,
    camera_q: Query<'w, 's, Entity, With<ShipCamera>>,
}

/// Ceiling on the massif warmup hold. Cold tile streaming to a wild mountain
/// site is ~20–60 s; past this something is wedged and the capture proceeds
/// (with a warning) rather than hanging the host.
const MASSIF_STREAM_TIMEOUT_S: f64 = 180.0;
/// udlod path: how long the finest resident LOD at the view must hold without
/// improving before the ground counts as streamed (wall clock — headless frame
/// rates make frame counts meaningless against ~30 ms/tile bakes).
const MASSIF_LOD_PLATEAU_S: f64 = 6.0;

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

fn initialize_capture_server(
    cfg: Res<ScreenshotConfig>,
    mut server: ResMut<PersistentCaptureServer>,
) {
    server.boot_scene = Some(cfg.scene_name());
    server.boot_width = cfg.width;
    server.boot_height = cfg.height;
    server.compatible_presets = compatible_presets(&cfg);
    server.settle_frames = env_parse("THALOS_CAPTURE_SETTLE_FRAMES").unwrap_or(60);
    server.publish(false);
}

/// Presets whose app-builder inputs and render-target extent match the booted
/// world. Camera framing and live diagnostic resources can change in-process;
/// target body, spawn scenario, hub wiring, and viewport-sized render resources
/// cannot.
fn compatible_presets(boot: &ScreenshotConfig) -> Vec<String> {
    debug_assert_eq!(
        ScreenshotPreset::ALL
            .iter()
            .map(|preset| preset.name())
            .collect::<Vec<_>>(),
        CAPTURE_PRESETS,
        "capture protocol and runtime preset catalogs diverged"
    );
    let mut scenes = ScreenshotPreset::ALL
        .iter()
        .copied()
        .filter(|preset| *preset != ScreenshotPreset::LatestPerspective)
        .filter(|preset| {
            preset.target_body_name() == boot.target_body_name()
                && preset.spawn_situation() == boot.spawn_situation()
                && preset.boots_hub() == boot.boots_hub()
        })
        .filter(|preset| {
            let defaults = preset.defaults();
            defaults.width == boot.width && defaults.height == boot.height
        })
        .map(|preset| preset.name().to_owned())
        .collect::<Vec<_>>();
    if let Ok(catalog) = crate::viewpoints::load_catalog() {
        scenes.extend(
            catalog
                .viewpoints
                .iter()
                .filter(|viewpoint| {
                    viewpoint.body.eq_ignore_ascii_case(boot.target_body_name())
                        && SpawnSituation::from(viewpoint.spawn) == boot.spawn_situation()
                        && viewpoint.boots_hub == boot.boots_hub()
                        && viewpoint.viewport == [boot.width, boot.height]
                })
                .map(crate::viewpoints::viewpoint_scene_name),
        );
        if catalog.latest().is_some_and(|viewpoint| {
            viewpoint.body.eq_ignore_ascii_case(boot.target_body_name())
                && SpawnSituation::from(viewpoint.spawn) == boot.spawn_situation()
                && viewpoint.boots_hub == boot.boots_hub()
                && viewpoint.viewport == [boot.width, boot.height]
        }) {
            scenes.push("latest-perspective".to_owned());
        }
    }
    let boot_scene = boot.scene_name();
    if !scenes.contains(&boot_scene) {
        scenes.push(boot_scene);
    }
    scenes.sort();
    scenes.dedup();
    scenes
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
    surfaces: Res<crate::terrain_registry::BodySurfaceRegistry>,
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

    if !server
        .compatible_presets
        .iter()
        .any(|scene| scene == &request.preset)
    {
        server.respond(
            &request.id,
            false,
            format!(
                "server booted for {}; {} needs a different boot world",
                server.boot_scene.as_deref().unwrap_or("unknown"),
                request.preset
            ),
            None,
        );
        return;
    }

    let mut next = match ScreenshotConfig::for_scene(&request.preset) {
        Ok(next) => next,
        Err(error) => {
            server.respond(&request.id, false, error, None);
            return;
        }
    };
    next.apply_overrides(&request.overrides);
    if !next
        .target_body_name()
        .eq_ignore_ascii_case(cfg.target_body_name())
        || next.spawn_situation() != cfg.spawn_situation()
        || next.boots_hub() != cfg.boots_hub()
        || next.width != server.boot_width
        || next.height != server.boot_height
    {
        server.respond(
            &request.id,
            false,
            format!(
                "{} changed to a different boot world/context; restart the capture host",
                request.preset
            ),
            None,
        );
        return;
    }
    // A body with no constructed surface renders as a blank world. That is fine
    // for a body merely present in the scene (it degrades locally), but a shot
    // *of* it would be invalid evidence dressed as a valid PNG — refuse it with
    // the repair command instead. See `BodySurfaceRegistry::degraded`.
    if let Some(degraded) = surfaces.degraded_by_name(next.target_body_name()) {
        server.respond(
            &request.id,
            false,
            format!(
                "{} has no terrain surface: {}",
                degraded.body_name, degraded.reason
            ),
            None,
        );
        return;
    }
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
    // Re-arm the massif streaming hold: a new request may frame a different
    // (cold) site. When the ground is already streamed the gate re-passes in
    // one settled check, so warm re-captures stay fast.
    driver.terrain_ready = false;
    driver.terrain_wait_s = 0.0;
    driver.terrain_best_lod_m = None;
    driver.terrain_lod_hold_s = 0.0;
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
    mut contact_shadow: ResMut<ContactShadowConfig>,
    mut cloud_shadow: ResMut<crate::rendering::clouds::CloudShadowConfig>,
    mut terrain_materials: ResMut<Assets<BodyTerrainMaterial>>,
    mut tile_materials: ResMut<Assets<thalos_body_render::tiles::material::TileTerrainMaterial>>,
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
    clouds.surface_density_coupling = match runtime
        .get("THALOS_SCREENSHOT_CLOUD_DENSITY_COUPLING")
        .map(|raw| raw.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("legacy") | Some("legacy-bias") | Some("off") | Some("0") => 0.0,
        Some("shared") | Some("shared-envelope") | Some("on") | Some("1") | None => 1.0,
        Some(other) => {
            warn!(
                target: "thalos::screenshot",
                "unknown THALOS_SCREENSHOT_CLOUD_DENSITY_COUPLING={other:?}; using shared-envelope"
            );
            1.0
        }
    };
    clouds.tier_diagnostic = match runtime
        .get("THALOS_SCREENSHOT_CLOUD_TIER")
        .map(|raw| raw.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("near") | Some("near-only") | Some("-1") => -1.0,
        Some("composite") | Some("both") | Some("0") | None => 0.0,
        Some("far") | Some("far-only") | Some("1") => 1.0,
        Some(other) => {
            warn!(
                target: "thalos::screenshot",
                "unknown THALOS_SCREENSHOT_CLOUD_TIER={other:?}; using composite"
            );
            0.0
        }
    };
    clouds.far_pixel_footprint = match runtime
        .get("THALOS_SCREENSHOT_CLOUD_FAR_FILTER")
        .map(|raw| raw.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("chord") | Some("chord-mip") | Some("legacy") | Some("0") => 0.0,
        Some("pixel") | Some("pixel-footprint") | Some("on") | Some("1") | None => 1.0,
        Some(other) => {
            warn!(
                target: "thalos::screenshot",
                "unknown THALOS_SCREENSHOT_CLOUD_FAR_FILTER={other:?}; using pixel-footprint"
            );
            1.0
        }
    };
    clouds.far_coverage_preserving = match runtime
        .get("THALOS_SCREENSHOT_CLOUD_FAR_AGGREGATION")
        .map(|raw| raw.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("stacked") | Some("legacy") | Some("0") => 0.0,
        Some("coverage") | Some("coverage-preserving") | Some("on") | Some("1") | None => 1.0,
        Some(other) => {
            warn!(
                target: "thalos::screenshot",
                "unknown THALOS_SCREENSHOT_CLOUD_FAR_AGGREGATION={other:?}; using coverage-preserving"
            );
            1.0
        }
    };
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
    contact_shadow.apply_capture_mode(runtime.get("THALOS_CONTACT_SHADOW"));
    cloud_shadow.apply_capture_mode(runtime.get("THALOS_CLOUD_SHADOW"));
    let inspection = terrain_inspection_override(runtime.get("THALOS_TERRAIN_INSPECTION"));
    for (_, material) in terrain_materials.iter_mut() {
        material.extras.inspection.x = inspection;
    }
    // Same override on the tile renderer's materials, so the `terrain-lighting`
    // axis reads the *live* value on the persistent host for whichever renderer
    // owns the body. Without this the tile path would honour only its
    // boot-time env read and silently render one mode for every variant of a
    // warm-host comparison — the labelled-but-identical failure the axis
    // contract exists to prevent.
    for (_, material) in tile_materials.iter_mut() {
        material.extension.params.inspect = inspection as u32;
    }
    info!(
        target: "thalos::diagnostic::capture",
        event = "cloud_probe_configuration",
        quality = cfg.cloud_quality.name(),
        view_steps = clouds.clouds_raymarch_steps_count,
        shadow_steps = clouds.clouds_shadow_raymarch_steps_count,
        reconstruction,
        coverage = clouds.clouds_coverage,
        "cloud probe configuration"
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
    runtime: Res<CaptureRuntimeOverrides>,
    mut over: ResMut<crate::rendering::plume::PlumeDebugOverride>,
) {
    if !cfg.is_changed() && !runtime.is_changed() {
        return;
    }
    if !matches!(
        cfg.preset,
        ScreenshotPreset::Plume | ScreenshotPreset::PlumeSkyline
    ) {
        over.throttle = None;
        over.ambient_pressure_pa = None;
        over.ignition = None;
        return;
    }
    // `PlumeSkyline` flies inside the atmosphere at a known altitude, so its
    // back-pressure is already deterministic — pinning it would only decouple
    // the probe from the air the composites are integrating through.
    let pressure = match cfg.preset {
        ScreenshotPreset::PlumeSkyline => None,
        _ => Some(
            runtime
                .get("THALOS_PLUME_PRESSURE")
                .and_then(|raw| raw.trim().parse::<f32>().ok())
                .or_else(|| env_parse::<f32>("THALOS_PLUME_PRESSURE"))
                .unwrap_or(101_325.0)
                .max(0.0),
        ),
    };
    over.throttle = Some(1.0);
    over.ambient_pressure_pa = pressure;
    over.ignition = Some(1.0);
    info!(
        target: "thalos::diagnostic::capture",
        event = "plume_probe_configuration",
        throttle = 1.0,
        ambient_pressure_pa = pressure.unwrap_or(f32::NAN),
        pressure_source = if pressure.is_some() { "pinned" } else { "atmosphere" },
        "plume probe configuration"
    );
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
        target: "thalos::diagnostic::capture",
        event = "apron_probe",
        view_distance_km = view_dist_km,
        samples = %lods.join(" "),
        "apron probe"
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
    readiness: TerrainReadiness,
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
        )
        .map(CaptureFocus::from),
        ScreenshotPreset::ForestStand => forest_site_context(
            &sim,
            &solar,
            &height_sources,
            homeworld.0,
            &mut driver.forest_site_dir,
        )
        .map(CaptureFocus::from),
        ScreenshotPreset::Ocean | ScreenshotPreset::OceanSlopes => ocean_site_context(
            &sim,
            &solar,
            &height_sources,
            homeworld.0,
            &mut driver.ocean_site_dir,
        )
        .map(CaptureFocus::from),
        ScreenshotPreset::MiraOrbit
        | ScreenshotPreset::MiraSurface
        | ScreenshotPreset::MiraDisc
        | ScreenshotPreset::MiraApproach
        | ScreenshotPreset::MiraRim => daylight_surface_context(
            &sim,
            &solar,
            &height_sources,
            homeworld.0,
            &surfaces,
            &mut driver.airless_site,
            match cfg.preset {
                // The disc aims at the body centre, so the site only decides
                // which face turns toward the camera — keep it well lit.
                ScreenshotPreset::MiraDisc => AirlessSunGeometry::SubSolar,
                // Grazing light for long rim shadows across the macro bands.
                ScreenshotPreset::MiraApproach => AirlessSunGeometry::Grazing,
                // The rim probe sits back in `Oblique`: grazing light at close
                // range buries the floor and central peak in shadow and drives
                // the BL-33 speckle to its worst (MIRA-V6). Large craters are
                // reachable in this band now that the landmark set is
                // size-stratified rather than age-gated.
                _ => AirlessSunGeometry::Oblique,
            },
            match cfg.preset {
                // The rim framing must *contain* a crater, so it needs the
                // biggest one available rather than the registry's typical
                // ~10 km pick, which is too small to read as a rim from low.
                ScreenshotPreset::MiraRim => LandmarkChoice::MostLegible,
                _ => LandmarkChoice::Typical,
            },
        ),
        ScreenshotPreset::MiraEva => {
            eva_surface_context(&sim, &solar, &height_sources, homeworld.0).map(CaptureFocus::from)
        }
        // Frame the craft itself, not a surface site.
        ScreenshotPreset::Plume => craft_context(&sim).map(CaptureFocus::from),
        ScreenshotPreset::PlumeSkyline => {
            plume_skyline_context(&sim, &solar).map(CaptureFocus::from)
        }
        ScreenshotPreset::MassifAerial
        | ScreenshotPreset::MassifRidge
        | ScreenshotPreset::MassifValley => {
            let (lat_deg, lon_deg) = massif_site(cfg.preset);
            fixed_site_context(&sim, &solar, &height_sources, homeworld.0, lat_deg, lon_deg)
                .map(CaptureFocus::from)
        }
        _ => match cfg.framing {
            // The sun-relative and disc framings are only reachable from the
            // airless presets above, which resolve their own site; a non-Mira
            // preset adopting one still gets a sane focus here.
            ScreenshotFraming::GodView
            | ScreenshotFraming::SunRelativeGodView { .. }
            | ScreenshotFraming::BodyDisc { .. }
            | ScreenshotFraming::LocalCloud {
                site_sun_elevation_deg: None,
                ..
            } => hub_context(&sim, &solar, &height_sources, &registry, homeworld.0)
                .map(CaptureFocus::from),
            ScreenshotFraming::LocalCloud {
                site_sun_elevation_deg: Some(sun_elevation_deg),
                ..
            } => cloud_site_context(
                &sim,
                &solar,
                &height_sources,
                homeworld.0,
                sun_elevation_deg,
            )
            .map(CaptureFocus::from),
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

    if let Some(viewpoint) = pose_cfg.viewpoint.as_ref() {
        if let Err(error) = crate::viewpoints::pose_viewpoint(
            viewpoint,
            &sim.system.bodies,
            &solar,
            root,
            &mut cell,
            &mut transform,
            &mut projection,
        ) {
            warn!(target: "thalos::screenshot", "could not pose viewpoint: {error}");
            return;
        }
    } else if pose_cfg.preset == ScreenshotPreset::MiraEva {
        pose_eva_camera(pose_cfg, &ctx.hub, root, &mut transform, &mut cell);
    } else {
        pose_camera(
            pose_cfg,
            &ctx,
            &sim,
            &solar,
            root,
            &mut transform,
            &mut cell,
        );
    }

    if driver.captured {
        if persistent {
            return;
        }
        // Screenshot readback is asynchronous. Do not start the fixed flush
        // tail until Bevy has removed the `Capturing` marker; on slower cold
        // runs the old 24-frame countdown could exit first, close the result
        // channel, and invalidate an otherwise healthy render.
        if !readiness.active_captures.is_empty() {
            return;
        }
        driver.tail += 1;
        if driver.tail >= cfg.tail_frames {
            info!(target: "thalos::screenshot", "headless capture flushed — exiting");
            exit.write(AppExit::Success);
        }
        return;
    }

    // Massif presets: hold the warmup countdown until the streamed ground at
    // the (fixed, cold, wild) site has actually refined — frame counts alone
    // fire while only the coarse mip pyramid is resident, which renders the
    // 5.8 km massif as soft ~800 m hills (the 2026-07-24 baseline failure).
    if matches!(
        cfg.preset,
        ScreenshotPreset::MassifAerial
            | ScreenshotPreset::MassifRidge
            | ScreenshotPreset::MassifValley
    ) && !driver.terrain_ready
    {
        let dt = readiness.time.delta_secs_f64();
        let ready = if crate::rendering::tile_terrain::tile_renderer_enabled() {
            // Tile path: the eye follows the posed camera, so "the desired
            // selection is fully resident" is exactly "the site is streamed".
            readiness
                .tile_roots
                .iter()
                .find(|(_, b)| b.body_id == homeworld.0)
                .is_some_and(|(root, _)| root.coverage_ready() && root.settled())
        } else {
            // udlod path: wall-clock plateau of the finest resident LOD at
            // the streamer's view position (mirrors `surface_settle`).
            match crate::surface_settle::resident_lod_under_view(
                &sim,
                &readiness.udlod_trees,
                &readiness.udlod_terrains,
                &readiness.camera_q,
            ) {
                Some(m)
                    if driver
                        .terrain_best_lod_m
                        .is_none_or(|best| m < best * 0.999) =>
                {
                    driver.terrain_best_lod_m = Some(m);
                    driver.terrain_lod_hold_s = 0.0;
                    false
                }
                Some(_) => {
                    driver.terrain_lod_hold_s += dt;
                    driver.terrain_lod_hold_s >= MASSIF_LOD_PLATEAU_S
                }
                None => false,
            }
        };
        if ready {
            driver.terrain_ready = true;
            info!(
                target: "thalos::screenshot",
                "massif terrain streamed ({:.1} s) — starting warmup",
                driver.terrain_wait_s,
            );
        } else {
            driver.terrain_wait_s += dt;
            if driver.terrain_wait_s < MASSIF_STREAM_TIMEOUT_S {
                return;
            }
            driver.terrain_ready = true;
            warn!(
                target: "thalos::screenshot",
                "massif terrain still streaming after {:.0} s — capturing anyway",
                driver.terrain_wait_s,
            );
        }
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
        cfg.scene_name(),
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

/// Pose one catalog-backed scripted agent view in the running game.
///
/// This reuses the same focus search and framing code as headless capture. It
/// intentionally changes only the camera: diagnostic render overrides (slope
/// false-colour, plume pressure, cloud coverage, temporal slew) remain capture
/// concerns and are described in the manager status.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pose_scripted_viewpoint(
    driver: &str,
    sim: &SimulationState,
    solar: &SolarSystemState,
    height_sources: &HeightSourceRegistry,
    registry: &StructureRegistry,
    surfaces: &BodySurfaceRegistry,
    root: &Grid,
    cell: &mut CellCoord,
    transform: &mut Transform,
) -> Result<(BodyId, String), String> {
    let preset = ScreenshotPreset::try_parse(driver)
        .filter(|preset| *preset != ScreenshotPreset::LatestPerspective)
        .ok_or_else(|| format!("unknown scripted viewpoint driver {driver:?}"))?;
    let cfg = preset.defaults();
    let body_id = sim
        .system
        .name_to_id
        .get(preset.target_body_name())
        .copied()
        .ok_or_else(|| format!("body {:?} is not authored", preset.target_body_name()))?;
    let mut dry_site = None;
    let mut forest_site = None;
    let mut ocean_site = None;
    let mut airless_site = None;
    let focus = match preset {
        ScreenshotPreset::DryBelt => {
            dry_site_context(sim, solar, height_sources, body_id, &mut dry_site)
                .map(CaptureFocus::from)
        }
        ScreenshotPreset::ForestStand => {
            forest_site_context(sim, solar, height_sources, body_id, &mut forest_site)
                .map(CaptureFocus::from)
        }
        ScreenshotPreset::Ocean | ScreenshotPreset::OceanSlopes => {
            ocean_site_context(sim, solar, height_sources, body_id, &mut ocean_site)
                .map(CaptureFocus::from)
        }
        ScreenshotPreset::MiraOrbit
        | ScreenshotPreset::MiraSurface
        | ScreenshotPreset::MiraDisc
        | ScreenshotPreset::MiraApproach
        | ScreenshotPreset::MiraRim => daylight_surface_context(
            sim,
            solar,
            height_sources,
            body_id,
            surfaces,
            &mut airless_site,
            match preset {
                ScreenshotPreset::MiraDisc => AirlessSunGeometry::SubSolar,
                ScreenshotPreset::MiraApproach => AirlessSunGeometry::Grazing,
                _ => AirlessSunGeometry::Oblique,
            },
            if preset == ScreenshotPreset::MiraRim {
                LandmarkChoice::MostLegible
            } else {
                LandmarkChoice::Typical
            },
        ),
        ScreenshotPreset::MiraEva => {
            eva_surface_context(sim, solar, height_sources, body_id).map(CaptureFocus::from)
        }
        ScreenshotPreset::Plume => craft_context(sim).map(CaptureFocus::from),
        ScreenshotPreset::PlumeSkyline => plume_skyline_context(sim, solar).map(CaptureFocus::from),
        ScreenshotPreset::MassifAerial
        | ScreenshotPreset::MassifRidge
        | ScreenshotPreset::MassifValley => {
            let (lat_deg, lon_deg) = massif_site(preset);
            fixed_site_context(sim, solar, height_sources, body_id, lat_deg, lon_deg)
                .map(CaptureFocus::from)
        }
        _ => match cfg.framing {
            ScreenshotFraming::GodView
            | ScreenshotFraming::SunRelativeGodView { .. }
            | ScreenshotFraming::BodyDisc { .. }
            | ScreenshotFraming::LocalCloud {
                site_sun_elevation_deg: None,
                ..
            } => hub_context(sim, solar, height_sources, registry, body_id).map(CaptureFocus::from),
            ScreenshotFraming::LocalCloud {
                site_sun_elevation_deg: Some(sun_elevation_deg),
                ..
            } => cloud_site_context(sim, solar, height_sources, body_id, sun_elevation_deg)
                .map(CaptureFocus::from),
        },
    }
    .ok_or_else(|| format!("scripted viewpoint {driver:?} is not ready in this world"))?;

    if preset == ScreenshotPreset::MiraEva {
        pose_eva_camera(&cfg, &focus.hub, root, transform, cell);
    } else {
        pose_camera(&cfg, &focus, sim, solar, root, transform, cell);
    }
    Ok((
        body_id,
        format!(
            "Viewing scripted agent viewpoint {driver}; capture-only diagnostic overrides are unchanged"
        ),
    ))
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
    // Height sources / site searches work in the SURFACE body-fixed frame (the
    // frame the terrain renders in) — for tidally-locked moons that is NOT the
    // ephemeris orientation. One authority: `surface_orientation_authored`.
    let surface_q = crate::rendering::transforms::surface_orientation_authored(
        &sim.system.bodies,
        body_id,
        states,
    )
    .unwrap_or_else(|| body_state.orientation.normalize());
    let sun_world = (-body_state.position).normalize_or_zero();
    let sun_world = if sun_world == DVec3::ZERO {
        DVec3::Y
    } else {
        sun_world
    };

    let dir_body = match *cached_dir {
        Some(dir) => dir,
        None => {
            let dir = find_ocean_site(hs.as_ref(), surface_q, sun_world);
            let up_world = (surface_q * dir).normalize();
            let sun_elevation_deg = up_world.dot(sun_world).clamp(-1.0, 1.0).asin().to_degrees();
            let depth_m = -hs
                .sample_height_m(dir.as_vec3(), OCEAN_SITE_LOD_M)
                .unwrap_or(0.0);
            info!(
                target: "thalos::diagnostic::capture",
                event = "ocean_site",
                depth_m,
                sun_elevation_deg,
                "ocean capture site"
            );
            *cached_dir = Some(dir);
            dir
        }
    };

    let up_world = (surface_q * dir_body).normalize();
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
    cached_site: &mut Option<(DVec3, Option<f64>)>,
    geometry: AirlessSunGeometry,
    choice: LandmarkChoice,
) -> Option<CaptureFocus> {
    let states = solar.states.as_deref()?;
    let body_state = states.get(body_id)?;
    let radius_m = sim.system.bodies.get(body_id)?.radius_m;
    let hs = height_sources.get(body_id)?;
    // Landmark dirs + height sources live in the SURFACE frame (see the
    // ocean-context note) — using the ephemeris orientation on a tidally
    // locked moon posed the rim camera ~130° of longitude away from its
    // crater.
    let surface_q = crate::rendering::transforms::surface_orientation_authored(
        &sim.system.bodies,
        body_id,
        states,
    )
    .unwrap_or_else(|| body_state.orientation.normalize());
    let sun_inertial = (-body_state.position).normalize_or_zero();
    let sun_world = if sun_inertial == DVec3::ZERO {
        DVec3::Y
    } else {
        sun_inertial
    };
    let sun_body = (surface_q.inverse() * sun_world).normalize();
    let (dir_body, landmark_radius_m) = match *cached_site {
        Some(site) => site,
        None => {
            let site = find_rugged_airless_site(
                hs.as_ref(),
                sun_body,
                surfaces.airless_landmarks(body_id),
                geometry,
                choice,
            );
            let incidence_deg = site.0.dot(sun_body).clamp(-1.0, 1.0).acos().to_degrees();
            info!(
                target: "thalos::diagnostic::capture",
                event = "airless_survey_site",
                solar_incidence_deg = incidence_deg,
                geometry = ?geometry,
                "airless survey capture site"
            );
            *cached_site = Some(site);
            site
        }
    };
    let up_world = (surface_q * dir_body).normalize();
    let height_m = hs
        .sample_height_m(dir_body.as_vec3(), DRY_SITE_LOD_M)
        .unwrap_or(0.0) as f64;
    let surface_r = radius_m + height_m;
    Some(CaptureFocus {
        hub: HubContext {
            body_id,
            center_world: body_state.position + up_world * surface_r,
            up_world,
            pad_r: surface_r,
        },
        landmark_radius_m,
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

    let surface_q = crate::rendering::transforms::surface_orientation_authored(
        &sim.system.bodies,
        body_id,
        states,
    )
    .unwrap_or_else(|| body_state.orientation.normalize());
    let craft_world = sim.simulation.craft_state().translation.position;
    let up_world = (craft_world - body_state.position)
        .try_normalize()
        .unwrap_or_else(|| (-body_state.position).try_normalize().unwrap_or(DVec3::Y));
    let dir_body = (surface_q.inverse() * up_world).normalize();
    let height_m = height_source
        .sample_height_m(dir_body.as_vec3(), 1.0)
        .unwrap_or(0.0) as f64;
    let surface_r = radius_m + height_m;

    // The EVA site is a *fixed* body-fixed direction (the canonical spawn), with
    // no sun constraint — unlike every other airless preset, which searches a
    // lighting band. Log the resulting incidence so a black frame can be told
    // apart from a render bug (BL-34).
    static EVA_SUN_LOGGED: std::sync::Once = std::sync::Once::new();
    EVA_SUN_LOGGED.call_once(|| {
        if let Some(sun_body) =
            sun_direction_world(solar, body_id).map(|sun| (surface_q.inverse() * sun).normalize())
        {
            let incidence_deg = dir_body.dot(sun_body).clamp(-1.0, 1.0).acos().to_degrees();
            info!(
                target: "thalos::diagnostic::capture",
                event = "eva_site",
                solar_incidence_deg = incidence_deg,
                illumination = if incidence_deg >= 90.0 { "night" } else { "lit" },
                "EVA capture site"
            );
        }
    });
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

/// Focus context centered on the player craft with **local up** as the pole,
/// for [`ScreenshotPreset::PlumeSkyline`].
///
/// [`craft_context`] poles on the ship's nose, which is right for a hero shot
/// orbiting the stack but wrong here twice over: the horizon rolls with the
/// craft's attitude, and — the part that matters — `elevation_deg` then means
/// "angle off the craft's waist" rather than "angle off the local horizontal".
/// This probe exists to hold the camera at a known pitch *above the horizontal*,
/// because that is the regime where the fullscreen composites can sort ahead of
/// world transparency (see [`thalos_body_render::composite_order`]). Poling on
/// the radial makes the framing say exactly that, and keeps it true no matter
/// how the descending craft happens to be pointed.
fn plume_skyline_context(sim: &SimulationState, solar: &SolarSystemState) -> Option<HubContext> {
    /// Shift the focus down the stack toward the bell, as [`craft_context`]
    /// does, so the column rather than the pod centres the frame.
    const FOCUS_OFFSET_M: f64 = -4.0;
    let body_id = sim.simulation.dominant_body();
    // No body state yet (the frame the solar cache has not filled): fall back to
    // the nose-poled hero framing rather than returning no focus at all. It
    // rolls with the craft, but a rolled frame still exercises the ordering —
    // an unposed capture is simply lost.
    let Some(body_state) = solar
        .states
        .as_deref()
        .and_then(|states| states.get(body_id))
    else {
        return craft_context(sim);
    };
    let craft = sim.simulation.ship_state().position;
    let up_world = match (craft - body_state.position).try_normalize() {
        Some(up) => up,
        None => return craft_context(sim),
    };
    let nose = (sim.simulation.attitude().orientation * DVec3::Y)
        .try_normalize()
        .unwrap_or(DVec3::Y);
    Some(HubContext {
        body_id,
        center_world: craft + nose * FOCUS_OFFSET_M,
        up_world,
        pad_r: 0.0,
    })
}

/// Find a visibly structured airless site while keeping the sun oblique enough
/// for relief and Hapke backscatter to read. Candidates use a Fibonacci sphere;
/// each is scored by the elevation range of a ~25 km neighborhood.
/// What solar geometry an airless capture wants at its focus site.
///
/// The Mira reference framings differ mainly in **where the sun is**, because
/// that decides what carries the image: a near-full disc needs the sun behind
/// the camera so albedo provinces read, while an oblique approach needs grazing
/// light so rim shadows do. Expressed as an acceptance band on
/// `dot(site_dir, sun_dir)` — `1.0` is sub-solar, `0.0` is the terminator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AirlessSunGeometry {
    /// Sun near-overhead (≲26° incidence). Flat, shadow-poor light in which
    /// relief nearly vanishes and albedo province structure dominates.
    SubSolar,
    /// 41–72° incidence — clearly lit, but far enough from noon to reveal
    /// crater walls and exercise Hapke's angular response.
    Oblique,
    /// 72–85° incidence. Grazing light: long rim shadows, maximum relief
    /// legibility, and the regime where terrain self-shadowing shows.
    Grazing,
}

impl AirlessSunGeometry {
    /// Acceptance band on `dot(site_dir, sun_dir)`.
    ///
    /// Kept away from `dot ≈ 0` even for [`Self::Grazing`]: at the terminator
    /// itself the site is too dark to verify anything.
    fn light_band(self) -> std::ops::RangeInclusive<f64> {
        match self {
            Self::SubSolar => 0.90..=1.0,
            Self::Oblique => 0.30..=0.75,
            Self::Grazing => 0.09..=0.31,
        }
    }
}

/// How the airless site search picks among landmark craters inside the lighting
/// band.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LandmarkChoice {
    /// First acceptable landmark in registry order. The registry sorts toward
    /// ~10 km radius, so this yields a typical mid-size crater — what the
    /// existing survey probes are calibrated against.
    Typical,
    /// The landmark with the most *surviving relief*. Required by framings that
    /// must contain a crater rather than survey terrain near one. Ranked on
    /// rendered depth rather than radius, because the widest feature on an old
    /// body is routinely one that degradation has flattened.
    MostLegible,
}

fn find_rugged_airless_site(
    source: &dyn HeightSource,
    sun_dir: DVec3,
    landmarks: &[AirlessLandmark],
    geometry: AirlessSunGeometry,
    choice: LandmarkChoice,
) -> (DVec3, Option<f64>) {
    let band = geometry.light_band();
    let in_band = || {
        landmarks
            .iter()
            .filter(|landmark| band.contains(&landmark.dir.dot(sun_dir)))
    };
    let landmark = match choice {
        // Relief is measured from the surface now (see `airless_landmarks`),
        // so skip fully-degraded ghosts — the first in-band entry can be a
        // crater the bake relaxed to a plain (measured ~0 m), which frames
        // featureless ground.
        LandmarkChoice::Typical => in_band()
            .find(|landmark| landmark.relief_m > 500.0)
            .or_else(|| in_band().next()),
        LandmarkChoice::MostLegible => in_band().max_by(|a, b| a.relief_m.total_cmp(&b.relief_m)),
    };
    if let Some(landmark) = landmark {
        info!(
            target: "thalos::diagnostic::capture",
            event = "airless_landmark",
            radius_m = landmark.radius_m,
            relief_m = landmark.relief_m,
            choice = ?choice,
            "airless landmark selected"
        );
        return (landmark.dir, Some(f64::from(landmark.radius_m)));
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
        if !band.contains(&dir.dot(sun_dir)) {
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
    (best, None)
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
    if let Some(viewpoint) = cfg.viewpoint.as_ref() {
        return serde_json::json!({
            "kind": "viewpoint",
            "id": viewpoint.id.as_str(),
            "body": viewpoint.body.as_str(),
            "position_body_m": viewpoint.camera_position_body_m,
            "rotation_body_xyzw": viewpoint.camera_rotation_body_xyzw,
            "vertical_fov_rad": viewpoint.vertical_fov_rad,
            "recorded_sim_time_s": viewpoint.sim_time_s,
        })
        .to_string();
    }
    match cfg.framing {
        ScreenshotFraming::GodView => format!(
            "{{\"kind\":\"god_view\",\"azimuth_deg\":{:.4},\"elevation_deg\":{:.4},\"distance_m\":{:.3}}}",
            cfg.azimuth_deg, cfg.elevation_deg, cfg.distance_m
        ),
        ScreenshotFraming::SunRelativeGodView {
            sun_azimuth_deg,
            dark_side,
            landmark_radii,
        } => format!(
            "{{\"kind\":\"sun_relative_god_view\",\"sun_azimuth_deg\":{sun_azimuth_deg:.4},\"dark_side\":\"{dark_side:?}\",\"landmark_radii\":{},\"elevation_deg\":{:.4},\"distance_m\":{:.3}}}",
            landmark_radii
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "null".to_string()),
            cfg.elevation_deg,
            cfg.distance_m
        ),
        ScreenshotFraming::BodyDisc {
            phase_deg,
            dark_side,
        } => format!(
            "{{\"kind\":\"body_disc\",\"phase_deg\":{phase_deg:.4},\"dark_side\":\"{dark_side:?}\",\"distance_m\":{:.3}}}",
            cfg.distance_m
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
            "\"coverage_bytes\":{},\"surface_density_bytes\":{},",
            "\"total_bytes\":{}}}}}\n"
        ),
        unix_ms,
        cfg.scene_name(),
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
        memory.surface_density_bytes,
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
/// Unit world direction from `body` toward the sun.
///
/// Bodies orbit the origin-centred star, so the direction home from the body is
/// the sun direction. One definition, shared by every sun-relative framing.
fn sun_direction_world(solar: &SolarSystemState, body_id: BodyId) -> Option<DVec3> {
    let position = solar.states.as_deref()?.get(body_id)?.position;
    (-position).try_normalize()
}

fn pose_camera(
    cfg: &ScreenshotConfig,
    focus: &CaptureFocus,
    sim: &SimulationState,
    solar: &SolarSystemState,
    root: &Grid,
    transform: &mut Transform,
    cell: &mut CellCoord,
) {
    let ctx = &focus.hub;
    match cfg.framing {
        ScreenshotFraming::GodView => {
            pose_god_view_camera(cfg, ctx, root, transform, cell);
        }
        ScreenshotFraming::SunRelativeGodView {
            sun_azimuth_deg,
            dark_side,
            landmark_radii,
        } => pose_sun_relative_god_view(
            cfg,
            focus,
            solar,
            root,
            transform,
            cell,
            sun_azimuth_deg,
            dark_side,
            landmark_radii,
        ),
        ScreenshotFraming::BodyDisc {
            phase_deg,
            dark_side,
        } => pose_body_disc_camera(
            cfg, ctx, sim, solar, root, transform, cell, phase_deg, dark_side,
        ),
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

/// Screen-space right vector for a camera looking along `forward` with `up`.
///
/// Bevy cameras look down local `-Z` with `+X` right and `+Y` up, so
/// `right = forward × up` in the right-handed basis `looking_to` builds.
fn screen_right(forward: DVec3, up: DVec3) -> DVec3 {
    forward.cross(up).normalize_or_zero()
}

/// `GodView`, but the boom bearing is measured from the sun and the terminator
/// is pinned to a requested side of frame.
///
/// The mirror ambiguity (see [`FrameSide`]) is resolved by evaluating both
/// bearings and keeping the one that lands the sun on the opposite side from the
/// requested dark half — never by nudging an absolute azimuth, which would only
/// hold for one site.
#[allow(clippy::too_many_arguments)]
fn pose_sun_relative_god_view(
    cfg: &ScreenshotConfig,
    focus: &CaptureFocus,
    solar: &SolarSystemState,
    root: &Grid,
    transform: &mut Transform,
    cell: &mut CellCoord,
    sun_azimuth_deg: f32,
    dark_side: FrameSide,
    landmark_radii: Option<f32>,
) {
    let ctx = &focus.hub;
    let up = ctx.up_world;
    let boom_m = match (landmark_radii, focus.landmark_radius_m) {
        (Some(radii), Some(radius_m)) => (radius_m * f64::from(radii)).max(1_000.0),
        _ => cfg.distance_m,
    };
    let Some(sun_world) = sun_direction_world(solar, ctx.body_id) else {
        // No solar state yet — fall back rather than pose from a zero vector.
        pose_god_view_camera(cfg, ctx, root, transform, cell);
        return;
    };

    // Sun bearing projected onto the site's horizon. Degenerate only with the
    // sun exactly overhead, where every bearing is equivalent anyway.
    let sun_horizon = (sun_world - up * sun_world.dot(up))
        .try_normalize()
        .unwrap_or_else(|| {
            let seed = if up.dot(DVec3::Y).abs() < 0.99 {
                DVec3::Y
            } else {
                DVec3::X
            };
            seed.cross(up).normalize()
        });
    let sun_perp = up.cross(sun_horizon).normalize();

    let az = (sun_azimuth_deg as f64).to_radians();
    let elev = (cfg.elevation_deg as f64).to_radians();

    // The two mirror-image bearings about the sun/up plane. Both give identical
    // lighting character; they differ only in which side of frame goes dark.
    let mut chosen = None;
    for sign in [1.0_f64, -1.0] {
        let horiz = sun_horizon * az.cos() + sun_perp * (az.sin() * sign);
        let offset_dir = horiz * elev.cos() + up * elev.sin();
        let camera_world = ctx.center_world + offset_dir * boom_m;
        let forward = (ctx.center_world - camera_world).normalize();
        let look_up = if forward.dot(up).abs() > 0.99 {
            sun_perp
        } else {
            up
        };
        let right = screen_right(forward, look_up);
        let sun_on_right = sun_world.dot(right) > 0.0;
        // Dark half is opposite the sun.
        let matches_request = match dark_side {
            FrameSide::Left => sun_on_right,
            FrameSide::Right => !sun_on_right,
        };
        if matches_request || chosen.is_none() {
            chosen = Some((camera_world, forward, look_up));
            if matches_request {
                break;
            }
        }
    }

    let Some((camera_world, forward, look_up)) = chosen else {
        return;
    };
    let (next_cell, local) = root.translation_to_grid(camera_world);
    *cell = next_cell;
    *transform =
        Transform::from_translation(local).looking_to(forward.as_vec3(), look_up.as_vec3());
}

/// Frame the whole body as a disc at a chosen phase angle, rolled so the
/// terminator runs vertically.
///
/// Aims at the body centre, so the disc is centred no matter which site the
/// search picked. The camera's up axis is the sun-rotation axis, which puts the
/// sun in the screen-horizontal plane — that is what makes the terminator
/// vertical rather than an arbitrary diagonal.
#[allow(clippy::too_many_arguments)]
fn pose_body_disc_camera(
    cfg: &ScreenshotConfig,
    ctx: &HubContext,
    sim: &SimulationState,
    solar: &SolarSystemState,
    root: &Grid,
    transform: &mut Transform,
    cell: &mut CellCoord,
    phase_deg: f32,
    dark_side: FrameSide,
) {
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let Some(body_state) = states.get(ctx.body_id) else {
        return;
    };
    let center = body_state.position;
    let Some(sun_world) = sun_direction_world(solar, ctx.body_id) else {
        return;
    };

    // Which hemisphere to present.
    //
    // The phase angle fixes how *much* of the disc is lit but not *which* face
    // turns toward the camera, and the two are independent — so left free, the
    // framing lands on whatever hemisphere the site search happened to pick. For
    // a tidally locked moon the interesting one is the near side: that is where
    // the authored mare provinces live (`procellarum`'s near-side half-angle),
    // and a mare-poor far side is exactly why an earlier disc capture read as a
    // near-uniform grey ball. Aim at the parent when there is one.
    let face_target = sim
        .system
        .bodies
        .get(ctx.body_id)
        .and_then(|body| body.parent)
        .and_then(|parent| states.get(parent))
        .and_then(|parent| (parent.position - center).try_normalize());

    let phase = (phase_deg as f64).to_radians();

    // Camera direction on the phase cone about the sun. With a face target we
    // take the point on that cone closest to it — decompose the target into its
    // sun-parallel and perpendicular parts and rebuild at exactly `phase`, which
    // maximises how much of the wanted face is visible without giving up the
    // requested lighting. Without one, any perpendicular seed will do.
    let perp_seed = face_target
        .map(|target| target - sun_world * target.dot(sun_world))
        .and_then(|perp| perp.try_normalize())
        .unwrap_or_else(|| {
            let spin_axis = (body_state.orientation * DVec3::Y).normalize_or_zero();
            sun_world
                .cross(spin_axis)
                .cross(sun_world)
                .try_normalize()
                .unwrap_or_else(|| sun_world.cross(DVec3::X).cross(sun_world).normalize())
        });

    // `distance_m` is measured from the surface, matching every other framing.
    let range = ctx.pad_r + cfg.distance_m;

    // View direction is fixed by the face target and the phase — flipping the
    // perpendicular instead would swing the camera to the *other* hemisphere and
    // throw the face away.
    let cam_dir = (sun_world * phase.cos() + perp_seed * phase.sin()).normalize();
    let camera_world = center + cam_dir * range;
    let forward = (center - camera_world).normalize();

    // Which side goes dark is a roll choice, not a position one. The up axis must
    // stay ⊥ to both sun and view (that is what keeps the terminator vertical),
    // which leaves exactly two options 180° apart; they mirror the frame.
    //
    // Flipping the *perpendicular* cannot do this: it flips the derived axis too,
    // so `right` comes out identical and both candidates land the sun on the same
    // side — which silently ignored the request until a capture showed it.
    let Some(axis_seed) = sun_world.cross(cam_dir).try_normalize() else {
        return;
    };
    let sun_on_right = sun_world.dot(screen_right(forward, axis_seed)) > 0.0;
    let want_sun_on_right = matches!(dark_side, FrameSide::Left);
    let axis = if sun_on_right == want_sun_on_right {
        axis_seed
    } else {
        -axis_seed
    };
    let (next_cell, local) = root.translation_to_grid(camera_world);
    *cell = next_cell;
    *transform = Transform::from_translation(local).looking_to(forward.as_vec3(), axis.as_vec3());
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

/// Fixed showcase sites for the NTR-X4 massif presets, derived offline from
/// the diffusion 90 m detail raster
/// (`assets/terrain_packages/thalos_diffusion/thalos_site_detail_6144_90m.f32`,
/// tangent-drape pixel→lat/lon per `DiffusionSurface::detail_px`):
///
/// - **peak**: the NE massif's 5799 m summit at raster px (3302, 2515) —
///   20.7 km E / 50.1 km N of the spaceport site (7.6, 178).
/// - **valley face**: 40 % of the way up from the 1836 m valley floor at
///   px (3188, 2382) toward the peak, i.e. a focus on the massif's NW face;
///   the valley→peak bearing is 139° true.
///
/// Re-derive with the raster scan if the diffusion window is re-exported
/// (`scratch: find_massif.py` recipe lives in the NTR-X4 notes; the peak is
/// simply the regional argmax, so any rescan reproduces it).
fn massif_site(preset: ScreenshotPreset) -> (f64, f64) {
    match preset {
        ScreenshotPreset::MassifValley => (8.6307, 178.2639),
        _ => (8.5015, 178.3756),
    }
}

/// Focus for the fixed-lat/lon showcase presets: no site search, just the
/// authored direction resolved in the surface body-fixed frame with the local
/// terrain height sampled under it. Mirrors [`dry_site_context`]'s output
/// shape; deterministic by construction, so nothing is cached.
fn fixed_site_context(
    sim: &SimulationState,
    solar: &SolarSystemState,
    height_sources: &HeightSourceRegistry,
    body_id: BodyId,
    lat_deg: f64,
    lon_deg: f64,
) -> Option<HubContext> {
    let states = solar.states.as_deref()?;
    let body_state = states.get(body_id)?;
    let radius_m = sim.system.bodies.get(body_id)?.radius_m;
    let hs = height_sources.get(body_id)?;
    let surface_q = crate::rendering::transforms::surface_orientation_authored(
        &sim.system.bodies,
        body_id,
        states,
    )
    .unwrap_or_else(|| body_state.orientation.normalize());

    // Same lat/lon → direction convention as `DiffusionSurface` (lat from
    // `dir.y`, lon = atan2(z, x)), so raster-derived coordinates land exactly.
    let (lat_r, lon_r) = (lat_deg.to_radians(), lon_deg.to_radians());
    let dir_body = DVec3::new(
        lat_r.cos() * lon_r.cos(),
        lat_r.sin(),
        lat_r.cos() * lon_r.sin(),
    )
    .normalize();

    let up_world = (surface_q * dir_body).normalize();
    let height_m = hs
        .sample_height_m(dir_body.as_vec3(), DRY_SITE_LOD_M)
        .unwrap_or(0.0)
        .max(0.0) as f64;
    let pad_r = radius_m + height_m;
    // One line per capture session (the context is re-resolved every frame —
    // gate on first resolve via the height being sampled once is overkill;
    // dedup happens in the log reader). Kept cheap and INFO so the capture log
    // records which backing/height the fixed site actually resolved to.
    static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    LOGGED.get_or_init(|| {
        info!(
            target: "thalos::diagnostic::capture",
            event = "fixed_site",
            lat_deg,
            lon_deg,
            height_m,
            diffusion = crate::terrain_registry::thalos_diffusion_enabled(),
            "fixed capture site"
        );
    });
    Some(HubContext {
        body_id,
        center_world: body_state.position + up_world * pad_r,
        up_world,
        pad_r,
    })
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
    let surface_q = crate::rendering::transforms::surface_orientation_authored(
        &sim.system.bodies,
        body_id,
        states,
    )
    .unwrap_or_else(|| body_state.orientation.normalize());

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
                (surface_q.inverse() * sun_inertial).normalize()
            };
            let d = find_driest_site(hs.as_ref(), sun_body);
            *cached_dir = Some(d);
            let moisture = hs.landcover_moisture(d);
            let lat_deg = d.y.clamp(-1.0, 1.0).asin().to_degrees();
            info!(
                target: "thalos::diagnostic::capture",
                event = "dry_belt_site",
                lat_deg,
                macro_moisture = moisture,
                "dry-belt capture site"
            );
            d
        }
    };

    let up_world = (surface_q * dir_body).normalize();
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

/// Focus for the [`ScreenshotPreset::ForestStand`] colour-coupling probe: a
/// wet, stand-covered sunlit land site, framed at the surface. Mirrors
/// [`dry_site_context`] exactly except for the score being maximised.
fn forest_site_context(
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
    let surface_q = crate::rendering::transforms::surface_orientation_authored(
        &sim.system.bodies,
        body_id,
        states,
    )
    .unwrap_or_else(|| body_state.orientation.normalize());

    let dir_body = match *cached_dir {
        Some(d) => d,
        None => {
            let sun_inertial = (-body_state.position).normalize_or_zero();
            let sun_body = if sun_inertial == DVec3::ZERO {
                DVec3::Y
            } else {
                (surface_q.inverse() * sun_inertial).normalize()
            };
            let d = find_forest_site(hs.as_ref(), sun_body);
            *cached_dir = Some(d);
            let moisture = hs.landcover_moisture(d);
            let site_h = hs
                .sample_height_m(d.as_vec3(), DRY_SITE_LOD_M)
                .unwrap_or(0.0);
            let stand = hs.canopy_coverage(d, site_h, DRY_SITE_LOD_M);
            let lat_deg = d.y.clamp(-1.0, 1.0).asin().to_degrees();
            info!(
                target: "thalos::diagnostic::capture",
                event = "forest_stand_site",
                lat_deg,
                macro_moisture = moisture,
                stand_coverage = stand,
                "forest-stand capture site"
            );
            d
        }
    };

    let up_world = (surface_q * dir_body).normalize();
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

/// Spiral the daylight hemisphere around `sun_dir_body` (same walk as
/// [`find_driest_site`]) and return the body-fixed direction that **maximises**
/// wet-belt forest, scored on the canonical canopy coverage
/// ([`HeightSource::canopy_coverage`], `thalos_terrain::canopy`) — which is both
/// what the ground paints *and* what the trees are placed from, so a
/// high-coverage site is guaranteed to frame real trees. This used to score
/// `moisture + 1.2 × stand` against two independent fields precisely because a
/// wet site could land in a stand gap and frame empty ground; unifying the
/// fields removed the need to hedge. Land-only, low-latitude, falls back to the
/// sub-stellar point.
fn find_forest_site(hs: &dyn HeightSource, sun_dir_body: DVec3) -> DVec3 {
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
    const MAX_ANGLE_RAD: f64 = 1.10;
    let mut best: Option<(f32, DVec3)> = None; // (score, dir) — maximise
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
                continue; // want forested LAND, not ocean
            }
            let score = hs.canopy_coverage(dir, h, DRY_SITE_LOD_M);
            if best.is_none_or(|(bs, _)| score > bs) {
                best = Some((score, dir));
            }
        }
    }
    best.map(|(_, d)| d).unwrap_or(sun)
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

    let surface_q = crate::rendering::transforms::surface_orientation_authored(
        &sim.system.bodies,
        body_id,
        states,
    )
    .unwrap_or_else(|| body_state.orientation.normalize());
    let sun_body = (surface_q.inverse() * sun_world).normalize();
    let seed = if sun_body.dot(DVec3::Y).abs() < 0.99 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let across_terminator = (seed - sun_body * seed.dot(sun_body)).normalize();
    let elevation = (sun_elevation_deg as f64).to_radians();
    let up_body = (sun_body * elevation.sin() + across_terminator * elevation.cos()).normalize();
    let up_world = (surface_q * up_body).normalize();

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
