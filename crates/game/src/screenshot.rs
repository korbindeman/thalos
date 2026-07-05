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
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use bevy::{
    asset::RenderAssetUsages,
    camera::{ImageRenderTarget, RenderTarget},
    math::DVec3,
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
        view::screenshot::{Capturing, Screenshot, save_to_disk},
    },
    window::{CursorIcon, SystemCursorIcon},
};
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_input::game::GameInputIntent;
use thalos_physics_local::HeightSourceRegistry;

use crate::camera::ShipCamera;
use crate::loading::AppState;
use crate::rendering::ground_terrain::BodyTerrain;
use crate::rendering::{SimulationState, SolarSystemState};
use thalos_body_render::renderer_tile_lod_m_at;
use thalos_body_render::udlod::prelude::{TerrainViewComponents, TileAtlas, TileTree};
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
}

impl ScreenshotPreset {
    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            // Truthy / unnamed → the default preset.
            "" | "1" | "true" | "yes" | "on" | "spaceport" | "spaceport-aerial" | "aerial"
            | "base" => Self::SpaceportAerial,
            "hub" | "space-center" | "spacecenter" | "play" => Self::Hub,
            other => {
                eprintln!(
                    "  Unknown THALOS_SCREENSHOT preset '{other}'; using spaceport-aerial."
                );
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
        }
    }

    /// Whether this preset boots the space-center hub route (spaceport built
    /// with no craft placed, hub opened on reveal) — `main.rs` arms
    /// `HubSpaceportBuild` + `OpenSpaceCenterOnStart` for it, exactly like the
    /// start screen's PLAY.
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
            },
            // Matches the hub's establishing view (`HUB_ESTABLISHING_DISTANCE_M`)
            // so the capture shows what PLAY shows.
            Self::Hub => ScreenshotConfig {
                preset: self,
                out: PathBuf::from("tools/screenshots/hub.png"),
                width: 1920,
                height: 1080,
                azimuth_deg: 35.0,
                elevation_deg: 42.0,
                distance_m: 4000.0,
                warmup_frames: 240,
                tail_frames: 24,
            },
        }
    }
}

/// A headless screenshot request, resolved from `THALOS_SCREENSHOT*` env vars.
///
/// The pose is a god-view around a preset-resolved focus point (the spaceport
/// pad centre): [`azimuth_deg`](Self::azimuth_deg) sweeps around the local
/// vertical, [`elevation_deg`](Self::elevation_deg) tilts up from the horizon
/// (90° = straight down), and [`distance_m`](Self::distance_m) is the boom
/// length. Every field is overridable so you can save *specific angles + points*
/// without recompiling.
#[derive(Resource, Clone, Debug)]
pub struct ScreenshotConfig {
    pub preset: ScreenshotPreset,
    /// Output PNG path (relative to the working dir). Its parent is created.
    pub out: PathBuf,
    pub width: u32,
    pub height: u32,
    /// Camera azimuth around the focus, degrees (0 = local east, +CCW).
    pub azimuth_deg: f32,
    /// Camera elevation above the local horizon, degrees (90 = straight down).
    pub elevation_deg: f32,
    /// Boom distance from the focus, metres.
    pub distance_m: f64,
    /// Frames to render (posing the aerial camera) after reaching `Running`
    /// before the capture — lets pipelines compile and shadows / atmosphere /
    /// tiles converge to the new framing.
    pub warmup_frames: u32,
    /// Frames to keep running after the capture so the async GPU readback flushes
    /// to disk before the app exits.
    pub tail_frames: u32,
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
    pub fn from_env() -> Option<Self> {
        let raw = env::var("THALOS_SCREENSHOT").ok()?;
        let mut cfg = ScreenshotPreset::parse(&raw).defaults();

        if let Some(out) = env::var_os("THALOS_SCREENSHOT_OUT") {
            cfg.out = PathBuf::from(out);
        }
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
        Some(cfg)
    }
}

fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    env::var(key).ok().and_then(|s| s.trim().parse::<T>().ok())
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
}

pub struct HeadlessScreenshotPlugin;

impl Plugin for HeadlessScreenshotPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScreenshotDriver>()
            .add_systems(Startup, setup_screenshot_target)
            .add_systems(Update, (retarget_ship_camera, hide_overlays))
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
            .add_systems(
                Update,
                probe_apron_lod.run_if(in_state(AppState::Running)),
            );
    }
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
    let Some(camera) = camera_q.iter().next() else { return };
    let Some(tree) = tile_trees.get(&(terrain_entity, camera)) else {
        return;
    };
    let r = sim.system.bodies[site.body_id].radius_m + site.elevation_m;
    let across = site.center_dir.cross(site.heading_tangent).normalize();
    let pad = site.center_dir * r;
    let view_dist_km = (tree.view_position() - pad).length() / 1000.0;
    let hs = height_sources.get(site.body_id);
    let offs = [-1200.0f64, -560.0, -520.0, -470.0, -350.0, 0.0, 350.0, 470.0, 520.0, 560.0, 1200.0];
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
fn hide_overlays(
    mut photo: ResMut<crate::photo_mode::PhotoMode>,
    mut overlays: ParamSet<(
        Query<&mut Visibility, With<crate::hud::HudPanel>>,
        Query<&mut Visibility, With<crate::photo_mode::HideInPhotoMode>>,
    )>,
) {
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

/// Pose the aerial camera every `Running` frame, then capture once warmed up and
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
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(target) = driver.target.clone() else {
        return;
    };
    if !driver.retargeted {
        return; // wait until the camera renders into our target
    }

    // Resolve the focus (spaceport pad centre) and pose the camera. If anything
    // isn't ready yet, hold the frame counter so warmup only starts once we're
    // actually framing the scene.
    let Some(ctx) = hub_context(&sim, &solar, &height_sources, &registry, homeworld.0) else {
        return;
    };
    let Ok(root) = root_grid.single() else {
        return;
    };
    let Ok((mut transform, mut cell)) = camera.single_mut() else {
        return;
    };
    pose_camera(&cfg, &ctx, root, &mut transform, &mut cell);

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
        "capturing {} ({}x{}) az={}° el={}° dist={:.0}m",
        cfg.out.display(),
        cfg.width,
        cfg.height,
        cfg.azimuth_deg,
        cfg.elevation_deg,
        cfg.distance_m,
    );
    commands
        .spawn(Screenshot::image(target))
        .observe(save_to_disk(cfg.out.clone()));
    driver.captured = true;
}

/// Place the ship camera at the god-view pose defined by `cfg` around `ctx`'s
/// focus. Mirrors [`crate::god_view::drive_god_view`], minus the input handling
/// and pitch clamp (so near-top-down elevations are reachable). Detail systems
/// (scatter, shadows) follow the camera via `rendering::view_anchor`.
fn pose_camera(
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
    let look_up = if to_focus.dot(up).abs() > 0.99 { north } else { up };

    let (next_cell, local) = root.translation_to_grid(camera_world);
    *cell = next_cell;
    *transform =
        Transform::from_translation(local).looking_to(to_focus.as_vec3(), look_up.as_vec3());
}
