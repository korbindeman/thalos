#![allow(clippy::too_many_arguments, clippy::type_complexity)]

mod aero;
mod autopilot;
mod bake_check;
mod body_tree_panel;
mod bridge;
mod camera;
mod control_bus;
mod controls;
mod coords;
mod debug;
mod engine;
mod flight_plan_view;
mod freecam;
mod fuel;
mod hud;
mod input;
mod loading;
mod local_physics;
mod main_menu;
mod maneuver;
mod map_view;
mod navball;
mod navigation;
mod pause_menu;
mod perf_log;
mod photo_mode;
mod player_controller;
mod reflection_probe;
mod regime;
mod relaunch;
mod rendering;
mod runway;
mod scenario_menu;
mod screenshot;
mod settings_menu;
mod ship_view;
mod shipyard_editor;
mod sim_clock;
mod sky_render;
mod solar_system_state;
mod spawn;
mod staging;
mod star_flare;
mod structures;
mod surface_settle;
mod target;
mod terrain_registry;
mod velocity_frame;
mod view;
mod warp_to_maneuver;

use std::sync::Arc;

use bevy::asset::AssetPlugin;
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use bevy::render::{
    RenderPlugin,
    settings::{Backends, RenderCreation, WgpuSettings},
};
use bevy::window::{
    MonitorSelection, PresentMode, PrimaryWindow, VideoModeSelection, WindowMode, WindowResolution,
};
use thalos_body_render::BodyRenderPlugin;
use thalos_input::game::GameInputPlugin;
use thalos_input::settings::InputSettings;
use thalos_physics_canonical::{
    body_trajectory_provider::BodyTrajectoryProvider,
    canonical::{AuthorityMode, Epoch, WorldPhysicsConfig},
    debug_orbits::debug_parking_orbit_relative_state,
    gravity_mode::GravityMode,
    simulation::{Simulation, SimulationConfig},
    types::{AttitudeState, ShipParameters, VesselKind},
};
use thalos_world::StateVector;
use thalos_world::parsing::load_solar_system_from_dir;

use aero::GameAeroPlugin;
use autopilot::AutopilotPlugin;
use body_tree_panel::BodyTreePanelPlugin;
use bridge::BridgePlugin;
use camera::CameraPlugin;
use controls::ControlLocksPlugin;
use debug::DebugPlugin;
use engine::EnginePlugin;
use flight_plan_view::FlightPlanViewPlugin;
use freecam::FreeCamPlugin;
use fuel::FuelPlugin;
use hud::HudPlugin;
use input::GameInputGatePlugin;
use loading::LoadingScreenPlugin;
use local_physics::GameLocalPhysicsPlugin;
use maneuver::ManeuverPlugin;
use map_view::MapViewPlugin;
use navball::NavballPlugin;
use navigation::NavigationPlugin;
use pause_menu::PauseMenuPlugin;
use settings_menu::SettingsMenuPlugin;
use perf_log::PerfLogPlugin;
use photo_mode::PhotoModePlugin;
use player_controller::PlayerControllerPlugin;
use rendering::RenderingPlugin;
use scenario_menu::ScenarioMenuPlugin;
use screenshot::ScreenshotPlugin;
use ship_view::ShipViewPlugin;
use sim_clock::SimClockPlugin;
use solar_system_state::{SimulationState, SolarSystemStatePlugin};
use spawn::{SpawnPlugin, SpawnSituation};
use target::TargetPlugin;
use terrain_registry::SharedTerrainRegistry;
use view::ViewPlugin;
use warp_to_maneuver::WarpToManeuverPlugin;

// ---------------------------------------------------------------------------
// System ordering
// ---------------------------------------------------------------------------

/// Execution stages within `Update`, ordered so that physics advances before
/// positions are written, and positions are written before the camera reads them.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum SimStage {
    /// Bridge: advance sim_time and ship state.
    Physics,
    /// Rendering: update body/ship transforms from sim state.
    Sync,
    /// Camera: compute camera transform from body transforms.
    Camera,
}

// ---------------------------------------------------------------------------
// Runtime body-state provider
// ---------------------------------------------------------------------------

/// Patched-conics runtime span (10,000 Julian years).
const RUNTIME_TIME_SPAN: f64 = 3.156e11;

/// Bevy-side handle on the shared terrain registry. Holds the same inner
/// `Arc<RwLock<...>>` as the propagator's terrain provider — inserting a
/// surface here is immediately visible to the propagator's collision
/// detection. The wrapper exists because `Resource` is a Bevy derive and
/// `SharedTerrainRegistry` (in `terrain_registry`) is plain non-Bevy code.
#[derive(Resource, Clone)]
pub struct GameTerrainRegistry(pub SharedTerrainRegistry);

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Window present mode. `THALOS_VSYNC=off` (or `0`/`false`/`no`) selects
/// `AutoNoVsync` so frame times run uncapped for profiling while still letting
/// wgpu fall back to a supported non-vsync present mode; anything else keeps
/// the vsync default (`AutoVsync`).
fn present_mode_from_env() -> PresentMode {
    match std::env::var("THALOS_VSYNC") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" | "no" => PresentMode::AutoNoVsync,
            _ => PresentMode::AutoVsync,
        },
        Err(_) => PresentMode::AutoVsync,
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

/// Primary-window configuration. Borderless fullscreen remains the default,
/// but `THALOS_WINDOW_MODE=windowed` is a useful Windows swapchain-stability
/// fallback when a driver returns a generic surface-acquire error. Optional:
/// `THALOS_WINDOW_SIZE=1600x900`.
fn window_from_env() -> Window {
    let present_mode = present_mode_from_env();
    let size = std::env::var("THALOS_WINDOW_SIZE")
        .ok()
        .and_then(|value| parse_window_size(&value))
        .unwrap_or((1600, 900));

    let mode = match std::env::var("THALOS_WINDOW_MODE") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "windowed" | "window" => WindowMode::Windowed,
            "exclusive" | "fullscreen" | "true-fullscreen" | "true_fullscreen" => {
                WindowMode::Fullscreen(MonitorSelection::Primary, VideoModeSelection::Current)
            }
            "borderless" | "borderless-fullscreen" | "borderless_fullscreen" | "" => {
                WindowMode::BorderlessFullscreen(MonitorSelection::Primary)
            }
            other => {
                eprintln!(
                    "Unknown THALOS_WINDOW_MODE={other:?}; using borderless fullscreen. \
                     Expected windowed, borderless, or fullscreen."
                );
                WindowMode::BorderlessFullscreen(MonitorSelection::Primary)
            }
        },
        Err(_) => WindowMode::BorderlessFullscreen(MonitorSelection::Primary),
    };

    // Dev/diagnostic: force a window scale factor (overrides the OS HiDPI
    // scale). `THALOS_SCALE=1.0` etc. Used to isolate fractional-scale text
    // rendering bugs.
    let mut resolution = WindowResolution::new(size.0, size.1);
    if let Ok(scale) = std::env::var("THALOS_SCALE")
        && let Ok(value) = scale.trim().parse::<f32>()
        && value > 0.0
    {
        resolution = resolution.with_scale_factor_override(value);
    }

    Window {
        title: "Thalos".into(),
        mode,
        resolution,
        // Dev/perf hook: `THALOS_VSYNC=off` uncaps the framerate so
        // frame-time deltas are observable when profiling. Default keeps
        // vsync on.
        present_mode,
        ..default()
    }
}

/// Work around a Bevy 0.18 text-rendering bug: at **fractional** window scale
/// factors (a 150 % HiDPI display reports 1.5, etc.) glyphs rasterise at
/// inconsistent sizes — text looks broken (non-uniform, "not monospace").
/// Integer scale factors render cleanly. So we compensate through the UI
/// scale so the *effective* UI scale (window scale × UI scale) lands on the
/// nearest integer (≥ 1) once the window's real scale is known.
///
/// An earlier version snapped the *window* scale-factor override instead, but
/// `bevy_winit::changed_windows` treats a scale-factor change as
/// logical-size-preserving and physically resizes the window — on a 150 %
/// display the borderless-fullscreen window grew to 4/3 of the monitor. The
/// window is left untouched now: `UiScale` covers Bevy UI (rasterised at
/// `window scale × UiScale`) and `EguiContextSettings::scale_factor` covers
/// the egui panels (`window scale × scale_factor`), so the two stay mutually
/// consistent.
///
/// The UI ends up slightly larger or smaller than the OS-intended size (e.g.
/// 1.5 → 2.0) but crisp; a user who prefers a specific scale can pin the
/// window scale with `THALOS_SCALE=1` (handled in [`window_from_env`], which
/// wins here). Remove this once the upstream Bevy fractional-scale text bug
/// is fixed.
fn compensate_fractional_ui_scale(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut ui_scale: ResMut<UiScale>,
    mut egui_settings: Query<&mut bevy_egui::EguiContextSettings>,
    mut compensated_log: Local<bool>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    // A manual `THALOS_SCALE` override already pinned the window scale (winit
    // honours it from creation, no resize involved) — leave the UI alone.
    if window.resolution.scale_factor_override().is_some() {
        return;
    }
    let os = window.resolution.base_scale_factor();
    if os <= 0.0 {
        return; // scale not reported by winit yet
    }
    let ratio = os.round().max(1.0) / os;
    if (ui_scale.0 - ratio).abs() > 1.0e-4 {
        ui_scale.0 = ratio;
        if !*compensated_log {
            info!(
                "compensating fractional window scale {os:.3} with UI scale ×{ratio:.3} \
                 (crisp-text workaround for Bevy fractional-scale rendering; \
                 override with THALOS_SCALE)"
            );
            *compensated_log = true;
        }
    }
    // Egui contexts can spawn after startup (and after the ratio is known),
    // so keep late arrivals in step instead of writing once.
    for mut settings in &mut egui_settings {
        if (settings.scale_factor - ratio).abs() > 1.0e-4 {
            settings.scale_factor = ratio;
        }
    }
}

fn backends_from_env() -> Option<Backends> {
    let value = std::env::var("THALOS_WGPU_BACKEND").ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" | "all" | "" => None,
        "dx12" | "d3d12" => Some(Backends::DX12),
        "vulkan" | "vk" => Some(Backends::VULKAN),
        "metal" => Some(Backends::METAL),
        "gl" | "opengl" => Some(Backends::GL),
        other => {
            eprintln!(
                "Unknown THALOS_WGPU_BACKEND={other:?}; using Bevy/wgpu default. \
                 Expected auto, dx12, vulkan, metal, or gl."
            );
            None
        }
    }
}

fn wgpu_settings_from_env() -> WgpuSettings {
    let mut settings = WgpuSettings::default();
    if let Some(backends) = backends_from_env() {
        settings.backends = Some(backends);
    }
    settings
}

fn main() {
    // ------------------------------------------------------------------
    // 1. Load the solar system definition from the RON asset files.
    // ------------------------------------------------------------------
    let system = load_solar_system_from_dir(std::path::Path::new("assets"))
        .expect("Failed to load solar system from assets/");

    // ------------------------------------------------------------------
    // 1a. Pre-flight: ensure every procedural body has a current bake.
    //
    // Missing or stale bakes are auto-repaired by shelling out to
    // `thalos_bake_dump all` before Bevy boots, so a source edit under
    // `crates/terrain_gen/` doesn't require the user to remember
    // `just bake all` between iterations. When all bakes are current
    // the check is microseconds (per-body `peek_key` only).
    // ------------------------------------------------------------------
    bake_check::ensure_bakes_or_exit(&system.bodies);

    // ------------------------------------------------------------------
    // 2. Print a startup banner.
    // ------------------------------------------------------------------
    println!("╔══════════════════════════════════════════╗");
    println!("║             T H A L O S                  ║");
    println!("╚══════════════════════════════════════════╝");
    println!("  System:           {}", system.name);
    println!("  Bodies:           {}", system.bodies.len());

    // ------------------------------------------------------------------
    // 3. Build the gravity model.
    //
    //    The mode is the savegame-pinned strategy that picks both the body
    //    motion source and the ship propagator. Hardcoded here until save
    //    files exist to deserialize it from.
    // ------------------------------------------------------------------
    let world_config = WorldPhysicsConfig::classic();
    world_config
        .validate_supported()
        .expect("Realistic world preset is deferred until M7");
    let gravity_mode = GravityMode::PatchedConics;
    println!(
        "  Gravity mode:    {:?} ({:.0}-year span).",
        gravity_mode,
        RUNTIME_TIME_SPAN / 3.156e7,
    );
    // Shared terrain registry: filled in by `finalize_planet_generation`
    // as each body's surface finishes baking. The propagator holds a clone
    // of the same inner Arc<RwLock<...>>, so live propagation and trajectory
    // prediction collide against the same surface the renderer is showing.
    let terrain_registry = SharedTerrainRegistry::new();
    let gravity_impls = gravity_mode.build_with_terrain(
        &system,
        RUNTIME_TIME_SPAN,
        Arc::new(terrain_registry.clone()),
    );
    let ephemeris: Arc<dyn BodyTrajectoryProvider> = Arc::clone(&gravity_impls.body_trajectory);

    // ------------------------------------------------------------------
    // 4. Resolve the player's absolute initial state.
    //
    //    Spawn modes are picked by `just game [mode]` (default `orbit`, or
    //    `just game eva` / `landing` / `final`). The canonical CraftState is
    //    the player either way — KSP-style: one craft, Ship or EVA,
    //    distinguished by `VesselKind`.
    //
    //    - `orbit` (default): the ship in a low Thalos parking orbit, derived
    //      from the homeworld via `debug_parking_orbit_relative_state`.
    //    - `eva`: the player on foot at the Thalos sub-stellar point. The
    //      pose is a body-fixed direction plus a safe drop margin above the
    //      body radius; terrain heights aren't known yet (the registry is
    //      populated later as bakes load), so we err above the worst-case
    //      elevation and let local physics resolve the drop to the surface.
    // ------------------------------------------------------------------
    let homeworld_name = "Thalos";
    let homeworld_id = system
        .name_to_id
        .get(homeworld_name)
        .copied()
        .unwrap_or_else(|| {
            // Fall back to the first non-star body if "Thalos" isn't present.
            system
                .bodies
                .iter()
                .find(|b| b.parent.is_some())
                .map(|b| b.id)
                .expect("No non-star body found to use as homeworld fallback")
        });

    let homeworld = &system.bodies[homeworld_id];
    let homeworld_state = ephemeris.state(homeworld_id, Epoch::ZERO);

    // Spawn mode arrives as a CLI arg from `just game [mode]` (works in any
    // shell), falling back to the `THALOS_SPAWN` env var for a direct
    // `cargo run`. Unset / `menu` → boot to the start screen.
    let spawn_request = std::env::args()
        .nth(1)
        .filter(|arg| !arg.trim().is_empty())
        .or_else(|| std::env::var("THALOS_SPAWN").ok())
        .unwrap_or_default();
    let request = spawn_request.trim().to_ascii_lowercase();
    // `just game shipyard` opens straight into the in-game ship editor: the
    // sim seeds the default parking orbit behind it (and stays paused while
    // the editor is open), so closing the editor drops into normal flight.
    let open_shipyard = matches!(request.as_str(), "shipyard" | "editor" | "vab");
    let auto_run = spawn::AutoRun::from_env();
    // Bare launch → start screen, behind the boot load of the placeholder
    // parking-orbit world. `THALOS_AUTO_RUN` skips the menu (straight into
    // orbit) so autonomous agents keep their one-shot launch flow.
    let wants_menu = matches!(request.as_str(), "" | "menu" | "title");
    let menu_boot = wants_menu && !open_shipyard && !auto_run.enabled;
    let situation = if open_shipyard || wants_menu {
        SpawnSituation::ShipOrbit
    } else {
        SpawnSituation::from_request(&spawn_request)
    };

    let (ship_state, vessel_kind, ship_params, attitude) = if situation == SpawnSituation::Eva {
        // Sub-stellar point so the player wakes up in daylight: the direction
        // from Thalos toward Pyros (heliocentric origin) is
        // `-homeworld_state.position`.
        let up = (-homeworld_state.position).normalize();
        // 12 km drop margin: has to clear Thalos's max peaks (~±8 km from
        // body radius) since the terrain registry isn't loaded yet — anything
        // less can spawn inside a mountain. `attach_terrain_patch_when_close`
        // then hands translation to Avian and the collider catches the fall.
        let spawn_offset = up * (homeworld.radius_m + 12_000.0);
        let surface_velocity = homeworld_state.angular_velocity.cross(spawn_offset);
        let state = StateVector {
            position: homeworld_state.position + spawn_offset,
            velocity: homeworld_state.velocity + surface_velocity,
        };
        // Stand upright: body +Z (dorsal) along local-up, body +Y (nose)
        // tangent to the surface roughly eastward. Matches the EVA
        // controller's `level_orientation` so the player faces forward.
        let east_seed = homeworld_state.orientation * DVec3::Y;
        let east_tangent = (east_seed - up * east_seed.dot(up)).normalize();
        let right = up.cross(east_tangent).normalize();
        let basis = DMat3::from_cols(right, east_tangent, up);
        let attitude = AttitudeState {
            orientation: DQuat::from_mat3(&basis),
            angular_velocity: DVec3::ZERO,
        };
        println!(
            "  Player (EVA):    standing on {} (daylight side)",
            homeworld.name
        );
        (state, VesselKind::Eva, ShipParameters::eva(), attitude)
    } else {
        // `orbit` starts in the authored parking orbit. Deferred descent modes
        // (`landing`, `final`) use that orbit only as the placeholder behind
        // the loading screen: `spawn::refine_descent_spawn` swaps in the
        // terrain-aware approach state on the first `Running` frame, once
        // heights are available to place the ship above ground over land.
        // The parking orbit is derived from the homeworld (a debug spawn, not
        // authored world data) rather than stored on `SolarSystemDefinition`.
        let rel = debug_parking_orbit_relative_state(homeworld);
        // Level orbital flight: nose along prograde, dorsal radial-out, shared
        // with the navball and control axes so "upright" stays aligned. Real
        // ship params (MOI, torque, masses) are pushed in by `spawn_player_ship`
        // once `apollo.ron` loads. The same helper rebuilds this orbit for a
        // post-destruction respawn, so the two never drift.
        let (state, attitude) = spawn::orbit_parking_state(rel, &homeworld_state);
        if let Some(label) = situation.descent_label() {
            println!(
                "  Ship:            {label} on {} (over land)",
                homeworld.name
            );
        } else if situation.is_runway() {
            // The runway scenarios also seed the parking orbit as a placeholder
            // behind the loading screen; `runway::finish_runway_spawn` installs
            // the real runway + aircraft state on the first `Running` frame.
            println!(
                "  Aircraft:        runway scenario on {} (placed once terrain loads)",
                homeworld.name
            );
        } else if menu_boot {
            println!("  Start screen:    pick a scenario in-game (just game <mode> skips it)");
        } else {
            let altitude_km = (rel.position.length() - homeworld.radius_m) / 1000.0;
            println!(
                "  Ship:            {:.0} km orbit around {}",
                altitude_km, homeworld.name
            );
        }
        (state, VesselKind::Ship, ShipParameters::default(), attitude)
    };

    // ------------------------------------------------------------------
    // 5. Build and run the Bevy app.
    // ------------------------------------------------------------------
    let window = window_from_env();
    let wgpu_settings = wgpu_settings_from_env();

    App::new()
        // The shipyard editor is a separate scene: while it is open, no game
        // logic runs. Gating the three simulation stages on `editor_closed`
        // (on top of the pause gate) freezes physics, world sync, and the game
        // camera so the editor owns the frame entirely — the flight world is
        // suspended, not just hidden. See `crate::shipyard_editor`.
        .configure_sets(
            Update,
            (
                SimStage::Physics
                    .run_if(pause_menu::not_game_paused.and(shipyard_editor::editor_closed)),
                SimStage::Sync.run_if(shipyard_editor::editor_closed),
                SimStage::Camera
                    .run_if(pause_menu::not_game_paused.and(shipyard_editor::editor_closed)),
            )
                .chain(),
        )
        .insert_resource(ClearColor(Color::srgb(0.02, 0.01, 0.04)))
        .insert_resource(
            InputSettings::load_from_path("assets/input.ron")
                .expect("Failed to load input bindings from assets/input.ron"),
        )
        .add_plugins(
            DefaultPlugins
                .build()
                .disable::<bevy::transform::TransformPlugin>()
                .set(WindowPlugin {
                    primary_window: Some(window),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(wgpu_settings),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: "../../assets".to_string(),
                    ..default()
                })
                .set(bevy::log::LogPlugin {
                    // Keep our own crates at INFO; silence Bevy's chatty
                    // startup-info categories so the terrain diagnostic and
                    // other game logs aren't buried. Override via RUST_LOG.
                    filter: "info,\
                             wgpu=error,naga=warn,bevy_app=warn,\
                             bevy_render=warn,bevy_diagnostic=warn,\
                             bevy_winit=warn,bevy_egui=warn,\
                             bevy_pbr=warn,bevy_asset=warn,\
                             cosmic_text=warn,gilrs_core=warn,gilrs=warn,\
                             offset_allocator=warn"
                        .to_string(),
                    ..default()
                }),
        )
        // `BodyRenderPlugin` adds the ground terrain stack
        // (`thalos_udlod::TerrainPlugin`, which adds `BigSpaceDefaultPlugins`
        // unconditionally) plus impostor materials and shared shading
        // libraries. Adding it again here would panic on duplicate
        // registration.
        .add_plugins(BodyRenderPlugin)
        // BRP server (port 15702) for agent-driven inspection / mutation.
        // Always on in dev; the listener is idle when no client is
        // connected. See docs/tooling.md for the MCP workflow.
        //
        // `BrpExtrasPlugin` adds `FrameTimeDiagnosticsPlugin` itself
        // (its `diagnostics` feature backs the `brp_extras/get_diagnostics`
        // method), so we do not add it separately here.
        .add_plugins(bevy_brp_extras::BrpExtrasPlugin)
        .add_plugins(bevy_egui::EguiPlugin::default())
        // The dedicated UI camera in `view::spawn_ui_camera` owns the
        // primary egui context; disable auto-attach so it doesn't bind
        // to whichever scene camera spawns first and disappear when that
        // camera goes inactive.
        .insert_resource(bevy_egui::EguiGlobalSettings {
            auto_create_primary_context: false,
            ..default()
        })
        .insert_resource({
            let mut simulation = Simulation::new(
                ship_state,
                gravity_impls,
                system.bodies.clone(),
                SimulationConfig::default(),
            );
            // Vessel kind, parameters, and attitude were resolved per
            // spawn mode above. For EVA, `ShipParameters::eva()` keeps the
            // throttle / attitude-command pipes wired but motion-free so HUD
            // readouts stay coherent; for a ship, the sentinel defaults are
            // replaced by `spawn_player_ship` once `apollo.ron` loads.
            simulation.set_vessel_kind(vessel_kind);
            simulation.set_ship_params(ship_params);
            simulation.set_attitude(attitude);
            // Start in `OnRails` (the default coast authority). For an EVA
            // surface spawn the brief Kepler arc holds the player up until
            // `attach_terrain_patch_when_close` hands translation to Avian and
            // the collider catches the fall; for a ship it is just the orbit.
            simulation.transition_authority(AuthorityMode::OnRails { trajectory: 0 });
            SimulationState {
                simulation,
                system,
                ephemeris,
            }
        })
        .insert_resource(GameTerrainRegistry(terrain_registry))
        .insert_resource(situation)
        // Open the editor only once the world finishes loading — opening it
        // during `Loading` would gate off the very systems that complete the
        // load (see `OpenShipyardOnStart`).
        .insert_resource(shipyard_editor::OpenShipyardOnStart(open_shipyard))
        // Where the boot load reveals to: the start screen for a bare
        // launch, straight into the scenario otherwise.
        .insert_resource(loading::LoadDestination(if menu_boot {
            loading::AppState::MainMenu
        } else {
            loading::AppState::Running
        }))
        // Every spawn situation starts paused (warp 0×); `THALOS_AUTO_RUN`
        // resumes to 1× as soon as the loading screen clears (for agents).
        .insert_resource(auto_run)
        // The body the parking-orbit / on-foot scenarios anchor to, so the
        // destruction scenario menu can rebuild them on respawn.
        .insert_resource(spawn::Homeworld(homeworld_id))
        .add_plugins(SolarSystemStatePlugin)
        .add_plugins(bevy::prelude::MeshPickingPlugin)
        // Opt-in picking: body meshes (and any other Pickable-less mesh) would
        // otherwise absorb rays before the maneuver arrows, since the hover-map
        // builder treats unmarked entities as opaque blockers. At zoomed-out
        // ranges those bodies sit on top of the arrow's screen region and made
        // the arrows silently unhoverable.
        .insert_resource(bevy::picking::mesh_picking::MeshPickingSettings {
            require_markers: true,
            ..default()
        })
        .add_plugins(GameInputPlugin)
        .add_plugins(GameInputGatePlugin)
        .add_plugins(SimClockPlugin)
        .add_plugins(CameraPlugin)
        .add_plugins(FreeCamPlugin)
        .add_plugins(reflection_probe::ReflectionProbePlugin)
        .add_plugins(sky_render::SkyRenderPlugin)
        .add_plugins(star_flare::LensFlarePlugin)
        .add_plugins(LoadingScreenPlugin)
        .add_plugins(SpawnPlugin)
        .add_plugins(surface_settle::SurfaceSettlePlugin)
        .add_plugins(structures::StructuresPlugin)
        .add_plugins(runway::RunwayPlugin)
        .add_plugins(RenderingPlugin)
        .add_plugins(GameLocalPhysicsPlugin)
        .add_plugins(GameAeroPlugin)
        .add_plugins(PlayerControllerPlugin)
        .add_plugins(MapViewPlugin)
        .add_plugins(BridgePlugin)
        .add_plugins(regime::RegimePlugin)
        .add_plugins(FuelPlugin)
        .add_plugins(staging::StagingPlugin)
        .add_plugins(EnginePlugin)
        .add_plugins(TargetPlugin)
        .add_plugins(velocity_frame::VelocityFramePlugin)
        .add_plugins(FlightPlanViewPlugin)
        .add_plugins(ManeuverPlugin)
        .add_plugins(NavigationPlugin)
        .add_plugins(AutopilotPlugin)
        .add_plugins(ControlLocksPlugin)
        .add_plugins(control_bus::ControlBusPlugin)
        .add_plugins(WarpToManeuverPlugin)
        .add_plugins(HudPlugin)
        .add_plugins(PauseMenuPlugin)
        .add_plugins(main_menu::MainMenuPlugin)
        .add_plugins(SettingsMenuPlugin)
        .add_plugins(PerfLogPlugin)
        .add_plugins(ScenarioMenuPlugin)
        .add_plugins(NavballPlugin)
        .add_plugins(PhotoModePlugin)
        .add_plugins(ScreenshotPlugin)
        // Snap fractional HiDPI scale factors to an integer so UI text renders
        // crisply (Bevy 0.18 fractional-scale text bug). Runs every frame but
        // no-ops once the override is set.
        .add_systems(Update, compensate_fractional_ui_scale)
        .add_plugins(ViewPlugin)
        .add_plugins(ShipViewPlugin)
        .add_plugins(relaunch::RelaunchPlugin)
        .add_plugins(shipyard_editor::ShipyardEditorPlugin)
        .add_plugins(BodyTreePanelPlugin)
        .add_plugins(DebugPlugin)
        .run();
}
