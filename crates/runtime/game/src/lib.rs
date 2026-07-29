#![allow(clippy::too_many_arguments, clippy::type_complexity)]

mod aero;
mod autopilot;
mod base_editor;
mod body_tree_panel;
mod bridge;
mod camera;
mod camera_optics;
pub mod capture_health;
mod control_bus;
mod controls;
mod coords;
mod debug;
mod engine;
mod flight_config;
mod flight_plan_view;
mod freecam;
mod fuel;
mod game_context;
mod god_view;
mod graphics_settings;
mod hud;
mod input;
mod loading;
mod local_physics;
mod main_menu;
mod maneuver;
mod map_view;
mod mem_diag;
mod navball;
mod navigation;
mod pause_menu;
mod perf;
mod photo_mode;
mod player_controller;
mod reflection_probe;
mod regime;
mod relaunch;
mod rendering;
mod runtime_diagnostics;
mod runway;
mod scenario_menu;
mod screenshot;
mod settings;
mod settings_menu;
mod ship_view;
mod shipyard_editor;
mod sim_clock;
mod sky_render;
pub mod solar_system_state;
mod space_center;
mod spawn;
mod staging;
mod star_flare;
mod structures;
mod surface_settle;
mod target;
mod terrain_registry;
mod units_settings;
mod velocity_frame;
mod view;
pub mod viewpoints;
mod warp_to_maneuver;
mod window_settings;

use std::sync::Arc;
use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::asset::AssetPlugin;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use bevy::render::{
    RenderPlugin,
    settings::{Backends, RenderCreation, WgpuSettings},
};
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;
use thalos_body_render::BodyRenderPlugin;
use thalos_input::game::GameInputPlugin;
use thalos_input::settings::InputSettings;
use thalos_physics_canonical::{
    body_trajectory_provider::BodyTrajectoryProvider,
    canonical::{AuthorityMode, Epoch, WorldPhysicsConfig},
    debug_orbits::{debug_parking_orbit_relative_state, debug_polar_parking_orbit_relative_state},
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
use photo_mode::PhotoModePlugin;
use player_controller::PlayerControllerPlugin;
use rendering::RenderingPlugin;
use scenario_menu::ScenarioMenuPlugin;
use screenshot::ScreenshotPlugin;
use settings_menu::SettingsMenuPlugin;
use ship_view::ShipViewPlugin;
use sim_clock::SimClockPlugin;
use solar_system_state::{SimulationState, SolarSystemStatePlugin};
use spawn::{SpawnPlugin, SpawnSituation};
use target::TargetPlugin;
use terrain_registry::BodySurfaceRegistry;
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

/// `THALOS_WGPU_BACKEND` selection. Outer `None` = variable unset or
/// unparseable (fall through to the Thalos default); inner `None` =
/// explicit `auto` (let wgpu pick).
fn backends_from_env() -> Option<Option<Backends>> {
    let value = std::env::var("THALOS_WGPU_BACKEND").ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" | "all" | "" => Some(None),
        "dx12" | "d3d12" => Some(Some(Backends::DX12)),
        "vulkan" | "vk" => Some(Some(Backends::VULKAN)),
        "metal" => Some(Some(Backends::METAL)),
        "gl" | "opengl" => Some(Some(Backends::GL)),
        other => {
            eprintln!(
                "Unknown THALOS_WGPU_BACKEND={other:?}; using the Thalos default. \
                 Expected auto, dx12, vulkan, metal, or gl."
            );
            None
        }
    }
}

fn wgpu_settings_from_env() -> WgpuSettings {
    let mut settings = WgpuSettings::default();
    let backends = backends_from_env().unwrap_or_else(|| {
        // Thalos default: Vulkan on Windows. wgpu's own default prefers DX12
        // there, and DX12 is this project's documented unstable path
        // (swapchain-acquire panics, silent device death, and a full
        // DeviceLost wedge — see docs/development/tooling.md). Other platforms keep the
        // wgpu default (Metal on macOS); THALOS_WGPU_BACKEND=auto restores
        // the wgpu default everywhere.
        cfg!(target_os = "windows").then_some(Backends::VULKAN)
    });
    if let Some(backends) = backends {
        settings.backends = Some(backends);
    }
    settings
}

/// Builds the canonical Thalos application from the process environment.
///
/// Interactive and headless launchers both use this builder so there is one
/// plugin graph, one world construction path, and one renderer configuration.
#[derive(Clone, Copy, Debug, Default)]
pub struct AppBuilder;

impl AppBuilder {
    pub const fn new() -> Self {
        Self
    }

    pub fn build(self) -> App {
        // ------------------------------------------------------------------
        // 1. Load the solar system definition from the RON asset files.
        // ------------------------------------------------------------------
        let system = load_solar_system_from_dir(std::path::Path::new("assets"))
            .expect("Failed to load solar system from assets/");
        // Per-body surface failures degrade that body only (they are recorded
        // on the registry and reported here); this `expect` now fires only for
        // a failure global to the whole registry.
        let body_surfaces = BodySurfaceRegistry::load(
            &system.bodies,
            std::path::Path::new("assets/terrain_packages"),
        )
        .expect("Failed to load body terrain surfaces");
        for degraded in body_surfaces.degraded_bodies() {
            println!(
                "  ! {} has no terrain surface: {}",
                degraded.body_name, degraded.reason
            );
        }

        // ------------------------------------------------------------------
        // 1a. (Retired in terrain-rewrite 0b-1.) Procedural bodies now generate
        // terrain at runtime via `ProceduralSurface` behind the `SurfaceQuery`
        // seam, so there is no pre-baked artifact to validate or auto-repair.
        // `bake_check` is deleted in 0b-2.
        // ------------------------------------------------------------------

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
        for (body_id, surface) in body_surfaces.iter() {
            terrain_registry.insert(body_id, surface);
        }
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
        // Resolve launch/capture intent before choosing the anchor body. Named Mira
        // modes are ordinary game scenarios; headless presets use the same seam.
        let spawn_request = std::env::args()
            .nth(1)
            .filter(|arg| !arg.trim().is_empty())
            .or_else(|| std::env::var("THALOS_SPAWN").ok())
            .unwrap_or_default();
        let request = spawn_request.trim().to_ascii_lowercase();
        let screenshot_config = screenshot::ScreenshotConfig::from_env();
        let persistent_capture = screenshot_config.is_some()
            && std::env::var("THALOS_CAPTURE_SERVER")
                .ok()
                .is_some_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"));
        let homeworld_name = screenshot_config
            .as_ref()
            .map(|cfg| cfg.target_body_name())
            .unwrap_or_else(|| {
                if request.starts_with("mira") {
                    "Mira"
                } else {
                    "Thalos"
                }
            });
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
        // `just game shipyard` opens straight into the in-game ship editor: the
        // sim seeds the default parking orbit behind it (and stays paused while
        // the editor is open), so closing the editor drops into normal flight.
        // Headless screenshot mode (`THALOS_SCREENSHOT`): the whole game boots
        // off-screen (no window, no winit — driven by `ScheduleRunnerPlugin`, like
        // `just preview`), renders one scripted camera pose to a PNG, and exits. Its
        // presence forces the preset's scenario so the captured world is fully built,
        // and never the start screen / shipyard. See `crate::screenshot`.
        let headless = screenshot_config.is_some();

        let open_shipyard = matches!(request.as_str(), "shipyard" | "editor" | "vab") && !headless;
        // `just game hub` boots straight into the space-center hub: the PLAY path
        // without the start screen — placeholder parking orbit, spaceport built
        // behind the loading pass (no craft placed), hub opened on reveal. The
        // headless `hub` screenshot preset rides the same route.
        let boot_hub = matches!(request.as_str(), "hub" | "space-center" | "spacecenter")
            || screenshot_config
                .as_ref()
                .is_some_and(|cfg| cfg.boots_hub());
        let auto_run = spawn::AutoRun::from_env();
        // Bare launch → start screen, behind the boot load of the placeholder
        // parking-orbit world. `THALOS_AUTO_RUN` skips the menu (straight into
        // orbit) so autonomous agents keep their one-shot launch flow.
        let wants_menu = matches!(request.as_str(), "" | "menu" | "title");
        let menu_boot = wants_menu && !open_shipyard && !auto_run.enabled && !headless;
        let situation = if let Some(cfg) = &screenshot_config {
            cfg.spawn_situation()
        } else if open_shipyard || wants_menu || boot_hub {
            SpawnSituation::ShipOrbit
        } else {
            let body_request = request
                .strip_prefix("mira-")
                .or_else(|| request.strip_prefix("mira_"))
                .unwrap_or_else(|| {
                    if request == "mira" {
                        "orbit"
                    } else {
                        request.as_str()
                    }
                });
            SpawnSituation::from_request(body_request)
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
            // `orbit` / `polar` start in a debug parking orbit. Deferred descent
            // modes (`landing`, `final`) use the equatorial orbit only as the
            // placeholder behind the loading screen: `spawn::refine_descent_spawn`
            // swaps in the terrain-aware approach state on the first `Running`
            // frame, once heights are available to place the ship above ground
            // over land. Parking orbits are derived from the homeworld (a debug
            // spawn, not authored world data) rather than stored on
            // `SolarSystemDefinition`.
            let rel = if situation == SpawnSituation::PolarOrbit {
                debug_polar_parking_orbit_relative_state(homeworld)
            } else {
                debug_parking_orbit_relative_state(homeworld)
            };
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
            } else if situation.is_spaceport() {
                // Spaceport scenarios seed the parking orbit as a placeholder
                // behind the loading screen; `runway::finish_runway_spawn`
                // installs the real runway/pad craft state once terrain loads.
                if situation == SpawnSituation::Launch {
                    println!(
                        "  Rocket:          Saturn on {} launchpad (placed once terrain loads)",
                        homeworld.name
                    );
                } else {
                    println!(
                        "  Aircraft:        runway scenario on {} (placed once terrain loads)",
                        homeworld.name
                    );
                }
            } else if menu_boot {
                println!("  Start screen:    pick a scenario in-game (just game <mode> skips it)");
            } else {
                let altitude_km = (rel.position.length() - homeworld.radius_m) / 1000.0;
                let plane = if situation == SpawnSituation::PolarOrbit {
                    "polar "
                } else {
                    ""
                };
                println!(
                    "  Ship:            {:.0} km {plane}orbit around {}",
                    altitude_km, homeworld.name
                );
            }
            (state, VesselKind::Ship, ShipParameters::default(), attitude)
        };

        // ------------------------------------------------------------------
        // 5. Build and run the Bevy app.
        // ------------------------------------------------------------------
        // Unified persisted preferences (window + graphics + units) from one
        // settings.ron (project-local in debug, OS app-data in release; see
        // `crate::settings`), plus the THALOS_WINDOW_MODE / THALOS_WINDOW_SIZE /
        // THALOS_VSYNC session overrides. Loaded before the app so the initial
        // window honours them.
        let mut app_settings = settings::load();
        if headless {
            let graphics = std::env::var("THALOS_SCREENSHOT_GRAPHICS")
                .ok()
                .map(|raw| {
                    thalos_capture_protocol::CaptureGraphicsOverrides::parse(&raw).unwrap_or_else(
                        |error| panic!("invalid THALOS_SCREENSHOT_GRAPHICS: {error}"),
                    )
                })
                .unwrap_or_default();
            app_settings.graphics = graphics_settings::GraphicsSettings::for_capture(graphics);
        }
        let win_overrides = window_settings::overrides_from_env();
        let window = window_settings::initial_window(&app_settings.window, &win_overrides);
        let wgpu_settings = wgpu_settings_from_env();

        // Headless screenshot mode renders off-screen with no window; a normal launch
        // opens the configured primary window.
        let window_plugin = if headless {
            WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                close_when_requested: false,
                ..default()
            }
        } else {
            WindowPlugin {
                primary_window: Some(window),
                ..default()
            }
        };
        let mut default_plugins = DefaultPlugins
            .build()
            .disable::<bevy::transform::TransformPlugin>()
            .set(window_plugin)
            .set(RenderPlugin {
                // 0.19: RenderCreation::Automatic takes a Box<WgpuSettings>.
                render_creation: RenderCreation::Automatic(Box::new(wgpu_settings)),
                ..default()
            })
            .set(AssetPlugin {
                // Under `cargo run` Bevy resolves this against
                // CARGO_MANIFEST_DIR (apps/game or tools/capture_host, both
                // two levels below the workspace). The dx-launched capture
                // host has no manifest dir at runtime; its client sets
                // BEVY_ASSET_ROOT to the workspace root, which Bevy prepends
                // verbatim — so the relative hop must collapse to "assets"
                // there or every asset resolves outside the workspace.
                file_path: if std::env::var_os("BEVY_ASSET_ROOT").is_some() {
                    "assets".to_string()
                } else {
                    "../../assets".to_string()
                },
                ..default()
            })
            .set(bevy::log::LogPlugin {
                // Keep our own crates at INFO; silence Bevy's chatty startup
                // categories so human-facing game status stays readable.
                // `thalos::diagnostic::*` INFO events are separately written
                // to JSONL and omitted by the console formatter below.
                // Override collection levels via RUST_LOG.
                filter: "info,\
                     wgpu=error,naga=warn,bevy_app=warn,\
                     bevy_render=warn,bevy_diagnostic=warn,\
                     bevy_winit=warn,\
                     bevy_pbr=warn,bevy_asset=warn,\
                     cosmic_text=warn,gilrs_core=warn,gilrs=warn,\
                     offset_allocator=warn"
                    .to_string(),
                // Counts ERROR events so a headless capture that logged a
                // shader/pipeline validation failure can exit non-zero instead
                // of writing a PNG and reporting success (BL-20). See
                // `capture_health`.
                custom_layer: capture_health::runtime_layers,
                fmt_layer: runtime_diagnostics::human_console_layer,
                ..default()
            });
        // No winit event loop in headless mode — `ScheduleRunnerPlugin` (added below)
        // drives the frames instead.
        if headless {
            default_plugins = default_plugins.disable::<WinitPlugin>();
        }

        let mut app = App::new();
        app
            // The `SimStage` sets are gated by the `GameContext` sub-state (the
            // single in-`Running` mode authority — see `crate::game_context` and
            // `docs/gameplay/ui_flow.md`), replacing the former `*_closed` boolean helpers:
            // - **VAB** (`GameContext::Vab`) is a separate scene that hides the
            //   world, so it gates *all three* sets off (`not_vab`) — physics, world
            //   sync, and the game camera all freeze; the editor owns the frame.
            // - The **base editor** and **space-center hub** are *in-world* overlays:
            //   they keep the world visible and frozen (via `SimClock`, like a warp-0
            //   pause), so only the **Camera** set is gated off for them
            //   (`flight_or_no_context`) — their shared god-view camera owns the view,
            //   but the terrain-streaming `Sync` must keep running or the ground goes
            //   black. Outside `Running` (Loading / MainMenu) the helpers read as "no
            //   modal", so `Sync`/`Camera` keep running behind the loading screen.
            .configure_sets(
                Update,
                (
                    SimStage::Physics
                        .run_if(pause_menu::not_game_paused.and_then(game_context::not_vab)),
                    SimStage::Sync.run_if(game_context::not_vab),
                    SimStage::Camera.run_if(
                        pause_menu::not_game_paused.and_then(game_context::flight_or_no_context),
                    ),
                )
                    .chain(),
            )
            .insert_resource(ClearColor(Color::srgb(0.02, 0.01, 0.04)))
            // The three settings sections become three separate resources (so every
            // consumer is unchanged); `settings::AppSettingsPlugin` persists them
            // back to the one file. Window must be inserted here (pre-app) since it
            // shaped the initial window above.
            .insert_resource(app_settings.window)
            .insert_resource(win_overrides)
            .insert_resource(app_settings.graphics)
            .insert_resource(app_settings.units)
            .insert_resource(
                InputSettings::load_from_path("assets/input.ron")
                    .expect("Failed to load input bindings from assets/input.ron"),
            )
            // `default_plugins` is pre-built above (window / render / asset / log),
            // with the window + winit disabled in headless screenshot mode.
            .add_plugins(default_plugins)
            // `BodyRenderPlugin` adds the ground terrain stack
            // (`thalos_udlod::TerrainPlugin`, which adds `BigSpaceDefaultPlugins`
            // unconditionally) plus impostor materials and shared shading
            // libraries. Adding it again here would panic on duplicate
            // registration.
            .add_plugins(BodyRenderPlugin)
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
            .insert_resource(body_surfaces)
            .init_resource::<terrain_registry::RenderedGroundRegistry>()
            .insert_resource(situation)
            // `just game hub` / the headless hub preset: build the spaceport (no
            // craft) behind the boot loading pass and open the space-center hub on
            // reveal — the same arming the start screen's PLAY does at runtime.
            // `register_boot_steps` adds the PLACEMENT step when the build is armed.
            .insert_resource(space_center::HubSpaceportBuild { pending: boot_hub })
            // Boot routing: which `GameContext` the boot load reveals into (retiring
            // the old `Open{Shipyard,SpaceCenter}OnStart` one-shot flags). The
            // context switch is applied on `OnEnter(Running)` by
            // `game_context::apply_initial_context`; `None` reveals into Flight.
            .insert_resource(game_context::InitialContext(if open_shipyard {
                Some(game_context::GameContext::Vab)
            } else if boot_hub {
                Some(game_context::GameContext::SpaceCenter)
            } else {
                None
            }))
            // Where the boot load reveals to: the start screen for a bare
            // launch, straight into the scenario otherwise.
            .insert_resource(loading::LoadDestination(if menu_boot {
                loading::AppState::MainMenu
            } else {
                loading::AppState::Running
            }))
            // A bare menu boot defers the world entirely: no bodies, ship, or sky
            // are spawned until the menu starts a scenario (see
            // `loading::WorldState`). Every boot starts `Absent`; a scenario boot
            // queues `Live` from a `Startup` system rather than inserting the
            // `Live` state at build: Bevy runs the *initial* `StateTransition`
            // BEFORE `PreStartup` (see `bevy_app::main_schedule`), so a
            // build-inserted `Live` fires the `OnEnter(WorldState::Live)`
            // world-spawn chain before `Startup` has inserted its resources
            // (`RealSpaceRoot` etc.) — every spawn system panics on missing
            // resources. Queued from `Startup`, the transition applies at the
            // first regular `StateTransition` — same frame, after the `Startup`
            // command flush, behind the loading screen.
            .insert_state(loading::WorldState::Absent)
            .add_systems(
                Startup,
                move |mut next: ResMut<NextState<loading::WorldState>>| {
                    if !menu_boot {
                        next.set(loading::WorldState::Live);
                    }
                },
            )
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
            .add_plugins(game_context::GameContextPlugin)
            .add_plugins(SpawnPlugin)
            .add_plugins(surface_settle::SurfaceSettlePlugin)
            .add_plugins(structures::StructuresPlugin)
            .add_plugins(runway::RunwayPlugin)
            .add_plugins(RenderingPlugin)
            .add_plugins(GameLocalPhysicsPlugin)
            .add_plugins(GameAeroPlugin)
            .add_plugins(flight_config::FlightConfigPlugin)
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
            // The shared game-UI kit: design tokens, frosted-glass panels, and the
            // widget library every menu/editor screen composes (crates/interface/ui).
            .add_plugins(thalos_ui::ThalosUiPlugin)
            .add_plugins(HudPlugin)
            .add_plugins(PauseMenuPlugin)
            .add_plugins(main_menu::MainMenuPlugin)
            .add_plugins(SettingsMenuPlugin)
            // FPS/FRAME_TIME diagnostics for `hud/fps_overlay` (not part of
            // `DefaultPlugins`, so add it explicitly).
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            // Always-on perf telemetry: the F3 debug view, the runtime.jsonl
            // perf lane, and the opt-in full-rate recorder. Also owns
            // RenderDiagnosticsPlugin (GPU pass timings) for every lane —
            // interactive and headless.
            .add_plugins(perf::PerfPlugin)
            .add_plugins(ScenarioMenuPlugin)
            .add_plugins(NavballPlugin)
            .add_plugins(PhotoModePlugin)
            .add_plugins(ScreenshotPlugin)
            // Applies WindowSettings to the live window (mode / vsync / monitor /
            // windowed size) and folds the user UI scale into the fractional-HiDPI
            // crisp-text compensation.
            .add_plugins(window_settings::WindowSettingsPlugin)
            // Graphics preferences — e.g. the volumetric-cloud toggle read by
            // `rendering::clouds::drive_clouds`. (Registers the type; the resource
            // is inserted above and persisted by `AppSettingsPlugin`.)
            .add_plugins(graphics_settings::GraphicsSettingsPlugin)
            // Measurement-unit preference — metric vs imperial, read by the HUD
            // formatters in `hud::format`.
            .add_plugins(units_settings::UnitsSettingsPlugin)
            .add_plugins(ViewPlugin)
            .add_plugins(ShipViewPlugin)
            .add_plugins(relaunch::RelaunchPlugin)
            .add_plugins(shipyard_editor::ShipyardEditorPlugin)
            .add_plugins(base_editor::BaseEditorPlugin)
            // Shared god-view camera (base editor + space-center hub) and the KSP-style
            // space-center hub itself.
            .add_plugins(god_view::GodViewPlugin)
            .add_plugins(space_center::SpaceCenterPlugin)
            .add_plugins(BodyTreePanelPlugin)
            .add_plugins(mem_diag::MemDiagPlugin)
            .add_plugins(DebugPlugin);

        // Headless graphics settings are request-scoped and must never rewrite
        // the player's persisted preferences. Interactive launches keep the
        // unified settings autosave; capture hosts deliberately do not.
        if !headless {
            app.add_plugins(settings::AppSettingsPlugin);
        }

        // F8/F9 are a developer collaboration surface: the interactive app
        // gets the egui catalog manager and the quick-save prompt, while the
        // headless host only consumes the same authored JSON data.
        if screenshot_config.is_none() {
            app.add_plugins(viewpoints::ViewpointManagerPlugin)
                .add_plugins(viewpoints::quick_save::QuickSaveViewpointPlugin);
        }

        // Headless screenshot: the fixed-step runner (no winit event loop) plus the
        // off-screen capture driver + its resolved config, layered over the fully
        // built game app. The persistent agent lane keeps this same process alive
        // for later capture requests; the cold verification lane still exits after
        // one image.
        if let Some(config) = screenshot_config {
            app.add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
                1.0 / 60.0,
            )))
            .add_plugins(screenshot::HeadlessScreenshotPlugin {
                persistent: persistent_capture,
            })
            .insert_resource(config);
        }

        app
    }

    pub fn run(self) {
        self.build().run();
    }
}
