mod autopilot;
mod bake_check;
mod body_tree_panel;
mod bridge;
mod camera;
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
mod maneuver;
mod map_view;
mod navball;
mod navigation;
mod pause_menu;
mod photo_mode;
mod player_controller;
mod reflection_probe;
mod rendering;
mod screenshot;
mod ship_view;
mod sky_render;
mod solar_system_state;
mod star_flare;
mod target;
mod view;
mod warp_to_maneuver;

use std::sync::Arc;

use bevy::asset::AssetPlugin;
use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use bevy::window::{MonitorSelection, WindowMode};
use thalos_input::game::GameInputPlugin;
use thalos_input::settings::InputSettings;
use thalos_physics_canonical::{
    body_trajectory_provider::BodyTrajectoryProvider,
    canonical::{AuthorityMode, Epoch, WorldPhysicsConfig},
    gravity_mode::GravityMode,
    parsing::load_solar_system_from_dir,
    simulation::{Simulation, SimulationConfig},
    terrain_provider::SharedTerrainRegistry,
    types::{AttitudeState, ShipParameters, StateVector, VesselKind},
};
use thalos_terrain_render::ThalosTerrainPlugin;

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
use screenshot::ScreenshotPlugin;
use ship_view::ShipViewPlugin;
use solar_system_state::{SimulationState, SolarSystemStatePlugin};
use target::TargetPlugin;
use thalos_planet_rendering::PlanetRenderingPlugin;
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
/// `SharedTerrainRegistry` lives in the pure-Rust `thalos_physics_canonical` crate.
#[derive(Resource, Clone)]
pub struct GameTerrainRegistry(pub SharedTerrainRegistry);

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

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
    //    Default vessel is EVA (player on foot) standing on the Thalos
    //    surface. The canonical CraftState is the player — KSP-style:
    //    one craft, EVA or Ship, distinguished by `VesselKind`.
    //
    //    Spawn pose is a fixed body-fixed direction over Thalos plus a
    //    safe drop margin above the body radius. Terrain heights aren't
    //    known at this point (the registry is populated later by the
    //    rendering crate as bakes load), so we err above the worst-case
    //    elevation and let local physics resolve the few-metre drop to
    //    the rendered surface.
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

    // Spawn at the sub-stellar point so the player wakes up in
    // daylight. The direction from Thalos toward Pyros (origin in
    // heliocentric inertial) is `-homeworld_state.position`; expressed
    // in body-fixed coordinates that's the body-fixed direction
    // pointing at the star right now.
    let sun_dir_inertial = (-homeworld_state.position).normalize();
    let spawn_dir_inertial = sun_dir_inertial;
    // Drop margin above the body radius. Has to clear Thalos's max
    // peaks (~±8 km from body radius) since the terrain registry isn't
    // loaded yet — anything less spawns inside a mountain. The local
    // physics bubble will pin us to the rendered surface within the
    // first few frames once the surface arrives.
    let spawn_drop_height_m = 12_000.0;
    let spawn_offset = spawn_dir_inertial * (homeworld.radius_m + spawn_drop_height_m);
    let surface_velocity = homeworld_state.angular_velocity.cross(spawn_offset);
    let ship_state = StateVector {
        position: homeworld_state.position + spawn_offset,
        velocity: homeworld_state.velocity + surface_velocity,
    };

    println!(
        "  Player (EVA):    standing on {} (daylight side)",
        homeworld.name
    );

    // ------------------------------------------------------------------
    // 5. Build and run the Bevy app.
    // ------------------------------------------------------------------
    App::new()
        .configure_sets(
            Update,
            (
                SimStage::Physics.run_if(pause_menu::not_game_paused),
                SimStage::Sync,
                SimStage::Camera.run_if(pause_menu::not_game_paused),
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
                    primary_window: Some(Window {
                        title: "Thalos".into(),
                        mode: WindowMode::BorderlessFullscreen(MonitorSelection::Primary),
                        ..default()
                    }),
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
        // `ThalosTerrainPlugin` wraps `thalos_udlod::TerrainPlugin`, which
        // adds `BigSpaceDefaultPlugins` itself when the `high_precision`
        // feature is enabled. Adding the plugin again here would panic on
        // duplicate registration.
        .add_plugins(ThalosTerrainPlugin)
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
            // Mark the player as EVA before anything else reads vessel
            // kind. `set_ship_params` then pushes the EVA parameter
            // preset (90 kg, no thrust, no torque) — the throttle and
            // attitude-command pipes are still wired up but produce no
            // motion, so HUD readouts stay coherent.
            simulation.set_vessel_kind(VesselKind::Eva);
            simulation.set_ship_params(ShipParameters::eva());
            // Stand upright at the spawn point: body +Z (dorsal) along
            // local-up (radial-out from the homeworld), body +Y (nose)
            // tangent to the surface roughly along the body's eastward
            // direction. Matches the convention used by the EVA
            // controller's `level_orientation`, so the player faces
            // forward when they start walking.
            let up = spawn_dir_inertial;
            let east_seed = homeworld_state.orientation * DVec3::Y;
            let east_tangent = (east_seed - up * east_seed.dot(up)).normalize();
            let right = up.cross(east_tangent).normalize();
            let basis = bevy::math::DMat3::from_cols(right, east_tangent, up);
            simulation.set_attitude(AttitudeState {
                orientation: DQuat::from_mat3(&basis),
                angular_velocity: DVec3::ZERO,
            });
            // Start in `OnRails`: the brief Kepler arc holds the player
            // up until `attach_terrain_patch_when_close` fires (the
            // 500 m drop puts us well inside the AGL handoff band), at
            // which point `manage_authority` hands translation to Avian
            // and the terrain collider catches the fall.
            simulation.transition_authority(AuthorityMode::OnRails { trajectory: 0 });
            SimulationState {
                simulation,
                system,
                ephemeris,
            }
        })
        .insert_resource(GameTerrainRegistry(terrain_registry))
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
        .add_plugins(PlanetRenderingPlugin)
        .add_plugins(GameInputPlugin)
        .add_plugins(GameInputGatePlugin)
        .add_plugins(CameraPlugin)
        .add_plugins(FreeCamPlugin)
        .add_plugins(reflection_probe::ReflectionProbePlugin)
        .add_plugins(sky_render::SkyRenderPlugin)
        .add_plugins(star_flare::LensFlarePlugin)
        .add_plugins(LoadingScreenPlugin)
        .add_plugins(RenderingPlugin)
        .add_plugins(GameLocalPhysicsPlugin)
        .add_plugins(PlayerControllerPlugin)
        .add_plugins(MapViewPlugin)
        .add_plugins(BridgePlugin)
        .add_plugins(FuelPlugin)
        .add_plugins(EnginePlugin)
        .add_plugins(TargetPlugin)
        .add_plugins(FlightPlanViewPlugin)
        .add_plugins(ManeuverPlugin)
        .add_plugins(NavigationPlugin)
        .add_plugins(AutopilotPlugin)
        .add_plugins(ControlLocksPlugin)
        .add_plugins(WarpToManeuverPlugin)
        .add_plugins(HudPlugin)
        .add_plugins(PauseMenuPlugin)
        .add_plugins(NavballPlugin)
        .add_plugins(PhotoModePlugin)
        .add_plugins(ScreenshotPlugin)
        .add_plugins(ViewPlugin)
        .add_plugins(ShipViewPlugin)
        .add_plugins(BodyTreePanelPlugin)
        .add_plugins(DebugPlugin)
        .run();
}
