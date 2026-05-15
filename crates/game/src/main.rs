mod autopilot;
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
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use bevy::window::{MonitorSelection, WindowMode};
use thalos_input::game::GameInputPlugin;
use thalos_input::settings::InputSettings;
use thalos_physics::{
    body_trajectory_provider::BodyTrajectoryProvider,
    canonical::{Epoch, WorldPhysicsConfig},
    gravity_mode::GravityMode,
    parsing::load_solar_system_from_dir,
    simulation::{Simulation, SimulationConfig},
    terrain_provider::SharedTerrainRegistry,
    types::{AttitudeState, StateVector},
};
use thalos_terrain::ThalosTerrainPlugin;

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
/// `SharedTerrainRegistry` lives in the pure-Rust `thalos_physics` crate.
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
    // 4. Resolve the ship's absolute initial state.
    //
    //    ShipDefinition.initial_state is relative to the homeworld.
    //    Add the homeworld's t=0 ephemeris state to get heliocentric coords.
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

    let homeworld_state = ephemeris.state(homeworld_id, Epoch::ZERO);
    let rel = system.ship.initial_state;
    let ship_state = StateVector {
        position: homeworld_state.position + rel.position,
        velocity: homeworld_state.velocity + rel.velocity,
    };

    let homeworld = &system.bodies[homeworld_id];
    let altitude_km = (rel.position.length() - homeworld.radius_m) / 1000.0;
    println!(
        "  Ship:            {:.0} km orbit around {}",
        altitude_km, homeworld.name,
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
        // `ThalosTerrainPlugin` wraps `bevy_terrain::TerrainPlugin`, which
        // adds `BigSpaceDefaultPlugins` itself when the `high_precision`
        // feature is enabled. Adding the plugin again here would panic on
        // duplicate registration.
        .add_plugins(ThalosTerrainPlugin)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
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
            // Spawn in level orbital flight: body +Y (nose) along prograde,
            // body +Z (dorsal) along local-up (radial-out from the
            // homeworld). This is the convention shared by the navball
            // (which puts the zenith of the local frame at the top of
            // the sphere texture) and the controls (pitch around the
            // wing axis, yaw around dorsal, roll around the nose), so
            // all three stay aligned when the craft is "upright".
            let prograde = rel.velocity.normalize();
            let radial_raw = rel.position.normalize();
            // Project radial perpendicular to prograde — for circular
            // orbits these are already orthogonal, but for eccentric
            // ones the velocity has a radial component.
            let dorsal = (radial_raw - radial_raw.dot(prograde) * prograde).normalize();
            // Right wing: body +X = body +Y × body +Z (right-hand rule).
            let right = prograde.cross(dorsal);
            let basis = bevy::math::DMat3::from_cols(right, prograde, dorsal);
            simulation.set_attitude(AttitudeState {
                orientation: DQuat::from_mat3(&basis),
                angular_velocity: DVec3::ZERO,
            });
            SimulationState {
                simulation,
                system,
                ephemeris,
                world_config,
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
