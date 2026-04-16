mod bridge;
mod camera;
mod coords;
mod ghost_bodies;
mod hud;
mod maneuver;
mod rendering;
mod sky_render;
mod star_flare;
mod target;
mod trajectory_rendering;

use std::sync::Arc;

use bevy::asset::AssetPlugin;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::prelude::*;
use bevy::window::{MonitorSelection, PresentMode, WindowMode};
use thalos_physics::{
    body_state_provider::BodyStateProvider,
    patched_conics::PatchedConics,
    simulation::{Simulation, SimulationConfig},
    types::{StateVector, load_solar_system},
};

use bridge::BridgePlugin;
use camera::CameraPlugin;
use ghost_bodies::GhostBodiesPlugin;
use hud::HudPlugin;
use maneuver::ManeuverPlugin;
use rendering::{RenderingPlugin, SimulationState};
use target::TargetPlugin;
use thalos_planet_rendering::PlanetRenderingPlugin;
use trajectory_rendering::TrajectoryRenderingPlugin;

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

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    // ------------------------------------------------------------------
    // 1. Load the solar system definition from the RON asset file.
    // ------------------------------------------------------------------
    let ron_source = std::fs::read_to_string("assets/solar_system.ron")
        .expect("Could not read assets/solar_system.ron — run from the workspace root");

    let system = load_solar_system(&ron_source).expect("Failed to parse solar_system.ron");

    // ------------------------------------------------------------------
    // 2. Print a startup banner.
    // ------------------------------------------------------------------
    println!("╔══════════════════════════════════════════╗");
    println!("║             T H A L O S                  ║");
    println!("╚══════════════════════════════════════════╝");
    println!("  System:           {}", system.name);
    println!("  Bodies:           {}", system.bodies.len());

    // ------------------------------------------------------------------
    // 3. Build the runtime patched-conics provider.
    // ------------------------------------------------------------------
    println!(
        "  Using patched-conics runtime body states ({:.0}-year span).",
        RUNTIME_TIME_SPAN / 3.156e7,
    );
    let ephemeris: Arc<dyn BodyStateProvider> =
        Arc::new(PatchedConics::new(&system, RUNTIME_TIME_SPAN));

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

    let homeworld_state = ephemeris.query_body(homeworld_id, 0.0);
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
            (SimStage::Physics, SimStage::Sync, SimStage::Camera).chain(),
        )
        .insert_resource(ClearColor(Color::srgb(0.02, 0.01, 0.04)))
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Thalos".into(),
                        present_mode: PresentMode::AutoNoVsync,
                        mode: WindowMode::BorderlessFullscreen(MonitorSelection::Primary),
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: "../../assets".to_string(),
                    ..default()
                }),
        )
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(bevy_egui::EguiPlugin::default())
        .insert_resource({
            let simulation = Simulation::new(
                ship_state,
                system.ship.thrust_acceleration,
                Arc::clone(&ephemeris),
                system.bodies.clone(),
                SimulationConfig::default(),
            );
            SimulationState {
                simulation,
                system,
                ephemeris,
            }
        })
        .add_plugins(bevy::prelude::MeshPickingPlugin)
        .add_plugins(PlanetRenderingPlugin)
        .add_plugins(CameraPlugin)
        .add_plugins(sky_render::SkyRenderPlugin)
        .add_plugins(star_flare::LensFlarePlugin)
        .add_plugins(RenderingPlugin)
        .add_plugins(BridgePlugin)
        .add_plugins(TrajectoryRenderingPlugin)
        .add_plugins(TargetPlugin)
        .add_plugins(GhostBodiesPlugin)
        .add_plugins(ManeuverPlugin)
        .add_plugins(HudPlugin)
        .run();
}
