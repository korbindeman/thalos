//! HUD overlay using egui — time controls, camera focus, throttle, fuel.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use thalos_physics::body_state_provider::BodyStateProvider;
use thalos_physics::orbital_math::cartesian_to_elements;
use thalos_physics::types::StateVector;
use thalos_shipyard::{FuelTank, PartResources, Resource};

use crate::camera::{CameraFocus, ShipCameraMode};
use crate::fuel::ThrottleState;
use crate::photo_mode::not_in_photo_mode;
use crate::rendering::{CelestialBody, SimulationState};
use crate::view::ViewMode;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            bevy_egui::EguiPrimaryContextPass,
            time_control_panel.run_if(not_in_photo_mode),
        );
    }
}

fn time_control_panel(
    mut contexts: EguiContexts,
    mut sim: ResMut<SimulationState>,
    focus: Res<CameraFocus>,
    view: Res<ViewMode>,
    cam_mode: Res<ShipCameraMode>,
    bodies: Query<(&CelestialBody, &Name)>,
    diagnostics: Res<DiagnosticsStore>,
    throttle: Res<ThrottleState>,
    tanks: Query<&PartResources, With<FuelTank>>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    egui::TopBottomPanel::top("time_control").show(ctx, |ui| {
        ui.horizontal(|ui| {
            // Camera focus indicator.
            let focus_name = focus
                .target
                .and_then(|entity| bodies.get(entity).ok())
                .map(|(_, name)| name.as_str().to_string())
                .unwrap_or_else(|| "\u{2014}".to_string());
            ui.label(format!("Focus: {}", focus_name));
            if *view == ViewMode::Ship {
                ui.separator();
                ui.label(format!("Cam: {} (V)", cam_mode.label()));
            }
            ui.separator();

            // Warp controls.
            ui.label("Warp:");
            if ui.button("<").clicked() {
                sim.simulation.warp.decrease();
            }
            ui.label(sim.simulation.warp.label());
            if ui.button(">").clicked() {
                sim.simulation.warp.increase();
            }

            ui.separator();

            // Simulation time display.
            let t = sim.simulation.sim_time();
            let days = (t / 86400.0) as u64;
            let hours = ((t % 86400.0) / 3600.0) as u64;
            let minutes = ((t % 3600.0) / 60.0) as u64;
            ui.label(format!("T+ {}d {:02}h {:02}m", days, hours, minutes));

            ui.separator();

            let fps = diagnostics
                .get(&FrameTimeDiagnosticsPlugin::FPS)
                .and_then(|d| d.smoothed())
                .unwrap_or(0.0);
            ui.label(format!("FPS: {:.0}", fps));

            ui.separator();
            ui.label(format!("Thr: {:>3.0}%", throttle.commanded * 100.0));
            // Show "(cut)" when fuel limited the actual thrust below
            // commanded, so the player knows the engine is starving
            // even before a tank reads zero.
            if throttle.commanded > 0.0
                && throttle.effective + 1e-3 < throttle.commanded
            {
                ui.colored_label(egui::Color32::from_rgb(220, 110, 60), "(cut)");
            }

            // Per-resource fuel readout, summed across every tank.
            // Each resource only shows when at least one tank holds it.
            for res in [Resource::Methane, Resource::Lox] {
                let (amount, capacity) = tanks
                    .iter()
                    .filter_map(|t| t.pools.get(&res).map(|p| (p.amount, p.capacity)))
                    .fold((0.0_f32, 0.0_f32), |(a, c), (pa, pc)| (a + pa, c + pc));
                if capacity > 0.0 {
                    let pct = (amount / capacity) * 100.0;
                    let label = format!("{}: {:>3.0}%", res.display_name(), pct);
                    if pct < 10.0 {
                        ui.colored_label(egui::Color32::from_rgb(220, 110, 60), label);
                    } else {
                        ui.label(label);
                    }
                }
            }

            ui.separator();

            // Orbital readouts — altitude / apoapsis / periapsis, all
            // measured above the SOI body's surface (sea-level).
            let ship = sim.simulation.ship_state();
            let anchor_id = sim.simulation.dominant_body();
            let body = &sim.simulation.bodies()[anchor_id];
            let body_state = sim
                .simulation
                .ephemeris()
                .query_body(anchor_id, sim.simulation.sim_time());
            let rel = StateVector {
                position: ship.position - body_state.position,
                velocity: ship.velocity - body_state.velocity,
            };
            let altitude_m = rel.position.length() - body.radius_m;
            ui.label(format!("Alt: {}", format_altitude(altitude_m)));
            match cartesian_to_elements(rel, body.gm) {
                Some(el) => {
                    let ap_label = if el.apoapsis_m.is_finite() {
                        format_altitude(el.apoapsis_m - body.radius_m)
                    } else {
                        // Hyperbolic / parabolic — apoapsis is undefined.
                        "\u{2014}".to_string()
                    };
                    ui.label(format!("Ap: {}", ap_label));
                    ui.label(format!(
                        "Pe: {}",
                        format_altitude(el.periapsis_m - body.radius_m)
                    ));
                }
                None => {
                    ui.label("Ap: \u{2014}");
                    ui.label("Pe: \u{2014}");
                }
            }
        });
    });
}

/// Format an altitude in meters as a short human-readable string.
/// Uses km up to 9999, then Mm, then Gm — keeps the HUD compact across
/// the ~100 km low-orbit to ~10⁹ km interplanetary range.
fn format_altitude(meters: f64) -> String {
    let abs = meters.abs();
    if abs < 9_999_500.0 {
        format!("{:.1} km", meters / 1_000.0)
    } else if abs < 9_999_500_000.0 {
        format!("{:.1} Mm", meters / 1_000_000.0)
    } else {
        format!("{:.2} Gm", meters / 1_000_000_000.0)
    }
}
