//! HUD overlay using egui — time controls and camera focus indicator.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

use crate::camera::CameraFocus;
use crate::rendering::{CelestialBody, ShowOrbits, SimulationState};

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(bevy_egui::EguiPrimaryContextPass, time_control_panel);
    }
}

fn time_control_panel(
    mut contexts: EguiContexts,
    mut sim: ResMut<SimulationState>,
    focus: Res<CameraFocus>,
    bodies: Query<(&CelestialBody, &Name)>,
    diagnostics: Res<DiagnosticsStore>,
    mut show_orbits: ResMut<ShowOrbits>,
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
            ui.separator();

            // Warp controls.
            ui.label("Warp:");
            if ui.button("<").clicked() {
                sim.simulation.decrease_warp();
            }
            ui.label(sim.simulation.warp_label());
            if ui.button(">").clicked() {
                sim.simulation.increase_warp();
            }

            if sim.simulation.is_observation_mode() {
                ui.label(
                    egui::RichText::new("SIM OFF")
                        .color(egui::Color32::from_rgb(255, 180, 0))
                        .strong(),
                )
                .on_hover_text(
                    "Observation mode: ship physics and trajectory prediction are \
                     paused. Lower warp to resume.",
                );
            }

            ui.separator();

            // Simulation time display.
            let t = sim.simulation.sim_time();
            let days = (t / 86400.0) as u64;
            let hours = ((t % 86400.0) / 3600.0) as u64;
            let minutes = ((t % 3600.0) / 60.0) as u64;
            ui.label(format!("T+ {}d {:02}h {:02}m", days, hours, minutes));

            ui.separator();

            let orbits_label = if show_orbits.0 { "Orbits: On" } else { "Orbits: Off" };
            if ui.button(orbits_label).clicked() {
                show_orbits.0 = !show_orbits.0;
            }

            ui.separator();

            let fps = diagnostics
                .get(&FrameTimeDiagnosticsPlugin::FPS)
                .and_then(|d| d.smoothed())
                .unwrap_or(0.0);
            ui.label(format!("FPS: {:.0}", fps));
        });
    });
}
