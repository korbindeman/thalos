//! HUD overlay using egui — time controls, camera focus, throttle, delta-v.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use thalos_physics::orbital_math::cartesian_to_elements;
use thalos_physics::types::StateVector;
use thalos_shipyard::{
    DeltaVEnvironment, DeltaVInputs, PartResources, aggregate_resource_totals, estimate_delta_v,
};

use crate::camera::{CameraFocus, CameraFocusTarget, ShipCameraMode};
use crate::controls::ControlLocks;
use crate::fuel::{ActivePropulsion, ThrottleState};
use crate::photo_mode::not_in_photo_mode;
use crate::rendering::{CelestialBody, PlayerShip, SimulationState};
use crate::view::ViewMode;
use crate::warp_to_maneuver::{WarpToManeuver, find_next_maneuver};

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            bevy_egui::EguiPrimaryContextPass,
            time_control_panel.run_if(not_in_photo_mode),
        );
    }
}

pub fn time_control_panel(
    mut contexts: EguiContexts,
    mut sim: ResMut<SimulationState>,
    focus: Res<CameraFocus>,
    view: Res<ViewMode>,
    cam_mode: Res<ShipCameraMode>,
    locks: Res<ControlLocks>,
    bodies: Query<(&CelestialBody, &Name)>,
    ship_name: Query<&Name, With<PlayerShip>>,
    diagnostics: Res<DiagnosticsStore>,
    throttle: Res<ThrottleState>,
    active_propulsion: Res<ActivePropulsion>,
    resources: Query<&PartResources>,
    mut warp_to: ResMut<WarpToManeuver>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    egui::TopBottomPanel::top("time_control").show(ctx, |ui| {
        ui.horizontal(|ui| {
            // Camera focus indicator.
            let focus_name = match focus.target {
                CameraFocusTarget::Body(body_id) => bodies
                    .iter()
                    .find(|(body, _)| body.body_id == body_id)
                    .map(|(_, name)| name.as_str().to_string()),
                CameraFocusTarget::Ship => ship_name
                    .single()
                    .ok()
                    .map(|name| name.as_str().to_string()),
                CameraFocusTarget::Ghost(ghost_focus) => bodies
                    .iter()
                    .find(|(body, _)| body.body_id == ghost_focus.body_id)
                    .map(|(_, name)| format!("Ghost: {}", name.as_str())),
                CameraFocusTarget::None => None,
            }
            .unwrap_or_else(|| "\u{2014}".to_string());
            ui.label(format!("Focus: {}", focus_name));
            if *view == ViewMode::Ship {
                ui.separator();
                ui.label(format!("Cam: {} (V)", cam_mode.label()));
            }
            ui.separator();

            // Warp controls. Disabled per [`ControlLocks::warp`] —
            // the same flag the keybinds in
            // [`crate::bridge::handle_warp_controls`] read. Whichever
            // programmatic system holds the lock (today: scheduled-burn
            // autopilot during the lead-down) owns warp; UI tweaks
            // would just get clobbered.
            ui.label("Warp:");
            ui.add_enabled_ui(!locks.warp, |ui| {
                if ui.button("<").clicked() {
                    sim.simulation.warp.decrease();
                    warp_to.cancel();
                }
                ui.label(sim.simulation.warp.label());
                if ui.button(">").clicked() {
                    sim.simulation.warp.increase();
                    warp_to.cancel();
                }

                // Warp-to-next-maneuver toggle. The label flips to a
                // stop sign while engaged so the player has unambiguous
                // on/off feedback without a separate indicator. The
                // status string to the right — populated each tick by
                // [`crate::warp_to_maneuver`] — shows the remaining
                // time (e.g. "Maneuver in 1h 23m").
                let label = if warp_to.active {
                    "■"
                } else {
                    "→ Maneuver"
                };
                if ui.button(label).clicked() {
                    if warp_to.active {
                        warp_to.cancel();
                    } else if find_next_maneuver(sim.simulation.sim_time(), &sim.simulation)
                        .is_some()
                    {
                        warp_to.active = true;
                    }
                }
            });
            if let Some(target) = warp_to.current.as_ref() {
                ui.colored_label(
                    egui::Color32::from_rgb(120, 200, 255),
                    target.label(sim.simulation.sim_time()),
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
            if throttle.commanded > 0.0 && throttle.effective + 1e-3 < throttle.commanded {
                ui.colored_label(egui::Color32::from_rgb(220, 110, 60), "(cut)");
            }

            let resource_totals = aggregate_resource_totals(resources.iter());
            let delta_v = estimate_delta_v(
                DeltaVEnvironment::Vacuum,
                DeltaVInputs {
                    dry_mass_kg: active_propulsion.dry_mass_kg,
                    wet_mass_kg: active_propulsion.wet_mass_kg,
                    total_thrust_n: active_propulsion.total_thrust_n,
                    mass_flow_kg_per_s: active_propulsion.mass_flow_kg_per_s,
                    power_draw_kw: active_propulsion.power_draw_kw,
                    reactant_fractions: &active_propulsion.reactant_fractions,
                    resources: &resource_totals,
                },
            );
            let dv_label = format!("Vac Δv: {}", format_delta_v(delta_v.delta_v_m_per_s));
            if active_propulsion.total_thrust_n > 0.0 && delta_v.delta_v_m_per_s <= 1.0 {
                ui.colored_label(egui::Color32::from_rgb(220, 110, 60), dv_label);
            } else {
                ui.label(dv_label);
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

fn format_delta_v(meters_per_second: f64) -> String {
    if meters_per_second.abs() >= 9_999.5 {
        format!("{:.2} km/s", meters_per_second / 1_000.0)
    } else {
        format!("{:.0} m/s", meters_per_second)
    }
}
