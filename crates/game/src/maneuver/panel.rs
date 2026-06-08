use bevy::math::DVec3;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

use super::state::{ManeuverEvent, ManeuverPlan, NodeBurnPhase, NodeDeltaV, SelectedNode};
use crate::rendering::SimulationState;

/// Bottom panel: node editor shown when a node is selected.
pub(super) fn node_editor_panel(
    mut contexts: EguiContexts,
    plan: Res<ManeuverPlan>,
    mut selected: ResMut<SelectedNode>,
    sim: Option<Res<SimulationState>>,
    mut node_dv: ResMut<NodeDeltaV>,
    mut writer: bevy::ecs::message::MessageWriter<ManeuverEvent>,
) {
    let Some(sel_id) = selected.id else { return };
    let Some(node_idx) = plan.nodes.iter().position(|n| n.id == sel_id) else {
        selected.id = None;
        return;
    };

    let burn_time = plan.nodes[node_idx].time;
    let total_dv = plan.nodes[node_idx].delta_v.length();
    let phase = plan.nodes[node_idx].phase;
    let executed = phase == NodeBurnPhase::Executed;
    let executing = phase == NodeBurnPhase::Executing;
    let sim_time = sim.as_ref().map(|s| s.simulation.sim_time()).unwrap_or(0.0);
    let time_until = burn_time - sim_time;

    // Tsiolkovsky burn duration at the ship's *current* mass — close
    // enough for the panel display; the live integrator picks up the
    // mass-anchor at the burn's actual start time.
    let burn_duration = sim
        .as_ref()
        .map(|s| s.simulation.estimated_burn_duration(total_dv))
        .unwrap_or(0.0);

    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::TopBottomPanel::bottom("node_editor").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.heading("Maneuver Node");
            ui.separator();
            ui.label(format!("Node #{}", sel_id.0));
            if executed {
                ui.separator();
                ui.colored_label(egui::Color32::from_rgb(120, 180, 120), "EXECUTED");
            } else if executing {
                ui.separator();
                ui.colored_label(egui::Color32::from_rgb(230, 200, 90), "BURNING");
            }
            ui.separator();
            ui.label(format!("T{:+.0}s", time_until));
            ui.separator();
            ui.label(format!("Total \u{0394}v: {:.1} m/s", total_dv));
            if !executed {
                ui.label(format!("Est. burn: {:.1}s", burn_duration));
            }
            ui.separator();
            let delete_label = if executed { "Dismiss" } else { "Delete" };
            if ui.button(delete_label).clicked() {
                writer.write(ManeuverEvent::DeleteNode { id: sel_id });
                selected.id = None;
            }
        });

        if executed {
            // A spent burn is a record, not a plan — show its components but
            // don't offer editing handles that would imply it can still change.
            ui.horizontal(|ui| {
                ui.label(format!(
                    "P: {:.1}   N: {:.1}   R: {:.1} m/s",
                    node_dv.prograde, node_dv.normal, node_dv.radial
                ));
            });
            return;
        }

        ui.horizontal(|ui| {
            let mut pg = node_dv.prograde as f32;
            let mut nm = node_dv.normal as f32;
            let mut rd = node_dv.radial as f32;

            let changed_pg = ui
                .add(
                    egui::DragValue::new(&mut pg)
                        .speed(1.0)
                        .prefix("P: ")
                        .suffix(" m/s"),
                )
                .changed();
            let changed_nm = ui
                .add(
                    egui::DragValue::new(&mut nm)
                        .speed(1.0)
                        .prefix("N: ")
                        .suffix(" m/s"),
                )
                .changed();
            let changed_rd = ui
                .add(
                    egui::DragValue::new(&mut rd)
                        .speed(1.0)
                        .prefix("R: ")
                        .suffix(" m/s"),
                )
                .changed();

            if changed_pg || changed_nm || changed_rd {
                node_dv.prograde = pg as f64;
                node_dv.normal = nm as f64;
                node_dv.radial = rd as f64;
                writer.write(ManeuverEvent::AdjustNode {
                    id: sel_id,
                    delta_v: DVec3::new(pg as f64, nm as f64, rd as f64),
                });
            }
        });
    });
}
