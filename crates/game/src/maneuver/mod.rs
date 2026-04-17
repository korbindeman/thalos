mod helpers;
mod interaction;
mod panel;
mod render;
mod state;

use bevy::picking::hover::HoverMap;
use bevy::prelude::*;

use interaction::{
    arrow_drag_end, arrow_drag_start, handle_maneuver_events, maneuver_input,
    slide_sphere_drag_end, slide_sphere_drag_start, sync_node_delta_v,
};
use panel::node_editor_panel;
use render::{
    manage_arrow_handles, manage_node_markers, spawn_snap_indicator, update_arrow_transforms,
    update_snap_indicator,
};
use state::{ArrowHitbox, ArrowStretchState, NodeSlideSphere, SelectedNodeView};

pub use state::{InteractionMode, ManeuverEvent, ManeuverPlan, NodeDeltaV, SelectedNode};

/// Block camera rotation whenever a maneuver element is hovered or any
/// non-Idle interaction mode is active.
fn update_camera_block(
    hover_map: Res<HoverMap>,
    hitboxes: Query<Entity, Or<(With<ArrowHitbox>, With<NodeSlideSphere>)>>,
    mode: Res<InteractionMode>,
    mut block: ResMut<crate::camera::BlockCameraInput>,
) {
    let pointer_on_element = hover_map
        .0
        .values()
        .any(|hovers| hovers.keys().any(|e| hitboxes.get(*e).is_ok()));

    block.0 = pointer_on_element || !matches!(*mode, InteractionMode::Idle);
}

/// Recomputes the cached world position and orbital frame for the selected node.
fn update_selected_node_view(
    selected: Res<SelectedNode>,
    plan: Res<ManeuverPlan>,
    sim: Option<Res<crate::rendering::SimulationState>>,
    body_states: Res<crate::rendering::FrameBodyStates>,
    origin: Res<crate::coords::RenderOrigin>,
    flight_plan_view: Res<crate::flight_plan_view::FlightPlanView>,
    mut selected_view: ResMut<SelectedNodeView>,
    mut last_selected: Local<Option<state::NodeId>>,
    mut last_plan_size: Local<usize>,
) {
    if *last_selected != selected.id {
        info!(
            "[maneuver] SelectedNode changed: {:?} -> {:?} (plan has {} nodes: ids {:?})",
            *last_selected,
            selected.id,
            plan.nodes.len(),
            plan.nodes.iter().map(|n| n.id).collect::<Vec<_>>()
        );
        *last_selected = selected.id;
    }
    if *last_plan_size != plan.nodes.len() {
        info!(
            "[maneuver] plan.nodes size changed: {} -> {} (ids {:?})",
            *last_plan_size,
            plan.nodes.len(),
            plan.nodes.iter().map(|n| n.id).collect::<Vec<_>>()
        );
        *last_plan_size = plan.nodes.len();
    }
    let Some(ref sim) = sim else {
        selected_view.world_pos = None;
        selected_view.frame = None;
        return;
    };
    let Some(prediction) = sim.simulation.prediction() else {
        selected_view.world_pos = None;
        selected_view.frame = None;
        return;
    };
    let Some(ref states) = body_states.states else {
        selected_view.world_pos = None;
        selected_view.frame = None;
        return;
    };

    match helpers::selected_node_world_and_frame(
        selected.id,
        &plan,
        prediction,
        states,
        &origin,
        &sim.system,
        sim.ephemeris.as_ref(),
        &flight_plan_view,
    ) {
        Some((world_pos, frame)) => {
            if selected_view.world_pos.is_none() && selected.id.is_some() {
                info!(
                    "[maneuver] selected_view.world_pos restored (id={:?})",
                    selected.id
                );
            }
            selected_view.world_pos = Some(world_pos);
            selected_view.frame = Some(frame);
        }
        None => {
            if selected_view.world_pos.is_some() {
                let node_info = selected
                    .id
                    .and_then(|id| plan.nodes.iter().find(|n| n.id == id))
                    .map(|n| format!("id={:?} time={:.2} ref_body={}", n.id, n.time, n.reference_body));
                let seg_ranges: Vec<(f64, f64)> = prediction
                    .segments
                    .iter()
                    .chain(prediction.baseline.iter())
                    .filter(|s| !s.samples.is_empty())
                    .map(|s| (s.samples[0].time, s.samples.last().unwrap().time))
                    .collect();
                warn!(
                    "[maneuver] selected_view.world_pos LOST — node {:?}, segments {:?}",
                    node_info, seg_ranges
                );
            }
            selected_view.world_pos = None;
            selected_view.frame = None;
        }
    }
}

pub struct ManeuverPlugin;

impl Plugin for ManeuverPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<ManeuverEvent>()
            .init_resource::<ManeuverPlan>()
            .init_resource::<NodeDeltaV>()
            .init_resource::<SelectedNode>()
            .init_resource::<SelectedNodeView>()
            .init_resource::<InteractionMode>()
            .init_resource::<ArrowStretchState>()
            .add_systems(Startup, spawn_snap_indicator)
            .add_systems(
                Update,
                (
                    update_camera_block,
                    maneuver_input,
                    handle_maneuver_events.after(maneuver_input),
                    sync_node_delta_v.after(handle_maneuver_events),
                    update_selected_node_view.after(sync_node_delta_v),
                    manage_arrow_handles.after(update_selected_node_view),
                    update_arrow_transforms.after(manage_arrow_handles),
                    manage_node_markers.after(update_selected_node_view),
                    update_snap_indicator.after(maneuver_input),
                )
                    .before(crate::SimStage::Physics),
            )
            .add_observer(arrow_drag_start)
            .add_observer(arrow_drag_end)
            .add_observer(slide_sphere_drag_start)
            .add_observer(slide_sphere_drag_end)
            .add_systems(bevy_egui::EguiPrimaryContextPass, node_editor_panel);
    }
}
