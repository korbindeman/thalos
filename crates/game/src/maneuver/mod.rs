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
use state::{ArrowHitbox, ArrowStretchState, NodeSlideSphere, SelectedNodeView, SlidePreview};

pub use state::{GameNode, InteractionMode, ManeuverEvent, ManeuverPlan, NodeDeltaV, SelectedNode};

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
///
/// While [`InteractionMode::SlidingNode`] is active, the slide handler has
/// already written a [`SlidePreview`] from the chosen sample on the orbit
/// being dragged along — prefer it over re-sampling the (throttle-stale)
/// prediction, which can otherwise snap the marker onto the wrong leg
/// mid-drag.
fn update_selected_node_view(
    selected: Res<SelectedNode>,
    plan: Res<ManeuverPlan>,
    sim: Option<Res<crate::rendering::SimulationState>>,
    body_states: Res<crate::rendering::FrameBodyStates>,
    origin: Res<crate::coords::RenderOrigin>,
    scale: Res<crate::coords::WorldScale>,
    flight_plan_view: Res<crate::flight_plan_view::FlightPlanView>,
    mode: Res<InteractionMode>,
    slide_preview: Res<SlidePreview>,
    mut selected_view: ResMut<SelectedNodeView>,
) {
    if matches!(*mode, InteractionMode::SlidingNode)
        && let (Some(world_pos), Some(frame)) = (slide_preview.world_pos, slide_preview.frame)
    {
        selected_view.world_pos = Some(world_pos);
        selected_view.frame = Some(frame);
        return;
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
        &scale,
        &sim.system,
        sim.ephemeris.as_ref(),
        &flight_plan_view,
    ) {
        Some((world_pos, frame)) => {
            selected_view.world_pos = Some(world_pos);
            selected_view.frame = Some(frame);
        }
        None => {
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
            .init_resource::<SlidePreview>()
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
                    manage_arrow_handles
                        .after(update_selected_node_view)
                        .run_if(crate::photo_mode::not_in_photo_mode.and(crate::view::in_map_view)),
                    update_arrow_transforms
                        .after(manage_arrow_handles)
                        .run_if(crate::photo_mode::not_in_photo_mode.and(crate::view::in_map_view)),
                    manage_node_markers
                        .after(update_selected_node_view)
                        .run_if(crate::photo_mode::not_in_photo_mode.and(crate::view::in_map_view)),
                    update_snap_indicator
                        .after(maneuver_input)
                        .run_if(crate::photo_mode::not_in_photo_mode.and(crate::view::in_map_view)),
                )
                    .before(crate::SimStage::Physics),
            )
            .add_observer(arrow_drag_start)
            .add_observer(arrow_drag_end)
            .add_observer(slide_sphere_drag_start)
            .add_observer(slide_sphere_drag_end)
            .add_systems(
                bevy_egui::EguiPrimaryContextPass,
                node_editor_panel
                    .run_if(crate::photo_mode::not_in_photo_mode.and(crate::view::in_map_view)),
            );
    }
}
