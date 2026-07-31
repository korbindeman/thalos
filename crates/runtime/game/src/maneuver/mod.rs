mod helpers;
mod interaction;
mod panel;
mod render;
mod state;

use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use thalos_input::enhanced::ContextActivity;
use thalos_input::game::GameManeuverPrecisionContext;

use interaction::{
    arrow_drag_end, arrow_drag_start, end_drag_on_release, handle_maneuver_events, maneuver_input,
    slide_sphere_drag_end, slide_sphere_drag_start, sync_node_delta_v,
};
use panel::{
    handle_buttons as maneuver_editor_buttons, setup as setup_maneuver_editor,
    update_editor as update_maneuver_editor,
};
use render::{
    manage_arrow_handles, manage_node_markers, spawn_snap_indicator, update_arrow_transforms,
    update_snap_indicator,
};
use state::{ArrowHitbox, ArrowStretchState, NodeSlideSphere, SelectedNodeView, SlidePreview};

pub use state::{
    GameNode, InteractionMode, ManeuverEvent, ManeuverPlan, NodeBurnPhase, NodeDeltaV, NodeSource,
    SelectedNode,
};

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

/// Precision modifier bindings are only active during a maneuver edit that
/// actually consumes them. This keeps Shift/Ctrl free for throttle control in
/// normal flight.
fn sync_maneuver_precision_context(
    mut commands: Commands,
    mode: Res<InteractionMode>,
    contexts: Query<(Entity, &ContextActivity<GameManeuverPrecisionContext>)>,
) {
    let active = matches!(
        *mode,
        InteractionMode::DraggingArrow { .. } | InteractionMode::SlidingNode
    );
    let target = if active {
        ContextActivity::<GameManeuverPrecisionContext>::ACTIVE
    } else {
        ContextActivity::<GameManeuverPrecisionContext>::INACTIVE
    };
    for (entity, context) in &contexts {
        if **context != active {
            commands.entity(entity).insert(target);
        }
    }
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
    body_states: Res<crate::rendering::SolarSystemState>,
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
        sim.simulation.trajectory_branches(),
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
                Startup,
                setup_maneuver_editor.after(crate::hud::theme::init_theme),
            )
            // Outside the `not_game_paused` tuple below, deliberately: a drag
            // must end when the button comes up even if the release lands with
            // the pause menu open. It reads only the input intent and the mode,
            // so it is safe to run in any state.
            .add_systems(
                Update,
                end_drag_on_release
                    .before(update_camera_block)
                    .before(sync_maneuver_precision_context)
                    .before(maneuver_input)
                    .before(crate::SimStage::Physics),
            )
            .add_systems(
                Update,
                (
                    update_camera_block,
                    sync_maneuver_precision_context,
                    maneuver_input,
                    handle_maneuver_events.after(maneuver_input),
                    sync_node_delta_v.after(handle_maneuver_events),
                    update_selected_node_view.after(sync_node_delta_v),
                    manage_arrow_handles
                        .after(update_selected_node_view)
                        .run_if(
                            crate::photo_mode::not_in_photo_mode.and_then(crate::view::in_map_view),
                        ),
                    update_arrow_transforms.after(manage_arrow_handles).run_if(
                        crate::photo_mode::not_in_photo_mode.and_then(crate::view::in_map_view),
                    ),
                    manage_node_markers.after(update_selected_node_view).run_if(
                        crate::photo_mode::not_in_photo_mode.and_then(crate::view::in_map_view),
                    ),
                    update_snap_indicator.after(maneuver_input).run_if(
                        crate::photo_mode::not_in_photo_mode.and_then(crate::view::in_map_view),
                    ),
                )
                    .run_if(crate::pause_menu::not_game_paused)
                    .before(crate::SimStage::Physics),
            )
            .add_observer(arrow_drag_start)
            .add_observer(arrow_drag_end)
            .add_observer(slide_sphere_drag_start)
            .add_observer(slide_sphere_drag_end)
            // The maneuver editor is native Bevy UI: `update_maneuver_editor`
            // owns its visibility (selection + map-view gate), so it runs
            // unconditionally; `maneuver_editor_buttons` only fires on visible
            // (pickable) controls.
            .add_systems(Update, (update_maneuver_editor, maneuver_editor_buttons));
    }
}
