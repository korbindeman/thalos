use bevy::math::DVec3;
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use thalos_input::game::GameInputIntent;

use super::super::helpers::{
    closest_node, closest_trail_point, closest_trail_point_on_orbit, orbit_sensitivity_scale,
    orbital_frame_mat3, slide_search_segments,
};
use super::super::state::{
    ArrowHitbox, InteractionMode, ManeuverEvent, ManeuverPlan, NodeDeltaV, NodeSlideSphere,
    SELECT_THRESHOLD_PX, SelectedNode, SlidePreview,
};
use crate::camera::ActiveCamera;
use crate::coords::{RenderOrigin, WorldScale};
use crate::flight_plan_view::FlightPlanView;
use crate::rendering::{SimulationState, SolarSystemState};

/// Enforce the invariant **a drag mode implies the primary button is held**.
///
/// Deliberately its own system, running unconditionally and reading nothing
/// fallible. The release handling used to live inside [`maneuver_input`], below
/// that system's cursor / camera / simulation / prediction early-returns and
/// behind its `not_game_paused` gate — so releasing the button while any of
/// those held (pointer off the window, an aircraft with no orbital prediction,
/// the pause menu up) left [`InteractionMode`] in `DraggingArrow` *forever*.
/// A stuck drag mode keeps `GameManeuverPrecisionContext` active, and that
/// context consumes Shift/Ctrl above `GameFlightContext` in priority, so the
/// throttle ramp went dead for the rest of the session
/// (INC-20260730T222419Z). `arrow_drag_end` / `slide_sphere_drag_end` still
/// handle the ordinary case; they cannot cover this one, because a `DragEnd`
/// delivered to a despawned hitbox never reaches them.
pub(in crate::maneuver) fn end_drag_on_release(
    intent: Res<GameInputIntent>,
    mut mode: ResMut<InteractionMode>,
    mut plan: ResMut<ManeuverPlan>,
    mut slide_preview: ResMut<SlidePreview>,
) {
    if intent.primary_pressed {
        return;
    }
    match *mode {
        InteractionMode::DraggingArrow { .. } => *mode = InteractionMode::Idle,
        InteractionMode::SlidingNode => {
            // Mirror `slide_sphere_drag_end`: force the final rebuild against
            // the released node time, and drop the drag-local preview pose so
            // the next `update_selected_node_view` samples the fresh
            // prediction.
            *mode = InteractionMode::Idle;
            plan.dirty = true;
            slide_preview.world_pos = None;
            slide_preview.frame = None;
        }
        // Not drags: `PlacingNode` is a click-to-place mode toggled by N, and
        // neither it nor `Idle` holds the precision context open.
        InteractionMode::PlacingNode { .. } | InteractionMode::Idle => {}
    }
}

/// Main input system for maneuver nodes.
pub(in crate::maneuver) fn maneuver_input(
    intent: Res<GameInputIntent>,
    time: Res<Time<Real>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), With<ActiveCamera>>,
    sim: Option<Res<SimulationState>>,
    body_states: Res<SolarSystemState>,
    origin: Res<RenderOrigin>,
    scale: Res<WorldScale>,
    flight_plan_view: Res<FlightPlanView>,
    // Read-only: the one write this system used to make (`dirty` on a released
    // slide) moved to `end_drag_on_release`, so it no longer takes the lock.
    plan: Res<ManeuverPlan>,
    mut mode: ResMut<InteractionMode>,
    mut selected: ResMut<SelectedNode>,
    mut node_dv: ResMut<NodeDeltaV>,
    mut slide_preview: ResMut<SlidePreview>,
    picking: (
        Res<HoverMap>,
        Query<Entity, Or<(With<ArrowHitbox>, With<NodeSlideSphere>)>>,
    ),
    mut writer: bevy::ecs::message::MessageWriter<ManeuverEvent>,
) {
    let (hover_map, hitboxes) = picking;

    let pointer_on_arrow = hover_map
        .0
        .values()
        .any(|hovers| hovers.keys().any(|entity| hitboxes.get(*entity).is_ok()));

    if intent.toggle_place_node {
        if matches!(*mode, InteractionMode::PlacingNode { .. }) {
            *mode = InteractionMode::Idle;
        } else {
            *mode = InteractionMode::PlacingNode {
                snap_time: None,
                snap_world_pos: None,
                snap_anchor_body: None,
            };
        }
    }

    if intent.delete_node
        && let Some(id) = selected.id
    {
        writer.write(ManeuverEvent::DeleteNode { id });
        selected.id = None;
    }

    let Ok(window) = windows.single() else { return };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_transform)) = camera_q.single() else {
        return;
    };
    let Some(ref sim) = sim else { return };
    let Some(prediction) = sim.simulation.prediction() else {
        if let InteractionMode::PlacingNode {
            snap_time,
            snap_world_pos,
            snap_anchor_body,
            ..
        } = &mut *mode
        {
            *snap_time = None;
            *snap_world_pos = None;
            *snap_anchor_body = None;
        }
        return;
    };
    let Some(ref states) = body_states.states else {
        return;
    };

    match &mut *mode {
        InteractionMode::DraggingArrow {
            axis,
            axis_screen_dir,
            drag_origin,
            rate_sign,
            ..
        } => {
            // Reaching this arm means the button is still down:
            // `end_drag_on_release` clears the mode the frame it comes up.
            let screen_delta = cursor_pos - *drag_origin;
            let displacement = screen_delta.dot(*axis_screen_dir);
            let raw_rate = displacement as f64 * 10.0;

            // Scale drag gain by the post-burn orbit's semi-major axis so
            // the mapping from tug-pixels to trajectory shift stays
            // roughly uniform as the user stretches the trajectory out.
            let sensitivity_scale = selected
                .id
                .and_then(|id| plan.nodes.iter().find(|n| n.id == id))
                .and_then(|node| {
                    orbit_sensitivity_scale(
                        prediction,
                        node.time,
                        node.delta_v,
                        node.reference_body,
                        sim.ephemeris.as_ref(),
                        &sim.system,
                    )
                })
                .unwrap_or(1.0);

            // Precision modifiers: Shift = 10× finer, Ctrl = 100× finer.
            let mut modifier_scale = 1.0;
            if intent.precision_fine {
                modifier_scale *= 0.1;
            }
            if intent.precision_ultra {
                modifier_scale *= 0.01;
            }

            let rate = raw_rate * sensitivity_scale * modifier_scale;
            let dt = time.delta_secs_f64();

            // Arrow stretch visual follows the raw drag signal so it still
            // animates when sensitivity scaling drives `rate` very small.
            *rate_sign = if raw_rate.abs() < 0.01 {
                0.0
            } else {
                raw_rate.signum() as f32
            };

            let axis = *axis;
            match axis {
                0 => node_dv.prograde += rate * dt,
                1 => node_dv.normal += rate * dt,
                2 => node_dv.radial += rate * dt,
                _ => {}
            }

            if let Some(id) = selected.id {
                writer.write(ManeuverEvent::AdjustNode {
                    id,
                    delta_v: DVec3::new(node_dv.prograde, node_dv.normal, node_dv.radial),
                });
            }
            return;
        }

        InteractionMode::SlidingNode => {
            // As above: the button is still held, or `end_drag_on_release`
            // would already have ended the slide and run its cleanup.
            if let Some(sel_id) = selected.id {
                let node_time = plan
                    .nodes
                    .iter()
                    .find(|n| n.id == sel_id)
                    .map(|n| n.time)
                    .unwrap_or(0.0);
                let branch_stack = sim.simulation.trajectory_branches();
                let coasts = slide_search_segments(&plan, branch_stack, sel_id);
                let closest = closest_trail_point_on_orbit(
                    &coasts,
                    prediction,
                    node_time,
                    states,
                    &origin,
                    &scale,
                    &sim.system,
                    sim.ephemeris.as_ref(),
                    &flight_plan_view,
                    camera,
                    cam_transform,
                    cursor_pos,
                );
                if let Some(closest) = closest {
                    // The slide-rebuild throttle in `handle_maneuver_events`
                    // can leave the cached prediction up to ~100 ms behind
                    // `node.time`. Capture the marker pose straight from
                    // the chosen sample so the rendered slide sphere stays
                    // pinned to the orbit the user is dragging along, even
                    // when sampling the (stale) prediction at the new time
                    // would otherwise resolve onto a different leg.
                    let body = sim.ephemeris.state(
                        closest.anchor_body,
                        thalos_physics_canonical::canonical::Epoch(closest.time),
                    );
                    let frame = orbital_frame_mat3(
                        closest.sample_position,
                        closest.sample_velocity,
                        body.position,
                        body.velocity,
                    );
                    slide_preview.world_pos = Some(closest.world_pos);
                    slide_preview.frame = Some(frame);

                    writer.write(ManeuverEvent::SlideNode {
                        id: sel_id,
                        new_time: closest.time,
                    });
                }
            }
            return;
        }

        InteractionMode::PlacingNode {
            snap_time,
            snap_world_pos,
            snap_anchor_body,
        } => {
            let branch_stack = sim.simulation.trajectory_branches();
            let closest = closest_trail_point(
                prediction,
                branch_stack,
                states,
                &origin,
                &scale,
                &sim.system,
                sim.ephemeris.as_ref(),
                &flight_plan_view,
                camera,
                cam_transform,
                cursor_pos,
            );
            *snap_time = closest.as_ref().map(|p| p.time);
            *snap_world_pos = closest.as_ref().map(|p| p.world_pos);
            *snap_anchor_body = closest.as_ref().map(|p| p.anchor_body);

            if intent.primary_started
                && let (Some(trail_time), Some(reference_body)) = (*snap_time, *snap_anchor_body)
            {
                writer.write(ManeuverEvent::PlaceNode {
                    trail_time,
                    reference_body,
                });
                *mode = InteractionMode::Idle;
            }
            return;
        }

        InteractionMode::Idle => {}
    }

    if intent.primary_started && !pointer_on_arrow {
        if let Some(id) = closest_node(
            &plan,
            prediction,
            sim.simulation.trajectory_branches(),
            states,
            &origin,
            &scale,
            &sim.system,
            sim.ephemeris.as_ref(),
            &flight_plan_view,
            camera,
            cam_transform,
            cursor_pos,
            SELECT_THRESHOLD_PX,
        ) {
            if selected.id == Some(id) {
                *mode = InteractionMode::SlidingNode;
            } else {
                selected.id = Some(id);
            }
        } else {
            selected.id = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;

    fn dragging_arrow() -> InteractionMode {
        InteractionMode::DraggingArrow {
            axis: 0,
            positive: true,
            axis_screen_dir: Vec2::X,
            drag_origin: Vec2::ZERO,
            rate_sign: 0.0,
        }
    }

    /// The world deliberately has **no** window, camera, `SimulationState`,
    /// prediction, or `SolarSystemState` — every early-return condition in
    /// `maneuver_input` at once. That is the state the release used to be lost
    /// in; the invariant must hold without any of them.
    fn release_in_a_bare_world(mode: InteractionMode) -> World {
        let mut world = World::new();
        world.insert_resource(GameInputIntent {
            primary_pressed: false,
            ..Default::default()
        });
        world.insert_resource(mode);
        world.insert_resource(ManeuverPlan::default());
        world.insert_resource(SlidePreview::default());
        world.run_system_once(end_drag_on_release).unwrap();
        world
    }

    #[test]
    fn releasing_the_button_ends_an_arrow_drag_with_nothing_else_available() {
        let world = release_in_a_bare_world(dragging_arrow());
        assert!(matches!(
            *world.resource::<InteractionMode>(),
            InteractionMode::Idle
        ));
    }

    #[test]
    fn releasing_the_button_ends_a_slide_and_runs_its_cleanup() {
        let mut world = World::new();
        world.insert_resource(GameInputIntent {
            primary_pressed: false,
            ..Default::default()
        });
        world.insert_resource(InteractionMode::SlidingNode);
        world.insert_resource(ManeuverPlan::default());
        world.insert_resource(SlidePreview {
            world_pos: Some(Vec3::ONE),
            frame: Some(Mat3::IDENTITY),
        });
        world.run_system_once(end_drag_on_release).unwrap();

        assert!(matches!(
            *world.resource::<InteractionMode>(),
            InteractionMode::Idle
        ));
        // The final rebuild must still be requested, and the drag-local pose
        // dropped — otherwise the marker stays pinned to a stale sample.
        assert!(world.resource::<ManeuverPlan>().dirty);
        assert!(world.resource::<SlidePreview>().world_pos.is_none());
        assert!(world.resource::<SlidePreview>().frame.is_none());
    }

    #[test]
    fn a_held_button_leaves_the_drag_running() {
        let mut world = World::new();
        world.insert_resource(GameInputIntent {
            primary_pressed: true,
            ..Default::default()
        });
        world.insert_resource(dragging_arrow());
        world.insert_resource(ManeuverPlan::default());
        world.insert_resource(SlidePreview::default());
        world.run_system_once(end_drag_on_release).unwrap();

        assert!(matches!(
            *world.resource::<InteractionMode>(),
            InteractionMode::DraggingArrow { .. }
        ));
    }

    #[test]
    fn placing_a_node_survives_a_button_release() {
        // `PlacingNode` is a click-to-place mode, not a drag: releasing the
        // button that armed it must not cancel it.
        let world = release_in_a_bare_world(InteractionMode::PlacingNode {
            snap_time: None,
            snap_world_pos: None,
            snap_anchor_body: None,
        });
        assert!(matches!(
            *world.resource::<InteractionMode>(),
            InteractionMode::PlacingNode { .. }
        ));
    }
}
