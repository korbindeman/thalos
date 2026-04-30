use bevy::math::DVec3;
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

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
use crate::rendering::{FrameBodyStates, SimulationState};

/// Main input system for maneuver nodes.
pub(in crate::maneuver) fn maneuver_input(
    input: (
        Res<ButtonInput<KeyCode>>,
        Res<ButtonInput<MouseButton>>,
        Res<Time>,
    ),
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), With<ActiveCamera>>,
    sim: Option<Res<SimulationState>>,
    body_states: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    scale: Res<WorldScale>,
    flight_plan_view: Res<FlightPlanView>,
    mut plan: ResMut<ManeuverPlan>,
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
    let (keys, mouse, time) = input;
    let (hover_map, hitboxes) = picking;

    let pointer_on_arrow = hover_map
        .0
        .values()
        .any(|hovers| hovers.keys().any(|entity| hitboxes.get(*entity).is_ok()));

    if keys.just_pressed(KeyCode::KeyN) {
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

    if keys.just_pressed(KeyCode::Escape) && matches!(*mode, InteractionMode::PlacingNode { .. }) {
        *mode = InteractionMode::Idle;
    }

    if keys.just_pressed(KeyCode::Delete) || keys.just_pressed(KeyCode::Backspace) {
        if let Some(id) = selected.id {
            writer.write(ManeuverEvent::DeleteNode { id });
            selected.id = None;
        }
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
            if mouse.pressed(MouseButton::Left) {
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
                if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
                    modifier_scale *= 0.1;
                }
                if keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight) {
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
            } else {
                *mode = InteractionMode::Idle;
            }
        }

        InteractionMode::SlidingNode => {
            if mouse.pressed(MouseButton::Left) {
                if let Some(sel_id) = selected.id {
                    let node_time = plan
                        .nodes
                        .iter()
                        .find(|n| n.id == sel_id)
                        .map(|n| n.time)
                        .unwrap_or(0.0);
                    let closest = if flight_plan_view.focused_ghost().is_some() {
                        closest_trail_point(
                            prediction,
                            states,
                            &origin,
                            &scale,
                            &sim.system,
                            sim.ephemeris.as_ref(),
                            &flight_plan_view,
                            camera,
                            cam_transform,
                            cursor_pos,
                        )
                    } else {
                        let coasts = slide_search_segments(&plan, prediction, sel_id);
                        closest_trail_point_on_orbit(
                            &coasts,
                            prediction,
                            node_time,
                            states,
                            &origin,
                            &scale,
                            &sim.system,
                            &flight_plan_view,
                            camera,
                            cam_transform,
                            cursor_pos,
                        )
                    };
                    if let Some(closest) = closest {
                        // The slide-rebuild throttle in `handle_maneuver_events`
                        // can leave the cached prediction up to ~100 ms behind
                        // `node.time`. Capture the marker pose straight from
                        // the chosen sample so the rendered slide sphere stays
                        // pinned to the orbit the user is dragging along, even
                        // when sampling the (stale) prediction at the new time
                        // would otherwise resolve onto a different leg.
                        let body = sim.ephemeris.query_body(closest.anchor_body, closest.time);
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
            } else {
                // Fallback: mouse released without `slide_sphere_drag_end`
                // having fired (e.g. picking entity despawned mid-drag). Mirror
                // its cleanup so we don't leak preview state or skip the final
                // rebuild.
                *mode = InteractionMode::Idle;
                plan.dirty = true;
                slide_preview.world_pos = None;
                slide_preview.frame = None;
            }
        }

        InteractionMode::PlacingNode {
            snap_time,
            snap_world_pos,
            snap_anchor_body,
        } => {
            let closest = closest_trail_point(
                prediction,
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

            if mouse.just_pressed(MouseButton::Left) {
                if let (Some(trail_time), Some(reference_body)) = (*snap_time, *snap_anchor_body) {
                    writer.write(ManeuverEvent::PlaceNode {
                        trail_time,
                        reference_body,
                    });
                    *mode = InteractionMode::Idle;
                }
            }
            return;
        }

        InteractionMode::Idle => {}
    }

    if mouse.just_pressed(MouseButton::Left) && !pointer_on_arrow {
        if let Some(id) = closest_node(
            &plan,
            prediction,
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
