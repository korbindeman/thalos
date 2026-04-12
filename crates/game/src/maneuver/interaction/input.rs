use bevy::math::DVec3;
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use super::super::helpers::{closest_node, closest_trail_point};
use super::super::state::{
    ArrowHitbox, InteractionMode, ManeuverEvent, ManeuverPlan, NodeDeltaV, NodeSlideSphere,
    SELECT_THRESHOLD_PX, SelectedNode,
};
use crate::camera::OrbitCamera;
use crate::coords::RenderOrigin;
use crate::rendering::{FrameBodyStates, SimulationState};

/// Main input system for maneuver nodes.
pub(in crate::maneuver) fn maneuver_input(
    input: (
        Res<ButtonInput<KeyCode>>,
        Res<ButtonInput<MouseButton>>,
        Res<Time>,
    ),
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), With<OrbitCamera>>,
    sim: Option<Res<SimulationState>>,
    body_states: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    plan: ResMut<ManeuverPlan>,
    mut mode: ResMut<InteractionMode>,
    mut selected: ResMut<SelectedNode>,
    mut node_dv: ResMut<NodeDeltaV>,
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
                snap_dominant_body: None,
            };
        }
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
        if let InteractionMode::PlacingNode { snap_time, snap_world_pos, snap_dominant_body, .. } = &mut *mode {
            *snap_time = None;
            *snap_world_pos = None;
            *snap_dominant_body = None;
        }
        return;
    };
    let Some(ref states) = body_states.states else {
        return;
    };

    match &mut *mode {
        InteractionMode::DraggingArrow { axis, axis_screen_dir, drag_origin, rate_sign, .. } => {
            if mouse.pressed(MouseButton::Left) {
                let screen_delta = cursor_pos - *drag_origin;
                let displacement = screen_delta.dot(*axis_screen_dir);
                let rate = displacement as f64 * 10.0;
                let dt = time.delta_secs_f64();

                *rate_sign = if rate.abs() < 0.01 {
                    0.0
                } else {
                    rate.signum() as f32
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
                    if let Some(closest) =
                        closest_trail_point(prediction, states, &origin, camera, cam_transform, cursor_pos)
                    {
                        writer.write(ManeuverEvent::SlideNode {
                            id: sel_id,
                            new_time: closest.time,
                        });
                    }
                }
                return;
            } else {
                *mode = InteractionMode::Idle;
            }
        }

        InteractionMode::PlacingNode { snap_time, snap_world_pos, snap_dominant_body } => {
            let closest =
                closest_trail_point(prediction, states, &origin, camera, cam_transform, cursor_pos);
            *snap_time = closest.as_ref().map(|p| p.time);
            *snap_world_pos = closest.as_ref().map(|p| p.world_pos);
            *snap_dominant_body = closest.as_ref().map(|p| p.dominant_body);

            if mouse.just_pressed(MouseButton::Left) {
                if let (Some(trail_time), Some(reference_body)) = (*snap_time, *snap_dominant_body) {
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
