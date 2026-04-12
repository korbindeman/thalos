use bevy::picking::prelude::{DragEnd, DragStart, Pointer};
use bevy::prelude::*;

use super::super::state::{
    ArrowHandle, ArrowHitbox, InteractionMode, BASE_ARROW_LEN, HITBOX_CAPSULE_RADIUS,
    NodeSlideSphere, SLIDE_SPHERE_RADIUS, SelectedNodeView,
};
use crate::camera::{CameraFocus, OrbitCamera};
use crate::coords::RENDER_SCALE;

/// Handle drag start on an arrow hitbox — record which axis and compute the
/// screen-space direction for projecting mouse delta to delta-v.
pub(in crate::maneuver) fn arrow_drag_start(
    trigger: On<Pointer<DragStart>>,
    handles: Query<&ArrowHandle, With<ArrowHitbox>>,
    mut mode: ResMut<InteractionMode>,
    selected_view: Res<SelectedNodeView>,
    focus: Res<CameraFocus>,
    camera_q: Query<(&Camera, &GlobalTransform), With<OrbitCamera>>,
) {
    let event = trigger.event();
    let entity = event.entity;
    let Ok(handle) = handles.get(entity) else {
        return;
    };
    let Ok((camera, cam_transform)) = camera_q.single() else {
        return;
    };
    let Some(world_pos) = selected_view.world_pos else {
        return;
    };
    let Some(frame) = selected_view.frame else {
        return;
    };

    let s = (focus.distance * RENDER_SCALE) as f32;
    let arrow_len = BASE_ARROW_LEN * s;
    let sphere_gap = (SLIDE_SPHERE_RADIUS + HITBOX_CAPSULE_RADIUS) * s;
    let dir = frame.col(handle.axis).normalize();
    let positive_tip = world_pos + dir * (sphere_gap + arrow_len);

    let Some(node_screen) = camera.world_to_viewport(cam_transform, world_pos).ok() else {
        return;
    };
    let Some(tip_screen) = camera.world_to_viewport(cam_transform, positive_tip).ok() else {
        return;
    };

    *mode = InteractionMode::DraggingArrow {
        axis: handle.axis,
        positive: handle.positive,
        axis_screen_dir: (tip_screen - node_screen).normalize_or_zero(),
        drag_origin: event.pointer_location.position,
        rate_sign: 0.0,
    };
}

/// Clear drag state when arrow drag ends.
pub(in crate::maneuver) fn arrow_drag_end(
    trigger: On<Pointer<DragEnd>>,
    handles: Query<&ArrowHandle, With<ArrowHitbox>>,
    mut mode: ResMut<InteractionMode>,
) {
    if handles.get(trigger.event().entity).is_ok() {
        if matches!(*mode, InteractionMode::DraggingArrow { .. }) {
            *mode = InteractionMode::Idle;
        }
    }
}

/// Start sliding the node along the trajectory when dragging the center sphere.
pub(in crate::maneuver) fn slide_sphere_drag_start(
    trigger: On<Pointer<DragStart>>,
    spheres: Query<(), With<NodeSlideSphere>>,
    mut mode: ResMut<InteractionMode>,
) {
    if spheres.get(trigger.event().entity).is_ok() {
        *mode = InteractionMode::SlidingNode;
    }
}

/// Stop sliding when the drag ends.
pub(in crate::maneuver) fn slide_sphere_drag_end(
    trigger: On<Pointer<DragEnd>>,
    spheres: Query<(), With<NodeSlideSphere>>,
    mut mode: ResMut<InteractionMode>,
) {
    if spheres.get(trigger.event().entity).is_ok() {
        if matches!(*mode, InteractionMode::SlidingNode) {
            *mode = InteractionMode::Idle;
        }
    }
}
