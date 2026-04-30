use bevy::picking::prelude::{DragEnd, DragStart, Pointer};
use bevy::prelude::*;

use super::super::state::{
    ArrowHandle, ArrowHitbox, BASE_ARROW_LEN, HITBOX_CAPSULE_RADIUS, InteractionMode, ManeuverPlan,
    NodeSlideSphere, SLIDE_SPHERE_RADIUS, SelectedNodeView, SlidePreview,
};
use crate::camera::{ActiveCamera, CameraFocus};
use crate::coords::WorldScale;

/// Handle drag start on an arrow hitbox — record which axis and compute the
/// screen-space direction for projecting mouse delta to delta-v.
pub(in crate::maneuver) fn arrow_drag_start(
    trigger: On<Pointer<DragStart>>,
    handles: Query<&ArrowHandle, With<ArrowHitbox>>,
    mut mode: ResMut<InteractionMode>,
    selected_view: Res<SelectedNodeView>,
    focus: Res<CameraFocus>,
    scale: Res<WorldScale>,
    camera_q: Query<(&Camera, &GlobalTransform), With<ActiveCamera>>,
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

    let s = (focus.distance * scale.0) as f32;
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
///
/// Force `plan.dirty = true` on release so the trajectory rebuilds against the
/// final node time even if `handle_maneuver_events` throttled the most recent
/// slide samples (see `SLIDE_REBUILD_THROTTLE_S`). Clear the [`SlidePreview`]
/// so the next frame's `update_selected_node_view` falls back to sampling the
/// freshly-rebuilt prediction.
pub(in crate::maneuver) fn slide_sphere_drag_end(
    trigger: On<Pointer<DragEnd>>,
    spheres: Query<(), With<NodeSlideSphere>>,
    mut mode: ResMut<InteractionMode>,
    mut plan: ResMut<ManeuverPlan>,
    mut slide_preview: ResMut<SlidePreview>,
) {
    if spheres.get(trigger.event().entity).is_ok() {
        if matches!(*mode, InteractionMode::SlidingNode) {
            *mode = InteractionMode::Idle;
            plan.dirty = true;
            slide_preview.world_pos = None;
            slide_preview.frame = None;
        }
    }
}
