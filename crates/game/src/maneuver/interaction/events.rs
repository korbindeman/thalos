use bevy::math::DVec3;
use bevy::prelude::*;

use super::super::state::{GameNode, ManeuverEvent, ManeuverPlan, NodeDeltaV, SelectedNode};

/// Sync NodeDeltaV when selection changes.
pub(in crate::maneuver) fn sync_node_delta_v(
    selected: Res<SelectedNode>,
    plan: Res<ManeuverPlan>,
    mut node_dv: ResMut<NodeDeltaV>,
) {
    if !selected.is_changed() {
        return;
    }
    if let Some(id) = selected.id {
        if let Some(node) = plan.nodes.iter().find(|n| n.id == id) {
            node_dv.prograde = node.delta_v.x;
            node_dv.normal = node.delta_v.y;
            node_dv.radial = node.delta_v.z;
        }
    } else {
        *node_dv = NodeDeltaV::default();
    }
}

/// Handle maneuver events: place, adjust, slide, delete.
///
/// `SlideNode` events fire every frame during a drag (one per cursor sample),
/// and each `dirty = true` flip drives a full flight-plan reprop in
/// [`crate::bridge::update_prediction`]. On non-trivial plans that puts the
/// rebuild on the critical path at 60 Hz and sliding feels laggy.
///
/// We always update `node.time` (so the visual marker tracks the cursor every
/// frame) but only flip `plan.dirty` at most every [`SLIDE_REBUILD_THROTTLE_S`]
/// seconds during a drag. The drag-end observer
/// ([`super::observers::slide_sphere_drag_end`]) forces a final
/// `dirty = true` so the trajectory always reflects the released position.
pub(in crate::maneuver) fn handle_maneuver_events(
    mut events: bevy::ecs::message::MessageReader<ManeuverEvent>,
    mut plan: ResMut<ManeuverPlan>,
    mut selected: ResMut<SelectedNode>,
    time: Res<Time>,
) {
    for event in events.read() {
        match event.clone() {
            ManeuverEvent::PlaceNode {
                trail_time,
                reference_body,
                rail,
            } => {
                let id = plan.next_node_id();
                plan.nodes.push(GameNode {
                    id,
                    time: trail_time,
                    delta_v: DVec3::ZERO,
                    reference_body,
                    rail,
                });
                selected.id = Some(id);
                plan.dirty = true;
            }
            ManeuverEvent::AdjustNode { id, delta_v } => {
                if let Some(node) = plan.nodes.iter_mut().find(|n| n.id == id) {
                    node.delta_v = delta_v;
                    plan.dirty = true;
                }
            }
            ManeuverEvent::SlideNode { id, new_time } => {
                if !new_time.is_finite() {
                    warn!("[maneuver] ignoring SlideNode with non-finite time {new_time}");
                    continue;
                }
                if let Some(node) = plan.nodes.iter_mut().find(|n| n.id == id) {
                    node.time = new_time;
                    let now = time.elapsed_secs_f64();
                    if now - plan.last_slide_apply_secs >= SLIDE_REBUILD_THROTTLE_S {
                        plan.dirty = true;
                        plan.last_slide_apply_secs = now;
                    }
                }
            }
            ManeuverEvent::DeleteNode { id } => {
                let Some(delete_time) = plan.nodes.iter().find(|n| n.id == id).map(|n| n.time)
                else {
                    continue;
                };
                let before = plan.nodes.len();
                plan.nodes
                    .retain(|n| n.time < delete_time - NODE_TIME_EPSILON_S);
                if plan.nodes.len() != before {
                    plan.dirty = true;
                }
            }
        }
    }
}

/// Minimum interval between slide-driven flight-plan rebuilds (seconds).
///
/// 100 ms (10 Hz) keeps the live trajectory preview responsive without
/// stalling the frame budget on plans where a single reprop costs tens of
/// milliseconds.
const SLIDE_REBUILD_THROTTLE_S: f64 = 0.1;
const NODE_TIME_EPSILON_S: f64 = 1e-6;
