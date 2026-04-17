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
pub(in crate::maneuver) fn handle_maneuver_events(
    mut events: bevy::ecs::message::MessageReader<ManeuverEvent>,
    mut plan: ResMut<ManeuverPlan>,
) {
    for event in events.read() {
        match event.clone() {
            ManeuverEvent::PlaceNode {
                trail_time,
                reference_body,
            } => {
                let id = plan.next_node_id();
                plan.nodes.push(GameNode {
                    id,
                    time: trail_time,
                    delta_v: DVec3::ZERO,
                    reference_body,
                });
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
                    plan.dirty = true;
                }
            }
            ManeuverEvent::DeleteNode { id } => {
                let before = plan.nodes.len();
                plan.nodes.retain(|n| n.id != id);
                if plan.nodes.len() != before {
                    info!("[maneuver] DeleteNode consumed id={:?}", id);
                    plan.dirty = true;
                }
            }
        }
    }
}
