use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics::trajectory::TrajectoryPrediction;
use thalos_physics::types::BodyState;

use super::state::{GameNode, ManeuverPlan, NodeId};
use crate::coords::{RenderOrigin, sample_render_pos};

pub(super) struct ClosestTrailPoint {
    pub time: f64,
    pub world_pos: Vec3,
    pub dominant_body: usize,
    pub screen_distance: f32,
}

/// Compute the prograde/normal/radial frame as a Mat3 from ship + body state.
///
/// Columns: [prograde, normal, radial] matching the ManeuverNode delta_v axes.
fn orbital_frame_mat3(
    ship_pos: DVec3,
    ship_vel: DVec3,
    body_pos: DVec3,
    body_vel: DVec3,
) -> Mat3 {
    let rel_vel = ship_vel - body_vel;
    let rel_pos = ship_pos - body_pos;

    let prograde = if rel_vel.length_squared() > 1e-20 {
        rel_vel.normalize()
    } else {
        DVec3::X
    };

    let radial_raw = if rel_pos.length_squared() > 1e-20 {
        rel_pos.normalize()
    } else {
        DVec3::Y
    };

    let normal_raw = radial_raw.cross(prograde);
    let normal = if normal_raw.length_squared() > 1e-20 {
        normal_raw.normalize()
    } else {
        let arb = if prograde.dot(DVec3::Y).abs() < 0.9 {
            DVec3::Y
        } else {
            DVec3::X
        };
        prograde.cross(arb).normalize()
    };

    let radial = prograde.cross(normal);

    Mat3::from_cols(
        prograde.as_vec3(),
        normal.as_vec3(),
        radial.as_vec3(),
    )
}

/// Find the render-space position and orbital frame for a given simulation time
/// by searching the prediction for the closest sample.
pub(super) fn node_world_pos_and_frame(
    prediction: &TrajectoryPrediction,
    time: f64,
    body_states: &[BodyState],
    origin: &RenderOrigin,
) -> Option<(Vec3, Mat3)> {
    for seg in &prediction.segments {
        if seg.samples.is_empty() {
            continue;
        }
        let first = seg.samples[0].time;
        let last = seg.samples.last().unwrap().time;
        if time < first || time > last {
            continue;
        }

        let idx = seg
            .samples
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (a.time - time)
                    .abs()
                    .partial_cmp(&(b.time - time).abs())
                    .unwrap()
            })
            .map(|(i, _)| i)?;

        let sample = &seg.samples[idx];
        let world_pos = sample_render_pos(sample, body_states, origin);

        let body_vel = body_states
            .get(sample.dominant_body)
            .map(|bs| bs.velocity)
            .unwrap_or(DVec3::ZERO);

        let frame = orbital_frame_mat3(
            sample.position,
            sample.velocity,
            sample.dominant_body_pos,
            body_vel,
        );

        return Some((world_pos, frame));
    }
    None
}

pub(super) fn node_world_position(
    node: &GameNode,
    prediction: &TrajectoryPrediction,
    body_states: &[BodyState],
    origin: &RenderOrigin,
) -> Option<Vec3> {
    node_world_pos_and_frame(prediction, node.time, body_states, origin).map(|(pos, _)| pos)
}

pub(super) fn selected_node_world_and_frame(
    selected_id: Option<NodeId>,
    plan: &ManeuverPlan,
    prediction: &TrajectoryPrediction,
    body_states: &[BodyState],
    origin: &RenderOrigin,
) -> Option<(Vec3, Mat3)> {
    let id = selected_id?;
    let node = plan.nodes.iter().find(|n| n.id == id)?;
    node_world_pos_and_frame(prediction, node.time, body_states, origin)
}

pub(super) fn closest_node(
    plan: &ManeuverPlan,
    prediction: &TrajectoryPrediction,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    camera: &Camera,
    cam_transform: &GlobalTransform,
    cursor_pos: Vec2,
    max_distance: f32,
) -> Option<NodeId> {
    let mut best = None;
    let mut best_dist = max_distance;

    for node in &plan.nodes {
        let Some(world_pos) = node_world_position(node, prediction, body_states, origin) else {
            continue;
        };
        let Some(screen_pos) = camera.world_to_viewport(cam_transform, world_pos).ok() else {
            continue;
        };
        let d = (screen_pos - cursor_pos).length();
        if d < best_dist {
            best_dist = d;
            best = Some(node.id);
        }
    }

    best
}

pub(super) fn closest_trail_point(
    prediction: &TrajectoryPrediction,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    camera: &Camera,
    cam_transform: &GlobalTransform,
    cursor_pos: Vec2,
) -> Option<ClosestTrailPoint> {
    let mut best: Option<ClosestTrailPoint> = None;

    for seg in &prediction.segments {
        for sample in &seg.samples {
            let world_pos = sample_render_pos(sample, body_states, origin);
            let Some(screen_pos) = camera.world_to_viewport(cam_transform, world_pos).ok() else {
                continue;
            };
            let d = (screen_pos - cursor_pos).length();
            let is_better = best.as_ref().map_or(true, |b| d < b.screen_distance);
            if is_better {
                best = Some(ClosestTrailPoint {
                    time: sample.time,
                    world_pos,
                    dominant_body: sample.dominant_body,
                    screen_distance: d,
                });
            }
        }
    }

    best
}

pub(super) fn overlay_marker_transform(
    world_pos: Vec3,
    camera_rotation: Quat,
    marker_scale: f32,
) -> Transform {
    Transform {
        translation: world_pos,
        rotation: camera_rotation,
        scale: Vec3::splat(marker_scale),
    }
}
