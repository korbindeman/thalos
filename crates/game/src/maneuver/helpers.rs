use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics::body_state_provider::BodyStateProvider;
use thalos_physics::maneuver::{delta_v_to_world, orbital_frame};
use thalos_physics::trajectory::{FlightPlan, Trajectory};
use thalos_physics::types::{BodyState, SolarSystemDefinition};

use super::state::{GameNode, ManeuverPlan, NodeId};
use crate::coords::{RenderOrigin, sample_render_pos};
use crate::flight_plan_view::FlightPlanView;

pub(super) struct ClosestTrailPoint {
    pub time: f64,
    pub world_pos: Vec3,
    pub anchor_body: usize,
    pub screen_distance: f32,
}

/// Compute the prograde/normal/radial frame as a Mat3 from ship + body state.
///
/// Columns: [prograde, normal, radial] matching the ManeuverNode delta_v axes.
fn orbital_frame_mat3(ship_pos: DVec3, ship_vel: DVec3, body_pos: DVec3, body_vel: DVec3) -> Mat3 {
    let [prograde, normal, radial] = orbital_frame(ship_pos, ship_vel, body_pos, body_vel);
    Mat3::from_cols(prograde.as_vec3(), normal.as_vec3(), radial.as_vec3())
}

/// Find the render-space position and orbital frame for a given simulation
/// time by searching the prediction for the closest sample.
///
/// The orbital frame is computed relative to the sample's `soi_body` queried
/// at the sample's time — so a node placed inside a moon's SOI gets its
/// prograde/normal/radial axes aligned to that moon, matching the node's
/// `reference_body` and the physics burn frame.
pub(super) fn node_world_pos_and_frame(
    prediction: &FlightPlan,
    time: f64,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    system: &SolarSystemDefinition,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
) -> Option<(Vec3, Mat3)> {
    // Include the baseline (full pre-maneuver orbit) in the search: when the
    // user slides the first node, `closest_trail_point_on_leg` snaps it to a
    // baseline sample, and that time may sit outside the current leg-0 coast
    // until the prediction is rebuilt one frame later.
    let segs = prediction.segments.iter().chain(prediction.baseline.iter());
    for seg in segs {
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

        let _ = system;
        let pin_for = |id| flight_plan_view.pin_for_body(id, body_states);
        let sample = &seg.samples[idx];
        let world_pos = sample_render_pos(sample, pin_for, origin);

        let ref_state = ephemeris.query_body(sample.anchor_body, sample.time);
        let frame = orbital_frame_mat3(
            sample.position,
            sample.velocity,
            ref_state.position,
            ref_state.velocity,
        );

        return Some((world_pos, frame));
    }
    None
}

pub(super) fn node_world_position(
    node: &GameNode,
    prediction: &FlightPlan,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    system: &SolarSystemDefinition,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
) -> Option<Vec3> {
    node_world_pos_and_frame(
        prediction,
        node.time,
        body_states,
        origin,
        system,
        ephemeris,
        flight_plan_view,
    )
    .map(|(pos, _)| pos)
}

pub(super) fn selected_node_world_and_frame(
    selected_id: Option<NodeId>,
    plan: &ManeuverPlan,
    prediction: &FlightPlan,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    system: &SolarSystemDefinition,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
) -> Option<(Vec3, Mat3)> {
    let id = selected_id?;
    let node = plan.nodes.iter().find(|n| n.id == id)?;
    node_world_pos_and_frame(
        prediction,
        node.time,
        body_states,
        origin,
        system,
        ephemeris,
        flight_plan_view,
    )
}

/// Drag-to-Δv gain multiplier derived from the post-burn orbit size.
///
/// Downstream trajectory sensitivity to an impulsive Δv scales with the
/// semi-major axis `a` of the orbit the ship is being placed onto — the
/// same 1 m/s nudge shifts apoapsis by orders of magnitude more once the
/// trajectory has been stretched into an interplanetary transfer.
/// Returning `A_REF / a` lets the caller damp drag input proportionally
/// so dragging the handles feels uniform instead of exploding past the
/// right Δv as the user pulls the trajectory outward.
///
/// `node_delta_v` is the node's current local-frame Δv; applying it to
/// pre-burn velocity is what makes the scale tighten up *during* the
/// drag — using pre-burn state alone would leave a first-node LEO→far-
/// transfer drag unchanged and defeat the whole point.
///
/// Uses current radius as a fallback for hyperbolic/parabolic orbits.
pub(super) fn orbit_sensitivity_scale(
    prediction: &FlightPlan,
    node_time: f64,
    node_delta_v: DVec3,
    reference_body: usize,
    ephemeris: &dyn BodyStateProvider,
    system: &SolarSystemDefinition,
) -> Option<f64> {
    const A_REF: f64 = 1.0e7;
    const MIN_SCALE: f64 = 1.0e-4;

    let ship_state = prediction.state_at(node_time)?;
    let body = system.bodies.get(reference_body)?;
    if body.gm <= 0.0 {
        return None;
    }
    let body_state = ephemeris.query_body(reference_body, node_time);

    let dv_world = delta_v_to_world(
        node_delta_v,
        ship_state.velocity,
        ship_state.position,
        body_state.position,
        body_state.velocity,
    );

    let r_rel = ship_state.position - body_state.position;
    let v_rel = (ship_state.velocity + dv_world) - body_state.velocity;
    let r = r_rel.length();
    if r < 1.0 {
        return None;
    }
    let v_sq = v_rel.length_squared();

    // Vis-viva: 1/a = 2/r - v²/μ. Non-positive means unbound — fall back to r.
    let inv_a = 2.0 / r - v_sq / body.gm;
    let effective_a = if inv_a > 1.0e-20 { 1.0 / inv_a } else { r };

    Some((A_REF / effective_a).clamp(MIN_SCALE, 1.0))
}

pub(super) fn closest_node(
    plan: &ManeuverPlan,
    prediction: &FlightPlan,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    system: &SolarSystemDefinition,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
    camera: &Camera,
    cam_transform: &GlobalTransform,
    cursor_pos: Vec2,
    max_distance: f32,
) -> Option<NodeId> {
    let mut best = None;
    let mut best_dist = max_distance;

    for node in &plan.nodes {
        let Some(world_pos) = node_world_position(
            node,
            prediction,
            body_states,
            origin,
            system,
            ephemeris,
            flight_plan_view,
        ) else {
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

/// Find which leg of the prediction a given time falls within.
fn leg_index_for_time(prediction: &FlightPlan, time: f64) -> usize {
    for (i, leg) in prediction.legs().iter().enumerate() {
        if let Some(end_time) = leg.leg_end_time() {
            if time <= end_time + 1e-6 {
                return i;
            }
        }
    }
    prediction.legs().len().saturating_sub(1)
}

/// Find the closest trail point constrained to the node's own leg and the
/// leg it creates (the next one).
///
/// Leg `i` contains the coast leading *up to* the node; leg `i+1` contains
/// the post-burn coast *after* the node.  Searching both lets the node slide
/// backward along the pre-maneuver trajectory and forward along the
/// post-maneuver trajectory.  Only coast sub-segments are searched so the
/// node cannot land inside another maneuver's burn phase.
///
/// When either leg has a stable orbit, coast samples form a closed loop on
/// screen, so the screen-space search wraps around the revolution naturally.
pub(super) fn closest_trail_point_on_leg(
    prediction: &FlightPlan,
    node_time: f64,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    system: &SolarSystemDefinition,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
    camera: &Camera,
    cam_transform: &GlobalTransform,
    cursor_pos: Vec2,
) -> Option<ClosestTrailPoint> {
    let leg_idx = leg_index_for_time(prediction, node_time);
    let legs = prediction.legs();

    // Collect coast segments to search.
    let mut coasts: Vec<&thalos_physics::trajectory::NumericSegment> = Vec::new();

    if leg_idx == 0 {
        if let Some(baseline) = &prediction.baseline {
            // Baseline covers the full original orbit — use it instead of
            // the truncated leg 0 coast + divergent leg 1 coast.
            coasts.push(baseline);
        } else {
            // No maneuvers → leg 0 IS the full trajectory.
            if let Some(leg) = legs.first() {
                coasts.push(&leg.coast_segment);
            }
        }
    } else {
        // Later legs: search the node's own leg and the next.
        for &idx in &[leg_idx, leg_idx + 1] {
            if let Some(leg) = legs.get(idx) {
                coasts.push(&leg.coast_segment);
            }
        }
    }

    let mut best: Option<ClosestTrailPoint> = None;

    let _ = (system, ephemeris);
    let pin_for = |id| flight_plan_view.pin_for_body(id, body_states);
    for coast in coasts {
        if coast.samples.is_empty() {
            continue;
        }
        for sample in coast.samples.iter() {
            let world_pos = sample_render_pos(sample, &pin_for, origin);
            let Some(screen_pos) = camera.world_to_viewport(cam_transform, world_pos).ok()
            else {
                continue;
            };
            let d = (screen_pos - cursor_pos).length();
            let is_better = best.as_ref().map_or(true, |b| d < b.screen_distance);
            if is_better {
                best = Some(ClosestTrailPoint {
                    time: sample.time,
                    world_pos,
                    anchor_body: sample.anchor_body,
                    screen_distance: d,
                });
            }
        }
    }

    best
}

pub(super) fn closest_trail_point(
    prediction: &FlightPlan,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    system: &SolarSystemDefinition,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
    camera: &Camera,
    cam_transform: &GlobalTransform,
    cursor_pos: Vec2,
) -> Option<ClosestTrailPoint> {
    let mut best: Option<ClosestTrailPoint> = None;

    let _ = (system, ephemeris);
    let pin_for = |id| flight_plan_view.pin_for_body(id, body_states);
    for seg in prediction.segments.iter() {
        for sample in seg.samples.iter() {
            let world_pos = sample_render_pos(sample, &pin_for, origin);
            let Some(screen_pos) = camera.world_to_viewport(cam_transform, world_pos).ok() else {
                continue;
            };
            let d = (screen_pos - cursor_pos).length();
            let is_better = best.as_ref().map_or(true, |b| d < b.screen_distance);
            if is_better {
                best = Some(ClosestTrailPoint {
                    time: sample.time,
                    world_pos,
                    anchor_body: sample.anchor_body,
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
