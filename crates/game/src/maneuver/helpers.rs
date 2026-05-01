use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics::body_state_provider::BodyStateProvider;
use thalos_physics::maneuver::{delta_v_to_world, orbital_frame};
use thalos_physics::trajectory::{FlightPlan, NumericSegment, Trajectory};
use thalos_physics::types::{BodyId, BodyState, SolarSystemDefinition, TrajectorySample};

use super::state::{GameNode, ManeuverPlan, NodeId, RailFrame, TrajectoryRail};
use crate::coords::{RenderOrigin, WorldScale, sample_render_pos};
use crate::flight_plan_view::FlightPlanView;

pub(super) struct ClosestTrailPoint {
    pub time: f64,
    pub world_pos: Vec3,
    pub anchor_body: usize,
    pub screen_distance: f32,
    /// Ship state at this sample, in `BodyStateProvider` (heliocentric)
    /// coordinates — i.e. the sample's stored `position` / `velocity`.
    /// Carried out of the search so the slide handler can compute the
    /// orbital frame for the slide preview without re-sampling the
    /// prediction (which may be stale during throttled slides).
    pub sample_position: DVec3,
    pub sample_velocity: DVec3,
    pub rail: Option<TrajectoryRail>,
}

/// Compute the prograde/normal/radial frame as a Mat3 from ship + body state.
///
/// Columns: [prograde, normal, radial] matching the ManeuverNode delta_v axes.
pub(super) fn orbital_frame_mat3(
    ship_pos: DVec3,
    ship_vel: DVec3,
    body_pos: DVec3,
    body_vel: DVec3,
) -> Mat3 {
    let [prograde, normal, radial] = orbital_frame(ship_pos, ship_vel, body_pos, body_vel);
    Mat3::from_cols(prograde.as_vec3(), normal.as_vec3(), radial.as_vec3())
}

/// Find the render-space position and orbital frame for a maneuver node at
/// `time`.
///
/// Anchored to the *pre-burn* trajectory: burns are centered on `node.time`
/// (`[t − d/2, t + d/2]`), so sampling the burn segment at `t` returns a
/// partially-thrusted state that drifts off the unperturbed orbit as Δv
/// grows. `FlightPlan::pre_burn_state_at` does the right thing — it samples
/// the pre-burn coast directly when it covers `t`, and Kepler-extrapolates
/// across the burn window otherwise. The orbital frame is computed from
/// pre-burn velocity for the same reason: prograde/normal/radial must stay
/// rigid while the user drags a handle.
///
/// Falls back to the baseline (full pre-maneuver orbit) for the slide-snap
/// latency case: `closest_trail_point_on_leg` may snap a slide to a baseline
/// sample whose time sits outside the current legs until the prediction is
/// rebuilt one frame later.
pub(super) fn node_world_pos_and_frame(
    node: &GameNode,
    prediction: &FlightPlan,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    scale: &WorldScale,
    system: &SolarSystemDefinition,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
) -> Option<(Vec3, Mat3)> {
    if let Some(rail) = &node.rail {
        return rail_world_pos_and_frame(
            rail,
            node.time,
            body_states,
            origin,
            scale,
            ephemeris,
            flight_plan_view,
        );
    }

    let time = node.time;
    let reference_body = node.reference_body;
    if !node_visible_at_time(prediction, system, flight_plan_view, reference_body, time) {
        return None;
    }

    let sample = prediction
        .pre_burn_state_at(time, ephemeris, &system.bodies)
        .or_else(|| baseline_sample_at(prediction, time, ephemeris))?;

    let ref_state = ephemeris.query_body(reference_body, sample.time);
    let render_sample = TrajectorySample {
        anchor_body: reference_body,
        ref_pos: ref_state.position,
        ..sample
    };
    let pin = flight_plan_view.pin_for_body(reference_body, sample.time, body_states);
    let world_pos = sample_render_pos(&render_sample, pin, origin, scale);

    let frame = orbital_frame_mat3(
        sample.position,
        sample.velocity,
        ref_state.position,
        ref_state.velocity,
    );

    Some((world_pos, frame))
}

fn node_visible_at_time(
    prediction: &FlightPlan,
    system: &SolarSystemDefinition,
    flight_plan_view: &FlightPlanView,
    reference_body: usize,
    time: f64,
) -> bool {
    if let Some(focus_ghost) = flight_plan_view.focused_ghost() {
        return focus_ghost.body_id == reference_body
            && flight_plan_view.focused_ghost_contains_epoch(time);
    }

    !flight_plan_view.epoch_hidden_in_focus(prediction, system, time)
}

/// Build a sample on the baseline (unperturbed) orbit at `time`.
fn baseline_sample_at(
    prediction: &FlightPlan,
    time: f64,
    ephemeris: &dyn BodyStateProvider,
) -> Option<thalos_physics::types::TrajectorySample> {
    let baseline = prediction.baseline.as_ref()?;
    let first = baseline.samples.first()?;
    let last = baseline.samples.last()?;
    if time < first.time - 1e-6 || time > last.time + 1e-6 {
        return None;
    }
    let state = baseline.state_at(time)?;
    let anchor = last.anchor_body;
    let body_state = ephemeris.query_body(anchor, time);
    Some(thalos_physics::types::TrajectorySample {
        time,
        position: state.position,
        velocity: state.velocity,
        anchor_body: anchor,
        ref_pos: body_state.position,
    })
}

fn rail_world_pos_and_frame(
    rail: &TrajectoryRail,
    time: f64,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    scale: &WorldScale,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
) -> Option<(Vec3, Mat3)> {
    let state = rail.state_at(time)?;
    let world_pos = rail_world_position_for_state(
        rail,
        time,
        state.position,
        body_states,
        origin,
        scale,
        ephemeris,
        flight_plan_view,
    )?;
    let reference = ephemeris.query_body(rail.reference_body, time);
    let frame = orbital_frame_mat3(
        state.position,
        state.velocity,
        reference.position,
        reference.velocity,
    );
    Some((world_pos, frame))
}

fn rail_visible(rail: &TrajectoryRail, flight_plan_view: &FlightPlanView) -> bool {
    match rail.frame {
        RailFrame::Body { .. } => flight_plan_view.focused_ghost().is_none(),
        RailFrame::Ghost { body_id, .. } => flight_plan_view
            .focused_ghost()
            .is_some_and(|focus| focus.body_id == body_id),
    }
}

fn rail_world_position_for_state(
    rail: &TrajectoryRail,
    time: f64,
    position: DVec3,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    scale: &WorldScale,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
) -> Option<Vec3> {
    if !rail_visible(rail, flight_plan_view) {
        return None;
    }

    match rail.frame {
        RailFrame::Body { body_id } => {
            let body_state = ephemeris.query_body(body_id, time);
            let sample = TrajectorySample {
                time,
                position,
                velocity: DVec3::ZERO,
                anchor_body: body_id,
                ref_pos: body_state.position,
            };
            let pin = flight_plan_view.pin_for_body(body_id, time, body_states);
            Some(sample_render_pos(&sample, pin, origin, scale))
        }
        RailFrame::Ghost {
            body_id,
            parent_id,
            relative_position,
            projection_epoch,
            encounter_epoch,
            soi_radius,
        } => {
            let pin = flight_plan_view
                .focused_ghost()
                .filter(|focus| focus.body_id == body_id)
                .map(|focus| flight_plan_view.pin_for_ghost_focus(focus, body_states))
                .unwrap_or_else(|| {
                    flight_plan_view
                        .ghosts()
                        .iter()
                        .find(|ghost| {
                            ghost.body_id == body_id
                                && (ghost.encounter_epoch - encounter_epoch).abs()
                                    <= crate::coords::GHOST_FOCUS_EPOCH_TOLERANCE_S
                        })
                        .or_else(|| {
                            flight_plan_view
                                .focused_ghost_ref()
                                .filter(|ghost| ghost.body_id == body_id)
                        })
                        .or_else(|| {
                            flight_plan_view
                                .ghosts()
                                .iter()
                                .filter(|ghost| ghost.body_id == body_id)
                                .min_by(|a, b| {
                                    let da = (a.encounter_epoch - encounter_epoch).abs();
                                    let db = (b.encounter_epoch - encounter_epoch).abs();
                                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                                })
                        })
                        .map(|ghost| flight_plan_view.pin_for_ghost(ghost, body_states))
                        .unwrap_or_else(|| {
                            let parent_pin = flight_plan_view.pin_for_body(
                                parent_id,
                                projection_epoch,
                                body_states,
                            );
                            parent_pin + relative_position
                        })
                });
            Some(ghost_local_world_pos_at_pin(
                time, position, body_id, pin, soi_radius, origin, scale, ephemeris,
            ))
        }
    }
}

pub(super) fn trajectory_rail_points(
    rail: &TrajectoryRail,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    scale: &WorldScale,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
) -> Vec<Vec3> {
    if !rail_visible(rail, flight_plan_view) {
        return Vec::new();
    }

    rail_candidate_times(rail)
        .into_iter()
        .filter_map(|time| {
            let state = rail.state_at(time)?;
            rail_world_position_for_state(
                rail,
                time,
                state.position,
                body_states,
                origin,
                scale,
                ephemeris,
                flight_plan_view,
            )
        })
        .collect()
}

pub(super) fn node_world_position(
    node: &GameNode,
    prediction: &FlightPlan,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    scale: &WorldScale,
    system: &SolarSystemDefinition,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
) -> Option<Vec3> {
    node_world_pos_and_frame(
        node,
        prediction,
        body_states,
        origin,
        scale,
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
    scale: &WorldScale,
    system: &SolarSystemDefinition,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
) -> Option<(Vec3, Mat3)> {
    let id = selected_id?;
    let node = plan.nodes.iter().find(|n| n.id == id)?;
    node_world_pos_and_frame(
        node,
        prediction,
        body_states,
        origin,
        scale,
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

    // Pre-burn state — sampling the prediction directly at `node_time`
    // would land in the (centered) burn segment for finite burns and add
    // the full local Δv on top of an already partially-thrusted velocity,
    // so the post-burn `effective_a` would be wrong.
    let ship_sample = prediction.pre_burn_state_at(node_time, ephemeris, &system.bodies)?;
    let body = system.bodies.get(reference_body)?;
    if body.gm <= 0.0 {
        return None;
    }
    let body_state = ephemeris.query_body(reference_body, node_time);

    let dv_world = delta_v_to_world(
        node_delta_v,
        ship_sample.velocity,
        ship_sample.position,
        body_state.position,
        body_state.velocity,
    );

    let r_rel = ship_sample.position - body_state.position;
    let v_rel = (ship_sample.velocity + dv_world) - body_state.velocity;
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
    scale: &WorldScale,
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
            scale,
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

/// Pick the coast segment(s) that represent the orbit a node is currently
/// sliding along.
///
/// A node sits on the *pre-burn* orbit — the orbit it would still be on if
/// its burn never fired. Constraining the slide to that single orbit keeps
/// the node from wandering across maneuver boundaries onto a different
/// trajectory.
///
/// - **First node by time:** the pre-burn orbit is the unperturbed coast
///   from `sim_time`. Use [`FlightPlan::baseline`] — propagated to one full
///   revolution, so the slide closes back at the ship and wraps cleanly.
///   `legs[0].coast_segment` is the wrong choice here: it is only the
///   segment of the same orbit truncated at `node[0].time − d/2`, so a
///   finite-burn node on the first slot would have *no* baseline available
///   to wrap around.
/// - **Subsequent nodes:** sliding node `M` (chronologically the M-th, 0-
///   indexed) walks along `legs[M].coast_segment` — the coast that the
///   previous burn deposits the ship onto, leading up to this node's burn.
///   We do not include the post-burn leg `legs[M+1]`, which lives on a
///   different orbit; sliding into it would let the user hop across
///   trajectories mid-drag.
pub(super) fn slide_search_segments<'a>(
    plan: &ManeuverPlan,
    prediction: &'a FlightPlan,
    slide_id: NodeId,
) -> Vec<&'a NumericSegment> {
    // Sort nodes by time to find the slide_id's chronological index.
    let mut sorted: Vec<&GameNode> = plan.nodes.iter().collect();
    sorted.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some(idx) = sorted.iter().position(|n| n.id == slide_id) else {
        return Vec::new();
    };

    if idx == 0 {
        // First node: prefer baseline (full revolution from sim_time).
        if let Some(baseline) = &prediction.baseline {
            return vec![baseline];
        }
        // Baseline is only built when maneuvers exist — but we *do* have a
        // maneuver here, so this branch is unreachable in normal operation.
        // Fall through to leg 0's coast as a defensive fallback.
        if let Some(leg) = prediction.legs().first() {
            return vec![&leg.coast_segment];
        }
        return Vec::new();
    }

    if let Some(leg) = prediction.legs().get(idx) {
        return vec![&leg.coast_segment];
    }
    Vec::new()
}

fn body_rail_from_segment(segment: &NumericSegment) -> Option<TrajectoryRail> {
    let first = segment.samples.first()?;
    Some(TrajectoryRail {
        frame: RailFrame::Body {
            body_id: first.anchor_body,
        },
        reference_body: first.anchor_body,
        samples: segment.samples.clone(),
    })
}

fn rail_from_segment_interval(
    segment: &NumericSegment,
    frame: RailFrame,
    reference_body: BodyId,
    start: f64,
    end: f64,
    ephemeris: &dyn BodyStateProvider,
) -> Option<TrajectoryRail> {
    let (segment_start, segment_end) = segment.epoch_range();
    let start = start.max(segment_start);
    let end = end.min(segment_end);
    if end <= start {
        return None;
    }

    let mut times = Vec::with_capacity(segment.samples.len() + 2);
    times.push(start);
    times.extend(
        segment
            .samples
            .iter()
            .map(|sample| sample.time)
            .filter(|time| *time > start && *time < end),
    );
    times.push(end);
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    let mut samples = Vec::with_capacity(times.len());
    for time in times {
        let Some(state) = segment.state_at(time) else {
            continue;
        };
        let reference = ephemeris.query_body(reference_body, time);
        samples.push(TrajectorySample {
            time,
            position: state.position,
            velocity: state.velocity,
            anchor_body: reference_body,
            ref_pos: reference.position,
        });
    }

    if samples.len() < 2 {
        return None;
    }

    Some(TrajectoryRail {
        frame,
        reference_body,
        samples,
    })
}

fn push_body_rail(runs: &mut Vec<TrajectoryRail>, samples: &mut Vec<TrajectorySample>) {
    if samples.len() < 2 {
        samples.clear();
        return;
    }

    let body_id = samples[0].anchor_body;
    runs.push(TrajectoryRail {
        frame: RailFrame::Body { body_id },
        reference_body: body_id,
        samples: std::mem::take(samples),
    });
}

fn visible_trajectory_rails(
    prediction: &FlightPlan,
    system: &SolarSystemDefinition,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
) -> Vec<TrajectoryRail> {
    if flight_plan_view.focused_ghost().is_some() {
        return focused_ghost_rails(prediction, system, ephemeris, flight_plan_view);
    }

    let mut rails = Vec::new();
    for segment in prediction.segments.iter() {
        let Some(first) = segment.samples.first() else {
            continue;
        };
        let mut run = Vec::new();
        for sample in &segment.samples {
            let same_anchor = sample.anchor_body == first.anchor_body;
            let visible = same_anchor
                && !flight_plan_view.epoch_hidden_in_focus(prediction, system, sample.time);
            if visible {
                run.push(*sample);
            } else {
                push_body_rail(&mut rails, &mut run);
            }
        }
        push_body_rail(&mut rails, &mut run);
    }
    rails
}

fn focused_ghost_rails(
    prediction: &FlightPlan,
    system: &SolarSystemDefinition,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
) -> Vec<TrajectoryRail> {
    let Some(focus) = flight_plan_view.focused_ghost() else {
        return Vec::new();
    };
    let Some(ghost) = flight_plan_view.focused_ghost_ref().or_else(|| {
        flight_plan_view
            .ghosts()
            .iter()
            .filter(|ghost| ghost.body_id == focus.body_id)
            .min_by(|a, b| {
                let da = (a.encounter_epoch - focus.encounter_epoch).abs();
                let db = (b.encounter_epoch - focus.encounter_epoch).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
    }) else {
        return Vec::new();
    };
    let Some(window) = ghost.trajectory_window else {
        return Vec::new();
    };
    let end = window.exit_epoch.unwrap_or(window.end_epoch);
    if end <= window.start_epoch {
        return Vec::new();
    }

    let soi_radius = system
        .bodies
        .get(ghost.body_id)
        .map(|body| body.soi_radius_m)
        .unwrap_or(f64::INFINITY);
    let frame = RailFrame::Ghost {
        body_id: ghost.body_id,
        parent_id: focus.parent_id,
        relative_position: focus.relative_position,
        projection_epoch: focus.projection_epoch,
        encounter_epoch: focus.encounter_epoch,
        soi_radius,
    };

    let mut rails = Vec::new();
    for leg in prediction.legs() {
        if let Some(burn) = &leg.burn_segment
            && let Some(rail) = rail_from_segment_interval(
                burn,
                frame.clone(),
                ghost.body_id,
                window.start_epoch,
                end,
                ephemeris,
            )
        {
            rails.push(rail);
        }
        if let Some(rail) = rail_from_segment_interval(
            &leg.coast_segment,
            frame.clone(),
            ghost.body_id,
            window.start_epoch,
            end,
            ephemeris,
        ) {
            rails.push(rail);
        }
    }
    rails
}

fn rail_candidate_times(rail: &TrajectoryRail) -> Vec<f64> {
    let Some((start, end)) = rail.epoch_range() else {
        return Vec::new();
    };
    if end <= start {
        return Vec::new();
    }

    let subdivisions = rail.samples.len().clamp(96, 512);
    let mut times = Vec::with_capacity(rail.samples.len() + subdivisions + 1);
    times.extend(rail.samples.iter().map(|sample| sample.time));
    for i in 0..=subdivisions {
        times.push(start + (end - start) * (i as f64 / subdivisions as f64));
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    times
}

/// Find the closest trail point along a single orbit.
///
/// The search runs in screen space across all samples in `coasts`. A pure
/// argmin would flicker at the seam where a closed orbit's first and last
/// samples project to almost the same pixel: tiny cursor jitter would snap
/// the node a full period away. To suppress that, when the two screen-
/// closest samples are within ~8 px of each other AND their times are
/// further apart than half the searched span (which only happens at a
/// closed-orbit's wrap, not for adjacent samples), we tiebreak by closeness
/// to `prev_time`. The user crossing the seam intentionally still wraps —
/// once they've dragged past the closure point one candidate becomes
/// clearly closer (>8 px gap) and wins regardless of time.
pub(super) fn closest_trail_point_on_orbit(
    coasts: &[&NumericSegment],
    prediction: &FlightPlan,
    prev_time: f64,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    scale: &WorldScale,
    system: &SolarSystemDefinition,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
    camera: &Camera,
    cam_transform: &GlobalTransform,
    cursor_pos: Vec2,
) -> Option<ClosestTrailPoint> {
    // Total time covered by the searched coasts. For a closed-orbit baseline
    // this is one orbital period; for a partial coast it's that arc's
    // duration. Half of it is the seam threshold below — adjacent samples
    // (~stride seconds apart) never meet it; only the closure of a one-rev
    // segment does.
    let total_span: f64 = coasts
        .iter()
        .filter_map(|c| {
            let f = c.samples.first()?.time;
            let l = c.samples.last()?.time;
            Some(l - f)
        })
        .sum();
    let seam_time_threshold = total_span * 0.5;

    // Track the two screen-closest samples for the seam tiebreak below.
    let mut best_a: Option<ClosestTrailPoint> = None;
    let mut best_b: Option<ClosestTrailPoint> = None;
    let focused_ghost = flight_plan_view.focused_ghost_ref();
    let focused_ghost_pin = flight_plan_view
        .focused_ghost()
        .map(|focus| flight_plan_view.pin_for_ghost_focus(focus, body_states));

    for coast in coasts {
        let Some(first) = coast.samples.first() else {
            continue;
        };
        let rail = body_rail_from_segment(coast);
        // Per-leg relock makes all samples in a coast share an anchor,
        // so the pin is constant across the segment — compute once.
        let pin = flight_plan_view.pin_for_body(first.anchor_body, first.time, body_states);
        for sample in &coast.samples {
            let (world_pos, anchor_body) = if let Some(ghost) = focused_ghost {
                if !flight_plan_view.focused_ghost_contains_epoch(sample.time) {
                    continue;
                }
                let pin = focused_ghost_pin
                    .unwrap_or_else(|| flight_plan_view.pin_for_ghost(ghost, body_states));
                let soi_radius = system
                    .bodies
                    .get(ghost.body_id)
                    .map(|body| body.soi_radius_m)
                    .unwrap_or(f64::INFINITY);
                (
                    ghost_local_world_pos_at_pin(
                        sample.time,
                        sample.position,
                        ghost.body_id,
                        pin,
                        soi_radius,
                        origin,
                        scale,
                        ephemeris,
                    ),
                    ghost.body_id,
                )
            } else {
                if flight_plan_view.epoch_hidden_in_focus(prediction, system, sample.time) {
                    continue;
                }
                (
                    sample_render_pos(sample, pin, origin, scale),
                    sample.anchor_body,
                )
            };
            let Some(screen_pos) = camera.world_to_viewport(cam_transform, world_pos).ok() else {
                continue;
            };
            let d = (screen_pos - cursor_pos).length();
            let candidate = ClosestTrailPoint {
                time: sample.time,
                world_pos,
                anchor_body,
                screen_distance: d,
                sample_position: sample.position,
                sample_velocity: sample.velocity,
                rail: rail.clone(),
            };
            match (&best_a, &best_b) {
                (None, _) => best_a = Some(candidate),
                (Some(a), _) if d < a.screen_distance => {
                    best_b = best_a.take();
                    best_a = Some(candidate);
                }
                (Some(_), None) => best_b = Some(candidate),
                (Some(_), Some(b)) if d < b.screen_distance => best_b = Some(candidate),
                _ => {}
            }
        }
    }

    let a = best_a?;
    let Some(b) = best_b else {
        return Some(a);
    };

    // Wrap-seam hysteresis: when the two best candidates are nearly tied in
    // screen distance but separated by more than half the searched span,
    // we're at a closed-orbit closure point — pick the one whose time
    // continues from `prev_time` to avoid snap-flicker.
    const SEAM_DIST_PX: f32 = 8.0;
    if (b.screen_distance - a.screen_distance) < SEAM_DIST_PX
        && (a.time - b.time).abs() > seam_time_threshold
        && (b.time - prev_time).abs() < (a.time - prev_time).abs()
    {
        return Some(b);
    }
    Some(a)
}

pub(super) fn closest_trail_point_on_rail(
    rail: &TrajectoryRail,
    prev_time: f64,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    scale: &WorldScale,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
    camera: &Camera,
    cam_transform: &GlobalTransform,
    cursor_pos: Vec2,
) -> Option<ClosestTrailPoint> {
    let (start, end) = rail.epoch_range()?;
    let seam_time_threshold = (end - start) * 0.5;

    let mut best_a: Option<ClosestTrailPoint> = None;
    let mut best_b: Option<ClosestTrailPoint> = None;

    for time in rail_candidate_times(rail) {
        let Some(state) = rail.state_at(time) else {
            continue;
        };
        let Some(world_pos) = rail_world_position_for_state(
            rail,
            time,
            state.position,
            body_states,
            origin,
            scale,
            ephemeris,
            flight_plan_view,
        ) else {
            continue;
        };
        let Some(screen_pos) = camera.world_to_viewport(cam_transform, world_pos).ok() else {
            continue;
        };
        let d = (screen_pos - cursor_pos).length();
        let candidate = ClosestTrailPoint {
            time,
            world_pos,
            anchor_body: rail.reference_body,
            screen_distance: d,
            sample_position: state.position,
            sample_velocity: state.velocity,
            rail: Some(rail.clone()),
        };
        match (&best_a, &best_b) {
            (None, _) => best_a = Some(candidate),
            (Some(a), _) if d < a.screen_distance => {
                best_b = best_a.take();
                best_a = Some(candidate);
            }
            (Some(_), None) => best_b = Some(candidate),
            (Some(_), Some(b)) if d < b.screen_distance => best_b = Some(candidate),
            _ => {}
        }
    }

    let a = best_a?;
    let Some(b) = best_b else {
        return Some(a);
    };

    const SEAM_DIST_PX: f32 = 8.0;
    if (b.screen_distance - a.screen_distance) < SEAM_DIST_PX
        && (a.time - b.time).abs() > seam_time_threshold
        && (b.time - prev_time).abs() < (a.time - prev_time).abs()
    {
        return Some(b);
    }
    Some(a)
}

pub(super) fn closest_trail_point(
    prediction: &FlightPlan,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    scale: &WorldScale,
    system: &SolarSystemDefinition,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
    camera: &Camera,
    cam_transform: &GlobalTransform,
    cursor_pos: Vec2,
) -> Option<ClosestTrailPoint> {
    let mut best: Option<ClosestTrailPoint> = None;

    for rail in visible_trajectory_rails(prediction, system, ephemeris, flight_plan_view) {
        update_best_on_rail(
            &mut best,
            Some(rail),
            body_states,
            origin,
            scale,
            ephemeris,
            flight_plan_view,
            camera,
            cam_transform,
            cursor_pos,
        );
    }

    best
}

#[allow(clippy::too_many_arguments)]
fn update_best_on_rail(
    best: &mut Option<ClosestTrailPoint>,
    rail: Option<TrajectoryRail>,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    scale: &WorldScale,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
    camera: &Camera,
    cam_transform: &GlobalTransform,
    cursor_pos: Vec2,
) {
    let Some(rail) = rail else {
        return;
    };

    for time in rail_candidate_times(&rail) {
        let Some(state) = rail.state_at(time) else {
            continue;
        };
        let Some(world_pos) = rail_world_position_for_state(
            &rail,
            time,
            state.position,
            body_states,
            origin,
            scale,
            ephemeris,
            flight_plan_view,
        ) else {
            continue;
        };
        let Some(screen_pos) = camera.world_to_viewport(cam_transform, world_pos).ok() else {
            continue;
        };
        let d = (screen_pos - cursor_pos).length();
        let is_better = best.as_ref().is_none_or(|b| d < b.screen_distance);
        if is_better {
            *best = Some(ClosestTrailPoint {
                time,
                world_pos,
                anchor_body: rail.reference_body,
                screen_distance: d,
                sample_position: state.position,
                sample_velocity: state.velocity,
                rail: Some(rail.clone()),
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn ghost_local_world_pos_at_pin(
    time: f64,
    craft_position: DVec3,
    body_id: usize,
    pin: DVec3,
    soi_radius: f64,
    origin: &RenderOrigin,
    scale: &WorldScale,
    ephemeris: &dyn BodyStateProvider,
) -> Vec3 {
    let body = ephemeris.query_body(body_id, time);
    let relative = craft_position - body.position;
    let display_relative =
        if soi_radius.is_finite() && soi_radius > 0.0 && relative.length() > soi_radius {
            relative
                .try_normalize()
                .map(|direction| direction * soi_radius)
                .unwrap_or(relative)
        } else {
            relative
        };
    ((display_relative + pin - origin.position) * scale.0).as_vec3()
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
