//! Trajectory line rendering for the predicted ship path.
//!
//! Each leg has a single anchor body (set by the per-leg relock in
//! [`thalos_physics::trajectory::propagate_flight_plan`]), so we
//! compute the rendering pin once per leg via
//! [`FlightPlanView::pin_for_body`] and reuse it for every sample in
//! that leg. `sample_render_pos` then applies the standard
//! `(sample.pos − sample.ref_pos) + pin − origin` formula.
//!
//! Encounter windows also get an overlay sampled directly from
//! [`FlightPlan::state_at`]. That overlay subtracts the encountered
//! body's ephemeris at each sample time, then draws the relative ship
//! path around the matching ghost pin. This keeps a future SOI's local
//! trajectory visible even when the current patched-conics leg is still
//! locked to a departure-frame anchor.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics::body_state_provider::BodyStateProvider;
use thalos_physics::trajectory::{FlightPlan, NumericSegment, Trajectory, TrajectoryEventKind};
use thalos_physics::types::{BodyId, SolarSystemDefinition, TrajectorySample};

use crate::coords::{RenderGhostFocus, RenderOrigin, WorldScale, sample_render_pos};
use crate::rendering::{FrameBodyStates, SimulationState};

use super::view::{FlightPlanView, Ghost, GhostPhase};

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

pub(super) fn render_trajectory(
    sim: Option<Res<SimulationState>>,
    origin: Res<RenderOrigin>,
    scale: Res<WorldScale>,
    cache: Res<FrameBodyStates>,
    view: Res<FlightPlanView>,
    mut gizmos: Gizmos,
) {
    let Some(sim) = sim else { return };
    let Some(prediction) = sim.simulation.prediction() else {
        return;
    };
    let Some(ref body_states) = cache.states else {
        return;
    };

    let focused_ghost = view.focused_ghost();
    if focused_ghost.is_some()
        && render_ghost_encounter_windows(
            prediction,
            &view,
            sim.ephemeris.as_ref(),
            &sim.system,
            body_states,
            &origin,
            &scale,
            &mut gizmos,
        )
    {
        return;
    }

    let mut prev_end: Option<(Vec3, BodyId)> = None;

    for (leg_idx, leg) in prediction.legs().iter().enumerate() {
        let is_ghost_leg = leg_idx > 0;

        if let Some(burn) = &leg.burn_segment {
            // Per-segment relock: burn and coast within one leg can
            // carry distinct anchors, so each gets its own pin.
            let burn_pin = segment_pin(burn, &view, body_states);
            let burn_anchor = segment_anchor(burn);

            // Bridge only within the same anchor frame. Across SOI
            // frame changes a straight line would be a visual artifact,
            // not a physical trajectory.
            if let (Some((prev, prev_anchor)), Some(first), Some(anchor)) =
                (prev_end, burn.samples.first(), burn_anchor)
                && prev_anchor == anchor
            {
                let first_pos = sample_render_pos(first, burn_pin, &origin, &scale);
                let color = ghost_adjust(Color::srgba(1.0, 1.0, 1.0, 0.5), is_ghost_leg);
                gizmos.line(prev, first_pos, color);
            }
            render_burn_segment(
                burn,
                burn_pin,
                prediction,
                &view,
                &sim.system,
                &origin,
                &scale,
                &mut gizmos,
            );
            if let Some(last) = burn.samples.last() {
                if let Some(anchor) = burn_anchor {
                    prev_end = Some((sample_render_pos(last, burn_pin, &origin, &scale), anchor));
                }
            }
        }

        let coast_pin = segment_pin(&leg.coast_segment, &view, body_states);
        let coast_anchor = segment_anchor(&leg.coast_segment);
        if let (Some((prev, prev_anchor)), Some(first), Some(anchor)) =
            (prev_end, leg.coast_segment.samples.first(), coast_anchor)
            && prev_anchor == anchor
        {
            let first_pos = sample_render_pos(first, coast_pin, &origin, &scale);
            let color = ghost_adjust(Color::srgba(1.0, 1.0, 1.0, 0.5), is_ghost_leg);
            gizmos.line(prev, first_pos, color);
        }
        prev_end = render_segment(
            &leg.coast_segment,
            coast_pin,
            is_ghost_leg,
            prediction,
            &view,
            &sim.system,
            &origin,
            &scale,
            &mut gizmos,
        )
        .zip(coast_anchor);
    }

    // Baseline: original trajectory without maneuvers. Pinned to its
    // own first-sample anchor at its first-sample time, which means a
    // maneuver that shifts an active-plan encounter doesn't drag the
    // baseline along — they have independent ghost lookups.
    if let Some(baseline) = &prediction.baseline
        && !baseline.samples.is_empty()
    {
        let pin = segment_pin(baseline, &view, body_states);
        render_segment(
            baseline,
            pin,
            true,
            prediction,
            &view,
            &sim.system,
            &origin,
            &scale,
            &mut gizmos,
        );
    }

    let _ = render_ghost_encounter_windows(
        prediction,
        &view,
        sim.ephemeris.as_ref(),
        &sim.system,
        body_states,
        &origin,
        &scale,
        &mut gizmos,
    );

    render_focus_soi_transition_markers(
        prediction,
        &view,
        sim.ephemeris.as_ref(),
        &sim.system,
        body_states,
        &origin,
        &scale,
        &mut gizmos,
    );
}

// ---------------------------------------------------------------------------
// Pin computation
// ---------------------------------------------------------------------------

/// Pin in physics-space metres for a relocked segment. The relock
/// guarantees every sample in the segment shares the first sample's
/// anchor, so reading the first sample is sufficient — burn and coast
/// each get their own pin even when they belong to the same leg.
fn segment_pin(
    segment: &NumericSegment,
    view: &FlightPlanView,
    body_states: &[thalos_physics::types::BodyState],
) -> bevy::math::DVec3 {
    segment
        .samples
        .first()
        .map(|s| view.pin_for_body(s.anchor_body, s.time, body_states))
        .unwrap_or(bevy::math::DVec3::ZERO)
}

fn segment_anchor(segment: &NumericSegment) -> Option<BodyId> {
    segment.samples.first().map(|sample| sample.anchor_body)
}

// ---------------------------------------------------------------------------
// Segment rendering
// ---------------------------------------------------------------------------

fn render_burn_segment(
    segment: &NumericSegment,
    pin: bevy::math::DVec3,
    prediction: &FlightPlan,
    view: &FlightPlanView,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
) {
    if segment.samples.len() < 2 {
        return;
    }
    let burn_color = Color::srgba(1.0, 0.65, 0.1, 1.0);
    for pair in segment.samples.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if view.interval_hidden_in_focus(prediction, system, a.time, b.time) {
            continue;
        }
        gizmos.line(
            sample_render_pos(a, pin, origin, scale),
            sample_render_pos(b, pin, origin, scale),
            burn_color,
        );
    }
}

fn render_segment(
    segment: &NumericSegment,
    pin: bevy::math::DVec3,
    is_ghost: bool,
    prediction: &FlightPlan,
    view: &FlightPlanView,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
) -> Option<Vec3> {
    if segment.samples.is_empty() {
        return None;
    }

    if segment.is_stable_orbit {
        return render_stable_orbit_segment(
            segment, pin, is_ghost, prediction, view, system, origin, scale, gizmos,
        );
    }

    render_open_samples(
        &segment.samples,
        pin,
        is_ghost,
        prediction,
        view,
        system,
        origin,
        scale,
        gizmos,
    )
}

fn render_open_samples(
    samples: &[TrajectorySample],
    pin: bevy::math::DVec3,
    is_ghost: bool,
    prediction: &FlightPlan,
    view: &FlightPlanView,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
) -> Option<Vec3> {
    if samples.is_empty() {
        return None;
    }

    let total = samples.len();
    for k in 0..total.saturating_sub(1) {
        let a = &samples[k];
        let b = &samples[k + 1];
        if view.interval_hidden_in_focus(prediction, system, a.time, b.time) {
            continue;
        }

        let progress_a = k as f32 / total as f32;
        let progress_b = (k + 1) as f32 / total as f32;
        let alpha_a = 0.3 + 0.7 * (1.0 - progress_a);
        let alpha_b = 0.3 + 0.7 * (1.0 - progress_b);

        let p_a = sample_render_pos(a, pin, origin, scale);
        let p_b = sample_render_pos(b, pin, origin, scale);

        let color_a = ghost_adjust(line_color(a, b, system, alpha_a), is_ghost);
        let color_b = ghost_adjust(line_color(b, a, system, alpha_b), is_ghost);
        gizmos.line_gradient(p_a, p_b, color_a, color_b);
    }

    Some(sample_render_pos(&samples[total - 1], pin, origin, scale))
}

fn render_stable_orbit_segment(
    segment: &NumericSegment,
    pin: bevy::math::DVec3,
    is_ghost: bool,
    prediction: &FlightPlan,
    view: &FlightPlanView,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
) -> Option<Vec3> {
    let loop_start = segment
        .stable_orbit_start_index
        .unwrap_or(0)
        .min(segment.samples.len().saturating_sub(1));

    if loop_start > 0 {
        render_open_samples(
            &segment.samples[..=loop_start],
            pin,
            is_ghost,
            prediction,
            view,
            system,
            origin,
            scale,
            gizmos,
        );
    }

    render_stable_orbit(
        &segment.samples[loop_start..],
        pin,
        is_ghost,
        prediction,
        view,
        system,
        origin,
        scale,
        gizmos,
    )
}

fn render_stable_orbit(
    samples: &[TrajectorySample],
    pin: bevy::math::DVec3,
    is_ghost: bool,
    prediction: &FlightPlan,
    view: &FlightPlanView,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
) -> Option<Vec3> {
    if samples.len() < 2 {
        return None;
    }

    let anchor = samples[0].anchor_body;
    let [r, g, b] = system
        .bodies
        .get(anchor)
        .map(|bd| bd.color)
        .unwrap_or([1.0, 1.0, 1.0]);
    let color = ghost_adjust(Color::srgba(r, g, b, 1.0), is_ghost);

    for pair in samples.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if view.interval_hidden_in_focus(prediction, system, a.time, b.time) {
            continue;
        }
        gizmos.line(
            sample_render_pos(a, pin, origin, scale),
            sample_render_pos(b, pin, origin, scale),
            color,
        );
    }
    Some(sample_render_pos(samples.last()?, pin, origin, scale))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn render_focus_soi_transition_markers(
    prediction: &FlightPlan,
    view: &FlightPlanView,
    ephemeris: &dyn BodyStateProvider,
    system: &SolarSystemDefinition,
    body_states: &[thalos_physics::types::BodyState],
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
) {
    if view.focused_ghost().is_some() {
        return;
    }

    let focus_body = view.focus_body();
    let bodies = &system.bodies;
    let Some(focus_def) = system.bodies.get(focus_body) else {
        return;
    };
    let focus_parent = focus_def.parent;

    let pin = view.pin_for_body(focus_body, prediction.epoch_range().0, body_states);
    let marker_radius = transition_marker_radius(focus_def);

    for event in prediction.events() {
        let marker_color = match event.kind {
            // Leaving the focused body's SOI for its parent.
            TrajectoryEventKind::SoiExit if Some(event.body) == focus_parent => {
                Color::srgba(1.0, 0.48, 0.20, 1.0)
            }
            // Entering the focused body's SOI from its parent.
            TrajectoryEventKind::SoiEntry if event.body == focus_body => {
                Color::srgba(0.30, 0.85, 1.0, 1.0)
            }
            // Entering a child body's SOI while still viewing the parent.
            TrajectoryEventKind::SoiEntry
                if bodies.get(event.body).and_then(|body| body.parent) == Some(focus_body) =>
            {
                Color::srgba(0.30, 0.85, 1.0, 1.0)
            }
            // Exiting a child body's SOI back into the focused body.
            TrajectoryEventKind::SoiExit if event.body == focus_body => {
                Color::srgba(1.0, 0.48, 0.20, 1.0)
            }
            _ => continue,
        };

        let focus_state = ephemeris.query_body(focus_body, event.epoch);
        let relative = event.craft_state.position - focus_state.position;
        let center = relative + pin;
        draw_cross_marker(center, marker_radius, marker_color, origin, scale, gizmos);
    }
}

fn render_ghost_encounter_windows(
    prediction: &FlightPlan,
    view: &FlightPlanView,
    ephemeris: &dyn BodyStateProvider,
    system: &SolarSystemDefinition,
    body_states: &[thalos_physics::types::BodyState],
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
) -> bool {
    let Some(focus_ghost) = view.focused_ghost() else {
        return false;
    };

    let mut rendered = false;
    for ghost in view.ghosts() {
        if !is_focused_ghost(ghost, focus_ghost) {
            continue;
        }
        if ghost.phase == GhostPhase::Retired {
            continue;
        }
        let Some(window) = ghost.trajectory_window else {
            continue;
        };
        if window.end_epoch <= window.start_epoch {
            continue;
        }

        let pin = view.pin_for_ghost_focus(focus_ghost, body_states);
        rendered = true;

        let soi_radius = system
            .bodies
            .get(ghost.body_id)
            .map(|body| body.soi_radius_m)
            .unwrap_or(f64::INFINITY);
        let end = window.exit_epoch.unwrap_or(window.end_epoch);
        let start = window.start_epoch;
        for leg in prediction.legs() {
            if let Some(burn) = &leg.burn_segment {
                render_ghost_segment(
                    burn, true, ghost, ephemeris, pin, soi_radius, start, end, system, origin,
                    scale, gizmos,
                );
            }
            render_ghost_segment(
                &leg.coast_segment,
                false,
                ghost,
                ephemeris,
                pin,
                soi_radius,
                start,
                end,
                system,
                origin,
                scale,
                gizmos,
            );
        }

        draw_window_marker(
            prediction,
            ghost,
            ephemeris,
            pin,
            window.start_epoch,
            soi_radius,
            true,
            system,
            origin,
            scale,
            gizmos,
            Color::srgba(0.35, 0.85, 1.0, 0.95),
        );
        if let Some(closest_epoch) = window.closest_epoch {
            draw_window_marker(
                prediction,
                ghost,
                ephemeris,
                pin,
                closest_epoch,
                soi_radius,
                false,
                system,
                origin,
                scale,
                gizmos,
                Color::srgba(1.0, 0.88, 0.25, 1.0),
            );
        }
        if let Some(exit_epoch) = window.exit_epoch {
            draw_window_marker(
                prediction,
                ghost,
                ephemeris,
                pin,
                exit_epoch,
                soi_radius,
                true,
                system,
                origin,
                scale,
                gizmos,
                Color::srgba(1.0, 0.45, 0.25, 0.9),
            );
        }
    }
    rendered
}

fn transition_marker_radius(body: &thalos_physics::types::BodyDefinition) -> f64 {
    if body.soi_radius_m.is_finite() && body.soi_radius_m > body.radius_m {
        let min = body.radius_m * 2.0;
        let max = (body.soi_radius_m * 0.04).max(min);
        (body.soi_radius_m * 0.012).clamp(min, max)
    } else {
        body.radius_m.max(1.0) * 4.0
    }
}

fn draw_cross_marker(
    center: DVec3,
    radius: f64,
    color: Color,
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
) {
    for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
        let a = ((center - axis * radius - origin.position) * scale.0).as_vec3();
        let b = ((center + axis * radius - origin.position) * scale.0).as_vec3();
        gizmos.line(a, b, color);
    }
}

fn is_focused_ghost(ghost: &Ghost, focus: RenderGhostFocus) -> bool {
    focus.matches(ghost.body_id, ghost.encounter_epoch)
}

#[allow(clippy::too_many_arguments)]
fn render_ghost_segment(
    segment: &NumericSegment,
    is_burn: bool,
    ghost: &Ghost,
    ephemeris: &dyn BodyStateProvider,
    pin: DVec3,
    soi_radius: f64,
    start: f64,
    end: f64,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
) -> bool {
    let points = ghost_segment_points(
        segment, ghost, ephemeris, pin, soi_radius, start, end, origin, scale,
    );
    if points.len() < 2 {
        return false;
    }

    let total = points.len();
    for k in 0..total.saturating_sub(1) {
        let alpha_a = 0.9 - 0.45 * (k as f32 / total as f32);
        let alpha_b = 0.9 - 0.45 * ((k + 1) as f32 / total as f32);
        let color_a = if is_burn {
            Color::srgba(1.0, 0.65, 0.1, alpha_a)
        } else {
            encounter_color(ghost.body_id, system, alpha_a, 0.20)
        };
        let color_b = if is_burn {
            Color::srgba(1.0, 0.65, 0.1, alpha_b)
        } else {
            encounter_color(ghost.body_id, system, alpha_b, 0.20)
        };
        gizmos.line_gradient(points[k].1, points[k + 1].1, color_a, color_b);
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn ghost_segment_points(
    segment: &NumericSegment,
    ghost: &Ghost,
    ephemeris: &dyn BodyStateProvider,
    pin: DVec3,
    soi_radius: f64,
    start: f64,
    end: f64,
    origin: &RenderOrigin,
    scale: &WorldScale,
) -> Vec<(f64, Vec3)> {
    let mut points = Vec::new();
    let mut inside_started = false;
    let (segment_start, segment_end) = segment.epoch_range();
    let start = start.max(segment_start);
    let end = end.min(segment_end);

    if end <= start {
        return points;
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

    for (i, t) in times.into_iter().enumerate() {
        let Some(state) = segment.state_at(t) else {
            continue;
        };
        let sample = relative_render_point(
            t,
            state.position,
            ghost.body_id,
            ephemeris,
            pin,
            soi_radius,
            origin,
            scale,
        );

        if sample.inside {
            points.push((t, sample.position));
            inside_started = true;
        } else if i == 0 {
            points.push((t, sample.position));
            inside_started = true;
        } else if inside_started {
            points.push((t, sample.position));
            break;
        }
    }

    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    points.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6);
    points
}

struct RelativeRenderPoint {
    position: Vec3,
    inside: bool,
}

fn relative_render_point(
    time: f64,
    craft_position: DVec3,
    body_id: BodyId,
    ephemeris: &dyn BodyStateProvider,
    pin: DVec3,
    soi_radius: f64,
    origin: &RenderOrigin,
    scale: &WorldScale,
) -> RelativeRenderPoint {
    let body_state = ephemeris.query_body(body_id, time);
    let relative = craft_position - body_state.position;
    let inside = !soi_radius.is_finite() || relative.length() <= soi_radius * 1.000_001;
    let display_relative = if inside || !soi_radius.is_finite() || soi_radius <= 0.0 {
        relative
    } else {
        relative
            .try_normalize()
            .map(|direction| direction * soi_radius)
            .unwrap_or(relative)
    };
    RelativeRenderPoint {
        position: ((display_relative + pin - origin.position) * scale.0).as_vec3(),
        inside,
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_window_marker(
    prediction: &FlightPlan,
    ghost: &Ghost,
    ephemeris: &dyn BodyStateProvider,
    pin: DVec3,
    epoch: f64,
    soi_radius: f64,
    clamp_to_soi_boundary: bool,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
    color: Color,
) {
    let Some(state) = prediction.state_at(epoch) else {
        return;
    };
    let body_state = ephemeris.query_body(ghost.body_id, epoch);
    let mut relative = state.position - body_state.position;
    if soi_radius.is_finite() && relative.length() > soi_radius * 1.000_001 {
        if !clamp_to_soi_boundary {
            return;
        }
        relative = relative
            .try_normalize()
            .map(|direction| direction * soi_radius)
            .unwrap_or(relative);
    }
    let center = relative + pin;

    let marker_radius = system
        .bodies
        .get(ghost.body_id)
        .map(transition_marker_radius)
        .unwrap_or(1.0);
    draw_cross_marker(center, marker_radius, color, origin, scale, gizmos);
}

fn encounter_color(body_id: BodyId, system: &SolarSystemDefinition, alpha: f32, mix: f32) -> Color {
    let [r, g, b] = body_color(body_id, system);
    Color::srgba(
        r + (1.0 - r) * mix,
        g + (1.0 - g) * mix,
        b + (1.0 - b) * mix,
        alpha,
    )
}

fn ghost_adjust(color: Color, is_ghost: bool) -> Color {
    if !is_ghost {
        return color;
    }
    let srgba = color.to_srgba();
    let mix = 0.3;
    Color::srgba(
        srgba.red + (1.0 - srgba.red) * mix,
        srgba.green + (1.0 - srgba.green) * mix,
        srgba.blue + (1.0 - srgba.blue) * mix,
        srgba.alpha * 0.6,
    )
}

fn line_color(
    this: &TrajectorySample,
    other: &TrajectorySample,
    system: &SolarSystemDefinition,
    alpha: f32,
) -> Color {
    // Per-leg anchor relock makes the in-leg anchor uniform, but the
    // cross-segment guard here is harmless and stays useful if a future
    // anchor mode (e.g. weighted barycenter) emits varying anchors
    // within a single leg.
    let [r0, g0, b0] = body_color(this.anchor_body, system);
    if this.anchor_body == other.anchor_body {
        return Color::srgba(r0, g0, b0, alpha);
    }
    let [r1, g1, b1] = body_color(other.anchor_body, system);
    Color::srgba(0.5 * (r0 + r1), 0.5 * (g0 + g1), 0.5 * (b0 + b1), alpha)
}

fn body_color(id: BodyId, system: &SolarSystemDefinition) -> [f32; 3] {
    system
        .bodies
        .get(id)
        .map(|bd| bd.color)
        .unwrap_or([1.0, 1.0, 1.0])
}
