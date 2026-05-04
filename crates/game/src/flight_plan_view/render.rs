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
//!
//! Discrete events (SOI entry/exit, closest approach, apsis) are
//! rendered as pickable mesh markers by `markers.rs`, not here. This
//! module only emits the continuous trajectory lines.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics::trajectory::{FlightPlan, NumericSegment, Trajectory};
use thalos_physics::types::{BodyDefinition, BodyId, TrajectorySample};

use crate::coords::{RenderGhostFocus, RenderOrigin, WorldScale, sample_render_pos};
use crate::map_view::MapSnapshot;

use super::view::{FlightPlanView, Ghost, GhostPhase};

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

pub(super) fn render_trajectory(
    origin: Res<RenderOrigin>,
    scale: Res<WorldScale>,
    snapshot: Res<MapSnapshot>,
    view: Res<FlightPlanView>,
    mut gizmos: Gizmos,
) {
    let Some(prediction) = snapshot.flight_plan.as_ref() else {
        return;
    };
    if snapshot.body_states.is_empty() {
        return;
    }
    let body_states = snapshot.body_states.as_slice();

    let focused_ghost = view.focused_ghost();
    if focused_ghost.is_some()
        && render_ghost_encounter_windows(
            prediction,
            &view,
            &snapshot,
            &snapshot.body_defs,
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
                &snapshot.body_defs,
                &origin,
                &scale,
                &mut gizmos,
            );
            if let Some(last) = burn.samples.last()
                && let Some(anchor) = burn_anchor
            {
                prev_end = Some((sample_render_pos(last, burn_pin, &origin, &scale), anchor));
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
            &snapshot.body_defs,
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
            &snapshot.body_defs,
            &origin,
            &scale,
            &mut gizmos,
        );
    }

    let _ = render_ghost_encounter_windows(
        prediction,
        &view,
        &snapshot,
        &snapshot.body_defs,
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
) -> DVec3 {
    segment
        .samples
        .first()
        .map(|s| view.pin_for_body(s.anchor_body, s.time, body_states))
        .unwrap_or(DVec3::ZERO)
}

fn segment_anchor(segment: &NumericSegment) -> Option<BodyId> {
    segment.samples.first().map(|sample| sample.anchor_body)
}

// ---------------------------------------------------------------------------
// Segment rendering
// ---------------------------------------------------------------------------

fn render_burn_segment(
    segment: &NumericSegment,
    pin: DVec3,
    prediction: &FlightPlan,
    view: &FlightPlanView,
    system: &[BodyDefinition],
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
    pin: DVec3,
    is_ghost: bool,
    prediction: &FlightPlan,
    view: &FlightPlanView,
    system: &[BodyDefinition],
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
    pin: DVec3,
    is_ghost: bool,
    prediction: &FlightPlan,
    view: &FlightPlanView,
    system: &[BodyDefinition],
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
    pin: DVec3,
    is_ghost: bool,
    prediction: &FlightPlan,
    view: &FlightPlanView,
    system: &[BodyDefinition],
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
    pin: DVec3,
    is_ghost: bool,
    prediction: &FlightPlan,
    view: &FlightPlanView,
    system: &[BodyDefinition],
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
) -> Option<Vec3> {
    if samples.len() < 2 {
        return None;
    }

    let anchor = samples[0].anchor_body;
    let [r, g, b] = system
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
// Ghost encounter windows
// ---------------------------------------------------------------------------

fn render_ghost_encounter_windows(
    prediction: &FlightPlan,
    view: &FlightPlanView,
    snapshot: &MapSnapshot,
    system: &[BodyDefinition],
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
        let soi_radius = system
            .get(ghost.body_id)
            .map(|body| body.soi_radius_m)
            .unwrap_or(f64::INFINITY);
        let end = window.exit_epoch.unwrap_or(window.end_epoch);
        let start = window.start_epoch;
        for leg in prediction.legs() {
            if let Some(burn) = &leg.burn_segment {
                rendered |= render_ghost_segment(
                    burn, true, ghost, snapshot, pin, soi_radius, start, end, system, origin,
                    scale, gizmos,
                );
            }
            rendered |= render_ghost_segment(
                &leg.coast_segment,
                false,
                ghost,
                snapshot,
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
    }
    rendered
}

fn is_focused_ghost(ghost: &Ghost, focus: RenderGhostFocus) -> bool {
    focus.matches(ghost.body_id, ghost.encounter_epoch)
}

#[allow(clippy::too_many_arguments)]
fn render_ghost_segment(
    segment: &NumericSegment,
    is_burn: bool,
    ghost: &Ghost,
    snapshot: &MapSnapshot,
    pin: DVec3,
    soi_radius: f64,
    start: f64,
    end: f64,
    system: &[BodyDefinition],
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
) -> bool {
    let points = ghost_segment_points(
        segment, ghost, snapshot, pin, soi_radius, start, end, origin, scale,
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
    snapshot: &MapSnapshot,
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
        let Some(sample) = relative_render_point(
            t,
            state.position,
            ghost.body_id,
            snapshot,
            pin,
            soi_radius,
            origin,
            scale,
        ) else {
            continue;
        };

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
    snapshot: &MapSnapshot,
    pin: DVec3,
    soi_radius: f64,
    origin: &RenderOrigin,
    scale: &WorldScale,
) -> Option<RelativeRenderPoint> {
    let body_state = snapshot.body_state_at(body_id, time)?;
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
    Some(RelativeRenderPoint {
        position: ((display_relative + pin - origin.position) * scale.0).as_vec3(),
        inside,
    })
}

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

fn encounter_color(body_id: BodyId, system: &[BodyDefinition], alpha: f32, mix: f32) -> Color {
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
    system: &[BodyDefinition],
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

fn body_color(id: BodyId, system: &[BodyDefinition]) -> [f32; 3] {
    system.get(id).map(|bd| bd.color).unwrap_or([1.0, 1.0, 1.0])
}
