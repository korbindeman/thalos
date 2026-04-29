//! Trajectory line rendering for the predicted ship path.
//!
//! Each leg has a single anchor body (set by the per-leg relock in
//! [`thalos_physics::trajectory::propagate_flight_plan`]), so we
//! compute the rendering pin once per leg via
//! [`FlightPlanView::pin_for_body`] and reuse it for every sample in
//! that leg. `sample_render_pos` then applies the standard
//! `(sample.pos − sample.ref_pos) + pin − origin` formula.
//!
//! This means the trajectory is drawn in a per-leg "frozen anchor"
//! frame: every sample in a Mira-anchored leg is shown as if Mira sat
//! at `pin` for the whole leg, even though the ship is physically
//! moving past Mira's actual heliocentric trajectory. That's the price
//! patched-conics visualisation pays for legibility — see the
//! [`super::view`] module docstring for the full story.

use bevy::prelude::*;
use thalos_physics::trajectory::NumericSegment;
use thalos_physics::types::{BodyId, SolarSystemDefinition, TrajectorySample};

use crate::coords::{RenderOrigin, WorldScale, sample_render_pos};
use crate::rendering::{FrameBodyStates, SimulationState};

use super::view::FlightPlanView;

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

    let mut prev_end_pos: Option<Vec3> = None;

    for (leg_idx, leg) in prediction.legs().iter().enumerate() {
        let is_ghost_leg = leg_idx > 0;

        if let Some(burn) = &leg.burn_segment {
            // Per-segment relock: burn and coast within one leg can
            // carry distinct anchors, so each gets its own pin.
            let burn_pin = segment_pin(burn, &view, body_states);

            // Bridge gizmo from the previous segment's last point to
            // this burn's first sample. Frames may differ across
            // segments so the bridge can't always meet exactly, but a
            // thin line preserves the visual continuity of the
            // planned path.
            if let (Some(prev), Some(first)) = (prev_end_pos, burn.samples.first()) {
                let first_pos = sample_render_pos(first, burn_pin, &origin, &scale);
                let color = ghost_adjust(Color::srgba(1.0, 1.0, 1.0, 0.5), is_ghost_leg);
                gizmos.line(prev, first_pos, color);
            }
            render_burn_segment(burn, burn_pin, &origin, &scale, &mut gizmos);
            if let Some(last) = burn.samples.last() {
                prev_end_pos = Some(sample_render_pos(last, burn_pin, &origin, &scale));
            }
        }

        let coast_pin = segment_pin(&leg.coast_segment, &view, body_states);
        if let (Some(prev), Some(first)) = (prev_end_pos, leg.coast_segment.samples.first()) {
            let first_pos = sample_render_pos(first, coast_pin, &origin, &scale);
            let color = ghost_adjust(Color::srgba(1.0, 1.0, 1.0, 0.5), is_ghost_leg);
            gizmos.line(prev, first_pos, color);
        }
        prev_end_pos = render_segment(
            &leg.coast_segment,
            coast_pin,
            is_ghost_leg,
            &sim.system,
            &origin,
            &scale,
            &mut gizmos,
        );
    }

    // Baseline: original trajectory without maneuvers. Pinned to its
    // own first-sample anchor at its first-sample time, which means a
    // maneuver that shifts an active-plan encounter doesn't drag the
    // baseline along — they have independent ghost lookups.
    if let Some(baseline) = &prediction.baseline
        && !baseline.samples.is_empty()
    {
        let pin = segment_pin(baseline, &view, body_states);
        render_segment(baseline, pin, true, &sim.system, &origin, &scale, &mut gizmos);
    }
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

// ---------------------------------------------------------------------------
// Segment rendering
// ---------------------------------------------------------------------------

fn render_burn_segment(
    segment: &NumericSegment,
    pin: bevy::math::DVec3,
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
) {
    if segment.samples.len() < 2 {
        return;
    }
    let burn_color = Color::srgba(1.0, 0.65, 0.1, 1.0);
    let points = segment
        .samples
        .iter()
        .map(|s| sample_render_pos(s, pin, origin, scale));
    gizmos.linestrip(points, burn_color);
}

fn render_segment(
    segment: &NumericSegment,
    pin: bevy::math::DVec3,
    is_ghost: bool,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    scale: &WorldScale,
    gizmos: &mut Gizmos,
) -> Option<Vec3> {
    if segment.samples.is_empty() {
        return None;
    }

    if segment.is_stable_orbit {
        return render_stable_orbit_segment(segment, pin, is_ghost, system, origin, scale, gizmos);
    }

    render_open_samples(&segment.samples, pin, is_ghost, system, origin, scale, gizmos)
}

fn render_open_samples(
    samples: &[TrajectorySample],
    pin: bevy::math::DVec3,
    is_ghost: bool,
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

    // The last sample sits at `time + period`, which for a closed Kepler
    // orbit coincides with samples[0] in the anchor frame. Include it so
    // the linestrip's final edge closes the loop; dropping it leaves a
    // visible gap of one sample spacing (~2.8° for 128 samples).
    gizmos.linestrip(
        samples
            .iter()
            .map(|s| sample_render_pos(s, pin, origin, scale)),
        color,
    );
    Some(sample_render_pos(samples.last()?, pin, origin, scale))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
    Color::srgba(
        0.5 * (r0 + r1),
        0.5 * (g0 + g1),
        0.5 * (b0 + b1),
        alpha,
    )
}

fn body_color(id: BodyId, system: &SolarSystemDefinition) -> [f32; 3] {
    system
        .bodies
        .get(id)
        .map(|bd| bd.color)
        .unwrap_or([1.0, 1.0, 1.0])
}
