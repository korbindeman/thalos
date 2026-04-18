//! Trajectory line rendering for the predicted ship path.
//!
//! Every sample is drawn through [`sample_render_pos`], which applies the
//! gravity-weighted barycenter rule from `docs/orbital_mechanics.md` §7.2:
//! `render_pos = (sample.pos − ref(sample.t)) + ref(now)` where
//! `ref = Σ wᵢ · rᵢ` uses the sample's cached top-K body weights. This
//! collapses to anchor-relative rendering when a single body dominates
//! and smoothly transitions through Hill-sphere regions — one continuous
//! rule, no run splitting, no bridge lines.

use bevy::prelude::*;
use thalos_physics::trajectory::NumericSegment;
use thalos_physics::types::{SolarSystemDefinition, TrajectorySample};

use crate::coords::{RenderOrigin, sample_render_pos};
use crate::rendering::{FrameBodyStates, ShowOrbits, SimulationState};

use super::view::FlightPlanView;

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

pub(super) fn render_trajectory(
    sim: Option<Res<SimulationState>>,
    origin: Res<RenderOrigin>,
    cache: Res<FrameBodyStates>,
    view: Res<FlightPlanView>,
    show_orbits: Res<ShowOrbits>,
    mut gizmos: Gizmos,
) {
    if !show_orbits.0 {
        return;
    }
    let Some(sim) = sim else { return };
    let Some(prediction) = sim.simulation.prediction() else {
        return;
    };
    let Some(ref body_states) = cache.states else {
        return;
    };

    let pin_for = |body_id| view.pin_for_body(body_id, body_states);
    let mut prev_end_pos: Option<Vec3> = None;

    for (leg_idx, leg) in prediction.legs().iter().enumerate() {
        let is_ghost_leg = leg_idx > 0;

        if let Some(burn) = &leg.burn_segment {
            if let (Some(prev), Some(first)) = (prev_end_pos, burn.samples.first()) {
                let first_pos = sample_render_pos(first, &pin_for, &origin);
                let color = ghost_adjust(Color::srgba(1.0, 1.0, 1.0, 0.5), is_ghost_leg);
                gizmos.line(prev, first_pos, color);
            }
            render_burn_segment(burn, &pin_for, &origin, &mut gizmos);
            if let Some(last) = burn.samples.last() {
                prev_end_pos = Some(sample_render_pos(last, &pin_for, &origin));
            }
        }

        if let (Some(prev), Some(first)) = (prev_end_pos, leg.coast_segment.samples.first()) {
            let first_pos = sample_render_pos(first, &pin_for, &origin);
            let color = ghost_adjust(Color::srgba(1.0, 1.0, 1.0, 0.5), is_ghost_leg);
            gizmos.line(prev, first_pos, color);
        }
        prev_end_pos = render_segment(
            &leg.coast_segment,
            &pin_for,
            is_ghost_leg,
            &sim.system,
            &origin,
            &mut gizmos,
        );
    }

    // Baseline: original trajectory without maneuvers (dimmed).
    if let Some(baseline) = &prediction.baseline
        && !baseline.samples.is_empty()
    {
        render_segment(baseline, &pin_for, true, &sim.system, &origin, &mut gizmos);
    }
}

// ---------------------------------------------------------------------------
// Segment rendering
// ---------------------------------------------------------------------------

fn render_burn_segment(
    segment: &NumericSegment,
    pin_for: &impl Fn(usize) -> bevy::math::DVec3,
    origin: &RenderOrigin,
    gizmos: &mut Gizmos,
) {
    if segment.samples.len() < 2 {
        return;
    }
    let burn_color = Color::srgba(1.0, 0.65, 0.1, 1.0);
    let points = segment
        .samples
        .iter()
        .map(|s| sample_render_pos(s, pin_for, origin));
    gizmos.linestrip(points, burn_color);
}

fn render_segment(
    segment: &NumericSegment,
    pin_for: &impl Fn(usize) -> bevy::math::DVec3,
    is_ghost: bool,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    gizmos: &mut Gizmos,
) -> Option<Vec3> {
    if segment.samples.is_empty() {
        return None;
    }

    if segment.is_stable_orbit {
        return render_stable_orbit_segment(segment, pin_for, is_ghost, system, origin, gizmos);
    }

    render_open_samples(&segment.samples, pin_for, is_ghost, system, origin, gizmos)
}

fn render_open_samples(
    samples: &[TrajectorySample],
    pin_for: &impl Fn(usize) -> bevy::math::DVec3,
    is_ghost: bool,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
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

        let p_a = sample_render_pos(a, pin_for, origin);
        let p_b = sample_render_pos(b, pin_for, origin);

        let color_a = ghost_adjust(line_color(a, b, system, alpha_a), is_ghost);
        let color_b = ghost_adjust(line_color(b, a, system, alpha_b), is_ghost);
        gizmos.line_gradient(p_a, p_b, color_a, color_b);
    }

    Some(sample_render_pos(&samples[total - 1], pin_for, origin))
}

fn render_stable_orbit_segment(
    segment: &NumericSegment,
    pin_for: &impl Fn(usize) -> bevy::math::DVec3,
    is_ghost: bool,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    gizmos: &mut Gizmos,
) -> Option<Vec3> {
    let loop_start = segment
        .stable_orbit_start_index
        .unwrap_or(0)
        .min(segment.samples.len().saturating_sub(1));

    if loop_start > 0 {
        render_open_samples(
            &segment.samples[..=loop_start],
            pin_for,
            is_ghost,
            system,
            origin,
            gizmos,
        );
    }

    render_stable_orbit(
        &segment.samples[loop_start..],
        pin_for,
        is_ghost,
        system,
        origin,
        gizmos,
    )
}

fn render_stable_orbit(
    samples: &[TrajectorySample],
    pin_for: &impl Fn(usize) -> bevy::math::DVec3,
    is_ghost: bool,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
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

    let last = samples.len() - 1;
    let end_pos = sample_render_pos(&samples[last - 1], pin_for, origin);
    gizmos.linestrip(
        samples[..last]
            .iter()
            .map(|s| sample_render_pos(s, pin_for, origin)),
        color,
    );
    Some(end_pos)
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
    // Under patched conics each sample has a single anchor body. When two
    // consecutive samples land in different SOIs (i.e. we just crossed a
    // Hill-sphere boundary), interpolate half-and-half between the two body
    // colours so the line reads as a soft handoff rather than a hard cut.
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

fn body_color(id: usize, system: &SolarSystemDefinition) -> [f32; 3] {
    system
        .bodies
        .get(id)
        .map(|bd| bd.color)
        .unwrap_or([1.0, 1.0, 1.0])
}
