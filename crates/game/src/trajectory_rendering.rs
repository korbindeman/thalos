//! Trajectory line rendering for the predicted ship path.
//!
//! Each [`TrajectorySample`] carries an absolute heliocentric position, but we
//! render every segment *relative to its anchor body's position at the sample
//! time*.  Anchor body is the smallest SOI containing the ship — geometric
//! containment, not perturbation magnitude — so the rendering frame never
//! flickers mid-orbit.  This makes orbits appear as clean ellipses even as
//! bodies move.
//!
//! Line segments are still tinted by the per-sample `dominant_body` (the
//! strongest gravitational source), so the user sees a smooth color gradient
//! when a perturber's pull rises near an SOI boundary.

use bevy::math::{DVec3, Isometry3d, Vec3A};
use bevy::prelude::*;
use thalos_physics::trajectory::{NumericSegment, cone_width};
use thalos_physics::types::{BodyId, BodyKind, SolarSystemDefinition, TrajectorySample};

use crate::coords::{RENDER_SCALE, RenderOrigin, compute_segment_pins, sample_render_pos};
use crate::rendering::FrameBodyStates;

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------

/// Cone width (metres) at which the trajectory begins to fade out.
const CONE_FADE_THRESHOLD: f64 = 1e6;

/// Cone width (metres) at which the trajectory is fully transparent.
const CONE_INVISIBLE_THRESHOLD: f64 = 2e6;

/// Minimum `perturbation_ratio` for a non-star body to merit a ghost marker.
const GHOST_BODY_PERTURBATION_THRESHOLD: f64 = 0.05;

use crate::rendering::{ShowOrbits, SimulationState};

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct TrajectoryRenderingPlugin;

impl Plugin for TrajectoryRenderingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            render_trajectory
                .after(crate::rendering::cache_body_states)
                .in_set(crate::SimStage::Sync),
        );
    }
}

// ---------------------------------------------------------------------------
// Rendering system
// ---------------------------------------------------------------------------

fn render_trajectory(
    sim: Option<Res<SimulationState>>,
    origin: Res<RenderOrigin>,
    cache: Res<FrameBodyStates>,
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

    for (leg_idx, leg) in prediction.legs().iter().enumerate() {
        // Burn sub-segment is rendered first (when present) as a bright
        // thrust arc — never ghost-dimmed, so the player always sees where
        // the burn happened regardless of how far into the plan it is.
        if let Some(burn) = &leg.burn_segment {
            let pins = compute_segment_pins(&burn.samples, body_states, &sim.system, false);
            render_burn_segment(burn, &pins, &origin, &mut gizmos);
        }

        // Coast sub-segment uses the standard trajectory rendering. Legs
        // after the first get the "ghost" (post-maneuver) dim because the
        // ship has had at least one burn applied by then.
        let is_ghost = leg_idx > 0;
        let pins = compute_segment_pins(
            &leg.coast_segment.samples,
            body_states,
            &sim.system,
            leg_idx == 0,
        );
        render_segment(
            &leg.coast_segment,
            &pins,
            is_ghost,
            &sim.system,
            &origin,
            &mut gizmos,
        );
    }
}

/// Render a burn sub-segment as a bright orange arc so the player sees the
/// thrust interval without relying on ghost/ghost-less dimming.
fn render_burn_segment(
    segment: &NumericSegment,
    pins: &[DVec3],
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
        .zip(pins.iter())
        .map(|(s, &pin)| sample_render_pos(s, pin, origin));
    gizmos.linestrip(points, burn_color);
}

// ---------------------------------------------------------------------------
// Segment rendering
// ---------------------------------------------------------------------------

fn render_segment(
    segment: &NumericSegment,
    pins: &[DVec3],
    is_ghost: bool,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    gizmos: &mut Gizmos,
) {
    if segment.samples.is_empty() {
        return;
    }

    if segment.is_stable_orbit {
        render_stable_orbit_segment(segment, pins, is_ghost, system, origin, gizmos);
        return;
    }

    render_open_samples(&segment.samples, pins, is_ghost, system, origin, gizmos);
}

fn render_open_samples(
    samples: &[TrajectorySample],
    pins: &[DVec3],
    is_ghost: bool,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    gizmos: &mut Gizmos,
) {
    let total_points = samples.len();
    for i in 0..total_points.saturating_sub(1) {
        let a = &samples[i];
        let b = &samples[i + 1];

        let cone_alpha = fade_alpha(a);
        if cone_alpha <= 0.0 {
            break;
        }

        // Progress-based fade combined with cone fade.
        let progress_a = i as f32 / total_points as f32;
        let progress_b = (i + 1) as f32 / total_points as f32;
        let alpha_a = cone_alpha * (0.3 + 0.7 * (1.0 - progress_a));
        let alpha_b = cone_alpha * (0.3 + 0.7 * (1.0 - progress_b));

        let pin_a = pins[i];
        let pin_b = pins[i + 1];
        let p_a = sample_render_pos(a, pin_a, origin);
        let p_b = sample_render_pos(b, pin_b, origin);

        // --- Trajectory line ---
        let color_a = ghost_adjust(line_color(a, b, system, alpha_a), is_ghost);
        let color_b = ghost_adjust(line_color(b, a, system, alpha_b), is_ghost);
        gizmos.line_gradient(p_a, p_b, color_a, color_b);

        // --- Cone proxy (wider translucent parallel offset lines) ---
        let cw_a = cone_width(a) as f32 * RENDER_SCALE as f32;
        let cw_b = cone_width(b) as f32 * RENDER_SCALE as f32;
        if cw_a > 0.0 || cw_b > 0.0 {
            let cone_color_a = ghost_adjust(
                color_with_alpha(body_color(a.dominant_body, system), alpha_a * 0.35),
                is_ghost,
            );
            let cone_color_b = ghost_adjust(
                color_with_alpha(body_color(b.dominant_body, system), alpha_b * 0.35),
                is_ghost,
            );

            let perp_a = velocity_perpendicular(a);
            let perp_b = velocity_perpendicular(b);

            let offset_a = perp_a * cw_a;
            let offset_b = perp_b * cw_b;

            gizmos.line_gradient(p_a + offset_a, p_b + offset_b, cone_color_a, cone_color_b);
            gizmos.line_gradient(p_a - offset_a, p_b - offset_b, cone_color_a, cone_color_b);
        }

        maybe_draw_ghost(a, pin_a, system, origin, gizmos);
    }

    // Ghost check for the final sample.
    if let Some(last) = samples.last()
        && fade_alpha(last) > 0.0
        && let Some(&pin) = pins.last()
    {
        maybe_draw_ghost(last, pin, system, origin, gizmos);
    }
}

// ---------------------------------------------------------------------------
// Stable orbit rendering
// ---------------------------------------------------------------------------

fn render_stable_orbit_segment(
    segment: &NumericSegment,
    pins: &[DVec3],
    is_ghost: bool,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    gizmos: &mut Gizmos,
) {
    let loop_start = segment
        .stable_orbit_start_index
        .unwrap_or(0)
        .min(segment.samples.len().saturating_sub(1));

    if loop_start > 0 {
        render_open_samples(
            &segment.samples[..=loop_start],
            &pins[..=loop_start],
            is_ghost,
            system,
            origin,
            gizmos,
        );
    }

    render_stable_orbit(
        &segment.samples[loop_start..],
        &pins[loop_start..],
        is_ghost,
        system,
        origin,
        gizmos,
    );
}

fn render_stable_orbit(
    samples: &[TrajectorySample],
    pins: &[DVec3],
    is_ghost: bool,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    gizmos: &mut Gizmos,
) {
    if samples.len() < 2 {
        return;
    }

    let anchor = samples[0].anchor_body;
    let [r, g, b] = system
        .bodies
        .get(anchor)
        .map(|bd| bd.color)
        .unwrap_or([1.0, 1.0, 1.0]);
    let color = ghost_adjust(Color::srgba(r, g, b, 1.0), is_ghost);

    // Drop the last sample: the orbit tracker fires as soon as cumulative
    // sweep hits 2π, so the final sample overshoots the start by one step.
    // Without this, a closing edge would cut a chord across the orbit.
    let last = samples.len() - 1;
    gizmos.linestrip(
        samples[..last]
            .iter()
            .zip(pins[..last].iter())
            .map(|(s, &pin)| sample_render_pos(s, pin, origin)),
        color,
    );
}

/// Desaturate and dim a color for ghost (post-maneuver) trajectory segments.
/// Mixes 30% toward white and reduces alpha by 40%.
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Color for the trajectory line at sample `this`, blending toward `other`'s
/// body color using `perturbation_ratio` as the transition signal.
fn line_color(
    this: &TrajectorySample,
    other: &TrajectorySample,
    system: &SolarSystemDefinition,
    alpha: f32,
) -> Color {
    let [r0, g0, b0] = body_color(this.dominant_body, system);

    if this.dominant_body == other.dominant_body {
        return Color::srgba(r0, g0, b0, alpha);
    }

    let [r1, g1, b1] = body_color(other.dominant_body, system);
    let t = this.perturbation_ratio.clamp(0.0, 1.0) as f32;
    Color::srgba(
        r0 + (r1 - r0) * t,
        g0 + (g1 - g0) * t,
        b0 + (b1 - b0) * t,
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

fn color_with_alpha([r, g, b]: [f32; 3], alpha: f32) -> Color {
    Color::srgba(r, g, b, alpha)
}

/// Alpha for the trajectory at this sample, fading to 0 as cone width grows
/// from `CONE_FADE_THRESHOLD` to `CONE_INVISIBLE_THRESHOLD`.
fn fade_alpha(sample: &TrajectorySample) -> f32 {
    let cw = cone_width(sample);
    if cw >= CONE_INVISIBLE_THRESHOLD {
        return 0.0;
    }
    if cw <= CONE_FADE_THRESHOLD {
        return 1.0;
    }
    let t = (cw - CONE_FADE_THRESHOLD) / (CONE_INVISIBLE_THRESHOLD - CONE_FADE_THRESHOLD);
    (1.0 - t) as f32
}

/// Compute a render-space unit vector perpendicular to a trajectory sample's
/// velocity. Uses `cross(velocity_dir, Y)` and falls back to `X` when the
/// velocity is nearly parallel to Y.
fn velocity_perpendicular(sample: &TrajectorySample) -> Vec3 {
    let vel = sample.velocity;
    if vel.length_squared() < 1e-20 {
        return Vec3::X;
    }
    let dir = vel.normalize();
    let cross = dir.cross(bevy::math::DVec3::Y);
    if cross.length_squared() < 1e-12 {
        Vec3::X
    } else {
        cross.normalize().as_vec3()
    }
}

/// Draw a ghost body sphere gizmo at the anchor body's position when the
/// sample's anchor is a non-star body with meaningful gravitational influence.
fn maybe_draw_ghost(
    sample: &TrajectorySample,
    pin: DVec3,
    system: &SolarSystemDefinition,
    origin: &RenderOrigin,
    gizmos: &mut Gizmos,
) {
    if sample.perturbation_ratio < GHOST_BODY_PERTURBATION_THRESHOLD {
        return;
    }

    let Some(body_def) = system.bodies.get(sample.anchor_body) else {
        return;
    };

    if body_def.kind == BodyKind::Star {
        return;
    }

    let ghost_pos = ((pin - origin.position) * RENDER_SCALE).as_vec3();

    let [r, g, b] = body_def.color;
    let alpha = (sample.perturbation_ratio as f32 * 0.6).clamp(0.2, 0.6);
    let color = Color::srgba(r, g, b, alpha);

    // Match the actual body's rendered radius so the ghost is a proper
    // same-size wireframe twin, not a tiny marker.
    let radius = (body_def.radius_m * RENDER_SCALE) as f32;

    gizmos.sphere(
        Isometry3d::from_translation(Vec3A::from(ghost_pos)),
        radius,
        color,
    );
}
