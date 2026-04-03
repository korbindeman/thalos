//! Trajectory line and cone rendering for the predicted ship path.
//!
//! # Design
//!
//! Each [`TrajectorySample`] carries an absolute heliocentric position, but we
//! render every segment *relative to its dominant body's position at the sample
//! time*.  This makes orbits appear as clean ellipses even as bodies move.
//!
//! Line segments are colored by dominant body.  At body transitions the two
//! adjacent colors are linearly blended using the sample's `perturbation_ratio`
//! as the interpolant, giving a smooth visual gradient at sphere-of-influence
//! boundaries.
//!
//! Cone width is rendered as a thicker translucent parallel line on either side
//! of the trajectory.  A full tube mesh would give a more accurate
//! representation but is left as future work.
//!
//! When cone width exceeds [`CONE_FADE_THRESHOLD`] both the trajectory line and
//! the cone proxy fade to transparent, matching the physics termination point.

use std::collections::HashMap;

use bevy::math::{Isometry3d, Vec3A};
use bevy::prelude::*;
use thalos_physics::ephemeris::Ephemeris;
use thalos_physics::trajectory::{cone_width, TrajectorySegment};
use thalos_physics::types::{BodyId, BodyKind, SolarSystemDefinition, TrajectorySample};

// ---------------------------------------------------------------------------
// Scale
// ---------------------------------------------------------------------------

/// 1 render unit = 1 000 km.  All physics positions (metres) are multiplied
/// by this constant before being passed to Bevy.
const RENDER_SCALE: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------

/// Cone width (metres) at which the trajectory begins to fade out.
/// Should match `PredictionConfig::cone_fade_threshold`.
const CONE_FADE_THRESHOLD: f64 = 1e6;

/// Cone width (metres) at which the trajectory is fully transparent and stops
/// being drawn.
const CONE_INVISIBLE_THRESHOLD: f64 = 2e6;

/// Minimum `perturbation_ratio` (and gravitational dominance) for a non-star
/// body to merit a ghost marker along the trajectory.
const GHOST_BODY_PERTURBATION_THRESHOLD: f64 = 0.05;

/// Radius (render units) of the ghost body wireframe sphere gizmo.
const GHOST_SPHERE_RADIUS: f32 = 0.5;

use crate::rendering::{RenderOrigin, SimulationState};

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct TrajectoryRenderingPlugin;

impl Plugin for TrajectoryRenderingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, render_trajectory.in_set(crate::SimStage::Sync));
    }
}

// ---------------------------------------------------------------------------
// Rendering system
// ---------------------------------------------------------------------------

fn render_trajectory(
    sim: Option<Res<SimulationState>>,
    origin: Res<RenderOrigin>,
    mut gizmos: Gizmos,
) {
    let Some(sim) = sim else { return };
    let Some(prediction) = sim.simulation.prediction() else { return };

    // Cache body positions at sim_time so we don't re-query the ephemeris for
    // every sample point (sim_time is constant for the whole frame).
    let sim_time = sim.simulation.sim_time();
    let mut body_pos_now_cache: HashMap<BodyId, bevy::math::DVec3> = HashMap::new();

    for segment in &prediction.segments {
        render_segment(segment, &sim.system, &sim.ephemeris, &origin, sim_time, &mut gizmos, &mut body_pos_now_cache);
    }
}

// ---------------------------------------------------------------------------
// Segment rendering
// ---------------------------------------------------------------------------

fn render_segment(
    segment: &TrajectorySegment,
    system: &SolarSystemDefinition,
    ephemeris: &Ephemeris,
    origin: &RenderOrigin,
    sim_time: f64,
    gizmos: &mut Gizmos,
    body_pos_now_cache: &mut HashMap<BodyId, bevy::math::DVec3>,
) {
    if segment.samples.is_empty() {
        return;
    }

    // For a stable orbit the termination sample wraps back close to the start,
    // so we can draw the full loop cleanly as a single line strip.  The cone
    // is suppressed because the orbit is stable and uncertainty has not grown.
    if segment.is_stable_orbit {
        render_stable_orbit(segment, system, ephemeris, origin, sim_time, gizmos, body_pos_now_cache);
        return;
    }

    // Walk consecutive pairs of samples and draw each edge individually.
    // This lets us vary color and alpha per segment without building a buffer.
    let samples = &segment.samples;
    let total_points = samples.len();
    for i in 0..total_points.saturating_sub(1) {
        let a = &samples[i];
        let b = &samples[i + 1];

        let cone_alpha = fade_alpha(a);
        if cone_alpha <= 0.0 {
            break; // Remainder of segment is fully faded.
        }

        // Progress-based fade: trajectory fades from full opacity at the start
        // to 30% at the end, giving a visual sense of time/distance.
        let progress_a = i as f32 / total_points as f32;
        let progress_b = (i + 1) as f32 / total_points as f32;
        let alpha_a = cone_alpha * (0.3 + 0.7 * (1.0 - progress_a));
        let alpha_b = cone_alpha * (0.3 + 0.7 * (1.0 - progress_b));

        let p_a = sample_render_pos_cached(a, system, ephemeris, origin, sim_time, body_pos_now_cache);
        let p_b = sample_render_pos_cached(b, system, ephemeris, origin, sim_time, body_pos_now_cache);

        // --- Trajectory line ------------------------------------------------
        let color_a = line_color(a, b, system, alpha_a);
        let color_b = line_color(b, a, system, alpha_b);
        gizmos.line_gradient(p_a, p_b, color_a, color_b);

        // --- Cone proxy (wider translucent parallel offset lines) -----------
        let cw_a = cone_width(a) as f32 * RENDER_SCALE as f32;
        let cw_b = cone_width(b) as f32 * RENDER_SCALE as f32;
        if cw_a > 0.0 || cw_b > 0.0 {
            let cone_color_a = color_with_alpha(body_color(a.dominant_body, system), alpha_a * 0.35);
            let cone_color_b = color_with_alpha(body_color(b.dominant_body, system), alpha_b * 0.35);

            // Perpendicular to velocity at each sample, giving correct cone
            // orientation even on inclined orbits.
            let perp_a = velocity_perpendicular(a);
            let perp_b = velocity_perpendicular(b);

            let offset_a = perp_a * cw_a;
            let offset_b = perp_b * cw_b;

            gizmos.line_gradient(p_a + offset_a, p_b + offset_b, cone_color_a, cone_color_b);
            gizmos.line_gradient(p_a - offset_a, p_b - offset_b, cone_color_a, cone_color_b);
        }

        // --- Ghost body markers ---------------------------------------------
        // Draw a wireframe sphere at the dominant body's position at time `a`
        // whenever that body is a non-star with meaningful gravitational pull.
        maybe_draw_ghost(a, system, ephemeris, origin, sim_time, gizmos, body_pos_now_cache);
    }

    // Ghost check for the final sample (loop above stops at len-1).
    if let Some(last) = samples.last()
        && fade_alpha(last) > 0.0
    {
        maybe_draw_ghost(last, system, ephemeris, origin, sim_time, gizmos, body_pos_now_cache);
    }
}

// ---------------------------------------------------------------------------
// Stable orbit rendering
// ---------------------------------------------------------------------------

fn render_stable_orbit(
    segment: &TrajectorySegment,
    system: &SolarSystemDefinition,
    ephemeris: &Ephemeris,
    origin: &RenderOrigin,
    sim_time: f64,
    gizmos: &mut Gizmos,
    body_pos_now_cache: &mut HashMap<BodyId, bevy::math::DVec3>,
) {
    let samples = &segment.samples;
    if samples.is_empty() {
        return;
    }

    // Collect the render-space positions of the closed loop.
    let points: Vec<Vec3> = samples
        .iter()
        .map(|s| sample_render_pos_cached(s, system, ephemeris, origin, sim_time, body_pos_now_cache))
        .collect();

    // Use the dominant body color of the first sample; a stable orbit has a
    // single dominant body throughout.
    let dominant = samples[0].dominant_body;
    let [r, g, b] = system
        .bodies
        .get(dominant)
        .map(|bd| bd.color)
        .unwrap_or([1.0, 1.0, 1.0]);
    let color = Color::srgba(r, g, b, 1.0);

    // Draw the loop as a strip and close it back to the first point.
    gizmos.linestrip(points.iter().copied().chain(std::iter::once(points[0])), color);
    // No cone drawn for stable orbits — uncertainty has stopped growing.
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a sample's absolute heliocentric position into a Bevy render-space
/// Vec3.  The sample is expressed relative to the dominant body's position at
/// the sample time (preserving the clean elliptical shape), then placed at the
/// body's *current* position in origin-relative render space.
///
/// Caches the `query_body(id, sim_time)` result
/// per body — `sim_time` is constant for the entire frame so the lookup only
/// needs to happen once per distinct dominant body.
fn sample_render_pos_cached(
    sample: &TrajectorySample,
    system: &SolarSystemDefinition,
    ephemeris: &Ephemeris,
    origin: &RenderOrigin,
    sim_time: f64,
    body_pos_now_cache: &mut HashMap<BodyId, bevy::math::DVec3>,
) -> Vec3 {
    let body_count = system.bodies.len();
    let (body_pos_at_sample, body_pos_now) = if sample.dominant_body < body_count {
        let pos_now = *body_pos_now_cache
            .entry(sample.dominant_body)
            .or_insert_with(|| ephemeris.query_body(sample.dominant_body, sim_time).position);
        (
            ephemeris.query_body(sample.dominant_body, sample.time).position,
            pos_now,
        )
    } else {
        (bevy::math::DVec3::ZERO, bevy::math::DVec3::ZERO)
    };

    let rel = sample.position - body_pos_at_sample;
    let body_render_pos = (body_pos_now - origin.position) * RENDER_SCALE;
    (rel * RENDER_SCALE + body_render_pos).as_vec3()
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

/// Color for the trajectory line at sample `this`, blending toward `other`'s
/// body color using `perturbation_ratio` as the transition signal.
///
/// When both samples share the same dominant body the color is pure.  When
/// they differ, we linearly interpolate between the two body colors weighted
/// by `perturbation_ratio` so the transition is smooth near SOI boundaries.
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
    // `perturbation_ratio` grows toward 1 as the body approaches a SOI
    // boundary; use it to smoothly blend into the next body's color.
    let t = this.perturbation_ratio.clamp(0.0, 1.0) as f32;
    Color::srgba(
        r0 + (r1 - r0) * t,
        g0 + (g1 - g0) * t,
        b0 + (b1 - b0) * t,
        alpha,
    )
}

/// Return the `[r, g, b]` color of a body, defaulting to white if the id is
/// out of range.
fn body_color(id: BodyId, system: &SolarSystemDefinition) -> [f32; 3] {
    system
        .bodies
        .get(id)
        .map(|bd| bd.color)
        .unwrap_or([1.0, 1.0, 1.0])
}

/// Build a `Color::srgba` from a `[r, g, b]` array and a separate alpha.
fn color_with_alpha([r, g, b]: [f32; 3], alpha: f32) -> Color {
    Color::srgba(r, g, b, alpha)
}

/// Compute a render-space unit vector perpendicular to a trajectory sample's
/// velocity.  Uses `cross(velocity_dir, Y)` and falls back to `X` when the
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

/// Draw a ghost body sphere gizmo at the dominant body's position when the
/// sample has a non-star dominant body with meaningful gravitational influence.
fn maybe_draw_ghost(
    sample: &TrajectorySample,
    system: &SolarSystemDefinition,
    ephemeris: &Ephemeris,
    origin: &RenderOrigin,
    sim_time: f64,
    gizmos: &mut Gizmos,
    body_pos_now_cache: &mut HashMap<BodyId, bevy::math::DVec3>,
) {
    if sample.perturbation_ratio < GHOST_BODY_PERTURBATION_THRESHOLD {
        return;
    }

    let Some(body_def) = system.bodies.get(sample.dominant_body) else {
        return;
    };

    // Stars are always visible in the scene; no ghost needed.
    if body_def.kind == BodyKind::Star {
        return;
    }

    // Place the ghost at the dominant body's current position (origin-relative)
    // so it aligns with the body's actual scene entity.
    let body_pos_now = *body_pos_now_cache
        .entry(sample.dominant_body)
        .or_insert_with(|| ephemeris.query_body(sample.dominant_body, sim_time).position);
    let ghost_pos = ((body_pos_now - origin.position) * RENDER_SCALE).as_vec3();

    let [r, g, b] = body_def.color;
    let alpha = (sample.perturbation_ratio as f32 * 0.6).clamp(0.1, 0.5);
    let color = Color::srgba(r, g, b, alpha);

    // Draw a wireframe sphere as a lightweight ghost proxy.
    // TODO: Replace with a billboard label rendering the ghost annotation.
    gizmos.sphere(
        Isometry3d::from_translation(Vec3A::from(ghost_pos)),
        GHOST_SPHERE_RADIUS,
        color,
    );

    // Ghost label text (e.g. "Thalos T+2d 6h") would be rendered here using
    // a world-space UI text component.  Skipped for MVP — add when the UI
    // layer supports world-space labels.
}
