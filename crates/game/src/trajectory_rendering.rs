//! Trajectory line rendering for the predicted ship path.
//!
//! Each [`TrajectorySample`] carries an absolute heliocentric position, but we
//! render every segment *relative to its dominant body's position at the sample
//! time*.  This makes orbits appear as clean ellipses even as bodies move.
//!
//! Line segments are colored by dominant body.  At body transitions the two
//! adjacent colors are linearly blended using the sample's `perturbation_ratio`
//! as the interpolant, giving a smooth visual gradient at sphere-of-influence
//! boundaries.

use bevy::math::{Isometry3d, Vec3A};
use bevy::prelude::*;
use thalos_physics::trajectory::{TrajectorySegment, cone_width};
use thalos_physics::types::{BodyId, BodyKind, SolarSystemDefinition, TrajectorySample};

use crate::coords::{RENDER_SCALE, RenderOrigin, sample_render_pos};
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

/// Radius (render units) of the ghost body wireframe sphere gizmo.
const GHOST_SPHERE_RADIUS: f32 = 0.5;

use crate::rendering::SimulationState;

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
    mut gizmos: Gizmos,
) {
    let Some(sim) = sim else { return };
    let Some(prediction) = sim.simulation.prediction() else {
        return;
    };
    let Some(ref body_states) = cache.states else {
        return;
    };

    for (i, segment) in prediction.segments.iter().enumerate() {
        let is_ghost = i > 0;
        render_segment(segment, is_ghost, &sim.system, body_states, &origin, &mut gizmos);
    }
}

// ---------------------------------------------------------------------------
// Segment rendering
// ---------------------------------------------------------------------------

fn render_segment(
    segment: &TrajectorySegment,
    is_ghost: bool,
    system: &SolarSystemDefinition,
    body_states: &[thalos_physics::types::BodyState],
    origin: &RenderOrigin,
    gizmos: &mut Gizmos,
) {
    if segment.samples.is_empty() {
        return;
    }

    if segment.is_stable_orbit {
        render_stable_orbit_segment(segment, is_ghost, system, body_states, origin, gizmos);
        return;
    }

    render_open_samples(
        &segment.samples,
        is_ghost,
        system,
        body_states,
        origin,
        gizmos,
    );
}

fn render_open_samples(
    samples: &[TrajectorySample],
    is_ghost: bool,
    system: &SolarSystemDefinition,
    body_states: &[thalos_physics::types::BodyState],
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

        let p_a = sample_render_pos(a, body_states, origin);
        let p_b = sample_render_pos(b, body_states, origin);

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

        maybe_draw_ghost(a, system, body_states, origin, gizmos);
    }

    // Ghost check for the final sample.
    if let Some(last) = samples.last()
        && fade_alpha(last) > 0.0
    {
        maybe_draw_ghost(last, system, body_states, origin, gizmos);
    }
}

// ---------------------------------------------------------------------------
// Stable orbit rendering
// ---------------------------------------------------------------------------

fn render_stable_orbit_segment(
    segment: &TrajectorySegment,
    is_ghost: bool,
    system: &SolarSystemDefinition,
    body_states: &[thalos_physics::types::BodyState],
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
            is_ghost,
            system,
            body_states,
            origin,
            gizmos,
        );
    }

    render_stable_orbit(
        &segment.samples[loop_start..],
        is_ghost,
        system,
        body_states,
        origin,
        gizmos,
    );
}

fn render_stable_orbit(
    samples: &[TrajectorySample],
    is_ghost: bool,
    system: &SolarSystemDefinition,
    body_states: &[thalos_physics::types::BodyState],
    origin: &RenderOrigin,
    gizmos: &mut Gizmos,
) {
    if samples.len() < 2 {
        return;
    }

    let dominant = samples[0].dominant_body;
    let [r, g, b] = system
        .bodies
        .get(dominant)
        .map(|bd| bd.color)
        .unwrap_or([1.0, 1.0, 1.0]);
    let color = ghost_adjust(Color::srgba(r, g, b, 1.0), is_ghost);

    let first = sample_render_pos(&samples[0], body_states, origin);
    gizmos.linestrip(
        samples
            .iter()
            .map(|s| sample_render_pos(s, body_states, origin))
            .chain(std::iter::once(first)),
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

/// Draw a ghost body sphere gizmo at the dominant body's position when the
/// sample has a non-star dominant body with meaningful gravitational influence.
fn maybe_draw_ghost(
    sample: &TrajectorySample,
    system: &SolarSystemDefinition,
    body_states: &[thalos_physics::types::BodyState],
    origin: &RenderOrigin,
    gizmos: &mut Gizmos,
) {
    if sample.perturbation_ratio < GHOST_BODY_PERTURBATION_THRESHOLD {
        return;
    }

    let Some(body_def) = system.bodies.get(sample.dominant_body) else {
        return;
    };

    if body_def.kind == BodyKind::Star {
        return;
    }

    let body_pos_now = body_states
        .get(sample.dominant_body)
        .map(|bs| bs.position)
        .unwrap_or(bevy::math::DVec3::ZERO);
    let ghost_pos = ((body_pos_now - origin.position) * RENDER_SCALE).as_vec3();

    let [r, g, b] = body_def.color;
    let alpha = (sample.perturbation_ratio as f32 * 0.6).clamp(0.1, 0.5);
    let color = Color::srgba(r, g, b, alpha);

    gizmos.sphere(
        Isometry3d::from_translation(Vec3A::from(ghost_pos)),
        GHOST_SPHERE_RADIUS,
        color,
    );
}
