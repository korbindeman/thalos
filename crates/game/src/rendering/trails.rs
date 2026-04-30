//! Orbit-trail data + per-frame gizmo drawing. Trails are recomputed
//! periodically (`ORBIT_TRAIL_RECOMPUTE_INTERVAL`) and on view-scale
//! change; the draw pass fades them as the camera zooms inside the
//! parent body's visible billboard so close-up framing doesn't show
//! self-clipping orbit rings.

use bevy::prelude::*;

use super::screen_marker_radius;
use super::types::{CelestialBody, FrameBodyStates, SimulationState};
use crate::camera::{ActiveCamera, CameraFocus, CameraFocusTarget, OrbitCamera};
use crate::coords::{RenderOrigin, WorldScale, to_render_pos};
use crate::flight_plan_view::GhostBody;

/// Points sampled along each orbit for gizmo line drawing.
const ORBIT_SAMPLES: usize = 256;

/// Minimum sim-time advance (seconds) before orbit trails are recomputed.
/// Trails are also recomputed on the first frame.
const ORBIT_TRAIL_RECOMPUTE_INTERVAL: f64 = 3600.0;

/// Distance-to-parent / camera-distance ratio at which orbit trails start fading.
const ORBIT_FADE_START: f64 = 20.0;
/// Ratio at which orbit trails are fully hidden.
const ORBIT_FADE_END: f64 = 100.0;

/// Aggressive fade for the focus body's own orbit and its siblings (bodies
/// sharing the same parent as the focus). Expressed as
/// `focus_orbit_radius / cam_dist` — when the camera is well inside the
/// focus body's orbit around its parent, these lines clutter the view.
const SIBLING_FADE_START: f64 = 3.0;
const SIBLING_FADE_END: f64 = 10.0;

/// Stores orbit trail render data for each body. Recomputed periodically
/// as sim time advances so trails reflect the forward trajectory.
#[derive(Resource)]
pub(super) struct OrbitLines {
    lines: Vec<Option<OrbitLine>>,
    /// Sim time when trails were last computed.
    last_compute_time: f64,
}

struct OrbitLine {
    points: Vec<Vec3>,
    color: Color,
    parent_id: usize,
    /// Max distance from the parent across all sample points, in
    /// render units. Used to compare the orbit's screen-space size
    /// against the parent body's visible billboard so child orbits
    /// can be hidden once they converge with the parent's icon.
    max_radius: f32,
}

/// Convert an sRGB component (0..1) to linear light.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Recompute orbit trails when sim time has advanced enough, or on first frame.
pub(super) fn recompute_orbit_trails(
    mut commands: Commands,
    sim: Res<SimulationState>,
    scale: Res<WorldScale>,
    existing: Option<Res<OrbitLines>>,
) {
    let sim_time = sim.simulation.sim_time();

    // Force a recompute on scale change so cached render-space points
    // don't linger at the old scale after a view toggle.
    let scale_changed = scale.is_changed();
    if let Some(ref orbit_lines) = existing
        && !scale_changed
    {
        let elapsed = sim_time - orbit_lines.last_compute_time;
        if elapsed < ORBIT_TRAIL_RECOMPUTE_INTERVAL {
            return;
        }
    }

    let trails = sim.simulation.body_orbit_trails(ORBIT_SAMPLES);
    let bodies = sim.simulation.bodies();

    let lines: Vec<Option<OrbitLine>> = trails
        .into_iter()
        .enumerate()
        .map(|(i, trail)| {
            let trail = trail?;
            let body = &bodies[i];
            let [r, g, b] = body.color;
            let orbit_color = Color::linear_rgba(
                srgb_to_linear(r) * 0.4,
                srgb_to_linear(g) * 0.4,
                srgb_to_linear(b) * 0.4,
                0.6,
            );
            let points: Vec<Vec3> = trail
                .points
                .iter()
                .map(|p| to_render_pos(*p, &scale))
                .collect();
            let max_radius = points
                .iter()
                .map(|p| p.length())
                .reduce(f32::max)
                .unwrap_or(0.0);
            Some(OrbitLine {
                points,
                color: orbit_color,
                parent_id: trail.parent_id,
                max_radius,
            })
        })
        .collect();

    commands.insert_resource(OrbitLines {
        lines,
        last_compute_time: sim_time,
    });
}

pub(super) fn draw_orbits(
    mut gizmos: Gizmos,
    orbit_lines: Option<Res<OrbitLines>>,
    cache: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    scale: Res<WorldScale>,
    focus: Res<CameraFocus>,
    ghosts: Query<&GhostBody>,
    bodies: Query<(&CelestialBody, &Transform)>,
    sim: Option<Res<SimulationState>>,
    camera_query: Query<&Transform, (With<ActiveCamera>, With<OrbitCamera>)>,
) {
    let Some(orbit_lines) = orbit_lines else {
        return;
    };
    let Some(ref states) = cache.states else {
        return;
    };
    let Some(sim) = sim else {
        return;
    };

    // Focus body id and its parent id, if any.
    let focus_body_id = match focus.target {
        CameraFocusTarget::Body(body_id) => Some(body_id),
        CameraFocusTarget::Ghost(entity) => ghosts.get(entity).ok().map(|ghost| ghost.body_id),
        CameraFocusTarget::Ship => Some(crate::camera::find_reference_body(
            sim.simulation.ship_state().position,
            sim.simulation.bodies(),
            states,
        )),
        _ => None,
    };
    let focus_parent_id = focus_body_id.and_then(|id| sim.simulation.bodies()[id].parent);

    // Determine the camera focus position in metres.
    let focus_pos = match focus.target {
        CameraFocusTarget::Ship => sim.simulation.ship_state().position,
        CameraFocusTarget::Ghost(_) => origin.position,
        _ => focus_body_id
            .and_then(|id| states.get(id))
            .map(|s| s.position)
            .unwrap_or(bevy::math::DVec3::ZERO),
    };

    let cam_dist = focus.distance;

    // Per-body visible billboard radius in render units. For non-stars
    // this is the larger of the impostor's render radius and the
    // icon's render-space radius (sized from that body's own camera
    // distance, i.e. a constant on-screen size). Stars never render as
    // an icon, so we use their render radius unconditionally.
    //
    // We hide a child's orbit once its ring is smaller than the
    // parent's visible billboard — at that point the orbit ellipse
    // visually merges into the parent, regardless of camera distance,
    // and tuning gizmo `depth_bias` can't disentangle them. Tying the
    // cull to *visible billboard* size (rather than icon-mode alone)
    // means the rule kicks in smoothly across the icon→impostor
    // crossfade, instead of popping moon orbits in/out at the
    // threshold where the parent's icon disappears.
    let parent_visual_radius: Vec<f32> = if let Ok(cam_tf) = camera_query.single() {
        let cam_pos = cam_tf.translation;
        let mut radii = vec![0.0f32; sim.simulation.bodies().len()];
        for (body, body_tf) in bodies.iter() {
            let visual_radius = if body.is_star {
                body.render_radius
            } else {
                let icon_radius = screen_marker_radius(body_tf.translation, cam_pos);
                body.render_radius.max(icon_radius)
            };
            radii[body.body_id] = visual_radius;
        }
        radii
    } else {
        Vec::new()
    };

    for (i, line) in orbit_lines.lines.iter().enumerate() {
        let Some(line) = line else { continue };

        let parent_visual = parent_visual_radius
            .get(line.parent_id)
            .copied()
            .unwrap_or(0.0);
        if line.max_radius < parent_visual {
            continue;
        }

        let parent_pos_m = states
            .get(line.parent_id)
            .map(|s| s.position)
            .unwrap_or(bevy::math::DVec3::ZERO);
        let dist_to_parent = (parent_pos_m - focus_pos).length();
        let ratio = dist_to_parent / cam_dist;

        // Focus body's own orbit, or a sibling (same parent). Apply an
        // aggressive fade so close-up views aren't cluttered by the orbit
        // ring cutting through the focus body.
        let is_self_or_sibling = Some(i) == focus_body_id
            || (focus_parent_id.is_some() && Some(line.parent_id) == focus_parent_id);

        let (fade_start, fade_end) = if is_self_or_sibling {
            (SIBLING_FADE_START, SIBLING_FADE_END)
        } else {
            (ORBIT_FADE_START, ORBIT_FADE_END)
        };

        if ratio > fade_end {
            continue;
        }

        let parent_render_pos = to_render_pos(parent_pos_m - origin.position, &scale);

        if ratio > fade_start {
            let t = (ratio - fade_start) / (fade_end - fade_start);
            let alpha = (1.0 - t) as f32;
            let faded = line.color.with_alpha(line.color.alpha() * alpha);
            gizmos.linestrip(line.points.iter().map(|p| *p + parent_render_pos), faded);
        } else {
            gizmos.linestrip(
                line.points.iter().map(|p| *p + parent_render_pos),
                line.color,
            );
        }
    }
}
