//! Shared coordinate-system helpers for the game crate.
//!
//! Physics uses heliocentric inertial (ecliptic XZ, Y up), metres, f64.
//! Rendering uses origin-relative scaled coordinates, render units, f32.
//! 1 render unit = 1 / RENDER_SCALE metres = 1,000 km.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics::types::{BodyKind, BodyState, SolarSystemDefinition, TrajectorySample};

/// Metres → render units.  1 render unit = 1,000 km.
pub const RENDER_SCALE: f64 = 1e-6;

/// The physics-space position (metres, f64) that maps to the render-space
/// origin.  Updated every frame to the camera focus body's position so that
/// objects near the camera always have small render-space coordinates,
/// preserving f32 precision at any zoom level.
#[derive(Resource, Default)]
pub struct RenderOrigin {
    pub position: DVec3,
}

/// Convert a physics DVec3 (metres, f64) to a Bevy Vec3 (render units, f32).
#[inline]
pub fn to_render_pos(v: DVec3) -> Vec3 {
    (v * RENDER_SCALE).as_vec3()
}

/// Convert a trajectory sample to render space by expressing it relative to
/// its sample-time anchor and then placing that body-relative offset at a
/// pin position in heliocentric space. Pin is precomputed per run of samples
/// sharing an anchor — see [`compute_segment_pins`].
pub fn sample_render_pos(sample: &TrajectorySample, pin: DVec3, origin: &RenderOrigin) -> Vec3 {
    let rel = sample.position - sample.anchor_body_pos;
    let pin_render = (pin - origin.position) * RENDER_SCALE;
    (rel * RENDER_SCALE + pin_render).as_vec3()
}

/// Compute per-sample pin positions for a segment.
///
/// Samples are grouped into contiguous runs sharing an anchor body. Each run
/// gets one pin that is reused for every sample in the run:
///
/// - **Star anchor**: pin = star current position (effectively heliocentric).
/// - **Live run**: the first run of the first segment represents the ship's
///   current location, so its pin = anchor body current position. This keeps
///   the live orbit glued to where the body actually appears on screen.
/// - **Future encounter**: pin = anchor body position at the run's periapsis
///   (sample minimising `|position - anchor_body_pos|`). This places the
///   encounter loop around a *ghost* body at the position the anchor will
///   have at closest approach, instead of on top of its current position.
pub fn compute_segment_pins(
    samples: &[TrajectorySample],
    body_states: &[BodyState],
    system: &SolarSystemDefinition,
    is_first_segment: bool,
) -> Vec<DVec3> {
    let mut pins = vec![DVec3::ZERO; samples.len()];
    let mut i = 0;
    let mut run_index = 0;
    while i < samples.len() {
        let anchor = samples[i].anchor_body;
        let mut j = i + 1;
        while j < samples.len() && samples[j].anchor_body == anchor {
            j += 1;
        }
        let run = &samples[i..j];
        let is_live_run = is_first_segment && run_index == 0;
        let pin = run_pin(run, anchor, body_states, system, is_live_run);
        for slot in &mut pins[i..j] {
            *slot = pin;
        }
        i = j;
        run_index += 1;
    }
    pins
}

fn run_pin(
    run: &[TrajectorySample],
    anchor: thalos_physics::types::BodyId,
    body_states: &[BodyState],
    system: &SolarSystemDefinition,
    is_live: bool,
) -> DVec3 {
    let body_now = body_states
        .get(anchor)
        .map(|bs| bs.position)
        .unwrap_or(DVec3::ZERO);

    let is_star = system
        .bodies
        .get(anchor)
        .map(|b| b.kind == BodyKind::Star)
        .unwrap_or(false);
    if is_star || is_live {
        return body_now;
    }

    // Future encounter: pin = anchor body position at run periapsis.
    let mut best_d2 = f64::MAX;
    let mut best_pin = body_now;
    for s in run {
        let d2 = (s.position - s.anchor_body_pos).length_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best_pin = s.anchor_body_pos;
        }
    }
    best_pin
}
