//! Shared coordinate-system helpers for the game crate.
//!
//! Physics uses heliocentric inertial (ecliptic XZ, Y up), metres, f64.
//! Rendering uses origin-relative scaled coordinates, render units, f32.
//! 1 render unit = 1 / RENDER_SCALE metres = 1,000 km.

use bevy::prelude::*;
use thalos_physics::types::{BodyState, TrajectorySample};

/// Metres → render units.  1 render unit = 1,000 km.
pub const RENDER_SCALE: f64 = 1e-6;

/// The physics-space position (metres, f64) that maps to the render-space
/// origin.  Updated every frame to the camera focus body's position so that
/// objects near the camera always have small render-space coordinates,
/// preserving f32 precision at any zoom level.
#[derive(Resource, Default)]
pub struct RenderOrigin {
    pub position: bevy::math::DVec3,
}

/// Convert a physics DVec3 (metres, f64) to a Bevy Vec3 (render units, f32).
#[inline]
pub fn to_render_pos(v: bevy::math::DVec3) -> Vec3 {
    (v * RENDER_SCALE).as_vec3()
}

/// Convert a trajectory sample's absolute heliocentric position into a Bevy
/// render-space Vec3.  The sample is expressed relative to the dominant body's
/// position at the sample time (preserving the clean elliptical shape), then
/// placed at the body's *current* position in origin-relative render space.
pub fn sample_render_pos(
    sample: &TrajectorySample,
    body_states: &[BodyState],
    origin: &RenderOrigin,
) -> Vec3 {
    let (body_pos_at_sample, body_pos_now) = if let Some(bs) = body_states.get(sample.dominant_body)
    {
        (sample.dominant_body_pos, bs.position)
    } else {
        (bevy::math::DVec3::ZERO, bevy::math::DVec3::ZERO)
    };

    let rel = sample.position - body_pos_at_sample;
    let body_render_pos = (body_pos_now - origin.position) * RENDER_SCALE;
    (rel * RENDER_SCALE + body_render_pos).as_vec3()
}
