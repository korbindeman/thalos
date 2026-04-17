//! Shared coordinate-system helpers for the game crate.
//!
//! Physics uses heliocentric inertial (ecliptic XZ, Y up), metres, f64.
//! Rendering uses origin-relative scaled coordinates, render units, f32.
//! 1 render unit = 1 / RENDER_SCALE metres = 1,000 km.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics::types::{BodyId, TrajectorySample};

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

/// Convert a trajectory sample to render space using per-leg anchor locking:
///
/// ```text
///   render_pos = (sample.pos − sample.ref_pos) + pin(anchor)
/// ```
///
/// `sample.ref_pos` is the anchor body's position at sample time; `pin`
/// is the world-space position the leg is drawn around — typically the
/// anchor body's current position, or the anchor's ghost position when
/// a ghost exists (see `FlightPlanView::pin_for_body`). This keeps the
/// encounter's trajectory and its ghost mesh coincident in world space.
#[inline]
pub fn sample_render_pos(
    sample: &TrajectorySample,
    pin_for: impl Fn(BodyId) -> DVec3,
    origin: &RenderOrigin,
) -> Vec3 {
    let pin = pin_for(sample.anchor_body);
    let rel = sample.position - sample.ref_pos;
    ((rel + pin - origin.position) * RENDER_SCALE).as_vec3()
}
