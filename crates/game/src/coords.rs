//! Shared coordinate-system helpers for the game crate.
//!
//! Physics uses heliocentric inertial (ecliptic XZ, Y up), metres, f64.
//! Rendering uses origin-relative scaled coordinates, render units, f32.
//!
//! Map and ship view live on **separate render layers**, each with a
//! fixed compile-time scale:
//!
//! - **Map view** ([`MAP_LAYER`], [`MAP_SCALE`] = 1e-6): 1 render unit
//!   = 1,000 km. Keeps solar-system distances inside f32 range at the
//!   cost of making metre-sized objects microscopic.
//! - **Ship view** ([`SHIP_LAYER`], [`SHIP_SCALE`] = 1.0): 1 render unit
//!   = 1 m. Ship parts and nearby bodies render at physical size; distant
//!   bodies sit far out in world space but Bevy's reverse-Z depth handles
//!   the range.
//!
//! Each view owns its own camera; transform systems target one layer at
//! a time and bake the corresponding scale as a const so there is no
//! cross-frame inconsistency.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics::types::{BodyId, TrajectorySample};

/// Map view: 1 render unit = 1,000 km.
pub const MAP_SCALE: f64 = 1.0e-6;

/// Ship view: 1 render unit = 1 m.
pub const SHIP_SCALE: f64 = 1.0;

/// Render layer for map-view entities (orbit-scale).
pub const MAP_LAYER: usize = 1;

/// Render layer for ship-view entities (metre-scale).
pub const SHIP_LAYER: usize = 2;

/// Metres → render-units scale factor for **map-view** systems
/// (orbit trails, maneuver UI, body parent transforms). Always
/// [`MAP_SCALE`]; the field is kept as a resource only so existing
/// consumers don't all need to migrate to a `const` reference at once.
///
/// Ship-view systems should use [`SHIP_SCALE`] directly.
#[derive(Resource, Debug, Clone, Copy)]
pub struct WorldScale(pub f64);

impl Default for WorldScale {
    fn default() -> Self {
        Self(MAP_SCALE)
    }
}

/// The physics-space position (metres, f64) that maps to the render-space
/// origin. Updated every frame to the camera focus body's (or player
/// ship's) position so that objects near the camera always have small
/// render-space coordinates, preserving f32 precision.
#[derive(Resource, Default)]
pub struct RenderOrigin {
    pub position: DVec3,
}

/// The body whose frame the trajectory and ghost system are conceptually
/// drawn in. Distinct from [`RenderOrigin`] so that camera-following
/// ("origin tracks ship") is decoupled from frame semantics ("trajectory
/// is in Mira's frame while ship is in Mira's SOI").
///
/// Resolution rules:
/// - Camera target is a celestial body → `focus_body = body.id`
/// - Camera target is a ghost body → `focus_body = ghost.body_id`
/// - Camera target is the player ship → `focus_body = ship's current SOI body`
/// - No camera target → `focus_body = 0` (the star)
///
/// Consumers:
/// - `FlightPlanView::rebuild` reads this to suppress focus-body ghosts
///   (so the focus body sits at the trajectory's center, not at a
///   parent-anchored projection).
/// - `FlightPlanView::pin_for_body` returns the focus body's current
///   heliocentric position when asked for it, grounding ghost
///   parent-chain recursion at the focus.
#[derive(Resource, Debug, Clone, Copy)]
pub struct RenderFrame {
    pub focus_body: BodyId,
}

impl Default for RenderFrame {
    fn default() -> Self {
        Self { focus_body: 0 }
    }
}

/// Convert a physics DVec3 (metres, f64) to a Bevy Vec3 (render units,
/// f32) at the current world scale.
#[inline]
pub fn to_render_pos(v: DVec3, scale: &WorldScale) -> Vec3 {
    (v * scale.0).as_vec3()
}

/// Convert a trajectory sample to render space using per-leg anchor
/// locking:
///
/// ```text
///   render_pos = ((sample.pos − sample.ref_pos) + pin − origin) · scale
/// ```
///
/// `sample.ref_pos` is the anchor body's position at sample time; `pin`
/// is the world-space position the leg is drawn around — typically the
/// anchor body's current position, or the anchor's ghost position when
/// a ghost exists (see `FlightPlanView::pin_for_body`). This keeps the
/// encounter's trajectory and its ghost mesh coincident in world space.
///
/// `pin` is constant within a leg (samples within a leg share an
/// anchor after the per-leg relock in `propagate_flight_plan`), so
/// callers compute it once per leg/segment and pass the same value
/// for every sample in that leg.
#[inline]
pub fn sample_render_pos(
    sample: &TrajectorySample,
    pin: DVec3,
    origin: &RenderOrigin,
    scale: &WorldScale,
) -> Vec3 {
    let rel = sample.position - sample.ref_pos;
    ((rel + pin - origin.position) * scale.0).as_vec3()
}
