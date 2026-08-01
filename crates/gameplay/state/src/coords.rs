//! Shared coordinate-system vocabulary for the running game.
//!
//! Physics uses heliocentric inertial (ecliptic XZ, Y up), metres, f64.
//! Map rendering uses origin-relative scaled coordinates, render units, f32.
//! Ship-view bodies and the player ship use BigSpace cells in metres.
//!
//! Map and ship view live on **separate render layers**, each with a
//! fixed compile-time scale:
//!
//! - **Map view** ([`MAP_LAYER`], [`MAP_SCALE`] = 1e-6): 1 render unit
//!   = 1,000 km. Keeps solar-system distances inside f32 range at the
//!   cost of making metre-sized objects microscopic.
//! - **Ship view** ([`SHIP_LAYER`], [`SHIP_SCALE`] = 1.0): 1 render unit
//!   = 1 m. Ship parts and nearby bodies render at physical size inside
//!   BigSpace cells.
//!
//! Each view owns its own camera; transform systems target one layer at
//! a time and bake the corresponding scale as a const so there is no
//! cross-frame inconsistency.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics_canonical::types::TrajectorySample;
use thalos_world::BodyId;

/// Map view: 1 render unit = 1,000 km.
pub const MAP_SCALE: f64 = 1.0e-6;

/// Ship view: 1 render unit = 1 m.
pub const SHIP_SCALE: f64 = 1.0;

/// Render layer for map-view entities (orbit-scale).
pub const MAP_LAYER: usize = 1;

/// Render layer for ship-view entities (metre-scale).
pub const SHIP_LAYER: usize = 2;

/// Render layer for the in-game shipyard editor's build scene (parts under
/// construction, editor lights, hangar backdrop). Only the editor camera
/// renders it, so the build world never bleeds into the flight or map views.
/// (3 = navball render target, 4 = nav attitude indicator.)
pub const EDITOR_LAYER: usize = 5;

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

/// The physics-space position (metres, f64) that maps to the map render-space
/// origin. Updated every frame to the camera focus body's (or player ship's)
/// position so map-view objects near the camera have small f32 coordinates.
///
/// **Map/orbit space only.** This tracks the camera *focus pivot*, not the view:
/// in flight it is the craft, so it drifts from the actual render frame by the
/// whole camera↔craft separation whenever the view leaves the craft (freecam,
/// god view). Metre-scale placement among big_space content — the sun-shadow
/// cascade cameras and anything else outside the hierarchy — must use
/// `RealSpaceOrigin` (the runtime's `rendering::real_space`) instead.
#[derive(Resource, Default)]
pub struct RenderOrigin {
    pub position: DVec3,
}

/// A focused future encounter marker. Stored separately from
/// [`RenderFrame::focus_body`] because a ghost is not the same frame as
/// the real body at `sim_time`; it is a body frame pinned to a specific
/// future epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderGhostFocus {
    pub body_id: BodyId,
    pub parent_id: BodyId,
    pub relative_position: DVec3,
    pub projection_epoch: f64,
    pub encounter_epoch: f64,
}

pub const GHOST_FOCUS_EPOCH_TOLERANCE_S: f64 = 1.0;

impl RenderGhostFocus {
    pub fn matches(self, body_id: BodyId, encounter_epoch: f64) -> bool {
        self.body_id == body_id
            && (self.encounter_epoch - encounter_epoch).abs() <= GHOST_FOCUS_EPOCH_TOLERANCE_S
    }
}

/// The body/frame the trajectory and ghost system are conceptually drawn
/// in. Distinct from [`RenderOrigin`] so that camera-following ("origin
/// tracks ship") is decoupled from frame semantics ("trajectory is in
/// Mira's frame while ship is in Mira's SOI").
///
/// Resolution rules:
/// - Camera target is a celestial body → `focus_body = body.id`
/// - Camera target is a ghost body → `focus_body = ghost.body_id`,
///   `focus_ghost = Some(...)`
/// - Camera target is the player ship → `focus_body = ship's current SOI body`
/// - No camera target → `focus_body = 0` (the star)
///
/// Consumers:
/// - `FlightPlanView::rebuild` reads this to preserve the selected
///   ghost encounter context across prediction rebuilds.
/// - `FlightPlanView::pin_for_body` uses it to decide whether the
///   focused body should resolve to the real body at `sim_time` or a
///   selected ghost at its future encounter epoch.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Default)]
pub struct RenderFrame {
    pub focus_body: BodyId,
    pub focus_ghost: Option<RenderGhostFocus>,
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

/// Render-space radius for a marker that should keep the same screen size as
/// the body icon billboards.
#[inline]
pub fn screen_marker_radius(world_pos: Vec3, camera_pos: Vec3) -> f32 {
    (world_pos - camera_pos).length().max(1.0) * SCREEN_MARKER_RADIUS
}

/// Radius of screen-stable marker billboards as a fraction of camera distance
/// (in render units). Bodies whose rendered sphere is smaller than this get
/// replaced by a fixed-size circle billboard, and map-view marker overlays
/// use the same value so every marker family has the same screen size.
pub const SCREEN_MARKER_RADIUS: f32 = 0.006;
