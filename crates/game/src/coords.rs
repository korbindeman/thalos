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

use bevy::camera::visibility::RenderLayers;
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

/// `RenderLayers` containing only the map layer.
pub fn map_layer() -> RenderLayers {
    RenderLayers::layer(MAP_LAYER)
}

/// `RenderLayers` containing only the ship layer.
pub fn ship_layer() -> RenderLayers {
    RenderLayers::layer(SHIP_LAYER)
}

/// `RenderLayers` containing both view layers, for entities (sky, lens
/// flare) that should appear in both views regardless of which camera
/// is active.
pub fn both_view_layers() -> RenderLayers {
    RenderLayers::from_layers(&[MAP_LAYER, SHIP_LAYER])
}

/// Metres → render-units scale factor. Mutated on view change; read by
/// every system that needs to place something in the scene.
///
/// **Deprecated:** systems should use [`MAP_SCALE`] / [`SHIP_SCALE`]
/// directly along with their target render layer. This resource exists
/// during the migration to per-view fixed-scale systems.
#[derive(Resource, Debug, Clone, Copy)]
pub struct WorldScale(pub f64);

impl Default for WorldScale {
    fn default() -> Self {
        Self(MAP_SCALE)
    }
}

impl WorldScale {
    /// Map view: 1 render unit = 1,000 km.
    pub const MAP: f64 = MAP_SCALE;
    /// Ship view: 1 render unit = 1 m.
    pub const SHIP: f64 = SHIP_SCALE;

    #[inline]
    pub fn get(&self) -> f64 {
        self.0
    }

    /// Metres-per-render-unit — used by shaders that bake world-unit
    /// params (atmosphere scale heights, ring radii).
    #[inline]
    pub fn meters_per_render_unit(&self) -> f32 {
        (1.0 / self.0) as f32
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
///   render_pos = ((sample.pos − sample.ref_pos) + pin(anchor) − origin) · scale
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
    scale: &WorldScale,
) -> Vec3 {
    let pin = pin_for(sample.anchor_body);
    let rel = sample.position - sample.ref_pos;
    ((rel + pin - origin.position) * scale.0).as_vec3()
}
