//! Pluggable surface-elevation source for terrain-aware propagation.
//!
//! [`TerrainProvider`] mirrors the [`crate::body_trajectory_provider::BodyTrajectoryProvider`]
//! pattern: a trait the [`crate::ship_propagator::ShipPropagator`] queries
//! during collision detection. Live `Simulation::step` and trajectory
//! prediction share the same propagator instance, so they collide against the
//! same surface — matching the "one propagator everywhere" invariant in
//! CLAUDE.md.
//!
//! The concrete `PlanetSurface`-backed implementation lives in the game crate
//! (it needs `thalos_terrain` runtime data); this crate stays free of a
//! terrain dependency and ships only the trait plus a flat fallback. For tests
//! and headless builds with no surface data, [`FlatTerrain`] reports zero
//! elevation everywhere; the propagator falls back to mean radius.

use glam::DVec3;

use crate::types::BodyId;

/// Surface-elevation source. Lookups are by [`BodyId`] and a unit direction
/// in the body's body-fixed frame (the same frame the rendered cubemap is
/// authored in).
pub trait TerrainProvider: Send + Sync {
    /// Elevation above mean radius in metres, in the direction `dir_body`.
    /// Implementations with no data for a body must return 0.0; the caller
    /// then treats the surface as a sphere of `radius_m`.
    fn surface_elevation_m(&self, body: BodyId, dir_body: DVec3) -> f64;

    /// Conservative upper bound on the absolute elevation for `body` in
    /// metres. The propagator uses this to fast-path skip per-direction
    /// queries when the ship is comfortably above the highest possible
    /// terrain. Returns 0.0 when no data is available.
    fn max_elevation_m(&self, body: BodyId) -> f64;
}

/// Spherical-surface terrain provider: zero elevation everywhere, so the
/// propagator collides against `radius_m` exactly like the pre-terrain
/// behaviour. Used as the default until a real provider is plugged in, and
/// in tests that don't need surface data.
#[derive(Debug, Default, Clone, Copy)]
pub struct FlatTerrain;

impl TerrainProvider for FlatTerrain {
    fn surface_elevation_m(&self, _body: BodyId, _dir_body: DVec3) -> f64 {
        0.0
    }

    fn max_elevation_m(&self, _body: BodyId) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_terrain_reports_zero_everywhere() {
        let t = FlatTerrain;
        assert_eq!(t.surface_elevation_m(0, DVec3::X), 0.0);
        assert_eq!(t.surface_elevation_m(7, -DVec3::Y), 0.0);
        assert_eq!(t.max_elevation_m(0), 0.0);
    }
}
