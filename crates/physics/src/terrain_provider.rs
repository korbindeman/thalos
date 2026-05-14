//! Pluggable surface-elevation source for terrain-aware propagation.
//!
//! [`TerrainProvider`] mirrors the [`crate::body_trajectory_provider::BodyTrajectoryProvider`]
//! pattern: a trait the [`crate::ship_propagator::ShipPropagator`] queries
//! during collision detection, plus a concrete shared registry the game
//! crate updates as planet surfaces finish baking. Live `Simulation::step`
//! and trajectory prediction share the same propagator instance, so they
//! collide against the same surface — matching the "one propagator
//! everywhere" invariant in CLAUDE.md.
//!
//! For tests and headless builds with no surface data, [`FlatTerrain`]
//! reports zero elevation everywhere; the propagator falls back to mean
//! radius and behaves as before.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use glam::{DVec3, Vec3};
use thalos_terrain_gen::{Cubemap, PlanetSurface, cubemap::dir_to_face_uv};

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

/// Thread-safe registry of baked planet surfaces, keyed by [`BodyId`]. The
/// game crate holds one of these, inserts an entry per body as its surface
/// finishes baking, and hands the same handle to the propagator at
/// construction time so prediction and live propagation see the same data.
///
/// Reads are taken behind an `RwLock`; the propagator's hot path runs
/// hundreds of reads per frame, which is well below the cost of any
/// realistic write rate (surfaces are inserted once at startup and only
/// re-inserted on rare hot-reload).
#[derive(Default, Clone)]
pub struct SharedTerrainRegistry {
    inner: Arc<RwLock<HashMap<BodyId, Arc<PlanetSurface>>>>,
}

impl SharedTerrainRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, body: BodyId, surface: Arc<PlanetSurface>) {
        self.inner.write().unwrap().insert(body, surface);
    }

    pub fn contains(&self, body: BodyId) -> bool {
        self.inner.read().unwrap().contains_key(&body)
    }
}

impl TerrainProvider for SharedTerrainRegistry {
    fn surface_elevation_m(&self, body: BodyId, dir_body: DVec3) -> f64 {
        let guard = self.inner.read().unwrap();
        let Some(surface) = guard.get(&body) else {
            return 0.0;
        };
        let dir = dir_body.normalize_or_zero();
        if dir.length_squared() < 0.5 {
            return 0.0;
        }
        sample_height_cubemap_m(
            &surface.static_surface.height_cubemap,
            dir.as_vec3(),
            surface.static_surface.height_range,
        ) as f64
    }

    fn max_elevation_m(&self, body: BodyId) -> f64 {
        let guard = self.inner.read().unwrap();
        guard
            .get(&body)
            .map(|s| s.static_surface.height_range as f64)
            .unwrap_or(0.0)
    }
}

fn sample_height_cubemap_m(cubemap: &Cubemap<u16>, dir: Vec3, range_m: f32) -> f32 {
    let (face, u, v) = dir_to_face_uv(dir);
    let res = cubemap.resolution();
    let x = ((u * res as f32) as u32).min(res - 1);
    let y = ((v * res as f32) as u32).min(res - 1);
    let texel = cubemap.get(face, x, y);
    ((texel as f32 / u16::MAX as f32) * 2.0 - 1.0) * range_m
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

    #[test]
    fn shared_registry_reports_zero_for_unknown_body() {
        let registry = SharedTerrainRegistry::new();
        assert_eq!(registry.surface_elevation_m(0, DVec3::X), 0.0);
        assert_eq!(registry.max_elevation_m(0), 0.0);
        assert!(!registry.contains(0));
    }
}
