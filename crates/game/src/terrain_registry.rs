//! `PlanetSurface`-backed terrain-elevation provider for the propagator.
//!
//! This is the concrete [`TerrainProvider`] the propagator queries during
//! collision detection. It lives in the game crate (not
//! `thalos_physics_canonical`) because it depends on `thalos_terrain` runtime
//! data — keeping the physics crate free of a terrain dependency. The game
//! holds one registry, inserts an entry per body as its surface finishes
//! baking, and hands the same handle to the propagator at construction time so
//! prediction and live propagation see the same data.
//!
//! Reads are taken behind an `RwLock`; the propagator's hot path runs hundreds
//! of reads per frame, well below the cost of any realistic write rate
//! (surfaces are inserted once at startup and only re-inserted on rare
//! hot-reload).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use bevy::math::{DVec3, Vec3};
use thalos_physics_canonical::terrain_provider::TerrainProvider;
use thalos_terrain::{Cubemap, PlanetSurface, cubemap::dir_to_face_uv};
use thalos_world::BodyId;

/// Thread-safe registry of baked planet surfaces, keyed by [`BodyId`].
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
    fn shared_registry_reports_zero_for_unknown_body() {
        let registry = SharedTerrainRegistry::new();
        assert_eq!(registry.surface_elevation_m(0, DVec3::X), 0.0);
        assert_eq!(registry.max_elevation_m(0), 0.0);
        assert!(!registry.contains(0));
    }
}
