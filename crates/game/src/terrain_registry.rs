//! `ProceduralSurface`-backed terrain-elevation provider for the propagator.
//!
//! This is the concrete [`TerrainProvider`] the propagator queries during
//! collision detection. It lives in the game crate (not
//! `thalos_physics_canonical`) because it depends on `thalos_terrain` runtime
//! data — keeping the physics crate free of a terrain dependency. The game
//! holds one registry, inserts an entry per procedural body when it spawns, and
//! hands the same handle to the propagator at construction time so prediction
//! and live propagation see the same surface.
//!
//! Reads are taken behind an `RwLock`; the propagator's hot path runs hundreds
//! of reads per frame, well below the cost of any realistic write rate
//! (surfaces are inserted once at startup).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use bevy::math::DVec3;
use thalos_physics_canonical::terrain_provider::TerrainProvider;
use thalos_terrain::{ProceduralSurface, SurfaceQuery};
use thalos_world::BodyId;

/// Coarse LOD (metres per sample) for orbital collision queries. The propagator
/// only needs "does this orbit dip below the ground", so a coarse sample —
/// skipping the fine procedural octaves — is both sufficient and cheap.
const PROPAGATOR_LOD_M: f32 = 1000.0;

/// Thread-safe registry of procedural surfaces, keyed by [`BodyId`].
#[derive(Default, Clone)]
pub struct SharedTerrainRegistry {
    inner: Arc<RwLock<HashMap<BodyId, ProceduralSurface>>>,
}

impl SharedTerrainRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, body: BodyId, surface: ProceduralSurface) {
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
        // Orbital collision is coarse; the f32 direction is adequate here
        // (sub-metre precision is irrelevant to "does the orbit hit ground").
        surface.sample_height_m(dir.as_vec3(), PROPAGATOR_LOD_M) as f64
    }

    fn max_elevation_m(&self, body: BodyId) -> f64 {
        let guard = self.inner.read().unwrap();
        guard
            .get(&body)
            .map(|s| s.height_range_m() as f64)
            .unwrap_or(0.0)
    }
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
