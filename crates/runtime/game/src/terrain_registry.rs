//! Canonical per-body terrain surfaces and the propagator projection.
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
use bevy::prelude::{Resource, Vec3};
use thalos_body_render::GpuAtlasMirrorHandle;
use thalos_physics_canonical::terrain_provider::TerrainProvider;
use thalos_terrain::{
    BodyArchetype, DynamicSurfaceState, PackageSurface, PlanetSurface, ProceduralSurface,
    SurfaceQuery, TerrainCompileContext, TerrainCompileOptions, TerrainConfig, cache,
    compile_dynamic_surface_layers, compile_tectonics_from_config, load_static_package,
};
use thalos_world::{BodyDefinition, BodyId, BodyKind};

/// Coarse LOD (metres per sample) for orbital collision queries. The propagator
/// only needs "does this orbit dip below the ground", so a coarse sample —
/// skipping the fine procedural octaves — is both sufficient and cheap.
const PROPAGATOR_LOD_M: f32 = 1000.0;

/// Runtime-only projection of renderer residency data. The canonical height
/// queries themselves remain in `thalos_physics_local::HeightSourceRegistry`
/// behind the renderer-independent `thalos_terrain::HeightSource` contract.
#[derive(Resource, Default, Clone)]
pub struct GpuHeightMirrorRegistry {
    gpu_mirrors: HashMap<BodyId, GpuAtlasMirrorHandle>,
}

impl GpuHeightMirrorRegistry {
    pub fn insert(&mut self, body_id: BodyId, mirror: GpuAtlasMirrorHandle) {
        self.gpu_mirrors.insert(body_id, mirror);
    }

    pub fn get(&self, body_id: BodyId) -> Option<GpuAtlasMirrorHandle> {
        self.gpu_mirrors.get(&body_id).cloned()
    }
}

/// One constructed surface per terrain-bearing body.
///
/// Every game consumer clones the same `Arc`: UDLOD, map terrain, impostor
/// projections, height sources, and the propagator. This is the canonical
/// N-body construction seam; consumers never select or instantiate a concrete
/// generator themselves.
#[derive(Resource, Default, Clone)]
pub struct BodySurfaceRegistry {
    surfaces: HashMap<BodyId, Arc<dyn SurfaceQuery>>,
    fingerprints: HashMap<BodyId, u64>,
    airless_landmarks: HashMap<BodyId, Vec<AirlessLandmark>>,
}

impl BodySurfaceRegistry {
    pub fn load(bodies: &[BodyDefinition], package_dir: &std::path::Path) -> Result<Self, String> {
        let mut registry = Self::default();
        for body in bodies.iter().filter(|body| body.terrain.is_some()) {
            let (surface, fingerprint): (Arc<dyn SurfaceQuery>, u64) = match &body.terrain {
                TerrainConfig::Feature(feature)
                    if feature.archetype == BodyArchetype::AirlessImpactMoon =>
                {
                    let context = terrain_context(body);
                    let options = TerrainCompileOptions::default();
                    let key = cache::terrain_cache_key(
                        &body.terrain,
                        body.tectonics.as_ref(),
                        &context,
                        options,
                    );
                    let path = cache::cache_path(package_dir, &body.name);
                    let package = load_static_package(&path, &body.name, key).map_err(|error| {
                        format!(
                            "{} requires an offline terrain package: {error}. Run `just bake {}`",
                            body.name, body.name
                        )
                    })?;
                    let package_fingerprint = package.manifest.artifact_fingerprint();
                    let craters = package.static_surface.craters.clone();
                    let dynamic_layers = compile_dynamic_surface_layers(&body.terrain, &context)
                        .map_err(|error| format!("{} dynamic terrain: {error}", body.name))?;
                    let tectonics =
                        compile_tectonics_from_config(body.tectonics.as_ref(), &context);
                    let surface = PlanetSurface {
                        static_surface: package.static_surface,
                        dynamic_layers,
                        tectonics,
                    };
                    let surface_arc: Arc<dyn SurfaceQuery> = Arc::new(PackageSurface::new(
                        package.manifest,
                        surface,
                        DynamicSurfaceState::default(),
                    ));
                    // Landmarks are derived AFTER the surface exists so their
                    // relief can be measured from the one height authority
                    // rather than estimated by a parallel analytic model (which
                    // max-selection turns into a bug: the "most legible" pick
                    // is exactly the crater the model most overestimates).
                    registry.airless_landmarks.insert(
                        body.id,
                        airless_landmarks(&craters, surface_arc.as_ref(), body.radius_m),
                    );
                    (surface_arc, package_fingerprint)
                }
                _ => (
                    Arc::new(ProceduralSurface::new(body.radius_m as f32, body.id as u32)),
                    thalos_terrain::GENERATOR_VERSION ^ body.id as u64,
                ),
            };
            registry.surfaces.insert(body.id, surface);
            registry.fingerprints.insert(body.id, fingerprint);
        }
        Ok(registry)
    }

    pub fn surface(&self, body: BodyId) -> Option<Arc<dyn SurfaceQuery>> {
        self.surfaces.get(&body).cloned()
    }

    pub fn fingerprint(&self, body: BodyId) -> Option<u64> {
        self.fingerprints.get(&body).copied()
    }

    /// Mid-sized baked crater centres, ordered by usefulness for a close
    /// surface survey. This is package metadata for diagnostics/cinematics;
    /// terrain consumers still depend only on `SurfaceQuery`.
    pub fn airless_landmarks(&self, body: BodyId) -> &[AirlessLandmark] {
        self.airless_landmarks
            .get(&body)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub fn iter(&self) -> impl Iterator<Item = (BodyId, Arc<dyn SurfaceQuery>)> + '_ {
        self.surfaces
            .iter()
            .map(|(body, surface)| (*body, Arc::clone(surface)))
    }
}

/// Total landmarks retained per airless body.
const LANDMARK_BUDGET: usize = 256;

/// Pick the landmark craters cinematic framings may lock onto.
///
/// Two sets, concatenated, because two different questions get asked of this
/// list and one ordering cannot answer both:
///
/// - **Typical (first half).** Young (`age <= 1 Gyr`) craters nearest ~10 km,
///   which is what a survey framing wants: sharp rims, unambiguous morphology.
///   Callers taking the first acceptable landmark keep exactly the prior
///   behaviour, so the existing probes are unaffected.
/// - **Most legible (second half).** Big *and* still sharp — a framing that
///   must *contain* a crater rather than survey near one needs these. The
///   candidate pool is pre-ranked by the cheap analytic
///   `radius x degradation_factor`, but the kept set and the stored `relief_m`
///   come from **sampling the surface itself**. The analytic estimate must not
///   be the stored value: a max-over-candidates consumer (the rim framing's
///   `MostLegible`) selects exactly the crater the model most *overestimates*
///   — on Mira that was a claimed 5.8 km-relief basin whose baked ground
///   samples ±0.5 m-scale plains, so the framing hovered over nothing
///   (found 2026-07-24 during NTR-X1 verification).
///
/// Duplicates between the halves are harmless and deliberately not filtered:
/// "first in band" and "largest in band" both give the same answer with or
/// without them.
fn airless_landmarks(
    craters: &[thalos_terrain::Crater],
    surface: &dyn SurfaceQuery,
    body_radius_m: f64,
) -> Vec<AirlessLandmark> {
    const RADIUS_BAND_M: std::ops::RangeInclusive<f32> = 4_000.0..=30_000.0;
    let half = LANDMARK_BUDGET / 2;

    let in_band = || {
        craters
            .iter()
            .filter(|crater| RADIUS_BAND_M.contains(&crater.radius_m))
    };

    // Surviving relief measured from the height authority: centre sample plus
    // two rings (floor at 0.55 r, rim at 1.0 r), max − min. Coarse LOD — this
    // asks "does the crater read at framing scale", not for micro-detail.
    let measured_relief = |crater: &thalos_terrain::Crater| -> f32 {
        let dir = crater.center.as_dvec3().normalize();
        let arc = f64::from(crater.radius_m) / body_radius_m;
        let seed = if dir.dot(DVec3::Y).abs() < 0.95 {
            DVec3::Y
        } else {
            DVec3::X
        };
        let tangent_a = seed.cross(dir).normalize();
        let tangent_b = dir.cross(tangent_a).normalize();
        let lod_m = (crater.radius_m / 8.0).max(64.0);
        let mut lo = surface.sample_height_m(dir.as_vec3(), lod_m);
        let mut hi = lo;
        for ring_frac in [0.55_f64, 1.0] {
            let ring_arc = arc * ring_frac;
            for k in 0..8 {
                let a = std::f64::consts::TAU * k as f64 / 8.0;
                let ring = tangent_a * a.cos() + tangent_b * a.sin();
                let sample_dir = (dir * ring_arc.cos() + ring * ring_arc.sin()).normalize();
                let h = surface.sample_height_m(sample_dir.as_vec3(), lod_m);
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        hi - lo
    };

    let mut typical: Vec<&thalos_terrain::Crater> =
        in_band().filter(|crater| crater.age_gyr <= 1.0).collect();
    typical.sort_by(|a, b| {
        (a.radius_m - 10_000.0)
            .abs()
            .total_cmp(&(b.radius_m - 10_000.0).abs())
    });
    typical.truncate(half);

    // Analytic pre-rank bounds how many craters we pay to measure; measured
    // relief then decides what survives into the legible half.
    let legibility = |crater: &thalos_terrain::Crater| {
        crater.radius_m * thalos_terrain::degradation_factor(crater.radius_m, crater.age_gyr)
    };
    let mut pool: Vec<&thalos_terrain::Crater> = in_band().collect();
    pool.sort_by(|a, b| legibility(b).total_cmp(&legibility(a)));
    pool.truncate(half * 2);
    let mut legible: Vec<(&thalos_terrain::Crater, f32)> = pool
        .into_iter()
        .map(|crater| (crater, measured_relief(crater)))
        .collect();
    legible.sort_by(|a, b| b.1.total_cmp(&a.1));
    legible.truncate(half);

    typical
        .into_iter()
        .map(|crater| (crater, measured_relief(crater)))
        .chain(legible)
        .map(|(crater, relief_m)| AirlessLandmark {
            dir: crater.center.as_dvec3().normalize(),
            radius_m: crater.radius_m,
            relief_m,
        })
        .collect()
}

/// One crater a cinematic framing may lock onto.
///
/// Carries `relief_m` because radius alone is the wrong thing to rank by: an
/// ancient basin can be the widest feature on the body while `degradation_factor`
/// has relaxed it to almost flat ground, so "pick the largest" frames an empty
/// plain. `relief_m` is the depth that actually survives to the surface, which is
/// what decides whether a crater *reads* as one. Framing distance still scales
/// off `radius_m` — that is the feature's true extent.
#[derive(Clone, Copy, Debug)]
pub struct AirlessLandmark {
    pub dir: DVec3,
    pub radius_m: f32,
    pub relief_m: f32,
}

fn terrain_context(body: &BodyDefinition) -> TerrainCompileContext {
    TerrainCompileContext {
        body_name: body.name.clone(),
        radius_m: body.radius_m as f32,
        gravity_m_s2: body.surface_gravity_m_s2() as f32,
        rotation_hours: (body.rotation_period_s > 0.0)
            .then_some((body.rotation_period_s / 3600.0) as f32),
        obliquity_deg: Some(body.axial_tilt_rad.to_degrees() as f32),
        tidal_axis: matches!(body.kind, BodyKind::Moon).then_some(Vec3::Z),
        axial_tilt_rad: body.axial_tilt_rad as f32,
    }
}

/// Thread-safe body-surface projection used by canonical propagation.
#[derive(Default, Clone)]
pub struct SharedTerrainRegistry {
    inner: Arc<RwLock<HashMap<BodyId, Arc<dyn SurfaceQuery>>>>,
}

impl SharedTerrainRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, body: BodyId, surface: Arc<dyn SurfaceQuery>) {
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
