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
#[cfg(feature = "legacy-udlod")]
use thalos_body_render::GpuAtlasMirrorHandle;
use thalos_body_render::{ImpostorAlbedo, RenderedGround};
use thalos_physics_canonical::terrain_provider::TerrainProvider;
#[cfg(not(debug_assertions))]
use thalos_terrain::load_static_package_artifact;
use thalos_terrain::{
    BodyArchetype, DynamicSurfaceState, PackageSurface, PlanetSurface, ProceduralSurface,
    SurfaceQuery, TerrainCompileContext, TerrainConfig, cache, compile_dynamic_surface_layers,
    compile_tectonics_from_config,
};
#[cfg(debug_assertions)]
use thalos_terrain::{TerrainCompileOptions, load_static_package};
use thalos_world::{BodyDefinition, BodyId, BodyKind};

/// Coarse LOD (metres per sample) for orbital collision queries. The propagator
/// only needs "does this orbit dip below the ground", so a coarse sample —
/// skipping the fine procedural octaves — is both sufficient and cheap.
const PROPAGATOR_LOD_M: f32 = 1000.0;

/// Runtime-only projection of renderer residency data — **one entry per body,
/// whichever renderer currently draws its ground** (udlod's atlas mirror, or
/// the standard-path tile renderer's height mirror). The canonical height
/// queries themselves remain in `thalos_physics_local::HeightSourceRegistry`
/// behind the renderer-independent `thalos_terrain::HeightSource` contract.
///
/// Surface-detail consumers (grass / trees / rocks) read the residency gate
/// from here, so they work on either renderer without knowing which is up.
/// The tile driver **replaces** a body's entry when it takes the body over.
#[derive(Resource, Default, Clone)]
pub struct RenderedGroundRegistry {
    grounds: HashMap<BodyId, RenderedGround>,
}

impl RenderedGroundRegistry {
    pub fn insert(&mut self, body_id: BodyId, ground: RenderedGround) {
        self.grounds.insert(body_id, ground);
    }

    /// Drop a body's rendered ground — its renderer has released it (the tile
    /// root following the view anchor to another body). Consumers fall back to
    /// the canonical surface, which is the same thing they do for a body that
    /// never had a rendered ground at all.
    pub fn remove(&mut self, body_id: BodyId) {
        self.grounds.remove(&body_id);
    }

    pub fn get(&self, body_id: BodyId) -> Option<RenderedGround> {
        self.grounds.get(&body_id).cloned()
    }

    /// udlod's concrete atlas handle for `body_id`, or `None` when the body is
    /// unknown or drawn by another renderer.
    #[cfg(feature = "legacy-udlod")]
    pub fn udlod_handle(&self, body_id: BodyId) -> Option<GpuAtlasMirrorHandle> {
        self.grounds.get(&body_id)?.udlod_handle()
    }
}

/// The macro appearance of each body, as baked once at spawn.
///
/// The same bake feeds the distant body the player **looks at**
/// (`SolidPlanetMaterial`'s albedo cube) and the planet a stainless hull
/// **reflects** from orbit (`reflection_probe`). Two consumers, one authority —
/// a reflected coastline that disagreed with the coastline beside it would be
/// worse than no coastline at all. Absent for solid-colour and degraded bodies,
/// which is the same "no cube" case the impostor shader already handles.
#[derive(Resource, Default, Clone)]
pub struct ImpostorAlbedoRegistry {
    bakes: HashMap<BodyId, Arc<ImpostorAlbedo>>,
}

impl ImpostorAlbedoRegistry {
    pub fn insert(&mut self, body_id: BodyId, bake: Arc<ImpostorAlbedo>) {
        self.bakes.insert(body_id, bake);
    }

    pub fn get(&self, body_id: BodyId) -> Option<Arc<ImpostorAlbedo>> {
        self.bakes.get(&body_id).cloned()
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
    degraded: HashMap<BodyId, DegradedSurface>,
}

/// A body whose canonical surface could not be constructed (a missing or stale
/// offline package, a dynamic-layer compile failure).
///
/// Surface construction is **per body, and one body's failure is local to it**:
/// the registry keeps the process up and records the body here instead of
/// aborting the whole app. A stale `Mira.bin` must not block a Thalos capture —
/// see INC-20260724T182643Z-unrelated-body-package-aborts-boot. Consumers see a
/// degraded body as surface-less through the existing `Option` accessors; the
/// capture server refuses to certify a shot whose *target* body is degraded, so
/// the failure still cannot masquerade as valid evidence.
#[derive(Clone, Debug)]
pub struct DegradedSurface {
    pub body_name: String,
    pub reason: String,
}

/// NTR-X2a: whether this session renders Thalos through the learned terrain
/// path. The Cargo features define capability/default; `THALOS_TERRAIN` is a
/// runtime override for controlled A/Bs.
pub fn thalos_diffusion_enabled() -> bool {
    match configured_thalos_diffusion() {
        Ok(enabled) => enabled,
        Err(error) => panic!("invalid Thalos terrain configuration: {error}"),
    }
}

fn configured_thalos_diffusion() -> Result<bool, String> {
    static ENABLED: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    ENABLED
        .get_or_init(|| {
            select_thalos_diffusion(
                std::env::var("THALOS_TERRAIN").ok().as_deref(),
                cfg!(feature = "neural-terrain"),
                cfg!(feature = "neural-terrain-default"),
            )
        })
        .clone()
}

fn select_thalos_diffusion(
    override_value: Option<&str>,
    neural_available: bool,
    neural_default: bool,
) -> Result<bool, String> {
    if neural_default && !neural_available {
        return Err("neural-terrain-default requires neural-terrain".to_string());
    }
    let requested = match override_value
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        None => neural_default,
        Some(value)
            if matches_ignore_ascii_case(value, &["diffusion", "neural", "1", "true", "on"]) =>
        {
            true
        }
        Some(value) if matches_ignore_ascii_case(value, &["procedural", "0", "false", "off"]) => {
            false
        }
        Some(value) => {
            return Err(format!(
                "unknown THALOS_TERRAIN={value:?}; expected neural, diffusion, or procedural"
            ));
        }
    };
    if requested && !neural_available {
        return Err(
            "neural terrain was requested, but this binary was built without the neural-terrain feature"
                .to_string(),
        );
    }
    Ok(requested)
}

fn matches_ignore_ascii_case(value: &str, candidates: &[&str]) -> bool {
    candidates
        .iter()
        .any(|candidate| value.eq_ignore_ascii_case(candidate))
}

impl BodySurfaceRegistry {
    pub fn load(bodies: &[BodyDefinition], package_dir: &std::path::Path) -> Result<Self, String> {
        let mut registry = Self::default();
        let diffusion_enabled = configured_thalos_diffusion()?;
        for body in bodies.iter().filter(|body| body.terrain.is_some()) {
            if body.name == "Thalos" && diffusion_enabled {
                #[cfg(feature = "neural-terrain")]
                let dir = package_dir.join("thalos_diffusion");
                #[cfg(feature = "neural-terrain")]
                match thalos_terrain::DiffusionSurface::load(
                    &dir,
                    body.radius_m as f32,
                    body.id as u32,
                ) {
                    Ok(surface) => {
                        match surface.conditioning_generator_version() {
                            Some(version) if version != thalos_terrain::GENERATOR_VERSION => {
                                bevy::log::warn!(
                                    "Thalos learned macro relief was conditioned with terrain generator {version}, current generator is {}. The package remains usable, but it will not show the current tectonic provinces until it is regenerated",
                                    thalos_terrain::GENERATOR_VERSION
                                );
                            }
                            None => {
                                bevy::log::warn!(
                                    "Thalos learned macro relief has no conditioning-generator provenance. Regenerate the terrain package before using it as evidence for current tectonic structure"
                                );
                            }
                            _ => {}
                        }
                        let fingerprint =
                            thalos_terrain::GENERATOR_VERSION ^ surface.content_fingerprint;
                        bevy::log::info!(
                            "Thalos: terrain-diffusion surface active (fingerprint {fingerprint:#x})"
                        );
                        registry.surfaces.insert(body.id, Arc::new(surface));
                        registry.fingerprints.insert(body.id, fingerprint);
                        continue;
                    }
                    Err(error) => {
                        return Err(format!(
                            "neural terrain was selected but {} failed to load: {error}. \
                             Re-download the complete build or run `just terrain-assets` in a developer checkout",
                            dir.display()
                        ));
                    }
                }
                #[cfg(not(feature = "neural-terrain"))]
                unreachable!("terrain selection rejects neural mode when the feature is absent");
            }
            let built = match &body.terrain {
                TerrainConfig::Feature(feature)
                    if feature.archetype == BodyArchetype::AirlessImpactMoon =>
                {
                    build_airless_package_surface(body, package_dir)
                }
                _ => Ok(BuiltSurface {
                    surface: Arc::new(ProceduralSurface::new(body.radius_m as f32, body.id as u32)),
                    fingerprint: thalos_terrain::GENERATOR_VERSION ^ body.id as u64,
                    landmarks: Vec::new(),
                }),
            };
            match built {
                Ok(built) => {
                    if !built.landmarks.is_empty() {
                        registry.airless_landmarks.insert(body.id, built.landmarks);
                    }
                    registry.surfaces.insert(body.id, built.surface);
                    registry.fingerprints.insert(body.id, built.fingerprint);
                }
                Err(reason) => {
                    // Local failure, local consequence: this body loses its
                    // surface, every other body loads normally, and the process
                    // stays up. Aborting here used to make one stale package
                    // (Mira) block unrelated work (a Thalos capture) —
                    // INC-20260724T182643Z.
                    bevy::log::error!(
                        "{}: terrain surface unavailable — {reason}. \
                         This body will not render; other bodies are unaffected.",
                        body.name
                    );
                    registry.degraded.insert(
                        body.id,
                        DegradedSurface {
                            body_name: body.name.clone(),
                            reason,
                        },
                    );
                }
            }
        }
        Ok(registry)
    }

    pub fn surface(&self, body: BodyId) -> Option<Arc<dyn SurfaceQuery>> {
        self.surfaces.get(&body).cloned()
    }

    /// Name-keyed lookup for consumers that address bodies by authored name
    /// (the capture presets' `target_body_name`).
    pub fn degraded_by_name(&self, name: &str) -> Option<&DegradedSurface> {
        self.degraded
            .values()
            .find(|entry| entry.body_name.eq_ignore_ascii_case(name))
    }

    pub fn degraded_bodies(&self) -> impl Iterator<Item = &DegradedSurface> + '_ {
        self.degraded.values()
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

/// One body's constructed surface, before it is filed into the registry.
struct BuiltSurface {
    surface: Arc<dyn SurfaceQuery>,
    fingerprint: u64,
    landmarks: Vec<AirlessLandmark>,
}

/// Construct an airless body's surface from its offline package.
///
/// Split out of `BodySurfaceRegistry::load` so a per-body failure is an
/// ordinary `Err` the caller records as degraded, rather than a `?` that
/// aborts the load of every *other* body with it.
fn build_airless_package_surface(
    body: &BodyDefinition,
    package_dir: &std::path::Path,
) -> Result<BuiltSurface, String> {
    let context = terrain_context(body);
    #[cfg(debug_assertions)]
    let options = TerrainCompileOptions::default();
    let path = cache::cache_path(package_dir, &body.name);
    #[cfg(debug_assertions)]
    let package = {
        let key =
            cache::terrain_cache_key(&body.terrain, body.tectonics.as_ref(), &context, options);
        load_static_package(&path, &body.name, key)
    };
    #[cfg(not(debug_assertions))]
    let package = load_static_package_artifact(&path, &body.name);
    let package = package.map_err(|error| {
        format!(
            "requires an offline terrain package: {error}. Run `just bake {}`",
            body.name
        )
    })?;
    let fingerprint = package.manifest.artifact_fingerprint();
    let craters = package.static_surface.craters.clone();
    let dynamic_layers = compile_dynamic_surface_layers(&body.terrain, &context)
        .map_err(|error| format!("dynamic terrain: {error}"))?;
    let tectonics = compile_tectonics_from_config(body.tectonics.as_ref(), &context);
    let surface = PlanetSurface {
        static_surface: package.static_surface,
        dynamic_layers,
        tectonics,
    };
    let surface: Arc<dyn SurfaceQuery> = Arc::new(PackageSurface::new(
        package.manifest,
        surface,
        DynamicSurfaceState::default(),
    ));
    // Landmarks are derived AFTER the surface exists so their relief can be
    // measured from the one height authority rather than estimated by a
    // parallel analytic model (which max-selection turns into a bug: the "most
    // legible" pick is exactly the crater the model most overestimates).
    let landmarks = airless_landmarks(&craters, surface.as_ref(), body.radius_m);
    Ok(BuiltSurface {
        surface,
        fingerprint,
        landmarks,
    })
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

    // Surviving relief measured from the height authority, and measured
    // *bowl-shaped*: median of the rim ring (1.0 r) minus the floor (centre
    // and the 0.4 r ring's median). A max−min window would re-create the
    // selection bug one level down — over ~256 candidates the max−min winner
    // is whichever window happens to straddle a neighbouring deep crater, not
    // an actual bowl under the landmark. Medians make one stray sample
    // harmless; a degraded crater (rim ≈ floor) correctly measures ~0.
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
        let ring_median = |ring_frac: f64| -> f32 {
            let ring_arc = arc * ring_frac;
            let mut hs: Vec<f32> = (0..8)
                .map(|k| {
                    let a = std::f64::consts::TAU * k as f64 / 8.0;
                    let ring = tangent_a * a.cos() + tangent_b * a.sin();
                    let sample_dir = (dir * ring_arc.cos() + ring * ring_arc.sin()).normalize();
                    surface.sample_height_m(sample_dir.as_vec3(), lod_m)
                })
                .collect();
            hs.sort_by(f32::total_cmp);
            (hs[3] + hs[4]) * 0.5
        };
        let centre = surface.sample_height_m(dir.as_vec3(), lod_m);
        let floor = centre.min(ring_median(0.4));
        let rim = ring_median(1.0);
        (rim - floor).max(0.0)
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
    fn terrain_selection_separates_capability_default_and_override() {
        assert_eq!(select_thalos_diffusion(None, false, false), Ok(false));
        assert_eq!(select_thalos_diffusion(None, true, false), Ok(false));
        assert_eq!(select_thalos_diffusion(None, true, true), Ok(true));
        assert_eq!(
            select_thalos_diffusion(Some("procedural"), true, true),
            Ok(false)
        );
        assert_eq!(
            select_thalos_diffusion(Some("neural"), true, false),
            Ok(true)
        );
    }

    #[test]
    fn terrain_selection_rejects_unavailable_or_unknown_modes() {
        assert!(select_thalos_diffusion(Some("neural"), false, false).is_err());
        assert!(select_thalos_diffusion(Some("surprise"), true, false).is_err());
        assert!(select_thalos_diffusion(None, false, true).is_err());
    }

    #[test]
    fn shared_registry_reports_zero_for_unknown_body() {
        let registry = SharedTerrainRegistry::new();
        assert_eq!(registry.surface_elevation_m(0, DVec3::X), 0.0);
        assert_eq!(registry.max_elevation_m(0), 0.0);
        assert!(!registry.contains(0));
    }
}
