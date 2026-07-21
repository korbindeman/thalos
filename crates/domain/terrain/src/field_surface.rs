//! `FieldSurface` — the field-DAG-era live backing for the Query API seam.
//!
//! Migration phase P2 (see docs/planet-generation-pipeline-migration.md):
//! `FieldSurface` is the strangler-fig seam between the analytic terrain
//! evaluator and the Query API. It owns a body's terrain definition (today the
//! oceanic/continental params) plus a **reusable continent-intent cache**, and
//! produces both the baked [`StaticSurfaceData`] ([`FieldSurface::bake`]) and
//! on-demand [`SurfaceQuery`] samples from one source.
//!
//! Why it exists: the continent-shape kernel is ~80% of the oceanic field's
//! per-sample cost and is re-run from scratch on every bake today. By caching it
//! ([`build_continent_intent_cache`]) and holding the `FieldSurface` across edits,
//! a re-bake after a *non-shape* edit (palette, sea level, ocean fraction —
//! ocean fraction biases sea level *outside* the kernel) skips the kernel
//! entirely. This is the "intent layer" of the field-DAG made concrete; the
//! kernel will later decompose into named scalar intent fields, but the cache
//! semantics stay the same.
//!
//! The cache is materialised at the bake's output resolution, so the baked
//! cubemaps are bit-identical to the direct (uncached) kernel at texel centres —
//! no new impostor/ground divergence. The win is reuse, not coarsening.

use std::sync::Arc;

use glam::Vec3;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{Cubemap, default_resolution};
use crate::generic_terrestrial_field::{
    OceanicContinentalField, OceanicContinentalParams, RuntimeTerrainDetail,
    build_continent_intent_cache, sample_oceanic_continental,
};
use crate::query::{SurfaceQuery, SurfaceSample};
use crate::static_surface::StaticSurfaceData;
use crate::surface_color::{SurfaceColorSpec, paint_oceanic_surface_albedo};
use crate::surface_field::bake_surface_field_into_builder;
use crate::types::{Composition, Material};

/// Seabed/land material palette for the oceanic route, in `OceanicSample`
/// material-id order: abyssal floor, continental shelf, coastal sand,
/// continental soil, exposed rock, snow. Held on the `FieldSurface` so the bake
/// and the `SurfaceQuery` albedo agree on the same six materials.
fn oceanic_materials() -> Vec<Material> {
    vec![
        Material {
            albedo: [0.038, 0.034, 0.028],
            roughness: 0.76,
        },
        Material {
            albedo: [0.130, 0.108, 0.075],
            roughness: 0.64,
        },
        Material {
            albedo: [0.690, 0.590, 0.390],
            roughness: 0.72,
        },
        Material {
            albedo: [0.120, 0.220, 0.075],
            roughness: 0.80,
        },
        Material {
            albedo: [0.440, 0.310, 0.205],
            roughness: 0.86,
        },
        Material {
            albedo: [0.870, 0.875, 0.890],
            roughness: 0.42,
        },
    ]
}

/// The continent-shape params the intent cache depends on. Equal keys ⇒ the
/// cache is reused across edits/bakes; a changed key rebuilds it. Continent
/// shape is governed by the macro/warp/coast seeds; ocean fraction is *not*
/// here (it biases sea level outside the kernel), so changing it reuses the
/// cache and makes the re-bake cheap.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ContinentIntentKey {
    seed_macro: u32,
    seed_warp: u32,
    seed_coast: u32,
}

impl ContinentIntentKey {
    fn of(params: &OceanicContinentalParams) -> Self {
        Self {
            seed_macro: params.seed_macro,
            seed_warp: params.seed_warp,
            seed_coast: params.seed_coast,
        }
    }
}

/// Field-DAG-era live surface backing (oceanic/continental route).
pub struct FieldSurface {
    radius_m: f32,
    params: OceanicContinentalParams,
    composition: Composition,
    age_gyr: f32,
    obliquity_rad: f32,
    /// Seed for the surface-color painter (the body root seed).
    color_seed: u64,
    materials: Vec<Material>,
    intent_cache: Option<Arc<Cubemap<f32>>>,
    intent_cache_key: ContinentIntentKey,
    intent_cache_res: u32,
}

impl FieldSurface {
    /// Build an oceanic/continental field backing.
    pub fn oceanic(
        radius_m: f32,
        params: OceanicContinentalParams,
        composition: Composition,
        age_gyr: f32,
        obliquity_rad: f32,
        color_seed: u64,
    ) -> Self {
        Self {
            radius_m,
            params,
            composition,
            age_gyr,
            obliquity_rad,
            color_seed,
            materials: oceanic_materials(),
            intent_cache: None,
            intent_cache_key: ContinentIntentKey::of(&params),
            intent_cache_res: 0,
        }
    }

    pub fn params(&self) -> OceanicContinentalParams {
        self.params
    }

    pub fn radius_m(&self) -> f32 {
        self.radius_m
    }

    /// Replace the terrain params for the next bake/sample. Drops the intent
    /// cache only if a continent-shape param actually changed.
    pub fn set_params(&mut self, params: OceanicContinentalParams) {
        if ContinentIntentKey::of(&params) != self.intent_cache_key {
            self.intent_cache = None;
        }
        self.params = params;
    }

    /// Build (or reuse) the continent intent cache at `res`.
    fn intent_cache_at(&mut self, res: u32) -> Arc<Cubemap<f32>> {
        let key = ContinentIntentKey::of(&self.params);
        let reusable = self.intent_cache_res == res
            && self.intent_cache_key == key
            && self.intent_cache.is_some();
        if !reusable {
            self.intent_cache = Some(Arc::new(build_continent_intent_cache(self.params, res)));
            self.intent_cache_res = res;
            self.intent_cache_key = key;
        }
        self.intent_cache.clone().expect("intent cache just built")
    }

    /// Bake the full static surface at `resolution` (None → radius-derived).
    ///
    /// Reuses the continent intent cache across calls with the same resolution
    /// and continent-shape key, so repeated bakes after non-shape edits skip the
    /// (dominant) kernel cost.
    pub fn bake(&mut self, resolution: Option<u32>) -> StaticSurfaceData {
        let res = resolution.unwrap_or_else(|| default_resolution(self.radius_m));
        let cache = self.intent_cache_at(res);

        let mut builder = BodyBuilder::new(
            self.radius_m,
            self.color_seed,
            self.composition,
            Some(res),
            self.age_gyr,
            None,
            self.obliquity_rad,
        );
        builder.materials = self.materials.clone();

        let field = OceanicContinentalField::with_intent_cache(self.params, self.radius_m, cache);
        bake_surface_field_into_builder(&mut builder, &field);
        builder.runtime_detail = RuntimeTerrainDetail::OceanicContinental(self.params);
        builder.sea_level_m = Some(0.0);

        paint_oceanic_surface_albedo(
            &mut builder,
            &SurfaceColorSpec::aging_oceanic_homeworld(self.color_seed, 0.0),
        );
        builder.build()
    }
}

impl SurfaceQuery for FieldSurface {
    fn sample(&self, dir: Vec3, lod_m: f32) -> SurfaceSample {
        self.sample_d(dir.as_dvec3(), lod_m)
    }

    fn sample_d(&self, dir: glam::DVec3, lod_m: f32) -> SurfaceSample {
        let s = sample_oceanic_continental(
            self.params,
            self.radius_m as f64,
            dir,
            lod_m,
            self.intent_cache.as_deref(),
        );
        // Coarse trait path: albedo is the dominant material's flat colour. The
        // relief/climate-graded albedo lives in the baked cubemap
        // (`paint_surface_albedo`); precise per-direction colour is a later
        // refinement once the painter has a per-direction core.
        let albedo = self
            .materials
            .get(s.material_id as usize)
            .map(|m| Vec3::from_array(m.albedo))
            .unwrap_or(Vec3::splat(0.5));
        SurfaceSample {
            height_m: s.height_m,
            albedo_linear: albedo,
            roughness: s.roughness,
            moisture: 0.0,
        }
    }

    fn radius_m(&self) -> f32 {
        self.radius_m
    }

    fn height_range_m(&self) -> f32 {
        self.params.height_range_hint_m()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn params() -> OceanicContinentalParams {
        OceanicContinentalParams::from_seed_parts(1003, 1003 ^ 0x9999, 4_800.0, 0.62)
    }

    fn field() -> FieldSurface {
        FieldSurface::oceanic(
            3_186_000.0,
            params(),
            Composition::new(0.68, 0.30, 0.0, 0.02, 0.0),
            5.5,
            0.0,
            1003,
        )
    }

    fn dirs() -> Vec<Vec3> {
        vec![
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::NEG_X,
            Vec3::new(0.31, -0.42, 0.85).normalize(),
            Vec3::new(-0.67, 0.22, 0.71).normalize(),
            Vec3::new(0.1, -0.9, 0.4).normalize(),
        ]
    }

    /// The cached intent path must agree with the direct kernel at the cache's
    /// own texel centres (it reads the same value there), so a `FieldSurface`
    /// sample backed by an output-res cache matches the uncached params sampler
    /// within bilinear tolerance off-centre. Here we check the geometric height
    /// is close, not bit-identical, since arbitrary dirs land between texels.
    #[test]
    fn cached_height_tracks_direct_kernel() {
        let mut f = field();
        // Materialise an intent cache at a representative resolution.
        let _ = f.intent_cache_at(512);
        let p = params();
        for dir in dirs() {
            let cached = f.sample_height_m(dir, 64.0);
            let direct = p.sample_height_dm(3_186_000.0, dir.as_dvec3(), 64.0);
            let tol = (direct.abs() * 0.02).max(50.0);
            assert!(
                (cached - direct).abs() <= tol,
                "cached {cached} vs direct {direct} at {dir:?} (tol {tol})"
            );
        }
    }

    /// Reusing the cache (same shape key + res) must not rebuild it; changing a
    /// continent-shape seed must invalidate it.
    #[test]
    fn intent_cache_reuse_and_invalidation() {
        let mut f = field();
        let a = f.intent_cache_at(256);
        let b = f.intent_cache_at(256);
        assert!(Arc::ptr_eq(&a, &b), "same key+res must reuse the cache Arc");

        let mut shifted = params();
        shifted.seed_macro ^= 0xABCD;
        f.set_params(shifted);
        let c = f.intent_cache_at(256);
        assert!(
            !Arc::ptr_eq(&a, &c),
            "a continent-shape seed change must rebuild the cache"
        );
    }

    /// A non-shape edit (ocean fraction) reuses the cache.
    #[test]
    fn ocean_fraction_edit_reuses_cache() {
        let mut f = field();
        let a = f.intent_cache_at(256);
        let mut wetter = params();
        wetter.ocean_fraction = 0.75;
        f.set_params(wetter);
        let b = f.intent_cache_at(256);
        assert!(
            Arc::ptr_eq(&a, &b),
            "ocean fraction is outside the kernel; cache must be reused"
        );
    }

    /// Determinism: a sample is stable across repeated calls.
    #[test]
    fn sampling_is_deterministic() {
        let f = field();
        let dir = Vec3::new(0.3, 0.5, -0.8).normalize();
        let a = f.sample(dir, 32.0);
        let b = f.sample(dir, 32.0);
        assert_eq!(a.height_m.to_bits(), b.height_m.to_bits());
        assert_eq!(a.roughness.to_bits(), b.roughness.to_bits());
    }
}
