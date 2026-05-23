//! Basic `GenericTerrestrial` surface evaluator.
//!
//! This is the P2A vertical-slice terrain definition for Thalos: one
//! all-land, single-biome continental surface. The same evaluator is used by
//! the bake (`SurfaceField`) and by the Query API's runtime height path, so the
//! ground LOD does not inherit the legacy P0 HMF cascade.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::noise::{fbm3, hmf_ridged_3d};
use crate::seeding::splitmix64;
use crate::surface_field::{
    BiomeMix, SurfaceField, SurfaceFieldSample, SurfaceMaterialMix, smoothstep,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BasicContinentalParams {
    pub seed_macro: u32,
    pub seed_regional: u32,
    pub seed_hills: u32,
    pub seed_mountains: u32,
    pub seed_fine: u32,
    pub relief_scale_m: f32,
}

impl BasicContinentalParams {
    pub fn from_seed_parts(shape_seed: u64, detail_seed: u64, relief_scale_m: f32) -> Self {
        Self {
            seed_macro: splitmix64(shape_seed ^ 0xB451_C07A_1A1D_0001) as u32,
            seed_regional: splitmix64(shape_seed ^ 0xB451_C07A_1A1D_0002) as u32,
            seed_hills: splitmix64(detail_seed ^ 0xB451_C07A_1A1D_0003) as u32,
            seed_mountains: splitmix64(detail_seed ^ 0xB451_C07A_1A1D_0004) as u32,
            seed_fine: splitmix64(detail_seed ^ 0xB451_C07A_1A1D_0005) as u32,
            relief_scale_m: relief_scale_m.max(750.0),
        }
    }

    /// Conservative absolute-height envelope for runtime consumers that encode
    /// height into a fixed range before sampling the direct evaluator.
    pub fn height_range_hint_m(self) -> f32 {
        self.relief_scale_m * 2.9
    }

    pub fn sample_height_m(self, radius_m: f32, dir: Vec3, sample_scale_m: f32) -> f32 {
        sample_basic_continental_height_m(self, radius_m, dir, sample_scale_m)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RuntimeTerrainDetail {
    /// P0 compatibility: baked cubemap + legacy runtime HMF uplift.
    LegacyHmf,
    /// P2A: evaluate the basic continental terrain function directly. This is
    /// the same function used to bake the cubemap, so the ground mesh no
    /// longer receives a separate old detail layer. Runtime ground geometry
    /// currently samples it at full detail regardless of tile LOD to avoid
    /// parent/child handoff contouring during the vertical slice.
    BasicContinental(BasicContinentalParams),
}

impl Default for RuntimeTerrainDetail {
    fn default() -> Self {
        Self::LegacyHmf
    }
}

pub struct BasicContinentalField {
    params: BasicContinentalParams,
    radius_m: f32,
}

impl BasicContinentalField {
    pub fn new(params: BasicContinentalParams, radius_m: f32) -> Self {
        Self { params, radius_m }
    }

    pub fn params(&self) -> BasicContinentalParams {
        self.params
    }
}

impl SurfaceField for BasicContinentalField {
    fn sample(&self, dir: Vec3, sample_scale_m: f32) -> SurfaceFieldSample {
        let dir = dir.normalize_or_zero();
        let height_m = self
            .params
            .sample_height_m(self.radius_m, dir, sample_scale_m);
        let roughness = roughness_at(self.params, self.radius_m, dir, sample_scale_m);

        SurfaceFieldSample::new(
            height_m,
            SurfaceMaterialMix::single(0),
            BiomeMix::single(0),
            roughness,
            dir,
        )
    }
}

fn sample_basic_continental_height_m(
    params: BasicContinentalParams,
    radius_m: f32,
    dir: Vec3,
    sample_scale_m: f32,
) -> f32 {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return 0.0;
    }

    let p_m = dir * radius_m.max(1.0);
    let relief = params.relief_scale_m;
    let lod_m = sample_scale_m.max(1.0);

    let macro_n = fbm3_band(p_m, 1_250_000.0, params.seed_macro, 5, lod_m);
    let regional_n = fbm3_band(p_m, 360_000.0, params.seed_regional, 5, lod_m);
    let hills_n = fbm3_band(p_m, 95_000.0, params.seed_hills, 4, lod_m);

    // Smooth ridge-lines for local mountains. This uses the same continuous
    // noise primitive as the rest of the field; there is no stepped Voronoi or
    // positive-only legacy uplift here.
    let mountain_weight = band_weight(42_000.0, lod_m);
    let ridge = hmf_ridged_3d(p_m / 42_000.0, params.seed_mountains, 4.0, 0.52, 2.0, 0.98);
    let uplift_mask = smoothstep(-0.15, 0.58, macro_n * 0.65 + regional_n * 0.35);
    let mountains = ridge * uplift_mask * mountain_weight;

    let fine_weight = band_weight(9_000.0, lod_m);
    let fine_n = fbm3(
        p_m.x / 9_000.0,
        p_m.y / 9_000.0,
        p_m.z / 9_000.0,
        params.seed_fine,
        3,
        0.50,
        2.03,
    ) * fine_weight;

    let undulation_weight = band_weight(1_800.0, lod_m);
    let undulation_n = fbm3(
        p_m.x / 1_800.0,
        p_m.y / 1_800.0,
        p_m.z / 1_800.0,
        params.seed_fine ^ 0x8D27_6B45,
        3,
        0.48,
        2.11,
    ) * undulation_weight;

    let texture_weight = band_weight(420.0, lod_m);
    let texture_n = fbm3(
        p_m.x / 420.0,
        p_m.y / 420.0,
        p_m.z / 420.0,
        params.seed_fine ^ 0xC36E_1A91,
        2,
        0.45,
        2.17,
    ) * texture_weight;

    let height_m = relief
        * (0.86
            + macro_n * 0.42
            + regional_n * 0.22
            + hills_n * 0.10
            + mountains * 0.50
            + fine_n * 0.055
            + undulation_n * 0.055
            + texture_n * 0.014);
    height_m.max(120.0)
}

fn roughness_at(
    params: BasicContinentalParams,
    radius_m: f32,
    dir: Vec3,
    sample_scale_m: f32,
) -> f32 {
    let p_m = dir.normalize_or_zero() * radius_m.max(1.0);
    let lod_m = sample_scale_m.max(1.0);
    let hills = fbm3_band(p_m, 95_000.0, params.seed_hills, 4, lod_m).abs();
    let ridge = hmf_ridged_3d(p_m / 42_000.0, params.seed_mountains, 4.0, 0.52, 2.0, 0.98);
    let local = fbm3_band(p_m, 1_800.0, params.seed_fine ^ 0x8D27_6B45, 3, lod_m).abs();
    (0.76 + hills * 0.06 + ridge * band_weight(42_000.0, lod_m) * 0.08 + local * 0.04)
        .clamp(0.62, 0.94)
}

fn fbm3_band(p_m: Vec3, wavelength_m: f32, seed: u32, octaves: u32, lod_m: f32) -> f32 {
    fbm3(
        p_m.x / wavelength_m,
        p_m.y / wavelength_m,
        p_m.z / wavelength_m,
        seed,
        octaves,
        0.53,
        2.03,
    ) * band_weight(wavelength_m, lod_m)
}

fn band_weight(wavelength_m: f32, lod_m: f32) -> f32 {
    // Fade bands in only when the sample can represent at least a few samples
    // per wavelength. This keeps both cubemap bakes and UDLOD tiles smooth
    // instead of aliasing unresolved relief into steps.
    smoothstep(lod_m * 3.0, lod_m * 6.0, wavelength_m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> BasicContinentalParams {
        BasicContinentalParams::from_seed_parts(1234, 5678, 4_500.0)
    }

    #[test]
    fn basic_continental_sampling_is_deterministic() {
        let params = params();
        let dir = Vec3::new(0.31, -0.42, 0.85).normalize();
        let a = params.sample_height_m(3_186_000.0, dir, 1.0);
        let b = params.sample_height_m(3_186_000.0, dir, 1.0);
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn basic_continental_height_stays_inside_hint_for_representative_samples() {
        let params = params();
        let hint = params.height_range_hint_m();
        let dirs = [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            -Vec3::X,
            -Vec3::Y,
            -Vec3::Z,
            Vec3::new(0.31, -0.42, 0.85).normalize(),
            Vec3::new(-0.67, 0.22, 0.71).normalize(),
        ];

        for dir in dirs {
            let height = params.sample_height_m(3_186_000.0, dir, 1.0);
            assert!(height.is_finite(), "height must be finite for {dir:?}");
            assert!(
                height.abs() <= hint,
                "height {height} exceeded hint ±{hint} for {dir:?}"
            );
        }
    }

    #[test]
    fn coarser_sample_scale_suppresses_unresolved_bands() {
        let params = params();
        let dir = Vec3::new(0.31, -0.42, 0.85).normalize();
        let fine = params.sample_height_m(3_186_000.0, dir, 1.0);
        let coarse = params.sample_height_m(3_186_000.0, dir, 1_000_000.0);

        // The exact values are seed-dependent; this asserts the LOD parameter
        // is wired into the evaluator instead of being ignored by the bake path.
        assert_ne!(fine.to_bits(), coarse.to_bits());
    }
}
