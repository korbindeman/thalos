//! Basic `GenericTerrestrial` surface evaluator.
//!
//! This is the P2A vertical-slice terrain definition for Thalos: one
//! all-land, single-biome continental surface. The same evaluator is used by
//! the bake (`SurfaceField`) and by the Query API's runtime height path, so the
//! ground LOD does not inherit the legacy P0 HMF cascade.

use std::sync::Arc;

use glam::{DVec3, Vec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::cubemap::{Cubemap, CubemapFace, face_uv_to_dir};
use crate::noise::{eroded_ridged_3d_f64, fbm3_f64};
use crate::seeding::splitmix64;
use crate::surface_color::{
    AGING_OCEANIC_BIOME_BEACH, AGING_OCEANIC_BIOME_FOREST, AGING_OCEANIC_BIOME_GRASSLAND,
    AGING_OCEANIC_BIOME_OCEAN, AGING_OCEANIC_BIOME_ROCK, AGING_OCEANIC_BIOME_SHELF,
    AGING_OCEANIC_BIOME_SNOW, AGING_OCEANIC_BIOME_STEPPE, AGING_OCEANIC_BIOME_TUNDRA,
};
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct OceanicContinentalParams {
    pub seed_macro: u32,
    pub seed_warp: u32,
    pub seed_coast: u32,
    pub seed_islands: u32,
    pub seed_hills: u32,
    pub seed_mountains: u32,
    pub seed_fine: u32,
    pub relief_scale_m: f32,
    pub ocean_fraction: f32,
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

    /// f32-direction entry (bake path: cubemap texels, where f32 addressing is
    /// already well below texel size). Promotes to the f64 evaluator.
    pub fn sample_height_m(self, radius_m: f32, dir: Vec3, sample_scale_m: f32) -> f32 {
        sample_basic_continental_height_dm(self, radius_m as f64, dir.as_dvec3(), sample_scale_m)
    }

    /// f64-direction entry (runtime ground LOD). Keeps the sample position
    /// precise at planet scale so the surface does not quantise into ~0.25 m
    /// body-local plateaus when viewed on foot.
    pub fn sample_height_dm(self, radius_m: f64, dir: DVec3, sample_scale_m: f32) -> f32 {
        sample_basic_continental_height_dm(self, radius_m, dir, sample_scale_m)
    }
}

impl OceanicContinentalParams {
    pub fn from_seed_parts(
        shape_seed: u64,
        detail_seed: u64,
        relief_scale_m: f32,
        ocean_fraction: f32,
    ) -> Self {
        Self {
            seed_macro: splitmix64(shape_seed ^ 0x0CEA_11C0_771A_0001) as u32,
            seed_warp: splitmix64(shape_seed ^ 0x0CEA_11C0_771A_0002) as u32,
            seed_coast: splitmix64(shape_seed ^ 0x0CEA_11C0_771A_0003) as u32,
            seed_islands: splitmix64(shape_seed ^ 0x0CEA_11C0_771A_0004) as u32,
            seed_hills: splitmix64(detail_seed ^ 0x0CEA_11C0_771A_0005) as u32,
            seed_mountains: splitmix64(detail_seed ^ 0x0CEA_11C0_771A_0006) as u32,
            seed_fine: splitmix64(detail_seed ^ 0x0CEA_11C0_771A_0007) as u32,
            relief_scale_m: relief_scale_m.max(2_500.0),
            ocean_fraction: ocean_fraction.clamp(0.35, 0.82),
        }
    }

    pub fn height_range_hint_m(self) -> f32 {
        (self.relief_scale_m * 1.75 + 5_200.0).max(7_500.0)
    }

    pub fn sample_height_m(self, radius_m: f32, dir: Vec3, sample_scale_m: f32) -> f32 {
        // Runtime/bake-without-cache path: evaluate the continent kernel
        // directly (exact). The cached path lives on `OceanicContinentalField`.
        sample_oceanic_continental(self, radius_m as f64, dir.as_dvec3(), sample_scale_m, None)
            .height_m
    }

    pub fn sample_height_dm(self, radius_m: f64, dir: DVec3, sample_scale_m: f32) -> f32 {
        sample_oceanic_continental(self, radius_m, dir, sample_scale_m, None).height_m
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
    /// P2A.5: signed oceanic terrain for Thalos. The evaluator produces both
    /// exposed land and underwater seabed height/materials; water is a separate
    /// render surface at `sea_level_m`, not a terrain material.
    OceanicContinental(OceanicContinentalParams),
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

pub struct OceanicContinentalField {
    params: OceanicContinentalParams,
    radius_m: f32,
    /// Optional cached continent **intent** ([`build_continent_intent_cache`]).
    /// When present, the expensive 12-shape continent kernel is read from this
    /// coarse cubemap instead of recomputed per sample — the bulk of the field
    /// cost. `None` reproduces the direct (uncached) evaluation exactly.
    intent_cache: Option<Arc<Cubemap<f32>>>,
}

impl BasicContinentalField {
    pub fn new(params: BasicContinentalParams, radius_m: f32) -> Self {
        Self { params, radius_m }
    }

    pub fn params(&self) -> BasicContinentalParams {
        self.params
    }
}

impl OceanicContinentalField {
    pub fn new(params: OceanicContinentalParams, radius_m: f32) -> Self {
        Self {
            params,
            radius_m,
            intent_cache: None,
        }
    }

    /// Like [`OceanicContinentalField::new`] but reads the continent kernel from
    /// a prebuilt [`build_continent_intent_cache`] cube. The kernel is the
    /// dominant cost of the field, so reusing one cache across bakes (e.g. live
    /// editor edits that don't move continents) makes re-bakes cheap.
    pub fn with_intent_cache(
        params: OceanicContinentalParams,
        radius_m: f32,
        intent_cache: Arc<Cubemap<f32>>,
    ) -> Self {
        Self {
            params,
            radius_m,
            intent_cache: Some(intent_cache),
        }
    }

    pub fn params(&self) -> OceanicContinentalParams {
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

impl SurfaceField for OceanicContinentalField {
    fn sample(&self, dir: Vec3, sample_scale_m: f32) -> SurfaceFieldSample {
        let dir = dir.normalize_or_zero();
        let sample = sample_oceanic_continental(
            self.params,
            self.radius_m as f64,
            dir.as_dvec3(),
            sample_scale_m,
            self.intent_cache.as_deref(),
        );
        SurfaceFieldSample::new(
            sample.height_m,
            SurfaceMaterialMix::single(sample.material_id),
            sample.biome_mix,
            sample.roughness,
            dir,
        )
    }
}

/// Broad continental swell plus the `0..1` relief-control field that decides
/// where the surface reads as flat plain (`0`) versus eroded hill country (`1`).
///
/// Shared by the height and roughness evaluators so they agree on the
/// plains/hills split. The control field is its *own* low-frequency noise (not
/// just an elevation threshold) so plains and hills interleave irregularly; a
/// light elevation bias keeps uplands a touch rougher than basins without the
/// split simply tracking altitude.
fn continental_relief(params: BasicContinentalParams, p_m: DVec3, lod_m: f32) -> (f32, f32) {
    let macro_n = fbm3_band(p_m, 1_250_000.0, params.seed_macro, 5, lod_m);
    let regional_n = fbm3_band(p_m, 360_000.0, params.seed_regional, 5, lod_m);
    let continent = macro_n * 0.62 + regional_n * 0.30;

    let control_n = fbm3_band(p_m, 240_000.0, params.seed_hills, 4, lod_m);
    // Bias the onset hard toward plains: most of the surface should be genuine,
    // usable plains, with hill country a clear minority. The weak continental
    // coupling keeps hills from blanketing every upland.
    let relief_control = smoothstep(0.18, 0.55, control_n * 0.88 + continent * 0.18);
    (continent, relief_control)
}

fn sample_basic_continental_height_dm(
    params: BasicContinentalParams,
    radius_m: f64,
    dir: DVec3,
    sample_scale_m: f32,
) -> f32 {
    let dir = dir.normalize_or_zero();
    if dir == DVec3::ZERO {
        return 0.0;
    }

    // Sample position in body-local metres, kept in f64: at planet scale the
    // f32 ULP here is ~0.25 m, which is what quantised the surface into
    // axis-aligned plateaus before. See `noise::*_f64`.
    let p_m = dir * radius_m.max(1.0);
    let relief = params.relief_scale_m;
    let lod_m = sample_scale_m.max(1.0);

    // Broad continental shape + where the terrain is dissected vs flat.
    let (continent, relief_control) = continental_relief(params, p_m, lod_m);

    // The visible relief is spent at *small* wavelengths so slopes actually
    // read as hills/scarps rather than continental swells. Two eroded bands,
    // both squared-ridge swiss turbulence (sharp crests, slope-damped flat
    // valley floors), gated entirely by the relief control so plains stay flat:
    //
    // - a broad highland swell (~55 km) the hills sit on, and
    // - steep hills (~3 km base, octaves down to ~40 m) that carry most of the
    //   amplitude — this is the band that makes the terrain read as hill
    //   country instead of a gentle dome.
    let swell_wl_m = 55_000.0;
    let swell = eroded_ridged_band(p_m, swell_wl_m, params.seed_mountains, 5, 0.5, 2.0, 1.0, lod_m);

    let hill_wl_m = 3_200.0;
    let hills = eroded_ridged_band(
        p_m,
        hill_wl_m,
        params.seed_mountains ^ 0x5151_3737,
        7,
        0.5,
        2.08,
        1.6,
        lod_m,
    );

    let hill_h = (swell * 0.28 + hills * 0.50) * relief_control;

    // Gentle rolling base — present everywhere (plains *and* hills) at low
    // amplitude, so genuine plains read as softly rolling, usable ground rather
    // than a dead-flat plane. Two wavelengths plus fine texture; slopes stay a
    // few degrees so plains remain landable and walkable.
    let roll_lo = fbm3_band(p_m, 4_000.0, params.seed_fine ^ 0x9E37_79B9, 3, lod_m);
    let roll_hi = fbm3_band(p_m, 1_300.0, params.seed_fine ^ 0x2545_F491, 3, lod_m);
    let texture_n = fbm3_band(p_m, 320.0, params.seed_fine ^ 0xC36E_1A91, 2, lod_m);
    let rolling = roll_lo * 0.040 + roll_hi * 0.014 + texture_n * 0.006;

    let height_m = relief * (0.92 + continent * 0.42 + hill_h + rolling);
    height_m.max(120.0)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OceanicSample {
    pub(crate) height_m: f32,
    pub(crate) material_id: u8,
    pub(crate) biome_mix: BiomeMix,
    pub(crate) roughness: f32,
}

const OCEANIC_MAT_ABYSSAL: u8 = 0;
const OCEANIC_MAT_SHELF_SEDIMENT: u8 = 1;
const OCEANIC_MAT_COASTAL_SAND: u8 = 2;
const OCEANIC_MAT_CONTINENTAL_SOIL: u8 = 3;
const OCEANIC_MAT_ROCK: u8 = 4;
const OCEANIC_MAT_SNOW: u8 = 5;

pub(crate) fn sample_oceanic_continental(
    params: OceanicContinentalParams,
    radius_m: f64,
    dir: DVec3,
    sample_scale_m: f32,
    intent_cache: Option<&Cubemap<f32>>,
) -> OceanicSample {
    let dir = dir.normalize_or_zero();
    if dir == DVec3::ZERO {
        return OceanicSample {
            height_m: 0.0,
            material_id: OCEANIC_MAT_CONTINENTAL_SOIL,
            biome_mix: BiomeMix::single(AGING_OCEANIC_BIOME_GRASSLAND),
            roughness: 0.78,
        };
    }

    let p_m = dir * radius_m.max(1.0);
    let lod_m = sample_scale_m.max(1.0);
    let relief = params.relief_scale_m;

    let warp_wl_m = 1_850_000.0;
    let warp = DVec3::new(
        fbm3_band(p_m, warp_wl_m, params.seed_warp, 4, lod_m) as f64,
        fbm3_band(p_m, warp_wl_m, params.seed_warp ^ 0xA53A_9E21, 4, lod_m) as f64,
        fbm3_band(p_m, warp_wl_m, params.seed_warp ^ 0xC2B2_AE35, 4, lod_m) as f64,
    ) * 420_000.0;
    let q_m = p_m + warp;

    // Continent intent: read the cached coarse kernel when available, else
    // evaluate it directly. The cache is bit-exact at its own texel centres and
    // ~lossless between them (the kernel has no content finer than ~100 km).
    let continent_base = match intent_cache {
        Some(cache) => cache.sample_bilinear(dir.as_vec3()),
        None => continent_kernel_base(params, dir),
    };
    let continent_shape = continent_shape_from_base(params, dir, continent_base, lod_m);
    let macro_n = fbm3_band(q_m, 1_100_000.0, params.seed_macro, 5, lod_m);
    let regional_n = fbm3_band(q_m, 420_000.0, params.seed_macro ^ 0x6D2B_79F5, 5, lod_m);
    let coast_n = fbm3_band(q_m, 125_000.0, params.seed_coast, 4, lod_m);
    let island_n = fbm3_band(q_m, 190_000.0, params.seed_islands, 4, lod_m);

    let archipelago = archipelago_chain_signal(params, dir, lod_m);
    let ocean_bias = (params.ocean_fraction - 0.62) * 0.45;
    let continent_edge = 1.0 - smoothstep(0.12, 0.42, continent_shape.abs());
    let continent_potential = continent_shape
        + macro_n * 0.08
        + regional_n * 0.18 * (0.35 + continent_edge * 0.65)
        + coast_n * 0.36 * continent_edge
        + archipelago
        + island_n.max(0.0) * 0.05 * smoothstep(-0.42, -0.05, continent_shape)
        - ocean_bias;

    let land = continent_potential;
    let coastness = 1.0 - smoothstep(0.018, 0.220, land.abs());

    let roll_lo = fbm3_band(p_m, 7_500.0, params.seed_fine ^ 0x9E37_79B9, 3, lod_m);
    let roll_hi = fbm3_band(p_m, 2_100.0, params.seed_fine ^ 0x2545_F491, 3, lod_m);
    let texture = fbm3_band(p_m, 520.0, params.seed_fine ^ 0xC36E_1A91, 2, lod_m);
    let rolling = relief * (roll_lo * 0.030 + roll_hi * 0.011 + texture * 0.004);

    if land <= 0.0 {
        let depth_t = smoothstep(0.0, 0.82, -land);
        let shelf_t = 1.0 - smoothstep(0.035, 0.23, -land);
        let abyssal_noise = fbm3_band(p_m, 95_000.0, params.seed_hills ^ 0xA0A0_5151, 4, lod_m);
        let ridge = eroded_ridged_band(
            p_m,
            180_000.0,
            params.seed_mountains ^ 0x0CE4_1111,
            4,
            0.52,
            2.0,
            0.6,
            lod_m,
        );
        let shelf_depth_m = 18.0 + smoothstep(0.0, 0.18, -land) * 360.0;
        let abyssal_depth_m = 1_250.0 + depth_t * 3_900.0;
        let depth_m = shelf_depth_m * shelf_t + abyssal_depth_m * (1.0 - shelf_t);
        let relief_m = abyssal_noise * 260.0 + ridge * 420.0;
        let height_m = (-depth_m + relief_m).min(-2.0);
        let shelf = height_m > -520.0;
        return OceanicSample {
            height_m,
            material_id: if shelf {
                OCEANIC_MAT_SHELF_SEDIMENT
            } else {
                OCEANIC_MAT_ABYSSAL
            },
            biome_mix: BiomeMix::from_weighted([
                (AGING_OCEANIC_BIOME_OCEAN, 1.0 - shelf_t),
                (AGING_OCEANIC_BIOME_SHELF, shelf_t),
            ]),
            roughness: if shelf { 0.64 } else { 0.76 },
        };
    }

    let inland = smoothstep(0.0, 0.62, land);
    let relief_control = smoothstep(
        0.12,
        0.58,
        fbm3_band(p_m, 240_000.0, params.seed_hills, 4, lod_m) * 0.82 + macro_n * 0.18,
    );
    let mountain_control = smoothstep(
        0.25,
        0.76,
        fbm3_band(p_m, 680_000.0, params.seed_mountains, 5, lod_m) + regional_n * 0.22,
    ) * inland;
    let mountain_ridges = eroded_ridged_band(
        p_m,
        58_000.0,
        params.seed_mountains ^ 0x5151_3737,
        6,
        0.50,
        2.05,
        1.25,
        lod_m,
    );
    let hill_ridges = eroded_ridged_band(
        p_m,
        5_200.0,
        params.seed_hills ^ 0xD1B5_4A32,
        6,
        0.50,
        2.08,
        1.45,
        lod_m,
    );

    let coastal_plain_m = 18.0 + smoothstep(0.0, 0.20, land) * 210.0;
    let continental_rise_m = inland.powf(0.72) * relief * 0.30;
    let hill_m = relief * relief_control * (hill_ridges * 0.28 + mountain_ridges * 0.08);
    let mountain_m = relief * mountain_control * (0.18 + mountain_ridges * 0.72);
    let height_m = coastal_plain_m + continental_rise_m + hill_m + mountain_m + rolling;
    let height_m = height_m.max(2.0);

    let latitude_abs = dir.y.clamp(-1.0, 1.0).asin().abs() as f32 / std::f32::consts::FRAC_PI_2;
    let roughness = (0.58 + relief_control * 0.14 + mountain_control * 0.16 + coastness * 0.04)
        .clamp(0.50, 0.94);
    let beach_w =
        (smoothstep(0.92, 0.995, coastness) + smoothstep(18.0, 2.0, height_m)).clamp(0.0, 1.0);
    let snow_w = (smoothstep(3_100.0, 4_900.0, height_m)
        + smoothstep(0.58, 0.86, latitude_abs) * smoothstep(2_600.0, 3_800.0, height_m))
    .clamp(0.0, 1.0);
    let rock_w = (smoothstep(2_000.0, 3_400.0, height_m)
        + smoothstep(0.55, 0.82, mountain_control))
    .clamp(0.0, 1.0)
        * (1.0 - snow_w * 0.65);
    let tundra_w = smoothstep(0.60, 0.86, latitude_abs) * (1.0 - snow_w * 0.85);
    let forest_w = smoothstep(0.16, 0.58, macro_n + island_n * 0.25)
        * smoothstep(0.0, 0.55, 1.0 - latitude_abs)
        * (1.0 - beach_w)
        * (1.0 - rock_w * 0.55)
        * (1.0 - snow_w);
    let grass_w = smoothstep(0.38, 0.02, relief_control)
        * smoothstep(-0.20, 0.18, -regional_n)
        * (1.0 - beach_w)
        * (1.0 - tundra_w * 0.50);
    let steppe_w = (1.0 - forest_w * 0.55 - grass_w * 0.35 - beach_w * 0.65).max(0.10)
        * (1.0 - snow_w * 0.55)
        * (1.0 - rock_w * 0.35);

    let material_id = if beach_w > 0.72 {
        OCEANIC_MAT_COASTAL_SAND
    } else if snow_w > 0.55 {
        OCEANIC_MAT_SNOW
    } else if rock_w > 0.58 {
        OCEANIC_MAT_ROCK
    } else {
        OCEANIC_MAT_CONTINENTAL_SOIL
    };
    let biome_mix = BiomeMix::from_weighted([
        (AGING_OCEANIC_BIOME_BEACH, beach_w),
        (AGING_OCEANIC_BIOME_SNOW, snow_w),
        (AGING_OCEANIC_BIOME_ROCK, rock_w),
        (AGING_OCEANIC_BIOME_TUNDRA, tundra_w),
        (AGING_OCEANIC_BIOME_FOREST, forest_w),
        (AGING_OCEANIC_BIOME_GRASSLAND, grass_w),
        (AGING_OCEANIC_BIOME_STEPPE, steppe_w),
    ]);

    OceanicSample {
        height_m,
        material_id,
        biome_mix,
        roughness,
    }
}

/// LOD-independent continent **intent** base ("continentalness" in `0..1`): the
/// expensive 12-shape + 6-seaway + macro topology that defines where land is. It
/// has no content finer than ~100 km, so this is the layer cached coarsely by
/// [`build_continent_intent_cache`] and reused across edits / shared by every
/// output texel instead of being recomputed per sample (it is ~80% of the field
/// eval cost). The LOD-gated coastline jitter is layered on by
/// [`continent_shape_from_base`].
fn continent_kernel_base(params: OceanicContinentalParams, dir: DVec3) -> f32 {
    // Legacy-Thalos-style macro topology: broad elongated continent fields
    // first, then warp the sea-level contour. This avoids both circular stamp
    // blobs and all-noise speckle; the 0.5 continentalness contour is what
    // becomes the coastline.
    const SHAPES: [ContinentShapeSeed; 12] = [
        // Edge-wrapping western continent and its southern shoulder.
        ContinentShapeSeed::new(-178.0, 4.0, 0.58, 0.42, 20.0, 0.62),
        ContinentShapeSeed::new(-136.0, -18.0, 0.78, 0.46, -54.0, 0.88),
        ContinentShapeSeed::new(-104.0, 18.0, 0.50, 0.30, 42.0, 0.46),
        // North-central mass, broken by seaways into the main reference shape.
        ContinentShapeSeed::new(-46.0, 22.0, 0.82, 0.38, 4.0, 0.86),
        ContinentShapeSeed::new(-4.0, 28.0, 0.64, 0.32, -28.0, 0.62),
        ContinentShapeSeed::new(24.0, 2.0, 0.46, 0.28, 54.0, 0.42),
        // Southern/central peninsula and island-continent around the main basin.
        ContinentShapeSeed::new(30.0, -28.0, 0.48, 0.30, -16.0, 0.42),
        ContinentShapeSeed::new(68.0, -12.0, 0.52, 0.32, 40.0, 0.48),
        // Eastern land complex and wrap-around continuation.
        ContinentShapeSeed::new(108.0, 22.0, 0.70, 0.38, -24.0, 0.70),
        ContinentShapeSeed::new(142.0, -8.0, 0.62, 0.34, 34.0, 0.58),
        ContinentShapeSeed::new(174.0, 16.0, 0.42, 0.28, -8.0, 0.34),
        ContinentShapeSeed::new(166.0, -28.0, 0.34, 0.22, 62.0, 0.26),
    ];

    const SEAWAYS: [ContinentShapeSeed; 6] = [
        // Elongated negative masks: major ocean cuts, gulfs, and straits.
        // These are what prevent the continent fields from reading as convex
        // blobs while avoiding the circular hole-punch artifact.
        ContinentShapeSeed::new(-82.0, 4.0, 0.34, 0.13, 8.0, 0.46),
        ContinentShapeSeed::new(10.0, 4.0, 0.30, 0.14, -34.0, 0.30),
        ContinentShapeSeed::new(86.0, 4.0, 0.42, 0.15, 14.0, 0.58),
        ContinentShapeSeed::new(-150.0, -7.0, 0.22, 0.10, 58.0, 0.20),
        ContinentShapeSeed::new(132.0, 4.0, 0.24, 0.11, -58.0, 0.22),
        ContinentShapeSeed::new(42.0, -34.0, 0.22, 0.10, 82.0, 0.22),
    ];

    let mut continentalness = 0.0_f32;
    for (i, shape) in SHAPES.iter().enumerate() {
        let warped = domain_warp_d(
            dir,
            params.seed_macro ^ (i as u32).wrapping_mul(0x57A7_EC01),
            0.9,
            0.16,
        );
        let local = continent_shape_continentalness(*shape, warped, params.seed_coast, i as u32);
        continentalness = 1.0 - (1.0 - continentalness) * (1.0 - local * shape.weight);
    }
    for (i, cut) in SEAWAYS.iter().enumerate() {
        let warped = domain_warp_d(
            dir,
            params.seed_coast ^ (i as u32).wrapping_mul(0xEA7E_5EA1),
            1.1,
            0.08,
        );
        let local = continent_shape_continentalness(*cut, warped, params.seed_warp, i as u32 + 37);
        continentalness -= local * cut.weight;
    }
    continentalness = continentalness.clamp(0.0, 1.0);

    let macro_warped = domain_warp_d(dir, params.seed_warp ^ 0xC0AC_7AA1, 0.7, 0.22);
    let macro_n = fbm3_f64(
        macro_warped.x * 1.5,
        macro_warped.y * 1.5,
        macro_warped.z * 1.5,
        params.seed_warp ^ 0xC0AC_7AB2,
        4,
        0.55,
        2.0,
    ) * 0.18;
    (continentalness + macro_n).clamp(0.0, 1.0)
}

/// Compose the base/cached continentalness with the LOD-gated coastline-jitter
/// detail (one warped fBM band at ~75 km, faded out at coarse LOD), then convert
/// `0..1` continentalness to signed land potential. Cut a little above 0.5 so
/// overlapping warped shapes form separate continents with flooded straits
/// instead of one continuous equatorial belt.
fn continent_shape_from_base(
    params: OceanicContinentalParams,
    dir: DVec3,
    base: f32,
    lod_m: f32,
) -> f32 {
    let weight = band_weight(75_000.0, lod_m);
    if weight <= 0.0 {
        // Coastline-jitter band below the sample LOD's Nyquist → no detail.
        return base - 0.60;
    }
    let d = base * 2.0 - 1.0;
    let coast_proximity = (1.0 - d * d).max(0.0);
    let detail_warped = domain_warp_d(dir, params.seed_coast ^ 0xC0DE_7A11, 4.0, 0.12);
    let detail = fbm3_f64(
        detail_warped.x * 16.0,
        detail_warped.y * 16.0,
        detail_warped.z * 16.0,
        params.seed_coast ^ 0xC0DE_7A12,
        5,
        0.55,
        2.0,
    ) * 0.42
        * coast_proximity
        * weight;
    (base + detail).clamp(0.0, 1.0) - 0.60
}

/// Coarse cubemap of [`continent_kernel_base`] — the cacheable continent
/// **intent** layer. The kernel's finest content is ~100 km, so a coarse cube
/// resolves it and bilinear sampling is effectively lossless at the scales that
/// matter. Building it once and reusing it across bakes skips the kernel — the
/// dominant (~80%) cost of the oceanic field eval. Parallel over texels.
pub fn build_continent_intent_cache(
    params: OceanicContinentalParams,
    resolution: u32,
) -> Cubemap<f32> {
    let res = resolution.max(1) as usize;
    let mut cache = Cubemap::<f32>::new(resolution.max(1));
    for face in CubemapFace::ALL {
        let data = cache.face_data_mut(face);
        data.par_iter_mut().enumerate().for_each(|(i, texel)| {
            let x = i % res;
            let y = i / res;
            let u = (x as f32 + 0.5) / res as f32;
            let v = (y as f32 + 0.5) / res as f32;
            let dir = face_uv_to_dir(face, u, v).as_dvec3();
            *texel = continent_kernel_base(params, dir);
        });
    }
    cache
}

fn archipelago_chain_signal(params: OceanicContinentalParams, dir: DVec3, lod_m: f32) -> f32 {
    // Localized island arcs, not full great-circle stripes. Each arc is a
    // short window on a great circle with noisy segmentation, so it creates
    // chains around ocean margins without drawing planet-spanning lines.
    let mut signal: f32 = 0.0;
    for i in 0..7u32 {
        let seed = params
            .seed_islands
            .wrapping_add(i.wrapping_mul(0x85EB_CA6B));
        let center = seeded_unit_vector(seed, i + 17);
        let raw_tangent = seeded_unit_vector(seed ^ 0x27D4_EB2D, i + 41);
        let tangent = (raw_tangent - center * raw_tangent.dot(center)).normalize_or_zero();
        if tangent == DVec3::ZERO {
            continue;
        }
        let normal = center.cross(tangent).normalize_or_zero();
        let along = dir.dot(tangent).asin() as f32;
        let cross = dir.dot(normal).asin().abs() as f32;
        let length = 0.30 + pseudo01(seed ^ 0xB529_7A4D) * 0.34;
        let width = 0.018 + pseudo01(seed ^ 0x68E3_1DA4) * 0.022;
        let band = smoothstep(width * 1.8, width * 0.35, cross);
        let window = smoothstep(length, length * 0.68, along.abs());
        let segment = fbm3_f64(
            dir.x * 12.0 + i as f64 * 1.7,
            dir.y * 12.0,
            dir.z * 12.0,
            seed ^ 0xC13F_A9A9,
            4,
            0.56,
            2.1,
        );
        let broken = smoothstep(-0.18, 0.50, segment);
        signal = signal.max(band * window * broken);
    }
    signal * 0.42 * band_weight(75_000.0, lod_m)
}

#[derive(Clone, Copy)]
struct ContinentShapeSeed {
    lon_deg: f32,
    lat_deg: f32,
    major_rad: f32,
    minor_rad: f32,
    axis_deg: f32,
    weight: f32,
}

impl ContinentShapeSeed {
    const fn new(
        lon_deg: f32,
        lat_deg: f32,
        major_rad: f32,
        minor_rad: f32,
        axis_deg: f32,
        weight: f32,
    ) -> Self {
        Self {
            lon_deg,
            lat_deg,
            major_rad,
            minor_rad,
            axis_deg,
            weight,
        }
    }
}

fn continent_shape_continentalness(
    shape: ContinentShapeSeed,
    dir: DVec3,
    seed: u32,
    index: u32,
) -> f32 {
    let center = lat_lon_dir(shape.lat_deg, shape.lon_deg);
    let (axis_major, axis_minor) = tangent_axes_for_shape(center, shape.axis_deg);
    let dot = center.dot(dir).clamp(-1.0, 1.0);
    let tangent = dir - center * dot;
    let x = tangent.dot(axis_major) as f32 / (shape.major_rad.sin()).max(0.04);
    let y = tangent.dot(axis_minor) as f32 / (shape.minor_rad.sin()).max(0.04);
    let d = (x * x + y * y).sqrt();

    let broad = fbm3_f64(
        dir.x * 3.1,
        dir.y * 3.1,
        dir.z * 3.1,
        seed ^ index.wrapping_mul(0xC041_57A7),
        4,
        0.55,
        2.0,
    );
    let fine = fbm3_f64(
        dir.x * 8.0,
        dir.y * 8.0,
        dir.z * 8.0,
        seed ^ index.wrapping_mul(0xC0A5_7011) ^ 0x5EA1_0001,
        3,
        0.50,
        2.1,
    );
    let edge = 1.0 + broad * 0.20 + fine * 0.07;
    (1.0 - smoothstep(edge * 0.78, edge * 1.12, d)).clamp(0.0, 1.0)
}

fn lat_lon_dir(lat_deg: f32, lon_deg: f32) -> DVec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let clat = lat.cos() as f64;
    DVec3::new(
        clat * lon.cos() as f64,
        lat.sin() as f64,
        clat * lon.sin() as f64,
    )
}

fn tangent_axes_for_shape(center: DVec3, axis_deg: f32) -> (DVec3, DVec3) {
    let north = DVec3::Y;
    let east = north.cross(center).normalize_or_zero();
    let north_tangent = (north - center * north.dot(center))
        .normalize_or_zero()
        .lerp(DVec3::X, 0.0);
    let angle = axis_deg.to_radians();
    let major =
        (east * angle.cos() as f64 + north_tangent * angle.sin() as f64).normalize_or_zero();
    let minor = center.cross(major).normalize_or_zero();
    (major, minor)
}

fn domain_warp_d(dir: DVec3, seed: u32, frequency: f32, strength: f32) -> DVec3 {
    let p = dir * frequency as f64;
    let warp = DVec3::new(
        fbm3_f64(p.x, p.y, p.z, seed, 3, 0.5, 2.0) as f64,
        fbm3_f64(p.x, p.y, p.z, seed ^ 0xA53A_9E1D, 3, 0.5, 2.0) as f64,
        fbm3_f64(p.x, p.y, p.z, seed ^ 0xC2B2_AE35, 3, 0.5, 2.0) as f64,
    );
    let tangent_warp = warp - dir * warp.dot(dir);
    (dir + tangent_warp * strength as f64).normalize_or_zero()
}

fn seeded_unit_vector(seed: u32, index: u32) -> DVec3 {
    DVec3::new(
        pseudo_signed(seed.wrapping_add(index.wrapping_mul(0x9E37_79B9))) as f64,
        pseudo_signed(seed.wrapping_add(index.wrapping_mul(0x85EB_CA6B)) ^ 0xB529_7A4D) as f64,
        pseudo_signed(seed.wrapping_add(index.wrapping_mul(0xC2B2_AE35)) ^ 0x68E3_1DA4) as f64,
    )
    .normalize_or_zero()
}

fn pseudo01(seed: u32) -> f32 {
    let h = splitmix64(seed as u64) >> 40;
    h as f32 / ((1u32 << 24) as f32)
}

fn pseudo_signed(seed: u32) -> f32 {
    pseudo01(seed) * 2.0 - 1.0
}

fn roughness_at(
    params: BasicContinentalParams,
    radius_m: f32,
    dir: Vec3,
    sample_scale_m: f32,
) -> f32 {
    let p_m = dir.normalize_or_zero().as_dvec3() * radius_m.max(1.0) as f64;
    let lod_m = sample_scale_m.max(1.0);
    let (_continent, relief_control) = continental_relief(params, p_m, lod_m);
    let local = fbm3_band(p_m, 1_800.0, params.seed_fine ^ 0x8D27_6B45, 3, lod_m).abs();
    // Hill country reads rougher than the plains.
    (0.74 + relief_control * 0.12 + local * 0.04).clamp(0.62, 0.95)
}

fn fbm3_band(p_m: DVec3, wavelength_m: f32, seed: u32, octaves: u32, lod_m: f32) -> f32 {
    let weight = band_weight(wavelength_m, lod_m);
    if weight <= 0.0 {
        // Band is below the sample LOD's Nyquist — its contribution is exactly
        // `noise * 0.0`. Skip the multi-octave fBM eval entirely so coarse /
        // orbital sampling does not pay for invisible detail. Bit-equivalent to
        // the old `fbm * 0.0` for the dominant (positive) sign; numerically
        // identical for all.
        return 0.0;
    }
    let wl = wavelength_m as f64;
    fbm3_f64(p_m.x / wl, p_m.y / wl, p_m.z / wl, seed, octaves, 0.53, 2.03) * weight
}

/// [`eroded_ridged_3d_f64`] for one wavelength band, LOD-gated like
/// [`fbm3_band`]: when the band is below the sample LOD's Nyquist its weight is
/// 0, so skip the (expensive) ridged eval entirely.
#[allow(clippy::too_many_arguments)]
fn eroded_ridged_band(
    p_m: DVec3,
    wavelength_m: f32,
    seed: u32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
    offset: f32,
    lod_m: f32,
) -> f32 {
    let weight = band_weight(wavelength_m, lod_m);
    if weight <= 0.0 {
        return 0.0;
    }
    eroded_ridged_3d_f64(
        p_m / wavelength_m as f64,
        seed,
        octaves,
        persistence,
        lacunarity,
        offset,
    ) * weight
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
