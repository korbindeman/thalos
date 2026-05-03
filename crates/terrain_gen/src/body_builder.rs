use glam::Vec3;

use crate::body_data::BodyData;
use crate::cubemap::{Cubemap, CubemapAccumulator, default_resolution};
use crate::spatial_index::IcoBuckets;
use crate::stages::{BasinDef, MAT_HIGHLAND};
use crate::types::{
    BiomeParams, Channel, Composition, Crater, DetailNoiseParams, Material, Volcano,
};

/// Mutable build-time state for surface generation.
///
/// Stages mutate this during pipeline execution.  After all stages run,
/// `build()` finalizes accumulators into immutable cubemaps and produces
/// the GPU-facing `BodyData`.
pub struct BodyBuilder {
    pub radius_m: f32,
    pub seed: u64,
    pub composition: Composition,
    pub cubemap_resolution: u32,
    /// Body age in Gyr. Used by Cratering for age distribution and by the
    /// shader detail layer. Single source of truth across stages.
    pub body_age_gyr: f32,
    /// Unit vector toward the parent body for tidally locked moons. `None`
    /// for non-locked bodies. Stages requiring near/far asymmetry read this
    /// instead of carrying their own per-stage axis.
    pub tidal_axis: Option<Vec3>,
    /// Axial tilt in radians. Used by biome seeding to bias polar regions.
    /// Defaults to 0 when not plumbed through.
    pub axial_tilt_rad: f32,

    /// Accumulating cubemap contributions before bake.
    pub height_contributions: CubemapAccumulator,
    pub albedo_contributions: CubemapAccumulator,
    /// Signed multiplicative albedo modulation from megabasin-scale features.
    /// Per-texel: negative darkens (basin interior, eroded lobes), positive
    /// brightens (ejecta apron, anorthositic melt sheets). Read by the
    /// SpaceWeather base pass: `albedo *= 1.0 + basin_albedo_field`.
    ///
    /// Megabasin writes this with soft noise-warped boundaries and Imbrium-
    /// style radial ejecta sculpture so basins don't read as the hard discs
    /// a clean biome-id threshold produces.
    pub basin_albedo_field: Cubemap<f32>,
    /// Per-texel material index (R8). Initialized to MAT_HIGHLAND at builder
    /// construction; stages overwrite as needed (MareFlood flips flooded
    /// regions to MAT_MARE, future cryovolcanism overwrites with ice, etc.).
    /// Finalized into `BodyData::material_cubemap` without any transformation
    /// — stages see the same buffer the GPU will see.
    pub material_cubemap: Cubemap<u8>,
    /// Per-texel surface roughness (0..1, encoded R8Unorm). Default ~0.85
    /// at construction; written by `bake_surface_field_into_builder` (or any
    /// future stage). Consumed by the impostor shader for the PBR microsurface
    /// term.
    pub roughness_cubemap: Cubemap<u8>,
    /// Per-texel object-space normal (RGBA8, alpha unused). Default per-texel
    /// outward direction so flat-sphere impostors look correct without a
    /// `SurfaceField` bake; overwritten by the bake routine when present.
    /// Encoding: `(n * 0.5 + 0.5) * 255`. Stored linear, sample as `Rgba8Unorm`.
    pub normal_cubemap: Cubemap<[u8; 4]>,
    /// Cutoff below which craters stay SSBO-only; at-or-above, Cratering
    /// rasterizes into the cubemap. Written by the Cratering stage from its
    /// own parameter; the sampler and shader read it from BodyData to avoid
    /// double-counting baked craters.
    pub cubemap_bake_threshold_m: f32,

    /// Mid-frequency feature lists (stages append to these).
    pub craters: Vec<Crater>,
    pub volcanoes: Vec<Volcano>,
    pub channels: Vec<Channel>,
    /// Megabasin definitions written by the Megabasin stage and read by
    /// later stages (MareFlood selects flood targets from these).
    pub megabasins: Vec<BasinDef>,

    /// High-frequency detail parameters.
    pub detail_params: DetailNoiseParams,

    /// Materials palette.  Immutable after Differentiate stage.
    pub materials: Vec<Material>,

    /// Biome palette. Registered by the Biomes stage; indexed by `biome_map`.
    pub biomes: Vec<BiomeParams>,
    /// Per-texel biome assignment (R8, indexes `biomes`). Defaults to 0 at
    /// construction; the Biomes stage paints it.
    pub biome_map: Cubemap<u8>,

    /// Sea level above which texels render as land. Set by the Ocean
    /// terrain config or by future hydrology work; `None` on airless bodies.
    pub sea_level_m: Option<f32>,

    /// Per-stage seed, set by the caller before each stage.
    pub stage_seed: u64,
}

impl BodyBuilder {
    /// Create a new builder with default (empty) state.
    ///
    /// If `cubemap_resolution` is 0, a default is computed from `radius_m`.
    pub fn new(
        radius_m: f32,
        seed: u64,
        composition: Composition,
        cubemap_resolution: u32,
        body_age_gyr: f32,
        tidal_axis: Option<Vec3>,
        axial_tilt_rad: f32,
    ) -> Self {
        let resolution = if cubemap_resolution == 0 {
            default_resolution(radius_m)
        } else {
            cubemap_resolution
        };

        Self {
            radius_m,
            seed,
            composition,
            cubemap_resolution: resolution,
            body_age_gyr,
            tidal_axis,
            axial_tilt_rad,
            height_contributions: CubemapAccumulator::new(resolution),
            albedo_contributions: CubemapAccumulator::new(resolution),
            basin_albedo_field: Cubemap::<f32>::new(resolution),
            material_cubemap: {
                let mut mat = Cubemap::<u8>::new(resolution);
                for face in crate::cubemap::CubemapFace::ALL {
                    for v in mat.face_data_mut(face) {
                        *v = MAT_HIGHLAND as u8;
                    }
                }
                mat
            },
            roughness_cubemap: {
                // 0.85 is "moderately rough" — fits regolith / dry rock /
                // sand. Bodies with a SurfaceField bake overwrite per texel.
                let default = crate::surface_field::quantize_unit_to_u8(0.85);
                let mut r = Cubemap::<u8>::new(resolution);
                for face in crate::cubemap::CubemapFace::ALL {
                    for v in r.face_data_mut(face) {
                        *v = default;
                    }
                }
                r
            },
            normal_cubemap: crate::surface_field::default_normal_cubemap(resolution),
            // Defaults to +∞ so "nothing gets baked, everything goes to SSBO"
            // until a Cratering stage runs and sets its real value.
            cubemap_bake_threshold_m: f32::INFINITY,
            craters: Vec::new(),
            volcanoes: Vec::new(),
            channels: Vec::new(),
            megabasins: Vec::new(),
            detail_params: DetailNoiseParams {
                body_radius_m: radius_m,
                body_age_gyr,
                ..DetailNoiseParams::default()
            },
            materials: Vec::new(),
            biomes: Vec::new(),
            biome_map: Cubemap::<u8>::new(resolution),
            sea_level_m: None,
            stage_seed: 0,
        }
    }

    /// The per-stage seed set by the caller. Stages should use this (or
    /// derive sub-seeds from it) for all RNG, not `self.seed` directly.
    pub fn stage_seed(&self) -> u64 {
        self.stage_seed
    }

    /// Finalize into immutable `BodyData`.
    ///
    /// Bakes accumulators into cubemaps, builds the spatial index, and drops
    /// build-time-only fields (composition, seed, scratch state).
    pub fn build(self) -> BodyData {
        let (height_cubemap, height_range) = self.height_contributions.finalize_height();
        let albedo_cubemap = self.albedo_contributions.finalize_albedo();
        let mean_albedo = mean_albedo_linear(&albedo_cubemap);

        let feature_index = IcoBuckets::build(
            &self.craters,
            &self.volcanoes,
            &self.channels,
            self.radius_m,
        );

        BodyData {
            radius_m: self.radius_m,
            cubemap_bake_threshold_m: self.cubemap_bake_threshold_m,
            height_cubemap,
            height_range,
            albedo_cubemap,
            material_cubemap: self.material_cubemap,
            roughness_cubemap: self.roughness_cubemap,
            normal_cubemap: self.normal_cubemap,
            craters: self.craters,
            volcanoes: self.volcanoes,
            channels: self.channels,
            feature_index,
            detail_params: self.detail_params,
            materials: self.materials,
            mean_albedo,
            sea_level_m: self.sea_level_m,
        }
    }
}

/// Averages a baked sRGB8 albedo cubemap into a single linear-RGB tint.
///
/// All texels weighted equally — the cubemap-face → sphere area distortion
/// is small enough that planetshine, the only consumer, is insensitive
/// to it.
fn mean_albedo_linear(albedo: &Cubemap<[u8; 4]>) -> [f32; 3] {
    let mut sum = [0.0f64; 3];
    let mut count = 0u64;
    for face in crate::cubemap::CubemapFace::ALL {
        for texel in albedo.face_data(face) {
            sum[0] += srgb8_to_linear(texel[0]) as f64;
            sum[1] += srgb8_to_linear(texel[1]) as f64;
            sum[2] += srgb8_to_linear(texel[2]) as f64;
            count += 1;
        }
    }
    let n = count.max(1) as f64;
    [
        (sum[0] / n) as f32,
        (sum[1] / n) as f32,
        (sum[2] / n) as f32,
    ]
}

fn srgb8_to_linear(byte: u8) -> f32 {
    let s = byte as f32 / 255.0;
    if s <= 0.040_45 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_composition() -> Composition {
        Composition::new(0.9, 0.05, 0.0, 0.05, 0.0)
    }

    #[test]
    fn build_with_no_stages_produces_valid_body_data() {
        let builder = BodyBuilder::new(869_000.0, 42, test_composition(), 0, 4.5, None, 0.0);
        assert_eq!(builder.cubemap_resolution, 1024);

        let body = builder.build();
        assert_eq!(body.radius_m, 869_000.0);
        assert!(body.craters.is_empty());
        assert!(body.volcanoes.is_empty());
        assert!(body.channels.is_empty());
        assert!(body.materials.is_empty());
        assert!(body.sea_level_m.is_none());
    }

    #[test]
    fn explicit_resolution_used() {
        let builder = BodyBuilder::new(100.0, 1, test_composition(), 64, 4.5, None, 0.0);
        assert_eq!(builder.cubemap_resolution, 64);
    }
}
