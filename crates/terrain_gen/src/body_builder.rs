use crate::body_data::BodyData;
use crate::cubemap::{CubemapAccumulator, default_resolution};
use crate::spatial_index::IcoBuckets;
use crate::types::{
    Channel, Composition, Crater, DetailNoiseParams, DrainageNetwork, Material, PlateMap, Volcano,
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

    /// Accumulating cubemap contributions before bake.
    pub height_contributions: CubemapAccumulator,
    pub albedo_contributions: CubemapAccumulator,

    /// Mid-frequency feature lists (stages append to these).
    pub craters: Vec<Crater>,
    pub volcanoes: Vec<Volcano>,
    pub channels: Vec<Channel>,

    /// High-frequency detail parameters.
    pub detail_params: DetailNoiseParams,

    /// Materials palette.  Immutable after Differentiate stage.
    pub materials: Vec<Material>,

    /// Optional global structures.
    pub plates: Option<PlateMap>,
    pub drainage: Option<DrainageNetwork>,

    /// Per-stage seed, set by the pipeline runner before each stage.
    pub(crate) stage_seed: u64,
}

impl BodyBuilder {
    /// Create a new builder with default (empty) state.
    ///
    /// If `cubemap_resolution` is 0, a default is computed from `radius_m`.
    pub fn new(radius_m: f32, seed: u64, composition: Composition, cubemap_resolution: u32) -> Self {
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
            height_contributions: CubemapAccumulator::new(resolution),
            albedo_contributions: CubemapAccumulator::new(resolution),
            craters: Vec::new(),
            volcanoes: Vec::new(),
            channels: Vec::new(),
            detail_params: DetailNoiseParams::default(),
            materials: Vec::new(),
            plates: None,
            drainage: None,
            stage_seed: 0,
        }
    }

    /// The per-stage seed set by the pipeline runner.  Stages should use this
    /// (or derive sub-seeds from it) for all RNG, not `self.seed` directly.
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

        let feature_index = IcoBuckets::build(
            &self.craters,
            &self.volcanoes,
            &self.channels,
            self.radius_m,
        );

        BodyData {
            radius_m: self.radius_m,
            height_cubemap,
            height_range,
            albedo_cubemap,
            craters: self.craters,
            volcanoes: self.volcanoes,
            channels: self.channels,
            feature_index,
            detail_params: self.detail_params,
            materials: self.materials,
            plates: self.plates,
            drainage: self.drainage,
        }
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
        let builder = BodyBuilder::new(869_000.0, 42, test_composition(), 0);
        assert_eq!(builder.cubemap_resolution, 512);

        let body = builder.build();
        assert_eq!(body.radius_m, 869_000.0);
        assert!(body.craters.is_empty());
        assert!(body.volcanoes.is_empty());
        assert!(body.channels.is_empty());
        assert!(body.materials.is_empty());
        assert!(body.plates.is_none());
        assert!(body.drainage.is_none());
    }

    #[test]
    fn explicit_resolution_used() {
        let builder = BodyBuilder::new(100.0, 1, test_composition(), 64);
        assert_eq!(builder.cubemap_resolution, 64);
    }
}
