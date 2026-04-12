use crate::cubemap::Cubemap;
use crate::spatial_index::IcoBuckets;
use crate::types::{
    Channel, Crater, DetailNoiseParams, DrainageNetwork, Material, PlateMap, Volcano,
};

/// Immutable, GPU-facing surface data for a celestial body.
///
/// Produced by `BodyBuilder::build()` after all pipeline stages have run.
/// The renderer reads this; it never runs the pipeline.
pub struct BodyData {
    pub radius_m: f32,

    /// Craters with `radius_m >= cubemap_bake_threshold_m` are rasterized
    /// into `height_cubemap` by the Cratering stage. The sampler and shader
    /// SSBO iteration paths must skip these to avoid double-counting the
    /// contribution (cubemap already holds it).
    pub cubemap_bake_threshold_m: f32,

    /// Low-frequency baked height layer.  R16Unorm encoding:
    /// `real_meters = (texel / 65535 * 2 - 1) * height_range`.
    pub height_cubemap: Cubemap<u16>,
    /// The ± range in meters that the height cubemap encodes.
    pub height_range: f32,
    /// Low-frequency baked albedo layer.  sRGB RGBA8.
    pub albedo_cubemap: Cubemap<[u8; 4]>,
    /// Per-texel material index into `materials`. R8Uint. Replaces the old
    /// albedo-alpha hack as the source of truth for highland/mare tagging.
    /// Stages write this directly (MareFlood marks flooded regions with
    /// MAT_MARE; everything else stays at the initial MAT_HIGHLAND).
    pub material_cubemap: Cubemap<u8>,

    /// Mid-frequency discrete features.
    pub craters: Vec<Crater>,
    pub volcanoes: Vec<Volcano>,
    pub channels: Vec<Channel>,

    /// Shared spatial index over all feature arrays.
    pub feature_index: IcoBuckets,

    /// High-frequency statistical detail noise parameters.
    pub detail_params: DetailNoiseParams,

    /// Materials palette, indexed by `material_id` on features.
    pub materials: Vec<Material>,

    /// Optional global structures from coherent stages.
    pub plates: Option<PlateMap>,
    pub drainage: Option<DrainageNetwork>,
}
