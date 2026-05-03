use serde::{Deserialize, Serialize};

use crate::cubemap::Cubemap;
use crate::spatial_index::IcoBuckets;
use crate::types::{Channel, Crater, DetailNoiseParams, Material, Volcano};

/// Immutable, GPU-facing surface data for a celestial body.
///
/// Produced by `BodyBuilder::build()` after all pipeline stages have run.
/// The renderer reads this; it never runs the pipeline.
#[derive(Serialize, Deserialize)]
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
    /// Per-texel material index into `materials`. R8Uint. Stages write this
    /// directly (MareFlood marks flooded regions with MAT_MARE; everything
    /// else stays at the initial MAT_HIGHLAND).
    pub material_cubemap: Cubemap<u8>,
    /// Per-texel surface roughness (R8Unorm; `roughness = byte / 255.0`).
    /// Consumed by the impostor shader for PBR microsurface response.
    pub roughness_cubemap: Cubemap<u8>,
    /// Per-texel object-space normal (RGBA8, linear; alpha unused).
    /// Decoding: `n = (texel.rgb * 2.0 - 1.0)`. Sample as `Rgba8Unorm`,
    /// not sRGB. Already includes height-derived bumps + any anisotropic
    /// perturbation the field provided, so the shader does not need to
    /// finite-difference the height cube at runtime.
    pub normal_cubemap: Cubemap<[u8; 4]>,

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

    /// Linear-RGB mean of the baked `albedo_cubemap`. Used by consumers
    /// that need a single tint for the body without reading the cubemap
    /// — e.g. planetshine illuminating a moon from its parent. Computed
    /// once at `BodyBuilder::build()` so callers don't reimplement the
    /// average.
    pub mean_albedo: [f32; 3],

    /// Sea level above the heightfield reference. Bodies with this set
    /// render the impostor's water BRDF wherever `height < sea_level`;
    /// airless bodies leave it `None`.
    pub sea_level_m: Option<f32>,
}
