use serde::{Deserialize, Serialize};

use crate::cubemap::Cubemap;
use crate::drainage::DrainageGraph;
use crate::icosphere::Icosphere;
use crate::province::ProvinceDef;
use crate::spatial_index::IcoBuckets;
use crate::types::{
    Channel, Crater, DetailNoiseParams, DrainageNetwork, Material, PlateMap, Volcano,
};

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
    /// Per-texel material index into `materials`. R8Uint. Replaces the old
    /// albedo-alpha hack as the source of truth for highland/mare tagging.
    /// Stages write this directly (MareFlood marks flooded regions with
    /// MAT_MARE; everything else stays at the initial MAT_HIGHLAND).
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

    /// Optional global structures from coherent stages.
    pub plates: Option<PlateMap>,
    pub drainage: Option<DrainageNetwork>,

    /// Icosphere mesh used by the Thalos pipeline as its canonical working
    /// representation. `None` on bodies whose pipeline is cubemap-only
    /// (Mira-family).
    pub sphere: Option<Icosphere>,
    /// Per-vertex province ID on `sphere`, produced by `TectonicSkeleton`.
    pub vertex_provinces: Option<Vec<u32>>,
    /// Per-vertex "home craton" province ID on `sphere` — the Voronoi
    /// craton this vertex was assigned to before any boundary rewrite.
    /// Lets downstream stages read the continental-vs-oceanic side of a
    /// boundary vertex.
    pub vertex_craton_provinces: Option<Vec<u32>>,
    /// Province table, indexed by `ProvinceDef::id`.
    pub provinces: Vec<ProvinceDef>,

    /// Per-vertex elevation in meters on `sphere`, produced by
    /// `CoarseElevation` (Stage 2) and refined by `HydrologicalCarving`
    /// (Stage 3). Reference elevation is nominal zero.
    pub vertex_elevations_m: Option<Vec<f32>>,
    /// Per-vertex sediment thickness in meters, produced by Stage 3.
    pub vertex_sediment_m: Option<Vec<f32>>,
    /// Drainage graph produced by Stage 3. Persistent; consumed by
    /// gameplay code (navigable rivers, settlement placement).
    pub drainage_graph: Option<DrainageGraph>,
    /// Sea level relative to the elevation reference (nominal zero).
    /// Picked by Stage 3 to hit the target ocean fraction.
    pub sea_level_m: Option<f32>,
}
