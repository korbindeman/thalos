use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Bulk composition as mass fractions. Fractions must sum to 1.0.
///
/// Stages meaningfully consume only `silicate`, `iron`, and `ice` in the
/// current pipeline, but all fields exist so the data model stays stable
/// as later stages are added.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
pub struct Composition {
    pub silicate: f64,
    pub iron: f64,
    pub ice: f64,
    pub volatiles: f64,
    pub hydrogen_helium: f64,
}

impl Composition {
    pub const SUM_TOLERANCE: f64 = 1e-6;

    pub fn new(silicate: f64, iron: f64, ice: f64, volatiles: f64, hydrogen_helium: f64) -> Self {
        let total = silicate + iron + ice + volatiles + hydrogen_helium;
        assert!(
            (total - 1.0).abs() < Self::SUM_TOLERANCE,
            "composition mass fractions must sum to 1.0, got {total}"
        );
        Self {
            silicate,
            iron,
            ice,
            volatiles,
            hydrogen_helium,
        }
    }
}

/// A discrete crater feature stored in the mid-frequency SSBO layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Crater {
    pub center: Vec3,
    pub radius_m: f32,
    pub depth_m: f32,
    pub rim_height_m: f32,
    pub age_gyr: f32,
    pub material_id: u32,
}

impl Crater {
    /// Outer influence radius for spatial indexing (ejecta blanket extent).
    /// McGetchin et al. (1973): ~90% of ejecta falls within 5R of crater center.
    pub fn influence_radius_m(&self) -> f32 {
        self.radius_m * 5.0
    }
}

/// A discrete volcanic feature.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Volcano {
    pub center: Vec3,
    pub radius_m: f32,
    pub height_m: f32,
    pub material_id: u32,
}

impl Volcano {
    pub fn influence_radius_m(&self) -> f32 {
        self.radius_m * 1.5
    }
}

/// A hand-anchored aeolian dune-sea region. Carries everything needed by both
/// the bake stage and the impostor's per-fragment dune synthesis.
///
/// Two morphological bands compose the visible dune signature
/// (see `docs/gen/dunes.md` §C.3):
/// - **Draa** (~2–5 km): rasterized into the height + albedo cubemaps at
///   bake time. Sun-shadowed silhouettes that read from orbit.
/// - **Dune** (~30–500 m): synthesized per fragment in the impostor shader
///   using `axis_tangent`, `lambda_dune_m`, `amplitude_dune_m`, `alpha_skew`.
///   Sub-cubemap-resolution on bodies the size of Vaelen — cannot be baked.
///
/// Per-region `axis_tangent` is constant for v1: parallel-transport drift
/// across a few-degree region on a 1130 km radius is small enough to ignore.
/// The full sphere wind field treatment lives in `dunes.md §C.1` if region
/// sizes ever grow.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DuneSea {
    /// Unit-vector direction of the region center on the body sphere.
    pub center: Vec3,
    /// Angular radius of the region's full-strength core, in radians.
    pub radius_rad: f32,
    /// Soft-edge feathering width in radians. Acts as the smoothstep
    /// half-width that fades dune contribution to zero past `radius_rad`.
    pub feather_rad: f32,
    /// Wind axis at the region center, expressed in the local tangent
    /// plane (unit vector tangent to `center`). Crests align perpendicular
    /// to this for a transverse field.
    pub axis_tangent: Vec3,

    /// Draa-scale wavelength. Baked into the height cubemap.
    pub lambda_draa_m: f32,
    /// Draa-scale amplitude.
    pub amplitude_draa_m: f32,

    /// Dune-scale wavelength. Synthesized per fragment in the impostor.
    pub lambda_dune_m: f32,
    /// Dune-scale amplitude.
    pub amplitude_dune_m: f32,

    /// Stoss-fraction of the asymmetric ridge in [0.5, 0.95]. ~0.85 gives
    /// a ~5.7:1 stoss/slip slope ratio matching dry granular media's
    /// angle of repose. See `docs/gen/dunes.md §B.2`.
    pub alpha_skew: f32,

    /// Anisotropic cross-wind warp amplitude, in unit-sphere coordinates.
    /// Displaces the sample point along `center × axis_tangent` by an
    /// fbm-driven amount before computing the wind phase. Crests stay
    /// spaced along the wind axis but meander cross-wind, producing
    /// the sinuous Namib-style read instead of straight bars. A value of
    /// ~0.4 × λ_draa / radius_m is a reasonable starting point.
    pub warp_amp_unit: f32,
    /// Spatial frequency of the warp fbm (cycles per unit-sphere-radius).
    /// Higher = tighter meanders.
    pub warp_freq: f32,

    /// Linear-RGB color the crest tint pulls toward. The bake stage and
    /// the impostor mix surface albedo toward this on dune crests. Per
    /// region so different bodies can express their own active-dune
    /// signature (e.g. saturated gold on a rust-desert, warm bone on a
    /// coastal beach, subtle frost on an icy moon).
    pub albedo_crest_lin: [f32; 3],
    /// Strength in [0, 1] of the crest-tint mix. 0 disables the tint
    /// entirely; the surrounding terrain color shows through unchanged.
    pub crest_strength: f32,

    /// Per-region seed for warp/jitter noise streams.
    pub seed: u64,
}

impl DuneSea {
    /// Outer angular extent for spatial indexing.
    pub fn influence_radius_rad(&self) -> f32 {
        self.radius_rad + self.feather_rad
    }
}

/// A linear/curved surface feature: rift, graben, ancient riverbed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Channel {
    pub points: Vec<Vec3>,
    pub width_m: f32,
    pub depth_m: f32,
    pub material_id: u32,
}

impl Channel {
    pub fn influence_radius_m(&self) -> f32 {
        self.width_m * 2.0
    }
}

/// A surface material, indexed by `material_id` on features.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Material {
    pub albedo: [f32; 3],
    pub roughness: f32,
}

/// Data-driven polar ice veneer parameters for a seasonal surface overlay.
/// The static compiler carries these in `DynamicSurfaceFeature` instead of
/// baking them into terrain cubemaps.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
pub struct IceCapSpec {
    /// Body-local spin axis. North caps grow toward `axis`, south caps grow
    /// toward `-axis`.
    #[serde(default = "default_ice_cap_axis")]
    pub axis: Vec3,
    #[serde(default = "default_ice_cap_include_pole")]
    pub north: bool,
    #[serde(default = "default_ice_cap_include_pole")]
    pub south: bool,
    /// Latitude where the cap starts fading in.
    #[serde(default = "default_ice_cap_edge_latitude_deg")]
    pub edge_latitude_deg: f32,
    /// Latitude where coverage reaches full strength.
    #[serde(default = "default_ice_cap_solid_latitude_deg")]
    pub solid_latitude_deg: f32,
    /// Low-frequency boundary warp in degrees so the edge does not read as a
    /// perfect circle of latitude.
    #[serde(default = "default_ice_cap_edge_noise_deg")]
    pub edge_noise_deg: f32,
    /// Contrast applied to the latitudinal coverage ramp. Higher values keep
    /// the lobed edge but make the ice/sand boundary read as a sharper cap.
    #[serde(default = "default_ice_cap_edge_sharpness")]
    pub edge_sharpness: f32,
    #[serde(default = "default_ice_cap_noise_frequency")]
    pub noise_frequency: f32,
    /// Maximum vertical veneer thickness at full coverage.
    #[serde(default = "default_ice_cap_max_thickness_m")]
    pub max_thickness_m: f32,
    #[serde(default = "default_ice_cap_albedo_linear")]
    pub albedo_linear: [f32; 3],
    #[serde(default = "default_ice_cap_dust_albedo_linear")]
    pub dust_albedo_linear: [f32; 3],
    /// Blend strength from existing albedo toward the ice/dust color.
    #[serde(default = "default_ice_cap_albedo_strength")]
    pub albedo_strength: f32,
    #[serde(default = "default_ice_cap_roughness")]
    pub roughness: f32,
    #[serde(default = "default_ice_cap_roughness_strength")]
    pub roughness_strength: f32,
    /// How much effective obliquity modulates this cap. `0` means the authored
    /// latitudes are literal; `1` uses the default climate response.
    #[serde(default = "default_ice_cap_obliquity_response")]
    pub obliquity_response: f32,
}

impl Default for IceCapSpec {
    fn default() -> Self {
        Self {
            axis: default_ice_cap_axis(),
            north: default_ice_cap_include_pole(),
            south: default_ice_cap_include_pole(),
            edge_latitude_deg: default_ice_cap_edge_latitude_deg(),
            solid_latitude_deg: default_ice_cap_solid_latitude_deg(),
            edge_noise_deg: default_ice_cap_edge_noise_deg(),
            edge_sharpness: default_ice_cap_edge_sharpness(),
            noise_frequency: default_ice_cap_noise_frequency(),
            max_thickness_m: default_ice_cap_max_thickness_m(),
            albedo_linear: default_ice_cap_albedo_linear(),
            dust_albedo_linear: default_ice_cap_dust_albedo_linear(),
            albedo_strength: default_ice_cap_albedo_strength(),
            roughness: default_ice_cap_roughness(),
            roughness_strength: default_ice_cap_roughness_strength(),
            obliquity_response: default_ice_cap_obliquity_response(),
        }
    }
}

/// Runtime surface feature that is intentionally not baked into the static
/// terrain cubemaps.
///
/// These descriptors travel with `BodyData` so the renderer/editor/runtime can
/// draw or simulate seasonal/changeable surface state without making it part
/// of the immutable geomorphology. Polar caps use this path because their
/// extent is seasonal; active dune overlays can move here once wind transport
/// becomes time-dependent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DynamicSurfaceFeature {
    SeasonalIceCap(IceCapSpec),
}

impl DynamicSurfaceFeature {
    pub fn kind_label(&self) -> &'static str {
        match self {
            Self::SeasonalIceCap(_) => "seasonal_ice_cap",
        }
    }
}

fn default_ice_cap_axis() -> Vec3 {
    Vec3::Y
}

fn default_ice_cap_include_pole() -> bool {
    true
}

fn default_ice_cap_edge_latitude_deg() -> f32 {
    68.0
}

fn default_ice_cap_solid_latitude_deg() -> f32 {
    82.0
}

fn default_ice_cap_edge_noise_deg() -> f32 {
    4.0
}

fn default_ice_cap_edge_sharpness() -> f32 {
    0.35
}

fn default_ice_cap_noise_frequency() -> f32 {
    2.0
}

fn default_ice_cap_max_thickness_m() -> f32 {
    220.0
}

fn default_ice_cap_albedo_linear() -> [f32; 3] {
    [0.82, 0.86, 0.88]
}

fn default_ice_cap_dust_albedo_linear() -> [f32; 3] {
    [0.56, 0.54, 0.50]
}

fn default_ice_cap_albedo_strength() -> f32 {
    0.86
}

fn default_ice_cap_roughness() -> f32 {
    0.48
}

fn default_ice_cap_roughness_strength() -> f32 {
    0.82
}

fn default_ice_cap_obliquity_response() -> f32 {
    1.0
}

/// Parameters for the high-frequency statistical detail noise layer.
/// Drives per-fragment crater synthesis in the shader.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetailNoiseParams {
    pub body_radius_m: f32,
    pub d_min_m: f32,
    pub d_max_m: f32,
    pub sfd_alpha: f32,
    pub global_k_per_km2: f32,
    pub d_sc_m: f32,
    pub body_age_gyr: f32,
    pub seed: u64,
}

impl Default for DetailNoiseParams {
    fn default() -> Self {
        Self {
            body_radius_m: 1.0,
            d_min_m: 0.0,
            d_max_m: 0.0,
            sfd_alpha: 2.0,
            global_k_per_km2: 0.0,
            d_sc_m: 1.0,
            body_age_gyr: 4.5,
            seed: 0,
        }
    }
}

/// Placeholder — drainage networks land with the Hydrology stage.
pub type DrainageNetwork = ();

/// Whether a plate carries buoyant felsic (continental) crust or dense mafic
/// (oceanic) crust. Drives everything downstream: subduction type, orogen
/// eligibility, ocean-floor-age applicability, base isostatic elevation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum PlateKind {
    Continental,
    Oceanic,
}

/// One tectonic plate on the sphere. Represented kinematically by an Euler
/// pole + angular velocity rather than by an integrated trajectory; boundary
/// motions and types are derived analytically from pairs of Euler poles at
/// the boundary midpoint. See `docs/gen/thalos_processes.md §Plates`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plate {
    pub id: u16,
    pub kind: PlateKind,
    /// Voronoi seed position (unit vector on the sphere).
    pub centroid: Vec3,
    /// Rotation axis of this plate relative to the mantle (unit vector).
    pub euler_pole: Vec3,
    /// Signed rotation rate around `euler_pole`, in rad/Myr. Sign encodes
    /// rotation sense.
    pub angular_velocity_rad_per_myr: f32,
}

/// Qualitative classification of a plate boundary's current motion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryKind {
    Convergent,
    Divergent,
    Transform,
}

/// A boundary segment between two plates. Attributes are populated by the
/// Tectonics stage after walking the plate adjacency graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Boundary {
    /// Ordered pair of plate IDs sharing this boundary.
    pub plates: (u16, u16),
    pub kind: BoundaryKind,
    /// Relative plate velocity magnitude at a representative midpoint, in
    /// m/Myr. Drives orogen intensity accumulation and ocean-floor spreading.
    pub relative_speed_m_per_myr: f32,
    /// Time this boundary has been in its current configuration (convergent /
    /// divergent / transform). Older configurations have had more time to
    /// build orogens or spread ocean floor.
    pub establishment_age_myr: f32,
    /// Whether this boundary is currently moving. In Thalos's declining era
    /// roughly 20% are active; the rest are stagnant but still carry their
    /// historical record.
    pub is_active: bool,
    /// Accumulated unitless orogen intensity over this boundary's active
    /// lifetime. Zero for divergent/transform and for non-continental-
    /// participating convergent boundaries.
    pub cumulative_orogeny: f32,
}

/// Global tectonic structure produced by the Plates + Tectonics stages.
///
/// `plate_id_cubemap` stores per-texel plate assignments; downstream stages
/// look up their cell's plate via `plate_id_cubemap.sample_nearest(dir)` and
/// index into `plates` by the returned id.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlateMap {
    pub plates: Vec<Plate>,
    pub boundaries: Vec<Boundary>,
    /// Per-texel plate ID. u16 covers more plates than we'd ever want.
    pub plate_id_cubemap: crate::cubemap::Cubemap<u16>,
}

/// Numeric identifier for a biome, indexing into `BodyBuilder::biomes`.
pub type BiomeId = u8;

/// A biome is a named region type with its own surface parameters. The
/// Biomes stage registers a palette of these and paints `BodyBuilder::biome_map`
/// with biome ids; downstream stages can later read the map to vary their
/// behavior per region (crater density, weathering rate, base albedo).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BiomeParams {
    pub name: String,
    /// Base linear albedo for mature (fully space-weathered) surface.
    pub albedo: f32,
    /// Base linear albedo for fresh (recently exposed) surface. Used by
    /// SpaceWeather as the target color for crater rims, ejecta, and rays.
    /// Defaults to `albedo * 1.9` if omitted.
    #[serde(default)]
    pub fresh_albedo: Option<f32>,
    /// RGB tint (multiplicative, linear). A per-biome chromatic signature —
    /// e.g. anorthosite slightly cool, KREEP slightly warm. Small values
    /// (~0.02–0.08 deviation from 1.0) keep Moon-like realism. Defaults to
    /// (1,1,1) if omitted.
    #[serde(default = "default_tint")]
    pub tint: [f32; 3],
    /// Amplitude of low-freq tonal variation (±fraction) on top of base
    /// albedo. Default 0.18 matches the previous single-biome behavior.
    #[serde(default = "default_tonal")]
    pub tonal_amp: f32,
    /// Surface roughness (affects shading model).
    pub roughness: f32,
    /// Fraction of bare ground visible through this biome's surface
    /// cover. 1.0 = fully exposed regolith (desert, badlands, tundra).
    /// 0.0 = full canopy hiding the substrate (closed-canopy forest).
    /// Used as a multiplier on `IronOverlay` strength so iron-rich
    /// catchments stain visibly through deserts but stay buried under
    /// rainforest. Default 1.0 keeps existing palettes back-compatible.
    #[serde(default = "default_iron_visibility")]
    pub iron_visibility: f32,
}

fn default_tint() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
fn default_tonal() -> f32 {
    0.18
}
fn default_iron_visibility() -> f32 {
    1.0
}
