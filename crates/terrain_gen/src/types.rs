use glam::Vec3;
use serde::Deserialize;

/// Bulk composition as mass fractions. Fractions must sum to 1.0.
///
/// Stages meaningfully consume only `silicate`, `iron`, and `ice` in the
/// current pipeline, but all fields exist so the data model stays stable
/// as later stages are added.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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

/// A linear/curved surface feature: rift, graben, ancient riverbed.
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
pub struct Material {
    pub albedo: [f32; 3],
    pub roughness: f32,
}

/// Parameters for the high-frequency statistical detail noise layer.
/// Drives per-fragment crater synthesis in the shader.
#[derive(Clone, Debug)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum PlateKind {
    Continental,
    Oceanic,
}

/// One tectonic plate on the sphere. Represented kinematically by an Euler
/// pole + angular velocity rather than by an integrated trajectory; boundary
/// motions and types are derived analytically from pairs of Euler poles at
/// the boundary midpoint. See `docs/gen/thalos_processes.md §Plates`.
#[derive(Clone, Debug)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryKind {
    Convergent,
    Divergent,
    Transform,
}

/// A boundary segment between two plates. Attributes are populated by the
/// Tectonics stage after walking the plate adjacency graph.
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug, Deserialize)]
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
}

fn default_tint() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
fn default_tonal() -> f32 {
    0.18
}
