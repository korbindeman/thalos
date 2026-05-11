//! Continuous surface field for the `AgingOceanicHomeworld` archetype.
//!
//! ## How plate-guided coastlines emerge
//!
//! The tectonic graph (Voronoi mesh + plate kinds + boundary records) is the
//! *prior*: it says "ocean roughly here, land roughly there, mountain belt
//! roughly along this line." The visible terrain is **never** the cell mesh
//! — that's a step-function and would read as Voronoi-blocky from any
//! distance. The visible terrain is a continuous 3-term sum:
//!
//! 1. **Smooth prior elevation**, computed from the cells via K-nearest
//!    inverse-distance-weighted blending. `continentalness ∈ [0, 1]` combines
//!    a smooth plate prior with a separate continent-shape field. Continental
//!    plates are structural scaffolding, not fully exposed land polygons; the
//!    visible continents are low-frequency, elongated masks seeded from that
//!    scaffold. The elevation base is
//!    `lerp(OCEANIC_BASE, CONTINENTAL_BASE, continentalness)`. Both quantities
//!    transition smoothly across cell edges.
//!
//! 2. **Smooth boundary contribution**, gating mountain/trench/ridge/rift by
//!    a smoothed boundary-distance field (also K-nearest blended) and a
//!    `continentalness`-driven smoothstep that picks the right kind for the
//!    local geology — convergent in continent → mountains; convergent in
//!    ocean → trench; transitional → arc.
//!
//! 3. **Low-frequency coastline shaping**. Continental plates are not
//!    treated as solid above-water polygons: broad breakup noise pushes
//!    seaways, shelves, peninsulas, and uplands through the smooth plate
//!    prior. Fine shoreline texture is deliberately left out here; the
//!    structural pass should not create noisy, fuzzy coastlines.
//!
//! 4. **Bake-time ocean connectivity cleanup**. The continuous field is local
//!    by design, so fBM can push small continental basins below sea level.
//!    `enforce_single_connected_ocean` runs after cubemap bake, keeps the
//!    largest connected below-sea-level component, and dry-fills every other
//!    water component into low coastal plain. This gives Thalos one global
//!    ocean without inventing isolated, hydrologically impossible seas.
//!
//! Cells guide the structure (where continents and mountain belts live);
//! low-frequency shaping gives the coastlines room to diverge from the
//! plate graph. Neither alone is enough — discrete cells produce blocky
//! outputs; pure noise produces no coherent continents.
//!
//! ## What's deferred
//!
//! High-frequency detail (rivers, individual peaks, cliff faces) is meant to
//! come from the user's downstream erosion-filter pass on a separate fBM
//! heightmap. This field's job is the *structural* shape — continent
//! placement, mountain-belt geometry, coastline contour. The two layers
//! compose: this layer's smooth-and-fractal output gives erosion something
//! plausible to operate on.

use glam::Vec3;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{Cubemap, CubemapFace, dir_to_face_uv, face_uv_to_dir};
use crate::noise::fbm3;
use crate::seeding::{Rng, sub_seed};
use crate::surface_color::{
    AGING_OCEANIC_BIOME_BEACH, AGING_OCEANIC_BIOME_BOREAL, AGING_OCEANIC_BIOME_DESERT,
    AGING_OCEANIC_BIOME_FOREST, AGING_OCEANIC_BIOME_GRASSLAND, AGING_OCEANIC_BIOME_OCEAN,
    AGING_OCEANIC_BIOME_ROCK, AGING_OCEANIC_BIOME_SHELF, AGING_OCEANIC_BIOME_SNOW,
    AGING_OCEANIC_BIOME_STEPPE, AGING_OCEANIC_BIOME_TUNDRA,
};
use crate::surface_field::{
    BiomeMix, SurfaceField, SurfaceFieldSample, SurfaceMaterialMix, cube_face_texel_scale_m, mix3,
    quantize_unit_to_u8, smoothstep,
};
use crate::tectonics::{BoundaryKind, PlateKind, TectonicSystem};

// Material IDs published in the AgingOceanicHomeworld palette. The
// dominant id only feeds CPU samplers; the impostor reads the baked
// albedo cubemap directly.
pub const AGING_OCEANIC_MAT_ABYSSAL: u8 = 0;
pub const AGING_OCEANIC_MAT_CONTINENTAL_LOW: u8 = 1;
pub const AGING_OCEANIC_MAT_CONTINENTAL_HIGH: u8 = 2;
pub const AGING_OCEANIC_MAT_BEACH: u8 = 3;
pub const AGING_OCEANIC_MAT_PEAK: u8 = 4;

// ---------------------------------------------------------------------------
// Prior elevation: plate-kind base levels
// ---------------------------------------------------------------------------

/// Base elevation of an oceanic plate interior, in meters. Earth's abyssal
/// plain averages around -4000 m; biased a touch shallower so divergent
/// ridge bumps still read above-base without crowding sea level.
const OCEANIC_BASE_M: f32 = -3500.0;

/// Base elevation of continental plate interiors, before boundary and
/// coastline shaping. This is deliberately close to the old field: plates
/// define broad continental crust, not the final shoreline contour.
const CONTINENTAL_BASE_M: f32 = 420.0;

// ---------------------------------------------------------------------------
// Cell smoothing
// ---------------------------------------------------------------------------

/// Number of nearest cells to blend per sample. K=9 lets ragged BFS plate
/// edges leak into the smoothed continentalness contour without showing
/// hard Voronoi edges, giving the silhouette natural irregularity that
/// the macro warp below then amplifies. Larger K (we ran K=13 historically)
/// over-rounds the continents into smooth circles; smaller K starts
/// exposing cell-edge wrinkles in the boundary distance field too.
const SMOOTHING_K: usize = 9;

/// Regularizer for inverse-distance weighting. At typical inter-cell angles
/// (~0.05 rad on Thalos at 2k cells, so d² ≈ 0.0025), this keeps weights
/// finite when the sample sits exactly on a cell center while still letting
/// the closest cells dominate the blend.
const SHEPARD_EPS_RAD2: f32 = 3.0e-4;

// ---------------------------------------------------------------------------
// Continent shape localization
// ---------------------------------------------------------------------------
//
// A tectonic plate can be long and awkwardly wrapped on the sphere. Treating
// every cell in a continental plate as above-water continental crust therefore
// tends to create meridional land stripes. The plate graph should decide the
// rough tectonic neighborhood of continental crust, but the visible continent
// outlines need their own low-frequency shape field.

/// Primary-continent major-axis radius before hydrosphere scaling.
const PRIMARY_CONTINENT_MAJOR_RAD: f32 = 0.78;

/// Secondary-continent major-axis radius before hydrosphere scaling.
const SECONDARY_CONTINENT_MAJOR_RAD: f32 = 0.62;

/// Maximum extra major-axis radius for the primary continent when the body
/// config asks it to grow faster. This keeps the authored "main continent"
/// control useful without letting a high multiplier become a global stripe.
const PRIMARY_CONTINENT_EXTRA_MAJOR_RAD: f32 = 0.14;

// ---------------------------------------------------------------------------
// Boundary contributions
// ---------------------------------------------------------------------------

/// Half-width of the boundary-falloff gaussian, in meters. At distance
/// equal to this, the contribution is `1/e ≈ 0.37` of peak.
const BOUNDARY_HALF_WIDTH_M: f32 = 250_000.0;

/// Peak amplitudes for boundary contributions. Convergent in deep continent
/// → mountain belt; convergent in deep ocean → trench; convergent in
/// transitional zone → island arc (mountain bump weakens, trench partial).
/// Divergent gets ridges in ocean and rift valleys on continent.
const MOUNTAIN_BUMP_M: f32 = 2_800.0;
const TRENCH_DEPTH_M: f32 = 1_250.0;
const RIDGE_BUMP_M: f32 = 650.0;
const RIFT_DEPTH_M: f32 = 350.0;
const TRANSFORM_JITTER_M: f32 = 90.0;

// ---------------------------------------------------------------------------
// Coastline shaping
// ---------------------------------------------------------------------------
//
// The tectonic prior decides where continental crust can exist. The coastline
// pass below cuts and extends the sea-level contour without replacing the
// broad plate-guided landform.

const COAST_MARGIN_MACRO_AMP_M: f32 = 1500.0;

/// Peak ridge bump added to continent interiors. Ridge-style fbm
/// concentrates amplitude near zero-crossings of the underlying noise,
/// producing oriented mountain ranges instead of uniformly rolling hills.
/// Tuned so interior ridge crests reach roughly the floor of the boundary
/// mountain belts: secondary relief that's clearly readable from orbit
/// without competing with belt mountains for visual dominance.
const INTERIOR_RIDGE_AMP_M: f32 = 1100.0;

/// Smooth fbm added on top of the ridge term so flanks and valley floors
/// aren't a single uniform elevation. Gives the impostor's per-fragment
/// height-derived normal something below the ridge band to read.
const INTERIOR_FILL_AMP_M: f32 = 220.0;

// ---------------------------------------------------------------------------
// Macro continentalness warp
// ---------------------------------------------------------------------------
//
// Domain-warped fBM modulation added to the smoothed continentalness *before*
// any height contributions read it. This is the key lever for breaking smooth
// circular continent silhouettes into Earth-like irregularity: the warp can
// push interior cells across the 0.5 sea-level contour, carving inland seas,
// archipelagos, and elongated peninsulas without invalidating the underlying
// tectonic structure. Mountain belts and trenches still anchor to plate
// edges; only the visible coastline shape diverges.
//
// The connectivity-cleanup pass dry-fills any below-sea-level component that
// isn't the largest one, so isolated inland seas the warp may carve get
// reclaimed as low coastal plain. That's intentional — it keeps Thalos's
// "single global ocean" topological invariant.

/// Peak warp added to (or subtracted from) continentalness. Continentalness
/// is a [0, 1] value; ±0.18 means a sample one octave-band off the plate
/// edge can swing across the coastline threshold. Values much above 0.20
/// start carving large interior basins that the connectivity cleanup
/// truncates visibly.
const MACRO_CONTINENT_WARP_AMP: f32 = 0.18;

/// Spatial frequency of the warp's fBM in cycles per radian. ~1.5 keeps
/// features at continent-scale; higher frequencies look like noisy
/// shoreline scallops instead of macro shape variation.
const MACRO_CONTINENT_WARP_FREQ: f32 = 1.5;

/// Domain-warp parameters applied *before* the warp fBM is sampled, so the
/// warp itself isn't axis-aligned and doesn't produce vertical/horizontal
/// streaks. Amplitude is in unit-sphere offsets.
const MACRO_CONTINENT_WARP_DOMAIN_FREQ: f32 = 0.7;
const MACRO_CONTINENT_WARP_DOMAIN_AMP: f32 = 0.22;

// ---------------------------------------------------------------------------
// Coastline detail
// ---------------------------------------------------------------------------
//
// The macro warp above breaks the smoothed-plate-circle silhouette at the
// 1000-km scale. This second pass adds *coastline-scale* fractal detail —
// the 30–200 km wiggle that makes a coastline look like a coastline rather
// than a smooth blob. Gated by proximity to the unwarped coast contour so
// it can't rip islands out of deep ocean or punch interior seas into
// continental hearts.
//
// Why this isn't just "more octaves on the macro warp": the macro warp is
// an **fBm with gain 0.55**, so each successive octave loses ~half its
// amplitude. By the 4th octave the high-frequency band has too little
// energy to push continentalness across the 0.5 sea-level threshold, so
// the visible silhouette stays smooth no matter how many octaves you
// stack. A separate term with its own (high) amplitude budget at high
// frequency is the only way to deliver the 30–200 km wiggle the
// coastline needs.

/// Peak coastline-detail wiggle in continentalness units. Larger than the
/// macro warp because it's gated to a narrow band around the coast — far
/// from the coast it tapers to zero, so the connectivity-cleanup pass
/// won't see big new disconnected basins. Tuned to push the silhouette
/// into Earth-like fingered peninsulas, deep bays, and small
/// archipelagos rather than smoothed-blob coasts.
const COASTLINE_DETAIL_AMP: f32 = 0.55;

/// Base spatial frequency of the coastline-detail fBm in cycles per radian.
/// 16 puts the lowest octave at ~200 km features on a 3186 km body. The
/// term lives in the coastline-scale band (12–200 km) where the macro
/// warp leaves off; the macro warp covers 0.5–4000 km from its own
/// multi-octave fBm. See `COASTLINE_DETAIL_OCTAVES` for the band layout
/// and the Nyquist behaviour at the top end.
const COASTLINE_DETAIL_FREQ: f32 = 16.0;

/// Coastline-detail fBm shape parameters. With 5 octaves at gain 0.55 the
/// bands run 16, 32, 64, 128, 256 cycles/rad → wavelengths 199, 99, 50,
/// 25, 12 km on a 3186 km body. The 5th octave technically sits past the
/// 512² Nyquist limit (~162 cycles/rad), but the geometric gain
/// attenuates it to ~4% of total amplitude — well below the perceptual
/// threshold where aliasing residue could read as fake structure. Net
/// effect is finer near-coast wiggle without visible artefacts. Lower
/// octaves still carry non-trivial weight (octave 3 ~14%) so the
/// coastline reads at every zoom from far orbit down to near approach.
const COASTLINE_DETAIL_OCTAVES: u32 = 5;
const COASTLINE_DETAIL_GAIN: f32 = 0.55;

/// Pre-warp domain parameters for the coastline detail. The
/// frequency-4 warp keeps the high-frequency fBm from looking
/// grid-aligned; the amplitude is the lever for how much the coast
/// outline swirls and curves rather than running radially out from
/// continent centroids — at 0.12 (≈ 60 km of tangential displacement)
/// peninsulas form curving fingers and bays carve back into the interior
/// at oblique angles rather than head-on.
const COASTLINE_DETAIL_DOMAIN_FREQ: f32 = 4.0;
const COASTLINE_DETAIL_DOMAIN_AMP: f32 = 0.12;

/// Isolated water components are raised to low, wet-looking coastal plain
/// rather than hard-clamped exactly to sea level. The old depth contributes a
/// little residual relief so former basins still read as broad lowlands.
const DRY_FILL_BASE_M: f32 = 30.0;
const DRY_FILL_DEPTH_RELIEF: f32 = 0.10;
const DRY_FILL_MAX_DEPTH_M: f32 = 650.0;

/// Roughness response at altitude. Water is very smooth; steep rocky
/// terrain is moderately rough; deep ocean roughness is dominated by the
/// shader's water BRDF anyway, but we publish a sane PBR fallback.
fn altitude_roughness(relative_height_m: f32) -> f32 {
    if relative_height_m < 0.0 {
        // Water surface — the impostor's water BRDF takes over here, but
        // publish a low value for any consumer that reads roughness
        // directly.
        0.05
    } else {
        // Land roughness ramps from beach (semi-smooth) to peaks (rocky).
        smoothstep(0.0, 3500.0, relative_height_m) * 0.5 + 0.45
    }
}

/// Slope-aware roughness: the altitude-only base plus an additive bump on
/// steep faces, where exposed rock and scree read rougher than the soil
/// or vegetation default. Slope thresholds match the rock-coloring band
/// in `surface_albedo` so the rough-cliff signal coincides with the
/// rock-color signal.
fn surface_roughness(relative_height_m: f32, slope: f32) -> f32 {
    let base = altitude_roughness(relative_height_m);
    if relative_height_m < 0.0 {
        return base;
    }
    // Slope band matches `surface_albedo`'s rock signal so rough texture
    // and rock color coincide. Altitudes already drive the base; the
    // slope bump is what pushes ridge flanks past the smooth-soil reading.
    let slope_t = smoothstep(0.003, 0.018, slope);
    (base + slope_t * 0.18).min(0.95)
}

/// Altitude-banded albedo: deep seabed → shallow seabed → coastal beach
/// → lowland → highland → snow peaks. Smooth blends keep the bands from
/// reading as hard color banding from orbit.
#[allow(dead_code)]
fn altitude_albedo(relative_height_m: f32) -> [f32; 3] {
    // Anchor colors. All linear-RGB. Tuned to read against the impostor's
    // water tint and atmospheric optics.
    let abyssal = [0.030, 0.045, 0.060];
    let shelf = [0.075, 0.115, 0.140];
    let beach = [0.660, 0.575, 0.405];
    let lowland = [0.260, 0.310, 0.180];
    let highland = [0.350, 0.290, 0.220];
    let peak = [0.880, 0.880, 0.900];

    let mut color = abyssal;
    color = mix3(color, shelf, smoothstep(-3500.0, -300.0, relative_height_m));
    color = mix3(color, beach, smoothstep(-200.0, 50.0, relative_height_m));
    color = mix3(color, lowland, smoothstep(40.0, 350.0, relative_height_m));
    color = mix3(
        color,
        highland,
        smoothstep(400.0, 1800.0, relative_height_m),
    );
    color = mix3(color, peak, smoothstep(2400.0, 4000.0, relative_height_m));
    color
}

/// Bilinear blend across four palette corners arranged as
/// `(dry,sparse) – (dry,dense) – (wet,sparse) – (wet,dense)`, indexed by
/// `lushness` (sparse → dense) and `moisture` (dry → wet). Used to give
/// each biome continuous tonal variation along two independent climate
/// axes instead of one flat fill colour.
#[allow(dead_code)]
fn bilinear_palette(
    dry_sparse: [f32; 3],
    dry_dense: [f32; 3],
    wet_sparse: [f32; 3],
    wet_dense: [f32; 3],
    lushness: f32,
    moisture: f32,
) -> [f32; 3] {
    let dry = mix3(dry_sparse, dry_dense, lushness);
    let wet = mix3(wet_sparse, wet_dense, lushness);
    mix3(dry, wet, moisture)
}

fn biome_band(value: f32, lo: f32, peak: f32, hi: f32) -> f32 {
    smoothstep(lo, peak, value) * (1.0 - smoothstep(peak, hi, value))
}

fn sharpen_biome_weight(weight: f32) -> f32 {
    weight.max(0.0).powf(1.25)
}

fn altitude_material(relative_height_m: f32) -> u8 {
    if relative_height_m < -100.0 {
        AGING_OCEANIC_MAT_ABYSSAL
    } else if relative_height_m < 50.0 {
        AGING_OCEANIC_MAT_BEACH
    } else if relative_height_m < 1800.0 {
        AGING_OCEANIC_MAT_CONTINENTAL_LOW
    } else if relative_height_m < 3000.0 {
        AGING_OCEANIC_MAT_CONTINENTAL_HIGH
    } else {
        AGING_OCEANIC_MAT_PEAK
    }
}

/// Smooth prior derived from K-nearest cells. All four fields are
/// continuous scalar functions of `dir`; nothing here is a Voronoi step.
/// The three boundary-kind weights sum to 1 when there's any boundary
/// nearby (and to 0 in single-plate edge cases).
struct SmoothCell {
    /// Weighted fraction of nearby cells that are continental. 0 = deep
    /// ocean, 1 = deep continent, 0.5 = at the smooth coastline.
    continentalness: f32,
    /// Distance to the nearest plate boundary, IDW-smoothed across cells
    /// so the gaussian falloff doesn't show cell-edge banding.
    boundary_distance_m: f32,
    /// Soft membership in each boundary kind, weighted by IDW so the
    /// transitions between regimes (mountain / trench / ridge / rift /
    /// transform) are continuous instead of cell-quantized.
    convergent_w: f32,
    divergent_w: f32,
    transform_w: f32,
}

#[derive(Clone, Debug)]
struct CoastalEmbayment {
    center_dir: Vec3,
    axis_inland: Vec3,
    axis_alongshore: Vec3,
    inland_rad: f32,
    alongshore_rad: f32,
    depth_m: f32,
    edge_seed: u32,
}

#[derive(Clone, Debug)]
struct ContinentShape {
    center_dir: Vec3,
    axis_major: Vec3,
    axis_minor: Vec3,
    major_rad: f32,
    minor_rad: f32,
    edge_seed: u32,
}

fn build_continent_shapes(
    tectonics: &TectonicSystem,
    seed: u64,
    ocean_fraction: f32,
) -> Vec<ContinentShape> {
    let land_fraction = (1.0 - ocean_fraction.clamp(0.05, 0.95)).clamp(0.05, 0.75);
    let radius_scale = (land_fraction / 0.30).sqrt().clamp(0.70, 1.35);
    let primary_t = (tectonics.config.primary_size_multiplier.clamp(1.0, 4.0) - 1.0) / 3.0;

    let mut out = Vec::new();
    for plate in &tectonics.plates {
        if plate.kind != PlateKind::Continental {
            continue;
        }

        let plate_idx = plate.id.0 as usize;
        let seed_dir = tectonics.mesh.cells[plate.seed_cell as usize];
        let center = (seed_dir * 0.82 + plate.centroid_dir * 0.18)
            .try_normalize()
            .unwrap_or(seed_dir);

        let mut rng = Rng::new(seed ^ ((plate.id.0 as u64) << 32));
        let random_dir = {
            let d = rng.unit_vector();
            Vec3::new(d.x as f32, d.y as f32, d.z as f32)
        };
        let axis_major = tangent_toward(center, random_dir);
        let axis_minor = center.cross(axis_major).try_normalize().unwrap_or_else(|| {
            let fallback = if center.y.abs() < 0.9 {
                Vec3::Y
            } else {
                Vec3::X
            };
            tangent_toward(center, fallback)
        });

        let is_primary = plate_idx == 0;
        let base_major = if is_primary {
            PRIMARY_CONTINENT_MAJOR_RAD + PRIMARY_CONTINENT_EXTRA_MAJOR_RAD * primary_t
        } else {
            SECONDARY_CONTINENT_MAJOR_RAD
        };
        let major_jitter = rng.range_f64(0.90, 1.12) as f32;
        let major_rad = (base_major * radius_scale * major_jitter).clamp(0.38, 1.18);
        let elongation = rng.range_f64(1.20, 1.85) as f32;
        let minor_rad = (major_rad / elongation).clamp(0.30, 0.82);

        out.push(ContinentShape {
            center_dir: center,
            axis_major,
            axis_minor,
            major_rad,
            minor_rad,
            edge_seed: sub_seed(seed, &format!("continent:{}", plate.id.0)) as u32,
        });
    }
    out
}

fn build_coastal_embayments(tectonics: &TectonicSystem, seed: u64) -> Vec<CoastalEmbayment> {
    let mut by_cont_plate: Vec<Vec<Vec3>> = vec![Vec::new(); tectonics.plates.len()];
    for boundary in &tectonics.boundaries {
        let a_kind = tectonics.plates[boundary.plate_a.0 as usize].kind;
        let b_kind = tectonics.plates[boundary.plate_b.0 as usize].kind;
        match (a_kind, b_kind) {
            (PlateKind::Continental, PlateKind::Oceanic) => {
                by_cont_plate[boundary.plate_a.0 as usize].push(boundary.midpoint_dir);
            }
            (PlateKind::Oceanic, PlateKind::Continental) => {
                by_cont_plate[boundary.plate_b.0 as usize].push(boundary.midpoint_dir);
            }
            _ => {}
        }
    }

    let mut out = Vec::new();
    for plate in &tectonics.plates {
        if plate.kind != PlateKind::Continental {
            continue;
        }
        let candidates = &by_cont_plate[plate.id.0 as usize];
        if candidates.is_empty() {
            continue;
        }

        let mut rng = Rng::new(seed ^ ((plate.id.0 as u64) << 32));
        let count = (candidates.len() / 80).clamp(3, 7);
        for i in 0..count {
            let midpoint = candidates[(rng.next_u64() as usize) % candidates.len()];
            let inward = tangent_toward(midpoint, plate.centroid_dir);
            let alongshore = midpoint.cross(inward).try_normalize().unwrap_or(Vec3::Z);
            let center = rotate_on_sphere(midpoint, inward, rng.range_f64(0.015, 0.105) as f32);
            out.push(CoastalEmbayment {
                center_dir: center,
                axis_inland: inward,
                axis_alongshore: alongshore,
                inland_rad: rng.range_f64(0.055, 0.18) as f32,
                alongshore_rad: rng.range_f64(0.075, 0.30) as f32,
                depth_m: rng.range_f64(750.0, 2200.0) as f32,
                edge_seed: sub_seed(seed, &format!("coast:{}:{i}", plate.id.0)) as u32,
            });
        }
    }
    out
}

fn tangent_toward(origin: Vec3, target: Vec3) -> Vec3 {
    let tangent = target - origin * target.dot(origin);
    if let Some(t) = tangent.try_normalize() {
        return t;
    }
    let fallback = if origin.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    (fallback - origin * fallback.dot(origin))
        .try_normalize()
        .unwrap_or(Vec3::X)
}

fn rotate_on_sphere(dir: Vec3, tangent_axis: Vec3, angle_rad: f32) -> Vec3 {
    (dir * angle_rad.cos() + tangent_axis * angle_rad.sin())
        .try_normalize()
        .unwrap_or(dir)
}

/// Surface field that paints a plate-guided oceanic homeworld from a
/// tectonic graph. See module docs for the algorithm.
pub struct AgingOceanicField {
    tectonics: TectonicSystem,
    /// Seed for the transform-boundary jitter term, separate from the
    /// continent seed so a transform jitter retune doesn't shift coastlines.
    transform_seed: u32,
    /// Seed for continent-scale breakup and climate color variation.
    continent_seed: u32,
    climate_seed: u32,
    /// Cached body radius from the tectonic system.
    #[allow(dead_code)]
    body_radius_m: f32,
    /// Independent continent silhouettes seeded from the continental plate
    /// scaffold. These stop land/water topology from being a direct copy of
    /// plate polygons.
    continents: Vec<ContinentShape>,
    /// Coastline-only negative shapes seeded from continental/oceanic
    /// boundaries. Plates guide where these can happen; they do not become
    /// shoreline outlines.
    embayments: Vec<CoastalEmbayment>,
}

impl AgingOceanicField {
    pub fn new(tectonics: TectonicSystem, root_seed: u64, ocean_fraction: f32) -> Self {
        let transform_seed = sub_seed(root_seed, "aging_oceanic.transform_jitter") as u32;
        let continent_seed = sub_seed(root_seed, "aging_oceanic.continent_breakup") as u32;
        let climate_seed = sub_seed(root_seed, "aging_oceanic.climate_color") as u32;
        let body_radius_m = tectonics.body_radius_m;
        let continents = build_continent_shapes(
            &tectonics,
            sub_seed(root_seed, "aging_oceanic.continent_shapes"),
            ocean_fraction,
        );
        let embayments = build_coastal_embayments(
            &tectonics,
            sub_seed(root_seed, "aging_oceanic.coastal_embayments"),
        );
        Self {
            tectonics,
            transform_seed,
            continent_seed,
            climate_seed,
            body_radius_m,
            continents,
            embayments,
        }
    }

    /// K-nearest IDW blend of cell properties. The K closest cells are
    /// weighted by `1/(angle² + ε)` (Shepard's method); the result is a
    /// smooth function of `dir` with no cell-edge discontinuities.
    fn smooth_cell(&self, dir: Vec3) -> SmoothCell {
        // Find the top-K cells by dot product (largest dot = smallest angle).
        // Insertion-sort into a sorted-descending array — O(N·K) per sample,
        // same complexity as the existing nearest-cell scan with K=1.
        let cells = &self.tectonics.mesh.cells;
        let mut top: [(f32, u32); SMOOTHING_K] = [(f32::NEG_INFINITY, 0); SMOOTHING_K];
        for (i, &c) in cells.iter().enumerate() {
            let dot = c.dot(dir);
            if dot > top[SMOOTHING_K - 1].0 {
                top[SMOOTHING_K - 1] = (dot, i as u32);
                let mut j = SMOOTHING_K - 1;
                while j > 0 && top[j].0 > top[j - 1].0 {
                    top.swap(j, j - 1);
                    j -= 1;
                }
            }
        }

        // IDW weights from angular distance to each top-K cell.
        let mut weights = [0.0_f32; SMOOTHING_K];
        let mut total_w = 0.0_f32;
        for i in 0..SMOOTHING_K {
            let dot = top[i].0.clamp(-1.0, 1.0);
            // angle² ≈ 2·(1 − dot) for small angles; use the exact form so
            // antipodal samples don't degenerate.
            let angle = dot.acos();
            let w = 1.0 / (angle * angle + SHEPARD_EPS_RAD2);
            weights[i] = w;
            total_w += w;
        }
        let inv_total = 1.0 / total_w.max(1.0e-9);

        // Continentalness plate prior: weighted indicator of plate-kind =
        // continental. The visible continent outline is applied later as an
        // independent low-frequency mask; keeping this as a prior preserves
        // tectonic context without making whole plates become land.
        let mut continentalness = 0.0_f32;
        for i in 0..SMOOTHING_K {
            let cell_idx = top[i].1 as usize;
            let plate_id = self.tectonics.cell_plate[cell_idx];
            let kind = self.tectonics.plates[plate_id.0 as usize].kind;
            if kind == PlateKind::Continental {
                continentalness += weights[i];
            }
        }
        continentalness *= inv_total;

        // Smoothed boundary distance + smoothed boundary-kind membership.
        // Each top-K cell carries its own "nearest boundary" (and that
        // boundary's kind); we weight the kinds by the cell weights so
        // the convergent/divergent/transform regimes blend continuously
        // across cell edges instead of switching abruptly.
        let mut boundary_distance_m = 0.0_f32;
        let mut convergent_w = 0.0_f32;
        let mut divergent_w = 0.0_f32;
        let mut transform_w = 0.0_f32;
        for i in 0..SMOOTHING_K {
            let cell_idx = top[i].1 as usize;
            boundary_distance_m +=
                weights[i] * self.tectonics.fields.cell_boundary_distance_m[cell_idx];

            if let Some(b_idx) = self.tectonics.fields.cell_nearest_boundary[cell_idx] {
                match self.tectonics.boundaries[b_idx as usize].kind {
                    BoundaryKind::Convergent => convergent_w += weights[i],
                    BoundaryKind::Divergent => divergent_w += weights[i],
                    BoundaryKind::Transform => transform_w += weights[i],
                }
            }
        }
        boundary_distance_m *= inv_total;
        // Normalize kind weights to a probability simplex (each in [0, 1],
        // sum ≤ 1). Sum < 1 only when some top-K cells have no nearest
        // boundary (single-plate edge case).
        convergent_w *= inv_total;
        divergent_w *= inv_total;
        transform_w *= inv_total;

        SmoothCell {
            continentalness,
            boundary_distance_m,
            convergent_w,
            divergent_w,
            transform_w,
        }
    }

    fn continent_shape_continentalness(&self, dir: Vec3) -> f32 {
        let mut combined = 0.0_f32;
        for continent in &self.continents {
            let warped = self.domain_warp(dir, continent.edge_seed ^ 0x57A7_EC01, 0.9, 0.16);
            let dot = continent.center_dir.dot(warped).clamp(-1.0, 1.0);
            let tangent = warped - continent.center_dir * dot;
            let x = tangent.dot(continent.axis_major) / continent.major_rad.sin().max(0.04);
            let y = tangent.dot(continent.axis_minor) / continent.minor_rad.sin().max(0.04);
            let d = (x * x + y * y).sqrt();

            let edge_p = warped * 3.1;
            let broad = fbm3(
                edge_p.x,
                edge_p.y,
                edge_p.z,
                continent.edge_seed ^ 0xC041_57A7,
                4,
                0.55,
                2.0,
            );
            let fine_p = warped * 8.0;
            let fine = fbm3(
                fine_p.x,
                fine_p.y,
                fine_p.z,
                continent.edge_seed ^ 0xC0A57_011,
                3,
                0.50,
                2.1,
            );
            let edge = 1.0 + broad * 0.16 + fine * 0.06;
            let local = 1.0 - smoothstep(edge * 0.78, edge * 1.12, d);
            combined = 1.0 - (1.0 - combined) * (1.0 - local.clamp(0.0, 1.0));
        }
        combined.clamp(0.0, 1.0)
    }

    /// Boundary contribution to elevation. All three regime contributions
    /// (convergent, divergent, transform) are computed with the smooth
    /// continentalness gating, then blended by their respective kind
    /// weights. This means a sample sitting between cells whose nearest
    /// boundaries are different kinds gets a continuous interpolation —
    /// no cell-edge switching of regime.
    fn boundary_contribution(&self, smooth: &SmoothCell, dir: Vec3) -> f32 {
        let d = smooth.boundary_distance_m;
        let falloff = (-(d / BOUNDARY_HALF_WIDTH_M).powi(2)).exp();

        // `cont` smoothsteps continentalness so the transition between
        // mountain-belt regime and trench regime is gradual, producing
        // island-arc geometry in mixed zones rather than a hard switch.
        let cont = smoothstep(0.30, 0.76, smooth.continentalness);

        let convergent = MOUNTAIN_BUMP_M * cont + (-TRENCH_DEPTH_M * 0.55) * (1.0 - cont);
        let divergent = RIDGE_BUMP_M * 0.65 * (1.0 - cont) + (-RIFT_DEPTH_M) * cont;
        let transform = {
            // Position-dependent sign so adjacent transform stripes don't
            // all push the same way. Keep it low-amplitude so transform
            // boundaries don't add noisy shoreline scallops.
            let p = dir * 12.0;
            fbm3(p.x, p.y, p.z, self.transform_seed, 3, 0.5, 2.0) * TRANSFORM_JITTER_M
        };

        let raw = convergent * smooth.convergent_w
            + divergent * smooth.divergent_w
            + transform * smooth.transform_w;
        let scar_p = dir * 3.4;
        let scar_visibility = 0.22
            + 0.78
                * smoothstep(
                    0.30,
                    0.82,
                    fbm3(
                        scar_p.x,
                        scar_p.y,
                        scar_p.z,
                        self.transform_seed ^ 0x57A1_5CA7,
                        4,
                        0.55,
                        2.0,
                    ) * 0.5
                        + 0.5,
                );
        raw * falloff * scar_visibility
    }

    fn coastal_margin_weight(&self, continentalness: f32) -> f32 {
        smoothstep(0.10, 0.78, continentalness) * (1.0 - smoothstep(0.94, 1.0, continentalness))
    }

    fn coastline_shape(&self, smooth: &SmoothCell, dir: Vec3) -> f32 {
        let margin = self.coastal_margin_weight(smooth.continentalness);
        if margin <= 0.0 {
            return 0.0;
        }

        let warped = self.domain_warp(dir, self.continent_seed ^ 0xC04A_5711, 1.0, 0.18);
        let macro_p = warped * 1.45;
        let macro_n = fbm3(
            macro_p.x,
            macro_p.y,
            macro_p.z,
            self.continent_seed ^ 0xA771_D5EA,
            3,
            0.55,
            2.0,
        );
        let mut embayment_h = 0.0_f32;
        for embayment in &self.embayments {
            let warped = self.domain_warp(dir, embayment.edge_seed, 1.4, 0.05);
            let dot = embayment.center_dir.dot(warped).clamp(-1.0, 1.0);
            let tangent = warped - embayment.center_dir * dot;
            let x = tangent.dot(embayment.axis_inland) / embayment.inland_rad.sin().max(0.025);
            let y =
                tangent.dot(embayment.axis_alongshore) / embayment.alongshore_rad.sin().max(0.025);
            let d2 = x * x + y * y;
            let edge_p = warped * 1.8;
            let ragged = fbm3(
                edge_p.x,
                edge_p.y,
                edge_p.z,
                embayment.edge_seed ^ 0xEA7E_5EA1,
                2,
                0.50,
                2.1,
            );
            let threshold = 1.0 + ragged * 0.03;
            let local = 1.0 - smoothstep(threshold * 0.55, threshold * 1.12, d2);
            embayment_h -= local.clamp(0.0, 1.0) * embayment.depth_m;
        }

        macro_n * COAST_MARGIN_MACRO_AMP_M * margin + embayment_h * margin
    }

    fn interior_relief(&self, continentalness: f32, dir: Vec3) -> f32 {
        let interior = smoothstep(0.68, 0.98, continentalness);
        if interior <= 0.0 {
            return 0.0;
        }

        // Ridge term: light domain warp keeps ranges from looking
        // axis-aligned; squaring `(1 - |fbm|)` sharpens the crests so
        // ridges read as oriented spines rather than fuzzy bumps. The
        // multi-octave fbm is centred near zero, so the zero-crossings
        // (where ridges live) form a connected network across the
        // continent rather than isolated peaks.
        let warped = self.domain_warp(dir, self.continent_seed ^ 0x71D6_4E11, 0.6, 0.10);
        let p_ridge = warped * 1.6;
        let raw = fbm3(
            p_ridge.x,
            p_ridge.y,
            p_ridge.z,
            self.continent_seed ^ 0x1A71_CE11,
            6,
            0.55,
            2.0,
        );
        let ridge = 1.0 - raw.abs();
        let ridge_relief = ridge * ridge * INTERIOR_RIDGE_AMP_M;

        // Fill term: low-amplitude smooth fbm so range flanks and valley
        // floors carry small undulations on top of the ridge skeleton.
        let p_fill = dir * 2.4;
        let fill_relief = fbm3(
            p_fill.x,
            p_fill.y,
            p_fill.z,
            self.continent_seed ^ 0x4F11_BA52,
            5,
            0.50,
            2.0,
        ) * INTERIOR_FILL_AMP_M;

        (ridge_relief + fill_relief) * interior
    }

    /// Coastline-scale fractal wiggle added to continentalness near the
    /// shoreline. Gated by `coast_proximity` derived from the *already
    /// warped* continentalness so the detail rides the actual coast,
    /// not the smoothed plate prior. Tapers to zero deep in continent
    /// (continentalness > 0.95) and deep in ocean (< 0.05) so it can't
    /// invent islands in mid-ocean or interior seas in the heartland.
    fn coastline_detail(&self, dir: Vec3, continentalness: f32) -> f32 {
        // Soft bell-curve gate: strongest at the coast (continentalness 0.5),
        // tapers to zero at deep ocean (0.0) and deep continent (1.0). Wider
        // than a plain `coast ± ε` smoothstep — texels whose own
        // continentalness reads as deep continent or deep ocean still
        // receive partial detail, because under K-IDW smoothing those
        // texels can sit one bilinear-step from the coast contour and
        // therefore influence the silhouette rendered by the GPU.
        let d = continentalness * 2.0 - 1.0;
        let coast_proximity = (1.0 - d * d).max(0.0);
        let warped = self.domain_warp(
            dir,
            self.continent_seed ^ 0xC0DE_7A11,
            COASTLINE_DETAIL_DOMAIN_FREQ,
            COASTLINE_DETAIL_DOMAIN_AMP,
        );
        let p = warped * COASTLINE_DETAIL_FREQ;
        let n = fbm3(
            p.x,
            p.y,
            p.z,
            self.continent_seed ^ 0xC0DE_7A12,
            COASTLINE_DETAIL_OCTAVES,
            COASTLINE_DETAIL_GAIN,
            2.0,
        );
        n * COASTLINE_DETAIL_AMP * coast_proximity
    }

    /// Macro warp added to smoothed continentalness before any height
    /// contributions read it. Domain-warped low-frequency fBM in [-1, 1],
    /// scaled to ±[`MACRO_CONTINENT_WARP_AMP`]. Carves inland seas and
    /// pushes peninsulas through the smoothed plate prior so visible
    /// continent silhouettes diverge from the plate-edge contour.
    fn macro_continent_warp(&self, dir: Vec3) -> f32 {
        let warped = self.domain_warp(
            dir,
            self.continent_seed ^ 0xC0AC_7AA1,
            MACRO_CONTINENT_WARP_DOMAIN_FREQ,
            MACRO_CONTINENT_WARP_DOMAIN_AMP,
        );
        let p = warped * MACRO_CONTINENT_WARP_FREQ;
        fbm3(
            p.x,
            p.y,
            p.z,
            self.continent_seed ^ 0xC0AC_7AB2,
            4,
            0.55,
            2.0,
        ) * MACRO_CONTINENT_WARP_AMP
    }

    fn domain_warp(&self, dir: Vec3, seed: u32, frequency: f32, strength: f32) -> Vec3 {
        let p = dir * frequency;
        let warp = Vec3::new(
            fbm3(p.x, p.y, p.z, seed, 3, 0.5, 2.0),
            fbm3(p.x, p.y, p.z, seed ^ 0xA53A_9E1D, 3, 0.5, 2.0),
            fbm3(p.x, p.y, p.z, seed ^ 0xC2B2_AE35, 3, 0.5, 2.0),
        );
        let tangent_warp = warp - dir * warp.dot(dir);
        (dir + tangent_warp * strength)
            .try_normalize()
            .unwrap_or(dir)
    }

    #[allow(dead_code)]
    fn surface_albedo(
        &self,
        relative_height_m: f32,
        dir: Vec3,
        slope: f32,
        coast_distance_m: f32,
    ) -> [f32; 3] {
        if relative_height_m < 0.0 {
            return altitude_albedo(relative_height_m);
        }

        let abs_lat = dir.y.clamp(-1.0, 1.0).asin().abs() / std::f32::consts::FRAC_PI_2;
        let warm_lat = 1.0 - smoothstep(0.52, 0.86, abs_lat);
        let cold = smoothstep(0.48, 0.86, abs_lat);
        let highland = smoothstep(900.0, 2300.0, relative_height_m);
        let coastal = 1.0 - smoothstep(160_000.0, 1_900_000.0, coast_distance_m);
        let interior = smoothstep(350_000.0, 2_600_000.0, coast_distance_m);

        let macro_warp = self.domain_warp(dir, self.climate_seed ^ 0x31A7_0B10, 0.75, 0.28);
        let macro_p = macro_warp * 1.45;
        let macro_moisture = (fbm3(
            macro_p.x,
            macro_p.y,
            macro_p.z,
            self.climate_seed ^ 0x31A7_0B11,
            5,
            0.58,
            2.0,
        ) * 0.5
            + 0.5)
            .clamp(0.0, 1.0);

        let lobe_warp = self.domain_warp(dir, self.climate_seed ^ 0x31A7_0B20, 1.55, 0.20);
        let lobe_p = lobe_warp * 3.4;
        let wet_lobes = (fbm3(
            lobe_p.x,
            lobe_p.y,
            lobe_p.z,
            self.climate_seed ^ 0x31A7_0B21,
            4,
            0.55,
            2.0,
        ) * 0.5
            + 0.5)
            .clamp(0.0, 1.0);
        let dry_lobes = (fbm3(
            lobe_p.x + 11.7,
            lobe_p.y - 4.3,
            lobe_p.z + 8.1,
            self.climate_seed ^ 0x31A7_0B22,
            4,
            0.55,
            2.0,
        ) * 0.5
            + 0.5)
            .clamp(0.0, 1.0);

        let ridge_p = self.domain_warp(dir, self.climate_seed ^ 0x31A7_0B30, 0.9, 0.16) * 2.65;
        let ridge_raw = fbm3(
            ridge_p.x,
            ridge_p.y,
            ridge_p.z,
            self.climate_seed ^ 0x31A7_0B31,
            5,
            0.55,
            2.0,
        );
        let dry_corridors = (1.0 - ridge_raw.abs()).powf(2.0);
        let orographic_dry = dry_corridors
            * (smoothstep(700.0, 2100.0, relative_height_m) * 0.65 + slope * 26.0).clamp(0.0, 1.0);

        let mut moisture = 0.16 + macro_moisture * 0.48 + coastal * 0.24 + (wet_lobes - 0.5) * 0.30
            - interior * 0.28
            - orographic_dry * 0.40
            - highland * 0.12;
        moisture = moisture.clamp(0.0, 1.0);

        let dryness = (1.0 - moisture + dry_lobes * 0.30 + interior * 0.20 + highland * 0.12
            - coastal * 0.16)
            .clamp(0.0, 1.0);

        let veg_warp = self.domain_warp(dir, self.climate_seed ^ 0x1E55_4411, 1.5, 0.10);
        let veg_p = veg_warp * 6.0;
        let lushness = (fbm3(
            veg_p.x,
            veg_p.y,
            veg_p.z,
            self.climate_seed ^ 0x1E55_4BBA,
            3,
            0.55,
            2.0,
        ) * 0.5
            + 0.5)
            .clamp(0.0, 1.0);

        let desert_shape = smoothstep(0.42, 0.80, dry_lobes + interior * 0.22);
        let forest_shape = smoothstep(0.28, 0.78, wet_lobes + coastal * 0.16);
        let slope_t = smoothstep(0.005, 0.020, slope);
        let altitude_gate = smoothstep(850.0, 1550.0, relative_height_m);

        let forest_w = sharpen_biome_weight(
            smoothstep(0.48, 0.78, moisture)
                * forest_shape
                * (1.0 - cold * 0.68)
                * (1.0 - highland * 0.44),
        ) * 1.18;
        let grass_w = sharpen_biome_weight(
            biome_band(moisture + coastal * 0.08, 0.22, 0.56, 0.90)
                * (1.0 - cold * 0.30)
                * (1.0 - highland * 0.18),
        ) * 0.84;
        let steppe_w = sharpen_biome_weight(
            biome_band(dryness, 0.22, 0.56, 0.94)
                * warm_lat
                * (1.0 - coastal * 0.28)
                * (1.0 - highland * 0.12),
        ) * 1.10;
        let desert_w = sharpen_biome_weight(
            smoothstep(0.46, 0.80, dryness) * desert_shape * warm_lat * (1.0 - coastal * 0.42),
        ) * 1.35;
        let boreal_w = sharpen_biome_weight(
            cold * smoothstep(0.28, 0.72, moisture + forest_shape * 0.10) * (1.0 - highland * 0.35),
        );
        let tundra_w = sharpen_biome_weight(
            cold * (1.0 - smoothstep(0.40, 0.78, moisture))
                * (0.55 + highland * 0.55)
                * (1.0 - coastal * 0.18),
        );

        let total = forest_w + grass_w + steppe_w + desert_w + boreal_w + tundra_w;
        let climate_strength = smoothstep(0.006, 0.085, total);

        // Forest pulled significantly darker — dense canopy reads
        // near-black green from orbit, and the brightness contrast against
        // the lighter grassland is what reads as richness rather than as
        // a saturated colour pop.
        let forest = bilinear_palette(
            [0.100, 0.155, 0.055],
            [0.055, 0.108, 0.040],
            [0.065, 0.140, 0.045],
            [0.022, 0.068, 0.022],
            lushness,
            moisture,
        );
        // Grassland desaturated at the bright corner — the previous
        // [0.41, 0.46, 0.15] read as eye-watering yellow-green. R+B raised
        // and G eased so the savanna sits closer to khaki than to
        // electric-green.
        let grassland = bilinear_palette(
            [0.340, 0.390, 0.180],
            [0.240, 0.330, 0.140],
            [0.215, 0.330, 0.130],
            [0.115, 0.230, 0.090],
            lushness,
            moisture,
        );
        let steppe = bilinear_palette(
            [0.450, 0.395, 0.205],
            [0.355, 0.320, 0.150],
            [0.385, 0.380, 0.180],
            [0.275, 0.290, 0.130],
            lushness,
            moisture,
        );
        let desert = bilinear_palette(
            [0.625, 0.530, 0.330],
            [0.535, 0.435, 0.265],
            [0.540, 0.480, 0.295],
            [0.425, 0.380, 0.215],
            lushness,
            moisture,
        );
        let boreal = bilinear_palette(
            [0.195, 0.235, 0.150],
            [0.115, 0.175, 0.115],
            [0.155, 0.230, 0.145],
            [0.065, 0.145, 0.105],
            lushness,
            moisture,
        );
        let tundra = bilinear_palette(
            [0.370, 0.370, 0.290],
            [0.285, 0.315, 0.230],
            [0.310, 0.355, 0.250],
            [0.210, 0.285, 0.200],
            lushness,
            moisture,
        );

        let altitude_color = altitude_albedo(relative_height_m);
        let mut climate_land = altitude_color;
        if total > 1e-6 {
            climate_land = [0.0, 0.0, 0.0];
            for (color, weight) in [
                (forest, forest_w),
                (grassland, grass_w),
                (steppe, steppe_w),
                (desert, desert_w),
                (boreal, boreal_w),
                (tundra, tundra_w),
            ] {
                let w = weight / total;
                climate_land[0] += color[0] * w;
                climate_land[1] += color[1] * w;
                climate_land[2] += color[2] * w;
            }
        }

        let mut land = mix3(altitude_color, climate_land, climate_strength);

        let exposed_soil = mix3([0.500, 0.320, 0.175], [0.385, 0.275, 0.160], lushness);
        let rock = mix3([0.500, 0.300, 0.185], [0.410, 0.260, 0.165], lushness);
        let snow = [0.870, 0.870, 0.885];

        let soil_t = (slope_t * altitude_gate * 0.30 + highland * 0.24 * dryness).min(0.58);
        land = mix3(land, exposed_soil, soil_t);

        let altitude_rock = smoothstep(1750.0, 2550.0, relative_height_m);
        let rock_t = (altitude_rock * 0.42 + slope_t * altitude_gate * 0.26).min(0.70);
        land = mix3(land, rock, rock_t);

        let mottle_p = dir * 40.0;
        let mottle = fbm3(
            mottle_p.x,
            mottle_p.y,
            mottle_p.z,
            self.climate_seed ^ 0xCA77_E110,
            2,
            0.50,
            2.0,
        );
        let m = 1.0 + mottle * 0.045;
        land = [land[0] * m, land[1] * m, land[2] * m];

        // Snow only on the very top of peaks — narrow summit ribbons,
        // not blob-covering swaths. Peaks reach ~4000 m on Thalos so
        // 3300–3900 m reserves snow for the top ~500–700 m.
        land = mix3(land, snow, smoothstep(3300.0, 3900.0, relative_height_m));

        // Preserve the wet beach band; above that, climate + slope coloring
        // takes over from the altitude-only ramp.
        mix3(
            altitude_color,
            land,
            smoothstep(75.0, 320.0, relative_height_m),
        )
    }

    /// Weighted biome identity for the shared surface-color painter.
    ///
    /// This mirrors the climate logic that used to live inside
    /// `surface_albedo`, but it returns semantic weights instead of final
    /// color. The painter owns the palette and relief grading.
    fn surface_biome_mix(
        &self,
        relative_height_m: f32,
        dir: Vec3,
        slope: f32,
        coast_distance_m: f32,
    ) -> BiomeMix {
        let water_depth_m = -relative_height_m;
        let ocean_w = smoothstep(60.0, 1_600.0, water_depth_m);
        let shelf_w =
            smoothstep(1_900.0, 90.0, water_depth_m) * smoothstep(-80.0, -900.0, relative_height_m);
        let beach_w = (1.0 - smoothstep(35.0, 260.0, relative_height_m))
            * smoothstep(-60.0, 35.0, relative_height_m);
        if relative_height_m < -80.0 {
            return BiomeMix::from_weighted([
                (AGING_OCEANIC_BIOME_OCEAN, ocean_w.max(0.10)),
                (AGING_OCEANIC_BIOME_SHELF, shelf_w),
                (AGING_OCEANIC_BIOME_BEACH, beach_w * 0.25),
            ]);
        }

        let abs_lat = dir.y.clamp(-1.0, 1.0).asin().abs() / std::f32::consts::FRAC_PI_2;
        let warm_lat = 1.0 - smoothstep(0.52, 0.86, abs_lat);
        let cold = smoothstep(0.48, 0.86, abs_lat);
        let highland = smoothstep(900.0, 2300.0, relative_height_m);
        let coastal = 1.0 - smoothstep(160_000.0, 1_900_000.0, coast_distance_m);
        let interior = smoothstep(350_000.0, 2_600_000.0, coast_distance_m);

        let macro_warp = self.domain_warp(dir, self.climate_seed ^ 0x31A7_0B10, 0.75, 0.28);
        let macro_p = macro_warp * 1.45;
        let macro_moisture = (fbm3(
            macro_p.x,
            macro_p.y,
            macro_p.z,
            self.climate_seed ^ 0x31A7_0B11,
            5,
            0.58,
            2.0,
        ) * 0.5
            + 0.5)
            .clamp(0.0, 1.0);

        let lobe_warp = self.domain_warp(dir, self.climate_seed ^ 0x31A7_0B20, 1.55, 0.20);
        let lobe_p = lobe_warp * 3.4;
        let wet_lobes = (fbm3(
            lobe_p.x,
            lobe_p.y,
            lobe_p.z,
            self.climate_seed ^ 0x31A7_0B21,
            4,
            0.55,
            2.0,
        ) * 0.5
            + 0.5)
            .clamp(0.0, 1.0);
        let dry_lobes = (fbm3(
            lobe_p.x + 11.7,
            lobe_p.y - 4.3,
            lobe_p.z + 8.1,
            self.climate_seed ^ 0x31A7_0B22,
            4,
            0.55,
            2.0,
        ) * 0.5
            + 0.5)
            .clamp(0.0, 1.0);

        let ridge_p = self.domain_warp(dir, self.climate_seed ^ 0x31A7_0B30, 0.9, 0.16) * 2.65;
        let ridge_raw = fbm3(
            ridge_p.x,
            ridge_p.y,
            ridge_p.z,
            self.climate_seed ^ 0x31A7_0B31,
            5,
            0.55,
            2.0,
        );
        let dry_corridors = (1.0 - ridge_raw.abs()).powf(2.0);
        let orographic_dry = dry_corridors
            * (smoothstep(700.0, 2100.0, relative_height_m) * 0.65 + slope * 26.0).clamp(0.0, 1.0);

        let moisture = (0.23 + macro_moisture * 0.50 + coastal * 0.24 + (wet_lobes - 0.5) * 0.30
            - interior * 0.20
            - orographic_dry * 0.30
            - highland * 0.10)
            .clamp(0.0, 1.0);
        let dryness = (1.0 - moisture + dry_lobes * 0.24 + interior * 0.13 + highland * 0.10
            - coastal * 0.16)
            .clamp(0.0, 1.0);
        let desert_shape = smoothstep(0.42, 0.80, dry_lobes + interior * 0.22);
        let forest_shape = smoothstep(0.28, 0.78, wet_lobes + coastal * 0.16);
        let slope_t = smoothstep(0.005, 0.020, slope);
        let altitude_gate = smoothstep(850.0, 1550.0, relative_height_m);

        let forest_w = sharpen_biome_weight(
            smoothstep(0.30, 0.66, moisture)
                * forest_shape
                * (1.0 - cold * 0.58)
                * (1.0 - highland * 0.44),
        ) * 2.15;
        let grass_w = sharpen_biome_weight(
            biome_band(moisture + coastal * 0.08, 0.22, 0.56, 0.90)
                * (1.0 - cold * 0.30)
                * (1.0 - highland * 0.18),
        ) * 0.54;
        let steppe_w = sharpen_biome_weight(
            biome_band(dryness, 0.22, 0.56, 0.94)
                * warm_lat
                * (1.0 - coastal * 0.28)
                * (1.0 - highland * 0.12),
        ) * 0.92;
        let desert_w = sharpen_biome_weight(
            smoothstep(0.46, 0.80, dryness) * desert_shape * warm_lat * (1.0 - coastal * 0.42),
        ) * 1.08;
        let boreal_w = sharpen_biome_weight(
            cold * smoothstep(0.22, 0.66, moisture + forest_shape * 0.14) * (1.0 - highland * 0.35),
        ) * 1.25;
        let tundra_w = sharpen_biome_weight(
            cold * (1.0 - smoothstep(0.40, 0.78, moisture))
                * (0.55 + highland * 0.55)
                * (1.0 - coastal * 0.18),
        );
        let rock_w = (smoothstep(1_650.0, 2_850.0, relative_height_m) * 0.42
            + slope_t * altitude_gate * 0.36)
            .clamp(0.0, 0.74);
        let snow_w = smoothstep(3_250.0, 3_900.0, relative_height_m);

        BiomeMix::from_weighted([
            (AGING_OCEANIC_BIOME_BEACH, beach_w * 1.40),
            (
                AGING_OCEANIC_BIOME_FOREST,
                forest_w * (1.0 - rock_w * 0.45) * (1.0 - snow_w),
            ),
            (
                AGING_OCEANIC_BIOME_GRASSLAND,
                grass_w * (1.0 - rock_w * 0.35) * (1.0 - snow_w),
            ),
            (
                AGING_OCEANIC_BIOME_STEPPE,
                steppe_w * (1.0 - rock_w * 0.30) * (1.0 - snow_w),
            ),
            (
                AGING_OCEANIC_BIOME_DESERT,
                desert_w * (1.0 - rock_w * 0.25) * (1.0 - snow_w),
            ),
            (
                AGING_OCEANIC_BIOME_BOREAL,
                boreal_w * (1.0 - rock_w * 0.30) * (1.0 - snow_w),
            ),
            (
                AGING_OCEANIC_BIOME_TUNDRA,
                tundra_w * (1.0 - rock_w * 0.20) * (1.0 - snow_w),
            ),
            (AGING_OCEANIC_BIOME_ROCK, rock_w * (1.0 - snow_w * 0.55)),
            (AGING_OCEANIC_BIOME_SNOW, snow_w),
        ])
    }

    /// Repaint the baked cubemaps after connectivity cleanup so dry-filled
    /// interior basins use the same relative-to-sea-level palette as the
    /// original field samples.
    pub fn repaint_baked_surface(&self, builder: &mut BodyBuilder, sea_level_m: f32) {
        let res = builder.cubemap_resolution as usize;
        let pixel_size_m = cube_face_texel_scale_m(builder.radius_m, builder.cubemap_resolution);

        // Snapshot the finalized height cube so the slope sampler can reach
        // across face boundaries while we write albedo/roughness/material
        // back per face. The clone is one transient cube copy (~6 MB at
        // 512²).
        let heights_full = builder.height_contributions.height.clone();

        // Distance from every texel to the nearest below-sea-level texel,
        // via multi-source BFS over the cubemap grid. Drives the
        // continentality term in `surface_albedo` (deep continental
        // interiors read drier than coasts at the same latitude). One
        // O(N) pass over the ~1.6 M texels.
        let coast_distances = compute_coast_distance(&heights_full, sea_level_m, res, pixel_size_m);
        let face_len = res * res;

        for face in CubemapFace::ALL {
            let heights = heights_full.face_data(face);
            let materials = builder.material_cubemap.face_data_mut(face);
            let roughness = builder.roughness_cubemap.face_data_mut(face);
            let biome_weights = builder.biome_weights_cubemap.face_data_mut(face);
            let face_offset = face as usize * face_len;

            for (i, &height_m) in heights.iter().enumerate() {
                let x = i % res;
                let y = i / res;
                let u = (x as f32 + 0.5) / res as f32;
                let v = (y as f32 + 0.5) / res as f32;
                let dir = face_uv_to_dir(face, u, v);
                let relative_height_m = height_m - sea_level_m;
                let slope = sample_slope(&heights_full, face, x, y, res, pixel_size_m);
                let coast_d = coast_distances[face_offset + i];
                let biome_mix = self.surface_biome_mix(relative_height_m, dir, slope, coast_d);
                biome_weights[i] = crate::surface_field::BiomeMixTexel::from_mix(biome_mix);
                materials[i] = altitude_material(relative_height_m);
                roughness[i] = quantize_unit_to_u8(surface_roughness(relative_height_m, slope));
            }
        }
    }
}

impl SurfaceField for AgingOceanicField {
    fn sample(&self, dir: Vec3, _sample_scale_m: f32) -> SurfaceFieldSample {
        let dir = dir.normalize_or_zero();
        let mut smooth = self.smooth_cell(dir);

        // Continent silhouettes are their own low-frequency field. The
        // smoothed plate prior still nudges the result, but land no longer
        // directly inherits whole plate polygons.
        let plate_prior = smooth.continentalness;
        let continent_shape = self.continent_shape_continentalness(dir);
        smooth.continentalness = (continent_shape * (0.90 + 0.10 * plate_prior)).clamp(0.0, 1.0);

        // Macro warp: shift continentalness by a low-frequency domain-warped
        // fBM. This is what breaks the smoothed-plate-circle silhouette
        // into Earth-like irregularity. All downstream readers of
        // continentalness (plate base, boundary regime gating, coastline
        // shaping, interior relief, ocean cleanup) follow the warped value.
        let warp = self.macro_continent_warp(dir);
        smooth.continentalness = (smooth.continentalness + warp).clamp(0.0, 1.0);

        // Coastline detail: high-frequency wiggle gated to the coast region.
        // Adds the 30–200 km fractal irregularity that makes coastlines look
        // like coastlines instead of smooth analog curves. Uses the
        // already-warped continentalness for gating so the detail rides
        // the actual silhouette, not the unwarped plate prior.
        let detail = self.coastline_detail(dir, smooth.continentalness);
        smooth.continentalness = (smooth.continentalness + detail).clamp(0.0, 1.0);

        // Smooth elevation prior: broad continental crust follows the plate
        // graph, but coastline shaping below moves the sea-level contour.
        let plate_base =
            OCEANIC_BASE_M + (CONTINENTAL_BASE_M - OCEANIC_BASE_M) * smooth.continentalness;

        let boundary_h = self.boundary_contribution(&smooth, dir);
        let coastline_h = self.coastline_shape(&smooth, dir);
        let relief_h = self.interior_relief(smooth.continentalness, dir);

        let height_m = plate_base + boundary_h + coastline_h + relief_h;
        // Slope is computed from finalized height-cube neighbors during
        // `repaint_baked_surface`; the per-direction sample path doesn't
        // see neighbors, so pass slope=0. Repaint overwrites both albedo
        // and roughness anyway, this output only serves direct callers.
        // Slope and coast distance are computed in the bake-time repaint
        // pass; the per-direction sample path doesn't see neighbors, so
        // pass 0. Repaint overwrites both albedo and roughness anyway —
        // this output only serves direct callers of `SurfaceField::sample`.
        let roughness = altitude_roughness(height_m);
        let material_mix = SurfaceMaterialMix::single(altitude_material(height_m));
        let biome_mix = self.surface_biome_mix(height_m, dir, 0.0, 0.0);

        SurfaceFieldSample {
            height_m,
            material_mix,
            biome_mix,
            roughness,
            // Anisotropy left to the height-derived normal in the impostor.
            // Pass `dir` to mean "no analytical perturbation."
            normal_local: dir,
        }
    }
}

/// Remove all disconnected below-sea-level components except the largest one.
///
/// This is intentionally a bake-space pass rather than part of
/// `AgingOceanicField::sample`: ocean connectivity is global topology, while
/// the field sampler is a local continuous function. Running here guarantees
/// the finalized impostor height cubemap has one connected ocean.
pub fn enforce_single_connected_ocean(builder: &mut BodyBuilder, sea_level_m: f32) {
    let res = builder.height_contributions.resolution() as usize;
    let texel_count = res * res * CubemapFace::ALL.len();
    let (labels, largest_label, component_count) = {
        let heights = &builder.height_contributions.height;
        let mut labels = vec![0_u32; texel_count];
        let mut component_sizes: Vec<usize> = Vec::new();
        let mut queue: Vec<usize> = Vec::new();
        let mut next_label = 0_u32;

        for idx in 0..texel_count {
            if labels[idx] != 0 || !is_water_texel(heights, idx, res, sea_level_m) {
                continue;
            }

            next_label += 1;
            let label = next_label;
            labels[idx] = label;
            queue.clear();
            queue.push(idx);

            let mut read = 0usize;
            let mut size = 0usize;
            while read < queue.len() {
                let current = queue[read];
                read += 1;
                size += 1;

                let (face, x, y, _) = split_texel_index(current, res);
                for (dx, dy) in [(1_i32, 0_i32), (-1, 0), (0, 1), (0, -1)] {
                    let neighbor = adjacent_texel_index(face, x, y, dx, dy, res);
                    if neighbor == current {
                        continue;
                    }
                    if labels[neighbor] == 0 && is_water_texel(heights, neighbor, res, sea_level_m)
                    {
                        labels[neighbor] = label;
                        queue.push(neighbor);
                    }
                }
            }

            component_sizes.push(size);
        }

        let (largest_idx, _) = component_sizes
            .iter()
            .copied()
            .enumerate()
            .max_by_key(|&(_, size)| size)
            .unwrap_or((0, 0));
        (labels, (largest_idx + 1) as u32, component_sizes.len())
    };

    if component_count <= 1 {
        return;
    }

    for (idx, &label) in labels.iter().enumerate() {
        if label != 0 && label != largest_label {
            dry_fill_water_texel(builder, idx, res, sea_level_m);
        }
    }
}

fn is_water_texel(heights: &Cubemap<f32>, index: usize, res: usize, sea_level_m: f32) -> bool {
    let (face, _, _, local) = split_texel_index(index, res);
    heights.face_data(face)[local] < sea_level_m
}

fn dry_fill_water_texel(builder: &mut BodyBuilder, index: usize, res: usize, sea_level_m: f32) {
    let (face, _, _, local) = split_texel_index(index, res);
    let filled_height = {
        let heights = builder.height_contributions.height.face_data_mut(face);
        let old_height = heights[local];
        let filled_height = dry_fill_height_m(old_height, sea_level_m);
        heights[local] = filled_height;
        filled_height
    };

    builder.material_cubemap.face_data_mut(face)[local] = altitude_material(filled_height);
    builder.roughness_cubemap.face_data_mut(face)[local] =
        quantize_unit_to_u8(altitude_roughness(filled_height));
}

fn dry_fill_height_m(height_m: f32, sea_level_m: f32) -> f32 {
    let depth = (sea_level_m - height_m).max(0.0).min(DRY_FILL_MAX_DEPTH_M);
    sea_level_m + DRY_FILL_BASE_M + depth * DRY_FILL_DEPTH_RELIEF
}

fn split_texel_index(index: usize, res: usize) -> (CubemapFace, usize, usize, usize) {
    let face_len = res * res;
    let face_idx = index / face_len;
    let local = index % face_len;
    let x = local % res;
    let y = local / res;
    (CubemapFace::ALL[face_idx], x, y, local)
}

fn texel_index(face: CubemapFace, x: usize, y: usize, res: usize) -> usize {
    face as usize * res * res + y * res + x
}

fn adjacent_texel_index(
    face: CubemapFace,
    x: usize,
    y: usize,
    dx: i32,
    dy: i32,
    res: usize,
) -> usize {
    let nx = x as i32 + dx;
    let ny = y as i32 + dy;
    if (0..res as i32).contains(&nx) && (0..res as i32).contains(&ny) {
        return texel_index(face, nx as usize, ny as usize, res);
    }

    let u = (x as f32 + 0.5 + dx as f32) / res as f32;
    let v = (y as f32 + 0.5 + dy as f32) / res as f32;
    let dir = face_uv_to_dir(face, u, v);
    let (neighbor_face, neighbor_u, neighbor_v) = dir_to_face_uv(dir);
    let neighbor_x = ((neighbor_u * res as f32).floor() as usize).min(res - 1);
    let neighbor_y = ((neighbor_v * res as f32).floor() as usize).min(res - 1);
    texel_index(neighbor_face, neighbor_x, neighbor_y, res)
}

/// Central-difference slope magnitude at a cubemap texel, computed in
/// rise/run units (slope 0.1 ≈ 5.7° flank). Crosses face boundaries via
/// `adjacent_texel_index`; the cubesphere face metric isn't perfectly
/// uniform but the small distortion is fine for the "is this a flank"
/// decision the slope feeds into.
fn sample_slope(
    heights: &Cubemap<f32>,
    face: CubemapFace,
    x: usize,
    y: usize,
    res: usize,
    pixel_size_m: f32,
) -> f32 {
    let h_xp = sample_height_by_index(heights, adjacent_texel_index(face, x, y, 1, 0, res), res);
    let h_xn = sample_height_by_index(heights, adjacent_texel_index(face, x, y, -1, 0, res), res);
    let h_yp = sample_height_by_index(heights, adjacent_texel_index(face, x, y, 0, 1, res), res);
    let h_yn = sample_height_by_index(heights, adjacent_texel_index(face, x, y, 0, -1, res), res);

    let dx = (h_xp - h_xn) / (2.0 * pixel_size_m);
    let dy = (h_yp - h_yn) / (2.0 * pixel_size_m);
    (dx * dx + dy * dy).sqrt()
}

fn sample_height_by_index(heights: &Cubemap<f32>, index: usize, res: usize) -> f32 {
    let (face, _, _, local) = split_texel_index(index, res);
    heights.face_data(face)[local]
}

/// Multi-source BFS distance from every cubemap texel to the nearest
/// below-sea-level texel, returned as a flat per-texel buffer keyed by
/// `face * res² + y * res + x`. Water texels seed at distance 0; land
/// texels get the shortest 4-connected path through their neighbors,
/// crossing face boundaries via `adjacent_texel_index`. Step cost is
/// `pixel_size_m` (the face-center scale), so the result reads in metres.
/// 4-connectivity gives Manhattan-on-cubemap distance — biased up to
/// ~40 % vs. true Euclidean at diagonals — but the smoothstep that
/// consumes this field absorbs that error.
fn compute_coast_distance(
    heights: &Cubemap<f32>,
    sea_level_m: f32,
    res: usize,
    pixel_size_m: f32,
) -> Vec<f32> {
    let face_len = res * res;
    let texel_count = face_len * 6;
    let mut distance = vec![f32::INFINITY; texel_count];
    let mut queue: Vec<usize> = Vec::with_capacity(texel_count);

    for face in CubemapFace::ALL {
        let face_data = heights.face_data(face);
        let face_offset = face as usize * face_len;
        for (local, &h) in face_data.iter().enumerate() {
            if h < sea_level_m {
                let idx = face_offset + local;
                distance[idx] = 0.0;
                queue.push(idx);
            }
        }
    }

    let mut read = 0;
    while read < queue.len() {
        let current_idx = queue[read];
        read += 1;
        let (face, x, y, _) = split_texel_index(current_idx, res);
        let next_d = distance[current_idx] + pixel_size_m;

        for (dx, dy) in [(1_i32, 0_i32), (-1, 0), (0, 1), (0, -1)] {
            let neighbor_idx = adjacent_texel_index(face, x, y, dx, dy, res);
            if distance[neighbor_idx].is_infinite() {
                distance[neighbor_idx] = next_d;
                queue.push(neighbor_idx);
            }
        }
    }

    distance
}
