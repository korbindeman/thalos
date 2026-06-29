//! Runtime procedural surface — the single planet-wide terrain generator
//! behind the [`SurfaceQuery`] seam.
//!
//! This is the replacement for the old baked-cubemap pipeline. It is a pure
//! function of `(direction, lod)`: no bake, no disk artifact, no dependency on
//! the (deleted) feature-compiler / cubemap / cache machinery. The whole field
//! is evaluated analytically in **f64 body-local coordinates** so the surface is
//! sphere-continuous (the same physical point returns the same height regardless
//! of which cube face or tile LOD is sampling it) and precise at planet scale
//! (an f32 `dir * radius` would quantise sample positions to a ~0.25 m lattice on
//! a 3000 km body and terrace the ground on foot — see
//! [`crate::query`]'s precision note).
//!
//! Detail is **LOD-aware** via a continuous octave count
//! ([`octaves_for_lod`]): coarse far tiles evaluate fewer octaves (cheaper, and
//! anti-aliased), near tiles fade additional octaves in smoothly so a tile
//! cascading from N → N+1 octaves across an LOD boundary blends rather than
//! steps.
//!
//! ## Continents & oceans
//!
//! The macro land/sea structure is a **bimodal hypsometric** model (Earth's
//! hypsometric curve is bimodal; airless bodies without plate tectonics are
//! unimodal). A single low-frequency *continentalness* field — distinct
//! cellular "plates", each randomly land/ocean-biased, plus an organic fBm
//! overlay for real coastlines ([`ProceduralSurface::continentalness`]) — is
//! mapped through [`hypsometric_height`] into signed height about sea level
//! (= the reference radius, 0 m): flat abyssal plains, a narrow continental
//! shelf + slope, then a raised continental platform. The relief cascade
//! (hills / swell / ridged mountains) rides on that base, gated to land; the
//! separate water renderer floods everything below 0 m.
//!
//! Two functions are the **plate-tectonics seams** — a later backend replaces
//! their bodies to drive continents and mountain belts / island arcs from real
//! plate margins, without touching the hypsometric remap or the relief cascade:
//! [`ProceduralSurface::continentalness`] (land/sea structure) and
//! [`ProceduralSurface::orogeny`] (where mountains are tall vs plains). The
//! runway-siting scaffold ([`ProceduralSurface::runway_land_bias`] /
//! [`ProceduralSurface::runway_plains_factor`]) is a temporary nudge keeping the
//! fixed runway pad on flat inland ground until terrain-aware auto-siting lands.
//!
//! This is the Slice-0 generator: competent but deliberately simple. Real
//! plate-driven structure, finer mountain placement, and the material/biome
//! weight model are later slices; procedural shading is later still.

use std::sync::LazyLock;

use bevy_erosion_filter::cpu::{ErosionFilterParams, erosion_filter};
use glam::{DVec3, Vec2, Vec3};

use crate::query::{SurfaceQuery, SurfaceSample};

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/// Domain-warp wavelength (m) and displacement (m). The warp breaks up the
/// grid-aligned look of the underlying lattice noise before any height field is
/// sampled.
const WARP_WL_M: f64 = 60_000.0;
const WARP_AMP_M: f64 = 4_000.0;

// --- Continents & oceans (the macro land/sea structure) --------------------
//
// A single low-frequency *continentalness* field `c` ([`ProceduralSurface::
// continentalness`]) is mapped through a BIMODAL hypsometric transfer
// ([`hypsometric_height`]) into signed height about sea level (= the reference
// radius, 0 m). That bimodality is what makes the orbital view read as
// continents + ocean basins instead of uniform fBm bumps: flat abyssal plains,
// a narrow shelf + continental slope, then a raised continental platform (Earth
// has the same bimodal hypsometric curve; airless bodies without plate
// tectonics are unimodal). `continentalness` is the SEPARABLE seam a future
// plate-tectonics backend replaces (continents/mountain belts/island arcs from
// plate margins) without touching the remap or the relief cascade.

/// Continent "plate" cell size (m). The sphere is partitioned into cellular
/// (Worley) cells, each randomly land- or ocean-biased — a cheap analytic
/// stand-in for tectonic plates that yields *distinct* continents separated by
/// ocean (pure fBm + threshold gives one connected supercontinent instead).
/// Smaller → more, smaller landmasses. ~2.8 Mm gives roughly a dozen plates on
/// Thalos → a handful of continents.
const CONTINENT_CELL_M: f64 = 2_800_000.0;
/// How sharply each plate reads as a flat plateau vs blending into its
/// neighbours (units of 1/cell). Higher → flatter plate interiors + tighter
/// coasts; lower → softer, more blended masses.
const PLATE_SHARPNESS: f64 = 8.0;
/// Blend of the plate field vs the organic fBm detail in the continentalness.
const PLATE_WEIGHT: f64 = 0.85;
const CONTINENT_DETAIL_WEIGHT: f64 = 0.45;

/// Organic continent-detail wavelength (m): fragments plate coasts into
/// peninsulas, bays, and island chains. Octaves carry the fractal coastline
/// down to ~20 km; finer shoreline crinkle comes from the relief cascade.
const CONTINENT_WL_M: f64 = 1_500_000.0;
const CONTINENT_OCTAVES: f64 = 6.0;
/// Continent-scale domain warp: shears coastlines so masses fold and drift
/// rather than reading as round cells/blobs. Independent of the relief warp.
const CONTINENT_WARP_WL_M: f64 = 2_000_000.0;
const CONTINENT_WARP_AMP_M: f64 = 320_000.0;

/// Continentalness threshold for the coastline. Tuned for ~40 % land; verify
/// with `cargo run -p thalos_terrain --example world_map` (it prints the
/// area-weighted land fraction). Higher → less land.
const CONTINENT_C0: f64 = 0.105;
/// Half-width (in continentalness) of the land/sea transition that gates the
/// relief layers (hills/swell/mountains).
const LAND_MASK_W: f64 = 0.03;

/// Land hypsometry: a steep continental slope easing onto a platform of height
/// `LAND_PLATFORM_M`, plus a gentle linear interior gain so continent interiors
/// ride higher than their coasts. `LAND_K` sets how fast the slope saturates.
const LAND_PLATFORM_M: f64 = 420.0;
const LAND_K: f64 = 0.10;
const LAND_INTERIOR_GAIN_M: f64 = 650.0;

/// Ocean hypsometry: a shallow continental shelf shoulder dropping over the
/// continental slope to a flat abyssal plain. Widths are in continentalness.
const SHELF_DEPTH_M: f64 = 180.0;
const ABYSS_DEPTH_M: f64 = 4_000.0;
const SHELF_WIDTH_C: f64 = 0.05;
const SLOPE_WIDTH_C: f64 = 0.12;

/// Soft cap on how far sub-sea relief may breach the surface, so shallow
/// shelves crinkle into low islets/beaches instead of a field of flat-topped
/// mesas at the waterline.
const SHELF_BREACH_CAP_M: f64 = 28.0;

/// Rolling hills layer: mid wavelength, land-masked, LOD-aware octaves.
const HILLS_WL_M: f64 = 6_000.0;
const HILLS_AMP_M: f64 = 420.0;

/// Lowland swell: an ever-present gentle undulation, only lightly land-gated,
/// so plains (and the seabed) roll at the few-hundred-metre scale instead of
/// reading as a flat plane meeting the sky in a razor-straight horizon. It sits
/// between the hills layer and the shader's metre-scale micro-relief and, being
/// fBm, the LOD octave plan cascades it into finer ripples toward the camera.
const SWELL_WL_M: f64 = 1_400.0;
const SWELL_AMP_M: f64 = 25.0;

/// Mountain layer: ridged multifractal, LOD-aware. Its amplitude is no longer
/// gated by base altitude — it blends `PLAINS_MTN_AMP_M → MONTANE_MTN_AMP_M`
/// by the decorrelated [`ProceduralSurface::biome_weight`], so ruggedness is a
/// property of *which region* you're in, not how high the continent happens to
/// be. Land-gated only (no mountains rising out of the seabed).
const MOUNTAIN_WL_M: f64 = 20_000.0;
/// Ridged amplitude in plains regions (gentle — plains read as plains).
const PLAINS_MTN_AMP_M: f64 = 220.0;
/// Ridged amplitude in montane regions (tall enough that peaks reach the
/// shader's treeline/snow bands once the uplift is added).
const MONTANE_MTN_AMP_M: f64 = 3_500.0;

/// Orogeny field: a long-wavelength fBm **decorrelated** from the continent
/// field (own seed), thresholded into a montane weight in `[0, 1]`. It governs
/// where mountains are tall vs where the land reads as plains — ruggedness is a
/// property of *which region* you're in, not of how high the continent happens
/// to be. Shorter wavelength than the continents so a single continent can hold
/// both plains and a mountain belt.
///
/// This is the second plate-tectonics seam (alongside [`ProceduralSurface::
/// continentalness`]): a future backend replaces the body of [`ProceduralSurface
/// ::orogeny`] with a warped plate-margin / island-arc structure and mountain
/// amplitude follows it, without touching the height composition below. Per the
/// project's coast/relief separation rule, coastline detail must never feed this
/// field.
const OROGENY_WL_M: f64 = 420_000.0;
const OROGENY_OCTAVES: f64 = 4.0;
/// Montane where the field sits in its upper range; the gap is the transition
/// band (kept wide so the parameter blend doesn't smear character abruptly).
const OROGENY_LO: f64 = 0.05;
const OROGENY_HI: f64 = 0.45;
/// Base-elevation lift applied across montane land so ranges sit on raised
/// ground and their peaks clear the shader's treeline/snow lines.
const MONTANE_UPLIFT_M: f64 = 800.0;

// --- Runway-scenario siting scaffold ---------------------------------------
//
// The runway sits at a fixed body-fixed lat/lon (see `thalos_game::runway`).
// The flattened runway pad must sit on dry, flat land, so the continent field
// is nudged to guarantee that: a broad gentle continentalness bump centred on
// the site (so it is reliably mid-continent rather than a platform in open
// ocean) and a light orogeny suppression around it (so the immediate basin is
// plains). Both are smooth and wide enough to just read as "the site happens to
// be in a flat continental interior". The authored horizon massifs are additive
// and unaffected. Remove this whole scaffold when terrain-aware auto-siting
// picks a natural flat spot instead.

const RUNWAY_SITE_LAT_DEG: f64 = 7.6;
const RUNWAY_SITE_LON_DEG: f64 = 178.0;
/// Peak continentalness added at the site centre, and the lateral reach (m) over
/// which it eases to zero.
const RUNWAY_BIAS_AMP: f64 = 0.55;
const RUNWAY_BIAS_REACH_M: f64 = 1_400_000.0;
/// Orogeny is scaled down to this factor at the site centre, easing back to 1
/// over `RUNWAY_PLAINS_REACH_M`, so the runway basin is plains.
const RUNWAY_PLAINS_MIN: f64 = 0.10;
const RUNWAY_PLAINS_REACH_M: f64 = 350_000.0;

/// Body-fixed unit direction to the runway site (built once).
static RUNWAY_SITE_DIR: LazyLock<DVec3> =
    LazyLock::new(|| latlon_dir(RUNWAY_SITE_LAT_DEG, RUNWAY_SITE_LON_DEG));

/// Finest cascade depth for the LOD-aware layers.
const MAX_OCTAVES: f64 = 11.0;

// ---------------------------------------------------------------------------
// Authored mountain massifs (near the runway)
// ---------------------------------------------------------------------------
//
// Hand-placed mountain ranges so the runway scenario has scenery on the horizon
// (the planet's procedural montane belts are hundreds of km away from the fixed
// runway site). Each is a *localised* feature in the otherwise planet-wide
// field: a smooth elongated base swell modulated by a low-frequency ridged
// multifractal ("the basic base geometry"), then sculpted with
// `bevy_erosion_filter` to carve realistic drainage ridges and gullies — the
// same CPU erosion path the cold-desert shield volcano uses. They are fully
// LOD-invariant (always evaluated, fixed erosion octaves) so a range stays the
// same height whether viewed from the runway threshold or a distant approach;
// the cost is footprint-gated, so tiles outside every range pay almost nothing.
//
// Placement is body-fixed lat/lon. The runway site is lat 7.6°, lon 178.0°,
// takeoff heading 30° (see `thalos_game::runway`). Add/retune sites freely —
// this whole block is iteration scratch space; see the `runway_relief` /
// `runway_skyline` examples for visual tuning.

/// One authored mountain range. Shape parameters (`MASSIF_*` constants below)
/// are shared; each site sets only its placement, orientation, footprint, and a
/// seed `salt` so ranges don't share an identical ridge pattern.
struct MassifSite {
    lat_deg: f64,
    lon_deg: f64,
    /// Long-axis azimuth (degrees from north, toward east). Pick it broadside to
    /// the intended viewing direction so the range reads as a wall, not a spur.
    long_axis_deg: f64,
    /// Footprint half-extents (m): half-length along the long axis, half-width
    /// across it.
    half_len_m: f64,
    half_wid_m: f64,
    salt: u32,
}

const MASSIF_SITES: [MassifSite; 2] = [
    // Down the runway takeoff heading (30°), which points south-west of the
    // site; ~55 km out, broadside (long axis 30°) so it fills the down-runway
    // view as a wall.
    MassifSite {
        lat_deg: 7.105,
        lon_deg: 177.137,
        long_axis_deg: 30.0,
        half_len_m: 46_000.0,
        half_wid_m: 22_000.0,
        // 0 → the approved down-runway range, unchanged by the multi-site refactor.
        salt: 0x0000,
    },
    // North-east of the site (off to the side of the runway), a longer, deeper
    // range for scenery in the other direction.
    MassifSite {
        lat_deg: 8.48,
        lon_deg: 178.51,
        long_axis_deg: 120.0,
        half_len_m: 64_000.0,
        half_wid_m: 24_000.0,
        salt: 0x7C3E,
    },
];

/// Stretched-radius fraction inside which the base swell is at full height; from
/// here it eases to zero at the footprint edge (stretched radius 1.0).
const MASSIF_PLATEAU: f64 = 0.18;

/// Broad smooth uplift at the range centre (m) — the gentle base swell. Kept
/// low so the ridged spine (not a domed plateau) defines the relief.
const MASSIF_BASE_AMP_M: f64 = 950.0;
/// Ridged-multifractal relief riding on the swell (m): the multi-peak spine.
const MASSIF_RIDGE_AMP_M: f64 = 3_050.0;
/// Wavelength of the largest ridged feature (m) and its octave count. Shorter
/// than the range length so several distinct summits march along the crest.
const MASSIF_RIDGE_WL_M: f64 = 17_000.0;
const MASSIF_RIDGE_OCTAVES: f64 = 7.0;
/// Wavelength of the along-crest tall/short modulation (m).
const MASSIF_CREST_WL_M: f64 = 48_000.0;

/// Multiplier on the erosion-filter height delta (m of carved relief). The
/// filter's own `scale * strength` already sets the per-octave displacement;
/// this trims the final contribution.
const MASSIF_EROSION_GAIN: f64 = 0.85;

/// Worst-case massif column height, folded into [`HEIGHT_RANGE_M`].
const MASSIF_PEAK_M: f64 = MASSIF_BASE_AMP_M + MASSIF_RIDGE_AMP_M + 900.0;

/// Precomputed per-site tangent frame `(anchor, across, long_axis)`. Constant
/// (derived only from the placement constants), so it's built once rather than
/// per height sample. `across` ⟂ `long_axis`, both tangent at the anchor.
static MASSIF_FRAMES: LazyLock<[(DVec3, DVec3, DVec3); MASSIF_SITES.len()]> = LazyLock::new(|| {
    MASSIF_SITES.map(|site| {
        let anchor = latlon_dir(site.lat_deg, site.lon_deg);
        let east = DVec3::Y.cross(anchor).normalize();
        let north = anchor.cross(east).normalize();
        let az = site.long_axis_deg.to_radians();
        let long_axis = (north * az.cos() + east * az.sin()).normalize();
        let across = anchor.cross(long_axis).normalize();
        (anchor, across, long_axis)
    })
});

/// Worst-case positive land column: continental platform + interior gain +
/// montane uplift + the taller of the procedural montane ridge or an authored
/// massif + hills + swell + margin.
const LAND_PEAK_M: f64 = LAND_PLATFORM_M
    + LAND_INTERIOR_GAIN_M
    + MONTANE_UPLIFT_M
    + MONTANE_MTN_AMP_M.max(MASSIF_PEAK_M)
    + HILLS_AMP_M
    + SWELL_AMP_M
    + 600.0;

/// Vertical envelope the encoder/collider size from: the larger of the
/// worst-case land column and the abyssal ocean depth (plus seabed relief).
/// Heights are clamped to `±HEIGHT_RANGE_M` on encode. (Widening this coarsens
/// the u16 height-atlas quantisation everywhere — here ~0.2 m/step, acceptable.)
const HEIGHT_RANGE_M: f32 = LAND_PEAK_M.max(ABYSS_DEPTH_M + HILLS_AMP_M + 400.0) as f32;

// ---------------------------------------------------------------------------
// Surface
// ---------------------------------------------------------------------------

/// The runtime procedural surface for one body.
///
/// Cheap to construct and `Copy`-by-value-cheap to clone (just a few scalars),
/// so the two construction sites that need it (the ground tile provider and the
/// near-surface height source) build identical instances from the same body
/// params and stay in lockstep without sharing an `Arc`.
#[derive(Debug, Clone, Copy)]
pub struct ProceduralSurface {
    radius_m: f64,
    seed: u32,
}

impl ProceduralSurface {
    pub fn new(radius_m: f32, seed: u32) -> Self {
        Self {
            radius_m: radius_m.max(1.0) as f64,
            seed,
        }
    }

    /// Geometric height (m above the reference radius) **and** the montane
    /// orogeny weight at body-fixed unit direction `dir`, evaluated at `lod_m`
    /// metres per sample. The orogeny weight is returned alongside the height
    /// (it's computed in the height path anyway, since it drives uplift and the
    /// mountain amplitude) so the albedo path reuses it for free.
    fn height_and_orogeny(&self, dir: DVec3, lod_m: f32) -> (f64, f64) {
        let dir = dir.normalize_or_zero();
        if dir == DVec3::ZERO {
            return (0.0, 0.0);
        }
        // Body-local position in metres (f64), the precision-critical step.
        let p = dir * self.radius_m;

        // Relief-scale domain warp (breaks up the lattice for the hill / mountain
        // bands; the continent field warps itself at its own scale).
        let wp = p / WARP_WL_M;
        let warp = DVec3::new(
            fbm(wp, self.seed ^ 0x1111, 2.0),
            fbm(wp + DVec3::splat(31.4), self.seed ^ 0x2222, 2.0),
            fbm(wp - DVec3::splat(17.2), self.seed ^ 0x3333, 2.0),
        ) * WARP_AMP_M;
        let pw = p + warp;

        // --- Macro: continents & oceans ------------------------------------
        // Continentalness (the separable plate-tectonics seam) → bimodal
        // hypsometric height about sea level (0 m). LOD-invariant: low-frequency,
        // so cost is fixed, it never aliases, and a parent→child tile handoff
        // never terraces the macro shape.
        let c = self.continentalness(p) + self.runway_land_bias(dir);
        let macro_h = hypsometric_height(c);
        let land_mask = smoothstep(CONTINENT_C0 - LAND_MASK_W, CONTINENT_C0 + LAND_MASK_W, c);

        // Montane orogeny, decorrelated from the continent field, suppressed to
        // plains around the runway site.
        let orogeny = self.orogeny(pw) * self.runway_plains_factor(dir);

        // --- Relief cascade riding on the macro base -----------------------
        // Montane uplift on land so ranges sit high enough to reach the shader's
        // treeline / snow bands.
        let uplift = MONTANE_UPLIFT_M * orogeny * land_mask;

        // Rolling hills, stronger on land.
        let hills_oct = octaves_for_lod(lod_m, HILLS_WL_M);
        let hills = fbm(pw / HILLS_WL_M, self.seed ^ 0x5151, hills_oct)
            * HILLS_AMP_M
            * (0.35 + 0.65 * land_mask);

        // Lowland / seabed swell: barely land-gated so neither plains nor the
        // abyssal floor go dead flat. fBm, so the LOD octave plan fades finer
        // ripples in toward the camera.
        let swell_oct = octaves_for_lod(lod_m, SWELL_WL_M);
        let swell = fbm(pw / SWELL_WL_M, self.seed ^ 0x57E1, swell_oct)
            * SWELL_AMP_M
            * (0.55 + 0.45 * land_mask);

        // Ridged mountains: amplitude blends plains↔montane by orogeny, gated to
        // land.
        let mtn_oct = octaves_for_lod(lod_m, MOUNTAIN_WL_M);
        let mtn_amp = lerp(orogeny, PLAINS_MTN_AMP_M, MONTANE_MTN_AMP_M) * land_mask;
        let mountains = ridged(pw / MOUNTAIN_WL_M, self.seed ^ 0x9A9A, mtn_oct) * mtn_amp;

        // Combine relief with the macro base, soft-capping how far sub-sea relief
        // may breach the surface (so shallow shelves crinkle into islets, not a
        // field of waterline mesas).
        let height = combine_macro_and_relief(macro_h, uplift + hills + swell + mountains);

        // Authored, erosion-sculpted mountain ranges near the runway. Additive
        // and footprint-gated (zero outside every envelope), so they don't
        // perturb the rest of the planet and aren't subject to the shelf cap.
        // `rock` skews the macro albedo greyer where a range stands.
        let (massif_m, rock) = self.mountain_massifs(p);

        (height + massif_m, orogeny.max(rock))
    }

    /// Continentalness in `~[-1, 1]` at body-local position `p` (m), warped at
    /// continent scale. This is the SEPARABLE macro field: a later
    /// plate-tectonics backend replaces the body of this one function (and
    /// [`Self::orogeny`]) to drive continents / mountain belts / island arcs from
    /// plate margins, without touching the hypsometric remap or relief cascade.
    fn continentalness(&self, p: DVec3) -> f64 {
        // Continent-scale domain warp: shears coastlines so masses fold and drift
        // instead of reading as round cells/blobs.
        let cwp = p / CONTINENT_WARP_WL_M;
        let warp = DVec3::new(
            fbm(cwp, self.seed ^ 0xCA01, 3.0),
            fbm(cwp + DVec3::splat(53.7), self.seed ^ 0xCA02, 3.0),
            fbm(cwp - DVec3::splat(12.9), self.seed ^ 0xCA03, 3.0),
        ) * CONTINENT_WARP_AMP_M;
        let cp = p + warp;

        // Distinct land/ocean plates (Worley cells, each randomly biased) + an
        // organic fBm overlay that breaks the cell edges into real coastlines.
        let plate = plate_value(cp / CONTINENT_CELL_M, self.seed ^ 0x71A7);
        let detail = fbm(cp / CONTINENT_WL_M, self.seed ^ 0xC0FF, CONTINENT_OCTAVES);
        PLATE_WEIGHT * plate + CONTINENT_DETAIL_WEIGHT * detail
    }

    /// Broad gentle continentalness bump centred on the runway site so the
    /// flattened pad is reliably inland on a continent (see the runway-siting
    /// scaffold note). Zero past `RUNWAY_BIAS_REACH_M`.
    fn runway_land_bias(&self, dir: DVec3) -> f64 {
        let ang = dir.dot(*RUNWAY_SITE_DIR).clamp(-1.0, 1.0).acos();
        let reach = RUNWAY_BIAS_REACH_M / self.radius_m;
        let t = (1.0 - ang / reach).clamp(0.0, 1.0);
        RUNWAY_BIAS_AMP * t * t * (3.0 - 2.0 * t)
    }

    /// Multiplier in `[RUNWAY_PLAINS_MIN, 1]` that suppresses orogeny near the
    /// runway site so its basin reads as plains. `1` everywhere else.
    fn runway_plains_factor(&self, dir: DVec3) -> f64 {
        let ang = dir.dot(*RUNWAY_SITE_DIR).clamp(-1.0, 1.0).acos();
        let reach = RUNWAY_PLAINS_REACH_M / self.radius_m;
        let t = (1.0 - ang / reach).clamp(0.0, 1.0);
        1.0 - (1.0 - RUNWAY_PLAINS_MIN) * t * t * (3.0 - 2.0 * t)
    }

    /// Summed contribution of every authored mountain range at body-local
    /// position `p` (m). Sites are far apart, so this is a plain sum; `rock` is
    /// the max footprint envelope. Returns `(0, 0)` away from all ranges — the
    /// planet-wide common case — for free.
    fn mountain_massifs(&self, p: DVec3) -> (f64, f64) {
        let mut height = 0.0;
        let mut rock = 0.0_f64;
        for (site, frame) in MASSIF_SITES.iter().zip(MASSIF_FRAMES.iter()) {
            let (h, r) = self.massif_contribution(p, site, frame);
            height += h;
            rock = rock.max(r);
        }
        (height, rock)
    }

    /// One range's contribution. `frame` is its precomputed `(anchor, across,
    /// long_axis)`; `rock_weight ∈ [0, 1]` is the footprint envelope.
    fn massif_contribution(
        &self,
        p: DVec3,
        site: &MassifSite,
        frame: &(DVec3, DVec3, DVec3),
    ) -> (f64, f64) {
        let (anchor, across, long_axis) = *frame;

        // Local tangent-plane metric coords (valid to curvature error well under
        // a metre across the footprint on a 3000 km body).
        let rel = p - anchor * self.radius_m;
        // Cheap radial reject: the stretched ellipse fits inside a disc of
        // radius `half_len`, so anything farther can't be in the footprint. This
        // short-circuits the planet-wide common case (samples nowhere near a
        // range) before any dot products or noise.
        if rel.length_squared() > site.half_len_m * site.half_len_m {
            return (0.0, 0.0);
        }
        let x = rel.dot(across); // across the range
        let y = rel.dot(long_axis); // along the range

        // Precise footprint test.
        let u = x / site.half_wid_m;
        let v = y / site.half_len_m;
        if u * u + v * v > 1.0 {
            return (0.0, 0.0);
        }

        let base_h = self.massif_base(x, y, site);
        let env = massif_envelope(x, y, site);
        if env <= 0.0 {
            return (0.0, 0.0);
        }

        // Base slope (analytic-ish, via finite difference of the smooth base) so
        // the erosion filter knows which way water would run and carves gullies
        // down the flanks. eps below the finest ridged octave.
        let eps = 60.0;
        let dh_dx = ((self.massif_base(x + eps, y, site) - self.massif_base(x - eps, y, site))
            / (2.0 * eps)) as f32;
        let dh_dy = ((self.massif_base(x, y + eps, site) - self.massif_base(x, y - eps, site))
            / (2.0 * eps)) as f32;

        let params = massif_erosion_params();
        let offset = massif_erosion_offset(self.seed ^ site.salt);
        let fade = ((base_h / MASSIF_PEAK_M) * 2.0 - 1.0).clamp(-1.0, 1.0) as f32;
        let base = Vec3::new(base_h as f32, dh_dx, dh_dy);
        let res = erosion_filter(
            Vec2::new(x as f32 + offset.x, y as f32 + offset.y),
            base,
            fade,
            &params,
        );
        let erosion_h = res.delta.x as f64 * MASSIF_EROSION_GAIN * env;

        (base_h + erosion_h, env)
    }

    /// Smooth base swell of a range at local `(x, y)` metres: an elongated hump
    /// modulated by a low-frequency ridged multifractal for a multi-peak spine.
    /// Pure and cheap (one ridged eval) so the slope finite-difference can call
    /// it a few times.
    fn massif_base(&self, x: f64, y: f64, site: &MassifSite) -> f64 {
        let env = massif_envelope(x, y, site);
        if env <= 0.0 {
            return 0.0;
        }
        // 2-D ridged slice (z = 0): deterministic, multi-ridge spine.
        let ridge = ridged(
            DVec3::new(x, y, 0.0) / MASSIF_RIDGE_WL_M,
            self.seed ^ 0x4D54 ^ site.salt,
            MASSIF_RIDGE_OCTAVES,
        );
        // Long-wavelength modulation along the crest so the range has tall
        // massifs and lower saddles instead of a uniform spine — a varied
        // skyline. fBm in [-1, 1] → factor in [0.6, 1.0].
        let crest = 0.6
            + 0.4
                * (0.5
                    + 0.5
                        * fbm(
                            DVec3::new(y, 0.0, 0.0) / MASSIF_CREST_WL_M,
                            self.seed ^ 0x6E57 ^ site.salt,
                            3.0,
                        ));
        env * (MASSIF_BASE_AMP_M + MASSIF_RIDGE_AMP_M * ridge * crest)
    }

    /// Decorrelated montane-orogeny weight in `[0, 1]` at warped position `pw`.
    /// A long-wavelength field independent of the continent field, so mountains
    /// form their own regions rather than tracking base altitude. Kept as one
    /// function so a future plate-margin / island-arc structure can replace the
    /// body without touching the height composition.
    fn orogeny(&self, pw: DVec3) -> f64 {
        let b = fbm(pw / OROGENY_WL_M, self.seed ^ 0xB10E, OROGENY_OCTAVES);
        smoothstep(OROGENY_LO, OROGENY_HI, b)
    }

    /// Provisional macro albedo (linear RGB) by altitude band. This is a
    /// stand-in so Slice 0 renders something plausible; Slice 2 replaces the
    /// baked-albedo path with procedural in-shader materials.
    fn albedo_at(&self, height_m: f64, orogeny: f64) -> Vec3 {
        // Linear-RGB band anchors.
        let shore = Vec3::new(0.30, 0.27, 0.18); // tan/sand near 0 m
        let lowland = Vec3::new(0.08, 0.16, 0.06); // green
        let upland = Vec3::new(0.12, 0.11, 0.08); // brown
        let rock = Vec3::new(0.13, 0.12, 0.11); // grey rock
        let snow = Vec3::new(0.62, 0.64, 0.68); // snow

        let h = height_m;
        let mut c = mix(shore, lowland, smoothstep(0.0, 120.0, h) as f32);
        c = mix(c, upland, smoothstep(120.0, 900.0, h) as f32);
        c = mix(c, rock, smoothstep(900.0, 2_400.0, h) as f32);
        c = mix(c, snow, smoothstep(2_700.0, 3_400.0, h) as f32);
        // Montane macro tone skews greyer/rockier even at moderate altitude;
        // the shader's slope/altitude bands add the real scree + snow on top.
        c = mix(c, rock, (0.35 * orogeny) as f32);
        c
    }
}

impl SurfaceQuery for ProceduralSurface {
    fn sample(&self, dir: Vec3, lod_m: f32) -> SurfaceSample {
        self.sample_d(dir.as_dvec3(), lod_m)
    }

    fn sample_d(&self, dir: DVec3, lod_m: f32) -> SurfaceSample {
        let (height_m, orogeny) = self.height_and_orogeny(dir, lod_m);
        SurfaceSample {
            height_m: height_m as f32,
            albedo_linear: self.albedo_at(height_m, orogeny),
            roughness: 0.92,
        }
    }

    fn sample_height_m(&self, dir: Vec3, lod_m: f32) -> f32 {
        self.height_and_orogeny(dir.as_dvec3(), lod_m).0 as f32
    }

    fn radius_m(&self) -> f32 {
        self.radius_m as f32
    }

    fn height_range_m(&self) -> f32 {
        HEIGHT_RANGE_M
    }
}

// ---------------------------------------------------------------------------
// LOD plan
// ---------------------------------------------------------------------------

/// Continuous octave count for a layer of base wavelength `base_wl_m` viewed at
/// `lod_m` metres per sample. Fractional values fade the top octave in smoothly
/// (see [`fbm`]) so a tile gaining an octave across an LOD boundary blends
/// rather than steps. `lod_m <= 0` requests full detail.
fn octaves_for_lod(lod_m: f32, base_wl_m: f64) -> f64 {
    if lod_m <= 0.0 {
        return MAX_OCTAVES;
    }
    let ratio = base_wl_m / (2.0 * lod_m as f64);
    if ratio <= 1.0 {
        return 1.0;
    }
    (ratio.log2() + 1.0).clamp(1.0, MAX_OCTAVES)
}

// ---------------------------------------------------------------------------
// Hypsometry (continentalness → signed macro height)
// ---------------------------------------------------------------------------

/// Map continentalness `c` to a signed macro height (m about sea level = 0 m),
/// producing Earth-like **bimodal** hypsometry: flat abyssal plains, a narrow
/// continental shelf + slope, and a raised continental platform. The steep
/// segment at the [`CONTINENT_C0`] crossing is the shelf break / coastline, so
/// the two modes (deep ocean, high land) carry most of the surface area and the
/// transition between them carries little — the orbital silhouette that raw fBm
/// (unimodal) can't produce. Continuous at `c == CONTINENT_C0` (both branches
/// give 0).
fn hypsometric_height(c: f64) -> f64 {
    if c >= CONTINENT_C0 {
        let t = c - CONTINENT_C0;
        // Steep continental slope saturating onto a platform, plus a gentle
        // linear interior gain so continent interiors ride higher than coasts.
        LAND_PLATFORM_M * (1.0 - (-t / LAND_K).exp()) + LAND_INTERIOR_GAIN_M * t
    } else {
        let s = CONTINENT_C0 - c;
        // Shallow shelf shoulder, then the continental slope dropping to a flat
        // abyssal plain (the smoothstep saturates → constant depth far out).
        let shelf = SHELF_DEPTH_M * smoothstep(0.0, SHELF_WIDTH_C, s);
        let abyss = (ABYSS_DEPTH_M - SHELF_DEPTH_M)
            * smoothstep(SHELF_WIDTH_C, SHELF_WIDTH_C + SLOPE_WIDTH_C, s);
        -(shelf + abyss)
    }
}

/// Add relief to the macro base, soft-limiting how far **sub-sea** relief may
/// breach the surface. Land (`macro_h >= 0`) and any point that stays below sea
/// level pass through unchanged; only a sub-sea point whose relief would lift it
/// above 0 is saturated toward [`SHELF_BREACH_CAP_M`], so shallow shelves read
/// as low islets / beaches rather than a field of flat-topped waterline mesas.
fn combine_macro_and_relief(macro_h: f64, relief: f64) -> f64 {
    let h = macro_h + relief;
    if macro_h >= 0.0 || h <= 0.0 {
        return h;
    }
    SHELF_BREACH_CAP_M * (1.0 - (-h / SHELF_BREACH_CAP_M).exp())
}

// ---------------------------------------------------------------------------
// Noise (self-contained f64 gradient/Perlin + fractal layers)
// ---------------------------------------------------------------------------

/// Fractional-octave fBm in `~[-1, 1]`. The final (fractional) octave is
/// amplitude-weighted by `frac(octaves)` for smooth LOD blending.
fn fbm(p: DVec3, seed: u32, octaves: f64) -> f64 {
    let full = octaves.floor().max(0.0) as u32;
    let frac = octaves - full as f64;
    let mut amp = 0.5_f64;
    let mut freq = 1.0_f64;
    let mut sum = 0.0_f64;
    let mut norm = 0.0_f64;
    for i in 0..full {
        sum += amp * perlin3(p * freq, seed.wrapping_add(i.wrapping_mul(1013)));
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    if frac > 0.0 {
        sum += amp * frac * perlin3(p * freq, seed.wrapping_add(full.wrapping_mul(1013)));
        norm += amp * frac;
    }
    sum / norm.max(1e-6)
}

/// Fractional-octave ridged multifractal in `~[0, 1]` with sharp crests.
fn ridged(p: DVec3, seed: u32, octaves: f64) -> f64 {
    let full = octaves.floor().max(0.0) as u32;
    let frac = octaves - full as f64;
    let mut amp = 0.5_f64;
    let mut freq = 1.0_f64;
    let mut sum = 0.0_f64;
    let mut norm = 0.0_f64;
    for i in 0..full {
        let n = 1.0 - perlin3(p * freq, seed.wrapping_add(i.wrapping_mul(1013))).abs();
        sum += amp * n * n;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    if frac > 0.0 {
        let n = 1.0 - perlin3(p * freq, seed.wrapping_add(full.wrapping_mul(1013))).abs();
        sum += amp * frac * n * n;
        norm += amp * frac;
    }
    sum / norm.max(1e-6)
}

/// 3D gradient (Perlin) noise in `~[-1, 1]`, f64.
fn perlin3(p: DVec3, seed: u32) -> f64 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let zi = p.z.floor();
    let xf = p.x - xi;
    let yf = p.y - yi;
    let zf = p.z - zi;
    let (x0, y0, z0) = (xi as i64, yi as i64, zi as i64);

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let g = |cx: i64, cy: i64, cz: i64, fx: f64, fy: f64, fz: f64| {
        let grad = GRAD3[(hash3(cx, cy, cz, seed) % 12) as usize];
        grad[0] * fx + grad[1] * fy + grad[2] * fz
    };

    let n000 = g(x0, y0, z0, xf, yf, zf);
    let n100 = g(x0 + 1, y0, z0, xf - 1.0, yf, zf);
    let n010 = g(x0, y0 + 1, z0, xf, yf - 1.0, zf);
    let n110 = g(x0 + 1, y0 + 1, z0, xf - 1.0, yf - 1.0, zf);
    let n001 = g(x0, y0, z0 + 1, xf, yf, zf - 1.0);
    let n101 = g(x0 + 1, y0, z0 + 1, xf - 1.0, yf, zf - 1.0);
    let n011 = g(x0, y0 + 1, z0 + 1, xf, yf - 1.0, zf - 1.0);
    let n111 = g(x0 + 1, y0 + 1, z0 + 1, xf - 1.0, yf - 1.0, zf - 1.0);

    let x00 = lerp(u, n000, n100);
    let x10 = lerp(u, n010, n110);
    let x01 = lerp(u, n001, n101);
    let x11 = lerp(u, n011, n111);
    let y0l = lerp(v, x00, x10);
    let y1l = lerp(v, x01, x11);
    lerp(w, y0l, y1l)
}

/// The 12 classic Perlin edge gradients.
const GRAD3: [[f64; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

/// Integer hash of a lattice cell + seed → well-mixed u32.
fn hash3(x: i64, y: i64, z: i64, seed: u32) -> u32 {
    let mut h = seed.wrapping_mul(0x9E37_79B1);
    h ^= (x as u32).wrapping_mul(0x85EB_CA77);
    h = h.rotate_left(13);
    h ^= (y as u32).wrapping_mul(0xC2B2_AE3D);
    h = h.rotate_left(13);
    h ^= (z as u32).wrapping_mul(0x27D4_EB2F);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2545_F491);
    h ^ (h >> 13)
}

/// Smoothly-interpolated random per-cell value in `~[-1, 1]` — a cellular
/// (Worley) "plate" field. Each integer lattice cell carries a jittered feature
/// point and a random value; the result is a soft-min distance-weighted blend of
/// the 27 nearest cells' values, so cell interiors read as distinct plateaus
/// (continents / ocean basins) while boundaries blend into smooth coastlines.
/// [`PLATE_SHARPNESS`] sets how plateau-like vs blended the field is.
fn plate_value(p: DVec3, seed: u32) -> f64 {
    let pi = DVec3::new(p.x.floor(), p.y.floor(), p.z.floor());
    let mut wsum = 0.0_f64;
    let mut vsum = 0.0_f64;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let cell = pi + DVec3::new(dx as f64, dy as f64, dz as f64);
                let (cx, cy, cz) = (cell.x as i64, cell.y as i64, cell.z as i64);
                // Jittered feature point inside the cell.
                let hp = hash3(cx, cy, cz, seed);
                let jitter = DVec3::new(
                    (hp & 0xff) as f64 / 255.0,
                    ((hp >> 8) & 0xff) as f64 / 255.0,
                    ((hp >> 16) & 0xff) as f64 / 255.0,
                );
                let fp = cell + jitter;
                // Decorrelated random value for this plate, in [-1, 1].
                let hv = hash3(cx, cy, cz, seed ^ 0x5BD1_C0DE);
                let val = (hv as f64 / u32::MAX as f64) * 2.0 - 1.0;
                let d = (fp - p).length();
                let w = (-d * PLATE_SHARPNESS).exp();
                wsum += w;
                vsum += w * val;
            }
        }
    }
    vsum / wsum.max(1e-9)
}

fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}

fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0).max(f64::EPSILON)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Mountain-massif helpers
// ---------------------------------------------------------------------------

/// Body-fixed unit direction for a latitude/longitude (degrees). Matches the
/// convention in `thalos_game::runway::latlon_dir`.
fn latlon_dir(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()).normalize()
}

/// Footprint envelope in `[0, 1]`: full inside [`MASSIF_PLATEAU`] of the
/// stretched radius, easing to zero at the edge (stretched radius 1).
fn massif_envelope(x: f64, y: f64, site: &MassifSite) -> f64 {
    let u = x / site.half_wid_m;
    let v = y / site.half_len_m;
    let rr = (u * u + v * v).sqrt();
    1.0 - smoothstep(MASSIF_PLATEAU, 1.0, rr)
}

/// Erosion-filter parameters tuned for a km-scale mountain range (the crate
/// defaults target a ~22-unit toy mesh). `scale` is the largest erosion
/// wavelength in metres; effective per-octave displacement is `scale * strength`
/// (~1.4 km here), accumulated over the octaves. `onset` is eased down so the
/// gentle base flanks still trigger the gully carving.
fn massif_erosion_params() -> ErosionFilterParams {
    let d = ErosionFilterParams::default();
    ErosionFilterParams {
        scale: 6_500.0,
        strength: 0.14,
        gully_weight: 0.55,
        detail: 1.7,
        octaves: 7,
        onset: d.onset * 0.16,
        assumed_slope: Vec2::new(0.5, 0.95),
        ..d
    }
}

/// Stable per-seed offset so the erosion field doesn't lock to the local origin.
fn massif_erosion_offset(seed: u32) -> Vec2 {
    let h = hash3(seed as i64, 0x517E, 0x4D54, seed ^ 0xA53F);
    Vec2::new((h & 0xffff) as f32 * 13.0, (h >> 16) as f32 * 13.0)
}
