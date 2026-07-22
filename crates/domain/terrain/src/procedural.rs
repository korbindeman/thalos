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

/// Continentalness threshold for the coastline. Tuned for the authored **35 %
/// land area** (docs/lore/solar_system.md §II — Thalos is an ocean-fed
/// homeworld); verify with `just map` (it prints the area-weighted land
/// fraction). Higher → less land.
const CONTINENT_C0: f64 = 0.143;

// --- Coast-character refinement (BL-6) --------------------------------------
//
// Near the threshold the continentalness is refined by a ruggedness-modulated
// fractal domain warp (crenulation) + a rugged-coast islet sprinkle — see
// `ProceduralSurface::continentalness`. Regional variety: depositional coasts
// stay sweeping beach arcs, erosional coasts get rias/coves/archipelagos.

/// Regional coast-character wavelength (m): the scale on which coasts switch
/// between smooth/depositional and rugged/erosional stretches.
const COAST_RUGGED_WL_M: f64 = 6_000_000.0;
/// Crenulation warp base wavelength (m); ~4 octaves reach ≈ WL/8.
const COAST_CREN_WL_M: f64 = 220_000.0;
const COAST_CREN_OCT: f64 = 4.0;
/// Coastline-displacement amplitude (m) of the crenulation warp at the two
/// character extremes. Smooth coasts keep a gentle sway; rugged coasts fold
/// into headlands, coves, and rias.
const COAST_CREN_AMP_SMOOTH_M: f64 = 15_000.0;
const COAST_CREN_AMP_RUGGED_M: f64 = 85_000.0;
/// Offshore islet/skerry sprinkle: additive continentalness on rugged coasts
/// (macro islands are legitimate — only LOD-faded relief may never cross the
/// waterline). Wavelength ~90 km over 3 octaves → islands 10–45 km.
/// **Thresholded to clumps** (BL-10): only where the sprinkle noise rises past
/// `COAST_ISLE_BIAS` does it add land, so islands are sparse, distinct groups
/// with real channels between them — the old signed, everywhere-active term
/// dusted rugged shores with semi-submerged fringe fragments (the "marshy
/// semi-island" look) and carved as much as it added.
const COAST_ISLE_WL_M: f64 = 90_000.0;
const COAST_ISLE_BIAS: f64 = 0.22;
const COAST_ISLE_GAIN_C: f64 = 0.028;
/// Half-width (continentalness) of the refinement gate about `CONTINENT_C0`.
/// Must comfortably exceed the largest |c_warped + isle − c_base| so warped
/// coast excursions are never truncated at the band edge; the interior/deep
/// ocean outside the band skip the refinement cost entirely.
const COAST_REFINE_BAND_C: f64 = 0.11;
/// Half-width (in continentalness) of the land/sea transition that gates the
/// relief layers (hills/swell/mountains).
const LAND_MASK_W: f64 = 0.03;

/// Land hypsometry: a steep continental slope easing onto a platform of height
/// `LAND_PLATFORM_M`, plus a gentle linear interior gain so continent interiors
/// ride higher than their coasts. `LAND_K` sets how fast the slope saturates.
///
/// Authored to the lore (docs/lore/solar_system.md §II): Thalos is a
/// **geologically old** world — eroded, subdued land that keeps typical
/// terrain low in the lush ecological bands (the planet "looks lush" from
/// orbit), with ruggedness supplied by the decorrelated montane regions, not
/// by a high-riding platform. The old 420 m + 650 m values pushed mean land
/// toward ~1 km and let the altitude bands crush the climate palette
/// (TM-P3 rebalance, 2026-07-20).
const LAND_PLATFORM_M: f64 = 300.0;
const LAND_K: f64 = 0.10;
const LAND_INTERIOR_GAIN_M: f64 = 400.0;

/// Ocean hypsometry: a broad shallow continental shelf shoulder dropping over
/// the continental slope to the abyssal plain. Widths are in continentalness;
/// the wide shelf gives a gentle shallow band at the coast (the natural-coast
/// look) rather than land plunging straight to deep water.
const SHELF_DEPTH_M: f64 = 150.0;
const ABYSS_DEPTH_M: f64 = 4_000.0;
const SHELF_WIDTH_C: f64 = 0.09;
const SLOPE_WIDTH_C: f64 = 0.16;

/// Foreshore drop: a fast depth gain immediately off the waterline (saturating
/// at `FORESHORE_DEPTH_M` over `FORESHORE_WIDTH_C` of continentalness —
/// roughly the first kilometre or two of shore at typical coastal `c`
/// gradients). Beaches read as beaches: wade in and the bottom falls away,
/// instead of the shelf shoulder's ~1 m/km leaving kilometres of ankle-deep
/// see-through water (the INC-0003 "mushy shoreline"). Sized so that even
/// archipelago zones — where the continentalness field meanders near its
/// threshold for hundreds of km and the shelf shoulder alone stays in the
/// −5…−25 m band — drop below the shallow-colour band promptly: real
/// island seas read as water with pale fringes, not one vast translucent
/// bank. The wide gentle shelf beyond is unchanged. ADR-20260720T185957Z-coastline-as-authored-data.
const FORESHORE_DEPTH_M: f64 = 15.0;
const FORESHORE_WIDTH_C: f64 = 2.5e-4;

/// Beach berm: the land-side mirror of the foreshore drop (BL-10). Land lifts
/// a few metres off the waterline within the first ~km of shore, so the coast
/// is a defined sand strand instead of kilometres of near-sea-level flats —
/// the "marshy" look, and the zone where the water renderer's wet-edge feather
/// smeared widest. Together with the foreshore this makes the waterline a
/// crisp crossing: −15 m … +4 m over ~2 km instead of ±2 m over 20.
const BEACH_RISE_M: f64 = 4.0;
const BEACH_RISE_WIDTH_C: f64 = 2.5e-4;

/// Seabed relief: a rolling abyssal-hills / seamount layer so the deep ocean
/// floor is not a dead-flat plane (which reads as a uniform flat ocean and feeds
/// the water renderer no depth variation). Gated to deep water (see
/// `deep_factor` in the height path) so it never churns the shelf or the coast.
const SEABED_WL_M: f64 = 170_000.0;
const SEABED_AMP_M: f64 = 950.0;
/// Depth band over which seabed relief fades in: none on the shelf (shallower
/// than `SEABED_FADE_HI_M`), full in the deep (below `SEABED_FADE_LO_M`).
const SEABED_FADE_HI_M: f64 = -300.0;
const SEABED_FADE_LO_M: f64 = -1_500.0;

/// Depth (m) the awash-reef cap saturates to: sub-sea relief that would breach
/// the surface instead shoals asymptotically to this just-below-the-surface
/// depth. Relief must NEVER actually cross sea level offshore — the relief
/// cascade is LOD-aware, so a relief-defined islet field appears/moves with
/// camera distance (the probe measured 40% → 13% breach coverage across LODs),
/// and the old `+14 m` breach cap turned shallow shelves into flat-topped mesa
/// fields pocked with circular noise-dip holes — halftone speckle from orbit
/// (INC-0003). Islands come only from the LOD-invariant macro field. Deep
/// enough that reef tops read as pale submerged bathymetry — at the original
/// 2 m they sat inside the shoreline wet-edge feather and rendered as
/// surface-breaking sand-green "scum" rings. Deepening the saturation (6 →
/// 12 → 25 m) never killed the island shoal halos, because the fold only
/// touches relief that actually BREACHES — the halos are mostly *legal*
/// non-breaching relief tops parked just below the surface (−0.1…−5 m,
/// inside the shallow-tint band). The offshore shallow clearance below is
/// the structural fix; this constant now only sets how deep large breaches
/// saturate.
const AWASH_REEF_DEPTH_M: f64 = 25.0;

/// Offshore shallow clearance (BL-10): away from the macro foreshore, the
/// combined seabed may not occupy the shallow band at all — heights in
/// `(−OFFSHORE_SHALLOW_CLEAR_M, 0)` are compressed to its bottom
/// `OFFSHORE_SHALLOW_KEEP` fraction, so relief tops (folded or merely
/// near-surface) sit below the shallow-colour e-folding and open water stays
/// deep blue. Feathered on MACRO depth (full effect only where the macro
/// seabed is deeper than 2× the clearance), so the genuine beach foreshore —
/// where the macro field itself is shallow — keeps its pale shallows. The
/// shallow tint thereby becomes an exclusive property of real coastlines.
/// 14 m: the compressed band bottoms out near −12.7 m, ≥ 1.5 e-foldings of
/// the 8 m shallow-colour ramp even on slant paths — offshore tops tint
/// ≤ ~15 %, reading as faint deep-bank texture rather than pale speckle.
const OFFSHORE_SHALLOW_CLEAR_M: f64 = 14.0;
const OFFSHORE_SHALLOW_KEEP: f64 = 0.15;

/// Elevation half-width (m about sea level) of the coastal band in which the
/// LOD-aware relief cascade is faded out, so the **shoreline is the crossing of
/// the LOD-invariant macro (continent) field**, not of the relief on a near-flat
/// coast. Without this the waterline wandered kilometres between the coarse LOD a
/// coast is drawn at from orbit and the fine LOD up close (the relief octaves
/// fade in/out with camera distance, and a gentle coast turns a few metres of
/// height change into kilometres of horizontal shift — see
/// `examples/coastline_lod.rs`). Wider → a broader smooth foreshore but a more
/// rock-steady waterline; on a flat coast the smoothed apron is ~`COAST_BAND_M /
/// slope` wide, which is where the pinning is needed most.
const COAST_BAND_M: f64 = 60.0;

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

/// Medium-scale ridged band riding on the coarse mountain field. fBm amplitude
/// decays geometrically, so a single ridged layer's mid-frequency octaves carry
/// almost no relief and ranges read "melted"/soft. This second band restores
/// visible relief in the ~0.5–3 km gap. It is **multifractal-coupled** — scaled
/// by the coarse ridged value so the detail sharpens crests and stays quiet in
/// the basins — and gated to montane land so plains/ocean stay smooth.
const MID_MOUNTAIN_WL_M: f64 = 5_000.0;
const MID_MOUNTAIN_AMP_M: f64 = 700.0;

// --- Macro landcover (moisture) --------------------------------------------
//
// The planet-scale moisture/landcover field (docs/world/terrain_macro.md Phase 1).
// This is the f64, unlimited-wavelength macro layer the 4 km-wrapped shader
// noise structurally cannot carry: it bakes into the tile albedo attachment's
// alpha channel and rides `SurfaceSample::moisture`, and the terrain shader /
// grass builders add only a fine (≤125 m) wrapped detail tier on top.
// Semantics match the shader's historical field: `[-1, 1]`, + wet, − dry.
// Gradient (Perlin) fBm, so there is no value-noise lattice weave.

/// Climatic wet/dry provinces — the continent-interior-vs-coast scale.
const MOIST_REGION_WL_M: f64 = 700_000.0;
const MOIST_REGION_W: f64 = 0.50;
/// Regional mosaic (broad forest belts / dry basins).
const MOIST_LOCAL_WL_M: f64 = 90_000.0;
const MOIST_LOCAL_W: f64 = 0.24;
/// Stand/valley patchiness — the finest macro tier; its LOD-aware octaves
/// cascade down to ~0.5 km, where the shader's wrapped fine tier takes over.
const MOIST_STAND_WL_M: f64 = 9_000.0;
const MOIST_STAND_W: f64 = 0.26;
const MOIST_STAND_MAX_OCT: f64 = 5.0;
/// Contrast gain so regions actually reach the forest/dry extremes of the
/// transfer curves (mirrors the shader's old `MOISTURE_CONTRAST` intent).
const MOIST_CONTRAST: f64 = 1.45;

/// Ecotone mosaic gate: the 90 km / 9 km mosaic tiers may flip the cover only
/// near climate *transitions*. Where the geographic trend (latitude belts +
/// continentality) has strongly committed — a desert-belt core, a rainforest
/// core, the polar desert — the landscape is coherent, and the mosaic fades
/// to `1 − MOIST_CORE_COHERENCE`. Without this the dry belt read as splotchy
/// green/tan camo from orbit: the belt's mean dryness sits mid-transfer, so
/// full-amplitude 9–90 km noise oscillated the palette across the thresholds
/// everywhere (TM-P3 orbital screenshot, 2026-07-20). The 700 km province
/// tier stays ungated — provinces are the structure, not the splotch.
const MOIST_CORE_COHERENCE: f64 = 0.65;
/// |latitude + continental| band over which the gate engages.
const MOIST_CORE_LO: f64 = 0.20;
const MOIST_CORE_HI: f64 = 0.50;

/// Broad value-tone mottle folded into the baked albedo (~±10 % value at
/// ~30 km), replacing the shader's wrapped ~1 km "region tone drift" tier.
const TONE_WL_M: f64 = 30_000.0;
const TONE_AMP: f64 = 0.10;

// --- Climate (latitude → cold lift / warmth) -------------------------------
//
// Phase 2 of docs/world/terrain_macro.md: a minimal insolation model expressed as a
// **cold lift** — how many metres the ecological altitude bands (lush belt,
// treeline, snowline) descend at a given latitude. Zero in the tropics (the
// fixed bands stay the authored look at the runway site, lat 7.6°), rising to
// past-the-snowline at the poles so ice caps emerge at sea level. `warmth`
// gates the hot-desert sand palette: dry ground is sand only where the climate
// is warm, cold steppe elsewhere.
//
// These are MIRRORED in `thalos::landcover` (landcover.wgsl) — the terrain
// shader and GPU grass evaluate the same curve from the same constants; keep
// them in lockstep.

/// Full band descent at the poles (m). Snow bands start at 3.1 km, so a full
/// lift buries the poles under the cap with margin.
const CLIMATE_COLD_LIFT_MAX_M: f64 = 3_600.0;
/// `sin(latitude)` where cooling begins (~27°) and the span it ramps over.
/// The power curve keeps mid-latitudes mild — Earth-ish treeline descent
/// against Thalos's high-sitting land (platform + interior gain ≈ 0.4–1 km):
/// ~850 m of lift at 50°, ~1.8 km at 60°, treeline under the lowlands ~66°+,
/// snowline at sea level ~75°+ — so temperate lands stay green taiga-like and
/// ice is a *polar* feature, not a mid-latitude one.
const CLIMATE_LAT_LO: f64 = 0.45;
const CLIMATE_LAT_SPAN: f64 = 0.55;
const CLIMATE_LAT_POW: f64 = 2.6;
/// Cold-lift band over which `warmth` fades 1 → 0 (hot tropics → cool
/// temperate). Dry ground reads as hot-desert sand only while warm.
const CLIMATE_WARM_LO_M: f64 = 500.0;
const CLIMATE_WARM_HI_M: f64 = 1_600.0;

/// Metres the ecological bands descend at `sin_lat = |sin(latitude)|`.
/// Mirrored as `climate_cold_lift` in landcover.wgsl.
pub fn climate_cold_lift_m(sin_lat: f64) -> f64 {
    let t = ((sin_lat.abs() - CLIMATE_LAT_LO) / CLIMATE_LAT_SPAN).clamp(0.0, 1.0);
    CLIMATE_COLD_LIFT_MAX_M * t.powf(CLIMATE_LAT_POW)
}

/// Hot-climate weight in `[0, 1]` from the cold lift (1 = tropics, 0 = cool).
/// Mirrored as `climate_warmth` in landcover.wgsl.
pub fn climate_warmth(cold_lift_m: f64) -> f64 {
    1.0 - smoothstep(CLIMATE_WARM_LO_M, CLIMATE_WARM_HI_M, cold_lift_m)
}

// --- Moisture geography (latitude belts + continentality) ------------------
//
// Structure layered under the noise tiers so moisture provinces sit where a
// planet would put them: a wet equatorial belt, the subtropical dry belt
// (deserts), a mid-latitude storm track, polar desert, and drier continent
// interiors than coasts (continentality, from the continentalness field).

/// Whole-planet wet bias: it's a 65 %-ocean world (lore: 35 % land), so the
/// default land cover is living grass/forest; the dry belts and interiors
/// carve *into* that. Without it the negative terms stack into
/// near-planet-wide steppe.
const MOIST_BIAS: f64 = 0.06;
const MOIST_EQUATOR_AMT: f64 = 0.25;
/// Subtropical dry belt. Strong enough that its *core* (with continentality
/// and the dry side of the regional noise) actually crosses the soil/sand
/// transfer thresholds (dryness ≳ 0.88) — at the old 0.40 the belt only
/// reached "slightly drier grass" and the planet had zero deserts (TM-P3,
/// measured with `just map`). Lush stays the planet-wide default; this belt
/// and continental interiors are where the lore's rust-red ground bares.
const MOIST_SUBTROPIC_AMT: f64 = 0.70;
const MOIST_MIDLAT_AMT: f64 = 0.22;
const MOIST_POLAR_AMT: f64 = 0.30;
/// Interior drying: full effect deep inside a continent (continentalness well
/// above the coast threshold [`CONTINENT_C0`]).
const MOIST_CONTINENTAL_AMT: f64 = 0.26;

/// Latitude term of the macro moisture, in `[-1, 1]`-ish (added pre-clamp).
fn latitude_moisture(sin_lat: f64) -> f64 {
    let s = sin_lat.abs();
    let equator = MOIST_EQUATOR_AMT * (1.0 - smoothstep(0.08, 0.35, s));
    let subtropic =
        -MOIST_SUBTROPIC_AMT * (smoothstep(0.28, 0.40, s) * (1.0 - smoothstep(0.55, 0.70, s)));
    let midlat = MOIST_MIDLAT_AMT * (smoothstep(0.60, 0.72, s) * (1.0 - smoothstep(0.86, 0.94, s)));
    let polar = -MOIST_POLAR_AMT * smoothstep(0.90, 0.98, s);
    MOIST_BIAS + equator + subtropic + midlat + polar
}

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
// The runway sits at a fixed body-fixed lat/lon (see `thalos_runtime::runway`).
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
// takeoff heading 30° (see `thalos_runtime::runway`). Add/retune sites freely —
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
/// Kept fairly short so the skyline reads as many summits + spurs (rich at the
/// medium-to-far distance the range is actually viewed from), not a few big
/// lumps.
const MASSIF_RIDGE_WL_M: f64 = 11_000.0;
const MASSIF_RIDGE_OCTAVES: f64 = 7.0;
/// Medium domain warp applied to the ridged spine so its summits fold into
/// curved ridgelines and side-spurs (geological) instead of radially smooth
/// blobs — the main lever for a non-simplistic medium-far skyline. Independent
/// of the planet-scale `WARP_*`.
const MASSIF_WARP_WL_M: f64 = 7_000.0;
const MASSIF_WARP_AMP_M: f64 = 1_800.0;
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
    + (MONTANE_MTN_AMP_M + MID_MOUNTAIN_AMP_M).max(MASSIF_PEAK_M)
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
/// Fingerprint of the terrain generator's *output*.
///
/// **Bump this whenever a change to [`ProceduralSurface`] (or anything it
/// samples) alters the heights/albedo/roughness it produces.** Consumers that
/// memoize generated tiles across runs — the ground-LOD disk tile cache
/// (`thalos_runtime::rendering::tile_cache`) — fold this into their cache key, so a
/// stale cache is addressed by a *different* key and simply never read again.
/// Forgetting to bump it means a cached run keeps rendering the old terrain
/// while the code says otherwise, which is a maddening thing to debug.
///
/// (Terrain generation is under active iteration; if you are mid-tuning and want
/// to sidestep this entirely, run with `THALOS_TILE_CACHE=0`.)
pub const GENERATOR_VERSION: u64 = 17;

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

    /// Geometric height (m above the reference radius), the montane orogeny
    /// weight, **and** the continentalness `c` at body-fixed unit direction
    /// `dir`, evaluated at `lod_m` metres per sample. Orogeny and `c` are
    /// computed in the height path anyway (uplift / mountain amplitude / the
    /// land-sea macro), so the albedo + moisture paths reuse them for free.
    fn height_and_orogeny(&self, dir: DVec3, lod_m: f32) -> (f64, f64, f64) {
        let dir = dir.normalize_or_zero();
        if dir == DVec3::ZERO {
            return (0.0, 0.0, 0.0);
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

        // Rolling hills, mostly on land — only a low floor at sea so the shelf
        // stays smooth and doesn't speckle into islets at the waterline.
        let hills_oct = octaves_for_lod(lod_m, HILLS_WL_M);
        let hills = fbm(pw / HILLS_WL_M, self.seed ^ 0x5151, hills_oct)
            * HILLS_AMP_M
            * (0.18 + 0.82 * land_mask);

        // Lowland / seabed swell: barely land-gated so neither plains nor the
        // abyssal floor go dead flat. fBm, so the LOD octave plan fades finer
        // ripples in toward the camera.
        let swell_oct = octaves_for_lod(lod_m, SWELL_WL_M);
        let swell = fbm(pw / SWELL_WL_M, self.seed ^ 0x57E1, swell_oct)
            * SWELL_AMP_M
            * (0.55 + 0.45 * land_mask);

        // Ridged mountains: a coarse range-defining band (amplitude blends
        // plains↔montane by orogeny, gated to land) plus a medium band that
        // rides on it. The medium band is multiplied by the coarse ridged value
        // (multifractal coupling: sharp on the ranges, quiet in the basins) and
        // gated to montane land, so ranges carry visible relief down to a few
        // hundred metres instead of melting into a single fBm's vanishing tail.
        let mtn_oct = octaves_for_lod(lod_m, MOUNTAIN_WL_M);
        let mtn_amp = lerp(orogeny, PLAINS_MTN_AMP_M, MONTANE_MTN_AMP_M) * land_mask;
        let mtn_base = ridged(pw / MOUNTAIN_WL_M, self.seed ^ 0x9A9A, mtn_oct);
        let mid_oct = octaves_for_lod(lod_m, MID_MOUNTAIN_WL_M);
        let mid = ridged(pw / MID_MOUNTAIN_WL_M, self.seed ^ 0x3D7B, mid_oct);
        let mountains =
            mtn_base * mtn_amp + mid * MID_MOUNTAIN_AMP_M * mtn_base * (orogeny * land_mask);

        // Seabed relief: rolling abyssal hills / seamounts, faded in by depth so
        // the deep floor isn't a flat plane while the shelf and coast stay clean.
        // (`smoothstep` needs increasing edges, so invert the shallow→deep ramp.)
        let deep = 1.0 - smoothstep(SEABED_FADE_LO_M, SEABED_FADE_HI_M, macro_h);
        let seabed_oct = octaves_for_lod(lod_m, SEABED_WL_M);
        let seabed = fbm(pw / SEABED_WL_M, self.seed ^ 0x5EAB, seabed_oct) * SEABED_AMP_M * deep;

        // Combine relief with the macro base: sea-level crossings belong to the
        // LOD-invariant macro field alone — sub-sea relief shoals to awash
        // reefs instead of breaching (see `combine_macro_and_relief`).
        let height = combine_macro_and_relief(macro_h, uplift + hills + swell + mountains + seabed);

        // Authored, erosion-sculpted mountain ranges near the runway. Additive
        // and footprint-gated (zero outside every envelope), so they don't
        // perturb the rest of the planet and aren't subject to the shelf cap.
        // `rock` skews the macro albedo greyer where a range stands.
        let (massif_m, rock) = self.mountain_massifs(p);

        (height + massif_m, orogeny.max(rock), c)
    }

    /// Continentalness in `~[-1, 1]` at body-local position `p` (m), warped at
    /// continent scale. This is the SEPARABLE macro field: a later
    /// plate-tectonics backend replaces the body of this one function (and
    /// [`Self::orogeny`]) to drive continents / mountain belts / island arcs from
    /// plate margins, without touching the hypsometric remap or relief cascade.
    fn continentalness(&self, p: DVec3) -> f64 {
        let c_base = self.continentalness_base(p);

        // ── Coast-character refinement (BL-6) ───────────────────────────────
        // The base field's coasts are smooth, gently wavy lines — no bays,
        // capes, rias, or archipelagos at the 20–300 km scale, and every coast
        // has the same character. Refine the field near its threshold with a
        // ruggedness-modulated fractal DOMAIN WARP: displacing where the base
        // is *evaluated* crenulates the shoreline into coves/headlands without
        // the closed-pocket lattice that additive threshold noise produces.
        // A low-frequency ruggedness field picks the regional character —
        // depositional coasts (low) stay sweeping beach arcs, erosional coasts
        // (high) get deep crenulation plus a sparse offshore islet sprinkle.
        //
        // Cost-gated to the coastal band: away from the threshold the refined
        // and base fields agree to well under the gate width, so the planet's
        // interior and deep ocean never pay the second base evaluation.
        // Everything here is fixed-octave → LOD-invariant, so the shoreline
        // stays exclusively macro-owned (the INC-0003 invariant).
        let t = ((c_base - CONTINENT_C0).abs() / COAST_REFINE_BAND_C).min(1.0);
        if t >= 1.0 {
            return c_base;
        }

        // Regional coast character in [0, 1]: 0 = depositional/smooth,
        // 1 = erosional/rugged. Remapped hard so both characters actually
        // occur as regions rather than everything sitting mid-scale.
        let rug_n = fbm(p / COAST_RUGGED_WL_M, self.seed ^ 0x5EA5, 2.0);
        let rug = smoothstep(-0.35, 0.45, rug_n);

        // Fractal crenulation warp. Amplitude in metres of coastline
        // displacement; wavelengths from COAST_CREN_WL_M down ~4 octaves.
        let amp_m =
            COAST_CREN_AMP_SMOOTH_M + (COAST_CREN_AMP_RUGGED_M - COAST_CREN_AMP_SMOOTH_M) * rug;
        let cq = p / COAST_CREN_WL_M;
        let cren_warp = DVec3::new(
            fbm(cq, self.seed ^ 0xC4E1, COAST_CREN_OCT),
            fbm(cq + DVec3::splat(41.3), self.seed ^ 0xC4E2, COAST_CREN_OCT),
            fbm(cq - DVec3::splat(27.8), self.seed ^ 0xC4E3, COAST_CREN_OCT),
        ) * amp_m;
        let c_warped = self.continentalness_base(p + cren_warp);

        // Sparse offshore islets/skerries, rugged coasts only. This term IS
        // allowed to cross the threshold (macro islands are legitimate — the
        // forbidden thing is LOD-faded relief crossing it), and its ruggedness
        // gate keeps it a regional trait instead of a planet-wide lattice.
        // Positive-thresholded: land is added only in distinct clumps (BL-10 —
        // no fringe dust, no carved semi-submerged channels).
        let isle_n = fbm(p / COAST_ISLE_WL_M, self.seed ^ 0x151E, 3.0);
        let isle = (isle_n - COAST_ISLE_BIAS).max(0.0) * COAST_ISLE_GAIN_C * rug;

        // Ease the refinement in across the gate so the band edge is seamless.
        // (The local `smoothstep` clamps `edge1 − edge0` positive, so reversed
        // edges silently return 0 — invert explicitly instead.)
        let w = 1.0 - smoothstep(0.55, 1.0, t);
        c_base + (c_warped + isle - c_base) * w
    }

    /// The unrefined continentalness: plates + continent-scale warp + organic
    /// detail. [`Self::continentalness`] evaluates this once everywhere and a
    /// second time (domain-warped) inside the coastal refinement band.
    fn continentalness_base(&self, p: DVec3) -> f64 {
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
        // Medium domain warp: fold the ridged spine into curved ridgelines /
        // spurs instead of a radially smooth swell. Warps the (x, y) sample
        // position before the ridged slice; `y` (along-crest) stays un-warped for
        // the tall/short crest modulation below.
        let w = DVec3::new(x, y, 0.0) / MASSIF_WARP_WL_M;
        let wx = x + fbm(w, self.seed ^ 0x7A11 ^ site.salt, 3.0) * MASSIF_WARP_AMP_M;
        let wy = y + fbm(w + DVec3::splat(13.7), self.seed ^ 0x7A22 ^ site.salt, 3.0)
            * MASSIF_WARP_AMP_M;

        // 2-D ridged slice (z = 0): deterministic, multi-ridge spine.
        let ridge = ridged(
            DVec3::new(wx, wy, 0.0) / MASSIF_RIDGE_WL_M,
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

    /// Planet-scale macro landcover moisture in `[-1, 1]` (+ wet, − dry) at
    /// body-local position `p` (m), evaluated at `lod_m` metres per sample
    /// (`<= 0` requests full detail). Geographic structure (latitude belts from
    /// `sin_lat`, continentality from the continentalness `c` — interiors
    /// drier than coasts) layered with three decorrelated gradient-fBm tiers —
    /// climatic provinces, regional mosaic, stand/valley patchiness — with only
    /// the finest tier LOD-aware. This is the f64 macro layer the 4 km-wrapped
    /// shader noise cannot carry (docs/world/terrain_macro.md); the shader and the
    /// grass builders add a fine (≤125 m) wrapped detail tier on top.
    pub fn macro_moisture(&self, p: DVec3, lod_m: f32, c: f64, sin_lat: f64) -> f64 {
        let region = fbm(p / MOIST_REGION_WL_M, self.seed ^ 0x4C43_0001, 3.0);
        let local = fbm(p / MOIST_LOCAL_WL_M, self.seed ^ 0x4C43_0002, 3.0);
        let stand_oct = octaves_for_lod(lod_m, MOIST_STAND_WL_M).min(MOIST_STAND_MAX_OCT);
        let stand = fbm(p / MOIST_STAND_WL_M, self.seed ^ 0x4C43_0003, stand_oct);
        // Interior gate sized to the actual continentalness range: plate
        // interiors sit at c ≈ 0.7–1.0 (`PLATE_WEIGHT · 1 ± detail`), so the
        // old (0.45, 1.15) gate never applied more than ~60 % of the drying.
        let continental = -MOIST_CONTINENTAL_AMT * smoothstep(0.30, 0.90, c);
        let geo = latitude_moisture(sin_lat) + continental;
        // Ecotone gate (see MOIST_CORE_*): mosaic tiers fade where the
        // geographic trend has committed, so belt cores are coherent and the
        // patchwork lives at the climate transitions. `geo` is measured
        // relative to the neutral bias, not zero.
        let mosaic = 1.0
            - MOIST_CORE_COHERENCE
                * smoothstep(MOIST_CORE_LO, MOIST_CORE_HI, (geo - MOIST_BIAS).abs());
        let noise = (region * MOIST_REGION_W
            + (local * MOIST_LOCAL_W + stand * MOIST_STAND_W) * mosaic)
            * MOIST_CONTRAST;
        (noise + geo).clamp(-1.0, 1.0)
    }

    /// Broad value-tone mottle in `[-1, 1]` for the baked albedo (~30 km).
    fn macro_tone(&self, p: DVec3) -> f64 {
        fbm(p / TONE_WL_M, self.seed ^ 0x70E4_0001, 3.0)
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

    /// Provisional macro albedo (linear RGB) by **climate-shifted** altitude
    /// band, laterally varied by the macro landcover moisture (lush green ↔
    /// dry tan lowlands, dark forest on the wettest ground, hot-desert sand on
    /// the driest *warm* ground) and a broad value-tone mottle, so the
    /// distant-body impostor and the ground shader's macro tint show the same
    /// regions the moisture + climate fields paint. `cold_lift_m` descends the
    /// upland/rock/snow bands with latitude (polar caps at sea level); `warmth`
    /// gates sand vs cold steppe. Still a stand-in for the later procedural
    /// material model.
    fn albedo_from_bands(&self, t: &MacroBandTs, tone: f64) -> Vec3 {
        // Linear-RGB band anchors.
        let shore = Vec3::new(0.30, 0.27, 0.18); // tan/sand near 0 m
        let lowland_lush = Vec3::new(0.08, 0.16, 0.06); // green
        let lowland_dry = Vec3::new(0.135, 0.140, 0.070); // dry-grass tan
        let laterite = Vec3::new(0.112, 0.074, 0.042); // rust-red bare soil (= landcover C_SOIL)
        let sand = Vec3::new(0.225, 0.190, 0.125); // hot-desert sand
        let forest = Vec3::new(0.040, 0.095, 0.032); // dark canopy green
        let tundra = Vec3::new(0.088, 0.096, 0.078); // cold muted moss/lichen
        let upland = Vec3::new(0.092, 0.104, 0.082); // grey-green upland (temperate, not brown)
        let rock = Vec3::new(0.118, 0.120, 0.122); // neutral grey rock
        let snow = Vec3::new(0.62, 0.64, 0.68); // snow

        // Lowland palette by regional dryness; the wettest regions skew toward
        // closed-canopy forest (darker from orbit), the driest ground bares the
        // lore's iron-rich lateritic soil (docs/lore/solar_system.md §II:
        // "rust-red ground showing through where forest cover thins" — the
        // same C_SOIL step the ground shader's `vegetation_color` paints),
        // going hot-desert sand only in *warm* climates — cold dry ground
        // stays tan steppe / bare soil. Cold climates mute the living cover
        // toward tundra well before the (shifted) rock/snow bands take over.
        let mut lowland = mix(lowland_lush, lowland_dry, t.dry as f32);
        lowland = mix(lowland, laterite, t.soil as f32);
        lowland = mix(lowland, sand, t.sand as f32);
        lowland = mix(lowland, forest, t.forest as f32);

        // Ecological altitude: the bands descend with latitude, so high
        // latitudes go rock/snow at ever-lower ground (polar caps at 0 m).
        // Tundra sits BETWEEN upland and rock: it is the cold-climate cover
        // and must be able to claim the upland grey at high latitude (inside
        // the lowland chain the eco-shifted upland band crushed it and the
        // planet had zero tundra), while the scree/snow of the true alpine
        // zone still override it.
        let mut c = mix(shore, lowland, t.lowland as f32);
        c = mix(c, upland, t.upland as f32);
        c = mix(c, tundra, t.tundra as f32);
        c = mix(c, rock, t.rock as f32);
        c = mix(c, snow, t.snow as f32);
        // Montane macro tone skews greyer/rockier even at moderate altitude;
        // the shader's slope/altitude bands add the real scree + snow on top.
        c = mix(c, rock, t.oro_rock as f32);
        // Broad value-tone mottle (~30 km) — the baked home of what used to be
        // the shader's wrapped ~1 km tone-drift tier.
        c * (1.0 + (tone * TONE_AMP) as f32)
    }

    /// One-evaluation variant of [`SurfaceQuery::sample_d`] that also returns
    /// the dominant [`MacroBiome`] class. Both the albedo mix chain and the
    /// classification read the same [`MacroBandTs`] evaluation, so the class
    /// map can never drift from what the palette actually renders. Consumed by
    /// the `world_map` biome export; classification costs a handful of
    /// multiplies on top of the sample.
    pub fn sample_biome_d(&self, dir: DVec3, lod_m: f32) -> (SurfaceSample, MacroBiome) {
        let dir = dir.normalize_or_zero();
        let (height_m, orogeny, c) = self.height_and_orogeny(dir, lod_m);
        let p = dir * self.radius_m;
        let sin_lat = dir.y.abs();
        let moisture = self.macro_moisture(p, lod_m, c, sin_lat);
        let tone = self.macro_tone(p);
        let cold_lift = climate_cold_lift_m(sin_lat);
        let warmth = climate_warmth(cold_lift);
        let bands = macro_band_ts(height_m, orogeny, moisture, cold_lift, warmth);
        let sample = SurfaceSample {
            height_m: height_m as f32,
            albedo_linear: self.albedo_from_bands(&bands, tone),
            roughness: 0.92,
            moisture: moisture as f32,
        };
        (sample, classify_macro(&bands, height_m))
    }
}

/// Discrete dominant-class view of the macro landcover — the argmax of the
/// same nested band weights the macro albedo blends continuously (see
/// [`macro_band_ts`]). This is the read-only stats/iteration view consumed by
/// the `world_map` biome export, and the seam the biome-expansion work
/// (docs/world/terrain_macro.md Phase 2 remainder) grows explicit classes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroBiome {
    Ocean,
    Beach,
    Grassland,
    Steppe,
    Desert,
    Forest,
    Tundra,
    Upland,
    Rock,
    Snow,
}

impl MacroBiome {
    pub const ALL: [MacroBiome; 10] = [
        MacroBiome::Ocean,
        MacroBiome::Beach,
        MacroBiome::Grassland,
        MacroBiome::Steppe,
        MacroBiome::Desert,
        MacroBiome::Forest,
        MacroBiome::Tundra,
        MacroBiome::Upland,
        MacroBiome::Rock,
        MacroBiome::Snow,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MacroBiome::Ocean => "ocean",
            MacroBiome::Beach => "beach",
            MacroBiome::Grassland => "grassland",
            MacroBiome::Steppe => "steppe",
            MacroBiome::Desert => "desert",
            MacroBiome::Forest => "forest",
            MacroBiome::Tundra => "tundra",
            MacroBiome::Upland => "upland",
            MacroBiome::Rock => "rock",
            MacroBiome::Snow => "snow",
        }
    }
}

/// The blend factors of the macro-albedo mix chains, computed once and shared
/// by [`ProceduralSurface::albedo_from_bands`] (the continuous palette) and
/// [`classify_macro`] (the discrete class) so the two views cannot drift.
struct MacroBandTs {
    // Lowland palette chain (regional dryness / climate).
    dry: f64,
    soil: f64,
    sand: f64,
    forest: f64,
    tundra: f64,
    // Altitude band chain (`lowland` keys off raw height — the beach strip
    // stays at the physical coast; the rest off climate-shifted eco altitude).
    lowland: f64,
    upland: f64,
    rock: f64,
    snow: f64,
    oro_rock: f64,
}

/// The altitude thresholds MIRROR the ground shader's ecological bands
/// (`landcover.wgsl` `LUSH_LO/HI` = 1500/2400, `TREELINE_LO/HI` = 2400/3000,
/// snow above ~3.1 km) so the orbital palette and the ground agree — the
/// one-world rule. The pre-TM-P3 bands (upland from **120 m**!) painted
/// nearly all land grey from orbit while the ground below rendered lush, and
/// crushed the climate palette (59 % of land classified upland, deserts /
/// steppe / tundra ≈ 0 %). Snow saturates at 3600 m = `CLIMATE_COLD_LIFT_MAX_M`
/// so the poles cap fully. The dryness thresholds mirror the ground's
/// `vegetation_color` transfer (dry-grass 0.55–0.88, soil 0.88–0.98, sand
/// 0.80–0.95 × warmth, forest below 0.42).
fn macro_band_ts(
    height_m: f64,
    orogeny: f64,
    moisture: f64,
    cold_lift_m: f64,
    warmth: f64,
) -> MacroBandTs {
    let dryness = (0.5 - 0.5 * moisture).clamp(0.0, 1.0);
    let h = height_m + cold_lift_m;
    MacroBandTs {
        dry: smoothstep(0.55, 0.88, dryness),
        soil: smoothstep(0.88, 0.98, dryness),
        sand: smoothstep(0.80, 0.95, dryness) * warmth,
        forest: 1.0 - smoothstep(0.28, 0.58, dryness),
        tundra: smoothstep(1_400.0, 2_600.0, cold_lift_m),
        // The tan shore band hugs the physical beach (the +4 m berm face —
        // BL-10), not the first 60 m of elevation: sand is a coastline
        // feature, and the old wide band washed whole coastal plains tan.
        lowland: smoothstep(2.0, 9.0, height_m),
        upland: smoothstep(1_500.0, 2_400.0, h),
        rock: smoothstep(2_400.0, 3_000.0, h),
        snow: smoothstep(3_000.0, 3_600.0, h),
        oro_rock: 0.35 * orogeny,
    }
}

/// Expand the nested `mix` chains into absolute per-class weights (each mix
/// `t` scales everything blended before it by `1 − t`) and return the argmax.
fn classify_macro(t: &MacroBandTs, height_m: f64) -> MacroBiome {
    if height_m < 0.0 {
        return MacroBiome::Ocean;
    }
    // Lowland palette chain: lush → dry → soil → sand → forest.
    // Bare laterite soil and hot sand both classify as Desert (hot desert =
    // sand, cold/temperate barrens = soil).
    let no_forest = 1.0 - t.forest;
    let low_lush = (1.0 - t.dry) * (1.0 - t.soil) * (1.0 - t.sand) * no_forest;
    let low_dry = t.dry * (1.0 - t.soil) * (1.0 - t.sand) * no_forest;
    let low_desert = (t.soil * (1.0 - t.sand) + t.sand) * no_forest;
    let low_forest = t.forest;

    // Altitude band chain: shore → lowland → upland → tundra → rock → snow →
    // oro-rock (tundra between upland and rock — see `albedo_from_bands`).
    let tail = (1.0 - t.rock) * (1.0 - t.snow) * (1.0 - t.oro_rock);
    let after_low = (1.0 - t.upland) * (1.0 - t.tundra) * tail;
    let w_beach = (1.0 - t.lowland) * after_low;
    let w_low = t.lowland * after_low;
    let w_upland = t.upland * (1.0 - t.tundra) * tail;
    let w_tundra = t.tundra * tail;
    let w_rock = t.rock * (1.0 - t.snow) * (1.0 - t.oro_rock) + t.oro_rock;
    let w_snow = t.snow * (1.0 - t.oro_rock);

    let weights = [
        (MacroBiome::Beach, w_beach),
        (MacroBiome::Grassland, w_low * low_lush),
        (MacroBiome::Steppe, w_low * low_dry),
        (MacroBiome::Desert, w_low * low_desert),
        (MacroBiome::Forest, w_low * low_forest),
        (MacroBiome::Tundra, w_tundra),
        (MacroBiome::Upland, w_upland),
        (MacroBiome::Rock, w_rock),
        (MacroBiome::Snow, w_snow),
    ];
    weights
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("non-empty")
        .0
}

impl SurfaceQuery for ProceduralSurface {
    fn sample(&self, dir: Vec3, lod_m: f32) -> SurfaceSample {
        self.sample_d(dir.as_dvec3(), lod_m)
    }

    fn sample_d(&self, dir: DVec3, lod_m: f32) -> SurfaceSample {
        self.sample_biome_d(dir, lod_m).0
    }

    fn sample_height_m(&self, dir: Vec3, lod_m: f32) -> f32 {
        self.height_and_orogeny(dir.as_dvec3(), lod_m).0 as f32
    }

    fn landcover_moisture(&self, dir: DVec3) -> f32 {
        let dir = dir.normalize_or_zero();
        if dir == DVec3::ZERO {
            return 0.0;
        }
        let p = dir * self.radius_m;
        // Same continentalness the height path uses (incl. the runway-siting
        // bias), so point queries agree with the baked field.
        let c = self.continentalness(p) + self.runway_land_bias(dir);
        self.macro_moisture(p, 0.0, c, dir.y.abs()) as f32
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
        // linear interior gain so continent interiors ride higher than coasts,
        // plus the beach berm lifting the strand promptly off the waterline
        // (the land-side mirror of the foreshore drop — BL-10).
        LAND_PLATFORM_M * (1.0 - (-t / LAND_K).exp())
            + LAND_INTERIOR_GAIN_M * t
            + BEACH_RISE_M * (1.0 - (-t / BEACH_RISE_WIDTH_C).exp())
    } else {
        let s = CONTINENT_C0 - c;
        // Shallow shelf shoulder, then the continental slope dropping to a flat
        // abyssal plain (saturating → constant depth far out).
        //
        // The shoulder is a quadratic EASE-OUT (`x(2 − x)`: full slope at the
        // waterline, flat at the shelf edge), NOT a smoothstep. smoothstep's
        // zero-derivative start left the first third of the shelf hugging sea
        // level — tens of kilometres of seabed only ~1–3 m deep, inside the
        // renderer's f32 depth-error floor at orbital distances, which z-fought
        // the analytic sea sphere into dotted land-through-water speckle
        // (INC-0003). The linear start clears the error floor within roughly a
        // coarse-LOD texel of the shore while keeping the same shelf depth and
        // width overall.
        let x = (s / SHELF_WIDTH_C).min(1.0);
        let shelf = SHELF_DEPTH_M * x * (2.0 - x);
        // Immediate foreshore drop off the waterline (beach shelf lip): a few
        // metres of depth within the first ~km of shore, so the water there
        // reads as water instead of an ankle-deep translucent apron. ADR-20260720T185957Z-coastline-as-authored-data.
        let foreshore = FORESHORE_DEPTH_M * (1.0 - (-s / FORESHORE_WIDTH_C).exp());
        let abyss = (ABYSS_DEPTH_M - SHELF_DEPTH_M)
            * smoothstep(SHELF_WIDTH_C, SHELF_WIDTH_C + SLOPE_WIDTH_C, s);
        -(shelf + foreshore + abyss)
    }
}

/// Add relief to the macro base, keeping the **shoreline a property of the
/// LOD-invariant macro field** and the **ocean a single connected body**.
///
/// Two rules, both about the sea-level crossing:
///
/// 1. **Coastal fade.** The relief cascade ([`ProceduralSurface::
///    height_and_orogeny`]) is LOD-aware (its octaves fade with camera distance),
///    so if it defined the waterline the coast would move as you fly in — badly,
///    because a gentle coast turns a few metres of LOD height wobble into
///    kilometres of horizontal shift. So relief is faded to zero as the macro
///    surface approaches the waterline (over [`COAST_BAND_M`]): near the shore
///    `h → macro_h`, whose zero crossing is LOD-invariant. Relief resumes at full
///    strength inland/offshore, so coastal hills and the seabed keep their texture.
///
/// 2. **No sea on land, no land in the sea.** Macro **land** (`macro_h >= 0`) is
///    floored at the waterline: relief may not carve an isolated basin below sea
///    level, so the ocean stays the single connected body the macro field defines
///    (closed land basins are future inland lakes, not sea — deferred). Macro
///    **seabed** never breaches at all: relief that would cross the surface
///    shoals to an awash reef saturating at [`AWASH_REEF_DEPTH_M`] below sea
///    level. The 0-crossing is *exclusively* the macro field's — LOD-aware
///    relief defining any waterline (mainland or islet) makes that waterline
///    move with camera distance (INC-0003).
fn combine_macro_and_relief(macro_h: f64, relief: f64) -> f64 {
    let coast_fade = smoothstep(0.0, COAST_BAND_M, macro_h.abs());
    let h = macro_h + relief * coast_fade;
    if macro_h >= 0.0 {
        return h.max(0.0);
    }
    // Would-be breach: fold it into an awash shoal. Continuous at h = 0
    // (→ 0⁻) and monotone in |breach|, saturating just below the surface.
    let mut hs = if h <= 0.0 {
        h
    } else {
        -AWASH_REEF_DEPTH_M * (1.0 - (-h / AWASH_REEF_DEPTH_M).exp())
    };
    // Offshore shallow clearance (BL-10): where the MACRO seabed is deep,
    // relief may not park the combined seabed inside the shallow-tint band —
    // compress (−CLEAR, 0) onto its bottom sliver so near-surface tops (the
    // island "shoal halo" speckle) drop out of the pale band. `w` feathers
    // the rule in over macro depth CLEAR → 2×CLEAR, so the true foreshore
    // (macro itself shallow) is untouched and there is no spatial seam.
    // (The local `smoothstep` clamps reversed edges to 0 — key on the
    // positive macro DEPTH so the edges increase.)
    if hs > -OFFSHORE_SHALLOW_CLEAR_M {
        let w = smoothstep(
            OFFSHORE_SHALLOW_CLEAR_M,
            2.0 * OFFSHORE_SHALLOW_CLEAR_M,
            -macro_h,
        );
        let compressed =
            -OFFSHORE_SHALLOW_CLEAR_M + (hs + OFFSHORE_SHALLOW_CLEAR_M) * OFFSHORE_SHALLOW_KEEP;
        hs = hs + (compressed - hs) * w;
    }
    hs
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

/// WGSL-parity smoothstep, **including descending edges** (`edge0 > edge1`
/// inverts the ramp). Do NOT "guard" the denominator with `.max(EPSILON)`:
/// that turns every descending-edge call into an inverted hard step at
/// `edge0` — the INC-0005 forest-on-desert bug, where
/// `smoothstep(0.42, 0.18, dryness)` returned 1.0 on all ground drier than
/// 0.42 and the macro albedo painted forest onto the dry belts for months.
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let denom = edge1 - edge0;
    if denom.abs() < f64::EPSILON {
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / denom).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Mountain-massif helpers
// ---------------------------------------------------------------------------

/// Body-fixed unit direction for a latitude/longitude (degrees). Matches the
/// convention in `thalos_runtime::runway::latlon_dir`.
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
/// defaults target a ~22-unit toy mesh).
///
/// Two independent knobs to keep straight (see `bevy_erosion_filter::cpu`):
/// - **Feature scale.** The largest carved gully wavelength ≈ `scale *
///   cell_scale` (`cell_scale` defaults to 0.7). `octaves` extend the drainage
///   `octaves-1` halvings finer (each ×2 finer, ×`gain` = 0.5 shallower), so the
///   finest detail ≈ `scale * cell_scale / 2^(octaves-1)`.
/// - **Carve depth ∝ `scale * strength`**, summed over the octaves. So to make
///   the gullies *finer* without making them *shallower*, drop `scale` and raise
///   `strength` to hold that product.
///
/// Here: dominant gully ≈ 4000 × 0.7 ≈ 2.8 km, finest ≈ 90 m over **6** octaves
/// — deliberately stopping short of the sub-100 m detail that only reads in an
/// extreme close-up (which the player never gets), keeping the carve energy in
/// the medium band that's visible at flying/approach distance. Depth ∝ 4000 ×
/// 0.23 ≈ 920 (≈ unchanged). `onset` is eased down so the gentle base flanks
/// still carve.
fn massif_erosion_params() -> ErosionFilterParams {
    let d = ErosionFilterParams::default();
    ErosionFilterParams {
        scale: 4_000.0,
        strength: 0.23,
        gully_weight: 0.55,
        detail: 1.7,
        octaves: 6,
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
