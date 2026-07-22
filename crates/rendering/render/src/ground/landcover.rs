//! Shared large-scale **landcover field** — the moisture / palette / coverage
//! field the grass samples so it AGREES with the terrain ground.
//!
//! This is a faithful CPU mirror of the moisture + vegetation-colour model in
//! `body_terrain.wgsl` (`fbm3_periodic` → `moisture` → the `veg` colour inside
//! `eval_material_stack`). Grass blades and the terrain are then coloured from
//! the SAME field, which buys three things the grass was missing:
//!
//! 1. **Dynamic coloration** — grass fields pick up the terrain's large-scale
//!    palette variation (lush green ↔ dry tan ↔ forest), not one flat green.
//! 2. **Coverage variation** — grass density can track the same field (thinning
//!    on dry/bare patches), like the trees track `forest_coverage`.
//! 3. **Cheap distant coverage** — because blade colour is computed identically
//!    to the terrain albedo, the clipmap handoff (blades fade out, the ground's
//!    own grass colour carries the far field) is seamless: no colour seam where
//!    the blades stop.
//!
//! The field is a pure analytic function of the **body-fixed position** (metres)
//! plus altitude — the same `(direction, height)` the grass clump already has.
//! The noise is periodic at [`DETAIL_COORD_PERIOD_M`]; the shader keeps its
//! sample coordinate near the origin via a per-period phase offset, so this
//! mirror reduces the body position modulo that period before going to `f32`,
//! and because every `DETAIL_COORD_PERIOD_M × scale` is an integer the wrapped
//! lattice cells match the shader's exactly.
//!
//! **Keep the constants below in sync with `body_terrain.wgsl`.** They are a
//! deliberate mirror; if the terrain palette/scales move, these must move too or
//! grass and ground will disagree. (The proper long-term home is the
//! `ProceduralSurface` seam providing the field once, consumed by both — see
//! `docs/world/vegetation.md`.)

use bevy::math::{DVec3, Vec3};

// === Mirror of landcover.wgsl — keep in sync ===============================
// Fine tiers only: the MACRO moisture (≥ ~500 m) is the f64
// `ProceduralSurface::macro_moisture` field (docs/world/terrain_macro.md), passed in
// by the caller (grass builders read it via `HeightSource::landcover_moisture`;
// the terrain shader decodes it from the albedo attachment's alpha).
/// Noise wrap period (m). Every `* SCALE` below is integer, so the wrapped
/// lattice matches the shader regardless of the phase reduction.
const DETAIL_COORD_PERIOD_M: f64 = 4000.0;
const MOISTURE_SCALE: f32 = 0.008; // ~125 m medium patches
const MOISTURE_DETAIL_AMT: f32 = 0.30; // signed fine-deviation amplitude
const MACRO_VAR_SCALE: f32 = 0.004; // ~250 m mottle
const MACRO_VAR_FINE_AMT: f32 = 0.6; // post-slim fine-tier amplitude
const MACRO_VAR_AMT: f32 = 0.14;
const SNOW_LINE_NOISE_M: f32 = 400.0;

const LUSH_LO_M: f32 = 1500.0;
/// Forest gone above here (top of the lush band). `pub(crate)` so the tree
/// scatter's biome gate keys its treeline off the SAME constant the ground
/// palette uses — one definition of where woody cover stops.
pub(crate) const LUSH_HI_M: f32 = 2400.0;
const TREELINE_LO_M: f32 = 2400.0;
/// Alpine/scree fully taken over above here. `pub(crate)` for the tree scatter
/// gate (see [`LUSH_HI_M`]).
pub(crate) const TREELINE_HI_M: f32 = 3000.0;

const C_FOREST: Vec3 = Vec3::new(0.034, 0.084, 0.028);
const C_GRASS: Vec3 = Vec3::new(0.072, 0.152, 0.050);
const C_DRYGRASS: Vec3 = Vec3::new(0.138, 0.150, 0.074);
const C_SOIL: Vec3 = Vec3::new(0.112, 0.074, 0.042);
const C_SAND: Vec3 = Vec3::new(0.225, 0.190, 0.125);
const C_ALPINE: Vec3 = Vec3::new(0.082, 0.094, 0.074);
// ===========================================================================

/// A landcover sample at one surface point: the vegetation colour the terrain
/// paints there (the grass tint), a `[0, 1]` coverage multiplier, and the raw
/// moisture in `[-1, 1]` (`+` wet, `−` dry) for callers that want to gate on it.
#[derive(Clone, Copy, Debug)]
pub struct LandcoverSample {
    /// Linear-space vegetation colour — what the terrain ground reads as here.
    /// Use it as the grass blade base tint so blades match the ground.
    pub veg_color: Vec3,
    /// Grass coverage / density multiplier in `[0, 1]` (thins on dry → bare-soil
    /// patches, matching where the terrain paints soil).
    pub coverage: f32,
    /// Raw moisture in `[-1, 1]`.
    pub moisture: f32,
}

/// Sample the landcover field at a body-fixed surface position (metres from the
/// body centre) and its altitude above the reference radius (metres).
///
/// `macro_moisture` is the planet-scale macro field in `[-1, 1]` at this point
/// (`HeightSource::landcover_moisture` / `SurfaceQuery::landcover_moisture`);
/// this function adds only the wrapped fine detail tier the terrain shader
/// adds, so blade and ground stay the same material. `sin_lat` is
/// `|body_dir.y|` — the climate input that descends the ecological bands with
/// latitude and gates the hot-desert sand palette (the exact
/// `thalos_terrain::climate_*` curve the terrain shader mirrors). Pass
/// `(0.0, 0.0)` for standalone (preview) consumers with no macro field.
pub fn sample_landcover(
    body_pos_m: DVec3,
    altitude_m: f32,
    macro_moisture: f32,
    sin_lat: f32,
) -> LandcoverSample {
    // Reduce modulo the wrap period in f64 (keeps the f32 noise precise and
    // matches the shader's near-origin sample coordinate).
    let reduced = body_pos_m - (body_pos_m / DETAIL_COORD_PERIOD_M).round() * DETAIL_COORD_PERIOD_M;
    let p = reduced.as_vec3();

    let cold_lift = thalos_terrain::climate_cold_lift_m(sin_lat as f64) as f32;
    let warmth = thalos_terrain::climate_warmth(cold_lift as f64) as f32;
    let moisture = (macro_moisture + moisture_detail_at(p)).clamp(-1.0, 1.0);
    let macro_var = macro_var_at(p);
    LandcoverSample {
        veg_color: vegetation_color(altitude_m + cold_lift, moisture, macro_var, warmth),
        coverage: grass_coverage(moisture),
        moisture,
    }
}

/// Vegetation colour the terrain paints at `(altitude, moisture)` — the exact
/// `veg` branch of `eval_material_stack` (mirrored from the shared
/// `landcover.wgsl::vegetation_color`), times the macro mottle the terrain
/// applies to the whole ground. This is the grass blade tint, so blade == ground.
fn vegetation_color(eco_altitude_m: f32, moisture: f32, macro_var: f32, warmth: f32) -> Vec3 {
    let jitter = macro_var * SNOW_LINE_NOISE_M;
    let lush = smoothstep(LUSH_HI_M, LUSH_LO_M, eco_altitude_m + jitter); // 1 low, 0 high
    let alpine = smoothstep(TREELINE_LO_M, TREELINE_HI_M, eco_altitude_m + jitter);
    let dryness = (0.5 - 0.5 * moisture).clamp(0.0, 1.0); // + wet → 0, − dry → 1
    let forest_amt = smoothstep(0.58, 0.28, dryness) * lush;

    let mut grass_c = C_GRASS.lerp(C_DRYGRASS, smoothstep(0.55, 0.88, dryness));
    grass_c = grass_c.lerp(C_SOIL, smoothstep(0.88, 0.98, dryness));
    grass_c = grass_c.lerp(C_SAND, smoothstep(0.80, 0.95, dryness) * warmth);
    let mut veg = grass_c.lerp(C_FOREST, forest_amt);
    veg = veg.lerp(C_ALPINE, alpine);

    // Low-frequency value mottle (terrain: `ground *= 1 + variation·MACRO_VAR_AMT`).
    veg * (1.0 + macro_var * MACRO_VAR_AMT)
}

/// Grass coverage `[0, 1]` from the moisture field: full on lush/temperate
/// ground, thinning to ~0 on the driest patches (where the terrain switches to
/// bare soil), so the grass carpet breaks up into the same patches the ground
/// shows. Forest thinning is handled separately by the driver's `forest_cull`.
fn grass_coverage(moisture: f32) -> f32 {
    let dryness = (0.5 - 0.5 * moisture).clamp(0.0, 1.0);
    (1.0 - smoothstep(0.70, 0.94, dryness)).clamp(0.0, 1.0)
}

// --- Moisture / macro-variation fields (mirror of the WGSL) ----------------

fn moisture_detail_at(p: Vec3) -> f32 {
    let lc_med = fbm3_periodic(
        p * MOISTURE_SCALE,
        3,
        DETAIL_COORD_PERIOD_M as f32 * MOISTURE_SCALE,
    );
    (lc_med - 0.5) * 2.0 * MOISTURE_DETAIL_AMT
}

fn macro_var_at(p: Vec3) -> f32 {
    let macro_fine = (fbm3_periodic(
        p * MACRO_VAR_SCALE,
        3,
        DETAIL_COORD_PERIOD_M as f32 * MACRO_VAR_SCALE,
    ) - 0.5)
        * 2.0;
    (macro_fine * MACRO_VAR_FINE_AMT).clamp(-1.0, 1.0)
}

// --- Value-noise fbm (exact port of body_terrain.wgsl) ---------------------

fn fbm3_periodic(mut p: Vec3, octaves: i32, mut period: f32) -> f32 {
    let mut amp = 0.5;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for _ in 0..octaves {
        sum += amp * value_noise_3d_periodic(p, period);
        norm += amp;
        p *= 2.0;
        period *= 2.0;
        amp *= 0.5;
    }
    sum / norm.max(1.0e-5)
}

fn value_noise_3d_periodic(x: Vec3, period: f32) -> f32 {
    let i = x.floor();
    let f = x - i;
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let c = |dx: f32, dy: f32, dz: f32| hash13(wrap_lattice(i + Vec3::new(dx, dy, dz), period));
    let n000 = c(0.0, 0.0, 0.0);
    let n100 = c(1.0, 0.0, 0.0);
    let n010 = c(0.0, 1.0, 0.0);
    let n110 = c(1.0, 1.0, 0.0);
    let n001 = c(0.0, 0.0, 1.0);
    let n101 = c(1.0, 0.0, 1.0);
    let n011 = c(0.0, 1.0, 1.0);
    let n111 = c(1.0, 1.0, 1.0);
    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);
    mix(nxy0, nxy1, u.z)
}

/// WGSL `hash13` (Dave Hoskins). Uses `floor`-based fract (NOT Rust `f32::fract`,
/// which keeps the sign for negatives) so it matches the shader bit-for-bit.
fn hash13(p: Vec3) -> f32 {
    let mut p3 = frac3(p * 0.1031);
    let d = p3.dot(Vec3::new(p3.z, p3.y, p3.x) + Vec3::splat(31.32));
    p3 += Vec3::splat(d);
    frac((p3.x + p3.y) * p3.z)
}

fn wrap_lattice(p: Vec3, period: f32) -> Vec3 {
    p - (p / period).floor() * period
}

#[inline]
fn frac(x: f32) -> f32 {
    x - x.floor()
}

#[inline]
fn frac3(v: Vec3) -> Vec3 {
    v - v.floor()
}

#[inline]
fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// WGSL `smoothstep` — handles `edge0 > edge1` (the inverted-band idiom) the same
/// way the shader does.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moisture_in_range_and_varies() {
        // The fine tier should produce visible variation over a transect and
        // stay in range; the macro offset shifts the whole field.
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for k in 0..200 {
            let p = DVec3::new(3_186_000.0 + k as f64 * 12.0, 0.0, 1234.0);
            let s = sample_landcover(p, 1000.0, 0.0, 0.0);
            assert!(s.moisture >= -1.0 && s.moisture <= 1.0);
            assert!(s.coverage >= 0.0 && s.coverage <= 1.0);
            assert!(s.veg_color.min_element() >= 0.0);
            lo = lo.min(s.moisture);
            hi = hi.max(s.moisture);
        }
        assert!(
            hi - lo > 0.15,
            "fine moisture should vary across a transect"
        );
        // The macro offset carries through (clamped to range).
        let p = DVec3::new(3_186_000.0, 0.0, 1234.0);
        let wet = sample_landcover(p, 1000.0, 0.8, 0.0);
        let dry = sample_landcover(p, 1000.0, -0.8, 0.0);
        assert!(wet.moisture > dry.moisture);
    }

    #[test]
    fn polar_climate_shifts_bands() {
        // The same low ground reads alpine/tundra-tinted at polar latitude:
        // the cold lift pushes the eco altitude past the treeline, so the
        // colour must differ from the tropical sample.
        let p = DVec3::new(3_186_000.0, 0.0, 1234.0);
        let tropical = sample_landcover(p, 300.0, 0.0, 0.05);
        let polar = sample_landcover(p, 300.0, 0.0, 0.99);
        assert_ne!(tropical.veg_color, polar.veg_color);
    }

    #[test]
    fn deterministic() {
        let p = DVec3::new(1.0e6, 2.0e5, -3.0e5);
        let a = sample_landcover(p, 800.0, 0.2, 0.3);
        let b = sample_landcover(p, 800.0, 0.2, 0.3);
        assert_eq!(a.veg_color, b.veg_color);
        assert_eq!(a.moisture, b.moisture);
    }
}
