//! Stage 6 — regolith and space weathering (maturity pass).
//!
//! Per spec §Stage 6: on airless bodies, maturity is the dominant source
//! of visible variation — more than composition.  Fresh surfaces are bright
//! (low maturity), old surfaces dark (high maturity).  This pass finalises
//! the maturity field by combining:
//!
//! - Global baseline from body age (older bodies start more weathered).
//! - Per-sample noise for natural variation.
//! - Per-crater age blending (the crater stamping pre-set `crater_age_gyr`).
//! - Slope-based reset (steep slopes shed regolith and expose fresh material).
//!
//! Maturity is per-sample; this stage does not modify elevation or material.
//!
//! Slope is sampling-dependent — a Fibonacci lattice and an equirect grid
//! need different finite-difference schemes — so the caller is responsible
//! for supplying a per-sample slope magnitude (m/m).  The pipeline provides
//! [`equirect_slope_magnitude`] for the common equirect case; other
//! samplings can pass `None` to skip the slope reset.

use crate::descriptor::PlanetDescriptor;
use crate::noise::fbm3;
use crate::seeding::sub_seed;
use crate::surface::SurfaceState;

pub const STAGE_NAME: &str = "maturity";

/// Reference body age (Gyr) at which the global baseline maturity is 1.0.
pub const REFERENCE_AGE_GYR: f64 = 4.5;
/// Per-sample noise amplitude (added to baseline before clamping).
pub const NOISE_AMPLITUDE: f64 = 0.10;
/// Slope (m/m) at and above which a sample is fully reset to fresh.
pub const SLOPE_FULL_RESET: f64 = 0.4; // ~22°, real crater wall slopes
/// Slope (m/m) below which the slope-based reset is inactive.
pub const SLOPE_INACTIVE: f64 = 0.05;
/// Per-crater age scale (Gyr): a crater's "fresh" maturity blends back
/// toward the baseline over this many gigayears.  Real lunar fresh
/// craters (Tycho, Copernicus) are ~100 Myr — we want only the youngest
/// ~3% of stamped craters to read bright, so this is short.  Matches
/// `FRESH_AGE_GYR` on the shader side so the two populations age at the
/// same rate.
pub const CRATER_AGE_BLEND_GYR: f64 = 0.15;

/// Run Stage 6.  Reads `state.crater_age_gyr` (set by the crater stamps)
/// and optionally a precomputed per-sample slope magnitude in m/m.  Writes
/// only `state.maturity`.
pub fn run(
    state: &mut SurfaceState,
    desc: &PlanetDescriptor,
    top_seed: u64,
    slope_magnitude: Option<&[f64]>,
) {
    let stage_seed = sub_seed(top_seed, STAGE_NAME);

    // Global baseline: linear in age, clamped to [0.3, 1.0].  Even very
    // young bodies are not perfectly fresh because micrometeorite gardening
    // is fast on geological timescales.
    let baseline = (desc.age_gyr / REFERENCE_AGE_GYR).clamp(0.3, 1.0);

    if let Some(slope) = slope_magnitude {
        debug_assert_eq!(slope.len(), state.len());
    }

    for i in 0..state.len() {
        let p = state.points[i];
        let noise = fbm3(p.x * 16.0, p.y * 16.0, p.z * 16.0, stage_seed, 3, 0.5, 2.0);

        // Start from the global baseline + noise.
        let mut m = (baseline + noise * NOISE_AMPLITUDE).clamp(0.0, 1.0);

        // Per-crater age blend: a fresh crater (age = 0) overrides toward
        // 0; an old crater (age >> CRATER_AGE_BLEND_GYR) leaves the
        // baseline alone.
        let crater_age = state.crater_age_gyr[i];
        if crater_age.is_finite() {
            // Existing maturity from the crater stamp tapers from 0 at
            // the crater centre to 1 at the ejecta edge — preserve that
            // shape but blend toward baseline as age accumulates.
            let stamp_maturity = state.maturity[i];
            let t = (crater_age / CRATER_AGE_BLEND_GYR).clamp(0.0, 1.0);
            // Smoothstep blend from stamp value (fresh) to baseline (old).
            let blend = t * t * (3.0 - 2.0 * t);
            let from_crater = stamp_maturity * (1.0 - blend) + m * blend;
            m = m.min(from_crater); // never weather *backwards*
        }

        // Slope-based reset disabled — produced unrealistic bright rings
        // on large crater rims.  Keeping the infrastructure (constants,
        // slope_magnitude argument) so it can be re-enabled with better
        // calibration later.
        let _ = slope_magnitude;

        state.maturity[i] = m.clamp(0.0, 1.0);
    }
}

/// Compute per-sample slope magnitude (m/m) for an equirect-sampled
/// surface state.  The result has the same length as the state and is
/// suitable for [`run`]'s `slope_magnitude` argument.
///
/// `width × height` must equal `state.len()`.  The state's `points` are
/// assumed to be in row-major order matching
/// [`crate::sampling::equirect_lattice`].  Polar rows zero out the
/// east derivative (cos lat → 0 would otherwise blow up).
pub fn equirect_slope_magnitude(state: &SurfaceState, width: usize, height: usize) -> Vec<f64> {
    assert_eq!(state.len(), width * height);
    let r = state.reference_radius_m;
    let dlon = std::f64::consts::TAU / width as f64;
    let dlat = std::f64::consts::PI / height as f64;
    let arc_n = r * dlat;
    let polar_cutoff = 0.05;

    let elev = &state.elevation_m;
    let idx = |x: usize, y: usize| y * width + x;
    let mut out = vec![0.0; state.len()];

    for y in 0..height {
        let v = (y as f64 + 0.5) / height as f64;
        let lat = (v - 0.5) * std::f64::consts::PI;
        let cos_lat = lat.cos();
        let de_factor = if cos_lat.abs() < polar_cutoff {
            0.0
        } else {
            1.0 / (2.0 * r * cos_lat * dlon)
        };
        let (yn, ys, dn_factor) = if y == 0 {
            (0, 1, 1.0 / arc_n)
        } else if y == height - 1 {
            (height - 2, height - 1, 1.0 / arc_n)
        } else {
            (y - 1, y + 1, 1.0 / (2.0 * arc_n))
        };
        for x in 0..width {
            let xe = (x + 1) % width;
            let xw = (x + width - 1) % width;
            let dh_de = (elev[idx(xe, y)] - elev[idx(xw, y)]) * de_factor;
            let dh_dn = (elev[idx(x, ys)] - elev[idx(x, yn)]) * dn_factor;
            out[idx(x, y)] = (dh_de * dh_de + dh_dn * dh_dn).sqrt();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_bodies;
    use crate::sampling::{equirect_lattice, fibonacci_lattice};

    #[test]
    fn baseline_increases_with_age() {
        let pts = fibonacci_lattice(256);
        let mut young_desc = reference_bodies::luna();
        young_desc.age_gyr = 0.5;
        let mut old_desc = reference_bodies::luna();
        old_desc.age_gyr = 4.5;

        let mut young = SurfaceState::new(pts.clone(), 1_737_400.0);
        let mut old = SurfaceState::new(pts, 1_737_400.0);
        run(&mut young, &young_desc, 1, None);
        run(&mut old, &old_desc, 1, None);

        let mean_young: f64 = young.maturity.iter().sum::<f64>() / young.len() as f64;
        let mean_old: f64 = old.maturity.iter().sum::<f64>() / old.len() as f64;
        assert!(mean_old > mean_young, "old not more weathered: {mean_old} vs {mean_young}");
    }

    #[test]
    fn fresh_crater_stays_fresh_after_pass() {
        // Pre-set a fresh crater on one sample (age 0, maturity 0); the
        // pass must not push it back toward the baseline.
        let pts = fibonacci_lattice(64);
        let desc = reference_bodies::luna();
        let mut s = SurfaceState::new(pts, 1_737_400.0);
        s.crater_age_gyr[0] = 0.0;
        s.maturity[0] = 0.0;
        run(&mut s, &desc, 1, None);
        assert!(s.maturity[0] < 0.1, "fresh crater weathered: {}", s.maturity[0]);
    }

    #[test]
    fn old_crater_blends_toward_baseline() {
        let pts = fibonacci_lattice(64);
        let desc = reference_bodies::luna();
        let mut s = SurfaceState::new(pts, 1_737_400.0);
        s.crater_age_gyr[0] = 4.0; // very old
        s.maturity[0] = 0.0; // started fresh
        run(&mut s, &desc, 1, None);
        // After 4 Gyr blend the crater is fully aged toward baseline (~1).
        assert!(s.maturity[0] > 0.5, "old crater not aged: {}", s.maturity[0]);
    }

    #[test]
    fn equirect_slope_magnitude_is_zero_on_flat_state() {
        let w = 32;
        let h = 16;
        let pts = equirect_lattice(w, h);
        let s = SurfaceState::new(pts, 1_737_400.0);
        let slope = equirect_slope_magnitude(&s, w, h);
        assert!(slope.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let pts = fibonacci_lattice(256);
        let desc = reference_bodies::luna();
        let mut a = SurfaceState::new(pts.clone(), 1_737_400.0);
        let mut b = SurfaceState::new(pts, 1_737_400.0);
        run(&mut a, &desc, 7, None);
        run(&mut b, &desc, 7, None);
        assert_eq!(a.maturity, b.maturity);
    }
}
