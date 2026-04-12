//! Single-crater stamping, tested in isolation.
//!
//! Per spec §Implementation order step 3: "one crater, correct simple
//! profile, correct rim, correct ejecta falloff."  This is the load-bearing
//! function for whether airless bodies look like real moons.  This module
//! owns three morphologies (simple bowl, complex flat-floor with central
//! peak, peak-ring) plus the per-crater perturbation (mild ellipticity,
//! rim irregularity, asymmetric ejecta) the spec requires.
//!
//! All geometry is measured in surface arc length from the crater centre
//! (`s = θ · R_body`), which collapses to Euclidean distance for small
//! craters and correctly wraps for large ones.  Elevation modifications are
//! **additive** so existing terrain under the stamp is preserved outside
//! the crater's area of effect and older features show through fresh
//! ejecta, matching spec §Stage 4.

use glam::DVec3;

use crate::seeding::Rng;
use crate::surface::{MaterialId, SurfaceState};

// --- calibration constants ---------------------------------------------------
//
// Values from spec §Physical calibration notes.  Exposed as constants so
// callers and tests can reference them; tune in one place.

/// Depth-to-diameter ratio for fresh simple craters.
pub const SIMPLE_CRATER_DEPTH_RATIO: f64 = 0.2;
/// Rim-height-to-diameter ratio for fresh simple craters.
pub const SIMPLE_CRATER_RIM_RATIO: f64 = 0.04;
/// Outer radius of the ejecta blanket, in units of crater radius measured
/// from the crater centre.  Spec: "roughly 2–3 crater radii."
pub const SIMPLE_CRATER_EJECTA_EXTENT: f64 = 2.5;
/// Interior profile exponent in `h(r) = −depth + (depth + rim) · r^n`.
/// Larger → steeper walls near the rim, wider flatter floor.  2.5 gives a
/// visibly recognisable bowl with the rim peak at r = 1.
pub const SIMPLE_CRATER_INTERIOR_EXPONENT: f64 = 2.5;

// --- types -------------------------------------------------------------------

/// A single simple crater, in physical units.
#[derive(Clone, Copy, Debug)]
pub struct SimpleCrater {
    /// Direction from body centre to the crater centre.  Normalised on stamp.
    pub center: DVec3,
    /// Rim-to-rim diameter in metres.
    pub diameter_m: f64,
}

// --- stamping ----------------------------------------------------------------

/// Stamp a simple (bowl-shaped) crater into `state`.
///
/// The elevation delta is:
/// ```text
///   r ∈ [0, 1]   (interior): Δh = −depth + (depth + rim) · r^n
///   r ∈ (1, E]   (ejecta):   Δh = rim / r^3
///   r >  E       (outside):  unchanged
/// ```
/// where `r = s / R_crater`, `s` is arc length from the crater centre,
/// `n = SIMPLE_CRATER_INTERIOR_EXPONENT`, and `E = SIMPLE_CRATER_EJECTA_EXTENT`.
///
/// At `r = 1` the two branches agree at `+rim` (C⁰ continuous at the rim),
/// and the rim is a local maximum — i.e. the rim peaks at the rim radius,
/// as the spec requires.
///
/// Affected samples have their material set to fresh excavation/ejecta,
/// their crater age reset to 0, and their maturity reset to 0 (fresh).
pub fn stamp_simple_crater(
    state: &mut SurfaceState,
    body_radius_m: f64,
    crater: SimpleCrater,
) {
    let crater_radius = crater.diameter_m * 0.5;
    let depth = crater.diameter_m * SIMPLE_CRATER_DEPTH_RATIO;
    let rim = crater.diameter_m * SIMPLE_CRATER_RIM_RATIO;

    // Early-out threshold in dot-product space: samples with
    // `dot(p, center) < cos(theta_max)` are outside the area of effect.
    let max_arc = crater_radius * SIMPLE_CRATER_EJECTA_EXTENT;
    let max_theta = max_arc / body_radius_m;
    let cos_max = max_theta.cos();

    let center = crater.center.normalize();

    for i in 0..state.len() {
        let cos_theta = state.points[i].dot(center).clamp(-1.0, 1.0);
        if cos_theta < cos_max {
            continue;
        }
        let theta = cos_theta.acos();
        let s = theta * body_radius_m;
        let r = s / crater_radius;

        let (delta_elev, mat) = if r <= 1.0 {
            let h = -depth + (depth + rim) * r.powf(SIMPLE_CRATER_INTERIOR_EXPONENT);
            (h, MaterialId::FreshExcavation)
        } else {
            (ejecta_profile(r, rim), MaterialId::FreshEjecta)
        };

        // Smoothstep maturity taper: fresh (0) at the crater centre,
        // blending toward baseline weathered (1) at the outer ejecta
        // extent.  Without this taper every footprint texel would share
        // `maturity = 0`, producing a hard uniform-bright disc with a
        // visible step at `r = EJECTA_EXTENT`; the smoothstep fades the
        // crater into its surroundings.
        let new_maturity = fresh_crater_maturity(r);

        state.elevation_m[i] += delta_elev;
        state.material[i] = mat;
        state.crater_age_gyr[i] = 0.0;
        // `min` so overlapping younger craters don't age the fresher sample
        // they overwrite — an impact can only freshen the surface.
        state.maturity[i] = state.maturity[i].min(new_maturity);
    }
}

// --- generalized crater stamping (simple / complex / peak-ring) -------------

/// Threshold (multiples of `D_sc`) above which a complex crater grows a
/// peak ring instead of a single central peak.  Spec: "well above transition."
pub const PEAK_RING_THRESHOLD_FACTOR: f64 = 4.0;

/// Depth-to-diameter ratio for the largest complex craters.  Real craters
/// flatten substantially with size; spec calls for ~0.05 at the largest end.
pub const COMPLEX_MIN_DEPTH_RATIO: f64 = 0.05;
/// Fraction of the crater radius occupied by the flat floor in a complex
/// crater (the remaining radial range is the terraced wall).
const COMPLEX_FLOOR_FRACTION: f64 = 0.55;
/// Central-peak height as a fraction of crater depth.  Real moon peaks
/// (Copernicus, Tycho) are ~10–15% of the crater depth — an earlier
/// value of 0.35 made the peak dominate the floor visually and read as
/// a "bright center" instead of a subtle rise.
const COMPLEX_PEAK_HEIGHT_FRAC: f64 = 0.15;
/// Central-peak base radius as a fraction of crater radius.  Narrower
/// than the old 0.25 — the Gaussian halo used to extend across most of
/// the floor and produce a visible bright shoulder.
const COMPLEX_PEAK_BASE_FRAC: f64 = 0.15;
/// Peak-ring radial position as a fraction of crater radius.
const PEAK_RING_RADIUS_FRAC: f64 = 0.5;

/// Maximum eccentricity of the elliptical area-of-effect mask.  Raised
/// from 0.15 to 0.25: real craters are noticeably non-circular, especially
/// at oblique impact angles.
const MAX_ELLIPTICITY: f64 = 0.25;
/// Amplitude of the rim irregularity, in fractions of the rim height.
/// Raised from 0.25 to 0.45 so crater rims have visibly collapsed or
/// built-up sections, matching real crater morphology.
const RIM_IRREGULARITY: f64 = 0.45;
/// Amplitude of the ejecta-blanket asymmetry; the ejecta along one
/// direction is scaled by `1 + AMP·cos(θ − dir)`.
const EJECTA_ASYMMETRY: f64 = 0.4;
/// Maximum depth reduction for the oldest craters.  A 4 Gyr crater on a
/// 4 Gyr body retains only `1 − MAX_DEGRADATION` of its original depth,
/// modelling infill by subsequent ejecta and micrometeorite gardening.
const MAX_DEGRADATION: f64 = 0.7;
/// Number of terrace steps in complex crater walls.  Real lunar craters
/// (Copernicus, Aristarchus) show 2–4 slump terraces.
const TERRACE_STEPS: f64 = 3.0;
/// Amplitude of the terrace undulation, in fractions of crater depth.
const TERRACE_AMPLITUDE: f64 = 0.04;

/// A single crater of any morphology.  The morphology selected at stamp
/// time is determined by `diameter_m` versus the body's simple-to-complex
/// transition diameter.
#[derive(Clone, Copy, Debug)]
pub struct Crater {
    /// Direction from body centre to the crater centre.  Normalised on stamp.
    pub center: DVec3,
    /// Rim-to-rim diameter in metres.
    pub diameter_m: f64,
    /// Time before present at which the impact happened, gigayears.  Used
    /// by the maturity stage; recorded into `crater_age_gyr` on every
    /// affected sample (taking the minimum, so younger craters win).
    pub age_gyr: f64,
    /// Per-crater perturbation seed.  Drives ellipticity, rim noise, and
    /// ejecta asymmetry so two adjacent craters look distinct.
    pub perturb_seed: u64,
}

/// Smoothly interpolate the depth-to-diameter ratio between simple and the
/// floor for the largest complex craters.  Diameter normalised by `d_sc`.
fn complex_depth_ratio(diameter_over_dsc: f64) -> f64 {
    // At d/d_sc = 1 we match the simple ratio; at d/d_sc → ∞ we approach
    // COMPLEX_MIN_DEPTH_RATIO.  Smooth exponential decay.
    let t = (-(diameter_over_dsc - 1.0).max(0.0) / 3.0).exp();
    COMPLEX_MIN_DEPTH_RATIO + (SIMPLE_CRATER_DEPTH_RATIO - COMPLEX_MIN_DEPTH_RATIO) * t
}

/// Compute a degradation factor in `[1 − MAX_DEGRADATION, 1]` based on
/// crater age relative to body age.  Fresh craters (age 0) return 1.0;
/// the oldest craters return `1 − MAX_DEGRADATION`.  Uses a sqrt curve
/// because most erosion happens early (ejecta bombardment from other
/// impacts is heaviest on the young, heavily-cratered surface).
fn degradation_factor(crater_age_gyr: f64, body_age_gyr: f64) -> f64 {
    if body_age_gyr <= 0.0 {
        return 1.0;
    }
    let t = (crater_age_gyr / body_age_gyr).clamp(0.0, 1.0);
    1.0 - MAX_DEGRADATION * t.sqrt()
}

/// Stamp a crater of any morphology.  Selects simple, complex, or peak-ring
/// based on `diameter_m / d_sc_m`, applies per-crater perturbation and
/// age-dependent degradation, and writes the appropriate material tag.
///
/// `d_sc_m` is the body's simple-to-complex transition diameter (metres),
/// from `DerivedProperties::simple_to_complex_transition_m`.
///
/// `body_age_gyr` is the body's total age in gigayears, used to scale
/// the degradation of older craters.
pub fn stamp_crater(
    state: &mut SurfaceState,
    body_radius_m: f64,
    crater: Crater,
    d_sc_m: f64,
    body_age_gyr: f64,
) {
    // Per-crater RNG drives the perturbation knobs once, deterministically.
    let mut rng = Rng::new(crater.perturb_seed);
    let ellipticity = rng.range_f64(0.0, MAX_ELLIPTICITY);
    let ellipse_orientation = rng.range_f64(0.0, std::f64::consts::TAU);
    let rim_phase = rng.range_f64(0.0, std::f64::consts::TAU);
    let rim_lobes = (rng.range_f64(3.0, 7.0)).floor(); // 3..6 lobes
    let ejecta_phase = rng.range_f64(0.0, std::f64::consts::TAU);
    let terrace_phase = rng.range_f64(0.0, std::f64::consts::TAU);

    let crater_radius = crater.diameter_m * 0.5;
    let normalised = crater.diameter_m / d_sc_m;
    let is_complex = normalised >= 1.0;
    let is_peak_ring = normalised >= PEAK_RING_THRESHOLD_FACTOR;

    // Age-dependent degradation: old craters are shallower with softer rims.
    let degrade = degradation_factor(crater.age_gyr, body_age_gyr);

    let depth_ratio = if is_complex {
        complex_depth_ratio(normalised)
    } else {
        SIMPLE_CRATER_DEPTH_RATIO
    };
    let depth = crater.diameter_m * depth_ratio * degrade;
    let rim = crater.diameter_m * SIMPLE_CRATER_RIM_RATIO * degrade;

    // Build a local east/north basis at the crater centre so we can compute
    // an azimuth for the elliptical mask, rim phase, and ejecta asymmetry.
    let center = crater.center.normalize();
    let world_up = if center.y.abs() < 0.95 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let east = world_up.cross(center).normalize();
    let north = center.cross(east).normalize();

    // Maximum ejecta extent (using the major axis of the ellipse).
    let max_arc = crater_radius * SIMPLE_CRATER_EJECTA_EXTENT * (1.0 + ellipticity);
    let max_theta = max_arc / body_radius_m;
    let cos_max = max_theta.cos();

    // Both simple and complex craters share the FreshExcavation tag for
    // their interior; the maturity stage and material palette differentiate
    // by maturity, not by morphology.  Kept as a `let` so the dispatch is
    // explicit and the future "complex craters write a distinct tag" hook
    // is one line away.
    let body_material = MaterialId::FreshExcavation;

    for i in 0..state.len() {
        let p = state.points[i];
        let cos_theta = p.dot(center).clamp(-1.0, 1.0);
        if cos_theta < cos_max {
            continue;
        }
        let theta = cos_theta.acos();
        let s_arc = theta * body_radius_m;

        // Azimuth in the local east/north plane around the crater centre.
        let e_proj = p.dot(east);
        let n_proj = p.dot(north);
        let azimuth = e_proj.atan2(n_proj); // 0 = north, +π/2 = east

        // Elliptical area-of-effect: scale arc length by an azimuth-dependent
        // factor in [1 - e, 1 + e] so the same nominal radius traces an
        // ellipse on the surface.
        let ellipse_factor =
            1.0 + ellipticity * (2.0 * (azimuth - ellipse_orientation)).cos();
        let r = (s_arc / crater_radius) / ellipse_factor;

        if r > SIMPLE_CRATER_EJECTA_EXTENT {
            continue;
        }

        // Profile in the normalised radial coordinate `r`.
        let (delta_elev, mat) = if r <= 1.0 {
            let h = if is_peak_ring {
                peak_ring_profile(r, depth, rim)
            } else if is_complex {
                complex_profile(r, depth, rim)
            } else {
                simple_profile(r, depth, rim)
            };
            (h, body_material)
        } else {
            // Ejecta blanket: tapered inverse-cube from the rim, modulated
            // by an azimuth-dependent asymmetry term.
            let asym = 1.0 + EJECTA_ASYMMETRY * (azimuth - ejecta_phase).cos();
            (ejecta_profile(r, rim) * asym, MaterialId::FreshEjecta)
        };

        // Wall terracing for complex craters: sinusoidal undulation along
        // the wall ramp, producing step-like features that mimic slump
        // terraces.  Amplitude fades near the rim so it blends smoothly
        // into the rim irregularity below.
        let terrace = if is_complex && r > COMPLEX_FLOOR_FRACTION && r < 1.0 {
            let wall_t = (r - COMPLEX_FLOOR_FRACTION) / (1.0 - COMPLEX_FLOOR_FRACTION);
            let wave = (wall_t * TERRACE_STEPS * std::f64::consts::PI + terrace_phase).sin();
            TERRACE_AMPLITUDE * depth * wave * (1.0 - wall_t)
        } else {
            0.0
        };

        // Mild rim irregularity: only inside `r ≈ [0.85, 1.15]` and only
        // a fraction of the rim height — keeps the average rim profile
        // intact while breaking up the perfect circle.
        let irregular = if (0.85..=1.15).contains(&r) {
            let lobes = (rim_lobes * azimuth + rim_phase).sin();
            RIM_IRREGULARITY * rim * lobes * (1.0 - 4.0 * (r - 1.0).powi(2)).max(0.0)
        } else {
            0.0
        };

        let new_maturity = fresh_crater_maturity(r);

        state.elevation_m[i] += delta_elev + irregular + terrace;
        state.material[i] = mat;
        // `min` so an older crater age never overwrites a younger one
        // already recorded by an overlapping younger crater.  NaN
        // (uninitialised) compares as "no recorded impact"; treat NaN as
        // "no constraint" and adopt this crater's age unconditionally.
        let prev = state.crater_age_gyr[i];
        state.crater_age_gyr[i] = if prev.is_nan() {
            crater.age_gyr
        } else {
            prev.min(crater.age_gyr)
        };
        state.maturity[i] = state.maturity[i].min(new_maturity);
    }
}

/// Simple bowl profile in the normalised radial coordinate `r ∈ [0, 1]`.
fn simple_profile(r: f64, depth: f64, rim: f64) -> f64 {
    -depth + (depth + rim) * r.powf(SIMPLE_CRATER_INTERIOR_EXPONENT)
}

/// Ejecta-blanket profile for `r ∈ (1, EJECTA_EXTENT]`.  Base shape is
/// an inverse-cube falloff from the rim; the tapered smoothstep makes it
/// **C¹-continuous at the extent boundary** so the bake's finite-
/// difference slope pass doesn't pick up a step at `r = EJECTA_EXTENT`.
/// Without the taper you get thin circular artifacts around every large
/// crater where the ejecta cut off abruptly.
fn ejecta_profile(r: f64, rim: f64) -> f64 {
    let t = ((r - 1.0) / (SIMPLE_CRATER_EJECTA_EXTENT - 1.0)).clamp(0.0, 1.0);
    // Smoothstep fade that hits 1 at r=1 and 0 at r=EJECTA_EXTENT, with
    // zero first derivative at both endpoints.
    let smooth = t * t * (3.0 - 2.0 * t);
    let fade = 1.0 - smooth;
    (rim / r.powi(3)) * fade
}

/// Width of the bright fresh-rim annulus in units of crater radius (the
/// Gaussian sigma in the freshness dip below).  Smaller → tighter rim
/// highlight; ~0.15 lines up with what real fresh-rim annuli look like
/// against the floor on the Moon.
pub const RIM_FRESHNESS_SIGMA: f64 = 0.22;

/// Maturity contribution of a fresh impact at radius `r` (in units of
/// crater radius).  Returns a value in `[0, 1]` where 0 = bright fresh and
/// 1 = baseline weathered.
///
/// Shape:
/// - Floor (`r ≪ 1`): baseline (1).  Real crater floors are not the
///   brightest part — the rim and ejecta are.
/// - Rim (`r ≈ 1`): a tight Gaussian dip toward 0.
/// - Ejecta apron (`r > 1`): a smooth quadratic taper from the rim back
///   to baseline at the ejecta extent.
///
/// The interior taper through the wall is also handled by the Gaussian's
/// inner shoulder, so the inner crater wall reads as the "bright wall"
/// real moon photos show against the dark floor.
pub fn fresh_crater_maturity(r: f64) -> f64 {
    let dr = r - 1.0;
    let dip = (-(dr * dr) / (2.0 * RIM_FRESHNESS_SIGMA * RIM_FRESHNESS_SIGMA)).exp();
    let ejecta_freshness = if r > 1.0 && r < SIMPLE_CRATER_EJECTA_EXTENT {
        let t = (r - 1.0) / (SIMPLE_CRATER_EJECTA_EXTENT - 1.0);
        // Quadratic falloff: most of the freshness is just outside the rim.
        let one_minus_t = 1.0 - t;
        one_minus_t * one_minus_t
    } else {
        0.0
    };
    let freshness = dip.max(ejecta_freshness);
    (1.0 - freshness).clamp(0.0, 1.0)
}

/// Complex crater profile: shallower flat floor for `r < FLOOR_FRACTION`,
/// terraced wall rising to the rim, plus a Gaussian central peak.
fn complex_profile(r: f64, depth: f64, rim: f64) -> f64 {
    let floor_h = -depth;
    // Base profile: flat floor → smooth ramp → rim.
    let base = if r <= COMPLEX_FLOOR_FRACTION {
        floor_h
    } else {
        let t = (r - COMPLEX_FLOOR_FRACTION) / (1.0 - COMPLEX_FLOOR_FRACTION);
        // Smoothstep ramp from floor to rim.  This produces a recognisable
        // terraced-wall feel without a separate terrace generator.
        let s = t * t * (3.0 - 2.0 * t);
        floor_h + (depth + rim) * s
    };
    // Central peak: small Gaussian centred at r=0.
    let peak_sigma = COMPLEX_PEAK_BASE_FRAC;
    let peak = COMPLEX_PEAK_HEIGHT_FRAC * depth * (-(r * r) / (2.0 * peak_sigma * peak_sigma)).exp();
    base + peak
}

/// Peak-ring profile: like a complex crater but the central peak is
/// replaced by a concentric ring of peaks at `r ≈ PEAK_RING_RADIUS_FRAC`.
fn peak_ring_profile(r: f64, depth: f64, rim: f64) -> f64 {
    let floor_h = -depth;
    let base = if r <= COMPLEX_FLOOR_FRACTION {
        floor_h
    } else {
        let t = (r - COMPLEX_FLOOR_FRACTION) / (1.0 - COMPLEX_FLOOR_FRACTION);
        let s = t * t * (3.0 - 2.0 * t);
        floor_h + (depth + rim) * s
    };
    let ring_sigma = 0.08;
    let dr = r - PEAK_RING_RADIUS_FRAC;
    let ring = COMPLEX_PEAK_HEIGHT_FRAC * depth
        * (-(dr * dr) / (2.0 * ring_sigma * ring_sigma)).exp();
    base + ring
}

// --- tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a surface state with a handful of points laid along a single
    /// great circle passing through the crater centre (north pole), so we
    /// can place samples at exact known radii from the crater centre.
    fn handpicked_state(body_radius_m: f64, crater_radius_m: f64, radii: &[f64]) -> SurfaceState {
        let points: Vec<DVec3> = radii
            .iter()
            .map(|&r| {
                let theta = r * crater_radius_m / body_radius_m;
                DVec3::new(theta.sin(), theta.cos(), 0.0)
            })
            .collect();
        SurfaceState::new(points, body_radius_m)
    }

    fn assert_close(actual: f64, expected: f64, abs_tol: f64, label: &str) {
        let err = (actual - expected).abs();
        assert!(
            err <= abs_tol,
            "{label}: expected {expected}, got {actual}, abs err {err:.3e} > tol {abs_tol:.3e}"
        );
    }

    const BODY_R: f64 = 1_000_000.0;
    const DIAM: f64 = 100_000.0;

    fn fresh_crater() -> SimpleCrater {
        SimpleCrater {
            center: DVec3::Y,
            diameter_m: DIAM,
        }
    }

    #[test]
    fn center_is_at_minus_depth() {
        let mut s = handpicked_state(BODY_R, DIAM * 0.5, &[0.0]);
        stamp_simple_crater(&mut s, BODY_R, fresh_crater());
        let depth = DIAM * SIMPLE_CRATER_DEPTH_RATIO;
        assert_close(s.elevation_m[0], -depth, 1e-6, "center depth");
        assert_eq!(s.material[0], MaterialId::FreshExcavation);
    }

    #[test]
    fn rim_is_at_plus_rim_height() {
        let mut s = handpicked_state(BODY_R, DIAM * 0.5, &[1.0]);
        stamp_simple_crater(&mut s, BODY_R, fresh_crater());
        let rim = DIAM * SIMPLE_CRATER_RIM_RATIO;
        assert_close(s.elevation_m[0], rim, 1e-6, "rim height");
        // The rim is the boundary between interior and ejecta; we pick the
        // interior branch at r == 1, so material is FreshExcavation.
        assert_eq!(s.material[0], MaterialId::FreshExcavation);
    }

    #[test]
    fn ejecta_decays_smoothly_to_zero_at_extent() {
        let mut s = handpicked_state(BODY_R, DIAM * 0.5, &[1.5, 2.0, 2.499]);
        stamp_simple_crater(&mut s, BODY_R, fresh_crater());
        let rim = DIAM * SIMPLE_CRATER_RIM_RATIO;
        // Helper: the same tapered formula that `ejecta_profile` uses.
        let tapered = |r: f64| -> f64 {
            let t = ((r - 1.0) / (SIMPLE_CRATER_EJECTA_EXTENT - 1.0)).clamp(0.0, 1.0);
            let smooth = t * t * (3.0 - 2.0 * t);
            let fade = 1.0 - smooth;
            rim / r.powi(3) * fade
        };
        assert_close(s.elevation_m[0], tapered(1.5), 1e-6, "ejecta r=1.5");
        assert_close(s.elevation_m[1], tapered(2.0), 1e-6, "ejecta r=2.0");
        // Near the extent boundary, elevation must be *nearly* zero so
        // the bake's finite-difference slope pass doesn't pick up a
        // step.  `tapered(2.499)` is on the order of 1e-9 · rim.
        let near_boundary = tapered(2.499);
        assert!(
            near_boundary.abs() < rim * 1e-5,
            "ejecta r≈2.5 should be ~0, got {near_boundary}"
        );
        assert_close(s.elevation_m[2], near_boundary, 1e-6, "ejecta r=2.499");
        assert_eq!(s.material[0], MaterialId::FreshEjecta);
    }

    #[test]
    fn samples_beyond_ejecta_extent_are_untouched() {
        let mut s = handpicked_state(BODY_R, DIAM * 0.5, &[3.0, 5.0]);
        // Give each a pre-existing elevation so "untouched" is observable.
        s.elevation_m[0] = 123.0;
        s.elevation_m[1] = -456.0;
        let initial_material = s.material.clone();
        stamp_simple_crater(&mut s, BODY_R, fresh_crater());
        assert_eq!(s.elevation_m[0], 123.0);
        assert_eq!(s.elevation_m[1], -456.0);
        assert_eq!(s.material, initial_material);
        assert!(s.crater_age_gyr[0].is_nan());
    }

    #[test]
    fn stamping_is_additive_over_existing_terrain() {
        let mut s = handpicked_state(BODY_R, DIAM * 0.5, &[0.0, 1.0, 2.0]);
        s.elevation_m[0] = 500.0;
        s.elevation_m[1] = -200.0;
        s.elevation_m[2] = 77.0;
        stamp_simple_crater(&mut s, BODY_R, fresh_crater());
        let depth = DIAM * SIMPLE_CRATER_DEPTH_RATIO;
        let rim = DIAM * SIMPLE_CRATER_RIM_RATIO;
        // Expected ejecta contribution at r=2 via the tapered formula.
        let ejecta_r2 = {
            let t = (2.0 - 1.0) / (SIMPLE_CRATER_EJECTA_EXTENT - 1.0);
            let smooth = t * t * (3.0 - 2.0 * t);
            rim / 8.0 * (1.0 - smooth)
        };
        assert_close(s.elevation_m[0], 500.0 - depth, 1e-6, "additive center");
        assert_close(s.elevation_m[1], -200.0 + rim, 1e-6, "additive rim");
        assert_close(s.elevation_m[2], 77.0 + ejecta_r2, 1e-6, "additive ejecta");
    }

    #[test]
    fn rim_is_a_local_maximum_and_profile_is_monotonic_interior() {
        // Interior: elevation should increase monotonically from -depth at
        // the centre to +rim at r=1.  Rim should exceed both the interior
        // just inside (r=0.95) and the ejecta just outside (r=1.05).
        let mut s = handpicked_state(
            BODY_R,
            DIAM * 0.5,
            &[0.0, 0.25, 0.5, 0.75, 0.95, 1.0, 1.05, 1.25, 1.5, 2.0],
        );
        stamp_simple_crater(&mut s, BODY_R, fresh_crater());
        // Interior monotonic rise.
        for i in 0..5 {
            assert!(
                s.elevation_m[i + 1] > s.elevation_m[i],
                "interior not monotonic at index {i}: {} → {}",
                s.elevation_m[i],
                s.elevation_m[i + 1]
            );
        }
        // Rim is highest in the sequence.
        let rim_val = s.elevation_m[5];
        for (i, &e) in s.elevation_m.iter().enumerate() {
            if i == 5 {
                continue;
            }
            assert!(
                e < rim_val,
                "elevation at index {i} ({e}) not below rim ({rim_val})"
            );
        }
    }

    #[test]
    fn stamp_resets_age_and_tapers_maturity() {
        // Sample at r = 0 (floor center), r = 1 (rim), r = 1.5 (ejecta),
        // r = 3 (beyond ejecta).
        let mut s = handpicked_state(BODY_R, DIAM * 0.5, &[0.0, 1.0, 1.5, 3.0]);
        stamp_simple_crater(&mut s, BODY_R, fresh_crater());
        // Age is reset on every affected sample (up to the ejecta extent);
        // the sample at r = 3.0 is beyond extent and untouched.
        for i in 0..3 {
            assert_eq!(s.crater_age_gyr[i], 0.0);
        }
        assert!(s.crater_age_gyr[3].is_nan());
        // Floor sample (r=0) is at baseline — the new model puts brightness
        // on the rim and ejecta, not the floor.
        assert!(s.maturity[0] > 0.95, "floor not at baseline: {}", s.maturity[0]);
        // Rim sample (r=1) is at the freshness peak — brightest.
        assert!(s.maturity[1] < 0.05, "rim not fresh: {}", s.maturity[1]);
        // Ejecta sample (r=1.5) is in between.
        assert!(
            s.maturity[2] > s.maturity[1] && s.maturity[2] < s.maturity[0],
            "ejecta not in between: {}",
            s.maturity[2]
        );
        // Untouched sample keeps the baseline maturity.
        assert_eq!(s.maturity[3], 1.0);
    }

    #[test]
    fn fresh_crater_maturity_floor_is_baseline_rim_is_fresh() {
        // Quick standalone check of the new function shape.
        assert!(fresh_crater_maturity(0.0) > 0.99); // floor center → baseline
        assert!(fresh_crater_maturity(0.5) > 0.90); // floor mid → near baseline
        assert!(fresh_crater_maturity(1.0) < 0.01); // rim crest → fully fresh
        assert!(fresh_crater_maturity(0.85) < 0.5); // inner wall → partway fresh
        assert!(fresh_crater_maturity(1.25) < 0.7); // immediate ejecta → fresh
        assert!(fresh_crater_maturity(2.49) > 0.99); // ejecta edge → near baseline
        assert!((fresh_crater_maturity(3.0) - 1.0).abs() < 1e-12); // outside → baseline
    }

    // ---------- complex / peak-ring tests ----------------------------------

    fn unified_crater(diam: f64) -> Crater {
        Crater {
            center: DVec3::Y,
            diameter_m: diam,
            age_gyr: 1.0,
            perturb_seed: 12345,
        }
    }

    /// Pre-allocate a Fibonacci surface dense enough to land samples
    /// inside any crater bigger than ~1% of body radius.
    fn dense_state(body_r: f64) -> SurfaceState {
        SurfaceState::new(crate::sampling::fibonacci_lattice(16384), body_r)
    }

    #[test]
    fn unified_stamp_simple_matches_dispatch() {
        // For diameters below d_sc the unified stamp_crater must select
        // the simple branch and produce a recognisable bowl.  A 100 km
        // crater on a 1737 km body comfortably contains many lattice
        // samples; d_sc = 200 km keeps it firmly simple.
        let body_r = 1_737_400.0;
        let mut s = dense_state(body_r);
        let diam = 100_000.0;
        stamp_crater(&mut s, body_r, unified_crater(diam), 200_000.0, 4.0);
        let min = s.elevation_m.iter().cloned().fold(f64::MAX, f64::min);
        let max = s.elevation_m.iter().cloned().fold(f64::MIN, f64::max);
        let depth = diam * SIMPLE_CRATER_DEPTH_RATIO;
        assert!(min < -0.5 * depth, "simple bowl too shallow: min = {min}");
        assert!(max > 0.0, "simple rim missing: max = {max}");
    }

    #[test]
    fn degradation_factor_is_correct() {
        // Fresh crater → no degradation.
        assert!((degradation_factor(0.0, 4.0) - 1.0).abs() < 1e-12);
        // Oldest crater → maximum degradation.
        let oldest = degradation_factor(4.0, 4.0);
        assert!((oldest - (1.0 - MAX_DEGRADATION)).abs() < 1e-12);
        // Middle-aged crater is between.
        let mid = degradation_factor(2.0, 4.0);
        assert!(mid > oldest && mid < 1.0);
        // Zero body age → no degradation.
        assert!((degradation_factor(0.0, 0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn complex_profile_has_flat_floor_and_central_peak() {
        let depth = 5000.0;
        let rim = 200.0;
        // The central-peak Gaussian sits *on top of* the flat floor with a
        // halo that extends well into the floor region; this is the intended
        // morphology, not a bug.  At r=0 the centre must rise above the
        // floor; far from the centre but still inside the floor region the
        // surface should be near `-depth`.
        let centre_h = complex_profile(0.0, depth, rim);
        let edge_floor = complex_profile(0.5, depth, rim);
        // Centre is above the floor (peak present).
        assert!(centre_h > -depth, "central peak missing: {centre_h}");
        // Centre is still inside the bowl (peak does not breach the rim).
        assert!(centre_h < 0.0, "central peak above zero: {centre_h}");
        // Edge of floor (just before the wall ramp) is within ~25% of the
        // depth from −depth — the Gaussian halo has decayed substantially.
        assert!(
            edge_floor < -0.7 * depth,
            "edge of floor not near -depth: {edge_floor}"
        );
        // Rim transition: at r=1 we should be at +rim, modulo the
        // negligible far tail of the central-peak Gaussian.
        let rim_h = complex_profile(1.0, depth, rim);
        assert!((rim_h - rim).abs() < rim * 0.01, "rim wrong: {rim_h}");
    }

    #[test]
    fn peak_ring_profile_has_ring_above_floor_at_half_radius() {
        let depth = 8000.0;
        let rim = 300.0;
        let centre_h = peak_ring_profile(0.0, depth, rim);
        let ring_h = peak_ring_profile(PEAK_RING_RADIUS_FRAC, depth, rim);
        // Sample well away from the ring's narrow Gaussian (sigma = 0.08).
        let floor_h = peak_ring_profile(0.2, depth, rim);
        // The ring sits above the floor.
        assert!(ring_h > floor_h, "ring not above floor: ring={ring_h} floor={floor_h}");
        // No central peak: centre and floor agree.
        assert!((centre_h - floor_h).abs() < depth * 0.05);
        // Centre and floor are both at -depth (no peak there).
        assert!((centre_h - (-depth)).abs() < depth * 0.05);
    }

    #[test]
    fn complex_depth_ratio_decreases_with_size() {
        let s = complex_depth_ratio(1.0);
        let m = complex_depth_ratio(3.0);
        let l = complex_depth_ratio(10.0);
        assert!(s > m && m > l, "depth ratio not monotonic: {s} {m} {l}");
        assert!(l >= COMPLEX_MIN_DEPTH_RATIO);
        assert!((s - SIMPLE_CRATER_DEPTH_RATIO).abs() < 1e-12);
    }

    #[test]
    fn perturbation_changes_crater_for_different_seeds() {
        let body_r = 1_737_400.0;
        let pts = crate::sampling::fibonacci_lattice(8192);
        let mut a = SurfaceState::new(pts.clone(), body_r);
        let mut b = SurfaceState::new(pts, body_r);
        let mut crater = unified_crater(50_000.0);
        crater.perturb_seed = 1;
        stamp_crater(&mut a, body_r, crater, 15_000.0, 4.0);
        crater.perturb_seed = 2;
        stamp_crater(&mut b, body_r, crater, 15_000.0, 4.0);
        // The two stamps must produce different elevation fields.
        assert_ne!(a.elevation_m, b.elevation_m);
    }

    #[test]
    fn stamp_crater_records_age_and_takes_minimum() {
        let body_r = 1_737_400.0;
        let mut s = dense_state(body_r);
        // Stamp an old crater.
        stamp_crater(
            &mut s,
            body_r,
            Crater {
                center: DVec3::Y,
                diameter_m: 100_000.0,
                age_gyr: 4.0,
                perturb_seed: 1,
            },
            15_000.0,
            4.0,
        );
        // Then a younger overlapping one.
        stamp_crater(
            &mut s,
            body_r,
            Crater {
                center: DVec3::Y,
                diameter_m: 50_000.0,
                age_gyr: 1.0,
                perturb_seed: 2,
            },
            15_000.0,
            4.0,
        );
        // Samples right at +Y should now read 1.0 (the minimum / younger).
        let centre_idx = s
            .points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (1.0 - a.dot(DVec3::Y))
                    .partial_cmp(&(1.0 - b.dot(DVec3::Y)))
                    .unwrap()
            })
            .unwrap()
            .0;
        assert!(s.crater_age_gyr[centre_idx] <= 1.0 + 1e-12);
    }

    #[test]
    fn profile_produces_finite_values_everywhere_on_global_lattice() {
        // Sanity: stamp a crater into a real Fibonacci-sampled surface and
        // check no NaN/inf leaks through.
        use crate::sampling::fibonacci_lattice;
        let pts = fibonacci_lattice(2048);
        let mut s = SurfaceState::new(pts, BODY_R);
        stamp_simple_crater(&mut s, BODY_R, fresh_crater());
        assert!(s.elevation_m.iter().all(|e| e.is_finite()));
        // And at least one sample should have been modified.
        assert!(s.elevation_m.iter().any(|&e| e != 0.0));
    }
}
