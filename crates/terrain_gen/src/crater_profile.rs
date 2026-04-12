//! Single source of truth for explicit-crater morphometry and radial profiles.
//!
//! Both the `Cratering` stage (bake path) and the CPU `sample()` function
//! (read path) evaluate the same math from this module. The shader's detail
//! noise layer uses a different, simplified profile family defined in
//! `assets/shaders/planet_impostor.wgsl` — see `sample::sample_detail_noise`
//! for the matching CPU port. These two profile families are intentionally
//! separate: explicit craters use full Pike/Krüger morphometry, while the
//! statistical detail tail uses the simpler shader profile for performance
//! and continuity with the hash-based synthesis.
//!
//! All profiles take normalized radial distance `t = surface_distance / radius`
//! and return an additive height delta in meters. Negative values carve into
//! the terrain; positive values raise it.

/// Crater morphology classes, determined by size.
///
/// Thresholds scaled for Mira (~870 km radius, ~half lunar surface gravity).
/// Simple-to-complex transition scales inversely with gravity; at ~0.5 g_lunar
/// the transition shifts from ~15 km to ~30 km diameter.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Morphology {
    Simple,
    Complex,
    PeakRing,
    MultiRing,
}

/// Radius thresholds for morphology transitions (meters).
const SIMPLE_MAX: f32    = 15_000.0;   // < 30 km diameter
const COMPLEX_MAX: f32   = 68_500.0;   // < 137 km diameter
const PEAK_RING_MAX: f32 = 150_000.0;  // < 300 km diameter

pub(crate) fn morphology_for_radius(radius_m: f32) -> Morphology {
    if radius_m < SIMPLE_MAX      { Morphology::Simple }
    else if radius_m < COMPLEX_MAX  { Morphology::Complex }
    else if radius_m < PEAK_RING_MAX { Morphology::PeakRing }
    else                           { Morphology::MultiRing }
}

/// Crater depth and rim height from radius, using Pike (1977) lunar morphometry.
///
/// Returns `(depth_m, rim_height_m)`. Depth is measured from pre-impact terrain
/// to crater floor; rim height is measured from pre-impact terrain to rim crest.
pub(crate) fn crater_dimensions(radius_m: f32) -> (f32, f32) {
    let d_km = radius_m * 2.0 / 1000.0; // diameter in km
    match morphology_for_radius(radius_m) {
        Morphology::Simple => {
            // Pike (1977): d/D ≈ 0.196, h_rim/D = 0.036
            (radius_m * 0.392, radius_m * 0.072)
        }
        Morphology::Complex => {
            // Pike (1977): d = 0.196 D^0.301 km, h_rim = 0.236 D^0.399 km
            let depth_m = 196.0 * d_km.powf(0.301);
            let rim_m   = 236.0 * d_km.powf(0.399);
            (depth_m, rim_m)
        }
        Morphology::PeakRing => {
            // Extrapolated Pike trend, d/D ≈ 0.03-0.04
            let depth_m = 140.0 * d_km.powf(0.25);
            let rim_m   = 200.0 * d_km.powf(0.38);
            (depth_m, rim_m)
        }
        Morphology::MultiRing => {
            // Very shallow; isostatic relaxation dominates
            let depth_m = 100.0 * d_km.powf(0.20);
            let rim_m   = 160.0 * d_km.powf(0.35);
            (depth_m, rim_m)
        }
    }
}

/// Hermite smoothstep: 3t^2 - 2t^3 on [0, 1], used inside profile zones.
#[inline]
fn s01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Ranged smoothstep: smooth 0→1 transition as `x` moves across `[edge0, edge1]`.
#[inline]
pub(crate) fn smoothstep_range(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 { return if x >= edge1 { 1.0 } else { 0.0 }; }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Crater profile functions
// ---------------------------------------------------------------------------
//
// All profiles use normalized radial distance t = surface_distance / radius.
// The ejecta zone (t > 1) uses the McGetchin, Settle & Head (1973) ejecta
// thickness law: thickness = 0.14 × R^0.74 × (r/R)^-3, with R and thickness
// in meters. This is independent of rim height and gives a physically
// correct apron that is lower than the rim crest — producing the "protruding
// rim" silhouette. Ejecta is cut off at t = 5 (~90% of mass within 5R).

const RIM_SIGMA: f32 = 0.10;
const EJECTA_MAX_T: f32 = 5.0;

/// Narrow Gaussian ridge centered on the rim crest (t = 1), normalized
/// so its peak value is 1.0. Width is controlled by `RIM_SIGMA`.
#[inline]
fn rim_ridge(t: f32) -> f32 {
    let x = (t - 1.0) / RIM_SIGMA;
    (-x * x).exp()
}

/// McGetchin ejecta apron thickness at normalized radial distance t ≥ 1.
#[inline]
fn ejecta_apron(t: f32, radius_m: f32) -> f32 {
    if t < 1.0 || t > EJECTA_MAX_T { return 0.0; }
    0.14 * radius_m.powf(0.74) * t.powi(-3)
}

/// Simple (bowl-shaped) crater profile.
///
/// Zones: parabolic bowl (0 < t < 1) rising into a sharp Gaussian rim ridge
/// at t ≈ 1, plus a McGetchin ejecta apron (1 ≤ t ≤ 5). The ridge is narrow
/// and sits above the surrounding ejecta so the crest reads as a distinct
/// topographic high — essential for crisp terminator shadows.
pub(crate) fn simple_profile(t: f32, depth: f32, rim_h: f32, radius_m: f32) -> f32 {
    let bowl = if t < 1.0 { -depth * (1.0 - t * t) } else { 0.0 };
    let ridge = rim_h * rim_ridge(t);
    bowl + ridge + ejecta_apron(t, radius_m)
}

/// Complex crater profile (flat floor, central peak, terraced wall).
pub(crate) fn complex_profile(t: f32, depth: f32, rim_h: f32, radius_m: f32) -> f32 {
    let d_km = radius_m * 2.0 / 1000.0;

    // Krüger et al. (2018): h_peak = 0.0049 × D^1.297 km (D in km)
    let h_peak = 4.9 * d_km.powf(1.297); // meters above the floor

    // Pike (1977): D_floor = 0.187 × D^1.249 → t_floor = 0.187 × D^0.249
    let t_floor = (0.187 * d_km.powf(0.249)).clamp(0.30, 0.75);

    // t_peak = 0.168 (Krüger et al.)
    let t_peak = 0.168_f32;

    if t < t_peak {
        let s = t / t_peak;
        -depth + h_peak * (1.0 - s * s)
    } else if t < t_floor {
        -depth
    } else if t < 1.0 {
        let s = (t - t_floor) / (1.0 - t_floor);
        -depth + (depth + rim_h) * s01(s)
    } else if t <= 5.0 {
        rim_h * t.powi(-3)
    } else {
        0.0
    }
}

/// Peak-ring basin profile.
pub(crate) fn peak_ring_profile(t: f32, depth: f32, rim_h: f32) -> f32 {
    let ring_h = depth * 0.2;
    if t < 0.25 {
        -depth
    } else if t < 0.35 {
        let s = (t - 0.25) / 0.1;
        -depth + ring_h * (std::f32::consts::PI * s).sin()
    } else if t < 0.55 {
        -depth
    } else if t < 1.0 {
        let s = (t - 0.55) / 0.45;
        -depth + (depth + rim_h) * s01(s)
    } else if t <= 5.0 {
        rim_h * t.powi(-3)
    } else {
        0.0
    }
}

/// Multi-ring basin profile.
pub(crate) fn multi_ring_profile(t: f32, depth: f32, rim_h: f32) -> f32 {
    let inner_ring_h = depth * 0.15;
    let outer_ring_h = depth * 0.10;
    if t < 0.2 {
        -depth
    } else if t < 0.3 {
        let s = (t - 0.2) / 0.1;
        -depth + inner_ring_h * (std::f32::consts::PI * s).sin()
    } else if t < 0.5 {
        -depth * 0.9
    } else if t < 0.6 {
        let s = (t - 0.5) / 0.1;
        -depth * 0.9 + outer_ring_h * (std::f32::consts::PI * s).sin()
    } else if t < 1.0 {
        let s = (t - 0.6) / 0.4;
        -depth * 0.5 + (depth * 0.5 + rim_h) * s01(s)
    } else if t <= 5.0 {
        rim_h * t.powi(-3)
    } else {
        0.0
    }
}

/// Dispatch to the morphology-specific profile.
pub(crate) fn crater_profile(
    t: f32,
    depth: f32,
    rim_h: f32,
    radius_m: f32,
    morph: Morphology,
) -> f32 {
    match morph {
        Morphology::Simple    => simple_profile(t, depth, rim_h),
        Morphology::Complex   => complex_profile(t, depth, rim_h, radius_m),
        Morphology::PeakRing  => peak_ring_profile(t, depth, rim_h),
        Morphology::MultiRing => multi_ring_profile(t, depth, rim_h),
    }
}

// ---------------------------------------------------------------------------
// Diffusion degradation
// ---------------------------------------------------------------------------
//
// Topographic diffusion (Soderblom 1970, Fassett & Thomson 2014):
//   ∂h/∂t = κ∇²h,  κ ≈ 5.5 m²/Myr (lunar)
//
// Fractional depth retained ≈ exp(-C_DIFF × κt / D²)
// C_DIFF = 14.5 calibrated so a 300 m crater at 3 Ga retains ~7% of depth,
// while a 10 km crater retains ~99.8% (Fassett & Thomson 2014 Fig. 4).

const KAPPA_M2_PER_MYR: f32 = 5.5;
const C_DIFF: f32 = 14.5;
const MIN_RETENTION: f32 = 0.03;

pub(crate) fn degradation_factor(radius_m: f32, age_gyr: f32) -> f32 {
    let k = KAPPA_M2_PER_MYR * age_gyr * 1000.0;
    let d_m = radius_m * 2.0;
    (-C_DIFF * k / (d_m * d_m)).exp().max(MIN_RETENTION)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_profile_shape() {
        let depth = 1000.0;
        let rim_h = 100.0;
        assert!((simple_profile(0.0, depth, rim_h) + depth).abs() < 1.0);
        assert!(simple_profile(1.0, depth, rim_h) > 0.0);
        assert!(simple_profile(3.0, depth, rim_h) > 0.0);
        assert!(simple_profile(3.0, depth, rim_h) < rim_h);
        assert!(simple_profile(5.5, depth, rim_h).abs() < 0.01);
    }

    #[test]
    fn complex_profile_has_central_peak() {
        let depth = 3000.0;
        let rim_h = 500.0;
        let radius_m = 30_000.0;
        let center = complex_profile(0.0, depth, rim_h, radius_m);
        let floor  = complex_profile(0.2, depth, rim_h, radius_m);
        assert!(center > floor, "center={center} floor={floor}");
    }

    #[test]
    fn complex_profile_ejecta_extends_to_5r() {
        let depth = 1000.0;
        let rim_h = 500.0;
        let radius_m = 30_000.0;
        assert!(complex_profile(4.0, depth, rim_h, radius_m) > 0.0);
        assert!(complex_profile(5.5, depth, rim_h, radius_m).abs() < 0.01);
    }

    #[test]
    fn morphology_thresholds() {
        assert!(matches!(morphology_for_radius(5_000.0),   Morphology::Simple));
        assert!(matches!(morphology_for_radius(30_000.0),  Morphology::Complex));
        assert!(matches!(morphology_for_radius(100_000.0), Morphology::PeakRing));
        assert!(matches!(morphology_for_radius(200_000.0), Morphology::MultiRing));
    }

    #[test]
    fn crater_dimensions_scale() {
        let (d_small, _) = crater_dimensions(2_000.0);
        let (d_large, _) = crater_dimensions(4_000.0);
        assert!(d_large > d_small);
    }

    #[test]
    fn pike_simple_depth_ratio() {
        // Simple craters: d/D ≈ 0.196 (Pike 1977)
        let (depth, _) = crater_dimensions(10_000.0);
        let ratio = depth / (10_000.0 * 2.0);
        assert!((ratio - 0.196).abs() < 0.005, "d/D={ratio}");
    }

    #[test]
    fn degradation_small_old_crater_is_eroded() {
        // 300m diameter crater at 3 Gyr should retain ~7% depth (Fassett & Thomson 2014)
        let f = degradation_factor(150.0, 3.0);
        assert!(f < 0.12, "degradation factor={f}");
    }

    #[test]
    fn degradation_large_crater_survives() {
        // 10km diameter at 3 Gyr should retain >99% depth
        let f = degradation_factor(5_000.0, 3.0);
        assert!(f > 0.99, "degradation factor={f}");
    }
}
