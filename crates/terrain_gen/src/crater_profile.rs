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

const RIM_SIGMA: f32 = 0.13;
const EJECTA_MAX_T: f32 = 5.0;
/// Maximum widening of the rim ridge σ at full softness. Pohn & Offield
/// class-1 ghost craters show rims that are barely distinguishable from
/// the background — modelled here as a 4× wider Gaussian.
const RIM_SOFT_BROADEN: f32 = 3.0;

/// Narrow Gaussian ridge centered on the rim crest (t = 1), normalized
/// so its peak value is 1.0. The base width is `RIM_SIGMA`; `softness ∈
/// [0, 1]` widens the Gaussian linearly to model the diffusive rounding
/// of degraded rims (an old crater shows a low broad scarp instead of a
/// sharp ridge).
#[inline]
fn rim_ridge(t: f32, softness: f32) -> f32 {
    let sigma = RIM_SIGMA * (1.0 + softness * RIM_SOFT_BROADEN);
    let x = (t - 1.0) / sigma;
    (-x * x).exp()
}

/// McGetchin ejecta apron thickness at normalized radial distance t ≥ 1.
///
/// A Gaussian rise gate (same σ as `rim_ridge`) makes the apron start at 0
/// right at the rim crest (t = 1) and grow to its full thickness by the
/// time the rim ridge has faded — so the apron sits below the crest and
/// never produces a discontinuity.
///
/// `ejecta_scale ∈ [0, 1]` is a direct multiplier on the apron thickness.
/// Used by the bake loop to carve an uprange suppression wedge for oblique
/// impacts. Angular ray streaks are an albedo phenomenon (exposed fresh
/// material) and live in the SpaceWeather stage's albedo bake — never in
/// the height channel. `softness` widens the gate to match the broadened
/// rim ridge, keeping the apron continuous with the crest.
#[inline]
fn ejecta_apron(t: f32, radius_m: f32, ejecta_scale: f32, softness: f32) -> f32 {
    if !(1.0..=EJECTA_MAX_T).contains(&t) { return 0.0; }
    let thickness = 0.14 * radius_m.powf(0.74) * t.powi(-3);
    let gate = 1.0 - rim_ridge(t, softness);
    thickness * gate * ejecta_scale.max(0.0)
}

/// Simple (bowl-shaped) crater profile.
///
/// Zones: parabolic bowl (0 < t < 1) rising into a sharp Gaussian rim ridge
/// at t ≈ 1, plus a McGetchin ejecta apron (1 ≤ t ≤ 5). The ridge is narrow
/// and sits above the surrounding ejecta so the crest reads as a distinct
/// topographic high — essential for crisp terminator shadows.
///
/// `softness ∈ [0, 1]` widens the rim ridge so degraded simple craters
/// read as soft saucers. Bowl shape is unchanged because Simple is
/// already morphologically minimal — only the rim sharpness varies.
pub(crate) fn simple_profile(
    t: f32,
    depth: f32,
    rim_h: f32,
    radius_m: f32,
    ejecta_scale: f32,
    softness: f32,
) -> f32 {
    let bowl = if t < 1.0 { -depth * (1.0 - t * t) } else { 0.0 };
    let ridge = rim_h * rim_ridge(t, softness);
    bowl + ridge + ejecta_apron(t, radius_m, ejecta_scale, softness)
}

/// Number of terraces in a complex crater wall. Real complex craters
/// typically show 2-4 stepped benches from gravitational slumping.
const TERRACE_COUNT: f32 = 3.0;
/// Fraction of each step's radial width occupied by the riser (steep scarp
/// between benches); the rest is the flat terrace top.
const TERRACE_RISER_FRAC: f32 = 0.35;

/// Map a smooth 0..1 wall ramp to a stepped terraced ramp that still
/// reaches 1.0 at s=1. The benches are flat tops; the risers are smoothed
/// ramps between them. Output range: [0, 1].
///
/// `phase` shifts the step quantization by up to ~0.5 bench widths. A
/// parabolic taper forces the effective phase to zero at s=0 and s=1 so
/// the floor and rim boundary conditions stay exact. Used by the bake
/// loop to jitter terrace positions per angle → breaks the concentric
/// cookie-cutter look of complex craters.
#[inline]
fn terraced_ramp(s: f32, phase: f32) -> f32 {
    let s = s.clamp(0.0, 1.0);
    let taper = 4.0 * s * (1.0 - s);
    let eff_phase = phase * taper;
    let scaled = s * TERRACE_COUNT;
    let step = (scaled - eff_phase).floor();
    let local = scaled - eff_phase - step;
    let riser_start = 1.0 - TERRACE_RISER_FRAC;
    let riser_progress = if local <= riser_start {
        0.0
    } else {
        (local - riser_start) / TERRACE_RISER_FRAC
    };
    ((step + s01(riser_progress)) / TERRACE_COUNT).clamp(0.0, 1.0)
}

/// Maximum fractional perturbation of central peak shape by angular noise.
/// 0.35 lets the peak bulge or recede by ±35% at different azimuths, which
/// reliably breaks the cookie-cutter "smooth paraboloid" look without ever
/// inverting the profile into a hole.
const PEAK_WARP_EPS: f32 = 0.35;

/// Number of off-center sub-peaks that ride on top of the main central
/// peak. Real complex craters show clustered massifs (e.g. Tycho's
/// central peak is a multi-summit ridge, not a smooth cone). Each sub-
/// peak is a small Gaussian bump at a hashed offset; together they
/// break the cookie-cutter symmetry without adding much cost.
const SUB_PEAK_COUNT: u32 = 3;

/// Sub-peak placement: angular offset and radial fraction of the parent
/// peak's footprint. Hashed from the crater seed by the bake loop and
/// passed in as `peak_subs`. The amplitude term is a fraction of the
/// main peak height.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SubPeak {
    /// Tangent-plane azimuth of the sub-peak (radians).
    pub az: f32,
    /// Radial offset from crater center as a fraction of `peak_t`.
    /// Values in [0.2, 0.95] keep sub-peaks inside the main peak base.
    pub r_frac: f32,
    /// Sub-peak height as a fraction of the main `h_peak`.
    pub amp: f32,
    /// Sub-peak width (Gaussian σ) as a fraction of `peak_t`.
    pub sigma_frac: f32,
}

/// Bundle of `SubPeak` data for a Complex crater. Fixed-size to avoid
/// allocations in the bake hot path.
pub(crate) type SubPeaks = [SubPeak; SUB_PEAK_COUNT as usize];

/// Complex crater profile (flat floor, central peak, terraced wall).
///
/// Additive decomposition mirroring `simple_profile`:
///   h(t) = base(t) + rim_h × rim_ridge(t) + ejecta_apron(t, R)
///
/// The `base` term carries the floor, central peak, and terraced wall for
/// t < 1 and returns 0 for t ≥ 1. The ridge and apron handle the crest and
/// exterior, so the total profile is continuous at t = 1 by construction.
///
/// `peak_noise ∈ [-1, 1]` is a per-texel angular noise sample used to warp
/// the central peak's effective size so it reads as a rugged mountain
/// instead of a smooth symmetric cone. `peak_subs` adds 0..N hashed sub-
/// peak bumps for additional ruggedness; pass `&Default::default()` for a
/// smooth peak.
///
/// `softness ∈ [0, 1]` lerps the morphology toward a soft Hermite saucer
/// (Pohn class 1): the central peak shrinks, the flat floor narrows
/// toward zero radius, terraces fade into a smooth wall ramp, and the
/// rim ridge widens. At full softness only a smooth bowl remains.
#[allow(clippy::too_many_arguments)]
pub(crate) fn complex_profile(
    t: f32,
    depth: f32,
    rim_h: f32,
    radius_m: f32,
    wall_phase: f32,
    peak_noise: f32,
    peak_azimuth: f32,
    peak_subs: &SubPeaks,
    ejecta_scale: f32,
    softness: f32,
) -> f32 {
    let d_km = radius_m * 2.0 / 1000.0;

    // Krüger et al. (2018): h_peak = 0.0049 × D^1.297 km (D in km)
    let h_peak_pristine = 4.9 * d_km.powf(1.297); // meters above the floor
    // Peak fades linearly with softness — fully degraded craters have no
    // visible central peak.
    let h_peak = h_peak_pristine * (1.0 - softness);

    // Pike (1977): D_floor = 0.187 × D^1.249 → t_floor = 0.187 × D^0.249
    let t_floor_pristine = (0.187 * d_km.powf(0.249)).clamp(0.30, 0.75);
    // Floor narrows toward zero so the wall ramp covers the whole interior
    // at full softness, producing a smooth saucer.
    let t_floor = t_floor_pristine * (1.0 - softness).max(1e-3);

    // t_peak = 0.168 (Krüger et al.)
    let t_peak_base = 0.168_f32 * (1.0 - softness * 0.5);

    // Angular warp on the peak: effective peak radius grows/shrinks with
    // azimuth, giving the central mountain an irregular footprint. A
    // positive noise shrinks s (peak extends further out); negative pushes
    // the interior of the peak down earlier. The warp amplitude also
    // modulates h_peak so shrunken lobes aren't unnaturally tall.
    let peak_warp = 1.0 + PEAK_WARP_EPS * peak_noise;
    let peak_t = (t_peak_base * peak_warp).max(1e-4);
    let peak_h = h_peak * (0.75 + 0.25 * peak_warp);

    let base = if t < peak_t {
        let s = t / peak_t;
        let main_peak = peak_h * (1.0 - s * s);
        // Sub-peak rubble: each Gaussian bump at hashed offset (az, r_frac).
        // Distance is computed in the local tangent frame using `peak_azimuth`
        // as the texel's azimuth around crater center.
        let mut sub_h = 0.0_f32;
        if peak_h > 0.0 {
            for sp in peak_subs.iter() {
                if sp.amp <= 0.0 { continue; }
                // Position of this sub-peak in (r/peak_t, azimuth) polar
                // coordinates → squared chord distance to the texel.
                let r_a = s;
                let r_b = sp.r_frac;
                let mut d_az = (peak_azimuth - sp.az).abs();
                let pi = std::f32::consts::PI;
                let tau = std::f32::consts::TAU;
                if d_az > pi { d_az = tau - d_az; }
                let dist2 =
                    r_a * r_a + r_b * r_b - 2.0 * r_a * r_b * d_az.cos();
                let sigma = sp.sigma_frac.max(0.05);
                let g = (-dist2 / (2.0 * sigma * sigma)).exp();
                sub_h += peak_h * sp.amp * g;
            }
        }
        -depth + main_peak + sub_h
    } else if t < t_floor {
        -depth
    } else if t < 1.0 {
        // Wall ramp: -depth at t_floor → 0 at t = 1. The terraced ramp
        // is lerped toward a smooth Hermite ramp by `softness` so degraded
        // craters lose their stepped wall benches.
        let s = (t - t_floor) / (1.0 - t_floor);
        let stepped = terraced_ramp(s, wall_phase);
        let smooth = s * s * (3.0 - 2.0 * s);
        let ramp = stepped * (1.0 - softness) + smooth * softness;
        -depth * (1.0 - ramp)
    } else {
        0.0
    };

    base + rim_h * rim_ridge(t, softness) + ejecta_apron(t, radius_m, ejecta_scale, softness)
}

/// Peak-ring basin profile. Additive decomposition: interior base (with a
/// peak ring replacing the central peak) + rim ridge + ejecta apron.
///
/// Ring sits at t ≈ 0.5 — empirical ring-to-rim diameter ratio ≈ 0.47 for
/// real peak-ring basins (Schrödinger: 150/320; Pike & Spudis 1987). The
/// ring is a narrow sinusoidal bump between t_floor and t_wall.
pub(crate) fn peak_ring_profile(
    t: f32,
    depth: f32,
    rim_h: f32,
    radius_m: f32,
    wall_phase: f32,
    ejecta_scale: f32,
    softness: f32,
) -> f32 {
    // Ring height fades with softness; large basins relax over Gyr but
    // the floor itself stays roughly flat.
    let ring_h = depth * 0.2 * (1.0 - softness);
    const T_RING_LO: f32 = 0.45;
    const T_RING_HI: f32 = 0.55;
    let base = if t < T_RING_LO {
        -depth
    } else if t < T_RING_HI {
        let s = (t - T_RING_LO) / (T_RING_HI - T_RING_LO);
        -depth + ring_h * (std::f32::consts::PI * s).sin()
    } else if t < 1.0 {
        let s = (t - T_RING_HI) / (1.0 - T_RING_HI);
        let stepped = terraced_ramp(s, wall_phase);
        let smooth = s * s * (3.0 - 2.0 * s);
        let ramp = stepped * (1.0 - softness) + smooth * softness;
        -depth * (1.0 - ramp)
    } else {
        0.0
    };
    base + rim_h * rim_ridge(t, softness) + ejecta_apron(t, radius_m, ejecta_scale, softness)
}

/// Multi-ring basin profile. Concentric interior rings at √2 diameter
/// spacing (Pike & Spudis 1987): with the rim at t=1, rings fall at
/// t = 1/√2 ≈ 0.707 and t = 0.5. Modelled as two Gaussian bumps on a
/// smooth shallow bowl — multi-ring basins are heavily isostatically
/// relaxed and do not show terraced walls.
pub(crate) fn multi_ring_profile(
    t: f32,
    depth: f32,
    rim_h: f32,
    radius_m: f32,
    wall_phase: f32,
    ejecta_scale: f32,
    softness: f32,
) -> f32 {
    let _ = wall_phase;
    const T_INNER: f32 = 0.5;
    const T_OUTER: f32 = std::f32::consts::FRAC_1_SQRT_2; // ≈ 0.707
    const RING_SIGMA: f32 = 0.05;
    let ring_amp = 1.0 - softness;
    let inner_ring_h = depth * 0.15 * ring_amp;
    let outer_ring_h = depth * 0.10 * ring_amp;
    let base = if t < 1.0 {
        let bowl = -depth * (1.0 - t).clamp(0.0, 1.0);
        let g_inner = (-((t - T_INNER) / RING_SIGMA).powi(2)).exp();
        let g_outer = (-((t - T_OUTER) / RING_SIGMA).powi(2)).exp();
        bowl + inner_ring_h * g_inner + outer_ring_h * g_outer
    } else {
        0.0
    };
    base + rim_h * rim_ridge(t, softness) + ejecta_apron(t, radius_m, ejecta_scale, softness)
}

/// Dispatch to the morphology-specific profile.
///
/// `wall_phase` jitters terrace step boundaries for complex and larger
/// morphologies (in [-1, 1]; pass 0.0 for unjittered). `peak_noise` warps
/// the central peak shape for Complex craters (same range). `peak_azimuth`
/// is the texel azimuth around the crater center used to position the
/// hashed sub-peak rubble bumps in `peak_subs` (Complex only).
///
/// `ejecta_scale ∈ [0, 1]` is a direct multiplier on the radial ejecta
/// apron thickness. Pass 1.0 for the full uniform blanket; lower values
/// are used by the bake loop to carve an uprange suppression wedge for
/// oblique impacts. `wall_phase`, `peak_noise`, `peak_azimuth`, and
/// `peak_subs` are ignored by Simple.
///
/// `softness ∈ [0, 1]` softens crater morphology toward a bowl saucer
/// to model diffusive degradation (Pohn & Offield class progression).
/// Pass `degradation_softness(radius_m, age_gyr)` from the bake loop;
/// pass 0.0 for the pristine shape.
#[allow(clippy::too_many_arguments)]
pub(crate) fn crater_profile(
    t: f32,
    depth: f32,
    rim_h: f32,
    radius_m: f32,
    morph: Morphology,
    wall_phase: f32,
    peak_noise: f32,
    peak_azimuth: f32,
    peak_subs: &SubPeaks,
    ejecta_scale: f32,
    softness: f32,
) -> f32 {
    match morph {
        Morphology::Simple    => simple_profile(t, depth, rim_h, radius_m, ejecta_scale, softness),
        Morphology::Complex   => complex_profile(
            t, depth, rim_h, radius_m, wall_phase, peak_noise,
            peak_azimuth, peak_subs, ejecta_scale, softness,
        ),
        Morphology::PeakRing  => peak_ring_profile(t, depth, rim_h, radius_m, wall_phase, ejecta_scale, softness),
        Morphology::MultiRing => multi_ring_profile(t, depth, rim_h, radius_m, wall_phase, ejecta_scale, softness),
    }
}

// ---------------------------------------------------------------------------
// Degradation: diffusion + viscous relaxation
// ---------------------------------------------------------------------------
//
// Two processes erode crater topography over time, acting on opposite ends
// of the size spectrum:
//
// 1. Topographic diffusion (Soderblom 1970, Fassett & Thomson 2014):
//      ∂h/∂t = κ∇²h,  κ ≈ 5.5 m²/Myr (lunar)
//      Fractional depth retained ≈ exp(-C_DIFF × κt / D²)
//    The D² in the denominator makes this dominate small craters (a 300 m
//    crater at 3 Gyr retains ~7%) while leaving large craters untouched.
//
// 2. Viscous / isostatic relaxation:
//      Large impact basins flow under their own weight over Gyr timescales,
//      lifting the floor and lowering the rim. Scales positively with size
//      (bigger loads relax faster). This is what produces lunar "ghost
//      basins" — ancient 50–200 km craters that read as faint circular
//      discolorations with smaller craters scattered inside.
//      Model: retained ≈ exp(-((D - D_relax_threshold) / D_relax_ref) × age/τ)
//      Below the threshold, no relaxation. Above it, linear in excess
//      diameter × age.
//
// C_DIFF = 14.5 calibrated so a 300 m crater at 3 Ga retains ~7% of depth,
// while a 10 km crater retains ~99.8% (Fassett & Thomson 2014 Fig. 4).
// D_RELAX_THRESHOLD = 30 km (simple→complex transition, where relaxation
// starts mattering). D_RELAX_REF = 100 km, TAU = 3 Gyr → a 100 km crater
// at 3 Gyr retains ~50% of its depth, a 200 km crater retains ~18%.

const KAPPA_M2_PER_MYR: f32 = 5.5;
const C_DIFF: f32 = 14.5;
const MIN_RETENTION: f32 = 0.03;

const D_RELAX_THRESHOLD_M: f32 = 30_000.0;
const D_RELAX_REF_M: f32 = 100_000.0;
const RELAX_TAU_GYR: f32 = 3.0;

/// Morphology softening factor ∈ [0, 1]. Independent of `degradation_factor`:
///
/// - `degradation_factor` shrinks crater amplitude (depth, rim height).
/// - `degradation_softness` reshapes the morphology — wider rim, narrower
///   floor, smaller central peak, smoother walls — toward a soft saucer.
///
/// Same diffusion kinetics (`κ ≈ 5.5 m²/Myr`) but with a separate constant
/// chosen so visible softening kicks in well before total depth erasure.
/// At default tuning, a 5 km crater at 3 Gyr is ~70% softened (clearly
/// degraded) while a 30 km crater at the same age is barely affected
/// (~3%) — matches the observed Pohn & Offield class spread.
pub(crate) fn degradation_softness(radius_m: f32, age_gyr: f32) -> f32 {
    let d_m = radius_m * 2.0;
    let k = KAPPA_M2_PER_MYR * age_gyr * 1000.0;
    const C_SOFT: f32 = 7270.0;
    (1.0 - (-C_SOFT * k / (d_m * d_m)).exp()).clamp(0.0, 1.0)
}

pub(crate) fn degradation_factor(radius_m: f32, age_gyr: f32) -> f32 {
    let d_m = radius_m * 2.0;

    let k = KAPPA_M2_PER_MYR * age_gyr * 1000.0;
    let diffusion = (-C_DIFF * k / (d_m * d_m)).exp();

    let relaxation = if d_m <= D_RELAX_THRESHOLD_M {
        1.0
    } else {
        let excess = (d_m - D_RELAX_THRESHOLD_M) / D_RELAX_REF_M;
        (-excess * (age_gyr / RELAX_TAU_GYR)).exp()
    };

    (diffusion * relaxation).max(MIN_RETENTION)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_SUBS: SubPeaks = [SubPeak { az: 0.0, r_frac: 0.0, amp: 0.0, sigma_frac: 0.5 }; 3];

    #[test]
    fn simple_profile_shape() {
        let depth = 1000.0;
        let rim_h = 100.0;
        let radius_m = 5_000.0;
        // Center ≈ -depth
        assert!((simple_profile(0.0, depth, rim_h, radius_m, 1.0, 0.0) + depth).abs() < 1.0);
        // Rim crest peaks at ≈ rim_h
        let crest = simple_profile(1.0, depth, rim_h, radius_m, 1.0, 0.0);
        assert!((crest - rim_h).abs() < 1.0, "crest={crest}");
        // Rim ridge stands above the exterior apron just outside
        let outside = simple_profile(1.15, depth, rim_h, radius_m, 1.0, 0.0);
        assert!(outside < crest, "outside={outside} crest={crest}");
        // Apron fades by 5R
        assert!(simple_profile(5.5, depth, rim_h, radius_m, 1.0, 0.0).abs() < 0.1);
    }

    #[test]
    fn simple_profile_rim_protrudes_above_surroundings() {
        // For a 5 km crater the rim must be higher than both inside and
        // outside samples at ±0.2 rim units.
        let (depth, rim_h) = crater_dimensions(5_000.0);
        let crest = simple_profile(1.0, depth, rim_h, 5_000.0, 1.0, 0.0);
        let inside = simple_profile(0.8, depth, rim_h, 5_000.0, 1.0, 0.0);
        let outside = simple_profile(1.2, depth, rim_h, 5_000.0, 1.0, 0.0);
        assert!(crest > inside, "crest={crest} inside={inside}");
        assert!(crest > outside, "crest={crest} outside={outside}");
    }

    #[test]
    fn complex_profile_wall_is_terraced() {
        // Terraced ramp should produce non-monotone second derivative —
        // detectable by sampling three points on the wall and checking
        // that the middle sample doesn't sit on the secant.
        let (depth, rim_h) = crater_dimensions(40_000.0);
        let radius_m = 40_000.0;
        let t_floor = 0.55_f32;
        let t0 = t_floor + 0.05;
        let t1 = t_floor + 0.20;
        let t2 = t_floor + 0.35;
        let h0 = complex_profile(t0, depth, rim_h, radius_m, 0.0, 0.0, 0.0, &NO_SUBS, 1.0, 0.0);
        let h1 = complex_profile(t1, depth, rim_h, radius_m, 0.0, 0.0, 0.0, &NO_SUBS, 1.0, 0.0);
        let h2 = complex_profile(t2, depth, rim_h, radius_m, 0.0, 0.0, 0.0, &NO_SUBS, 1.0, 0.0);
        let secant = h0 + (h2 - h0) * ((t1 - t0) / (t2 - t0));
        assert!((h1 - secant).abs() > 1.0, "wall looks smooth: {h0} {h1} {h2}");
    }

    #[test]
    fn complex_profile_has_central_peak() {
        let depth = 3000.0;
        let rim_h = 500.0;
        let radius_m = 30_000.0;
        let center = complex_profile(0.0, depth, rim_h, radius_m, 0.0, 0.0, 0.0, &NO_SUBS, 1.0, 0.0);
        let floor  = complex_profile(0.2, depth, rim_h, radius_m, 0.0, 0.0, 0.0, &NO_SUBS, 1.0, 0.0);
        assert!(center > floor, "center={center} floor={floor}");
    }

    #[test]
    fn complex_profile_ejecta_extends_to_5r() {
        let depth = 1000.0;
        let rim_h = 500.0;
        let radius_m = 30_000.0;
        assert!(complex_profile(4.0, depth, rim_h, radius_m, 0.0, 0.0, 0.0, &NO_SUBS, 1.0, 0.0) > 0.0);
        assert!(complex_profile(5.5, depth, rim_h, radius_m, 0.0, 0.0, 0.0, &NO_SUBS, 1.0, 0.0).abs() < 0.01);
    }

    #[test]
    fn complex_softening_removes_peak() {
        // At full softness the central peak must be gone — degraded
        // complex craters read as smooth bowls.
        let (depth, rim_h) = crater_dimensions(40_000.0);
        let radius_m = 40_000.0;
        let pristine_center = complex_profile(0.0, depth, rim_h, radius_m, 0.0, 0.0, 0.0, &NO_SUBS, 1.0, 0.0);
        let soft_center     = complex_profile(0.0, depth, rim_h, radius_m, 0.0, 0.0, 0.0, &NO_SUBS, 1.0, 1.0);
        // Pristine: floor + central peak. Soft: just floor.
        assert!(pristine_center > soft_center,
            "pristine={pristine_center} soft={soft_center}");
        // Soft-center should be at -depth (no peak above floor).
        assert!((soft_center + depth).abs() < depth * 0.05,
            "soft center should be ≈ -depth, got {soft_center}");
    }

    #[test]
    fn complex_softening_widens_rim_ridge() {
        // A degraded rim should be wider but lower at offsets from t=1.
        let (_, rim_h) = crater_dimensions(20_000.0);
        let r = 20_000.0;
        let pristine_off = complex_profile(1.30, 0.0, rim_h, r, 0.0, 0.0, 0.0, &NO_SUBS, 0.0, 0.0);
        let soft_off     = complex_profile(1.30, 0.0, rim_h, r, 0.0, 0.0, 0.0, &NO_SUBS, 0.0, 1.0);
        assert!(soft_off > pristine_off,
            "softened rim should still contribute at t=1.30: pristine={pristine_off} soft={soft_off}");
    }

    #[test]
    fn peak_ring_basin_has_ring_near_half_radius() {
        // Peak-ring basins: interior ring sits at t ≈ 0.5 (Schrödinger
        // analogue; Pike & Spudis 1987). The ring must lie higher than
        // the surrounding floor at t=0.3 and t=0.65.
        let depth = 4000.0;
        let rim_h = 800.0;
        let radius_m = 100_000.0;
        let floor_in  = peak_ring_profile(0.30, depth, rim_h, radius_m, 0.0, 0.0, 0.0);
        let ring      = peak_ring_profile(0.50, depth, rim_h, radius_m, 0.0, 0.0, 0.0);
        let floor_out = peak_ring_profile(0.65, depth, rim_h, radius_m, 0.0, 0.0, 0.0);
        assert!(ring > floor_in,  "ring={ring} floor_in={floor_in}");
        assert!(ring > floor_out, "ring={ring} floor_out={floor_out}");
    }

    #[test]
    fn multi_ring_basin_rings_at_sqrt2_spacing() {
        // Two interior rings at t = 0.5 and t = 1/√2 ≈ 0.707 must both
        // sit above the valley at t = 0.6 between them.
        let depth = 6000.0;
        let rim_h = 1000.0;
        let radius_m = 200_000.0;
        let inner  = multi_ring_profile(0.50,  depth, rim_h, radius_m, 0.0, 0.0, 0.0);
        let valley = multi_ring_profile(0.60,  depth, rim_h, radius_m, 0.0, 0.0, 0.0);
        let outer  = multi_ring_profile(0.707, depth, rim_h, radius_m, 0.0, 0.0, 0.0);
        assert!(inner > valley, "inner={inner} valley={valley}");
        assert!(outer > valley, "outer={outer} valley={valley}");
    }

    #[test]
    fn degradation_softness_small_old_crater_is_softened() {
        // 5 km crater at 3 Gyr → ~70% softened (Pohn class 3-4 ish).
        let s = degradation_softness(2_500.0, 3.0);
        assert!(s > 0.6, "softness={s}");
    }

    #[test]
    fn degradation_softness_large_crater_resists() {
        // 30 km crater at 3 Gyr → barely softened.
        let s = degradation_softness(15_000.0, 3.0);
        assert!(s < 0.2, "softness={s}");
    }

    #[test]
    fn terraced_ramp_anchors_at_endpoints() {
        // Boundary conditions must hold for any phase to keep floor/rim
        // heights exact — relied on by complex_profile et al.
        for &phase in &[-0.5_f32, -0.25, 0.0, 0.25, 0.5] {
            assert!(terraced_ramp(0.0, phase).abs() < 1e-5, "phase={phase}");
            assert!((terraced_ramp(1.0, phase) - 1.0).abs() < 1e-5, "phase={phase}");
        }
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
