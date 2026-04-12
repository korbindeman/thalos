//! Stage 4 — main crater population.
//!
//! Per spec §Stage 4: sample crater diameters from a power-law approximating
//! observed lunar cumulative SFD, with total count proportional to
//! `surface_area × age × impact_flux_multiplier` against a lunar-calibrated
//! reference.  Sort oldest-to-youngest and stamp in age order so younger
//! craters overprint older.
//!
//! This is the workhorse stage; output starts looking like a moon here.
//!
//! Calibration (spec §Physical calibration): lunar cumulative crater density
//! at 1 km diameter on a 4 Gyr surface is on the order of 10⁻² per km².  We
//! pick `LUNAR_DENSITY_AT_1KM_PER_KM2 = 0.01` and derive everything else by
//! scaling against that point.

use glam::DVec3;

use crate::crater::{Crater, stamp_crater};
use crate::derived::DerivedProperties;
use crate::descriptor::PlanetDescriptor;
use crate::seeding::{Rng, sub_seed};
use crate::surface::SurfaceState;

pub const STAGE_NAME: &str = "main_craters";

/// Cumulative crater density at D = 1 km on a 4 Gyr lunar surface,
/// craters per square kilometre.  Spec calibration point.
pub const LUNAR_DENSITY_AT_1KM_PER_KM2: f64 = 1e-2;
/// Reference age, gigayears, at which the calibration density holds.
pub const LUNAR_REFERENCE_AGE_GYR: f64 = 4.0;
/// Power-law cumulative exponent for the main-population SFD: `N(>D) ∝ D^-α`.
/// Lunar main range is roughly α ≈ 2; we use 2.0 for v0.1 and accept the
/// known small/large breaks as future tuning.
pub const SFD_ALPHA: f64 = 2.0;
/// Default smallest crater the main-population stage stamps, in metres.
/// Lowered from 2 km to 800 m: on a 4 Gyr surface the highlands should
/// be saturated with overlapping craters at all scales.  This increases
/// crater count ~6× but fills in the unrealistic "clean" gaps between
/// larger craters.  Callers that know their bake resolution can override
/// via [`StageConfig`] to match the texel size.
pub const DEFAULT_MIN_CRATER_DIAMETER_M: f64 = 800.0;
/// Largest crater the main-population stage stamps, in metres.  Anything
/// bigger is a giant basin (Stage 2).
pub const MAX_CRATER_DIAMETER_M: f64 = 200_000.0;

/// Per-call configuration for the main-crater stage.  Lets the caller
/// pin the smallest crater to the resolution they're going to bake at,
/// and cap the *largest* baked crater so a per-fragment shader pass can
/// own the smaller half of the population without double-stamping.
#[derive(Clone, Copy, Debug)]
pub struct StageConfig {
    pub min_diameter_m: f64,
    pub max_diameter_m: f64,
}

impl Default for StageConfig {
    fn default() -> Self {
        Self {
            min_diameter_m: DEFAULT_MIN_CRATER_DIAMETER_M,
            max_diameter_m: MAX_CRATER_DIAMETER_M,
        }
    }
}

/// Compute the total number of craters to stamp in `[d_min, d_max]` for a
/// body of given surface area, age, and flux multiplier.
///
/// Derivation: lunar cumulative `N(>D) = K · D^-α` per km² with
/// `K = LUNAR_DENSITY_AT_1KM_PER_KM2` (D in km).  The cumulative count
/// above `d_min` per km² is `K · d_min^-α`.  Multiply by surface area in
/// km², by `age / reference_age` (counts scale linearly with exposure), and
/// by `flux_multiplier`.  Subtract the count above `d_max` to get the
/// `[d_min, d_max]` count.
pub fn total_count(
    surface_area_m2: f64,
    age_gyr: f64,
    flux_multiplier: f64,
    d_min_m: f64,
    d_max_m: f64,
) -> u64 {
    let area_km2 = surface_area_m2 * 1e-6;
    let d_min_km = d_min_m * 1e-3;
    let d_max_km = d_max_m * 1e-3;
    let above_min = LUNAR_DENSITY_AT_1KM_PER_KM2 * d_min_km.powf(-SFD_ALPHA);
    let above_max = LUNAR_DENSITY_AT_1KM_PER_KM2 * d_max_km.powf(-SFD_ALPHA);
    let per_km2 = (above_min - above_max).max(0.0);
    let exposure = (age_gyr / LUNAR_REFERENCE_AGE_GYR) * flux_multiplier;
    (per_km2 * area_km2 * exposure).round() as u64
}

/// Run Stage 4.  Generates craters, sorts them oldest-to-youngest, and
/// stamps them in age order.  No-ops cleanly when the body has zero age or
/// zero impact flux (count rounds to zero).
pub fn run(
    state: &mut SurfaceState,
    desc: &PlanetDescriptor,
    derived: &DerivedProperties,
    top_seed: u64,
    config: StageConfig,
) {
    let surface_area = 4.0 * std::f64::consts::PI * desc.radius_m * desc.radius_m;
    let d_min = config.min_diameter_m;
    let d_max = config.max_diameter_m;
    if d_max <= d_min {
        return;
    }
    let count = total_count(
        surface_area,
        desc.age_gyr,
        desc.impact_flux_multiplier,
        d_min,
        d_max,
    );
    if count == 0 {
        return;
    }

    let stage_seed = sub_seed(top_seed, STAGE_NAME);
    let mut rng = Rng::new(stage_seed);

    // Sample all craters first so we can sort them by age before stamping.
    let mut craters: Vec<Crater> = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let diameter_m = rng.power_law(d_min, d_max, SFD_ALPHA);
        let center = rng.unit_vector();
        // Age sampled uniformly across the body's lifetime.  More
        // sophisticated bombardment histories (Late Heavy Bombardment
        // spike, etc.) can be added by reshaping this distribution.
        let age_gyr = rng.range_f64(0.0, desc.age_gyr);
        let perturb_seed = rng.next_u64();
        craters.push(Crater {
            center,
            diameter_m,
            age_gyr,
            perturb_seed,
        });
    }

    // Oldest first → youngest last, so younger craters overprint older.
    craters.sort_by(|a, b| b.age_gyr.partial_cmp(&a.age_gyr).unwrap());

    let d_sc = derived.simple_to_complex_transition_m;

    // Collect large craters that will generate secondaries before stamping,
    // so secondaries can be interleaved in age order with primaries.
    let secondary_threshold_m = 20_000.0; // only primaries > 20 km spawn secondaries
    let mut secondaries: Vec<Crater> = Vec::new();
    for c in &craters {
        if c.diameter_m >= secondary_threshold_m {
            generate_secondaries(c, desc.radius_m, &mut rng, &mut secondaries);
        }
    }

    // Merge secondaries into the primary list and re-sort by age so the
    // overprint order remains correct.
    craters.extend(secondaries);
    craters.sort_by(|a, b| b.age_gyr.partial_cmp(&a.age_gyr).unwrap());

    for c in craters {
        stamp_crater(state, desc.radius_m, c, d_sc, desc.age_gyr);
    }
}

/// Minimum number of secondary craters per large primary.
const SECONDARIES_PER_PRIMARY_MIN: u32 = 6;
/// Maximum number of secondary craters per large primary.
const SECONDARIES_PER_PRIMARY_MAX: u32 = 20;
/// Secondary crater diameter as a fraction of primary diameter.
const SECONDARY_DIAMETER_FRAC_MIN: f64 = 0.02;
const SECONDARY_DIAMETER_FRAC_MAX: f64 = 0.06;
/// Distance range for secondaries, in multiples of primary crater radius.
const SECONDARY_DIST_MIN: f64 = 1.8;
const SECONDARY_DIST_MAX: f64 = 5.0;
/// Number of radial rays secondaries cluster around.
const SECONDARY_RAY_COUNT: f64 = 5.0;
/// Angular spread of secondaries around each ray, radians.
const SECONDARY_RAY_SPREAD: f64 = 0.3;

/// Generate secondary craters for a single large primary impact.
/// Secondaries are placed along radial rays at 1.8–5× primary radius,
/// with diameters 2–6% of the primary.  They inherit the primary's age.
fn generate_secondaries(
    primary: &Crater,
    body_radius_m: f64,
    rng: &mut Rng,
    out: &mut Vec<Crater>,
) {
    let primary_radius = primary.diameter_m * 0.5;
    let center = primary.center.normalize();

    // Build a local basis at the primary center.
    let world_up = if center.y.abs() < 0.95 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let east = world_up.cross(center).normalize();
    let north = center.cross(east).normalize();

    // Ray directions (evenly spaced + random offset).
    let ray_offset = rng.range_f64(0.0, std::f64::consts::TAU);

    let count = rng.range_f64(
        SECONDARIES_PER_PRIMARY_MIN as f64,
        SECONDARIES_PER_PRIMARY_MAX as f64 + 1.0,
    ) as u32;

    for _ in 0..count {
        // Pick a ray and scatter around it.
        let ray_idx = rng.range_f64(0.0, SECONDARY_RAY_COUNT).floor();
        let ray_azimuth = ray_offset + ray_idx * std::f64::consts::TAU / SECONDARY_RAY_COUNT;
        let azimuth = ray_azimuth + rng.range_f64(-SECONDARY_RAY_SPREAD, SECONDARY_RAY_SPREAD);

        let dist_radii = rng.range_f64(SECONDARY_DIST_MIN, SECONDARY_DIST_MAX);
        let arc_dist = dist_radii * primary_radius;
        let theta = arc_dist / body_radius_m;

        // Direction on the sphere at (theta, azimuth) from the primary center.
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let dir = center * cos_t
            + (east * azimuth.sin() + north * azimuth.cos()) * sin_t;

        let diam_frac = rng.range_f64(SECONDARY_DIAMETER_FRAC_MIN, SECONDARY_DIAMETER_FRAC_MAX);
        let diameter_m = primary.diameter_m * diam_frac;

        out.push(Crater {
            center: dir,
            diameter_m,
            age_gyr: primary.age_gyr, // same age as parent impact
            perturb_seed: rng.next_u64(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derived::compute;
    use crate::reference_bodies;
    use crate::sampling::fibonacci_lattice;

    #[test]
    fn lunar_count_is_within_an_order_of_magnitude_of_observation() {
        // Lunar surface area ~ 3.79e7 km² → at α=2 and K=1e-2, with
        // [0.8, 200] km range and α=2, the per-km² count is
        //   K · (0.8^-2 - 200^-2) ≈ 0.01 · 1.5625 = 1.56e-2.
        // Total ≈ 5.9e5 craters on a 4 Gyr lunar surface.
        let area = 4.0 * std::f64::consts::PI * 1_737_400.0_f64.powi(2);
        let n = total_count(area, 4.0, 1.0, DEFAULT_MIN_CRATER_DIAMETER_M, MAX_CRATER_DIAMETER_M);
        // Within an order of magnitude of 5e5.
        assert!(n > 1e4 as u64 && n < 1e7 as u64, "lunar count out of range: {n}");
    }

    #[test]
    fn count_scales_with_age_and_flux() {
        let area = 1e12;
        let n_old = total_count(area, 4.0, 1.0, 2000.0, 200_000.0);
        let n_young = total_count(area, 1.0, 1.0, 2000.0, 200_000.0);
        let n_active = total_count(area, 4.0, 4.0, 2000.0, 200_000.0);
        assert!(n_old > n_young);
        assert!(n_active > n_old);
    }

    #[test]
    fn zero_age_produces_no_craters() {
        let area = 1e12;
        let n = total_count(area, 0.0, 1.0, 2000.0, 200_000.0);
        assert_eq!(n, 0);
    }

    #[test]
    fn run_is_deterministic_for_fixed_seed() {
        let pts = fibonacci_lattice(2048);
        let desc = reference_bodies::luna();
        let d = compute(&desc);
        let mut a = SurfaceState::new(pts.clone(), desc.radius_m);
        let mut b = SurfaceState::new(pts, desc.radius_m);
        run(&mut a, &desc, &d, 99, StageConfig::default());
        run(&mut b, &desc, &d, 99, StageConfig::default());
        assert_eq!(a.elevation_m, b.elevation_m);
    }

    #[test]
    fn luna_run_produces_visible_relief() {
        // 2048 lattice points is sparse — most small craters miss every
        // sample — but a 4 Gyr lunar exposure produces enough big ones
        // that some elevation must be non-zero.
        let pts = fibonacci_lattice(2048);
        let desc = reference_bodies::luna();
        let d = compute(&desc);
        let mut s = SurfaceState::new(pts, desc.radius_m);
        run(&mut s, &desc, &d, 1, StageConfig::default());
        let max = s.elevation_m.iter().cloned().fold(f64::MIN, f64::max);
        let min = s.elevation_m.iter().cloned().fold(f64::MAX, f64::min);
        assert!(max > 50.0, "no positive relief: {max}");
        assert!(min < -50.0, "no negative relief: {min}");
    }

    #[test]
    fn rhea_run_produces_visible_relief() {
        // Cold small moon: still produces craters (impacts depend only on
        // surface area, age, flux — not heat).
        let pts = fibonacci_lattice(2048);
        let desc = reference_bodies::rhea();
        let d = compute(&desc);
        let mut s = SurfaceState::new(pts, desc.radius_m);
        run(&mut s, &desc, &d, 1, StageConfig::default());
        let max = s.elevation_m.iter().cloned().fold(f64::MIN, f64::max);
        let min = s.elevation_m.iter().cloned().fold(f64::MAX, f64::min);
        assert!(max > 0.0 && min < 0.0, "Rhea no relief: [{min}, {max}]");
    }
}
