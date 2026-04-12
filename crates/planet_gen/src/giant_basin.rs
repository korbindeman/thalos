//! Stage 2 — giant basins.
//!
//! Per spec §Stage 2: distribute a small number of very large impacts
//! (typically 3–15 for a lunar-scale body, scaled by surface area and
//! impact flux multiplier).  Diameters from the high tail of the size
//! distribution.  Basin floors are tagged with [`MaterialId::BasinFloor`]
//! so the mare-flooding stage can find them.
//!
//! Runs *before* Stage 4 so smaller craters correctly overprint basin
//! rims and floors.
//!
//! Architecturally a giant basin is a very large complex/peak-ring crater,
//! so we reuse [`crate::crater::stamp_crater`] with diameters above the
//! main population's `MAX_CRATER_DIAMETER_M`.  After stamping we tag the
//! deepest fraction of each basin as [`MaterialId::BasinFloor`] in a
//! second pass — this is what mare flooding scans for.

use crate::crater::{Crater, stamp_crater};
use crate::derived::DerivedProperties;
use crate::descriptor::PlanetDescriptor;
use crate::main_craters::MAX_CRATER_DIAMETER_M;
use crate::seeding::{Rng, sub_seed};
use crate::surface::{MaterialId, SurfaceState};

pub const STAGE_NAME: &str = "giant_basins";

/// Diameter range for giant basins, metres.  Lower bound is the upper
/// bound of the main crater population (so the two SFDs don't overlap),
/// upper bound is a fraction of the body's circumference (set per-body
/// at run time as `min(MAX_BASIN_DIAMETER_M, 0.6 × 2πR)`).
pub const MIN_BASIN_DIAMETER_M: f64 = MAX_CRATER_DIAMETER_M;
pub const MAX_BASIN_DIAMETER_M: f64 = 2_500_000.0;
/// Diameter SFD exponent for the basin tail.  Slightly steeper than the
/// main population: very large basins are very rare.
pub const BASIN_SFD_ALPHA: f64 = 2.5;

/// Reference number of basins on a lunar-area surface at flux = 1.
const LUNAR_BASIN_REFERENCE_COUNT: f64 = 8.0;

/// Compute basin count for a given surface area and flux multiplier.
/// Linear in surface area against the lunar reference, multiplied by
/// `flux_multiplier`.  Capped to a sane minimum/maximum so very small
/// (Deimos) or very large bodies don't produce silly counts.
pub fn basin_count(surface_area_m2: f64, flux_multiplier: f64) -> u64 {
    const LUNAR_AREA_M2: f64 = 4.0 * std::f64::consts::PI * 1_737_400.0_f64 * 1_737_400.0_f64;
    let area_ratio = surface_area_m2 / LUNAR_AREA_M2;
    let n = (LUNAR_BASIN_REFERENCE_COUNT * area_ratio * flux_multiplier).round();
    n.clamp(0.0, 30.0) as u64
}

/// Run Stage 2.
pub fn run(
    state: &mut SurfaceState,
    desc: &PlanetDescriptor,
    derived: &DerivedProperties,
    top_seed: u64,
) {
    let surface_area = 4.0 * std::f64::consts::PI * desc.radius_m * desc.radius_m;
    let count = basin_count(surface_area, desc.impact_flux_multiplier);
    if count == 0 {
        return;
    }

    let stage_seed = sub_seed(top_seed, STAGE_NAME);
    let mut rng = Rng::new(stage_seed);

    // Cap the maximum basin diameter at a fraction of the body's
    // circumference; otherwise we'd try to stamp basins larger than the
    // body itself on a small moon.
    let circumference = std::f64::consts::TAU * desc.radius_m;
    let body_max = (0.6 * circumference).min(MAX_BASIN_DIAMETER_M);
    let body_min = MIN_BASIN_DIAMETER_M.min(body_max * 0.5);

    let d_sc = derived.simple_to_complex_transition_m;

    // Generate, sort by age, and stamp.  Same age-ordering invariant as
    // Stage 4: older first.
    let mut basins: Vec<Crater> = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let diameter_m = rng.power_law(body_min, body_max, BASIN_SFD_ALPHA);
        let center = rng.unit_vector();
        // Basins formed early in the body's history.
        let age_gyr = rng.range_f64(0.7 * desc.age_gyr, desc.age_gyr);
        let perturb_seed = rng.next_u64();
        basins.push(Crater {
            center,
            diameter_m,
            age_gyr,
            perturb_seed,
        });
    }
    basins.sort_by(|a, b| b.age_gyr.partial_cmp(&a.age_gyr).unwrap());

    for basin in &basins {
        stamp_crater(state, desc.radius_m, *basin, d_sc, desc.age_gyr);
    }

    // Second pass: mark the deepest interior of each basin as BasinFloor
    // for the mare-flooding stage.  We use a simple geometric criterion
    // (within `floor_fraction × R_basin` of the basin centre) rather than
    // a depth threshold, because the elevation field has been mutated
    // multiple times and the absolute depth depends on overlapping
    // perturbations.
    const FLOOR_FRACTION: f64 = 0.55;
    for basin in &basins {
        let floor_arc = basin.diameter_m * 0.5 * FLOOR_FRACTION;
        let floor_theta = floor_arc / desc.radius_m;
        let cos_floor = floor_theta.cos();
        let center = basin.center.normalize();
        for i in 0..state.len() {
            let cos_t = state.points[i].dot(center).clamp(-1.0, 1.0);
            if cos_t >= cos_floor {
                state.material[i] = MaterialId::BasinFloor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derived::compute;
    use crate::reference_bodies;
    use crate::sampling::fibonacci_lattice;

    #[test]
    fn lunar_basin_count_is_in_spec_range() {
        let area = 4.0 * std::f64::consts::PI * 1_737_400.0_f64.powi(2);
        let n = basin_count(area, 1.0);
        assert!(n >= 3 && n <= 15, "lunar basin count {n} not in [3, 15]");
    }

    #[test]
    fn deimos_produces_zero_or_few_basins() {
        let area = 4.0 * std::f64::consts::PI * 6_200.0_f64.powi(2);
        let n = basin_count(area, 1.0);
        assert!(n <= 1, "tiny moon basin count too high: {n}");
    }

    #[test]
    fn flux_multiplier_scales_count() {
        let area = 4.0 * std::f64::consts::PI * 1_737_400.0_f64.powi(2);
        let n1 = basin_count(area, 1.0);
        let n2 = basin_count(area, 2.0);
        assert!(n2 > n1);
    }

    #[test]
    fn run_marks_basin_floor_material() {
        let pts = fibonacci_lattice(8192);
        let desc = reference_bodies::luna();
        let d = compute(&desc);
        let mut s = SurfaceState::new(pts, desc.radius_m);
        run(&mut s, &desc, &d, 1);
        let n_floor = s
            .material
            .iter()
            .filter(|m| **m == MaterialId::BasinFloor)
            .count();
        assert!(n_floor > 0, "no basin floor samples written");
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let pts = fibonacci_lattice(2048);
        let desc = reference_bodies::luna();
        let d = compute(&desc);
        let mut a = SurfaceState::new(pts.clone(), desc.radius_m);
        let mut b = SurfaceState::new(pts, desc.radius_m);
        run(&mut a, &desc, &d, 7);
        run(&mut b, &desc, &d, 7);
        assert_eq!(a.elevation_m, b.elevation_m);
    }
}
