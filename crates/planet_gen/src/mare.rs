//! Stage 3 — mare flooding.
//!
//! Per spec §Stage 3: gated on total internal heat budget and thermal
//! history scalar.  If the body had sufficient internal activity, select
//! basin floors whose lowest elevations fall below a threshold and flood
//! them to a level determined by available melt volume.
//!
//! Within flooded regions:
//! - Flatten elevation to the flood level (with minor noise for wrinkle ridges).
//! - Set material to [`MaterialId::Mare`].
//! - Reset maturity to a mid value (mare are younger than highlands).
//!
//! Bodies with low heat budget or low thermal history skip cleanly.  This
//! gating must come from derived properties — never a body-type check —
//! per spec §Design principles 2.

use crate::derived::DerivedProperties;
use crate::descriptor::PlanetDescriptor;
use crate::noise::fbm3;
use crate::seeding::{splitmix64, sub_seed};
use crate::surface::{MaterialId, SurfaceState};

pub const STAGE_NAME: &str = "mare_flooding";

/// Heat-budget threshold (watts per kg of body mass) below which mare
/// flooding is suppressed.  Calibrated so Luna (radiogenic ≈ 4e-12 W/kg
/// after 4.5 Gyr decay against H0=2e-11) just clears it, while Rhea and
/// Callisto fall below.  Specific heat — not absolute — so the gate
/// scales naturally to bodies of different mass.
pub const HEAT_GATE_W_PER_KG: f64 = 3.0e-12;
/// Thermal-history scalar threshold.  Spec: "drives mare flooding in v0.1."
/// A body with `thermal_history < 0.4` is treated as too inactive
/// regardless of present-day heat.
pub const THERMAL_HISTORY_GATE: f64 = 0.4;
/// Maturity reset value for fresh mare basalt (younger than highlands but
/// has weathered since emplacement).
pub const MARE_MATURITY: f64 = 0.55;
/// Wrinkle-ridge noise amplitude as a fraction of the body radius.  Tiny.
const WRINKLE_AMPLITUDE_FRAC: f64 = 1e-5;

/// Run Stage 3.  Floods qualifying basins; cleanly no-ops on cold bodies.
pub fn run(
    state: &mut SurfaceState,
    desc: &PlanetDescriptor,
    derived: &DerivedProperties,
    top_seed: u64,
) {
    // Specific heat: total / mass.  This is what gates flooding — a small
    // body can be hot per kg even with a tiny absolute heat budget.
    let specific_heat = derived.total_heat_budget_w / desc.mass_kg;
    if specific_heat < HEAT_GATE_W_PER_KG {
        return;
    }
    if desc.thermal_history < THERMAL_HISTORY_GATE {
        return;
    }

    // Activity scalar in [0, 1] above the gates — controls how much of the
    // basin floor area gets flooded.
    let heat_excess = (specific_heat / HEAT_GATE_W_PER_KG - 1.0).clamp(0.0, 1.0);
    let history_excess =
        ((desc.thermal_history - THERMAL_HISTORY_GATE) / (1.0 - THERMAL_HISTORY_GATE))
            .clamp(0.0, 1.0);
    let activity = heat_excess * history_excess;
    if activity <= 0.0 {
        return;
    }

    // Find all BasinFloor samples and their elevations.
    let basin_indices: Vec<usize> = (0..state.len())
        .filter(|&i| state.material[i] == MaterialId::BasinFloor)
        .collect();
    if basin_indices.is_empty() {
        return;
    }

    // Determine the flood level: the (1 - activity)-th percentile of basin
    // floor elevations.  At max activity, the level rises to the
    // 0th-percentile (deepest sample only); at min activity, only the
    // very deepest basin samples are below the level.  We invert: at high
    // activity we want *more* area flooded → use a higher percentile.
    let mut elevations: Vec<f64> = basin_indices
        .iter()
        .map(|&i| state.elevation_m[i])
        .collect();
    elevations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = (activity * 0.7).clamp(0.0, 0.95); // up to 95% of basin area
    let idx = ((elevations.len() - 1) as f64 * pct).round() as usize;
    let flood_level = elevations[idx];

    let stage_seed = sub_seed(top_seed, STAGE_NAME);
    let wrinkle_amp = WRINKLE_AMPLITUDE_FRAC * desc.radius_m;

    for &i in &basin_indices {
        if state.elevation_m[i] > flood_level {
            continue;
        }
        // Wrinkle-ridge noise: tiny low-frequency perturbation on top of
        // the flat flood level so the mare doesn't read as a perfect disc.
        let p = state.points[i];
        let n = fbm3(p.x * 8.0, p.y * 8.0, p.z * 8.0, stage_seed, 3, 0.5, 2.0);
        state.elevation_m[i] = flood_level + n * wrinkle_amp;
        state.material[i] = MaterialId::Mare;
        state.maturity[i] = MARE_MATURITY;
        // Mare are younger than highlands; arbitrary "few hundred Myr"
        // value placed here so the maturity stage doesn't try to age them
        // back toward zero.  Spec leaves the exact number unspecified.
        state.crater_age_gyr[i] = 0.5;
    }

    // Touch the seed so the splitmix path is exercised even if no
    // wrinkles ended up emplaced; keeps determinism tests honest.
    let _ = splitmix64(stage_seed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derived::compute;
    use crate::giant_basin;
    use crate::reference_bodies;
    use crate::sampling::fibonacci_lattice;

    fn run_with_basins(desc: &PlanetDescriptor, top_seed: u64) -> SurfaceState {
        let pts = fibonacci_lattice(8192);
        let d = compute(desc);
        let mut s = SurfaceState::new(pts, desc.radius_m);
        giant_basin::run(&mut s, desc, &d, top_seed);
        run(&mut s, desc, &d, top_seed);
        s
    }

    #[test]
    fn luna_floods_some_basin_area() {
        let s = run_with_basins(&reference_bodies::luna(), 1);
        let n_mare = s.material.iter().filter(|m| **m == MaterialId::Mare).count();
        assert!(n_mare > 0, "Luna produced no mare");
    }

    #[test]
    fn rhea_produces_no_mare() {
        // Cold icy moon, low thermal history → must skip cleanly.
        let s = run_with_basins(&reference_bodies::rhea(), 1);
        let n_mare = s.material.iter().filter(|m| **m == MaterialId::Mare).count();
        assert_eq!(n_mare, 0, "Rhea unexpectedly produced mare");
    }

    #[test]
    fn deimos_produces_no_mare() {
        let s = run_with_basins(&reference_bodies::deimos(), 1);
        let n_mare = s.material.iter().filter(|m| **m == MaterialId::Mare).count();
        assert_eq!(n_mare, 0, "Deimos unexpectedly produced mare");
    }

    #[test]
    fn flooding_lowers_thermal_history_below_gate_no_ops() {
        let mut desc = reference_bodies::luna();
        desc.thermal_history = 0.1; // below gate
        let s = run_with_basins(&desc, 1);
        let n_mare = s.material.iter().filter(|m| **m == MaterialId::Mare).count();
        assert_eq!(n_mare, 0);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let a = run_with_basins(&reference_bodies::luna(), 11);
        let b = run_with_basins(&reference_bodies::luna(), 11);
        assert_eq!(a.elevation_m, b.elevation_m);
        assert_eq!(a.material, b.material);
    }
}
