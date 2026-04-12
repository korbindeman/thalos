//! Stage 1 — primordial crustal topography.
//!
//! Per spec §Stage 1: low-frequency coherent noise representing ancient
//! crustal thickness variation.  Amplitude scales inversely with surface
//! gravity (low-g bodies support taller primordial relief) and with the
//! square root of radius.  A few octaves, low total amplitude — typically
//! a few kilometres peak-to-peak for a lunar-scale body.
//!
//! Intentionally gentle: on airless rocky bodies the impact stages dominate
//! visible relief.  This pass sets a soft substrate for impacts to overprint.
//!
//! No material or maturity change.

use crate::derived::DerivedProperties;
use crate::derived::constants::G_MOON;
use crate::noise::fbm3;
use crate::seeding::{Rng, sub_seed};
use crate::surface::SurfaceState;

/// Stable stage identifier used for sub-seeding.
pub const STAGE_NAME: &str = "primordial_topography";

/// Peak-to-peak amplitude of the primordial relief on a lunar reference
/// body (g ≈ 1.625 m/s², R ≈ 1 737 km), in metres.  Raised from 3 km to
/// 4.5 km for more visible highland/lowland variation; impacts will still
/// dominate visible relief but the substrate undulates convincingly.
pub const REFERENCE_AMPLITUDE_M: f64 = 4_500.0;
/// Reference radius for the √R amplitude term, metres (Luna).
const REFERENCE_RADIUS_M: f64 = 1_737_400.0;
/// Wavelength of the lowest octave, in body radii.  The `points` passed
/// in are unit vectors, so we scale them by `1 / FUND_WAVELENGTH_IN_RADII`
/// to get the noise input coordinate.
const FUND_WAVELENGTH_IN_RADII: f64 = 1.0;
/// Raised from 4 to 6 octaves: adds medium-frequency roughness that
/// breaks up the smooth inter-crater plains.  Real highlands look
/// chaotic at all scales from aeons of bombardment.
const OCTAVES: u32 = 6;
const PERSISTENCE: f64 = 0.5;
const LACUNARITY: f64 = 2.0;

/// Compute the amplitude scaling for a body with gravity `g` and radius `r`.
///
/// `A(g, R) = A_ref · (g_moon / g) · √(R / R_moon)`
pub fn amplitude(g: f64, radius_m: f64) -> f64 {
    REFERENCE_AMPLITUDE_M * (G_MOON / g) * (radius_m / REFERENCE_RADIUS_M).sqrt()
}

/// Run the primordial-topography stage.  Additive over existing elevation
/// (Stage 0 starts at zero, so on a fresh surface this *sets* the relief;
/// on a stack it adds onto whatever came before, preserving invariants).
pub fn run(state: &mut SurfaceState, derived: &DerivedProperties, top_seed: u64) {
    let amp = amplitude(derived.surface_gravity, state.reference_radius_m);
    // One per-stage noise seed; no per-sample randomness — the spatial
    // variation comes entirely from the coordinate-dependent noise.
    let noise_seed = sub_seed(top_seed, STAGE_NAME);
    // Random rotation of the noise field so two bodies with the same top
    // seed but different sub-seeds don't line up their continents.  We use
    // a deterministic small offset per body.
    let mut rng = Rng::new(noise_seed);
    let ox = rng.range_f64(-1000.0, 1000.0);
    let oy = rng.range_f64(-1000.0, 1000.0);
    let oz = rng.range_f64(-1000.0, 1000.0);

    let inv_wavelength = 1.0 / FUND_WAVELENGTH_IN_RADII;

    for i in 0..state.len() {
        let p = state.points[i];
        let v = fbm3(
            p.x * inv_wavelength + ox,
            p.y * inv_wavelength + oy,
            p.z * inv_wavelength + oz,
            noise_seed,
            OCTAVES,
            PERSISTENCE,
            LACUNARITY,
        );
        // v roughly in [-1, 1] → ±(amp/2), i.e. peak-to-peak = amp.
        state.elevation_m[i] += v * 0.5 * amp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derived::compute;
    use crate::reference_bodies;
    use crate::sampling::fibonacci_lattice;
    use crate::surface::MaterialId;

    #[test]
    fn amplitude_is_larger_on_low_gravity_bodies() {
        // Smaller body with lower gravity → taller primordial relief
        // (inverse-g dominates at equal radius).
        let a_high = amplitude(5.0, 1e6);
        let a_low = amplitude(1.0, 1e6);
        assert!(a_low > a_high);
    }

    #[test]
    fn luna_amplitude_is_a_few_kilometres() {
        let d = compute(&reference_bodies::luna());
        let a = amplitude(d.surface_gravity, 1_737_400.0);
        assert!(a > 1_000.0 && a < 10_000.0, "Luna primordial amp {a}");
    }

    #[test]
    fn run_perturbs_elevation_without_touching_material_or_maturity() {
        let pts = fibonacci_lattice(1024);
        let d = compute(&reference_bodies::luna());
        let mut s = SurfaceState::new(pts, 1_737_400.0);
        run(&mut s, &d, 1234);
        // At least some elevation is non-zero.
        assert!(s.elevation_m.iter().any(|&e| e.abs() > 100.0));
        // All values finite.
        assert!(s.elevation_m.iter().all(|e| e.is_finite()));
        // No material writes.
        assert!(s.material.iter().all(|&m| m == MaterialId::PrimordialCrust));
        // No maturity writes.
        assert!(s.maturity.iter().all(|&m| m == 1.0));
    }

    #[test]
    fn deterministic_for_fixed_seed_and_points() {
        let pts = fibonacci_lattice(256);
        let d = compute(&reference_bodies::luna());
        let mut a = SurfaceState::new(pts.clone(), 1_737_400.0);
        let mut b = SurfaceState::new(pts, 1_737_400.0);
        run(&mut a, &d, 42);
        run(&mut b, &d, 42);
        assert_eq!(a.elevation_m, b.elevation_m);
    }

    #[test]
    fn different_top_seeds_produce_different_fields() {
        let pts = fibonacci_lattice(256);
        let d = compute(&reference_bodies::luna());
        let mut a = SurfaceState::new(pts.clone(), 1_737_400.0);
        let mut b = SurfaceState::new(pts, 1_737_400.0);
        run(&mut a, &d, 1);
        run(&mut b, &d, 2);
        assert_ne!(a.elevation_m, b.elevation_m);
    }

    #[test]
    fn peak_to_peak_is_bounded_by_amplitude() {
        let pts = fibonacci_lattice(4096);
        let d = compute(&reference_bodies::luna());
        let mut s = SurfaceState::new(pts, 1_737_400.0);
        run(&mut s, &d, 7);
        let max = s.elevation_m.iter().cloned().fold(f64::MIN, f64::max);
        let min = s.elevation_m.iter().cloned().fold(f64::MAX, f64::min);
        let p2p = max - min;
        let amp = amplitude(d.surface_gravity, 1_737_400.0);
        // Value noise is nominally in [-1, 1]; 0.5 · amp on each side gives
        // the nominal peak-to-peak.  Allow 30% slack for fbm sum reaching
        // its bound (6 octaves can overshoot more than 4).
        assert!(
            p2p <= amp * 1.3,
            "primordial p2p {p2p} exceeds amp {amp} × 1.3"
        );
    }
}
