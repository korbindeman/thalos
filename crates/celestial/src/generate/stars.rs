//! Procedural star catalog generation.
//!
//! This generator takes the observational-astronomy shortcut: instead
//! of simulating the IMF × volume × extinction chain and hoping the
//! integral comes out right, we sample the distributions that matter
//! for what the player sees directly.
//!
//!   1. Apparent magnitude is drawn from `N(m) ∝ 10^(0.6 m)` — the
//!      classical differential star-count law that matches observed
//!      naked-eye-to-Hipparcos populations to first order.
//!   2. Spectral type is drawn from an approximate distribution
//!      fitted to the colours of stars brighter than mag ~6 in real
//!      catalogues. The distribution is weighted toward A/F/G/K
//!      (which is what dominates the *visible* sky) rather than
//!      toward M dwarfs (which dominate by raw count but are almost
//!      never bright enough to see).
//!   3. Spectral type maps to an effective temperature; temperature
//!      drives the blackbody spectrum used by the renderer.
//!   4. Distance falls out of `m - M = 5(log₁₀ d - 1)` and is used
//!      by the telescope layer for parallax later. We do not model
//!      extinction.
//!   5. Directions are uniform on the sphere. The diffuse milky-way
//!      glow belongs to the nebula layer, not to this point catalog.
//!
//! Every step is independent — swap the spectral-type distribution
//! for a real catalog import without touching any other function.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::catalog::Universe;
use crate::coords::{UnitVector3, uniform_sphere};
use crate::spectrum::Spectrum;
use crate::sources::Star;

#[derive(Debug, Clone)]
pub struct StarGenParams {
    pub seed: u64,
    pub count: usize,
    /// Stars fainter than this apparent magnitude are discarded.
    pub faint_magnitude_limit: f32,
    /// Brightest apparent magnitude the sampler may produce. The
    /// differential count law is unbounded below — we cap at this
    /// so a single mag-(-1.5) Sirius doesn't hijack the rest of the
    /// distribution. Real sky has ~20 stars brighter than mag 1, so
    /// we set the floor around there.
    pub bright_magnitude_floor: f32,
}

/// Sample an apparent magnitude from `N(m) ∝ 10^(0.6 m)` via inverse
/// transform sampling.
///
/// The CDF of the differential count on `[m_min, m_max]` is
/// `F(m) = (10^(0.6 m) - 10^(0.6 m_min)) / (10^(0.6 m_max) - 10^(0.6 m_min))`,
/// which inverts to `m = log₁₀(F · (10^(0.6 m_max) - 10^(0.6 m_min)) + 10^(0.6 m_min)) / 0.6`.
fn sample_apparent_magnitude(rng: &mut ChaCha8Rng, m_min: f32, m_max: f32) -> f32 {
    let u: f32 = rng.random_range(0.0f32..1.0);
    let a = 10f32.powf(0.6 * m_min);
    let b = 10f32.powf(0.6 * m_max);
    let x = a + (b - a) * u;
    x.log10() / 0.6
}

/// Spectral-type bucket: a letter, a temperature range, and its
/// relative prevalence among *visible-to-mag-6* stars (not among all
/// stars in the galaxy). Numbers are loose fits to tables in e.g.
/// Allen's Astrophysical Quantities; precision doesn't matter here
/// because the generator is approximate anyway and any real catalog
/// import would sidestep this function entirely.
struct SpectralBucket {
    weight: f32,
    t_lo: f32,
    t_hi: f32,
}

const VISIBLE_SPECTRAL_BUCKETS: &[SpectralBucket] = &[
    // O: vanishingly rare but very bright
    SpectralBucket { weight: 0.005, t_lo: 30_000.0, t_hi: 40_000.0 },
    // B: hot blue giants
    SpectralBucket { weight: 0.10,  t_lo: 10_000.0, t_hi: 30_000.0 },
    // A: blue-white, lots of Vega-alikes among visible stars
    SpectralBucket { weight: 0.22,  t_lo:  7_500.0, t_hi: 10_000.0 },
    // F: white
    SpectralBucket { weight: 0.17,  t_lo:  6_000.0, t_hi:  7_500.0 },
    // G: yellow (Sun)
    SpectralBucket { weight: 0.15,  t_lo:  5_200.0, t_hi:  6_000.0 },
    // K: orange
    SpectralBucket { weight: 0.20,  t_lo:  3_700.0, t_hi:  5_200.0 },
    // M: red dwarfs and red giants — only local ones make the cutoff
    SpectralBucket { weight: 0.16,  t_lo:  2_400.0, t_hi:  3_700.0 },
];

fn sample_temperature(rng: &mut ChaCha8Rng) -> f32 {
    let total: f32 = VISIBLE_SPECTRAL_BUCKETS.iter().map(|b| b.weight).sum();
    let mut pick: f32 = rng.random_range(0.0..total);
    for bucket in VISIBLE_SPECTRAL_BUCKETS {
        if pick < bucket.weight {
            // Uniform within the bucket's temperature range.
            let u: f32 = rng.random_range(0.0f32..1.0);
            return bucket.t_lo + (bucket.t_hi - bucket.t_lo) * u;
        }
        pick -= bucket.weight;
    }
    5_778.0
}

/// Rough inverse of the main-sequence mass→abs-mag fit we had before,
/// but driven from temperature for consistency with the spectral
/// buckets. Uses `M_V ≈ 4.83 - 2.5 log₁₀((T/T☉)^4 · (M/M☉)^2)`
/// collapsed with a temperature-only M∝T^0.54 relation to keep
/// everything on the main sequence.
fn temperature_to_absolute_magnitude(temperature_k: f32) -> f32 {
    const T_SUN: f32 = 5_778.0;
    const M_SUN: f32 = 4.83;
    let ratio = temperature_k / T_SUN;
    // Effective L/L☉ ≈ ratio^5.7 across the main sequence.
    let luminosity = ratio.powf(5.7);
    M_SUN - 2.5 * luminosity.log10()
}

fn sample_direction(rng: &mut ChaCha8Rng) -> UnitVector3 {
    let u1: f32 = rng.random_range(0.0..1.0);
    let u2: f32 = rng.random_range(0.0..1.0);
    uniform_sphere(u1, u2)
}

pub fn populate(universe: &mut Universe, params: &StarGenParams) {
    let mut rng = ChaCha8Rng::seed_from_u64(params.seed);

    for _ in 0..params.count {
        let apparent =
            sample_apparent_magnitude(&mut rng, params.bright_magnitude_floor, params.faint_magnitude_limit);
        let temperature_k = sample_temperature(&mut rng);
        let abs_mag = temperature_to_absolute_magnitude(temperature_k);
        // m - M = 5 log₁₀(d) - 5  →  d_pc = 10^((m-M+5)/5)
        let distance_pc = 10f32.powf((apparent - abs_mag + 5.0) / 5.0).max(1.0);

        let position = sample_direction(&mut rng);
        let spectrum = Spectrum::Blackbody {
            temperature_k,
            scale: 1.0,
        };

        universe.stars.push(Star {
            position,
            spectrum,
            apparent_magnitude: apparent,
            distance_pc,
            proper_motion: [0.0, 0.0],
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_requested_count() {
        let mut universe = Universe::new(42);
        populate(
            &mut universe,
            &StarGenParams {
                seed: 42,
                count: 5_000,
                faint_magnitude_limit: 8.5,
                bright_magnitude_floor: -1.5,
            },
        );
        assert_eq!(universe.stars.len(), 5_000);
    }

    #[test]
    fn magnitude_sampling_stays_in_range() {
        let mut universe = Universe::new(1);
        populate(
            &mut universe,
            &StarGenParams {
                seed: 1,
                count: 10_000,
                faint_magnitude_limit: 6.0,
                bright_magnitude_floor: -1.5,
            },
        );
        for star in &universe.stars {
            assert!(star.apparent_magnitude >= -1.5);
            assert!(star.apparent_magnitude <= 6.0);
        }
    }

    #[test]
    fn temperature_sampler_covers_all_buckets() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut hot = false;
        let mut cool = false;
        for _ in 0..5_000 {
            let t = sample_temperature(&mut rng);
            if t > 20_000.0 {
                hot = true;
            }
            if t < 4_000.0 {
                cool = true;
            }
        }
        assert!(hot, "expected at least one B/O sample in 5k");
        assert!(cool, "expected at least one M sample in 5k");
    }
}
