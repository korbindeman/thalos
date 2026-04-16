//! Procedural galaxy catalog generation.
//!
//! Mirrors the observational shortcut the star generator uses:
//! sample apparent magnitudes from a galaxy count law, assign a
//! morphological type (elliptical vs spiral) from a simple mix,
//! draw an angular size from a log-normal distribution, then fill
//! in the Sérsic parameters the renderer consumes. No clustering
//! or large-scale structure yet — galaxies are uniform on the
//! sphere for now.
//!
//! Counts and size priors are loose fits to Sloan Digital Sky
//! Survey photometry; they're good enough to produce a plausible
//! backdrop of resolvable galaxies without a real catalog. Real
//! data (NGC / PGC / SDSS) drops in later behind the same struct.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::catalog::Universe;
use crate::coords::uniform_sphere;
use crate::sources::Galaxy;
use crate::spectrum::Spectrum;

#[derive(Debug, Clone)]
pub struct GalaxyGenParams {
    pub seed: u64,
    pub count: usize,
    /// Faintest galaxy apparent magnitude to sample.
    pub faint_magnitude_limit: f32,
    /// Brightest apparent magnitude (a few bright local-group analogs
    /// are visible to the naked eye around mag 3–4).
    pub bright_magnitude_floor: f32,
}

/// Galaxy differential count law is steeper than the stellar one —
/// roughly `N(m) ∝ 10^(0.4 m)` in the local universe before the
/// survey flattens around z ~ 0.5. We keep the constant to 0.4 for
/// simplicity.
fn sample_apparent_magnitude(rng: &mut ChaCha8Rng, m_min: f32, m_max: f32) -> f32 {
    let u: f32 = rng.random_range(0.0f32..1.0);
    let a = 10f32.powf(0.4 * m_min);
    let b = 10f32.powf(0.4 * m_max);
    let x = a + (b - a) * u;
    x.log10() / 0.4
}

/// Log-normal angular radius in arcseconds, loosely fit to SDSS
/// half-light radii for galaxies at the survey's magnitude range.
fn sample_effective_radius_arcsec(rng: &mut ChaCha8Rng, apparent_magnitude: f32) -> f32 {
    // Brighter galaxies tend to be larger on-sky (they're nearby).
    // Rough inverse trend: log10(r_eff) ≈ 1.6 - 0.15 * m, then add
    // a Gaussian-ish jitter via two uniform draws.
    let mean_log = 1.6 - 0.15 * apparent_magnitude;
    let jitter: f32 = rng.random_range(-0.35f32..0.35);
    let log_r = mean_log + jitter;
    10f32.powf(log_r).clamp(0.4, 3_600.0)
}

enum Morphology {
    Elliptical,
    Spiral,
}

fn sample_morphology(rng: &mut ChaCha8Rng) -> Morphology {
    // ~35% elliptical, 65% spiral among visible galaxies is close
    // enough to SDSS morphological classifications for our purposes.
    if rng.random::<f32>() < 0.35 {
        Morphology::Elliptical
    } else {
        Morphology::Spiral
    }
}

/// Convert morphology to (sersic_n, temperature) — temperature is a
/// stand-in for the dominant stellar population colour: ellipticals
/// are old and warm, spirals are a blue-ish mix dominated by their
/// young-arm populations.
fn morphology_to_sersic_and_temperature(morph: &Morphology, rng: &mut ChaCha8Rng) -> (f32, f32) {
    match morph {
        Morphology::Elliptical => {
            let n: f32 = rng.random_range(3.0f32..4.5);
            let t: f32 = rng.random_range(3_800.0f32..4_800.0);
            (n, t)
        }
        Morphology::Spiral => {
            let n: f32 = rng.random_range(0.8f32..1.6);
            let t: f32 = rng.random_range(5_500.0f32..7_500.0);
            (n, t)
        }
    }
}

/// Hand-tuned featured galaxies injected before the random catalog.
/// These guarantee a few clearly resolvable extragalactic objects at
/// Andromeda/Triangulum scale so the sky isn't just a wash of
/// sub-pixel blobs. Positions and PAs are seed-derived so the layout
/// still changes between universes.
struct Featured {
    apparent_magnitude: f32,
    effective_radius_arcmin: f32,
    sersic_n: f32,
    axis_ratio: f32,
    temperature_k: f32,
}

const FEATURED: &[Featured] = &[
    // Big face-ish spiral (Andromeda-scale). Larger than real
    // Andromeda so the procedural structure reads clearly rather
    // than collapsing to a few pixels.
    Featured {
        apparent_magnitude: 3.0,
        effective_radius_arcmin: 110.0,
        sersic_n: 1.1,
        axis_ratio: 0.60,
        temperature_k: 6_200.0,
    },
    // Medium elliptical, round and warm.
    Featured {
        apparent_magnitude: 4.0,
        effective_radius_arcmin: 22.0,
        sersic_n: 4.0,
        axis_ratio: 0.82,
        temperature_k: 4_200.0,
    },
    // Edge-on spiral, long and thin.
    Featured {
        apparent_magnitude: 4.4,
        effective_radius_arcmin: 50.0,
        sersic_n: 1.0,
        axis_ratio: 0.22,
        temperature_k: 6_000.0,
    },
];

fn inject_featured(universe: &mut Universe, rng: &mut ChaCha8Rng) {
    for f in FEATURED {
        let u1: f32 = rng.random_range(0.0f32..1.0);
        let u2: f32 = rng.random_range(0.0f32..1.0);
        let position = crate::coords::uniform_sphere(u1, u2);
        let position_angle_rad: f32 = rng.random_range(0.0f32..std::f32::consts::PI);
        let effective_radius_rad =
            f.effective_radius_arcmin * std::f32::consts::PI / (180.0 * 60.0);
        universe.galaxies.push(Galaxy {
            position,
            spectrum: Spectrum::Blackbody {
                temperature_k: f.temperature_k,
                scale: 1.0,
            },
            apparent_magnitude: f.apparent_magnitude,
            effective_radius_rad,
            sersic_n: f.sersic_n,
            axis_ratio: f.axis_ratio,
            position_angle_rad,
            redshift: 0.0,
        });
    }
}

pub fn populate(universe: &mut Universe, params: &GalaxyGenParams) {
    let mut rng = ChaCha8Rng::seed_from_u64(params.seed);

    inject_featured(universe, &mut rng);

    for _ in 0..params.count {
        let apparent = sample_apparent_magnitude(
            &mut rng,
            params.bright_magnitude_floor,
            params.faint_magnitude_limit,
        );
        let morph = sample_morphology(&mut rng);
        let (sersic_n, temperature_k) = morphology_to_sersic_and_temperature(&morph, &mut rng);

        let r_arcsec = sample_effective_radius_arcsec(&mut rng, apparent);
        let effective_radius_rad = r_arcsec * std::f32::consts::PI / (180.0 * 3600.0);

        let axis_ratio: f32 = match morph {
            Morphology::Elliptical => rng.random_range(0.55f32..1.0),
            // Spirals span wider axis ratios because we see them at
            // random inclinations from face-on (1.0) to edge-on (~0.15).
            Morphology::Spiral => rng.random_range(0.18f32..1.0),
        };
        let position_angle_rad: f32 = rng.random_range(0.0f32..std::f32::consts::PI);

        let u1: f32 = rng.random_range(0.0f32..1.0);
        let u2: f32 = rng.random_range(0.0f32..1.0);
        let position = uniform_sphere(u1, u2);

        universe.galaxies.push(Galaxy {
            position,
            spectrum: Spectrum::Blackbody { temperature_k, scale: 1.0 },
            apparent_magnitude: apparent,
            effective_radius_rad,
            sersic_n,
            axis_ratio,
            position_angle_rad,
            redshift: 0.0,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_requested_count() {
        let mut u = Universe::new(0);
        populate(
            &mut u,
            &GalaxyGenParams {
                seed: 0,
                count: 500,
                faint_magnitude_limit: 12.0,
                bright_magnitude_floor: 4.0,
            },
        );
        assert_eq!(u.galaxies.len(), 500 + FEATURED.len());
    }

    #[test]
    fn sersic_buckets_in_range() {
        let mut u = Universe::new(1);
        populate(
            &mut u,
            &GalaxyGenParams {
                seed: 1,
                count: 200,
                faint_magnitude_limit: 10.0,
                bright_magnitude_floor: 4.0,
            },
        );
        for g in &u.galaxies {
            assert!(g.sersic_n >= 0.8 && g.sersic_n <= 4.5);
            assert!(g.axis_ratio > 0.0 && g.axis_ratio <= 1.0);
        }
    }
}
