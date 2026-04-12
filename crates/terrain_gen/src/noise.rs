//! Hand-rolled 3D value noise on the sphere.
//!
//! v0.1 needs only coherent low-frequency noise for primordial topography
//! (and, later, mild rim-perturbation jitter).  A full Perlin / simplex
//! implementation would be overkill; this module provides trilinearly
//! interpolated value noise over a hashed integer lattice, plus an fBm
//! stacker.  No external dependency, deterministic across platforms.
//!
//! For whole-body generation at low frequencies the difference between
//! value noise and gradient noise is imperceptible — the spec explicitly
//! calls Stage 1 "intentionally gentle" on airless bodies.

use crate::seeding::splitmix64;

/// Hash three integer lattice coords + a seed to a f64 in `[-1, 1)`.
#[inline]
fn hash3(ix: i32, iy: i32, iz: i32, seed: u64) -> f64 {
    let mut h = seed;
    h ^= (ix as i64 as u64).wrapping_mul(0x9E3779B97F4A7C15);
    h = splitmix64(h);
    h ^= (iy as i64 as u64).wrapping_mul(0xBF58476D1CE4E5B9);
    h = splitmix64(h);
    h ^= (iz as i64 as u64).wrapping_mul(0x94D049BB133111EB);
    h = splitmix64(h);
    // Map the top 53 bits to [-1, 1).
    let u = (h >> 11) as f64 / (1u64 << 53) as f64;
    u * 2.0 - 1.0
}

/// Smoothstep fade, `6t^5 − 15t^4 + 10t^3`.
#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// 3D value noise at a point, seeded.  Returns a value in roughly `[-1, 1]`.
pub fn value_noise_3d(x: f64, y: f64, z: f64, seed: u64) -> f64 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let fx = fade(x - xi as f64);
    let fy = fade(y - yi as f64);
    let fz = fade(z - zi as f64);

    let c000 = hash3(xi, yi, zi, seed);
    let c100 = hash3(xi + 1, yi, zi, seed);
    let c010 = hash3(xi, yi + 1, zi, seed);
    let c110 = hash3(xi + 1, yi + 1, zi, seed);
    let c001 = hash3(xi, yi, zi + 1, seed);
    let c101 = hash3(xi + 1, yi, zi + 1, seed);
    let c011 = hash3(xi, yi + 1, zi + 1, seed);
    let c111 = hash3(xi + 1, yi + 1, zi + 1, seed);

    let x00 = c000 + (c100 - c000) * fx;
    let x10 = c010 + (c110 - c010) * fx;
    let x01 = c001 + (c101 - c001) * fx;
    let x11 = c011 + (c111 - c011) * fx;

    let y0 = x00 + (x10 - x00) * fy;
    let y1 = x01 + (x11 - x01) * fy;

    y0 + (y1 - y0) * fz
}

/// Fractal Brownian motion stacker over [`value_noise_3d`].
///
/// Returns roughly `[-1, 1]`; amplitude decay is geometric in `persistence`
/// and frequency grows by `lacunarity` per octave.  Typical values:
/// `octaves = 4..6`, `persistence ≈ 0.5`, `lacunarity ≈ 2.0`.
pub fn fbm3(
    x: f64,
    y: f64,
    z: f64,
    seed: u64,
    octaves: u32,
    persistence: f64,
    lacunarity: f64,
) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for o in 0..octaves {
        // Use a per-octave sub-seed so lower octaves are stable when the
        // octave count changes, and so two calls to fbm3 with different
        // seeds decorrelate from the first octave on.
        let osubseed = splitmix64(seed.wrapping_add(o as u64));
        sum += amp * value_noise_3d(x * freq, y * freq, z * freq, osubseed);
        norm += amp;
        amp *= persistence;
        freq *= lacunarity;
    }
    sum / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_noise_is_deterministic() {
        let a = value_noise_3d(1.23, 4.56, 7.89, 42);
        let b = value_noise_3d(1.23, 4.56, 7.89, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn value_noise_in_approximate_range() {
        let mut hi = f64::MIN;
        let mut lo = f64::MAX;
        for i in 0..1000 {
            let t = i as f64 * 0.37;
            let v = value_noise_3d(t, t * 1.5, t * 0.8, 99);
            hi = hi.max(v);
            lo = lo.min(v);
        }
        assert!(hi <= 1.0 && lo >= -1.0, "range violated: [{lo}, {hi}]");
        assert!(hi > 0.3 && lo < -0.3, "range too narrow: [{lo}, {hi}]");
    }

    #[test]
    fn fbm_is_continuous() {
        // Nearby inputs produce nearby outputs.
        let a = fbm3(1.0, 2.0, 3.0, 0, 4, 0.5, 2.0);
        let b = fbm3(1.001, 2.0, 3.0, 0, 4, 0.5, 2.0);
        assert!((a - b).abs() < 0.05, "fbm discontinuous: {a} vs {b}");
    }

    #[test]
    fn fbm_varies_with_seed() {
        let a = fbm3(1.0, 2.0, 3.0, 0, 4, 0.5, 2.0);
        let b = fbm3(1.0, 2.0, 3.0, 1, 4, 0.5, 2.0);
        assert_ne!(a, b);
    }
}
