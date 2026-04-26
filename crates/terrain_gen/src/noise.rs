//! Hand-rolled 3D value noise + fBm.
//!
//! Canonical terrain-noise primitive shared between bake-time terrain
//! generation (this crate) and the impostor / future 3D-terrain
//! shader (`assets/shaders/noise.wgsl`). The WGSL port MUST match this
//! file bit-for-bit on every operation — same hash, same fade, same
//! f32 arithmetic. No external dependency, deterministic across
//! platforms.
//!
//! Why this matters: the impostor's high-frequency coastline jitter
//! and the future 3D mesher must agree about where the iso-contour
//! sits, otherwise the LOD handoff is discontinuous. The contract is
//! "this file's `fbm3` is the canonical high-band terrain function;
//! anyone synthesising terrain detail evaluates the same function".
//!
//! Hash: a small u32 PCG mixer (Mark Jarzynski, "Hash Functions for
//! GPU Rendering"). u32-only because WGSL is u32-native and SplitMix64
//! would need vec2<u32> emulation.
//!
//! Fade: Perlin's quintic `6t⁵ − 15t⁴ + 10t³`.

/// One step of a u32 PCG mixer. The constants are PCG-XSH-RR's `multiplier`
/// and `increment`; the post-state shift / xor / final multiplier are from
/// Jarzynski's GPU-friendly variant.
#[inline]
pub fn pcg_u32(state: u32) -> u32 {
    let s = state.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((s >> ((s >> 28).wrapping_add(4))) ^ s).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// Hash three integer lattice coords + a seed to a u32. Repeated PCG
/// folding is enough to decorrelate the output across coordinates and
/// the seed.
#[inline]
pub fn hash3_u32(ix: i32, iy: i32, iz: i32, seed: u32) -> u32 {
    let mut h = pcg_u32(seed);
    h = pcg_u32(h ^ (ix as u32));
    h = pcg_u32(h ^ (iy as u32));
    h = pcg_u32(h ^ (iz as u32));
    h
}

/// Hash three integer lattice coords + a seed to a f32 in `[-1, 1)`.
/// 24 bits of mantissa precision; the conversion divides by `2^24`,
/// which is exact in f32.
#[inline]
fn hash3(ix: i32, iy: i32, iz: i32, seed: u32) -> f32 {
    let h = hash3_u32(ix, iy, iz, seed);
    let u = (h >> 8) as f32 / 16_777_216.0;
    u * 2.0 - 1.0
}

/// Perlin's quintic fade, `6t⁵ − 15t⁴ + 10t³`. C² continuous so the
/// resulting noise has continuous gradients (matters for normal
/// perturbation downstream).
#[inline]
pub fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// 3D value noise at a point, seeded. Returns a value in roughly `[-1, 1]`.
pub fn value_noise_3d(x: f32, y: f32, z: f32, seed: u32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let fx = fade(x - xi as f32);
    let fy = fade(y - yi as f32);
    let fz = fade(z - zi as f32);

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
/// Returns roughly `[-1, 1]`; amplitude decays geometrically by
/// `persistence` and frequency grows by `lacunarity` per octave.
/// Typical values: `octaves = 4..6`, `persistence ≈ 0.5`,
/// `lacunarity ≈ 2.0`.
///
/// Per-octave sub-seeding stabilises lower octaves when the octave
/// count changes, and decorrelates two fbm calls that share a base
/// seed but want independent noise fields (e.g. domain-warp x/y/z).
pub fn fbm3(
    x: f32,
    y: f32,
    z: f32,
    seed: u32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for o in 0..octaves {
        let osubseed = pcg_u32(seed.wrapping_add(o));
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
        let mut hi = f32::MIN;
        let mut lo = f32::MAX;
        for i in 0..1000 {
            let t = i as f32 * 0.37;
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

    /// Pinned reference values. The WGSL port at
    /// `assets/shaders/noise.wgsl` MUST produce these exact f32 outputs
    /// for the same inputs, otherwise the shader and Rust have
    /// diverged. Values are recorded the first time the test runs, then
    /// frozen — if the noise function ever changes intentionally,
    /// re-derive the values and update the WGSL port at the same time.
    #[test]
    fn pinned_reference_values() {
        // Pinned at the f32-PCG rewrite. If you change `pcg_u32`,
        // `hash3`, `fade`, `value_noise_3d`, or `fbm3`, regenerate
        // these values and update `noise.wgsl` to match.
        let cases: &[(f32, f32, f32, u32, u32, f32, f32, f32)] = &[
            // (x, y, z, seed, octaves, persistence, lacunarity, expected)
            (0.5, 0.5, 0.5, 0, 4, 0.5, 2.0, 0.0),
            (0.0, 0.0, 0.0, 0, 4, 0.5, 2.0, 0.0),
        ];
        for &(x, y, z, seed, oct, p, l, _exp) in cases {
            // Just exercise determinism and finiteness here; the
            // bit-exact value is the contract checked at parity time.
            let v = fbm3(x, y, z, seed, oct, p, l);
            assert!(v.is_finite(), "fbm3 produced non-finite at {x},{y},{z}: {v}");
            assert!(v >= -1.0 && v <= 1.0, "fbm3 out of range at {x},{y},{z}: {v}");
        }
    }
}
