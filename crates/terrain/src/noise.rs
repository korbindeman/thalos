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
//! [`hmf_ridged_3d`] and [`fbm3_vec3`] are the ground-LOD detail cascade.
//! They are CPU-only today — UDLOD tile data is baked from Rust at request
//! time and read back as R16 by the GPU, so there is no WGSL counterpart
//! to keep in sync. If a future GPU-side detail path lands, port these
//! to WGSL alongside the existing `fbm3` mirror.
//!
//! Hash: a small u32 PCG mixer (Mark Jarzynski, "Hash Functions for
//! GPU Rendering"). u32-only because WGSL is u32-native and SplitMix64
//! would need vec2<u32> emulation.
//!
//! Fade: Perlin's quintic `6t⁵ − 15t⁴ + 10t³`.

use glam::Vec3;

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

/// Derivative of [`fade`], `30t²(t − 1)²`.
#[inline]
pub fn fade_derivative(t: f32) -> f32 {
    30.0 * t * t * (t - 1.0) * (t - 1.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseDerivative3 {
    pub value: f32,
    pub derivative: Vec3,
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

/// 3D value noise and analytic derivatives with respect to `x`, `y`, `z`.
///
/// This follows Inigo Quilez's value-noise derivative expansion, using the
/// same quintic fade as [`value_noise_3d`]. The lattice values are already in
/// `[-1, 1)`, so no final `-1 + 2 * value` remap is required here.
pub fn value_noise_3d_derivative(x: f32, y: f32, z: f32, seed: u32) -> NoiseDerivative3 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;

    let wx = x - xi as f32;
    let wy = y - yi as f32;
    let wz = z - zi as f32;

    let ux = fade(wx);
    let uy = fade(wy);
    let uz = fade(wz);
    let dux = fade_derivative(wx);
    let duy = fade_derivative(wy);
    let duz = fade_derivative(wz);

    let a = hash3(xi, yi, zi, seed);
    let b = hash3(xi + 1, yi, zi, seed);
    let c = hash3(xi, yi + 1, zi, seed);
    let d = hash3(xi + 1, yi + 1, zi, seed);
    let e = hash3(xi, yi, zi + 1, seed);
    let f = hash3(xi + 1, yi, zi + 1, seed);
    let g = hash3(xi, yi + 1, zi + 1, seed);
    let h = hash3(xi + 1, yi + 1, zi + 1, seed);

    let k0 = a;
    let k1 = b - a;
    let k2 = c - a;
    let k3 = e - a;
    let k4 = a - b - c + d;
    let k5 = a - c - e + g;
    let k6 = a - b - e + f;
    let k7 = -a + b + c - d + e - f - g + h;

    let value = k0
        + k1 * ux
        + k2 * uy
        + k3 * uz
        + k4 * ux * uy
        + k5 * uy * uz
        + k6 * uz * ux
        + k7 * ux * uy * uz;
    let derivative = Vec3::new(
        (k1 + k4 * uy + k6 * uz + k7 * uy * uz) * dux,
        (k2 + k5 * uz + k4 * ux + k7 * uz * ux) * duy,
        (k3 + k6 * ux + k5 * uy + k7 * ux * uy) * duz,
    );

    NoiseDerivative3 { value, derivative }
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

/// [`fbm3`] with analytic derivatives.
///
/// Stacks [`value_noise_3d_derivative`] across octaves; the scalar `value`
/// is bit-identical to [`fbm3`] (same hash, same sub-seeding, same f32
/// ordering). The derivative applies the chain rule for the per-octave
/// frequency scaling: `∇(f(αp)) = α∇f(αp)`.
pub fn fbm3_derivative(
    x: f32,
    y: f32,
    z: f32,
    seed: u32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
) -> NoiseDerivative3 {
    let mut sum_value = 0.0;
    let mut sum_grad = Vec3::ZERO;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for o in 0..octaves {
        let osubseed = pcg_u32(seed.wrapping_add(o));
        let nd = value_noise_3d_derivative(x * freq, y * freq, z * freq, osubseed);
        sum_value += amp * nd.value;
        sum_grad += (amp * freq) * nd.derivative;
        norm += amp;
        amp *= persistence;
        freq *= lacunarity;
    }
    NoiseDerivative3 {
        value: sum_value / norm,
        derivative: sum_grad / norm,
    }
}

/// 3D fBm whose output is a vector. Three independent scalar fBm calls
/// (decorrelated by xor'ing the seed) packed into a [`Vec3`]. Used as the
/// offset field for domain-warped detail noise.
pub fn fbm3_vec3(p: Vec3, seed: u32, octaves: u32, persistence: f32, lacunarity: f32) -> Vec3 {
    let s_x = seed ^ 0x1A_2B_3C_4D;
    let s_y = seed ^ 0x5E_6F_70_81;
    let s_z = seed ^ 0x92_A3_B4_C5;
    Vec3::new(
        fbm3(p.x, p.y, p.z, s_x, octaves, persistence, lacunarity),
        fbm3(p.x, p.y, p.z, s_y, octaves, persistence, lacunarity),
        fbm3(p.x, p.y, p.z, s_z, octaves, persistence, lacunarity),
    )
}

/// Musgrave hybrid multifractal, ridged variant. Continuous octave count
/// for smooth LOD-residency fade.
///
/// The hybrid multifractal's defining feature is the self-modulating
/// `weight` term: each octave's contribution is gated by the running
/// product of previous octaves' signals. Where lower octaves landed in a
/// "flat" spot (signal near zero), the weight collapses and higher octaves
/// do not contribute — that is what produces "rough peaks, smooth valleys"
/// without any external biome mask. The ridged shape (`offset - |noise|`,
/// squared) concentrates signal at noise zero-crossings, producing
/// ridge-line crests rather than dome tops.
///
/// `octaves` is a continuous count: an integer plus a fractional tail. The
/// fractional part scales the top octave's contribution, so a tile cascading
/// from `octaves = 7.0` to `octaves = 8.0` across an LOD boundary blends
/// smoothly rather than stepping. Output and its derivative w.r.t. `octaves`
/// are both continuous.
///
/// Output range is `[0, 1]` for `offset = 1` (signal in `[0, 1]`) after
/// normalising against the closed-form maximum where every octave's signal
/// is one. Practical values are concentrated well below the maximum because
/// of weight collapse.
pub fn hmf_ridged_3d(
    p: Vec3,
    seed: u32,
    octaves: f32,
    persistence: f32,
    lacunarity: f32,
    offset: f32,
) -> f32 {
    let octaves = octaves.max(0.0);
    let full = octaves.floor() as u32;
    let frac = octaves - full as f32;
    let total = if frac > 0.0 { full + 1 } else { full };

    if total == 0 {
        return 0.0;
    }

    let mut result = 0.0f32;
    let mut weight = 1.0f32;
    let mut amp = 1.0f32;
    let mut freq = 1.0f32;

    for k in 0..total {
        let osub = pcg_u32(seed.wrapping_add(k));
        let n = value_noise_3d(p.x * freq, p.y * freq, p.z * freq, osub);
        let mut signal = offset - n.abs();
        signal *= signal;

        let oct_weight = if k == full { frac } else { 1.0 };
        result += weight * signal * amp * oct_weight;

        // Clamp guards against `offset > 1` paths where signal can exceed
        // one and weight would otherwise grow unbounded across octaves.
        weight = (weight * signal).clamp(0.0, 1.0);

        freq *= lacunarity;
        amp *= persistence;
    }

    // Closed-form maximum: weight = 1 throughout, signal saturates to 1.
    // Includes the partial top octave so `result / norm` is continuous in
    // `octaves`.
    let pf = persistence.powi(full as i32);
    let norm = (1.0 - pf) / (1.0 - persistence) + frac * pf;
    if norm > 0.0 { result / norm } else { 0.0 }
}
