//! SuperSimplex-based fBm and gradient domain warp.
//!
//! Thin pure-Rust replacement for the narrow slice of `fastnoise2` that the
//! `mare_flood` and `regolith` stages use: single noise type (SuperSimplex
//! 3D), textbook fBm with weighted_strength=0, gradient-vector domain warp,
//! and bulk per-face sampling. Backed by the `noise` crate.
//!
//! Not a FastNoise2 clone. Output does not match FastNoise2 bit-for-bit —
//! baked goldens must be regenerated when porting.

use glam::Vec3;
use noise::{NoiseFn, SuperSimplex};

/// Textbook fractional-Brownian-motion of SuperSimplex 3D, normalized so the
/// output falls in roughly `[-1, 1]`.
///
/// Input is consumed as-is; apply any outer `domain_scale` at the call site
/// by passing `dir * freq` to [`SsFbm::sample`].
pub struct SsFbm {
    noise: SuperSimplex,
    gain: f32,
    octaves: u32,
    lacunarity: f32,
    inv_amp_sum: f32,
}

impl SsFbm {
    pub fn new(seed: i32, gain: f32, octaves: u32, lacunarity: f32) -> Self {
        let mut amp = 1.0_f32;
        let mut sum = 0.0_f32;
        for _ in 0..octaves {
            sum += amp;
            amp *= gain;
        }
        Self {
            noise: SuperSimplex::new(seed as u32),
            gain,
            octaves,
            lacunarity,
            inv_amp_sum: if sum > 0.0 { 1.0 / sum } else { 1.0 },
        }
    }

    #[inline]
    pub fn sample(&self, p: Vec3) -> f32 {
        let mut freq = 1.0_f32;
        let mut amp = 1.0_f32;
        let mut sum = 0.0_f32;
        for _ in 0..self.octaves {
            let x = (p.x * freq) as f64;
            let y = (p.y * freq) as f64;
            let z = (p.z * freq) as f64;
            sum += amp * self.noise.get([x, y, z]) as f32;
            freq *= self.lacunarity;
            amp *= self.gain;
        }
        sum * self.inv_amp_sum
    }
}

/// Gradient-vector domain warp. Samples three axis-offset SuperSimplex fields
/// at `p * frequency` to form an offset vector, scales by `amplitude`, and
/// returns `p + offset`.
pub struct GradientWarp {
    noise: SuperSimplex,
    frequency: f32,
    amplitude: f32,
}

impl GradientWarp {
    pub fn new(seed: i32, amplitude: f32, frequency: f32) -> Self {
        Self {
            noise: SuperSimplex::new(seed as u32),
            frequency,
            amplitude,
        }
    }

    #[inline]
    pub fn warp(&self, p: Vec3) -> Vec3 {
        let x = (p.x * self.frequency) as f64;
        let y = (p.y * self.frequency) as f64;
        let z = (p.z * self.frequency) as f64;
        // Axis offsets decorrelate the three noise samples without needing
        // three independent SuperSimplex instances.
        let dx = self.noise.get([x + 19.19, y, z]) as f32;
        let dy = self.noise.get([x, y + 31.71, z]) as f32;
        let dz = self.noise.get([x, y, z + 47.43]) as f32;
        p + Vec3::new(dx, dy, dz) * self.amplitude
    }
}

/// Bulk-sample `out[i] = f(Vec3(xs[i], ys[i], zs[i]))`. Mirrors the per-face
/// sampling pattern the stages use with fastnoise2's `gen_position_array_3d`.
#[inline]
pub fn bulk_sample<F: Fn(Vec3) -> f32>(out: &mut [f32], xs: &[f32], ys: &[f32], zs: &[f32], f: F) {
    for i in 0..out.len() {
        out[i] = f(Vec3::new(xs[i], ys[i], zs[i]));
    }
}
