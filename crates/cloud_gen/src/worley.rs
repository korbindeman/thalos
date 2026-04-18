//! 3D Worley (cellular) noise, fBm, and a pre-baked tileable volume.
//!
//! Wedekind's technique calls Worley fBm many times per cubemap texel per
//! iteration (50 iters × 6 faces × N² × 6 gradient taps × 2 hemispheres ×
//! 3 octaves). A literal Worley evaluation costs 27 cell-distance
//! computations, which is too slow for a CPU bake even with rayon.
//!
//! The speedup: bake Worley once into a tileable 3D volume and trilinear-
//! sample it for every fBm call. The `lookup_north` / `lookup_south` /
//! `lookup_cover` routines in Wedekind's GLSL do the same thing (GLSL
//! `texture(sampler3D, p)` with repeat addressing).

use glam::Vec3;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Hash primitives
// ---------------------------------------------------------------------------

fn pcg(x: u32) -> u32 {
    let state = x.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

fn hash3(ix: i32, iy: i32, iz: i32, seed: u32) -> u32 {
    let mut h = (ix as u32).wrapping_mul(73856093);
    h ^= (iy as u32).wrapping_mul(19349663);
    h ^= (iz as u32).wrapping_mul(83492791);
    h = pcg(h ^ seed);
    h ^= seed.wrapping_mul(1540483477);
    pcg(h)
}

/// One feature point per cell, uniformly distributed in `[0, 1)³`.
fn feature_point(ix: i32, iy: i32, iz: i32, seed: u32) -> Vec3 {
    let h1 = hash3(ix, iy, iz, seed);
    let h2 = pcg(h1);
    let h3 = pcg(h2);
    let norm = 1.0 / u32::MAX as f32;
    Vec3::new(
        h1 as f32 * norm,
        h2 as f32 * norm,
        h3 as f32 * norm,
    )
}

// ---------------------------------------------------------------------------
// Tileable 3D Worley F1 distance
// ---------------------------------------------------------------------------

/// Tileable 3D Worley noise. `p` is in `[0, 1)` (tiled via modular
/// arithmetic), subdivided into `cells_per_edge³` cells each containing
/// one feature point. Returns the F1 distance to the nearest feature
/// point, normalised to approximately `[0, 1]`.
///
/// The raw F1 distance in a unit cube with one point per cell has
/// maximum value `sqrt(3)/2 ≈ 0.866` (a sample at the far corner from
/// the nearest feature, in the adjacent-cell worst case). We scale by
/// the cell size and divide by 0.866 so the output is roughly in
/// `[0, 1]` regardless of `cells_per_edge`.
pub fn worley_3d_tileable(p: Vec3, cells_per_edge: u32, seed: u32) -> f32 {
    let cells = cells_per_edge as i32;
    let cells_f = cells_per_edge as f32;
    let cp = p * cells_f;
    let pi = cp.floor();
    let pf = cp - pi;
    let base_ix = pi.x as i32;
    let base_iy = pi.y as i32;
    let base_iz = pi.z as i32;

    let mut min_dist_sq = f32::MAX;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let raw_x = base_ix + dx;
                let raw_y = base_iy + dy;
                let raw_z = base_iz + dz;
                let wrap_x = raw_x.rem_euclid(cells);
                let wrap_y = raw_y.rem_euclid(cells);
                let wrap_z = raw_z.rem_euclid(cells);
                let feature = feature_point(wrap_x, wrap_y, wrap_z, seed);
                let offset = Vec3::new(dx as f32, dy as f32, dz as f32);
                let diff = offset + feature - pf;
                let dist_sq = diff.length_squared();
                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                }
            }
        }
    }
    // `min_dist_sq` is measured in cell-space units (each cell is a unit
    // cube in the inner loop). The farthest possible F1 distance in a
    // 3D unit-cell grid is `sqrt(3)/2 ≈ 0.866` (the pathological corner
    // case — a sample equidistant from 8 features). Normalise so the
    // output is in `[0, 1]` regardless of `cells_per_edge`; keeping the
    // amplitude scale independent of the cell count is what lets fBm
    // octaves compose at consistent weights.
    let dist = min_dist_sq.sqrt();
    (dist / 0.866).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Pre-baked Worley volume (trilinear-sampled)
// ---------------------------------------------------------------------------

/// Tileable 3D f32 volume. Baked once from a seed, then sampled trilinearly
/// by `worley_fbm`. The CPU-equivalent of a GLSL `sampler3D` with repeat
/// addressing. All six cubemap faces and all 50 warp iterations hit the
/// same cached volumes.
#[derive(Clone)]
pub struct WorleyVolume {
    size: u32,
    data: Vec<f32>,
}

impl WorleyVolume {
    /// Bake a `size³` tileable Worley volume. The volume contains
    /// `cells_per_edge³` feature points positioned via the seed.
    ///
    /// Bake cost scales as `size³ × 27`. At `size=64, cells=8` this is
    /// ~7M distance computations — ~50 ms on a single core, <10 ms with
    /// rayon across 8 cores.
    pub fn bake(size: u32, cells_per_edge: u32, seed: u32) -> Self {
        assert!(size > 1, "volume size must be > 1");
        assert!(cells_per_edge > 0, "cells_per_edge must be > 0");
        let n = (size * size * size) as usize;
        let mut data = vec![0.0_f32; n];
        let inv = 1.0 / size as f32;
        data.par_iter_mut().enumerate().for_each(|(idx, out)| {
            let i = idx as u32;
            let z = (i / (size * size)) as f32;
            let y = ((i / size) % size) as f32;
            let x = (i % size) as f32;
            // Texel centres — offset by 0.5 so the volume is symmetric
            // about its midpoints and lookups at integer `p*size` land
            // on texel centres.
            let p = Vec3::new((x + 0.5) * inv, (y + 0.5) * inv, (z + 0.5) * inv);
            *out = worley_3d_tileable(p, cells_per_edge, seed);
        });
        Self { size, data }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    /// Trilinear sample of the volume with tiling (repeat) addressing.
    /// `p` is interpreted modulo 1 on each axis.
    pub fn sample(&self, p: Vec3) -> f32 {
        let sz = self.size as f32;
        // Wrap into `[0, 1)`, then scale to texel space with a half-texel
        // offset (matches the bake's centre-of-voxel convention).
        let px = p.x.rem_euclid(1.0) * sz - 0.5;
        let py = p.y.rem_euclid(1.0) * sz - 0.5;
        let pz = p.z.rem_euclid(1.0) * sz - 0.5;

        let xi = px.floor();
        let yi = py.floor();
        let zi = pz.floor();
        let fx = px - xi;
        let fy = py - yi;
        let fz = pz - zi;

        let s = self.size as i32;
        let x0 = (xi as i32).rem_euclid(s) as u32;
        let y0 = (yi as i32).rem_euclid(s) as u32;
        let z0 = (zi as i32).rem_euclid(s) as u32;
        let x1 = (x0 + 1) % self.size;
        let y1 = (y0 + 1) % self.size;
        let z1 = (z0 + 1) % self.size;

        let c000 = self.voxel(x0, y0, z0);
        let c100 = self.voxel(x1, y0, z0);
        let c010 = self.voxel(x0, y1, z0);
        let c110 = self.voxel(x1, y1, z0);
        let c001 = self.voxel(x0, y0, z1);
        let c101 = self.voxel(x1, y0, z1);
        let c011 = self.voxel(x0, y1, z1);
        let c111 = self.voxel(x1, y1, z1);

        let x00 = c000 + (c100 - c000) * fx;
        let x10 = c010 + (c110 - c010) * fx;
        let x01 = c001 + (c101 - c001) * fx;
        let x11 = c011 + (c111 - c011) * fx;
        let y0_ = x00 + (x10 - x00) * fy;
        let y1_ = x01 + (x11 - x01) * fy;
        y0_ + (y1_ - y0_) * fz
    }

    fn voxel(&self, x: u32, y: u32, z: u32) -> f32 {
        let idx = (z * self.size * self.size + y * self.size + x) as usize;
        self.data[idx]
    }
}

// ---------------------------------------------------------------------------
// Worley fBm
// ---------------------------------------------------------------------------

/// Weighted sum of Worley noise at doubling frequencies.
///
/// `octave_weights[i]` is the amplitude of the `2^i`-frequency octave.
/// Wedekind's defaults (see `CloudBakeConfig::wedekind_defaults`):
///
///   flow  octaves = [0.5, 0.25, 0.125]                 (3 octaves)
///   cloud octaves = [0.25, 0.25, 0.125, 0.125, 0.0625, 0.0625]  (6 octaves)
///
/// Weights do NOT need to sum to 1 — the caller normalises elsewhere via
/// the threshold/multiplier in the rendering path. We emit the raw
/// weighted sum.
pub fn worley_fbm(volume: &WorleyVolume, p: Vec3, octave_weights: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut freq = 1.0;
    for &w in octave_weights {
        sum += w * volume.sample(p * freq);
        freq *= 2.0;
    }
    sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worley_is_tileable() {
        // Sampling at `p` and `p + (1, 0, 0)` (or any integer offset) must
        // yield the same value — the wraparound is built into
        // `worley_3d_tileable`.
        let seed = 0xDEAD_BEEF;
        for &(x, y, z) in &[(0.1, 0.2, 0.3), (0.7, 0.05, 0.99), (0.33, 0.66, 0.42)] {
            let p = Vec3::new(x, y, z);
            let a = worley_3d_tileable(p, 4, seed);
            let b = worley_3d_tileable(p + Vec3::X, 4, seed);
            let c = worley_3d_tileable(p + Vec3::new(0.0, 1.0, -2.0), 4, seed);
            assert!((a - b).abs() < 1e-5, "X-tile mismatch at {:?}: {} vs {}", p, a, b);
            assert!((a - c).abs() < 1e-5, "YZ-tile mismatch at {:?}: {} vs {}", p, a, c);
        }
    }

    #[test]
    fn worley_output_in_range() {
        let seed = 42;
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    let p = Vec3::new(x as f32 / 16.0, y as f32 / 16.0, z as f32 / 16.0);
                    let v = worley_3d_tileable(p, 4, seed);
                    min = min.min(v);
                    max = max.max(v);
                }
            }
        }
        assert!(min >= 0.0, "min {} < 0", min);
        assert!(max <= 1.0, "max {} > 1", max);
        // Should cover a reasonable spread on a 16³ probe — the extreme
        // values near 0 and 1 are rare (corner cases of the cell layout),
        // but a typical sample window produces 0.2+ range.
        assert!(max - min > 0.2, "range {} too narrow", max - min);
    }

    #[test]
    fn volume_sample_matches_analytic_at_centers() {
        // At texel centres, the trilinear lookup should return the stored
        // value, which equals the analytic Worley at the same point.
        let seed = 0xC0FFEE;
        let size = 32;
        let cells = 4;
        let vol = WorleyVolume::bake(size, cells, seed);
        let inv = 1.0 / size as f32;
        for (ix, iy, iz) in [(0u32, 0u32, 0u32), (5, 7, 11), (31, 31, 31)] {
            let p = Vec3::new(
                (ix as f32 + 0.5) * inv,
                (iy as f32 + 0.5) * inv,
                (iz as f32 + 0.5) * inv,
            );
            let sampled = vol.sample(p);
            let analytic = worley_3d_tileable(p, cells, seed);
            assert!(
                (sampled - analytic).abs() < 1e-5,
                "mismatch at ({}, {}, {}): {} vs {}",
                ix,
                iy,
                iz,
                sampled,
                analytic
            );
        }
    }

    #[test]
    fn volume_sample_wraps_across_seam() {
        let seed = 0xBADD_F00D;
        let vol = WorleyVolume::bake(32, 4, seed);
        for &(x, y, z) in &[(0.01, 0.5, 0.5), (0.99, 0.5, 0.5), (0.5, 0.001, 0.999)] {
            let p = Vec3::new(x, y, z);
            let a = vol.sample(p);
            let b = vol.sample(p + Vec3::new(1.0, 0.0, 0.0));
            let c = vol.sample(p + Vec3::new(0.0, -2.0, 5.0));
            assert!((a - b).abs() < 1e-5, "wrap(X) mismatch at {:?}: {} vs {}", p, a, b);
            assert!((a - c).abs() < 1e-5, "wrap(YZ) mismatch at {:?}: {} vs {}", p, a, c);
        }
    }

    #[test]
    fn worley_is_deterministic_in_seed() {
        let p = Vec3::new(0.31, 0.42, 0.59);
        let a = worley_3d_tileable(p, 8, 123);
        let b = worley_3d_tileable(p, 8, 123);
        let c = worley_3d_tileable(p, 8, 124);
        assert_eq!(a, b);
        assert!((a - c).abs() > 1e-6);
    }
}
