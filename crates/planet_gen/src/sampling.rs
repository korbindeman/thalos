//! Sphere sampling helpers.
//!
//! The generator itself is sampling-agnostic — callers pass in whatever
//! points they like (cubesphere face tiles, icosahedral faces, arbitrary
//! LOD patches).  This module provides the Fibonacci lattice, which is a
//! cheap, roughly-uniform global sampling useful for tests, debugging, and
//! whole-body bakes where a specific tile layout isn't needed.

use glam::DVec3;

/// Equirectangular lattice of unit vectors with texel-centred sampling.
///
/// Row `y`, column `x` maps to longitude `((x + 0.5)/width − 0.5) · 2π` and
/// latitude `((y + 0.5)/height − 0.5) · π`.  This matches the UV convention
/// the planet impostor shader uses when sampling an equirect texture:
/// `u = atan2(n.z, n.x) / 2π + 0.5`, `v = asin(n.y) / π + 0.5`.
///
/// Points are returned in row-major order so the caller can feed the
/// resulting `SurfaceState` straight into a 2D texture of matching
/// dimensions.
pub fn equirect_lattice(width: usize, height: usize) -> Vec<DVec3> {
    let mut out = Vec::with_capacity(width * height);
    let inv_w = 1.0 / width as f64;
    let inv_h = 1.0 / height as f64;
    for y in 0..height {
        let v = (y as f64 + 0.5) * inv_h;
        let lat = (v - 0.5) * std::f64::consts::PI;
        let (sin_lat, cos_lat) = lat.sin_cos();
        for x in 0..width {
            let u = (x as f64 + 0.5) * inv_w;
            let lon = (u - 0.5) * std::f64::consts::TAU;
            let (sin_lon, cos_lon) = lon.sin_cos();
            out.push(DVec3::new(cos_lat * cos_lon, sin_lat, cos_lat * sin_lon));
        }
    }
    out
}

/// Deterministic Fibonacci (golden-angle) sphere lattice of `n` points.
///
/// The returned points are unit vectors.  Density is within a few percent
/// of uniform for any `n ≥ ~50`; at very small `n` there is visible
/// polar bias, which is unavoidable with this construction.
pub fn fibonacci_lattice(n: usize) -> Vec<DVec3> {
    if n == 0 {
        return Vec::new();
    }
    // Golden angle: π · (3 − √5).
    let phi = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let inv_n = 1.0 / n as f64;
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) * inv_n;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let theta = phi * i as f64;
            DVec3::new(r * theta.cos(), y, r * theta.sin())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_requested_count() {
        assert_eq!(fibonacci_lattice(0).len(), 0);
        assert_eq!(fibonacci_lattice(1).len(), 1);
        assert_eq!(fibonacci_lattice(1000).len(), 1000);
    }

    #[test]
    fn all_points_are_unit_vectors() {
        for p in fibonacci_lattice(1024) {
            let len = p.length();
            assert!((len - 1.0).abs() < 1e-12, "non-unit vector: {p:?} len={len}");
        }
    }

    #[test]
    fn distribution_mean_is_near_origin() {
        // A uniform sphere sampling has zero mean; Fibonacci is not perfectly
        // uniform but should be close for any decent count.
        let pts = fibonacci_lattice(4096);
        let sum: DVec3 = pts.iter().copied().sum();
        let mean = sum / pts.len() as f64;
        assert!(mean.length() < 5e-3, "mean {mean:?} too far from origin");
    }

    #[test]
    fn deterministic() {
        assert_eq!(fibonacci_lattice(128), fibonacci_lattice(128));
    }

    #[test]
    fn equirect_lattice_size_and_unit_length() {
        let pts = equirect_lattice(32, 16);
        assert_eq!(pts.len(), 32 * 16);
        for p in &pts {
            assert!((p.length() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn equirect_lattice_matches_shader_uv_convention() {
        // Sanity: the texel-centred point at the middle of the lattice
        // (u ≈ v ≈ 0.5) should map back to longitude 0, latitude 0,
        // i.e. +X in world space.
        let pts = equirect_lattice(1024, 512);
        // Row 255 + column 511 → (0.4995..., 0.4990...), near (+X).
        let p = pts[255 * 1024 + 511];
        assert!(p.x > 0.9999, "expected +X, got {p:?}");
        assert!(p.y.abs() < 0.01);
        assert!(p.z.abs() < 0.01);
    }
}
