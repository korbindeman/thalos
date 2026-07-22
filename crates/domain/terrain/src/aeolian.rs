//! Aeolian (wind-driven) primitives for dune generation.
//!
//! Math primitives for the layered-dune recipe in `docs/archive/gen/dunes.md`.
//! Used by both the bake-time draa-scale rasterization and the
//! per-fragment dune-scale synthesis in the impostor.
//!
//! Mirroring contract: any function here that the impostor evaluates per
//! fragment must have a bit-exact WGSL twin (cf. the `noise` module's
//! relationship with `crates/planet_rendering/src/shaders/noise.wgsl`).

use glam::Vec3;

/// Asymmetric ridge function — the core dune profile primitive.
///
/// Maps a wind-aligned phase to a [0, 1] saw-toothed shape with a gentle
/// stoss face (length `alpha`) and a sharp slip face (length `1 - alpha`).
///
/// At `alpha = 0.85` the slope ratio is ~5.7 : 1, which is what real dry
/// sand produces between its stoss face (~10–15°) and its slip face
/// (~32–34° angle of repose). Symmetric `(sin(phase) * 0.5 + 0.5).powf(k)`
/// will never look like sand — the asymmetry is the entire visual
/// signature. See `docs/archive/gen/dunes.md` §B.2.
///
/// `phase` is unnormalized; only its fractional part matters. `alpha` is
/// clamped to `[0.05, 0.95]` so neither face collapses.
#[inline]
pub fn asym_ridge(phase: f32, alpha: f32) -> f32 {
    let alpha = alpha.clamp(0.05, 0.95);
    let t = phase - phase.floor();
    if t < alpha {
        t / alpha
    } else {
        (1.0 - t) / (1.0 - alpha)
    }
}

/// Smooth membership weight of `dir` in a dune-sea region centered at
/// `center` with full-strength angular radius `radius_rad` and feathered
/// out to `radius_rad + feather_rad`. Returns 0 outside, 1 inside the
/// core, and a smooth fall-off through the feather band.
///
/// `dir` and `center` must both be unit vectors.
#[inline]
pub fn region_weight(dir: Vec3, center: Vec3, radius_rad: f32, feather_rad: f32) -> f32 {
    let cos_angle = dir.dot(center).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();
    let outer = radius_rad + feather_rad.max(0.0);
    if angle >= outer {
        return 0.0;
    }
    if angle <= radius_rad {
        return 1.0;
    }
    let t = (angle - radius_rad) / feather_rad.max(1.0e-6);
    let t = t.clamp(0.0, 1.0);
    1.0 - t * t * (3.0 - 2.0 * t)
}

/// Wind-aligned phase along a tangent axis, in units of wavelengths.
///
/// `dir` is the unit-sphere sample direction (where height is being
/// queried). `region_center` is the unit-vector center of the dune
/// region; `axis_tangent` is the wind direction in that region's local
/// tangent plane (unit, tangent to `region_center`). `lambda_m` is the
/// dune wavelength in meters and `body_radius_m` converts displacement
/// in unit-sphere coords into meters.
///
/// For the few-degree regions we're targeting, projecting `dir` into
/// the region's tangent plane and taking the linear component along
/// `axis_tangent` is well within sub-percent of exact great-circle arc,
/// well below dune wavelength — and avoids the f32 precision loss `acos`
/// suffers near 1.
#[inline]
pub fn wind_phase(
    dir: Vec3,
    region_center: Vec3,
    axis_tangent: Vec3,
    lambda_m: f32,
    body_radius_m: f32,
) -> f32 {
    let tangent_disp = dir - region_center * dir.dot(region_center);
    let along = tangent_disp.dot(axis_tangent.normalize_or_zero());
    along * body_radius_m / lambda_m.max(1.0e-3)
}
