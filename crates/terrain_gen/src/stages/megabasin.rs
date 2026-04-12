use glam::Vec3;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::noise::fbm3;
use crate::seeding::{splitmix64, Rng};
use crate::stage::Stage;
use super::util::for_all_texels;

/// A single large impact basin definition.
#[derive(Clone, Debug, Deserialize)]
pub struct BasinDef {
    /// Unit direction toward basin center.
    pub center_dir: Vec3,
    /// Radius of the outermost ring in meters.
    pub radius_m: f32,
    /// Depth at basin center in meters (positive = downward).
    pub depth_m: f32,
    /// Number of concentric inner-ring ridges for multi-ring basins.
    /// Rings are placed at √2 sub-multiples of `radius_m` (Pike & Spudis 1987).
    pub ring_count: u32,
}

/// Places the largest-scale impact basins and applies hemispheric asymmetry.
///
/// On top of the raw bowl, each basin receives:
///   * Noise-warped rim radius (breaks perfect circular symmetry)
///   * Flat-ish floor with terrace ridges on the wall (gravitational slumping)
///   * Peak ring at ~0.5 R for peak-ring class basins (> ~300 km diameter)
///   * Concentric inner rings at √2 spacing for multi-ring basins
///   * Raised ejecta blanket outside the rim with McGetchin r⁻³ falloff
///   * Radial angular "sculpture" on the ejecta (Imbrium-style streaks)
///
/// The result is a set of basins that read as embedded features rather than
/// flat depressions stamped into the surface.
///
/// Hemispheric lowering uses `BodyBuilder::tidal_axis` as the near-side
/// reference. If the builder has no tidal axis, hemispheric lowering is
/// skipped.
#[derive(Debug, Clone, Deserialize)]
pub struct Megabasin {
    pub basins: Vec<BasinDef>,
    /// Broad hemispheric lowering of the near side, in meters.
    pub hemispheric_lowering_m: f32,
}

struct BasinPrecomp {
    def: BasinDef,
    seed: u64,
    center: Vec3,
    /// Tangent-plane basis vectors at the basin center (for azimuth measurement).
    tan1: Vec3,
    tan2: Vec3,
    /// Morphology class flags.
    has_peak_ring: bool,
    has_inner_rings: bool,
    /// Per-basin ejecta angular sculpture parameters.
    lobe_count: f32,
    lobe_phase: f32,
    /// Per-basin warp / scale jitter.
    depth_jitter: f32,
}

const EJECTA_MAX_T: f32 = 3.0;

impl Stage for Megabasin {
    fn name(&self) -> &str { "megabasin" }
    fn dependencies(&self) -> &[&str] { &["differentiate"] }

    fn apply(&self, builder: &mut BodyBuilder) {
        let body_radius = builder.radius_m;
        let res = builder.cubemap_resolution;
        let seed = builder.stage_seed();
        let near = builder.tidal_axis;

        let precomp: Vec<BasinPrecomp> = self
            .basins
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let basin_seed = splitmix64(
                    seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut rng = Rng::new(basin_seed);
                let center = b.center_dir.normalize();
                // Build tangent frame at basin center for azimuth measurement.
                let up = if center.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
                let tan1 = up.cross(center).normalize();
                let tan2 = center.cross(tan1);

                let r_km = b.radius_m / 1000.0;
                // Morphology thresholds are per-body; Mira is half-lunar gravity
                // so transitions shift to larger sizes, but for megabasin scale
                // these size cutoffs are already well above the transitional band.
                let has_peak_ring = r_km >= 140.0;
                let has_inner_rings = r_km >= 200.0;

                let lobe_count = 6.0 + rng.next_f64() as f32 * 6.0; // 6..12
                let lobe_phase = rng.next_f64() as f32 * std::f32::consts::TAU;
                let depth_jitter = 1.0 + rng.next_f64_signed() as f32 * 0.08;

                BasinPrecomp {
                    def: b.clone(),
                    seed: basin_seed,
                    center,
                    tan1,
                    tan2,
                    has_peak_ring,
                    has_inner_rings,
                    lobe_count,
                    lobe_phase,
                    depth_jitter,
                }
            })
            .collect();

        for_all_texels(res, |face, x, y, dir| {
            let mut h = 0.0f32;

            // Hemispheric lowering: gentle cosine toward the near-side center.
            // Skipped on bodies without a tidal axis (no near/far asymmetry).
            if let Some(near) = near {
                let near_dot = dir.dot(near).max(0.0);
                h -= self.hemispheric_lowering_m * near_dot * near_dot;
            }

            for b in &precomp {
                let cos_d = dir.dot(b.center).clamp(-1.0, 1.0);
                let angular_dist = cos_d.acos();
                let surface_dist = angular_dist * body_radius;
                let t_raw = surface_dist / b.def.radius_m;
                if t_raw > EJECTA_MAX_T {
                    continue;
                }

                // Azimuth around basin center (for sculpture + rim warp).
                let tx = dir.dot(b.tan1);
                let ty = dir.dot(b.tan2);
                let azimuth = ty.atan2(tx);

                // Low-frequency radial warp: breaks perfect circular symmetry.
                let warp = fbm3(
                    dir.x as f64 * 2.5,
                    dir.y as f64 * 2.5,
                    dir.z as f64 * 2.5,
                    b.seed,
                    3,
                    0.5,
                    2.0,
                ) as f32;
                let t = t_raw / (1.0 + warp * 0.12);

                h += basin_profile(t, azimuth, b);
            }

            if h.abs() > 1e-6 {
                builder.height_contributions.add_height(face, x, y, h);
            }
        });

        // Publish basin definitions for downstream stages (MareFlood reads these).
        builder.megabasins = self.basins.clone();
    }
}

fn basin_profile(t: f32, azimuth: f32, b: &BasinPrecomp) -> f32 {
    if t < 1.0 {
        basin_interior(t, b)
    } else {
        basin_ejecta(t, azimuth, b)
    }
}

fn basin_interior(t: f32, b: &BasinPrecomp) -> f32 {
    let depth = b.def.depth_m * b.depth_jitter;
    let rim_h = depth * 0.12;

    // Cosine bowl: -depth at center, 0 at rim base.
    let bowl = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
    let mut h = -depth * bowl;

    // Terrace ripple on the wall: sinusoidal perturbation whose amplitude
    // peaks mid-wall and fades toward floor and rim.
    if t > 0.3 && t < 0.97 {
        let n_terraces = if b.has_inner_rings {
            4.0
        } else if b.has_peak_ring {
            3.0
        } else {
            2.0
        };
        let shape = ((1.0 - t) * (t - 0.3) * 6.0).max(0.0);
        let ripple = (n_terraces * std::f32::consts::PI * t).sin();
        h += depth * 0.05 * shape * ripple;
    }

    // Rim crest ridge (narrow Gaussian at t=1).
    let rim_dr = t - 1.0;
    h += rim_h * (-(rim_dr * rim_dr) / (2.0 * 0.04 * 0.04)).exp();

    // Peak ring at ~0.5 R for peak-ring class basins.
    if b.has_peak_ring {
        let dr = t - 0.5;
        let pr_h = depth * 0.18;
        h += pr_h * (-(dr * dr) / (2.0 * 0.055 * 0.055)).exp();
    }

    // Concentric inner rings at √2 spacing for multi-ring basins.
    if b.has_inner_rings {
        let sqrt2 = std::f32::consts::SQRT_2;
        for n in 1..b.def.ring_count.min(5) {
            let ring_t = 1.0 / sqrt2.powi(n as i32);
            if ring_t < 0.94 && ring_t > 0.12 {
                let dr = t - ring_t;
                let ring_h = depth * 0.07 * 0.8_f32.powi((n - 1) as i32);
                h += ring_h * (-(dr * dr) / (2.0 * 0.05 * 0.05)).exp();
            }
        }
    }

    h
}

fn basin_ejecta(t: f32, azimuth: f32, b: &BasinPrecomp) -> f32 {
    let depth = b.def.depth_m * b.depth_jitter;
    let rim_h = depth * 0.12;

    // McGetchin-style raised ejecta blanket: rim_h at t=1, r⁻³ decay outward.
    // This is the #1 feature missing from the old megabasin — without it
    // basins read as flat bowls stamped into the surface.
    let radial = rim_h / (t * t * t);

    // Angular sculpture (Imbrium-style radial streaks): cosine lobes around
    // the basin. Ramps on from t=1 so it matches the interior rim smoothly,
    // and fades to zero at EJECTA_MAX_T.
    let sculpture = (b.lobe_count * azimuth + b.lobe_phase).cos();
    let ramp_on = ((t - 1.0) / 0.25).clamp(0.0, 1.0);
    let fade_out = ((EJECTA_MAX_T - t) / (EJECTA_MAX_T - 1.0)).clamp(0.0, 1.0);
    let fade_smooth = fade_out * fade_out * (3.0 - 2.0 * fade_out);
    let sculpture_mod = 1.0 + 0.55 * sculpture * ramp_on * fade_smooth;

    // Smooth taper to zero at EJECTA_MAX_T so the blanket vanishes cleanly.
    let taper = fade_smooth;

    radial * sculpture_mod * taper
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_basin() -> BasinPrecomp {
        BasinPrecomp {
            def: BasinDef {
                center_dir: Vec3::Z,
                radius_m: 300_000.0,
                depth_m: 6_000.0,
                ring_count: 4,
            },
            seed: 1234,
            center: Vec3::Z,
            tan1: Vec3::X,
            tan2: Vec3::Y,
            has_peak_ring: true,
            has_inner_rings: true,
            lobe_count: 8.0,
            lobe_phase: 0.0,
            depth_jitter: 1.0,
        }
    }

    #[test]
    fn interior_is_deeper_than_rim() {
        let b = sample_basin();
        let center = basin_interior(0.0, &b);
        let rim = basin_interior(1.0, &b);
        assert!(center < rim, "center {center} should be below rim {rim}");
    }

    #[test]
    fn ejecta_raises_terrain_outside_rim() {
        let b = sample_basin();
        let e = basin_ejecta(1.2, 0.0, &b);
        assert!(e > 0.0, "ejecta should be positive outside rim, got {e}");
    }

    #[test]
    fn ejecta_vanishes_at_max_extent() {
        let b = sample_basin();
        let e = basin_ejecta(EJECTA_MAX_T, 0.0, &b);
        assert!(e.abs() < 1.0, "ejecta should vanish at t=EJECTA_MAX_T, got {e}");
    }

    #[test]
    fn rim_seam_is_continuous() {
        // Interior at t=1 and ejecta at t=1 should agree within a small fraction
        // of the rim height so there is no visible seam.
        let b = sample_basin();
        let interior = basin_interior(0.999, &b);
        let ejecta = basin_ejecta(1.001, 0.0, &b);
        let rim_h = b.def.depth_m * 0.12;
        assert!(
            (interior - ejecta).abs() < rim_h * 0.2,
            "rim seam discontinuous: interior={interior} ejecta={ejecta} rim_h={rim_h}"
        );
    }

    #[test]
    fn peak_ring_raises_midbasin() {
        let b = sample_basin();
        let h_peak = basin_interior(0.5, &b);
        let h_floor = basin_interior(0.2, &b);
        assert!(h_peak > h_floor, "peak ring at 0.5 should be above floor at 0.2");
    }
}
