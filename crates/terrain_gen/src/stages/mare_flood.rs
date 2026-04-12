use glam::Vec3;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::crater_profile::degradation_factor;
use crate::noise::fbm3;
use crate::seeding::{sub_seed, Rng};
use crate::stage::Stage;
use super::util::for_texels_in_cap;
use super::MAT_MARE;

/// A candidate region for mare flooding.
#[derive(Clone, Debug)]
struct FloodTarget {
    center: Vec3,
    search_radius_m: f32,
    fill_level: f32,
}

/// Floods the deepest basins on the near side with mare basalt.
///
/// Creates the mare/highland dichotomy: dark volcanic fills in the lowest
/// basins on the near side, while the far side remains bright highland.
/// This is the defining tidal asymmetry feature.
///
/// Runs multiple flooding episodes at slightly different noise-modulated fill
/// levels to simulate sequential eruptions (Hiesinger et al. 2011: lunar mare
/// resurfacing spans ~1 Gyr of eruption events). Ghost craters — pre-mare
/// craters whose rims protrude above the lava fill — emerge naturally from
/// age-ordered height stamping in the Cratering stage.
///
/// Reads `BodyBuilder::megabasins` for flood targets and
/// `BodyBuilder::tidal_axis` for the near-side reference. Bodies without
/// either won't flood anything.
#[derive(Debug, Clone, Deserialize)]
pub struct MareFlood {
    /// Maximum number of basins to flood, picking the deepest first.
    pub target_count: u32,
    /// How many additional large craters to flood beyond the megabasins.
    pub additional_crater_count: u32,
    /// How high the lava reaches relative to basin depth (0.0-1.0).
    pub fill_fraction: f32,
    /// Near-side preference strength (0.0 = uniform, 1.0 = exclusively near).
    pub near_side_bias: f32,
    /// Amplitude of boundary noise in meters.
    pub boundary_noise_amplitude_m: f32,
    /// Frequency multiplier for boundary noise.
    pub boundary_noise_freq: f64,
    /// Number of sequential flooding episodes per target.
    /// Each episode uses a slightly different fill level and noise seed,
    /// producing the irregular embayments and lava-flow lobes seen in real maria.
    pub episode_count: u32,
    /// Whether to add post-flood wrinkle ridges inside the maria.
    pub wrinkle_ridges: bool,
}

impl Stage for MareFlood {
    fn name(&self) -> &str { "mare_flood" }
    fn dependencies(&self) -> &[&str] { &["cratering"] }

    fn apply(&self, builder: &mut BodyBuilder) {
        let body_radius = builder.radius_m;
        let res = builder.cubemap_resolution;
        let seed = builder.stage_seed();
        // Bodies without a tidal axis don't get mare flooding (no near/far asymmetry).
        let Some(near) = builder.tidal_axis else {
            return;
        };

        // Pick the deepest basins on the near side, up to target_count.
        let near_threshold = (1.0 - self.near_side_bias).max(0.0);
        let mut basin_pool: Vec<&crate::stages::BasinDef> = builder
            .megabasins
            .iter()
            .filter(|b| b.center_dir.normalize().dot(near) >= near_threshold)
            .collect();
        basin_pool.sort_by(|a, b| b.depth_m.partial_cmp(&a.depth_m).unwrap());
        basin_pool.truncate(self.target_count as usize);

        // Collect flood targets: megabasins first, then largest near-side craters.
        let mut targets: Vec<FloodTarget> = Vec::new();

        // Megabasins: sample the height accumulator to find floor and rim levels.
        for basin in &basin_pool {
            let center = basin.center_dir.normalize();
            let floor_h = builder.height_contributions.height.sample_bilinear(center);

            // Sample rim: average height at ~250km angular distance (typical basin edge).
            // Use 8 sample points around the rim.
            let rim_angle = 250_000.0 / body_radius;
            let rim_h = sample_ring_height(
                &builder.height_contributions.height,
                center,
                rim_angle,
                8,
            );

            let fill_level = floor_h + self.fill_fraction * (rim_h - floor_h);
            // Search radius extends beyond the basin to catch embayments.
            let search_radius_m = 300_000.0;

            targets.push(FloodTarget { center, search_radius_m, fill_level });
        }

        // Additional large craters on the near side.
        let mut near_side_craters: Vec<(usize, f32)> = builder.craters.iter()
            .enumerate()
            .filter(|(_, c)| c.center.dot(near) > (1.0 - self.near_side_bias).max(0.0))
            .map(|(i, c)| (i, c.radius_m))
            .collect();
        near_side_craters.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for &(idx, _) in near_side_craters.iter().take(self.additional_crater_count as usize) {
            let crater = &builder.craters[idx];
            let center = crater.center.normalize();
            let floor_h = builder.height_contributions.height.sample_bilinear(center);

            // For craters, rim level is the pre-existing terrain around the rim.
            let rim_angle = crater.radius_m / body_radius;
            let rim_h = sample_ring_height(
                &builder.height_contributions.height,
                center,
                rim_angle * 1.2,
                8,
            );

            // Only flood if the crater is actually a depression.
            if floor_h >= rim_h { continue; }

            let fill_level = floor_h + self.fill_fraction * (rim_h - floor_h);
            let search_radius_m = crater.radius_m * 1.5;

            targets.push(FloodTarget { center, search_radius_m, fill_level });
        }

        // Flood each target over multiple sequential episodes.
        //
        // Each episode uses a different noise seed and a slightly higher fill level
        // to simulate distinct eruption pulses. The final episode sets the canonical
        // fill level; earlier episodes only mark texels that lie below that pulse's
        // threshold. Ghost craters (pre-mare craters with rims above the fill level)
        // emerge naturally: their interiors are flooded but their rims survive.
        let mut episode_rng = Rng::new(sub_seed(seed, "episodes"));
        for target in &targets {
            let search_angle = target.search_radius_m / body_radius;

            for ep in 0..self.episode_count {
                // Each episode slightly underfills relative to the final level,
                // varying by ±15% of the noise amplitude. The last episode reaches
                // the canonical fill_level to ensure complete flooding.
                let ep_frac = (ep + 1) as f32 / self.episode_count as f32;
                let ep_fill = target.fill_level * (0.85 + 0.15 * ep_frac);
                let ep_seed = episode_rng.next_u64();
                let ep_freq = self.boundary_noise_freq * (1.0 + ep as f64 * 0.3);

                for_texels_in_cap(res, target.center, search_angle, |face, x, y, dir, _dist| {
                    let current_h = builder.height_contributions.height.get(face, x, y);

                    // Boundary noise: makes mare edges irregular (domain-warped).
                    let noise = fbm3(
                        dir.x as f64 * ep_freq,
                        dir.y as f64 * ep_freq,
                        dir.z as f64 * ep_freq,
                        ep_seed,
                        4, 0.5, 2.0,
                    ) as f32;
                    let threshold = ep_fill + noise * self.boundary_noise_amplitude_m;

                    if current_h < threshold {
                        builder.height_contributions.height.set(face, x, y, ep_fill);
                        builder.material_cubemap.set(face, x, y, MAT_MARE as u8);
                    }
                });
            }
        }

        // Wrinkle ridges: narrow compressional ridges that form as the mare basalt
        // cools and contracts, following roughly concentric arcs around each basin.
        // Real dimensions: ~40–80 m height, ~300 m half-width, segmented into
        // chains of sinuous ~3 km sections. They are the main topographic relief
        // inside otherwise flat maria.
        if self.wrinkle_ridges {
            let mut ridge_rng = Rng::new(sub_seed(seed, "wrinkle_ridges"));
            for target in &targets {
                // 3–6 concentric ridge loops per mare.
                let loop_count = 3 + (ridge_rng.next_u64() & 3) as u32;
                for _ in 0..loop_count {
                    let frac = 0.25 + ridge_rng.next_f64() as f32 * 0.55; // 0.25–0.80 of search radius
                    let ring_r_m = target.search_radius_m * frac;
                    let ring_angle = ring_r_m / body_radius;
                    // Ridge half-width: ~300–600 m on the surface.
                    let sigma_m = 300.0 + ridge_rng.next_f64() as f32 * 300.0;
                    let ridge_h = 40.0 + ridge_rng.next_f64() as f32 * 60.0; // 40–100 m
                    let ridge_seed = ridge_rng.next_u64();
                    let seg_freq = 20.0 + ridge_rng.next_f64() as f32 * 30.0;

                    // Search cap: ring_angle plus a 4σ band for the Gaussian.
                    let cap_angle = ring_angle + (4.0 * sigma_m) / body_radius;
                    for_texels_in_cap(res, target.center, cap_angle, |face, x, y, dir, dist| {
                        // Only raise texels inside the mare.
                        if builder.material_cubemap.get(face, x, y) != MAT_MARE as u8 {
                            return;
                        }
                        let dr_m = (dist * body_radius) - ring_r_m;
                        if dr_m.abs() > sigma_m * 4.0 {
                            return;
                        }

                        // Segment along ridge length via fbm noise: positive
                        // regions become ridge segments, negative ones are gaps.
                        let seg = fbm3(
                            dir.x as f64 * seg_freq as f64,
                            dir.y as f64 * seg_freq as f64,
                            dir.z as f64 * seg_freq as f64,
                            ridge_seed,
                            3,
                            0.5,
                            2.0,
                        ) as f32;
                        let seg_factor = (seg - 0.1).max(0.0) * 1.5;
                        if seg_factor <= 0.0 {
                            return;
                        }

                        let gauss =
                            (-(dr_m * dr_m) / (2.0 * sigma_m * sigma_m)).exp();
                        let add = ridge_h * gauss * seg_factor;
                        if add > 0.5 {
                            builder.height_contributions.add_height(face, x, y, add);
                        }
                    });
                }
            }
        }

        // Cull buried craters and retag ghost craters.
        //
        // Any crater whose center texel is MAT_MARE has been covered by lava.
        // If its rim relief — after age-based diffusion, the same degradation
        // factor the Cratering stage applied when stamping height — is too
        // small to rise above the fill, it's geologically erased. Drop it from
        // the SSBO. Otherwise its rim still pokes above the basalt as a ghost
        // crater; retag its material_id so the renderer knows its interior is
        // mare.
        const GHOST_RIM_THRESHOLD_M: f32 = 10.0;
        builder.craters.retain_mut(|c| {
            let dir = c.center.normalize();
            let is_mare = builder.material_cubemap.sample_nearest(dir) == MAT_MARE as u8;
            if is_mare {
                let effective_rim = c.rim_height_m * degradation_factor(c.radius_m, c.age_gyr);
                if effective_rim < GHOST_RIM_THRESHOLD_M {
                    return false;
                }
                c.material_id = MAT_MARE;
            }
            true
        });
    }
}

/// Sample the average height on a ring at `angle` radians from `center`.
fn sample_ring_height(
    height: &crate::cubemap::Cubemap<f32>,
    center: Vec3,
    angle: f32,
    samples: u32,
) -> f32 {
    // Build a local tangent frame at `center`.
    let up = if center.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let right = center.cross(up).normalize();
    let forward = right.cross(center).normalize();

    let mut sum = 0.0f32;
    for i in 0..samples {
        let phi = std::f32::consts::TAU * i as f32 / samples as f32;
        let (sin_phi, cos_phi) = phi.sin_cos();
        let (sin_a, cos_a) = angle.sin_cos();
        let dir = center * cos_a + (right * cos_phi + forward * sin_phi) * sin_a;
        sum += height.sample_bilinear(dir.normalize());
    }
    sum / samples as f32
}
