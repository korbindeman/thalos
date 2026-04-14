use glam::Vec3;
use rayon::prelude::*;
use serde::Deserialize;

use super::MAT_MARE;
use super::noise_fbm::{GradientWarp, SsFbm, bulk_sample};
use super::util::{face_may_intersect_cap, face_position_arrays, for_face_texels_in_cap};
use crate::body_builder::BodyBuilder;
use crate::crater_profile::degradation_factor;
use crate::cubemap::CubemapFace;
use crate::seeding::{Rng, sub_seed};
use crate::stage::Stage;

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

    /// Oceanus-Procellarum-scale flooding. When enabled, a second pass
    /// floods a large irregular near-side region not tied to any single
    /// basin — modeling the real Moon's Procellarum, which is a thin-crust
    /// province rather than a basin. Set coverage to 0 to disable.
    #[serde(default)]
    pub procellarum: Option<ProcellarumConfig>,
}

/// Parameters for the Procellarum-style near-side flood pass.
#[derive(Debug, Clone, Deserialize)]
pub struct ProcellarumConfig {
    /// Absolute fill level in meters (height below which near-side texels
    /// inside the mask become mare). Use a value between the near-side
    /// highland floor and zero; negative values trace the lowlands.
    pub fill_level_m: f32,
    /// Half-angle (radians) of the near-side cosine window. Texels with
    /// `near·dot > cos(half_angle)` are eligible. Use ~1.2 rad (≈70°) to
    /// cover most of the near hemisphere, less to confine tighter.
    pub near_side_half_angle_rad: f32,
    /// Amplitude of the low-frequency mask field. The mask must exceed
    /// `mask_threshold` for a texel to flood, so this controls the
    /// irregularity of the boundary. Use ~0.4.
    pub mask_threshold: f32,
    /// Frequency of the mask field on the sphere. ~1.2 gives one or two
    /// large connected blobs; larger values fragment the coverage.
    pub mask_frequency: f64,
    /// Additional boundary fuzz amplitude in meters (threshold modulation).
    pub boundary_noise_amplitude_m: f32,
}

impl Stage for MareFlood {
    fn name(&self) -> &str {
        "mare_flood"
    }
    fn dependencies(&self) -> &[&str] {
        &["cratering"]
    }

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

            // Sample rim at the basin's own radius, averaged over 16 samples
            // around the ring. Using a fixed 250 km for all basins (as a
            // previous revision did) hit the basin interior for larger basins
            // and the ejecta blanket for smaller ones — so fill levels were
            // systematically wrong.
            let rim_angle = (basin.radius_m * 0.95) / body_radius;
            let rim_h =
                sample_ring_height(&builder.height_contributions.height, center, rim_angle, 16);

            let fill_level = floor_h + self.fill_fraction * (rim_h - floor_h);
            // Search cap extends to ~1.2 R so the flood can reach embayments
            // beyond the nominal rim without running off into the ejecta.
            let search_radius_m = basin.radius_m * 1.2;

            targets.push(FloodTarget {
                center,
                search_radius_m,
                fill_level,
            });
        }

        // Additional large craters on the near side.
        let mut near_side_craters: Vec<(usize, f32)> = builder
            .craters
            .iter()
            .enumerate()
            .filter(|(_, c)| c.center.dot(near) > (1.0 - self.near_side_bias).max(0.0))
            .map(|(i, c)| (i, c.radius_m))
            .collect();
        near_side_craters.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for &(idx, _) in near_side_craters
            .iter()
            .take(self.additional_crater_count as usize)
        {
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
            if floor_h >= rim_h {
                continue;
            }

            let fill_level = floor_h + self.fill_fraction * (rim_h - floor_h);
            let search_radius_m = crater.radius_m * 1.5;

            targets.push(FloodTarget {
                center,
                search_radius_m,
                fill_level,
            });
        }

        // Flood each target over multiple sequential episodes.
        //
        // Each episode uses a different noise seed and a slightly higher fill level
        // to simulate distinct eruption pulses. The last episode reaches the
        // canonical fill_level. Ghost craters (pre-mare craters with rims above the
        // fill level) emerge naturally: their interiors are flooded but their rims
        // survive.
        //
        // The boundary shape is driven by two layers of noise with domain warping:
        // a low-frequency "embayment" field that curves the coastline by multiple
        // km, plus a high-frequency field that breaks up any remaining straight
        // edges. The total amplitude is comparable to the basin's rim-to-floor
        // height so the mare outline actually departs from the rim contour.
        let mut episode_rng = Rng::new(sub_seed(seed, "episodes"));
        let boundary_noise_amplitude_m = self.boundary_noise_amplitude_m;
        for target in &targets {
            let search_angle = target.search_radius_m / body_radius;

            for ep in 0..self.episode_count {
                let ep_frac = (ep + 1) as f32 / self.episode_count as f32;
                let ep_fill = target.fill_level * (0.85 + 0.15 * ep_frac);
                let ep_seed_u = episode_rng.next_u64();
                let ep_seed = ep_seed_u as i32;
                let relief_seed = sub_seed(ep_seed_u, "relief") as i32;

                let lo_freq = (self.boundary_noise_freq * 0.6) as f32;
                let hi_freq = (self.boundary_noise_freq * (3.5 + ep as f64 * 0.5)) as f32;
                let warp_freq = (self.boundary_noise_freq * 0.25) as f32;
                let target_center = target.center;

                // Boundary noise: warped combination of a large-scale
                // embayment field (low freq) and a detail edge field
                // (high freq). Output range ≈ [-1, 1].
                let boundary_fbm_lo = SsFbm::new(ep_seed, 0.55, 4, 2.0);
                let boundary_fbm_hi = SsFbm::new(ep_seed.wrapping_add(0x9E37_79B9u32 as i32), 0.5, 3, 2.0);
                let boundary_warp = GradientWarp::new(
                    ep_seed.wrapping_add(0x517C_C1B7u32 as i32),
                    0.02,
                    warp_freq,
                );

                // Surface relief: small-amplitude simplex texture kept on
                // the mare surface so it isn't perfectly flat.
                let relief_fbm = SsFbm::new(relief_seed, 0.5, 3, 2.0);
                let relief_scale = hi_freq * 1.7;

                let boundary_fbm_lo_ref = &boundary_fbm_lo;
                let boundary_fbm_hi_ref = &boundary_fbm_hi;
                let boundary_warp_ref = &boundary_warp;
                let relief_fbm_ref = &relief_fbm;

                let h_faces = builder.height_contributions.height.faces_mut();
                let m_faces = builder.material_cubemap.faces_mut();
                h_faces
                    .par_iter_mut()
                    .zip(m_faces.par_iter_mut())
                    .enumerate()
                    .for_each(|(face_idx, (h_slice, m_slice))| {
                        let face = CubemapFace::ALL[face_idx];
                        if !face_may_intersect_cap(face, target_center, search_angle) {
                            return;
                        }

                        // Bulk sample the whole face once per episode.
                        let n = (res * res) as usize;
                        let (xs, ys, zs) = face_position_arrays(face, res);
                        let mut noise_out = vec![0.0f32; n];
                        let mut relief_out = vec![0.0f32; n];
                        bulk_sample(&mut noise_out, &xs, &ys, &zs, |dir| {
                            let pw = boundary_warp_ref.warp(dir);
                            0.7 * boundary_fbm_lo_ref.sample(pw * lo_freq)
                                + 0.3 * boundary_fbm_hi_ref.sample(pw * hi_freq)
                        });
                        bulk_sample(&mut relief_out, &xs, &ys, &zs, |dir| {
                            relief_fbm_ref.sample(dir * relief_scale)
                        });

                        for_face_texels_in_cap(
                            face,
                            res,
                            target_center,
                            search_angle,
                            |x, y, _dir, _dist| {
                                let idx = (y * res + x) as usize;
                                let current_h = h_slice[idx];
                                let threshold =
                                    ep_fill + noise_out[idx] * boundary_noise_amplitude_m;
                                if current_h < threshold {
                                    let new_h = ep_fill + relief_out[idx] * 25.0;
                                    h_slice[idx] = new_h;
                                    m_slice[idx] = MAT_MARE as u8;
                                }
                            },
                        );
                    });
            }
        }

        // Procellarum-scale near-side flood (optional).
        //
        // Floods a large irregular region of the near side defined by a
        // low-frequency mask noise AND an altitude threshold AND a
        // near-side cosine window. Models the real Moon's Oceanus
        // Procellarum, which is a thin-crust province covering much of
        // the near side rather than a single impact basin. Unlike the
        // basin-driven pass above, this pass doesn't follow any rim
        // contour — its boundary is entirely controlled by the mask.
        if let Some(proc) = &self.procellarum {
            let proc_seed = sub_seed(seed, "procellarum");
            let mask_seed = proc_seed as i32;
            let fuzz_seed = sub_seed(proc_seed, "fuzz") as i32;
            let surface_seed = sub_seed(proc_seed, "surface") as i32;

            let cos_window = proc.near_side_half_angle_rad.cos();
            let mask_freq = proc.mask_frequency as f32;
            let mask_threshold = proc.mask_threshold;
            let fill_level_m = proc.fill_level_m;
            let proc_boundary_amp = proc.boundary_noise_amplitude_m;

            // Low-freq mask, domain-warped for organic shapes.
            let mask_fbm = SsFbm::new(mask_seed, 0.55, 5, 2.0);
            let mask_warp = GradientWarp::new(
                mask_seed.wrapping_add(0x517C_C1B7u32 as i32),
                0.03,
                mask_freq * 0.4,
            );
            // Threshold fuzz — mild noise on the altitude cutoff.
            let fuzz_fbm = SsFbm::new(fuzz_seed, 0.5, 3, 2.0);
            // Subtle surface relief on the flooded surface.
            let relief_fbm = SsFbm::new(surface_seed, 0.5, 3, 2.0);

            let mask_fbm_ref = &mask_fbm;
            let mask_warp_ref = &mask_warp;
            let fuzz_fbm_ref = &fuzz_fbm;
            let relief_fbm_ref = &relief_fbm;

            let h_faces = builder.height_contributions.height.faces_mut();
            let m_faces = builder.material_cubemap.faces_mut();
            h_faces
                .par_iter_mut()
                .zip(m_faces.par_iter_mut())
                .enumerate()
                .for_each(|(face_idx, (h_slice, m_slice))| {
                    let face = CubemapFace::ALL[face_idx];
                    let n = (res * res) as usize;
                    let (xs, ys, zs) = face_position_arrays(face, res);
                    let mut mask_out = vec![0.0f32; n];
                    let mut fuzz_out = vec![0.0f32; n];
                    let mut relief_out = vec![0.0f32; n];
                    bulk_sample(&mut mask_out, &xs, &ys, &zs, |dir| {
                        let p = dir * mask_freq;
                        let pw = mask_warp_ref.warp(p);
                        mask_fbm_ref.sample(pw)
                    });
                    bulk_sample(&mut fuzz_out, &xs, &ys, &zs, |dir| {
                        fuzz_fbm_ref.sample(dir * mask_freq * 3.0)
                    });
                    bulk_sample(&mut relief_out, &xs, &ys, &zs, |dir| {
                        relief_fbm_ref.sample(dir * mask_freq * 5.0)
                    });

                    for y in 0..res {
                        for x in 0..res {
                            let idx = (y * res + x) as usize;
                            let dir = Vec3::new(xs[idx], ys[idx], zs[idx]);

                            let near_dot = dir.dot(near);
                            if near_dot < cos_window {
                                continue;
                            }

                            let near_weight =
                                ((near_dot - cos_window) / (1.0 - cos_window)).clamp(0.0, 1.0);
                            let mask_effective = mask_out[idx] + (near_weight - 0.5) * 0.6;
                            if mask_effective < mask_threshold {
                                continue;
                            }

                            let threshold = fill_level_m + fuzz_out[idx] * proc_boundary_amp;
                            let current_h = h_slice[idx];
                            if current_h < threshold {
                                let new_h = fill_level_m + relief_out[idx] * 30.0;
                                h_slice[idx] = new_h;
                                m_slice[idx] = MAT_MARE as u8;
                            }
                        }
                    }
                });
        }

        // Wrinkle ridges: narrow compressional ridges that form as the mare basalt
        // cools and contracts, following roughly concentric arcs around each basin.
        // Real dimensions: ~40–80 m height, ~300 m half-width, segmented into
        // chains of sinuous ~3 km sections. They are the main topographic relief
        // inside otherwise flat maria.
        if self.wrinkle_ridges {
            let mut ridge_rng = Rng::new(sub_seed(seed, "wrinkle_ridges"));
            for target in &targets {
                let loop_count = 3 + (ridge_rng.next_u64() & 3) as u32;
                for _ in 0..loop_count {
                    let frac = 0.25 + ridge_rng.next_f64() as f32 * 0.55;
                    let ring_r_m = target.search_radius_m * frac;
                    let ring_angle = ring_r_m / body_radius;
                    let sigma_m = 300.0 + ridge_rng.next_f64() as f32 * 300.0;
                    let ridge_h = 40.0 + ridge_rng.next_f64() as f32 * 60.0;
                    let ridge_seed = ridge_rng.next_u64() as i32;
                    let seg_freq = 20.0 + ridge_rng.next_f64() as f32 * 30.0;

                    let cap_angle = ring_angle + (4.0 * sigma_m) / body_radius;
                    let target_center = target.center;

                    // Segment mask: fBm noise along the ridge curve, used as
                    // an on/off factor to break the ring into lobes.
                    let seg_fbm = SsFbm::new(ridge_seed, 0.5, 3, 2.0);
                    let seg_fbm_ref = &seg_fbm;

                    let material_cubemap = &builder.material_cubemap;
                    builder
                        .height_contributions
                        .height
                        .faces_mut()
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(face_idx, h_slice)| {
                            let face = CubemapFace::ALL[face_idx];
                            if !face_may_intersect_cap(face, target_center, cap_angle) {
                                return;
                            }
                            let n = (res * res) as usize;
                            let (xs, ys, zs) = face_position_arrays(face, res);
                            let mut seg_out = vec![0.0f32; n];
                            bulk_sample(&mut seg_out, &xs, &ys, &zs, |dir| {
                                seg_fbm_ref.sample(dir * seg_freq)
                            });

                            for_face_texels_in_cap(
                                face,
                                res,
                                target_center,
                                cap_angle,
                                |x, y, _dir, dist| {
                                    if material_cubemap.get(face, x, y) != MAT_MARE as u8 {
                                        return;
                                    }
                                    let dr_m = (dist * body_radius) - ring_r_m;
                                    if dr_m.abs() > sigma_m * 4.0 {
                                        return;
                                    }
                                    let idx = (y * res + x) as usize;
                                    let seg_factor = (seg_out[idx] - 0.1).max(0.0) * 1.5;
                                    if seg_factor <= 0.0 {
                                        return;
                                    }
                                    let gauss = (-(dr_m * dr_m) / (2.0 * sigma_m * sigma_m)).exp();
                                    let add = ridge_h * gauss * seg_factor;
                                    if add > 0.5 {
                                        h_slice[idx] += add;
                                    }
                                },
                            );
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
    let up = if center.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
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
