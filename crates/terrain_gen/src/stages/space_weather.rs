use glam::Vec3;
use rayon::prelude::*;
use serde::Deserialize;

use super::MAT_MARE;
use super::util::for_face_texels_in_cap;
use crate::body_builder::BodyBuilder;
use crate::cubemap::{CubemapFace, face_uv_to_dir};
use crate::noise::fbm3;
use crate::seeding::Rng;
use crate::stage::Stage;
use crate::types::{BiomeParams, Crater};

/// Visual freshness of a young crater, normalized to a freshness threshold.
///
/// Returns 1.0 at age 0, 0.0 at `threshold_gyr`, with a quadratic falloff
/// so the brightest craters dominate visually but a few intermediate-age
/// craters still contribute a faint halo.
///
/// Note: this is intentionally NOT the log-kinetics maturity curve from
/// Lucey 2000. The log curve collapses to near-mature within ~50 Myr,
/// which means by the time a crater is even 5% of the threshold age it's
/// already invisible. We want a population of visibly fresh craters to
/// remain when the user looks at any face of the body, so we trade
/// physical accuracy for visual readability.
fn freshness(age_gyr: f32, threshold_gyr: f32) -> f32 {
    if age_gyr >= threshold_gyr || threshold_gyr <= 0.0 {
        return 0.0;
    }
    let t = 1.0 - age_gyr / threshold_gyr;
    t * t
}

/// Authors the final albedo cubemap using biome palette + crater weathering.
///
/// Pipeline:
/// 1. **Base albedo**: per-biome mature albedo modulated by that biome's own
///    low-freq tonal variation.
/// 2. **Per-crater signatures**: all baked craters get floor-darken, rim-
///    brighten, and r⁻³ ejecta halo. Strength has a persistent floor so
///    ancient craters still read strongly.
/// 3. **Young-crater halo**: extra brightening for recent impacts.
/// 4. **Ray systems**: radial bright streaks on the youngest craters.
///
/// Maria (if present) use a single fixed dark palette; highlands pull from
/// their per-biome parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct SpaceWeather {
    /// Optional override for non-mare texels. When present, forces every
    /// highland biome to use this mature albedo (grayscale) instead of the
    /// per-biome value. Kept for backward compatibility with bodies that
    /// don't yet have a rich multi-biome config — Mira-style bodies should
    /// leave this `None` and drive color from biome palette.
    #[serde(default)]
    pub highland_mature_albedo: Option<f32>,
    /// Optional override for the fresh-surface target on non-mare texels.
    #[serde(default)]
    pub highland_fresh_albedo: Option<f32>,
    /// Mare mature/fresh albedos — used for any texel tagged MAT_MARE.
    pub mare_mature_albedo: f32,
    pub mare_fresh_albedo: f32,
    /// RGB tint applied multiplicatively to mare texels. Default is
    /// neutral gray (lunar mare). Bodies with compositionally distinct
    /// flood-plain material (olivine-rich, carbonaceous, etc.) can set a
    /// chromatic tint here so the smooth plains read in their own color
    /// instead of a neutral darkening.
    #[serde(default = "default_mare_tint")]
    pub mare_tint: [f32; 3],
    /// Craters younger than this (Gyr) get fresh-ejecta brightening.
    pub young_crater_age_threshold: f32,
    /// Craters younger than this get ray systems.
    pub ray_age_threshold: f32,
    /// How far rays extend, in multiples of crater radius.
    pub ray_extent_radii: f32,
    /// Number of ray arms per crater.
    pub ray_count_per_crater: u32,
    /// Angular half-width of each ray arm (radians).
    pub ray_half_width: f32,
}

fn default_mare_tint() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

/// Precomputed ray data for a single rayed crater.
struct RayedCrater {
    center: Vec3,
    radius_m: f32,
    age_frac: f32,
    ray_angles: Vec<f32>,
    uprange_az: f32,
    exclusion_half: f32,
    tangent_right: Vec3,
    tangent_forward: Vec3,
}

#[inline]
fn biome_fresh(b: &BiomeParams) -> f32 {
    b.fresh_albedo.unwrap_or(b.albedo * 1.9)
}

/// SplitMix64 — used here to hash a crater center direction into a
/// per-crater pond pattern phase pair without pulling in `cratering.rs`.
#[inline]
fn sw_splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Stage for SpaceWeather {
    fn name(&self) -> &str {
        "space_weather"
    }
    fn dependencies(&self) -> &[&str] {
        &["cratering"]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        let res = builder.cubemap_resolution;
        let body_radius = builder.radius_m;
        let seed = builder.stage_seed();
        let bake_threshold = builder.cubemap_bake_threshold_m;

        // Biomes are optional: bodies without a Biomes stage fall back to the
        // legacy single-highland path via `highland_*_albedo` overrides. We
        // synthesize a neutral default biome so the per-texel lookup path
        // still works without a second code path.
        let biomes = if builder.biomes.is_empty() {
            vec![BiomeParams {
                name: "default".to_string(),
                albedo: self.highland_mature_albedo.unwrap_or(0.12),
                fresh_albedo: self.highland_fresh_albedo,
                tint: [1.0, 1.0, 1.0],
                tonal_amp: 0.18,
                roughness: 0.8,
            }]
        } else {
            builder.biomes.clone()
        };

        // Classify young and rayed craters.
        let young_craters: Vec<&Crater> = builder
            .craters
            .iter()
            .filter(|c| c.age_gyr < self.young_crater_age_threshold)
            .collect();

        let mut rng = Rng::new(seed);
        let rayed_craters: Vec<RayedCrater> = builder
            .craters
            .iter()
            .filter(|c| c.age_gyr < self.ray_age_threshold)
            .map(|c| {
                let center = c.center.normalize();
                let up = if center.y.abs() < 0.9 {
                    Vec3::Y
                } else {
                    Vec3::X
                };
                let right = center.cross(up).normalize();
                let forward = right.cross(center).normalize();

                // Variable ray count: 8-24 scaling with crater size (doc §4).
                let ray_count = ((8.0 + c.radius_m / 10_000.0 * 8.0) as u32).clamp(8, 24);
                let ray_angles: Vec<f32> = (0..ray_count)
                    .map(|_| rng.next_f64() as f32 * std::f32::consts::TAU)
                    .collect();

                let uprange_az = rng.next_f64() as f32 * std::f32::consts::TAU;
                let exclusion_half =
                    rng.range_f64(std::f64::consts::FRAC_PI_6, std::f64::consts::FRAC_PI_2) as f32;

                let age_frac = c.age_gyr / self.ray_age_threshold;

                RayedCrater {
                    center,
                    radius_m: c.radius_m,
                    age_frac,
                    ray_angles,
                    uprange_az,
                    exclusion_half,
                    tangent_right: right,
                    tangent_forward: forward,
                }
            })
            .collect();

        let mare_mature = self.mare_mature_albedo;
        let mare_fresh = self.mare_fresh_albedo;
        let mare_tint = self.mare_tint;
        let hl_mature_override = self.highland_mature_albedo;
        let hl_fresh_override = self.highland_fresh_albedo;
        let ray_half_width = self.ray_half_width;
        let ray_extent_radii = self.ray_extent_radii;
        let young_threshold = self.young_crater_age_threshold;

        // Per-crater albedo signature for ALL baked craters (not just young).
        // Real lunar craters retain visible color contrast for billions of
        // years: brighter rim/upper wall (continually micro-exposed bedrock),
        // darker floor (impact melt + dust pond), brighter ejecta apron.
        // PERSISTENCE_FLOOR controls how much of the fresh contrast ancient
        // craters retain.
        const PERSISTENCE_FLOOR: f32 = 0.55;
        // Weights for the radial zones. Floor darkening reads strongly, rim
        // crest is the primary visible feature.
        const FLOOR_MAX: f32 = 0.85;
        const RIM_MAX: f32 = 1.15;
        const EJECTA_MAX: f32 = 0.75;
        // Rim band width (fraction of radius either side of t=1).
        const RIM_HALF: f32 = 0.28;

        let baked_craters: Vec<&Crater> = builder
            .craters
            .iter()
            .filter(|c| c.radius_m >= bake_threshold)
            .collect();

        let young_ref = &young_craters;
        let rayed_ref = &rayed_craters;
        let baked_ref = &baked_craters;
        let material_cubemap = &builder.material_cubemap;
        let biome_map = &builder.biome_map;
        let basin_field = &builder.basin_albedo_field;
        let biomes_ref = &biomes;

        // Biome lookup with bounds safety — the Biomes stage may have painted
        // with an id ≥ biomes.len() if configured wrongly; clamp.
        let lookup_biome = |face: CubemapFace, x: u32, y: u32| -> &BiomeParams {
            let id = biome_map.get(face, x, y) as usize;
            &biomes_ref[id.min(biomes_ref.len() - 1)]
        };

        builder
            .albedo_contributions
            .albedo
            .faces_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(face_idx, slice)| {
                let face = CubemapFace::ALL[face_idx];

                // ── Pass 1: biome-aware base albedo with tonal variation ──
                //
                // Highland biomes carry their own `tonal_amp`; each biome
                // uses a private noise channel so neighboring biomes don't
                // share a pattern.
                const TONAL_FREQ: f64 = 3.5;
                let inv_res = 1.0 / res as f32;
                for y in 0..res {
                    for x in 0..res {
                        let idx = (y * res + x) as usize;
                        let is_mare = material_cubemap.get(face, x, y) == MAT_MARE as u8;
                        let (base_gray, tint) = if is_mare {
                            (mare_mature, mare_tint)
                        } else {
                            let biome = lookup_biome(face, x, y);
                            let biome_base = hl_mature_override.unwrap_or(biome.albedo);
                            let u = (x as f32 + 0.5) * inv_res;
                            let v = (y as f32 + 0.5) * inv_res;
                            let dir = face_uv_to_dir(face, u, v).normalize();
                            // Biome-private noise channel: xor the biome id
                            // into the seed so each biome gets an independent
                            // tonal field rather than sharing one pattern.
                            let bs = seed
                                ^ 0xA17B_5C2D_4E9F_1357
                                ^ ((biome_map.get(face, x, y) as u64)
                                    .wrapping_mul(0xB3F9_4C78_CE32_1A5D));
                            let n = fbm3(
                                dir.x as f64 * TONAL_FREQ,
                                dir.y as f64 * TONAL_FREQ,
                                dir.z as f64 * TONAL_FREQ,
                                bs,
                                4,
                                0.55,
                                2.1,
                            ) as f32;
                            let scale = 1.0 + biome.tonal_amp * n;
                            // Fold in megabasin albedo field: interior
                            // darkening and ejecta streaks ride on top of
                            // the biome base albedo so basin boundaries are
                            // soft/fractal instead of hard biome discs.
                            let bf = basin_field.get(face, x, y).clamp(-0.75, 0.5);
                            let basin_mul = (1.0 + bf).max(0.0);
                            ((biome_base * scale * basin_mul).max(0.0), biome.tint)
                        };
                        slice[idx] = [
                            base_gray * tint[0],
                            base_gray * tint[1],
                            base_gray * tint[2],
                            1.0,
                        ];
                    }
                }

                // ── Pass 1.5: per-crater albedo signature ──
                //
                // Three overlapping radial zones author a signed delta that
                // lerps toward either a `bright` or `dark` target chosen from
                // the underlying biome's fresh albedo.
                for crater in baked_ref {
                    let center = crater.center.normalize();
                    let radius_m = crater.radius_m;
                    // Extend sample cap to 2.5R (floor + rim + ejecta apron).
                    let cap_angle = (radius_m * 2.5) / body_radius;
                    let f = freshness(crater.age_gyr, young_threshold);
                    let strength = PERSISTENCE_FLOOR + (1.0 - PERSISTENCE_FLOOR) * f;

                    // Per-crater impact-melt pond hash. Real complex craters
                    // show dark melt-pool patches on the floor (Copernicus,
                    // Tycho); modelled here as low-freq angular + radial
                    // noise that boosts the floor-darken in scattered
                    // patches rather than uniformly across the floor.
                    let tangent_basis_up = if center.y.abs() < 0.9 {
                        Vec3::Y
                    } else {
                        Vec3::X
                    };
                    let crater_tangent = tangent_basis_up.cross(center).normalize();
                    let crater_bitangent = center.cross(crater_tangent);
                    let pond_seed = sw_splitmix64(
                        (center.x.to_bits() as u64)
                            ^ ((center.y.to_bits() as u64) << 17)
                            ^ ((center.z.to_bits() as u64) << 33),
                    );
                    let pond_phase_a =
                        (pond_seed as u32 as f32 / u32::MAX as f32) * std::f32::consts::TAU;
                    let pond_phase_b =
                        ((pond_seed >> 32) as u32 as f32 / u32::MAX as f32) * std::f32::consts::TAU;
                    let is_complex = radius_m >= 15_000.0;

                    for_face_texels_in_cap(
                        face,
                        res,
                        center,
                        cap_angle,
                        |x, y, dir, angular_dist| {
                            let surface_dist = angular_dist * body_radius;
                            let t = surface_dist / radius_m;

                            let mut delta = 0.0_f32;
                            // Floor darken — linear falloff from center to 0.55 R.
                            if t < 0.55 {
                                let base = 1.0 - t / 0.55;
                                let pond_boost = if is_complex {
                                    let proj = dir - center * dir.dot(center);
                                    let theta =
                                        proj.dot(crater_bitangent).atan2(proj.dot(crater_tangent));
                                    let n_az = (3.0 * theta + pond_phase_a).sin()
                                        + 0.5 * (5.0 * theta + pond_phase_b).sin();
                                    let n_r = (7.0 * t + pond_phase_a * 1.7).sin();
                                    let pond = (n_az + n_r) * 0.4;
                                    // Bias so only sparse patches darken.
                                    (pond - 0.15).max(0.0) * 0.7
                                } else {
                                    0.0
                                };
                                delta -= FLOOR_MAX * base * (1.0 + pond_boost);
                            }
                            // Rim crest brighten — wider triangular band at t=1.
                            let lo = 1.0 - RIM_HALF;
                            let hi = 1.0 + RIM_HALF;
                            if t >= lo && t <= hi {
                                let rim_w = 1.0 - (t - 1.0).abs() / RIM_HALF;
                                delta += RIM_MAX * rim_w;
                            }
                            // Ejecta apron with r⁻³ falloff, extended to 2.5 R.
                            if t > 1.0 && t < 2.5 {
                                let apron = t.powi(-3);
                                let fade = ((2.5 - t) / 1.5).clamp(0.0, 1.0);
                                delta += EJECTA_MAX * apron * fade;
                            }

                            if delta.abs() < 1e-3 {
                                return;
                            }

                            let idx = (y * res + x) as usize;
                            let current = slice[idx];
                            let is_mare = material_cubemap.get(face, x, y) == MAT_MARE as u8;
                            let (bright, dark, tint) = if is_mare {
                                (mare_fresh, mare_mature * 0.75, mare_tint)
                            } else {
                                let b = lookup_biome(face, x, y);
                                let mature = hl_mature_override.unwrap_or(b.albedo);
                                let fresh = hl_fresh_override.unwrap_or_else(|| biome_fresh(b));
                                (fresh, mature * 0.55, b.tint)
                            };
                            let target_gray = if delta >= 0.0 { bright } else { dark };
                            let mix = (delta.abs() * strength).min(1.0);
                            let target = [
                                target_gray * tint[0],
                                target_gray * tint[1],
                                target_gray * tint[2],
                            ];
                            slice[idx] = [
                                current[0] + (target[0] - current[0]) * mix,
                                current[1] + (target[1] - current[1]) * mix,
                                current[2] + (target[2] - current[2]) * mix,
                                1.0,
                            ];
                        },
                    );
                }

                // ── Pass 2: young-crater ejecta halo (additional brightening) ──
                for crater in young_ref {
                    let center = crater.center.normalize();
                    let ejecta_radius_m = crater.influence_radius_m();
                    let ejecta_angle = ejecta_radius_m / body_radius;
                    let f = freshness(crater.age_gyr, young_threshold);
                    let radius_m = crater.radius_m;

                    for_face_texels_in_cap(
                        face,
                        res,
                        center,
                        ejecta_angle,
                        |x, y, _dir, angular_dist| {
                            let surface_dist = angular_dist * body_radius;
                            let t_crater = surface_dist / radius_m;
                            let distance_falloff = if t_crater <= 1.0 {
                                1.0
                            } else {
                                t_crater.powi(-2).min(1.0)
                            };
                            let strength = f * distance_falloff;
                            if strength > 0.005 {
                                let idx = (y * res + x) as usize;
                                let current = slice[idx];
                                let is_mare = material_cubemap.get(face, x, y) == MAT_MARE as u8;
                                let (fresh_v, tint) = if is_mare {
                                    (mare_fresh, mare_tint)
                                } else {
                                    let b = lookup_biome(face, x, y);
                                    let fresh = hl_fresh_override.unwrap_or_else(|| biome_fresh(b));
                                    (fresh, b.tint)
                                };
                                let target =
                                    [fresh_v * tint[0], fresh_v * tint[1], fresh_v * tint[2]];
                                slice[idx] = [
                                    current[0] + (target[0] - current[0]) * strength,
                                    current[1] + (target[1] - current[1]) * strength,
                                    current[2] + (target[2] - current[2]) * strength,
                                    1.0,
                                ];
                            }
                        },
                    );
                }

                // ── Pass 3: ray systems ──
                for rc in rayed_ref {
                    let ray_extent_m = rc.radius_m * ray_extent_radii;
                    let ray_angle = ray_extent_m / body_radius;
                    let ray_fade = 1.0 - rc.age_frac;

                    for_face_texels_in_cap(
                        face,
                        res,
                        rc.center,
                        ray_angle,
                        |x, y, dir, angular_dist| {
                            let surface_dist = angular_dist * body_radius;
                            if surface_dist < rc.radius_m {
                                return;
                            }

                            let to_texel = (dir - rc.center * dir.dot(rc.center)).normalize();
                            let az_x = to_texel.dot(rc.tangent_right);
                            let az_y = to_texel.dot(rc.tangent_forward);
                            let azimuth = az_y.atan2(az_x);

                            // Soft ray profile: minimum angular distance to
                            // any ray arm, then a Gaussian-ish falloff across
                            // the arm's half-width. Gives crisp-but-not-hard
                            // ray edges that match real rayed craters.
                            let mut min_delta = std::f32::consts::PI;
                            for &ray_az in &rc.ray_angles {
                                let mut delta = (azimuth - ray_az).abs();
                                if delta > std::f32::consts::PI {
                                    delta = std::f32::consts::TAU - delta;
                                }
                                if delta < min_delta {
                                    min_delta = delta;
                                }
                            }
                            if min_delta >= ray_half_width * 1.6 {
                                return;
                            }
                            let w = (1.0 - min_delta / (ray_half_width * 1.6)).max(0.0);
                            let arm_w = w * w * (3.0 - 2.0 * w);

                            let mut delta_up = (azimuth - rc.uprange_az).abs();
                            if delta_up > std::f32::consts::PI {
                                delta_up = std::f32::consts::TAU - delta_up;
                            }
                            if delta_up < rc.exclusion_half {
                                return;
                            }

                            let dist_from_rim =
                                (surface_dist - rc.radius_m) / (ray_extent_m - rc.radius_m);
                            let brightness =
                                ray_fade * (1.0 - dist_from_rim).max(0.0) * arm_w * 0.85;
                            if brightness > 0.001 {
                                let idx = (y * res + x) as usize;
                                let current = slice[idx];
                                let biome = lookup_biome(face, x, y);
                                let fresh_v =
                                    hl_fresh_override.unwrap_or_else(|| biome_fresh(biome));
                                let target = [
                                    fresh_v * biome.tint[0],
                                    fresh_v * biome.tint[1],
                                    fresh_v * biome.tint[2],
                                ];
                                slice[idx] = [
                                    current[0] + (target[0] - current[0]) * brightness,
                                    current[1] + (target[1] - current[1]) * brightness,
                                    current[2] + (target[2] - current[2]) * brightness,
                                    1.0,
                                ];
                            }
                        },
                    );
                }
            });
    }
}
