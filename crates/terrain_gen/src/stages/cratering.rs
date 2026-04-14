use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::crater_profile::{
    crater_dimensions, crater_profile, degradation_factor, morphology_for_radius,
};
use crate::cubemap::CubemapFace;
use crate::seeding::Rng;
use crate::stage::Stage;
use crate::types::Crater;
use super::MAT_HIGHLAND;
use super::util::for_face_texels_in_cap_rows;

// ---------------------------------------------------------------------------
// Per-crater angular warping
// ---------------------------------------------------------------------------
//
// Real craters are not circular. To break the cookie-cutter look we sample an
// angular noise function `n(θ) ∈ [-1, 1]` in the crater's local tangent plane
// and use it to perturb the radial lookup before evaluating the (pure radial)
// profile from `crater_profile.rs`. A positive `n(θ)` means "effective radius
// at this azimuth is larger" → the rim, terrace boundaries, and floor all
// bulge outward together at that angle. The profile math stays untouched.
//
// Noise is a normalized sum of four sine harmonics with per-crater phases
// hashed from the crater center. Sum of amplitudes = 1.0, so |n| ≤ 1.

/// Max fractional radius perturbation. 0.08 ≈ ±8% warp — enough to break
/// perfect circles while keeping the morphometry recognizable.
const RADIAL_WARP_EPS: f32 = 0.08;

/// Max fractional rim-height perturbation. Real rims crenulate visibly;
/// ±25% gives a lively silhouette without ever dropping the crest below
/// the surrounding ejecta apron.
const RIM_WARP_EPS: f32 = 0.25;

/// Max terrace phase jitter, in units of one bench width. 0.4 shifts a
/// bench boundary by up to ~0.4 of its spacing, enough to desync adjacent
/// azimuths so walls read as asymmetric slump patterns.
const WALL_PHASE_EPS: f32 = 0.4;

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic 64-bit seed from a unit-sphere direction.
fn center_seed(c: glam::Vec3) -> u64 {
    let x = c.x.to_bits() as u64;
    let y = c.y.to_bits() as u64;
    let z = c.z.to_bits() as u64;
    splitmix64(x ^ (y << 13) ^ (z << 27) ^ 0xD1B5_4A32_D192_ED03)
}

/// Four independent phase offsets in [0, 2π) for the given channel.
fn phases_from_seed(seed: u64, channel: u64) -> [f32; 4] {
    let s0 = splitmix64(seed ^ channel.wrapping_mul(0xA076_1D64_78BD_642F));
    let s1 = splitmix64(s0);
    let s2 = splitmix64(s1);
    let s3 = splitmix64(s2);
    let tau = std::f32::consts::TAU;
    let to_phase = |s: u64| (s as u32 as f32 / u32::MAX as f32) * tau;
    [to_phase(s0), to_phase(s1), to_phase(s2), to_phase(s3)]
}

/// One uniform `f32` in [0, 1) from the given seed + channel.
fn unit_from_seed(seed: u64, channel: u64) -> f32 {
    let s = splitmix64(seed ^ channel.wrapping_mul(0xD1B5_4A32_D192_ED03));
    (s as u32 as f32) / (u32::MAX as f32 + 1.0)
}

/// Normalized sum of 4 sine harmonics in θ. Output range: [-1, 1].
#[inline]
fn angular_noise(theta: f32, phases: &[f32; 4]) -> f32 {
    0.45 * (theta + phases[0]).sin()
        + 0.28 * (2.0 * theta + phases[1]).sin()
        + 0.17 * (3.0 * theta + phases[2]).sin()
        + 0.10 * (5.0 * theta + phases[3]).sin()
}

/// Hermite smoothstep: 0 at edge0, 1 at edge1, smooth in between.
#[inline]
fn smoothstep01(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

/// Generates the full crater population via power-law SFD and bakes large
/// craters into the cubemap height field.
///
/// All craters are stored in `builder.craters` for the mid-frequency SSBO.
/// Craters above `cubemap_bake_threshold_m` are additionally rasterized
/// into the height accumulator for impostor-distance visibility.
///
/// Body age is read from `BodyBuilder::body_age_gyr` (single source).
#[derive(Debug, Clone, Deserialize)]
pub struct Cratering {
    pub total_count: u32,
    /// Cumulative SFD slope on the large-crater branch (N(>D) ∝ D⁻ᵅ).
    /// Lunar production function is ≈ 1.8–2.0 above the 1–2 km knee.
    pub sfd_slope: f64,
    /// Cumulative SFD slope on the small-crater branch (steeper). Lunar
    /// small-branch is ≈ 3.0–3.8. When set, the sampler uses a broken
    /// power law with the break at `sfd_break_radius_m`.
    #[serde(default)]
    pub sfd_slope_small: Option<f64>,
    /// Radius (meters) where the SFD transitions from small to large
    /// branch. Required when `sfd_slope_small` is set.
    #[serde(default)]
    pub sfd_break_radius_m: Option<f64>,
    pub min_radius_m: f64,
    pub max_radius_m: f64,
    /// Controls the age distribution skew toward older craters: age =
    /// body_age × (1 − u^age_bias). Larger values bias more strongly
    /// toward old. Lunar surfaces want ~2.0 (roughly half ancient,
    /// half distributed through later time), not 4.0 (which degrades
    /// small craters out of the visible population).
    #[serde(default = "default_age_bias")]
    pub age_bias: f64,
    pub cubemap_bake_threshold_m: f32,
    /// For crater parents at or above this radius, spawn a cluster of
    /// small secondary craters in the ejecta annulus (1–3 R). Produces
    /// visible clustering that random Poisson placement lacks. Set to 0
    /// (or a value above max_radius_m) to disable.
    #[serde(default = "default_secondary_parent_threshold")]
    pub secondary_parent_radius_m: f32,
    #[serde(default = "default_secondaries_per_parent")]
    pub secondaries_per_parent: u32,
}

fn default_age_bias() -> f64 { 2.0 }
fn default_secondary_parent_threshold() -> f32 { 50_000.0 }
fn default_secondaries_per_parent() -> u32 { 20 }

impl Stage for Cratering {
    fn name(&self) -> &str { "cratering" }
    fn dependencies(&self) -> &[&str] { &["differentiate"] }

    fn apply(&self, builder: &mut BodyBuilder) {
        let mut rng = Rng::new(builder.stage_seed());
        let body_radius = builder.radius_m;
        let body_age_gyr = builder.body_age_gyr;
        let age_bias = self.age_bias;

        // Closure that samples a crater radius from the configured SFD.
        // Uses broken power law if both small-branch fields are set,
        // otherwise falls back to the single-slope sampler.
        let sample_radius = |rng: &mut Rng| -> f64 {
            match (self.sfd_slope_small, self.sfd_break_radius_m) {
                (Some(alpha_small), Some(d_break))
                    if d_break > self.min_radius_m && d_break < self.max_radius_m =>
                {
                    rng.broken_power_law(
                        self.min_radius_m,
                        d_break,
                        self.max_radius_m,
                        alpha_small,
                        self.sfd_slope,
                    )
                }
                _ => rng.power_law(self.min_radius_m, self.max_radius_m, self.sfd_slope),
            }
        };

        // Generate primary crater population. ±15% size jitter breaks
        // cookie-cutter repetition (doc §1: "vary profile parameters by ±20%").
        //
        // Age distribution: bias toward old via age = body_age × (1 − u^age_bias).
        // The default age_bias = 2.0 yields ~65% of craters older than half
        // the body age — enough to read as ancient, gentle enough to leave
        // a healthy population of fresh small craters for texture.
        let primary_capacity =
            self.total_count as usize + self.secondaries_per_parent as usize * 64;
        let mut craters = Vec::with_capacity(primary_capacity);
        for _ in 0..self.total_count {
            let radius_m = sample_radius(&mut rng) as f32;
            let center = rng.unit_vector().as_vec3();

            let (base_depth, base_rim) = crater_dimensions(radius_m);
            let jitter = 1.0 + rng.next_f64_signed() as f32 * 0.15;
            let depth_m = base_depth * jitter;
            let rim_height_m = base_rim * jitter;

            let u = rng.next_f64();
            let age_gyr = (body_age_gyr as f64 * (1.0 - u.powf(age_bias))) as f32;

            craters.push(Crater {
                center,
                radius_m,
                depth_m,
                rim_height_m,
                age_gyr,
                material_id: MAT_HIGHLAND,
            });
        }

        // Secondary crater clustering: for each large parent, spawn a
        // cluster of smaller craters in its 1–3 R ejecta annulus. This
        // mimics secondary crater chains from ballistic ejecta and breaks
        // up the uniform-Poisson look. Secondaries inherit the parent's
        // age (they formed from the same impact) plus a tiny delta.
        let parent_threshold = self.secondary_parent_radius_m;
        let secondaries_per_parent = self.secondaries_per_parent as usize;
        if parent_threshold > 0.0 && secondaries_per_parent > 0 {
            let parent_indices: Vec<usize> = craters
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (c.radius_m >= parent_threshold).then_some(i))
                .collect();
            for &parent_idx in &parent_indices {
                let parent = craters[parent_idx].clone();
                let parent_center = parent.center.normalize();
                // Tangent basis at the parent center for angular offsets.
                let up = if parent_center.x.abs() < 0.9 {
                    glam::Vec3::X
                } else {
                    glam::Vec3::Y
                };
                let tangent = up.cross(parent_center).normalize();
                let bitangent = parent_center.cross(tangent);
                for _ in 0..secondaries_per_parent {
                    // Angular distance in [1R, 3R], uniform in area via √u.
                    let u_r = rng.next_f64() as f32;
                    let r_mult = 1.0 + 2.0 * u_r.sqrt();
                    let angular_dist = (parent.radius_m * r_mult) / body_radius;
                    let theta = rng.next_f64() as f32 * std::f32::consts::TAU;
                    let (sin_t, cos_t) = theta.sin_cos();
                    let offset = tangent * (cos_t * angular_dist)
                        + bitangent * (sin_t * angular_dist);
                    let child_center = (parent_center + offset).normalize();

                    // Secondary radius: fraction of parent, power-law biased
                    // small. Cap at parent_threshold / 4 so secondaries stay
                    // clearly smaller than primaries.
                    let sec_min = self.min_radius_m.max(parent.radius_m as f64 * 0.005);
                    let sec_max = (parent.radius_m as f64 * 0.15)
                        .min(parent_threshold as f64 * 0.25);
                    if sec_max <= sec_min { continue; }
                    let radius_m = rng.power_law(sec_min, sec_max, 2.5) as f32;

                    let (base_depth, base_rim) = crater_dimensions(radius_m);
                    let jitter = 1.0 + rng.next_f64_signed() as f32 * 0.15;

                    // Secondary craters form from low-velocity ballistic
                    // ejecta rather than hypervelocity impacts, so their
                    // depth/diameter is roughly half the primary value
                    // (Pike & Wilhelms 1978, also lunar survey data).
                    const SECONDARY_SCALE: f32 = 0.5;
                    craters.push(Crater {
                        center: child_center,
                        radius_m,
                        depth_m: base_depth * jitter * SECONDARY_SCALE,
                        rim_height_m: base_rim * jitter * SECONDARY_SCALE,
                        // Slightly younger than parent (post-impact fracturing
                        // retriggers) but essentially the same epoch.
                        age_gyr: (parent.age_gyr - rng.next_f64() as f32 * 0.05)
                            .max(0.0),
                        material_id: MAT_HIGHLAND,
                    });
                }
            }
        }

        // Force a handful of mid-sized craters to be young so SpaceWeather
        // has fresh bright crater + ray candidates. We deliberately exclude
        // the absolute-largest craters: real bright young craters (Tycho,
        // Copernicus) are tens of km across, not basin-class. The biggest
        // craters formed during heavy bombardment and are always ancient
        // and shallow — giving them ejecta halos looks wrong.
        //
        // Target band: 1.5%–9% of body radius. On Mira (869 km) that's
        // roughly 13 km – 78 km radius. Ages are forced into
        // [0, 0.08 * body_age] (≈ 0–0.36 Gyr on a 4.5 Gyr body) so freshness
        // stays > 0.5 even at the upper end of the bracket.
        const YOUNG_COUNT: usize = 16;
        let young_min_r = body_radius * 0.015;
        let young_max_r = body_radius * 0.09;
        let mut candidate_indices: Vec<usize> = craters
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                (c.radius_m >= young_min_r && c.radius_m <= young_max_r).then_some(i)
            })
            .collect();
        // Partial Fisher–Yates shuffle: draw `YOUNG_COUNT` unique picks
        // without replacement so no candidate is refreshed twice. Pulling
        // with replacement (old behavior) could pick the same crater
        // multiple times, wasting the fresh-crater budget.
        let picks = YOUNG_COUNT.min(candidate_indices.len());
        for k in 0..picks {
            let span = candidate_indices.len() - k;
            let j = k + (rng.next_f64() * span as f64) as usize;
            let j = j.min(candidate_indices.len() - 1);
            candidate_indices.swap(k, j);
            let i = candidate_indices[k];
            let u = rng.next_f64();
            craters[i].age_gyr = (body_age_gyr as f64 * 0.08 * u) as f32;
        }

        // Sort oldest-first so younger craters stamp over older ones in the cubemap.
        craters.sort_by(|a, b| b.age_gyr.partial_cmp(&a.age_gyr).unwrap());

        // Bake large craters to height cubemap with age-based diffusion degradation.
        let res = builder.cubemap_resolution;

        // Precompute per-crater bake parameters once. The tangent basis and
        // phase table drive angular radial warping in the inner loop.
        struct BakedCrater {
            center: glam::Vec3,
            radius_m: f32,
            depth_m: f32,
            rim_height_m: f32,
            morph: crate::crater_profile::Morphology,
            influence_angle: f32,
            tangent: glam::Vec3,
            bitangent: glam::Vec3,
            radial_phases: [f32; 4],
            rim_phases: [f32; 4],
            wall_phases: [f32; 4],
            peak_phases: [f32; 4],
            /// Azimuth (rad) pointing at the uprange side of the impact.
            /// Ejecta apron is suppressed within the wedge centered here,
            /// modelling the characteristic "zone of avoidance" of
            /// oblique impacts (Proclus-type, up to ~60° half-width).
            uprange_az: f32,
            /// Half-angle (rad) of the uprange wedge. Constant 30°–60°.
            wedge_half: f32,
            /// Obliqueness ∈ [0, 0.85]. Fraction of the apron that's
            /// carved away at the wedge center. 0 → uniform blanket.
            obliqueness: f32,
            /// Pre-Cratering terrain height at crater center. Used as the
            /// base elevation for absolute-mode interior writes so younger
            /// craters cookie-cut older ones overlapping their interior
            /// instead of accumulating additively into a noisy bowl.
            base_elevation_m: f32,
        }
        let height_snapshot = &builder.height_contributions.height;
        let baked: Vec<BakedCrater> = craters
            .iter()
            .filter(|c| c.radius_m >= self.cubemap_bake_threshold_m)
            .map(|c| {
                let degrad = degradation_factor(c.radius_m, c.age_gyr);
                let center = c.center.normalize();
                let base_elevation_m = height_snapshot.sample_bilinear(center);
                // Tangent basis for measuring azimuth around the crater.
                // Matches the construction used by `space_weather.rs`.
                let up = if center.y.abs() < 0.9 { glam::Vec3::Y } else { glam::Vec3::X };
                let tangent = up.cross(center).normalize();
                let bitangent = center.cross(tangent);
                let seed = center_seed(center);
                // Influence angle must cover the outward-bulged edge; scale by
                // (1 + eps) so we don't clip texels on positive-noise angles.
                let influence_angle =
                    (c.influence_radius_m() / body_radius) * (1.0 + RADIAL_WARP_EPS);
                let tau = std::f32::consts::TAU;
                let pi = std::f32::consts::PI;
                let uprange_az = unit_from_seed(seed, 10) * tau;
                // Half-angle in [30°, 60°]. Wider wedges read as strongly
                // asymmetric ejecta blankets; narrower as mild.
                let wedge_half = pi / 6.0 + unit_from_seed(seed, 11) * (pi / 6.0);
                // Obliqueness hashed so ~30% of craters exceed 0.5 (strong
                // wedge), but most are mild — matches the 1.0–1.2 axis-
                // ratio distribution for real oblique impacts.
                let oh = unit_from_seed(seed, 12);
                let obliqueness = (oh * oh) * 0.85;
                BakedCrater {
                    center,
                    radius_m: c.radius_m,
                    depth_m: c.depth_m * degrad,
                    rim_height_m: c.rim_height_m * degrad,
                    morph: morphology_for_radius(c.radius_m),
                    influence_angle,
                    tangent,
                    bitangent,
                    radial_phases: phases_from_seed(seed, 0),
                    rim_phases: phases_from_seed(seed, 1),
                    wall_phases: phases_from_seed(seed, 2),
                    peak_phases: phases_from_seed(seed, 3),
                    uprange_az,
                    wedge_half,
                    obliqueness,
                    base_elevation_m,
                }
            })
            .collect();

        // Strip-parallel bake. Each face is processed sequentially but split
        // into row strips that run in parallel. Within a strip we still loop
        // craters in age order so younger-over-older stamping is preserved
        // (each texel is touched by exactly one strip). This scales past the
        // old 6-face parallelism ceiling while keeping write ordering correct.
        const STRIP_ROWS: u32 = 16;
        let strip_len = STRIP_ROWS as usize * res as usize;
        let baked_ref = &baked;
        for (face_idx, slice) in builder
            .height_contributions
            .height
            .faces_mut()
            .iter_mut()
            .enumerate()
        {
            let face = CubemapFace::ALL[face_idx];
            slice
                .par_chunks_mut(strip_len)
                .enumerate()
                .for_each(|(strip_idx, strip)| {
                    let y_start = strip_idx as u32 * STRIP_ROWS;
                    let strip_rows = (strip.len() as u32) / res;
                    let y_end = y_start + strip_rows;
                    for c in baked_ref {
                        for_face_texels_in_cap_rows(
                            face,
                            res,
                            c.center,
                            c.influence_angle,
                            y_start,
                            y_end,
                            |x, y, dir, angular_dist| {
                                let surface_dist = angular_dist * body_radius;
                                let t_base = surface_dist / c.radius_m;

                                // Angular radial warp: sample noise along the
                                // crater's local tangent plane and perturb t
                                // so effective radius becomes r·(1 + ε·n(θ)).
                                let proj = dir - c.center * dir.dot(c.center);
                                let theta = proj
                                    .dot(c.bitangent)
                                    .atan2(proj.dot(c.tangent));
                                let n_r = angular_noise(theta, &c.radial_phases);
                                let t = t_base / (1.0 + RADIAL_WARP_EPS * n_r);

                                let n_h = angular_noise(theta, &c.rim_phases);
                                let rim_h = c.rim_height_m * (1.0 + RIM_WARP_EPS * n_h);

                                let n_w = angular_noise(theta, &c.wall_phases);
                                let wall_phase = WALL_PHASE_EPS * n_w;

                                let n_p = angular_noise(theta, &c.peak_phases);

                                // Uprange ejecta wedge. Angular distance
                                // from the crater's uprange direction; the
                                // apron is smoothly suppressed inside
                                // `wedge_half` by a factor of `obliqueness`.
                                let pi = std::f32::consts::PI;
                                let tau = std::f32::consts::TAU;
                                let mut d_az = (theta - c.uprange_az).abs();
                                if d_az > pi { d_az = tau - d_az; }
                                // 1.0 at wedge center, 0.0 at wedge edge.
                                let in_wedge =
                                    smoothstep01(c.wedge_half, c.wedge_half * 0.5, d_az);
                                let ejecta_scale =
                                    (1.0 - c.obliqueness * in_wedge).clamp(0.0, 1.0);

                                let h = crater_profile(
                                    t,
                                    c.depth_m,
                                    rim_h,
                                    c.radius_m,
                                    c.morph,
                                    wall_phase,
                                    n_p,
                                    ejecta_scale,
                                );

                                // Interior-absolute / exterior-additive blend.
                                // At t < XOVER_LO cookie-cut; at t > XOVER_HI
                                // add to existing terrain; smooth lerp between.
                                //
                                // Cookie-cut extends past the rim crest
                                // (out to 1.12 R) so a younger crater
                                // overlapping an older one wipes out the
                                // old rim in the overlap region instead of
                                // stacking two concentric ridges. Real
                                // lunar crater superposition looks like
                                // partial destruction, not ring addition.
                                const XOVER_LO: f32 = 0.96;
                                const XOVER_HI: f32 = 1.12;
                                let local_y = y - y_start;
                                let idx = (local_y * res + x) as usize;
                                let existing = strip[idx];
                                let absolute = c.base_elevation_m + h;
                                let additive = existing + h;
                                let w = smoothstep01(XOVER_LO, XOVER_HI, t);
                                let new_h = absolute * (1.0 - w) + additive * w;
                                if (new_h - existing).abs() > 1e-3 || h.abs() > 1e-3 {
                                    strip[idx] = new_h;
                                }
                            },
                        );
                    }
                });
        }

        builder.craters = craters;
        // Publish the threshold to BodyData so the sampler and shader can
        // skip baked craters during Layer 2 iteration (otherwise their
        // contribution is counted twice — once from the cubemap texel, once
        // from the SSBO).
        builder.cubemap_bake_threshold_m = self.cubemap_bake_threshold_m;
    }
}

