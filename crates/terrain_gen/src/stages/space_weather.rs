use glam::Vec3;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::seeding::Rng;
use crate::stage::Stage;
use crate::types::Crater;
use super::util::{for_all_texels, for_texels_in_cap};
use super::MAT_MARE;

/// Space-weathering maturity from crater age (Lucey et al. 2000).
///
/// Darkening follows logarithmic kinetics: rapid initial darkening in the
/// first ~100 Myr, asymptoting to saturation by ~1 Gyr.
///
/// Returns a value in [0, 1]: 0 = completely fresh, 1 = fully mature.
fn maturity(age_gyr: f32) -> f32 {
    let age_ma = age_gyr * 1000.0; // convert Gyr → Myr
    // doc formula: clamp(ln(age_Ma) / ln(1000), 0, 1)
    // At 1 Ma → 0 (fresh). At 1 Gyr → 1.0 (mature).
    if age_ma <= 0.0 {
        return 0.0;
    }
    (age_ma.ln() / 1000.0_f32.ln()).clamp(0.0, 1.0)
}

/// Authors the final albedo cubemap based on material, age, and weathering.
///
/// Three effects:
/// 1. Maturity darkening — all surfaces darken toward mature albedo with age.
/// 2. Fresh-crater brightening — young craters and ejecta are brighter.
/// 3. Ray systems — the very youngest craters get radial bright streaks.
#[derive(Debug, Clone, Deserialize)]
pub struct SpaceWeather {
    pub highland_mature_albedo: f32,
    pub highland_fresh_albedo: f32,
    pub mare_mature_albedo: f32,
    pub mare_fresh_albedo: f32,
    /// Craters younger than this (Gyr) get fresh-ejecta brightening.
    pub young_crater_age_threshold: f32,
    /// Craters younger than this get ray systems (very few).
    pub ray_age_threshold: f32,
    /// How far rays extend, in multiples of crater radius.
    pub ray_extent_radii: f32,
    /// Number of ray arms per crater.
    pub ray_count_per_crater: u32,
    /// Angular half-width of each ray arm (radians).
    pub ray_half_width: f32,
}

/// Precomputed ray data for a single rayed crater.
struct RayedCrater {
    center: Vec3,
    radius_m: f32,
    age_frac: f32,
    ray_angles: Vec<f32>,
    /// Azimuth of the uprange (impact-arrival) direction; rays are excluded
    /// in a sector around this angle to simulate oblique-impact asymmetry
    /// (Hawke et al. 2004: Proclus-type asymmetry).
    uprange_az: f32,
    /// Half-width of uprange exclusion sector (radians).
    exclusion_half: f32,
    /// Local tangent frame for computing azimuth.
    tangent_right: Vec3,
    tangent_forward: Vec3,
}

impl Stage for SpaceWeather {
    fn name(&self) -> &str { "space_weather" }
    fn dependencies(&self) -> &[&str] { &["mare_flood", "regolith"] }

    fn apply(&self, builder: &mut BodyBuilder) {
        let res = builder.cubemap_resolution;
        let body_radius = builder.radius_m;
        let seed = builder.stage_seed();

        // Classify young and rayed craters.
        let young_craters: Vec<&Crater> = builder.craters.iter()
            .filter(|c| c.age_gyr < self.young_crater_age_threshold)
            .collect();

        let mut rng = Rng::new(seed);
        let rayed_craters: Vec<RayedCrater> = builder.craters.iter()
            .filter(|c| c.age_gyr < self.ray_age_threshold)
            .map(|c| {
                let center = c.center.normalize();
                let up = if center.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
                let right = center.cross(up).normalize();
                let forward = right.cross(center).normalize();

                // Variable ray count: 8-24 scaling with crater size (doc §4).
                let ray_count = ((8.0 + c.radius_m / 10_000.0 * 8.0) as u32).clamp(8, 24);
                let ray_angles: Vec<f32> = (0..ray_count)
                    .map(|_| rng.next_f64() as f32 * std::f32::consts::TAU)
                    .collect();

                // Uprange exclusion: oblique impacts produce a "forbidden zone"
                // where ejecta is absent (Hawke et al. 2004). Exclude 30-90° sector.
                let uprange_az = rng.next_f64() as f32 * std::f32::consts::TAU;
                let exclusion_half = rng.range_f64(
                    std::f64::consts::FRAC_PI_6,   // 30°
                    std::f64::consts::FRAC_PI_2,   // 90°
                ) as f32;

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

        // Pass 1: Write base mature albedo for every texel.
        for_all_texels(res, |face, x, y, _dir| {
            let is_mare = builder.material_cubemap.get(face, x, y) == MAT_MARE as u8;
            let base = if is_mare {
                self.mare_mature_albedo
            } else {
                self.highland_mature_albedo
            };
            builder.albedo_contributions.albedo.set(face, x, y, [base, base, base, 1.0]);
        });

        // Pass 2: Brighten young crater vicinities using logarithmic maturity.
        //
        // Maturity follows logarithmic kinetics (Lucey et al. 2000): rapid initial
        // darkening (~20-30% reduction in first 100 Myr) asymptoting by ~1 Gyr.
        for crater in &young_craters {
            let center = crater.center.normalize();
            let ejecta_radius_m = crater.influence_radius_m();
            let ejecta_angle = ejecta_radius_m / body_radius;
            // freshness = 1 - maturity; 1.0 for brand-new, ~0.1 at 500 Myr
            let freshness = 1.0 - maturity(crater.age_gyr);

            for_texels_in_cap(res, center, ejecta_angle, |face, x, y, _dir, angular_dist| {
                let surface_dist = angular_dist * body_radius;
                let _t = surface_dist / ejecta_radius_m;

                // Distance falloff: ejecta brightness decays as t^-3 (power law).
                let t_crater = surface_dist / crater.radius_m;
                let distance_falloff = if t_crater <= 1.0 {
                    1.0 // inside crater: fully fresh
                } else {
                    t_crater.powi(-3).min(1.0)
                };
                let strength = freshness * distance_falloff;

                if strength > 0.005 {
                    let current = builder.albedo_contributions.albedo.get(face, x, y);
                    let is_mare = builder.material_cubemap.get(face, x, y) == MAT_MARE as u8;
                    let fresh = if is_mare { self.mare_fresh_albedo } else { self.highland_fresh_albedo };
                    let boosted = current[0] + (fresh - current[0]) * strength;
                    builder.albedo_contributions.albedo.set(
                        face, x, y,
                        [boosted, boosted, boosted, 1.0],
                    );
                }
            });
        }

        // Pass 3: Ray systems on the very youngest craters.
        for rc in &rayed_craters {
            let ray_extent_m = rc.radius_m * self.ray_extent_radii;
            let ray_angle = ray_extent_m / body_radius;
            let ray_fade = 1.0 - rc.age_frac; // fades as crater ages

            for_texels_in_cap(res, rc.center, ray_angle, |face, x, y, dir, angular_dist| {
                let surface_dist = angular_dist * body_radius;

                // Skip inside the crater itself (already brightened in pass 2).
                if surface_dist < rc.radius_m { return; }

                // Compute azimuth angle in the crater's tangent plane.
                let to_texel = (dir - rc.center * dir.dot(rc.center)).normalize();
                let az_x = to_texel.dot(rc.tangent_right);
                let az_y = to_texel.dot(rc.tangent_forward);
                let azimuth = az_y.atan2(az_x);

                // Check if this azimuth aligns with any ray arm.
                let mut on_ray = false;
                for &ray_az in &rc.ray_angles {
                    let mut delta = (azimuth - ray_az).abs();
                    if delta > std::f32::consts::PI { delta = std::f32::consts::TAU - delta; }
                    if delta < self.ray_half_width {
                        on_ray = true;
                        break;
                    }
                }

                if on_ray {
                    // Uprange exclusion: oblique impacts produce an asymmetric
                    // ray pattern with a forbidden sector (Hawke et al. 2004).
                    let mut delta_up = (azimuth - rc.uprange_az).abs();
                    if delta_up > std::f32::consts::PI {
                        delta_up = std::f32::consts::TAU - delta_up;
                    }
                    if delta_up < rc.exclusion_half {
                        return; // suppressed in uprange direction
                    }

                    // Ray brightness decays with distance from rim and crater age.
                    let dist_from_rim = (surface_dist - rc.radius_m) / (ray_extent_m - rc.radius_m);
                    let brightness = ray_fade * (1.0 - dist_from_rim).max(0.0) * 0.6;

                    if brightness > 0.001 {
                        let current = builder.albedo_contributions.albedo.get(face, x, y);
                        let boosted = current[0] + (self.highland_fresh_albedo - current[0]) * brightness;
                        builder.albedo_contributions.albedo.set(
                            face, x, y,
                            [boosted, boosted, boosted, 1.0],
                        );
                    }
                }
            });
        }
    }
}
