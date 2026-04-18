//! Climate — annual-mean temperature and precipitation.
//!
//! Produces two float cubemaps on `BodyBuilder`:
//! - `temperature_c`: annual-mean surface temperature in °C
//! - `precipitation_mm`: annual total precipitation in mm
//!
//! Temperature model:
//!   T = polar + (equator - polar) × (1 - |sin(lat)|^lat_exponent) - lapse × h_km
//! Annual-mean, not seasonal. Axial tilt is implicit in the pole-equator
//! gradient — changing the gradient changes how habitable high latitudes are.
//!
//! Precipitation model (zonal + wind advection + orographic):
//!   1. Zonal base — piecewise by latitude (Hadley / subtropical / Ferrel
//!      / polar front / polar).
//!   2. Continentality — trace N steps upwind, count ocean fraction. Moisture
//!      attenuates as the air passes over land.
//!   3. Orographic — rise along the last few upwind steps boosts precip;
//!      drops (leeward) produce rain shadow.
//!
//! Prevailing wind direction by latitude band follows Earth's three-cell
//! circulation: tropics easterlies, mid-latitudes westerlies, polar easterlies.
//!
//! Dependencies: Topography (reads height_contributions.height).

use glam::Vec3;
use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{Cubemap, CubemapFace, dir_to_face_uv, face_uv_to_dir};
use crate::noise::fbm3;
use crate::seeding::splitmix64;
use crate::stage::Stage;

/// Climate stage parameters. All authored per-body.
#[derive(Debug, Clone, Deserialize)]
pub struct Climate {
    /// Annual-mean temperature at the equator at sea level (°C).
    pub equator_temp_c: f32,
    /// Annual-mean temperature at the pole at sea level (°C).
    pub polar_temp_c: f32,
    /// Exponent on |sin(lat)| in the equator-pole gradient. 1.0 = linear;
    /// >1 concentrates the drop near the poles (wider tropical band).
    pub lat_exponent: f32,
    /// Lapse rate (°C per km of altitude). Earth = 6.5.
    pub lapse_c_per_km: f32,

    /// Piecewise zonal precipitation in mm/yr, one value per 10° of
    /// absolute latitude (0-10, 10-20, ..., 80-90). Nine entries.
    /// Earth-like: [2500, 2000, 400, 600, 1200, 1100, 800, 400, 200].
    pub zonal_precip_mm: [f32; 9],

    /// Number of upwind steps traced per cell. 8-16. Cost scales linearly.
    pub upwind_steps: u32,
    /// Arc length (rad) of each upwind step. ~0.04-0.06 works for a
    /// 3000 km body (a step covers ~120-200 km).
    pub step_arc_rad: f32,
    /// Per-land-step multiplier on moisture. <1 attenuates precipitation
    /// as air passes over land. 0.82-0.92 gives visible dry interiors.
    pub continentality_decay: f32,
    /// Orographic gain: precip multiplied by (1 + gain × rise_km).
    /// Rise is summed over upwind steps, weighted inversely by distance.
    pub orographic_gain_per_km: f32,
    /// Rain-shadow strength: precip divided by (1 + strength × drop_km).
    pub rain_shadow_per_km: f32,

    /// Noise amplitude as a fraction of the cell's precip. 0.10-0.25
    /// breaks up sharp zonal-band iso-contours.
    pub precip_noise_amp: f32,
    /// Noise frequency for precip jitter.
    pub precip_noise_frequency: f64,
}

impl Climate {
    /// Zonal base precipitation at absolute latitude (radians). Linearly
    /// interpolates between the nine 10° bins.
    fn zonal_at(&self, abs_lat_rad: f32) -> f32 {
        let bin_f = (abs_lat_rad / (std::f32::consts::FRAC_PI_2 / 9.0)).clamp(0.0, 8.999);
        let i = bin_f.floor() as usize;
        let t = bin_f - i as f32;
        let j = (i + 1).min(8);
        self.zonal_precip_mm[i] * (1.0 - t) + self.zonal_precip_mm[j] * t
    }
}

impl Stage for Climate {
    fn name(&self) -> &str {
        "climate"
    }
    fn dependencies(&self) -> &[&str] {
        &["topography"]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        let res = builder.cubemap_resolution;
        let inv_res = 1.0 / res as f32;
        let seed = builder.stage_seed();
        let noise_seed = splitmix64(seed ^ 0xC117_A7ED_12B0_0F91);

        let height = &builder.height_contributions.height;

        let mut temp = Cubemap::<f32>::new(res);
        let mut precip = Cubemap::<f32>::new(res);

        // Closure samples the height cubemap by nearest texel.  Nearest is
        // fine at climate scale (step ~120 km ≫ texel at 2048²).
        let sample_height = |d: Vec3| -> f32 {
            let (face, u, v) = dir_to_face_uv(d.normalize());
            let x = ((u * res as f32) as u32).min(res - 1);
            let y = ((v * res as f32) as u32).min(res - 1);
            height.face_data(face)[(y * res + x) as usize]
        };

        for face in CubemapFace::ALL {
            let t_out = temp.face_data_mut(face);
            let p_out = precip.face_data_mut(face);
            t_out
                .par_iter_mut()
                .zip(p_out.par_iter_mut())
                .enumerate()
                .for_each(|(idx, (t_cell, p_cell))| {
                    let x = (idx as u32) % res;
                    let y = (idx as u32) / res;
                    let u = (x as f32 + 0.5) * inv_res;
                    let v = (y as f32 + 0.5) * inv_res;
                    let dir = face_uv_to_dir(face, u, v).normalize();
                    let h = sample_height(dir);
                    let sin_lat = dir.y.abs().clamp(0.0, 1.0);
                    let abs_lat_rad = sin_lat.asin();

                    // Temperature: equator-pole gradient + lapse.
                    let lat_factor = 1.0 - sin_lat.powf(self.lat_exponent);
                    let sea_level_t = self.polar_temp_c
                        + (self.equator_temp_c - self.polar_temp_c) * lat_factor;
                    let altitude_t = self.lapse_c_per_km * (h.max(0.0) / 1000.0);
                    *t_cell = sea_level_t - altitude_t;

                    // Precipitation.
                    let zonal_base = self.zonal_at(abs_lat_rad);

                    // Prevailing wind tangent at this cell.
                    let wind_tan = prevailing_wind_tangent(dir, abs_lat_rad);

                    // Upwind trace. Rotation axis is perpendicular to both
                    // the cell direction and the wind tangent (i.e. locally
                    // perpendicular to the wind great-circle).
                    let steps = self.upwind_steps as usize;
                    let axis = dir.cross(wind_tan).normalize_or_zero();
                    let mut ocean_steps = 0u32;
                    let mut rise_km = 0.0f32;
                    let mut drop_km = 0.0f32;
                    let mut prev_h = h;
                    for i in 1..=steps {
                        // Negative angle → step upwind (opposite the wind).
                        let angle = -(i as f32) * self.step_arc_rad;
                        let p_i = rotate_around_axis(dir, axis, angle).normalize();
                        let h_i = sample_height(p_i);
                        if h_i <= 0.0 {
                            ocean_steps += 1;
                        }
                        // Weight near-steps more than far-steps.
                        let weight = 1.0 / i as f32;
                        // Rise going downwind (from p_i toward the cell): prev_h - h_i.
                        // `prev_h` starts at the cell and walks outward, so
                        // `prev_h - h_i` is the downwind rise between p_i and
                        // the nearer step.
                        let dh_km = (prev_h - h_i) / 1000.0;
                        if dh_km > 0.0 {
                            rise_km += dh_km * weight;
                        } else {
                            drop_km += (-dh_km) * weight;
                        }
                        prev_h = h_i;
                    }
                    let ocean_frac = ocean_steps as f32 / steps as f32;
                    let land_steps = steps as u32 - ocean_steps;
                    let continentality = ocean_frac
                        + (1.0 - ocean_frac) * self.continentality_decay.powi(land_steps as i32);
                    let mut p = zonal_base * continentality;

                    p *= 1.0 + self.orographic_gain_per_km * rise_km;
                    p /= 1.0 + self.rain_shadow_per_km * drop_km;

                    // Noise jitter.
                    let noise = fbm3(
                        dir.x as f64 * self.precip_noise_frequency,
                        dir.y as f64 * self.precip_noise_frequency,
                        dir.z as f64 * self.precip_noise_frequency,
                        noise_seed,
                        4,
                        0.55,
                        2.1,
                    ) as f32;
                    p *= 1.0 + noise * self.precip_noise_amp;

                    // Ocean: maritime climate — never drier than 70 % of the
                    // band's zonal value.
                    if h <= 0.0 {
                        p = p.max(zonal_base * 0.7);
                    }
                    *p_cell = p.max(0.0);
                });
        }

        builder.temperature_c = temp;
        builder.precipitation_mm = precip;
    }
}

/// Rotate a vector around a unit axis by `angle_rad` (Rodrigues).
fn rotate_around_axis(v: Vec3, axis: Vec3, angle_rad: f32) -> Vec3 {
    let (s, c) = angle_rad.sin_cos();
    v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c))
}

/// Prevailing wind tangent at a given surface direction. Earth's three-cell
/// circulation: tropics trade winds (east→west), mid-latitudes westerlies
/// (west→east), polar easterlies (east→west).
fn prevailing_wind_tangent(dir: Vec3, abs_lat_rad: f32) -> Vec3 {
    // Eastward tangent at this cell: cross(north_pole, dir). At the poles
    // this is the zero vector; fall back to an arbitrary tangent since the
    // wind circulation is undefined there anyway.
    let east = Vec3::Y.cross(dir);
    let east_len = east.length();
    if east_len < 1e-6 {
        return Vec3::ZERO;
    }
    let east_tan = east / east_len;
    let sign = if abs_lat_rad < 30.0_f32.to_radians() {
        -1.0 // trade winds: blow westward
    } else if abs_lat_rad < 60.0_f32.to_radians() {
        1.0 // westerlies: blow eastward
    } else {
        -1.0 // polar easterlies: blow westward
    };
    east_tan * sign
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::{Differentiate, Plates, Tectonics, Topography};
    use crate::stage::Pipeline;
    use crate::types::Composition;

    fn earth_like_climate() -> Climate {
        Climate {
            equator_temp_c: 27.0,
            polar_temp_c: -28.0,
            lat_exponent: 1.3,
            lapse_c_per_km: 6.5,
            zonal_precip_mm: [
                2500.0, 2000.0, 600.0, 700.0, 1200.0, 1100.0, 800.0, 400.0, 200.0,
            ],
            upwind_steps: 10,
            step_arc_rad: 0.05,
            continentality_decay: 0.90,
            orographic_gain_per_km: 0.5,
            rain_shadow_per_km: 0.4,
            precip_noise_amp: 0.15,
            precip_noise_frequency: 3.0,
        }
    }

    fn thalos_like_pipeline() -> Pipeline {
        Pipeline::new(vec![
            Box::new(Differentiate),
            Box::new(Plates {
                n_plates: 40,
                continental_area_fraction: 0.35,
                n_continental_seeds: 4,
                neighbour_continental_bias: 0.5,
                lloyd_iterations: 3,
            }),
            Box::new(Tectonics {
                active_boundary_fraction: 0.20,
                orogen_age_peak_gyr: 2.0,
            }),
            Box::new(crate::stages::OrogenDla {
                subdivision_level: 4,
                seed_density: 4.0,
                walkers_per_seed: 100,
                walker_step_budget: 200,
                launch_cap_km: 500.0,
                ridge_falloff_km: 40.0,
                depth_saturation: 4.0,
                samples_per_edge: 4,
                midpoint_jitter_frac: 0.3,
            }),
            Box::new(Topography {
                continental_baseline_m: 500.0,
                oceanic_baseline_m: -3500.0,
                continentalness_bandwidth: 0.02,
                peak_orogen_m: 7000.0,
                roughness_m: 700.0,
                orogen_age_scale_myr: 1500.0,
                noise_frequency: 2.5,
                warp_amplitude: 0.16,
                warp_frequency: 1.8,
                warp_amplitude_mid: 0.10,
                warp_frequency_mid: 5.0,
                warp_amplitude_hi: 0.04,
                warp_frequency_hi: 14.0,
                regional_height_lo_m: 2000.0,
                regional_frequency_lo: 0.5,
                regional_height_m: 1500.0,
                regional_frequency: 1.5,
                regional_height_hi_m: 600.0,
                regional_frequency_hi: 4.0,
                coastal_detail_m: 220.0,
                coastal_detail_frequency: 18.0,
                coastal_detail_scale_m: 1300.0,
                sea_level_percentile: 0.55,
            }),
            Box::new(earth_like_climate()),
        ])
    }

    #[test]
    fn climate_produces_equator_pole_gradient() {
        let mut b = BodyBuilder::new(
            3_186_000.0,
            1003,
            Composition::new(0.26, 0.70, 0.0, 0.04, 0.0),
            64,
            4.5,
            None,
            23.0_f32.to_radians(),
        );
        thalos_like_pipeline().run(&mut b);

        // Equator sample (+X): warmer than polar sample (+Y).
        let equator_t = b.temperature_c.get(CubemapFace::PosX, 32, 32);
        let pole_t = b.temperature_c.get(CubemapFace::PosY, 32, 32);
        assert!(
            equator_t > pole_t + 20.0,
            "expected strong gradient: equator={equator_t}°C pole={pole_t}°C",
        );
        assert!((-40.0..=45.0).contains(&equator_t), "equator out of range: {equator_t}");
        assert!((-60.0..=10.0).contains(&pole_t), "pole out of range: {pole_t}");
    }

    #[test]
    fn climate_produces_precipitation_variance() {
        let mut b = BodyBuilder::new(
            3_186_000.0,
            1003,
            Composition::new(0.26, 0.70, 0.0, 0.04, 0.0),
            64,
            4.5,
            None,
            23.0_f32.to_radians(),
        );
        thalos_like_pipeline().run(&mut b);

        let mut min_p = f32::MAX;
        let mut max_p = 0.0_f32;
        let mut sum = 0.0_f32;
        let mut n = 0u32;
        for face in CubemapFace::ALL {
            for &p in b.precipitation_mm.face_data(face) {
                min_p = min_p.min(p);
                max_p = max_p.max(p);
                sum += p;
                n += 1;
            }
        }
        let mean = sum / n as f32;
        // Strong zonal + orographic variation should span a wide range.
        assert!(min_p < 500.0, "min precip too high: {min_p}");
        assert!(max_p > 2000.0, "max precip too low: {max_p}");
        assert!(
            (300.0..=1800.0).contains(&mean),
            "mean precip out of range: {mean}",
        );
    }
}
