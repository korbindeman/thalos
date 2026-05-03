use glam::Vec3;
use rayon::prelude::*;
use serde::Deserialize;

use super::util::for_face_texels_in_cap_rows;
use crate::body_builder::BodyBuilder;
use crate::cubemap::CubemapFace;
use crate::noise::fbm3;
use crate::seeding::splitmix64;
use crate::stage::Stage;
use crate::surface_field::{mix3, smoothstep};
use crate::types::Crater;

/// Vaelen-specific color/provenance overprint for resolved impacts.
///
/// `Cratering` owns the relief. This stage only adds soft albedo memory: no
/// material-id rewrites, so impacts do not turn into categorical paint masks.
#[derive(Debug, Clone, Deserialize)]
pub struct VaelenImpactColor {
    /// Minimum crater radius that gets a baked albedo signature.
    ///
    /// Craters below this remain height/SSBO detail. Craters at and above this
    /// are large enough to be resolved in the cubemap, so they need color
    /// structure or they disappear under high sun.
    pub crater_min_radius_m: f32,
}

impl Stage for VaelenImpactColor {
    fn name(&self) -> &str {
        "vaelen_impact_color"
    }

    fn dependencies(&self) -> &[&str] {
        &["cratering"]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        let radius_m = builder.radius_m;
        let res = builder.cubemap_resolution;
        let mut marks = Vec::new();

        for basin in &builder.megabasins {
            let seed = basin
                .seed
                .unwrap_or_else(|| splitmix64(builder.seed ^ basin.radius_m.to_bits() as u64));
            marks.push(ImpactMark::Basin(BasinMark::new(
                basin.center_dir,
                basin.radius_m,
                seed,
            )));
        }

        for crater in builder
            .craters
            .iter()
            .filter(|crater| crater.radius_m >= self.crater_min_radius_m)
        {
            marks.push(ImpactMark::Crater(CraterMark::new(crater)));
        }

        if marks.is_empty() {
            return;
        }

        const STRIP_ROWS: u32 = 16;
        let strip_len = STRIP_ROWS as usize * res as usize;
        let albedo_faces = builder.albedo_contributions.albedo.faces_mut();

        for face_idx in 0..CubemapFace::ALL.len() {
            let face = CubemapFace::ALL[face_idx];
            let albedo_face = &mut albedo_faces[face_idx];

            albedo_face.par_chunks_mut(strip_len).enumerate().for_each(
                |(strip_idx, albedo_strip)| {
                    let y_start = strip_idx as u32 * STRIP_ROWS;
                    let strip_rows = (albedo_strip.len() as u32) / res;
                    let y_end = y_start + strip_rows;

                    for mark in &marks {
                        for_face_texels_in_cap_rows(
                            face,
                            res,
                            mark.center(),
                            mark.influence_angle(radius_m),
                            y_start,
                            y_end,
                            |x, y, dir, angular_dist| {
                                let local_y = y - y_start;
                                let idx = (local_y * res + x) as usize;
                                let color = &mut albedo_strip[idx];
                                let mut rgb = [color[0], color[1], color[2]];
                                mark.apply(dir, angular_dist, radius_m, &mut rgb);
                                color[0] = rgb[0].clamp(0.02, 0.92);
                                color[1] = rgb[1].clamp(0.02, 0.92);
                                color[2] = rgb[2].clamp(0.02, 0.92);
                                color[3] = 1.0;
                            },
                        );
                    }
                },
            );
        }
    }
}

enum ImpactMark {
    Basin(BasinMark),
    Crater(CraterMark),
}

impl ImpactMark {
    fn center(&self) -> Vec3 {
        match self {
            Self::Basin(mark) => mark.center,
            Self::Crater(mark) => mark.center,
        }
    }

    fn influence_angle(&self, body_radius_m: f32) -> f32 {
        match self {
            Self::Basin(mark) => (mark.radius_m * 2.65 / body_radius_m).min(std::f32::consts::PI),
            Self::Crater(mark) => (mark.radius_m * 2.75 / body_radius_m).min(std::f32::consts::PI),
        }
    }

    fn apply(&self, dir: Vec3, angular_dist: f32, body_radius_m: f32, albedo: &mut [f32; 3]) {
        match self {
            Self::Basin(mark) => mark.apply(dir, angular_dist, body_radius_m, albedo),
            Self::Crater(mark) => mark.apply(dir, angular_dist, body_radius_m, albedo),
        }
    }
}

struct BasinMark {
    center: Vec3,
    radius_m: f32,
    seed: u32,
    tangent: Vec3,
    bitangent: Vec3,
    evaporite_bias: f32,
}

impl BasinMark {
    fn new(center: Vec3, radius_m: f32, seed: u64) -> Self {
        let center = center.normalize();
        let (tangent, bitangent) = tangent_frame(center);
        Self {
            center,
            radius_m,
            seed: seed as u32,
            tangent,
            bitangent,
            evaporite_bias: unit_from_seed(seed, 11),
        }
    }

    fn apply(&self, dir: Vec3, angular_dist: f32, body_radius_m: f32, albedo: &mut [f32; 3]) {
        let t_raw = angular_dist * body_radius_m / self.radius_m.max(1.0);
        if t_raw > 2.65 {
            return;
        }

        let az = azimuth(dir, self.tangent, self.bitangent);
        let warp = fbm3(
            dir.x * 3.4,
            dir.y * 3.4,
            dir.z * 3.4,
            self.seed ^ 0xA317_5D49,
            3,
            0.55,
            2.1,
        ) * 0.18
            + fbm3(
                dir.x * 8.5,
                dir.y * 8.5,
                dir.z * 8.5,
                self.seed ^ 0x71D9_20AF,
                2,
                0.5,
                2.2,
            ) * 0.07;
        let t = t_raw / (1.0 + warp).max(0.35);

        let rim_presence = broken_ring_mask(az, self.seed);
        let floor_w = smoothstep(0.94, 0.24, t);
        let inner_floor_w = smoothstep(0.42, 0.08, t);
        let wall_w = gaussian(t, 0.76, 0.18) * (0.45 + 0.55 * rim_presence);
        let rim_w = gaussian(t, 1.0, 0.10) * rim_presence;
        let ejecta_w = if t > 1.0 {
            let fade = smoothstep(2.65, 1.0, t);
            let streak = 0.45 + 0.55 * (az * 8.0 + self.evaporite_bias * 6.11).cos().max(0.0);
            fade * streak / (t * t).max(1.0)
        } else {
            0.0
        };

        if floor_w > 0.01 {
            let sediment = mix3([0.56, 0.40, 0.24], [0.70, 0.60, 0.42], inner_floor_w * 0.45);
            *albedo = mix3(*albedo, sediment, floor_w * 0.16);
        }
        if inner_floor_w > 0.01 && self.evaporite_bias > 0.35 {
            let evap = inner_floor_w * (0.06 + 0.10 * self.evaporite_bias);
            *albedo = mix3(*albedo, [0.79, 0.70, 0.52], evap);
        }
        if wall_w > 0.01 {
            *albedo = mix3(*albedo, [0.27, 0.15, 0.10], wall_w * 0.08);
        }
        if rim_w > 0.01 {
            *albedo = mix3(*albedo, [0.60, 0.34, 0.18], rim_w * 0.10);
        }
        if ejecta_w > 0.01 {
            *albedo = mix3(*albedo, [0.50, 0.28, 0.16], ejecta_w * 0.07);
        }
    }
}

struct CraterMark {
    center: Vec3,
    radius_m: f32,
    age_gyr: f32,
    seed: u32,
    tangent: Vec3,
    bitangent: Vec3,
    pale_fill: f32,
}

impl CraterMark {
    fn new(crater: &Crater) -> Self {
        let center = crater.center.normalize();
        let seed = center_seed(center);
        let (tangent, bitangent) = tangent_frame(center);
        let age_fill = smoothstep(1.2, 4.0, crater.age_gyr);
        Self {
            center,
            radius_m: crater.radius_m,
            age_gyr: crater.age_gyr,
            seed: seed as u32,
            tangent,
            bitangent,
            pale_fill: age_fill * unit_from_seed(seed, 21),
        }
    }

    fn apply(&self, dir: Vec3, angular_dist: f32, body_radius_m: f32, albedo: &mut [f32; 3]) {
        let t_raw = angular_dist * body_radius_m / self.radius_m.max(1.0);
        if t_raw > 2.75 {
            return;
        }

        let az = azimuth(dir, self.tangent, self.bitangent);
        let rim_presence = broken_ring_mask(az + 0.7, self.seed);
        let floor_w = smoothstep(0.68, 0.16, t_raw);
        let inner_floor_w = smoothstep(0.40, 0.08, t_raw);
        let wall_w = gaussian(t_raw, 0.80, 0.17) * (0.52 + 0.48 * rim_presence);
        let inner_wall_shadow_w = gaussian(t_raw, 0.66, 0.10) * (0.45 + 0.55 * rim_presence);
        let rim_w = gaussian(t_raw, 1.0, 0.075) * rim_presence;
        let outer_rim_w = gaussian(t_raw, 1.12, 0.12) * (0.65 + 0.35 * rim_presence);
        let age_soft = smoothstep(0.4, 3.8, self.age_gyr);
        let freshness = 1.0 - age_soft;
        let resolved = smoothstep(3_000.0, 16_000.0, self.radius_m);
        let contrast = 0.68 + freshness * 0.32 + resolved * 0.22;
        let ejecta_w = if t_raw > 1.0 {
            let fade = smoothstep(2.75, 1.0, t_raw);
            let streak = 0.35
                + 0.65
                    * (az * (7.0 + self.radius_m * 0.00003) + self.seed as f32)
                        .cos()
                        .max(0.0);
            fade * streak / (t_raw * t_raw).max(1.0)
        } else {
            0.0
        };

        if floor_w > 0.01 {
            let dark_floor = floor_w * (0.20 - self.pale_fill * 0.05) * contrast;
            *albedo = mix3(*albedo, [0.17, 0.09, 0.065], dark_floor);

            let pale_w = inner_floor_w * self.pale_fill * 0.16;
            if pale_w > 0.01 {
                *albedo = mix3(*albedo, [0.63, 0.50, 0.34], pale_w);
            }
        }
        if inner_wall_shadow_w > 0.01 {
            *albedo = mix3(
                *albedo,
                [0.13, 0.07, 0.055],
                inner_wall_shadow_w * 0.16 * contrast,
            );
        }
        if wall_w > 0.01 {
            *albedo = mix3(
                *albedo,
                [0.25, 0.13, 0.085],
                wall_w * (0.11 + age_soft * 0.05) * contrast,
            );
        }
        if rim_w > 0.01 {
            *albedo = mix3(
                *albedo,
                [0.74, 0.43, 0.22],
                rim_w * (0.20 + freshness * 0.06) * contrast,
            );
        }
        if outer_rim_w > 0.01 {
            *albedo = mix3(*albedo, [0.58, 0.31, 0.15], outer_rim_w * 0.10 * contrast);
        }
        if ejecta_w > 0.01 {
            *albedo = mix3(
                *albedo,
                [0.62, 0.35, 0.18],
                ejecta_w * (0.10 + freshness * 0.05) * contrast,
            );
        }
    }
}

fn tangent_frame(center: Vec3) -> (Vec3, Vec3) {
    let up = if center.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent = up.cross(center).normalize();
    let bitangent = center.cross(tangent);
    (tangent, bitangent)
}

fn azimuth(dir: Vec3, tangent: Vec3, bitangent: Vec3) -> f32 {
    dir.dot(bitangent).atan2(dir.dot(tangent))
}

fn gaussian(x: f32, mean: f32, sigma: f32) -> f32 {
    let z = (x - mean) / sigma.max(1e-4);
    (-z * z).exp()
}

fn broken_ring_mask(azimuth: f32, seed: u32) -> f32 {
    let phase = seed as f32 * 0.000_000_37;
    let broad_a = 0.5 + 0.5 * (azimuth + phase).sin();
    let broad_b = 0.5 + 0.5 * (2.0 * azimuth + phase * 1.7 + 0.8).sin();
    let scallop = 0.78 + 0.22 * (7.0 * azimuth + phase * 0.6).sin();
    ((broad_a * 0.55 + broad_b * 0.55) * scallop).clamp(0.08, 1.0)
}

fn center_seed(center: Vec3) -> u64 {
    splitmix64(
        center.x.to_bits() as u64
            ^ ((center.y.to_bits() as u64) << 17)
            ^ ((center.z.to_bits() as u64) << 34),
    )
}

fn unit_from_seed(seed: u64, channel: u64) -> f32 {
    let h = splitmix64(seed ^ channel.wrapping_mul(0xA076_1D64_78BD_642F));
    (h as u32 as f32) / u32::MAX as f32
}
