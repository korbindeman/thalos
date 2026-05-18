use glam::Vec3;
use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{CubemapFace, face_uv_to_dir};
use crate::noise::fbm3;
use crate::stage::Stage;
use crate::surface_field::{mix3, smoothstep};

/// Generic relief-color grading pass driven by the baked biome-weights
/// cubemap and per-biome palettes.
///
/// Runs at the end of an archetype's pipeline, after stages that overwrite
/// albedo (cratering, dunes, impact splotches). For each texel:
///
/// 1. Sample center + 4 cardinal-neighbor heights to derive slope, ridge,
///    and hollow signals.
/// 2. For each biome with non-zero weight at this texel, evaluate the
///    biome's `ReliefPalette` and accumulate weighted color.
/// 3. Mix the result into the existing albedo at `strength`.
///
/// `random_variation` controls the amplitude of low-/mid-frequency fbm
/// added to each palette evaluation so adjacent same-biome texels don't
/// read as a flat color zone. Zero = no jitter.
///
/// Requires `builder.biome_weights_cubemap` to be populated (typically by
/// `bake_surface_field_into_builder`) and `builder.biome_palettes` to have
/// at least one entry. Otherwise a no-op.
#[derive(Debug, Clone, Deserialize)]
pub struct BiomeReliefColor {
    #[serde(default = "default_strength")]
    pub strength: f32,
    #[serde(default = "default_random_variation")]
    pub random_variation: f32,
}

impl Default for BiomeReliefColor {
    fn default() -> Self {
        Self {
            strength: default_strength(),
            random_variation: default_random_variation(),
        }
    }
}

impl Stage for BiomeReliefColor {
    fn name(&self) -> &str {
        "biome_relief_color"
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        let strength = self.strength.max(0.0);
        if strength <= 0.0 || builder.biome_palettes.is_empty() {
            return;
        }

        let res = builder.cubemap_resolution;
        let res_usize = res as usize;
        let radius_m = builder.radius_m;
        let height = builder.height_contributions.height.clone();
        let biome_weights = builder.biome_weights_cubemap.clone();
        let palettes = builder.biome_palettes.clone();
        let sample_angle = std::f32::consts::FRAC_PI_2 / res.max(1) as f32 * 1.35;
        let sample_distance_m = radius_m * sample_angle;
        let seed = builder.stage_seed() as u32;
        let random_variation = self.random_variation.max(0.0);

        for face in CubemapFace::ALL {
            let albedo = builder.albedo_contributions.albedo.face_data_mut(face);
            albedo.par_iter_mut().enumerate().for_each(|(i, texel)| {
                let x = i % res_usize;
                let y = i / res_usize;

                let mix_texel = biome_weights.get(face, x as u32, y as u32);
                if mix_texel.is_empty() {
                    return;
                }

                let u = (x as f32 + 0.5) / res as f32;
                let v = (y as f32 + 0.5) / res as f32;
                let dir = face_uv_to_dir(face, u, v);
                let center_h = height.get(face, x as u32, y as u32);
                let (tangent, bitangent) = tangent_frame(dir);

                let h_e = height.sample_bilinear((dir + tangent * sample_angle).normalize());
                let h_w = height.sample_bilinear((dir - tangent * sample_angle).normalize());
                let h_n = height.sample_bilinear((dir + bitangent * sample_angle).normalize());
                let h_s = height.sample_bilinear((dir - bitangent * sample_angle).normalize());
                let neighbor_mean = (h_e + h_w + h_n + h_s) * 0.25;
                let slope_m_per_m =
                    ((h_e - h_w).hypot(h_n - h_s) / (2.0 * sample_distance_m.max(1.0))).max(0.0);

                let slope_signal = smoothstep(0.018, 0.16, slope_m_per_m);
                let ridge_signal = smoothstep(18.0, 210.0, center_h - neighbor_mean);
                let hollow_signal = smoothstep(18.0, 250.0, neighbor_mean - center_h);

                // Two-band variation noise so palette samples aren't piecewise
                // constant across same-palette regions. Hash the salt with the
                // dominant biome id so neighboring biomes get decorrelated
                // patterns even when their palettes overlap.
                let dom = mix_texel.biome_ids[0] as u32;
                let broad = fbm3(
                    dir.x * 1.55,
                    dir.y * 1.55,
                    dir.z * 1.55,
                    seed ^ 0x91A7_2C3D ^ dom,
                    4,
                    0.55,
                    2.02,
                );
                let mottle = fbm3(
                    dir.x * 12.0,
                    dir.y * 12.0,
                    dir.z * 12.0,
                    seed ^ 0x51F1_6E23 ^ (dom << 8),
                    3,
                    0.52,
                    2.04,
                );
                let variation = (broad * 0.64 + mottle * 0.36) * random_variation;

                let mut target = [0.0_f32; 3];
                let mut total_w = 0.0_f32;
                for (biome_id, weight) in mix_texel.iter_weights() {
                    let Some(palette) = palettes.get(biome_id as usize) else {
                        continue;
                    };
                    let sample = palette.evaluate(
                        center_h,
                        slope_signal,
                        ridge_signal,
                        hollow_signal,
                        variation,
                    );
                    target[0] += sample[0] * weight;
                    target[1] += sample[1] * weight;
                    target[2] += sample[2] * weight;
                    total_w += weight;
                }
                if total_w <= 1.0e-5 {
                    return;
                }
                let target = [
                    target[0] / total_w,
                    target[1] / total_w,
                    target[2] / total_w,
                ];

                // The biome palette IS the body's color authority — see
                // `docs/terrain.md` lines 387-398. With `strength = 1.0`
                // the post-pass fully owns color in flat regions, so the
                // inline base color doesn't compete with the per-biome
                // weighted blend. Lower the stage's `strength` to fade
                // the post-pass back toward the inline color.
                let relief_strength = strength.clamp(0.0, 1.0);

                let alpha = texel[3].max(1.0e-5);
                let base = [texel[0] / alpha, texel[1] / alpha, texel[2] / alpha];
                let graded = mix3(base, target, relief_strength);

                texel[0] = graded[0] * alpha;
                texel[1] = graded[1] * alpha;
                texel[2] = graded[2] * alpha;
            });
        }
    }
}

fn default_strength() -> f32 {
    1.0
}

fn default_random_variation() -> f32 {
    0.78
}

fn tangent_frame(dir: Vec3) -> (Vec3, Vec3) {
    let up = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let tangent = up.cross(dir).normalize();
    let bitangent = dir.cross(tangent);
    (tangent, bitangent)
}
