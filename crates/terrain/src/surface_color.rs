//! Unified static surface-albedo painter.
//!
//! Height, materials, biome weights, craters, and dynamic-layer metadata are
//! produced elsewhere. This module is the single authority that turns those
//! signals into the baked static albedo cubemap.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::body_builder::BodyBuilder;
use crate::cubemap::{Cubemap, CubemapFace, face_uv_to_dir};
use crate::noise::fbm3;
use crate::seeding::splitmix64;
use crate::stages::util::for_face_texels_in_cap;
use crate::surface_field::{BiomeMixTexel, mix3, smoothstep};
use crate::types::Crater;

pub const AGING_OCEANIC_BIOME_OCEAN: u8 = 0;
pub const AGING_OCEANIC_BIOME_SHELF: u8 = 1;
pub const AGING_OCEANIC_BIOME_BEACH: u8 = 2;
pub const AGING_OCEANIC_BIOME_FOREST: u8 = 3;
pub const AGING_OCEANIC_BIOME_GRASSLAND: u8 = 4;
pub const AGING_OCEANIC_BIOME_STEPPE: u8 = 5;
pub const AGING_OCEANIC_BIOME_DESERT: u8 = 6;
pub const AGING_OCEANIC_BIOME_BOREAL: u8 = 7;
pub const AGING_OCEANIC_BIOME_TUNDRA: u8 = 8;
pub const AGING_OCEANIC_BIOME_ROCK: u8 = 9;
pub const AGING_OCEANIC_BIOME_SNOW: u8 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WaterAppearance {
    /// xyz = apparent deep-water linear RGB, w = minimum optical depth in m.
    pub color_depth: [f32; 4],
}

impl WaterAppearance {
    pub const fn new(color: [f32; 3], min_optical_depth_m: f32) -> Self {
        Self {
            color_depth: [color[0], color[1], color[2], min_optical_depth_m],
        }
    }
}

impl Default for WaterAppearance {
    fn default() -> Self {
        Self::new([0.012, 0.040, 0.090], 120.0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BiomeColorPalette {
    pub name: String,
    pub low: [f32; 3],
    pub mid: [f32; 3],
    pub high: [f32; 3],
    pub steep: [f32; 3],
    pub ridge: [f32; 3],
    pub hollow: [f32; 3],
    pub dry_sparse: [f32; 3],
    pub dry_dense: [f32; 3],
    pub wet_sparse: [f32; 3],
    pub wet_dense: [f32; 3],
    #[serde(default = "default_climate_strength")]
    pub climate_strength: f32,
    #[serde(default = "default_relief_strength")]
    pub relief_strength: f32,
    #[serde(default = "default_variation_strength")]
    pub variation_strength: f32,
}

fn default_climate_strength() -> f32 {
    0.65
}

fn default_relief_strength() -> f32 {
    0.70
}

fn default_variation_strength() -> f32 {
    1.0
}

impl BiomeColorPalette {
    pub fn from_relief(name: &str, relief: crate::surface_field::ReliefPalette) -> Self {
        Self {
            name: name.to_string(),
            low: relief.low,
            mid: relief.mid,
            high: relief.high,
            steep: relief.steep,
            ridge: relief.ridge,
            hollow: relief.hollow,
            dry_sparse: relief.mid,
            dry_dense: mix3(relief.mid, relief.hollow, 0.35),
            wet_sparse: mix3(relief.mid, relief.low, 0.35),
            wet_dense: mix3(relief.low, relief.hollow, 0.35),
            climate_strength: 0.20,
            relief_strength: 0.92,
            variation_strength: 0.90,
        }
    }

    fn evaluate(&self, context: &SurfaceColorContext, biome_variation: f32) -> [f32; 3] {
        let relief_strength = self.relief_strength.clamp(0.0, 1.0);
        let rel_h = context.relative_height_m;
        let low = smoothstep(90.0, -1_650.0, rel_h);
        let high = smoothstep(650.0, 3_600.0, rel_h);
        let summit = smoothstep(2_800.0, 4_700.0, rel_h);

        let mut relief = self.mid;
        relief = mix3(relief, self.low, low * 0.80);
        relief = mix3(
            relief,
            self.high,
            (high * 0.70 + summit * 0.20).clamp(0.0, 0.90),
        );
        relief = mix3(
            relief,
            self.hollow,
            context.hollow_signal * 0.24 * relief_strength,
        );
        relief = mix3(
            relief,
            self.steep,
            context.slope_signal * 0.30 * relief_strength,
        );
        relief = mix3(
            relief,
            self.ridge,
            ((context.ridge_signal * 0.24 + high * context.slope_signal * 0.18) * relief_strength)
                .clamp(0.0, 0.46),
        );

        let dry = mix3(self.dry_sparse, self.dry_dense, context.lushness);
        let wet = mix3(self.wet_sparse, self.wet_dense, context.lushness);
        let climate = mix3(dry, wet, context.moisture);
        let color = mix3(
            relief,
            climate,
            self.climate_strength.clamp(0.0, 1.0)
                * (1.0 - context.slope_signal * 0.14 * relief_strength),
        );

        let relief_color = mix3(color, relief, relief_strength * 0.24);
        let v = biome_variation.clamp(-1.0, 1.0) * self.variation_strength;
        [
            (relief_color[0] * (1.0 + v * 0.065)).clamp(0.012, 0.96),
            (relief_color[1] * (1.0 + v * 0.035)).clamp(0.012, 0.96),
            (relief_color[2] * (1.0 - v * 0.030)).clamp(0.010, 0.96),
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SurfaceColorOverprint {
    None,
    ColdDesert,
    AirlessImpact {
        highland_fresh: f32,
        mare_fresh: f32,
        mare_tint: [f32; 3],
        young_crater_age_threshold: f32,
        ray_age_threshold: f32,
        ray_extent_radii: f32,
        ray_half_width: f32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceColorSpec {
    pub palettes: Vec<BiomeColorPalette>,
    pub seed: u64,
    #[serde(default = "default_broad_frequency")]
    pub broad_frequency: f32,
    #[serde(default = "default_mottle_frequency")]
    pub mottle_frequency: f32,
    #[serde(default = "default_surface_variation")]
    pub variation_strength: f32,
    #[serde(default)]
    pub use_material_id_when_biome_empty: bool,
    #[serde(default)]
    pub sea_level_m: Option<f32>,
    #[serde(default)]
    pub water: Option<WaterAppearance>,
    #[serde(default = "default_color_saturation")]
    pub saturation: f32,
    #[serde(default = "default_color_contrast")]
    pub contrast: f32,
    #[serde(default = "default_color_gain")]
    pub gain: f32,
    #[serde(default = "default_overprint")]
    pub overprint: SurfaceColorOverprint,
}

fn default_broad_frequency() -> f32 {
    1.65
}

fn default_mottle_frequency() -> f32 {
    18.0
}

fn default_surface_variation() -> f32 {
    0.78
}

fn default_color_saturation() -> f32 {
    1.0
}

fn default_color_contrast() -> f32 {
    1.0
}

fn default_color_gain() -> f32 {
    1.0
}

fn default_overprint() -> SurfaceColorOverprint {
    SurfaceColorOverprint::None
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceColorContext {
    pub dir: Vec3,
    pub height_m: f32,
    pub relative_height_m: f32,
    pub mare_coverage: f32,
    pub slope_signal: f32,
    pub ridge_signal: f32,
    pub hollow_signal: f32,
    pub latitude_abs: f32,
    pub coast_distance_m: f32,
    pub moisture: f32,
    pub lushness: f32,
    pub broad_noise: f32,
    pub mottle_noise: f32,
}

pub fn paint_surface_albedo(builder: &mut BodyBuilder, spec: &SurfaceColorSpec) {
    if spec.palettes.is_empty() {
        return;
    }

    if let Some(water) = spec.water {
        builder.water_appearance = Some(water);
    }

    let res = builder.cubemap_resolution as usize;
    let radius_m = builder.radius_m;
    let height = builder.height_contributions.height.clone();
    let biome_weights = builder.biome_weights_cubemap.clone();
    let material_ids = builder.material_cubemap.clone();
    let mare_coverage = builder.mare_coverage_cubemap.clone();
    let sample_angle =
        std::f32::consts::FRAC_PI_2 / builder.cubemap_resolution.max(1) as f32 * 1.35;
    let sample_distance_m = radius_m * sample_angle;

    for face in CubemapFace::ALL {
        let albedo = builder.albedo_contributions.albedo.face_data_mut(face);
        for (i, texel) in albedo.iter_mut().enumerate() {
            let x = i % res;
            let y = i / res;
            let u = (x as f32 + 0.5) / res as f32;
            let v = (y as f32 + 0.5) / res as f32;
            let dir = face_uv_to_dir(face, u, v);
            let center_h = height.get(face, x as u32, y as u32);
            let relative_height_m = center_h - spec.sea_level_m.unwrap_or(0.0);
            let (tangent, bitangent) = tangent_frame(dir);

            let h_e = height.sample_bilinear((dir + tangent * sample_angle).normalize());
            let h_w = height.sample_bilinear((dir - tangent * sample_angle).normalize());
            let h_n = height.sample_bilinear((dir + bitangent * sample_angle).normalize());
            let h_s = height.sample_bilinear((dir - bitangent * sample_angle).normalize());
            let neighbor_mean = (h_e + h_w + h_n + h_s) * 0.25;
            let slope_m_per_m =
                ((h_e - h_w).hypot(h_n - h_s) / (2.0 * sample_distance_m.max(1.0))).max(0.0);

            let slope_signal = smoothstep(0.010, 0.095, slope_m_per_m);
            let ridge_signal = smoothstep(18.0, 220.0, center_h - neighbor_mean);
            let hollow_signal = smoothstep(18.0, 260.0, neighbor_mean - center_h);
            let broad_seed = splitmix64(spec.seed ^ 0x51F0_0DCA_10B1_EA55) as u32;
            let mottle_seed = splitmix64(spec.seed ^ 0xA1BE_D011_77A1_0B55) as u32 ^ broad_seed;
            let broad_noise = fbm3(
                dir.x * spec.broad_frequency,
                dir.y * spec.broad_frequency,
                dir.z * spec.broad_frequency,
                broad_seed,
                4,
                0.55,
                2.02,
            );
            let mottle_noise = fbm3(
                dir.x * spec.mottle_frequency,
                dir.y * spec.mottle_frequency,
                dir.z * spec.mottle_frequency,
                mottle_seed,
                3,
                0.52,
                2.04,
            );
            let latitude_abs = dir.y.clamp(-1.0, 1.0).asin().abs() / std::f32::consts::FRAC_PI_2;
            let moisture = (0.50 + broad_noise * 0.34 - latitude_abs * 0.10
                + (spec.sea_level_m.is_some() as u8 as f32) * 0.04)
                .clamp(0.0, 1.0);
            let lushness = (0.50 + mottle_noise * 0.30 + broad_noise * 0.12).clamp(0.0, 1.0);
            let mare = mare_coverage.get(face, x as u32, y as u32).clamp(0.0, 1.0);
            let context = SurfaceColorContext {
                dir,
                height_m: center_h,
                relative_height_m,
                mare_coverage: mare,
                slope_signal,
                ridge_signal,
                hollow_signal,
                latitude_abs,
                coast_distance_m: 0.0,
                moisture,
                lushness,
                broad_noise,
                mottle_noise,
            };

            let mix_texel = biome_weights.get(face, x as u32, y as u32);
            let material_id = material_ids.get(face, x as u32, y as u32);
            let mut color = evaluate_mix(mix_texel, material_id, spec, &context);
            if matches!(spec.overprint, SurfaceColorOverprint::ColdDesert) {
                color = cold_desert_grade(color);
            }
            color = apply_surface_grade(color, spec);
            *texel = [color[0], color[1], color[2], 1.0];
        }
    }

    if matches!(spec.overprint, SurfaceColorOverprint::AirlessImpact { .. }) {
        soften_mare_contacts(builder);
    }

    if let SurfaceColorOverprint::AirlessImpact {
        highland_fresh,
        mare_fresh,
        mare_tint,
        young_crater_age_threshold,
        ray_age_threshold: _,
        ray_extent_radii: _,
        ray_half_width: _,
    } = spec.overprint
    {
        apply_airless_crater_overprints(
            builder,
            AirlessOverprintConfig {
                highland_fresh,
                mare_fresh,
                mare_tint,
                young_crater_age_threshold,
            },
        );
    }
}

fn evaluate_mix(
    mix_texel: BiomeMixTexel,
    material_id: u8,
    spec: &SurfaceColorSpec,
    context: &SurfaceColorContext,
) -> [f32; 3] {
    if mix_texel.is_empty() {
        if matches!(spec.overprint, SurfaceColorOverprint::AirlessImpact { .. })
            && spec.use_material_id_when_biome_empty
            && spec.palettes.len() >= 2
        {
            return evaluate_airless_provenance_color(material_id, spec, context);
        }
        let palette_id = if spec.use_material_id_when_biome_empty {
            material_id as usize
        } else {
            0
        };
        let palette = &spec.palettes[palette_id.min(spec.palettes.len() - 1)];
        return palette.evaluate(
            context,
            (context.broad_noise * 0.64 + context.mottle_noise * 0.36) * spec.variation_strength,
        );
    }

    let mut color = [0.0_f32; 3];
    let mut total = 0.0_f32;
    for (biome_id, weight) in mix_texel.iter_weights() {
        let Some(palette) = spec.palettes.get(biome_id as usize) else {
            continue;
        };
        let biome_variation =
            (context.broad_noise * 0.64 + context.mottle_noise * 0.36) * spec.variation_strength;
        let sample = palette.evaluate(context, biome_variation);
        color[0] += sample[0] * weight;
        color[1] += sample[1] * weight;
        color[2] += sample[2] * weight;
        total += weight;
    }
    if total <= 1.0e-5 {
        spec.palettes[0].evaluate(context, 0.0)
    } else {
        [color[0] / total, color[1] / total, color[2] / total]
    }
}

fn evaluate_airless_provenance_color(
    material_id: u8,
    spec: &SurfaceColorSpec,
    context: &SurfaceColorContext,
) -> [f32; 3] {
    let variation =
        (context.broad_noise * 0.64 + context.mottle_noise * 0.36) * spec.variation_strength;
    let highland = spec.palettes[0].evaluate(context, variation);
    let mare = spec.palettes[1].evaluate(context, variation * 0.58);
    let provenance = context.mare_coverage.clamp(0.0, 1.0);

    if provenance <= 0.001 {
        let palette_id = material_id as usize;
        if palette_id >= 2 {
            return spec.palettes[palette_id.min(spec.palettes.len() - 1)]
                .evaluate(context, variation);
        }
    }

    mix3(highland, mare, provenance)
}

fn apply_surface_grade(color: [f32; 3], spec: &SurfaceColorSpec) -> [f32; 3] {
    let sat = spec.saturation.max(0.0);
    let contrast = spec.contrast.max(0.0);
    let gain = spec.gain.max(0.0);
    let luma = color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722;
    let saturated = [
        luma + (color[0] - luma) * sat,
        luma + (color[1] - luma) * sat,
        luma + (color[2] - luma) * sat,
    ];
    let pivot = 0.22;
    [
        ((pivot + (saturated[0] - pivot) * contrast) * gain).clamp(0.006, 0.98),
        ((pivot + (saturated[1] - pivot) * contrast) * gain).clamp(0.006, 0.98),
        ((pivot + (saturated[2] - pivot) * contrast) * gain).clamp(0.006, 0.98),
    ]
}

fn tangent_frame(dir: Vec3) -> (Vec3, Vec3) {
    let up = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let tangent = up.cross(dir).normalize();
    let bitangent = dir.cross(tangent);
    (tangent, bitangent)
}

fn cold_desert_grade(color: [f32; 3]) -> [f32; 3] {
    [
        (color[0] * 1.045 + 0.008).clamp(0.025, 0.94),
        (color[1] * 0.915 + color[0] * 0.016).clamp(0.025, 0.94),
        (color[2] * 0.84 + color[1] * 0.018).clamp(0.020, 0.94),
    ]
}

fn soften_mare_contacts(builder: &mut BodyBuilder) {
    let res = builder.cubemap_resolution;
    let mare_coverage = builder.mare_coverage_cubemap.clone();
    let source = builder.albedo_contributions.albedo.clone();

    for face in CubemapFace::ALL {
        let out = builder.albedo_contributions.albedo.face_data_mut(face);
        for y in 0..res {
            for x in 0..res {
                let center_mare = mare_coverage.get(face, x, y).clamp(0.0, 1.0);
                let mut mare = 0.0_f32;
                let mut total = 0.0_f32;
                let mut avg = [0.0_f32; 3];

                for oy in -3..=3 {
                    for ox in -3..=3 {
                        let nx = (x as i32 + ox).clamp(0, res as i32 - 1) as u32;
                        let ny = (y as i32 + oy).clamp(0, res as i32 - 1) as u32;
                        mare += mare_coverage.get(face, nx, ny).clamp(0.0, 1.0);
                        let c = source.get(face, nx, ny);
                        avg[0] += c[0];
                        avg[1] += c[1];
                        avg[2] += c[2];
                        total += 1.0;
                    }
                }

                let mare_fraction = mare / total.max(1.0);
                let gradient = (center_mare - mare_fraction).abs();
                if !(0.03..=0.97).contains(&mare_fraction) || gradient < 0.025 {
                    continue;
                }

                avg[0] /= total;
                avg[1] /= total;
                avg[2] /= total;

                let boundary = 1.0 - (mare_fraction * 2.0 - 1.0).abs();
                let mix = (boundary * gradient * 3.2).clamp(0.0, 0.42);
                let idx = (y * res + x) as usize;
                let current = out[idx];
                out[idx] = [
                    current[0] + (avg[0] - current[0]) * mix,
                    current[1] + (avg[1] - current[1]) * mix,
                    current[2] + (avg[2] - current[2]) * mix,
                    1.0,
                ];
            }
        }
    }
}

#[derive(Clone, Copy)]
struct AirlessOverprintConfig {
    highland_fresh: f32,
    mare_fresh: f32,
    mare_tint: [f32; 3],
    young_crater_age_threshold: f32,
}

fn apply_airless_crater_overprints(builder: &mut BodyBuilder, cfg: AirlessOverprintConfig) {
    let res = builder.cubemap_resolution;
    let body_radius = builder.radius_m;
    let bake_threshold = builder.cubemap_bake_threshold_m;
    let craters: Vec<Crater> = builder
        .craters
        .iter()
        .filter(|c| c.radius_m >= bake_threshold)
        .cloned()
        .collect();
    if craters.is_empty() {
        return;
    }
    let mare_coverage = builder.mare_coverage_cubemap.clone();

    for crater in &craters {
        let center = crater.center.normalize();
        let radius_m = crater.radius_m;
        let old_memory = smoothstep(12_000.0, 45_000.0, radius_m);
        let young_visibility = smoothstep(7_500.0, 14_000.0, radius_m);
        let freshness =
            crater_freshness(crater.age_gyr, cfg.young_crater_age_threshold) * young_visibility;
        // Ray systems are too high-contrast for the baked low-frequency
        // cubemap/equirect. They belong in an analytic detail layer; here they
        // become continent-sized bright islands once filtered down to map scale.
        if old_memory <= 0.001 && freshness <= 0.001 {
            continue;
        }
        let cap_angle = (radius_m * 2.5) / body_radius;
        // Ancient small craters should shape the height field, not pepper the
        // low-frequency albedo. Keep persistent color memory only on large
        // resolved craters; young mid-sized craters still get bright ejecta.
        let crater_mare =
            crater_mare_coverage(&mare_coverage, center, radius_m, body_radius).clamp(0.0, 1.0);
        let mare_burial =
            smoothstep(0.18, 0.78, crater_mare) * smoothstep(0.55, 2.4, crater.age_gyr);
        let persistence = (old_memory * 0.0015 * (1.0 - mare_burial * 0.96)
            + freshness * 0.026 * (1.0 - mare_burial * 0.35))
            .clamp(0.0, 0.08);

        for face in CubemapFace::ALL {
            let albedo = builder.albedo_contributions.albedo.face_data_mut(face);
            for_face_texels_in_cap(face, res, center, cap_angle, |x, y, _dir, angular_dist| {
                let surface_dist = angular_dist * body_radius;
                let t = surface_dist / radius_m;
                let mut delta = 0.0_f32;
                if t < 0.55 {
                    delta -= 0.85 * (1.0 - t / 0.55);
                }
                if (0.72..=1.28).contains(&t) {
                    let rim_w = 1.0 - (t - 1.0).abs() / 0.28;
                    delta += 1.05 * rim_w.max(0.0);
                }
                if t > 1.0 && t < 2.5 {
                    delta += 0.58 * t.powi(-3) * ((2.5 - t) / 1.5).clamp(0.0, 1.0);
                }

                if delta.abs() <= 0.001 {
                    return;
                }

                let idx = (y * res + x) as usize;
                let current = albedo[idx];
                let base = [current[0], current[1], current[2]];
                let local_mare = mare_coverage.get(face, x, y).clamp(0.0, 1.0);
                let target = if delta >= 0.0 {
                    let highland_fresh =
                        [cfg.highland_fresh, cfg.highland_fresh, cfg.highland_fresh];
                    let mare_fresh = [
                        cfg.mare_fresh * cfg.mare_tint[0],
                        cfg.mare_fresh * cfg.mare_tint[1],
                        cfg.mare_fresh * cfg.mare_tint[2],
                    ];
                    let excavated = (1.0 - freshness * 0.75).clamp(0.0, 1.0);
                    mix3(highland_fresh, mare_fresh, local_mare * excavated)
                } else {
                    mix3([0.055, 0.055, 0.058], [0.035, 0.035, 0.038], local_mare)
                };
                let local_burial =
                    smoothstep(0.25, 0.82, local_mare) * smoothstep(0.55, 2.4, crater.age_gyr);
                let mix = (delta.abs() * persistence * (1.0 - local_burial * 0.74)).clamp(0.0, 1.0);
                let out = mix3(base, target, mix);
                albedo[idx] = [out[0], out[1], out[2], 1.0];
            });
        }
    }
}

fn crater_mare_coverage(
    coverage: &Cubemap<f32>,
    center: Vec3,
    crater_radius_m: f32,
    body_radius_m: f32,
) -> f32 {
    let mut mare = coverage.sample_bilinear(center).clamp(0.0, 1.0) * 1.5;
    let mut total = 1.5;

    for &(radius_frac, weight, samples) in &[(0.42, 1.0, 8_u32), (0.78, 0.55, 12_u32)] {
        let sample_angle = (crater_radius_m * radius_frac) / body_radius_m;
        for i in 0..samples {
            let dir = ring_dir(
                center,
                sample_angle,
                i as f32 * std::f32::consts::TAU / samples as f32,
            );
            mare += coverage.sample_bilinear(dir).clamp(0.0, 1.0) * weight;
            total += weight;
        }
    }

    (mare / total.max(1.0)).clamp(0.0, 1.0)
}

fn ring_dir(center: Vec3, angle: f32, azimuth: f32) -> Vec3 {
    let up = if center.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let right = center.cross(up).normalize();
    let forward = right.cross(center).normalize();
    let (sin_phi, cos_phi) = azimuth.sin_cos();
    let (sin_a, cos_a) = angle.sin_cos();
    (center * cos_a + (right * cos_phi + forward * sin_phi) * sin_a).normalize()
}

fn crater_freshness(age_gyr: f32, threshold_gyr: f32) -> f32 {
    if threshold_gyr <= 0.0 || age_gyr >= threshold_gyr {
        0.0
    } else {
        let t = 1.0 - age_gyr / threshold_gyr;
        t * t
    }
}

fn pal(
    name: &str,
    low: [f32; 3],
    mid: [f32; 3],
    high: [f32; 3],
    steep: [f32; 3],
    ridge: [f32; 3],
    hollow: [f32; 3],
    dry_sparse: [f32; 3],
    dry_dense: [f32; 3],
    wet_sparse: [f32; 3],
    wet_dense: [f32; 3],
    climate_strength: f32,
    relief_strength: f32,
    variation_strength: f32,
) -> BiomeColorPalette {
    BiomeColorPalette {
        name: name.to_string(),
        low,
        mid,
        high,
        steep,
        ridge,
        hollow,
        dry_sparse,
        dry_dense,
        wet_sparse,
        wet_dense,
        climate_strength,
        relief_strength,
        variation_strength,
    }
}

impl SurfaceColorSpec {
    pub fn aging_oceanic_homeworld(seed: u64, sea_level_m: f32) -> Self {
        Self {
            palettes: vec![
                // Seabed palettes paint the actual ocean floor (basaltic
                // crust + abyssal sediment). The water column above is now
                // rendered by `BodyWaterMaterial` (terrain LOD) and the
                // impostor's `shade_water` (orbital), so the terrain bake
                // must NOT pre-tint sub-sea-level texels blue — that would
                // double-stain the visible water colour.
                pal(
                    "deep_ocean",
                    [0.022, 0.020, 0.016],
                    [0.038, 0.034, 0.028],
                    [0.062, 0.054, 0.044],
                    [0.034, 0.030, 0.024],
                    [0.075, 0.064, 0.052],
                    [0.018, 0.016, 0.013],
                    [0.040, 0.035, 0.028],
                    [0.030, 0.026, 0.020],
                    [0.046, 0.040, 0.032],
                    [0.034, 0.030, 0.024],
                    0.20,
                    0.80,
                    0.45,
                ),
                pal(
                    "shelf",
                    [0.090, 0.075, 0.052],
                    [0.130, 0.108, 0.075],
                    [0.180, 0.155, 0.110],
                    [0.075, 0.062, 0.044],
                    [0.195, 0.165, 0.115],
                    [0.055, 0.046, 0.032],
                    [0.140, 0.118, 0.082],
                    [0.110, 0.092, 0.064],
                    [0.150, 0.128, 0.090],
                    [0.115, 0.098, 0.070],
                    0.40,
                    0.65,
                    0.55,
                ),
                pal(
                    "beach",
                    [0.560, 0.455, 0.275],
                    [0.690, 0.590, 0.390],
                    [0.760, 0.685, 0.495],
                    [0.430, 0.330, 0.210],
                    [0.800, 0.720, 0.540],
                    [0.500, 0.390, 0.245],
                    [0.700, 0.555, 0.310],
                    [0.590, 0.475, 0.275],
                    [0.690, 0.610, 0.395],
                    [0.550, 0.500, 0.330],
                    0.46,
                    0.80,
                    0.70,
                ),
                pal(
                    "forest",
                    [0.030, 0.105, 0.026],
                    [0.052, 0.185, 0.032],
                    [0.095, 0.255, 0.046],
                    [0.026, 0.070, 0.026],
                    [0.125, 0.300, 0.060],
                    [0.014, 0.060, 0.016],
                    [0.105, 0.230, 0.048],
                    [0.030, 0.130, 0.024],
                    [0.055, 0.245, 0.036],
                    [0.010, 0.090, 0.014],
                    0.92,
                    0.16,
                    1.18,
                ),
                pal(
                    "grassland",
                    [0.135, 0.305, 0.060],
                    [0.220, 0.430, 0.075],
                    [0.330, 0.535, 0.115],
                    [0.125, 0.220, 0.070],
                    [0.390, 0.585, 0.145],
                    [0.075, 0.225, 0.040],
                    [0.385, 0.440, 0.105],
                    [0.220, 0.360, 0.070],
                    [0.165, 0.430, 0.060],
                    [0.050, 0.250, 0.026],
                    0.74,
                    0.24,
                    0.92,
                ),
                pal(
                    "steppe",
                    [0.265, 0.265, 0.090],
                    [0.410, 0.360, 0.105],
                    [0.540, 0.485, 0.165],
                    [0.235, 0.185, 0.095],
                    [0.600, 0.545, 0.200],
                    [0.200, 0.205, 0.075],
                    [0.520, 0.405, 0.120],
                    [0.350, 0.315, 0.088],
                    [0.365, 0.410, 0.105],
                    [0.210, 0.305, 0.065],
                    0.68,
                    0.38,
                    0.92,
                ),
                pal(
                    "desert",
                    [0.420, 0.265, 0.115],
                    [0.610, 0.405, 0.170],
                    [0.735, 0.535, 0.255],
                    [0.310, 0.185, 0.095],
                    [0.800, 0.590, 0.300],
                    [0.340, 0.220, 0.105],
                    [0.720, 0.470, 0.190],
                    [0.545, 0.350, 0.140],
                    [0.575, 0.445, 0.205],
                    [0.390, 0.315, 0.130],
                    0.60,
                    0.46,
                    0.92,
                ),
                pal(
                    "boreal",
                    [0.055, 0.135, 0.070],
                    [0.090, 0.205, 0.080],
                    [0.155, 0.285, 0.105],
                    [0.045, 0.100, 0.062],
                    [0.205, 0.340, 0.125],
                    [0.030, 0.095, 0.050],
                    [0.170, 0.255, 0.090],
                    [0.070, 0.170, 0.065],
                    [0.105, 0.260, 0.085],
                    [0.035, 0.145, 0.055],
                    0.80,
                    0.22,
                    0.86,
                ),
                pal(
                    "tundra",
                    [0.255, 0.300, 0.220],
                    [0.335, 0.365, 0.255],
                    [0.460, 0.450, 0.325],
                    [0.220, 0.235, 0.180],
                    [0.555, 0.535, 0.390],
                    [0.190, 0.245, 0.175],
                    [0.410, 0.390, 0.285],
                    [0.300, 0.325, 0.235],
                    [0.320, 0.375, 0.255],
                    [0.215, 0.295, 0.205],
                    0.58,
                    0.70,
                    0.70,
                ),
                pal(
                    "rock",
                    [0.320, 0.235, 0.160],
                    [0.440, 0.310, 0.205],
                    [0.610, 0.480, 0.315],
                    [0.180, 0.135, 0.105],
                    [0.700, 0.585, 0.395],
                    [0.250, 0.180, 0.125],
                    [0.500, 0.320, 0.175],
                    [0.385, 0.275, 0.160],
                    [0.420, 0.335, 0.220],
                    [0.310, 0.260, 0.180],
                    0.34,
                    0.94,
                    0.80,
                ),
                pal(
                    "snow",
                    [0.720, 0.735, 0.745],
                    [0.870, 0.875, 0.890],
                    [0.945, 0.950, 0.965],
                    [0.640, 0.655, 0.670],
                    [0.980, 0.985, 0.995],
                    [0.670, 0.690, 0.705],
                    [0.760, 0.760, 0.740],
                    [0.700, 0.710, 0.710],
                    [0.890, 0.905, 0.930],
                    [0.820, 0.845, 0.875],
                    0.50,
                    0.86,
                    0.45,
                ),
            ],
            seed,
            broad_frequency: 1.35,
            mottle_frequency: 18.0,
            variation_strength: 1.08,
            use_material_id_when_biome_empty: false,
            sea_level_m: Some(sea_level_m),
            water: Some(WaterAppearance::new([0.010, 0.055, 0.145], 140.0)),
            saturation: 1.42,
            contrast: 1.20,
            gain: 0.96,
            overprint: SurfaceColorOverprint::None,
        }
    }

    pub fn cold_desert(seed: u64, palettes: &[crate::surface_field::ReliefPalette]) -> Self {
        let converted = palettes
            .iter()
            .enumerate()
            .map(|(i, p)| BiomeColorPalette::from_relief(&format!("cold_desert_{i}"), *p))
            .collect();
        Self {
            palettes: converted,
            seed,
            broad_frequency: 1.80,
            mottle_frequency: 22.0,
            variation_strength: 0.74,
            use_material_id_when_biome_empty: false,
            sea_level_m: None,
            water: None,
            saturation: 1.0,
            contrast: 1.0,
            gain: 1.0,
            overprint: SurfaceColorOverprint::ColdDesert,
        }
    }

    pub fn airless_impact(
        seed: u64,
        highland_mature: f32,
        highland_fresh: f32,
        mare_mature: f32,
        mare_fresh: f32,
        mare_tint: [f32; 3],
        young_crater_age_threshold: f32,
        ray_age_threshold: f32,
        ray_extent_radii: f32,
        ray_half_width: f32,
    ) -> Self {
        let gray = |v: f32| [v, v, v];
        let mare = [
            mare_mature * mare_tint[0],
            mare_mature * mare_tint[1],
            mare_mature * mare_tint[2],
        ];
        Self {
            palettes: vec![
                pal(
                    "highland",
                    gray(highland_mature * 0.96),
                    gray(highland_mature),
                    gray(highland_fresh * 0.82),
                    gray(highland_mature * 0.92),
                    gray(highland_mature * 1.08),
                    gray(highland_mature * 0.93),
                    gray(highland_mature * 0.95),
                    gray(highland_mature * 0.82),
                    gray(highland_mature * 1.05),
                    gray(highland_mature * 0.90),
                    0.10,
                    0.14,
                    0.46,
                ),
                pal(
                    "mare",
                    [mare[0] * 0.97, mare[1] * 0.97, mare[2] * 0.97],
                    mare,
                    [
                        mare_fresh * mare_tint[0],
                        mare_fresh * mare_tint[1],
                        mare_fresh * mare_tint[2],
                    ],
                    [mare[0] * 0.96, mare[1] * 0.96, mare[2] * 0.96],
                    [mare[0] * 1.03, mare[1] * 1.03, mare[2] * 1.03],
                    [mare[0] * 0.97, mare[1] * 0.97, mare[2] * 0.97],
                    mare,
                    mare,
                    mare,
                    mare,
                    0.08,
                    0.04,
                    0.18,
                ),
                pal(
                    "fresh_ejecta",
                    gray(highland_fresh * 0.76),
                    gray(highland_fresh),
                    gray((highland_fresh * 1.12).min(0.90)),
                    gray(highland_fresh * 0.62),
                    gray((highland_fresh * 1.18).min(0.92)),
                    gray(highland_fresh * 0.70),
                    gray(highland_fresh),
                    gray(highland_fresh * 0.90),
                    gray(highland_fresh),
                    gray(highland_fresh * 0.92),
                    0.08,
                    0.12,
                    0.34,
                ),
                pal(
                    "mature_regolith",
                    gray(highland_mature * 0.62),
                    gray(highland_mature * 0.90),
                    gray(highland_mature * 1.20),
                    gray(highland_mature * 0.48),
                    gray(highland_mature * 1.35),
                    gray(highland_mature * 0.55),
                    gray(highland_mature * 0.88),
                    gray(highland_mature * 0.78),
                    gray(highland_mature * 0.96),
                    gray(highland_mature * 0.84),
                    0.08,
                    0.12,
                    0.40,
                ),
            ],
            seed,
            broad_frequency: 3.5,
            mottle_frequency: 18.0,
            variation_strength: 0.42,
            use_material_id_when_biome_empty: true,
            sea_level_m: None,
            water: None,
            saturation: 1.0,
            contrast: 1.0,
            gain: 1.0,
            overprint: SurfaceColorOverprint::AirlessImpact {
                highland_fresh,
                mare_fresh,
                mare_tint,
                young_crater_age_threshold,
                ray_age_threshold,
                ray_extent_radii,
                ray_half_width,
            },
        }
    }

    pub fn ocean(seed: u64, sea_level_m: f32, seabed: [f32; 3], water: WaterAppearance) -> Self {
        Self {
            palettes: vec![pal(
                "seabed", seabed, seabed, seabed, seabed, seabed, seabed, seabed, seabed, seabed,
                seabed, 0.0, 0.0, 0.25,
            )],
            seed,
            broad_frequency: 1.0,
            mottle_frequency: 8.0,
            variation_strength: 0.12,
            use_material_id_when_biome_empty: true,
            sea_level_m: Some(sea_level_m),
            water: Some(water),
            saturation: 1.0,
            contrast: 1.0,
            gain: 1.0,
            overprint: SurfaceColorOverprint::None,
        }
    }
}
