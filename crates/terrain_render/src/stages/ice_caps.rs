use glam::Vec3;
use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{CubemapFace, face_uv_to_dir};
use crate::noise::fbm3;
use crate::seeding::sub_seed;
use crate::stage::Stage;
use crate::surface_field::{mix3, quantize_unit_to_u8, smoothstep};
use crate::types::{IceCapSpec, Material};

/// Late, data-driven ice veneer for polar caps.
///
/// Deprecated debug-only baked ice veneer.
///
/// Production terrain compilers should not schedule this stage. Polar ice is
/// represented as `DynamicSurfaceLayers::ice_caps` so runtime state can be
/// shared by the impostor and future ground tiles.
#[derive(Debug, Clone, Deserialize)]
pub struct IceCaps {
    #[serde(default)]
    pub caps: Vec<IceCapSpec>,
}

impl Stage for IceCaps {
    fn name(&self) -> &str {
        "ice_caps"
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        if self.caps.is_empty() {
            return;
        }

        let layers = prepare_layers(builder, &self.caps);
        if layers.is_empty() {
            return;
        }

        let res = builder.cubemap_resolution as usize;

        for face in CubemapFace::ALL {
            let heights = builder.height_contributions.height.face_data_mut(face);
            let albedo = builder.albedo_contributions.albedo.face_data_mut(face);
            let roughness = builder.roughness_cubemap.face_data_mut(face);
            let materials = builder.material_cubemap.face_data_mut(face);

            heights
                .par_iter_mut()
                .zip(albedo.par_iter_mut())
                .zip(roughness.par_iter_mut())
                .zip(materials.par_iter_mut())
                .enumerate()
                .for_each(|(i, (((height, color), rough), material))| {
                    let x = i % res;
                    let y = i / res;
                    let u = (x as f32 + 0.5) / res as f32;
                    let v = (y as f32 + 0.5) / res as f32;
                    let dir = face_uv_to_dir(face, u, v);

                    let mut winning_coverage = 0.0_f32;
                    let mut winning_material = *material;

                    for layer in &layers {
                        let coverage = ice_cap_coverage(dir, layer);
                        if coverage <= 0.001 {
                            continue;
                        }

                        let texture = ice_texture(dir, layer);
                        let edge_band = (coverage * (1.0 - coverage) * 4.0).clamp(0.0, 1.0);
                        let clean_ice = (0.72 + texture * 0.18 + coverage * 0.22
                            - edge_band * 0.10)
                            .clamp(0.0, 1.0);
                        let cap_color = mix3(
                            layer.spec.dust_albedo_linear,
                            layer.spec.albedo_linear,
                            clean_ice,
                        );

                        let alpha = color[3].max(1.0e-5);
                        let base = [color[0] / alpha, color[1] / alpha, color[2] / alpha];
                        let blend = ((coverage + edge_band * 0.10) * layer.spec.albedo_strength)
                            .clamp(0.0, 1.0);
                        let mixed = mix3(base, cap_color, blend);
                        color[0] = mixed[0] * alpha;
                        color[1] = mixed[1] * alpha;
                        color[2] = mixed[2] * alpha;

                        let thickness = layer.spec.max_thickness_m
                            * coverage
                            * (0.72 + texture * 0.28).clamp(0.35, 1.0);
                        *height += thickness;

                        let rough_f = *rough as f32 / 255.0;
                        let rough_blend =
                            (coverage * layer.spec.roughness_strength).clamp(0.0, 1.0);
                        *rough = quantize_unit_to_u8(
                            rough_f + (layer.spec.roughness - rough_f) * rough_blend,
                        );

                        if coverage > winning_coverage {
                            winning_coverage = coverage;
                            winning_material = layer.material_id;
                        }
                    }

                    if winning_coverage >= 0.50 {
                        *material = winning_material;
                    }
                });
        }
    }
}

#[derive(Clone, Copy)]
struct IceCapLayer {
    spec: IceCapSpec,
    axis: Vec3,
    climate: IceCapClimate,
    edge_seed: u32,
    texture_seed: u32,
    material_id: u8,
}

#[derive(Clone, Copy)]
struct IceCapClimate {
    cold_abs_latitude_deg: f32,
    edge_shift_deg: f32,
    solid_shift_deg: f32,
    coverage_strength: f32,
}

fn prepare_layers(builder: &mut BodyBuilder, caps: &[IceCapSpec]) -> Vec<IceCapLayer> {
    let effective_obliquity_deg = effective_obliquity_deg(builder.axial_tilt_rad);

    caps.iter()
        .enumerate()
        .filter_map(|(i, spec)| {
            let axis = spec.axis.try_normalize()?;
            if !spec.north && !spec.south {
                return None;
            }

            let material_id = append_ice_material(builder, spec)?;
            let salt = format!("ice_cap:{i}");
            Some(IceCapLayer {
                spec: *spec,
                axis,
                climate: ice_cap_climate(effective_obliquity_deg, spec.obliquity_response),
                edge_seed: sub_seed(builder.stage_seed(), &format!("{salt}:edge")) as u32,
                texture_seed: sub_seed(builder.stage_seed(), &format!("{salt}:texture")) as u32,
                material_id,
            })
        })
        .collect()
}

fn append_ice_material(builder: &mut BodyBuilder, spec: &IceCapSpec) -> Option<u8> {
    if builder.materials.len() > u8::MAX as usize {
        return None;
    }

    let id = builder.materials.len() as u8;
    builder.materials.push(Material {
        albedo: spec.albedo_linear,
        roughness: spec.roughness,
    });
    Some(id)
}

fn ice_cap_coverage(dir: Vec3, layer: &IceCapLayer) -> f32 {
    let spec = layer.spec;
    let climate = layer.climate;
    let axis_dot = dir.dot(layer.axis).clamp(-1.0, 1.0);
    let sample_lat_deg = axis_dot.asin().to_degrees();
    let edge_noise = fbm3(
        dir.x * spec.noise_frequency,
        dir.y * spec.noise_frequency,
        dir.z * spec.noise_frequency,
        layer.edge_seed,
        4,
        0.55,
        2.03,
    );
    let lace_noise = fbm3(
        dir.x * spec.noise_frequency * 4.2,
        dir.y * spec.noise_frequency * 4.2,
        dir.z * spec.noise_frequency * 4.2,
        layer.edge_seed ^ 0xA71C_3E55,
        3,
        0.52,
        2.07,
    );
    let edge_latitude = (spec.edge_latitude_deg + climate.edge_shift_deg).clamp(0.0, 89.5);
    let solid_latitude = (spec.solid_latitude_deg + climate.solid_shift_deg)
        .max(edge_latitude + 0.5)
        .clamp(edge_latitude + 0.5, 90.0);
    let outer_half_width =
        (90.0 - edge_latitude + edge_noise * spec.edge_noise_deg).clamp(0.5, 90.0);
    let solid_half_width = (90.0 - solid_latitude + edge_noise * spec.edge_noise_deg * 0.38)
        .clamp(0.0, outer_half_width - 0.5);
    let lace = lace_noise * spec.edge_noise_deg * 0.30;

    let mut coverage = 0.0_f32;
    if spec.north {
        let center = climate.cold_abs_latitude_deg;
        let distance = (sample_lat_deg - center).abs();
        coverage = coverage.max(smoothstep(
            outer_half_width,
            solid_half_width,
            distance + lace,
        ));
    }
    if spec.south {
        let center = -climate.cold_abs_latitude_deg;
        let distance = (sample_lat_deg - center).abs();
        coverage = coverage.max(smoothstep(
            outer_half_width,
            solid_half_width,
            distance - lace,
        ));
    }

    let coverage = sharpen_coverage(coverage, spec.edge_sharpness);
    (coverage * climate.coverage_strength).clamp(0.0, 1.0)
}

fn ice_texture(dir: Vec3, layer: &IceCapLayer) -> f32 {
    let broad = fbm3(
        dir.x * 2.7,
        dir.y * 2.7,
        dir.z * 2.7,
        layer.texture_seed,
        4,
        0.55,
        2.01,
    );
    let mottle = fbm3(
        dir.x * 13.0,
        dir.y * 13.0,
        dir.z * 13.0,
        layer.texture_seed ^ 0x51F1_6E23,
        3,
        0.52,
        2.04,
    );
    (0.5 + broad * 0.34 + mottle * 0.16).clamp(0.0, 1.0)
}

fn effective_obliquity_deg(axial_tilt_rad: f32) -> f32 {
    let mut deg = axial_tilt_rad.abs().to_degrees() % 360.0;
    if deg > 180.0 {
        deg = 360.0 - deg;
    }
    if deg > 90.0 {
        deg = 180.0 - deg;
    }
    deg.clamp(0.0, 90.0)
}

fn ice_cap_climate(effective_obliquity_deg: f32, response: f32) -> IceCapClimate {
    let response = response.clamp(0.0, 1.0);
    let annual_min_latitude = minimum_annual_insolation_latitude_deg(effective_obliquity_deg);
    let migration = smoothstep(54.0, 64.0, effective_obliquity_deg);
    let moderate = smoothstep(4.0, 28.0, effective_obliquity_deg)
        * (1.0 - smoothstep(44.0, 56.0, effective_obliquity_deg));

    // Annual mean insolation sets the cold latitude, but below the ~54 degree
    // inversion point the rotational poles remain the cold attractors. Keep
    // the target pinned to the poles until that regime transition; otherwise
    // near-threshold tilts turn a pole-tuned cap width into a huge mid-latitude
    // sheet.
    let cold_abs_latitude_deg = 90.0 + (annual_min_latitude - 90.0) * migration * response;
    let edge_shift_deg = -2.2 * moderate * response;
    let solid_shift_deg = -1.0 * moderate * response;
    let strength = (1.0 + 0.08 * moderate).clamp(1.0, 1.08);

    IceCapClimate {
        cold_abs_latitude_deg,
        edge_shift_deg,
        solid_shift_deg,
        coverage_strength: 1.0 + (strength - 1.0) * response,
    }
}

fn sharpen_coverage(coverage: f32, edge_sharpness: f32) -> f32 {
    let sharpness = edge_sharpness.clamp(0.0, 1.0);
    let half_width = 0.5 - sharpness * 0.42;
    smoothstep(0.5 - half_width, 0.5 + half_width, coverage)
}

fn minimum_annual_insolation_latitude_deg(obliquity_deg: f32) -> f32 {
    let obliquity_rad = obliquity_deg.to_radians();
    let mut best_latitude = 90.0_f32;
    let mut best_insolation = f32::INFINITY;

    for i in 0..=180 {
        let latitude_deg = i as f32 * 0.5;
        let insolation = annual_mean_insolation(latitude_deg.to_radians(), obliquity_rad);
        if insolation < best_insolation {
            best_insolation = insolation;
            best_latitude = latitude_deg;
        }
    }

    best_latitude
}

fn annual_mean_insolation(latitude_rad: f32, obliquity_rad: f32) -> f32 {
    let latitude_rad = latitude_rad.clamp(
        -std::f32::consts::FRAC_PI_2 + 1.0e-4,
        std::f32::consts::FRAC_PI_2 - 1.0e-4,
    );
    let sin_obliquity = obliquity_rad.sin();
    let mut total = 0.0_f32;
    const STEPS: u32 = 360;

    for step in 0..STEPS {
        let solar_longitude = (step as f32 + 0.5) / STEPS as f32 * std::f32::consts::TAU;
        let declination = (sin_obliquity * solar_longitude.sin()).asin();
        total += daily_mean_insolation(latitude_rad, declination);
    }

    total / STEPS as f32
}

fn daily_mean_insolation(latitude_rad: f32, declination_rad: f32) -> f32 {
    let polar_night_test = -latitude_rad.tan() * declination_rad.tan();
    let hour_angle = if polar_night_test >= 1.0 {
        0.0
    } else if polar_night_test <= -1.0 {
        std::f32::consts::PI
    } else {
        polar_night_test.acos()
    };

    let value = hour_angle * latitude_rad.sin() * declination_rad.sin()
        + latitude_rad.cos() * declination_rad.cos() * hour_angle.sin();
    value.max(0.0)
}
