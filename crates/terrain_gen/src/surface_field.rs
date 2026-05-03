//! Continuous, render-agnostic spherical terrain fields.
//!
//! A `SurfaceField` is sampled by direction on the unit sphere. Projections
//! such as the current cubemap impostor bake are consumers of this contract,
//! not the source of truth.

use glam::Vec3;
use rayon::prelude::*;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{CubemapFace, face_uv_to_dir};

pub const MAX_SURFACE_MATERIAL_WEIGHTS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceMaterialWeight {
    pub material_id: u8,
    pub weight: f32,
}

impl SurfaceMaterialWeight {
    pub const NONE: Self = Self {
        material_id: 0,
        weight: 0.0,
    };

    pub const fn new(material_id: u8, weight: f32) -> Self {
        Self {
            material_id,
            weight,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceMaterialMix {
    pub weights: [SurfaceMaterialWeight; MAX_SURFACE_MATERIAL_WEIGHTS],
}

impl SurfaceMaterialMix {
    pub fn single(material_id: u8) -> Self {
        Self {
            weights: [
                SurfaceMaterialWeight::new(material_id, 1.0),
                SurfaceMaterialWeight::NONE,
                SurfaceMaterialWeight::NONE,
                SurfaceMaterialWeight::NONE,
            ],
        }
    }

    pub fn from_weighted<const N: usize>(candidates: [(u8, f32); N]) -> Self {
        let mut merged = [SurfaceMaterialWeight::NONE; N];
        let mut merged_count = 0usize;

        for (material_id, weight) in candidates {
            let weight = weight.max(0.0);
            if weight <= 0.0 {
                continue;
            }

            if let Some(existing) = merged[..merged_count]
                .iter_mut()
                .find(|entry| entry.material_id == material_id)
            {
                existing.weight += weight;
            } else {
                merged[merged_count] = SurfaceMaterialWeight::new(material_id, weight);
                merged_count += 1;
            }
        }

        if merged_count == 0 {
            return Self::single(0);
        }

        for i in 0..merged_count {
            for j in (i + 1)..merged_count {
                if merged[j].weight > merged[i].weight {
                    merged.swap(i, j);
                }
            }
        }

        let total: f32 = merged[..merged_count]
            .iter()
            .map(|entry| entry.weight)
            .sum();
        let mut weights = [SurfaceMaterialWeight::NONE; MAX_SURFACE_MATERIAL_WEIGHTS];
        for (dst, src) in weights.iter_mut().zip(merged[..merged_count].iter()) {
            *dst = SurfaceMaterialWeight::new(src.material_id, src.weight / total);
        }
        Self { weights }
    }

    pub fn dominant_material_id(self) -> u8 {
        self.weights[0].material_id
    }

    pub fn weight_for(self, material_id: u8) -> f32 {
        self.weights
            .iter()
            .filter(|entry| entry.material_id == material_id)
            .map(|entry| entry.weight)
            .sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceFieldSample {
    pub height_m: f32,
    pub albedo_linear: [f32; 3],
    pub material_mix: SurfaceMaterialMix,
    pub roughness: f32,
    /// Body-local normal contribution at this point. Use `dir` to mean
    /// "no analytical perturbation, fall back to the height-derived normal."
    /// Anisotropic processes (dune ripple direction, foliated rock) should
    /// rotate this away from `dir` to encode their orientation.
    pub normal_local: Vec3,
}

impl SurfaceFieldSample {
    pub fn new(
        height_m: f32,
        albedo_linear: [f32; 3],
        material_mix: SurfaceMaterialMix,
        roughness: f32,
        normal_local: Vec3,
    ) -> Self {
        Self {
            height_m,
            albedo_linear,
            material_mix,
            roughness,
            normal_local,
        }
    }
}

pub trait SurfaceField: Sync {
    fn sample(&self, dir: Vec3, sample_scale_m: f32) -> SurfaceFieldSample;
}

pub fn bake_surface_field_into_builder<F: SurfaceField>(builder: &mut BodyBuilder, field: &F) {
    let res = builder.cubemap_resolution as usize;
    let sample_scale_m = cube_face_texel_scale_m(builder.radius_m, builder.cubemap_resolution);

    for face in CubemapFace::ALL {
        let heights = builder.height_contributions.height.face_data_mut(face);
        let albedo = builder.albedo_contributions.albedo.face_data_mut(face);
        let materials = builder.material_cubemap.face_data_mut(face);
        let roughness = builder.roughness_cubemap.face_data_mut(face);
        let normals = builder.normal_cubemap.face_data_mut(face);

        heights
            .par_iter_mut()
            .zip(albedo.par_iter_mut())
            .zip(materials.par_iter_mut())
            .zip(roughness.par_iter_mut())
            .zip(normals.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((((height, color), material), rough), nrm))| {
                let x = i % res;
                let y = i / res;
                let u = (x as f32 + 0.5) / res as f32;
                let v = (y as f32 + 0.5) / res as f32;
                let dir = face_uv_to_dir(face, u, v);
                let sample = field.sample(dir, sample_scale_m);

                *height = sample.height_m;
                *color = [
                    sample.albedo_linear[0],
                    sample.albedo_linear[1],
                    sample.albedo_linear[2],
                    1.0,
                ];
                *material = sample.material_mix.dominant_material_id();
                *rough = quantize_unit_to_u8(sample.roughness);

                // Normal cube: encode the field's analytical contribution
                // plus the geometric outward direction. Height-derived bumps
                // are NOT folded in here — that requires 4 extra `field.sample()`
                // calls per texel for finite differencing, and the impostor
                // shader doesn't consume this cube anyway (it reconstructs
                // normals per-fragment from the filterable height cube). When
                // ground LOD comes online and needs pre-baked normals, add a
                // separate two-pass bake that finite-differences the finalized
                // height cubemap.
                let perturb = sample.normal_local - sample.normal_local.dot(dir) * dir;
                let final_normal = (dir + perturb).try_normalize().unwrap_or(dir);
                *nrm = encode_object_space_normal(final_normal);
            });
    }
}

/// Encode a unit body-local (object-space) normal as RGBA8: `(n * 0.5 + 0.5)
/// * 255` per channel; alpha = 255. Decoded in the shader as `tex.rgb * 2 - 1`.
/// The texture must be sampled as linear (`Rgba8Unorm`), not sRGB.
pub fn encode_object_space_normal(n: Vec3) -> [u8; 4] {
    let scaled = n * 0.5 + Vec3::splat(0.5);
    [
        quantize_unit_to_u8(scaled.x),
        quantize_unit_to_u8(scaled.y),
        quantize_unit_to_u8(scaled.z),
        255,
    ]
}

/// Quantize a 0..1 scalar to u8 with rounding and clamping.
pub fn quantize_unit_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Per-texel default normal cube: each texel encodes the body-local outward
/// direction at that texel (i.e. the geometric sphere normal). Bodies whose
/// pipelines don't run a `SurfaceField` bake fall back to this — the impostor
/// shader sees the same normal it would derive from a flat sphere, which is
/// the correct default in the absence of any height or anisotropy information.
pub fn default_normal_cubemap(resolution: u32) -> crate::cubemap::Cubemap<[u8; 4]> {
    let res = resolution as usize;
    let mut cube = crate::cubemap::Cubemap::<[u8; 4]>::new(resolution);
    for face in CubemapFace::ALL {
        let data = cube.face_data_mut(face);
        for (i, val) in data.iter_mut().enumerate() {
            let x = i % res;
            let y = i / res;
            let u = (x as f32 + 0.5) / res as f32;
            let v = (y as f32 + 0.5) / res as f32;
            let dir = face_uv_to_dir(face, u, v);
            *val = encode_object_space_normal(dir);
        }
    }
    cube
}

pub fn cube_face_texel_scale_m(radius_m: f32, cubemap_resolution: u32) -> f32 {
    radius_m * std::f32::consts::FRAC_PI_2 / cubemap_resolution.max(1) as f32
}

pub fn scale_visibility(sample_scale_m: f32, feature_wavelength_m: f32) -> f32 {
    smoothstep(
        sample_scale_m * 1.5,
        sample_scale_m * 4.0,
        feature_wavelength_m.max(0.0),
    )
}

pub fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < 1e-6 {
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_mix_keeps_dominant_weight_first() {
        let mix = SurfaceMaterialMix::from_weighted([(3, 0.45), (1, 0.4), (3, 0.2)]);

        assert_eq!(mix.dominant_material_id(), 3);
        assert!(mix.weight_for(3) > mix.weight_for(1));
    }

    #[test]
    fn texel_scale_is_planetary_arc_length() {
        let scale = cube_face_texel_scale_m(1_130_000.0, 2048);

        assert!(scale > 850.0);
        assert!(scale < 880.0);
    }
}
