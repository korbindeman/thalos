//! Continuous, render-agnostic spherical terrain fields.
//!
//! A `SurfaceField` is sampled by direction on the unit sphere. Projections
//! such as the current cubemap impostor bake are consumers of this contract,
//! not the source of truth.

use glam::Vec3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::body_builder::BodyBuilder;
use crate::cubemap::{CubemapFace, face_uv_to_dir};

pub const MAX_SURFACE_MATERIAL_WEIGHTS: usize = 4;
pub const MAX_BIOME_WEIGHTS: usize = 4;

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

/// One biome's weighted contribution at a sample point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiomeWeight {
    pub biome_id: u8,
    pub weight: f32,
}

impl BiomeWeight {
    pub const NONE: Self = Self {
        biome_id: 0,
        weight: 0.0,
    };

    pub const fn new(biome_id: u8, weight: f32) -> Self {
        Self { biome_id, weight }
    }
}

/// Top-K weighted mix of biomes at a sample point. Mirrors `SurfaceMaterialMix`
/// but for biome ids; both travel together in `SurfaceFieldSample` so the bake
/// can persist the full mix and a downstream relief stage can do proper
/// per-biome palette blending without re-evaluating the source field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiomeMix {
    pub weights: [BiomeWeight; MAX_BIOME_WEIGHTS],
}

impl BiomeMix {
    pub fn single(biome_id: u8) -> Self {
        Self {
            weights: [
                BiomeWeight::new(biome_id, 1.0),
                BiomeWeight::NONE,
                BiomeWeight::NONE,
                BiomeWeight::NONE,
            ],
        }
    }

    /// Build a normalized top-K mix from `(biome_id, weight)` candidates.
    /// Duplicate ids accumulate; the result is sorted by descending weight
    /// and truncated to `MAX_BIOME_WEIGHTS`. Empty input returns
    /// `single(0)`.
    pub fn from_weighted<const N: usize>(candidates: [(u8, f32); N]) -> Self {
        let mut merged = [BiomeWeight::NONE; N];
        let mut merged_count = 0usize;

        for (biome_id, weight) in candidates {
            let weight = weight.max(0.0);
            if weight <= 0.0 {
                continue;
            }
            if let Some(existing) = merged[..merged_count]
                .iter_mut()
                .find(|entry| entry.biome_id == biome_id)
            {
                existing.weight += weight;
            } else {
                merged[merged_count] = BiomeWeight::new(biome_id, weight);
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
        let mut weights = [BiomeWeight::NONE; MAX_BIOME_WEIGHTS];
        for (dst, src) in weights.iter_mut().zip(merged[..merged_count].iter()) {
            *dst = BiomeWeight::new(src.biome_id, src.weight / total);
        }
        Self { weights }
    }

    pub fn dominant_biome_id(self) -> u8 {
        self.weights[0].biome_id
    }

    pub fn weight_for(self, biome_id: u8) -> f32 {
        self.weights
            .iter()
            .filter(|entry| entry.biome_id == biome_id)
            .map(|entry| entry.weight)
            .sum()
    }
}

/// Storage form of a `BiomeMix` for the build-time `biome_weights_cubemap`.
/// Quantizes weights to u8 (255 = 1.0) so each texel is 8 bytes regardless of
/// `MAX_BIOME_WEIGHTS`.
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct BiomeMixTexel {
    pub biome_ids: [u8; MAX_BIOME_WEIGHTS],
    pub weights_q: [u8; MAX_BIOME_WEIGHTS],
}

impl BiomeMixTexel {
    pub fn single(biome_id: u8) -> Self {
        Self::from_mix(BiomeMix::single(biome_id))
    }

    pub fn from_mix(mix: BiomeMix) -> Self {
        let mut biome_ids = [0u8; MAX_BIOME_WEIGHTS];
        let mut weights_q = [0u8; MAX_BIOME_WEIGHTS];
        for (i, w) in mix.weights.iter().enumerate() {
            biome_ids[i] = w.biome_id;
            weights_q[i] = quantize_unit_to_u8(w.weight);
        }
        Self {
            biome_ids,
            weights_q,
        }
    }

    /// Iterate `(biome_id, normalized_weight)` for this texel, skipping zero
    /// entries. Weights are renormalized so they sum to 1 even after u8
    /// quantization rounding.
    pub fn iter_weights(self) -> impl Iterator<Item = (u8, f32)> {
        let total: f32 = self.weights_q.iter().map(|&q| q as f32).sum();
        let total = if total > 0.0 { total } else { 1.0 };
        (0..MAX_BIOME_WEIGHTS).filter_map(move |i| {
            let q = self.weights_q[i];
            if q == 0 {
                None
            } else {
                Some((self.biome_ids[i], q as f32 / total))
            }
        })
    }

    pub fn is_empty(self) -> bool {
        self.weights_q.iter().all(|&q| q == 0)
    }
}

/// Per-biome relief palette: six anchor colors selected by height/slope
/// signals at sample time. The palette IS the biome's visual identity at the
/// pipeline level — there's no separate marbling layer.
///
/// - `low` / `mid` / `high`: low/mid/high altitude in-biome
/// - `steep`: dominant on slopes (cliff/scarp shadow tone)
/// - `ridge`: high-relief crests
/// - `hollow`: depressions / floors
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReliefPalette {
    pub low: [f32; 3],
    pub mid: [f32; 3],
    pub high: [f32; 3],
    pub steep: [f32; 3],
    pub ridge: [f32; 3],
    pub hollow: [f32; 3],
}

impl ReliefPalette {
    /// Evaluate the palette for a given height + relief signal set.
    ///
    /// `height_m` is absolute, in meters. Anchor heights are tuned for
    /// terrestrial bodies with O(km) relief; for moon-scale bodies callers
    /// can pre-scale the height before passing it in.
    pub fn evaluate(
        self,
        height_m: f32,
        slope_signal: f32,
        ridge_signal: f32,
        hollow_signal: f32,
        variation: f32,
    ) -> [f32; 3] {
        let low = smoothstep(120.0, -1_650.0, height_m);
        let high = smoothstep(520.0, 4_900.0, height_m);
        let summit = smoothstep(2_900.0, 7_400.0, height_m);
        let steep = smoothstep(0.16, 0.72, slope_signal);
        let ridge = ridge_signal.clamp(0.0, 1.0);
        let hollow = hollow_signal.clamp(0.0, 1.0);

        let mut color = self.mid;
        color = mix3(color, self.low, low * 0.82);
        color = mix3(
            color,
            self.high,
            (high * 0.70 + summit * 0.16).clamp(0.0, 0.88),
        );
        color = mix3(color, self.hollow, hollow * 0.34);
        color = mix3(color, self.steep, steep * 0.46);
        color = mix3(
            color,
            self.ridge,
            (ridge * 0.34 + high * steep * 0.20).clamp(0.0, 0.56),
        );

        let warm = variation.clamp(-1.0, 1.0);
        let value = 1.0 + warm * 0.045;
        [
            (color[0] * value * (1.0 + warm * 0.060)).clamp(0.018, 0.96),
            (color[1] * value * (1.0 + warm * 0.018)).clamp(0.016, 0.96),
            (color[2] * value * (1.0 - warm * 0.046)).clamp(0.014, 0.96),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceFieldSample {
    pub height_m: f32,
    pub material_mix: SurfaceMaterialMix,
    /// Top-K weighted biome mix at this point. Persisted by the bake into
    /// `BodyBuilder::biome_weights_cubemap` so the unified surface-color
    /// painter can blend per-biome palettes after height/material generation.
    pub biome_mix: BiomeMix,
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
        material_mix: SurfaceMaterialMix,
        biome_mix: BiomeMix,
        roughness: f32,
        normal_local: Vec3,
    ) -> Self {
        Self {
            height_m,
            material_mix,
            biome_mix,
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
        let materials = builder.material_cubemap.face_data_mut(face);
        let roughness = builder.roughness_cubemap.face_data_mut(face);
        let normals = builder.normal_cubemap.face_data_mut(face);
        let biomes = builder.biome_weights_cubemap.face_data_mut(face);

        heights
            .par_iter_mut()
            .zip(materials.par_iter_mut())
            .zip(roughness.par_iter_mut())
            .zip(normals.par_iter_mut())
            .zip(biomes.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((((height, material), rough), nrm), biome))| {
                let x = i % res;
                let y = i / res;
                let u = (x as f32 + 0.5) / res as f32;
                let v = (y as f32 + 0.5) / res as f32;
                let dir = face_uv_to_dir(face, u, v);
                let sample = field.sample(dir, sample_scale_m);

                *height = sample.height_m;
                *material = sample.material_mix.dominant_material_id();
                *rough = quantize_unit_to_u8(sample.roughness);
                *biome = BiomeMixTexel::from_mix(sample.biome_mix);

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
