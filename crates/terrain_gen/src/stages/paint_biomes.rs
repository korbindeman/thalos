use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{Cubemap, CubemapFace, face_uv_to_dir};
use crate::noise::fbm3;
use crate::stage::Stage;
use crate::types::Material;

/// Paint the baked albedo as a continuous function of biome color plus
/// optional physical overlays (rock exposure, snow coverage, slope AO).
/// Biome ids drive per-texel roughness via the material palette, same as
/// before; colour decouples from the discrete biome assignment.
///
/// Ocean water is NOT painted here — below-sea-level texels receive the
/// raw biome color (treat the ocean biome's albedo/tint as seabed, not
/// surface water). The planet shader draws water + ice on top at render
/// time using `sea_level_m` from `BodyData`, so bake output stays a
/// "dry planet" read.
///
/// ── Why overlays instead of more biomes ────────────────────────────────
/// A discrete "ice" biome paints every cold cell the same flat white and
/// hands PaintBiomes a hard biome-id boundary at the ice edge. The 3×3
/// blend softens it by a few texels, but the dominant shape is still the
/// rule's iso-contour — a literal latitude line becomes a bullseye polar
/// cap, a crisp 4500 m altitude threshold becomes a grey ring of alpine.
///
/// Treating snow and rock as *continuous coverage fractions* over the
/// underlying biome base eliminates the iso-contour entirely: a pixel 2 °C
/// warmer than its neighbour carries slightly less snow, not zero snow.
/// Climate (temp) and terrain (slope, height) vary smoothly at the texel
/// scale, so every coverage derived from them varies smoothly too. The
/// 3×3 CPU blend and GPU bilinear filter are still there on top — they
/// only have to hide sub-texel noise now, not rule boundaries.
///
/// Composition order (each step paints on top of the previous):
///   1. **Base** = biome color (from the 3×3 biome-id blend).
///   2. **Rock overlay**: if `rock_overlay` is configured, on land cells
///      base = mix(base, rock_color, rock_fraction(slope, height)).
///   3. **Snow overlay**: if `snow_overlay` is configured, on land cells
///      base = mix(base, snow_color, snow_fraction(temp)). Sea ice is a
///      shader concern — it sits on water, not on seabed.
///   4. **Slope AO**: if `slope_ao` is configured, base *= slope_darken(slope).
///   5. **Tonal noise**: two-octave fbm, ±(biome.tonal_amp × tonal_mix),
///      so each biome retains its per-biome texture character.
///
/// Bodies that want the minimal behaviour leave every overlay at `None`
/// and get pure biome-color output + single tonal fbm octave.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PaintBiomes {
    #[serde(default)]
    pub rock_overlay: Option<RockOverlay>,
    #[serde(default)]
    pub snow_overlay: Option<SnowOverlay>,
    #[serde(default)]
    pub slope_ao: Option<SlopeAO>,
}

/// Expose bedrock on steep slopes and high altitude by blending the base
/// biome colour toward `color`. Coverage = smoothstep(slope_start,
/// slope_full, slope) with an additive height boost above `height_boost_from_m`.
#[derive(Debug, Clone, Deserialize)]
pub struct RockOverlay {
    /// Slope (dimensionless, ≈ rise/run as a tangent) where rock exposure
    /// starts increasing above 0.
    pub slope_start: f32,
    /// Slope where rock exposure saturates at 1.0.
    pub slope_full: f32,
    /// Altitude (m) above which rock exposure begins to increase even on
    /// flat cells (dry-peak effect). Set to a large value (e.g. 1e9) to
    /// disable the altitude boost.
    pub height_boost_from_m: f32,
    /// Coverage boost per km above `height_boost_from_m`. Added to the
    /// slope coverage before clamping into [0, 1].
    pub height_boost_per_km: f32,
    /// Rock colour (linear sRGB).
    pub color: [f32; 3],
    /// Per-texel tonal amplitude for the rock colour (fbm-based, fraction).
    pub tonal_amp: f32,
    /// Rock biome fbm frequency. Keep in-scale with biome tonal_frequency.
    pub tonal_frequency: f64,
}

/// Blend a snow/ice colour in on top of land. Coverage is driven by
/// temperature rather than latitude so equatorial peaks can carry
/// glaciers and warm polar land can stay bare. Sea ice is NOT painted
/// here — the shader handles it alongside the water surface.
#[derive(Debug, Clone, Deserialize)]
pub struct SnowOverlay {
    /// Temperature at or above which snow coverage is 0 (°C).
    pub temp_none_c: f32,
    /// Temperature at or below which snow coverage is 1.0 (°C).
    pub temp_full_c: f32,
    /// Amplitude of the fbm jitter on the effective temperature (°C).
    /// 2–6 °C keeps the edge lobed without eating large swathes of land.
    pub jitter_amp_c: f32,
    pub jitter_frequency: f64,
    /// Land snow colour.
    pub land_color: [f32; 3],
    /// Tonal amp on the snow colour (fbm-modulated). Small — snow is
    /// almost homogeneous at orbital scale.
    pub tonal_amp: f32,
}

/// Multiplicative darkening on steep slopes to approximate ambient
/// occlusion / macro-scale self-shadowing. Reads as mountain relief from
/// orbit even when the shader's sun direction hides the detailed normals.
#[derive(Debug, Clone, Deserialize)]
pub struct SlopeAO {
    /// Slope at which darkening starts.
    pub slope_start: f32,
    /// Slope at which darkening saturates.
    pub slope_full: f32,
    /// Maximum multiplicative darkening (0 = no effect; 0.4 = up to 60 %
    /// of brightness retained on the steepest slopes).
    pub max_darken: f32,
}

/// Half-extent of the smoothing kernel in texels. 3×3 at this offset ≈
/// ±2 texel CPU smoothing, ~10 km on Thalos at 2048²; bilinear filtering
/// on the GPU adds another ~1 texel, so biome transitions span ~25 km.
const KERNEL_OFFSET_TEXELS: f32 = 2.0;
const TONAL_FREQ_LO: f64 = 3.5;
const TONAL_FREQ_HI: f64 = 14.0;
const TONAL_HI_MIX: f32 = 0.35;

impl Stage for PaintBiomes {
    fn name(&self) -> &str {
        "paint_biomes"
    }
    fn dependencies(&self) -> &[&str] {
        &["biomes"]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        let res = builder.cubemap_resolution;
        let seed = builder.stage_seed();
        let biomes = builder.biomes.clone();
        assert!(
            !biomes.is_empty(),
            "PaintBiomes: builder.biomes is empty — ensure `Biomes` stage ran first",
        );

        // ── 1. Register a Material per biome with NEUTRAL-WHITE albedo. ─
        // The chromatic signal lives in the baked-albedo cube; material
        // entries here only carry roughness so the shader still has a
        // per-biome surface-property hook.
        let base_idx = builder.materials.len();
        for b in &biomes {
            builder.materials.push(Material {
                albedo: [1.0, 1.0, 1.0],
                roughness: b.roughness,
            });
        }

        // ── 2. Rewrite material_cubemap so each texel points at its biome's
        // material slot. u8 caps at 255 total materials — saturate just in case.
        {
            let biome_map = &builder.biome_map;
            let mat_cm = &mut builder.material_cubemap;
            for face in CubemapFace::ALL {
                let src = biome_map.face_data(face);
                let dst = mat_cm.face_data_mut(face);
                for i in 0..src.len() {
                    let id = (base_idx as u32).saturating_add(src[i] as u32);
                    dst[i] = id.min(u8::MAX as u32) as u8;
                }
            }
        }

        // ── 3. Precompute slope field. Reading it during the parallel
        // paint means paying neighbour-lookup overhead per texel anyway,
        // and this keeps the paint loop readable. Slope is stored as
        // magnitude of height gradient (m rise per m of arc length).
        let slope_field = compute_slope_cubemap(&builder.height_contributions.height, builder.radius_m);

        let inv_res = 1.0 / res as f32;
        let texel_offset = KERNEL_OFFSET_TEXELS * inv_res;
        let n_biomes = biomes.len();

        // Per-biome chromatic base colour (luminance × tint).
        let biome_colors: Vec<[f32; 3]> = biomes
            .iter()
            .map(|b| {
                [
                    b.albedo * b.tint[0],
                    b.albedo * b.tint[1],
                    b.albedo * b.tint[2],
                ]
            })
            .collect();

        let biome_map = &builder.biome_map;
        let height_field = &builder.height_contributions.height;
        let temperature_c = &builder.temperature_c;
        let rock_overlay = self.rock_overlay.clone();
        let snow_overlay = self.snow_overlay.clone();
        let slope_ao = self.slope_ao.clone();

        builder
            .albedo_contributions
            .albedo
            .faces_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(face_idx, slice)| {
                let face = CubemapFace::ALL[face_idx];
                let bm_face = biome_map.face_data(face);
                let h_face = height_field.face_data(face);
                let t_face = temperature_c.face_data(face);
                let s_face = slope_field.face_data(face);

                for y in 0..res {
                    for x in 0..res {
                        let idx = (y * res + x) as usize;
                        let center_u = (x as f32 + 0.5) * inv_res;
                        let center_v = (y as f32 + 0.5) * inv_res;
                        let dir = face_uv_to_dir(face, center_u, center_v).normalize();
                        let height_m = h_face[idx];
                        let temp_c = t_face[idx];
                        let slope = s_face[idx];

                        // ── A. Biome base colour (3×3 blended) ──
                        let mut color_sum = [0.0f32; 3];
                        let mut weight_sum = 0.0f32;
                        for dy in -1..=1i32 {
                            for dx in -1..=1i32 {
                                let su = (center_u + dx as f32 * texel_offset).clamp(0.0, 1.0);
                                let sv = (center_v + dy as f32 * texel_offset).clamp(0.0, 1.0);
                                let sx = (su * res as f32) as i32;
                                let sy = (sv * res as f32) as i32;
                                let sxc = sx.clamp(0, res as i32 - 1) as u32;
                                let syc = sy.clamp(0, res as i32 - 1) as u32;
                                let sample_idx = (syc * res + sxc) as usize;
                                let bid = bm_face[sample_idx] as usize;
                                if bid >= n_biomes {
                                    continue;
                                }
                                let c = biome_colors[bid];
                                let r = (dx * dx + dy * dy) as f32;
                                let weight = 1.0 / (1.0 + r);
                                color_sum[0] += c[0] * weight;
                                color_sum[1] += c[1] * weight;
                                color_sum[2] += c[2] * weight;
                                weight_sum += weight;
                            }
                        }
                        let mut color = if weight_sum > 0.0 {
                            [
                                color_sum[0] / weight_sum,
                                color_sum[1] / weight_sum,
                                color_sum[2] / weight_sum,
                            ]
                        } else {
                            let bid = (bm_face[idx] as usize).min(n_biomes - 1);
                            biome_colors[bid]
                        };

                        let center_bid = bm_face[idx] as usize;
                        let center_biome = &biomes[center_bid.min(n_biomes - 1)];

                        // Below-sea-level cells keep their biome base
                        // color (interpreted as seabed). Water + sea
                        // ice are drawn by the shader using
                        // `sea_level_m`, so no ocean tint is baked.
                        let is_ocean = height_m <= 0.0;

                        // ── B. Rock exposure overlay (land only) ──
                        if !is_ocean && let Some(ro) = &rock_overlay {
                            let slope_cov = smoothstep(ro.slope_start, ro.slope_full, slope);
                            let height_cov = if height_m >= ro.height_boost_from_m {
                                let km_above = (height_m - ro.height_boost_from_m) / 1000.0;
                                (km_above * ro.height_boost_per_km).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };
                            let rock_cov = (slope_cov + height_cov).clamp(0.0, 1.0);
                            // Per-texel rock tonal modulation so bare rock
                            // isn't a single flat slab either.
                            let rock_tone_seed = seed ^ 0xD7C4_1A3B_92F0_E465;
                            let rock_n = fbm3(
                                dir.x as f64 * ro.tonal_frequency,
                                dir.y as f64 * ro.tonal_frequency,
                                dir.z as f64 * ro.tonal_frequency,
                                rock_tone_seed,
                                4,
                                0.55,
                                2.1,
                            ) as f32;
                            let rock_scale = (1.0 + ro.tonal_amp * rock_n).max(0.0);
                            let rock_c = [
                                ro.color[0] * rock_scale,
                                ro.color[1] * rock_scale,
                                ro.color[2] * rock_scale,
                            ];
                            color = [
                                lerp(color[0], rock_c[0], rock_cov),
                                lerp(color[1], rock_c[1], rock_cov),
                                lerp(color[2], rock_c[2], rock_cov),
                            ];
                        }

                        // ── C. Snow coverage on land (temperature driven) ──
                        //       Sea ice is a shader-layer concern; below
                        //       sea level we leave the biome color alone.
                        if !is_ocean && let Some(so) = &snow_overlay {
                            let snow_seed = seed ^ 0xB3F0_71A4_9C28_DE15;
                            let t_jitter = fbm3(
                                dir.x as f64 * so.jitter_frequency,
                                dir.y as f64 * so.jitter_frequency,
                                dir.z as f64 * so.jitter_frequency,
                                snow_seed,
                                4,
                                0.55,
                                2.1,
                            ) as f32;
                            let t_eff = temp_c + t_jitter * so.jitter_amp_c;
                            // Fully snow at temp <= temp_full, none at temp >= temp_none.
                            let snow_cov = smoothstep(so.temp_none_c, so.temp_full_c, t_eff);
                            if snow_cov > 0.0 {
                                let snow_tone_seed = seed ^ 0x5F19_A6D2_30C7_4E83;
                                let snow_n = fbm3(
                                    dir.x as f64 * (so.jitter_frequency * 2.0),
                                    dir.y as f64 * (so.jitter_frequency * 2.0),
                                    dir.z as f64 * (so.jitter_frequency * 2.0),
                                    snow_tone_seed,
                                    3,
                                    0.55,
                                    2.1,
                                ) as f32;
                                let snow_scale = (1.0 + so.tonal_amp * snow_n).max(0.0);
                                let snow_c = [
                                    so.land_color[0] * snow_scale,
                                    so.land_color[1] * snow_scale,
                                    so.land_color[2] * snow_scale,
                                ];
                                color = [
                                    lerp(color[0], snow_c[0], snow_cov),
                                    lerp(color[1], snow_c[1], snow_cov),
                                    lerp(color[2], snow_c[2], snow_cov),
                                ];
                            }
                        }

                        // ── D. Slope AO darkening ──
                        if let Some(ao) = &slope_ao {
                            let ao_cov = smoothstep(ao.slope_start, ao.slope_full, slope);
                            let darken = 1.0 - ao.max_darken * ao_cov;
                            color = [color[0] * darken, color[1] * darken, color[2] * darken];
                        }

                        // ── E. Two-octave biome tonal noise ──
                        // A second higher-freq octave adds 1-3 texel speckle
                        // so within-biome patches don't read as a single flat
                        // slab even without overlays. The center biome's id
                        // keys the seed so adjacent biomes don't share a
                        // pattern.
                        let bs = seed
                            ^ 0xA17B_5C2D_4E9F_1357
                            ^ ((center_bid as u64).wrapping_mul(0xB3F9_4C78_CE32_1A5D));
                        let n_lo = fbm3(
                            dir.x as f64 * TONAL_FREQ_LO,
                            dir.y as f64 * TONAL_FREQ_LO,
                            dir.z as f64 * TONAL_FREQ_LO,
                            bs,
                            4,
                            0.55,
                            2.1,
                        ) as f32;
                        let n_hi = fbm3(
                            dir.x as f64 * TONAL_FREQ_HI,
                            dir.y as f64 * TONAL_FREQ_HI,
                            dir.z as f64 * TONAL_FREQ_HI,
                            bs.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                            3,
                            0.5,
                            2.3,
                        ) as f32;
                        let n = n_lo * (1.0 - TONAL_HI_MIX) + n_hi * TONAL_HI_MIX;
                        let tonal = (1.0 + center_biome.tonal_amp * n).max(0.0);

                        // Shader expects baked_tint = final / 2 so that
                        //   mat_albedo(1) × baked_tint × 2 × regional = final × regional.
                        let v0 = (color[0] * tonal * 0.5).clamp(0.0, 1.0);
                        let v1 = (color[1] * tonal * 0.5).clamp(0.0, 1.0);
                        let v2 = (color[2] * tonal * 0.5).clamp(0.0, 1.0);
                        slice[idx] = [v0, v1, v2, 1.0];
                    }
                }
            });
    }
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    // Handles both directions: edge0 < edge1 (standard) and edge0 > edge1
    // (inverted — useful for "below this value" coverage).
    let denom = edge1 - edge0;
    if denom.abs() < 1e-6 {
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / denom).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Compute per-texel slope as the magnitude of the height gradient
/// expressed in units of m rise per m of surface arc length. Uses local
/// face-space finite differences with edge clamping; the ±1-texel
/// discontinuity at face seams is acceptable because slope drives only
/// soft multiplicative overlays (rock, AO), never hard iso-contours.
fn compute_slope_cubemap(
    height: &Cubemap<f32>,
    body_radius_m: f32,
) -> Cubemap<f32> {
    let res = height.resolution();
    let mut out = Cubemap::<f32>::new(res);
    // Average face pixel size at the cube-face centre. Corners are ~40 %
    // shorter, but slope is a soft multiplier here; the error is a ~40 %
    // per-corner underestimate, well below the smoothstep band width.
    let pixel_m = body_radius_m * 2.0 / res as f32;
    let inv_2dx = 1.0 / (2.0 * pixel_m);
    for face in CubemapFace::ALL {
        let src = height.face_data(face);
        let dst = out.face_data_mut(face);
        for y in 0..res {
            for x in 0..res {
                let xp = (x + 1).min(res - 1);
                let xm = x.saturating_sub(1);
                let yp = (y + 1).min(res - 1);
                let ym = y.saturating_sub(1);
                let h_xp = src[(y * res + xp) as usize];
                let h_xm = src[(y * res + xm) as usize];
                let h_yp = src[(yp * res + x) as usize];
                let h_ym = src[(ym * res + x) as usize];
                let dh_x = (h_xp - h_xm) * inv_2dx;
                let dh_y = (h_yp - h_ym) * inv_2dx;
                dst[(y * res + x) as usize] = (dh_x * dh_x + dh_y * dh_y).sqrt();
            }
        }
    }
    out
}
