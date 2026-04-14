use fastnoise2::generator::prelude::*;
use fastnoise2::SafeNode;
use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::CubemapFace;
use crate::sample::sample_detail_noise;
use crate::stage::Stage;
use crate::types::DetailNoiseParams;
use super::util::{face_position_arrays, texel_dir};

/// Configures the regolith: high-frequency detail noise for the realtime
/// shader plus a cubemap bake of mid-frequency Voronoi crater texture that
/// fills the gap between explicit baked craters.
///
/// The bake uses the same multi-octave crater hash as the runtime shader
/// (`sample_detail_noise`), tuned for the size band the cubemap can resolve.
/// SFD continuity across the three layers is enforced by tying ranges to
/// the explicit cratering threshold:
///
/// ```text
///   shader hash   :  d_min_m       … bake_d_min_m       (sub-texel detail)
///   regolith bake :  bake_d_min_m  … 2 × bake_threshold  (cubemap mid band)
///   explicit bake :  ≥ 2 × bake_threshold (diameter)     (Cratering stage)
/// ```
///
/// fBm biome stacks were removed: standard fractional brownian motion reads
/// as eroded terrestrial terrain, never as a crater-saturated airless body.
/// See docs/gen/procedural_mira.md §6 for the rationale.
#[derive(Debug, Clone, Deserialize)]
pub struct Regolith {
    /// Typical depth of small-crater texture in meters (detail shader).
    pub amplitude_m: f32,
    /// Approximate spacing of smallest visible crater-like features.
    pub characteristic_wavelength_m: f32,
    /// Density multiplier relative to continued SFD from explicit craters.
    pub crater_density_multiplier: f32,

    /// Lower diameter bound for the bake-time Voronoi crater stack, in
    /// meters. Sets the handoff with the runtime shader hash layer (whose
    /// `d_max_m` is clamped here so it never overlaps the bake range).
    /// Default 600 m ≈ 1 cubemap texel on Mira at 2048².
    #[serde(default = "default_bake_d_min")]
    pub bake_d_min_m: f32,

    /// Multiplicative scale on the baked Voronoi crater contribution.
    /// 1.0 = unscaled profile depths from `sample_detail_noise`. Set to 0
    /// to disable baking entirely (useful for tests / small bodies).
    #[serde(default = "default_bake_scale")]
    pub bake_scale: f32,

    /// Half-amplitude of the regional density modulation. 0.25 ≈ ±25% on
    /// the Voronoi contribution, driven by a single low-frequency
    /// supersimplex weight field. Prevents the bake from looking globally
    /// uniform while keeping the underlying crater shapes intact.
    #[serde(default = "default_density_mod")]
    pub density_modulation: f32,

    /// Wavelength of the regional density modulation, in meters of surface
    /// distance. 250 km gives Mira ~6 distinct macro-regions across each
    /// hemisphere.
    #[serde(default = "default_density_wavelength")]
    pub density_wavelength_m: f32,
}

fn default_bake_d_min() -> f32 { 600.0 }
fn default_bake_scale() -> f32 { 1.0 }
fn default_density_mod() -> f32 { 0.25 }
fn default_density_wavelength() -> f32 { 250_000.0 }

impl Stage for Regolith {
    fn name(&self) -> &str { "regolith" }
    fn dependencies(&self) -> &[&str] { &["cratering"] }

    fn apply(&self, builder: &mut BodyBuilder) {
        // ── Detail-noise params for the realtime shader ────────────────
        //
        // `d_max_m` is capped at `bake_d_min_m` so the runtime shader hash
        // and the cubemap bake don't double-count the same craters. The
        // shader's own `hash_d_max = min(d_max_m, 500.0)` further restricts
        // it to <500 m on Mira; the cap here keeps the CPU sampler honest.
        let runtime_d_max = self.bake_d_min_m.max(80.0);
        builder.detail_params = DetailNoiseParams {
            body_radius_m: builder.radius_m,
            d_min_m: 80.0,
            d_max_m: runtime_d_max,
            sfd_alpha: 2.0,
            global_k_per_km2: self.crater_density_multiplier * 0.05,
            d_sc_m: 30_000.0,
            body_age_gyr: builder.body_age_gyr,
            seed: builder.stage_seed(),
        };

        // ── Bake-time Voronoi crater stack ─────────────────────────────
        let bake_thresh = builder.cubemap_bake_threshold_m;
        if self.bake_scale <= 0.0 || !bake_thresh.is_finite() || bake_thresh <= 0.0 {
            return;
        }

        let res = builder.cubemap_resolution;
        let body_radius = builder.radius_m;

        // Voronoi diameter band: `bake_d_min_m` to twice the explicit bake
        // threshold (the smallest baked explicit crater's diameter).
        let bake_d_max = bake_thresh * 2.0;
        if bake_d_max <= self.bake_d_min_m {
            return;
        }

        let bake_params = DetailNoiseParams {
            body_radius_m: body_radius,
            d_min_m: self.bake_d_min_m,
            d_max_m: bake_d_max,
            sfd_alpha: 2.0,
            global_k_per_km2: self.crater_density_multiplier * 0.05,
            d_sc_m: 30_000.0,
            body_age_gyr: builder.body_age_gyr,
            seed: builder.stage_seed() ^ 0xC0FF_EE15_BAAD_F00D,
        };

        // LOD chosen so the largest Voronoi crater (`d_max`) hits the upper
        // edge of `sample_detail_noise`'s 0.5..8 px smoothstep — i.e. the
        // biggest baked craters render at full amplitude, smaller ones
        // progressively fade. Treat `pixel_size_m = d_max / 8`.
        let pixel_size_m = bake_d_max / 8.0;
        let lod = pixel_size_m.log2();

        // ── Single supersimplex weight field for regional modulation ───
        //
        // Drives ±`density_modulation` amplitude variation on the Voronoi
        // output so the bake doesn't read as globally uniform. No biome
        // softmax — the crater shapes themselves are the main signal.
        let weight_freq = body_radius / self.density_wavelength_m.max(1.0);
        let weight_node: SafeNode = supersimplex()
            .fbm(0.5, 0.0, 3, 2.0)
            .domain_scale(weight_freq)
            .build()
            .0;
        let weight_seed = (builder.stage_seed() & 0xFFFF_FFFF) as i32;
        let modulation = self.density_modulation.clamp(0.0, 0.95);
        let scale = self.bake_scale;

        // ── Per-face parallel bake ─────────────────────────────────────
        builder
            .height_contributions
            .height
            .faces_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(face_idx, slice)| {
                let face = CubemapFace::ALL[face_idx];
                let n = (res * res) as usize;

                // Bulk-sample the regional weight field for this face.
                let (xs, ys, zs) = face_position_arrays(face, res);
                let mut weights = vec![0.0_f32; n];
                weight_node.gen_position_array_3d(
                    &mut weights,
                    &xs,
                    &ys,
                    &zs,
                    0.0,
                    0.0,
                    0.0,
                    weight_seed,
                );

                slice.par_iter_mut().enumerate().for_each(|(i, h)| {
                    let x = (i as u32) % res;
                    let y = (i as u32) / res;
                    let dir = texel_dir(face, x, y, res).normalize();
                    let (dh, _grad) = sample_detail_noise(&bake_params, dir, lod);
                    let w = 1.0 + modulation * weights[i].clamp(-1.0, 1.0);
                    *h += dh * w * scale;
                });
            });
    }
}
