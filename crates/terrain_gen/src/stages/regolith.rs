use fastnoise2::generator::cellular::CellularDistanceReturnType;
use fastnoise2::generator::prelude::*;
use fastnoise2::generator::DistanceFunction;
use fastnoise2::SafeNode;
use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::CubemapFace;
use crate::stage::Stage;
use crate::types::DetailNoiseParams;
use super::util::face_position_arrays;

/// Configures the regolith: both the high-frequency detail noise for the
/// realtime shader and the baked background terrain on the cubemap.
///
/// The bake-time contribution is critical — without it, the cubemap reads as
/// a plastic sphere between explicit craters. Instead of generic biased fBm
/// we now bake a three-biome blend (rugged highlands, subdued plains, ridged
/// dorsa) sampled through `fastnoise2`. Biome weights come from three
/// decorrelated supersimplex fields combined via softmax, yielding soft 2–5
/// km transitions characteristic of intra-highland variation on real airless
/// bodies.
#[derive(Debug, Clone, Deserialize)]
pub struct Regolith {
    /// Typical depth of small-crater texture in meters (detail shader).
    pub amplitude_m: f32,
    /// Approximate spacing of smallest visible crater-like features.
    pub characteristic_wavelength_m: f32,
    /// Density multiplier relative to continued SFD from explicit craters.
    pub crater_density_multiplier: f32,

    /// Reference amplitude in meters for the rugged-highlands biome. Other
    /// biomes are scaled relative to this. Set to 0 to disable baking.
    #[serde(default = "default_bake_amplitude")]
    pub bake_amplitude_m: f32,
    /// Dominant wavelength of the finest biome noise octave in meters.
    /// Sets the sampling frequency: `base_freq = radius_m / wavelength_m`.
    #[serde(default = "default_bake_wavelength")]
    pub bake_wavelength_m: f32,
    /// Octaves for each biome's fractal stack.
    #[serde(default = "default_bake_octaves")]
    pub bake_octaves: u32,

    /// How many biome cells fit across the sphere. Higher → more frequent
    /// biome transitions. Defaults to ~6.
    #[serde(default = "default_biome_cell_count")]
    pub biome_cell_count: f32,
    /// Softmax temperature for biome blending. Smaller = harder boundaries
    /// (0.1 ≈ near-sharp), larger = more diffuse (1.0 ≈ even mixture).
    #[serde(default = "default_biome_softness")]
    pub biome_softness: f32,
}

fn default_bake_amplitude() -> f32 { 80.0 }
fn default_bake_wavelength() -> f32 { 15_000.0 }
fn default_bake_octaves() -> u32 { 6 }
fn default_biome_cell_count() -> f32 { 6.0 }
fn default_biome_softness() -> f32 { 0.35 }

impl Stage for Regolith {
    fn name(&self) -> &str { "regolith" }
    fn dependencies(&self) -> &[&str] { &["cratering"] }

    fn apply(&self, builder: &mut BodyBuilder) {
        // ── Detail-noise params for the realtime shader ────────────────
        builder.detail_params = DetailNoiseParams {
            body_radius_m: builder.radius_m,
            d_min_m: 80.0,
            d_max_m: 60_000.0,
            sfd_alpha: 2.0,
            global_k_per_km2: self.crater_density_multiplier * 0.05,
            d_sc_m: 30_000.0,
            body_age_gyr: builder.body_age_gyr,
            seed: builder.stage_seed(),
        };

        if self.bake_amplitude_m <= 0.0 { return; }

        let res = builder.cubemap_resolution;
        let body_radius = builder.radius_m;
        let stage_seed = builder.stage_seed() as i32;

        // ── Build noise trees ──────────────────────────────────────────
        //
        // Input coords are raw unit-sphere directions (`dir.x, y, z` in
        // [-1, 1]). Each tree bakes its own `domain_scale` to set the
        // frequency. `base_freq` is chosen so one period at the finest
        // octave matches `bake_wavelength_m` of surface distance.
        let base_freq = body_radius / self.bake_wavelength_m;
        let weight_freq = base_freq * (self.biome_cell_count / 10.0).max(0.05);

        // Biome weight source — a smooth supersimplex fBm. We sample it
        // three times with decorrelated seeds to get three weight fields,
        // then softmax into per-texel weights that sum to 1.
        let weight_node: SafeNode = supersimplex()
            .fbm(0.5, 0.0, 2, 2.0)
            .domain_scale(weight_freq)
            .build()
            .0;

        // Biome A: Rugged Highlands — ridged supersimplex gives overlapping
        // rim-ridge impression; cellular distance carves crater-like pits.
        // Together they produce the hummocky rubble that dominates real
        // lunar highlands at impostor scale.
        let octs = self.bake_octaves.max(1) as i32;
        let biome_a: SafeNode = (supersimplex()
            .ridged(0.5, 0.0, octs, 2.0)
            .domain_scale(base_freq)
            + cellular_distance(
                1.0,
                DistanceFunction::Euclidean,
                0,
                1,
                CellularDistanceReturnType::Index0Sub1,
            )
            .domain_scale(base_freq * 0.7)
                * 0.6)
            .remap(-2.0, 2.0, -1.0, 1.0)
            .build()
            .0;

        // Biome B: Subdued Plains — low-amplitude supersimplex fBm, rolling
        // character. Emerges from post-impact ejecta infill that buried
        // earlier small craters.
        let biome_b: SafeNode = supersimplex()
            .fbm(0.5, 0.0, octs, 2.0)
            .domain_scale(base_freq * 0.6)
            .build()
            .0;

        // Biome C: Dorsa / Scarps — warped ridged noise produces linear
        // ridges and subtle scarp features. Higher amplitude than baseline.
        let biome_c: SafeNode = supersimplex()
            .ridged(0.55, 0.0, octs, 2.0)
            .domain_warp_gradient(0.25, base_freq * 0.4)
            .domain_scale(base_freq * 0.8)
            .build()
            .0;

        // Biome amplitudes in meters.
        let amp_a = self.bake_amplitude_m;
        let amp_b = self.bake_amplitude_m * 0.25;
        let amp_c = self.bake_amplitude_m * 1.3;

        // Three weight seeds chosen to be widely spaced in the seed space.
        let seed_w_a = stage_seed;
        let seed_w_b = stage_seed.wrapping_add(0x5EED_B10E_u32 as i32);
        let seed_w_c = stage_seed.wrapping_add(0x3F00_BA11_u32 as i32);

        let softness = self.biome_softness.max(0.05);

        // ── Per-face parallel bulk sample + blend ──────────────────────
        builder
            .height_contributions
            .height
            .faces_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(face_idx, slice)| {
                let face = CubemapFace::ALL[face_idx];
                let n = (res * res) as usize;
                let (xs, ys, zs) = face_position_arrays(face, res);

                // Biome weight fields — one tree sampled with three seeds.
                let mut wa = vec![0.0f32; n];
                let mut wb = vec![0.0f32; n];
                let mut wc = vec![0.0f32; n];
                weight_node.gen_position_array_3d(
                    &mut wa, &xs, &ys, &zs, 0.0, 0.0, 0.0, seed_w_a,
                );
                weight_node.gen_position_array_3d(
                    &mut wb, &xs, &ys, &zs, 0.0, 0.0, 0.0, seed_w_b,
                );
                weight_node.gen_position_array_3d(
                    &mut wc, &xs, &ys, &zs, 0.0, 0.0, 0.0, seed_w_c,
                );

                // Biome terrain outputs.
                let mut ha = vec![0.0f32; n];
                let mut hb = vec![0.0f32; n];
                let mut hc = vec![0.0f32; n];
                biome_a.gen_position_array_3d(
                    &mut ha, &xs, &ys, &zs, 0.0, 0.0, 0.0, stage_seed,
                );
                biome_b.gen_position_array_3d(
                    &mut hb, &xs, &ys, &zs, 0.0, 0.0, 0.0, stage_seed,
                );
                biome_c.gen_position_array_3d(
                    &mut hc, &xs, &ys, &zs, 0.0, 0.0, 0.0, stage_seed,
                );

                // Softmax blend and accumulate into the face slice.
                for i in 0..n {
                    let max = wa[i].max(wb[i]).max(wc[i]);
                    let ea = ((wa[i] - max) / softness).exp();
                    let eb = ((wb[i] - max) / softness).exp();
                    let ec = ((wc[i] - max) / softness).exp();
                    let s = ea + eb + ec;
                    let blended =
                        (ha[i] * amp_a * ea + hb[i] * amp_b * eb + hc[i] * amp_c * ec)
                            / s;
                    slice[i] += blended;
                }
            });
    }
}
