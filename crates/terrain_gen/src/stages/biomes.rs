use glam::Vec3;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{CubemapFace, face_uv_to_dir};
use crate::noise::fbm3;
use crate::seeding::splitmix64;
use crate::stage::Stage;
use crate::types::BiomeParams;

/// Selection rule for painting a biome onto the surface.
///
/// Multiple rules can coexist; they are evaluated per-texel in the order
/// declared and the first matching rule wins. A final catch-all (default)
/// biome should always be listed so every texel receives an id.
#[derive(Debug, Clone, Deserialize)]
pub enum BiomeRule {
    /// Catch-all. Always matches. Put at the end of the list.
    Default,
    /// Match texels near a megabasin center (by angular distance, in units
    /// of the basin's radius on the sphere). Used for basin-floor biomes.
    NearBasin { max_t: f32 },
    /// Match texels whose low-freq fbm noise exceeds `threshold`. Gives
    /// patchy regional assignment for highland subtypes (KREEP, melt).
    Fbm {
        frequency: f64,
        threshold: f32,
        octaves: u32,
    },
    /// Match texels whose absolute latitude (|dir.y| in the body-local
    /// bake frame, where +Y = north pole) exceeds `min_abs`. Intended for
    /// ice caps, boreal/tundra bands, and other climatic zones that
    /// follow the rotation axis.
    ///
    /// The raw `|dir.y| >= min_abs` test produces a perfect circle on the
    /// sphere. `soft_width` perturbs the effective threshold by a
    /// low-frequency fbm noise of that amplitude, so a realistic ice-cap
    /// boundary reads as lobed and fractal rather than a painted disc.
    /// Set `soft_width = 0.0` for a crisp circular boundary.
    Latitude { min_abs: f32, soft_width: f32 },
    /// Match texels whose baked height exceeds `m` meters. Intended for
    /// land placement on bodies that use the Topography stage: `m = 0.0`
    /// picks everything above sea level. Reads the height accumulator
    /// already written by upstream stages, so this must run after them.
    ///
    /// `jitter_amp` perturbs the effective threshold by a low-frequency
    /// fbm in METERS, so the iso-contour reads as a fractal coastline
    /// rather than a smooth altitude line. Default 0 = crisp threshold
    /// (back-compat). Typical: 200–600 m for visible roughening.
    HeightAbove {
        m: f32,
        #[serde(default)]
        jitter_amp_m: f32,
        #[serde(default = "default_height_jitter_freq")]
        jitter_frequency: f64,
    },
    /// Match texels whose baked height exceeds `m` meters. Inverse of
    /// `HeightAbove` — useful for ice-shelf / shallow-sea partitioning
    /// when one wants a biome strictly below a threshold.
    HeightBelow {
        m: f32,
        #[serde(default)]
        jitter_amp_m: f32,
        #[serde(default = "default_height_jitter_freq")]
        jitter_frequency: f64,
    },
    /// Whittaker-style climate rule. Matches cells whose climate falls in
    /// the given temperature and precipitation box:
    ///   min_temp_c ≤ T < max_temp_c  AND  min_precip_mm ≤ P < max_precip_mm
    /// Each bound is optional; `None` means unbounded on that side. Requires
    /// the Climate stage to have run first. The `jitter_amp` / `jitter_frequency`
    /// fields add fbm-based noise to the decision so climate iso-contours read
    /// as fractal transitions rather than crisp boxes.
    TempPrecip {
        #[serde(default)]
        min_temp_c: Option<f32>,
        #[serde(default)]
        max_temp_c: Option<f32>,
        #[serde(default)]
        min_precip_mm: Option<f32>,
        #[serde(default)]
        max_precip_mm: Option<f32>,
        /// Fbm jitter on the climate decision, as a fraction of the bound
        /// width. 0.05-0.15 breaks up hard iso-contours.
        #[serde(default)]
        jitter_amp: f32,
        /// Noise frequency for boundary jitter.
        #[serde(default = "default_climate_jitter_freq")]
        jitter_frequency: f64,
    },
    /// Orogen / mountain-chain rule. Matches cells whose cumulative orogen
    /// intensity (from the Tectonics stage) exceeds `min_intensity`.
    /// Optionally gated on `max_age_myr` so active young orogens can be
    /// distinguished from worn ancient ones. Intended for authoring a
    /// distinct "active mountain belt" biome that visibly traces the
    /// tectonic lineaments regardless of absolute altitude.
    ///
    /// `jitter_amp` perturbs the intensity threshold by an fbm noise
    /// (in intensity units, ~0.05–0.15 for visible roughening). Default
    /// 0 = crisp threshold.
    Orogen {
        min_intensity: f32,
        #[serde(default)]
        max_age_myr: Option<f32>,
        #[serde(default)]
        jitter_amp: f32,
        #[serde(default = "default_orogen_jitter_freq")]
        jitter_frequency: f64,
    },
}

fn default_climate_jitter_freq() -> f64 {
    3.0
}
fn default_height_jitter_freq() -> f64 {
    3.0
}
fn default_orogen_jitter_freq() -> f64 {
    3.5
}

/// Registers the body's biome palette and paints `biome_map` with biome ids.
///
/// Each biome has an associated `BiomeRule` that decides which texels it
/// owns. Rules are evaluated in declared order; the first match wins. The
/// last biome MUST have a `Default` rule so every texel is covered.
#[derive(Debug, Clone, Deserialize)]
pub struct Biomes {
    pub biomes: Vec<BiomeParams>,
    #[serde(default)]
    pub rules: Vec<BiomeRule>,
}

impl Stage for Biomes {
    fn name(&self) -> &str {
        "biomes"
    }
    // Biomes stage runs after any structural stages (Differentiate and, if
    // present, Megabasin) so `NearBasin` rules see a populated basin list.
    // Listing Differentiate as the hard dependency keeps bodies that don't
    // run Megabasin free to still use Biomes.
    fn dependencies(&self) -> &[&str] {
        &["differentiate"]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        assert!(
            !self.biomes.is_empty(),
            "Biomes stage requires at least one biome"
        );
        builder.biomes = self.biomes.clone();

        // If no rules were configured, fall back to the old behavior: paint
        // everything with biome 0. Preserves backwards compatibility for
        // simple single-biome pipelines.
        if self.rules.is_empty() {
            for face in CubemapFace::ALL {
                for v in builder.biome_map.face_data_mut(face) {
                    *v = 0;
                }
            }
            return;
        }

        assert_eq!(
            self.rules.len(),
            self.biomes.len(),
            "Biomes: rules.len() must equal biomes.len() (one rule per biome)",
        );
        assert!(
            matches!(self.rules.last(), Some(BiomeRule::Default)),
            "Biomes: last rule must be Default (catch-all)",
        );

        let has_near_basin_rule = self
            .rules
            .iter()
            .any(|r| matches!(r, BiomeRule::NearBasin { .. }));
        assert!(
            !has_near_basin_rule || !builder.megabasins.is_empty(),
            "Biomes: NearBasin rules require the Megabasin stage to run first \
             (megabasins list is empty)",
        );

        let basins = builder.megabasins.clone();
        let body_radius = builder.radius_m;
        let seed = builder.stage_seed();
        let res = builder.cubemap_resolution;
        let inv_res = 1.0 / res as f32;
        let rules = &self.rules;

        // Read-only views of intermediate fields needed by the rule
        // evaluator. Height for HeightAbove/HeightBelow; temperature +
        // precipitation for TempPrecip (Climate stage output); orogen
        // intensity + age for Orogen (Tectonics stage output). All default
        // to zero when the relevant upstream stage didn't run, which is
        // fine because the corresponding rules won't be used on bodies
        // that skip that stage.
        let height = &builder.height_contributions.height;
        let temperature = &builder.temperature_c;
        let precipitation = &builder.precipitation_mm;
        let orogen_intensity = &builder.orogen_intensity;
        let orogen_age_myr = &builder.orogen_age_myr;

        // Paint into a scratch cubemap and swap at the end. This avoids a
        // double-mut-borrow on the builder when a rule needs to read another
        // builder field (e.g. height) while we're writing biome_map.
        let mut scratch = crate::cubemap::Cubemap::<u8>::new(res);

        for face in CubemapFace::ALL {
            let data = scratch.face_data_mut(face);
            let height_face = height.face_data(face);
            let temp_face = temperature.face_data(face);
            let precip_face = precipitation.face_data(face);
            let orog_int_face = orogen_intensity.face_data(face);
            let orog_age_face = orogen_age_myr.face_data(face);
            for y in 0..res {
                for x in 0..res {
                    let idx = (y * res + x) as usize;
                    let u = (x as f32 + 0.5) * inv_res;
                    let v = (y as f32 + 0.5) * inv_res;
                    let dir = face_uv_to_dir(face, u, v).normalize();
                    let cx = CellCtx {
                        dir,
                        height_m: height_face[idx],
                        temp_c: temp_face[idx],
                        precip_mm: precip_face[idx],
                        orogen_intensity: orog_int_face[idx],
                        orogen_age_myr: orog_age_face[idx],
                    };

                    let mut chosen: u8 = (rules.len() as u8).saturating_sub(1);
                    for (rule_idx, rule) in rules.iter().enumerate() {
                        if rule_matches(rule, &cx, &basins, body_radius, seed, rule_idx) {
                            chosen = rule_idx as u8;
                            break;
                        }
                    }
                    data[idx] = chosen;
                }
            }
        }

        builder.biome_map = scratch;
    }
}

/// Per-cell context passed to `rule_matches`. Collecting these into a
/// struct keeps the rule dispatch signature readable as the rule set grows.
struct CellCtx {
    dir: Vec3,
    height_m: f32,
    temp_c: f32,
    precip_mm: f32,
    orogen_intensity: f32,
    orogen_age_myr: f32,
}

fn rule_matches(
    rule: &BiomeRule,
    cx: &CellCtx,
    basins: &[crate::stages::BasinDef],
    body_radius: f32,
    seed: u64,
    rule_idx: usize,
) -> bool {
    let dir = cx.dir;
    let cell_height_m = cx.height_m;
    match rule {
        BiomeRule::Default => true,
        BiomeRule::HeightAbove {
            m,
            jitter_amp_m,
            jitter_frequency,
        } => {
            let threshold = if *jitter_amp_m > 0.0 {
                let s = splitmix64(
                    seed ^ 0x6F2A_4D81_E573_C019
                        ^ (rule_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                let n = fbm3(
                    dir.x as f64 * *jitter_frequency,
                    dir.y as f64 * *jitter_frequency,
                    dir.z as f64 * *jitter_frequency,
                    s,
                    4,
                    0.55,
                    2.1,
                ) as f32;
                *m + n * *jitter_amp_m
            } else {
                *m
            };
            cell_height_m > threshold
        }
        BiomeRule::HeightBelow {
            m,
            jitter_amp_m,
            jitter_frequency,
        } => {
            let threshold = if *jitter_amp_m > 0.0 {
                let s = splitmix64(
                    seed ^ 0xC1B7_5298_3F4D_E061
                        ^ (rule_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                let n = fbm3(
                    dir.x as f64 * *jitter_frequency,
                    dir.y as f64 * *jitter_frequency,
                    dir.z as f64 * *jitter_frequency,
                    s,
                    4,
                    0.55,
                    2.1,
                ) as f32;
                *m + n * *jitter_amp_m
            } else {
                *m
            };
            cell_height_m <= threshold
        }
        BiomeRule::NearBasin { max_t } => {
            for b in basins {
                let c = b.center_dir.normalize();
                let cos_d = dir.dot(c).clamp(-1.0, 1.0);
                let surface_dist = cos_d.acos() * body_radius;
                if surface_dist / b.radius_m <= *max_t {
                    return true;
                }
            }
            false
        }
        BiomeRule::Latitude {
            min_abs,
            soft_width,
        } => {
            let abs_lat = dir.y.abs();
            if *soft_width <= 0.0 {
                return abs_lat >= *min_abs;
            }
            // Low-freq fbm jitter on the threshold — same pattern as the
            // Fbm rule's boundary-softening jitter. A few lobes per
            // hemisphere is plenty; the goal is a lobed ice-cap outline,
            // not a high-frequency rim.
            let rule_seed = splitmix64(
                seed ^ 0x5A7F_9B1C_32E4_D091
                    ^ (rule_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let jitter = fbm3(
                dir.x as f64 * 2.4,
                dir.y as f64 * 2.4,
                dir.z as f64 * 2.4,
                rule_seed,
                4,
                0.55,
                2.1,
            ) as f32;
            let effective_threshold = *min_abs - jitter * *soft_width;
            abs_lat >= effective_threshold
        }
        BiomeRule::Fbm {
            frequency,
            threshold,
            octaves,
        } => {
            // Each rule needs an independent noise field — fold the rule
            // index into the seed so two Fbm rules at different frequencies
            // don't sample correlated patterns.
            let rule_seed = splitmix64(
                seed ^ 0x8F17_C5D3_A209_B46F
                    ^ (rule_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let n = fbm3(
                dir.x as f64 * *frequency,
                dir.y as f64 * *frequency,
                dir.z as f64 * *frequency,
                rule_seed,
                *octaves,
                0.55,
                2.1,
            ) as f32;
            // High-frequency fractal dither on the threshold crossing.
            // Without this the Fbm iso-contour is a smooth curve at the
            // noise's characteristic scale, and any albedo gap between the
            // matched biome and the fallthrough biome shows up as a long
            // linear band. The jitter breaks that contour into a fractal
            // boundary so the transition reads like natural material
            // variation, not a terraced contour map. Uses an independent
            // noise octave from the same seed family.
            let jitter_seed = splitmix64(rule_seed ^ 0x4D58_A1F3_92BE_CA71);
            let jitter = fbm3(
                dir.x as f64 * *frequency * 7.0,
                dir.y as f64 * *frequency * 7.0,
                dir.z as f64 * *frequency * 7.0,
                jitter_seed,
                3,
                0.5,
                2.3,
            ) as f32;
            let effective = n + jitter * 0.22;
            effective >= *threshold
        }
        BiomeRule::TempPrecip {
            min_temp_c,
            max_temp_c,
            min_precip_mm,
            max_precip_mm,
            jitter_amp,
            jitter_frequency,
        } => {
            // Jitter the effective temperature and precipitation samples by
            // a fraction of each bound's range. Without jitter, climate-zone
            // boundaries read as smooth contours; the jitter breaks them
            // into fractal transitions.
            let (t_jitter, p_jitter) = if *jitter_amp > 0.0 {
                let rule_seed = splitmix64(
                    seed ^ 0x2D4E_71C0_9B8A_5F36
                        ^ (rule_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                let tj = fbm3(
                    dir.x as f64 * *jitter_frequency,
                    dir.y as f64 * *jitter_frequency,
                    dir.z as f64 * *jitter_frequency,
                    rule_seed,
                    3,
                    0.55,
                    2.1,
                ) as f32;
                let pj = fbm3(
                    dir.x as f64 * *jitter_frequency * 1.7,
                    dir.y as f64 * *jitter_frequency * 1.7,
                    dir.z as f64 * *jitter_frequency * 1.7,
                    splitmix64(rule_seed ^ 0xA1B2_C3D4_E5F6_0789),
                    3,
                    0.55,
                    2.1,
                ) as f32;
                (tj, pj)
            } else {
                (0.0, 0.0)
            };

            // For temperature: jitter is scaled by the band width if both
            // bounds are present; otherwise a fixed 5°C so the band still
            // breaks at open-ended climate zones.
            let temp_range = match (min_temp_c, max_temp_c) {
                (Some(lo), Some(hi)) => (hi - lo).abs().max(1.0),
                _ => 5.0,
            };
            let precip_range = match (min_precip_mm, max_precip_mm) {
                (Some(lo), Some(hi)) => (hi - lo).abs().max(1.0),
                _ => 200.0,
            };
            let t_eff = cx.temp_c + t_jitter * jitter_amp * temp_range;
            let p_eff = cx.precip_mm + p_jitter * jitter_amp * precip_range;

            if let Some(lo) = min_temp_c
                && t_eff < *lo
            {
                return false;
            }
            if let Some(hi) = max_temp_c
                && t_eff >= *hi
            {
                return false;
            }
            if let Some(lo) = min_precip_mm
                && p_eff < *lo
            {
                return false;
            }
            if let Some(hi) = max_precip_mm
                && p_eff >= *hi
            {
                return false;
            }
            true
        }
        BiomeRule::Orogen {
            min_intensity,
            max_age_myr,
            jitter_amp,
            jitter_frequency,
        } => {
            let effective_intensity = if *jitter_amp > 0.0 {
                let s = splitmix64(
                    seed ^ 0x4F8E_2A19_7C56_B043
                        ^ (rule_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                let n = fbm3(
                    dir.x as f64 * *jitter_frequency,
                    dir.y as f64 * *jitter_frequency,
                    dir.z as f64 * *jitter_frequency,
                    s,
                    4,
                    0.55,
                    2.1,
                ) as f32;
                cx.orogen_intensity + n * *jitter_amp
            } else {
                cx.orogen_intensity
            };
            if effective_intensity < *min_intensity {
                return false;
            }
            if let Some(max_age) = max_age_myr
                && cx.orogen_age_myr > *max_age
            {
                return false;
            }
            true
        }
    }
}
