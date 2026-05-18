use glam::Vec3;
use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BiomeRule {
    /// Catch-all. Always matches. Put at the end of the list.
    Default,
    /// Match texels near a megabasin center (by angular distance, in units
    /// of the basin's radius on the sphere). Used for basin-floor biomes.
    NearBasin { max_t: f32 },
    /// Match texels whose low-freq fbm noise exceeds `threshold`. Gives
    /// patchy regional assignment for highland subtypes (KREEP, melt).
    Fbm {
        frequency: f32,
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
        jitter_frequency: f32,
    },
    /// Match texels whose baked height exceeds `m` meters. Inverse of
    /// `HeightAbove` — useful for ice-shelf / shallow-sea partitioning
    /// when one wants a biome strictly below a threshold.
    HeightBelow {
        m: f32,
        #[serde(default)]
        jitter_amp_m: f32,
        #[serde(default = "default_height_jitter_freq")]
        jitter_frequency: f32,
    },
}

fn default_height_jitter_freq() -> f32 {
    3.0
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

        let height = &builder.height_contributions.height;

        // Paint into a scratch cubemap and swap at the end. This avoids a
        // double-mut-borrow on the builder when a rule needs to read another
        // builder field (e.g. height) while we're writing biome_map.
        let mut scratch = crate::cubemap::Cubemap::<u8>::new(res);

        for face in CubemapFace::ALL {
            let data = scratch.face_data_mut(face);
            let height_face = height.face_data(face);
            for y in 0..res {
                for x in 0..res {
                    let idx = (y * res + x) as usize;
                    let u = (x as f32 + 0.5) * inv_res;
                    let v = (y as f32 + 0.5) * inv_res;
                    let dir = face_uv_to_dir(face, u, v).normalize();
                    let cx = CellCtx {
                        dir,
                        height_m: height_face[idx],
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

/// Per-cell context passed to `rule_matches`.
struct CellCtx {
    dir: Vec3,
    height_m: f32,
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
                    dir.x * *jitter_frequency,
                    dir.y * *jitter_frequency,
                    dir.z * *jitter_frequency,
                    s as u32,
                    4,
                    0.55,
                    2.1,
                );
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
                    dir.x * *jitter_frequency,
                    dir.y * *jitter_frequency,
                    dir.z * *jitter_frequency,
                    s as u32,
                    4,
                    0.55,
                    2.1,
                );
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
                dir.x * 2.4,
                dir.y * 2.4,
                dir.z * 2.4,
                rule_seed as u32,
                4,
                0.55,
                2.1,
            );
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
                dir.x * *frequency,
                dir.y * *frequency,
                dir.z * *frequency,
                rule_seed as u32,
                *octaves,
                0.55,
                2.1,
            );
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
                dir.x * *frequency * 7.0,
                dir.y * *frequency * 7.0,
                dir.z * *frequency * 7.0,
                jitter_seed as u32,
                3,
                0.5,
                2.3,
            );
            let effective = n + jitter * 0.22;
            effective >= *threshold
        }
    }
}
