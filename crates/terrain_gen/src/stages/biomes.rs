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

        for face in CubemapFace::ALL {
            let data = builder.biome_map.face_data_mut(face);
            for y in 0..res {
                for x in 0..res {
                    let u = (x as f32 + 0.5) * inv_res;
                    let v = (y as f32 + 0.5) * inv_res;
                    let dir = face_uv_to_dir(face, u, v).normalize();

                    let mut chosen: u8 = (rules.len() as u8).saturating_sub(1);
                    for (idx, rule) in rules.iter().enumerate() {
                        if rule_matches(rule, dir, &basins, body_radius, seed, idx) {
                            chosen = idx as u8;
                            break;
                        }
                    }
                    data[(y * res + x) as usize] = chosen;
                }
            }
        }
    }
}

fn rule_matches(
    rule: &BiomeRule,
    dir: Vec3,
    basins: &[crate::stages::BasinDef],
    body_radius: f32,
    seed: u64,
    rule_idx: usize,
) -> bool {
    match rule {
        BiomeRule::Default => true,
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
    }
}
