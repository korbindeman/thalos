use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::stage::Stage;
use crate::types::Material;

/// Defines the materials palette for a body based on its bulk composition.
///
/// Reads `builder.composition` and `builder.body_age_gyr` and writes a
/// 4-entry palette at the fixed indices below. Downstream stages reference
/// materials by these indices — the palette is frozen for the rest of the
/// pipeline run once this stage completes.
///
///   0 = highland  — primordial crust (bright, rough)
///   1 = mare      — volcanic fill (dark, smoother)
///   2 = fresh ejecta — brightest, recently excavated subsurface
///   3 = mature regolith — terminal weathered surface layer
#[derive(Debug, Clone, Deserialize)]
pub struct Differentiate;

/// Material palette indices, frozen by `Differentiate` and referenced by
/// every downstream stage, the sampler, and the renderer. These are the
/// single source of truth — do not hardcode numeric `material_id` anywhere
/// else in the crate.
pub const MAT_HIGHLAND: u32 = 0;
pub const MAT_MARE: u32 = 1;
pub const MAT_FRESH_EJECTA: u32 = 2;
pub const MAT_MATURE_REGOLITH: u32 = 3;

// Roughness values are shared across compositions — they describe surface
// texture driven by process (how the material came to rest), not by what
// it's made of.
const ROUGHNESS_HIGHLAND: f32 = 0.8;
const ROUGHNESS_MARE: f32 = 0.5;
const ROUGHNESS_FRESH_EJECTA: f32 = 0.7;
const ROUGHNESS_MATURE_REGOLITH: f32 = 0.6;

impl Stage for Differentiate {
    fn name(&self) -> &str {
        "differentiate"
    }
    fn dependencies(&self) -> &[&str] {
        &[]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        let comp = &builder.composition;

        // Secondary age effect: older bodies have been exposure-darkened
        // on average, so their *mature* albedos trend slightly lower. Fresh
        // albedos represent recently-exposed material and are left alone.
        // At 4.5 Gyr this gives a 10% reduction; at 0 Gyr, no change.
        let age_darken = 1.0 - 0.1 * (builder.body_age_gyr / 4.5);

        // Pick per-material gray values based on dominant composition. Each
        // branch targets the ranges documented in docs/procedural_mira.md §4.
        // `fresh_ejecta` is keyed off fresh-highland since that's the
        // subsurface material being excavated. Linear albedo, not sRGB.
        let (fresh_ejecta, highland_mature, mare_mature) = if comp.ice > 0.5 {
            // Icy body (Glacis/Calyx): crystalline water ice dominates the
            // crust; "mare" is cryovolcanic plume fill or dirty ice. Fresh
            // ice is very bright; weathered ice is darker/dustier.
            (0.75, 0.40, 0.20)
        } else if comp.iron > 0.3 && comp.silicate < 0.5 {
            // Metallic / iron-silicate body (Ignis): dark throughout. Even
            // "fresh" metallic ejecta is only modestly brighter than mature
            // surface — metal and iron-rich silicates are intrinsically dark
            // and don't space-weather as dramatically as anorthosite.
            (0.10, 0.07, 0.05)
        } else {
            // Default: silicate-dominated rocky body (Mira and friends).
            // Anorthositic highlands vs basaltic mare — the canonical lunar
            // palette.
            (0.28, 0.12, 0.07)
        };

        // Mature regolith is the terminal-state weathered layer — slightly
        // darker than mature highland because it represents the most
        // gardened, space-weathered surface on the body.
        let regolith_base = highland_mature * 0.90;

        let highland_mature_aged = highland_mature * age_darken;
        let mare_mature_aged = mare_mature * age_darken;
        let regolith_aged = regolith_base * age_darken;

        builder.materials = vec![
            // MAT_HIGHLAND: primordial crust, bright, rough
            Material {
                albedo: gray(highland_mature_aged),
                roughness: ROUGHNESS_HIGHLAND,
            },
            // MAT_MARE: volcanic fill, dark, smoother than highland
            Material {
                albedo: gray(mare_mature_aged),
                roughness: ROUGHNESS_MARE,
            },
            // MAT_FRESH_EJECTA: brightest, not age-darkened
            Material {
                albedo: gray(fresh_ejecta),
                roughness: ROUGHNESS_FRESH_EJECTA,
            },
            // MAT_MATURE_REGOLITH: terminal weathered layer
            Material {
                albedo: gray(regolith_aged),
                roughness: ROUGHNESS_MATURE_REGOLITH,
            },
        ];
    }
}

/// Neutral gray albedo from a single scalar. Real lunar/rocky surfaces have
/// very slight spectral variation but it's well below visual threshold at
/// impostor distance — a single channel keeps the tuning surface small and
/// lets SpaceWeather add any color shift it wants later.
fn gray(v: f32) -> [f32; 3] {
    [v, v, v]
}
