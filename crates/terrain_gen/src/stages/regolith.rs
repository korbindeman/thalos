use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::stage::Stage;
use crate::types::DetailNoiseParams;

/// Configures the high-frequency detail noise layer for regolith texture.
///
/// This stage produces no cubemap output (below resolution). It sets up
/// the `DetailNoiseParams` that the UDLOD renderer will use at close range
/// to synthesize the salt-and-pepper texture of billions of micro-craters.
#[derive(Debug, Clone, Deserialize)]
pub struct Regolith {
    /// Typical depth of small-crater texture in meters.
    pub amplitude_m: f32,
    /// Approximate spacing of smallest visible crater-like features.
    pub characteristic_wavelength_m: f32,
    /// Density multiplier relative to continued SFD from explicit craters.
    pub crater_density_multiplier: f32,
}

impl Stage for Regolith {
    fn name(&self) -> &str { "regolith" }
    fn dependencies(&self) -> &[&str] { &["cratering"] }

    fn apply(&self, builder: &mut BodyBuilder) {
        builder.detail_params = DetailNoiseParams {
            body_radius_m: builder.radius_m,
            // d_min: ~smallest resolvable at close-up distance (~80 m).
            // d_max: matches Cratering.cubemap_bake_threshold × 2 (diameter),
            //        so the shader tail terminates exactly where the baked
            //        population begins — seamless handoff.
            // Shader covers the entire realistic crater range (80 m to
            // ~80 km diameter — 10 octaves from d_min). Large craters at
            // impostor distance render at pixel-accurate sharpness with
            // no cubemap-filter smearing, so baked cratering is only used
            // for rare extra-large features (≥ 60 km diameter).
            d_min_m: 80.0,
            d_max_m: 60_000.0,
            sfd_alpha: 2.0,
            // Neukum-derived density. At alpha=2 the shader's per-bin
            // formula expands to K×(d_lo⁻² − d_hi⁻²) which approximates
            // the cumulative SFD. 0.05 gives ≲1 expected crater per
            // Voronoi cell across every octave — sparse at close range
            // and anchored by mid/large craters at far range.
            global_k_per_km2: self.crater_density_multiplier * 0.05,
            // Simple/complex transition diameter on Mira (half-lunar gravity):
            // ~30 km. All shader craters are well below, so they use the
            // simple bowl profile.
            d_sc_m: 30_000.0,
            body_age_gyr: builder.body_age_gyr,
            seed: builder.stage_seed(),
        };
    }
}
