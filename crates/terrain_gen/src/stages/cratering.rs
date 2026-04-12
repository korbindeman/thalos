use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::crater_profile::{
    crater_dimensions, crater_profile, degradation_factor, morphology_for_radius,
};
use crate::seeding::Rng;
use crate::stage::Stage;
use crate::types::Crater;
use super::MAT_HIGHLAND;
use super::util::for_texels_in_cap;

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

/// Generates the full crater population via power-law SFD and bakes large
/// craters into the cubemap height field.
///
/// All craters are stored in `builder.craters` for the mid-frequency SSBO.
/// Craters above `cubemap_bake_threshold_m` are additionally rasterized
/// into the height accumulator for impostor-distance visibility.
///
/// Body age is read from `BodyBuilder::body_age_gyr` (single source).
#[derive(Debug, Clone, Deserialize)]
pub struct Cratering {
    pub total_count: u32,
    /// Cumulative SFD slope (N(>D) ∝ D⁻ᵅ). Lunar large-branch ≈ 1.8–2.0.
    pub sfd_slope: f64,
    pub min_radius_m: f64,
    pub max_radius_m: f64,
    pub cubemap_bake_threshold_m: f32,
}

impl Stage for Cratering {
    fn name(&self) -> &str { "cratering" }
    fn dependencies(&self) -> &[&str] { &["megabasin"] }

    fn apply(&self, builder: &mut BodyBuilder) {
        let mut rng = Rng::new(builder.stage_seed());
        let body_radius = builder.radius_m;
        let body_age_gyr = builder.body_age_gyr;

        // Generate crater population with per-crater ±15% size jitter to break
        // cookie-cutter repetition (doc §1: "vary profile parameters by ±20%").
        //
        // Age distribution: strong bias toward old (most craters formed during
        // the Late Heavy Bombardment). Use age = body_age_gyr * (1 - u^AGE_K)
        // with AGE_K = 4, which puts ~84% of craters older than body_age_gyr/2
        // (satisfies the "≥80% old" target from the spec).
        const AGE_K: f64 = 4.0;

        let mut craters = Vec::with_capacity(self.total_count as usize);
        for _ in 0..self.total_count {
            let radius_m =
                rng.power_law(self.min_radius_m, self.max_radius_m, self.sfd_slope) as f32;
            let center = rng.unit_vector().as_vec3();

            let (base_depth, base_rim) = crater_dimensions(radius_m);
            let jitter = 1.0 + rng.next_f64_signed() as f32 * 0.15;
            let depth_m = base_depth * jitter;
            let rim_height_m = base_rim * jitter;

            let u = rng.next_f64();
            let age_gyr = (body_age_gyr as f64 * (1.0 - u.powf(AGE_K))) as f32;

            craters.push(Crater {
                center,
                radius_m,
                depth_m,
                rim_height_m,
                age_gyr,
                material_id: MAT_HIGHLAND,
            });
        }

        // Force the 3 largest craters to be young so SpaceWeather has ray-system
        // candidates. Without this the age bias toward old makes young large
        // craters statistically invisible. We find them by partial sort on
        // radius, then overwrite their ages uniformly in [0, 0.15 * body_age].
        const YOUNG_LARGE_COUNT: usize = 3;
        if craters.len() >= YOUNG_LARGE_COUNT {
            // Indices of the top-N craters by radius.
            let mut idx: Vec<usize> = (0..craters.len()).collect();
            idx.select_nth_unstable_by(YOUNG_LARGE_COUNT - 1, |&a, &b| {
                craters[b].radius_m.partial_cmp(&craters[a].radius_m).unwrap()
            });
            for &i in &idx[..YOUNG_LARGE_COUNT] {
                let u = rng.next_f64();
                craters[i].age_gyr = (body_age_gyr as f64 * 0.15 * u) as f32;
            }
        }

        // Sort oldest-first so younger craters stamp over older ones in the cubemap.
        craters.sort_by(|a, b| b.age_gyr.partial_cmp(&a.age_gyr).unwrap());

        // Bake large craters to height cubemap with age-based diffusion degradation.
        let res = builder.cubemap_resolution;
        let acc = &mut builder.height_contributions;

        for crater in craters.iter().filter(|c| c.radius_m >= self.cubemap_bake_threshold_m) {
            let center = crater.center.normalize();
            let morph = morphology_for_radius(crater.radius_m);
            let radius = crater.radius_m;

            // Apply diffusion degradation: older/smaller craters are more eroded.
            let degrad = degradation_factor(radius, crater.age_gyr);
            let depth = crater.depth_m * degrad;
            let rim_h = crater.rim_height_m * degrad;

            // Influence extends to 5R (ejecta blanket covers ~90% within 5R).
            let influence_angle = crater.influence_radius_m() / body_radius;

            for_texels_in_cap(res, center, influence_angle, |face, x, y, _dir, angular_dist| {
                let surface_dist = angular_dist * body_radius;
                let t = surface_dist / radius;
                let h = crater_profile(t, depth, rim_h, radius, morph);
                if h.abs() > 1e-3 {
                    acc.add_height(face, x, y, h);
                }
            });
        }

        builder.craters = craters;
        // Publish the threshold to BodyData so the sampler and shader can
        // skip baked craters during Layer 2 iteration (otherwise their
        // contribution is counted twice — once from the cubemap texel, once
        // from the SSBO).
        builder.cubemap_bake_threshold_m = self.cubemap_bake_threshold_m;
    }
}

