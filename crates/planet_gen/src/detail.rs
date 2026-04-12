//! Stage 7 — detail hook.
//!
//! Per spec §Stage 7: this stage does **not** bake high-frequency detail
//! into the surface data.  Instead it records the parameters a runtime
//! consumer (a fragment shader, a closer-LOD bake, a virtualized-geometry
//! tile job) needs to synthesize sub-sample detail on demand.
//!
//! For v0.1 the consumer is the planet impostor shader, which runs a
//! per-fragment small-crater accumulation pass over a 3D-hashed cell grid
//! on the unit sphere.  The parameters here are exactly what that pass
//! needs:
//!
//! - Body radius (so the shader can convert unit-sphere distance to metres)
//! - SFD shape: power-law `α`, `[d_min, d_max]`
//! - Total density (craters per km²) integrated over `[d_min, d_max]`
//! - Simple-to-complex transition diameter (so the shader can dispatch
//!   morphology the same way the bake-side stamping does)
//! - Per-body deterministic seed
//!
//! `d_max` here is the *largest* crater the shader synthesizes — it must
//! match the `min_diameter_m` the main-crater bake stage was run with so
//! the two populations meet exactly at the boundary with no gap or
//! overlap.  [`crate::pipeline`] enforces this by deriving both from a
//! single split-diameter when it constructs the configs.

use crate::derived::DerivedProperties;
use crate::descriptor::PlanetDescriptor;
use crate::main_craters::SFD_ALPHA;
use crate::seeding::sub_seed;

pub const STAGE_NAME: &str = "detail";

/// Smallest crater the shader synthesizes, in metres.  This is a fixed
/// floor — going smaller costs per-fragment work without adding visible
/// detail at typical viewing distances.  Tune in one place.
pub const DEFAULT_SHADER_MIN_DIAMETER_M: f64 = 200.0;

/// Parameters for runtime detail synthesis.  Emitted by the pipeline
/// alongside the [`crate::SurfaceState`] and consumed by the renderer.
///
/// All distances are in metres; the consumer is responsible for any
/// non-physical scale conversions it needs.
#[derive(Clone, Copy, Debug)]
pub struct DetailParams {
    /// Body radius, metres.  The shader uses this to map unit-sphere
    /// positions to physical distances on the surface.
    pub body_radius_m: f64,
    /// Smallest crater the shader stamps, metres.
    pub d_min_m: f64,
    /// Largest crater the shader stamps, metres.  Equal to the `min`
    /// the bake-side main-crater stage was run with — the two populations
    /// meet exactly here.
    pub d_max_m: f64,
    /// Power-law cumulative exponent: `N(>D) ∝ D^-alpha`.
    pub sfd_alpha: f64,
    /// Global density constant: craters per km² at 1 km diameter, scaled
    /// by body age and impact flux.  Equivalent to `K_lunar × exposure`.
    /// The consumer computes per-octave density as
    /// `global_k × (d_lo^-α − d_hi^-α)` with `d` in km — a single
    /// straightforward formula that doesn't depend on how the octaves
    /// are sliced.
    pub global_k_per_km2: f64,
    /// Simple-to-complex crater transition diameter, metres.  Lets the
    /// shader pick the right morphology per crater the same way the bake
    /// does.
    pub d_sc_m: f64,
    /// Body age in gigayears.  The shader uses this as the upper bound
    /// of the uniform age distribution it samples from per crater, so
    /// most craters end up old (weathered) and only the youngest ~2%
    /// land inside the fresh-rim window.
    pub body_age_gyr: f64,
    /// Deterministic per-body seed for the shader's hash.  Stable across
    /// runs and platforms.
    pub seed: u64,
}

/// Compute [`DetailParams`] for the body.  `d_split_m` is the diameter at
/// which bake and shader meet — bake handles `[d_split, d_max_main]`,
/// shader handles `[d_min_shader, d_split]`.
pub fn compute(
    desc: &PlanetDescriptor,
    derived: &DerivedProperties,
    d_split_m: f64,
) -> DetailParams {
    use crate::main_craters::LUNAR_DENSITY_AT_1KM_PER_KM2;
    use crate::main_craters::LUNAR_REFERENCE_AGE_GYR;

    let d_min = DEFAULT_SHADER_MIN_DIAMETER_M.min(d_split_m * 0.5);
    let d_max = d_split_m;

    // `global_k` is the cumulative lunar density constant scaled by body
    // age and impact flux.  The consumer builds per-octave density as
    // `global_k × (d_lo^-α − d_hi^-α)` (d in km).
    let exposure = (desc.age_gyr / LUNAR_REFERENCE_AGE_GYR) * desc.impact_flux_multiplier;
    let global_k = LUNAR_DENSITY_AT_1KM_PER_KM2 * exposure;

    DetailParams {
        body_radius_m: desc.radius_m,
        d_min_m: d_min,
        d_max_m: d_max,
        sfd_alpha: SFD_ALPHA,
        global_k_per_km2: global_k,
        d_sc_m: derived.simple_to_complex_transition_m,
        body_age_gyr: desc.age_gyr,
        seed: sub_seed(desc.seed, STAGE_NAME),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derived::compute as compute_derived;
    use crate::reference_bodies;

    #[test]
    fn density_is_positive_for_lunar_split() {
        let desc = reference_bodies::luna();
        let d = compute_derived(&desc);
        let p = compute(&desc, &d, 5_000.0);
        assert!(p.global_k_per_km2 > 0.0);
        assert!(p.d_min_m < p.d_max_m);
        assert_eq!(p.d_max_m, 5_000.0);
        assert_eq!(p.body_radius_m, desc.radius_m);
    }

    #[test]
    fn deterministic() {
        let desc = reference_bodies::luna();
        let d = compute_derived(&desc);
        let a = compute(&desc, &d, 5_000.0);
        let b = compute(&desc, &d, 5_000.0);
        assert_eq!(a.seed, b.seed);
        assert_eq!(a.global_k_per_km2, b.global_k_per_km2);
    }

    #[test]
    fn higher_flux_increases_density() {
        let mut desc = reference_bodies::luna();
        let d = compute_derived(&desc);
        let p1 = compute(&desc, &d, 5_000.0);
        desc.impact_flux_multiplier = 3.0;
        let p2 = compute(&desc, &d, 5_000.0);
        assert!(p2.global_k_per_km2 > p1.global_k_per_km2);
    }

    #[test]
    fn shader_d_min_floors_at_default() {
        // For a normal d_split, d_min defaults to DEFAULT_SHADER_MIN_DIAMETER_M.
        let desc = reference_bodies::luna();
        let d = compute_derived(&desc);
        let p = compute(&desc, &d, 10_000.0);
        assert_eq!(p.d_min_m, DEFAULT_SHADER_MIN_DIAMETER_M);
    }

    #[test]
    fn shader_d_min_clamps_below_d_split() {
        // For an unusually low d_split (small body / weird config), d_min
        // must clamp to d_split * 0.5 so we don't end up with d_min ≥ d_max.
        let desc = reference_bodies::luna();
        let d = compute_derived(&desc);
        let p = compute(&desc, &d, 300.0);
        assert!(p.d_min_m < p.d_max_m);
        assert_eq!(p.d_min_m, 150.0);
    }
}
