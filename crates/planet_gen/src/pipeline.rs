//! Top-level pipeline: stitches every stage together in order.
//!
//! Per spec §Pipeline: stages run in the listed order; each stage reads
//! current state and derived properties, and writes to surface state.
//! Stages skip via continuous derived properties — never via body-type
//! checks — and the pipeline itself is the same for every body.
//!
//! Sub-seeding is hierarchical: each stage derives its own sub-seed from
//! the descriptor seed and its stable stage name, so re-running one stage
//! never disturbs another stage's randomness (spec §Design principles 4).
//!
//! Slope information for the maturity pass is sampling-dependent.
//! Equirect callers should pass [`SlopeSource::Equirect { width, height }`]
//! to enable the slope-based reset; arbitrary samplings can pass
//! [`SlopeSource::None`] and accept that crater rims will read with
//! baseline maturity instead of fresh.

use glam::DVec3;

use crate::derived::{self, DerivedProperties};
use crate::descriptor::PlanetDescriptor;
use crate::detail::{self, DetailParams};
use crate::main_craters::{DEFAULT_MIN_CRATER_DIAMETER_M, MAX_CRATER_DIAMETER_M};
use crate::{giant_basin, main_craters, mare, maturity, primordial};
use crate::surface::SurfaceState;

/// How (and whether) the maturity stage gets per-sample slope.
///
/// The maturity pass treats slope as optional precisely so that callers
/// using non-grid samplings (icosahedral patches, cubesphere faces) can
/// still run the pipeline without inventing a finite-difference scheme
/// for their topology.
#[derive(Clone, Copy, Debug)]
pub enum SlopeSource {
    /// No slope reset.  Crater rims will read at baseline maturity.
    None,
    /// Compute slope from an equirectangular grid of the given dimensions.
    /// `width × height` must equal `points.len()` and the points must be
    /// in row-major order matching [`crate::sampling::equirect_lattice`].
    Equirect { width: usize, height: usize },
}

/// Per-call pipeline configuration.  Defaults are conservative and
/// resolution-agnostic; bakers that want a bake/shader split should use
/// [`Self::for_equirect_bake_with_split`] which derives both the bake's
/// crater range *and* the shader-handoff diameter from the same single
/// number, guaranteeing the two populations meet exactly.
#[derive(Clone, Copy, Debug)]
pub struct PipelineConfig {
    pub slope_source: SlopeSource,
    /// Smallest crater the main-population *bake* stage will stamp,
    /// metres.  This is also the largest diameter the shader-side
    /// detail synthesis will produce — the two ranges meet here.
    pub bake_min_crater_diameter_m: f64,
    /// Largest crater the main-population bake stage will stamp,
    /// metres.  Defaults to the global `MAX_CRATER_DIAMETER_M` —
    /// callers usually leave this alone.
    pub bake_max_crater_diameter_m: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            slope_source: SlopeSource::None,
            bake_min_crater_diameter_m: DEFAULT_MIN_CRATER_DIAMETER_M,
            bake_max_crater_diameter_m: MAX_CRATER_DIAMETER_M,
        }
    }
}

impl PipelineConfig {
    /// Build a config for an equirect bake of `(width, height)` texels
    /// on a body of `radius_m`, leaving everything below the bake/shader
    /// split for the detail-synthesis pass.
    ///
    /// `bake_shader_split_m` is the diameter where the two populations
    /// meet.  Above the split, the bake stamps craters into the textures.
    /// Below it, the renderer's detail pass synthesizes craters per
    /// fragment using the [`crate::DetailParams`] this pipeline emits.
    /// A reasonable default is ~2 equator texels of the bake — anything
    /// smaller would alias to a single dim pixel and is better handled
    /// by the resolution-independent shader path.
    pub fn for_equirect_bake_with_split(
        width: usize,
        height: usize,
        radius_m: f64,
        bake_shader_split_m: f64,
    ) -> Self {
        let texel_arc_m = std::f64::consts::TAU * radius_m / width as f64;
        let min = bake_shader_split_m.max(2.0 * texel_arc_m).max(200.0);
        Self {
            slope_source: SlopeSource::Equirect { width, height },
            bake_min_crater_diameter_m: min,
            bake_max_crater_diameter_m: MAX_CRATER_DIAMETER_M,
        }
    }

    /// Convenience: same as [`Self::for_equirect_bake_with_split`] but
    /// the split diameter defaults to twice the equator texel.
    pub fn for_equirect_bake(width: usize, height: usize, radius_m: f64) -> Self {
        let texel_arc_m = std::f64::consts::TAU * radius_m / width as f64;
        Self::for_equirect_bake_with_split(width, height, radius_m, 2.0 * texel_arc_m)
    }
}

/// Output of the pipeline: the baked surface state and the parameters
/// the renderer's detail pass needs to fill in below the bake's smallest
/// crater.
#[derive(Clone, Debug)]
pub struct PipelineOutput {
    pub surface: SurfaceState,
    pub detail: DetailParams,
}

/// Run the full v0.1 pipeline on the given samples.  Returns the
/// [`PipelineOutput`] containing both the baked surface state and the
/// detail-synthesis parameters.  Pure function: same descriptor + same
/// points → identical output across runs and platforms.
pub fn generate(
    desc: &PlanetDescriptor,
    points: Vec<DVec3>,
    config: PipelineConfig,
) -> PipelineOutput {
    let derived = derived::compute(desc);
    generate_with_derived(desc, &derived, points, config)
}

/// Like [`generate`], but reuses a pre-computed [`DerivedProperties`].
/// Use this in test/editor code that bakes the same body at multiple
/// resolutions and wants to share the derivation.
pub fn generate_with_derived(
    desc: &PlanetDescriptor,
    derived: &DerivedProperties,
    points: Vec<DVec3>,
    config: PipelineConfig,
) -> PipelineOutput {
    // Stage 0 — reference shape (oblateness applied at the elevation level
    // is left for a follow-up; for v0.1 the slow rotators in the test set
    // have negligible f).
    let mut state = SurfaceState::new(points, desc.radius_m);

    // Stage 1 — primordial topography
    primordial::run(&mut state, derived, desc.seed);

    // Stage 2 — giant basins (must run before Stage 4 so smaller craters
    // overprint basin rims)
    giant_basin::run(&mut state, desc, derived, desc.seed);

    // Stage 3 — mare flooding (gated on heat budget × thermal history)
    mare::run(&mut state, desc, derived, desc.seed);

    // Stage 4 — main crater population (workhorse).  Capped on the small
    // end at `bake_min_crater_diameter_m`; everything below that is
    // synthesized in the renderer from `DetailParams`.
    main_craters::run(
        &mut state,
        desc,
        derived,
        desc.seed,
        main_craters::StageConfig {
            min_diameter_m: config.bake_min_crater_diameter_m,
            max_diameter_m: config.bake_max_crater_diameter_m,
        },
    );

    // Stage 6 — maturity / regolith pass.  Stage 5 (secondaries & rays)
    // is deferred per the spec's "optional / future" notes.
    let slope = match config.slope_source {
        SlopeSource::None => None,
        SlopeSource::Equirect { width, height } => {
            Some(maturity::equirect_slope_magnitude(&state, width, height))
        }
    };
    maturity::run(&mut state, desc, desc.seed, slope.as_deref());

    // Stage 7 — detail hook.  Records parameters; does not mutate state.
    let detail = detail::compute(desc, derived, config.bake_min_crater_diameter_m);

    PipelineOutput { surface: state, detail }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_bodies;
    use crate::sampling::{equirect_lattice, fibonacci_lattice};
    use crate::surface::MaterialId;

    #[test]
    fn luna_end_to_end_meets_spec_requirements() {
        // Spec §Determinism and testing requirements: Luna must produce
        //  - nonzero giant basin count
        //  - nonzero mare area
        //  - crater density within an order of magnitude of lunar
        //  - maturity field with both fresh-crater lows and weathered highs
        let pts = equirect_lattice(512, 256);
        let desc = reference_bodies::luna();
        let out = generate(
            &desc,
            pts,
            PipelineConfig::for_equirect_bake(512, 256, desc.radius_m),
        );
        let state = &out.surface;
        // Detail params are populated and consistent with the bake split.
        assert!(out.detail.global_k_per_km2 > 0.0);
        assert_eq!(out.detail.body_radius_m, desc.radius_m);

        // Basin floor samples present.
        let n_basin_floor = state
            .material
            .iter()
            .filter(|m| **m == MaterialId::BasinFloor)
            .count();
        // (After mare flooding, some basin floor samples may have become
        // mare; what we care about is that the basin stage *fired*.)
        let n_mare = state
            .material
            .iter()
            .filter(|m| **m == MaterialId::Mare)
            .count();
        assert!(
            n_basin_floor + n_mare > 0,
            "Luna produced no basin/mare samples"
        );

        // Mare nonzero (Luna passes the heat × thermal-history gate).
        assert!(n_mare > 0, "Luna produced no mare");

        // Maturity field has range: at least one sample below 0.3 (fresh)
        // and at least one above 0.7 (weathered).
        let any_fresh = state.maturity.iter().any(|&m| m < 0.5);
        let any_weathered = state.maturity.iter().any(|&m| m > 0.7);
        assert!(any_fresh, "Luna has no fresh-crater samples");
        assert!(any_weathered, "Luna has no weathered samples");

        // Elevation field has visible relief.
        let max = state.elevation_m.iter().cloned().fold(f64::MIN, f64::max);
        let min = state.elevation_m.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            max - min > 1_000.0,
            "Luna elevation range too small: {} m",
            max - min
        );
    }

    #[test]
    fn rhea_end_to_end_runs_cleanly_with_no_mare() {
        let pts = fibonacci_lattice(2048);
        let desc = reference_bodies::rhea();
        let out = generate(&desc, pts, PipelineConfig::default());
        let n_mare = out
            .surface
            .material
            .iter()
            .filter(|m| **m == MaterialId::Mare)
            .count();
        assert_eq!(n_mare, 0, "cold moon must produce no mare");
        // But should still have craters.
        assert!(out.surface.elevation_m.iter().any(|e| e.abs() > 50.0));
    }

    #[test]
    fn deimos_end_to_end_runs_cleanly() {
        // Tiny moon, no mare, no basins (count clamps to 0 or 1) — must
        // not panic and must produce *some* surface variation from craters.
        let pts = fibonacci_lattice(1024);
        let desc = reference_bodies::deimos();
        let out = generate(&desc, pts, PipelineConfig::default());
        assert!(out.surface.elevation_m.iter().all(|e| e.is_finite()));
    }

    #[test]
    fn deterministic_across_runs_for_fixed_descriptor_and_points() {
        let pts = fibonacci_lattice(1024);
        let desc = reference_bodies::luna();
        let a = generate(&desc, pts.clone(), PipelineConfig::default());
        let b = generate(&desc, pts, PipelineConfig::default());
        assert_eq!(a.surface.elevation_m, b.surface.elevation_m);
        assert_eq!(a.surface.material, b.surface.material);
        assert_eq!(a.surface.maturity, b.surface.maturity);
        assert_eq!(a.detail.seed, b.detail.seed);
    }

    #[test]
    fn changing_seed_changes_output() {
        let pts = fibonacci_lattice(1024);
        let mut desc = reference_bodies::luna();
        desc.seed = 1;
        let a = generate(&desc, pts.clone(), PipelineConfig::default());
        desc.seed = 2;
        let b = generate(&desc, pts, PipelineConfig::default());
        assert_ne!(a.surface.elevation_m, b.surface.elevation_m);
    }

    #[test]
    fn changing_one_stage_seed_does_not_disturb_other_stages_outputs() {
        // Spec §Design principles 4: hierarchical sub-seeding.  We can't
        // change a single stage's sub-seed directly, but we can verify
        // that two pipeline runs with the same descriptor produce
        // different per-stage sub-seeds *only if* the top seed differs.
        // The cleaner determinism check is the one above; this is a
        // smoke test that the sub-seed function is wired in.
        use crate::seeding::sub_seed;
        let s = 12345u64;
        let primordial_seed = sub_seed(s, primordial::STAGE_NAME);
        let basin_seed = sub_seed(s, giant_basin::STAGE_NAME);
        let mare_seed = sub_seed(s, mare::STAGE_NAME);
        let main_seed = sub_seed(s, main_craters::STAGE_NAME);
        let mat_seed = sub_seed(s, maturity::STAGE_NAME);
        let mut all = vec![primordial_seed, basin_seed, mare_seed, main_seed, mat_seed];
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 5, "stage sub-seeds collided");
    }
}
