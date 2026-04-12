//! Per-sample surface state shared by every pipeline stage.
//!
//! The generator operates on arbitrary points on the unit sphere — stages
//! are pure functions from (previous state, derived properties, sub-seed) to
//! next state.  Nothing here assumes a particular sampling layout (grid,
//! cubesphere face, icosahedral face, Fibonacci lattice); whatever the
//! caller hands in is what gets written.
//!
//! `SurfaceState` is struct-of-arrays for cache locality and so stage
//! functions can borrow individual fields without lifetime gymnastics.

use glam::DVec3;

/// Semantic material identifier.  Stages write tags; mapping to colors is
/// the consumer's job (per spec §"What v0.1 explicitly does not do").
///
/// This list is intentionally short for v0.1.  New materials are added by
/// extension, not by replacing existing variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialId {
    /// Highland-dominated primordial crust (anorthosite-like for
    /// silicate-rich bodies, ice for ice-dominated bodies, etc.).
    PrimordialCrust,
    /// Exposed lower crust from a giant basin excavation.
    BasinFloor,
    /// Mare basalt or compositionally-appropriate dark volcanic fill.
    Mare,
    /// Fresh impact-excavated bedrock; will blend toward the surrounding
    /// baseline in the maturity pass.
    FreshExcavation,
    /// Fresh ejecta blanket from an impact.
    FreshEjecta,
}

/// Mutable per-sample surface state.
///
/// Elevation is measured from `reference_radius_m`; oblateness is applied to
/// the reference sphere itself, not to elevation, so the two concerns stay
/// orthogonal.  Material / maturity / crater age are auxiliary fields the
/// later stages and consumers read.
#[derive(Clone, Debug)]
pub struct SurfaceState {
    /// Unit vectors from body centre, one per sample.  Immutable after
    /// construction — stages write into the other fields in parallel with
    /// these indices.
    pub points: Vec<DVec3>,
    /// Reference sphere radius in metres.  Stage 0 initial elevation is zero;
    /// subsequent stages add/subtract relative to this.
    pub reference_radius_m: f64,
    /// Elevation above reference radius, metres.
    pub elevation_m: Vec<f64>,
    /// Per-sample material tag.
    pub material: Vec<MaterialId>,
    /// Regolith/weathering maturity.  0.0 = fresh (bright), 1.0 = weathered
    /// (dark).  Stage 0 initialises to 1.0 per spec ("old, weathered" baseline).
    pub maturity: Vec<f64>,
    /// Age of the most recent impact that touched this sample, in
    /// gigayears-before-present.  `f64::NAN` means "no recorded impact"
    /// (pristine baseline).  The maturity stage reads this to blend fresh
    /// excavation toward baseline.
    pub crater_age_gyr: Vec<f64>,
}

impl SurfaceState {
    /// Stage 0 — reference shape initialisation.  Every sample starts at
    /// elevation zero on a sphere of the given reference radius, with
    /// default material and baseline weathered maturity.
    pub fn new(points: Vec<DVec3>, reference_radius_m: f64) -> Self {
        let n = points.len();
        Self {
            points,
            reference_radius_m,
            elevation_m: vec![0.0; n],
            material: vec![MaterialId::PrimordialCrust; n],
            maturity: vec![1.0; n],
            crater_age_gyr: vec![f64::NAN; n],
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::fibonacci_lattice;

    #[test]
    fn new_initialises_all_fields_to_baseline() {
        let points = fibonacci_lattice(64);
        let state = SurfaceState::new(points, 1_737_400.0);
        assert_eq!(state.len(), 64);
        assert!(state.elevation_m.iter().all(|&e| e == 0.0));
        assert!(state.material.iter().all(|&m| m == MaterialId::PrimordialCrust));
        assert!(state.maturity.iter().all(|&m| m == 1.0));
        assert!(state.crater_age_gyr.iter().all(|&a| a.is_nan()));
    }
}
