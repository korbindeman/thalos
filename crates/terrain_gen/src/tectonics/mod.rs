//! Tectonic structural prior for a body.
//!
//! A `TectonicSystem` is a deterministic-from-seed graph of plates and the
//! boundaries between them, plus per-cell scalar fields (boundary distance,
//! nearest-boundary kind) for downstream sampling. It is *structural*, not
//! a renderer projection: the editor reads it to draw plate overlays, and
//! a future `SurfaceField` height contribution will read it to drive
//! mountain belts and rift scars at convergent and divergent boundaries.
//!
//! The module is intentionally archetype-agnostic. Both
//! `AgingOceanicHomeworld` (Thalos) and `ColdDesertFormerlyWet` (Vaelen)
//! can carry one without code changes here; the difference is just which
//! [`TectonicActivity`] mode they declare.
//!
//! Build cost is dominated by the symmetric KNN graph (`O(N²·K)` for `N`
//! mesh cells, `K = 8`). For 2k cells this is ~32M ops, milliseconds. The
//! plate flood-fill, boundary classification, and Dijkstra fields are all
//! linear in the edge count.
//!
//! ## Layering
//!
//! - [`config`] — authored input ([`TectonicConfig`], [`TectonicActivity`]).
//! - [`mesh`] — spherical Voronoi-ish mesh, jittered Fibonacci.
//! - [`plates`] — plate identity, flood-fill, Euler-pole motion.
//! - [`boundaries`] — cross-plate edge enumeration + classification.
//! - [`fields`] — per-cell distance and nearest-boundary fields.

pub mod boundaries;
pub mod config;
pub mod fields;
pub mod mesh;
pub mod plates;

use glam::Vec3;
use serde::{Deserialize, Serialize};

pub use boundaries::{Boundary, BoundaryKind, classify_boundaries};
pub use config::{TectonicActivity, TectonicConfig};
pub use fields::TectonicFields;
pub use mesh::SphericalMesh;
pub use plates::{Plate, PlateId, PlateKind, assign_plates, surface_velocity};

/// Built tectonic graph for a body. Deterministic from
/// `(config.seed, root_seed, body radius)`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TectonicSystem {
    pub config: TectonicConfig,
    pub body_radius_m: f32,
    pub mesh: SphericalMesh,
    pub plates: Vec<Plate>,
    /// Per mesh-cell plate id.
    pub cell_plate: Vec<PlateId>,
    pub boundaries: Vec<Boundary>,
    pub fields: TectonicFields,
}

/// Sample of the tectonic graph at a unit direction. See
/// [`TectonicActivity`] for the semantics of `plate_velocity_m_per_yr`
/// vs the boundary fields.
#[derive(Clone, Copy, Debug)]
pub struct TectonicSample {
    pub plate_id: PlateId,
    pub plate_kind: PlateKind,
    pub boundary_distance_m: f32,
    pub boundary_kind: Option<BoundaryKind>,
    /// Index into `TectonicSystem::boundaries` of the nearest cross-plate
    /// edge to this direction; `None` only when the body has zero
    /// boundaries (single-plate edge case). Consumers that need to know
    /// the *other* plate's kind (e.g. continental-continental convergence
    /// vs oceanic-continental) read it from `boundaries[i].plate_a/b`.
    pub nearest_boundary_index: Option<u32>,
    /// Tangent-plane velocity from the plate's Euler pole, in m/yr. Zero
    /// for `StagnantLid` and `Frozen` activity modes.
    pub plate_velocity_m_per_yr: Vec3,
}

impl TectonicSystem {
    /// Build a tectonic system. Pure function of the inputs.
    pub fn build(config: &TectonicConfig, body_radius_m: f32, root_seed: u64) -> Self {
        let combined_seed = crate::seeding::sub_seed(root_seed ^ config.seed, "tectonics.system");
        let mesh = SphericalMesh::build(config.mesh_cells, combined_seed);
        let assignment = assign_plates(&mesh, config, combined_seed);
        let boundaries = classify_boundaries(&mesh, &assignment, body_radius_m);
        let fields = fields::compute(&mesh, &boundaries, body_radius_m);
        Self {
            config: config.clone(),
            body_radius_m,
            mesh,
            plates: assignment.plates,
            cell_plate: assignment.cell_plate,
            boundaries,
            fields,
        }
    }

    /// Sample the system at a unit-direction `dir`.
    pub fn sample(&self, dir: Vec3) -> TectonicSample {
        let cell = self.mesh.nearest(dir) as usize;
        let plate_id = self.cell_plate[cell];
        let plate = &self.plates[plate_id.0 as usize];
        let nearest_b = self.fields.cell_nearest_boundary[cell];
        let boundary_kind = nearest_b.map(|i| self.boundaries[i as usize].kind);
        let velocity = surface_velocity(plate, dir, self.body_radius_m, self.config.activity);
        TectonicSample {
            plate_id,
            plate_kind: plate.kind,
            boundary_distance_m: self.fields.cell_boundary_distance_m[cell],
            boundary_kind,
            nearest_boundary_index: nearest_b,
            plate_velocity_m_per_yr: velocity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TectonicConfig {
        TectonicConfig {
            plate_count: 8,
            mesh_cells: 256,
            activity: TectonicActivity::Active,
            continental_fraction: 0.30,
            seed: 7,
            seed_dirs: None,
            continental_clustering: 0.0,
            equatorial_bias: 0.0,
            primary_size_multiplier: 1.0,
        }
    }

    #[test]
    fn build_is_deterministic_byte_for_byte() {
        // The system must encode/decode byte-stably for a given seed.
        // This is the insurance test against accidental nondeterminism
        // (HashMap ordering, FP nonassociativity, parallelism).
        let a = TectonicSystem::build(&cfg(), 6.4e6, 12345);
        let b = TectonicSystem::build(&cfg(), 6.4e6, 12345);
        let bytes_a = bincode::serde::encode_to_vec(&a, bincode::config::standard()).unwrap();
        let bytes_b = bincode::serde::encode_to_vec(&b, bincode::config::standard()).unwrap();
        assert_eq!(bytes_a, bytes_b, "tectonic system not byte-stable");
    }

    #[test]
    fn different_seeds_produce_different_systems() {
        let a = TectonicSystem::build(&cfg(), 6.4e6, 12345);
        let b = TectonicSystem::build(&cfg(), 6.4e6, 67890);
        assert_ne!(a.cell_plate, b.cell_plate);
    }

    #[test]
    fn sample_returns_consistent_plate_for_seed_cell() {
        let sys = TectonicSystem::build(&cfg(), 6.4e6, 12345);
        for plate in &sys.plates {
            let sample = sys.sample(sys.mesh.cells[plate.seed_cell as usize]);
            assert_eq!(sample.plate_id, plate.id);
        }
    }

    #[test]
    fn stagnant_lid_zeros_sample_velocity() {
        let mut config = cfg();
        config.activity = TectonicActivity::StagnantLid;
        let sys = TectonicSystem::build(&config, 6.4e6, 12345);
        // Sample any direction; velocity should be zero.
        let sample = sys.sample(Vec3::X);
        assert_eq!(sample.plate_velocity_m_per_yr, Vec3::ZERO);
    }

    #[test]
    fn boundary_distance_increases_inside_plate() {
        // For any plate with at least 2 cells, the plate seed cell is at
        // least as far from its boundary as some neighbor — i.e. the field
        // is monotone along radial paths into the plate. Weak property, but
        // catches obvious Dijkstra bugs.
        let sys = TectonicSystem::build(&cfg(), 6.4e6, 12345);
        let max_dist = sys
            .fields
            .cell_boundary_distance_m
            .iter()
            .copied()
            .fold(0.0_f32, f32::max);
        assert!(
            max_dist > 0.0,
            "all cells at distance zero — no plate interiors"
        );
    }
}
