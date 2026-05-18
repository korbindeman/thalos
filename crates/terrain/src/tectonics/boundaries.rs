//! Plate boundaries: edge enumeration and convergent/divergent/transform
//! classification from plate Euler-pole motion at the boundary midpoint.
//!
//! Classification math (regardless of activity mode):
//!
//! 1. At the boundary midpoint `m` (unit-direction average of the two cell
//!    centers), compute each plate's surface velocity via `omega × r`. The
//!    result lies in the tangent plane at `m`.
//! 2. Take the boundary normal `n` as the tangent component of `cell_b -
//!    cell_a`, normalized. (Tangent component because both vectors live on
//!    the unit sphere; their difference has a small radial component we
//!    project out.)
//! 3. Compute `v_rel = v_b - v_a`. Decompose into:
//!    - `normal_comp = v_rel · n` (signed; positive = plates moving apart)
//!    - `tangent_comp = v_rel - normal_comp · n` (vector)
//! 4. Classify:
//!    - If `|normal_comp| < |tangent_comp|`: **Transform**. Magnitude is
//!      `|tangent_comp|`.
//!    - Else if `normal_comp < 0`: **Convergent**. Magnitude is
//!      `|normal_comp|`.
//!    - Else: **Divergent**. Magnitude is `normal_comp`.
//!
//! Boundary classification reads the encoded Euler poles regardless of
//! activity mode — it is the structural signature of past or present motion,
//! and that is what the editor and downstream height contributions need to
//! see for "stagnant lid" planets where the scars are visible but no
//! velocity is.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use super::mesh::SphericalMesh;
use super::plates::{PlateAssignment, PlateId, raw_surface_velocity};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub enum BoundaryKind {
    Convergent,
    Divergent,
    Transform,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Boundary {
    pub cell_a: u32,
    pub cell_b: u32,
    pub plate_a: PlateId,
    pub plate_b: PlateId,
    pub kind: BoundaryKind,
    /// Magnitude of the relative-velocity component classified, in m/yr.
    pub magnitude_m_per_yr: f32,
    /// Midpoint direction on the unit sphere — average of the two cell
    /// centers, normalized.
    pub midpoint_dir: Vec3,
}

/// Enumerate every cross-plate edge in the mesh and classify it.
///
/// Returns one [`Boundary`] per unordered cell pair `(a, b)` where `a < b`
/// and `cell_plate[a] != cell_plate[b]`.
pub fn classify_boundaries(
    mesh: &SphericalMesh,
    assignment: &PlateAssignment,
    radius_m: f32,
) -> Vec<Boundary> {
    let mut out = Vec::new();
    for a in 0..mesh.cells.len() as u32 {
        let pa = assignment.cell_plate[a as usize];
        for &b in &mesh.neighbors[a as usize] {
            if b <= a {
                continue;
            }
            let pb = assignment.cell_plate[b as usize];
            if pa == pb {
                continue;
            }
            let dir_a = mesh.cells[a as usize];
            let dir_b = mesh.cells[b as usize];
            let midpoint = (dir_a + dir_b).normalize_or_zero();
            if midpoint == Vec3::ZERO {
                continue;
            }
            let plate_a = &assignment.plates[pa.0 as usize];
            let plate_b = &assignment.plates[pb.0 as usize];
            let v_a = raw_surface_velocity(plate_a, midpoint, radius_m);
            let v_b = raw_surface_velocity(plate_b, midpoint, radius_m);
            let (kind, magnitude) = classify(midpoint, dir_a, dir_b, v_a, v_b);
            out.push(Boundary {
                cell_a: a,
                cell_b: b,
                plate_a: pa,
                plate_b: pb,
                kind,
                magnitude_m_per_yr: magnitude,
                midpoint_dir: midpoint,
            });
        }
    }
    out
}

/// Pure classification helper. Public for tests.
pub fn classify(
    midpoint: Vec3,
    dir_a: Vec3,
    dir_b: Vec3,
    v_a: Vec3,
    v_b: Vec3,
) -> (BoundaryKind, f32) {
    let raw = dir_b - dir_a;
    // Project out the radial component at the midpoint.
    let radial = midpoint * raw.dot(midpoint);
    let normal = (raw - radial).normalize_or_zero();
    if normal == Vec3::ZERO {
        // Degenerate (cells diametrically opposite or identical); treat as
        // a transform boundary with zero magnitude. Won't appear in practice
        // because adjacent mesh cells are always close.
        return (BoundaryKind::Transform, 0.0);
    }
    let v_rel = v_b - v_a;
    let normal_comp = v_rel.dot(normal);
    let tangent_vec = v_rel - normal * normal_comp;
    let tangent_mag = tangent_vec.length();
    if normal_comp.abs() < tangent_mag {
        (BoundaryKind::Transform, tangent_mag)
    } else if normal_comp < 0.0 {
        (BoundaryKind::Convergent, -normal_comp)
    } else {
        (BoundaryKind::Divergent, normal_comp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonics::config::{TectonicActivity, TectonicConfig};
    use crate::tectonics::mesh::SphericalMesh;
    use crate::tectonics::plates::{Plate, PlateId, PlateKind, assign_plates};

    /// Two synthetic plates whose Euler poles produce purely convergent
    /// motion at a specific midpoint should classify as Convergent.
    #[test]
    fn convergent_motion_classifies_convergent() {
        // Place the boundary at +Z. Plate A is to the +X side, plate B to
        // the -X side. We want A moving in -X and B moving in +X (toward
        // each other) at the midpoint.
        let radius = 6.4e6;
        // For motion in -X at point +Z: omega = +Y gives v = omega×r =
        // Y×Z = X (positive X). We want -X, so omega = -Y.
        let plate_a = Plate {
            id: PlateId(0),
            seed_cell: 0,
            centroid_dir: Vec3::new(1.0, 0.0, 0.5).normalize(),
            kind: PlateKind::Oceanic,
            euler_pole: -Vec3::Y,
            omega_rad_per_year: 1.0e-8,
        };
        let plate_b = Plate {
            id: PlateId(1),
            seed_cell: 1,
            centroid_dir: Vec3::new(-1.0, 0.0, 0.5).normalize(),
            kind: PlateKind::Oceanic,
            euler_pole: Vec3::Y,
            omega_rad_per_year: 1.0e-8,
        };
        let midpoint = Vec3::Z;
        let dir_a = Vec3::new(0.5, 0.0, 1.0).normalize();
        let dir_b = Vec3::new(-0.5, 0.0, 1.0).normalize();
        let v_a = raw_surface_velocity(&plate_a, midpoint, radius);
        let v_b = raw_surface_velocity(&plate_b, midpoint, radius);
        let (kind, mag) = classify(midpoint, dir_a, dir_b, v_a, v_b);
        assert_eq!(
            kind,
            BoundaryKind::Convergent,
            "expected convergent (v_a={v_a:?}, v_b={v_b:?})"
        );
        assert!(mag > 0.0);
    }

    /// Reversed Euler poles produce divergent motion.
    #[test]
    fn divergent_motion_classifies_divergent() {
        let radius = 6.4e6;
        let plate_a = Plate {
            id: PlateId(0),
            seed_cell: 0,
            centroid_dir: Vec3::new(1.0, 0.0, 0.5).normalize(),
            kind: PlateKind::Oceanic,
            euler_pole: Vec3::Y,
            omega_rad_per_year: 1.0e-8,
        };
        let plate_b = Plate {
            id: PlateId(1),
            seed_cell: 1,
            centroid_dir: Vec3::new(-1.0, 0.0, 0.5).normalize(),
            kind: PlateKind::Oceanic,
            euler_pole: -Vec3::Y,
            omega_rad_per_year: 1.0e-8,
        };
        let midpoint = Vec3::Z;
        let dir_a = Vec3::new(0.5, 0.0, 1.0).normalize();
        let dir_b = Vec3::new(-0.5, 0.0, 1.0).normalize();
        let v_a = raw_surface_velocity(&plate_a, midpoint, radius);
        let v_b = raw_surface_velocity(&plate_b, midpoint, radius);
        let (kind, mag) = classify(midpoint, dir_a, dir_b, v_a, v_b);
        assert_eq!(
            kind,
            BoundaryKind::Divergent,
            "expected divergent (v_a={v_a:?}, v_b={v_b:?})"
        );
        assert!(mag > 0.0);
    }

    /// Equal Euler poles in the same direction → no relative motion → tangent
    /// magnitude tiny, normal component tiny; falls into the transform branch
    /// (because |normal| < |tangent| ≈ 0). Magnitude near zero either way.
    #[test]
    fn no_relative_motion_classifies_with_zero_magnitude() {
        let radius = 6.4e6;
        let plate_a = Plate {
            id: PlateId(0),
            seed_cell: 0,
            centroid_dir: Vec3::Z,
            kind: PlateKind::Oceanic,
            euler_pole: Vec3::Y,
            omega_rad_per_year: 1.0e-8,
        };
        let plate_b = plate_a.clone();
        let midpoint = Vec3::Z;
        let dir_a = Vec3::new(0.1, 0.0, 1.0).normalize();
        let dir_b = Vec3::new(-0.1, 0.0, 1.0).normalize();
        let v_a = raw_surface_velocity(&plate_a, midpoint, radius);
        let v_b = raw_surface_velocity(&plate_b, midpoint, radius);
        let (_, mag) = classify(midpoint, dir_a, dir_b, v_a, v_b);
        assert!(mag < 1e-3, "expected near-zero magnitude, got {mag}");
    }

    #[test]
    fn boundary_enumeration_skips_within_plate_edges() {
        let mesh = SphericalMesh::build(256, 1);
        let cfg = TectonicConfig {
            plate_count: 4,
            mesh_cells: 256,
            activity: TectonicActivity::Active,
            continental_fraction: 0.5,
            seed: 7,
            seed_dirs: None,
            continental_clustering: 0.0,
            equatorial_bias: 0.0,
            primary_size_multiplier: 1.0,
        };
        let asgn = assign_plates(&mesh, &cfg, 42);
        let boundaries = classify_boundaries(&mesh, &asgn, 6.4e6);
        for b in &boundaries {
            assert_ne!(b.plate_a, b.plate_b, "within-plate edge in boundary list");
            assert!(b.cell_a < b.cell_b, "boundaries should be canonicalized");
        }
    }
}
