//! Per-cell scalar fields derived from the plate graph.
//!
//! The primary product is `boundary_distance_m`: the great-circle distance
//! along the mesh adjacency graph from each cell to the nearest cross-plate
//! edge. We also record the index of the nearest [`Boundary`] so a sample
//! can report the kind of the nearest scar without a second lookup.
//!
//! Implementation: multi-source Dijkstra (BFS with a binary heap), seeded
//! from every cell that participates in a boundary edge with distance 0.
//! Edge weights are great-circle distances in meters. For 2k cells with ~6
//! neighbors each, this finishes in well under a millisecond.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use super::boundaries::Boundary;
use super::mesh::SphericalMesh;

/// Per-cell distance and nearest-boundary fields.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TectonicFields {
    /// Distance from each cell center to the nearest plate boundary, along
    /// great-circle paths through the mesh adjacency graph. In meters.
    pub cell_boundary_distance_m: Vec<f32>,
    /// Index (into the `boundaries` slice passed to [`compute`]) of the
    /// nearest boundary edge for each cell. `None` only when the body has
    /// zero boundaries (single-plate edge case).
    pub cell_nearest_boundary: Vec<Option<u32>>,
}

/// Compute boundary-distance and nearest-boundary fields by multi-source
/// Dijkstra over the mesh adjacency graph.
pub fn compute(mesh: &SphericalMesh, boundaries: &[Boundary], radius_m: f32) -> TectonicFields {
    let n = mesh.cells.len();
    let mut dist = vec![f32::INFINITY; n];
    let mut nearest: Vec<Option<u32>> = vec![None; n];

    if boundaries.is_empty() {
        return TectonicFields {
            cell_boundary_distance_m: dist,
            cell_nearest_boundary: nearest,
        };
    }

    // Seed every cell that participates in a boundary with distance 0.
    // If a cell touches multiple boundaries, prefer the lower-magnitude
    // one only by edge order — Dijkstra picks the first one to settle and
    // any ties resolve later naturally. Tie-break behavior isn't visible
    // in the equirect because every nearby cell shares the same kind.
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    for (b_idx, b) in boundaries.iter().enumerate() {
        for &cell in &[b.cell_a, b.cell_b] {
            if dist[cell as usize] > 0.0 {
                dist[cell as usize] = 0.0;
                nearest[cell as usize] = Some(b_idx as u32);
                heap.push(HeapEntry {
                    distance: 0.0,
                    cell,
                });
            }
        }
    }

    while let Some(HeapEntry { distance, cell }) = heap.pop() {
        if distance > dist[cell as usize] {
            continue; // stale entry
        }
        let nearest_b = nearest[cell as usize];
        let cell_dir = mesh.cells[cell as usize];
        for &neighbor in &mesh.neighbors[cell as usize] {
            // Great-circle distance between two unit vectors:
            // `radius * acos(clamp(a·b, -1, 1))`.
            let n_dir = mesh.cells[neighbor as usize];
            let cosine = cell_dir.dot(n_dir).clamp(-1.0, 1.0);
            let edge_len = radius_m * cosine.acos();
            let alt = distance + edge_len;
            if alt < dist[neighbor as usize] {
                dist[neighbor as usize] = alt;
                nearest[neighbor as usize] = nearest_b;
                heap.push(HeapEntry {
                    distance: alt,
                    cell: neighbor,
                });
            }
        }
    }

    TectonicFields {
        cell_boundary_distance_m: dist,
        cell_nearest_boundary: nearest,
    }
}

/// Min-heap by distance: `BinaryHeap` is a max-heap, so we invert the
/// comparator. NaN is treated as max distance to keep the order total.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HeapEntry {
    distance: f32,
    cell: u32,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so smaller distance = higher priority. NaN sinks to bottom.
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.cell.cmp(&other.cell))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonics::boundaries::{BoundaryKind, classify_boundaries};
    use crate::tectonics::config::{TectonicActivity, TectonicConfig};
    use crate::tectonics::plates::{PlateId, assign_plates};

    #[test]
    fn boundary_cells_have_zero_distance() {
        let mesh = SphericalMesh::build(256, 1);
        let cfg = TectonicConfig {
            plate_count: 6,
            mesh_cells: 256,
            activity: TectonicActivity::Active,
            continental_fraction: 0.30,
            seed: 7,
            seed_dirs: None,
            continental_clustering: 0.0,
            equatorial_bias: 0.0,
            primary_size_multiplier: 1.0,
        };
        let asgn = assign_plates(&mesh, &cfg, 42);
        let boundaries = classify_boundaries(&mesh, &asgn, 6.4e6);
        let fields = compute(&mesh, &boundaries, 6.4e6);

        // Every cell touching any boundary must be at distance 0.
        for b in &boundaries {
            assert_eq!(fields.cell_boundary_distance_m[b.cell_a as usize], 0.0);
            assert_eq!(fields.cell_boundary_distance_m[b.cell_b as usize], 0.0);
        }
    }

    #[test]
    fn distances_are_finite_when_boundaries_exist() {
        let mesh = SphericalMesh::build(256, 1);
        let cfg = TectonicConfig {
            plate_count: 6,
            mesh_cells: 256,
            activity: TectonicActivity::Active,
            continental_fraction: 0.30,
            seed: 7,
            seed_dirs: None,
            continental_clustering: 0.0,
            equatorial_bias: 0.0,
            primary_size_multiplier: 1.0,
        };
        let asgn = assign_plates(&mesh, &cfg, 42);
        let boundaries = classify_boundaries(&mesh, &asgn, 6.4e6);
        assert!(
            !boundaries.is_empty(),
            "test premise: 6 plates should produce boundaries"
        );
        let fields = compute(&mesh, &boundaries, 6.4e6);
        for &d in &fields.cell_boundary_distance_m {
            assert!(
                d.is_finite(),
                "infinite cell distance with non-empty boundaries"
            );
        }
    }

    #[test]
    fn nearest_boundary_kind_is_recoverable() {
        let mesh = SphericalMesh::build(256, 1);
        let cfg = TectonicConfig {
            plate_count: 6,
            mesh_cells: 256,
            activity: TectonicActivity::Active,
            continental_fraction: 0.30,
            seed: 7,
            seed_dirs: None,
            continental_clustering: 0.0,
            equatorial_bias: 0.0,
            primary_size_multiplier: 1.0,
        };
        let asgn = assign_plates(&mesh, &cfg, 42);
        let boundaries = classify_boundaries(&mesh, &asgn, 6.4e6);
        let fields = compute(&mesh, &boundaries, 6.4e6);
        // Every cell should resolve to *some* boundary index.
        for &b in &fields.cell_nearest_boundary {
            assert!(b.is_some());
            let kind = boundaries[b.unwrap() as usize].kind;
            assert!(matches!(
                kind,
                BoundaryKind::Convergent | BoundaryKind::Divergent | BoundaryKind::Transform
            ));
        }
        // Silence unused warning when PlateId not consumed.
        let _ = PlateId(0);
    }
}
