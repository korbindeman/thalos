//! Roughly uniform spherical mesh used as the substrate for plate assignment
//! and boundary classification.
//!
//! The mesh is built from a jittered Fibonacci sphere (cheap, deterministic,
//! no spherical Delaunay needed). Adjacency is symmetric K-nearest-neighbor:
//! for each cell we collect the K closest by direction, then symmetrize via
//! union (`A ↔ B  ⇔  A ∈ KNN(B) ∨ B ∈ KNN(A)`). This gives every BFS/Dijkstra
//! consumer a graph with two-way edges, and produces visibly clean plate
//! boundaries for the cell counts we use (~1k–10k). A proper spherical
//! Delaunay via 3D convex hull would be the principled answer if union ever
//! produces visible artifacts at boundaries; defer until we see them.
//!
//! Determinism: every random quantity (jitter direction, jitter magnitude
//! envelope) is drawn from a [`crate::seeding::Rng`] seeded by the caller.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::seeding::Rng;

/// Number of K-nearest-neighbors collected per cell before symmetrization.
/// 8 is generous enough for Voronoi-like coverage of every neighbor on the
/// sphere at the cell counts we use; smaller values risk holes in the graph.
const KNN_K: usize = 8;

/// Cells are unit vectors on the sphere; adjacency is per-cell list of
/// neighbor cell indices.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SphericalMesh {
    /// Cell center directions, unit length.
    pub cells: Vec<Vec3>,
    /// `neighbors[i]` is the list of cell indices adjacent to cell `i`.
    /// Symmetric: `j ∈ neighbors[i]  ⇔  i ∈ neighbors[j]`. Sorted ascending
    /// for stable iteration.
    pub neighbors: Vec<Vec<u32>>,
}

impl SphericalMesh {
    /// Build a mesh of `n_cells` points using a jittered Fibonacci sphere.
    /// Adjacency is symmetric K-nearest-neighbor with K=8.
    pub fn build(n_cells: u32, seed: u64) -> Self {
        let n = n_cells.max(4) as usize;
        let mut rng = Rng::new(seed);

        // Fibonacci sphere with per-point jitter. Jitter amplitude is half
        // the average cell pitch (`~ sqrt(4/n)` for n points on unit sphere)
        // so points stay inside their nominal Voronoi cell most of the time.
        let golden = (1.0 + 5.0_f32.sqrt()) / 2.0;
        let jitter_amp = 0.5 * (4.0 / n as f32).sqrt();
        let mut cells: Vec<Vec3> = Vec::with_capacity(n);
        for i in 0..n {
            let y = 1.0 - (i as f32 + 0.5) / n as f32 * 2.0;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let theta = std::f32::consts::TAU * i as f32 / golden;
            let (s, c) = theta.sin_cos();
            let base = Vec3::new(c * r, y, s * r);
            // Jitter in a random tangent direction by an amount drawn from
            // [-jitter_amp, +jitter_amp]. Project back to the sphere.
            let j = rng.unit_vector();
            let jitter_dir = (Vec3::new(j.x as f32, j.y as f32, j.z as f32)
                - base * Vec3::new(j.x as f32, j.y as f32, j.z as f32).dot(base))
            .normalize_or_zero();
            let jitter_mag = (rng.next_f64_signed() as f32) * jitter_amp;
            cells.push((base + jitter_dir * jitter_mag).normalize());
        }

        let neighbors = build_symmetric_knn(&cells, KNN_K);
        Self { cells, neighbors }
    }

    /// Returns the index of the cell whose center has the largest dot product
    /// with `dir` (i.e. the closest cell on the sphere). Brute force; fine
    /// for the cell counts we run at and acceptable for cubemap baking.
    pub fn nearest(&self, dir: Vec3) -> u32 {
        let mut best = 0u32;
        let mut best_dot = f32::NEG_INFINITY;
        for (i, &c) in self.cells.iter().enumerate() {
            let d = c.dot(dir);
            if d > best_dot {
                best_dot = d;
                best = i as u32;
            }
        }
        best
    }
}

/// K-nearest-neighbor graph, then symmetrize via union. Returns per-cell
/// neighbor lists sorted ascending.
fn build_symmetric_knn(cells: &[Vec3], k: usize) -> Vec<Vec<u32>> {
    let n = cells.len();
    let mut directed: Vec<Vec<u32>> = vec![Vec::with_capacity(k); n];
    for i in 0..n {
        // Top-K by dot product, excluding self. Insertion-sort into a small
        // ordered list — O(N·K) per cell, fine for N ≤ 10k.
        let mut top: Vec<(f32, u32)> = Vec::with_capacity(k);
        for j in 0..n {
            if j == i {
                continue;
            }
            let d = cells[i].dot(cells[j]);
            if top.len() < k {
                top.push((d, j as u32));
                top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            } else if d > top[k - 1].0 {
                top[k - 1] = (d, j as u32);
                top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        directed[i] = top.into_iter().map(|(_, idx)| idx).collect();
    }

    // Union symmetrize: for every directed edge i → j, add j → i to
    // `directed[j]` if missing. Use Vec contains() rather than HashSet to
    // keep iteration order deterministic.
    let mut neighbors = directed.clone();
    for (i, directed_edges) in directed.iter().enumerate().take(n) {
        for &j in directed_edges {
            let row = &mut neighbors[j as usize];
            if !row.contains(&(i as u32)) {
                row.push(i as u32);
            }
        }
    }
    for row in &mut neighbors {
        row.sort_unstable();
    }
    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_size_matches_request() {
        let mesh = SphericalMesh::build(256, 42);
        assert_eq!(mesh.cells.len(), 256);
        assert_eq!(mesh.neighbors.len(), 256);
    }

    #[test]
    fn cells_are_unit_length() {
        let mesh = SphericalMesh::build(256, 42);
        for c in &mesh.cells {
            assert!((c.length() - 1.0).abs() < 1e-5, "cell not unit length: {c}");
        }
    }

    #[test]
    fn adjacency_is_symmetric() {
        let mesh = SphericalMesh::build(512, 99);
        for (i, row) in mesh.neighbors.iter().enumerate() {
            for &j in row {
                assert!(
                    mesh.neighbors[j as usize].contains(&(i as u32)),
                    "asymmetric edge {i} → {j}"
                );
            }
        }
    }

    #[test]
    fn every_cell_has_neighbors() {
        let mesh = SphericalMesh::build(512, 7);
        for (i, row) in mesh.neighbors.iter().enumerate() {
            assert!(!row.is_empty(), "cell {i} has no neighbors");
        }
    }

    #[test]
    fn same_seed_produces_byte_identical_mesh() {
        let a = SphericalMesh::build(256, 12345);
        let b = SphericalMesh::build(256, 12345);
        assert_eq!(a.cells, b.cells);
        assert_eq!(a.neighbors, b.neighbors);
    }

    #[test]
    fn nearest_recovers_seed_directions() {
        // A cell is necessarily the nearest cell to its own direction.
        let mesh = SphericalMesh::build(256, 1);
        for (i, &c) in mesh.cells.iter().enumerate() {
            let nearest = mesh.nearest(c);
            assert_eq!(nearest as usize, i, "cell {i} not nearest to itself");
        }
    }
}
