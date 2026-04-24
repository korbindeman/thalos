//! Icosahedral bucket spatial index for mid-frequency features on the sphere.
//!
//! Level-4 subdivision of the regular icosahedron produces 5120 triangular
//! cells (~15 km characteristic spacing on an 869 km body). Each cell stores
//! indices into the feature arrays for features whose influence radius
//! overlaps. Sampling iterates the cell containing a query direction plus its
//! neighbors (~18 cells total).
//!
//! Sized for Mira's 10 k-entry crater SSBO down to 500 m radius at SFD
//! slope −2: ~90 k features / 5120 cells ≈ 18 per cell, ~125 per fragment
//! including neighbors. Tractable for GPU fragment iteration.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::types::{Channel, Crater, Volcano};

/// Reference to a feature in one of the typed arrays on `BodyData`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureRef {
    Crater(u32),
    Volcano(u32),
    Channel(u32),
}

/// Icosahedral spatial index over sphere features.
///
/// Built once during `BodyBuilder::build()` from the populated feature arrays.
/// Immutable after construction — lives on `BodyData`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IcoBuckets {
    /// Cell vertex indices into `vertices` (3 per triangle).
    triangles: Vec<[u32; 3]>,
    /// Vertex positions on the unit sphere.
    vertices: Vec<Vec3>,
    /// Per-cell feature lists.  Indexed by triangle index.
    buckets: Vec<Vec<FeatureRef>>,
    /// Per-cell neighbor lists (cells sharing an edge or vertex).
    neighbors: Vec<Vec<u32>>,
}

impl IcoBuckets {
    /// Build an empty index (no features populated).
    pub fn empty() -> Self {
        let (vertices, triangles) = generate_icosphere(SUBDIVISION_LEVEL);
        let n = triangles.len();
        let neighbors = compute_neighbors(&triangles, vertices.len());
        Self {
            triangles,
            vertices,
            buckets: vec![Vec::new(); n],
            neighbors,
        }
    }

    /// Build the index from feature arrays.
    ///
    /// Each feature is inserted into every cell whose center is within
    /// the feature's influence angular radius on the unit sphere.
    pub fn build(
        craters: &[Crater],
        volcanoes: &[Volcano],
        channels: &[Channel],
        body_radius_m: f32,
    ) -> Self {
        let mut index = Self::empty();

        let cell_centers: Vec<Vec3> = index
            .triangles
            .iter()
            .map(|tri| {
                let a = index.vertices[tri[0] as usize];
                let b = index.vertices[tri[1] as usize];
                let c = index.vertices[tri[2] as usize];
                ((a + b + c) / 3.0).normalize()
            })
            .collect();

        // Find the cell whose center is closest to `dir`. Used to guarantee
        // each feature is registered in at least its home cell, even when the
        // feature's influence radius is smaller than the cell spacing
        // (~1.5° at subdivision 4). Without this, sub-cell-sized features
        // vanish from the index entirely.
        let home_cell = |dir: Vec3| -> usize {
            let mut best = 0usize;
            let mut best_dot = f32::NEG_INFINITY;
            for (idx, c) in cell_centers.iter().enumerate() {
                let d = dir.dot(*c);
                if d > best_dot {
                    best_dot = d;
                    best = idx;
                }
            }
            best
        };

        for (i, crater) in craters.iter().enumerate() {
            let dir = crater.center.normalize();
            let cos_angle = cos_angular_radius(crater.influence_radius_m(), body_radius_m);
            for (cell_idx, center) in cell_centers.iter().enumerate() {
                if dir.dot(*center) >= cos_angle {
                    index.buckets[cell_idx].push(FeatureRef::Crater(i as u32));
                }
            }
            let home = home_cell(dir);
            if !index.buckets[home].contains(&FeatureRef::Crater(i as u32)) {
                index.buckets[home].push(FeatureRef::Crater(i as u32));
            }
        }

        for (i, volcano) in volcanoes.iter().enumerate() {
            let dir = volcano.center.normalize();
            let cos_angle = cos_angular_radius(volcano.influence_radius_m(), body_radius_m);
            for (cell_idx, center) in cell_centers.iter().enumerate() {
                if dir.dot(*center) >= cos_angle {
                    index.buckets[cell_idx].push(FeatureRef::Volcano(i as u32));
                }
            }
            let home = home_cell(dir);
            if !index.buckets[home].contains(&FeatureRef::Volcano(i as u32)) {
                index.buckets[home].push(FeatureRef::Volcano(i as u32));
            }
        }

        for (i, channel) in channels.iter().enumerate() {
            // Insert channel into cells near any of its control points.
            let cos_angle = cos_angular_radius(channel.influence_radius_m(), body_radius_m);
            for point in &channel.points {
                let dir = point.normalize();
                for (cell_idx, center) in cell_centers.iter().enumerate() {
                    if dir.dot(*center) >= cos_angle
                        && !index.buckets[cell_idx].contains(&FeatureRef::Channel(i as u32))
                    {
                        index.buckets[cell_idx].push(FeatureRef::Channel(i as u32));
                    }
                }
                let home = home_cell(dir);
                if !index.buckets[home].contains(&FeatureRef::Channel(i as u32)) {
                    index.buckets[home].push(FeatureRef::Channel(i as u32));
                }
            }
        }

        index
    }

    /// Number of cells in the index.
    pub fn cell_count(&self) -> usize {
        self.triangles.len()
    }

    /// Read-only access to the per-cell feature buckets. Indexed by cell
    /// index (same order as `cell_count()`). Used by the renderer to flatten
    /// buckets into GPU SSBO format.
    pub fn buckets(&self) -> &[Vec<FeatureRef>] {
        &self.buckets
    }

    /// Find the cell index whose center is closest to `dir`.
    pub fn cell_for_dir(&self, dir: Vec3) -> usize {
        let dir = dir.normalize();
        let mut best = 0;
        let mut best_dot = f32::NEG_INFINITY;
        for (i, tri) in self.triangles.iter().enumerate() {
            let center = (self.vertices[tri[0] as usize]
                + self.vertices[tri[1] as usize]
                + self.vertices[tri[2] as usize])
                .normalize();
            let d = dir.dot(center);
            if d > best_dot {
                best_dot = d;
                best = i;
            }
        }
        best
    }

    /// Features in the cell containing `dir`.
    pub fn lookup(&self, dir: Vec3) -> &[FeatureRef] {
        &self.buckets[self.cell_for_dir(dir)]
    }

    /// Features in the cell containing `dir` plus all neighboring cells.
    pub fn lookup_with_neighbors(&self, dir: Vec3) -> impl Iterator<Item = FeatureRef> + '_ {
        let cell = self.cell_for_dir(dir);
        let center_iter = self.buckets[cell].iter().copied();
        let neighbor_iter = self.neighbors[cell]
            .iter()
            .flat_map(move |&n| self.buckets[n as usize].iter().copied());
        center_iter.chain(neighbor_iter)
    }
}

/// Subdivision level 4 = 5120 cells. Base icosahedron is level 0; each
/// subdivision quadruples triangle count. 20 × 4⁴ = 5120.
const SUBDIVISION_LEVEL: u32 = 4;

/// Cosine of the angular radius subtended by `linear_radius_m` on a
/// sphere of `body_radius_m`.  Used for overlap tests.
fn cos_angular_radius(linear_radius_m: f32, body_radius_m: f32) -> f32 {
    // angular_radius = linear_radius / body_radius (small-angle approx is
    // fine for features much smaller than the body, but we use cos for
    // correctness at all scales).
    (1.0 - 0.5 * (linear_radius_m / body_radius_m).powi(2)).max(-1.0)
}

use crate::icosphere::generate_icosphere;

/// For each triangle, compute the set of triangles that share at least one vertex.
fn compute_neighbors(triangles: &[[u32; 3]], num_vertices: usize) -> Vec<Vec<u32>> {
    // Build vertex → triangle adjacency.
    let mut vert_to_tris: Vec<Vec<u32>> = vec![Vec::new(); num_vertices];
    for (tri_idx, tri) in triangles.iter().enumerate() {
        for &v in tri {
            vert_to_tris[v as usize].push(tri_idx as u32);
        }
    }

    // For each triangle, neighbors = union of triangles sharing any vertex, minus self.
    let mut neighbors = Vec::with_capacity(triangles.len());
    for (tri_idx, tri) in triangles.iter().enumerate() {
        let mut nbrs = Vec::new();
        for &v in tri {
            for &other_tri in &vert_to_tris[v as usize] {
                if other_tri != tri_idx as u32 && !nbrs.contains(&other_tri) {
                    nbrs.push(other_tri);
                }
            }
        }
        neighbors.push(nbrs);
    }
    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icosphere_has_5120_cells() {
        let (verts, tris) = generate_icosphere(SUBDIVISION_LEVEL);
        assert_eq!(tris.len(), 5120); // 20 * 4^4
        assert_eq!(verts.len(), 2562); // V = 10 * 4^L + 2
    }

    #[test]
    fn empty_index_lookup() {
        let index = IcoBuckets::empty();
        assert!(index.cell_count() > 0);
        let features = index.lookup(Vec3::X);
        assert!(features.is_empty());
    }

    #[test]
    fn cell_for_opposite_dirs_differs() {
        let index = IcoBuckets::empty();
        let a = index.cell_for_dir(Vec3::X);
        let b = index.cell_for_dir(Vec3::NEG_X);
        assert_ne!(a, b);
    }
}
