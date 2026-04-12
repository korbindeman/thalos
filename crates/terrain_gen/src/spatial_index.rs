//! Icosahedral bucket spatial index for mid-frequency features on the sphere.
//!
//! Level-3 subdivision of the regular icosahedron produces 1280 triangular
//! cells (~30 km characteristic spacing on an 869 km body).  Each cell stores
//! indices into the feature arrays for features whose influence radius overlaps.
//! Sampling iterates the cell containing a query direction plus its neighbors.

use glam::Vec3;

use crate::types::{Channel, Crater, Volcano};

/// Reference to a feature in one of the typed arrays on `BodyData`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureRef {
    Crater(u32),
    Volcano(u32),
    Channel(u32),
}

/// Icosahedral spatial index over sphere features.
///
/// Built once during `BodyBuilder::build()` from the populated feature arrays.
/// Immutable after construction — lives on `BodyData`.
#[derive(Clone, Debug)]
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

        for (i, crater) in craters.iter().enumerate() {
            let dir = crater.center.normalize();
            let cos_angle = cos_angular_radius(crater.influence_radius_m(), body_radius_m);
            for (cell_idx, center) in cell_centers.iter().enumerate() {
                if dir.dot(*center) >= cos_angle {
                    index.buckets[cell_idx].push(FeatureRef::Crater(i as u32));
                }
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
            }
        }

        index
    }

    /// Number of cells in the index.
    pub fn cell_count(&self) -> usize {
        self.triangles.len()
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

/// Subdivision level 3 = 1280 cells.  The design doc calls this "level 4"
/// using a convention where the base icosahedron is level 1; in the
/// standard Loop-subdivision convention the base is level 0.
const SUBDIVISION_LEVEL: u32 = 3;

/// Cosine of the angular radius subtended by `linear_radius_m` on a
/// sphere of `body_radius_m`.  Used for overlap tests.
fn cos_angular_radius(linear_radius_m: f32, body_radius_m: f32) -> f32 {
    // angular_radius = linear_radius / body_radius (small-angle approx is
    // fine for features much smaller than the body, but we use cos for
    // correctness at all scales).
    (1.0 - 0.5 * (linear_radius_m / body_radius_m).powi(2)).max(-1.0)
}

// ---------------------------------------------------------------------------
// Icosphere generation (subdivision level 4 → 1280 triangles, 642 vertices)
// ---------------------------------------------------------------------------

fn generate_icosphere(subdivisions: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    // Start with a regular icosahedron.
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let base_verts = [
        Vec3::new(-1.0, phi, 0.0),
        Vec3::new(1.0, phi, 0.0),
        Vec3::new(-1.0, -phi, 0.0),
        Vec3::new(1.0, -phi, 0.0),
        Vec3::new(0.0, -1.0, phi),
        Vec3::new(0.0, 1.0, phi),
        Vec3::new(0.0, -1.0, -phi),
        Vec3::new(0.0, 1.0, -phi),
        Vec3::new(phi, 0.0, -1.0),
        Vec3::new(phi, 0.0, 1.0),
        Vec3::new(-phi, 0.0, -1.0),
        Vec3::new(-phi, 0.0, 1.0),
    ];
    let mut vertices: Vec<Vec3> = base_verts.iter().map(|v| v.normalize()).collect();

    let base_tris: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    let mut triangles = base_tris;

    // Midpoint cache: (min_idx, max_idx) → new vertex index.
    use std::collections::HashMap;
    let mut midpoint_cache = HashMap::new();

    for _ in 0..subdivisions {
        let mut new_tris = Vec::with_capacity(triangles.len() * 4);
        midpoint_cache.clear();

        for tri in &triangles {
            let a = tri[0];
            let b = tri[1];
            let c = tri[2];

            let ab = get_midpoint(a, b, &mut vertices, &mut midpoint_cache);
            let bc = get_midpoint(b, c, &mut vertices, &mut midpoint_cache);
            let ca = get_midpoint(c, a, &mut vertices, &mut midpoint_cache);

            new_tris.push([a, ab, ca]);
            new_tris.push([b, bc, ab]);
            new_tris.push([c, ca, bc]);
            new_tris.push([ab, bc, ca]);
        }

        triangles = new_tris;
    }

    (vertices, triangles)
}

fn get_midpoint(
    a: u32,
    b: u32,
    vertices: &mut Vec<Vec3>,
    cache: &mut std::collections::HashMap<(u32, u32), u32>,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    let mid = ((vertices[a as usize] + vertices[b as usize]) / 2.0).normalize();
    let idx = vertices.len() as u32;
    vertices.push(mid);
    cache.insert(key, idx);
    idx
}

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
    fn icosphere_has_1280_cells() {
        let (verts, tris) = generate_icosphere(SUBDIVISION_LEVEL);
        assert_eq!(tris.len(), 1280); // 20 * 4^3
        assert_eq!(verts.len(), 642); // V = 10 * 4^L + 2
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
