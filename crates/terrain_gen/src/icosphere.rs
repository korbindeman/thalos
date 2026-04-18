//! Icosphere mesh — vertex-subdivided icosahedron on the unit sphere.
//!
//! Owns sphere-topology scaffolding used by multiple stages: the existing
//! `IcoBuckets` spatial index (`spatial_index`) uses the triangle mesh,
//! while DLA-style ridge generation (`stages::orogen_dla`) walks the dual
//! graph (vertex positions + per-vertex adjacency).
//!
//! Subdivision level `L` yields `V = 10·4^L + 2` vertices and `T = 20·4^L`
//! triangles. Characteristic spacing on a unit sphere is roughly
//! `sqrt(4π / V)` radians — ~1.0° at `L=6` (40,962 vertices), or ~55 km on
//! a Thalos-sized (3186 km) body.

use glam::Vec3;
use std::collections::HashMap;

/// Icosphere mesh: positions + triangles + per-vertex neighbor lists.
#[derive(Clone, Debug)]
pub struct Icosphere {
    /// Unit-sphere vertex positions.
    pub vertices: Vec<Vec3>,
    /// Triangle indices into `vertices`.
    pub triangles: Vec<[u32; 3]>,
    /// Per-vertex neighbor indices (sorted, deduplicated). Each vertex has
    /// 5 neighbors (at the 12 original icosahedron corners) or 6 (every
    /// vertex introduced by subdivision) — standard icosphere property.
    pub vertex_neighbors: Vec<Vec<u32>>,
}

impl Icosphere {
    /// Build an icosphere at the given subdivision level.
    pub fn new(subdivision_level: u32) -> Self {
        let (vertices, triangles) = generate_icosphere(subdivision_level);
        let vertex_neighbors = compute_vertex_neighbors(&triangles, vertices.len());
        Self {
            vertices,
            triangles,
            vertex_neighbors,
        }
    }

    /// Brute-force nearest-vertex lookup. O(V). Intended for one-shot
    /// seeding (called a few hundred times during OrogenDla seeding) —
    /// don't call from a per-texel loop.
    pub fn nearest_vertex(&self, dir: Vec3) -> u32 {
        let mut best = 0u32;
        let mut best_dot = -2.0f32;
        for (i, v) in self.vertices.iter().enumerate() {
            let d = v.dot(dir);
            if d > best_dot {
                best_dot = d;
                best = i as u32;
            }
        }
        best
    }

    /// Characteristic angular spacing between neighboring vertices, in
    /// radians. Derived from the surface area per vertex. Useful for
    /// sizing rasterization caps.
    pub fn characteristic_spacing_rad(&self) -> f32 {
        (4.0 * std::f32::consts::PI / self.vertices.len() as f32).sqrt()
    }
}

/// Build the mesh for a subdivided icosahedron. Shared by `spatial_index`
/// (uses the triangle list) and `icosphere::Icosphere` (also uses the
/// vertex list).
pub(crate) fn generate_icosphere(subdivisions: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
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
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    let mut triangles = base_tris;

    let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();
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
    cache: &mut HashMap<(u32, u32), u32>,
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

/// For each vertex, collect the set of vertices it shares a triangle edge
/// with. Output lists are sorted and deduplicated for determinism.
fn compute_vertex_neighbors(triangles: &[[u32; 3]], num_vertices: usize) -> Vec<Vec<u32>> {
    let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); num_vertices];
    for tri in triangles {
        let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
        for (a, b) in edges {
            neighbors[a as usize].push(b);
            neighbors[b as usize].push(a);
        }
    }
    for list in &mut neighbors {
        list.sort_unstable();
        list.dedup();
    }
    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_counts_match_formula() {
        for l in 0..=5 {
            let ico = Icosphere::new(l);
            let expected_v = 10 * 4u32.pow(l) + 2;
            let expected_t = 20 * 4u32.pow(l);
            assert_eq!(ico.vertices.len(), expected_v as usize, "level {l} V");
            assert_eq!(ico.triangles.len(), expected_t as usize, "level {l} T");
        }
    }

    #[test]
    fn every_vertex_has_5_or_6_neighbors() {
        let ico = Icosphere::new(4);
        let mut five = 0;
        let mut six = 0;
        for nbs in &ico.vertex_neighbors {
            match nbs.len() {
                5 => five += 1,
                6 => six += 1,
                n => panic!("unexpected neighbor count: {n}"),
            }
        }
        assert_eq!(five, 12, "exactly 12 vertices should have degree 5");
        assert!(six > 0);
    }

    #[test]
    fn vertices_are_unit_length() {
        let ico = Icosphere::new(3);
        for v in &ico.vertices {
            assert!((v.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn nearest_vertex_is_consistent() {
        let ico = Icosphere::new(3);
        for (i, v) in ico.vertices.iter().enumerate() {
            assert_eq!(ico.nearest_vertex(*v), i as u32);
        }
    }
}
