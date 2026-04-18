//! Plates — Voronoi partition of the sphere into tectonic plates.
//!
//! Seeds `n_plates` centroids on the sphere via Lloyd-relaxed uniform
//! sampling, classifies each plate as Continental or Oceanic with loose
//! clustering to hit a target continental area fraction, and assigns each
//! plate an Euler pole + angular velocity. Writes `BodyBuilder::plates`.
//!
//! Uses brute-force nearest-centroid per texel (O(N_texels × N_plates)).
//! Parallelized across texels with rayon — for ~40 plates on a 2048²
//! cubemap that is ~1 B comparisons, on the order of 1 s of bake time per
//! Lloyd iteration.
//!
//! See `docs/gen/thalos_processes.md §Plates`.

use std::collections::VecDeque;

use glam::Vec3;
use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{Cubemap, CubemapFace, face_uv_to_dir};
use crate::seeding::Rng;
use crate::stage::Stage;
use crate::types::{Plate, PlateKind, PlateMap};

#[derive(Debug, Clone, Deserialize)]
pub struct Plates {
    #[serde(default = "default_n_plates")]
    pub n_plates: u32,
    #[serde(default = "default_continental_area_fraction")]
    pub continental_area_fraction: f32,
    #[serde(default = "default_n_continental_seeds")]
    pub n_continental_seeds: u32,
    #[serde(default = "default_neighbour_continental_bias")]
    pub neighbour_continental_bias: f32,
    #[serde(default = "default_lloyd_iterations")]
    pub lloyd_iterations: u32,
}

fn default_n_plates() -> u32 {
    40
}
fn default_continental_area_fraction() -> f32 {
    0.35
}
fn default_n_continental_seeds() -> u32 {
    4
}
fn default_neighbour_continental_bias() -> f32 {
    0.5
}
fn default_lloyd_iterations() -> u32 {
    3
}

impl Stage for Plates {
    fn name(&self) -> &str {
        "plates"
    }
    fn dependencies(&self) -> &[&str] {
        &[]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        let seed = builder.stage_seed();
        let res = builder.cubemap_resolution;
        let n = self.n_plates as usize;
        assert!(n >= 2, "Plates stage needs at least 2 plates, got {n}");

        let mut rng = Rng::new(seed);

        // 1. Seed N centroids uniformly on the sphere.
        let mut centroids: Vec<Vec3> = (0..n)
            .map(|_| rng.unit_vector().as_vec3().normalize())
            .collect();

        // 2. Lloyd relaxation — spread centroids more evenly.
        for _iter in 0..self.lloyd_iterations {
            let assignments = assign_plate_ids(&centroids, res);
            centroids = recompute_centroids(&assignments, res, &centroids);
        }

        // 3. Final Voronoi assignment.
        let plate_id_cubemap = assign_plate_ids(&centroids, res);

        // 4. Plate adjacency from the assigned cubemap.
        let adjacency = build_adjacency(&plate_id_cubemap, n);

        // 5. Classify continental / oceanic with loose clustering.
        let plate_kinds = classify_plate_types(
            &adjacency,
            n,
            self.n_continental_seeds.min(self.n_plates) as usize,
            self.continental_area_fraction,
            self.neighbour_continental_bias,
            &plate_id_cubemap,
            &mut rng,
        );

        // 6. Euler poles + angular velocities per plate.
        let plates: Vec<Plate> = (0..n)
            .map(|i| {
                let euler_pole = rng.unit_vector().as_vec3().normalize();
                let omega = sample_angular_velocity(&mut rng);
                Plate {
                    id: i as u16,
                    kind: plate_kinds[i],
                    centroid: centroids[i],
                    euler_pole,
                    angular_velocity_rad_per_myr: omega,
                }
            })
            .collect();

        builder.plates = Some(PlateMap {
            plates,
            boundaries: Vec::new(), // Tectonics populates this.
            plate_id_cubemap,
        });
    }
}

/// Brute-force Voronoi: for every texel, find the nearest centroid.
/// Rayon-parallel over (face, texel) pairs. Centroids are unit vectors,
/// so nearest by dot-product == nearest by great-circle distance.
fn assign_plate_ids(centroids: &[Vec3], res: u32) -> Cubemap<u16> {
    let mut cubemap = Cubemap::<u16>::new(res);
    let inv_res = 1.0 / res as f32;

    for face in CubemapFace::ALL {
        let data = cubemap.face_data_mut(face);
        data.par_iter_mut().enumerate().for_each(|(idx, val)| {
            let x = (idx as u32) % res;
            let y = (idx as u32) / res;
            let u = (x as f32 + 0.5) * inv_res;
            let v = (y as f32 + 0.5) * inv_res;
            let dir = face_uv_to_dir(face, u, v);

            let mut best_id = 0u16;
            let mut best_dot = -2.0f32;
            for (i, c) in centroids.iter().enumerate() {
                let d = dir.dot(*c);
                if d > best_dot {
                    best_dot = d;
                    best_id = i as u16;
                }
            }
            *val = best_id;
        });
    }
    cubemap
}

/// Lloyd step: each new centroid = normalized mean of its assigned texels'
/// directions. Plates with zero texels (shouldn't happen after the first
/// iteration but can on the first pass if two seeds coincided) keep their
/// previous position.
fn recompute_centroids(plate_ids: &Cubemap<u16>, res: u32, prev: &[Vec3]) -> Vec<Vec3> {
    let n = prev.len();
    let inv_res = 1.0 / res as f32;

    let mut accumulators = vec![Vec3::ZERO; n];
    let mut counts = vec![0u32; n];

    for face in CubemapFace::ALL {
        let data = plate_ids.face_data(face);
        for y in 0..res {
            for x in 0..res {
                let u = (x as f32 + 0.5) * inv_res;
                let v = (y as f32 + 0.5) * inv_res;
                let dir = face_uv_to_dir(face, u, v);
                let id = data[(y * res + x) as usize] as usize;
                accumulators[id] += dir;
                counts[id] += 1;
            }
        }
    }

    accumulators
        .into_iter()
        .zip(counts)
        .enumerate()
        .map(|(i, (acc, cnt))| {
            if cnt == 0 {
                prev[i]
            } else {
                (acc / cnt as f32).normalize()
            }
        })
        .collect()
}

/// Scan the plate-ID cubemap for inter-face-interior neighbour disagreements
/// and return an undirected adjacency list (Vec<neighbour-id>) per plate.
///
/// Ignores face seams — a small fraction of true adjacencies are missed at
/// cube edges, but for a 40-plate partition most pairs will have additional
/// interior adjacencies and the clustering step is tolerant of this.
fn build_adjacency(plate_ids: &Cubemap<u16>, n: usize) -> Vec<Vec<usize>> {
    let res = plate_ids.resolution();
    let mut edges: Vec<(u16, u16)> = Vec::new();

    for face in CubemapFace::ALL {
        let data = plate_ids.face_data(face);
        for y in 0..res {
            for x in 0..res {
                let p = data[(y * res + x) as usize];
                // Right neighbour.
                if x + 1 < res {
                    let q = data[(y * res + x + 1) as usize];
                    if p != q {
                        edges.push(if p < q { (p, q) } else { (q, p) });
                    }
                }
                // Down neighbour.
                if y + 1 < res {
                    let q = data[((y + 1) * res + x) as usize];
                    if p != q {
                        edges.push(if p < q { (p, q) } else { (q, p) });
                    }
                }
            }
        }
    }

    edges.sort_unstable();
    edges.dedup();

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (p, q) in edges {
        adjacency[p as usize].push(q as usize);
        adjacency[q as usize].push(p as usize);
    }
    adjacency
}

/// Pick `n_seeds` random plates as continental, then grow outward along the
/// adjacency graph. Each unvisited neighbour flips continental with
/// probability `neighbour_bias`, otherwise oceanic. Grows until the
/// aggregate texel-count of continental plates reaches `target_frac` of
/// total surface area. The "otherwise oceanic" branch is what gives loose
/// clusters — without it, growth would saturate into a single supercontinent.
fn classify_plate_types(
    adjacency: &[Vec<usize>],
    n: usize,
    n_seeds: usize,
    target_frac: f32,
    neighbour_bias: f32,
    plate_id_cubemap: &Cubemap<u16>,
    rng: &mut Rng,
) -> Vec<PlateKind> {
    // Texels per plate (area proxy).
    let mut texel_counts: Vec<u32> = vec![0; n];
    for face in CubemapFace::ALL {
        for &val in plate_id_cubemap.face_data(face) {
            texel_counts[val as usize] += 1;
        }
    }
    let total_texels: u32 = texel_counts.iter().sum();
    let target_continental = (target_frac * total_texels as f32) as u32;

    let mut kinds: Vec<Option<PlateKind>> = vec![None; n];

    // Fisher-Yates shuffle indices to pick seeds without replacement.
    let mut available: Vec<usize> = (0..n).collect();
    for i in (1..available.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        available.swap(i, j);
    }

    let n_seeds = n_seeds.min(n);
    let mut frontier: VecDeque<usize> = VecDeque::new();
    let mut continental_texels: u32 = 0;
    for &id in available.iter().take(n_seeds) {
        kinds[id] = Some(PlateKind::Continental);
        continental_texels += texel_counts[id];
        frontier.push_back(id);
    }

    while continental_texels < target_continental
        && let Some(current) = frontier.pop_front()
    {
        for &neighbour in &adjacency[current] {
            if kinds[neighbour].is_none() {
                if rng.next_f64() < neighbour_bias as f64 {
                    kinds[neighbour] = Some(PlateKind::Continental);
                    continental_texels += texel_counts[neighbour];
                    frontier.push_back(neighbour);
                    if continental_texels >= target_continental {
                        break;
                    }
                } else {
                    // Explicit oceanic — forms the cluster boundary.
                    kinds[neighbour] = Some(PlateKind::Oceanic);
                }
            }
        }
    }

    kinds
        .into_iter()
        .map(|k| k.unwrap_or(PlateKind::Oceanic))
        .collect()
}

/// Sample an angular velocity in rad/Myr with a distribution biased toward
/// Earth-like plate speeds (~0.1–1°/Myr) plus a 20% chance of being
/// near-stagnant. Sign is randomized.
fn sample_angular_velocity(rng: &mut Rng) -> f32 {
    const RAD_PER_DEG: f64 = std::f64::consts::PI / 180.0;
    let u = rng.next_f64();
    let magnitude = if u < 0.2 {
        // Near-stagnant tail: 0..0.1°/Myr.
        rng.range_f64(0.0, 0.1 * RAD_PER_DEG)
    } else {
        // Active range: 0.1..1°/Myr.
        rng.range_f64(0.1 * RAD_PER_DEG, 1.0 * RAD_PER_DEG)
    };
    let sign = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
    (magnitude * sign) as f32
}
