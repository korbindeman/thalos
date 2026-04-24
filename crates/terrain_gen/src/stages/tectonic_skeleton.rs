//! Stage 1 — Tectonic Skeleton.
//!
//! Partitions the sphere into N plates via *noise-warped Voronoi* so
//! plate boundaries are fractal rather than straight geodesic arcs.
//! Classifies boundaries as Suture/RiftScar/ActiveMargin and grows a
//! few HotspotTrack paths. Continental vs. oceanic assignment is
//! *deferred* to Stage 2 (CoarseElevation), which derives land/ocean
//! from an independent noise mask — plates and continents are
//! decoupled concerns, matching Earth physics: plate boundaries do
//! not follow coastlines.
//!
//! Algorithm:
//! 1. Seed `n_cratons` plate centroids and Lloyd-relax on the warped
//!    Voronoi distance function.
//! 2. Each vertex assigns to the plate minimising
//!        angular_dist(dir_warped(pos), centroid)
//!    where `dir_warped` is `pos` offset by a 3D fBm vector then
//!    renormalised. Plate boundaries become fractal at the warp's
//!    spatial frequency.
//! 3. Collect adjacent-plate boundary pairs, randomly classify
//!    `n_active_margins` as ActiveMargin, `n_rift_scars` as RiftScar,
//!    rest as Suture.
//! 4. Rewrite boundary-adjacent vertices to the boundary province
//!    (ActiveMargin > RiftScar > Suture priority).
//! 5. Grow `n_hotspot_tracks` linear tracks through plate interiors.
//!
//! See `docs/thalos_terrain_pipeline.md §Stage 1`.

use std::collections::{BTreeSet, HashMap};

use glam::Vec3;
use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{CubemapFace, face_uv_to_dir};
use crate::icosphere::Icosphere;
use crate::noise::fbm3;
use crate::province::{ProvinceDef, ProvinceKind};
use crate::seeding::{Rng, splitmix64};
use crate::stage::Stage;

#[derive(Debug, Clone, Deserialize)]
pub struct TectonicSkeleton {
    #[serde(default = "default_subdivision_level")]
    pub subdivision_level: u32,
    #[serde(default = "default_n_cratons")]
    pub n_cratons: u32,
    /// Amplitude of the 3D fBm offset applied to each vertex position
    /// before Voronoi assignment, in unit-sphere units. At 0 the
    /// partition is plain spherical Voronoi with straight-arc
    /// boundaries; ~0.3–0.5 produces interlocking fractal boundaries
    /// matching real plate outlines; >0.8 starts breaking contiguity
    /// (islands of one plate embedded inside another).
    #[serde(default = "default_plate_warp_amplitude")]
    pub plate_warp_amplitude: f32,
    /// Base frequency of the plate-warp fBm, in cycles across the
    /// unit sphere. 2–4 produces waviness at continent-scale
    /// wavelength.
    #[serde(default = "default_plate_warp_frequency")]
    pub plate_warp_frequency: f32,
    #[serde(default = "default_plate_warp_octaves")]
    pub plate_warp_octaves: u32,
    #[serde(default = "default_n_active_margins")]
    pub n_active_margins: u32,
    #[serde(default = "default_n_rift_scars")]
    pub n_rift_scars: u32,
    #[serde(default = "default_n_hotspot_tracks")]
    pub n_hotspot_tracks: u32,
    #[serde(default = "default_hotspot_track_length")]
    pub hotspot_track_length: u32,
    #[serde(default = "default_lloyd_iterations")]
    pub lloyd_iterations: u32,
    /// When true, paint the albedo cubemap with flat per-province
    /// colors so Stage 1 output is directly visible. Turn off once
    /// Stage 2+ produce real albedo.
    #[serde(default = "default_true")]
    pub debug_paint_albedo: bool,
}

fn default_subdivision_level() -> u32 {
    6
}
fn default_n_cratons() -> u32 {
    20
}
fn default_plate_warp_amplitude() -> f32 {
    0.4
}
fn default_plate_warp_frequency() -> f32 {
    3.0
}
fn default_plate_warp_octaves() -> u32 {
    5
}
fn default_n_active_margins() -> u32 {
    1
}
fn default_n_rift_scars() -> u32 {
    2
}
fn default_n_hotspot_tracks() -> u32 {
    4
}
fn default_hotspot_track_length() -> u32 {
    10
}
fn default_lloyd_iterations() -> u32 {
    3
}
fn default_true() -> bool {
    true
}

#[derive(Copy, Clone, Debug)]
enum BoundaryClass {
    Suture,
    RiftScar,
    ActiveMargin,
}

impl Stage for TectonicSkeleton {
    fn name(&self) -> &str {
        "tectonic_skeleton"
    }

    fn dependencies(&self) -> &[&str] {
        &[]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        assert!(
            self.n_cratons >= 2,
            "TectonicSkeleton needs at least 2 cratons, got {}",
            self.n_cratons
        );

        let seed = builder.stage_seed();
        let mut rng = Rng::new(seed);
        let warp_seed = splitmix64(seed ^ 0xBD_0A_7E_EF_00_11_22_33);

        // 1. Build the icosphere.
        let sphere = Icosphere::new(self.subdivision_level);
        let n_verts = sphere.vertices.len();
        let n_cr = self.n_cratons as usize;

        // 2. Seed plate centroids and Lloyd-relax on the warped
        //    Voronoi. Each iteration reassigns each vertex to the
        //    nearest centroid using the *warped* position, then
        //    recomputes centroids from the *original* positions so
        //    the centroid still represents the plate's geographic
        //    center.
        let mut centroids: Vec<Vec3> = (0..n_cr)
            .map(|_| rng.unit_vector().as_vec3().normalize())
            .collect();

        for _ in 0..self.lloyd_iterations {
            let assignments: Vec<u32> = sphere
                .vertices
                .par_iter()
                .map(|v| nearest_centroid_warped(&centroids, *v, self, warp_seed))
                .collect();
            let mut sums = vec![Vec3::ZERO; n_cr];
            let mut counts = vec![0u32; n_cr];
            for (vi, &cid) in assignments.iter().enumerate() {
                sums[cid as usize] += sphere.vertices[vi];
                counts[cid as usize] += 1;
            }
            for i in 0..n_cr {
                if counts[i] > 0 {
                    centroids[i] = (sums[i] / counts[i] as f32).normalize();
                }
            }
        }

        // 3. Final plate assignment using the warped distance.
        let vertex_craton: Vec<u32> = sphere
            .vertices
            .par_iter()
            .map(|v| nearest_centroid_warped(&centroids, *v, self, warp_seed))
            .collect();

        // 4. Gather unique adjacent-plate boundary pairs.
        let mut boundary_pairs: BTreeSet<(u32, u32)> = BTreeSet::new();
        for vi in 0..n_verts {
            let a = vertex_craton[vi];
            for &nb in &sphere.vertex_neighbors[vi] {
                let b = vertex_craton[nb as usize];
                if a < b {
                    boundary_pairs.insert((a, b));
                } else if b < a {
                    boundary_pairs.insert((b, a));
                }
            }
        }

        // 5. Classify boundaries. No continental-vs-oceanic prior
        //    anymore — just take the shuffled list and assign the
        //    first N_active as ActiveMargin, next N_rift as
        //    RiftScar, rest as Suture. Shuffle deterministically.
        let mut all_pairs: Vec<(u32, u32)> = boundary_pairs.iter().copied().collect();
        all_pairs.sort_by_key(|&(a, b)| {
            splitmix64(((a as u64) << 32 | b as u64) ^ seed)
        });

        let mut boundary_class: HashMap<(u32, u32), BoundaryClass> = HashMap::new();
        let mut idx = 0usize;
        for _ in 0..self.n_active_margins {
            if idx >= all_pairs.len() {
                break;
            }
            boundary_class.insert(all_pairs[idx], BoundaryClass::ActiveMargin);
            idx += 1;
        }
        for _ in 0..self.n_rift_scars {
            if idx >= all_pairs.len() {
                break;
            }
            boundary_class.insert(all_pairs[idx], BoundaryClass::RiftScar);
            idx += 1;
        }
        for &pair in &all_pairs[idx..] {
            boundary_class.insert(pair, BoundaryClass::Suture);
        }

        // 6. Build province table. Plate interiors are all Craton —
        //    stage 2's noise mask decides continental vs. oceanic
        //    downstream. Each Craton gets its own `elevation_bias_m`
        //    drawn from a uniform distribution so continental
        //    cratons read as a mix of lowland shields and elevated
        //    plateaus — Stage 2 adds this on top of the global
        //    continental bias, producing Tibet-vs-Russian-Plain
        //    variation between cratons rather than a single flat
        //    continental elevation.
        let mut provinces: Vec<ProvinceDef> = Vec::new();
        let mut craton_province_id = vec![0u32; n_cr];
        for i in 0..n_cr {
            let id = provinces.len() as u32;
            craton_province_id[i] = id;
            provinces.push(ProvinceDef {
                id,
                kind: ProvinceKind::Craton,
                age_myr: rng.range_f64(1_500.0, 3_000.0) as f32,
                elevation_bias_m: rng.range_f64(-200.0, 1_200.0) as f32,
            });
        }

        let mut boundary_province_id: HashMap<(u32, u32), u32> = HashMap::new();
        for &pair in &all_pairs {
            let class = boundary_class[&pair];
            let id = provinces.len() as u32;
            let (kind, age_myr, elevation_bias_m) = match class {
                BoundaryClass::Suture => (
                    ProvinceKind::Suture,
                    rng.range_f64(1_000.0, 2_500.0) as f32,
                    500.0,
                ),
                BoundaryClass::RiftScar => (
                    ProvinceKind::RiftScar,
                    rng.range_f64(500.0, 1_500.0) as f32,
                    -1_000.0,
                ),
                BoundaryClass::ActiveMargin => (
                    ProvinceKind::ActiveMargin,
                    rng.range_f64(0.0, 20.0) as f32,
                    2_000.0,
                ),
            };
            provinces.push(ProvinceDef {
                id,
                kind,
                age_myr,
                elevation_bias_m,
            });
            boundary_province_id.insert(pair, id);
        }

        // 7. Per-vertex province assignment. Each vertex starts on
        //    its home plate's Craton province, and is overridden
        //    with a boundary province if any neighbor belongs to a
        //    different plate (ActiveMargin > RiftScar > Suture
        //    priority).
        let vertex_craton_provinces: Vec<u32> = vertex_craton
            .iter()
            .map(|&cid| craton_province_id[cid as usize])
            .collect();
        let mut vertex_provinces: Vec<u32> = vertex_craton_provinces.clone();

        for vi in 0..n_verts {
            let a = vertex_craton[vi];
            let mut chosen: Option<u32> = None;
            let mut chosen_priority: u8 = 0;
            for &nb in &sphere.vertex_neighbors[vi] {
                let b = vertex_craton[nb as usize];
                if a == b {
                    continue;
                }
                let key = if a < b { (a, b) } else { (b, a) };
                let Some(&bid) = boundary_province_id.get(&key) else {
                    continue;
                };
                let priority = match provinces[bid as usize].kind {
                    ProvinceKind::ActiveMargin => 3,
                    ProvinceKind::RiftScar => 2,
                    ProvinceKind::Suture => 1,
                    _ => 0,
                };
                if priority > chosen_priority {
                    chosen = Some(bid);
                    chosen_priority = priority;
                }
            }
            if let Some(bid) = chosen {
                vertex_provinces[vi] = bid;
            }
        }

        // 8. Grow hotspot tracks. Tracks start on random plate-
        //    interior vertices and walk straight across the mesh,
        //    bending through whatever plate they encounter.
        //    Whether a track reads as a seamount chain or a
        //    volcanic belt on land is decided later by the Stage 2
        //    continent mask — hotspots don't care about plate
        //    identity.
        for _ in 0..self.n_hotspot_tracks {
            let mut seed_vi: Option<usize> = None;
            for _ in 0..64 {
                let cand = (rng.next_u64() as usize) % n_verts;
                if provinces[vertex_provinces[cand] as usize].kind == ProvinceKind::Craton {
                    seed_vi = Some(cand);
                    break;
                }
            }
            let Some(start) = seed_vi else {
                continue;
            };

            let hotspot_id = provinces.len() as u32;
            provinces.push(ProvinceDef {
                id: hotspot_id,
                kind: ProvinceKind::HotspotTrack,
                age_myr: rng.range_f64(0.0, 100.0) as f32,
                elevation_bias_m: 1_000.0,
            });

            let mut current = start;
            let mut prev = current;
            for step in 0..self.hotspot_track_length as usize {
                vertex_provinces[current] = hotspot_id;

                let candidates: Vec<u32> = sphere.vertex_neighbors[current]
                    .iter()
                    .copied()
                    .filter(|&nb| {
                        nb as usize != prev
                            && provinces[vertex_provinces[nb as usize] as usize].kind
                                == ProvinceKind::Craton
                    })
                    .collect();
                if candidates.is_empty() {
                    break;
                }

                let next = if step == 0 {
                    candidates[(rng.next_u64() as usize) % candidates.len()]
                } else {
                    let dir = (sphere.vertices[current] - sphere.vertices[prev])
                        .normalize_or_zero();
                    *candidates
                        .iter()
                        .max_by(|&&a, &&b| {
                            let da = (sphere.vertices[a as usize] - sphere.vertices[current])
                                .dot(dir);
                            let db = (sphere.vertices[b as usize] - sphere.vertices[current])
                                .dot(dir);
                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap()
                };
                prev = current;
                current = next as usize;
            }
        }

        // 9. Commit.
        builder.sphere = Some(sphere);
        builder.vertex_provinces = Some(vertex_provinces);
        builder.vertex_craton_provinces = Some(vertex_craton_provinces);
        builder.provinces = provinces;

        // 10. Debug albedo.
        if self.debug_paint_albedo {
            paint_province_debug(builder);
        }
    }
}

/// Voronoi assignment under a 3D fBm domain warp on the input
/// position. Standard domain-warping technique: offset `dir` by a
/// coherent noise vector, renormalise, then pick the nearest
/// centroid. Boundaries become fractal at the warp's spatial
/// frequency because two nearby vertices can warp to distant regions
/// of the sphere.
fn nearest_centroid_warped(
    centroids: &[Vec3],
    dir: Vec3,
    params: &TectonicSkeleton,
    warp_seed: u64,
) -> u32 {
    let f = params.plate_warp_frequency as f64;
    let oct = params.plate_warp_octaves;
    let wx = fbm3(
        dir.x as f64 * f,
        dir.y as f64 * f,
        dir.z as f64 * f,
        warp_seed,
        oct,
        0.5,
        2.0,
    );
    let wy = fbm3(
        dir.x as f64 * f + 17.31,
        dir.y as f64 * f + 17.31,
        dir.z as f64 * f + 17.31,
        warp_seed.wrapping_add(1),
        oct,
        0.5,
        2.0,
    );
    let wz = fbm3(
        dir.x as f64 * f + 41.17,
        dir.y as f64 * f + 41.17,
        dir.z as f64 * f + 41.17,
        warp_seed.wrapping_add(2),
        oct,
        0.5,
        2.0,
    );
    let amp = params.plate_warp_amplitude;
    let warped = Vec3::new(
        dir.x + amp * wx as f32,
        dir.y + amp * wy as f32,
        dir.z + amp * wz as f32,
    )
    .normalize_or_zero();

    let mut best = 0u32;
    let mut best_dot = f32::MIN;
    for (i, c) in centroids.iter().enumerate() {
        let d = c.dot(warped);
        if d > best_dot {
            best_dot = d;
            best = i as u32;
        }
    }
    best
}

/// Paint the albedo accumulator with a flat color per province kind
/// so Stage 1 output is visible in the editor. Cubemap is a derived
/// rendering output here — not the canonical data store.
fn paint_province_debug(builder: &mut BodyBuilder) {
    let sphere = builder.sphere.as_ref().unwrap();
    let vertex_provinces = builder.vertex_provinces.as_ref().unwrap();
    let provinces = &builder.provinces;
    let res = builder.cubemap_resolution;
    let inv = 1.0 / res as f32;

    for face in CubemapFace::ALL {
        let face_albedo = builder.albedo_contributions.albedo.face_data_mut(face);
        face_albedo
            .par_chunks_mut(res as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let mut last_v = 0u32;
                for (x, texel) in row.iter_mut().enumerate() {
                    let u = (x as f32 + 0.5) * inv;
                    let v = (y as f32 + 0.5) * inv;
                    let dir = face_uv_to_dir(face, u, v);
                    let vi = sphere.nearest_vertex_from(dir, last_v);
                    last_v = vi;
                    let pid = vertex_provinces[vi as usize];
                    let kind = provinces[pid as usize].kind;
                    let c = debug_color(kind);
                    *texel = [c[0], c[1], c[2], 1.0];
                }
            });
    }
}

fn debug_color(kind: ProvinceKind) -> [f32; 3] {
    match kind {
        ProvinceKind::Craton => [0.72, 0.65, 0.50],
        ProvinceKind::Suture => [0.50, 0.15, 0.15],
        ProvinceKind::RiftScar => [0.25, 0.55, 0.25],
        ProvinceKind::ArcRemnant => [0.95, 0.55, 0.18],
        ProvinceKind::ActiveMargin => [1.00, 0.15, 0.15],
        ProvinceKind::HotspotTrack => [0.60, 0.25, 0.80],
        ProvinceKind::OceanicBasin => [0.12, 0.22, 0.55],
    }
}
