//! Tectonics — classify plate boundaries and rasterize the per-cell
//! Voronoi boundary-distance field.
//!
//! For each adjacent plate pair, computes the relative velocity at a
//! representative boundary midpoint from the pair's Euler poles (the
//! "quasi-static" model: relative motions are self-consistent without
//! time integration). Classifies each boundary as convergent, divergent,
//! or transform by decomposing relative velocity into a component normal
//! and tangential to the Voronoi boundary. Draws an age for each boundary
//! and computes `cumulative_orogeny` so downstream stages can scale ridge
//! strength by boundary activity.
//!
//! Also rasterizes `boundary_distance_km` — the Voronoi distance from
//! each texel to its plate's nearest Voronoi edge — for diagnostics and
//! downstream stages that need it.
//!
//! Rationale for NOT writing `orogen_intensity` / `orogen_age_myr` here:
//! any per-cell falloff from a Voronoi distance field embeds the polygon
//! geometry of the plate partition directly into the output. See
//! `OrogenDla` for the dendritic-ridge replacement.
//!
//! See `docs/gen/thalos_processes.md §Tectonics`.

use std::collections::HashMap;

use glam::Vec3;
use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{CubemapFace, face_uv_to_dir};
use crate::seeding::Rng;
use crate::stage::Stage;
use crate::types::{Boundary, BoundaryKind, Plate, PlateKind};

#[derive(Debug, Clone, Deserialize)]
pub struct Tectonics {
    #[serde(default = "default_active_boundary_fraction")]
    pub active_boundary_fraction: f32,
    #[serde(default = "default_orogen_age_peak_gyr")]
    pub orogen_age_peak_gyr: f32,
}

fn default_active_boundary_fraction() -> f32 {
    0.20
}
fn default_orogen_age_peak_gyr() -> f32 {
    2.0
}

impl Stage for Tectonics {
    fn name(&self) -> &str {
        "tectonics"
    }
    fn dependencies(&self) -> &[&str] {
        &["plates"]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        let seed = builder.stage_seed();
        let res = builder.cubemap_resolution;
        let body_radius_m = builder.radius_m;
        let body_age_gyr = builder.body_age_gyr;

        let mut rng = Rng::new(seed);

        let plate_map = builder
            .plates
            .as_ref()
            .expect("Tectonics requires Plates to run first (builder.plates is None)");
        let plates: Vec<Plate> = plate_map.plates.clone();
        let n_plates = plates.len();

        let pairs = compute_plate_pairs(&plate_map.plate_id_cubemap, n_plates);

        let mut boundaries: Vec<Boundary> = pairs
            .iter()
            .map(|pp| {
                classify_boundary(
                    pp,
                    &plates,
                    body_radius_m,
                    self.active_boundary_fraction,
                    &mut rng,
                )
            })
            .collect();

        // Draw establishment age per boundary. Active boundaries are young
        // (50–500 Myr). Stagnant boundaries are skewed old, peaked around
        // `orogen_age_peak_gyr`, with a widened right tail.
        let age_peak_myr = self.orogen_age_peak_gyr * 1000.0;
        for b in &mut boundaries {
            b.establishment_age_myr = if b.is_active {
                (rng.range_f64(50.0, 500.0)) as f32
            } else {
                let u = rng.next_f64();
                let spread = 800.0;
                let t = if u < 0.5 {
                    age_peak_myr + (u * 2.0 - 1.0) as f32 * spread
                } else {
                    age_peak_myr + ((u - 0.5) * 2.0) as f32 * spread * 1.5
                };
                t.clamp(500.0, (body_age_gyr * 1000.0 - 200.0).max(600.0))
            };
        }

        for b in &mut boundaries {
            let a_kind = plates[b.plates.0 as usize].kind;
            let c_kind = plates[b.plates.1 as usize].kind;
            b.cumulative_orogeny = compute_cumulative_orogeny(
                b.kind,
                a_kind,
                c_kind,
                b.relative_speed_m_per_myr,
                b.establishment_age_myr,
            );
        }

        // Adjacency list for per-cell nearest-boundary distance lookup.
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_plates];
        for p in &pairs {
            adjacency[p.plate_a].push(p.plate_b);
            adjacency[p.plate_b].push(p.plate_a);
        }

        // Rasterize `boundary_distance_km` — Voronoi distance from each
        // texel to its plate's nearest edge. Still populated (diagnostics
        // + any downstream stage that wants it), but orogen intensity and
        // age are now `OrogenDla`'s job.
        let plate_id_cubemap = &plate_map.plate_id_cubemap;
        let inv_res = 1.0 / res as f32;
        let bd = &mut builder.boundary_distance_km;

        for face in CubemapFace::ALL {
            let plate_ids = plate_id_cubemap.face_data(face);
            let bd_out = bd.face_data_mut(face);
            bd_out.par_iter_mut().enumerate().for_each(|(idx, bd_v)| {
                let own_id = plate_ids[idx] as usize;
                let y = (idx as u32) / res;
                let x = (idx as u32) % res;
                let u = (x as f32 + 0.5) * inv_res;
                let v = (y as f32 + 0.5) * inv_res;
                let dir = face_uv_to_dir(face, u, v);
                let cos_d_own = dir.dot(plates[own_id].centroid).clamp(-1.0, 1.0);
                let d_own = cos_d_own.acos();
                let mut best_dist_km = f32::INFINITY;
                for &nb in &adjacency[own_id] {
                    let cd = dir.dot(plates[nb].centroid).clamp(-1.0, 1.0);
                    let d_nb = cd.acos();
                    let angular = ((d_nb - d_own) * 0.5).max(0.0);
                    let dist_km = angular * body_radius_m / 1000.0;
                    if dist_km < best_dist_km {
                        best_dist_km = dist_km;
                    }
                }
                *bd_v = best_dist_km;
            });
        }

        if let Some(pm) = builder.plates.as_mut() {
            pm.boundaries = boundaries;
        }
    }
}

struct PlatePair {
    plate_a: usize,
    plate_b: usize,
    midpoint: Vec3,
}

fn pair_key(a: u16, b: u16) -> (u16, u16) {
    if a < b { (a, b) } else { (b, a) }
}

fn compute_plate_pairs(
    plate_id_cubemap: &crate::cubemap::Cubemap<u16>,
    n_plates: usize,
) -> Vec<PlatePair> {
    let res = plate_id_cubemap.resolution();
    let inv_res = 1.0 / res as f32;

    let mut accumulators: HashMap<(u16, u16), (Vec3, u32)> = HashMap::new();

    for face in CubemapFace::ALL {
        let data = plate_id_cubemap.face_data(face);
        for y in 0..res {
            for x in 0..res {
                let p = data[(y * res + x) as usize];
                let u = (x as f32 + 0.5) * inv_res;
                let v = (y as f32 + 0.5) * inv_res;
                let dir = face_uv_to_dir(face, u, v);
                if x + 1 < res {
                    let q = data[(y * res + x + 1) as usize];
                    if p != q {
                        let entry = accumulators
                            .entry(pair_key(p, q))
                            .or_insert((Vec3::ZERO, 0));
                        entry.0 += dir;
                        entry.1 += 1;
                    }
                }
                if y + 1 < res {
                    let q = data[((y + 1) * res + x) as usize];
                    if p != q {
                        let entry = accumulators
                            .entry(pair_key(p, q))
                            .or_insert((Vec3::ZERO, 0));
                        entry.0 += dir;
                        entry.1 += 1;
                    }
                }
            }
        }
    }

    let _ = n_plates;
    // Sort by plate-id pair so hashmap iteration-order non-determinism
    // can't drift `is_active` / age RNG draws between runs.
    let mut entries: Vec<((u16, u16), (Vec3, u32))> = accumulators.into_iter().collect();
    entries.sort_by_key(|&((a, b), _)| (a, b));
    entries
        .into_iter()
        .map(|((a, b), (sum, _cnt))| PlatePair {
            plate_a: a as usize,
            plate_b: b as usize,
            midpoint: sum.normalize(),
        })
        .collect()
}

fn classify_boundary(
    pp: &PlatePair,
    plates: &[Plate],
    body_radius_m: f32,
    active_boundary_fraction: f32,
    rng: &mut Rng,
) -> Boundary {
    let p0 = &plates[pp.plate_a];
    let p1 = &plates[pp.plate_b];

    let v_a = p0.euler_pole.cross(pp.midpoint) * p0.angular_velocity_rad_per_myr;
    let v_b = p1.euler_pole.cross(pp.midpoint) * p1.angular_velocity_rad_per_myr;
    let relative = v_b - v_a;

    let to_b = p1.centroid - p0.centroid;
    let tangent_to_b = {
        let v = to_b - pp.midpoint * to_b.dot(pp.midpoint);
        if v.length_squared() > 1e-12 {
            v.normalize()
        } else {
            pick_tangent(pp.midpoint)
        }
    };

    let v_normal = relative.dot(tangent_to_b);
    let v_tangent_vec = relative - tangent_to_b * v_normal;
    let v_tangent = v_tangent_vec.length();

    let kind = if v_normal.abs() > v_tangent {
        if v_normal > 0.0 {
            BoundaryKind::Divergent
        } else {
            BoundaryKind::Convergent
        }
    } else {
        BoundaryKind::Transform
    };

    let relative_speed_m_per_myr = relative.length() * body_radius_m;
    let is_active = rng.next_f64() < active_boundary_fraction as f64;

    Boundary {
        plates: (pp.plate_a as u16, pp.plate_b as u16),
        kind,
        relative_speed_m_per_myr,
        establishment_age_myr: 0.0,
        is_active,
        cumulative_orogeny: 0.0,
    }
}

fn compute_cumulative_orogeny(
    kind: BoundaryKind,
    a: PlateKind,
    b: PlateKind,
    relative_speed_m_per_myr: f32,
    boundary_age_myr: f32,
) -> f32 {
    if kind != BoundaryKind::Convergent {
        return 0.0;
    }
    let continental_count =
        (a == PlateKind::Continental) as u32 + (b == PlateKind::Continental) as u32;
    if continental_count == 0 {
        return 0.15;
    }
    let continental_weight = continental_count as f32 / 2.0;
    let speed_ref = 1.0e5;
    let speed_norm = (relative_speed_m_per_myr / speed_ref).min(2.0);
    let age_norm = (boundary_age_myr / 1000.0).clamp(0.1, 3.0);
    (continental_weight * speed_norm * age_norm).min(1.0)
}

fn pick_tangent(dir: Vec3) -> Vec3 {
    let other = if dir.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    (other - dir * dir.dot(other)).normalize()
}
