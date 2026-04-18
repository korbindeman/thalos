//! OrogenDla — Diffusion-Limited Aggregation mountain ranges anchored to
//! convergent plate boundaries.
//!
//! Where `Tectonics` only classifies boundaries, `OrogenDla` grows dendritic
//! ridge networks out from them. Seeds are planted at icosphere vertices
//! nearest to sampled points along each convergent boundary. For each
//! boundary, random walkers on the icosphere graph wander until they
//! touch the cluster, then attach — classical DLA. The resulting branching
//! structure is rasterized to `orogen_intensity` with depth-weighted
//! intensity so ridge pixels near the seed end up taller than leaves.
//!
//! Determinism: serial DLA within a boundary (walker attachment order is
//! load-bearing), rayon-parallel across boundaries (each returns an
//! independent `BoundaryDlaResult`; merge step is order-independent).
//!
//! v1: single-resolution. Multi-resolution graph subdivision with
//! jittered midpoint upscale + dual-filter blur (both flagged in the
//! project notes) live in v2 if this v1 looks pixellated at the icosphere
//! → 2048² cubemap upsample.

use std::collections::HashMap;

use glam::Vec3;
use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{CubemapFace, face_uv_to_dir};
use crate::icosphere::Icosphere;
use crate::seeding::{Rng, splitmix64};
use crate::stage::Stage;
use super::util::for_face_texels_in_cap;
use crate::types::BoundaryKind;

#[derive(Debug, Clone, Deserialize)]
pub struct OrogenDla {
    /// Icosphere subdivision level used as the DLA graph. Level 6 gives
    /// ~40k vertices (~55 km spacing on Thalos) — mountain-range scale.
    #[serde(default = "default_subdivision_level")]
    pub subdivision_level: u32,
    /// Seeds dropped per convergent boundary, scaled by `cumulative_orogeny`
    /// so stronger boundaries (continental–continental, fast convergence)
    /// get more ridge material.
    #[serde(default = "default_seed_density")]
    pub seed_density: f32,
    /// Walkers launched per seed. Higher → denser branching.
    #[serde(default = "default_walkers_per_seed")]
    pub walkers_per_seed: u32,
    /// Random-walk step budget per walker.
    #[serde(default = "default_walker_step_budget")]
    pub walker_step_budget: u32,
    /// Launch-cap half-angle expressed in km on the body surface. Walkers
    /// spawn inside this radius of their seed and abandon if they wander
    /// outside — keeps growth localised to each boundary instead of
    /// bleeding into other tectonic regions.
    #[serde(default = "default_launch_cap_km")]
    pub launch_cap_km: f32,
    /// Gaussian falloff σ (km) used when rasterizing cluster vertices to
    /// the cubemap. Ridges are wider when this is larger. Should be
    /// roughly the icosphere vertex spacing so adjacent cluster vertices
    /// blend into continuous ridges rather than dots.
    #[serde(default = "default_ridge_falloff_km")]
    pub ridge_falloff_km: f32,
    /// Depth at which ridge intensity saturates near 1.0. Smooth
    /// asymptote `raw / (raw + 1)` where `raw = depth / depth_saturation`
    /// — depth-1 leaves contribute modestly, a seed at depth ~N_saturation
    /// reaches ~0.5, and deeper cells saturate smoothly without ever
    /// blowing past 1.0. Prevents fine detail from piling up on centres
    /// (the "clamped summit" trick from the project notes).
    #[serde(default = "default_depth_saturation")]
    pub depth_saturation: f32,
    /// How many sub-point stamps are made along each parent→child ridge
    /// edge. Higher values give smoother, more continuous ridges at the
    /// cost of more rasterization work. 6–10 is a good range for
    /// subdivision level 6–7.
    #[serde(default = "default_samples_per_edge")]
    pub samples_per_edge: u32,
    /// Fraction of the edge length used as midpoint jitter amplitude.
    /// Displaces interior samples in the sphere-tangent plane so ridges
    /// don't trace perfect geodesics. `0.0` disables; `0.3`–`0.5` gives
    /// natural wiggle without scrambling topology.
    #[serde(default = "default_midpoint_jitter_frac")]
    pub midpoint_jitter_frac: f32,
}

fn default_subdivision_level() -> u32 {
    6
}
fn default_seed_density() -> f32 {
    6.0
}
fn default_walkers_per_seed() -> u32 {
    400
}
fn default_walker_step_budget() -> u32 {
    400
}
fn default_launch_cap_km() -> f32 {
    500.0
}
fn default_ridge_falloff_km() -> f32 {
    60.0
}
fn default_depth_saturation() -> f32 {
    5.0
}
fn default_samples_per_edge() -> u32 {
    8
}
fn default_midpoint_jitter_frac() -> f32 {
    0.4
}

impl Stage for OrogenDla {
    fn name(&self) -> &str {
        "orogen_dla"
    }
    fn dependencies(&self) -> &[&str] {
        &["tectonics"]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        let seed = builder.stage_seed();
        let body_radius_m = builder.radius_m;
        let res = builder.cubemap_resolution;

        let plate_map = builder
            .plates
            .as_ref()
            .expect("OrogenDla requires Plates+Tectonics (builder.plates is None)");

        let ico = Icosphere::new(self.subdivision_level);

        // Collect convergent boundary pairs and the texel directions that
        // lie on each pair's Voronoi edge. A texel is "on" the pair's edge
        // if itself is plate A and a 4-neighbour is plate B (or vice-versa).
        let convergent_pairs: Vec<(u16, u16, f32, f32)> = plate_map
            .boundaries
            .iter()
            .filter(|b| b.kind == BoundaryKind::Convergent && b.cumulative_orogeny > 0.0)
            .map(|b| (
                b.plates.0,
                b.plates.1,
                b.cumulative_orogeny,
                b.establishment_age_myr,
            ))
            .collect();

        let pair_lookup: HashMap<(u16, u16), usize> = convergent_pairs
            .iter()
            .enumerate()
            .map(|(i, (a, b, _, _))| ((*a, *b), i))
            .collect();

        let mut boundary_dirs: Vec<Vec<Vec3>> = vec![Vec::new(); convergent_pairs.len()];
        let inv_res = 1.0 / res as f32;
        let plate_ids = &plate_map.plate_id_cubemap;
        for face in CubemapFace::ALL {
            let data = plate_ids.face_data(face);
            for y in 0..res {
                for x in 0..res {
                    let p = data[(y * res + x) as usize];
                    let u = (x as f32 + 0.5) * inv_res;
                    let v = (y as f32 + 0.5) * inv_res;
                    let dir = face_uv_to_dir(face, u, v);
                    if x + 1 < res {
                        let q = data[(y * res + x + 1) as usize];
                        if p != q {
                            if let Some(&pi) = pair_lookup.get(&pair_key(p, q)) {
                                boundary_dirs[pi].push(dir);
                            }
                        }
                    }
                    if y + 1 < res {
                        let q = data[((y + 1) * res + x) as usize];
                        if p != q {
                            if let Some(&pi) = pair_lookup.get(&pair_key(p, q)) {
                                boundary_dirs[pi].push(dir);
                            }
                        }
                    }
                }
            }
        }

        let body_radius_km = body_radius_m / 1000.0;
        let launch_cap_rad = self.launch_cap_km / body_radius_km;
        let launch_cap_cos = launch_cap_rad.cos();
        let walkers_per_seed = self.walkers_per_seed;
        let step_budget = self.walker_step_budget;
        let seed_density = self.seed_density;

        // Parallel across boundaries. Each returns its own cluster (no
        // shared mutable state). Results merged deterministically below.
        let boundary_results: Vec<BoundaryDlaResult> = (0..convergent_pairs.len())
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|bidx| {
                let (plate_a, plate_b, cumulative, age_myr) = convergent_pairs[bidx];
                let boundary_seed = splitmix64(
                    seed
                        ^ (plate_a as u64).wrapping_mul(0xABCD_1234_5678_9ABC)
                        ^ (plate_b as u64).wrapping_mul(0xDEAD_BEEF_CAFE_1234),
                );
                run_boundary_dla(
                    &ico,
                    &boundary_dirs[bidx],
                    cumulative,
                    age_myr,
                    seed_density,
                    walkers_per_seed,
                    step_budget,
                    launch_cap_cos,
                    boundary_seed,
                )
            })
            .collect();

        // Zero out the destination cubemaps before stamping.
        {
            let oi = &mut builder.orogen_intensity;
            let oa = &mut builder.orogen_age_myr;
            for face in CubemapFace::ALL {
                for v in oi.face_data_mut(face).iter_mut() {
                    *v = 0.0;
                }
                for v in oa.face_data_mut(face).iter_mut() {
                    *v = 0.0;
                }
            }
        }

        let depth_saturation = self.depth_saturation.max(1.0);
        let stamp_radius_km = self.ridge_falloff_km;
        let samples_per_edge = self.samples_per_edge.max(2);
        let midpoint_jitter_frac = self.midpoint_jitter_frac.max(0.0);

        // Rasterize per-boundary, serially. Each cluster cell with a
        // parent contributes a parent→child edge, stamped as a jittered
        // chain of sub-points so the ridge reads as a continuous line
        // instead of a single Gaussian bubble. Seeds with no children
        // also get a fallback point stamp (childless-seed safety net).
        let oi = &mut builder.orogen_intensity;
        let oa = &mut builder.orogen_age_myr;
        for (bidx, result) in boundary_results.iter().enumerate() {
            // Track which seeds have at least one child so we can stamp
            // orphans at the end.
            let mut seed_has_child: Vec<bool> = vec![false; ico.vertices.len()];
            for (vidx, cell) in result.cluster.iter().enumerate() {
                let Some(c) = cell else {
                    continue;
                };
                let Some(parent_idx) = result.parents[vidx] else {
                    continue; // seed
                };
                let parent_cell = result.cluster[parent_idx as usize]
                    .expect("parent must be a cluster cell");
                seed_has_child[parent_idx as usize] = true;

                let a_pos = ico.vertices[parent_idx as usize];
                let b_pos = ico.vertices[vidx];
                let a_weight = depth_weight(parent_cell.depth, depth_saturation);
                let b_weight = depth_weight(c.depth, depth_saturation);

                // Unique per-edge jitter seed so the same parent→child
                // pair always jitters the same way (determinism) but
                // different edges decorrelate.
                let edge_seed = splitmix64(
                    (bidx as u64).wrapping_mul(0xC2B2_AE35_4E1F_8901)
                        ^ (parent_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ (vidx as u64).wrapping_mul(0xD1B5_4A32_D192_7F5D),
                );
                rasterize_edge(
                    a_pos,
                    b_pos,
                    a_weight,
                    b_weight,
                    c.age_myr,
                    samples_per_edge,
                    midpoint_jitter_frac,
                    stamp_radius_km,
                    body_radius_km,
                    res,
                    edge_seed,
                    oi,
                    oa,
                );
            }

            // Childless seeds: fallback stamp so they don't vanish.
            for (vidx, cell) in result.cluster.iter().enumerate() {
                let Some(c) = cell else {
                    continue;
                };
                if result.parents[vidx].is_some() {
                    continue; // has parent, already drawn via its edge
                }
                if seed_has_child[vidx] {
                    continue; // drawn as endpoint of a child's edge
                }
                let w = depth_weight(c.depth, depth_saturation);
                stamp_point(
                    ico.vertices[vidx],
                    w,
                    c.age_myr,
                    stamp_radius_km,
                    body_radius_km,
                    res,
                    oi,
                    oa,
                );
            }
        }
    }
}

/// Depth → saturating weight. `raw / (raw + 1)` in `[0, 1)`.
#[inline]
fn depth_weight(depth: u32, depth_saturation: f32) -> f32 {
    let raw = depth as f32 / depth_saturation;
    raw / (raw + 1.0)
}

/// Stamp a single ridge point onto the cubemap with a sharp quartic
/// profile `(1 − r²)²` that peaks at the sample position and falls to
/// zero at `stamp_radius_km`. Uses max-accumulation so overlapping stamps
/// don't saturate past 1.0.
fn stamp_point(
    pos: Vec3,
    weight: f32,
    age_myr: f32,
    stamp_radius_km: f32,
    body_radius_km: f32,
    res: u32,
    oi: &mut crate::cubemap::Cubemap<f32>,
    oa: &mut crate::cubemap::Cubemap<f32>,
) {
    let stamp_radius_rad = stamp_radius_km / body_radius_km;
    for face in CubemapFace::ALL {
        let oi_face = oi.face_data_mut(face);
        let oa_face = oa.face_data_mut(face);
        for_face_texels_in_cap(
            face,
            res,
            pos,
            stamp_radius_rad,
            |x, y, _dir, angular_dist| {
                let dist_km = angular_dist * body_radius_km;
                let r = dist_km / stamp_radius_km;
                if r >= 1.0 {
                    return;
                }
                let p = 1.0 - r * r;
                let falloff = p * p;
                let contribution = weight * falloff;
                let idx = (y * res + x) as usize;
                if contribution > oi_face[idx] {
                    oi_face[idx] = contribution;
                    oa_face[idx] = age_myr;
                }
            },
        );
    }
}

/// Rasterize a parent→child ridge. The edge is split into
/// `samples_per_edge` sub-segments at jittered interior midpoints, then
/// each sub-segment is rasterized as a great-circle arc with a continuous
/// cross-section — so the ridge reads as a thick continuous line rather
/// than a chain of point stamps. Jitter perpendicular to the edge gives
/// each ridge a natural wiggle away from the underlying geodesic.
#[allow(clippy::too_many_arguments)]
fn rasterize_edge(
    a_pos: Vec3,
    b_pos: Vec3,
    a_weight: f32,
    b_weight: f32,
    age_myr: f32,
    samples_per_edge: u32,
    midpoint_jitter_frac: f32,
    stamp_radius_km: f32,
    body_radius_km: f32,
    res: u32,
    edge_seed: u64,
    oi: &mut crate::cubemap::Cubemap<f32>,
    oa: &mut crate::cubemap::Cubemap<f32>,
) {
    let arc_cos = a_pos.dot(b_pos).clamp(-1.0, 1.0);
    let arc_total = arc_cos.acos();
    if arc_total < 1e-6 {
        stamp_point(
            a_pos,
            a_weight.max(b_weight),
            age_myr,
            stamp_radius_km,
            body_radius_km,
            res,
            oi,
            oa,
        );
        return;
    }
    let edge_len_km = arc_total * body_radius_km;
    let jitter_amp_rad = edge_len_km * midpoint_jitter_frac / body_radius_km;

    // Build `samples_per_edge + 1` waypoints along the edge: endpoints
    // pinned, interior points slerp'd then perpendicularly jittered.
    let n_points = (samples_per_edge as usize + 1).max(2);
    let mut waypoints: Vec<Vec3> = Vec::with_capacity(n_points);
    let mut weights: Vec<f32> = Vec::with_capacity(n_points);

    let mid = ((a_pos + b_pos) * 0.5).normalize();
    let edge_dir = (b_pos - a_pos).normalize_or_zero();
    let t_edge = (edge_dir - mid * edge_dir.dot(mid)).normalize_or_zero();
    let t_perp = mid.cross(t_edge).normalize_or_zero();
    let mut rng = Rng::new(edge_seed);
    let sinarc = arc_total.sin().max(1e-6);
    for i in 0..n_points {
        let t = i as f32 / (n_points - 1) as f32;
        let sa = ((1.0 - t) * arc_total).sin() / sinarc;
        let sb = (t * arc_total).sin() / sinarc;
        let mut pos = a_pos * sa + b_pos * sb;
        if i > 0 && i < n_points - 1 && jitter_amp_rad > 0.0 {
            let u = rng.next_f64_signed() as f32 * jitter_amp_rad;
            let v = rng.next_f64_signed() as f32 * jitter_amp_rad;
            pos += t_edge * u + t_perp * v;
        }
        waypoints.push(pos.normalize());
        weights.push(a_weight * (1.0 - t) + b_weight * t);
    }

    // Rasterize each sub-arc as a continuous line segment.
    for i in 0..(n_points - 1) {
        rasterize_arc_segment(
            waypoints[i],
            waypoints[i + 1],
            weights[i],
            weights[i + 1],
            age_myr,
            stamp_radius_km,
            body_radius_km,
            res,
            oi,
            oa,
        );
    }
}

/// Rasterize a single great-circle arc segment with a uniform quartic
/// cross-section profile. For every texel inside the enclosing cap,
/// computes the angular distance to the arc (perpendicular when the
/// projection falls on the arc, else distance to the nearer endpoint)
/// and applies `(1 − r²)²` where `r = dist_km / stamp_radius_km`.
///
/// This is what converts a point-stamp chain into a continuous ridge —
/// no inter-stamp gaps or peaks, just a thick line with smooth flanks.
#[allow(clippy::too_many_arguments)]
fn rasterize_arc_segment(
    a: Vec3,
    b: Vec3,
    a_weight: f32,
    b_weight: f32,
    age_myr: f32,
    stamp_radius_km: f32,
    body_radius_km: f32,
    res: u32,
    oi: &mut crate::cubemap::Cubemap<f32>,
    oa: &mut crate::cubemap::Cubemap<f32>,
) {
    let arc_cos = a.dot(b).clamp(-1.0, 1.0);
    let arc_rad = arc_cos.acos();
    if arc_rad < 1e-6 {
        stamp_point(
            a,
            a_weight.max(b_weight),
            age_myr,
            stamp_radius_km,
            body_radius_km,
            res,
            oi,
            oa,
        );
        return;
    }
    let plane_cross = a.cross(b);
    let plane_len_sq = plane_cross.length_squared();
    if plane_len_sq < 1e-12 {
        return;
    }
    let plane_n = plane_cross / plane_len_sq.sqrt();

    let stamp_radius_rad = stamp_radius_km / body_radius_km;
    let mid = ((a + b) * 0.5).normalize();
    let cap_half_angle = arc_rad * 0.5 + stamp_radius_rad;

    for face in CubemapFace::ALL {
        let oi_face = oi.face_data_mut(face);
        let oa_face = oa.face_data_mut(face);
        for_face_texels_in_cap(face, res, mid, cap_half_angle, |x, y, dir, _| {
            let (dist_rad, t) = arc_distance(dir, a, b, plane_n, arc_rad);
            let dist_km = dist_rad * body_radius_km;
            if dist_km >= stamp_radius_km {
                return;
            }
            let r = dist_km / stamp_radius_km;
            let p = 1.0 - r * r;
            let profile = p * p;
            let weight = a_weight * (1.0 - t) + b_weight * t;
            let contribution = weight * profile;
            let idx = (y * res + x) as usize;
            if contribution > oi_face[idx] {
                oi_face[idx] = contribution;
                oa_face[idx] = age_myr;
            }
        });
    }
}

/// Angular distance from `p` to the great-circle arc `a → b` (shorter
/// arc), plus an arc parameter `t ∈ [0, 1]` at the closest point.
///
/// Algorithm:
/// 1. Decompose `p` into perpendicular (normal to the arc's great-circle
///    plane) and in-plane components.
/// 2. Project onto the great circle → `p_on_circle`.
/// 3. If the projection lies within the arc's angular span (checked via
///    `angle(a, p_proj) ≤ arc_rad` AND `angle(b, p_proj) ≤ arc_rad`),
///    the closest point on the arc is `p_on_circle`, and distance is
///    `asin(|perp|)`.
/// 4. Otherwise the closest point is the nearer endpoint.
fn arc_distance(p: Vec3, a: Vec3, b: Vec3, plane_n: Vec3, arc_rad: f32) -> (f32, f32) {
    let perp = p.dot(plane_n);
    let in_plane = p - plane_n * perp;
    let in_plane_len_sq = in_plane.length_squared();
    let (dist_to_circle, t_on_circle) = if in_plane_len_sq > 1e-12 {
        let p_on_circle = in_plane / in_plane_len_sq.sqrt();
        let cos_a_proj = a.dot(p_on_circle).clamp(-1.0, 1.0);
        let cos_b_proj = b.dot(p_on_circle).clamp(-1.0, 1.0);
        let ang_a = cos_a_proj.acos();
        let ang_b = cos_b_proj.acos();
        // Projection on the short arc iff both endpoint angles are ≤ arc_rad.
        if ang_a <= arc_rad && ang_b <= arc_rad {
            (perp.abs().asin(), (ang_a / arc_rad).clamp(0.0, 1.0))
        } else {
            // Outside arc — signal "fall through to endpoint distance".
            (f32::INFINITY, if ang_a < ang_b { 0.0 } else { 1.0 })
        }
    } else {
        // p is at the plane's pole — equidistant π/2 from every arc point.
        (std::f32::consts::FRAC_PI_2, 0.5)
    };

    let cos_d_a = p.dot(a).clamp(-1.0, 1.0);
    let cos_d_b = p.dot(b).clamp(-1.0, 1.0);
    let d_a = cos_d_a.acos();
    let d_b = cos_d_b.acos();
    let (endpoint_dist, endpoint_t) = if d_a < d_b { (d_a, 0.0) } else { (d_b, 1.0) };

    if dist_to_circle < endpoint_dist {
        (dist_to_circle, t_on_circle)
    } else {
        (endpoint_dist, endpoint_t)
    }
}

struct BoundaryDlaResult {
    /// Per-icosphere-vertex cluster cell (None if vertex is not in the
    /// cluster for this boundary).
    cluster: Vec<Option<ClusterCell>>,
    /// Per-icosphere-vertex parent index. `None` for seeds (no incoming
    /// edge) and for non-cluster cells. Walking this from a cluster cell
    /// traces the back-edge to its parent; used by the edge-based
    /// rasterizer to stamp ridges as continuous lines.
    parents: Vec<Option<u32>>,
}

#[derive(Clone, Copy)]
struct ClusterCell {
    age_myr: f32,
    /// Downstream-depth: 1 at leaves, `1 + max(child.depth)` elsewhere.
    /// Seeds end up at the max depth (longest downstream branch), which
    /// is exactly what we want for mountain-peak elevation.
    depth: u32,
}

#[allow(clippy::too_many_arguments)]
fn run_boundary_dla(
    ico: &Icosphere,
    boundary_dirs: &[Vec3],
    cumulative: f32,
    age_myr: f32,
    seed_density: f32,
    walkers_per_seed: u32,
    step_budget: u32,
    launch_cap_cos: f32,
    boundary_seed: u64,
) -> BoundaryDlaResult {
    let n_verts = ico.vertices.len();
    let mut cluster: Vec<Option<ClusterCell>> = vec![None; n_verts];
    let mut parents: Vec<Option<u32>> = vec![None; n_verts];
    let mut attached_order: Vec<u32> = Vec::new();

    if boundary_dirs.is_empty() {
        return BoundaryDlaResult { cluster, parents };
    }

    // Seeds: spread along the boundary arc via evenly-stepped sampling.
    // `cumulative_orogeny ∈ [0, 1]` scales seed count so stronger
    // boundaries get more ridge material. Dedup seeds that snap to the
    // same icosphere vertex.
    let n_seeds = (seed_density * cumulative).round().max(1.0) as usize;
    let n_seeds = n_seeds.min(boundary_dirs.len());
    let mut seeds: Vec<u32> = Vec::with_capacity(n_seeds);
    for i in 0..n_seeds {
        let d = boundary_dirs[i * boundary_dirs.len() / n_seeds];
        let v = ico.nearest_vertex(d);
        if !seeds.contains(&v) {
            seeds.push(v);
        }
    }
    for &vidx in &seeds {
        cluster[vidx as usize] = Some(ClusterCell {
            age_myr,
            depth: 1,
        });
        attached_order.push(vidx);
    }

    let mut rng = Rng::new(boundary_seed);
    let total_walkers = walkers_per_seed as usize * seeds.len();
    for w in 0..total_walkers {
        let anchor_idx = seeds[w % seeds.len()];
        let anchor_pos = ico.vertices[anchor_idx as usize];
        // Sample launch position inside the cap around the anchor seed.
        // Rejection sample against the icosphere vertex set.
        let Some(mut pos) = sample_cap_vertex(ico, anchor_pos, launch_cap_cos, &mut rng) else {
            continue;
        };
        if cluster[pos as usize].is_some() {
            continue; // launched on already-cluster cell; abandon
        }

        let mut attached_parent: Option<u32> = None;
        for _ in 0..step_budget {
            let nbs = &ico.vertex_neighbors[pos as usize];
            // Attachment check: any neighbour already in cluster?
            let mut stuck_to: Option<u32> = None;
            for &nb in nbs {
                if cluster[nb as usize].is_some() {
                    stuck_to = Some(nb);
                    break;
                }
            }
            if let Some(p) = stuck_to {
                attached_parent = Some(p);
                break;
            }
            // Step to random neighbour.
            let choice = (rng.next_u64() as usize) % nbs.len();
            pos = nbs[choice];
            // Out-of-cap → abandon. Stops walkers from bleeding into
            // other boundaries' territory.
            if ico.vertices[pos as usize].dot(anchor_pos) < launch_cap_cos {
                break;
            }
        }

        if let Some(parent) = attached_parent {
            let parent_age = cluster[parent as usize].unwrap().age_myr;
            cluster[pos as usize] = Some(ClusterCell {
                age_myr: parent_age,
                depth: 1,
            });
            parents[pos as usize] = Some(parent);
            attached_order.push(pos);
        }
    }

    // Compute downstream-depth via bottom-up traversal in reverse
    // attachment order. Because a child always attaches after its parent,
    // reversing `attached_order` visits children before parents — the
    // correct evaluation order for `depth = 1 + max(child.depth)`.
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); n_verts];
    for &vidx in &attached_order {
        if let Some(parent) = parents[vidx as usize] {
            children[parent as usize].push(vidx);
        }
    }
    for &vidx in attached_order.iter().rev() {
        let max_child_depth = children[vidx as usize]
            .iter()
            .map(|&c| cluster[c as usize].unwrap().depth)
            .max();
        if let Some(m) = max_child_depth {
            cluster[vidx as usize].as_mut().unwrap().depth = m + 1;
        }
    }

    BoundaryDlaResult { cluster, parents }
}

/// Rejection-sample a random icosphere vertex that falls inside the
/// spherical cap. Returns `None` after a hard attempt cap — relevant only
/// for degenerate tiny caps that contain no icosphere vertex at all.
fn sample_cap_vertex(
    ico: &Icosphere,
    center: Vec3,
    launch_cap_cos: f32,
    rng: &mut Rng,
) -> Option<u32> {
    for _ in 0..200 {
        let idx = (rng.next_u64() as usize) % ico.vertices.len();
        if ico.vertices[idx].dot(center) >= launch_cap_cos {
            return Some(idx as u32);
        }
    }
    None
}

fn pair_key(a: u16, b: u16) -> (u16, u16) {
    if a < b { (a, b) } else { (b, a) }
}
