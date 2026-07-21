//! Plates: identity, seeding, flood-fill assignment, oceanic/continental
//! flagging, and Euler-pole motion encoding.
//!
//! A plate is a connected set of mesh cells that share an Euler pole. The
//! flood-fill is a random-order BFS in Red Blob Games' style — each step
//! pulls a random frontier cell (not the front or back, which would produce
//! BFS or DFS), expands one of its unclaimed neighbors, and continues until
//! every cell is assigned. The resulting plate sizes are non-uniform, which
//! matches Earth's geology (Pacific is huge, Caribbean is tiny).
//!
//! All randomness flows from the seed; `Vec` frontiers with `swap_remove`
//! keep the iteration order deterministic across runs.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use super::config::{TectonicActivity, TectonicConfig};
use super::mesh::SphericalMesh;
use crate::seeding::{Rng, sub_seed};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
pub struct PlateId(pub u32);

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum PlateKind {
    Continental,
    Oceanic,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Plate {
    pub id: PlateId,
    /// Mesh cell index this plate was seeded from.
    pub seed_cell: u32,
    /// Mean of the plate's cell directions, normalized. Useful for placing
    /// editor overlays (motion-arrow origins, plate labels). Not guaranteed
    /// to lie inside the plate for irregular shapes — switch to
    /// "cell with max distance to nearest boundary" if mis-placement
    /// becomes visually obvious.
    pub centroid_dir: Vec3,
    pub kind: PlateKind,
    /// Unit vector through the sphere center about which this plate rotates.
    pub euler_pole: Vec3,
    /// Angular speed around `euler_pole`, signed (right-hand rule). Earth's
    /// fastest plates rotate at ~0.05–0.1 deg/Myr ≈ 9e-10 to 2e-9 rad/yr.
    /// We pick from a small range; the exact magnitude only affects the
    /// classification threshold tuning.
    pub omega_rad_per_year: f32,
}

/// Result of plate seeding + flood-fill.
pub struct PlateAssignment {
    pub plates: Vec<Plate>,
    /// Per mesh-cell plate id (always `Some` after a successful assignment).
    pub cell_plate: Vec<PlateId>,
}

/// Sub-seed stage names used by this module. Centralized so that adding a
/// new randomization step doesn't disturb existing streams.
const STAGE_SEED_SELECT: &str = "tectonics.plate_seed_select";
const STAGE_EULER: &str = "tectonics.euler_poles";
const STAGE_FLOOD: &str = "tectonics.flood_fill";

/// Number of candidates evaluated per Mitchell's best-candidate pick. Higher
/// gives tighter Poisson-disc-like distribution but more compute. 16 is the
/// quality/cost sweet spot — distributions visibly improve up to ~10 then
/// plateau, and at our typical mesh size (≤ 8k cells) the cost is trivial.
const MITCHELL_CANDIDATES: usize = 16;

/// Assign every mesh cell to a plate. Drives plate seed selection, oceanic/
/// continental flagging, Euler-pole assignment, and the random-order BFS.
pub fn assign_plates(
    mesh: &SphericalMesh,
    config: &TectonicConfig,
    root_seed: u64,
) -> PlateAssignment {
    let n_cells = mesh.cells.len();
    assert!(n_cells > 0, "tectonics: empty mesh");
    let plate_count = (config.plate_count as usize).clamp(1, n_cells);
    let n_continental =
        ((plate_count as f32) * config.continental_fraction.clamp(0.0, 1.0)).floor() as usize;
    let n_continental = n_continental.min(plate_count);
    let n_oceanic = plate_count - n_continental;

    // 1. Pick plate seed cells via Mitchell's best-candidate sampling.
    //    Random selection produces clustered seeds — Wendel's theorem says
    //    3 random points on a sphere ALWAYS lie in some hemisphere, and
    //    independent random samples have no mutual repulsion so plates
    //    naturally bunch up. Mitchell's picks the first seed randomly,
    //    then for each subsequent seed evaluates K candidates and keeps
    //    the one farthest from already-placed seeds.
    //
    //    Continental seeds are placed first — this guarantees continents
    //    are spread across the sphere, not just the plates. Oceanic seeds
    //    fill in afterward, treating continental seeds as repulsion
    //    constraints so the two sets interleave evenly.
    //
    //    Authored `seed_dirs` snap to the nearest mesh cell and bypass
    //    the candidate evaluation entirely, but they still feed into the
    //    repulsion set for subsequent picks.
    let mut rng_seeds = Rng::new(sub_seed(root_seed, STAGE_SEED_SELECT));
    let mut continental_cells: Vec<u32> = Vec::with_capacity(n_continental);
    let mut oceanic_cells: Vec<u32> = Vec::with_capacity(n_oceanic);

    // Authored seeds first, capped to the continental count. Authored
    // seeds for oceanic placement are not currently expressed in the
    // schema; reserve the syntax for later.
    if let Some(authored) = &config.seed_dirs {
        for &dir in authored.iter().take(n_continental) {
            let cell = mesh.nearest(dir.normalize_or_zero());
            if !continental_cells.contains(&cell) {
                continental_cells.push(cell);
            }
        }
    }

    // Continental seed strategy:
    //
    // 1. Primary seed: Mitchell's against (empty or authored) seed set, with
    //    a `|dir.y| * equatorial_bias` penalty so equator picks beat polar
    //    picks. With `equatorial_bias = 0` and no authored seed this collapses
    //    to a uniform random pick (current behavior).
    //
    // 2. Secondary seeds: drawn from a spherical cap around the primary,
    //    radius determined by `continental_clustering` (0 → full sphere,
    //    1 → ~30°). Mitchell repulsion among the in-cap candidates spaces
    //    the cluster out so they don't collide. With `clustering = 0` this
    //    is identical to the current Mitchell-only behavior.
    //
    // 3. Outlier (when `clustering > 0` and we have ≥3 continentals): the
    //    last continental seed is placed via global Mitchell repulsion
    //    against the cluster, producing an isolated landmass off in the
    //    far hemisphere — the "Australia" or "Antarctica" of the body.
    let clustering = config.continental_clustering.clamp(0.0, 1.0);
    let equatorial_bias = config.equatorial_bias.max(0.0);

    // Reserve an outlier slot when clustering is on and we have room for
    // a meaningful "primary + ≥1 cluster + 1 outlier" partition.
    let n_outlier = if clustering > 0.0 && n_continental >= 3 {
        1
    } else {
        0
    };

    // Primary: equator-biased Mitchell against any authored seeds. Skipped
    // when authored seeds already filled the primary slot.
    if continental_cells.is_empty() && n_continental > 0 {
        let primary =
            pick_primary_seed(mesh, &[&continental_cells], &mut rng_seeds, equatorial_bias);
        continental_cells.push(primary);
    }

    // Cluster seeds: fill up to `n_continental - n_outlier`. With
    // `clustering > 0`, candidates are drawn from a spherical cap around
    // the primary (continental_cells[0]); otherwise it's the original
    // global Mitchell repulsion.
    let cluster_target = n_continental.saturating_sub(n_outlier);
    let primary_dir = continental_cells
        .first()
        .map(|&c| mesh.cells[c as usize])
        .unwrap_or(Vec3::Z);
    let cluster_cos_min = clustering_cap_cos(clustering);
    while continental_cells.len() < cluster_target {
        let next = if clustering > 0.0 {
            mitchell_in_cap(
                mesh,
                primary_dir,
                cluster_cos_min,
                &[&continental_cells],
                &mut rng_seeds,
                MITCHELL_CANDIDATES,
            )
        } else {
            mitchell_best_candidate(
                mesh,
                &[&continental_cells],
                &mut rng_seeds,
                MITCHELL_CANDIDATES,
            )
        };
        continental_cells.push(next);
    }

    // Outlier: global Mitchell repulsion against the cluster.
    while continental_cells.len() < n_continental {
        let next = mitchell_best_candidate(
            mesh,
            &[&continental_cells],
            &mut rng_seeds,
            MITCHELL_CANDIDATES,
        );
        continental_cells.push(next);
    }

    // Oceanic seeds: same algorithm, but treat continental seeds as
    // additional repulsion constraints so oceanic plates fill the gaps.
    while oceanic_cells.len() < n_oceanic {
        let next = mitchell_best_candidate(
            mesh,
            &[&continental_cells, &oceanic_cells],
            &mut rng_seeds,
            MITCHELL_CANDIDATES,
        );
        oceanic_cells.push(next);
    }

    // Combine into a single seed list; record kinds in matching order so
    // step 3 (Euler poles) and step 4 (flood fill) see consistent indices.
    let mut seed_cells: Vec<u32> = Vec::with_capacity(plate_count);
    let mut plate_kinds: Vec<PlateKind> = Vec::with_capacity(plate_count);
    for &c in &continental_cells {
        seed_cells.push(c);
        plate_kinds.push(PlateKind::Continental);
    }
    for &c in &oceanic_cells {
        seed_cells.push(c);
        plate_kinds.push(PlateKind::Oceanic);
    }

    // 3. Euler poles + omega. Random unit vector through the body's center,
    //    angular speed in [-OMEGA_MAX, +OMEGA_MAX] rad/yr. The magnitude
    //    is small but consistent across activity modes; classification
    //    only cares about relative motion at boundaries.
    const OMEGA_MAX: f32 = 1.0e-8; // ~0.6 deg/Myr — Earth's faster plates
    let mut rng_euler = Rng::new(sub_seed(root_seed, STAGE_EULER));
    let plates: Vec<Plate> = (0..plate_count)
        .map(|i| {
            let dv = rng_euler.unit_vector();
            let euler_pole = Vec3::new(dv.x as f32, dv.y as f32, dv.z as f32);
            let omega = (rng_euler.next_f64_signed() as f32) * OMEGA_MAX;
            Plate {
                id: PlateId(i as u32),
                seed_cell: seed_cells[i],
                centroid_dir: mesh.cells[seed_cells[i] as usize], // refined after fill
                kind: plate_kinds[i],
                euler_pole,
                omega_rad_per_year: omega,
            }
        })
        .collect();

    // 4. Round-robin BFS flood fill — each plate gets one cell-expansion
    //    per round (primary continental gets `primary_size_multiplier`),
    //    until every plate's frontier is exhausted. Random-order BFS (Red
    //    Blob style: pick a random frontier cell from a shared pool,
    //    expand it) has a "rich get richer" dynamic — plates with larger
    //    frontiers are picked more often and grow faster, producing one
    //    or two megaplates plus many slivers. Round-robin with a per-plate
    //    frontier guarantees all plates grow at the same rate, producing
    //    comparable plate sizes regardless of how the BFS lottery would
    //    have unfolded.
    //
    //    The primary continental plate (always index 0 of the combined
    //    list) optionally gets `primary_size_multiplier` expansions per
    //    round instead of one, producing a dominant supercontinent. The
    //    multiplier is clamped to [1, 4] so the primary cannot eat the
    //    sphere.
    //
    //    Within a plate's expansion turn, the choice of *which* frontier
    //    cell to expand is still randomized (so plate shapes aren't
    //    perfect circles). The randomness is per-plate-per-round, which
    //    keeps the global determinism property (same seed → same output).
    let primary_extra_expansions =
        (config.primary_size_multiplier.clamp(1.0, 4.0) - 1.0).floor() as usize;
    let primary_idx_for_growth: Option<usize> = if n_continental > 0 { Some(0) } else { None };

    let mut cell_plate: Vec<i32> = vec![-1; n_cells];
    let mut plate_frontiers: Vec<Vec<u32>> = vec![Vec::new(); plate_count];
    for (plate_idx, &cell) in seed_cells.iter().enumerate() {
        cell_plate[cell as usize] = plate_idx as i32;
        plate_frontiers[plate_idx].push(cell);
    }
    let mut rng_flood = Rng::new(sub_seed(root_seed, STAGE_FLOOD));
    loop {
        let mut any_expansion = false;
        for (plate_idx, frontier) in plate_frontiers.iter_mut().enumerate().take(plate_count) {
            let expansions = if Some(plate_idx) == primary_idx_for_growth {
                1 + primary_extra_expansions
            } else {
                1
            };
            for _ in 0..expansions {
                if try_expand_plate(plate_idx, mesh, frontier, &mut cell_plate, &mut rng_flood) {
                    any_expansion = true;
                } else {
                    // Frontier exhausted — no point trying more expansions
                    // this round for this plate.
                    break;
                }
            }
        }
        if !any_expansion {
            break;
        }
    }

    // 5. Refine plate centroids: mean of member-cell directions, normalized.
    let mut sum: Vec<Vec3> = vec![Vec3::ZERO; plate_count];
    for (cell_idx, &p) in cell_plate.iter().enumerate() {
        debug_assert!(p >= 0, "cell {cell_idx} unassigned after flood fill");
        sum[p as usize] += mesh.cells[cell_idx];
    }
    let mut plates = plates;
    for (i, plate) in plates.iter_mut().enumerate() {
        plate.centroid_dir = sum[i].normalize_or_zero();
        if plate.centroid_dir == Vec3::ZERO {
            // Pathological case (cells cancel exactly). Fall back to seed.
            plate.centroid_dir = mesh.cells[plate.seed_cell as usize];
        }
    }

    let cell_plate: Vec<PlateId> = cell_plate.iter().map(|&p| PlateId(p as u32)).collect();
    PlateAssignment { plates, cell_plate }
}

/// Mitchell's best-candidate sampler on a spherical mesh. Picks a cell
/// far from all already-placed cells (across `existing` slices) by
/// evaluating `n_candidates` random cells and keeping the one with the
/// largest minimum angular distance to any existing seed.
///
/// `existing: &[&[u32]]` lets the caller treat multiple seed lists as one
/// unified repulsion set — useful when oceanic seeds need to repel from
/// continental ones too.
///
/// When `existing` is empty (first pick), the chosen cell is purely
/// random — no repulsion constraint to satisfy.
fn mitchell_best_candidate(
    mesh: &SphericalMesh,
    existing: &[&[u32]],
    rng: &mut Rng,
    n_candidates: usize,
) -> u32 {
    let n_cells = mesh.cells.len();
    let any_existing = existing.iter().any(|s| !s.is_empty());

    if !any_existing {
        return (rng.next_u64() % n_cells as u64) as u32;
    }

    let mut best_cell: u32 = 0;
    let mut best_min_dot: f32 = f32::INFINITY;
    let mut found = false;
    for _ in 0..n_candidates {
        let candidate = (rng.next_u64() % n_cells as u64) as u32;
        // Skip if already taken.
        let already = existing.iter().any(|slice| slice.contains(&candidate));
        if already {
            continue;
        }
        let cand_dir = mesh.cells[candidate as usize];
        // Find max dot product with any existing seed → that's the closest
        // existing seed by angle. We want to MINIMIZE this max-dot, i.e.
        // place the candidate as far from its nearest neighbor as possible.
        let mut max_dot: f32 = f32::NEG_INFINITY;
        for slice in existing {
            for &s in *slice {
                let dot = cand_dir.dot(mesh.cells[s as usize]);
                if dot > max_dot {
                    max_dot = dot;
                }
            }
        }
        if max_dot < best_min_dot {
            best_min_dot = max_dot;
            best_cell = candidate;
            found = true;
        }
    }

    // Fallback: every candidate collided with an existing seed (extremely
    // unlikely unless n_cells ≈ existing.len()). Pick the next available
    // cell linearly.
    if !found {
        for i in 0..n_cells {
            let candidate = i as u32;
            let already = existing.iter().any(|slice| slice.contains(&candidate));
            if !already {
                return candidate;
            }
        }
        // Mesh fully consumed — should never happen in practice given the
        // n_continental + n_oceanic ≤ plate_count ≤ n_cells invariant.
        return 0;
    }

    best_cell
}

/// Try to expand a single plate's frontier by one cell. Returns true if a
/// cell was claimed; false if the frontier is exhausted.
fn try_expand_plate(
    plate_idx: usize,
    mesh: &SphericalMesh,
    frontier: &mut Vec<u32>,
    cell_plate: &mut [i32],
    rng: &mut Rng,
) -> bool {
    while !frontier.is_empty() {
        let frontier_len = frontier.len();
        let i = (rng.next_u64() as usize) % frontier_len;
        let cell = frontier[i];
        let mut chosen_neighbor: Option<u32> = None;
        for &neighbor in &mesh.neighbors[cell as usize] {
            if cell_plate[neighbor as usize] < 0 {
                chosen_neighbor = Some(neighbor);
                break;
            }
        }
        if let Some(neighbor) = chosen_neighbor {
            cell_plate[neighbor as usize] = plate_idx as i32;
            frontier.push(neighbor);
            return true;
        } else {
            // Fully surrounded — drop and try another.
            frontier.swap_remove(i);
        }
    }
    false
}

/// Pick the primary continental seed using Mitchell's best-candidate
/// repulsion against `existing` seeds, with an additive
/// `|dir.y| * equatorial_bias` penalty so equator picks beat polar picks.
///
/// `equatorial_bias = 0` collapses to a uniform random pick when `existing`
/// is empty (current behavior), or to standard Mitchell otherwise.
fn pick_primary_seed(
    mesh: &SphericalMesh,
    existing: &[&[u32]],
    rng: &mut Rng,
    equatorial_bias: f32,
) -> u32 {
    let n_cells = mesh.cells.len();
    let any_existing = existing.iter().any(|s| !s.is_empty());

    if !any_existing && equatorial_bias <= 0.0 {
        return (rng.next_u64() % n_cells as u64) as u32;
    }

    let mut best_cell: u32 = 0;
    let mut best_score: f32 = f32::INFINITY;
    let mut found = false;
    for _ in 0..MITCHELL_CANDIDATES {
        let candidate = (rng.next_u64() % n_cells as u64) as u32;
        let already = existing.iter().any(|slice| slice.contains(&candidate));
        if already {
            continue;
        }
        let cand_dir = mesh.cells[candidate as usize];
        let mut max_dot: f32 = if any_existing { f32::NEG_INFINITY } else { 0.0 };
        for slice in existing {
            for &s in *slice {
                let dot = cand_dir.dot(mesh.cells[s as usize]);
                if dot > max_dot {
                    max_dot = dot;
                }
            }
        }
        let score = max_dot + cand_dir.y.abs() * equatorial_bias;
        if score < best_score {
            best_score = score;
            best_cell = candidate;
            found = true;
        }
    }

    if !found {
        // Every candidate was already taken (extremely unlikely). Fall back
        // to ordinary Mitchell, which has its own linear-scan fallback.
        return mitchell_best_candidate(mesh, existing, rng, MITCHELL_CANDIDATES);
    }

    best_cell
}

/// Mitchell's best-candidate restricted to a spherical cap of half-angle
/// `acos(cos_min)` around `center`. Used to draw cluster seeds tightly
/// around the primary continental seed.
fn mitchell_in_cap(
    mesh: &SphericalMesh,
    center: Vec3,
    cos_min: f32,
    existing: &[&[u32]],
    rng: &mut Rng,
    n_candidates: usize,
) -> u32 {
    let mut best_cell: u32 = 0;
    let mut best_min_dot: f32 = f32::INFINITY;
    let mut found = false;
    for _ in 0..n_candidates {
        let candidate = sample_cell_in_cap(mesh, center, cos_min, rng);
        let already = existing.iter().any(|slice| slice.contains(&candidate));
        if already {
            continue;
        }
        let cand_dir = mesh.cells[candidate as usize];
        let mut max_dot: f32 = f32::NEG_INFINITY;
        for slice in existing {
            for &s in *slice {
                let dot = cand_dir.dot(mesh.cells[s as usize]);
                if dot > max_dot {
                    max_dot = dot;
                }
            }
        }
        if max_dot < best_min_dot {
            best_min_dot = max_dot;
            best_cell = candidate;
            found = true;
        }
    }

    if !found {
        // Cap too crowded — fall back to a global Mitchell pick so we
        // still get a deterministic continental seed.
        return mitchell_best_candidate(mesh, existing, rng, n_candidates);
    }
    best_cell
}

/// Sample a mesh cell whose direction lies within the spherical cap of
/// half-angle `acos(cos_min)` around `center`. Uniform in solid angle
/// within the cap, then snapped to the nearest mesh cell.
fn sample_cell_in_cap(mesh: &SphericalMesh, center: Vec3, cos_min: f32, rng: &mut Rng) -> u32 {
    use std::f32::consts::TAU;
    // Uniform sampling in solid angle: cos(theta) ~ U(cos_min, 1).
    let t = rng.next_f64() as f32;
    let cos_theta = cos_min + (1.0 - cos_min) * t;
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = TAU * (rng.next_f64() as f32);
    let (sin_phi, cos_phi) = phi.sin_cos();
    // Local frame with z = +Z; rotate to world frame so z aligns with center.
    let local = Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
    let center = center.normalize_or_zero();
    let world = if center.dot(Vec3::Z).abs() > 0.9999 {
        if center.z >= 0.0 { local } else { -local }
    } else {
        Quat::from_rotation_arc(Vec3::Z, center) * local
    };
    mesh.nearest(world)
}

/// Cosine of the spherical-cap half-angle that secondary continental seeds
/// are drawn from, as a function of the clustering parameter.
///
/// `clustering = 0.0` → cap covers the full sphere (cos = -1), so cluster
/// sampling is indistinguishable from uniform spherical sampling.
/// `clustering = 1.0` → cap of ~30° around the primary (cos = cos(π/6)).
fn clustering_cap_cos(clustering: f32) -> f32 {
    use std::f32::consts::PI;
    let t = clustering.clamp(0.0, 1.0);
    let half_angle = PI * (1.0 - t) + (PI / 6.0) * t; // lerp(PI, PI/6)
    half_angle.cos()
}

/// Activity-gated plate velocity at a unit-direction `dir` on the surface.
/// Returns tangent vector in m/yr.
pub fn surface_velocity(
    plate: &Plate,
    dir: Vec3,
    radius_m: f32,
    activity: TectonicActivity,
) -> Vec3 {
    if !activity.live_velocity() {
        return Vec3::ZERO;
    }
    raw_surface_velocity(plate, dir, radius_m)
}

/// Raw surface velocity from `omega × r`, ignoring activity. Used by boundary
/// classification (which always reads the encoded historical motion).
pub fn raw_surface_velocity(plate: &Plate, dir: Vec3, radius_m: f32) -> Vec3 {
    let omega_vec = plate.euler_pole * plate.omega_rad_per_year;
    omega_vec.cross(dir.normalize_or_zero()) * radius_m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(plate_count: u32, mesh_cells: u32) -> TectonicConfig {
        TectonicConfig {
            plate_count,
            mesh_cells,
            activity: TectonicActivity::Active,
            continental_fraction: 0.30,
            seed: 7,
            seed_dirs: None,
            continental_clustering: 0.0,
            equatorial_bias: 0.0,
            primary_size_multiplier: 1.0,
        }
    }

    #[test]
    fn every_cell_assigned() {
        let mesh = SphericalMesh::build(512, 1);
        let cfg = config(8, 512);
        let asgn = assign_plates(&mesh, &cfg, 99);
        assert_eq!(asgn.cell_plate.len(), 512);
        for &p in &asgn.cell_plate {
            assert!((p.0 as usize) < asgn.plates.len(), "stray plate id");
        }
    }

    #[test]
    fn plate_count_matches_request() {
        let mesh = SphericalMesh::build(256, 1);
        let cfg = config(12, 256);
        let asgn = assign_plates(&mesh, &cfg, 42);
        assert_eq!(asgn.plates.len(), 12);
    }

    #[test]
    fn continental_count_matches_fraction() {
        let mesh = SphericalMesh::build(256, 1);
        let mut cfg = config(10, 256);
        cfg.continental_fraction = 0.30;
        let asgn = assign_plates(&mesh, &cfg, 42);
        let n_continental = asgn
            .plates
            .iter()
            .filter(|p| p.kind == PlateKind::Continental)
            .count();
        // floor(10 * 0.30) = 3
        assert_eq!(n_continental, 3);
    }

    #[test]
    fn assignment_is_deterministic() {
        let mesh = SphericalMesh::build(256, 1);
        let cfg = config(8, 256);
        let a = assign_plates(&mesh, &cfg, 42);
        let b = assign_plates(&mesh, &cfg, 42);
        assert_eq!(a.cell_plate, b.cell_plate);
        for (pa, pb) in a.plates.iter().zip(&b.plates) {
            assert_eq!(pa.seed_cell, pb.seed_cell);
            assert_eq!(pa.kind, pb.kind);
            assert_eq!(pa.euler_pole, pb.euler_pole);
            assert_eq!(pa.omega_rad_per_year, pb.omega_rad_per_year);
        }
    }

    #[test]
    fn authored_seed_dirs_are_honored() {
        let mesh = SphericalMesh::build(256, 1);
        let mut cfg = config(4, 256);
        let target = Vec3::new(1.0, 0.2, -0.3).normalize();
        cfg.seed_dirs = Some(vec![target]);
        let asgn = assign_plates(&mesh, &cfg, 42);
        let expected_cell = mesh.nearest(target);
        assert_eq!(asgn.plates[0].seed_cell, expected_cell);
    }

    #[test]
    fn stagnant_lid_zeroes_sampled_velocity() {
        let plate = Plate {
            id: PlateId(0),
            seed_cell: 0,
            centroid_dir: Vec3::Z,
            kind: PlateKind::Continental,
            euler_pole: Vec3::Y,
            omega_rad_per_year: 1.0e-8,
        };
        let dir = Vec3::X;
        let radius = 6.4e6;
        let active = surface_velocity(&plate, dir, radius, TectonicActivity::Active);
        let stagnant = surface_velocity(&plate, dir, radius, TectonicActivity::StagnantLid);
        assert!(active.length() > 0.0);
        assert_eq!(stagnant, Vec3::ZERO);
    }
}
