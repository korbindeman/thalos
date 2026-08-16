//! Irregular plate-growth field used by [`crate::ProceduralSurface`].
//!
//! Plate ownership is grown across a cube-sphere with a weighted multi-source
//! flood, rather than assigned to the nearest seed. This carries the useful
//! geometry from the `thalos_maps` prototype into the runtime path: plates
//! have different growth rates and preferred bearings, broad crustal noise
//! bends their fronts, and small plates are born in contested gaps. Boundary
//! process and distance are then propagated from the resulting contact graph.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, LazyLock, Mutex, Weak};

use glam::DVec3;

use crate::noise::fbm3;
use crate::procedural::TectonicSignals;
use crate::seeding::Rng;

const FACE_COUNT: usize = 6;
const FACE_RESOLUTION: usize = 128;
const CELL_COUNT: usize = FACE_COUNT * FACE_RESOLUTION * FACE_RESOLUTION;
const MAJOR_PLATES: usize = 9;
const MICROPLATES: usize = 3;
const PLATE_COUNT: usize = MAJOR_PLATES + MICROPLATES;
const GROWTH_PHASE_ONE_FRAC: f64 = 0.62;
const GROWTH_BEARING_WEIGHT: f64 = 0.72;
const GROWTH_NOISE_WEIGHT: f64 = 1.05;
pub(crate) const ANCIENT_WIDTH_M: f64 = 280_000.0;
const ACTIVE_WIDTH_M: f64 = 120_000.0;
const WIDTH_NOISE_SCALE: f64 = 9.0;
const WIDTH_SCALE_MIN: f64 = 0.45;
const WIDTH_SCALE_MAX: f64 = 1.55;
#[cfg(test)]
pub(crate) const MAX_ANCIENT_WIDTH_M: f64 = ANCIENT_WIDTH_M * WIDTH_SCALE_MAX;
const PRESERVATION_NOISE_SCALE: f64 = 11.0;
const PRESERVATION_FLOOR: f64 = 0.03;
/// Fraction of the collision response retained continuously along a convergent
/// contact. Regional preservation still decides where tall massifs survive;
/// this low spine keeps the surviving pieces legible as one mountain system.
const RANGE_CONTINUITY_FLOOR: f64 = 0.12;
const ACTIVE_LO: f64 = 0.50;
const ACTIVE_HI: f64 = 0.82;
const HINTERLAND_INNER_M: f64 = 45_000.0;
const HINTERLAND_OUTER_M: f64 = 900_000.0;
const FORELAND_INNER_M: f64 = 90_000.0;
const FORELAND_OUTER_M: f64 = 620_000.0;
const RIDGE_SWELL_WIDTH_M: f64 = 820_000.0;

const NEIGHBOR_OFFSETS: [(i32, i32); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

type CacheKey = (u64, u32);

static FIELD_CACHE: LazyLock<Mutex<HashMap<CacheKey, Weak<ProceduralTectonicField>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Copy, Debug, Default)]
struct SignalTexel {
    boundary_distance_m: f32,
    convergence: f32,
    divergence: f32,
    transform: f32,
    activity: f32,
    orogeny: f32,
    hinterland: f32,
    foreland: f32,
    ridge_swell: f32,
}

impl SignalTexel {
    fn lerp(self, other: Self, t: f64) -> Self {
        let t = t as f32;
        Self {
            boundary_distance_m: self.boundary_distance_m
                + (other.boundary_distance_m - self.boundary_distance_m) * t,
            convergence: self.convergence + (other.convergence - self.convergence) * t,
            divergence: self.divergence + (other.divergence - self.divergence) * t,
            transform: self.transform + (other.transform - self.transform) * t,
            activity: self.activity + (other.activity - self.activity) * t,
            orogeny: self.orogeny + (other.orogeny - self.orogeny) * t,
            hinterland: self.hinterland + (other.hinterland - self.hinterland) * t,
            foreland: self.foreland + (other.foreland - self.foreland) * t,
            ridge_swell: self.ridge_swell + (other.ridge_swell - self.ridge_swell) * t,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PlateTraits {
    growth_bias: f64,
    preferred_direction: DVec3,
    euler_pole: DVec3,
    angular_rate: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct BoundaryState {
    normal_speed: f64,
    normal_share: f64,
    motion_strength: f64,
    activity: f64,
    pair_hash: u32,
    plates: [u8; 2],
    hinterland_plate: u8,
}

/// Compact, bilinearly sampled process field shared by surfaces with the same
/// `(radius, seed)`. The cube-sphere grid avoids a polar singularity while the
/// cache prevents the diffusion and procedural wrappers from rebuilding it.
#[derive(Debug)]
pub(crate) struct ProceduralTectonicField {
    texels: Vec<SignalTexel>,
}

impl ProceduralTectonicField {
    pub(crate) fn shared(radius_m: f64, seed: u32) -> Arc<Self> {
        let key = (radius_m.to_bits(), seed);
        let mut cache = FIELD_CACHE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(field) = cache.get(&key).and_then(Weak::upgrade) {
            return field;
        }
        let field = Arc::new(Self::build(radius_m, seed));
        cache.insert(key, Arc::downgrade(&field));
        field
    }

    fn build(radius_m: f64, seed: u32) -> Self {
        let directions: Vec<DVec3> = (0..CELL_COUNT).map(cell_direction).collect();
        let neighbors: Vec<[u32; 8]> = (0..CELL_COUNT).map(cell_neighbors).collect();
        let mut traits = plate_traits(seed);
        let owners = grow_plates(&directions, &neighbors, &mut traits, seed);
        let boundary_states =
            classify_boundary_cells(&directions, &neighbors, &owners, &traits, seed);
        let (distance_m, nearest_boundary) =
            boundary_distance_field(&directions, &neighbors, &boundary_states, radius_m);

        let texels = directions
            .iter()
            .enumerate()
            .map(|(cell, &dir)| {
                let boundary = boundary_states[nearest_boundary[cell] as usize];
                signal_texel(dir, distance_m[cell], boundary, owners[cell], seed)
            })
            .collect();
        Self { texels }
    }

    pub(crate) fn sample(&self, dir: DVec3) -> TectonicSignals {
        let dir = dir.normalize_or_zero();
        if dir == DVec3::ZERO {
            return TectonicSignals::default();
        }
        let (face, u, v) = direction_to_face_uv(dir);
        let px = (u + 1.0) * 0.5 * FACE_RESOLUTION as f64 - 0.5;
        let py = (1.0 - v) * 0.5 * FACE_RESOLUTION as f64 - 0.5;
        let x0 = px.floor() as i32;
        let y0 = py.floor() as i32;
        let tx = px - f64::from(x0);
        let ty = py - f64::from(y0);
        let top = self
            .sample_cross_face(face, x0, y0)
            .lerp(self.sample_cross_face(face, x0 + 1, y0), tx);
        let bottom = self
            .sample_cross_face(face, x0, y0 + 1)
            .lerp(self.sample_cross_face(face, x0 + 1, y0 + 1), tx);
        let signal = top.lerp(bottom, ty);
        TectonicSignals {
            boundary_distance_m: f64::from(signal.boundary_distance_m),
            convergence: f64::from(signal.convergence).clamp(0.0, 1.0),
            divergence: f64::from(signal.divergence).clamp(0.0, 1.0),
            transform: f64::from(signal.transform).clamp(0.0, 1.0),
            activity: f64::from(signal.activity).clamp(0.0, 1.0),
            orogeny: f64::from(signal.orogeny).clamp(0.0, 1.0),
            hinterland: f64::from(signal.hinterland).clamp(0.0, 1.0),
            foreland: f64::from(signal.foreland).clamp(0.0, 1.0),
            ridge_swell: f64::from(signal.ridge_swell).clamp(0.0, 1.0),
        }
    }

    fn sample_cross_face(&self, face: usize, x: i32, y: i32) -> SignalTexel {
        if (0..FACE_RESOLUTION as i32).contains(&x) && (0..FACE_RESOLUTION as i32).contains(&y) {
            return self.texels[cell_index(face, x as usize, y as usize)];
        }
        let u = (f64::from(x) + 0.5) / FACE_RESOLUTION as f64 * 2.0 - 1.0;
        let v = 1.0 - (f64::from(y) + 0.5) / FACE_RESOLUTION as f64 * 2.0;
        let (next_face, next_u, next_v) = direction_to_face_uv(face_uv_to_direction(face, u, v));
        let next_x = (((next_u + 1.0) * 0.5 * FACE_RESOLUTION as f64).floor() as i32)
            .clamp(0, FACE_RESOLUTION as i32 - 1) as usize;
        let next_y = (((1.0 - next_v) * 0.5 * FACE_RESOLUTION as f64).floor() as i32)
            .clamp(0, FACE_RESOLUTION as i32 - 1) as usize;
        self.texels[cell_index(next_face, next_x, next_y)]
    }
}

fn plate_traits(seed: u32) -> [PlateTraits; PLATE_COUNT] {
    let mut rng = Rng::new(u64::from(seed) ^ 0x4752_4F57_5448_504C);
    let mut traits = std::array::from_fn(|plate| {
        let microplate = plate >= MAJOR_PLATES;
        PlateTraits {
            growth_bias: if microplate {
                rng.range_f64(3.5, 6.0)
            } else {
                (rng.next_f64_signed() * 1.15).exp()
            },
            preferred_direction: rng.unit_vector(),
            euler_pole: rng.unit_vector(),
            angular_rate: rng.range_f64(0.55, 1.30),
        }
    });
    let dominant = (0..MAJOR_PLATES)
        .min_by(|&a, &b| {
            traits[a]
                .growth_bias
                .partial_cmp(&traits[b].growth_bias)
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or(0);
    traits[dominant].growth_bias = rng.range_f64(0.28, 0.38);
    traits
}

fn grow_plates(
    directions: &[DVec3],
    neighbors: &[[u32; 8]],
    traits: &mut [PlateTraits; PLATE_COUNT],
    seed: u32,
) -> Vec<u8> {
    let mut owners = vec![u8::MAX; CELL_COUNT];
    let broad_noise: Vec<f64> = directions
        .iter()
        .enumerate()
        .map(|(cell, dir)| {
            let broad = f64::from(fbm3(
                (dir.x * 1.6) as f32,
                (dir.y * 1.6) as f32,
                (dir.z * 1.6) as f32,
                seed ^ 0x3130_4E01,
                4,
                0.52,
                2.0,
            ));
            let edge = unit_hash(hash_u32(cell as u32, seed ^ 0x3170_ED6E)) * 2.0 - 1.0;
            (1.0 + GROWTH_NOISE_WEIGHT * (0.72 * broad + 0.28 * edge)).max(0.12)
        })
        .collect();

    let mut heap = BinaryHeap::new();
    let mut seed_rng = Rng::new(u64::from(seed) ^ 0x504C_4154_4553_4544);
    let rotation = seed_rng.next_f64();
    for plate in 0..MAJOR_PLATES {
        let y = 1.0 - (plate as f64 + 0.5) / MAJOR_PLATES as f64 * 2.0;
        let radial = (1.0 - y * y).max(0.0).sqrt();
        let angle =
            std::f64::consts::TAU * ((plate as f64 * 0.618_033_988_749_894_8 + rotation).fract());
        let base = DVec3::new(radial * angle.cos(), y, radial * angle.sin());
        let jitter = seed_rng.unit_vector();
        let tangent = (jitter - base * jitter.dot(base)).normalize_or_zero();
        let seed_dir = (base + tangent * seed_rng.range_f64(-0.10, 0.10)).normalize();
        let seed_cell = direction_cell(seed_dir);
        heap.push(GrowthEntry {
            cost: 0.0,
            cell: seed_cell as u32,
            plate: plate as u8,
        });
    }

    let phase_one_target = (CELL_COUNT as f64 * GROWTH_PHASE_ONE_FRAC).round() as usize;
    let mut claimed = grow_until(
        &mut heap,
        &mut owners,
        0,
        phase_one_target,
        directions,
        neighbors,
        &broad_noise,
        traits,
    );

    let micro_seeds = contested_microplate_seeds(&owners, directions, neighbors, seed);
    for (offset, cell) in micro_seeds.into_iter().enumerate() {
        heap.push(GrowthEntry {
            cost: 0.0,
            cell: cell as u32,
            plate: (MAJOR_PLATES + offset) as u8,
        });
    }
    claimed = grow_until(
        &mut heap,
        &mut owners,
        claimed,
        CELL_COUNT,
        directions,
        neighbors,
        &broad_noise,
        traits,
    );
    debug_assert_eq!(claimed, CELL_COUNT);
    owners
}

#[allow(clippy::too_many_arguments)]
fn grow_until(
    heap: &mut BinaryHeap<GrowthEntry>,
    owners: &mut [u8],
    mut claimed: usize,
    target: usize,
    directions: &[DVec3],
    neighbors: &[[u32; 8]],
    broad_noise: &[f64],
    traits: &[PlateTraits; PLATE_COUNT],
) -> usize {
    while claimed < target {
        let Some(entry) = heap.pop() else {
            break;
        };
        let cell = entry.cell as usize;
        if owners[cell] != u8::MAX {
            continue;
        }
        owners[cell] = entry.plate;
        claimed += 1;
        let plate = traits[entry.plate as usize];
        let from = directions[cell];
        let preferred = (plate.preferred_direction - from * plate.preferred_direction.dot(from))
            .normalize_or_zero();
        for &neighbor in &neighbors[cell] {
            let neighbor = neighbor as usize;
            if owners[neighbor] != u8::MAX {
                continue;
            }
            let to = directions[neighbor];
            let tangent = (to - from * to.dot(from)).normalize_or_zero();
            let align = tangent.dot(preferred).abs();
            let bearing = 1.0 + GROWTH_BEARING_WEIGHT * (1.0 - align);
            let edge_angle = from.dot(to).clamp(-1.0, 1.0).acos();
            heap.push(GrowthEntry {
                cost: entry.cost + edge_angle * plate.growth_bias * bearing * broad_noise[neighbor],
                cell: neighbor as u32,
                plate: entry.plate,
            });
        }
    }
    claimed
}

fn contested_microplate_seeds(
    owners: &[u8],
    directions: &[DVec3],
    neighbors: &[[u32; 8]],
    seed: u32,
) -> [usize; MICROPLATES] {
    if MICROPLATES == 0 {
        return [usize::MAX; MICROPLATES];
    }
    let mut candidates: Vec<(u8, u32, usize)> = owners
        .iter()
        .enumerate()
        .filter_map(|(cell, &owner)| {
            if owner != u8::MAX {
                return None;
            }
            let mut seen = [false; MAJOR_PLATES];
            for &neighbor in &neighbors[cell] {
                let plate = owners[neighbor as usize];
                if (plate as usize) < MAJOR_PLATES {
                    seen[plate as usize] = true;
                }
            }
            let contested = seen.into_iter().filter(|&present| present).count() as u8;
            (contested >= 2).then_some((contested, hash_u32(cell as u32, seed ^ 0x6190_C057), cell))
        })
        .collect();
    candidates.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));

    let mut result = [usize::MAX; MICROPLATES];
    let mut count = 0;
    let min_separation_cos = 0.28_f64.cos();
    for &(_, _, cell) in &candidates {
        if result[..count]
            .iter()
            .all(|&other| directions[cell].dot(directions[other]) < min_separation_cos)
        {
            result[count] = cell;
            count += 1;
            if count == MICROPLATES {
                return result;
            }
        }
    }

    for (cell, &owner) in owners.iter().enumerate() {
        if count == MICROPLATES {
            break;
        }
        if owner == u8::MAX && !result[..count].contains(&cell) {
            result[count] = cell;
            count += 1;
        }
    }
    assert_eq!(
        count, MICROPLATES,
        "tectonic growth left no room for microplates"
    );
    result
}

fn classify_boundary_cells(
    directions: &[DVec3],
    neighbors: &[[u32; 8]],
    owners: &[u8],
    traits: &[PlateTraits; PLATE_COUNT],
    seed: u32,
) -> Vec<Option<BoundaryState>> {
    // The grown boundary is deliberately ragged at cell scale. Resolving motion
    // against each individual cross-cell edge would turn that raster stair-step
    // into alternating convergent/divergent stripes. Plate-centroid separation
    // supplies a stable, smoothly varying pair normal; ownership still comes
    // entirely from the irregular flood geometry.
    let mut centroid_sums = [DVec3::ZERO; PLATE_COUNT];
    for (cell, &owner) in owners.iter().enumerate() {
        centroid_sums[owner as usize] += directions[cell];
    }
    let centroids = centroid_sums.map(DVec3::normalize_or_zero);

    directions
        .iter()
        .enumerate()
        .map(|(cell, &dir)| {
            let plate_a = owners[cell] as usize;
            let velocity_a = plate_velocity(traits[plate_a], dir);
            let mut best: Option<(f64, BoundaryState)> = None;
            for &neighbor in &neighbors[cell] {
                let neighbor = neighbor as usize;
                let plate_b = owners[neighbor] as usize;
                if plate_a == plate_b {
                    continue;
                }
                let to = directions[neighbor];
                let pair_axis = centroids[plate_b] - centroids[plate_a];
                let stable_normal = pair_axis - dir * pair_axis.dot(dir);
                let cell_normal = to - dir * to.dot(dir);
                let normal = if stable_normal.length_squared() > 1.0e-12 {
                    stable_normal.normalize()
                } else {
                    cell_normal.normalize_or_zero()
                };
                let relative = plate_velocity(traits[plate_b], dir) - velocity_a;
                let speed = relative.length();
                let signed_normal = relative.dot(normal);
                let normal_speed = if speed > 1.0e-9 {
                    signed_normal / speed
                } else {
                    0.0
                };
                let pair_hash = symmetric_pair_hash(plate_a as u32, plate_b as u32, seed);
                let plates = [plate_a.min(plate_b) as u8, plate_a.max(plate_b) as u8];
                let state = BoundaryState {
                    normal_speed,
                    normal_share: normal_speed.abs(),
                    motion_strength: smoothstep(0.18, 1.35, speed),
                    activity: smoothstep(ACTIVE_LO, ACTIVE_HI, unit_hash(pair_hash)),
                    pair_hash,
                    plates,
                    hinterland_plate: plates[(pair_hash & 1) as usize],
                };
                let score = signed_normal.abs() + 0.18 * speed;
                if best.is_none_or(|(best_score, _)| score > best_score) {
                    best = Some((score, state));
                }
            }
            best.map(|(_, state)| state)
        })
        .collect()
}

fn boundary_distance_field(
    directions: &[DVec3],
    neighbors: &[[u32; 8]],
    boundaries: &[Option<BoundaryState>],
    radius_m: f64,
) -> (Vec<f64>, Vec<u32>) {
    let mut distance = vec![f64::INFINITY; CELL_COUNT];
    let mut nearest = vec![u32::MAX; CELL_COUNT];
    let mut heap = BinaryHeap::new();
    for (cell, boundary) in boundaries.iter().enumerate() {
        if boundary.is_some() {
            distance[cell] = 0.0;
            nearest[cell] = cell as u32;
            heap.push(DistanceEntry {
                distance_m: 0.0,
                cell: cell as u32,
            });
        }
    }
    while let Some(entry) = heap.pop() {
        let cell = entry.cell as usize;
        if entry.distance_m > distance[cell] {
            continue;
        }
        for &neighbor in &neighbors[cell] {
            let neighbor = neighbor as usize;
            let edge_m = radius_m
                * directions[cell]
                    .dot(directions[neighbor])
                    .clamp(-1.0, 1.0)
                    .acos();
            let candidate = entry.distance_m + edge_m;
            if candidate < distance[neighbor] {
                distance[neighbor] = candidate;
                nearest[neighbor] = nearest[cell];
                heap.push(DistanceEntry {
                    distance_m: candidate,
                    cell: neighbor as u32,
                });
            }
        }
    }
    assert!(nearest.iter().all(|&cell| cell != u32::MAX));
    (distance, nearest)
}

fn signal_texel(
    dir: DVec3,
    distance_m: f64,
    boundary: Option<BoundaryState>,
    owner: u8,
    seed: u32,
) -> SignalTexel {
    let Some(boundary) = boundary else {
        return SignalTexel {
            boundary_distance_m: distance_m as f32,
            ..SignalTexel::default()
        };
    };
    let width_noise = f64::from(fbm3(
        (dir.x * WIDTH_NOISE_SCALE) as f32,
        (dir.y * WIDTH_NOISE_SCALE) as f32,
        (dir.z * WIDTH_NOISE_SCALE) as f32,
        seed ^ boundary.pair_hash ^ 0x72A4_94B7,
        3,
        0.5,
        2.0,
    ));
    let width_mix = (0.5 + 0.5 * width_noise).clamp(0.0, 1.0);
    let width_scale = WIDTH_SCALE_MIN + (WIDTH_SCALE_MAX - WIDTH_SCALE_MIN) * width_mix;
    let ancient = 1.0 - smoothstep(0.0, ANCIENT_WIDTH_M * width_scale, distance_m);
    let active_core =
        boundary.activity * (1.0 - smoothstep(0.0, ACTIVE_WIDTH_M * width_scale, distance_m));
    let normal_share = smoothstep(0.24, 0.68, boundary.normal_share);
    let transform_share = 1.0 - normal_share;
    let convergence = if boundary.normal_speed < 0.0 {
        boundary.motion_strength * normal_share * ancient
    } else {
        0.0
    };
    let divergence = if boundary.normal_speed > 0.0 {
        boundary.motion_strength * normal_share * ancient
    } else {
        0.0
    };
    let transform = boundary.motion_strength * transform_share * ancient;
    let belt_age = (0.60 * ancient + 0.72 * active_core).clamp(0.0, 1.0);
    let massif_noise = f64::from(fbm3(
        (dir.x * 3.75) as f32,
        (dir.y * 3.75) as f32,
        (dir.z * 3.75) as f32,
        seed ^ boundary.pair_hash ^ 0xB10E_4A55,
        3,
        0.5,
        2.0,
    ));
    let massif = 0.82 + 0.18 * (0.5 + 0.5 * massif_noise);
    // Ancient contacts survive as regional massifs, not equally preserved
    // walls along every kilometre of boundary. Use a shorter, independent
    // field than the broad massif modulation: sharing the broad field made a
    // whole curved contact survive as one synthetic-looking ridge stroke.
    let preservation_noise = f64::from(fbm3(
        (dir.x * PRESERVATION_NOISE_SCALE) as f32,
        (dir.y * PRESERVATION_NOISE_SCALE) as f32,
        (dir.z * PRESERVATION_NOISE_SCALE) as f32,
        seed ^ boundary.pair_hash ^ 0xC048_6C3D,
        3,
        0.5,
        2.0,
    ));
    let preservation = PRESERVATION_FLOOR
        + (1.0 - PRESERVATION_FLOOR) * smoothstep(-0.25, 0.35, preservation_noise);
    let convergent_share = if boundary.normal_speed < 0.0 {
        normal_share
    } else {
        0.0
    };
    let range_survival = RANGE_CONTINUITY_FLOOR + (1.0 - RANGE_CONTINUITY_FLOOR) * preservation;
    let orogeny = (boundary.motion_strength
        * (convergent_share + 0.14 * transform_share)
        * belt_age
        * massif
        * range_survival)
        .clamp(0.0, 1.0);
    let collision_strength = if boundary.normal_speed < 0.0 {
        boundary.motion_strength * normal_share
    } else {
        0.0
    };
    // A collision is wider than its exposed peaks. One plate carries a broad
    // elevated hinterland while the other carries a lower foreland basin. The
    // side is stable per plate pair, and the narrow orogen core hides the
    // ownership transition at the contact. Reuse regional preservation with a
    // softer exponent so old belts retain broad context after their peaks have
    // eroded into separated massifs.
    let province_survival = preservation.powf(0.65);
    let hinterland_lobe = smoothstep(0.0, HINTERLAND_INNER_M, distance_m)
        * (1.0 - smoothstep(HINTERLAND_OUTER_M * 0.55, HINTERLAND_OUTER_M, distance_m));
    let foreland_lobe = smoothstep(0.0, FORELAND_INNER_M, distance_m)
        * (1.0 - smoothstep(FORELAND_OUTER_M * 0.55, FORELAND_OUTER_M, distance_m));
    let pair_member = boundary.plates.contains(&owner);
    let hinterland = if pair_member && owner == boundary.hinterland_plate {
        collision_strength * province_survival * hinterland_lobe
    } else {
        0.0
    };
    let foreland = if pair_member && owner != boundary.hinterland_plate {
        collision_strength * province_survival * foreland_lobe
    } else {
        0.0
    };
    let ridge_swell = if boundary.normal_speed > 0.0 {
        boundary.motion_strength
            * normal_share
            * (1.0 - smoothstep(0.0, RIDGE_SWELL_WIDTH_M * width_scale, distance_m))
    } else {
        0.0
    };
    SignalTexel {
        boundary_distance_m: distance_m as f32,
        convergence: convergence as f32,
        divergence: divergence as f32,
        transform: transform as f32,
        activity: boundary.activity as f32,
        orogeny: orogeny as f32,
        hinterland: hinterland as f32,
        foreland: foreland as f32,
        ridge_swell: ridge_swell as f32,
    }
}

fn plate_velocity(plate: PlateTraits, dir: DVec3) -> DVec3 {
    plate.euler_pole.cross(dir) * plate.angular_rate
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GrowthEntry {
    cost: f64,
    cell: u32,
    plate: u8,
}

impl Eq for GrowthEntry {}

impl Ord for GrowthEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.cell.cmp(&other.cell))
            .then_with(|| self.plate.cmp(&other.plate))
    }
}

impl PartialOrd for GrowthEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DistanceEntry {
    distance_m: f64,
    cell: u32,
}

impl Eq for DistanceEntry {}

impl Ord for DistanceEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance_m
            .partial_cmp(&self.distance_m)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.cell.cmp(&other.cell))
    }
}

impl PartialOrd for DistanceEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn cell_neighbors(cell: usize) -> [u32; 8] {
    let face_len = FACE_RESOLUTION * FACE_RESOLUTION;
    let face = cell / face_len;
    let local = cell % face_len;
    let x = local % FACE_RESOLUTION;
    let y = local / FACE_RESOLUTION;
    std::array::from_fn(|index| {
        let (dx, dy) = NEIGHBOR_OFFSETS[index];
        direction_cell(face_uv_to_direction(
            face,
            (x as f64 + 0.5 + f64::from(dx)) / FACE_RESOLUTION as f64 * 2.0 - 1.0,
            1.0 - (y as f64 + 0.5 + f64::from(dy)) / FACE_RESOLUTION as f64 * 2.0,
        )) as u32
    })
}

fn cell_direction(cell: usize) -> DVec3 {
    let face_len = FACE_RESOLUTION * FACE_RESOLUTION;
    let face = cell / face_len;
    let local = cell % face_len;
    let x = local % FACE_RESOLUTION;
    let y = local / FACE_RESOLUTION;
    let u = (x as f64 + 0.5) / FACE_RESOLUTION as f64 * 2.0 - 1.0;
    let v = 1.0 - (y as f64 + 0.5) / FACE_RESOLUTION as f64 * 2.0;
    face_uv_to_direction(face, u, v)
}

fn direction_cell(dir: DVec3) -> usize {
    let (face, u, v) = direction_to_face_uv(dir);
    let x = (((u + 1.0) * 0.5 * FACE_RESOLUTION as f64).floor() as i32)
        .clamp(0, FACE_RESOLUTION as i32 - 1) as usize;
    let y = (((1.0 - v) * 0.5 * FACE_RESOLUTION as f64).floor() as i32)
        .clamp(0, FACE_RESOLUTION as i32 - 1) as usize;
    cell_index(face, x, y)
}

fn cell_index(face: usize, x: usize, y: usize) -> usize {
    face * FACE_RESOLUTION * FACE_RESOLUTION + y * FACE_RESOLUTION + x
}

fn face_uv_to_direction(face: usize, u: f64, v: f64) -> DVec3 {
    let dir = match face {
        0 => DVec3::new(1.0, v, -u),
        1 => DVec3::new(-1.0, v, u),
        2 => DVec3::new(u, 1.0, -v),
        3 => DVec3::new(u, -1.0, v),
        4 => DVec3::new(u, v, 1.0),
        5 => DVec3::new(-u, v, -1.0),
        _ => unreachable!("cube face {face}"),
    };
    dir.normalize()
}

fn direction_to_face_uv(dir: DVec3) -> (usize, f64, f64) {
    let abs = dir.abs();
    if abs.x >= abs.y && abs.x >= abs.z {
        if dir.x >= 0.0 {
            (0, -dir.z / abs.x, dir.y / abs.x)
        } else {
            (1, dir.z / abs.x, dir.y / abs.x)
        }
    } else if abs.y >= abs.z {
        if dir.y >= 0.0 {
            (2, dir.x / abs.y, -dir.z / abs.y)
        } else {
            (3, dir.x / abs.y, dir.z / abs.y)
        }
    } else if dir.z >= 0.0 {
        (4, dir.x / abs.z, dir.y / abs.z)
    } else {
        (5, -dir.x / abs.z, dir.y / abs.z)
    }
}

fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn unit_hash(value: u32) -> f64 {
    f64::from(value) / f64::from(u32::MAX)
}

fn hash_u32(value: u32, seed: u32) -> u32 {
    let mut h = seed ^ value.wrapping_mul(0x9E37_79B1);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^ (h >> 16)
}

fn symmetric_pair_hash(a: u32, b: u32, seed: u32) -> u32 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    hash_u32(lo ^ hi.rotate_left(13), seed ^ 0x7EC7_0A1C)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RADIUS_M: f64 = 3_186_000.0;

    #[test]
    fn cube_mapping_round_trips_cell_centers() {
        for cell in (0..CELL_COUNT).step_by(97) {
            assert_eq!(direction_cell(cell_direction(cell)), cell);
        }
    }

    #[test]
    fn grown_field_is_deterministic_and_process_bounded() {
        let a = ProceduralTectonicField::build(RADIUS_M, 2);
        let b = ProceduralTectonicField::build(RADIUS_M, 2);
        assert_eq!(a.texels.len(), CELL_COUNT);
        for cell in (0..CELL_COUNT).step_by(131) {
            let left = a.texels[cell];
            let right = b.texels[cell];
            assert_eq!(
                left.boundary_distance_m.to_bits(),
                right.boundary_distance_m.to_bits()
            );
            assert_eq!(left.orogeny.to_bits(), right.orogeny.to_bits());
            for value in [
                left.convergence,
                left.divergence,
                left.transform,
                left.activity,
                left.orogeny,
            ] {
                assert!((0.0..=1.0).contains(&value));
            }
        }
    }

    #[test]
    fn flood_growth_keeps_every_plate_connected() {
        let directions: Vec<DVec3> = (0..CELL_COUNT).map(cell_direction).collect();
        let neighbors: Vec<[u32; 8]> = (0..CELL_COUNT).map(cell_neighbors).collect();
        let mut traits = plate_traits(2);
        let owners = grow_plates(&directions, &neighbors, &mut traits, 2);
        for plate in 0..PLATE_COUNT as u8 {
            let start = owners.iter().position(|&owner| owner == plate).unwrap();
            let mut seen = vec![false; CELL_COUNT];
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(cell) = stack.pop() {
                for &neighbor in &neighbors[cell] {
                    let neighbor = neighbor as usize;
                    if !seen[neighbor] && owners[neighbor] == plate {
                        seen[neighbor] = true;
                        stack.push(neighbor);
                    }
                }
            }
            assert!(
                owners
                    .iter()
                    .enumerate()
                    .all(|(cell, &owner)| owner != plate || seen[cell]),
                "plate {plate} is disconnected"
            );
        }
    }

    #[test]
    fn convergent_contacts_survive_as_separate_massifs() {
        let field = ProceduralTectonicField::build(RADIUS_M, 2);
        let mut quiet = 0_usize;
        let mut strong = 0_usize;
        for texel in &field.texels {
            if texel.boundary_distance_m > 35_000.0 || texel.convergence < 0.25 {
                continue;
            }
            if texel.orogeny < 0.08 {
                quiet += 1;
            }
            if texel.orogeny > 0.25 {
                strong += 1;
            }
        }

        assert!(quiet > 100, "quiet convergent contact texels: {quiet}");
        assert!(strong > 100, "strong convergent contact texels: {strong}");
    }

    #[test]
    fn plate_contacts_author_broad_asymmetric_provinces() {
        let field = ProceduralTectonicField::build(RADIUS_M, 2);
        let mut hinterland = 0_usize;
        let mut foreland = 0_usize;
        let mut ridge_swell = 0_usize;
        let mut mixed_collision_sides = 0_usize;

        for texel in &field.texels {
            hinterland += usize::from(texel.hinterland > 0.08);
            foreland += usize::from(texel.foreland > 0.08);
            ridge_swell += usize::from(texel.ridge_swell > 0.08);
            mixed_collision_sides += usize::from(texel.hinterland > 0.08 && texel.foreland > 0.08);
        }

        assert!(hinterland > 500, "hinterland texels: {hinterland}");
        assert!(foreland > 500, "foreland texels: {foreland}");
        assert!(ridge_swell > 500, "ridge-swell texels: {ridge_swell}");
        assert_eq!(mixed_collision_sides, 0);
    }
}
