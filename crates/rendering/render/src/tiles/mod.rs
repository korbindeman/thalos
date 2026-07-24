//! NTR-X1 — the probe-extracted tile terrain renderer (keystone:
//! ADR-20260723T142945Z, plan `ntr §6`): terrain as ordinary `Mesh` +
//! `StandardMaterial` entities streamed by a camera-driven, 2:1-balanced
//! cube-sphere quadtree, entirely on Bevy's standard render path.
//!
//! Ported from the standalone probe (`thalos-terrain-probe`, M0–M4) with the
//! Thalos adaptations: per-body radius, tiles parented to the body's rotating
//! big_space grid (big_space composes rotated-grid origin-relative transforms
//! in f64, so co-rotating children keep planet-scale precision), content from
//! the body's canonical `Arc<dyn SurfaceQuery>` (real albedo/roughness per
//! sample), and the selection eye supplied by the game from `ViewAnchor`
//! (body-fixed camera position — the one per-frame answer to "where is the
//! view?").
//!
//! Slice 1 (Mira): behind the game's `THALOS_TILE_RENDERER=1` toggle; udlod
//! keeps running everywhere else. Known limits, tracked in the backlog: plain
//! `StandardMaterial` (no `thalos::shadow` rig, no Hapke — airless shading
//! fidelity is the follow-up), no GPU height-mirror feed (CPU `HeightSource`
//! fallback serves colliders), heights-only displacement of `sample_d`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::math::DVec3;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::tasks::{Task, block_on, poll_once};
use big_space::prelude::*;
use thalos_terrain::SurfaceQuery;

/// Vertices per tile side (core grid, excluding halo).
pub const TILE_RES: usize = 65;
/// Halo rings included in every sampled grid (edge-exact normals).
pub const TILE_HALO: usize = 1;
/// Coarsest selected level (8×8 tiles per face).
pub const MIN_LEVEL: u8 = 3;
/// Split while camera distance < factor × tile arc.
const SPLIT_FACTOR: f64 = 3.0;
const MAX_IN_FLIGHT: usize = 24;

// --- cube-sphere addressing (probe tiles.rs, verbatim) -----------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileKey {
    /// Canonical cube face, 0..6 (+X, −X, +Y, −Y, +Z, −Z).
    pub face: u8,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl TileKey {
    pub fn tiles_per_side(self) -> u32 {
        1u32 << self.level
    }

    pub fn uv_rect(self) -> (f64, f64, f64, f64) {
        let n = self.tiles_per_side() as f64;
        let w = 2.0 / n;
        let u0 = -1.0 + w * self.x as f64;
        let v0 = -1.0 + w * self.y as f64;
        (u0, v0, u0 + w, v0 + w)
    }

    pub fn containing(face: u8, level: u8, u: f64, v: f64) -> Self {
        let n = 1u32 << level;
        let t = |c: f64| (((c + 1.0) * 0.5 * n as f64) as i64).clamp(0, n as i64 - 1) as u32;
        Self { face, level, x: t(u), y: t(v) }
    }

    pub fn containing_dir(dir: DVec3, level: u8) -> Self {
        let (face, u, v) = face_uv_of_dir(dir);
        Self::containing(face, level, u, v)
    }

    pub fn parent(self) -> Option<Self> {
        (self.level > 0).then(|| Self {
            face: self.face,
            level: self.level - 1,
            x: self.x / 2,
            y: self.y / 2,
        })
    }

    pub fn children(self) -> [Self; 4] {
        let (f, l, x, y) = (self.face, self.level + 1, self.x * 2, self.y * 2);
        [
            Self { face: f, level: l, x, y },
            Self { face: f, level: l, x: x + 1, y },
            Self { face: f, level: l, x, y: y + 1 },
            Self { face: f, level: l, x: x + 1, y: y + 1 },
        ]
    }

    /// Unit direction at fractional in-tile coords (halo addresses extend
    /// past [0, 1] — gnomonic extension stays valid across face edges).
    pub fn dir_at(self, s: f64, t: f64) -> DVec3 {
        let (u0, v0, u1, v1) = self.uv_rect();
        face_dir(self.face, u0 + (u1 - u0) * s, v0 + (v1 - v0) * t)
    }

    pub fn center_dir(self) -> DVec3 {
        self.dir_at(0.5, 0.5)
    }

    pub fn sample_spacing_m(self, radius_m: f64) -> f64 {
        let face_arc = radius_m * core::f64::consts::FRAC_PI_2;
        face_arc / self.tiles_per_side() as f64 / (TILE_RES - 1) as f64
    }
}

/// Cube-face → unit direction; ∂u × ∂v points outward on every face (the
/// mesher's winding self-test enforces the consequence — probe M0 postmortem).
pub fn face_dir(face: u8, u: f64, v: f64) -> DVec3 {
    let d = match face {
        0 => DVec3::new(1.0, v, -u),
        1 => DVec3::new(-1.0, v, u),
        2 => DVec3::new(u, 1.0, -v),
        3 => DVec3::new(u, -1.0, v),
        4 => DVec3::new(u, v, 1.0),
        5 => DVec3::new(-u, v, -1.0),
        _ => unreachable!("six faces"),
    };
    d.normalize()
}

/// Exact inverse of [`face_dir`] — the cross-face adjacency transform.
pub fn face_uv_of_dir(dir: DVec3) -> (u8, f64, f64) {
    let a = dir.abs();
    if a.x >= a.y && a.x >= a.z {
        if dir.x >= 0.0 {
            (0, -dir.z / dir.x, dir.y / dir.x)
        } else {
            (1, dir.z / -dir.x, dir.y / -dir.x)
        }
    } else if a.y >= a.z {
        if dir.y >= 0.0 {
            (2, dir.x / dir.y, -dir.z / dir.y)
        } else {
            (3, dir.x / -dir.y, dir.z / -dir.y)
        }
    } else if dir.z >= 0.0 {
        (4, dir.x / dir.z, dir.y / dir.z)
    } else {
        (5, dir.x / dir.z, -dir.y / dir.z)
    }
}

// --- tile payload + provider ---------------------------------------------------

/// One sampled tile: heights + linear albedo on the halo grid.
pub struct SurfaceTile {
    pub key: TileKey,
    pub sample_spacing_m: f64,
    pub heights_m: Vec<f32>,
    pub albedo_linear: Vec<[f32; 3]>,
}

impl SurfaceTile {
    pub fn grid_side() -> usize {
        TILE_RES + 2 * TILE_HALO
    }
}

/// The provider seam (ADR-20260722T105147Z part 1). Slice 1 wraps the body's
/// canonical `SurfaceQuery`; package/neural producers slot in behind the same
/// boundary.
pub trait TerrainTileProvider: Send + Sync {
    fn request(&self, key: TileKey, radius_m: f64) -> SurfaceTile;
}

/// `SurfaceQuery`-backed provider — samples the body's one height/albedo
/// authority per tile vertex (f64 directions; see `SurfaceQuery::sample_d`).
pub struct SurfaceQueryProvider {
    pub surface: Arc<dyn SurfaceQuery>,
}

impl TerrainTileProvider for SurfaceQueryProvider {
    fn request(&self, key: TileKey, radius_m: f64) -> SurfaceTile {
        let spacing = key.sample_spacing_m(radius_m);
        let side = SurfaceTile::grid_side();
        let step = 1.0 / (TILE_RES - 1) as f64;
        // Package/point-query sampling is the expensive half (the
        // raster->point->raster tax, ADR-20260722T105147Z) — spread rows
        // across the shared *bounded* eval pool, exactly like udlod's
        // `compute_tile_pixels` (never rayon's implicit global pool; see
        // `ground::tile_synthesis_pool`).
        let rows: Vec<(Vec<f32>, Vec<[f32; 3]>)> =
            crate::ground::tile_synthesis_pool::tile_eval_pool().install(|| {
                use rayon::prelude::*;
                (0..side)
                    .into_par_iter()
                    .map(|j| {
                        let mut h = Vec::with_capacity(side);
                        let mut a = Vec::with_capacity(side);
                        let t = (j as f64 - TILE_HALO as f64) * step;
                        for i in 0..side {
                            let s = (i as f64 - TILE_HALO as f64) * step;
                            let sample =
                                self.surface.sample_d(key.dir_at(s, t), spacing as f32);
                            h.push(sample.height_m);
                            a.push([
                                sample.albedo_linear.x,
                                sample.albedo_linear.y,
                                sample.albedo_linear.z,
                            ]);
                        }
                        (h, a)
                    })
                    .collect()
            });
        let mut heights = Vec::with_capacity(side * side);
        let mut albedo = Vec::with_capacity(side * side);
        for (h, a) in rows {
            heights.extend(h);
            albedo.extend(a);
        }
        SurfaceTile { key, sample_spacing_m: spacing, heights_m: heights, albedo_linear: albedo }
    }
}

// --- mesher (probe mesher.rs, albedo-driven) ------------------------------------

/// Skirt depth: chord sag + band-gate allowance (shared probe formula).
pub fn skirt_drop_m(sample_spacing_m: f64, radius_m: f64) -> f32 {
    let sag = sample_spacing_m * sample_spacing_m / (8.0 * radius_m);
    (sag * 4.0 + sample_spacing_m * 0.06).clamp(0.5, 150.0) as f32
}

struct BuiltTile {
    mesh: Mesh,
    /// Body-fixed f64 position of the mesh origin (displaced tile center).
    origin: DVec3,
    /// Radial deviation range (m) of the built interior vertices from the
    /// reference sphere — the mesh-side counterpart of the provider height
    /// range, so "heights sampled" vs "heights in the mesh" separate in the
    /// telemetry.
    mesh_h: (f32, f32),
}

/// Debug relief exaggeration (`THALOS_TILE_HEIGHT_SCALE`, default 1.0) —
/// makes "is displacement reaching the rendered mesh at all" undeniable.
fn debug_height_scale() -> f64 {
    static SCALE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *SCALE.get_or_init(|| {
        std::env::var("THALOS_TILE_HEIGHT_SCALE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(1.0)
    })
}

fn build_tile_mesh(tile: &SurfaceTile, radius_m: f64) -> BuiltTile {
    let key = tile.key;
    let halo = TILE_HALO;
    let side = SurfaceTile::grid_side();
    let res = TILE_RES;
    let step = 1.0 / (res - 1) as f64;
    let h_scale = debug_height_scale();

    let mut pos_grid: Vec<DVec3> = Vec::with_capacity(side * side);
    for j in 0..side {
        for i in 0..side {
            let s = (i as f64 - halo as f64) * step;
            let t = (j as f64 - halo as f64) * step;
            let h = tile.heights_m[j * side + i] as f64 * h_scale;
            pos_grid.push(key.dir_at(s, t) * (radius_m + h));
        }
    }
    let origin =
        key.center_dir() * (radius_m + tile.heights_m[(side / 2) * side + side / 2] as f64);

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(res * res);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(res * res);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(res * res);
    let mut indices: Vec<u32> = Vec::with_capacity((res - 1) * (res - 1) * 6);

    let mut mesh_h = (f32::INFINITY, f32::NEG_INFINITY);
    for j in 0..res {
        for i in 0..res {
            let (gi, gj) = (i + halo, j + halo);
            let p = pos_grid[gj * side + gi];
            let dev = (p.length() - radius_m) as f32;
            mesh_h.0 = mesh_h.0.min(dev);
            mesh_h.1 = mesh_h.1.max(dev);
            let rel = p - origin;
            positions.push([rel.x as f32, rel.y as f32, rel.z as f32]);
            let du = pos_grid[gj * side + gi + 1] - pos_grid[gj * side + gi - 1];
            let dv = pos_grid[(gj + 1) * side + gi] - pos_grid[(gj - 1) * side + gi];
            let mut n = du.cross(dv).normalize();
            let outward = key.dir_at(i as f64 * step, j as f64 * step);
            if n.dot(outward) < 0.0 {
                n = -n;
            }
            normals.push([n.x as f32, n.y as f32, n.z as f32]);
            let a = tile.albedo_linear[gj * side + gi];
            colors.push([a[0], a[1], a[2], 1.0]);
        }
    }
    for j in 0..res - 1 {
        for i in 0..res - 1 {
            let a = (j * res + i) as u32;
            let b = a + 1;
            let c = a + res as u32;
            let d = c + 1;
            // CCW in (s, t) → outward front faces.
            indices.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }

    // Winding self-test — a probe M0 contract obligation. Skipped under the
    // debug height exaggeration: extreme scales legitimately invert steep
    // triangles, which is the exaggeration doing its job, not a winding bug.
    if (h_scale - 1.0).abs() < 1e-9 {
        let tri = [indices[0] as usize, indices[1] as usize, indices[2] as usize];
        let (p0, p1, p2) = (
            DVec3::from(positions[tri[0]].map(f64::from)),
            DVec3::from(positions[tri[1]].map(f64::from)),
            DVec3::from(positions[tri[2]].map(f64::from)),
        );
        let n = (p1 - p0).cross(p2 - p0);
        debug_assert!(
            n.dot(key.center_dir()) > 0.0,
            "tile {key:?}: first triangle winds inward"
        );
    }

    let drop_m = skirt_drop_m(tile.sample_spacing_m, radius_m);
    let border: Vec<u32> = {
        let mut b = Vec::new();
        for i in 0..res {
            b.push(i as u32);
        }
        for j in 1..res {
            b.push((j * res + res - 1) as u32);
        }
        for i in (0..res - 1).rev() {
            b.push(((res - 1) * res + i) as u32);
        }
        for j in (1..res - 1).rev() {
            b.push((j * res) as u32);
        }
        b
    };
    let down = -key.center_dir();
    let base = positions.len() as u32;
    for &idx in &border {
        let p = positions[idx as usize];
        positions.push([
            p[0] + down.x as f32 * drop_m,
            p[1] + down.y as f32 * drop_m,
            p[2] + down.z as f32 * drop_m,
        ]);
        normals.push(normals[idx as usize]);
        colors.push(colors[idx as usize]);
    }
    let n_border = border.len() as u32;
    for k in 0..n_border {
        let k2 = (k + 1) % n_border;
        let (top_a, top_b) = (border[k as usize], border[k2 as usize]);
        let (bot_a, bot_b) = (base + k, base + k2);
        indices.extend_from_slice(&[top_a, bot_a, bot_b, top_a, bot_b, top_b]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    BuiltTile { mesh, origin, mesh_h }
}

// --- streaming (probe streaming.rs, per-root) ------------------------------------

/// The selection eye, written each frame by the game (from `ViewAnchor`):
/// body-fixed camera position for the root entity's body.
#[derive(Resource, Default)]
pub struct TileEye {
    pub target: Option<TileEyeTarget>,
}

pub struct TileEyeTarget {
    /// The entity carrying [`TileTerrainRoot`] (the body's big_space grid).
    pub root: Entity,
    /// Camera position in the body-fixed frame, meters.
    pub cam_body: DVec3,
}

struct StreamedTile {
    key: TileKey,
    built: BuiltTile,
    gen_micros: u64,
    /// Sampled height range (m) — diagnostic for flat-terrain regressions.
    h_range: (f32, f32),
}

/// One streaming tile terrain on a body grid entity. Insert on the body's
/// `RealSpaceBody` grid; tiles spawn as its co-rotating children.
#[derive(Component)]
pub struct TileTerrainRoot {
    pub radius_m: f64,
    pub provider: Arc<dyn TerrainTileProvider>,
    pub material: Handle<StandardMaterial>,
    pub max_level: u8,
    desired: HashSet<TileKey>,
    resident: HashMap<TileKey, Entity>,
    pending: HashMap<TileKey, Task<StreamedTile>>,
    gen_stats: GenStats,
    /// Latched true the first time the desired set is fully resident. After
    /// that, the hole-free despawn rule keeps coverage complete, so the
    /// impostor↔terrain swap can trust it without flickering back.
    covered_once: bool,
}

/// Rolling per-tile generation-time telemetry; logs every 200 landings so
/// the raster->point->raster tax stays measured (the tile-native package
/// provider is the planned fix).
#[derive(Default)]
struct GenStats {
    samples: Vec<u64>,
    total_landed: u64,
    h_min: f32,
    h_max: f32,
    mesh_min: f32,
    mesh_max: f32,
}

impl GenStats {
    fn record(&mut self, micros: u64, h_range: (f32, f32), mesh_h: (f32, f32)) {
        self.samples.push(micros);
        self.total_landed += 1;
        self.h_min = self.h_min.min(h_range.0);
        self.h_max = self.h_max.max(h_range.1);
        self.mesh_min = self.mesh_min.min(mesh_h.0);
        self.mesh_max = self.mesh_max.max(mesh_h.1);
        if self.samples.len() >= 200 {
            self.samples.sort_unstable();
            let mean = self.samples.iter().sum::<u64>() / self.samples.len() as u64;
            let p95 = self.samples[self.samples.len() * 95 / 100];
            info!(
                "tile terrain: {} tiles landed | gen mean {:.1} ms p95 {:.1} ms (last {}) | provider h [{:.1}, {:.1}] m | mesh h [{:.1}, {:.1}] m",
                self.total_landed,
                mean as f64 / 1000.0,
                p95 as f64 / 1000.0,
                self.samples.len(),
                self.h_min,
                self.h_max,
                self.mesh_min,
                self.mesh_max,
            );
            self.samples.clear();
        }
    }
}

impl TileTerrainRoot {
    pub fn new(
        radius_m: f64,
        provider: Arc<dyn TerrainTileProvider>,
        material: Handle<StandardMaterial>,
    ) -> Self {
        // Deepest level targets ~9 m sample spacing (probe budget).
        let face_arc = radius_m * core::f64::consts::FRAC_PI_2;
        let max_level = ((face_arc / ((TILE_RES - 1) as f64 * 9.0)).log2().ceil() as u8)
            .clamp(MIN_LEVEL + 1, 18);
        Self {
            radius_m,
            provider,
            material,
            max_level,
            desired: HashSet::new(),
            resident: HashMap::new(),
            pending: HashMap::new(),
            gen_stats: GenStats::default(),
            covered_once: false,
        }
    }

    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// True once the streamed terrain has (ever) fully covered the desired
    /// selection — the impostor↔terrain handoff criterion (the analogue of
    /// udlod's `pinned_tiles_ready`).
    pub fn coverage_ready(&self) -> bool {
        self.covered_once
    }


    pub fn settled(&self) -> bool {
        self.pending.is_empty()
            && self
                .desired
                .iter()
                .all(|k| self.resident.contains_key(k))
    }
}

fn tile_arc_m(level: u8, radius_m: f64) -> f64 {
    radius_m * core::f64::consts::FRAC_PI_2 / (1u64 << level) as f64
}

/// Pure selection: distance-split descent + cross-face 2:1 balance via
/// direction probes (probe M2/M3).
pub fn select_leaves(cam_body: DVec3, radius_m: f64, max_level: u8) -> HashSet<TileKey> {
    let want_split = |key: TileKey| -> bool {
        if key.level >= max_level {
            return false;
        }
        let arc = tile_arc_m(key.level, radius_m);
        let d = ((cam_body - key.center_dir() * radius_m).length() - arc * 0.75).max(1.0);
        d < SPLIT_FACTOR * arc
    };

    let mut leaves: HashSet<TileKey> = HashSet::new();
    let n = 1u32 << MIN_LEVEL;
    let mut stack: Vec<TileKey> = Vec::with_capacity(1024);
    for face in 0..6u8 {
        for y in 0..n {
            for x in 0..n {
                stack.push(TileKey { face, level: MIN_LEVEL, x, y });
            }
        }
    }
    while let Some(key) = stack.pop() {
        if want_split(key) {
            stack.extend(key.children());
        } else {
            leaves.insert(key);
        }
    }

    const PROBES: [(f64, f64); 4] = [(-0.5, 0.5), (1.5, 0.5), (0.5, -0.5), (0.5, 1.5)];
    for _ in 0..16 {
        let mut to_split: HashSet<TileKey> = HashSet::new();
        for &leaf in &leaves {
            if leaf.level <= MIN_LEVEL + 1 {
                continue;
            }
            let coarse = leaf.level - 2;
            for (s, t) in PROBES {
                let mut probe = TileKey::containing_dir(leaf.dir_at(s, t), leaf.level);
                if probe == leaf {
                    continue;
                }
                loop {
                    if leaves.contains(&probe) {
                        if probe.level <= coarse {
                            to_split.insert(probe);
                        }
                        break;
                    }
                    match probe.parent() {
                        Some(p) if p.level >= MIN_LEVEL => probe = p,
                        _ => break,
                    }
                }
            }
        }
        if to_split.is_empty() {
            break;
        }
        for key in to_split {
            leaves.remove(&key);
            for child in key.children() {
                leaves.insert(child);
            }
        }
    }
    leaves
}

fn is_self_or_descendant(key: TileKey, anchor: TileKey) -> bool {
    key.face == anchor.face
        && key.level >= anchor.level
        && (key.x >> (key.level - anchor.level)) == anchor.x
        && (key.y >> (key.level - anchor.level)) == anchor.y
}

#[allow(clippy::too_many_arguments)]
fn stream_tile_terrain(
    eye: Res<TileEye>,
    mut roots: Query<(Entity, &mut TileTerrainRoot, &Grid)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut commands: Commands,
) {
    let Some(target) = &eye.target else {
        return;
    };
    let Ok((root_entity, mut root, grid)) = roots.get_mut(target.root) else {
        return;
    };
    let cam = target.cam_body;
    let radius = root.radius_m;
    let max_level = root.max_level;
    let root_ref = &mut *root;

    root_ref.desired = select_leaves(cam, radius, max_level);

    // Cancel pending tiles nobody wants (task drop aborts).
    let desired = &root_ref.desired;
    root_ref.pending.retain(|key, _| desired.contains(key));

    // Admit missing, screen-size-priority (distance / tile size — absolute
    // nearest-first starves coarse merge-targets; probe M3 finding).
    let mut missing: Vec<TileKey> = root_ref
        .desired
        .iter()
        .filter(|k| !root_ref.resident.contains_key(k) && !root_ref.pending.contains_key(k))
        .copied()
        .collect();
    missing.sort_by(|a, b| {
        let pa = (cam - a.center_dir() * radius).length() / tile_arc_m(a.level, radius);
        let pb = (cam - b.center_dir() * radius).length() / tile_arc_m(b.level, radius);
        pa.total_cmp(&pb)
    });
    let budget = MAX_IN_FLIGHT.saturating_sub(root_ref.pending.len());
    // Dedicated pool: routing this through AsyncComputeTaskPool starves
    // Avian's collider-tree optimisation and hitches the main thread (the
    // documented reason `ground::tile_synthesis_pool` exists).
    let pool = crate::ground::tile_synthesis_pool::tile_synthesis_pool();
    for key in missing.into_iter().take(budget) {
        let provider = root_ref.provider.clone();
        let task = pool.spawn(async move {
            let started = std::time::Instant::now();
            let tile = provider.request(key, radius);
            let h_range = tile
                .heights_m
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &h| {
                    (lo.min(h), hi.max(h))
                });
            let built = build_tile_mesh(&tile, radius);
            StreamedTile { key, built, gen_micros: started.elapsed().as_micros() as u64, h_range }
        });
        root_ref.pending.insert(key, task);
    }

    // Land finished tiles as co-rotating children of the body grid.
    let mut landed: Vec<StreamedTile> = Vec::new();
    root_ref.pending.retain(|_, task| {
        if let Some(done) = block_on(poll_once(task)) {
            landed.push(done);
            false
        } else {
            true
        }
    });
    for done in landed {
        root_ref
            .gen_stats
            .record(done.gen_micros, done.h_range, done.built.mesh_h);
        let (cell, local) = grid.translation_to_grid(done.built.origin);
        let entity = commands
            .spawn((
                Mesh3d(meshes.add(done.built.mesh)),
                MeshMaterial3d(root_ref.material.clone()),
                Transform::from_translation(local),
                cell,
                ChildOf(root_entity),
            ))
            .id();
        root_ref.resident.insert(done.key, entity);
    }

    if !root_ref.covered_once
        && !root_ref.desired.is_empty()
        && root_ref.desired.iter().all(|k| root_ref.resident.contains_key(k))
    {
        root_ref.covered_once = true;
        info!(
            "tile terrain: first full coverage ({} tiles) — impostor handoff ready",
            root_ref.resident.len()
        );
    }

    // Hole-free despawn (probe M2 rule).
    let removable: Vec<TileKey> = root_ref
        .resident
        .keys()
        .filter(|k| !root_ref.desired.contains(k))
        .filter(|k| {
            let mut probe = **k;
            let ancestor_ok = loop {
                match probe.parent() {
                    Some(p) if p.level >= MIN_LEVEL => {
                        probe = p;
                        if root_ref.desired.contains(&probe) {
                            break root_ref.resident.contains_key(&probe);
                        }
                    }
                    _ => break true,
                }
            };
            ancestor_ok
                && root_ref
                    .desired
                    .iter()
                    .filter(|d| is_self_or_descendant(**d, **k))
                    .all(|d| root_ref.resident.contains_key(d))
        })
        .copied()
        .collect();
    for key in removable {
        if let Some(entity) = root_ref.resident.remove(&key) {
            commands.entity(entity).despawn();
        }
    }
}

/// Registers the streaming system; inert until a [`TileTerrainRoot`] exists
/// and the game writes [`TileEye`].
pub struct TileTerrainPlugin;

impl Plugin for TileTerrainPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TileEye>()
            .add_systems(Update, stream_tile_terrain);
    }
}
