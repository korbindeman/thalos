use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use bevy::math::{DVec2, UVec2, Vec2, Vec3};
use bevy::prelude::*;
use thalos_terrain::SurfaceQuery;
use thalos_udlod::math::{Coordinate, TerrainModel, TileCoordinate};
use thalos_udlod::prelude::TileAtlas;

use crate::ground::rendered_height::{TerrainPatchBasis, TerrainPatchMesh};
pub trait HeightSource: Send + Sync {
    /// Height in metres above the body's reference radius, evaluated at
    /// `dir` (a body-fixed unit direction). `tile_lod_m` is a scale hint for
    /// sources that synthesize height procedurally.
    fn sample_height_m(&self, dir: Vec3, tile_lod_m: f32) -> Option<f32>;

    /// Monotonically increasing counter that advances when this source's
    /// underlying data changes in a way that would alter
    /// `sample_height_m` outputs. Consumers (e.g. the terrain collider
    /// patch) snapshot this at build time and trigger a rebuild when it
    /// advances. Static sources should return `0`.
    fn revision(&self) -> u64 {
        0
    }

    /// Build a collider patch directly from this source's native geometry —
    /// e.g. the resident GPU atlas tiles the renderer meshes from — so the
    /// collider lines up with the drawn surface by construction. `center_dir`
    /// is the body-fixed unit direction to center the patch on; `max_resolution`
    /// caps the vertex-grid side. Returns `None` for sources with no tile
    /// geometry (procedural CPU pipeline, flat, baked cubemap) or when no tile
    /// is resident yet; callers fall back to the tangent-grid resample in
    /// [`crate::rendered_height::build_rendered_terrain_patch_from_source`].
    fn build_collider_patch(
        &self,
        center_dir: Vec3,
        max_resolution: u32,
    ) -> Option<TerrainPatchMesh> {
        let _ = (center_dir, max_resolution);
        None
    }
}

pub struct CpuPipelineHeightSource {
    surface: Arc<dyn SurfaceQuery>,
}

impl CpuPipelineHeightSource {
    pub fn new(surface: Arc<dyn SurfaceQuery>) -> Self {
        Self { surface }
    }
}

impl HeightSource for CpuPipelineHeightSource {
    fn sample_height_m(&self, dir: Vec3, tile_lod_m: f32) -> Option<f32> {
        Some(self.surface.sample_height_m(dir, tile_lod_m))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConstantHeightSource {
    height_m: f32,
}

impl ConstantHeightSource {
    pub fn new(height_m: f32) -> Self {
        Self { height_m }
    }

    pub fn zero() -> Self {
        Self::new(0.0)
    }
}

impl HeightSource for ConstantHeightSource {
    fn sample_height_m(&self, _dir: Vec3, _tile_lod_m: f32) -> Option<f32> {
        Some(self.height_m)
    }
}

pub type GpuAtlasMirrorHandle = Arc<RwLock<GpuAtlasHeightMirror>>;

pub struct GpuAtlasMirrorHeightSource {
    mirror: GpuAtlasMirrorHandle,
    fallback: CpuPipelineHeightSource,
}

impl GpuAtlasMirrorHeightSource {
    pub fn new(surface: Arc<dyn SurfaceQuery>) -> Self {
        Self {
            mirror: Arc::new(RwLock::new(GpuAtlasHeightMirror::default())),
            fallback: CpuPipelineHeightSource::new(surface),
        }
    }

    pub fn mirror(&self) -> GpuAtlasMirrorHandle {
        Arc::clone(&self.mirror)
    }
}

impl HeightSource for GpuAtlasMirrorHeightSource {
    fn sample_height_m(&self, dir: Vec3, tile_lod_m: f32) -> Option<f32> {
        if let Ok(mirror) = self.mirror.read()
            && let Some(height) = mirror.sample_height_m(dir)
        {
            return Some(height);
        }
        self.fallback.sample_height_m(dir, tile_lod_m)
    }

    fn revision(&self) -> u64 {
        self.mirror.read().map(|m| m.revision()).unwrap_or(0)
    }

    fn build_collider_patch(
        &self,
        center_dir: Vec3,
        max_resolution: u32,
    ) -> Option<TerrainPatchMesh> {
        self.mirror
            .read()
            .ok()?
            .build_collider_patch(center_dir, max_resolution)
    }
}

#[derive(Component, Clone)]
pub struct GpuAtlasHeightMirrorComponent {
    pub mirror: GpuAtlasMirrorHandle,
}

impl GpuAtlasHeightMirrorComponent {
    pub fn new(mirror: GpuAtlasMirrorHandle) -> Self {
        Self { mirror }
    }
}

#[derive(Default)]
pub struct GpuAtlasHeightMirror {
    model: Option<TerrainModel>,
    lod_count: u32,
    texture_size: u32,
    border_size: u32,
    min_height_m: f32,
    max_height_m: f32,
    tiles: HashMap<TileCoordinate, MirroredHeightTile>,
    /// Bumped whenever `sync_from_atlas` inserts, evicts, or updates a
    /// tile. Consumers compare against a stored snapshot to detect that
    /// the underlying height data has shifted (e.g. a finer LOD tile has
    /// just become resident) and rebuild any cached geometry derived
    /// from this mirror.
    revision: u64,
}

#[derive(Clone)]
struct MirroredHeightTile {
    atlas_index: u32,
    revision: u64,
    texels: MirroredHeightTexels,
}

#[derive(Clone)]
enum MirroredHeightTexels {
    R16(Vec<u16>),
    Rg16(Vec<[u16; 2]>),
    R32Float(Vec<f32>),
}

impl MirroredHeightTexels {
    fn len(&self) -> usize {
        match self {
            Self::R16(texels) => texels.len(),
            Self::Rg16(texels) => texels.len(),
            Self::R32Float(texels) => texels.len(),
        }
    }

    fn sample_unit(&self, texture_size: u32, border_size: u32, tile_uv: Vec2) -> Option<f32> {
        match self {
            Self::R16(texels) => sample_r16_tile(texels, texture_size, border_size, tile_uv)
                .map(|encoded| encoded as f32 / u16::MAX as f32),
            Self::Rg16(texels) => {
                sample_rg16_height_tile(texels, texture_size, border_size, tile_uv)
            }
            Self::R32Float(texels) => sample_f32_tile(texels, texture_size, border_size, tile_uv),
        }
    }

    fn unit_at(&self, index: usize) -> Option<f32> {
        match self {
            Self::R16(texels) => texels.get(index).map(|v| *v as f32 / u16::MAX as f32),
            Self::Rg16(texels) => texels.get(index).map(|v| decode_rg16_height(*v)),
            Self::R32Float(texels) => texels.get(index).copied(),
        }
    }
}

impl GpuAtlasHeightMirror {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn sync_from_atlas(&mut self, atlas: &TileAtlas) {
        let Some(height_index) = atlas.attachment_index("height") else {
            if !self.tiles.is_empty() {
                self.tiles.clear();
                self.revision = self.revision.wrapping_add(1);
            }
            return;
        };
        let Some(config) = atlas.attachment_configs().get(height_index as usize) else {
            if !self.tiles.is_empty() {
                self.tiles.clear();
                self.revision = self.revision.wrapping_add(1);
            }
            return;
        };

        self.model = Some(atlas.model().clone());
        self.lod_count = atlas.lod_count();
        self.texture_size = config.texture_size;
        self.border_size = config.border_size;
        self.min_height_m = atlas.model().min_height();
        self.max_height_m = atlas.model().max_height();

        let mut changed = false;

        let loaded = atlas.loaded_tiles();
        let loaded_coords: HashSet<TileCoordinate> =
            loaded.iter().map(|(coord, _)| *coord).collect();
        let before = self.tiles.len();
        self.tiles.retain(|coord, _| loaded_coords.contains(coord));
        if self.tiles.len() != before {
            changed = true;
        }

        for (coord, atlas_index) in loaded {
            let revision = atlas
                .attachment_slot_revision(height_index, atlas_index)
                .unwrap_or(0);
            let stale = self
                .tiles
                .get(&coord)
                .is_none_or(|tile| tile.atlas_index != atlas_index || tile.revision != revision);
            if !stale {
                continue;
            }
            let Some(texels) = atlas
                .attachment_data(height_index, atlas_index)
                .and_then(|data| {
                    data.as_r16()
                        .map(|texels| MirroredHeightTexels::R16(texels.to_vec()))
                        .or_else(|| {
                            data.as_rg16()
                                .map(|texels| MirroredHeightTexels::Rg16(texels.to_vec()))
                        })
                        .or_else(|| {
                            data.as_r32_float()
                                .map(|texels| MirroredHeightTexels::R32Float(texels.to_vec()))
                        })
                })
            else {
                continue;
            };
            self.tiles.insert(
                coord,
                MirroredHeightTile {
                    atlas_index,
                    revision,
                    texels,
                },
            );
            changed = true;
        }

        if changed {
            self.revision = self.revision.wrapping_add(1);
        }
    }

    pub fn sample_height_m(&self, dir: Vec3) -> Option<f32> {
        let model = self.model.as_ref()?;
        let dir = dir.normalize_or_zero();
        if dir == Vec3::ZERO || self.lod_count == 0 || self.texture_size == 0 {
            return None;
        }

        let sample_position = dir.as_dvec3() * model.scale();
        let coordinate = Coordinate::from_world_position(sample_position, model);

        for lod in (0..self.lod_count).rev() {
            let (tile_coord, tile_uv) = tile_lookup_at_lod(coordinate, lod);
            if let Some(tile) = self.tiles.get(&tile_coord) {
                let t = tile
                    .texels
                    .sample_unit(self.texture_size, self.border_size, tile_uv)?;
                return Some(self.min_height_m + (self.max_height_m - self.min_height_m) * t);
            }
        }
        None
    }

    /// Metric texel size (m) of the finest atlas tile currently resident at
    /// `dir`, or `None` if nothing is resident there. Note the pinned LOD-0 tile
    /// is *always* resident but kilometres-coarse, so callers that need the
    /// terrain to be genuinely detailed (e.g. grass building, so blades don't
    /// float above a coarse mesh) must check this, not mere presence.
    pub fn best_resident_texel_m(&self, dir: Vec3) -> Option<f32> {
        let model = self.model.as_ref()?;
        let dir = dir.normalize_or_zero();
        if dir == Vec3::ZERO || self.lod_count == 0 || self.texture_size == 0 {
            return None;
        }
        let inner = self
            .texture_size
            .saturating_sub(self.border_size * 2)
            .max(1) as f32;
        let sample_position = dir.as_dvec3() * model.scale();
        let coordinate = Coordinate::from_world_position(sample_position, model);
        for lod in (0..self.lod_count).rev() {
            let (tile_coord, _) = tile_lookup_at_lod(coordinate, lod);
            if self.tiles.contains_key(&tile_coord) {
                let face_arc_m =
                    std::f32::consts::FRAC_PI_2 * model.scale() as f32 / (1u32 << lod) as f32;
                return Some((face_arc_m / inner).max(0.0));
            }
        }
        None
    }

    /// Build a collider mesh from the finest resident tile under `center_dir`,
    /// one vertex per height texel at the tile's native resolution, each placed
    /// at the exact cube-sphere position the renderer uses
    /// ([`Coordinate::world_position`] applied to [`TileCoordinate::pixel_coordinate`]).
    /// The collider therefore lines up with the drawn surface by construction.
    ///
    /// A square window of up to `max_resolution` texels is centered on
    /// `center_dir` and clamped to the tile's logical (border-excluded) region,
    /// so every vertex is a real texel of this one tile — no cross-tile
    /// stitching. Returns `None` when no tile is resident (caller falls back to
    /// the tangent-grid resample).
    pub fn build_collider_patch(
        &self,
        center_dir: Vec3,
        max_resolution: u32,
    ) -> Option<TerrainPatchMesh> {
        let model = self.model.as_ref()?;
        let dir = center_dir.normalize_or_zero();
        if dir == Vec3::ZERO || self.lod_count == 0 || self.texture_size == 0 {
            return None;
        }
        let texture_size = self.texture_size;
        let border = self.border_size;
        let inner = texture_size.saturating_sub(border * 2);
        if inner < 2 {
            return None;
        }

        // Finest resident tile under the center direction, plus the in-tile uv
        // of that direction.
        let sample_position = dir.as_dvec3() * model.scale();
        let coordinate = Coordinate::from_world_position(sample_position, model);
        let (tile_coord, tile_uv) = (0..self.lod_count).rev().find_map(|lod| {
            let (tc, uv) = tile_lookup_at_lod(coordinate, lod);
            self.tiles.contains_key(&tc).then_some((tc, uv))
        })?;
        let tile = self.tiles.get(&tile_coord)?;
        let size = texture_size as usize;
        if tile.texels.len() < size * size {
            return None;
        }

        // Texel of `center_dir` within the tile's logical region — the inverse
        // of `pixel_coordinate`'s `in_tile_uv = (pixel + 0.5 - border) / inner`.
        let inner_f = inner as f64;
        let center_px = (tile_uv.x as f64 * inner_f + border as f64 - 0.5).round() as i64;
        let center_py = (tile_uv.y as f64 * inner_f + border as f64 - 0.5).round() as i64;

        // Square texel window at native resolution, clamped so every vertex is
        // a real texel inside the logical extent `[border, texture - border)`.
        let res = max_resolution.clamp(2, inner);
        let lo = border as i64;
        let hi = (texture_size - border) as i64; // exclusive
        let half = (res as i64 - 1) / 2;
        let x0 = (center_px - half).clamp(lo, hi - res as i64);
        let y0 = (center_py - half).clamp(lo, hi - res as i64);

        let translation = model.translation();
        let height_range = self.max_height_m - self.min_height_m;
        let n = res as usize;
        let mut vertices_body_m = Vec::with_capacity(n * n);
        for j in 0..res {
            let py = (y0 + j as i64) as u32;
            for i in 0..res {
                let px = (x0 + i as i64) as u32;
                let t = tile.texels.unit_at(py as usize * size + px as usize)?;
                let height = self.min_height_m + height_range * t;
                let coord = tile_coord.pixel_coordinate(UVec2::new(px, py), texture_size, border);
                // `translation` is zero for body-centered models, so this is the
                // body-fixed vertex; subtracting it keeps us correct if a model
                // is ever given a nonzero offset.
                vertices_body_m.push(coord.world_position(model, height) - translation);
            }
        }

        let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 2);
        for j in 0..(res - 1) {
            for i in 0..(res - 1) {
                let i0 = j * res + i;
                let i1 = i0 + 1;
                let i2 = i0 + res;
                let i3 = i2 + 1;
                indices.push([i0, i2, i1]);
                indices.push([i1, i2, i3]);
            }
        }

        let center_index = (res / 2) as usize * n + (res / 2) as usize;
        let center_surface_body_m = vertices_body_m[center_index];

        // Window's metric lateral half-extent (texel spacing × half the grid),
        // for window-relative collider rebuild scheduling. `res >= 2` so the
        // first two vertices (adjacent in x) always exist.
        let texel_spacing_m = (vertices_body_m[1] - vertices_body_m[0]).length();
        let half_extent_m = texel_spacing_m * (res as f64 - 1.0) * 0.5;

        Some(TerrainPatchMesh {
            vertices_body_m,
            indices,
            center_surface_body_m,
            basis: TerrainPatchBasis::from_normal(dir.as_dvec3()),
            half_extent_m,
        })
    }
}

pub(crate) fn sync_gpu_atlas_height_mirrors(
    mirrors: Query<(&TileAtlas, &GpuAtlasHeightMirrorComponent)>,
) {
    for (atlas, mirror) in &mirrors {
        if let Ok(mut mirror) = mirror.mirror.write() {
            mirror.sync_from_atlas(atlas);
        }
    }
}

fn tile_lookup_at_lod(coordinate: Coordinate, lod: u32) -> (TileCoordinate, Vec2) {
    let count = TileCoordinate::count(lod) as f64;
    let scaled = coordinate.uv * count;
    let max_xy = (count - 1.0).max(0.0);
    let x = scaled.x.floor().clamp(0.0, max_xy) as u32;
    let y = scaled.y.floor().clamp(0.0, max_xy) as u32;
    let tile_origin = DVec2::new(x as f64, y as f64);
    let tile_uv = (scaled - tile_origin).clamp(DVec2::ZERO, DVec2::ONE);
    (
        TileCoordinate {
            side: coordinate.side,
            lod,
            x,
            y,
        },
        tile_uv.as_vec2(),
    )
}

fn sample_r16_tile(
    texels: &[u16],
    texture_size: u32,
    border_size: u32,
    tile_uv: Vec2,
) -> Option<u16> {
    sample_tile_unit(texture_size, border_size, tile_uv, |index| {
        texels.get(index).map(|v| *v as f32)
    })
    .map(|v| v.round().clamp(0.0, u16::MAX as f32) as u16)
}

fn sample_rg16_height_tile(
    texels: &[[u16; 2]],
    texture_size: u32,
    border_size: u32,
    tile_uv: Vec2,
) -> Option<f32> {
    sample_tile_unit(texture_size, border_size, tile_uv, |index| {
        texels.get(index).map(|v| decode_rg16_height(*v))
    })
}

fn decode_rg16_height(texel: [u16; 2]) -> f32 {
    texel[0] as f32 / u16::MAX as f32 + texel[1] as f32 / (u16::MAX as f32 * u16::MAX as f32)
}

fn sample_f32_tile(
    texels: &[f32],
    texture_size: u32,
    border_size: u32,
    tile_uv: Vec2,
) -> Option<f32> {
    sample_tile_unit(texture_size, border_size, tile_uv, |index| {
        texels.get(index).copied()
    })
}

fn sample_tile_unit(
    texture_size: u32,
    border_size: u32,
    tile_uv: Vec2,
    mut sample_at: impl FnMut(usize) -> Option<f32>,
) -> Option<f32> {
    let size = texture_size as usize;
    if size == 0 {
        return None;
    }
    let center_size = texture_size.saturating_sub(border_size * 2).max(1);
    let scale = center_size as f32 / texture_size as f32;
    let offset = border_size as f32 / texture_size as f32;
    let atlas_uv = tile_uv * scale + Vec2::splat(offset);

    let max = texture_size as f32 - 1.001;
    let px = (atlas_uv.x * texture_size as f32 - 0.5).clamp(0.0, max);
    let py = (atlas_uv.y * texture_size as f32 - 0.5).clamp(0.0, max);
    let x0 = px.floor() as usize;
    let y0 = py.floor() as usize;
    let x1 = (x0 + 1).min(size - 1);
    let y1 = (y0 + 1).min(size - 1);
    let fx = px - px.floor();
    let fy = py - py.floor();

    let h00 = sample_at(y0 * size + x0)?;
    let h10 = sample_at(y0 * size + x1)?;
    let h01 = sample_at(y1 * size + x0)?;
    let h11 = sample_at(y1 * size + x1)?;
    let top = h00 + (h10 - h00) * fx;
    let bot = h01 + (h11 - h01) * fx;
    Some(top + (bot - top) * fy)
}
