use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use bevy::math::{DVec2, Vec2, Vec3};
use bevy::prelude::*;
use thalos_terrain::{DynamicSurfaceState, PlanetSurface};
use thalos_udlod::math::{Coordinate, TerrainModel, TileCoordinate};
use thalos_udlod::prelude::TileAtlas;

use crate::pipeline::rendered_height_m;

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
}

pub struct CpuPipelineHeightSource {
    surface: Arc<PlanetSurface>,
    dynamic_state: DynamicSurfaceState,
}

impl CpuPipelineHeightSource {
    pub fn new(surface: Arc<PlanetSurface>, dynamic_state: DynamicSurfaceState) -> Self {
        Self {
            surface,
            dynamic_state,
        }
    }
}

impl HeightSource for CpuPipelineHeightSource {
    fn sample_height_m(&self, dir: Vec3, tile_lod_m: f32) -> Option<f32> {
        Some(rendered_height_m(
            &self.surface,
            &self.dynamic_state,
            dir,
            tile_lod_m,
        ))
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
    pub fn new(surface: Arc<PlanetSurface>, dynamic_state: DynamicSurfaceState) -> Self {
        Self {
            mirror: Arc::new(RwLock::new(GpuAtlasHeightMirror::default())),
            fallback: CpuPipelineHeightSource::new(surface, dynamic_state),
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
    texels: Vec<u16>,
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
                .and_then(|data| data.as_r16())
            else {
                continue;
            };
            self.tiles.insert(
                coord,
                MirroredHeightTile {
                    atlas_index,
                    revision,
                    texels: texels.to_vec(),
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
                let encoded =
                    sample_r16_tile(&tile.texels, self.texture_size, self.border_size, tile_uv)?;
                let t = encoded as f32 / u16::MAX as f32;
                return Some(self.min_height_m + (self.max_height_m - self.min_height_m) * t);
            }
        }
        None
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
    let size = texture_size as usize;
    if texels.len() < size * size || size == 0 {
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

    let h00 = texels[y0 * size + x0] as f32;
    let h10 = texels[y0 * size + x1] as f32;
    let h01 = texels[y1 * size + x0] as f32;
    let h11 = texels[y1 * size + x1] as f32;
    let top = h00 + (h10 - h00) * fx;
    let bot = h01 + (h11 - h01) * fx;
    Some((top + (bot - top) * fy).round().clamp(0.0, u16::MAX as f32) as u16)
}
