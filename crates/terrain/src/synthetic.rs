//! [`SyntheticTileProvider`] — a deterministic procedural [`TileProvider`]
//! used as a stand-in until the M3-stage-2 `PipelineTileProvider` (which calls
//! into `thalos_terrain_gen`) lands.
//!
//! The pattern is intentionally crude: a coarse latitude/longitude ridge field
//! evaluated as a pure function of the world position returned by
//! [`TileCoordinate::pixel_coordinate`] → [`Coordinate::world_position`]. That
//! is the canonical pixel→position mapping the renderer samples with, so
//! neighbouring tiles produce bit-identical values on their shared border
//! pixels with no extra work.

use anyhow::Result;
use bevy::math::{DVec3, UVec2};
use bevy::tasks::{AsyncComputeTaskPool, Task};
use bevy_terrain::math::TileCoordinate;
use bevy_terrain::prelude::*;
use bevy_terrain::terrain_data::AttachmentData;

/// Deterministic procedural [`TileProvider`] used for Stage 1 of M3.
///
/// Stores the body's height range so the encoded R16 values map back to the
/// same physical height range the renderer reads from the [`TerrainModel`]
/// (whose `min_height`/`max_height` fields are crate-private upstream).
pub struct SyntheticTileProvider {
    min_height: f32,
    max_height: f32,
}

impl SyntheticTileProvider {
    pub fn new(min_height: f32, max_height: f32) -> Self {
        Self {
            min_height,
            max_height,
        }
    }
}

impl TileProvider for SyntheticTileProvider {
    fn supports_all_tiles(&self) -> bool {
        true
    }

    fn request_tile(
        &self,
        coord: TileCoordinate,
        model: &TerrainModel,
        attachments: &[AttachmentConfig],
    ) -> Task<Result<Vec<AttachmentData>>> {
        let model = model.clone();
        let attachments: Vec<AttachmentConfig> = attachments.to_vec();
        let min_height = self.min_height;
        let max_height = self.max_height;

        AsyncComputeTaskPool::get().spawn(async move {
            let mut datas = Vec::with_capacity(attachments.len());
            for cfg in &attachments {
                let data = match cfg.format {
                    AttachmentFormat::R16 => {
                        synthesize_height(&model, coord, cfg, min_height, max_height)
                    }
                    AttachmentFormat::Rgba8 => synthesize_zero_rgba8(cfg),
                    AttachmentFormat::Rg16 => synthesize_zero_rg16(cfg),
                    AttachmentFormat::Rgb8 => AttachmentData::None,
                };
                datas.push(data);
            }
            Ok(datas)
        })
    }
}

fn synthesize_height(
    model: &TerrainModel,
    coord: TileCoordinate,
    cfg: &AttachmentConfig,
    min_height: f32,
    max_height: f32,
) -> AttachmentData {
    let size = cfg.texture_size;
    let border = cfg.border_size;
    let span = (max_height - min_height).max(1.0);
    let mut out = Vec::with_capacity((size * size) as usize);

    for y in 0..size {
        for x in 0..size {
            let pixel = coord.pixel_coordinate(UVec2::new(x, y), size, border);
            // Extract the surface normal by differencing two heights. Works
            // regardless of the model's world translation — Stage 2 will spawn
            // terrains parented to body grids that are not centered at the
            // origin, so directly normalizing `world_position(model, 0.0)`
            // would not produce a valid surface normal there.
            let surface = pixel.world_position(model, 0.0);
            let lifted = pixel.world_position(model, 1.0);
            let dir = (lifted - surface).normalize();
            let h = ridge_height(dir);
            let t = ((h - min_height) / span).clamp(0.0, 1.0);
            out.push((t * u16::MAX as f32) as u16);
        }
    }
    AttachmentData::R16(out)
}

fn synthesize_zero_rgba8(cfg: &AttachmentConfig) -> AttachmentData {
    let count = (cfg.texture_size * cfg.texture_size) as usize;
    AttachmentData::Rgba8(vec![[0, 0, 0, 255]; count])
}

fn synthesize_zero_rg16(cfg: &AttachmentConfig) -> AttachmentData {
    let count = (cfg.texture_size * cfg.texture_size) as usize;
    AttachmentData::Rg16(vec![[0, 0]; count])
}

fn ridge_height(dir: DVec3) -> f32 {
    let lat = dir.y as f32;
    let lon = (dir.x as f32).atan2(dir.z as f32);
    let primary = (lat * 8.0).sin() * 1500.0;
    let secondary = (lon * 6.0).cos() * 800.0;
    let cross = ((lat + lon) * 12.0).sin() * 300.0;
    primary + secondary + cross
}
