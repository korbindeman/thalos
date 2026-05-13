//! [`PipelineTileProvider`] — bridges `bevy_terrain`'s [`TileProvider`] seam
//! to the Thalos synthesis pipeline.
//!
//! For each requested tile pixel the provider:
//!
//! 1. Maps `(coord, pixel)` → cubesphere [`Coordinate`] via
//!    [`TileCoordinate::pixel_coordinate`] — the canonical pixel→position
//!    mapping the renderer samples with, so tile borders are bit-identical.
//! 2. Lifts the coordinate to a body-local unit direction by differencing two
//!    [`Coordinate::world_position`] queries; that stays correct under any
//!    non-zero model translation (Stage 2/3 parents terrains to body grids
//!    whose origin is at the body centre, but we don't bake that assumption
//!    into the provider).
//! 3. Reads the matching baked cubemap texel directly out of
//!    [`StaticSurfaceData`].
//!
//! ## Why direct cubemap reads rather than `sample_static_surface`?
//!
//! `sample_static_surface` performs the full three-layer evaluation: cubemap
//! sample + SSBO crater iteration via the spatial index + statistical detail
//! noise. That's appropriate for fragment-shader-resolution evaluation (a
//! handful of samples per frame) but is far too expensive when materialising
//! a 512² tile attachment, which means **262 144 samples per tile per
//! attachment**. Empirically that path takes tens of seconds per tile, which
//! means the renderer perpetually samples the `INVALID_ATLAS_INDEX` fallback
//! (zero data) — the body shows up as a black sphere even with the impostor
//! hidden.
//!
//! For M3 the cubemap layer alone is enough: features large enough to read
//! from orbit are baked into the cubemap by the compiler (see
//! `cubemap_bake_threshold_m` in `StaticSurfaceData`), and the renderer
//! already does the right thing visually. Mid-size SSBO crater detail and
//! the statistical noise tail are deferred to a separate "detail" projection
//! the fragment shader will mix on top — same story as the impostor.
//!
//! The cubemap texel formats match the tile attachment formats one-to-one,
//! so the inner loops are pure indexed reads (no per-pixel arithmetic).
//!
//! The provider holds the [`PlanetSurface`] behind an [`Arc`] so multiple
//! tile requests can run concurrently without copying the (cubemap-heavy)
//! surface data.

use std::sync::Arc;

use anyhow::Result;
use bevy::math::{DVec3, UVec2, Vec3};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use bevy_terrain::math::TileCoordinate;
use bevy_terrain::prelude::*;
use bevy_terrain::terrain_data::AttachmentData;
use thalos_terrain_gen::{cubemap::dir_to_face_uv, PlanetSurface, StaticSurfaceData};

/// `bevy_terrain` `TileProvider` backed by the Thalos synthesis pipeline.
pub struct PipelineTileProvider {
    surface: Arc<PlanetSurface>,
    body_name: String,
}

impl PipelineTileProvider {
    pub fn new(body_name: impl Into<String>, surface: Arc<PlanetSurface>) -> Self {
        Self {
            surface,
            body_name: body_name.into(),
        }
    }
}

impl TileProvider for PipelineTileProvider {
    fn supports_all_tiles(&self) -> bool {
        true
    }

    fn request_tile(
        &self,
        coord: TileCoordinate,
        model: &TerrainModel,
        attachments: &[AttachmentConfig],
    ) -> Task<Result<Vec<AttachmentData>>> {
        let surface = self.surface.clone();
        let model = model.clone();
        let attachments: Vec<AttachmentConfig> = attachments.to_vec();
        let body_name = self.body_name.clone();

        AsyncComputeTaskPool::get().spawn(async move {
            let mut datas = Vec::with_capacity(attachments.len());
            for cfg in &attachments {
                let data = synthesize_attachment(&surface.static_surface, coord, &model, cfg, &body_name);
                datas.push(data);
            }
            Ok(datas)
        })
    }
}

fn synthesize_attachment(
    body: &StaticSurfaceData,
    coord: TileCoordinate,
    model: &TerrainModel,
    cfg: &AttachmentConfig,
    body_name: &str,
) -> AttachmentData {
    let size = cfg.texture_size;
    let border = cfg.border_size;
    let count = (size * size) as usize;

    match (cfg.format, cfg.name.as_str()) {
        (AttachmentFormat::R16, "height") => {
            let mut out = Vec::with_capacity(count);
            for y in 0..size {
                for x in 0..size {
                    let dir = pixel_direction(coord, UVec2::new(x, y), size, border, model);
                    // The cubemap encodes height as
                    // `texel/65535 * 2 - 1) * height_range`; bevy_terrain's
                    // R16 sampling decodes as `mix(min, max, texel/65535)`.
                    // With `min = -height_range`, `max = +height_range` the
                    // two encodings agree and we can copy the texel through.
                    out.push(cubemap_texel_nearest(&body.height_cubemap, dir));
                }
            }
            AttachmentData::R16(out)
        }
        (AttachmentFormat::Rgba8, "albedo") => {
            let mut out = Vec::with_capacity(count);
            for y in 0..size {
                for x in 0..size {
                    let dir = pixel_direction(coord, UVec2::new(x, y), size, border, model);
                    // Albedo cubemap stores sRGB-encoded bytes; the tile is
                    // uploaded as `Rgba8UnormSrgb`, so the byte representation
                    // is identical and we copy through.
                    out.push(cubemap_texel_nearest(&body.albedo_cubemap, dir));
                }
            }
            AttachmentData::Rgba8(out)
        }
        _ => {
            warn!(
                "PipelineTileProvider: unsupported attachment ({:?}, {:?}) on body {}; \
                 filling with zeros",
                cfg.name, cfg.format, body_name,
            );
            zero_attachment(cfg)
        }
    }
}

fn pixel_direction(
    coord: TileCoordinate,
    pixel: UVec2,
    texture_size: u32,
    border_size: u32,
    model: &TerrainModel,
) -> Vec3 {
    // Differencing two heights along the same `Coordinate` lifts to a
    // body-local surface normal regardless of `model.translation`, which is
    // important when Stage 2 parents terrains to body grids that are offset
    // from world origin.
    let pix = coord.pixel_coordinate(pixel, texture_size, border_size);
    let surface = pix.world_position(model, 0.0);
    let lifted = pix.world_position(model, 1.0);
    let dir: DVec3 = (lifted - surface).normalize();
    Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32)
}

/// Nearest-neighbour cubemap lookup. The cubemap face's UV space exactly
/// matches `dir_to_face_uv`'s output, so the inner loop is just an indexed
/// read into a `Vec`.
fn cubemap_texel_nearest<T>(cube: &thalos_terrain_gen::cubemap::Cubemap<T>, dir: Vec3) -> T
where
    T: Copy + Default,
{
    let (face, u, v) = dir_to_face_uv(dir);
    let res = cube.resolution();
    let x = ((u * res as f32) as u32).min(res - 1);
    let y = ((v * res as f32) as u32).min(res - 1);
    cube.get(face, x, y)
}

fn zero_attachment(cfg: &AttachmentConfig) -> AttachmentData {
    let count = (cfg.texture_size * cfg.texture_size) as usize;
    match cfg.format {
        AttachmentFormat::R16 => AttachmentData::R16(vec![0; count]),
        AttachmentFormat::Rg16 => AttachmentData::Rg16(vec![[0, 0]; count]),
        AttachmentFormat::Rgba8 => AttachmentData::Rgba8(vec![[0, 0, 0, 255]; count]),
        AttachmentFormat::Rgb8 => AttachmentData::None,
    }
}
