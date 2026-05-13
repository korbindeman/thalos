//! [`PipelineTileProvider`] — bridges `bevy_terrain`'s [`TileProvider`] seam
//! to the Thalos synthesis pipeline.
//!
//! For each requested tile pixel the provider:
//!
//! 1. Maps `(coord, pixel)` → cubesphere [`Coordinate`] via
//!    [`TileCoordinate::pixel_coordinate`] (the canonical pixel→position
//!    mapping the renderer samples with, so tile borders are bit-identical).
//! 2. Lifts the coordinate to a body-local direction by differencing two
//!    [`Coordinate::world_position`] queries (so the implementation is robust
//!    against any non-zero model translation Stage 2/3 might use).
//! 3. Queries [`sample_static_surface`] for height, normal, albedo, and
//!    roughness.
//! 4. Encodes the result into whichever attachments the renderer configured —
//!    height into `R16`, albedo into `Rgba8`, normal into `Rg16`, roughness
//!    into `R16`. Anything else falls back to zero data with a warning.
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
use thalos_terrain_gen::{PlanetSurface, sample_static_surface};

/// `bevy_terrain` `TileProvider` backed by the Thalos synthesis pipeline.
///
/// Holds the immutable [`PlanetSurface`] product. Tiles are synthesized on
/// `AsyncComputeTaskPool`; the synthesis itself is a pure function of
/// ellipsoid direction, so deterministic + seam-safe by construction.
pub struct PipelineTileProvider {
    surface: Arc<PlanetSurface>,
    /// The body's geometric radius in metres. Used to estimate the
    /// `meters-per-pixel` LOD parameter that `sample_static_surface`
    /// expects (`log2(pixel_size_m)`).
    body_radius_m: f32,
    height_range_m: f32,
    body_name: String,
}

impl PipelineTileProvider {
    pub fn new(
        body_name: impl Into<String>,
        surface: Arc<PlanetSurface>,
        body_radius_m: f32,
        height_range_m: f32,
    ) -> Self {
        Self {
            surface,
            body_radius_m,
            height_range_m,
            body_name: body_name.into(),
        }
    }
}

impl TileProvider for PipelineTileProvider {
    fn request_tile(
        &self,
        coord: TileCoordinate,
        model: &TerrainModel,
        attachments: &[AttachmentConfig],
    ) -> Task<Result<Vec<AttachmentData>>> {
        let surface = self.surface.clone();
        let model = model.clone();
        let attachments: Vec<AttachmentConfig> = attachments.to_vec();
        let body_radius_m = self.body_radius_m;
        let height_range_m = self.height_range_m;
        let body_name = self.body_name.clone();

        AsyncComputeTaskPool::get().spawn(async move {
            let mut datas = Vec::with_capacity(attachments.len());
            for cfg in &attachments {
                let data = synthesize_attachment(
                    &surface,
                    coord,
                    &model,
                    cfg,
                    body_radius_m,
                    height_range_m,
                    &body_name,
                );
                datas.push(data);
            }
            Ok(datas)
        })
    }
}

fn synthesize_attachment(
    surface: &PlanetSurface,
    coord: TileCoordinate,
    model: &TerrainModel,
    cfg: &AttachmentConfig,
    body_radius_m: f32,
    height_range_m: f32,
    body_name: &str,
) -> AttachmentData {
    let size = cfg.texture_size;
    let border = cfg.border_size;
    let pixel_size_m = approximate_pixel_size_m(coord, size, border, body_radius_m);
    let lod_param = pixel_size_m.log2();

    let count = (size * size) as usize;
    let static_surface = &surface.static_surface;

    match cfg.format {
        AttachmentFormat::R16 if cfg.name == "height" => {
            let mut out = Vec::with_capacity(count);
            for y in 0..size {
                for x in 0..size {
                    let dir = pixel_direction(coord, UVec2::new(x, y), size, border, model);
                    let h = sample_static_surface(static_surface, dir, lod_param).height;
                    out.push(encode_height_r16(h, height_range_m));
                }
            }
            AttachmentData::R16(out)
        }
        AttachmentFormat::R16 if cfg.name == "roughness" => {
            let mut out = Vec::with_capacity(count);
            for y in 0..size {
                for x in 0..size {
                    let dir = pixel_direction(coord, UVec2::new(x, y), size, border, model);
                    let r = sample_static_surface(static_surface, dir, lod_param).roughness;
                    out.push(encode_unit_r16(r));
                }
            }
            AttachmentData::R16(out)
        }
        AttachmentFormat::Rgba8 if cfg.name == "albedo" => {
            let mut out = Vec::with_capacity(count);
            for y in 0..size {
                for x in 0..size {
                    let dir = pixel_direction(coord, UVec2::new(x, y), size, border, model);
                    let sample = sample_static_surface(static_surface, dir, lod_param);
                    out.push(encode_albedo_srgb(sample.albedo));
                }
            }
            AttachmentData::Rgba8(out)
        }
        AttachmentFormat::Rg16 if cfg.name == "normal" => {
            let mut out = Vec::with_capacity(count);
            for y in 0..size {
                for x in 0..size {
                    let dir = pixel_direction(coord, UVec2::new(x, y), size, border, model);
                    let n = sample_static_surface(static_surface, dir, lod_param).normal;
                    out.push(encode_normal_rg16(n));
                }
            }
            AttachmentData::Rg16(out)
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
    // Differencing two heights produces a body-local surface normal regardless
    // of the model's world translation (Stage 2 parents terrains to body grids
    // whose origin is at the body center, but we want the impl to stay correct
    // if that ever changes).
    let pix = coord.pixel_coordinate(pixel, texture_size, border_size);
    let surface = pix.world_position(model, 0.0);
    let lifted = pix.world_position(model, 1.0);
    let dir: DVec3 = (lifted - surface).normalize();
    Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32)
}

/// Coarse meters-per-pixel estimate for the LOD parameter.
///
/// One cubesphere face covers a quadrant of the sphere → its diagonal arc
/// length is approximately `2 π R / 4`. Each tile at LOD `l` covers
/// `1/2^l` of that arc, and each tile texel covers `arc / inner_size` metres.
/// This is approximate (the cubesphere face is not isotropic) but good enough
/// for `log2(m)` LOD selection.
fn approximate_pixel_size_m(
    coord: TileCoordinate,
    texture_size: u32,
    border_size: u32,
    body_radius_m: f32,
) -> f32 {
    let face_arc_m = std::f32::consts::FRAC_PI_2 * body_radius_m;
    let tiles_per_face = (1u32 << coord.lod) as f32;
    let inner = (texture_size - 2 * border_size).max(1) as f32;
    face_arc_m / tiles_per_face / inner
}

fn encode_height_r16(height_m: f32, height_range_m: f32) -> u16 {
    // `StaticSurfaceData::height_range` is the ± range that the cubemap height
    // encodes. The renderer maps `0..1` to `[min_height, max_height]` where
    // `min = -range`, `max = +range`, so map `height` linearly into `0..1`.
    let span = (2.0 * height_range_m).max(1.0);
    let t = ((height_m + height_range_m) / span).clamp(0.0, 1.0);
    (t * u16::MAX as f32) as u16
}

fn encode_unit_r16(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * u16::MAX as f32) as u16
}

fn encode_albedo_srgb(albedo_linear: Vec3) -> [u8; 4] {
    let to_srgb = |c: f32| -> u8 {
        let c = c.clamp(0.0, 1.0);
        let srgb = if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (srgb * 255.0 + 0.5) as u8
    };
    [
        to_srgb(albedo_linear.x),
        to_srgb(albedo_linear.y),
        to_srgb(albedo_linear.z),
        255,
    ]
}

fn encode_normal_rg16(normal_local: Vec3) -> [u16; 2] {
    // Pack the object-space normal's xy components into RG; the renderer
    // reconstructs `z = sqrt(1 - x² - y²)`. Stage 2 doesn't enable
    // `LIGHTING` in the shader yet so this is currently visually unused,
    // but the attachment stays populated for the eventual PBR path.
    let n = normal_local.normalize_or_zero();
    let encode = |v: f32| -> u16 {
        let t = (0.5 * (v + 1.0)).clamp(0.0, 1.0);
        (t * u16::MAX as f32) as u16
    };
    [encode(n.x), encode(n.y)]
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
