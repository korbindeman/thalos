//! [`PipelineTileProvider`] — bridges `thalos_udlod`'s [`TileProvider`] seam
//! to the Thalos synthesis pipeline.
//!
//! As of P0 of the planet-generation migration
//! ([docs/planet-generation-pipeline-migration.md]), the surface itself is
//! evaluated by the **Query API seam** in `thalos_terrain`
//! ([`thalos_terrain::query`]): one band-limited surface shared by the
//! impostor, these UDLOD tiles, and the physics collider. This module is now
//! purely the UDLOD-side adapter:
//!
//! 1. Map each tile pixel's `TileCoordinate` to a body-local direction
//!    ([`pixel_direction`], using UDLOD's canonical cube-sphere projection).
//! 2. Evaluate the surface at that direction via
//!    [`thalos_terrain::surface_sample`].
//! 3. Encode the result into the configured tile attachments
//!    ([`encode_attachment`]).
//!
//! The detail cascade, base cubemap sampling, dynamic-layer compositing, and
//! sea-level capping that used to live here moved into
//! [`thalos_terrain::query`] so there is exactly one surface synthesiser. See
//! that module for the cascade details.
//!
//! # Known limitation: CPU/GPU bilinear stand-off
//!
//! Tile R16 data is bilinearly sampled by the GPU; [`rendered_height_m`]
//! evaluates the surface pointwise at the requested `dir`. Off pixel centres
//! the two values disagree by up to one peak-to-trough of the resolved detail
//! amplitude (O(10–20 cm) at sub-metre wavelength). Acceptable for v1; the
//! `GpuAtlasMirrorHeightSource` path closes most of this by sampling the
//! actual resident atlas tiles. Tracked separately.

use std::sync::Arc;

use anyhow::Result;
use bevy::math::{DVec3, UVec2, Vec3};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use rayon::prelude::*;
use thalos_terrain::{
    DynamicSurfaceState, PlanetSurface, StaticSurfaceData, surface_height_m,
    surface_height_range_m, surface_sample,
};
use thalos_udlod::math::TileCoordinate;
use thalos_udlod::prelude::*;
use thalos_udlod::terrain_data::AttachmentData;

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// `thalos_udlod` `TileProvider` backed by the Thalos synthesis pipeline.
pub struct PipelineTileProvider {
    surface: Arc<PlanetSurface>,
    dynamic_state: DynamicSurfaceState,
    height_range_m: f32,
    body_name: String,
}

impl PipelineTileProvider {
    pub fn new(
        body_name: impl Into<String>,
        surface: Arc<PlanetSurface>,
        dynamic_state: DynamicSurfaceState,
        height_range_m: f32,
    ) -> Self {
        Self {
            surface,
            dynamic_state,
            height_range_m,
            body_name: body_name.into(),
        }
    }
}

/// Vertical range (metres) the ground LOD must encode: static + dynamic +
/// procedural detail headroom. Thin wrapper over the Query API seam
/// ([`thalos_terrain::surface_height_range_m`]); used by `ground_terrain` to
/// set up the thalos_udlod `TerrainModel` height bounds.
pub fn rendered_height_range(surface: &PlanetSurface, state: &DynamicSurfaceState) -> f32 {
    surface_height_range_m(surface, state)
}

/// Returns the `tile_lod_m` the renderer currently uses at `world_position`
/// (in the terrain model's local frame, i.e. the body grid frame for
/// spherical terrains). Mirrors the GPU's tile-residency lookup: derives the
/// value from the deepest atlas tile actually resident at this direction, so
/// CPU height queries land on the same surface the GPU draws.
///
/// `None` is returned only when no ancestor tile is resident yet (early frames
/// before any bake has completed). Callers should fall back to a detail-free
/// query in that case — passing a fine `tile_lod_m` would engage the full
/// procedural cascade and produce a CPU/GPU height gap.
///
/// The atlas's `"height"` attachment is used to derive metres-per-texel.
/// Atlases without a height attachment return `None`.
pub fn renderer_tile_lod_m_at(
    tile_atlas: &TileAtlas,
    tile_tree: &TileTree,
    world_position: DVec3,
) -> Option<f32> {
    let resident_lod = tile_tree.best_resident_atlas_lod(world_position, tile_atlas.model())?;
    let height_cfg = tile_atlas
        .attachment_configs()
        .iter()
        .find(|c| c.name == "height")?;
    let inner_texels = height_cfg
        .texture_size
        .saturating_sub(height_cfg.border_size * 2)
        .max(1);
    let lod_div = (1u32 << resident_lod).max(1) as f32;
    let face_radians = std::f32::consts::FRAC_PI_2 / lod_div;
    Some((tile_atlas.model().scale() as f32 * face_radians / inner_texels as f32).max(1.0))
}

impl TileProvider for PipelineTileProvider {
    fn request_tile(
        &self,
        coord: TileCoordinate,
        model: &TerrainModel,
        attachments: &[AttachmentConfig],
    ) -> Task<Result<Vec<AttachmentData>>> {
        let surface = self.surface.clone();
        let dynamic_state = self.dynamic_state.clone();
        let height_range_m = self.height_range_m;
        let model = model.clone();
        let attachments: Vec<AttachmentConfig> = attachments.to_vec();
        let body_name = self.body_name.clone();

        AsyncComputeTaskPool::get().spawn(async move {
            // All requested attachments share the same per-pixel evaluation,
            // so resolve the largest texture size once, evaluate, then encode
            // each attachment from the shared buffer. Different sizes would
            // require separate passes; in practice every attachment in a body
            // config uses the same `texture_size`, so just assert that.
            let Some(first) = attachments.first() else {
                return Ok(Vec::new());
            };
            let size = first.texture_size;
            let border = first.border_size;
            debug_assert!(
                attachments
                    .iter()
                    .all(|c| c.texture_size == size && c.border_size == border),
                "PipelineTileProvider expects every attachment in a tile to share \
                 the same texture_size and border_size",
            );

            let pixels = compute_tile_pixels(&surface, &dynamic_state, coord, &model, size, border);

            let mut datas = Vec::with_capacity(attachments.len());
            for cfg in &attachments {
                datas.push(encode_attachment(cfg, &pixels, height_range_m, &body_name));
            }
            Ok(datas)
        })
    }
}

// ---------------------------------------------------------------------------
// Per-pixel evaluation (delegates to the Query API seam)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct TilePixel {
    height_m: f32,
    albedo_linear: Vec3,
    roughness: f32,
    /// Packed procedural material intent for the ground shader:
    /// R = vegetation/grass, G = soil/peat/sediment, B = exposed rock,
    /// A = wetness/cavity darkening. The channels are masks, not final color.
    material_rgba: [u8; 4],
}

fn compute_tile_pixels(
    surface: &PlanetSurface,
    dynamic_state: &DynamicSurfaceState,
    coord: TileCoordinate,
    model: &TerrainModel,
    size: u32,
    border: u32,
) -> Vec<TilePixel> {
    let body = &surface.static_surface;
    let tile_lod_m = tile_lod_m(body, coord, size, border);

    let count = (size * size) as usize;
    let mut pixels = vec![TilePixel::default(); count];

    pixels
        .par_chunks_mut(size as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, pixel) in row.iter_mut().enumerate() {
                let dir =
                    pixel_direction(coord, UVec2::new(x as u32, y as u32), size, border, model);
                let sample = surface_sample(surface, dynamic_state, dir, tile_lod_m);
                let material_rgba =
                    material_masks(surface, dynamic_state, dir, sample.height_m, tile_lod_m);
                *pixel = TilePixel {
                    height_m: sample.height_m,
                    albedo_linear: sample.albedo_linear,
                    roughness: sample.roughness,
                    material_rgba,
                };
            }
        });

    pixels
}

/// Canonical "what does the ground LOD render at this direction?" height query.
///
/// Thin wrapper over the Query API seam ([`thalos_terrain::surface_height_m`]),
/// kept under this name so the existing consumers (terrain colliders, the
/// character controller's height source, camera boom ray-casts, HUD altitude
/// readouts) need no churn. The seam is the single source of truth shared with
/// the atlas baker above.
///
/// Pass a small `tile_lod_m` (e.g. `0.5`) for full procedural detail near the
/// camera; pass the patch's vertex spacing when building a coarser collider
/// mesh so the mesh resolution matches the represented detail.
pub fn rendered_height_m(
    surface: &PlanetSurface,
    dynamic_state: &DynamicSurfaceState,
    dir: Vec3,
    tile_lod_m: f32,
) -> f32 {
    surface_height_m(surface, dynamic_state, dir, tile_lod_m)
}

fn material_masks(
    surface: &PlanetSurface,
    dynamic_state: &DynamicSurfaceState,
    dir: Vec3,
    height_m: f32,
    tile_lod_m: f32,
) -> [u8; 4] {
    let body = &surface.static_surface;
    let radius_m = body.radius_m.max(1.0);
    let step_m = tile_lod_m.clamp(2.0, 250.0);
    let angular_step = step_m / radius_m;

    let normal = dir.normalize_or_zero();
    if normal == Vec3::ZERO {
        return [128, 96, 32, 0];
    }
    let tangent_seed = if normal.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent = tangent_seed.cross(normal).normalize_or_zero();
    let bitangent = normal.cross(tangent).normalize_or_zero();

    let h_l = surface_height_m(
        surface,
        dynamic_state,
        rotate_dir(normal, tangent, -angular_step),
        tile_lod_m,
    );
    let h_r = surface_height_m(
        surface,
        dynamic_state,
        rotate_dir(normal, tangent, angular_step),
        tile_lod_m,
    );
    let h_d = surface_height_m(
        surface,
        dynamic_state,
        rotate_dir(normal, bitangent, -angular_step),
        tile_lod_m,
    );
    let h_u = surface_height_m(
        surface,
        dynamic_state,
        rotate_dir(normal, bitangent, angular_step),
        tile_lod_m,
    );

    let grad_x = (h_r - h_l) / (2.0 * step_m);
    let grad_y = (h_u - h_d) / (2.0 * step_m);
    let slope = (grad_x * grad_x + grad_y * grad_y).sqrt();
    let laplacian = ((h_l + h_r + h_d + h_u) * 0.25 - height_m) / step_m.max(1.0);

    let slope_rock = smoothstep(0.20, 0.75, slope);
    let high_rock = smoothstep(2_200.0, 6_000.0, height_m);
    let convex_rock = smoothstep(0.04, 0.20, -laplacian);
    let rock = (slope_rock * 0.82 + high_rock * 0.18 + convex_rock * 0.16).clamp(0.0, 0.95);

    let hollow = smoothstep(0.035, 0.18, laplacian);
    let wetness = (hollow * (1.0 - smoothstep(1_500.0, 4_500.0, height_m)) * (1.0 - rock * 0.7))
        .clamp(0.0, 1.0);
    let soil = (smoothstep(0.035, 0.28, slope) * (1.0 - smoothstep(0.45, 0.9, slope))
        + hollow * 0.45
        + wetness * 0.25)
        .clamp(0.0, 1.0)
        * (1.0 - rock * 0.65);
    let grass = ((1.0 - rock) * (1.0 - soil * 0.45) * (1.0 - wetness * 0.25)).clamp(0.0, 1.0);

    let sum = (grass + soil + rock).max(1.0e-4);
    [
        quantize_unit_to_u8(grass / sum),
        quantize_unit_to_u8(soil / sum),
        quantize_unit_to_u8(rock / sum),
        quantize_unit_to_u8(wetness),
    ]
}

fn rotate_dir(dir: Vec3, axis: Vec3, angle: f32) -> Vec3 {
    (dir * angle.cos() + axis * axis.dot(dir) * (1.0 - angle.cos()) + axis.cross(dir) * angle.sin())
        .normalize_or_zero()
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(f32::EPSILON)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

fn encode_attachment(
    cfg: &AttachmentConfig,
    pixels: &[TilePixel],
    height_range_m: f32,
    body_name: &str,
) -> AttachmentData {
    match (cfg.format, cfg.name.as_str()) {
        (AttachmentFormat::R16, "height") => AttachmentData::R16(
            pixels
                .iter()
                .map(|p| encode_height(p.height_m, height_range_m))
                .collect(),
        ),
        (AttachmentFormat::Rg16, "height") => AttachmentData::Rg16(
            pixels
                .iter()
                .map(|p| encode_height_rg16(p.height_m, height_range_m))
                .collect(),
        ),
        (AttachmentFormat::R32Float, "height") => AttachmentData::R32Float(
            pixels
                .iter()
                .map(|p| encode_height_unit(p.height_m, height_range_m))
                .collect(),
        ),
        (AttachmentFormat::Rgba8, "albedo") => AttachmentData::Rgba8(
            pixels
                .iter()
                .map(|p| linear_rgb_to_srgba8(p.albedo_linear))
                .collect(),
        ),
        (AttachmentFormat::R16, "roughness") => AttachmentData::R16(
            pixels
                .iter()
                .map(|p| quantize_unit_to_u16(p.roughness))
                .collect(),
        ),
        (AttachmentFormat::Rgba8, "material") => {
            AttachmentData::Rgba8(pixels.iter().map(|p| p.material_rgba).collect())
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

// ---------------------------------------------------------------------------
// Tile-coordinate helpers
// ---------------------------------------------------------------------------

fn tile_lod_m(
    body: &StaticSurfaceData,
    coord: TileCoordinate,
    texture_size: u32,
    border_size: u32,
) -> f32 {
    let inner_texels = texture_size.saturating_sub(border_size * 2).max(1);
    let face_radians = std::f32::consts::FRAC_PI_2 / TileCoordinate::count(coord.lod).max(1) as f32;
    (body.radius_m * face_radians / inner_texels as f32).max(1.0)
}

fn encode_height(height_m: f32, range: f32) -> u16 {
    (encode_height_unit(height_m, range) * 65535.0 + 0.5) as u16
}

fn encode_height_unit(height_m: f32, range: f32) -> f32 {
    if range <= f32::EPSILON {
        return 0.5;
    }
    (height_m / range).clamp(-1.0, 1.0) * 0.5 + 0.5
}

fn encode_height_rg16(height_m: f32, range: f32) -> [u16; 2] {
    let unit = encode_height_unit(height_m, range);
    let coarse = (unit * u16::MAX as f32).floor() / u16::MAX as f32;
    let residual = ((unit - coarse) * u16::MAX as f32).clamp(0.0, 1.0);
    [
        (coarse * u16::MAX as f32 + 0.5) as u16,
        (residual * u16::MAX as f32 + 0.5) as u16,
    ]
}

fn pixel_direction(
    coord: TileCoordinate,
    pixel: UVec2,
    texture_size: u32,
    border_size: u32,
    model: &TerrainModel,
) -> Vec3 {
    // Differencing two heights along the same `Coordinate` lifts to a
    // body-local surface normal regardless of `model.translation`.
    let pix =
        coord.stitched_pixel_coordinate(pixel, texture_size, border_size, model.is_spherical());
    let surface = pix.world_position(model, 0.0);
    let lifted = pix.world_position(model, 1.0);
    let dir: DVec3 = (lifted - surface).normalize();
    Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32)
}

fn linear_rgb_to_srgba8(linear: Vec3) -> [u8; 4] {
    [
        linear_to_srgb8(linear.x),
        linear_to_srgb8(linear.y),
        linear_to_srgb8(linear.z),
        255,
    ]
}

fn linear_to_srgb8(linear: f32) -> u8 {
    let linear = linear.clamp(0.0, 1.0);
    let srgb = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    quantize_unit_to_u8(srgb)
}

fn quantize_unit_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn quantize_unit_to_u16(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

fn zero_attachment(cfg: &AttachmentConfig) -> AttachmentData {
    let count = (cfg.texture_size * cfg.texture_size) as usize;
    match cfg.format {
        AttachmentFormat::R16 => AttachmentData::R16(vec![0; count]),
        AttachmentFormat::R32Float => AttachmentData::R32Float(vec![0.0; count]),
        AttachmentFormat::Rg16 => AttachmentData::Rg16(vec![[0, 0]; count]),
        AttachmentFormat::Rgba8 => AttachmentData::Rgba8(vec![[0, 0, 0, 255]; count]),
        AttachmentFormat::Rgb8 => AttachmentData::None,
    }
}
