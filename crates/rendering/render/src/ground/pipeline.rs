//! [`PipelineTileProvider`] — bridges `thalos_udlod`'s [`TileProvider`] seam
//! to the Thalos synthesis pipeline.
//!
//! As of P0 of the planet-generation migration
//! ([docs/archive/planet-generation-pipeline-migration.md]), the surface itself is
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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use bevy::math::{DVec2, DVec3, UVec2, Vec3};
use bevy::prelude::*;
use bevy::tasks::Task;
use rayon::prelude::*;
use thalos_terrain::SurfaceQuery;
use thalos_udlod::math::{Coordinate, TileCoordinate};
use thalos_udlod::prelude::*;
use thalos_udlod::terrain_data::AttachmentData;

use crate::ground::material_masks::material_masks_from_heights;
use crate::ground::tile_synthesis_pool::{tile_eval_pool, tile_synthesis_pool};

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// `thalos_udlod` `TileProvider` backed by the Thalos synthesis pipeline.
pub struct PipelineTileProvider {
    /// The terrain black box, behind the [`SurfaceQuery`] seam. The provider
    /// names no generation internals (`PlanetSurface`/`StaticSurfaceData`); the
    /// construction site supplies a `BakedSurface` (or any future backing) as
    /// `Arc<dyn SurfaceQuery>`.
    surface: Arc<dyn SurfaceQuery>,
    body_name: String,
    /// Memoized per-tile screen-space-error scale (see
    /// [`TileProvider::subdivision_scale`]). The tile tree queries this for every
    /// tile in the request set each frame, so the relief probe behind it must be
    /// evaluated at most once per coordinate.
    relief_scale: Mutex<HashMap<TileCoordinate, f64>>,
}

impl PipelineTileProvider {
    pub fn new(body_name: impl Into<String>, surface: Arc<dyn SurfaceQuery>) -> Self {
        Self {
            surface,
            body_name: body_name.into(),
            relief_scale: Mutex::new(HashMap::new()),
        }
    }
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
        let model = model.clone();
        let attachments: Vec<AttachmentConfig> = attachments.to_vec();
        let body_name = self.body_name.clone();

        tile_synthesis_pool().spawn(async move {
            let height_range_m = surface.height_range_m();
            if attachments.is_empty() {
                return Ok(Vec::new());
            }

            // Attachments may declare **different resolutions**: height carries
            // the silhouette and wants the full grid, while albedo/material are
            // macro-colour and mask anchors that read fine at half. The GPU atlas
            // already sizes each attachment's texture array independently, so the
            // only cost is here — evaluate the field once per *distinct* grid
            // (not once per attachment) and encode every attachment sharing that
            // grid from the one buffer.
            let mut grids: HashMap<(u32, u32), Vec<TilePixel>> = HashMap::new();
            for cfg in &attachments {
                grids
                    .entry((cfg.texture_size, cfg.border_size))
                    .or_insert_with(|| {
                        compute_tile_pixels(
                            surface.as_ref(),
                            coord,
                            &model,
                            cfg.texture_size,
                            cfg.border_size,
                        )
                    });
            }

            let mut datas = Vec::with_capacity(attachments.len());
            for cfg in &attachments {
                let pixels = &grids[&(cfg.texture_size, cfg.border_size)];
                let mut data = encode_attachment(cfg, pixels, height_range_m, &body_name);
                // Mip generation is the provider's job (see the `TileProvider`
                // contract): doing it here runs it on the synthesis pool instead
                // of the main thread in `TileAtlas::update`, and means the cache
                // wrappers store a fully-mipped payload — a cache hit then costs
                // neither synthesis nor mip filtering.
                data.generate_mipmaps(cfg.texture_size, cfg.mip_level_count);
                datas.push(data);
            }
            Ok(datas)
        })
    }

    /// Relief-aware refinement: a tile whose footprint is nearly flat does not
    /// need the same subdivision distance as a mountainside. Probes the surface
    /// on a coarse grid across the tile, turns the relative relief into a scale
    /// in `[SSE_FLAT_SCALE, 1]`, and memoizes it per coordinate (the tile tree
    /// asks every frame; the probe must run once).
    fn subdivision_scale(&self, coord: TileCoordinate, model: &TerrainModel) -> f64 {
        if let Some(scale) = self
            .relief_scale
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&coord)
        {
            return *scale;
        }

        let scale = compute_subdivision_scale(self.surface.as_ref(), coord, model);
        let mut memo = self.relief_scale.lock().unwrap_or_else(|p| p.into_inner());
        // Flying across a planet touches unboundedly many coordinates, so this memo
        // is a slow leak if left to grow. It's a pure function of the coordinate,
        // so dropping it wholesale is always safe — just re-probe. Cheaper and far
        // simpler than tracking recency for a value this cheap to recompute.
        if memo.len() >= RELIEF_MEMO_CAPACITY {
            memo.clear();
        }
        memo.insert(coord, scale);
        scale
    }
}

/// Coordinates retained by the relief memo before it is dropped and refilled.
/// Comfortably above the request set of any single view (hundreds), so steady-state
/// play never clears; a long traversal clears occasionally and re-probes.
const RELIEF_MEMO_CAPACITY: usize = 32_768;

/// Subdivision scale for a tile from its relative relief.
///
/// "Relative" is the point: absolute metres of relief mean nothing without the
/// tile's own footprint — 50 m across a 100 km LOD-2 tile is a plain, across a
/// 100 m LOD-15 tile it is a cliff. Dividing by the tile's arc length gives a
/// grade the threshold is scale-free in.
fn compute_subdivision_scale(
    surface: &dyn SurfaceQuery,
    coord: TileCoordinate,
    model: &TerrainModel,
) -> f64 {
    let radius_m = surface.radius_m();
    let tile_count = TileCoordinate::count(coord.lod).max(1) as f32;
    let tile_span_m = radius_m * std::f32::consts::FRAC_PI_2 / tile_count;
    if tile_span_m <= f32::EPSILON {
        return 1.0;
    }
    // Probe at the tile's own scale so the cascade resolves the relief this tile
    // would actually show, not sub-texel noise.
    let probe_lod_m = (tile_span_m / RELIEF_PROBE_GRID as f32).max(1.0);

    let mut min_h = f32::INFINITY;
    let mut max_h = f32::NEG_INFINITY;
    for j in 0..=RELIEF_PROBE_GRID {
        for i in 0..=RELIEF_PROBE_GRID {
            let u = i as f64 / RELIEF_PROBE_GRID as f64;
            let v = j as f64 / RELIEF_PROBE_GRID as f64;
            let uv = (DVec2::new(coord.x as f64 + u, coord.y as f64 + v)) / tile_count as f64;
            let dir = coordinate_direction(Coordinate::new(coord.side, uv), model);
            let h = surface.sample_height_m(dir.as_vec3(), probe_lod_m);
            min_h = min_h.min(h);
            max_h = max_h.max(h);
        }
    }
    if !min_h.is_finite() || !max_h.is_finite() {
        return 1.0;
    }

    let grade = ((max_h - min_h) / tile_span_m) as f64;
    // Full detail once the tile carries a real grade; ramp down toward
    // `SSE_FLAT_SCALE` as it flattens out.
    let t = (grade / RELIEF_FULL_DETAIL_GRADE).clamp(0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    SSE_FLAT_SCALE + (1.0 - SSE_FLAT_SCALE) * smooth
}

/// Samples per tile side for the relief probe (a `(N+1)²` grid).
const RELIEF_PROBE_GRID: u32 = 4;
/// Relief-over-span ratio at which a tile earns full distance-driven detail.
/// ~3% grade across the tile; below that it starts refining less.
const RELIEF_FULL_DETAIL_GRADE: f64 = 0.03;
/// Subdivision scale for dead-flat terrain (ocean, salt pan, a flattened pad).
/// Kept above the tile tree's `SSE_MIN_SCALE` floor so this is the binding value.
const SSE_FLAT_SCALE: f64 = 0.6;

/// Body-local unit direction for a cube-sphere coordinate. Differencing two
/// lifted positions keeps this valid regardless of `model.translation` (mirrors
/// [`pixel_direction`]).
fn coordinate_direction(coordinate: Coordinate, model: &TerrainModel) -> DVec3 {
    let surface = coordinate.world_position(model, 0.0);
    let lifted = coordinate.world_position(model, 1.0);
    (lifted - surface).normalize()
}

// ---------------------------------------------------------------------------
// Per-pixel evaluation (delegates to the Query API seam)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct TilePixel {
    height_m: f32,
    albedo_linear: Vec3,
    roughness: f32,
    /// Macro landcover moisture in `[-1, 1]` — encoded into the albedo
    /// attachment's **alpha** channel (linear even on the sRGB texture), where
    /// the ground shader decodes it and adds its wrapped fine detail tier
    /// (docs/world/terrain_macro.md). The attachment alpha is NOT opacity.
    moisture: f32,
    /// Packed procedural material intent for the ground shader:
    /// R = vegetation/grass, G = soil/peat/sediment, B = exposed rock,
    /// A = wetness/cavity darkening. The channels are masks, not final color.
    material_rgba: [u8; 4],
}

fn compute_tile_pixels(
    surface: &dyn SurfaceQuery,
    coord: TileCoordinate,
    model: &TerrainModel,
    size: u32,
    border: u32,
) -> Vec<TilePixel> {
    let tile_lod_m = tile_lod_m(surface.radius_m(), coord, size, border);

    let count = (size * size) as usize;
    let mut pixels = vec![TilePixel::default(); count];

    // Each pixel is an independent (and, for Thalos, expensive) field sample.
    // Parallelise across rows with rayon: a cold view — e.g. a teleport straight
    // to a runway — only needs a handful of tiles, so most cores would otherwise
    // sit idle while one worker thread evaluates 262 k samples serially. Spreading
    // each tile across all cores cuts its wall-clock latency roughly by the core
    // count, which is what lets the ground under a fresh surface spawn resolve in
    // a few seconds instead of tens. Under heavy streaming the synthesis pool's
    // workers already saturate the cores, so rayon just load-balances — no
    // throughput regression.
    tile_eval_pool().install(|| {
        pixels
            .par_chunks_mut(size as usize)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.iter_mut().enumerate() {
                    let dir =
                        pixel_direction(coord, UVec2::new(x as u32, y as u32), size, border, model);
                    let sample = surface.sample_d(dir, tile_lod_m);
                    *pixel = TilePixel {
                        height_m: sample.height_m,
                        albedo_linear: sample.albedo_linear,
                        roughness: sample.roughness,
                        moisture: sample.moisture,
                        material_rgba: [0, 0, 0, 255],
                    };
                }
            });
    });

    populate_material_masks(&mut pixels, size, tile_lod_m);

    pixels
}

fn populate_material_masks(pixels: &mut [TilePixel], size: u32, tile_lod_m: f32) {
    let size = size as usize;
    if size == 0 {
        return;
    }
    // The stencil reads neighbouring texels, which are `tile_lod_m` metres
    // apart — the divisor must be that real spacing. An earlier version clamped
    // it to 250 m, which inflated slope/curvature by `tile_lod_m / 250` on
    // every tile coarser than 250 m/texel (up to ~80× at planet-scale LODs):
    // the rock mask saturated planet-wide grey, and the laplacian-driven
    // wetness mask tightened specular into a km-scale glint mottle. Coarse
    // tiles now measure genuinely coarse (smoother) slopes, which is the
    // consistent box-filtered limit of the fine view; the altitude-band model
    // in the shader carries the "mountains read rocky from orbit" look.
    let step_m = tile_lod_m.max(1.0);

    let heights: Vec<f32> = pixels.iter().map(|p| p.height_m).collect();
    // Slope/curvature stencil over the (read-only) height buffer. Rows are
    // independent — each writes only its own pixels' `material_rgba` and reads
    // shared `heights` — so spread them across the same bounded eval pool the
    // field sweep uses, rather than running this 262 k-pixel pass serially
    // after every parallel field bake. Kept on `tile_eval_pool` (not rayon's
    // implicit global pool) so it stays inside the synthesis isolation budget.
    tile_eval_pool().install(|| {
        pixels
            .par_chunks_mut(size)
            .enumerate()
            .for_each(|(y, row)| {
                let y_d = y.saturating_sub(1);
                let y_u = (y + 1).min(size - 1);
                for (x, pixel) in row.iter_mut().enumerate() {
                    let x_l = x.saturating_sub(1);
                    let x_r = (x + 1).min(size - 1);
                    let h_l = heights[y * size + x_l];
                    let h_r = heights[y * size + x_r];
                    let h_d = heights[y_d * size + x];
                    let h_u = heights[y_u * size + x];
                    pixel.material_rgba = material_masks_from_heights(
                        heights[y * size + x],
                        h_l,
                        h_r,
                        h_d,
                        h_u,
                        step_m,
                    );
                }
            });
    });
}

/// Slope/curvature/altitude → packed material masks (R = grass, G = soil,
/// B = rock, A = wetness). `pub(crate)` so the grass decoration layer
/// (`vegetation`) places blades with the exact gate the shader's grass
/// channel is baked from.
/// Fractal-scale slope compensation (INC-0004 follow-up). fBm terrain's
/// measured slope shrinks as the measurement baseline grows (RMS slope
/// ∝ `L^(H−1)`), so a coarse tile's texel-baseline slope under-reports the
/// fine-scale steepness the rock/soil thresholds below were tuned against —
/// mountainsides would read green from orbit. Boosting the measured slope by
/// `(step / REF)^exp` restores the *statistics* of the fine view (the coarse
/// mask approximates the area fraction of steep ground), unlike the old fixed
/// 250 m divisor which inflated every slope unconditionally. The ratio is
/// clamped ≥ 1 so tiles at/below the reference spacing — the near field the
/// thresholds were tuned on, and the grass-placement callers — are exact
/// no-ops. Only the ROCK response reads the compensated slope: soil (low
/// threshold, 0.035) and curvature (wetness/hollow) are fine-scale phenomena —
/// compensating them repainted the plains with a brown soil mottle / the
/// km-scale specular glint mottle respectively.
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
                .map(|p| {
                    // Alpha carries the macro landcover moisture ([-1, 1] →
                    // [0, 255]; linear even on the sRGB texture). The ground
                    // shader decodes it and forces its own output alpha to 1.
                    let mut texel = linear_rgb_to_srgba8(p.albedo_linear);
                    texel[3] = ((p.moisture.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8;
                    texel
                })
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

fn tile_lod_m(radius_m: f32, coord: TileCoordinate, texture_size: u32, border_size: u32) -> f32 {
    let inner_texels = texture_size.saturating_sub(border_size * 2).max(1);
    let face_radians = std::f32::consts::FRAC_PI_2 / TileCoordinate::count(coord.lod).max(1) as f32;
    (radius_m * face_radians / inner_texels as f32).max(1.0)
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
) -> DVec3 {
    // Differencing two heights along the same `Coordinate` lifts to a
    // body-local surface normal regardless of `model.translation`. Kept in f64:
    // the surface evaluator multiplies this by the body radius (~3.2e6 m on
    // Thalos), where an f32 direction would quantise the sample position to a
    // ~0.25 m body-local lattice and terrace the ground on foot.
    let pix =
        coord.stitched_pixel_coordinate(pixel, texture_size, border_size, model.is_spherical());
    let surface = pix.world_position(model, 0.0);
    let lifted = pix.world_position(model, 1.0);
    (lifted - surface).normalize()
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
