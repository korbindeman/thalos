//! [`SyntheticTileProvider`] — a deterministic procedural [`TileProvider`]
//! used to isolate the renderer from the synthesis pipeline. Two modes:
//!
//! - **`Analytic3d`** — multi-octave 3D fBm (value noise) sampled from the
//!   unit direction returned by [`TileCoordinate::stitched_pixel_coordinate`]
//!   → [`Coordinate::world_position`]. The octave count is LOD-adaptive: a
//!   tile at LOD `L` evaluates `min(L + base, MAX_OCTAVES)` octaves so the
//!   per-texel sampling stays below Nyquist and the cross-LOD area-average
//!   invariant holds. Plain fBm gives rolling terrain; a `signum * abs^p`
//!   sharpening lifts the peaks relative to the plains so the result reads
//!   as "mountains and plains" rather than a uniform-amplitude noise field.
//! - **`Flat`** — every height texel has the same value and the albedo is
//!   a neutral mid-grey. The scale-reference checkerboard for this mode
//!   lives in `body_terrain.wgsl` (analytic anti-aliased 3D checker
//!   evaluated per-fragment in body-fixed metres, driven by the
//!   `BodyTerrainDebug` uniform); painting it into the albedo texture
//!   aliased badly at every viewing distance.
//!
//! Both modes bypass generated cubemaps so renderer seams can be isolated
//! from terrain-generation seams. The cross-tile and cross-LOD invariants
//! are covered by the test suite at the bottom of the file.

use anyhow::Result;
use bevy::math::{DVec2, DVec3, UVec2};
use bevy::tasks::Task;
use rayon::prelude::*;
use thalos_terrain::noise::fbm3;
use thalos_udlod::math::{Coordinate, TileCoordinate};
use thalos_udlod::prelude::*;
use thalos_udlod::terrain_data::AttachmentData;

use crate::ground::tile_synthesis_pool::tile_synthesis_pool;

/// Number of sub-samples per pixel side for area-averaging. UDLOD's atlas
/// requires that a coarse-LOD pixel approximately equals the area-average of
/// the fine-LOD pixels it covers; pointwise noise evaluation at each pixel
/// center breaks that invariant whenever the noise has frequency content
/// above the pixel's Nyquist rate. 4×4 (16 samples) eliminates the visible
/// tile-edge seams at the LODs streamed near the player; coarser LODs may
/// still alias mildly under-sampled noise frequencies but the resulting
/// per-tile bias is small compared to the tile size.
const SUPERSAMPLE_FACTOR: u32 = 4;

/// Deterministic procedural [`TileProvider`] used for Stage 1 of M3.
///
/// Stores the body's height range so the encoded R16 values map back to the
/// same physical height range the renderer reads from the [`TerrainModel`]
/// (whose `min_height`/`max_height` fields are crate-private upstream).
pub struct SyntheticTileProvider {
    min_height: f32,
    max_height: f32,
    mode: SyntheticTerrainMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyntheticTerrainMode {
    Analytic3d,
    Flat,
}

impl SyntheticTileProvider {
    pub fn new(min_height: f32, max_height: f32) -> Self {
        Self::with_mode(min_height, max_height, SyntheticTerrainMode::Analytic3d)
    }

    pub fn flat() -> Self {
        Self::with_mode(0.0, 0.0, SyntheticTerrainMode::Flat)
    }

    pub fn with_mode(min_height: f32, max_height: f32, mode: SyntheticTerrainMode) -> Self {
        Self {
            min_height,
            max_height,
            mode,
        }
    }
}

impl TileProvider for SyntheticTileProvider {
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
        let mode = self.mode;

        tile_synthesis_pool().spawn(async move {
            let mut datas = Vec::with_capacity(attachments.len());
            for cfg in &attachments {
                let data = match cfg.format {
                    AttachmentFormat::R16 if cfg.name == "height" => match mode {
                        SyntheticTerrainMode::Analytic3d => {
                            synthesize_height_r16(&model, coord, cfg, min_height, max_height)
                        }
                        SyntheticTerrainMode::Flat => synthesize_constant_r16(cfg, 0.5),
                    },
                    AttachmentFormat::Rg16 if cfg.name == "height" => match mode {
                        SyntheticTerrainMode::Analytic3d => {
                            synthesize_height_rg16(&model, coord, cfg, min_height, max_height)
                        }
                        SyntheticTerrainMode::Flat => synthesize_constant_rg16_height(cfg, 0.5),
                    },
                    AttachmentFormat::R32Float if cfg.name == "height" => match mode {
                        SyntheticTerrainMode::Analytic3d => {
                            synthesize_height_r32_float(&model, coord, cfg, min_height, max_height)
                        }
                        SyntheticTerrainMode::Flat => synthesize_constant_r32_float(cfg, 0.5),
                    },
                    AttachmentFormat::R16 => synthesize_constant_r16(cfg, 0.74),
                    AttachmentFormat::R32Float => synthesize_constant_r32_float(cfg, 0.74),
                    AttachmentFormat::Rgba8 if cfg.name == "material" => {
                        // Mostly grass, a little soil, no rock/wetness. The
                        // BodyTerrainMaterial shader treats this attachment as
                        // masks, not color, so don't feed synthetic albedo into
                        // this slot.
                        synthesize_constant_rgba8(cfg, [220, 35, 0, 0])
                    }
                    AttachmentFormat::Rgba8 => match mode {
                        SyntheticTerrainMode::Analytic3d => {
                            synthesize_albedo(&model, coord, cfg, min_height, max_height)
                        }
                        // Constant neutral grey — the visible checkerboard
                        // for Flat mode is applied per-fragment in
                        // `body_terrain.wgsl` and reads from a debug
                        // uniform on `BodyTerrainMaterial`.
                        SyntheticTerrainMode::Flat => {
                            synthesize_constant_rgba8(cfg, [140, 140, 140, 255])
                        }
                    },
                    AttachmentFormat::Rg16 => synthesize_zero_rg16(cfg),
                    AttachmentFormat::Rgb8 => AttachmentData::None,
                };
                datas.push(data);
            }
            Ok(datas)
        })
    }
}

/// Invoke `f` at each of `SUPERSAMPLE_FACTOR^2` sub-sample directions within
/// the pixel's face-UV footprint. The pixel center comes from
/// `stitched_pixel_coordinate` so border pixels integrate over the neighbour's
/// footprint, preserving the same-LOD bit-identical-border invariant.
///
/// `world_position` returns surface normals via differencing rather than
/// normalising directly — that keeps the math valid when the model is
/// translated away from the origin (Stage 2 spawns terrain parented to body
/// grids that are not centered at world zero).
fn for_each_subsample(
    coord: TileCoordinate,
    pixel: UVec2,
    texture_size: u32,
    border_size: u32,
    model: &TerrainModel,
    mut f: impl FnMut(DVec3, DVec3),
) {
    let center =
        coord.stitched_pixel_coordinate(pixel, texture_size, border_size, model.is_spherical());
    let inner = (texture_size - 2 * border_size) as f64;
    let footprint = 1.0 / (inner * TileCoordinate::count(coord.lod) as f64);
    let supers = SUPERSAMPLE_FACTOR as f64;
    for sub_j in 0..SUPERSAMPLE_FACTOR {
        for sub_i in 0..SUPERSAMPLE_FACTOR {
            let sub_u = (sub_i as f64 + 0.5) / supers - 0.5;
            let sub_v = (sub_j as f64 + 0.5) / supers - 0.5;
            let sub_offset = DVec2::new(sub_u, sub_v) * footprint;
            let sub_coord = Coordinate::new(center.side, center.uv + sub_offset);
            let surface = sub_coord.world_position(model, 0.0);
            let lifted = sub_coord.world_position(model, 1.0);
            // Flat-mode bodies have `min_height == max_height`, so the
            // displacement vector is zero. Fall back to the radial direction,
            // which on a body-centered sphere matches the surface normal.
            let dir = (lifted - surface)
                .try_normalize()
                .unwrap_or_else(|| surface.normalize());
            f(dir, surface);
        }
    }
}

fn synthesize_height_r16(
    model: &TerrainModel,
    coord: TileCoordinate,
    cfg: &AttachmentConfig,
    min_height: f32,
    max_height: f32,
) -> AttachmentData {
    let values = synthesize_height_unit_values(model, coord, cfg, min_height, max_height);
    AttachmentData::R16(
        values
            .into_iter()
            .map(|t| (t * u16::MAX as f32 + 0.5) as u16)
            .collect(),
    )
}

fn synthesize_height_rg16(
    model: &TerrainModel,
    coord: TileCoordinate,
    cfg: &AttachmentConfig,
    min_height: f32,
    max_height: f32,
) -> AttachmentData {
    AttachmentData::Rg16(
        synthesize_height_unit_values(model, coord, cfg, min_height, max_height)
            .into_iter()
            .map(encode_unit_rg16)
            .collect(),
    )
}

fn synthesize_height_r32_float(
    model: &TerrainModel,
    coord: TileCoordinate,
    cfg: &AttachmentConfig,
    min_height: f32,
    max_height: f32,
) -> AttachmentData {
    AttachmentData::R32Float(synthesize_height_unit_values(
        model, coord, cfg, min_height, max_height,
    ))
}

fn synthesize_height_unit_values(
    model: &TerrainModel,
    coord: TileCoordinate,
    cfg: &AttachmentConfig,
    min_height: f32,
    max_height: f32,
) -> Vec<f32> {
    let size = cfg.texture_size;
    let border = cfg.border_size;
    let span = (max_height - min_height).max(1.0);
    let frequency = analytic_frequency(model);
    let octaves = octaves_for_lod(coord.lod);
    let sample_count = (SUPERSAMPLE_FACTOR * SUPERSAMPLE_FACTOR) as f32;

    let mut out: Vec<f32> = vec![0.0; (size * size) as usize];
    out.par_chunks_mut(size as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, texel) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for_each_subsample(
                    coord,
                    UVec2::new(x as u32, y as u32),
                    size,
                    border,
                    model,
                    |dir, _surface| {
                        sum += analytic_height(dir, min_height, max_height, frequency, octaves);
                    },
                );
                let h = sum / sample_count;
                *texel = ((h - min_height) / span).clamp(0.0, 1.0);
            }
        });
    out
}

fn synthesize_albedo(
    model: &TerrainModel,
    coord: TileCoordinate,
    cfg: &AttachmentConfig,
    min_height: f32,
    max_height: f32,
) -> AttachmentData {
    let size = cfg.texture_size;
    let border = cfg.border_size;
    let span = (max_height - min_height).max(1.0);
    let frequency = analytic_frequency(model);
    let octaves = octaves_for_lod(coord.lod);
    let sample_count = SUPERSAMPLE_FACTOR * SUPERSAMPLE_FACTOR;

    let mut out: Vec<[u8; 4]> = vec![[0, 0, 0, 255]; (size * size) as usize];
    out.par_chunks_mut(size as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, texel) in row.iter_mut().enumerate() {
                let mut r = 0u32;
                let mut g = 0u32;
                let mut b = 0u32;
                for_each_subsample(
                    coord,
                    UVec2::new(x as u32, y as u32),
                    size,
                    border,
                    model,
                    |dir, _surface| {
                        let h = analytic_height(dir, min_height, max_height, frequency, octaves);
                        let t = ((h - min_height) / span).clamp(0.0, 1.0);
                        let slope_tint =
                            (analytic_signal(dir, frequency, octaves) * 0.5 + 0.5).clamp(0.0, 1.0);
                        let c = terrain_color(t, slope_tint);
                        r += c[0] as u32;
                        g += c[1] as u32;
                        b += c[2] as u32;
                    },
                );
                *texel = [
                    (r / sample_count) as u8,
                    (g / sample_count) as u8,
                    (b / sample_count) as u8,
                    255,
                ];
            }
        });

    AttachmentData::Rgba8(out)
}

fn synthesize_constant_r16(cfg: &AttachmentConfig, value: f32) -> AttachmentData {
    let count = (cfg.texture_size * cfg.texture_size) as usize;
    let value = (value.clamp(0.0, 1.0) * u16::MAX as f32 + 0.5) as u16;
    AttachmentData::R16(vec![value; count])
}

fn synthesize_constant_rg16_height(cfg: &AttachmentConfig, value: f32) -> AttachmentData {
    let count = (cfg.texture_size * cfg.texture_size) as usize;
    AttachmentData::Rg16(vec![encode_unit_rg16(value); count])
}

fn synthesize_constant_r32_float(cfg: &AttachmentConfig, value: f32) -> AttachmentData {
    let count = (cfg.texture_size * cfg.texture_size) as usize;
    AttachmentData::R32Float(vec![value.clamp(0.0, 1.0); count])
}

fn encode_unit_rg16(value: f32) -> [u16; 2] {
    let unit = value.clamp(0.0, 1.0);
    let coarse = (unit * u16::MAX as f32).floor() / u16::MAX as f32;
    let residual = ((unit - coarse) * u16::MAX as f32).clamp(0.0, 1.0);
    [
        (coarse * u16::MAX as f32 + 0.5) as u16,
        (residual * u16::MAX as f32 + 0.5) as u16,
    ]
}

fn synthesize_zero_rg16(cfg: &AttachmentConfig) -> AttachmentData {
    let count = (cfg.texture_size * cfg.texture_size) as usize;
    AttachmentData::Rg16(vec![[0, 0]; count])
}

fn synthesize_constant_rgba8(cfg: &AttachmentConfig, value: [u8; 4]) -> AttachmentData {
    let count = (cfg.texture_size * cfg.texture_size) as usize;
    AttachmentData::Rgba8(vec![value; count])
}

/// Surface-metres width of octave 0's wavelength. ~80 km gives the
/// continent-scale silhouette at coarse LODs; the cascade extends ~11 more
/// octaves below that, reaching ~40 m at the deepest LOD so a player on the
/// surface still sees varying terrain within a single fine-LOD tile (a tile
/// at LOD 15 on Thalos is ~150 m wide).
const ANALYTIC_BASE_WL_M: f32 = 80_000.0;

/// fBm seed; decoupled from any pipeline seed so changing the synthetic
/// provider doesn't reshuffle bodies that share the constant.
const ANALYTIC_SEED: u32 = 0xA17D_EE0D;

/// Minimum octave count, used at LOD 0 where the per-pixel footprint is
/// kilometres wide — adding fine octaves at this resolution aliases even
/// with 4×4 supersampling.
const ANALYTIC_OCTAVE_BASE: u32 = 2;
/// Maximum octave count, reached at fine LODs. With base wavelength 80 km
/// and lacunarity 2, octave 11 lands at ~40 m wavelength — fine enough that
/// a player at metre scale sees several wavelengths per tile.
const ANALYTIC_OCTAVE_MAX: u32 = 12;
/// fBm persistence. Per-octave amplitude decay. The classic `0.5` value
/// concentrates almost all energy in the lowest octaves, which leaves fine
/// LODs visually flat (per-tile spread shrinks geometrically with LOD).
/// `0.75` keeps enough amplitude in high octaves that fine-LOD tiles still
/// have several-tens-of-metres of relief from the highest octaves alone.
const ANALYTIC_PERSISTENCE: f32 = 0.75;
const ANALYTIC_LACUNARITY: f32 = 2.0;

/// Sharpening exponent applied to the fBm output: `sign(s) * |s|^p` with
/// `p > 1` flattens the low end (plains stay smooth) and lifts the high end
/// (peaks read as mountains). `1.4` was eyeballed against the supersample
/// budget — too sharp aliases visibly at coarse LODs.
const ANALYTIC_SHARPEN: f32 = 1.4;

/// Base frequency in unit-direction space such that one octave-0 wavelength
/// equals `ANALYTIC_BASE_WL_M` metres on the body's surface. Body-relative so
/// feature scale stays constant across bodies.
fn analytic_frequency(model: &TerrainModel) -> f32 {
    ((model.scale() as f32) / ANALYTIC_BASE_WL_M).max(1.0)
}

/// How many fBm octaves a tile at this LOD evaluates.
///
/// The cross-LOD area-average invariant the renderer relies on (a coarse
/// tile's pixel ≈ the area-average of its fine-LOD children) requires that
/// no octave have wavelength shorter than the per-pixel footprint. Each LOD
/// step doubles the per-pixel resolution, so we add one octave per LOD step.
/// Capping at `ANALYTIC_OCTAVE_MAX` keeps the deepest tiles' cost finite
/// without losing visible detail past the highest LOD a player typically
/// reaches.
fn octaves_for_lod(lod: u32) -> u32 {
    ANALYTIC_OCTAVE_BASE
        .saturating_add(lod)
        .min(ANALYTIC_OCTAVE_MAX)
}

fn analytic_height(
    dir: DVec3,
    min_height: f32,
    max_height: f32,
    frequency: f32,
    octaves: u32,
) -> f32 {
    let center = 0.5 * (min_height + max_height);
    // Leave ~10% headroom so the sharpened signal never clamps at the
    // R16 ends. The fBm itself is bounded to roughly [-1, 1]; after
    // sharpening (sign-preserving power < 2) it stays in the same range.
    let amplitude = 0.45 * (max_height - min_height).max(1.0);
    center + analytic_signal(dir, frequency, octaves) * amplitude
}

/// Multi-octave 3D fBm sampled from a unit direction, with a sign-preserving
/// power curve applied to push contrast toward mountains-vs-plains. Returns
/// roughly `[-1, 1]`.
///
/// Evaluated in f32 because the underlying `fbm3` is f32, but the input is
/// pre-scaled in f64 (the multiplication by `frequency` happens before the
/// downcast) to avoid losing precision at planet scale. `frequency` is
/// typically O(10²–10³) on Thalos, so `dir * frequency` in f64 is exact
/// where the equivalent f32 product would already be losing bits.
fn analytic_signal(dir: DVec3, frequency: f32, octaves: u32) -> f32 {
    let p = (dir * frequency as f64).as_vec3();
    let raw = fbm3(
        p.x,
        p.y,
        p.z,
        ANALYTIC_SEED,
        octaves,
        ANALYTIC_PERSISTENCE,
        ANALYTIC_LACUNARITY,
    );
    raw.signum() * raw.abs().powf(ANALYTIC_SHARPEN)
}

fn terrain_color(height_t: f32, tint: f32) -> [u8; 4] {
    let low = [86.0, 72.0, 58.0];
    let mid = [118.0, 112.0, 90.0];
    let high = [170.0, 166.0, 145.0];
    let ridge = [192.0, 188.0, 164.0];

    let base = if height_t < 0.55 {
        mix_rgb(low, mid, smoothstep(0.12, 0.55, height_t))
    } else {
        mix_rgb(mid, high, smoothstep(0.55, 0.95, height_t))
    };
    let rgb = mix_rgb(base, ridge, (tint * height_t).powf(2.0) * 0.22);
    [
        rgb[0].clamp(0.0, 255.0) as u8,
        rgb[1].clamp(0.0, 255.0) as u8,
        rgb[2].clamp(0.0, 255.0) as u8,
        255,
    ]
}

fn mix_rgb(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r16_from(data: &AttachmentData) -> &[u16] {
        match data {
            AttachmentData::R16(v) => v,
            _ => panic!("expected R16 attachment data"),
        }
    }

    fn height_cfg(size: u32, border: u32) -> AttachmentConfig {
        AttachmentConfig {
            name: "height".to_string(),
            texture_size: size,
            border_size: border,
            mip_level_count: 1,
            format: AttachmentFormat::R16,
        }
    }

    /// Synthesize one tile's height attachment via the same code path the
    /// async `request_tile` task uses.
    fn synth(
        coord: TileCoordinate,
        model: &TerrainModel,
        size: u32,
        border: u32,
        min_h: f32,
        max_h: f32,
    ) -> Vec<u16> {
        let cfg = height_cfg(size, border);
        let data = synthesize_height_r16(model, coord, &cfg, min_h, max_h);
        r16_from(&data).to_vec()
    }

    fn index(x: u32, y: u32, size: u32) -> usize {
        (y * size + x) as usize
    }

    #[test]
    fn synthetic_border_matches_same_face_right_neighbour_interior() {
        // 1 Mm sphere with mountain-scale relief, mid-LOD interior tile.
        let model = TerrainModel::sphere(DVec3::ZERO, 1_000_000.0, -1000.0, 1000.0);
        let size = 16u32;
        let border = 2u32;
        let min_h = model.min_height();
        let max_h = model.max_height();

        let a = TileCoordinate::new(0, 3, 3, 4);
        let b = TileCoordinate::new(0, 3, 4, 4); // right neighbour
        let a_pixels = synth(a, &model, size, border, min_h, max_h);
        let b_pixels = synth(b, &model, size, border, min_h, max_h);

        // For every row, A's right border column at x = size-1 was
        // stitched to read source at the world position of B's interior
        // pixel at x = border+1 (the first interior texel just inside B's
        // left border). They must encode to bit-identical R16.
        for y in 0..size {
            let a_border = a_pixels[index(size - 1, y, size)];
            // border_neighbour_pixel_offset(right, size, border) = (-center, 0)
            // (size-1) + -(size - 2*border) = 2*border - 1
            let mirror_x = 2 * border - 1;
            let b_interior = b_pixels[index(mirror_x, y, size)];
            assert_eq!(
                a_border, b_interior,
                "y={y}: A.right_border={a_border} != B.interior(x={mirror_x})={b_interior}"
            );
        }
    }

    /// For every border pixel of `a`, find the interior pixel on neighbour
    /// `b` whose `pixel_coordinate` is closest (within ULPs) to A's
    /// `stitched_pixel_coordinate`, then assert their synthesised R16
    /// values match. Covers same-face and cross-face cardinals uniformly.
    fn assert_borders_match_neighbour(
        a: TileCoordinate,
        b: TileCoordinate,
        a_border_pixels: impl Iterator<Item = UVec2>,
        model: &TerrainModel,
        size: u32,
        border: u32,
        min_h: f32,
        max_h: f32,
    ) {
        let a_pixels = synth(a, model, size, border, min_h, max_h);
        let b_pixels = synth(b, model, size, border, min_h, max_h);

        for pixel in a_border_pixels {
            let stitched = a.stitched_pixel_coordinate(pixel, size, border, model.is_spherical());
            assert_eq!(stitched.side, b.side);

            let mut best = (f64::INFINITY, UVec2::ZERO);
            for by in border..(size - border) {
                for bx in border..(size - border) {
                    let bc = b.pixel_coordinate(UVec2::new(bx, by), size, border);
                    let d = (bc.uv - stitched.uv).length_squared();
                    if d < best.0 {
                        best = (d, UVec2::new(bx, by));
                    }
                }
            }
            assert!(best.0 < 1.0e-12, "no close interior in B for {pixel:?}");

            let a_val = a_pixels[index(pixel.x, pixel.y, size)];
            let b_val = b_pixels[index(best.1.x, best.1.y, size)];
            assert_eq!(
                a_val, b_val,
                "border {pixel:?} on tile {a:?} = {a_val} != B interior {:?} = {b_val}",
                best.1
            );
        }
    }

    #[test]
    fn synthetic_cross_face_right_border_matches_neighbour_interior() {
        let model = TerrainModel::sphere(DVec3::ZERO, 1_000_000.0, -1000.0, 1000.0);
        let size = 16u32;
        let border = 2u32;
        let min_h = model.min_height();
        let max_h = model.max_height();
        let lod = 3u32;
        let count = TileCoordinate::count(lod);

        let a = TileCoordinate::new(0, lod, count - 1, 4);
        let b = a.neighbours(true).nth(1).unwrap();
        assert_ne!(b, TileCoordinate::INVALID);
        assert_ne!(a.side, b.side, "expected cross-face right neighbour");

        assert_borders_match_neighbour(
            a,
            b,
            (border..(size - border)).map(|y| UVec2::new(size - 1, y)),
            &model,
            size,
            border,
            min_h,
            max_h,
        );
    }

    #[test]
    fn synthetic_border_matches_same_face_bottom_neighbour_interior() {
        let model = TerrainModel::sphere(DVec3::ZERO, 1_000_000.0, -1000.0, 1000.0);
        let size = 16u32;
        let border = 2u32;
        let min_h = model.min_height();
        let max_h = model.max_height();

        let a = TileCoordinate::new(2, 3, 4, 4);
        let b = TileCoordinate::new(2, 3, 4, 5); // bottom neighbour
        let a_pixels = synth(a, &model, size, border, min_h, max_h);
        let b_pixels = synth(b, &model, size, border, min_h, max_h);

        for x in 0..size {
            let a_border = a_pixels[index(x, size - 1, size)];
            let mirror_y = 2 * border - 1;
            let b_interior = b_pixels[index(x, mirror_y, size)];
            assert_eq!(
                a_border, b_interior,
                "x={x}: A.bottom_border={a_border} != B.interior(y={mirror_y})={b_interior}"
            );
        }
    }

    /// UDLOD's atlas requires the cross-LOD invariant: a coarse-LOD pixel
    /// should approximately equal the area-average of the fine-LOD pixels
    /// covering the same world-space footprint. Pointwise noise evaluation at
    /// each pixel center breaks this whenever the noise has frequency content
    /// above the pixel's Nyquist rate, producing visible elevation seams at
    /// every atlas tile boundary where adjacent atlas tiles happen to be
    /// loaded at different LODs.
    ///
    /// This test confirms that supersampled synthesis satisfies the invariant
    /// within a tolerance comparable to a single contour band. Without
    /// `SUPERSAMPLE_FACTOR > 1` the assertion fails by tens to hundreds of
    /// metres of elevation; with 4× sub-sampling at the LODs streamed near
    /// the player the residual is sub-band.
    /// At fine LODs the per-tile spread must be visible from a player on
    /// the surface — if octave count + persistence are wrong the cascade
    /// degenerates and a LOD-15 tile (~150 m wide on Thalos) ends up
    /// effectively flat (the fine octaves don't reach pixel-scale
    /// wavelengths, the coarse octaves vary too slowly to show within one
    /// tile). Threshold here is a soft lower bound on "you can see
    /// terrain" — a 50 m spread inside one ~150 m tile reads as visible
    /// relief.
    #[test]
    fn fine_lod_tile_has_visible_relief() {
        let model = TerrainModel::sphere(DVec3::ZERO, 3_186_000.0, -2500.0, 2500.0);
        let size = 64u32;
        let border = 2u32;
        let min_h = model.min_height();
        let max_h = model.max_height();
        let span = (max_h - min_h) as f32;

        let lod = 15u32;
        let n = TileCoordinate::count(lod);
        let pixels = synth(
            TileCoordinate::new(0, lod, n / 2, n / 2),
            &model,
            size,
            border,
            min_h,
            max_h,
        );
        let (lo, hi) = pixels
            .iter()
            .fold((u16::MAX, 0u16), |(lo, hi), &p| (lo.min(p), hi.max(p)));
        let spread_m = (hi - lo) as f32 / u16::MAX as f32 * span;
        assert!(
            spread_m >= 50.0,
            "LOD-15 tile spread {spread_m:.1} m is too flat — fBm cascade isn't reaching pixel-scale wavelengths"
        );
    }

    #[test]
    fn synthetic_cross_lod_average_within_contour_band() {
        let model = TerrainModel::sphere(DVec3::ZERO, 1_000_000.0, -1000.0, 1000.0);
        let size = 16u32;
        let border = 2u32;
        let min_h = model.min_height();
        let max_h = model.max_height();
        let inner = (size - 2 * border) as i32;
        let half = inner / 2;

        let parent_lod = 6u32;
        let parent = TileCoordinate::new(0, parent_lod, 30, 30);
        let parent_pixels = synth(parent, &model, size, border, min_h, max_h);

        // 25 m contour band → ~820 u16 in a 2000 m total range. Use 1500 as a
        // soft ceiling so coarse-LOD residual aliasing has some room.
        let tolerance: i32 = 1500;

        // One pixel per child quadrant.
        for &(px_inner, py_inner) in &[(3i32, 3i32), (9, 3), (3, 9), (9, 9)] {
            let px = (px_inner + border as i32) as u32;
            let py = (py_inner + border as i32) as u32;
            let parent_val = parent_pixels[index(px, py, size)] as i32;

            let child_dx = (px_inner / half) as u32;
            let child_dy = (py_inner / half) as u32;
            let child = TileCoordinate::new(
                parent.side,
                parent_lod + 1,
                parent.x * 2 + child_dx,
                parent.y * 2 + child_dy,
            );
            let child_pixels = synth(child, &model, size, border, min_h, max_h);

            let cx0 = (px_inner - child_dx as i32 * half) * 2 + border as i32;
            let cy0 = (py_inner - child_dy as i32 * half) * 2 + border as i32;

            let mut child_sum: u32 = 0;
            for dy in 0..2 {
                for dx in 0..2 {
                    let cx = (cx0 + dx) as u32;
                    let cy = (cy0 + dy) as u32;
                    child_sum += child_pixels[index(cx, cy, size)] as u32;
                }
            }
            let child_avg = (child_sum / 4) as i32;
            let diff = (parent_val - child_avg).abs();

            assert!(
                diff < tolerance,
                "parent ({px}, {py}) lod {parent_lod} = {parent_val} vs child-quad avg = {child_avg}: diff {diff} > tolerance {tolerance}"
            );
        }
    }
}
