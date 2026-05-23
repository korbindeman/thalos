//! [`PipelineTileProvider`] — bridges `thalos_udlod`'s [`TileProvider`] seam
//! to the Thalos synthesis pipeline.
//!
//! Per-pixel evaluation runs in two stages:
//!
//! 1. **Cubemap base + dynamic overlays** — sample baked
//!    height/albedo/roughness from [`StaticSurfaceData`] for the pixel's
//!    body-local direction, with a temporary macro low-pass to hide the
//!    pre-rewrite Thalos terracing, then apply terrain-owned time-varying
//!    layers (seasonal ice, aeolian bedforms).
//! 2. **Procedural detail (LOD-adaptive)** — additively layer
//!    Musgrave ridged hybrid multifractal noise on top of the cubemap base.
//!    The cascade is sphere-continuous (3D position input, no cube-face
//!    seam), runs in metres against a fixed base wavelength, and uses a
//!    continuous octave count so adjacent tiles at neighbouring LODs blend
//!    smoothly. Two ingredients:
//!    - `thalos_terrain::noise::hmf_ridged_3d` — Musgrave hybrid
//!      multifractal whose self-modulating weight produces "rough peaks,
//!      smooth valleys" without an external biome mask. Ridged shape
//!      concentrates signal at noise zero-crossings → ridge crests.
//!    - `thalos_terrain::noise::fbm3_vec3` — vector-valued fBm warps the
//!      input position before HMF sampling, breaking the lattice-aligned
//!      character of plain ridged noise.
//!
//!    The cascade is positive-only: HMF returns `[0, 1]`, scaled by
//!    [`DETAIL_AMP_M`] and added on top of the macro height. The macro
//!    therefore acts as the sediment / tectonic floor, with HMF orogeny
//!    accumulating in rough regions.
//!
//! Each tile request computes the full per-pixel evaluation once into a
//! `Vec<TilePixel>` (parallelised across rows with `rayon`) and then encodes
//! each requested attachment from that shared buffer.
//!
//! # Known limitation: CPU/GPU bilinear stand-off
//!
//! Tile R16 data is bilinearly sampled by the GPU; [`rendered_height_m`]
//! evaluates the cascade pointwise at the requested `dir`. Off pixel
//! centres the two values disagree by up to one peak-to-trough of the
//! resolved detail amplitude. With HMF's top-octave amplitude of order
//! 10 cm at sub-metre wavelength, the EVA controller can float by O(10–20 cm)
//! relative to the rendered surface. Acceptable for v1; fixing it means
//! threading [`TerrainModel`] into the height query and bilinear-mixing
//! four texel-centre evaluations using UDLOD's stretched-cube projection.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;
use bevy::math::{DVec3, UVec2, Vec3};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use rayon::prelude::*;
use thalos_terrain::cubemap::Cubemap;
use thalos_terrain::noise::{fbm3_vec3, hmf_ridged_3d};
use thalos_terrain::{
    DynamicSurfaceState, PlanetSurface, StaticSurfaceData, apply_dynamic_surface_layers,
    cubemap::dir_to_face_uv,
};
use thalos_udlod::math::TileCoordinate;
use thalos_udlod::prelude::*;
use thalos_udlod::terrain_data::AttachmentData;

// ---------------------------------------------------------------------------
// Procedural-detail tuning constants
// ---------------------------------------------------------------------------

/// Hash seed for the high-frequency detail noise. Decoupled from the body
/// generator's seed so changing terrain gen doesn't reshuffle ground-LOD
/// detail and vice versa.
const DETAIL_NOISE_SEED: u32 = 0x1E_E0_57_07;

/// Base (octave 0) wavelength of the HMF cascade, in metres. Subsequent
/// octaves halve at `lacunarity=2`; with the 11-octave cap below, the
/// cascade bottoms out at `BASE_WL / 2^11 ≈ 0.49 m`.
const DETAIL_BASE_WL_M: f32 = 1000.0;

/// Peak amplitude of the HMF height contribution, in metres. HMF output is
/// normalised to `[0, 1]`, so the additive contribution to macro height
/// stays in `[0, DETAIL_AMP_M]`. Practical values are concentrated well
/// below the maximum because weight collapse damps flat regions.
const DETAIL_AMP_M: f32 = 250.0;

const DETAIL_PERSISTENCE: f32 = 0.5;
const DETAIL_LACUNARITY: f32 = 2.0;

/// Musgrave ridged-multifractal offset. `1.0` keeps the signal
/// `offset - |noise|` inside `[0, 1]`; values above one allow the weight
/// to ramp upward across octaves but make the closed-form normalisation
/// less tight.
const DETAIL_OFFSET: f32 = 1.0;

/// Cascade depth at the finest LOD. Eleven octaves from a 1 km base
/// bottoms out at `1 km / 2^11 ≈ 0.49 m` — sub-metre wavelength with
/// `~250 m × 0.5^11 ≈ 12 cm` amplitude at the deepest octave.
const MAX_DETAIL_OCTAVES: f32 = 11.0;

/// Hash seed for the domain-warp vector field. Independent from the HMF
/// seed so the two noise calls don't share lattice alignment.
const WARP_NOISE_SEED: u32 = 0x77_C0_DE_42;

/// Wavelength of the warp field's octave 0, in metres. Warp samples are
/// taken at `pos / WARP_WAVELENGTH_M`, then the result is scaled by
/// [`WARP_AMP_M`] and added back to `pos` before the HMF call.
const WARP_WAVELENGTH_M: f32 = 4000.0;

/// Maximum positional displacement of the domain warp, in metres. Roughly
/// one fifth of [`WARP_WAVELENGTH_M`] keeps the warp visible without
/// folding the field back over itself.
const WARP_AMP_M: f32 = 800.0;

const WARP_OCTAVES: u32 = 2;

/// Additional height-range margin reserved for procedural detail.
/// Matches [`DETAIL_AMP_M`] so the R16 quantisation has room for the
/// full positive HMF contribution above the static + dynamic envelope.
const DETAIL_HEIGHT_MARGIN_M: f32 = DETAIL_AMP_M;

/// Temporary macro low-pass over the baked height/color cubemaps.
///
/// Thalos's pre-rewrite terrestrial field contains visible contour-like
/// bands. Blur the macro source before UDLOD tiles and CPU height queries see
/// it so the surface reads like normal smooth game terrain while the terrain
/// generator is being replaced.
const MACRO_HEIGHT_SMOOTH_RADIUS_TEXELS: f32 = 4.0;
const MACRO_MATERIAL_SMOOTH_RADIUS_TEXELS: f32 = 3.0;

/// Per-tile detail plan: a continuous octave count. Fractional values
/// blend the top octave in smoothly so a tile cascading from N → N+1
/// across an LOD boundary does not pop. `0.0` disables detail.
#[derive(Clone, Copy, Debug)]
struct DetailPlan {
    octaves: f32,
}

/// Log the first ~10k tile evaluations so we can confirm the LOD plan in
/// the absence of a real-time profiler. Atomic counter; no allocation, no
/// lock contention beyond the increment.
fn should_log_tile() -> bool {
    static COUNT: AtomicU32 = AtomicU32::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    n < 10_000
}

/// Choose the cascade depth for a tile at `tile_lod_m`.
///
/// An octave with wavelength `W` is barely Nyquist-resolvable when
/// `tile_lod_m = W / 2`. For lac=2 and base wavelength `base_wl_m`,
/// octave `k` (0-indexed) has wavelength `base_wl_m / 2^k`. The
/// continuous resolvable-octave count is `log2(base_wl_m / (2 *
/// tile_lod_m)) + 1`, returned without flooring so HMF's fractional
/// top-octave weighting can fade the cascade in smoothly.
fn detail_plan_for_lod(tile_lod_m: f32, base_wl_m: f32) -> DetailPlan {
    if tile_lod_m <= 0.0 {
        return DetailPlan {
            octaves: MAX_DETAIL_OCTAVES,
        };
    }
    let ratio = base_wl_m / (2.0 * tile_lod_m);
    if ratio <= 1.0 {
        return DetailPlan { octaves: 0.0 };
    }
    let octaves = (ratio.log2() + 1.0).clamp(0.0, MAX_DETAIL_OCTAVES);
    DetailPlan { octaves }
}

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
/// procedural detail headroom. Used by `ground_terrain` to set up the
/// thalos_udlod `TerrainModel` height bounds.
pub fn rendered_height_range(surface: &PlanetSurface, state: &DynamicSurfaceState) -> f32 {
    (surface.static_surface.height_range
        + dynamic_height_margin(surface, state)
        + DETAIL_HEIGHT_MARGIN_M)
        .max(1.0)
}

/// Returns the `tile_lod_m` the renderer currently uses at `world_position`
/// (in the terrain model's local frame, i.e. the body grid frame for
/// spherical terrains). Mirrors the GPU's tile-residency lookup: derives
/// the value from the deepest atlas tile actually resident at this
/// direction, so CPU height queries land on the same surface the GPU
/// draws.
///
/// `None` is returned only when no ancestor tile is resident yet (early
/// frames before any bake has completed). Callers should fall back to a
/// detail-free query in that case — passing a fine `tile_lod_m` would
/// engage the full procedural cascade and produce a CPU/GPU height gap.
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
            // require separate passes; in practice every attachment in a
            // body config uses the same `texture_size`, so just assert that.
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

            let pixels = compute_tile_pixels(
                &surface,
                &dynamic_state,
                height_range_m,
                coord,
                &model,
                size,
                border,
            );

            let mut datas = Vec::with_capacity(attachments.len());
            for cfg in &attachments {
                datas.push(encode_attachment(cfg, &pixels, height_range_m, &body_name));
            }
            Ok(datas)
        })
    }
}

// ---------------------------------------------------------------------------
// Per-pixel evaluation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct TilePixel {
    height_m: f32,
    albedo_linear: Vec3,
    roughness: f32,
}

fn compute_tile_pixels(
    surface: &PlanetSurface,
    dynamic_state: &DynamicSurfaceState,
    _height_range_m: f32,
    coord: TileCoordinate,
    model: &TerrainModel,
    size: u32,
    border: u32,
) -> Vec<TilePixel> {
    let body = &surface.static_surface;
    let tile_lod_m = tile_lod_m(body, coord, size, border);
    let dynamic_lod = tile_lod_m.log2();

    // LOD-adaptive cascade depth. Pick once per tile (rather than per
    // pixel) so the result is coherent across the tile and the hot inner
    // loop stays branch-free.
    let plan = detail_plan_for_lod(tile_lod_m, DETAIL_BASE_WL_M);

    // Diagnostic: log the first few tiles so we can confirm the LOD plan is
    // engaging. Throttled by a once-per-tile-coord atomic flag.
    if should_log_tile() {
        info!(
            "PipelineTileProvider tile lod={} ({}, {}, {}) tile_lod_m={:.1} \
             octaves={:.2}",
            coord.lod, coord.side, coord.x, coord.y, tile_lod_m, plan.octaves,
        );
    }

    let count = (size * size) as usize;
    let mut pixels = vec![TilePixel::default(); count];

    pixels
        .par_chunks_mut(size as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, pixel) in row.iter_mut().enumerate() {
                let dir =
                    pixel_direction(coord, UVec2::new(x as u32, y as u32), size, border, model);
                *pixel = evaluate_pixel(PixelContext {
                    surface,
                    dynamic_state,
                    dir,
                    dynamic_lod,
                    plan,
                });
            }
        });

    pixels
}

struct PixelContext<'a> {
    surface: &'a PlanetSurface,
    dynamic_state: &'a DynamicSurfaceState,
    dir: Vec3,
    dynamic_lod: f32,
    plan: DetailPlan,
}

/// Stages 1+2 result: smoothed cubemap base sample with dynamic-layer overlays
/// applied. Shared between [`evaluate_pixel`] and the public
/// [`rendered_height_m`] so the renderer and physics agree on the same
/// pre-detail surface.
#[derive(Debug, Clone, Copy)]
struct BaseSample {
    height_m: f32,
    albedo_linear: Vec3,
    roughness: f32,
}

fn sample_base_with_dynamic(
    surface: &PlanetSurface,
    dynamic_state: &DynamicSurfaceState,
    dir: Vec3,
    dynamic_lod: f32,
) -> BaseSample {
    let body = &surface.static_surface;
    let mut height = sample_height_smoothed(&body.height_cubemap, dir, body.height_range);
    let mut albedo = sample_albedo_smoothed(&body.albedo_cubemap, dir);
    let mut roughness = static_roughness_smoothed(body, dir);
    if !surface.dynamic_layers.is_empty() {
        apply_dynamic_surface_layers(
            surface,
            dynamic_state,
            dir,
            dynamic_lod,
            &mut height,
            &mut albedo,
            &mut roughness,
        );
    }
    BaseSample {
        height_m: height,
        albedo_linear: albedo,
        roughness,
    }
}

/// Domain-warped ridged hybrid multifractal in metres. Evaluated in
/// body-local 3D so the field is sphere-continuous (the same physical
/// point on the planet returns the same value regardless of which cube
/// face is generating it). Returns `0.0` when the LOD plan disables
/// detail.
fn compute_detail_height(dir: Vec3, radius_m: f32, plan: DetailPlan) -> f32 {
    if plan.octaves <= 0.0 {
        return 0.0;
    }

    let p_3d_m = dir * radius_m;

    // Domain warp: low-frequency vector field offsets the position before
    // HMF sampling. Breaks the lattice-aligned look of plain ridged noise.
    let warp_sample = fbm3_vec3(
        p_3d_m / WARP_WAVELENGTH_M,
        WARP_NOISE_SEED,
        WARP_OCTAVES,
        DETAIL_PERSISTENCE,
        DETAIL_LACUNARITY,
    );
    let warped_m = p_3d_m + warp_sample * WARP_AMP_M;

    let hmf = hmf_ridged_3d(
        warped_m / DETAIL_BASE_WL_M,
        DETAIL_NOISE_SEED,
        plan.octaves,
        DETAIL_PERSISTENCE,
        DETAIL_LACUNARITY,
        DETAIL_OFFSET,
    );

    hmf * DETAIL_AMP_M
}

/// Minimum below-sea-level margin (m) preserved after capping detail uplift.
/// Keeps the bathymetry mesh strictly below the water icosphere
/// (`WATER_SURFACE_EPSILON_M = 2 m` above sea level) so the impostor's
/// macro-height water mask and the ground-LOD water sphere agree on where
/// land ends.
const SEA_LEVEL_CAP_EPSILON_M: f32 = 0.5;

/// Combine macro height and HMF detail uplift, capping in shallow bathymetry.
///
/// HMF detail is positive-only (`[0, DETAIL_AMP_M]`), so on continental
/// shelves where macro height sits within `DETAIL_AMP_M` of sea level the
/// raw `macro + detail` could crest the water surface and expose
/// land-tinted bake albedo where the impostor's macro-height water mask
/// shows ocean. Capping the uplift to `(sea_level - macro - ε)` preserves
/// underwater detail up to the available depth budget while guaranteeing
/// the mesh never breaches the water sphere. Above sea level the cap is
/// inactive.
fn combine_base_and_detail(base_height_m: f32, detail_h: f32, sea_level_m: Option<f32>) -> f32 {
    let Some(sea) = sea_level_m else {
        return base_height_m + detail_h;
    };
    if base_height_m >= sea {
        return base_height_m + detail_h;
    }
    let max_uplift = (sea - base_height_m - SEA_LEVEL_CAP_EPSILON_M).max(0.0);
    base_height_m + detail_h.min(max_uplift)
}

/// Canonical "what UDLOD renders at this direction" height query.
///
/// Returns the height (metres above the reference sphere) that the ground
/// LOD pipeline produces for a tile evaluated at `tile_lod_m` metres per
/// texel. This is the single source of truth shared between the atlas baker
/// (`PipelineTileProvider`) and every system that needs to agree with the
/// rendered ground — terrain colliders, character controllers, camera
/// boom ray-casts, HUD altitude readouts.
///
/// Pass a small `tile_lod_m` (e.g. `0.5`) for full procedural detail near
/// the camera; pass the patch's vertex spacing when building a coarser
/// collider mesh so the mesh resolution matches the represented detail.
///
/// Stages, in order:
/// 1. Cubemap base, low-pass sampled.
/// 2. Dynamic layers (ice caps, aeolian bedforms).
/// 3. LOD-adaptive procedural detail: domain-warped ridged HMF.
///
/// Evaluation is pointwise at `dir`; the GPU's bilinear of the encoded
/// R16 atlas differs by up to one peak-to-trough of the resolved detail
/// (O(10–20 cm) at sub-metre wavelength). See the module-level note on
/// the bilinear stand-off.
pub fn rendered_height_m(
    surface: &PlanetSurface,
    dynamic_state: &DynamicSurfaceState,
    dir: Vec3,
    tile_lod_m: f32,
) -> f32 {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return 0.0;
    }
    let dynamic_lod = tile_lod_m.max(1e-6).log2();
    let base = sample_base_with_dynamic(surface, dynamic_state, dir, dynamic_lod);
    let plan = detail_plan_for_lod(tile_lod_m, DETAIL_BASE_WL_M);
    let detail_h = compute_detail_height(dir, surface.static_surface.radius_m, plan);
    combine_base_and_detail(base.height_m, detail_h, surface.static_surface.sea_level_m)
}

fn evaluate_pixel(ctx: PixelContext<'_>) -> TilePixel {
    let base = sample_base_with_dynamic(ctx.surface, ctx.dynamic_state, ctx.dir, ctx.dynamic_lod);
    let detail_h = compute_detail_height(ctx.dir, ctx.surface.static_surface.radius_m, ctx.plan);
    let height_m = combine_base_and_detail(
        base.height_m,
        detail_h,
        ctx.surface.static_surface.sea_level_m,
    );
    TilePixel {
        height_m,
        albedo_linear: base.albedo_linear,
        roughness: base.roughness,
    }
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
// Sampling helpers
// ---------------------------------------------------------------------------

fn dynamic_height_margin(surface: &PlanetSurface, state: &DynamicSurfaceState) -> f32 {
    let mut margin = 0.0;

    for (index, layer) in surface.dynamic_layers.ice_caps.iter().enumerate() {
        let coverage_scale = state
            .ice_cap_state(index, layer)
            .map(|s| s.coverage_scale)
            .unwrap_or(1.0);
        let thickness_scale = state
            .ice_cap_state(index, layer)
            .map(|s| s.thickness_scale)
            .unwrap_or(1.0);
        if coverage_scale > 0.0 {
            margin += layer.spec.max_thickness_m.max(0.0) * thickness_scale.max(0.0);
        }
    }

    for (index, layer) in surface.dynamic_layers.active_dunes.iter().enumerate() {
        let Some(dune_state) = state.active_dune_state(index, layer) else {
            let region = &layer.region;
            margin += region.amplitude_draa_m.max(0.0) + region.amplitude_dune_m.max(0.0);
            continue;
        };
        if dune_state.coverage_scale > 0.0 {
            let region = &layer.region;
            margin += dune_state.amplitude_scale.max(0.0)
                * (region.amplitude_draa_m.max(0.0) + region.amplitude_dune_m.max(0.0));
        }
    }

    margin
}

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

fn sample_height_smoothed(cube: &Cubemap<u16>, dir: Vec3, height_range: f32) -> f32 {
    smooth_cubemap_scalar(
        cube.resolution(),
        dir,
        MACRO_HEIGHT_SMOOTH_RADIUS_TEXELS,
        |face, x, y| decode_height(cube.get(face, x, y), height_range),
    )
}

fn sample_albedo_smoothed(cube: &Cubemap<[u8; 4]>, dir: Vec3) -> Vec3 {
    smooth_cubemap_vec3(
        cube.resolution(),
        dir,
        MACRO_MATERIAL_SMOOTH_RADIUS_TEXELS,
        |face, x, y| srgb_rgba8_to_linear_rgb(cube.get(face, x, y)),
    )
}

fn static_roughness_smoothed(body: &StaticSurfaceData, dir: Vec3) -> f32 {
    smooth_cubemap_scalar(
        body.roughness_cubemap.resolution(),
        dir,
        MACRO_MATERIAL_SMOOTH_RADIUS_TEXELS,
        |face, x, y| {
            let texel = body.roughness_cubemap.get(face, x, y);
            if texel > 0 {
                texel as f32 / 255.0
            } else {
                let material_id = body.material_cubemap.get(face, x, y) as usize;
                body.materials
                    .get(material_id)
                    .map(|m| m.roughness)
                    .unwrap_or(0.5)
            }
        },
    )
}

fn smooth_cubemap_scalar(
    res: u32,
    dir: Vec3,
    radius_texels: f32,
    mut sample: impl FnMut(thalos_terrain::cubemap::CubemapFace, u32, u32) -> f32,
) -> f32 {
    let (face, u, v) = dir_to_face_uv(dir);
    let res_f = res as f32;
    let px = (u * res_f - 0.5).clamp(0.0, res_f - 1.0);
    let py = (v * res_f - 0.5).clamp(0.0, res_f - 1.0);
    let support = radius_texels.ceil() as i32;
    let center_x = px.floor() as i32;
    let center_y = py.floor() as i32;
    let max_i = res.saturating_sub(1) as i32;

    let mut sum = 0.0;
    let mut weight_sum = 0.0;
    for dy in -support..=support {
        let sy = (center_y + dy).clamp(0, max_i) as u32;
        let wy = smooth_kernel((sy as f32 - py).abs(), radius_texels);
        if wy <= 0.0 {
            continue;
        }
        for dx in -support..=support {
            let sx = (center_x + dx).clamp(0, max_i) as u32;
            let wx = smooth_kernel((sx as f32 - px).abs(), radius_texels);
            let weight = wx * wy;
            if weight <= 0.0 {
                continue;
            }
            sum += sample(face, sx, sy) * weight;
            weight_sum += weight;
        }
    }

    if weight_sum > 0.0 {
        sum / weight_sum
    } else {
        sample(face, px.round() as u32, py.round() as u32)
    }
}

fn smooth_cubemap_vec3(
    res: u32,
    dir: Vec3,
    radius_texels: f32,
    mut sample: impl FnMut(thalos_terrain::cubemap::CubemapFace, u32, u32) -> Vec3,
) -> Vec3 {
    let (face, u, v) = dir_to_face_uv(dir);
    let res_f = res as f32;
    let px = (u * res_f - 0.5).clamp(0.0, res_f - 1.0);
    let py = (v * res_f - 0.5).clamp(0.0, res_f - 1.0);
    let support = radius_texels.ceil() as i32;
    let center_x = px.floor() as i32;
    let center_y = py.floor() as i32;
    let max_i = res.saturating_sub(1) as i32;

    let mut sum = Vec3::ZERO;
    let mut weight_sum = 0.0;
    for dy in -support..=support {
        let sy = (center_y + dy).clamp(0, max_i) as u32;
        let wy = smooth_kernel((sy as f32 - py).abs(), radius_texels);
        if wy <= 0.0 {
            continue;
        }
        for dx in -support..=support {
            let sx = (center_x + dx).clamp(0, max_i) as u32;
            let wx = smooth_kernel((sx as f32 - px).abs(), radius_texels);
            let weight = wx * wy;
            if weight <= 0.0 {
                continue;
            }
            sum += sample(face, sx, sy) * weight;
            weight_sum += weight;
        }
    }

    if weight_sum > 0.0 {
        sum / weight_sum
    } else {
        sample(face, px.round() as u32, py.round() as u32)
    }
}

fn smooth_kernel(distance_texels: f32, radius_texels: f32) -> f32 {
    if radius_texels <= 0.0 {
        return if distance_texels <= 0.5 { 1.0 } else { 0.0 };
    }
    let t = (1.0 - distance_texels / radius_texels).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn decode_height(texel: u16, range: f32) -> f32 {
    (texel as f32 / 65535.0 * 2.0 - 1.0) * range
}

fn encode_height(height_m: f32, range: f32) -> u16 {
    if range <= f32::EPSILON {
        return 32768;
    }
    (((height_m / range).clamp(-1.0, 1.0) * 0.5 + 0.5) * 65535.0 + 0.5) as u16
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

fn srgb_rgba8_to_linear_rgb(texel: [u8; 4]) -> Vec3 {
    Vec3::new(
        srgb8_to_linear(texel[0]),
        srgb8_to_linear(texel[1]),
        srgb8_to_linear(texel[2]),
    )
}

fn linear_rgb_to_srgba8(linear: Vec3) -> [u8; 4] {
    [
        linear_to_srgb8(linear.x),
        linear_to_srgb8(linear.y),
        linear_to_srgb8(linear.z),
        255,
    ]
}

fn srgb8_to_linear(srgb: u8) -> f32 {
    let srgb = f32::from(srgb) / 255.0;
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
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
        AttachmentFormat::Rg16 => AttachmentData::Rg16(vec![[0, 0]; count]),
        AttachmentFormat::Rgba8 => AttachmentData::Rgba8(vec![[0, 0, 0, 255]; count]),
        AttachmentFormat::Rgb8 => AttachmentData::None,
    }
}
