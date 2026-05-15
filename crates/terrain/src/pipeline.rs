//! [`PipelineTileProvider`] — bridges `bevy_terrain`'s [`TileProvider`] seam
//! to the Thalos synthesis pipeline.
//!
//! Per-pixel evaluation runs in three stages:
//!
//! 1. **Cubemap base + dynamic overlays** — sample baked
//!    height/albedo/roughness from [`StaticSurfaceData`] for the pixel's
//!    body-local direction, then apply terrain-owned time-varying layers
//!    (seasonal ice, aeolian bedforms).
//! 2. **Procedural detail (LOD-adaptive)** — when the tile's per-texel
//!    footprint allows it, additively layer high-frequency noise + 2D gully
//!    erosion on top of the cubemap base:
//!    - `thalos_terrain_gen::noise::fbm3_derivative` provides scalar height
//!      plus analytic gradient evaluated in body-local 3D coords. The 3D
//!      sampling is sphere-continuous, so the same world-space position
//!      gives the same noise value regardless of which tile/face is
//!      generating it — and the impostor's WGSL port of the same `fbm3`
//!      lines up bit-for-bit at the LOD swap distance.
//!    - `bevy_erosion_filter::cpu::erosion_filter` (the 2D filter) runs
//!      per-tile in cube-face UV scaled to meters. Phacelle cell evaluation
//!      is in 2D (16-cell search vs 64-cell for the 3D variant — a
//!      worth-it-for-cost step within a face; cube-face boundary seams are
//!      a known limitation deferred to a later pass).
//!    - The **octave count** in both noise and erosion is derived from the
//!      tile's Nyquist resolution. Coarse tiles run 0–1 octaves (or skip
//!      detail entirely); fine tiles run the full cascade. The cascade
//!      parameters (scale, lacunarity, seeds) are fixed across LODs, so a
//!      world position produces the same features at every LOD — finer
//!      LODs just resolve more of them. With `gain=0.5`, the
//!      most-recently-added octave contributes ≤ ~6 % of total detail per
//!      step, so integer octave-count transitions don't pop visibly.
//! 3. **Color cascade** — when detail is active, port of the
//!    `bevy_erosion_filter` demo's cliff/dirt/snow/sand/grass/drainage
//!    cascade keyed on eroded height + occlusion + ridge map + normal.
//!    Mixed with the cubemap albedo by `detail_strength`.
//!
//! Each tile request computes the full per-pixel evaluation once into a
//! `Vec<TilePixel>` (parallelised across rows with `rayon`) and then encodes
//! each requested attachment from that shared buffer.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;
use bevy::math::{DVec3, UVec2, Vec2, Vec3, Vec4};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use bevy_erosion_filter::cpu::{ErosionFilterParams, erosion_filter};
use bevy_terrain::math::TileCoordinate;
use bevy_terrain::prelude::*;
use bevy_terrain::terrain_data::AttachmentData;
use rayon::prelude::*;
use thalos_terrain_gen::cubemap::{Cubemap, CubemapFace};
use thalos_terrain_gen::noise::fbm3_derivative;
use thalos_terrain_gen::{
    DynamicSurfaceState, PlanetSurface, StaticSurfaceData, apply_dynamic_surface_layers,
    cubemap::dir_to_face_uv,
};

// ---------------------------------------------------------------------------
// Procedural-detail tuning constants
// ---------------------------------------------------------------------------

/// Hash seed for the high-frequency detail noise. Decoupled from the body
/// generator's seed so changing terrain gen doesn't reshuffle ground-LOD
/// detail and vice versa.
const DETAIL_NOISE_SEED: u32 = 0x1E_E0_57_07;

/// Base (octave 0) wavelength of both the noise and erosion cascades, in
/// metres. Subsequent octaves halve at `lacunarity=2`; with the 5-octave
/// cap below, the cascade reaches `BASE_WL / 16 = 50 m`.
const DETAIL_BASE_WL_M: f32 = 800.0;

/// Peak amplitude (one-sided) of the noise contribution, in metres. fBm
/// output is bounded in roughly [-1, 1], so the noise contribution to
/// height stays in ±this band before the LOD-adaptive strength fade.
///
/// Cranked deliberately high for first-cut visibility — the
/// `DETAIL_HEIGHT_MARGIN_M` below has the headroom; tune down once the
/// pipeline is visibly producing relief.
const DETAIL_NOISE_AMP_M: f32 = 200.0;

const DETAIL_NOISE_PERSISTENCE: f32 = 0.5;
const DETAIL_NOISE_LACUNARITY: f32 = 2.0;

/// Maximum cascade depth we ever ask for. Both the noise and the erosion
/// filter use this as their upper octave bound; the LOD-adaptive logic
/// elsewhere picks the actual count per tile.
const MAX_DETAIL_OCTAVES: u32 = 5;

/// Conservative additional height-range margin reserved for procedural
/// detail. Used by [`rendered_height_range`] so the R16 encoding has the
/// extra headroom on top of the static + dynamic ranges. Headroom for the
/// noise envelope plus the erosion magnitude bound.
const DETAIL_HEIGHT_MARGIN_M: f32 = 600.0;

/// Erosion parameters cloned from `bevy_erosion_filter::cpu::Default` with
/// `scale` lifted to metres so octave 0's wavelength is roughly
/// `DETAIL_BASE_WL_M`. `octaves` and `strength` are overwritten per tile
/// by the LOD-adaptive selection.
fn erosion_params_base() -> ErosionFilterParams {
    ErosionFilterParams {
        // `erosion_filter`'s starting frequency is `1 / (scale * cell_scale)`
        // (≈ wavelength `scale * cell_scale`). With cell_scale=0.7, we want
        // scale ≈ BASE_WL / 0.7 so octave 0 lands near BASE_WL.
        scale: DETAIL_BASE_WL_M / 0.7,
        strength: 0.22,
        gully_weight: 0.5,
        detail: 1.5,
        rounding: Vec4::new(0.1, 0.0, 0.1, 2.0),
        onset: Vec4::new(1.25, 1.25, 2.8, 1.5),
        assumed_slope: Vec2::new(0.7, 1.0),
        cell_scale: 0.7,
        normalization: 0.5,
        octaves: MAX_DETAIL_OCTAVES as i32,
        lacunarity: 2.0,
        gain: 0.5,
    }
}

/// Per-tile detail plan: how many octaves to evaluate, and a global
/// strength multiplier that fades the first octave in smoothly so the
/// "0 octaves → 1 octave" transition isn't a step.
#[derive(Clone, Copy, Debug)]
struct DetailPlan {
    octaves: u32,
    strength: f32,
}

/// Choose the cascade depth + fade for a tile at `tile_lod_m`.
///
/// An octave with wavelength `W` is barely Nyquist-resolvable when
/// `tile_lod_m = W / 2`. For lac=2 and base wavelength `BASE_WL`, octave
/// `k` (0-indexed) has wavelength `BASE_WL / 2^k`. The number of octaves
/// resolvable is therefore `floor(log2(ratio))` where
/// `ratio = BASE_WL / (2 * tile_lod_m)`.
///
/// We **add 1** so the first sub-Nyquist octave still contributes, and
/// fade `strength` in across `ratio ∈ [1, 2]` so when the cascade first
/// engages it doesn't snap on at full amplitude.
/// Log the first ~32 tile evaluations so we can confirm the LOD plan in
/// the absence of a real-time profiler. Atomic counter; no allocation, no
/// lock contention beyond the increment.
fn should_log_tile() -> bool {
    static COUNT: AtomicU32 = AtomicU32::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    n < 32
}

fn detail_plan_for_lod(tile_lod_m: f32, base_wl_m: f32) -> DetailPlan {
    let ratio = if tile_lod_m > 0.0 {
        base_wl_m / (2.0 * tile_lod_m)
    } else {
        f32::INFINITY
    };
    if ratio < 1.0 {
        return DetailPlan {
            octaves: 0,
            strength: 0.0,
        };
    }
    let octaves = (ratio.log2().floor() as u32 + 1).min(MAX_DETAIL_OCTAVES);
    let strength = smoothstep(1.0, 2.0, ratio).clamp(0.0, 1.0);
    DetailPlan { octaves, strength }
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// `bevy_terrain` `TileProvider` backed by the Thalos synthesis pipeline.
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
/// bevy_terrain `TerrainModel` height bounds.
pub fn rendered_height_range(surface: &PlanetSurface, state: &DynamicSurfaceState) -> f32 {
    (surface.static_surface.height_range
        + dynamic_height_margin(surface, state)
        + DETAIL_HEIGHT_MARGIN_M)
        .max(1.0)
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
    height_range_m: f32,
    coord: TileCoordinate,
    model: &TerrainModel,
    size: u32,
    border: u32,
) -> Vec<TilePixel> {
    let body = &surface.static_surface;
    let tile_lod_m = tile_lod_m(body, coord, size, border);
    let dynamic_lod = tile_lod_m.log2();
    let has_dynamic_layers = !surface.dynamic_layers.is_empty();
    let uses_expanded_height_range = (height_range_m - body.height_range).abs() > 0.01;
    let radius_m = body.radius_m;
    let max_relief_m = height_range_m.max(100.0);
    let sea_level_m = body.sea_level_m.unwrap_or(0.0);
    let face_size_m = std::f32::consts::FRAC_PI_2 * radius_m;

    // LOD-adaptive cascade depth. Pick once per tile (rather than per
    // pixel) so the result is coherent across the tile and the hot inner
    // loop stays branch-free.
    let plan = detail_plan_for_lod(tile_lod_m, DETAIL_BASE_WL_M);
    let do_detail = plan.octaves > 0 && plan.strength > 0.0;
    let detail_noise_amp_m = DETAIL_NOISE_AMP_M * plan.strength;

    // Diagnostic: log the first few tiles so we can confirm the LOD plan is
    // engaging. Throttled by a once-per-tile-coord atomic flag.
    if should_log_tile() {
        info!(
            "PipelineTileProvider tile lod={} ({}, {}, {}) tile_lod_m={:.1} \
             octaves={} strength={:.2} do_detail={}",
            coord.lod,
            coord.side,
            coord.x,
            coord.y,
            tile_lod_m,
            plan.octaves,
            plan.strength,
            do_detail,
        );
    }

    let mut params = erosion_params_base();
    params.octaves = plan.octaves as i32;
    params.strength *= plan.strength;
    let detail_albedo_mix = plan.strength;

    let count = (size * size) as usize;
    let mut pixels = vec![TilePixel::default(); count];

    pixels
        .par_chunks_mut(size as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..size as usize {
                let dir =
                    pixel_direction(coord, UVec2::new(x as u32, y as u32), size, border, model);
                row[x] = evaluate_pixel(PixelContext {
                    surface,
                    dynamic_state,
                    body,
                    dir,
                    has_dynamic_layers,
                    uses_expanded_height_range,
                    dynamic_lod,
                    radius_m,
                    face_size_m,
                    max_relief_m,
                    sea_level_m,
                    do_detail,
                    detail_octaves: plan.octaves,
                    noise_amp_m: detail_noise_amp_m,
                    detail_albedo_mix,
                    params: &params,
                });
            }
        });

    pixels
}

struct PixelContext<'a> {
    surface: &'a PlanetSurface,
    dynamic_state: &'a DynamicSurfaceState,
    body: &'a StaticSurfaceData,
    dir: Vec3,
    has_dynamic_layers: bool,
    uses_expanded_height_range: bool,
    dynamic_lod: f32,
    radius_m: f32,
    face_size_m: f32,
    max_relief_m: f32,
    sea_level_m: f32,
    do_detail: bool,
    detail_octaves: u32,
    noise_amp_m: f32,
    detail_albedo_mix: f32,
    params: &'a ErosionFilterParams,
}

fn evaluate_pixel(ctx: PixelContext<'_>) -> TilePixel {
    // Stage 1+2: cubemap base + dynamic overlays. Run the dynamic-layer
    // sampler whenever we need linear-space components — for the detail
    // path we always need the linear albedo, even on bodies without dynamic
    // layers.
    let need_linear = ctx.has_dynamic_layers || ctx.uses_expanded_height_range || ctx.do_detail;

    let (base_h, base_albedo, base_roughness) = if need_linear {
        let (mut h, mut a, mut r) = static_sample_components(ctx.body, ctx.dir);
        if ctx.has_dynamic_layers {
            apply_dynamic_surface_layers(
                ctx.surface,
                ctx.dynamic_state,
                ctx.dir,
                ctx.dynamic_lod,
                &mut h,
                &mut a,
                &mut r,
            );
        }
        (h, a, r)
    } else {
        // Cheap fast-path: read the raw cubemap texels (the encoded
        // representations already match the tile attachment formats).
        let h = decode_height(
            cubemap_texel_nearest(&ctx.body.height_cubemap, ctx.dir),
            ctx.body.height_range,
        );
        let a = srgb_rgba8_to_linear_rgb(cubemap_texel_nearest(&ctx.body.albedo_cubemap, ctx.dir));
        let r = static_roughness(ctx.body, ctx.dir);
        (h, a, r)
    };

    if !ctx.do_detail {
        return TilePixel {
            height_m: base_h,
            albedo_linear: base_albedo,
            roughness: base_roughness,
        };
    }

    // Stage 3: procedural detail.
    //
    // - Noise lives in body-local 3D so adjacent cube faces see the same
    //   field at the same physical point (sphere-continuous, hash-stable).
    // - Erosion lives in tile-local 2D, parametrised by face UV scaled to
    //   metres. Adjacent tiles in the same face share their border 2D
    //   coordinates exactly; cube-face boundaries are the seams we accept.
    // - Both use `detail_octaves` chosen by the per-tile LOD planner.

    let (face, face_u, face_v) = dir_to_face_uv(ctx.dir);
    let p_2d = Vec2::new(face_u, face_v) * ctx.face_size_m;

    // Noise via our 3D quintic fbm3.
    let p_3d = ctx.dir * ctx.radius_m;
    let freq = 1.0 / DETAIL_BASE_WL_M;
    let nd = fbm3_derivative(
        p_3d.x * freq,
        p_3d.y * freq,
        p_3d.z * freq,
        DETAIL_NOISE_SEED,
        ctx.detail_octaves,
        DETAIL_NOISE_PERSISTENCE,
        DETAIL_NOISE_LACUNARITY,
    );
    let noise_h = nd.value * ctx.noise_amp_m;
    // Project the 3D noise gradient into the face's constant tangent basis
    // to produce a 2D gradient in face-UV-metres. d(value * amp) / d(p_3d)
    // = amp * freq * grad(noise) — apply that to the basis.
    let (face_tx_3d, face_ty_3d) = face_tangent_basis(face);
    let scale = ctx.noise_amp_m * freq;
    let noise_grad_2d =
        Vec2::new(nd.derivative.dot(face_tx_3d), nd.derivative.dot(face_ty_3d)) * scale;

    // 2D erosion. `params.strength` already has the LOD fade applied.
    let combined_h = base_h + noise_h;
    let fade_target = (combined_h / ctx.max_relief_m).clamp(-1.0, 1.0);
    let result = erosion_filter(
        p_2d,
        Vec3::new(combined_h, noise_grad_2d.x, noise_grad_2d.y),
        fade_target,
        ctx.params,
    );

    // Approximate "world-up component" of the perturbed surface normal for
    // the colour cascade's slope mask. `slope_mag` is the magnitude of the
    // final 2D gradient (in metres-per-metre); a flat surface has 0 →
    // normal.y = 1; vertical → ∞ → normal.y = 0.
    let final_grad_2d = noise_grad_2d + Vec2::new(result.delta.y, result.delta.z);
    let slope_mag = final_grad_2d.length();
    let detail_normal_y = (1.0 / (1.0 + slope_mag * slope_mag).sqrt()).clamp(0.0, 1.0);

    let delta_h = result.delta.x;
    let magnitude = result.magnitude;
    let ridge_map = result.ridge_map;
    let final_h = base_h + noise_h + delta_h;

    // Color cascade. Map "metres above sea relative to max relief" to the
    // demo's unit space where the water line sits at 0.43.
    let elevation_norm = (final_h - ctx.sea_level_m) / ctx.max_relief_m;
    let unit_h = elevation_norm * 0.5 + 0.43;
    let water_unit = 0.43;
    let erosion_delta_unit = if magnitude > 0.0 {
        delta_h / magnitude
    } else {
        0.0
    };
    let occlusion = (erosion_delta_unit + 0.5).clamp(0.0, 1.0);
    let ridgemap_unit = (ridge_map * 0.5 + 0.5).clamp(0.0, 1.0);
    let detail_albedo = terrain_color_cascade(
        unit_h,
        occlusion,
        ridgemap_unit,
        detail_normal_y,
        water_unit,
        erosion_delta_unit,
    );

    TilePixel {
        height_m: final_h,
        albedo_linear: base_albedo.lerp(detail_albedo, ctx.detail_albedo_mix),
        roughness: base_roughness,
    }
}

/// Constant 3D tangent basis (cube-face right, cube-face down) for each
/// cubemap face. Derived from the partial derivatives of the cube-face
/// → 3D direction mapping in [`face_uv_to_dir`].
fn face_tangent_basis(face: CubemapFace) -> (Vec3, Vec3) {
    match face {
        CubemapFace::PosX => (Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, -1.0, 0.0)),
        CubemapFace::NegX => (Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, -1.0, 0.0)),
        CubemapFace::PosY => (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        CubemapFace::NegY => (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
        CubemapFace::PosZ => (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
        CubemapFace::NegZ => (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
    }
}

// ---------------------------------------------------------------------------
// Color cascade — port of `terrain_demo.wgsl::terrain_albedo`
// ---------------------------------------------------------------------------

const CLIFF_COLOR: Vec3 = Vec3::new(0.22, 0.20, 0.20);
const DIRT_COLOR: Vec3 = Vec3::new(0.60, 0.50, 0.40);
const GRASS_COLOR1: Vec3 = Vec3::new(0.15, 0.30, 0.10);
const GRASS_COLOR2: Vec3 = Vec3::new(0.40, 0.50, 0.20);
const SAND_COLOR: Vec3 = Vec3::new(0.80, 0.70, 0.60);
const SNOW_COLOR: Vec3 = Vec3::new(1.0, 1.0, 1.0);
const DRAINAGE_COLOR: Vec3 = Vec3::new(1.0, 1.0, 1.0);
const GRASS_HEIGHT: f32 = 0.465;
const DRAINAGE_WIDTH: f32 = 0.3;

fn terrain_color_cascade(
    h: f32,
    occlusion: f32,
    ridgemap_unit: f32,
    normal_y: f32,
    water_unit: f32,
    erosion_delta: f32,
) -> Vec3 {
    let mut color = CLIFF_COLOR * smoothstep(0.4, 0.52, h);
    color = color.lerp(DIRT_COLOR, smoothstep_down(0.0, 0.6, occlusion));
    color = color.lerp(SNOW_COLOR, smoothstep(0.53, 0.6, h));
    color = color.lerp(
        SAND_COLOR,
        smoothstep_down(water_unit, water_unit + 0.005, h),
    );

    let grass_mix = GRASS_COLOR1.lerp(GRASS_COLOR2, smoothstep(0.4, 0.6, h - erosion_delta * 0.05));
    let grass_height_mask = smoothstep_down(
        GRASS_HEIGHT + 0.02,
        GRASS_HEIGHT + 0.05,
        h + 0.01 + (occlusion - 0.8) * 0.05,
    );
    // The demo gates grass with a tree mask too; ground LOD doesn't (yet)
    // synthesize trees, so the mask collapses to a pure slope test.
    let grass_normal_mask = smoothstep(0.8, 1.0, normal_y);
    color = color.lerp(grass_mix, grass_height_mask * grass_normal_mask);

    let drainage = ((1.0 - (ridgemap_unit / DRAINAGE_WIDTH).clamp(0.0, 1.0)) * 1.5).clamp(0.0, 1.0);
    color = color.lerp(DRAINAGE_COLOR, drainage);
    color
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

fn static_sample_components(body: &StaticSurfaceData, dir: Vec3) -> (f32, Vec3, f32) {
    let height = decode_height(
        cubemap_texel_nearest(&body.height_cubemap, dir),
        body.height_range,
    );
    let albedo = srgb_rgba8_to_linear_rgb(cubemap_texel_nearest(&body.albedo_cubemap, dir));
    let roughness = static_roughness(body, dir);
    (height, albedo, roughness)
}

fn static_roughness(body: &StaticSurfaceData, dir: Vec3) -> f32 {
    let texel = cubemap_texel_nearest(&body.roughness_cubemap, dir);
    if texel > 0 {
        texel as f32 / 255.0
    } else {
        let material_id = cubemap_texel_nearest(&body.material_cubemap, dir) as usize;
        body.materials
            .get(material_id)
            .map(|m| m.roughness)
            .unwrap_or(0.5)
    }
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
    let pix = coord.pixel_coordinate(pixel, texture_size, border_size);
    let surface = pix.world_position(model, 0.0);
    let lifted = pix.world_position(model, 1.0);
    let dir: DVec3 = (lifted - surface).normalize();
    Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32)
}

fn cubemap_texel_nearest<T>(cube: &Cubemap<T>, dir: Vec3) -> T
where
    T: Copy + Default,
{
    let (face, u, v) = dir_to_face_uv(dir);
    let res = cube.resolution();
    let x = ((u * res as f32) as u32).min(res - 1);
    let y = ((v * res as f32) as u32).min(res - 1);
    cube.get(face, x, y)
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

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn smoothstep_down(edge0: f32, edge1: f32, x: f32) -> f32 {
    1.0 - smoothstep(edge0, edge1, x)
}
