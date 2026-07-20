//! Bake a [`thalos_terrain::PlanetSurface`] into GPU resources.
//!
//! Three layers are produced:
//! - **Cubemaps** (layer 1, always): albedo (`Rgba8UnormSrgb`), height
//!   (`R16Unorm`), roughness (`R8Unorm`). Normals are reconstructed per-
//!   fragment in the impostor shader from the height cube; the bake's
//!   `normal_cubemap` lives in `StaticSurfaceData` for future ground LOD use.
//! - **Feature SSBOs** (layer 2): craters, cell index, feature ids,
//!   radial volcanic features.
//! - **Dynamic overlay textures/buffers** (layer 3): seasonal ice-cap buffers
//!   and active-dune cubemaps. These can be rebuilt or transform-sampled
//!   without touching the static terrain cubemaps.
//!
//! See `crates/terrain/src/sample.rs` for the full LOD contract.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::storage::ShaderBuffer;
use rayon::prelude::*;

use thalos_terrain::cubemap::{CubemapFace, face_uv_to_dir};
use thalos_terrain::{Cubemap, DynamicSurfaceState, PlanetSurface};

use crate::impostor::shader_types::{
    GpuCellRange, GpuCrater, GpuDuneSea, GpuIceCap, GpuRadialFeature,
};
use crate::impostor::texture::PlanetTextures;

/// Maximum face resolution uploaded for the flat planet impostor.
///
/// Ship-view terrain takes over at 4× body radius, where the impostor's
/// closest possible footprint is ~29° across. With Bevy's 45° vertical FOV
/// that is ~695 px on a 1080p viewport and ~927 px on a 1440p viewport, so
/// 1024² cube faces are enough for the current orbital view. Higher-resolution
/// `StaticSurfaceData` is still kept for the ground terrain path.
const IMPOSTOR_MAX_CUBEMAP_RESOLUTION: u32 = 1024;

// ---------------------------------------------------------------------------
// SSBO cell hash — CONTRACT WITH `planet_impostor.wgsl`.
//
// The shader walks a 3×3×3 neighborhood of a 3D cell grid on the unit sphere
// and hashes each (ix, iy, iz) tuple into a dense `cell_index` table. This
// module builds that same table CPU-side. Cell size, table size, and hash
// function MUST match the shader exactly — any drift and the shader reads
// the wrong bucket and the SSBO layer is silent.
//
// Constants mirror the WGSL side in `planet_impostor.wgsl`:
//   const SSBO_CELL_SIZE_UNIT: f32 = 0.06;
//   const CELL_TABLE_SIZE: u32 = 8192u;
//   const CELL_TABLE_MASK: u32 = 8191u;
// ---------------------------------------------------------------------------

const SSBO_CELL_SIZE_UNIT: f32 = 0.06;
const CELL_TABLE_SIZE: usize = 8192;
const CELL_TABLE_MASK: u32 = 8191;

/// WGSL `pcg` ported verbatim. Matches `fn pcg(x: u32) -> u32` in the shader.
fn pcg(x: u32) -> u32 {
    let state = x.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

/// WGSL `hash_cell` ported verbatim. `octave = 0` is reserved for the SSBO
/// cell index; the statistical hash layer uses octaves 1..=11.
fn hash_cell(ix: i32, iy: i32, iz: i32, seed_lo: u32, seed_hi: u32) -> u32 {
    let ux = ix as u32;
    let uy = iy as u32;
    let uz = iz as u32;
    let mut h = ux.wrapping_mul(73856093);
    h ^= uy.wrapping_mul(19349663);
    h ^= uz.wrapping_mul(83492791);
    h = pcg(h);
    // octave = 0 for SSBO layer
    h ^= 0_u32.wrapping_mul(2654435769);
    h ^= seed_lo;
    h = pcg(h);
    h ^= seed_hi.wrapping_mul(1540483477);
    pcg(h)
}

/// Build the dense 3D-cell-grid hash table from a crater population.
///
/// Each crater is inserted into the hash slot of its home cell. Because the
/// shader walks a 3×3×3 neighborhood and every SSBO-band crater (≤ 5 km
/// radius ⇒ ≤ 25 km ejecta reach) fits inside a single cell (~52 km on
/// Mira), home-cell-only insertion is sufficient for correctness.
///
/// Returns `(cell_index, feature_ids)` where `cell_index[i] = (start, count)`
/// into `feature_ids`. The table is always exactly `CELL_TABLE_SIZE` entries
/// long; empty slots have `count = 0` and the shader loops zero iterations.
fn build_ssbo_cell_table(
    craters: &[thalos_terrain::types::Crater],
    bake_threshold_m: f32,
    seed_lo: u32,
    seed_hi: u32,
) -> (Vec<GpuCellRange>, Vec<u32>) {
    let inv = 1.0_f32 / SSBO_CELL_SIZE_UNIT;
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); CELL_TABLE_SIZE];

    for (idx, crater) in craters.iter().enumerate() {
        // SSBO layer only covers craters below the cubemap bake threshold —
        // craters at/above are rendered via the cubemap texel and must not
        // be iterated here (double-count bug).
        if crater.radius_m >= bake_threshold_m {
            continue;
        }

        let c = crater.center.normalize();
        let cx = (c.x * inv).floor() as i32;
        let cy = (c.y * inv).floor() as i32;
        let cz = (c.z * inv).floor() as i32;

        let h = hash_cell(cx, cy, cz, seed_lo, seed_hi);
        let slot = (h & CELL_TABLE_MASK) as usize;
        buckets[slot].push(idx as u32);
    }

    let mut cell_index = Vec::<GpuCellRange>::with_capacity(CELL_TABLE_SIZE);
    let mut feature_ids = Vec::<u32>::new();
    for bucket in &buckets {
        let start = feature_ids.len() as u32;
        feature_ids.extend_from_slice(bucket);
        let count = feature_ids.len() as u32 - start;
        cell_index.push(GpuCellRange { start, count });
    }
    (cell_index, feature_ids)
}

fn radial_frame(center: Vec3) -> (Vec3, Vec3) {
    let up = if center.y.abs() < 0.95 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let east = up.cross(center).normalize_or_zero();
    let north = center.cross(east).normalize_or_zero();
    (east, north)
}

fn volcano_seed(base_seed: u64, index: usize) -> u32 {
    feature_seed(base_seed, index)
}

fn feature_seed(base_seed: u64, index: usize) -> u32 {
    let mut x = base_seed ^ ((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x as u32) ^ ((x >> 32) as u32)
}

fn bake_active_dune_overlay(
    surface: &PlanetSurface,
    state: &DynamicSurfaceState,
) -> (Cubemap<u16>, Cubemap<[u8; 4]>) {
    let body = &surface.static_surface;
    // Dynamic dunes evolve slowly. Keep their overlay texture moderate even
    // when the immutable terrain bake is higher resolution; runtime movement
    // is handled by transforming the texture sample direction, not by
    // per-fragment procedural dune synthesis.
    let resolution = body.height_cubemap.resolution().min(512);
    let mut height = Cubemap::<u16>::new(resolution);
    let mut albedo = Cubemap::<[u8; 4]>::new(resolution);
    if surface.dynamic_layers.active_dunes.is_empty() {
        return (height, albedo);
    }

    let inv = 1.0 / resolution as f32;
    let pixel_size_m = (std::f32::consts::TAU * body.radius_m / (resolution as f32 * 4.0)).max(1.0);
    let row_len = resolution as usize;
    let height_range = body.height_range.max(1.0);
    // Parallelize per-row within each face. `active_dune_overlay_at` reads
    // `surface` and `state` immutably, so rayon can split rows freely; output
    // chunks are non-overlapping per-face buffer slices.
    for face in CubemapFace::ALL {
        let height_face = height.face_data_mut(face);
        let albedo_face = albedo.face_data_mut(face);
        height_face
            .par_chunks_mut(row_len)
            .zip(albedo_face.par_chunks_mut(row_len))
            .enumerate()
            .for_each(|(y, (height_row, albedo_row))| {
                let v = (y as f32 + 0.5) * inv;
                for x in 0..row_len {
                    let u = (x as f32 + 0.5) * inv;
                    let dir = face_uv_to_dir(face, u, v).normalize();
                    let overlay = active_dune_overlay_at(surface, state, dir, pixel_size_m);
                    height_row[x] =
                        ((overlay.height_m / height_range).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                    albedo_row[x] = [
                        linear_to_srgb8(overlay.albedo.x),
                        linear_to_srgb8(overlay.albedo.y),
                        linear_to_srgb8(overlay.albedo.z),
                        quantize_unit_to_u8(overlay.albedo_strength),
                    ];
                }
            });
    }

    (height, albedo)
}

#[derive(Clone, Copy)]
struct DuneOverlayBake {
    height_m: f32,
    albedo: Vec3,
    albedo_strength: f32,
}

fn active_dune_overlay_at(
    surface: &PlanetSurface,
    state: &DynamicSurfaceState,
    dir: Vec3,
    pixel_size_m: f32,
) -> DuneOverlayBake {
    let mut out = DuneOverlayBake {
        height_m: 0.0,
        albedo: Vec3::ZERO,
        albedo_strength: 0.0,
    };
    let radius_m = surface.static_surface.radius_m;
    for (index, layer) in surface.dynamic_layers.active_dunes.iter().enumerate() {
        let dune = &layer.region;
        let dune_state = state
            .active_dune_state(index, layer)
            .cloned()
            .unwrap_or_else(|| thalos_terrain::ActiveDuneState {
                id: layer.id.clone(),
                mobility: layer.mobility,
                ..Default::default()
            });
        if dune_state.coverage_scale <= 0.0 || dune_state.amplitude_scale <= 0.0 {
            continue;
        }

        let center = dune.center.try_normalize().unwrap_or(dir);
        let angular_distance = dir.dot(center).clamp(-1.0, 1.0).acos();
        let outer = (dune.radius_rad + dune.feather_rad.max(0.0)).max(dune.radius_rad + 1e-5);
        let weight = ((1.0 - smoothstep(dune.radius_rad, outer, angular_distance))
            * dune_state.coverage_scale)
            .clamp(0.0, 1.0);
        if weight <= 0.0 {
            continue;
        }

        let draa_lod = smoothstep(4.0, 9.0, dune.lambda_draa_m / pixel_size_m);
        if draa_lod <= 0.001 {
            let broad = (weight * dune.crest_strength * 0.34).clamp(0.0, 0.18);
            if broad > out.albedo_strength {
                out.albedo = Vec3::from_array(dune.albedo_crest_lin) * 0.68;
                out.albedo_strength = broad * 0.42;
            }
            continue;
        }

        let axis = (dune.axis_tangent - center * dune.axis_tangent.dot(center))
            .try_normalize()
            .unwrap_or(Vec3::X);
        let across = center.cross(axis).try_normalize().unwrap_or(Vec3::Z);
        let local = dir - center * dir.dot(center);
        let along_m = local.dot(axis) * radius_m + dune_state.phase_offset_m;
        let cross_m = local.dot(across) * radius_m;
        let broad_warp = simple_value_noise(
            cross_m / radius_m * dune.warp_freq * 0.52,
            dune.seed ^ 0x6D2B_79F5,
        );
        let lace_warp = simple_value_noise(
            cross_m / radius_m * dune.warp_freq * 2.2 + 13.7,
            dune.seed ^ 0x9E37_79B9,
        );
        let meander_m = (broad_warp * 0.75 + lace_warp * 0.25) * dune.warp_amp_unit * radius_m;
        let wind_m = along_m + meander_m;
        let lobe = smoothstep(-0.20, 0.52, broad_warp * 0.62 + weight * 0.10);
        let wavelength_jitter = (1.0
            + simple_value_noise(
                cross_m / radius_m * dune.warp_freq * 0.8 + 19.3,
                dune.seed ^ 0xA24B_AED5,
            ) * 0.24)
            .clamp(0.72, 1.42);
        let ridge = asymmetric_ridge(
            wind_m / (dune.lambda_draa_m.max(1.0) * wavelength_jitter) + broad_warp * 0.35,
            dune.alpha_skew,
        ) * draa_lod;
        let body = (ridge * (0.16 + lobe * 1.05)).clamp(0.0, 1.0);
        let crest = body;
        out.height_m += weight * dune_state.amplitude_scale * dune.amplitude_draa_m.max(0.0) * body;

        let visual = weight * crest;
        if visual > out.albedo_strength {
            out.albedo = Vec3::from_array(dune.albedo_crest_lin);
            out.albedo_strength = (visual * dune.crest_strength).clamp(0.0, 1.0);
        }
    }
    out
}

fn asymmetric_ridge(phase: f32, alpha_skew: f32) -> f32 {
    let t = phase - phase.floor();
    let alpha = alpha_skew.clamp(0.05, 0.95);
    let tri = if t < alpha {
        t / alpha
    } else {
        1.0 - (t - alpha) / (1.0 - alpha)
    };
    tri.clamp(0.0, 1.0).powf(1.35)
}

fn simple_value_noise(x: f32, seed: u64) -> f32 {
    let i0 = x.floor() as i32;
    let i1 = i0 + 1;
    let t = smoothstep(0.0, 1.0, x - x.floor());
    let a = hash_1d(i0, seed);
    let b = hash_1d(i1, seed);
    (a * 2.0 - 1.0) * (1.0 - t) + (b * 2.0 - 1.0) * t
}

fn hash_1d(i: i32, seed: u64) -> f32 {
    let mut h = (i as u32).wrapping_mul(73856093);
    h ^= seed as u32;
    h = pcg(h);
    h ^= (seed >> 32) as u32;
    h = pcg(h);
    h as f32 / 4294967296.0
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() <= f32::EPSILON {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn quantize_unit_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
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

fn srgb8_to_linear(srgb: u8) -> f32 {
    let srgb = f32::from(srgb) / 255.0;
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}

/// CPU-side bake products, ready for main-thread asset insertion.
///
/// Produced by [`prepare_planet_bake`] off the main thread (heavy CPU work:
/// dune-overlay synthesis, cubemap byte copies, SSBO packing), then handed
/// to [`upload_prepared_bake`] to register the assets and produce final
/// `Handle<_>`s. Both halves together replace the legacy
/// [`bake_from_planet_surface`], which is now a thin wrapper.
pub struct PreparedPlanetBake {
    pub albedo: Image,
    pub height: Image,
    pub roughness: Image,
    pub active_dune_height: Image,
    pub active_dune_albedo: Image,
    pub craters: ShaderBuffer,
    pub cell_index: ShaderBuffer,
    pub feature_ids: ShaderBuffer,
    pub radial_features: ShaderBuffer,
    pub ice_caps: ShaderBuffer,
    pub active_dunes: ShaderBuffer,
}

/// Off-thread half of the bake: synthesise dune overlays, copy cubemap
/// bytes into `Image` structs, and pack SSBOs into `ShaderBuffer`s.
/// All work is pure CPU on the caller's thread — no asset-storage access.
pub fn prepare_planet_bake(
    surface: &PlanetSurface,
    state: &DynamicSurfaceState,
) -> PreparedPlanetBake {
    let body = &surface.static_surface;
    // --- Layer 1: cubemaps -------------------------------------------------
    let albedo = impostor_cubemap_image(&body.albedo_cubemap, TextureFormat::Rgba8UnormSrgb);
    let height = impostor_cubemap_image(&body.height_cubemap, TextureFormat::R16Unorm);
    let roughness = impostor_cubemap_image(&body.roughness_cubemap, TextureFormat::R8Unorm);
    let (active_dune_height_cube, active_dune_albedo_cube) =
        bake_active_dune_overlay(surface, state);
    let active_dune_height = cubemap_image(&active_dune_height_cube, TextureFormat::R16Unorm);
    let active_dune_albedo = cubemap_image(&active_dune_albedo_cube, TextureFormat::Rgba8UnormSrgb);
    // Note: `body.normal_cubemap` is intentionally NOT uploaded to the
    // impostor's bind group. 8-bit object-space encoding crushes shallow
    // slope angles (terminator depth, crater rim falloff), so the shader
    // reconstructs normals per-fragment via `perturb_normal_from_height`.
    // The baked normal cube remains in `StaticSurfaceData` for future ground LOD
    // consumers where per-fragment height finite differencing isn't free.

    // --- Layer 2: feature SSBOs --------------------------------------------
    let craters: Vec<GpuCrater> = body
        .craters
        .iter()
        .map(|c| GpuCrater {
            center: c.center,
            radius_m: c.radius_m,
            depth_m: c.depth_m,
            rim_height_m: c.rim_height_m,
            age_gyr: c.age_gyr,
            material_id: c.material_id,
        })
        .collect();

    // Build the 3D-cell-grid hash table that the shader walks. NOT the same
    // layout as `body.feature_index` — that's an icosphere-triangle bucket
    // list used by the CPU sampler. The shader uses a dense 8192-slot hash
    // over 3D cell coordinates; see `build_ssbo_cell_table` above.
    let seed_lo = body.detail_params.seed as u32;
    let seed_hi = (body.detail_params.seed >> 32) as u32;
    let (cell_index, feature_ids) = build_ssbo_cell_table(
        &body.craters,
        body.cubemap_bake_threshold_m,
        seed_lo,
        seed_hi,
    );

    let craters_buf = storage_buffer_from_slice(&craters);
    let cell_index_buf = storage_buffer_from_slice(&cell_index);
    let feature_ids_buf = storage_buffer_from_slice(&feature_ids);
    let radial_features: Vec<GpuRadialFeature> = body
        .volcanoes
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let center = v.center.normalize_or_zero();
            let (east, north) = radial_frame(center);
            GpuRadialFeature {
                center,
                radius_m: v.radius_m,
                east,
                height_m: v.height_m,
                north,
                erosion_scale_m: (v.radius_m / 7.0).clamp(8_000.0, 64_000.0),
                seed: volcano_seed(body.detail_params.seed, i),
                material_id: v.material_id,
                _pad0: 0,
                _pad1: 0,
            }
        })
        .collect();
    let radial_features_buf = storage_buffer_from_slice(&radial_features);
    let ice_caps: Vec<GpuIceCap> = surface
        .dynamic_layers
        .ice_caps
        .iter()
        .enumerate()
        .map(|(i, layer)| {
            let cap = layer.spec;
            let cap_state = state.ice_cap_state(i, layer).cloned().unwrap_or_else(|| {
                thalos_terrain::IceCapState {
                    id: layer.id.clone(),
                    ..Default::default()
                }
            });
            let mut flags = 0u32;
            if cap.north {
                flags |= 1;
            }
            if cap.south {
                flags |= 2;
            }
            let axis = cap.axis.try_normalize().unwrap_or(Vec3::Y);
            GpuIceCap {
                axis,
                flags,
                albedo_linear: Vec3::from_array(cap.albedo_linear),
                edge_latitude_deg: cap.edge_latitude_deg,
                dust_albedo_linear: Vec3::from_array(cap.dust_albedo_linear),
                solid_latitude_deg: cap.solid_latitude_deg,
                edge_noise_deg: cap.edge_noise_deg,
                edge_sharpness: cap.edge_sharpness,
                noise_frequency: cap.noise_frequency,
                max_thickness_m: cap.max_thickness_m,
                albedo_strength: cap.albedo_strength,
                roughness: cap.roughness,
                roughness_strength: cap.roughness_strength,
                obliquity_response: cap.obliquity_response,
                coverage_scale: cap_state.coverage_scale,
                edge_offset_deg: cap_state.edge_offset_deg,
                thickness_scale: cap_state.thickness_scale,
                dustiness: cap_state.dustiness,
                seed: feature_seed(body.detail_params.seed ^ 0x1CEC_AFE5_EA50_0001, i),
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }
        })
        .collect();
    let ice_caps_buf = storage_buffer_from_slice(&ice_caps);
    let active_dunes: Vec<GpuDuneSea> = surface
        .dynamic_layers
        .active_dunes
        .iter()
        .enumerate()
        .map(|(i, layer)| {
            let dune = &layer.region;
            let dune_state = state
                .active_dune_state(i, layer)
                .cloned()
                .unwrap_or_else(|| thalos_terrain::ActiveDuneState {
                    id: layer.id.clone(),
                    mobility: layer.mobility,
                    ..Default::default()
                });
            GpuDuneSea {
                center: dune.center.try_normalize().unwrap_or(Vec3::Y),
                radius_rad: dune.radius_rad,
                axis_tangent: dune.axis_tangent.try_normalize().unwrap_or(Vec3::X),
                feather_rad: dune.feather_rad,
                albedo_crest_lin: Vec3::from_array(dune.albedo_crest_lin),
                crest_strength: dune.crest_strength,
                lambda_draa_m: dune.lambda_draa_m,
                amplitude_draa_m: dune.amplitude_draa_m,
                lambda_dune_m: dune.lambda_dune_m,
                amplitude_dune_m: dune.amplitude_dune_m,
                alpha_skew: dune.alpha_skew,
                warp_amp_unit: dune.warp_amp_unit,
                warp_freq: dune.warp_freq,
                coverage_scale: dune_state.coverage_scale,
                phase_offset_m: dune_state.phase_offset_m,
                amplitude_scale: dune_state.amplitude_scale,
                mobility: dune_state.mobility,
                seed: feature_seed(dune.seed, i),
            }
        })
        .collect();
    let active_dunes_buf = storage_buffer_from_slice(&active_dunes);

    PreparedPlanetBake {
        albedo,
        height,
        roughness,
        active_dune_height,
        active_dune_albedo,
        craters: craters_buf,
        cell_index: cell_index_buf,
        feature_ids: feature_ids_buf,
        radial_features: radial_features_buf,
        ice_caps: ice_caps_buf,
        active_dunes: active_dunes_buf,
    }
}

/// Main-thread half of the bake: take CPU-prepared assets and insert them
/// into Bevy's asset storages, returning the [`PlanetTextures`] handle
/// bundle the material expects. No heavy work — just `Assets::add` calls.
pub fn upload_prepared_bake(
    prep: PreparedPlanetBake,
    images: &mut Assets<Image>,
    storage_buffers: &mut Assets<ShaderBuffer>,
) -> PlanetTextures {
    PlanetTextures {
        albedo: images.add(prep.albedo),
        height: images.add(prep.height),
        roughness: images.add(prep.roughness),
        active_dune_height: images.add(prep.active_dune_height),
        active_dune_albedo: images.add(prep.active_dune_albedo),
        craters: storage_buffers.add(prep.craters),
        cell_index: storage_buffers.add(prep.cell_index),
        feature_ids: storage_buffers.add(prep.feature_ids),
        radial_features: storage_buffers.add(prep.radial_features),
        ice_caps: storage_buffers.add(prep.ice_caps),
        active_dunes: storage_buffers.add(prep.active_dunes),
    }
}

/// Bake `PlanetSurface` into the full set of GPU resources consumed by
/// [`crate::PlanetMaterial`]. Equivalent to running [`prepare_planet_bake`]
/// followed by [`upload_prepared_bake`] on the calling thread, kept for
/// callers that don't need to split the work across thread boundaries.
pub fn bake_from_planet_surface(
    surface: &PlanetSurface,
    state: &DynamicSurfaceState,
    images: &mut Assets<Image>,
    storage_buffers: &mut Assets<ShaderBuffer>,
) -> PlanetTextures {
    let prep = prepare_planet_bake(surface, state);
    upload_prepared_bake(prep, images, storage_buffers)
}

/// Serialize the 6 faces of a `Cubemap<T>` into a contiguous byte buffer in
/// `CubemapFace::ALL` order.
fn cubemap_to_bytes<T: Copy + Default>(cubemap: &thalos_terrain::Cubemap<T>) -> Vec<u8> {
    let resolution = cubemap.resolution();
    let bytes_per_texel = std::mem::size_of::<T>();
    let mut data = Vec::with_capacity((resolution * resolution) as usize * bytes_per_texel * 6);
    for face in CubemapFace::ALL {
        let face_data = cubemap.face_data(face);
        let ptr = face_data.as_ptr() as *const u8;
        let len = std::mem::size_of_val(face_data);
        data.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, len) });
    }
    data
}

/// Create a Bevy `Image` from a `Cubemap<T>` with a cube view descriptor.
fn cubemap_image<T: Copy + Default>(
    cubemap: &thalos_terrain::Cubemap<T>,
    format: TextureFormat,
) -> Image {
    let resolution = cubemap.resolution();
    let data = cubemap_to_bytes(cubemap);
    let mut image = Image::new(
        Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        data,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    image
}

trait AverageTexel: Copy + Default {
    type Sum: Default;

    fn add(sum: &mut Self::Sum, value: Self);
    fn average(sum: Self::Sum, count: u32) -> Self;
}

impl AverageTexel for u8 {
    type Sum = u64;

    fn add(sum: &mut Self::Sum, value: Self) {
        *sum += u64::from(value);
    }

    fn average(sum: Self::Sum, count: u32) -> Self {
        ((sum + u64::from(count / 2)) / u64::from(count)).min(u64::from(u8::MAX)) as u8
    }
}

impl AverageTexel for u16 {
    type Sum = u64;

    fn add(sum: &mut Self::Sum, value: Self) {
        *sum += u64::from(value);
    }

    fn average(sum: Self::Sum, count: u32) -> Self {
        ((sum + u64::from(count / 2)) / u64::from(count)).min(u64::from(u16::MAX)) as u16
    }
}

impl AverageTexel for [u8; 4] {
    type Sum = [f32; 4];

    fn add(sum: &mut Self::Sum, value: Self) {
        sum[0] += srgb8_to_linear(value[0]);
        sum[1] += srgb8_to_linear(value[1]);
        sum[2] += srgb8_to_linear(value[2]);
        sum[3] += f32::from(value[3]) / 255.0;
    }

    fn average(sum: Self::Sum, count: u32) -> Self {
        let inv_count = 1.0 / count as f32;
        [
            linear_to_srgb8(sum[0] * inv_count),
            linear_to_srgb8(sum[1] * inv_count),
            linear_to_srgb8(sum[2] * inv_count),
            quantize_unit_to_u8(sum[3] * inv_count),
        ]
    }
}

fn impostor_cubemap_image<T: AverageTexel>(
    cubemap: &thalos_terrain::Cubemap<T>,
    format: TextureFormat,
) -> Image {
    if cubemap.resolution() <= IMPOSTOR_MAX_CUBEMAP_RESOLUTION {
        return cubemap_image(cubemap, format);
    }

    let downsampled = downsample_cubemap(cubemap, IMPOSTOR_MAX_CUBEMAP_RESOLUTION);
    cubemap_image(&downsampled, format)
}

fn downsample_cubemap<T: AverageTexel>(
    source: &thalos_terrain::Cubemap<T>,
    resolution: u32,
) -> thalos_terrain::Cubemap<T> {
    debug_assert!(resolution > 0);
    debug_assert!(resolution <= source.resolution());

    let source_res = source.resolution();
    let mut output = thalos_terrain::Cubemap::<T>::new(resolution);
    for face in CubemapFace::ALL {
        let src = source.face_data(face);
        let dst = output.face_data_mut(face);
        for y in 0..resolution {
            let y0 = y * source_res / resolution;
            let y1 = ((y + 1) * source_res).div_ceil(resolution).max(y0 + 1);
            for x in 0..resolution {
                let x0 = x * source_res / resolution;
                let x1 = ((x + 1) * source_res).div_ceil(resolution).max(x0 + 1);
                let mut sum = T::Sum::default();
                let mut count = 0;
                for sy in y0..y1.min(source_res) {
                    for sx in x0..x1.min(source_res) {
                        T::add(&mut sum, src[(sy * source_res + sx) as usize]);
                        count += 1;
                    }
                }
                dst[(y * resolution + x) as usize] = T::average(sum, count);
            }
        }
    }
    output
}

/// Pack a slice of Pod data into a [`ShaderBuffer`] without
/// touching the asset storage. Pair with [`Assets::add`] on the main
/// thread.
///
/// The buffer is cast via `bytemuck` — the slice's element type must match
/// the WGSL layout declared in `shader_types.rs`.
///
/// Empty slices are handled by allocating a single-element buffer of the
/// right stride; a zero-size GPU buffer is not a valid binding and wgpu
/// will reject it, so we always upload at least one element's worth of
/// zeroed data.
fn storage_buffer_from_slice<T: bytemuck::Pod + bytemuck::Zeroable>(data: &[T]) -> ShaderBuffer {
    let bytes: Vec<u8> = if data.is_empty() {
        // One zeroed element keeps the binding valid; shader loops read 0
        // elements because the accompanying count/range is zero.
        vec![0u8; std::mem::size_of::<T>()]
    } else {
        bytemuck::cast_slice(data).to_vec()
    };
    ShaderBuffer::new(&bytes, RenderAssetUsages::RENDER_WORLD)
}
