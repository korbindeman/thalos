//! Bake a [`thalos_terrain_gen::PlanetSurface`] into GPU resources.
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
//! See `crates/terrain_gen/src/sample.rs` for the full LOD contract.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::storage::ShaderStorageBuffer;

use thalos_terrain_gen::cubemap::{CubemapFace, face_uv_to_dir};
use thalos_terrain_gen::{Cubemap, DynamicSurfaceState, PlanetSurface};

use crate::shader_types::{GpuCellRange, GpuCrater, GpuDuneSea, GpuIceCap, GpuRadialFeature};
use crate::texture::PlanetTextures;

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
    craters: &[thalos_terrain_gen::types::Crater],
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
    for face in CubemapFace::ALL {
        let height_face = height.face_data_mut(face);
        let albedo_face = albedo.face_data_mut(face);
        for y in 0..resolution {
            let v = (y as f32 + 0.5) * inv;
            for x in 0..resolution {
                let u = (x as f32 + 0.5) * inv;
                let dir = face_uv_to_dir(face, u, v).normalize();
                let overlay = active_dune_overlay_at(surface, state, dir, pixel_size_m);
                let idx = (y * resolution + x) as usize;
                height_face[idx] = ((overlay.height_m / body.height_range.max(1.0)).clamp(0.0, 1.0)
                    * 65535.0
                    + 0.5) as u16;
                albedo_face[idx] = [
                    linear_to_srgb8(overlay.albedo.x),
                    linear_to_srgb8(overlay.albedo.y),
                    linear_to_srgb8(overlay.albedo.z),
                    quantize_unit_to_u8(overlay.albedo_strength),
                ];
            }
        }
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
            .unwrap_or_else(|| thalos_terrain_gen::ActiveDuneState {
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

/// Bake `PlanetSurface` into the full set of GPU resources consumed by
/// [`crate::PlanetMaterial`].
///
/// Uploads the cubemap layers (albedo, height, roughness) and the feature
/// storage buffers (craters, cell index, feature ids, radial features).
/// All handles are bundled into a single [`PlanetTextures`].
pub fn bake_from_planet_surface(
    surface: &PlanetSurface,
    state: &DynamicSurfaceState,
    images: &mut Assets<Image>,
    storage_buffers: &mut Assets<ShaderStorageBuffer>,
) -> PlanetTextures {
    let body = &surface.static_surface;
    // --- Layer 1: cubemaps -------------------------------------------------
    let albedo = create_cubemap_image(
        &body.albedo_cubemap,
        body.albedo_cubemap.resolution(),
        TextureFormat::Rgba8UnormSrgb,
        4,
        images,
    );
    let height = create_cubemap_image(
        &body.height_cubemap,
        body.height_cubemap.resolution(),
        TextureFormat::R16Unorm,
        2,
        images,
    );
    let roughness = create_cubemap_image(
        &body.roughness_cubemap,
        body.roughness_cubemap.resolution(),
        TextureFormat::R8Unorm,
        1,
        images,
    );
    let (active_dune_height_cube, active_dune_albedo_cube) =
        bake_active_dune_overlay(surface, state);
    let active_dune_height = create_cubemap_image(
        &active_dune_height_cube,
        active_dune_height_cube.resolution(),
        TextureFormat::R16Unorm,
        2,
        images,
    );
    let active_dune_albedo = create_cubemap_image(
        &active_dune_albedo_cube,
        active_dune_albedo_cube.resolution(),
        TextureFormat::Rgba8UnormSrgb,
        4,
        images,
    );
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

    let craters_handle = create_storage_buffer_from_slice(&craters, storage_buffers);
    let cell_index_handle = create_storage_buffer_from_slice(&cell_index, storage_buffers);
    let feature_ids_handle = create_storage_buffer_from_slice(&feature_ids, storage_buffers);
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
    let radial_features_handle =
        create_storage_buffer_from_slice(&radial_features, storage_buffers);
    let ice_caps: Vec<GpuIceCap> = surface
        .dynamic_layers
        .ice_caps
        .iter()
        .enumerate()
        .map(|(i, layer)| {
            let cap = layer.spec;
            let cap_state = state.ice_cap_state(i, layer).cloned().unwrap_or_else(|| {
                thalos_terrain_gen::IceCapState {
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
                seed: feature_seed(body.detail_params.seed ^ 0x1CE_CAFE5_EA50_0001, i),
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }
        })
        .collect();
    let ice_caps_handle = create_storage_buffer_from_slice(&ice_caps, storage_buffers);
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
                .unwrap_or_else(|| thalos_terrain_gen::ActiveDuneState {
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
    let active_dunes_handle = create_storage_buffer_from_slice(&active_dunes, storage_buffers);

    PlanetTextures {
        albedo,
        height,
        roughness,
        active_dune_height,
        active_dune_albedo,
        craters: craters_handle,
        cell_index: cell_index_handle,
        feature_ids: feature_ids_handle,
        radial_features: radial_features_handle,
        ice_caps: ice_caps_handle,
        active_dunes: active_dunes_handle,
    }
}

/// Serialize the 6 faces of a `Cubemap<T>` into a contiguous byte buffer in
/// `CubemapFace::ALL` order.
fn cubemap_to_bytes<T: Copy + Default>(
    cubemap: &thalos_terrain_gen::Cubemap<T>,
    resolution: u32,
    bytes_per_texel: usize,
) -> Vec<u8> {
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
    cubemap: &thalos_terrain_gen::Cubemap<T>,
    resolution: u32,
    format: TextureFormat,
    bytes_per_texel: usize,
) -> Image {
    let data = cubemap_to_bytes(cubemap, resolution, bytes_per_texel);
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

fn create_cubemap_image<T: Copy + Default>(
    cubemap: &thalos_terrain_gen::Cubemap<T>,
    resolution: u32,
    format: TextureFormat,
    bytes_per_texel: usize,
    images: &mut Assets<Image>,
) -> Handle<Image> {
    images.add(cubemap_image(cubemap, resolution, format, bytes_per_texel))
}

/// Project an equirectangular 2D image into an R8Unorm cubemap used as
/// cloud cover density. Luminance-weighted: density = 0.299·R + 0.587·G +
/// 0.114·B (per texel, linear 0–255 byte values).
///
/// `source` must be an RGBA8 (SRGB or linear) image — the format Bevy's
/// default JPG loader produces. Other formats panic rather than silently
/// miscolour.
pub fn equirect_to_cloud_cover_image(source: &Image, resolution: u32) -> Image {
    equirect_to_cloud_cover_image_with_rotation(source, resolution, Quat::IDENTITY)
}

/// Project an equirectangular 2D image into cloud-cover cubemap density,
/// applying `source_from_output_rotation` before sampling the source image.
///
/// The rotation maps each output cubemap direction into the source image's
/// direction space. For example, a rotation that maps output +Y to source +X
/// makes the source image's front-center longitude appear at the north pole.
pub(crate) fn equirect_to_cloud_cover_image_with_rotation(
    source: &Image,
    resolution: u32,
    source_from_output_rotation: Quat,
) -> Image {
    let fmt = source.texture_descriptor.format;
    assert!(
        matches!(
            fmt,
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
        ),
        "equirect_to_cloud_cover_image: expected Rgba8Unorm{{Srgb}}, got {fmt:?}",
    );
    let src_w = source.texture_descriptor.size.width as usize;
    let src_h = source.texture_descriptor.size.height as usize;
    let src_data = source
        .data
        .as_ref()
        .expect("equirect source image has no CPU data");

    let mut cover = Cubemap::<u8>::new(resolution);
    let inv = 1.0 / resolution as f32;
    for face in CubemapFace::ALL {
        let dst = cover.face_data_mut(face);
        for y in 0..resolution {
            let v = (y as f32 + 0.5) * inv;
            for x in 0..resolution {
                let u = (x as f32 + 0.5) * inv;
                let dir = source_from_output_rotation
                    * thalos_terrain_gen::cubemap::face_uv_to_dir(face, u, v);
                // Equirectangular: longitude from atan2(x, z), latitude
                // from asin(y). Maps to [0, 1] UV matching source image
                // layout (longitude → x, latitude → y, north pole at top).
                let lon = dir.z.atan2(dir.x);
                let lat = dir.y.clamp(-1.0, 1.0).asin();
                let su = (lon / std::f32::consts::TAU + 0.5).fract();
                let sv = 0.5 - lat / std::f32::consts::PI;
                let sx = ((su * src_w as f32) as usize).min(src_w - 1);
                let sy = ((sv * src_h as f32) as usize).min(src_h - 1);
                let i = (sy * src_w + sx) * 4;
                let r = src_data[i] as f32;
                let g = src_data[i + 1] as f32;
                let b = src_data[i + 2] as f32;
                let lum = 0.299 * r + 0.587 * g + 0.114 * b;
                dst[(y * resolution + x) as usize] = lum.clamp(0.0, 255.0) as u8;
            }
        }
    }
    cubemap_image(&cover, resolution, TextureFormat::R8Unorm, 1)
}

/// 1×1 black cubemap used when a body has no cloud layer. Binding slots
/// must still be populated — WGSL has no optional texture bindings — so
/// airless bodies get a blank cube that the shader multiplies by zero
/// coverage.
pub fn blank_cloud_cover_image(images: &mut Assets<Image>) -> Handle<Image> {
    let blank = Cubemap::<u8>::new(1);
    create_cubemap_image(&blank, 1, TextureFormat::R8Unorm, 1, images)
}

/// Upload a slice of Pod data as a read-only storage buffer.
///
/// The buffer is cast via `bytemuck` — the slice's element type must match
/// the WGSL layout declared in `shader_types.rs`.
///
/// Empty slices are handled by allocating a single-element buffer of the
/// right stride; a zero-size GPU buffer is not a valid binding and wgpu
/// will reject it, so we always upload at least one element's worth of
/// zeroed data.
fn create_storage_buffer_from_slice<T: bytemuck::Pod + bytemuck::Zeroable>(
    data: &[T],
    storage_buffers: &mut Assets<ShaderStorageBuffer>,
) -> Handle<ShaderStorageBuffer> {
    let bytes: Vec<u8> = if data.is_empty() {
        // One zeroed element keeps the binding valid; shader loops read 0
        // elements because the accompanying count/range is zero.
        vec![0u8; std::mem::size_of::<T>()]
    } else {
        bytemuck::cast_slice(data).to_vec()
    };
    let buffer = ShaderStorageBuffer::new(&bytes, RenderAssetUsages::RENDER_WORLD);
    storage_buffers.add(buffer)
}
