//! Bake a [`thalos_terrain_gen::BodyData`] into GPU resources.
//!
//! Three layers are produced:
//! - **Cubemaps** (layer 1, always): height (`R16Unorm`), albedo (`Rgba8UnormSrgb`),
//!   and material-id (`R8Uint`).
//! - **Feature SSBOs** (layer 2): craters, cell index, feature ids, materials.
//! - **Detail noise params** (layer 3) travel separately via `PlanetDetailParams`.
//!
//! See `crates/terrain_gen/src/sample.rs` for the full LOD contract.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::storage::ShaderStorageBuffer;

use thalos_cloud_gen::{CloudBakeConfig, bake_cloud_cover};
use thalos_terrain_gen::BodyData;
use thalos_terrain_gen::Cubemap;
use thalos_terrain_gen::cubemap::CubemapFace;

use crate::shader_types::{GpuCellRange, GpuCrater, GpuMaterial};
use crate::texture::PlanetTextures;

// Cubemap resolution for baked cloud cover. 256² is ~1.5 MB at R8Unorm
// and bakes in ~2 s on 8 cores; 512² would be 4× that. At orbital
// viewing scale (the only distance this impostor is rendered at), 256
// is visually indistinguishable from 512. If per-planet detail needs
// to climb later, bump this constant and expect a ~4× bake-time hit.
const CLOUD_COVER_RESOLUTION: u32 = 256;

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

/// Bake `BodyData` into the full set of GPU resources consumed by
/// [`crate::PlanetMaterial`].
///
/// This uploads the three cubemap layers and the four feature storage
/// buffers. All handles are bundled into a single [`PlanetTextures`].
pub fn bake_from_body_data(
    body: &BodyData,
    images: &mut Assets<Image>,
    storage_buffers: &mut Assets<ShaderStorageBuffer>,
) -> PlanetTextures {
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
    // Material cubemap is uploaded as a 2D array with 6 layers, NOT a cube
    // view. WGSL/naga has no `textureLoad` overload for `texture_cube<u32>`,
    // so the shader reads it as `texture_2d_array<u32>` — see the comment at
    // the top of `planet_impostor.wgsl` near `sample_material_id`. The face
    // order matches `CubemapFace::ALL`.
    let material_cube = create_2d_array_image(
        &body.material_cubemap,
        body.material_cubemap.resolution(),
        TextureFormat::R8Uint,
        1,
        images,
    );

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

    // Neutral fallback when the pipeline produced no material palette
    // (e.g. Thalos — the baked albedo cube is the sole colour source).
    // `create_storage_buffer_from_slice` pads an empty slice with one
    // zero-initialized element to keep the binding valid, but the shader
    // reads `materials[0]` unconditionally whenever `arrayLength(&materials)`
    // is nonzero, and a zeroed material has `albedo = vec3(0)` — which
    // zeros out `baked_albedo = mat_albedo * tint_mod * regional` and
    // turns the surface black. A neutral (0.5, 1.0) entry collapses the
    // formula to `baked_tint * regional`, matching the intent of the
    // `baked_tint = 0.5 → neutral` design at `planet_impostor.wgsl:1011`.
    let materials: Vec<GpuMaterial> = if body.materials.is_empty() {
        vec![GpuMaterial {
            albedo: Vec3::splat(0.5),
            roughness: 1.0,
        }]
    } else {
        body.materials
            .iter()
            .map(|m| GpuMaterial {
                albedo: Vec3::from(m.albedo),
                roughness: m.roughness,
            })
            .collect()
    };

    let craters_handle = create_storage_buffer_from_slice(&craters, storage_buffers);
    let cell_index_handle = create_storage_buffer_from_slice(&cell_index, storage_buffers);
    let feature_ids_handle = create_storage_buffer_from_slice(&feature_ids, storage_buffers);
    let materials_handle = create_storage_buffer_from_slice(&materials, storage_buffers);

    PlanetTextures {
        albedo,
        height,
        material_cube,
        craters: craters_handle,
        cell_index: cell_index_handle,
        feature_ids: feature_ids_handle,
        materials: materials_handle,
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

/// Nearest-neighbor, non-filtering sampler for integer textures. The layout
/// `#[sampler(7, sampler_type = "non_filtering")]` on `PlanetMaterial`
/// requires the actual sampler to also be non-filtering — wgpu enforces this
/// at bind-group creation time.
fn non_filtering_sampler() -> ImageSampler {
    ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Nearest,
        min_filter: ImageFilterMode::Nearest,
        mipmap_filter: ImageFilterMode::Nearest,
        ..default()
    })
}

/// Create a Bevy `Image` from a `Cubemap<T>` with a 2D array view descriptor.
/// Same data layout as `create_cubemap_image` but no cube-dimension view — the
/// shader reads it as `texture_2d_array<T>` and does its own face lookup.
/// Uses a nearest-neighbor sampler since integer textures can't be filtered.
fn create_2d_array_image<T: Copy + Default>(
    cubemap: &thalos_terrain_gen::Cubemap<T>,
    resolution: u32,
    format: TextureFormat,
    bytes_per_texel: usize,
    images: &mut Assets<Image>,
) -> Handle<Image> {
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
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image.sampler = non_filtering_sampler();
    images.add(image)
}

// ---------------------------------------------------------------------------
// Cloud cover bake
// ---------------------------------------------------------------------------

/// Bake the procedural cloud-cover cubemap for a body and upload it as an
/// R8Unorm cube. Density is linearly encoded: texel value `v / 255` is the
/// raw Worley-fBm weighted sum at the advected direction, in `[0, 1]`.
/// Thresholding / coverage scaling happens shader-side from
/// `AtmosphereBlock::cloud_albedo_coverage`.
///
/// Runs synchronously on the current thread. Typical cost at 256² is
/// ~2 s on a modern 8-core machine; if that becomes a load-time issue,
/// move this call onto the same worker thread that already runs terrain
/// generation.
pub fn bake_cloud_cover_image(seed: u64, images: &mut Assets<Image>) -> Handle<Image> {
    let cfg = CloudBakeConfig::wedekind_defaults(seed, CLOUD_COVER_RESOLUTION);
    let cover_f32 = bake_cloud_cover(&cfg);

    // Convert Cubemap<f32> → Cubemap<u8> for R8Unorm upload. Raw Worley
    // fBm peaks around 0.8, so `* 255` fills most of the 8-bit range
    // without wasting precision.
    let size = cover_f32.resolution();
    let mut cover_u8 = Cubemap::<u8>::new(size);
    for face in CubemapFace::ALL {
        let src = cover_f32.face_data(face);
        let dst = cover_u8.face_data_mut(face);
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            *d = (s.clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    create_cubemap_image(&cover_u8, size, TextureFormat::R8Unorm, 1, images)
}

/// Project an equirectangular 2D image into an R8Unorm cubemap used as
/// cloud cover density. Luminance-weighted: density = 0.299·R + 0.587·G +
/// 0.114·B (per texel, linear 0–255 byte values).
///
/// TEMPORARY: used to drop a reference storm-clouds photo onto Thalos
/// while the procedural cloud pipeline is being redesigned. Remove once
/// `thalos_cloud_gen` covers the Thalos use-case.
///
/// `source` must be an RGBA8 (SRGB or linear) image — the format Bevy's
/// default JPG loader produces. Other formats panic rather than silently
/// miscolour.
pub fn equirect_to_cloud_cover_image(source: &Image, resolution: u32) -> Image {
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
                let dir =
                    thalos_terrain_gen::cubemap::face_uv_to_dir(face, u, v);
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
