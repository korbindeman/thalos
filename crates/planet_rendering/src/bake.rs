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

use thalos_terrain_gen::BodyData;
use thalos_terrain_gen::cubemap::CubemapFace;

use crate::shader_types::{GpuCellRange, GpuCrater, GpuMaterial};
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

    let materials: Vec<GpuMaterial> = body
        .materials
        .iter()
        .map(|m| GpuMaterial {
            albedo: Vec3::from(m.albedo),
            roughness: m.roughness,
        })
        .collect();

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
fn create_cubemap_image<T: Copy + Default>(
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
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    images.add(image)
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
