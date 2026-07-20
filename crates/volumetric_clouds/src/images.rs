use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};

pub const IMAGE_SIZE: u32 = 1920;
pub const RENDER_WIDTH: u32 = 1920;
pub const RENDER_HEIGHT: u32 = 1080;

/// Resolution of the planet-fixed equirect coverage (weather) map. Texel size
/// at the equator is ~2·π·R / 512 ≈ 39 km for a 3186 km body — weather-scale
/// structure; the cloud base-shape atlas provides everything finer.
pub const COVERAGE_WIDTH: u32 = 512;
pub const COVERAGE_HEIGHT: u32 = 256;

/// Persistent GPU allocation owned by the current cloud renderer. CLOUD-0
/// publishes this alongside timings so later low-resolution/format work has an
/// exact memory baseline instead of an estimate copied into documentation.
#[derive(Debug, Clone, Copy)]
pub struct CloudTargetMemory {
    pub render_bytes: u64,
    pub distance_bytes: u64,
    pub history_bytes: u64,
    pub history_distance_bytes: u64,
    pub base_atlas_bytes: u64,
    pub worley_bytes: u64,
    pub coverage_bytes: u64,
    pub total_bytes: u64,
}

pub const fn cloud_target_memory() -> CloudTargetMemory {
    let render_pixels = RENDER_WIDTH as u64 * RENDER_HEIGHT as u64;
    let render_bytes = render_pixels * 16; // RGBA32F
    let distance_bytes = render_pixels * 4; // R32F
    let history_bytes = render_bytes;
    let history_distance_bytes = distance_bytes;
    let base_atlas_bytes = IMAGE_SIZE as u64 * IMAGE_SIZE as u64 * 16; // RGBA32F
    let worley_bytes = 32 * 32 * 32 * 16; // RGBA32F 3-D
    let coverage_bytes = COVERAGE_WIDTH as u64 * COVERAGE_HEIGHT as u64; // R8
    let total_bytes = render_bytes
        + distance_bytes
        + history_bytes
        + history_distance_bytes
        + base_atlas_bytes
        + worley_bytes
        + coverage_bytes;
    CloudTargetMemory {
        render_bytes,
        distance_bytes,
        history_bytes,
        history_distance_bytes,
        base_atlas_bytes,
        worley_bytes,
        coverage_bytes,
        total_bytes,
    }
}

pub struct CloudImages {
    pub cloud_render_image: Handle<Image>,
    pub cloud_atlas_image: Handle<Image>,
    pub cloud_worley_image: Handle<Image>,
    pub cloud_distance_image: Handle<Image>,
    pub coverage_image: Handle<Image>,
    pub history_image: Handle<Image>,
    pub history_distance_image: Handle<Image>,
}

pub fn build_images(mut images: ResMut<Assets<Image>>) -> CloudImages {
    let mut cloud_render_image = Image::new_fill(
        Extent3d {
            width: RENDER_WIDTH,
            height: RENDER_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 4 * 4 * 2],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    cloud_render_image.texture_descriptor.usage = TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING;

    let mut cloud_atlas_image = Image::new_fill(
        Extent3d {
            width: IMAGE_SIZE,
            height: IMAGE_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 4 * 4 * 2],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    cloud_atlas_image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    let mut cloud_worley_image = Image::new_fill(
        Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 32,
        },
        TextureDimension::D3,
        &[0; 4 * 4 * 2],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    cloud_worley_image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    // Per-pixel nearest cloud-hit distance (metres from the camera; large
    // sentinel where the ray hit no cloud). Written by the `update` raymarch,
    // read by the game's `body_sky` composite for true depth occlusion.
    let mut cloud_distance_image = Image::new_fill(
        Extent3d {
            width: RENDER_WIDTH,
            height: RENDER_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 4],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    cloud_distance_image.texture_descriptor.usage = TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING;

    // Temporal-history copies of the render + distance textures, snapshotted
    // by the render node AFTER each `update` dispatch. The raymarch reads its
    // history (same-pixel accumulation, motion reprojection, the saved camera
    // rows) exclusively from these, so it never races the textures it is
    // writing this frame — in-pass history reads showed up as coherent streak
    // artifacts in motion.
    let mut history_image = Image::new_fill(
        Extent3d {
            width: RENDER_WIDTH,
            height: RENDER_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 4 * 4 * 2],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    history_image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;

    let mut history_distance_image = Image::new_fill(
        Extent3d {
            width: RENDER_WIDTH,
            height: RENDER_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 4],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    history_distance_image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;

    // Planet-fixed equirect coverage (weather) map: R8, u = longitude
    // (atan2(z, x), wrapping), v = colatitude (acos(y), clamped). Defaults to
    // full coverage so the scalar `clouds_coverage` knob alone reproduces the
    // pre-weather behaviour until a consumer writes a real field. Kept in
    // MAIN_WORLD too so the game can regenerate it at runtime.
    let mut coverage_image = Image::new_fill(
        Extent3d {
            width: COVERAGE_WIDTH,
            height: COVERAGE_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0xFF],
        TextureFormat::R8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    coverage_image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    coverage_image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::ClampToEdge,
        ..ImageSamplerDescriptor::linear()
    });

    CloudImages {
        cloud_render_image: images.add(cloud_render_image),
        cloud_atlas_image: images.add(cloud_atlas_image),
        cloud_worley_image: images.add(cloud_worley_image),
        cloud_distance_image: images.add(cloud_distance_image),
        coverage_image: images.add(coverage_image),
        history_image: images.add(history_image),
        history_distance_image: images.add(history_distance_image),
    }
}
