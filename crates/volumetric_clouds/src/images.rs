use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};

pub const IMAGE_SIZE: u32 = 1920;

/// Resolution of the planet-fixed equirect coverage (weather) map. Texel size
/// at the equator is ~2·π·R / 512 ≈ 39 km for a 3186 km body — weather-scale
/// structure; the cloud base-shape atlas provides everything finer.
pub const COVERAGE_WIDTH: u32 = 512;
pub const COVERAGE_HEIGHT: u32 = 256;

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
            width: 1920,
            height: 1080,
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
            width: 1920,
            height: 1080,
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
            width: 1920,
            height: 1080,
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
            width: 1920,
            height: 1080,
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
