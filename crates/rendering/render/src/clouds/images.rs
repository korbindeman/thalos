use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
        TextureViewDimension,
    },
};

// Two-thirds of the canonical 1080p presentation target. Half-resolution
// erased the authored 450 m boundary scale during the final upscale; this
// keeps that structure visible while retaining a 2.25x pixel-cost reduction
// versus a full-resolution march.
pub const RENDER_WIDTH: u32 = 1280;
pub const RENDER_HEIGHT: u32 = 720;
// 64³ keeps the highest-frequency generated Worley channel above the
// trilinear interpolation floor. At 32³ its ~11 cells/axis had fewer than
// three texels per cell and the resulting lattice showed through in shaded
// cloud boundaries. This costs 4 MiB but does not add runtime texture fetches.
pub const VOLUME_SIZE: u32 = 64;

/// Edge resolution of the view-anchored cloud sun-transmittance cascade
/// (CLOUD-5 / W2 near tier — `docs/rendering/clouds.md` §3.5). 512² against the
/// game-side extent ladder puts one texel at 23 m on the ground and ~120 m from
/// cruise altitude; cloud shadow edges are penumbra-soft at those scales, so
/// resolution buys nothing further, while the march cost is quadratic in it.
pub const CLOUD_SHADOW_SIZE: u32 = 512;

/// Per-face resolution of the canonical cubemap weather projection. The game
/// runtime field must use the same face size.
pub const WEATHER_FACE_SIZE: u32 = 1024;

/// Mip levels carried by the weather cube (1024 → 8 px). Must match the
/// producer's chain (`CloudWeatherField::MIP_LEVELS` game-side).
pub const WEATHER_MIP_LEVELS: u32 = 8;

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
    pub surface_density_bytes: u64,
    pub sun_transmittance_bytes: u64,
    pub total_bytes: u64,
}

pub const fn cloud_target_memory_for(width: u32, height: u32) -> CloudTargetMemory {
    let render_pixels = width as u64 * height as u64;
    let render_bytes = render_pixels * 16; // RGBA32F
    let distance_bytes = render_pixels * 4; // R32F
    let history_bytes = render_bytes;
    let history_distance_bytes = distance_bytes;
    let base_atlas_bytes = 0; // CLOUD-3 deleted the extruded 2-D shape atlas.
    let worley_bytes = VOLUME_SIZE as u64 * VOLUME_SIZE as u64 * VOLUME_SIZE as u64 * 16;
    let coverage_bytes = (6 * cube_layer_mip_bytes(WEATHER_FACE_SIZE, WEATHER_MIP_LEVELS)) as u64;
    let surface_density_bytes = coverage_bytes;
    // RGBA16F cascade (viewport-independent — it is anchored to the view, not
    // sized by it).
    let sun_transmittance_bytes = CLOUD_SHADOW_SIZE as u64 * CLOUD_SHADOW_SIZE as u64 * 8;
    let total_bytes = render_bytes
        + distance_bytes
        + history_bytes
        + history_distance_bytes
        + base_atlas_bytes
        + worley_bytes
        + coverage_bytes
        + surface_density_bytes
        + sun_transmittance_bytes;
    CloudTargetMemory {
        render_bytes,
        distance_bytes,
        history_bytes,
        history_distance_bytes,
        base_atlas_bytes,
        worley_bytes,
        coverage_bytes,
        surface_density_bytes,
        sun_transmittance_bytes,
        total_bytes,
    }
}

pub struct CloudImages {
    pub cloud_render_image: Handle<Image>,
    pub cloud_worley_image: Handle<Image>,
    pub cloud_distance_image: Handle<Image>,
    pub weather_image: Handle<Image>,
    pub surface_density_image: Handle<Image>,
    pub history_image: Handle<Image>,
    pub history_distance_image: Handle<Image>,
    pub cloud_shadow_image: Handle<Image>,
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

    let mut cloud_worley_image = Image::new_fill(
        Extent3d {
            width: VOLUME_SIZE,
            height: VOLUME_SIZE,
            depth_or_array_layers: VOLUME_SIZE,
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

    // View-anchored cloud sun-transmittance cascade (r = beam transmittance
    // toward the sun through the whole deck, 1 = unshadowed). RGBA16F because
    // it is both a compute STORAGE target and a FILTERED sampled texture on
    // every surface receiver: R32Float is not filterable without
    // `Float32Filterable`, and the 8-bit storage formats band a term that gets
    // multiplied straight into direct sunlight. Cleared to 1 so a receiver that
    // samples before the first dispatch is fully lit rather than black.
    let mut cloud_shadow_image = Image::new_fill(
        Extent3d {
            width: CLOUD_SHADOW_SIZE,
            height: CLOUD_SHADOW_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        // half-float 1.0 in all four channels.
        &[0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    cloud_shadow_image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    // Bilinear + clamp: receivers sample it as a smooth world-space field and
    // fade to "lit" at the border, so edge clamping can never wrap a shadow
    // around the cascade.
    cloud_shadow_image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        ..ImageSamplerDescriptor::linear()
    });

    // Clear by default: authored `None` is authoritative and the game only
    // uploads a field for a body with `CloudClimate`.
    let weather_image = cloud_weather_image(
        vec![0; 6 * cube_layer_mip_bytes(WEATHER_FACE_SIZE, WEATHER_MIP_LEVELS)],
        WEATHER_FACE_SIZE,
        WEATHER_MIP_LEVELS,
    );
    let surface_density_image = cloud_weather_image(
        vec![0; 6 * cube_layer_mip_bytes(WEATHER_FACE_SIZE, WEATHER_MIP_LEVELS)],
        WEATHER_FACE_SIZE,
        WEATHER_MIP_LEVELS,
    );

    CloudImages {
        cloud_render_image: images.add(cloud_render_image),
        cloud_worley_image: images.add(cloud_worley_image),
        cloud_distance_image: images.add(cloud_distance_image),
        weather_image: images.add(weather_image),
        surface_density_image: images.add(surface_density_image),
        history_image: images.add(history_image),
        history_distance_image: images.add(history_distance_image),
        cloud_shadow_image: images.add(cloud_shadow_image),
    }
}

pub const fn cloud_target_memory() -> CloudTargetMemory {
    cloud_target_memory_for(RENDER_WIDTH, RENDER_HEIGHT)
}

/// Byte length of one cube layer's full mip chain for `face_size`/`mips`.
const fn cube_layer_mip_bytes(face_size: u32, mips: u32) -> usize {
    let mut total = 0usize;
    let mut size = face_size as usize;
    let mut level = 0;
    while level < mips {
        total += size * size * 4;
        size = if size > 1 { size / 2 } else { 1 };
        level += 1;
    }
    total
}

/// Build the filterable cubemap image used by both near-volume weather and the
/// orbital cloud-cover projection. `rgba` is layer-major with `mips` levels
/// per face (face0[mip0..], face1[mip0..], …); far projections rely on the
/// chain for footprint filtering (disc/limb views alias into ring moiré on a
/// single-level cube).
pub fn cloud_weather_image(rgba: Vec<u8>, face_size: u32, mips: u32) -> Image {
    let expected = 6 * cube_layer_mip_bytes(face_size, mips);
    assert_eq!(rgba.len(), expected, "cloud weather RGBA8 cube byte count");
    // `Image::new`'s size debug-assert compares data length against the
    // level-0 volume only, so the mip chain must be attached after the
    // descriptor declares its mip count (our assert above is the mip-aware
    // equivalent).
    let mut image = Image::new_uninit(
        Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.mip_level_count = mips;
    image.data = Some(rgba);
    image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        ..ImageSamplerDescriptor::linear()
    });
    image
}
