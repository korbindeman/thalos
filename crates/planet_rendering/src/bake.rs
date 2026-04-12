//! Bake a [`thalos_terrain_gen::BodyData`] into GPU cubemap textures.
//!
//! Two cubemap textures are produced:
//! - **Albedo** (`Rgba8UnormSrgb`) — 6-face sRGB colour.
//! - **Height** (`R16Unorm`) — 6-face displacement from `radius_m`.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};

use thalos_terrain_gen::BodyData;
use thalos_terrain_gen::cubemap::CubemapFace;

use crate::texture::PlanetTextures;

/// Bake `BodyData` cubemaps into GPU `Image` assets.
pub fn bake_from_body_data(
    body: &BodyData,
    images: &mut Assets<Image>,
) -> PlanetTextures {
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

    PlanetTextures { albedo, height }
}

/// Create a Bevy `Image` from a `Cubemap<T>` with the right format and view descriptor.
fn create_cubemap_image<T: Copy + Default + thalos_terrain_gen::cubemap::bytemuck_compat::Pod>(
    cubemap: &thalos_terrain_gen::Cubemap<T>,
    resolution: u32,
    format: TextureFormat,
    bytes_per_texel: usize,
    images: &mut Assets<Image>,
) -> Handle<Image> {
    let mut data = Vec::with_capacity(
        (resolution * resolution) as usize * bytes_per_texel * 6,
    );
    for face in CubemapFace::ALL {
        let face_data = cubemap.face_data(face);
        let ptr = face_data.as_ptr() as *const u8;
        let len = std::mem::size_of_val(face_data);
        data.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, len) });
    }

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

/// Build flat cubemap textures (solid colour albedo + zero height).
///
/// Used for bodies without procedural terrain.  Small (4×4 per face)
/// to minimise GPU memory.
pub fn generate_flat_cubemap(
    color: [f32; 3],
    albedo_scale: f32,
    images: &mut Assets<Image>,
) -> PlanetTextures {
    const RES: u32 = 4;
    let r = (color[0] * albedo_scale * 255.0).clamp(0.0, 255.0) as u8;
    let g = (color[1] * albedo_scale * 255.0).clamp(0.0, 255.0) as u8;
    let b = (color[2] * albedo_scale * 255.0).clamp(0.0, 255.0) as u8;

    let face_texels = (RES * RES) as usize;
    let albedo_data: Vec<u8> = (0..face_texels * 6)
        .flat_map(|_| [r, g, b, 255u8])
        .collect();

    let mut albedo_image = Image::new(
        Extent3d { width: RES, height: RES, depth_or_array_layers: 6 },
        TextureDimension::D2,
        albedo_data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    albedo_image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });

    // Zero-height cubemap: 32768 = midpoint in R16Unorm centred encoding.
    let height_data: Vec<u8> = (0..face_texels * 6)
        .flat_map(|_| 32768_u16.to_le_bytes())
        .collect();

    let mut height_image = Image::new(
        Extent3d { width: RES, height: RES, depth_or_array_layers: 6 },
        TextureDimension::D2,
        height_data,
        TextureFormat::R16Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    height_image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });

    PlanetTextures {
        albedo: images.add(albedo_image),
        height: images.add(height_image),
    }
}
