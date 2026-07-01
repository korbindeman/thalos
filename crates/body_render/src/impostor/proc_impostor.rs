//! Bake a low-frequency impostor albedo cubemap from the runtime terrain
//! generator (`thalos_terrain::SurfaceQuery`).
//!
//! The distant-body impostor (`SolidPlanetMaterial`) samples this cube by the
//! body-fixed surface normal, so a procedural planet reads as its real
//! continents + oceans from orbit / the map instead of a flat solid colour —
//! without streaming the full UDLOD terrain at whole-planet distance. Ocean is
//! baked in directly: below sea level (`height < 0`) the texel takes a
//! depth-graded water colour; above, the surface albedo.
//!
//! This is deliberately a *low-frequency* bake (a coarse `lod_m` per texel), the
//! same role the old `PlanetSurface` impostor cubemap played — the near view is
//! UDLOD terrain; the impostor only needs the macro appearance.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};
use rayon::prelude::*;

use thalos_terrain::SurfaceQuery;
use thalos_terrain::cubemap::{CubemapFace, face_uv_to_dir};

/// Deep-water linear-RGB tint (open ocean).
const OCEAN_DEEP: Vec3 = Vec3::new(0.008, 0.028, 0.065);
/// Shallow-water linear-RGB tint (near shore, blends toward deep with depth).
const OCEAN_SHALLOW: Vec3 = Vec3::new(0.020, 0.090, 0.110);
/// Depth (m) over which shallow blends fully into deep.
const OCEAN_DEPTH_SCALE_M: f32 = 400.0;

fn linear_to_srgb_u8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// Bake an `Rgba8UnormSrgb` cube: continents (surface albedo) + oceans (water
/// colour where `height < 0`). `resolution` is the per-face edge in texels.
pub fn bake_impostor_albedo_cube(surface: &dyn SurfaceQuery, resolution: u32) -> Image {
    let res = resolution.max(4) as usize;
    let radius = surface.radius_m().max(1.0);
    // One texel spans ~ this arc on the sphere; feed it as the LOD so the bake
    // takes the matching (coarse) octave count and doesn't alias.
    let lod_m = (std::f32::consts::TAU * radius / (4.0 * res as f32)).max(1.0);

    // Faces in `CubemapFace::ALL` order (the order Bevy's cube view expects),
    // baked in parallel then concatenated.
    let faces: Vec<Vec<u8>> = CubemapFace::ALL
        .par_iter()
        .map(|&face| {
            let mut bytes = vec![0u8; res * res * 4];
            let mut i = 0usize;
            for y in 0..res {
                let v = (y as f32 + 0.5) / res as f32;
                for x in 0..res {
                    let u = (x as f32 + 0.5) / res as f32;
                    let dir = face_uv_to_dir(face, u, v).normalize();
                    let s = surface.sample(dir, lod_m);
                    let is_water = s.height_m < 0.0;
                    let col = if is_water {
                        let t = (-s.height_m / OCEAN_DEPTH_SCALE_M).clamp(0.0, 1.0);
                        OCEAN_SHALLOW.lerp(OCEAN_DEEP, t)
                    } else {
                        s.albedo_linear
                    };
                    bytes[i] = linear_to_srgb_u8(col.x);
                    bytes[i + 1] = linear_to_srgb_u8(col.y);
                    bytes[i + 2] = linear_to_srgb_u8(col.z);
                    // Alpha = water mask (255 = ocean, 0 = land). The impostor
                    // shader shades ocean texels through the shared water BRDF
                    // (Fresnel + sun glint) instead of the land Hapke BRDF. Alpha
                    // is linear in `Rgba8UnormSrgb` (only RGB gets the sRGB curve),
                    // so this reads back as a clean 0/1 flag.
                    bytes[i + 3] = if is_water { 255 } else { 0 };
                    i += 4;
                }
            }
            bytes
        })
        .collect();

    let data: Vec<u8> = faces.into_iter().flatten().collect();
    cube_image(res as u32, data, TextureFormat::Rgba8UnormSrgb)
}

/// A 1×1×6 opaque-black cube for bodies with no baked impostor (solid-colour
/// bodies fall back to the flat `SolidPlanetParams::albedo`, gated by
/// `albedo.w`, and never sample this).
pub fn blank_impostor_cube() -> Image {
    cube_image(1, vec![0u8; 4 * 6], TextureFormat::Rgba8UnormSrgb)
}

fn cube_image(res: u32, data: Vec<u8>, format: TextureFormat) -> Image {
    let mut image = Image::new(
        Extent3d {
            width: res,
            height: res,
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
