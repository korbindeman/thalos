//! Runtime upload for vegetation atlases emitted by `thalos_texgen_tool`.

use bevy::asset::RenderAssetUsages;
use bevy::image::{CompressedImageFormats, Image, ImageSampler, ImageType};
use bevy::render::render_resource::TextureUsages;

// Stable atlas layout consumed by mesh packing and shaders. The offline
// generator owns the pixels; changing this contract requires updating both.
pub const ATLAS_N: u32 = 4;
pub const LEAF_CELL_FIRST: u32 = 0;
pub const LEAF_CELL_COUNT: u32 = 11;
pub const NEEDLE_CELL: u32 = 11;
pub const BARK_CELL_FIRST: u32 = 13;
pub const BARK_CELL_COUNT: u32 = 3;
pub const GRASS_CARD_VARIANTS: u32 = 4;

#[inline]
pub fn leaf_code(cell: u32, corner: u32) -> f32 {
    (cell * 4 + (corner & 3)) as f32
}

const FOLIAGE_ATLAS_PNG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../assets/generated/vegetation/foliage_atlas.png"
));
const FOLIAGE_MATERIAL_ATLAS_PNG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../assets/generated/vegetation/foliage_material_atlas.png"
));
const GRASS_CARD_ATLAS_PNG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../assets/generated/vegetation/grass_card_atlas.png"
));

const FOLIAGE_ATLAS_SIZE_PX: f32 = 1024.0;

/// Decode the stable `cell * 4 + corner` payload into foliage-atlas UVs.
///
/// Keeping this on the CPU lets ordinary Bevy PBR materials consume the same
/// mesh payload as Thalos's specialized tree shader.
pub fn atlas_uv(code: f32) -> [f32; 2] {
    let cell = (code / 4.0).floor();
    let corner = code - cell * 4.0;
    let col = cell % ATLAS_N as f32;
    let row = (cell / ATLAS_N as f32).floor();
    let corner_u = if corner == 1.0 || corner == 2.0 {
        1.0
    } else {
        0.0
    };
    let corner_v = if corner == 2.0 || corner == 3.0 {
        1.0
    } else {
        0.0
    };
    let cell_size = 1.0 / ATLAS_N as f32;
    let texel = 1.0 / FOLIAGE_ATLAS_SIZE_PX;
    [
        col * cell_size + texel + corner_u * (cell_size - 2.0 * texel),
        row * cell_size + texel + corner_v * (cell_size - 2.0 * texel),
    ]
}

/// Baked sRGBA foliage albedo atlas.
pub fn build_foliage_atlas() -> Image {
    decode(FOLIAGE_ATLAS_PNG, true)
}

/// Baked linear normal/roughness atlas.
pub fn build_foliage_material_atlas() -> Image {
    decode(FOLIAGE_MATERIAL_ATLAS_PNG, false)
}

/// Baked linear grass-card modulation/coverage atlas.
pub fn build_grass_card_atlas() -> Image {
    decode(GRASS_CARD_ATLAS_PNG, false)
}

fn decode(bytes: &[u8], is_srgb: bool) -> Image {
    let mut image = Image::from_buffer(
        bytes,
        ImageType::Extension("png"),
        CompressedImageFormats::NONE,
        is_srgb,
        ImageSampler::linear(),
        RenderAssetUsages::RENDER_WORLD,
    )
    .expect("offline-generated vegetation atlas must be a valid PNG");
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_uv_corners_stay_inside_the_selected_cell() {
        for cell in 0..ATLAS_N * ATLAS_N {
            let min = atlas_uv(leaf_code(cell, 0));
            let max = atlas_uv(leaf_code(cell, 2));
            let cell_min = [
                (cell % ATLAS_N) as f32 / ATLAS_N as f32,
                (cell / ATLAS_N) as f32 / ATLAS_N as f32,
            ];
            let cell_max = [cell_min[0] + 0.25, cell_min[1] + 0.25];
            assert!(min[0] > cell_min[0] && min[1] > cell_min[1]);
            assert!(max[0] < cell_max[0] && max[1] < cell_max[1]);
        }
    }
}
