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

const FOLIAGE_ATLAS_PNG: &[u8] =
    include_bytes!("../../../../../assets/generated/vegetation/foliage_atlas.png");
const FOLIAGE_MATERIAL_ATLAS_PNG: &[u8] =
    include_bytes!("../../../../../assets/generated/vegetation/foliage_material_atlas.png");
const GRASS_CARD_ATLAS_PNG: &[u8] =
    include_bytes!("../../../../../assets/generated/vegetation/grass_card_atlas.png");

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
