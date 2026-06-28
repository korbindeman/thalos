//! Foliage atlas — thin Bevy wrapper over the procedural generator in
//! [`thalos_texgen`]. The generation (leaf clusters + conifer needles + bark, the
//! atlas layout, and [`leaf_code`]) lives in that Bevy-free crate so it can be
//! shared by the runtime (here), the object preview, and the offline bake; this
//! module only uploads the result into a GPU [`Image`].

use bevy::asset::RenderAssetUsages;
use bevy::image::{Image, ImageSampler};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};

// Re-export the atlas layout + packing so `tree_mesh` / consumers keep importing
// them from here; `thalos_texgen` is the source of truth.
pub use thalos_texgen::{
    ATLAS_N, BARK_CELL_COUNT, BARK_CELL_FIRST, LEAF_CELL_COUNT, LEAF_CELL_FIRST, NEEDLE_CELL,
    leaf_code,
};

/// Build the foliage atlas as a GPU `Image` (sRGBA8), ready to bind on
/// `TreeMaterial`. Filterable, clamped, mip-free (the scale-fade shrinks far
/// cards, so leaf shimmer is bounded without mips for now).
pub fn build_foliage_atlas() -> Image {
    let tex = thalos_texgen::foliage_atlas();
    let mut img = Image::new(
        Extent3d {
            width: tex.width,
            height: tex.height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        tex.rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    img.sampler = ImageSampler::linear();
    img
}
