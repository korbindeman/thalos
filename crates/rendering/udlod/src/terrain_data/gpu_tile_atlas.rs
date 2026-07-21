use crate::{
    terrain::TerrainComponents,
    terrain_data::{
        tile_atlas::{AtlasAttachment, AtlasTileAttachmentWithData, TileAtlas},
        AttachmentFormat,
    },
};
use bevy::{
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        Extract, MainWorld,
    },
};
use itertools::Itertools;
use std::{iter, mem};

#[derive(Clone, Copy, Debug)]
pub(crate) struct AtlasBufferInfo {
    pub(crate) texture_size: u32,
    pub(crate) border_size: u32,
    pub(crate) center_size: u32,
    format: AttachmentFormat,
    mip_level_count: u32,

    actual_side_size: u32,
}

impl AtlasBufferInfo {
    fn new(attachment: &AtlasAttachment) -> Self {
        // Todo: adjust this code for pixel sizes larger than 4 byte
        // This approach is currently limited to 1, 2, and 4 byte sized pixels
        // Extending it to 8 and 16 sized pixels should be quite easy.
        // However 3, 6, 12 sized pixels do and will not work!
        // For them to work properly we will need to write into a texture instead of buffer.

        let format = attachment.format;
        let texture_size = attachment.texture_size;
        let border_size = attachment.border_size;
        let center_size = attachment.center_size;
        let mip_level_count = attachment.mip_level_count;

        let pixel_size = format.pixel_size();

        let actual_side_size = texture_size * pixel_size;

        Self {
            texture_size,
            border_size,
            center_size,
            mip_level_count,
            actual_side_size,
            format,
        }
    }

    fn image_copy_texture<'a>(
        &'a self,
        texture: &'a Texture,
        index: u32,
        mip_level: u32,
    ) -> TexelCopyTextureInfo<'a> {
        TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: Origin3d {
                z: index,
                ..default()
            },
            aspect: TextureAspect::All,
        }
    }

    fn image_copy_size(&self, mip_level: u32) -> Extent3d {
        Extent3d {
            width: self.texture_size >> mip_level,
            height: self.texture_size >> mip_level,
            depth_or_array_layers: 1,
        }
    }
}

pub(crate) struct GpuAtlasAttachment {
    pub(crate) buffer_info: AtlasBufferInfo,

    pub(crate) atlas_texture: Texture,

    pub(crate) upload_tiles: Vec<AtlasTileAttachmentWithData>,
}

impl GpuAtlasAttachment {
    pub(crate) fn new(
        device: &RenderDevice,
        attachment: &AtlasAttachment,
        atlas_size: u32,
    ) -> Self {
        let name = attachment.name.clone();
        let buffer_info = AtlasBufferInfo::new(attachment);

        // dbg!(&buffer_info);

        let atlas_texture = device.create_texture(&TextureDescriptor {
            label: Some(&format!("{name}_attachment")),
            size: Extent3d {
                width: buffer_info.texture_size,
                height: buffer_info.texture_size,
                depth_or_array_layers: atlas_size,
            },
            mip_level_count: attachment.mip_level_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: buffer_info.format.render_format(),
            usage: TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC
                | TextureUsages::TEXTURE_BINDING,
            view_formats: &[buffer_info.format.processing_format()],
        });

        Self {
            buffer_info,
            atlas_texture,
            upload_tiles: default(),
        }
    }

    fn upload_tiles(&mut self, queue: &RenderQueue) {
        for tile in self.upload_tiles.drain(..) {
            let mut start = 0;

            for mip_level in 0..self.buffer_info.mip_level_count {
                let side_size = self.buffer_info.actual_side_size >> mip_level;
                let texture_size = self.buffer_info.texture_size >> mip_level;
                let end = start + (side_size * texture_size) as usize;

                queue.write_texture(
                    self.buffer_info.image_copy_texture(
                        &self.atlas_texture,
                        tile.tile.atlas_index,
                        mip_level,
                    ),
                    &tile.data.bytes()[start..end],
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(side_size),
                        rows_per_image: Some(texture_size),
                    },
                    self.buffer_info.image_copy_size(mip_level),
                );

                start = end;
            }
        }
    }
}

/// Stores the GPU representation of the [`TileAtlas`] (array textures)
/// alongside the data to update it.
///
/// All attachments of newly loaded tiles are copied into their according atlas attachment.
#[derive(Component)]
pub struct GpuTileAtlas {
    /// Stores the atlas attachments of the terrain.
    pub(crate) attachments: Vec<GpuAtlasAttachment>,
    pub(crate) is_spherical: bool,
}

impl GpuTileAtlas {
    /// The raw atlas texture array of attachment `index` (layers = atlas
    /// slots, full per-tile mip chain). Exposed so external passes (the
    /// `body_render` sky/ocean pass) can bind and sample the resident height
    /// tiles resolved through [`super::gpu_tile_tree::GpuTileTree`].
    pub fn attachment_texture(&self, index: usize) -> Option<&Texture> {
        self.attachments
            .get(index)
            .map(|attachment| &attachment.atlas_texture)
    }

    /// Creates a new gpu tile atlas and initializes its attachment textures.
    fn new(device: &RenderDevice, tile_atlas: &TileAtlas) -> Self {
        let attachments = tile_atlas
            .attachments
            .iter()
            .map(|attachment| GpuAtlasAttachment::new(device, attachment, tile_atlas.atlas_size))
            .collect_vec();

        Self {
            attachments,
            is_spherical: tile_atlas.model.is_spherical(),
        }
    }

    /// Initializes the [`GpuTileAtlas`] of newly created terrains.
    pub(crate) fn initialize(
        device: Res<RenderDevice>,
        mut gpu_tile_atlases: ResMut<TerrainComponents<GpuTileAtlas>>,
        mut tile_atlases: Extract<Query<(Entity, &TileAtlas), Added<TileAtlas>>>,
    ) {
        for (terrain, tile_atlas) in tile_atlases.iter_mut() {
            gpu_tile_atlases.insert(terrain, GpuTileAtlas::new(&device, tile_atlas));
        }
    }

    /// Extracts the tiles that have finished loading from all [`TileAtlas`]es into the
    /// corresponding [`GpuTileAtlas`]es.
    pub(crate) fn extract(
        mut main_world: ResMut<MainWorld>,
        mut gpu_tile_atlases: ResMut<TerrainComponents<GpuTileAtlas>>,
    ) {
        let mut tile_atlases = main_world.query::<(Entity, &mut TileAtlas)>();

        let mut live = Vec::new();
        for (terrain, mut tile_atlas) in tile_atlases.iter_mut(&mut main_world) {
            live.push(terrain);
            let Some(gpu_tile_atlas) = gpu_tile_atlases.get_mut(&terrain) else {
                continue;
            };

            for (attachment, gpu_attachment) in
                iter::zip(&mut tile_atlas.attachments, &mut gpu_tile_atlas.attachments)
            {
                mem::swap(
                    &mut attachment.uploading_tiles,
                    &mut gpu_attachment.upload_tiles,
                );
            }
        }
        gpu_tile_atlases.retain(|terrain, _| live.contains(terrain));
    }

    /// Queues the attachments of the tiles that have finished loading to be copied into the
    /// corresponding atlas attachments.
    pub(crate) fn prepare(
        queue: Res<RenderQueue>,
        mut gpu_tile_atlases: ResMut<TerrainComponents<GpuTileAtlas>>,
    ) {
        for gpu_tile_atlas in gpu_tile_atlases.values_mut() {
            for attachment in &mut gpu_tile_atlas.attachments {
                attachment.upload_tiles(&queue);
            }
        }
    }
}
