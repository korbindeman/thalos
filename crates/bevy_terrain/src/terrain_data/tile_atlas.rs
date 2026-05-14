use crate::{
    formats::TC,
    math::{TerrainModel, TileCoordinate},
    prelude::{AttachmentConfig, AttachmentFormat},
    terrain::TerrainConfig,
    terrain_data::{
        tile_provider::{DiskTileProvider, TileProvider},
        tile_tree::{TileLookup, TileTree, TileTreeEntry},
        AttachmentData, INVALID_ATLAS_INDEX, INVALID_LOD,
    },
    terrain_view::TerrainViewComponents,
};
use anyhow::Result;
use bevy::{
    platform::collections::{HashMap, HashSet},
    prelude::*,
    render::render_resource::*,
    tasks::{futures_lite::future, AsyncComputeTaskPool, Task},
};
use image::{DynamicImage, ImageBuffer, Luma, LumaA, Rgb, Rgba};
use itertools::Itertools;
use std::{collections::VecDeque, fs, mem, ops::DerefMut};

pub type Rgb8Image = ImageBuffer<Rgb<u8>, Vec<u8>>;
pub type Rgba8Image = ImageBuffer<Rgba<u8>, Vec<u8>>;
pub type R16Image = ImageBuffer<Luma<u16>, Vec<u16>>;
pub type Rg16Image = ImageBuffer<LumaA<u16>, Vec<u16>>;

pub(crate) const STORE_PNG: bool = false;

#[derive(Copy, Clone, Debug, Default, ShaderType)]
pub struct AtlasTile {
    pub(crate) coordinate: TileCoordinate,
    #[shader(size(16))]
    pub(crate) atlas_index: u32,
}

impl AtlasTile {
    pub fn new(tile_coordinate: TileCoordinate, atlas_index: u32) -> Self {
        Self {
            coordinate: tile_coordinate,
            atlas_index,
        }
    }
    pub fn attachment(self, attachment_index: u32) -> AtlasTileAttachment {
        AtlasTileAttachment {
            coordinate: self.coordinate,
            atlas_index: self.atlas_index,
            attachment_index,
        }
    }
}

impl From<AtlasTileAttachment> for AtlasTile {
    fn from(tile: AtlasTileAttachment) -> Self {
        Self {
            coordinate: tile.coordinate,
            atlas_index: tile.atlas_index,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct AtlasTileAttachment {
    pub(crate) coordinate: TileCoordinate,
    pub(crate) atlas_index: u32,
    pub(crate) attachment_index: u32,
}

#[derive(Clone)]
pub(crate) struct AtlasTileAttachmentWithData {
    pub(crate) tile: AtlasTileAttachment,
    pub(crate) data: AttachmentData,
    pub(crate) texture_size: u32,
}

impl AtlasTileAttachmentWithData {
    pub(crate) fn start_saving(self, path: String) -> Task<AtlasTileAttachment> {
        AsyncComputeTaskPool::get().spawn(async move {
            if STORE_PNG {
                let path = self.tile.coordinate.path(&path, "png");

                let image = match self.data {
                    AttachmentData::Rgba8(data) => {
                        let data = data.into_iter().flatten().collect_vec();
                        DynamicImage::from(
                            Rgba8Image::from_raw(self.texture_size, self.texture_size, data)
                                .unwrap(),
                        )
                    }
                    AttachmentData::R16(data) => DynamicImage::from(
                        R16Image::from_raw(self.texture_size, self.texture_size, data).unwrap(),
                    ),
                    AttachmentData::Rg16(data) => {
                        let data = data.into_iter().flatten().collect_vec();
                        DynamicImage::from(
                            Rg16Image::from_raw(self.texture_size, self.texture_size, data)
                                .unwrap(),
                        )
                    }
                    AttachmentData::None => panic!("Attachment has not data."),
                };

                image.save(&path).unwrap();

                println!("Finished saving tile: {path}");
            } else {
                let path = self.tile.coordinate.path(&path, "bin");

                fs::write(path, self.data.bytes()).unwrap();

                // println!("Finished saving tile: {path}");
            }

            self.tile
        })
    }

}

/// An attachment of a [`TileAtlas`].
pub struct AtlasAttachment {
    pub(crate) name: String,
    pub(crate) path: String,
    pub(crate) texture_size: u32,
    pub(crate) center_size: u32,
    pub(crate) border_size: u32,
    scale: f32,
    offset: f32,
    pub(crate) mip_level_count: u32,
    pub(crate) format: AttachmentFormat,
    pub(crate) data: Vec<AttachmentData>,

    pub(crate) saving_tiles: Vec<Task<AtlasTileAttachment>>,
    pub(crate) uploading_tiles: Vec<AtlasTileAttachmentWithData>,
    pub(crate) downloading_tiles: Vec<Task<AtlasTileAttachmentWithData>>,
}

impl AtlasAttachment {
    fn new(config: &AttachmentConfig, tile_atlas_size: u32, path: &str) -> Self {
        let name = config.name.clone();
        let path = format!("assets/{path}/data/{name}");
        let center_size = config.texture_size - 2 * config.border_size;

        Self {
            name,
            path,
            texture_size: config.texture_size,
            center_size,
            border_size: config.border_size,
            scale: center_size as f32 / config.texture_size as f32,
            offset: config.border_size as f32 / config.texture_size as f32,
            mip_level_count: config.mip_level_count,
            format: config.format,
            data: vec![AttachmentData::None; tile_atlas_size as usize],
            saving_tiles: default(),
            uploading_tiles: default(),
            downloading_tiles: default(),
        }
    }

    fn update(&mut self, atlas_state: &mut TileAtlasState) {
        self.downloading_tiles.retain_mut(|tile| {
            future::block_on(future::poll_once(tile)).map_or(true, |tile| {
                atlas_state.downloaded_tile_attachment(tile.tile);
                self.data[tile.tile.atlas_index as usize] = tile.data;
                false
            })
        });

        self.saving_tiles.retain_mut(|task| {
            future::block_on(future::poll_once(task)).map_or(true, |tile| {
                atlas_state.saved_tile_attachment(tile);
                false
            })
        });
    }

    fn save(&mut self, tile: AtlasTileAttachment) {
        self.saving_tiles.push(
            AtlasTileAttachmentWithData {
                tile: tile,
                data: self.data[tile.atlas_index as usize].clone(),
                texture_size: self.texture_size,
            }
            .start_saving(self.path.clone()),
        );
    }

    fn sample(&self, lookup: TileLookup) -> Vec4 {
        if lookup.atlas_index == INVALID_ATLAS_INDEX {
            return Vec4::splat(0.0); // Todo: Handle this better
        }

        let data = &self.data[lookup.atlas_index as usize];
        let uv = lookup.atlas_uv * self.scale + self.offset;

        data.sample(uv, self.texture_size)
    }
}

/// The current state of a tile of a [`TileAtlas`].
///
/// This indicates, whether the tile is loading or loaded and ready to be used.
#[derive(Clone, Copy)]
enum LoadingState {
    /// The tile is loading, but can not be used yet.
    Loading,
    /// The tile is loaded and can be used.
    Loaded,
}

/// The internal representation of a present tile in a [`TileAtlas`].
struct TileState {
    /// Indicates whether or not the tile is loading or loaded.
    state: LoadingState,
    /// The index of the tile inside the atlas.
    atlas_index: u32,
    /// The count of [`TileTrees`] that have requested this tile.
    requests: u32,
}

/// One in-flight per-tile load issued via the [`TileProvider`].
pub(crate) struct LoadingTile {
    coord: TileCoordinate,
    atlas_index: u32,
    task: Task<Result<Vec<AttachmentData>>>,
}

pub(crate) struct TileAtlasState {
    tile_states: HashMap<TileCoordinate, TileState>,
    unused_tiles: VecDeque<AtlasTile>,
    pub(crate) existing_tiles: HashSet<TileCoordinate>,
    /// When `true`, the [`TileProvider`] can synthesise data for any
    /// [`TileCoordinate`]; tile requests bypass the `existing_tiles` gate
    /// (which exists to keep disk providers from chasing missing `.bin`/`.png`
    /// files). Set once at construction from
    /// [`TileProvider::supports_all_tiles`].
    supports_all_tiles: bool,

    to_load: VecDeque<(TileCoordinate, u32)>,
    loading_tiles: Vec<LoadingTile>,
    load_slots: u32,
    to_save: VecDeque<AtlasTileAttachment>,
    pub(crate) save_slots: u32,
    pub(crate) max_save_slots: u32,

    pub(crate) download_slots: u32,
    pub(crate) max_download_slots: u32,

    pub(crate) max_atlas_write_slots: u32,
}

impl TileAtlasState {
    fn new(
        atlas_size: u32,
        existing_tiles: HashSet<TileCoordinate>,
        supports_all_tiles: bool,
    ) -> Self {
        let unused_tiles = (0..atlas_size)
            .map(|atlas_index| AtlasTile::new(TileCoordinate::INVALID, atlas_index))
            .collect();

        Self {
            tile_states: default(),
            unused_tiles,
            existing_tiles,
            supports_all_tiles,
            to_save: default(),
            to_load: default(),
            loading_tiles: default(),
            save_slots: 64,
            max_save_slots: 64,
            load_slots: 64,
            download_slots: 128,
            max_download_slots: 128,
            max_atlas_write_slots: 32,
        }
    }

    fn update(
        &mut self,
        provider: &dyn TileProvider,
        model: &TerrainModel,
        attachment_configs: &[AttachmentConfig],
        attachments: &mut [AtlasAttachment],
    ) {
        while self.save_slots > 0 {
            if let Some(tile) = self.to_save.pop_front() {
                attachments[tile.attachment_index as usize].save(tile);
                self.save_slots -= 1;
            } else {
                break;
            }
        }

        while self.load_slots > 0 {
            if let Some((coord, atlas_index)) = self.to_load.pop_front() {
                let task = provider.request_tile(coord, model, attachment_configs);
                self.loading_tiles.push(LoadingTile {
                    coord,
                    atlas_index,
                    task,
                });
                self.load_slots -= 1;
            } else {
                break;
            }
        }

        let mut completed = Vec::new();
        self.loading_tiles.retain_mut(|loading| {
            future::block_on(future::poll_once(&mut loading.task)).map_or(true, |result| {
                completed.push((loading.coord, loading.atlas_index, result));
                false
            })
        });

        for (coord, atlas_index, result) in completed {
            self.load_slots += 1;
            match result {
                Ok(datas) => {
                    if datas.len() != attachments.len() {
                        panic!(
                            "TileProvider returned {} attachments for tile {coord}, expected {}",
                            datas.len(),
                            attachments.len()
                        );
                    }

                    for (attachment_index, mut data) in datas.into_iter().enumerate() {
                        let attachment = &mut attachments[attachment_index];
                        data.generate_mipmaps(attachment.texture_size, attachment.mip_level_count);
                        attachment.uploading_tiles.push(AtlasTileAttachmentWithData {
                            tile: AtlasTileAttachment {
                                coordinate: coord,
                                atlas_index,
                                attachment_index: attachment_index as u32,
                            },
                            data: data.clone(),
                            texture_size: attachment.texture_size,
                        });
                        attachment.data[atlas_index as usize] = data;
                    }
                    trace!("loaded tile {coord} into atlas slot {atlas_index}");
                    self.loaded_tile(coord);
                }
                Err(e) => {
                    // Tile remains in `Loading` state; renderer continues to fall
                    // back to the parent LOD. Slot was returned above.
                    warn!("tile load failed for {coord}: {e}");
                }
            }
        }
    }

    fn loaded_tile(&mut self, coord: TileCoordinate) {
        let tile_state = self.tile_states.get_mut(&coord).unwrap();
        tile_state.state = LoadingState::Loaded;
    }

    fn saved_tile_attachment(&mut self, _tile: AtlasTileAttachment) {
        self.save_slots += 1;
    }

    fn downloaded_tile_attachment(&mut self, _tile: AtlasTileAttachment) {
        self.download_slots += 1;
    }

    fn get_tile(&mut self, tile_coordinate: TileCoordinate) -> AtlasTile {
        if tile_coordinate == TileCoordinate::INVALID {
            return AtlasTile::new(TileCoordinate::INVALID, INVALID_ATLAS_INDEX);
        }

        let atlas_index = if self.existing_tiles.contains(&tile_coordinate) {
            self.tile_states.get(&tile_coordinate).unwrap().atlas_index
        } else {
            INVALID_ATLAS_INDEX
        };

        AtlasTile::new(tile_coordinate, atlas_index)
    }

    fn allocate_tile(&mut self) -> u32 {
        let unused_tile = self.unused_tiles.pop_front().expect("Atlas out of indices");

        self.tile_states.remove(&unused_tile.coordinate);

        unused_tile.atlas_index
    }

    fn get_or_allocate_tile(&mut self, tile_coordinate: TileCoordinate) -> AtlasTile {
        if tile_coordinate == TileCoordinate::INVALID {
            return AtlasTile::new(TileCoordinate::INVALID, INVALID_ATLAS_INDEX);
        }

        self.existing_tiles.insert(tile_coordinate);

        let atlas_index = if let Some(tile) = self.tile_states.get(&tile_coordinate) {
            tile.atlas_index
        } else {
            let atlas_index = self.allocate_tile();

            self.tile_states.insert(
                tile_coordinate,
                TileState {
                    requests: 1,
                    state: LoadingState::Loaded,
                    atlas_index,
                },
            );

            atlas_index
        };

        AtlasTile::new(tile_coordinate, atlas_index)
    }

    fn request_tile(&mut self, tile_coordinate: TileCoordinate) {
        // Disk providers gate requests on `existing_tiles` (populated from
        // `config.tc`) to avoid chasing missing files; synthesised providers
        // set `supports_all_tiles` and skip the gate so any coordinate the
        // tile tree asks for gets queued.
        if !self.supports_all_tiles && !self.existing_tiles.contains(&tile_coordinate) {
            return;
        }
        // Insert into `existing_tiles` so the (otherwise disk-oriented)
        // `get_tile` lookup and `save_tile_config` enumeration both see the
        // synthesised tiles too.
        if self.supports_all_tiles {
            self.existing_tiles.insert(tile_coordinate);
        }

        let mut tile_states = mem::take(&mut self.tile_states);

        // check if the tile is already present else start loading it
        if let Some(tile) = tile_states.get_mut(&tile_coordinate) {
            if tile.requests == 0 {
                // the tile is now used again
                self.unused_tiles
                    .retain(|unused_tile| tile.atlas_index != unused_tile.atlas_index);
            }

            tile.requests += 1;
        } else {
            // Todo: implement better loading strategy
            let atlas_index = self.allocate_tile();

            tile_states.insert(
                tile_coordinate,
                TileState {
                    requests: 1,
                    state: LoadingState::Loading,
                    atlas_index,
                },
            );

            trace!("queueing tile load: {tile_coordinate} -> atlas slot {atlas_index}");
            self.to_load.push_back((tile_coordinate, atlas_index));
        }

        self.tile_states = tile_states;
    }

    fn release_tile(&mut self, tile_coordinate: TileCoordinate) {
        // Same gate as `request_tile`: skip if the disk-mode gate is engaged
        // and this coord was never registered.
        if !self.supports_all_tiles && !self.existing_tiles.contains(&tile_coordinate) {
            return;
        }

        let tile = self
            .tile_states
            .get_mut(&tile_coordinate)
            .expect("Tried releasing a tile, which is not present.");
        tile.requests -= 1;

        if tile.requests == 0 {
            // the tile is not used anymore
            self.unused_tiles
                .push_back(AtlasTile::new(tile_coordinate, tile.atlas_index));
        }
    }

    fn get_best_tile(&self, tile_coordinate: TileCoordinate) -> TileTreeEntry {
        let mut best_tile_coordinate = tile_coordinate;

        loop {
            if best_tile_coordinate == TileCoordinate::INVALID
                || best_tile_coordinate.lod == INVALID_LOD
            {
                // highest lod is not loaded
                return TileTreeEntry {
                    atlas_index: INVALID_ATLAS_INDEX,
                    atlas_lod: INVALID_LOD,
                };
            }

            if let Some(atlas_tile) = self.tile_states.get(&best_tile_coordinate) {
                if matches!(atlas_tile.state, LoadingState::Loaded) {
                    // found best loaded tile
                    return TileTreeEntry {
                        atlas_index: atlas_tile.atlas_index,
                        atlas_lod: best_tile_coordinate.lod,
                    };
                }
            }

            best_tile_coordinate = best_tile_coordinate.parent();
        }
    }
}

/// A sparse storage of all terrain attachments, which streams data in and out of memory
/// depending on the decisions of the corresponding [`TileTree`]s.
///
/// A tile is considered present and assigned an [`u32`] as soon as it is
/// requested by any tile_tree. Then the tile atlas will start loading all of its attachments
/// by storing the [`TileCoordinate`] (for one frame) in `load_events` for which
/// attachment-loading-systems can listen.
/// Tiles that are not being used by any tile_tree anymore are cached (LRU),
/// until new atlas indices are required.
///
/// The [`u32`] can be used for accessing the attached data in systems by the CPU
/// and in shaders by the GPU.
///
/// `TileAtlas` doubles as the visibility marker for terrain entities: when added to an
/// entity, the on-add hook pushes its `TypeId` into the entity's `VisibilityClass`, so
/// the standard `check_visibility` system tracks terrains and the queue can find them
/// via `RenderVisibleEntities::iter::<TileAtlas>()`.
#[derive(Component)]
#[require(bevy::camera::visibility::VisibilityClass)]
#[component(on_add = bevy::camera::visibility::add_visibility_class::<TileAtlas>)]
pub struct TileAtlas {
    pub(crate) attachments: Vec<AtlasAttachment>,
    pub(crate) attachment_configs: Vec<AttachmentConfig>,
    // stores the attachment data
    pub(crate) state: TileAtlasState,
    pub(crate) provider: Box<dyn TileProvider>,
    pub(crate) path: String,
    pub(crate) atlas_size: u32,
    pub(crate) lod_count: u32,
    pub(crate) model: TerrainModel,
}

impl TileAtlas {
    /// Creates a new [`TileAtlas`] backed by a [`DiskTileProvider`] reading
    /// preprocessed tiles from `assets/{config.path}/data/`.
    ///
    /// New code should prefer [`TileAtlas::with_provider`], which makes the
    /// data source explicit at the call site. This convenience constructor is
    /// kept so existing users don't need to update call sites when upgrading.
    pub fn new(config: &TerrainConfig) -> Self {
        let provider = Box::new(DiskTileProvider::new(config.path.clone()));
        Self::with_provider(config, provider)
    }

    /// Creates a new [`TileAtlas`] using the given [`TileProvider`] as its
    /// tile data source. This is the recommended constructor — passing the
    /// provider explicitly makes the data source visible at the call site
    /// and avoids the implicit disk dependency.
    pub fn with_provider(config: &TerrainConfig, provider: Box<dyn TileProvider>) -> Self {
        let attachments = config
            .attachments
            .iter()
            .map(|attachment| AtlasAttachment::new(attachment, config.atlas_size, &config.path))
            .collect_vec();

        let existing_tiles = Self::load_tile_config(&config.path);
        let supports_all_tiles = provider.supports_all_tiles();

        let state = TileAtlasState::new(config.atlas_size, existing_tiles, supports_all_tiles);

        Self {
            model: config.model.clone(),
            attachments,
            attachment_configs: config.attachments.clone(),
            state,
            provider,
            path: config.path.to_string(),
            atlas_size: config.atlas_size,
            lod_count: config.lod_count,
        }
    }

    pub fn get_tile(&mut self, tile_coordinate: TileCoordinate) -> AtlasTile {
        self.state.get_tile(tile_coordinate)
    }

    pub fn get_or_allocate_tile(&mut self, tile_coordinate: TileCoordinate) -> AtlasTile {
        self.state.get_or_allocate_tile(tile_coordinate)
    }

    pub fn save(&mut self, tile: AtlasTileAttachment) {
        self.state.to_save.push_back(tile);
    }

    pub(super) fn get_best_tile(&self, tile_coordinate: TileCoordinate) -> TileTreeEntry {
        self.state.get_best_tile(tile_coordinate)
    }

    pub(super) fn sample_attachment(&self, tile_lookup: TileLookup, attachment_index: u32) -> Vec4 {
        self.attachments[attachment_index as usize].sample(tile_lookup)
    }

    /// Updates the tile atlas according to all corresponding tile_trees.
    pub(crate) fn update(
        mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
        mut tile_atlases: Query<&mut TileAtlas>,
    ) {
        for mut tile_atlas in tile_atlases.iter_mut() {
            let TileAtlas {
                state,
                attachments,
                attachment_configs,
                provider,
                model,
                ..
            } = tile_atlas.deref_mut();

            state.update(provider.as_ref(), model, attachment_configs, attachments);

            for attachment in attachments {
                attachment.update(state);
            }
        }

        for (&(terrain, _view), tile_tree) in tile_trees.iter_mut() {
            let mut tile_atlas = tile_atlases.get_mut(terrain).unwrap();

            for tile_coordinate in tile_tree.released_tiles.drain(..) {
                tile_atlas.state.release_tile(tile_coordinate);
            }

            for tile_coordinate in tile_tree.requested_tiles.drain(..) {
                tile_atlas.state.request_tile(tile_coordinate);
            }
        }
    }

    /// Saves the tile configuration of the terrain, which stores the [`TileCoordinate`]s of all the tiles
    /// of the terrain.
    pub(crate) fn save_tile_config(&self) {
        let tc = TC {
            tiles: self.state.existing_tiles.iter().copied().collect_vec(),
        };

        tc.save_file(format!("assets/{}/config.tc", &self.path))
            .unwrap();
    }

    /// Loads the tile configuration of the terrain, which stores the [`TileCoordinate`]s of all the tiles
    /// of the terrain.
    pub(crate) fn load_tile_config(path: &str) -> HashSet<TileCoordinate> {
        match TC::load_file(format!("assets/{}/config.tc", path)) {
            Ok(tc) => tc.tiles.into_iter().collect(),
            // Missing `config.tc` is the normal case for runtime-synthesised
            // providers (e.g. `PipelineTileProvider`); the disk path simply
            // isn't in use. Log at `debug` so it's discoverable without
            // spamming stdout on every terrain creation.
            Err(_) => {
                debug!("no preprocessed tile config at assets/{path}/config.tc; \
                        using empty existing-tile set (expected for synthesised providers)");
                HashSet::default()
            }
        }
    }
}
