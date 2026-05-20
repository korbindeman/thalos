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
use std::{collections::VecDeque, fs, ops::DerefMut};

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
    revisions: Vec<u64>,
    next_revision: u64,

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
            revisions: vec![0; tile_atlas_size as usize],
            next_revision: 1,
            saving_tiles: default(),
            uploading_tiles: default(),
            downloading_tiles: default(),
        }
    }

    fn set_data(&mut self, atlas_index: u32, data: AttachmentData) {
        let index = atlas_index as usize;
        self.data[index] = data;
        self.revisions[index] = self.next_revision;
        self.next_revision = self.next_revision.wrapping_add(1).max(1);
    }

    fn update(&mut self, atlas_state: &mut TileAtlasState) {
        let mut downloaded_tiles = Vec::new();
        self.downloading_tiles.retain_mut(|tile| {
            future::block_on(future::poll_once(tile)).is_none_or(|tile| {
                downloaded_tiles.push(tile);
                false
            })
        });
        for tile in downloaded_tiles {
            atlas_state.downloaded_tile_attachment(tile.tile);
            self.set_data(tile.tile.atlas_index, tile.data);
        }

        self.saving_tiles.retain_mut(|task| {
            future::block_on(future::poll_once(task)).is_none_or(|tile| {
                atlas_state.saved_tile_attachment(tile);
                false
            })
        });
    }

    fn save(&mut self, tile: AtlasTileAttachment) {
        self.saving_tiles.push(
            AtlasTileAttachmentWithData {
                tile,
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
    /// Monotonic ownership token for `atlas_index` when this tile acquired
    /// the slot. Async tile work captures this value and may only write its
    /// result if the coordinate still owns the same slot generation.
    generation: u64,
    /// The count of [`TileTrees`] that have requested this tile.
    requests: u32,
}

/// A queued tile load that has not yet been handed to the provider.
struct QueuedTileLoad {
    coord: TileCoordinate,
    atlas_index: u32,
    generation: u64,
}

/// One in-flight per-tile load issued via the [`TileProvider`].
pub(crate) struct LoadingTile {
    coord: TileCoordinate,
    atlas_index: u32,
    generation: u64,
    task: Task<Result<Vec<AttachmentData>>>,
}

pub(crate) struct TileAtlasState {
    tile_states: HashMap<TileCoordinate, TileState>,
    unused_tiles: VecDeque<AtlasTile>,
    /// Monotonic ownership token per atlas slot. Incremented every time a
    /// slot is allocated to a new coordinate so stale async loads can be
    /// discarded instead of overwriting the new owner's texture data.
    slot_generations: Vec<u64>,
    pub(crate) existing_tiles: HashSet<TileCoordinate>,
    /// When `true`, the [`TileProvider`] can synthesise data for any
    /// [`TileCoordinate`]; tile requests bypass the `existing_tiles` gate
    /// (which exists to keep disk providers from chasing missing `.bin`/`.png`
    /// files). Set once at construction from
    /// [`TileProvider::supports_all_tiles`].
    supports_all_tiles: bool,

    /// Tiles pinned by [`TileAtlas::pin_tile`]: each pin contributes one
    /// extra refcount that the atlas never releases for the lifetime of the
    /// atlas. Used to keep `LOD 0` (and any other ancestors the renderer
    /// needs as a fallback) permanently resident so `get_best_tile` can
    /// always walk up to a loaded ancestor.
    pinned_tiles: Vec<TileCoordinate>,

    to_load: VecDeque<QueuedTileLoad>,
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
            slot_generations: vec![0; atlas_size as usize],
            existing_tiles,
            supports_all_tiles,
            pinned_tiles: default(),
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

    fn pin_tile(&mut self, tile_coordinate: TileCoordinate) {
        if tile_coordinate == TileCoordinate::INVALID {
            return;
        }
        self.request_tile(tile_coordinate);
        self.pinned_tiles.push(tile_coordinate);
    }

    fn pinned_tiles_ready(&self) -> bool {
        self.pinned_tiles.iter().all(|coord| {
            self.tile_states
                .get(coord)
                .map(|t| matches!(t.state, LoadingState::Loaded))
                .unwrap_or(false)
        })
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
            let Some(queued) = self.to_load.pop_front() else {
                break;
            };

            if !self.tile_load_is_current(queued.coord, queued.atlas_index, queued.generation) {
                trace!(
                    "discarding stale queued tile load: {} -> atlas slot {} gen {}",
                    queued.coord,
                    queued.atlas_index,
                    queued.generation
                );
                continue;
            }

            let task = provider.request_tile(queued.coord, model, attachment_configs);
            self.loading_tiles.push(LoadingTile {
                coord: queued.coord,
                atlas_index: queued.atlas_index,
                generation: queued.generation,
                task,
            });
            self.load_slots -= 1;
        }

        let mut completed = Vec::new();
        self.loading_tiles.retain_mut(|loading| {
            future::block_on(future::poll_once(&mut loading.task)).is_none_or(|result| {
                completed.push((
                    loading.coord,
                    loading.atlas_index,
                    loading.generation,
                    result,
                ));
                false
            })
        });

        for (coord, atlas_index, generation, result) in completed {
            self.load_slots += 1;

            if !self.tile_load_is_current(coord, atlas_index, generation) {
                trace!(
                    "discarding stale completed tile load: {coord} -> atlas slot {atlas_index} gen {generation}"
                );
                continue;
            }

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
                        attachment
                            .uploading_tiles
                            .push(AtlasTileAttachmentWithData {
                                tile: AtlasTileAttachment {
                                    coordinate: coord,
                                    atlas_index,
                                    attachment_index: attachment_index as u32,
                                },
                                data: data.clone(),
                                texture_size: attachment.texture_size,
                            });
                        attachment.set_data(atlas_index, data);
                    }
                    trace!("loaded tile {coord} into atlas slot {atlas_index} gen {generation}");
                    self.loaded_tile(coord, atlas_index, generation);
                }
                Err(e) => {
                    // Tile remains in `Loading` state; renderer continues to fall
                    // back to the parent LOD. Slot was returned above.
                    warn!("tile load failed for {coord}: {e}");
                }
            }
        }
    }

    fn tile_load_is_current(
        &self,
        coord: TileCoordinate,
        atlas_index: u32,
        generation: u64,
    ) -> bool {
        self.tile_states.get(&coord).is_some_and(|state| {
            matches!(state.state, LoadingState::Loading)
                && state.atlas_index == atlas_index
                && state.generation == generation
        })
    }

    fn loaded_tile(&mut self, coord: TileCoordinate, atlas_index: u32, generation: u64) {
        if let Some(tile_state) = self.tile_states.get_mut(&coord) {
            if tile_state.atlas_index == atlas_index && tile_state.generation == generation {
                tile_state.state = LoadingState::Loaded;
            }
        }
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

        let atlas_index = self
            .tile_states
            .get(&tile_coordinate)
            .map(|tile| tile.atlas_index)
            .unwrap_or(INVALID_ATLAS_INDEX);

        AtlasTile::new(tile_coordinate, atlas_index)
    }

    fn allocate_tile(&mut self) -> (u32, u64) {
        let unused_tile = self.unused_tiles.pop_front().expect("Atlas out of indices");

        self.tile_states.remove(&unused_tile.coordinate);

        let generation = &mut self.slot_generations[unused_tile.atlas_index as usize];
        *generation = generation.wrapping_add(1).max(1);

        (unused_tile.atlas_index, *generation)
    }

    fn get_or_allocate_tile(&mut self, tile_coordinate: TileCoordinate) -> AtlasTile {
        if tile_coordinate == TileCoordinate::INVALID {
            return AtlasTile::new(TileCoordinate::INVALID, INVALID_ATLAS_INDEX);
        }

        self.existing_tiles.insert(tile_coordinate);

        let atlas_index = if let Some(tile) = self.tile_states.get(&tile_coordinate) {
            tile.atlas_index
        } else {
            let (atlas_index, generation) = self.allocate_tile();

            self.tile_states.insert(
                tile_coordinate,
                TileState {
                    requests: 1,
                    state: LoadingState::Loaded,
                    atlas_index,
                    generation,
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

        // check if the tile is already present else start loading it
        if let Some(tile) = self.tile_states.get_mut(&tile_coordinate) {
            if tile.requests == 0 {
                // the tile is now used again
                self.unused_tiles
                    .retain(|unused_tile| tile.atlas_index != unused_tile.atlas_index);
            }

            tile.requests += 1;
            return;
        }

        // Todo: implement better loading strategy
        let (atlas_index, generation) = self.allocate_tile();

        self.tile_states.insert(
            tile_coordinate,
            TileState {
                requests: 1,
                state: LoadingState::Loading,
                atlas_index,
                generation,
            },
        );

        trace!(
            "queueing tile load: {tile_coordinate} -> atlas slot {atlas_index} gen {generation}"
        );
        self.to_load.push_back(QueuedTileLoad {
            coord: tile_coordinate,
            atlas_index,
            generation,
        });
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

    fn loaded_tiles(&self) -> Vec<(TileCoordinate, u32)> {
        self.tile_states
            .iter()
            .filter_map(|(coordinate, state)| {
                (state.requests > 0 && matches!(state.state, LoadingState::Loaded))
                    .then_some((*coordinate, state.atlas_index))
            })
            .collect()
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

    /// The terrain model this atlas was built for. Exposed so callers can
    /// translate world positions onto the cube-sphere when querying tile
    /// state via `TileTree::best_resident_atlas_lod`.
    pub fn model(&self) -> &TerrainModel {
        &self.model
    }

    /// Maximum LOD depth tracked by this atlas. The deepest tracked LOD is
    /// `lod_count() - 1`.
    pub fn lod_count(&self) -> u32 {
        self.lod_count
    }

    /// Attachment configurations as declared when the atlas was built.
    /// Needed to derive a tile's metres-per-texel from its LOD without
    /// hard-coding texture/border sizes at the call site.
    pub fn attachment_configs(&self) -> &[AttachmentConfig] {
        &self.attachment_configs
    }

    /// Find an attachment by name in this atlas.
    pub fn attachment_index(&self, name: &str) -> Option<u32> {
        self.attachment_configs
            .iter()
            .position(|config| config.name == name)
            .map(|index| index as u32)
    }

    /// All atlas tiles whose attachment data is loaded and usable.
    pub fn loaded_tiles(&self) -> Vec<(TileCoordinate, u32)> {
        self.state.loaded_tiles()
    }

    /// CPU-side attachment data for one loaded atlas slot.
    pub fn attachment_data(
        &self,
        attachment_index: u32,
        atlas_index: u32,
    ) -> Option<&AttachmentData> {
        self.attachments
            .get(attachment_index as usize)?
            .data
            .get(atlas_index as usize)
    }

    /// Monotonic per-slot revision for attachment data. Consumers can use
    /// this to mirror only slots that changed after a tile load or GPU
    /// readback completion.
    pub fn attachment_slot_revision(&self, attachment_index: u32, atlas_index: u32) -> Option<u64> {
        self.attachments
            .get(attachment_index as usize)?
            .revisions
            .get(atlas_index as usize)
            .copied()
    }

    /// Permanently keeps `tile_coordinate` resident for the lifetime of the
    /// atlas. Each call adds one refcount that is never released, so the
    /// tile stays loaded even when no [`TileTree`] currently requests it.
    ///
    /// Used at terrain spawn to pin the root LODs (typically `LOD 0` for
    /// every face). With those pinned, [`TileAtlasState::get_best_tile`] can
    /// always walk `parent()` up to a resident ancestor and the renderer
    /// never samples an `INVALID_ATLAS_INDEX` slot (the GPU texture-array
    /// OOB sample that decodes to `min_height`, leaving the mesh below the
    /// ellipsoid floor as a visible hole).
    ///
    /// Calling multiple times for the same coordinate is allowed; each call
    /// adds one extra refcount. The pin is dropped when the [`TileAtlas`]
    /// itself is dropped.
    pub fn pin_tile(&mut self, tile_coordinate: TileCoordinate) {
        self.state.pin_tile(tile_coordinate);
    }

    /// Returns `true` once every tile pinned via [`Self::pin_tile`] has
    /// finished loading into the atlas. Until this returns `true`, the
    /// renderer cannot guarantee a resident-ancestor fallback for arbitrary
    /// tile coordinates; gate visibility on this so terrain doesn't appear
    /// before its root LODs are ready.
    pub fn pinned_tiles_ready(&self) -> bool {
        self.state.pinned_tiles_ready()
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
    pub fn update(
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
                debug!(
                    "no preprocessed tile config at assets/{path}/config.tc; \
                        using empty existing-tile set (expected for synthesised providers)"
                );
                HashSet::default()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_state(atlas_size: u32) -> TileAtlasState {
        TileAtlasState::new(atlas_size, HashSet::default(), true)
    }

    #[test]
    fn reusing_slot_removes_previous_coordinate_owner() {
        let mut state = synthetic_state(1);
        let first = TileCoordinate::new(0, 2, 1, 1);
        let second = TileCoordinate::new(0, 2, 2, 2);

        state.request_tile(first);
        state.release_tile(first);
        state.request_tile(second);

        assert!(state.tile_states.get(&first).is_none());
        assert_eq!(state.tile_states.get(&second).unwrap().atlas_index, 0);
    }

    #[test]
    fn stale_queued_load_is_not_current_after_slot_reuse() {
        let mut state = synthetic_state(1);
        let first = TileCoordinate::new(0, 3, 3, 3);
        let second = TileCoordinate::new(0, 3, 4, 4);

        state.request_tile(first);
        let queued = state.to_load.front().unwrap();
        let stale_atlas_index = queued.atlas_index;
        let stale_generation = queued.generation;

        state.release_tile(first);
        state.request_tile(second);

        assert!(!state.tile_load_is_current(first, stale_atlas_index, stale_generation));
        assert!(state.tile_load_is_current(second, stale_atlas_index, stale_generation + 1));
    }
}
