//! Runtime-provider-first UDLOD terrain rendering for Thalos.
//!
//! # Status: LEGACY — end-of-life
//!
//! Thalos's default ground renderer is `thalos_body_render::tiles` (terrain as
//! ordinary `Mesh` + `StandardMaterial` on Bevy's standard path, keystone
//! ADR-20260723T142945Z). This crate and its WGSL stack are kept as the A/B
//! baseline (`THALOS_TILE_RENDERER=0`) and for bodies the tile driver has not
//! installed on yet. **Defect-driven fixes only** — do not build new terrain
//! features here; the crate is deleted once the remaining `ntr §6` rows close.
//!
//! This crate is an in-tree fork of Kurt Kühnert's `bevy_terrain`.
//! Upstream's offline raster preprocessing path has been removed: Thalos
//! streams sparse terrain tiles from runtime [`TileProvider`](prelude::TileProvider)
//! implementations, which may synthesize data, read a Thalos cache, or later
//! enqueue GPU jobs that write directly into atlas slots.
//!
//! The crate still owns the terrain-rendering machinery: chunked tile atlas
//! storage, tile-tree residency, parent-LOD fallback, CPU-balanced draw tile
//! selection, UDLOD mesh generation, attachment sampling, material integration,
//! and the Taylor-series precision path used at planet scale.

pub mod big_space;
pub mod debug;
pub mod math;
pub mod plugin;
pub mod render;
pub mod shaders;
pub mod terrain;
pub mod terrain_data;
pub mod terrain_view;
pub mod util;

pub mod prelude {
    //! `use thalos_udlod::prelude::*;` to import common components, bundles, and plugins.
    // #[doc(hidden)]

    pub use crate::big_space::{BigSpaceCommands, PreciseRotation, ReferenceFrame};

    pub use crate::{
        debug::{
            camera::{DebugCameraBundle, DebugCameraController},
            DebugTerrainMaterial, LoadingImages, TerrainDebugPlugin,
        },
        math::TerrainModel,
        plugin::TerrainPlugin,
        render::terrain_material::TerrainMaterialPlugin,
        terrain::{TerrainBundle, TerrainConfig},
        terrain_data::{
            prune_tile_cache, static_namespace,
            tile_atlas::TileAtlas,
            tile_tree::{TerrainStreamingPaused, TileTree},
            AttachmentConfig, AttachmentFormat, DiskTileCacheProvider, MemoryTileCacheProvider,
            NamespaceFn, SharedTileCache, TileProvider,
        },
        terrain_view::{TerrainViewComponents, TerrainViewConfig},
    };
}
