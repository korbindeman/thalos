//! Thalos integration of the forked `thalos_udlod` UDLOD renderer.
//!
//! M3 staging ([docs/terrain.md](../../docs/terrain.md), "Rendering: ground
//! LOD"):
//!
//! - **Stage 1**: wire the fork into the workspace + run a deterministic
//!   [`SyntheticTileProvider`] end-to-end via `examples/playground.rs`.
//! - **Stage 2**: [`PipelineTileProvider`] forwards tile requests through
//!   the `thalos_terrain::query` seam (`surface_sample` / `surface_height_m`),
//!   so the same geometric surface drives the UDLOD mesh, terrain collider,
//!   height sources, PNG dumps, and impostor bake substrate.
//! - **Stage 3**: per-body wiring lives in `thalos_game`'s
//!   `rendering::ground_terrain` module.
//!
//! ## Plugin layering
//!
//! [`ThalosTerrainPlugin`] adds [`thalos_udlod::prelude::TerrainPlugin`] +
//! [`bevy::pbr::MaterialPlugin<BodyTerrainMaterial>`] and embeds the body
//! terrain shader. `thalos_udlod::TerrainPlugin` in turn adds
//! [`big_space::prelude::BigSpaceDefaultPlugins`] unconditionally, so
//! consumers that already register `BigSpaceDefaultPlugins` directly must
//! drop that registration to avoid the duplicate-plugin panic.

use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;
use thalos_udlod::prelude::{TerrainMaterialPlugin, TerrainPlugin};

mod body_material;
mod height_source;
mod pipeline;
#[cfg(feature = "playground")]
mod playground_material;
mod rendered_height;
mod sky_material;
mod synthetic;
mod tile_synthesis_pool;
mod water_material;

pub use body_material::{
    BodySkyExtra, BodyTerrainDebug, BodyTerrainExtras, BodyTerrainMaterial, BodyTerrainShadow,
    MAX_TERRAIN_SHADOW_CASTERS,
};
pub use height_source::{
    ConstantHeightSource, CpuPipelineHeightSource, GpuAtlasHeightMirror,
    GpuAtlasHeightMirrorComponent, GpuAtlasMirrorHandle, GpuAtlasMirrorHeightSource, HeightSource,
};
pub use pipeline::{
    PipelineTileProvider, rendered_height_m, rendered_height_range, renderer_tile_lod_m_at,
};
#[cfg(feature = "playground")]
pub use playground_material::PlaygroundMaterial;
pub use rendered_height::{
    TerrainPatchBasis, TerrainPatchConfig, TerrainPatchMesh, build_rendered_terrain_patch,
    build_rendered_terrain_patch_from_source,
};
pub use sky_material::BodySkyMaterial;
pub use synthetic::{SyntheticTerrainMode, SyntheticTileProvider};
pub use tile_synthesis_pool::tile_synthesis_pool;
pub use water_material::{BodyWaterMaterial, BodyWaterParams};

pub struct ThalosTerrainPlugin;

impl Plugin for ThalosTerrainPlugin {
    fn build(&self, app: &mut App) {
        // `thalos::lighting` and `thalos::atmosphere` shader libraries live in
        // the shared `thalos_planet_lighting` crate. Add its plugin defensively
        // so terrain works in apps that don't also load
        // `PlanetRenderingPlugin` (e.g. headless examples).
        if !app.is_plugin_added::<thalos_planet_lighting::PlanetLightingPlugin>() {
            app.add_plugins(thalos_planet_lighting::PlanetLightingPlugin);
        }
        app.add_plugins(TerrainPlugin);
        app.add_systems(
            Last,
            height_source::sync_gpu_atlas_height_mirrors
                .after(thalos_udlod::prelude::TileAtlas::update),
        );
        app.add_plugins(TerrainMaterialPlugin::<BodyTerrainMaterial>::default());
        // Sky and water both use the standard Bevy MaterialPlugin — they
        // render through the regular forward pipeline (fullscreen quad and
        // icosphere mesh respectively), not thalos_udlod's UDLOD pipeline.
        app.add_plugins(MaterialPlugin::<BodySkyMaterial>::default());
        app.add_plugins(MaterialPlugin::<BodyWaterMaterial>::default());
        body_material::embed_body_terrain_shader(app);
        sky_material::embed_body_sky_shader(app);
        water_material::embed_body_water_shader(app);
        #[cfg(feature = "playground")]
        playground_material::embed_playground_shader(app);
    }
}
