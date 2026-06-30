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
mod ground_patch;
mod height_source;
mod landcover;
mod pipeline;
#[cfg(feature = "playground")]
mod playground_material;
mod rendered_height;
mod rock_material;
mod rock_mesh;
mod scatter;
mod sky_material;
mod synthetic;
mod tile_lattice;
mod tile_synthesis_pool;
mod tree_atlas;
mod tree_impostor;
mod tree_material;
mod tree_mesh;
mod vegetation;
mod water_material;

pub use body_material::{
    BodySkyExtra, BodyTerrainDebug, BodyTerrainExtras, BodyTerrainMaterial, BodyTerrainShadow,
    CASCADE_COUNT, MAX_TERRAIN_SHADOW_CASTERS, MAX_TERRAIN_SHADOW_QUADS, ShadowCascadeBlock,
    TerrainShadingStyle,
};
pub use ground_patch::{GroundPatchMaterial, GroundPatchMaterialPlugin};
pub use height_source::{
    ConstantHeightSource, CpuPipelineHeightSource, GpuAtlasHeightMirror,
    GpuAtlasHeightMirrorComponent, GpuAtlasMirrorHandle, GpuAtlasMirrorHeightSource, HeightSource,
};
pub use landcover::{LandcoverSample, sample_landcover};
pub use pipeline::{
    PipelineTileProvider, rendered_height_m, rendered_height_range, renderer_tile_lod_m_at,
};
#[cfg(feature = "playground")]
pub use playground_material::PlaygroundMaterial;
pub use rendered_height::{
    TerrainPatchBasis, TerrainPatchConfig, TerrainPatchMesh, build_rendered_terrain_patch,
    build_rendered_terrain_patch_from_source,
};
pub use rock_material::{RockMaterial, RockMaterialPlugin};
pub use rock_mesh::{RockMeshData, RockMeshParams, build_rock_mesh, build_rock_mesh_data};
pub use scatter::{
    PlacementSample, ScatterClass, ScatterRegion, ScatterTreatment, VegInstance, VegLayer,
    VegScatterInput, VegScatterTile, VegSpeciesPlacement, build_scatter_tile, classify_scatter,
    clump_field, combine_impostor_tile_mesh, combine_rock_tile_mesh, combine_tree_tile_mesh,
    placement_gate,
};
pub use sky_material::BodySkyMaterial;
pub use synthetic::{SyntheticTerrainMode, SyntheticTileProvider};
pub use tile_lattice::{TileKey, TileLattice, cube_dir, cube_face_uv, tiles_per_side};
pub use tile_synthesis_pool::tile_synthesis_pool;
pub use tree_atlas::{
    ATLAS_N, BARK_CELL_COUNT, BARK_CELL_FIRST, build_foliage_atlas, build_foliage_material_atlas,
};
pub use tree_impostor::{
    BakeParams, IMPOSTOR_MAX_SPECIES, ImpostorAtlasLayout, ImpostorParams, TreeBakeMaterial,
    TreeImpostorMaterial, TreeImpostorMaterialPlugin, hemioct_decode, impostor_bake_rotation,
    make_impostor_atlas, recenter_tree_mesh, tree_bounding_sphere,
};
pub use tree_material::{TreeMaterial, TreeMaterialPlugin, fallback_shadow_map};
pub use tree_mesh::{
    CanopyStyle, TreeMeshData, TreeMeshParams, build_tree_mesh, build_tree_mesh_data,
};
pub use vegetation::{
    GRASS_TILE_SIZE_M, GrassBladeLod, GrassClumpParams, GrassFieldParams, GrassMaterial,
    GrassMaterialPlugin, GrassParams, GrassProfile, GrassTileBuildInput, GrassTileKey,
    GrassTileMesh, build_grass_clump_mesh, build_grass_field_mesh, build_grass_tile_mesh,
    grass_tile_frame, grass_tile_key, grass_tiles_per_side,
};
pub use water_material::{BodyWaterMaterial, BodyWaterParams};

pub struct ThalosTerrainPlugin;

impl Plugin for ThalosTerrainPlugin {
    fn build(&self, app: &mut App) {
        // `thalos::lighting` and `thalos::atmosphere` shader libraries live in
        // the shared `thalos_planet_lighting` crate. Add its plugin defensively
        // so terrain works in apps that don't also load
        // `PlanetRenderingPlugin` (e.g. headless examples).
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
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
        // Grass blades + scattered trees/shrubs render through the regular
        // forward pipeline too — decoration meshes, not UDLOD geometry.
        // Grass is in the depth prepass (it ships a matching `grass_prepass.wgsl`,
        // so the main color pass early-Z-rejects the heavy blade overdraw). Trees
        // and rocks opt OUT via `Material::enable_prepass` (their vertex displaces —
        // wind / per-tree + per-rock scale-fade — and they have no matching prepass
        // shader yet, so a standard rest-pose prepass would mismatch the main depth
        // and flicker). Impostors are already prepass-safe (degenerate POSITION →
        // standard prepass draws nothing). Adding tree/rock prepass shaders later
        // brings them into the early-Z too.
        app.add_plugins(MaterialPlugin::<GrassMaterial>::default());
        app.add_plugins(MaterialPlugin::<TreeMaterial>::default());
        app.add_plugins(MaterialPlugin::<RockMaterial>::default());
        // Far-band tree impostors render through the forward pipeline too; the
        // bake material is rendered only by the startup off-screen bake cameras.
        app.add_plugins(MaterialPlugin::<TreeImpostorMaterial>::default());
        app.add_plugins(MaterialPlugin::<TreeBakeMaterial>::default());
        body_material::embed_body_terrain_shader(app);
        sky_material::embed_body_sky_shader(app);
        water_material::embed_body_water_shader(app);
        vegetation::embed_grass_shader(app);
        tree_material::embed_tree_shader(app);
        rock_material::embed_rock_shader(app);
        tree_impostor::embed_tree_impostor_shaders(app);
        #[cfg(feature = "playground")]
        playground_material::embed_playground_shader(app);
    }
}
