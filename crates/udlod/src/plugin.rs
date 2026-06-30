use crate::{
    math::{generate_terrain_model_approximation, TerrainModelApproximation},
    render::{
        culling_bind_group::CullingBindGroup,
        terrain_bind_group::TerrainData,
        terrain_view_bind_group::TerrainViewData,
        tiling_prepass::{
            init_tiling_prepass_pipelines, queue_tiling_prepass, TilingPrepassItem,
            TilingPrepassPipelines,
        },
    },
    shaders::{load_terrain_shaders, InternalShaders},
    terrain::TerrainComponents,
    terrain_data::{
        gpu_tile_atlas::GpuTileAtlas, gpu_tile_tree::GpuTileTree, tile_atlas::TileAtlas,
        tile_tree::TileTree,
    },
    terrain_view::TerrainViewComponents,
};
use bevy::{
    prelude::*,
    render::{render_resource::*, Render, RenderApp, RenderStartup, RenderSystems},
};

/// The plugin for the terrain renderer.
pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(crate::big_space::BigSpacePlugin);

        app.init_resource::<InternalShaders>()
            .init_resource::<TerrainViewComponents<TileTree>>()
            .init_resource::<TerrainViewComponents<TerrainModelApproximation>>()
            .add_systems(
                Last,
                (
                    TileTree::compute_requests,
                    TileAtlas::update,
                    TileTree::adjust_to_tile_atlas,
                    TileTree::approximate_height,
                    TileTree::compute_draw_set,
                    generate_terrain_model_approximation,
                )
                    .chain(),
            );

        app.sub_app_mut(RenderApp)
            .init_resource::<TerrainComponents<GpuTileAtlas>>()
            .init_resource::<TerrainComponents<TerrainData>>()
            .init_resource::<TerrainViewComponents<GpuTileTree>>()
            .init_resource::<TerrainViewComponents<TerrainViewData>>()
            .init_resource::<TerrainViewComponents<CullingBindGroup>>()
            .init_resource::<TerrainViewComponents<TilingPrepassItem>>()
            .init_resource::<SpecializedComputePipelines<TilingPrepassPipelines>>()
            .add_systems(RenderStartup, init_tiling_prepass_pipelines)
            .add_systems(
                ExtractSchedule,
                (
                    GpuTileAtlas::initialize,
                    GpuTileAtlas::extract.after(GpuTileAtlas::initialize),
                    GpuTileTree::initialize,
                    GpuTileTree::extract.after(GpuTileTree::initialize),
                    TerrainData::initialize.after(GpuTileAtlas::initialize),
                    TerrainData::extract.after(TerrainData::initialize),
                    TerrainViewData::initialize.after(GpuTileTree::initialize),
                    TerrainViewData::extract.after(TerrainViewData::initialize),
                ),
            )
            .add_systems(
                Render,
                (
                    (
                        GpuTileTree::prepare,
                        GpuTileAtlas::prepare,
                        TerrainData::prepare,
                        TerrainViewData::prepare,
                        CullingBindGroup::prepare,
                    )
                        .in_set(RenderSystems::Prepare),
                    queue_tiling_prepass.in_set(RenderSystems::Queue),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        load_terrain_shaders(app);

        // Bevy 0.19 replaced the node-based render graph with render-pass
        // systems. The former `TilingPrepassNode` was already a no-op (the GPU
        // tiling prepass was superseded by the CPU draw-set computation in
        // `TileTree::compute_draw_set`), so it is not ported — there is no GPU
        // work to schedule. The compute pipelines / bind groups remain wired
        // (`queue_tiling_prepass`) so the dispatch path can be reinstated as a
        // `Core3d`-schedule system if we ever bisect back to it.
    }
}
