use crate::terrain_data::gpu_tile_tree::GpuTileTree;
use crate::{
    debug::DebugTerrain,
    render::{
        culling_bind_group::{culling_layout_descriptor, CullingBindGroup},
        terrain_bind_group::{terrain_layout_descriptor, TerrainData},
        terrain_view_bind_group::{
            prepare_indirect_layout_descriptor, refine_tiles_layout_descriptor, TerrainViewData,
        },
    },
    shaders::{PREPARE_PREPASS_SHADER, REFINE_TILES_SHADER},
    terrain::TerrainComponents,
    terrain_data::gpu_tile_atlas::GpuTileAtlas,
    terrain_view::TerrainViewComponents,
};
use bevy::{
    prelude::*,
    render::{
        render_graph::{self, RenderLabel},
        render_resource::*,
        renderer::RenderContext,
    },
    shader::ShaderDefVal,
};

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct TilingPrepassLabel;

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct TilingPrepassPipelineKey: u32 {
        const NONE           = 0;
        const REFINE_TILES   = 1 << 0;
        const PREPARE_ROOT   = 1 << 1;
        const PREPARE_NEXT   = 1 << 2;
        const PREPARE_RENDER = 1 << 3;
        const SPHERICAL      = 1 << 4;
        const TEST1          = 1 << 5;
        const TEST2          = 1 << 6;
        const TEST3          = 1 << 7;
    }
}

impl TilingPrepassPipelineKey {
    pub fn from_debug(debug: &DebugTerrain) -> Self {
        let mut key = TilingPrepassPipelineKey::NONE;

        if debug.test1 {
            key |= TilingPrepassPipelineKey::TEST1;
        }
        if debug.test2 {
            key |= TilingPrepassPipelineKey::TEST2;
        }
        if debug.test3 {
            key |= TilingPrepassPipelineKey::TEST3;
        }

        key
    }

    pub fn shader_defs(&self) -> Vec<ShaderDefVal> {
        let mut shader_defs = Vec::new();

        if self.contains(TilingPrepassPipelineKey::SPHERICAL) {
            shader_defs.push("SPHERICAL".into());
        }
        if self.contains(TilingPrepassPipelineKey::TEST1) {
            shader_defs.push("TEST1".into());
        }
        if self.contains(TilingPrepassPipelineKey::TEST2) {
            shader_defs.push("TEST2".into());
        }
        if self.contains(TilingPrepassPipelineKey::TEST3) {
            shader_defs.push("TEST3".into());
        }

        shader_defs
    }
}

pub(crate) struct TilingPrepassItem {
    refine_tiles_pipeline: CachedComputePipelineId,
    prepare_root_pipeline: CachedComputePipelineId,
    prepare_next_pipeline: CachedComputePipelineId,
    prepare_render_pipeline: CachedComputePipelineId,
}

impl TilingPrepassItem {
    fn pipelines<'a>(
        &'a self,
        pipeline_cache: &'a PipelineCache,
    ) -> Option<(
        &'a ComputePipeline,
        &'a ComputePipeline,
        &'a ComputePipeline,
        &'a ComputePipeline,
    )> {
        Some((
            pipeline_cache.get_compute_pipeline(self.refine_tiles_pipeline)?,
            pipeline_cache.get_compute_pipeline(self.prepare_root_pipeline)?,
            pipeline_cache.get_compute_pipeline(self.prepare_next_pipeline)?,
            pipeline_cache.get_compute_pipeline(self.prepare_render_pipeline)?,
        ))
    }
}

#[derive(Resource)]
pub struct TilingPrepassPipelines {
    pub(crate) prepare_indirect_layout: BindGroupLayoutDescriptor,
    pub(crate) refine_tiles_layout: BindGroupLayoutDescriptor,
    culling_data_layout: BindGroupLayoutDescriptor,
    terrain_layout: BindGroupLayoutDescriptor,
    prepare_prepass_shader: Handle<Shader>,
    refine_tiles_shader: Handle<Shader>,
}

/// Initializes the [`TilingPrepassPipelines`] resource. Runs as a [`RenderStartup`] system
/// (consistent with the rest of `bevy_pbr`'s 0.18+ pipeline initialization) instead of the older
/// [`Plugin::finish`] + [`FromWorld`] pattern.
pub fn init_tiling_prepass_pipelines(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(TilingPrepassPipelines {
        prepare_indirect_layout: prepare_indirect_layout_descriptor(),
        refine_tiles_layout: refine_tiles_layout_descriptor(),
        culling_data_layout: culling_layout_descriptor(),
        terrain_layout: terrain_layout_descriptor(),
        prepare_prepass_shader: asset_server.load(PREPARE_PREPASS_SHADER),
        refine_tiles_shader: asset_server.load(REFINE_TILES_SHADER),
    });
}

impl SpecializedComputePipeline for TilingPrepassPipelines {
    type Key = TilingPrepassPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut layout: Vec<BindGroupLayoutDescriptor> = default();
        let mut shader: Handle<Shader> = default();
        let mut entry_point: Option<std::borrow::Cow<'static, str>> = None;

        let shader_defs = key.shader_defs();

        if key.contains(TilingPrepassPipelineKey::REFINE_TILES) {
            layout = vec![
                self.culling_data_layout.clone(),
                self.terrain_layout.clone(),
                self.refine_tiles_layout.clone(),
            ];
            shader = self.refine_tiles_shader.clone();
            entry_point = Some("refine_tiles".into());
        }
        if key.contains(TilingPrepassPipelineKey::PREPARE_ROOT) {
            layout = vec![
                self.culling_data_layout.clone(),
                self.terrain_layout.clone(),
                self.refine_tiles_layout.clone(),
                self.prepare_indirect_layout.clone(),
            ];
            shader = self.prepare_prepass_shader.clone();
            entry_point = Some("prepare_root".into());
        }
        if key.contains(TilingPrepassPipelineKey::PREPARE_NEXT) {
            layout = vec![
                self.culling_data_layout.clone(),
                self.terrain_layout.clone(),
                self.refine_tiles_layout.clone(),
                self.prepare_indirect_layout.clone(),
            ];
            shader = self.prepare_prepass_shader.clone();
            entry_point = Some("prepare_next".into());
        }
        if key.contains(TilingPrepassPipelineKey::PREPARE_RENDER) {
            layout = vec![
                self.culling_data_layout.clone(),
                self.terrain_layout.clone(),
                self.refine_tiles_layout.clone(),
                self.prepare_indirect_layout.clone(),
            ];
            shader = self.prepare_prepass_shader.clone();
            entry_point = Some("prepare_render".into());
        }

        ComputePipelineDescriptor {
            label: Some("tiling_prepass_pipeline".into()),
            layout,
            push_constant_ranges: default(),
            shader,
            shader_defs,
            entry_point,
            zero_initialize_workgroup_memory: false,
        }
    }
}

pub struct TilingPrepassNode;

impl render_graph::Node for TilingPrepassNode {
    fn run<'w>(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        _context: &mut RenderContext<'w>,
        _world: &'w World,
    ) -> Result<(), render_graph::NodeRunError> {
        // The GPU compute prepass (`prepare_root` → `refine_tiles` ×N →
        // `prepare_render`) is no longer dispatched. The draw set is now
        // produced on the CPU by [`TileTree::compute_draw_set`] with
        // explicit 2:1 LOD-gap enforcement across cube-face neighbours —
        // the GPU predicate was per-tile-independent and could emit
        // gap-≥-2 jumps at face seams, which show up as elevation shears
        // because the CDLOD morph only spans one LOD.
        //
        // `TerrainViewData::prepare` uploads the CPU draw set into the
        // same `final_tile_buffer` the vertex shader reads, and sets the
        // `indirect_buffer` (vertex_count, instance_count) for
        // `draw_indirect`. The compute pipelines and bind groups remain
        // wired so we can bisect against this path by un-no-op-ing the
        // dispatches; once the new approach beds in, the WGSL prepass
        // files and the temporary-tiles / parameters buffers can be
        // removed.
        Ok(())
    }
}

pub(crate) fn queue_tiling_prepass(
    debug: Option<Res<DebugTerrain>>,
    pipeline_cache: Res<PipelineCache>,
    prepass_pipelines: ResMut<TilingPrepassPipelines>,
    mut pipelines: ResMut<SpecializedComputePipelines<TilingPrepassPipelines>>,
    mut prepass_items: ResMut<TerrainViewComponents<TilingPrepassItem>>,
    gpu_tile_trees: Res<TerrainViewComponents<GpuTileTree>>,
    gpu_tile_atlases: Res<TerrainComponents<GpuTileAtlas>>,
) {
    for &(terrain, view) in gpu_tile_trees.keys() {
        let gpu_tile_atlas = gpu_tile_atlases.get(&terrain).unwrap();

        let mut key = TilingPrepassPipelineKey::NONE;

        if gpu_tile_atlas.is_spherical {
            key |= TilingPrepassPipelineKey::SPHERICAL;
        }

        if let Some(debug) = &debug {
            key |= TilingPrepassPipelineKey::from_debug(debug);
        }

        let refine_tiles_pipeline = pipelines.specialize(
            &pipeline_cache,
            &prepass_pipelines,
            key | TilingPrepassPipelineKey::REFINE_TILES,
        );
        let prepare_root_pipeline = pipelines.specialize(
            &pipeline_cache,
            &prepass_pipelines,
            key | TilingPrepassPipelineKey::PREPARE_ROOT,
        );
        let prepare_next_pipeline = pipelines.specialize(
            &pipeline_cache,
            &prepass_pipelines,
            key | TilingPrepassPipelineKey::PREPARE_NEXT,
        );
        let prepare_render_pipeline = pipelines.specialize(
            &pipeline_cache,
            &prepass_pipelines,
            key | TilingPrepassPipelineKey::PREPARE_RENDER,
        );

        prepass_items.insert(
            (terrain, view),
            TilingPrepassItem {
                refine_tiles_pipeline,
                prepare_root_pipeline,
                prepare_next_pipeline,
                prepare_render_pipeline,
            },
        );
    }
}
