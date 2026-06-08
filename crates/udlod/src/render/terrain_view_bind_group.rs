use crate::{
    math::{TerrainModelApproximation, TileCoordinate},
    terrain_data::{gpu_tile_tree::GpuTileTree, tile_tree::TileTree},
    terrain_view::TerrainViewComponents,
    util::StaticBuffer,
};
use bevy::{
    ecs::{
        query::ROQueryItem,
        system::{
            lifetimeless::{Read, SRes},
            SystemParamItem,
        },
    },
    prelude::*,
    render::{
        render_phase::{PhaseItem, RenderCommand, RenderCommandResult, TrackedRenderPass},
        render_resource::{binding_types::*, *},
        renderer::{RenderDevice, RenderQueue},
        sync_world::MainEntity,
        Extract,
    },
};

/// Descriptor for the indirect-buffer bind group used by the prepare-prepass compute pass.
pub(crate) fn prepare_indirect_layout_descriptor() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "prepare_indirect_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::COMPUTE,
            storage_buffer::<Indirect>(false), // indirect buffer
        ),
    )
}

/// Descriptor for the refine-tiles bind group used by the tiling-prepass compute pipelines.
pub(crate) fn refine_tiles_layout_descriptor() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "refine_tiles_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<TerrainViewConfigUniform>(false), // terrain view config
                uniform_buffer::<TerrainModelApproximation>(false), // model view approximation
                storage_buffer_read_only_sized(false, None),       // tile_tree
                storage_buffer_read_only_sized(false, None),       // origins
                storage_buffer_sized(false, None),                 // final tiles
                storage_buffer_sized(false, None),                 // temporary tiles
                storage_buffer::<Parameters>(false),               // parameters
            ),
        ),
    )
}

/// Descriptor for the per-view bind group used by the terrain render pipeline.
pub(crate) fn terrain_view_layout_descriptor() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "terrain_view_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<TerrainViewConfigUniform>(false), // terrain view config
                uniform_buffer::<TerrainModelApproximation>(false), // model view approximation
                storage_buffer_read_only_sized(false, None),       // tile_tree
                storage_buffer_read_only_sized(false, None),       // origins
                storage_buffer_read_only_sized(false, None),       // tiles
            ),
        ),
    )
}

#[derive(Default, ShaderType)]
pub(crate) struct Indirect {
    x_or_vertex_count: u32,
    y_or_instance_count: u32,
    z_or_base_vertex: u32,
    base_instance: u32,
}

#[derive(Default, ShaderType)]
struct Parameters {
    tile_count: u32,
    counter: i32,
    child_index: i32,
    final_index: i32,
}

#[derive(Default, ShaderType)]
struct TerrainViewConfigUniform {
    tree_size: u32,
    geometry_tile_count: u32,
    refinement_count: u32,
    grid_size: f32,
    vertices_per_row: u32,
    vertices_per_tile: u32,
    morph_distance: f32,
    blend_distance: f32,
    load_distance: f32,
    subdivision_distance: f32,
    morph_range: f32,
    blend_range: f32,
    precision_threshold_distance: f32,
}

impl TerrainViewConfigUniform {
    fn from_tile_tree(tile_tree: &TileTree) -> Self {
        TerrainViewConfigUniform {
            tree_size: tile_tree.tree_size,
            geometry_tile_count: tile_tree.geometry_tile_count,
            refinement_count: tile_tree.refinement_count,
            grid_size: tile_tree.grid_size as f32,
            vertices_per_row: 2 * (tile_tree.grid_size + 2),
            vertices_per_tile: 2 * tile_tree.grid_size * (tile_tree.grid_size + 2),
            morph_distance: tile_tree.morph_distance as f32,
            blend_distance: tile_tree.blend_distance as f32,
            load_distance: tile_tree.load_distance as f32,
            subdivision_distance: tile_tree.subdivision_distance as f32,
            precision_threshold_distance: tile_tree.precision_threshold_distance as f32,
            morph_range: tile_tree.morph_range,
            blend_range: tile_tree.blend_range,
        }
    }
}

#[allow(dead_code)]
pub struct TerrainViewData {
    view_config_buffer: StaticBuffer<TerrainViewConfigUniform>,
    terrain_model_approximation_buffer: StaticBuffer<TerrainModelApproximation>,
    pub(super) indirect_buffer: StaticBuffer<Indirect>,
    pub(super) prepare_indirect_bind_group: BindGroup,
    pub(super) refine_tiles_bind_group: BindGroup,
    pub(super) terrain_view_bind_group: BindGroup,
    final_tile_buffer: StaticBuffer<()>,
    /// CPU-balanced draw set extracted from main world this frame. Written
    /// to [`Self::final_tile_buffer`] in `prepare` and consumed by the
    /// vertex shader as `geometry_tiles`. Empty when the camera has no
    /// terrain in view; the indirect-draw vertex count goes to zero and
    /// the draw call is a no-op.
    draw_set: Vec<TileCoordinate>,
    /// Vertices per drawn tile — needed in `prepare` to set the
    /// `vertex_count` for `draw_indirect`. Mirrors
    /// `view_config_buffer.value().vertices_per_tile` but cached so
    /// `prepare` doesn't depend on whether the view-config has been set
    /// this frame yet.
    vertices_per_tile: u32,
}

impl TerrainViewData {
    fn new(
        device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        tile_tree: &TileTree,
        gpu_tile_tree: &GpuTileTree,
    ) -> Self {
        // Todo: figure out a better way of limiting the tile buffer size
        let tile_buffer_size =
            TileCoordinate::min_size().get() * tile_tree.geometry_tile_count as BufferAddress;

        let view_config_buffer =
            StaticBuffer::empty(None, device, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        // `final_tile_buffer` and `indirect_buffer` are now written from
        // the CPU each frame (the CPU-balanced draw set replaces the GPU
        // refine pass), so both need `COPY_DST`. The GPU compute prepass
        // is still wired up but no longer dispatched — keeping the bind
        // groups intact lets us bisect against the old path by toggling
        // the dispatch back on, and avoids churning the bind-group
        // descriptors.
        let indirect_buffer = StaticBuffer::empty(
            None,
            device,
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        );
        let parameter_buffer =
            StaticBuffer::<Parameters>::empty(None, device, BufferUsages::STORAGE);
        let temporary_tile_buffer =
            StaticBuffer::<()>::empty_sized(None, device, tile_buffer_size, BufferUsages::STORAGE);
        let final_tile_buffer = StaticBuffer::<()>::empty_sized(
            None,
            device,
            tile_buffer_size,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let terrain_model_approximation_buffer = StaticBuffer::<TerrainModelApproximation>::empty(
            None,
            device,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        let prepare_indirect_layout =
            pipeline_cache.get_bind_group_layout(&prepare_indirect_layout_descriptor());
        let refine_tiles_layout =
            pipeline_cache.get_bind_group_layout(&refine_tiles_layout_descriptor());
        let terrain_view_layout =
            pipeline_cache.get_bind_group_layout(&terrain_view_layout_descriptor());

        let prepare_indirect_bind_group = device.create_bind_group(
            "prepare_indirect_bind_group",
            &prepare_indirect_layout,
            &BindGroupEntries::single(&indirect_buffer),
        );
        let refine_tiles_bind_group = device.create_bind_group(
            "refine_tiles_bind_group",
            &refine_tiles_layout,
            &BindGroupEntries::sequential((
                &view_config_buffer,
                &terrain_model_approximation_buffer,
                &gpu_tile_tree.tile_tree_buffer,
                &gpu_tile_tree.origins_buffer,
                &final_tile_buffer,
                &temporary_tile_buffer,
                &parameter_buffer,
            )),
        );
        let terrain_view_bind_group = device.create_bind_group(
            "terrain_view_bind_group",
            &terrain_view_layout,
            &BindGroupEntries::sequential((
                &view_config_buffer,
                &terrain_model_approximation_buffer,
                &gpu_tile_tree.tile_tree_buffer,
                &gpu_tile_tree.origins_buffer,
                &final_tile_buffer,
            )),
        );

        Self {
            view_config_buffer,
            terrain_model_approximation_buffer,
            indirect_buffer,
            prepare_indirect_bind_group,
            refine_tiles_bind_group,
            terrain_view_bind_group,
            final_tile_buffer,
            draw_set: Vec::new(),
            vertices_per_tile: 0,
        }
    }

    #[allow(dead_code)]
    pub(super) fn refinement_count(&self) -> u32 {
        self.view_config_buffer.value().refinement_count
    }

    pub(crate) fn initialize(
        device: Res<RenderDevice>,
        pipeline_cache: Res<PipelineCache>,
        mut terrain_view_data: ResMut<TerrainViewComponents<TerrainViewData>>,
        gpu_tile_trees: Res<TerrainViewComponents<GpuTileTree>>,
        tile_trees: Extract<Res<TerrainViewComponents<TileTree>>>,
    ) {
        for (&(terrain, view), tile_tree) in tile_trees.iter() {
            if terrain_view_data.contains_key(&(terrain, view)) {
                continue;
            }

            let Some(gpu_tile_tree) = gpu_tile_trees.get(&(terrain, view)) else {
                continue;
            };

            terrain_view_data.insert(
                (terrain, view),
                TerrainViewData::new(&device, &pipeline_cache, tile_tree, gpu_tile_tree),
            );
        }
    }

    pub(crate) fn extract(
        mut terrain_view_data: ResMut<TerrainViewComponents<TerrainViewData>>,
        tile_trees: Extract<Res<TerrainViewComponents<TileTree>>>,
        terrain_model_approximations: Extract<
            Res<TerrainViewComponents<TerrainModelApproximation>>,
        >,
    ) {
        terrain_view_data.retain(|key, _| tile_trees.contains_key(key));

        for (&(terrain, view), tile_tree) in tile_trees.iter() {
            let Some(terrain_view_data) = terrain_view_data.get_mut(&(terrain, view)) else {
                continue;
            };

            let config = TerrainViewConfigUniform::from_tile_tree(tile_tree);
            terrain_view_data.vertices_per_tile = config.vertices_per_tile;
            terrain_view_data.view_config_buffer.set_value(config);

            if let Some(approximation) = terrain_model_approximations.get(&(terrain, view)) {
                terrain_view_data
                    .terrain_model_approximation_buffer
                    .set_value(approximation.clone());
            }

            // Pull the latest CPU-balanced draw set across the world
            // boundary. The vector is short (≪ atlas capacity in
            // practice) so cloning is cheap; we hand it to the buffer
            // serializer in `prepare` rather than keeping a reference,
            // since the main-world resource may be reset before
            // `prepare` runs.
            terrain_view_data.draw_set.clear();
            terrain_view_data
                .draw_set
                .extend_from_slice(&tile_tree.draw_set);
        }
    }

    pub(crate) fn prepare(
        queue: Res<RenderQueue>,
        mut terrain_view_data: ResMut<TerrainViewComponents<TerrainViewData>>,
    ) {
        for data in &mut terrain_view_data.values_mut() {
            data.view_config_buffer.update(&queue);
            data.terrain_model_approximation_buffer.update(&queue);

            // Upload the CPU draw set to the buffer the vertex shader
            // reads (the old `final_tile_buffer` of the GPU prepass).
            // Serialise via `encase::StorageBuffer` so the layout matches
            // WGSL's `array<TileCoordinate>` exactly — `TileCoordinate`
            // already derives `ShaderType`.
            if !data.draw_set.is_empty() {
                let mut scratch = encase::StorageBuffer::new(Vec::<u8>::new());
                scratch.write(&data.draw_set).unwrap();
                data.final_tile_buffer
                    .update_bytes(&queue, scratch.as_ref());
            }

            // `draw_indirect` reads `[vertex_count, instance_count,
            // first_vertex, first_instance]` from this buffer. The vertex
            // shader rebuilds the per-tile mesh from `vertex_index`, so
            // one big draw call with `vertex_count = vertices_per_tile *
            // tile_count` produces the entire terrain.
            //
            // The WGSL `Indirect` view of this buffer is only a
            // `vec3<u32> workgroup_count` (3 u32s); the 4th u32
            // (`first_instance`) stays zeroed from buffer creation
            // because we never overwrite it.
            let tile_count = data.draw_set.len() as u32;
            let vertex_count = data.vertices_per_tile.saturating_mul(tile_count);
            data.indirect_buffer.set_value(Indirect {
                x_or_vertex_count: vertex_count,
                y_or_instance_count: if tile_count == 0 { 0 } else { 1 },
                z_or_base_vertex: 0,
                base_instance: 0,
            });
            data.indirect_buffer.update(&queue);
        }
    }
}

pub struct SetTerrainViewBindGroup<const I: usize>;

impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetTerrainViewBindGroup<I> {
    type Param = SRes<TerrainViewComponents<TerrainViewData>>;
    // TerrainViewComponents is keyed on main-world (terrain, view) entity IDs,
    // so resolve the render-world view to its MainEntity here.
    type ViewQuery = Read<MainEntity>;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        terrain_view_data: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let data = terrain_view_data
            .into_inner()
            .get(&(item.main_entity().id(), view.id()))
            .unwrap();

        pass.set_bind_group(I, &data.terrain_view_bind_group, &[]);
        RenderCommandResult::Success
    }
}

pub(crate) struct DrawTerrainCommand;

impl<P: PhaseItem> RenderCommand<P> for DrawTerrainCommand {
    type Param = SRes<TerrainViewComponents<TerrainViewData>>;
    type ViewQuery = Read<MainEntity>;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        terrain_view_data: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let data = terrain_view_data
            .into_inner()
            .get(&(item.main_entity().id(), view.id()))
            .unwrap();

        pass.draw_indirect(&data.indirect_buffer, 0);

        RenderCommandResult::Success
    }
}
