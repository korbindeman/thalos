use crate::{
    terrain_data::gpu_tile_tree::GpuTileTree, terrain_view::TerrainViewComponents,
    util::StaticBuffer,
};
use bevy::{
    platform::collections::HashMap,
    prelude::*,
    render::{
        render_resource::{binding_types::*, *},
        renderer::RenderDevice,
        sync_world::MainEntity,
        view::ExtractedView,
    },
};
use std::ops::Deref;

/// Descriptor for the culling bind group layout. Resolved to a [`BindGroupLayout`]
/// via [`PipelineCache::get_bind_group_layout`] at bind-group creation time.
pub(crate) fn culling_layout_descriptor() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "culling_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::COMPUTE,
            uniform_buffer::<CullingUniform>(false), // culling data
        ),
    )
}

pub fn planes(view_projection: &Mat4) -> [Vec4; 5] {
    let row3 = view_projection.row(3);
    let mut planes = [default(); 5];
    for (i, plane) in planes.iter_mut().enumerate() {
        let row = view_projection.row(i / 2);
        *plane = if (i & 1) == 0 && i != 4 {
            row3 + row
        } else {
            row3 - row
        };
    }

    planes
}

#[derive(Default, ShaderType)]
pub struct CullingUniform {
    world_position: Vec3,
    view_proj: Mat4,
    planes: [Vec4; 5],
}

impl From<&ExtractedView> for CullingUniform {
    fn from(view: &ExtractedView) -> Self {
        Self {
            world_position: view.world_from_view.translation(),
            view_proj: view.world_from_view.to_matrix().inverse(),
            planes: default(),
        }
    }
}

#[derive(Component)]
pub struct CullingBindGroup(BindGroup);

impl Deref for CullingBindGroup {
    type Target = BindGroup;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl CullingBindGroup {
    fn new(
        device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        culling_uniform: CullingUniform,
    ) -> Self {
        let culling_buffer = StaticBuffer::<CullingUniform>::create(
            None,
            device,
            &culling_uniform,
            BufferUsages::UNIFORM,
        );

        let layout = pipeline_cache.get_bind_group_layout(&culling_layout_descriptor());
        let bind_group = device.create_bind_group(
            None,
            &layout,
            &BindGroupEntries::single(&culling_buffer),
        );

        Self(bind_group)
    }

    pub(crate) fn prepare(
        device: Res<RenderDevice>,
        pipeline_cache: Res<PipelineCache>,
        gpu_tile_trees: Res<TerrainViewComponents<GpuTileTree>>,
        extracted_views: Query<(&MainEntity, &ExtractedView)>,
        mut culling_bind_groups: ResMut<TerrainViewComponents<CullingBindGroup>>,
    ) {
        // gpu_tile_trees is keyed on main-world entity IDs (extracted from the
        // main world's TileTree resource). ExtractedView lives on render-world
        // entities whose IDs are unstable; we bridge via the MainEntity
        // component that points back to the main-world camera.
        let view_by_main: HashMap<Entity, &ExtractedView> = extracted_views
            .iter()
            .map(|(main_entity, view)| (main_entity.id(), view))
            .collect();

        for &(terrain, view) in gpu_tile_trees.keys() {
            let Some(&extracted_view) = view_by_main.get(&view) else {
                continue;
            };

            culling_bind_groups.insert(
                (terrain, view),
                CullingBindGroup::new(&device, &pipeline_cache, extracted_view.into()),
            );
        }
    }
}
