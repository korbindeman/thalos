//! Renders terrain entities through a custom indirect-draw pipeline that piggybacks on Bevy
//! 0.18's standard `MaterialPlugin<M>` infrastructure for asset extraction and material bind
//! group allocation, but uses a bespoke queue + draw command tuple to actually emit the draw.
//!
//! Bevy 0.18's [`MaterialPipeline`], [`RenderMaterialInstances`] and [`PreparedMaterial`] are
//! all type-erased. Adding [`MaterialPlugin::<M>::default()`] to the app gives us:
//!
//!   * `init_asset::<M>` (so users can `Assets<M>::add(...)` materials),
//!   * extraction of `MeshMaterial3d<M>` components into `RenderMaterialInstances`,
//!   * a per-`M` `MaterialBindGroupAllocator` (registered via `add_material_bind_group_allocator`)
//!     which the standard `SetMaterialBindGroup<I>` looks up at render time.
//!
//! On top of that we add our own `queue_terrain<M>` system that walks `RenderVisibleEntities`
//! filtered by [`TileAtlas`], pulls the matching material instance out of `RenderMaterialInstances`
//! (filtered by `TypeId::of::<M>()`), specializes our `TerrainRenderPipeline<M>` against the
//! `TerrainPipelineFlags` for the view's MSAA + debug state, and queues an [`Opaque3d`] phase
//! item with our [`DrawTerrain`] command tuple. The draw command then invokes
//! [`DrawTerrainCommand`] which issues the indirect draw whose parameters were filled in by the
//! compute prepass in [`crate::render::tiling_prepass`].

use crate::{
    debug::DebugTerrain,
    render::{
        terrain_bind_group::{terrain_layout_descriptor, SetTerrainBindGroup},
        terrain_view_bind_group::{
            terrain_view_layout_descriptor, DrawTerrainCommand, SetTerrainViewBindGroup,
        },
    },
    shaders::{DEFAULT_FRAGMENT_SHADER, DEFAULT_VERTEX_SHADER},
    terrain::TerrainComponents,
    terrain_data::{gpu_tile_atlas::GpuTileAtlas, tile_atlas::TileAtlas},
};
use bevy::{
    core_pipeline::core_3d::{Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey},
    ecs::system::SystemChangeTick,
    image::BevyDefault,
    pbr::{
        MaterialPlugin, MeshPipelineViewLayoutKey, MeshPipelineViewLayouts, SetMaterialBindGroup,
        SetMeshViewBindGroup,
    },
    prelude::*,
    render::{
        render_phase::{
            AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, InputUniformIndex,
            SetItemPipeline, ViewBinnedRenderPhases,
        },
        render_resource::*,
        view::{ExtractedView, RenderVisibleEntities, ViewTarget},
        Render, RenderApp, RenderStartup, RenderSystems,
    },
    shader::{ShaderDefVal, ShaderRef},
};
use bevy::pbr::{PreparedMaterial, RenderMaterialInstances};
use bevy::render::erased_render_asset::ErasedRenderAssets;
use bevy::render::renderer::RenderDevice;
use std::{any::TypeId, hash::Hash, marker::PhantomData};

pub struct TerrainPipelineKey<M: Material> {
    pub flags: TerrainPipelineFlags,
    pub bind_group_data: M::Data,
}

impl<M: Material> Eq for TerrainPipelineKey<M> where M::Data: PartialEq {}

impl<M: Material> PartialEq for TerrainPipelineKey<M>
where
    M::Data: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.flags == other.flags && self.bind_group_data == other.bind_group_data
    }
}

impl<M: Material> Clone for TerrainPipelineKey<M>
where
    M::Data: Clone,
{
    fn clone(&self) -> Self {
        Self {
            flags: self.flags,
            bind_group_data: self.bind_group_data.clone(),
        }
    }
}

impl<M: Material> Hash for TerrainPipelineKey<M>
where
    M::Data: Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.flags.hash(state);
        self.bind_group_data.hash(state);
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct TerrainPipelineFlags: u32 {
        const NONE               = 0;
        const SPHERICAL          = 1 <<  0;
        const WIREFRAME          = 1 <<  1;
        const SHOW_DATA_LOD      = 1 <<  2;
        const SHOW_GEOMETRY_LOD  = 1 <<  3;
        const SHOW_TILE_TREE     = 1 <<  4;
        const SHOW_PIXELS        = 1 <<  5;
        const SHOW_UV            = 1 <<  6;
        const SHOW_NORMALS       = 1 <<  7;
        const MORPH              = 1 <<  8;
        const BLEND              = 1 <<  9;
        const TILE_TREE_LOD      = 1 << 10;
        const LIGHTING           = 1 << 11;
        const SAMPLE_GRAD        = 1 << 12;
        const HIGH_PRECISION     = 1 << 13;
        const TEST1              = 1 << 14;
        const TEST2              = 1 << 15;
        const TEST3              = 1 << 16;
        const HDR                = 1 << 17;
        const MSAA_RESERVED_BITS = TerrainPipelineFlags::MSAA_MASK_BITS << TerrainPipelineFlags::MSAA_SHIFT_BITS;
    }
}

impl TerrainPipelineFlags {
    const MSAA_MASK_BITS: u32 = 0b111111;
    const MSAA_SHIFT_BITS: u32 = 32 - 6;

    pub fn from_msaa_samples(msaa_samples: u32) -> Self {
        let msaa_bits = ((msaa_samples - 1) & Self::MSAA_MASK_BITS) << Self::MSAA_SHIFT_BITS;
        TerrainPipelineFlags::from_bits(msaa_bits).unwrap()
    }

    pub fn from_debug(debug: &DebugTerrain) -> Self {
        let mut key = TerrainPipelineFlags::NONE;

        if debug.wireframe {
            key |= TerrainPipelineFlags::WIREFRAME;
        }
        if debug.show_data_lod {
            key |= TerrainPipelineFlags::SHOW_DATA_LOD;
        }
        if debug.show_geometry_lod {
            key |= TerrainPipelineFlags::SHOW_GEOMETRY_LOD;
        }
        if debug.show_tile_tree {
            key |= TerrainPipelineFlags::SHOW_TILE_TREE;
        }
        if debug.show_pixels {
            key |= TerrainPipelineFlags::SHOW_PIXELS;
        }
        if debug.show_uv {
            key |= TerrainPipelineFlags::SHOW_UV;
        }
        if debug.show_normals {
            key |= TerrainPipelineFlags::SHOW_NORMALS;
        }
        if debug.morph {
            key |= TerrainPipelineFlags::MORPH;
        }
        if debug.blend {
            key |= TerrainPipelineFlags::BLEND;
        }
        if debug.tile_tree_lod {
            key |= TerrainPipelineFlags::TILE_TREE_LOD;
        }
        if debug.lighting {
            key |= TerrainPipelineFlags::LIGHTING;
        }
        if debug.sample_grad {
            key |= TerrainPipelineFlags::SAMPLE_GRAD;
        }
        if debug.high_precision {
            key |= TerrainPipelineFlags::HIGH_PRECISION;
        }
        if debug.test1 {
            key |= TerrainPipelineFlags::TEST1;
        }
        if debug.test2 {
            key |= TerrainPipelineFlags::TEST2;
        }
        if debug.test3 {
            key |= TerrainPipelineFlags::TEST3;
        }

        key
    }

    pub fn msaa_samples(&self) -> u32 {
        ((self.bits() >> Self::MSAA_SHIFT_BITS) & Self::MSAA_MASK_BITS) + 1
    }

    pub fn polygon_mode(&self) -> PolygonMode {
        match self.contains(TerrainPipelineFlags::WIREFRAME) {
            true => PolygonMode::Line,
            false => PolygonMode::Fill,
        }
    }

    pub fn shader_defs(&self) -> Vec<ShaderDefVal> {
        let mut shader_defs = Vec::new();

        if self.contains(TerrainPipelineFlags::SPHERICAL) {
            shader_defs.push("SPHERICAL".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_DATA_LOD) {
            shader_defs.push("SHOW_DATA_LOD".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_GEOMETRY_LOD) {
            shader_defs.push("SHOW_GEOMETRY_LOD".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_TILE_TREE) {
            shader_defs.push("SHOW_TILE_TREE".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_PIXELS) {
            shader_defs.push("SHOW_PIXELS".into())
        }
        if self.contains(TerrainPipelineFlags::SHOW_UV) {
            shader_defs.push("SHOW_UV".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_NORMALS) {
            shader_defs.push("SHOW_NORMALS".into())
        }
        if self.contains(TerrainPipelineFlags::MORPH) {
            shader_defs.push("MORPH".into());
        }
        if self.contains(TerrainPipelineFlags::BLEND) {
            shader_defs.push("BLEND".into());
        }
        if self.contains(TerrainPipelineFlags::TILE_TREE_LOD) {
            shader_defs.push("TILE_TREE_LOD".into());
        }
        if self.contains(TerrainPipelineFlags::LIGHTING) {
            shader_defs.push("LIGHTING".into());
        }
        if self.contains(TerrainPipelineFlags::SAMPLE_GRAD) {
            shader_defs.push("SAMPLE_GRAD".into());
        }
        if self.contains(TerrainPipelineFlags::HIGH_PRECISION) {
            shader_defs.push("HIGH_PRECISION".into());
        }
        if self.contains(TerrainPipelineFlags::TEST1) {
            shader_defs.push("TEST1".into());
        }
        if self.contains(TerrainPipelineFlags::TEST2) {
            shader_defs.push("TEST2".into());
        }
        if self.contains(TerrainPipelineFlags::TEST3) {
            shader_defs.push("TEST3".into());
        }

        shader_defs
    }
}

/// The pipeline used to render the terrain entities.
#[derive(Resource)]
pub struct TerrainRenderPipeline<M: Material> {
    pub(crate) view_layout: BindGroupLayoutDescriptor,
    pub(crate) view_layout_multisampled: BindGroupLayoutDescriptor,
    pub(crate) terrain_layout: BindGroupLayoutDescriptor,
    pub(crate) terrain_view_layout: BindGroupLayoutDescriptor,
    pub(crate) material_layout: BindGroupLayoutDescriptor,
    pub vertex_shader: Handle<Shader>,
    pub fragment_shader: Handle<Shader>,
    marker: PhantomData<M>,
}

/// Initializes the [`TerrainRenderPipeline<M>`] resource. Runs as a [`RenderStartup`] system
/// instead of through [`Plugin::finish`] + [`FromWorld`] so that resource availability
/// (specifically [`MeshPipelineViewLayouts`], which is itself populated in
/// [`MeshRenderPlugin::finish`]) doesn't depend on plugin registration order.
pub fn init_terrain_render_pipeline<M: Material>(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mesh_view_layouts: Res<MeshPipelineViewLayouts>,
    render_device: Res<RenderDevice>,
) {
    let view_layout = mesh_view_layouts
        .get_view_layout(MeshPipelineViewLayoutKey::empty())
        .main_layout
        .clone();
    let view_layout_multisampled = mesh_view_layouts
        .get_view_layout(MeshPipelineViewLayoutKey::MULTISAMPLED)
        .main_layout
        .clone();
    let material_layout = M::bind_group_layout_descriptor(&render_device);

    let vertex_shader = match M::vertex_shader() {
        ShaderRef::Default => asset_server.load(DEFAULT_VERTEX_SHADER),
        ShaderRef::Handle(handle) => handle,
        ShaderRef::Path(path) => asset_server.load(path),
    };
    let fragment_shader = match M::fragment_shader() {
        ShaderRef::Default => asset_server.load(DEFAULT_FRAGMENT_SHADER),
        ShaderRef::Handle(handle) => handle,
        ShaderRef::Path(path) => asset_server.load(path),
    };

    commands.insert_resource(TerrainRenderPipeline::<M> {
        view_layout,
        view_layout_multisampled,
        terrain_layout: terrain_layout_descriptor(),
        terrain_view_layout: terrain_view_layout_descriptor(),
        material_layout,
        vertex_shader,
        fragment_shader,
        marker: PhantomData,
    });
}

impl<M: Material> SpecializedRenderPipeline for TerrainRenderPipeline<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    type Key = TerrainPipelineKey<M>;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = key.flags.shader_defs();

        let mut layout = match key.flags.msaa_samples() {
            1 => vec![self.view_layout.clone()],
            _ => {
                shader_defs.push("MULTISAMPLED".into());
                vec![self.view_layout_multisampled.clone()]
            }
        };

        layout.push(self.terrain_layout.clone());
        layout.push(self.terrain_view_layout.clone());
        layout.push(self.material_layout.clone());

        let vertex_shader_defs = shader_defs.clone();
        let mut fragment_shader_defs = shader_defs;
        fragment_shader_defs.push("FRAGMENT".into());

        RenderPipelineDescriptor {
            label: Some("terrain_pipeline".into()),
            layout,
            push_constant_ranges: default(),
            vertex: VertexState {
                shader: self.vertex_shader.clone(),
                entry_point: Some("vertex".into()),
                shader_defs: vertex_shader_defs,
                buffers: Vec::new(),
            },
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: key.flags.polygon_mode(),
                conservative: false,
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
            },
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs: fragment_shader_defs,
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    // Match the view's color target format. `ViewTarget` picks
                    // `Rgba16Float` whenever the camera has `Hdr` enabled and
                    // `Rgba8UnormSrgb` (== `bevy_default`) otherwise; queue_terrain
                    // sets the `HDR` flag from `ExtractedView::hdr`.
                    format: if key.flags.contains(TerrainPipelineFlags::HDR) {
                        ViewTarget::TEXTURE_FORMAT_HDR
                    } else {
                        TextureFormat::bevy_default()
                    },
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: key.flags.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            zero_initialize_workgroup_memory: false,
        }
    }
}

/// The draw command tuple used to render terrain entities. The `SetMaterialBindGroup<3>` looks up
/// the bound material via the per-item `MainEntity` and the global [`RenderMaterialInstances`]
/// table, so it doesn't need a generic `M`.
pub(crate) type DrawTerrain = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetTerrainBindGroup<1>,
    SetTerrainViewBindGroup<2>,
    SetMaterialBindGroup<3>,
    DrawTerrainCommand,
);

/// Queues all visible terrain entities for the [`Opaque3d`] phase. Iterates one [`Opaque3d`]
/// phase per camera; for each visible terrain entity (identified via the [`TileAtlas`] visibility
/// class), looks up its material via [`RenderMaterialInstances`], filters to material type `M`,
/// builds a [`TerrainPipelineKey`] from the view's MSAA + the (optional) [`DebugTerrain`] state,
/// specializes the pipeline, and adds it to the phase as a non-mesh binned item.
#[allow(clippy::too_many_arguments)]
pub(crate) fn queue_terrain<M: Material>(
    draw_functions: Res<DrawFunctions<Opaque3d>>,
    debug: Option<Res<DebugTerrain>>,
    render_materials: Res<ErasedRenderAssets<PreparedMaterial>>,
    pipeline_cache: Res<PipelineCache>,
    terrain_pipeline: Res<TerrainRenderPipeline<M>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<TerrainRenderPipeline<M>>>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    gpu_tile_atlases: Res<TerrainComponents<GpuTileAtlas>>,
    render_material_instances: Res<RenderMaterialInstances>,
    views: Query<(&ExtractedView, &RenderVisibleEntities, &Msaa)>,
    change_tick: SystemChangeTick,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    let Some(draw_function) = draw_functions.read().get_id::<DrawTerrain>() else {
        return;
    };

    for (view, visible_entities, msaa) in &views {
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        for &(render_entity, main_entity) in visible_entities.iter::<TileAtlas>() {
            let Some(material_instance) = render_material_instances.instances.get(&main_entity)
            else {
                continue;
            };
            // Skip materials of other types - we are queuing only entities whose material is `M`.
            if material_instance.asset_id.type_id() != TypeId::of::<M>() {
                continue;
            }
            let Some(material) = render_materials.get(material_instance.asset_id) else {
                continue;
            };
            let Some(gpu_tile_atlas) = gpu_tile_atlases.get(&*main_entity) else {
                continue;
            };

            let mut flags = TerrainPipelineFlags::from_msaa_samples(msaa.samples());

            if gpu_tile_atlas.is_spherical {
                flags |= TerrainPipelineFlags::SPHERICAL;
            }

            if view.hdr {
                flags |= TerrainPipelineFlags::HDR;
            }

            if let Some(debug) = &debug {
                flags |= TerrainPipelineFlags::from_debug(debug);
            } else {
                flags |= TerrainPipelineFlags::LIGHTING
                    | TerrainPipelineFlags::MORPH
                    | TerrainPipelineFlags::BLEND
                    | TerrainPipelineFlags::SAMPLE_GRAD;
            }

            let key = TerrainPipelineKey {
                flags,
                bind_group_data: material.properties.material_key.to_key::<M::Data>(),
            };
            let pipeline = pipelines.specialize(&pipeline_cache, &terrain_pipeline, key);

            let batch_set_key = Opaque3dBatchSetKey {
                pipeline,
                draw_function,
                material_bind_group_index: Some(*material.binding.group),
                vertex_slab: default(),
                index_slab: None,
                lightmap_slab: None,
            };
            let bin_key = Opaque3dBinKey {
                asset_id: material_instance.asset_id,
            };
            opaque_phase.add(
                batch_set_key,
                bin_key,
                (render_entity, main_entity),
                InputUniformIndex::default(),
                BinnedRenderPhaseType::NonMesh,
                change_tick.this_run(),
            );
        }
    }
}

/// This plugin adds a custom material for a terrain.
///
/// It can be used to render the terrain using a custom vertex and fragment shader.
///
/// Internally it adds [`MaterialPlugin::<M>::default()`] for the standard material asset
/// extraction + bind group allocator infrastructure, then layers our own queue + render command
/// tuple on top to drive the indirect draw filled in by the compute prepass.
pub struct TerrainMaterialPlugin<M: Material>(PhantomData<M>);

impl<M: Material> Default for TerrainMaterialPlugin<M> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<M: Material> Plugin for TerrainMaterialPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<M>::default());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<SpecializedRenderPipelines<TerrainRenderPipeline<M>>>()
                .add_render_command::<Opaque3d, DrawTerrain>()
                .add_systems(RenderStartup, init_terrain_render_pipeline::<M>)
                .add_systems(Render, queue_terrain::<M>.in_set(RenderSystems::QueueMeshes));
        }
    }
}

