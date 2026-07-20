//! Fullscreen-quad atmospheric sky for one body.
//!
//! Renders the in-scatter integral of `thalos::atmosphere` for every view
//! ray, premultiplied with opacity. Blended over the celestial background so
//! stars dim where the daytime sky is bright (the in-scatter alpha boost
//! crushes them) and re-emerge only as the sky darkens toward night/twilight.
//!
//! Reuses the same atmosphere uniforms as [`crate::BodyTerrainMaterial`] so a
//! single per-frame update system writes both.
//!
//! ## Manual `AsBindGroup` (ADR-0006)
//!
//! The analytic-ocean branch samples **signed sea height straight from the
//! udlod height-tile atlas** — the exact texels the visible terrain mesh is
//! displaced from — so water coverage/colour are a projection of the one
//! terrain field instead of a depth-buffer comparison. Those resources
//! (attachment-0 texture array, tile-tree + origins storage buffers) live in
//! udlod's render-world registries, not in `Assets`, so the derive can't bind
//! them: this material implements `AsBindGroup` by hand, keeping the derive's
//! exact layout for bindings 0–10 and appending the tile lookup at 11–14.
//! The material is mutated every frame by the game's sky update, so the bind
//! group re-prepares every frame and the lookup never goes stale across
//! terrain despawn/respawn.

use bevy::asset::embedded_asset;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::ecs::system::SystemParamItem;
use bevy::image::Image;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, texture_2d, texture_2d_array, texture_cube,
    texture_depth_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    encase, AsBindGroup, AsBindGroupError, BindGroupLayout, BindGroupLayoutEntries,
    BindGroupLayoutEntry, BindingResources, BufferInitDescriptor, BufferUsages,
    CompareFunction, FilterMode, MipmapFilterMode, OwnedBindingResource, RenderPipelineDescriptor,
    SamplerBindingType, SamplerDescriptor, ShaderStages, SpecializedMeshPipelineError,
    TextureSampleType, TextureViewDimension, UnpreparedBindGroup,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::{FallbackImage, GpuImage};
use bevy::shader::ShaderRef;

use thalos_udlod::terrain::TerrainComponents;
use thalos_udlod::terrain_data::gpu_tile_atlas::GpuTileAtlas;
use thalos_udlod::terrain_data::gpu_tile_tree::GpuTileTree;
use thalos_udlod::terrain_view::TerrainViewComponents;

use crate::shading::AtmosphereBlock;

use crate::ground::body_material::BodySkyExtra;

#[derive(Asset, TypePath, Clone, Default)]
pub struct BodySkyMaterial {
    pub atmosphere: AtmosphereBlock,
    pub atmosphere_extra: BodySkyExtra,
    /// Scene-depth texture written by the game crate's `CopySceneDepthNode`
    /// each frame between `Node3d::MainOpaquePass` and
    /// `Node3d::MainTransparentPass`. Sampled with `textureLoad` in the
    /// fragment shader to clip the atmosphere raymarch at opaque geometry,
    /// which is what produces aerial perspective on terrain pixels.
    pub scene_depth: Handle<Image>,
    /// Reference cloud-cover cubemap shared with the impostor material.
    /// Bodies without a registered overlay bind the same blank cube fallback.
    pub cloud_cover: Handle<Image>,
    /// Per-body multi-scatter LUT (32×32 `Rgba16Float`), baked once at spawn by
    /// `thalos_planet_lighting::bake_multi_scatter_lut` and never updated — the
    /// atmosphere parameters are static. `body_sky.wgsl` samples it at every
    /// view step via `integrate_atmosphere_multiscatter` to add the
    /// second-order in-scatter that single scattering omits. That term is what
    /// gives the daytime dome its blue luminance and lifts the in-scatter into
    /// the range where the sky-luminance alpha boost washes out stars at noon;
    /// without it the midday sky is physically dim and the celestial backdrop
    /// bleeds through. Indexed by `(u = (sun·zenith + 1) / 2, v = h / atmos_top)`.
    pub multi_scatter_lut: Handle<Image>,
    /// High-fidelity volumetric cloud layer rendered by `thalos_volumetric_clouds`
    /// (RGBA32F: rgb = premultiplied in-scatter, a = transmittance), composited
    /// over the atmosphere in-scatter as the final step of the fullscreen pass.
    /// Sampled with `textureLoad` (unfilterable float, no sampler). The game
    /// binds the live cloud texture for the active cloud body and a 1×1
    /// "clear" fallback (a = 1) for every other body, so the composite is a
    /// no-op where there are no clouds.
    pub cloud_layer: Handle<Image>,
    /// Per-pixel nearest cloud-hit distance from the same raymarch (R32F,
    /// metres from the camera; ≥ 1e8 sentinel = no cloud on the ray). Lets the
    /// composite occlude clouds against opaque geometry by true depth rather
    /// than by the geometric shell-band approximation. Bodies without an
    /// active cloud layer bind a 1×1 far-sentinel fallback.
    pub cloud_distance: Handle<Image>,
    /// Per-body coast/bathymetry cube (ADR-0005): signed terrain height about
    /// sea level, `R16Unorm`-encoded over `±COAST_ATLAS_HEIGHT_RANGE_M`, baked
    /// once at spawn by [`crate::bake_coast_bathymetry_cube`] from the same
    /// `SurfaceQuery` the tiles bake from. Since ADR-0006 it is the **coarse
    /// tail** of the sea-field cascade: the ocean branch samples the resident
    /// height tiles first and falls back to this cube where no tile is
    /// resident (cold streaming, terrain despawned, beyond the impostor swap).
    /// Airless / no-ocean bodies bind [`crate::blank_coast_cube`] and never
    /// sample it (`ocean.y` gates the branch).
    pub coast_atlas: Handle<Image>,
    /// Main-world entity of this body's udlod terrain (the one carrying
    /// `TileAtlas`), used to resolve the height-tile atlas + tile tree in
    /// `unprepared_bind_group`. `None` (or a stale entity after a terrain
    /// respawn, until the game's per-frame sky update refreshes it) binds
    /// fallbacks and force-disables the tile lookup in the uniform, so the
    /// shader falls back to the coast atlas. Sole writer: the game's
    /// `update_body_terrain_atmosphere`.
    pub terrain_entity: Option<Entity>,
}

impl AsBindGroup for BodySkyMaterial {
    type Data = ();
    type Param = (
        SRes<RenderAssets<GpuImage>>,
        SRes<FallbackImage>,
        Option<SRes<TerrainComponents<GpuTileAtlas>>>,
        Option<SRes<TerrainViewComponents<GpuTileTree>>>,
    );

    fn label() -> &'static str {
        "body_sky_material"
    }

    fn bind_group_data(&self) -> Self::Data {}

    fn unprepared_bind_group(
        &self,
        _layout: &BindGroupLayout,
        render_device: &RenderDevice,
        (images, fallback, gpu_atlases, gpu_tile_trees): &mut SystemParamItem<'_, '_, Self::Param>,
        _force_no_bindless: bool,
    ) -> Result<UnpreparedBindGroup, AsBindGroupError> {
        let image = |handle: &Handle<Image>| -> Result<&GpuImage, AsBindGroupError> {
            images.get(handle).ok_or(AsBindGroupError::RetryNextUpdate)
        };

        let scene_depth = image(&self.scene_depth)?;
        let cloud_cover = image(&self.cloud_cover)?;
        let multi_scatter_lut = image(&self.multi_scatter_lut)?;
        let cloud_layer = image(&self.cloud_layer)?;
        let cloud_distance = image(&self.cloud_distance)?;
        let coast_atlas = image(&self.coast_atlas)?;

        // Resolve this body's resident height-tile atlas + tile tree from
        // udlod's render-world registries. The tile tree is keyed per
        // `(terrain, view)`; terrain only streams for the ship camera, so the
        // first entry matching our terrain entity is the right one (if a
        // second terrain-streaming view ever appears, the lookup serves it
        // the ship camera's tree — coverage stays correct, resolution near
        // that view's ground point may be coarse).
        let tile_resources = self.terrain_entity.and_then(|terrain| {
            let atlas_texture = gpu_atlases
                .as_ref()?
                .get(&terrain)?
                .attachment_texture(0)?
                .clone();
            let gpu_tile_tree = gpu_tile_trees
                .as_ref()?
                .iter()
                .find(|((t, _view), _)| *t == terrain)
                .map(|(_, tree)| tree)?;
            Some((
                atlas_texture,
                gpu_tile_tree.tile_tree_buffer().clone(),
                gpu_tile_tree.origins_buffer().clone(),
            ))
        });

        // If the lookup resources are missing (no terrain, cold spawn frame,
        // respawn churn), bind fallbacks AND force the uniform's enable flag
        // off so the shader never samples them as heights.
        let mut extra = self.atmosphere_extra;
        if tile_resources.is_none() {
            extra.tile_lookup.x = 0.0;
        }

        let uniform = |bytes: &[u8]| {
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("body_sky_uniform"),
                usage: BufferUsages::UNIFORM,
                contents: bytes,
            })
        };
        let mut atmos_bytes = encase::UniformBuffer::new(Vec::new());
        atmos_bytes.write(&self.atmosphere).unwrap();
        let atmos_buffer = uniform(atmos_bytes.as_ref());
        let mut extra_bytes = encase::UniformBuffer::new(Vec::new());
        extra_bytes.write(&extra).unwrap();
        let extra_buffer = uniform(extra_bytes.as_ref());

        let (tile_atlas_view, tile_tree_buffer, origins_buffer) = match tile_resources {
            Some((texture, tree, origins)) => {
                (texture.create_view(&Default::default()), tree, origins)
            }
            None => {
                // 16-byte zeroed dummy satisfies both storage-buffer bindings;
                // the shader is gated off them by `tile_lookup.x = 0`.
                let dummy = render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("body_sky_tile_lookup_dummy"),
                    usage: BufferUsages::STORAGE,
                    contents: &[0u8; 16],
                });
                (
                    fallback.d2_array.texture_view.clone(),
                    dummy.clone(),
                    dummy,
                )
            }
        };

        // Linear min/mag/mip sampler for footprint-matched mip sampling of the
        // height atlas (the per-tile mip chains the providers bake).
        let tile_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("body_sky_tile_height_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });

        let bindings = vec![
            (0, OwnedBindingResource::Buffer(atmos_buffer)),
            (1, OwnedBindingResource::Buffer(extra_buffer)),
            (
                2,
                OwnedBindingResource::TextureView(
                    TextureViewDimension::D2,
                    scene_depth.texture_view.clone(),
                ),
            ),
            (
                3,
                OwnedBindingResource::TextureView(
                    TextureViewDimension::Cube,
                    cloud_cover.texture_view.clone(),
                ),
            ),
            (
                4,
                OwnedBindingResource::Sampler(
                    SamplerBindingType::Filtering,
                    cloud_cover.sampler.clone(),
                ),
            ),
            (
                5,
                OwnedBindingResource::TextureView(
                    TextureViewDimension::D2,
                    multi_scatter_lut.texture_view.clone(),
                ),
            ),
            (
                6,
                OwnedBindingResource::Sampler(
                    SamplerBindingType::Filtering,
                    multi_scatter_lut.sampler.clone(),
                ),
            ),
            (
                7,
                OwnedBindingResource::TextureView(
                    TextureViewDimension::D2,
                    cloud_layer.texture_view.clone(),
                ),
            ),
            (
                8,
                OwnedBindingResource::TextureView(
                    TextureViewDimension::D2,
                    cloud_distance.texture_view.clone(),
                ),
            ),
            (
                9,
                OwnedBindingResource::TextureView(
                    TextureViewDimension::Cube,
                    coast_atlas.texture_view.clone(),
                ),
            ),
            (
                10,
                OwnedBindingResource::Sampler(
                    SamplerBindingType::Filtering,
                    coast_atlas.sampler.clone(),
                ),
            ),
            (
                11,
                OwnedBindingResource::TextureView(TextureViewDimension::D2Array, tile_atlas_view),
            ),
            (
                12,
                OwnedBindingResource::Sampler(SamplerBindingType::Filtering, tile_sampler),
            ),
            (13, OwnedBindingResource::Buffer(tile_tree_buffer)),
            (14, OwnedBindingResource::Buffer(origins_buffer)),
        ];

        Ok(UnpreparedBindGroup {
            bindings: BindingResources(bindings),
        })
    }

    fn bind_group_layout_entries(
        _render_device: &RenderDevice,
        _force_no_bindless: bool,
    ) -> Vec<BindGroupLayoutEntry> {
        // Bindings 0–10 mirror what the `AsBindGroup` derive generated before
        // the manual impl (same order, types, and all-stages visibility);
        // 11–14 are the ADR-0006 height-tile lookup.
        BindGroupLayoutEntries::with_indices(
            ShaderStages::all(),
            (
                (0, uniform_buffer::<AtmosphereBlock>(false)),
                (1, uniform_buffer::<BodySkyExtra>(false)),
                (2, texture_depth_2d()),
                (3, texture_cube(TextureSampleType::Float { filterable: true })),
                (4, sampler(SamplerBindingType::Filtering)),
                (5, texture_2d(TextureSampleType::Float { filterable: true })),
                (6, sampler(SamplerBindingType::Filtering)),
                (
                    7,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
                (
                    8,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
                (9, texture_cube(TextureSampleType::Float { filterable: true })),
                (10, sampler(SamplerBindingType::Filtering)),
                (
                    11,
                    texture_2d_array(TextureSampleType::Float { filterable: true }),
                ),
                (12, sampler(SamplerBindingType::Filtering)),
                (13, storage_buffer_read_only_sized(false, None)),
                (14, storage_buffer_read_only_sized(false, None)),
            ),
        )
        .to_vec()
    }
}

impl Material for BodySkyMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/body_sky.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/body_sky.wgsl".into()
    }

    // Premultiplied: `rgb = in_scatter` is additive over the background, and
    // `(1 − alpha)` = mean atmospheric transmittance dims whatever was behind
    // (stars, galaxies, terrain). The fullscreen pass now overdraws onto
    // terrain too (depth_compare = Always) so its in-scatter and
    // transmittance composite uniformly across the frame.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Premultiplied
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Fullscreen quad — no culling.
        descriptor.primitive.cull_mode = None;
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            // The terrain atmosphere pass renders on every pixel and clips
            // the raymarch with sampled scene depth instead of via
            // depth-compare. Disable both depth write and depth test.
            depth.depth_write_enabled = Some(false);
            depth.depth_compare = Some(CompareFunction::Always);
        }
        Ok(())
    }
}

pub(crate) fn embed_body_sky_shader(app: &mut App) {
    embedded_asset!(app, "body_sky.wgsl");
}
