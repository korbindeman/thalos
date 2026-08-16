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
//! ## Shared optical `AsBindGroup` (ADR-20260720T185958Z-water-projects-one-signed-sea-field)
//!
//! [`BodyOceanMaterial`](super::BodyOceanMaterial) delegates this binding
//! implementation so the analytic-ocean branch samples **signed sea height straight from the
//! udlod height-tile atlas** — the exact texels the visible terrain mesh is
//! displaced from — so water coverage/colour are a projection of the one
//! terrain field instead of a depth-buffer comparison. Those resources
//! (attachment-0 texture array, tile-tree + origins storage buffers) live in
//! udlod's render-world registries, not in `Assets`, so the derive can't bind
//! them: this shared optical contract implements `AsBindGroup` by hand, keeping the derive's
//! exact layout for bindings 0–6, appending the tile lookup at 7–10, and
//! binding the shared mipmapped ocean slope field at 11–12.
//! The material is mutated every frame by the game's sky update, so the bind
//! group re-prepares every frame and the lookup never goes stale across
//! terrain despawn/respawn.

use bevy::asset::embedded_asset;
use bevy::ecs::system::SystemParamItem;
use bevy::ecs::system::lifetimeless::SRes;
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
    AsBindGroup, AsBindGroupError, BindGroupLayout, BindGroupLayoutEntries, BindGroupLayoutEntry,
    BindingResources, BufferInitDescriptor, BufferUsages, CompareFunction, FilterMode,
    MipmapFilterMode, OwnedBindingResource, RenderPipelineDescriptor, SamplerBindingType,
    SamplerDescriptor, ShaderStages, SpecializedMeshPipelineError, TextureSampleType,
    TextureViewDimension, UnpreparedBindGroup, encase,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::{FallbackImage, GpuImage};
use bevy::shader::ShaderRef;

use crate::clouds::CloudShadowBlock;
use crate::shading::AtmosphereBlock;

use crate::ground::body_material::BodySkyExtra;

#[derive(Asset, TypePath, Clone, Default)]
pub struct BodySkyMaterial {
    pub atmosphere: AtmosphereBlock,
    pub atmosphere_extra: BodySkyExtra,
    /// Scene-depth texture written by `thalos_render_foundation` each frame
    /// between Bevy's main opaque and transparent passes. Sampled with
    /// `textureLoad` in the
    /// fragment shader to clip the atmosphere raymarch at opaque geometry,
    /// which is what produces aerial perspective on terrain pixels.
    pub scene_depth: Handle<Image>,
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
    /// Per-body coast/bathymetry cube (ADR-20260720T185957Z-coastline-as-authored-data): signed terrain height about
    /// sea level, `R16Unorm`-encoded over `±COAST_ATLAS_HEIGHT_RANGE_M`, baked
    /// once at spawn by [`crate::bake_coast_bathymetry_cube`] from the same
    /// `SurfaceQuery` the tiles bake from. Since ADR-20260720T185958Z-water-projects-one-signed-sea-field it is the **coarse
    /// tail** of the sea-field cascade: the ocean branch samples the resident
    /// height tiles first and falls back to this cube where no tile is
    /// resident (cold streaming, terrain despawned, beyond the impostor swap).
    /// Airless / no-ocean bodies bind [`crate::blank_coast_cube`] and never
    /// sample it (`ocean.y` gates the branch).
    pub coast_atlas: Handle<Image>,
    /// Shared periodic broadband ocean slopes. The sky shader samples this at
    /// multiple physical scales with explicit anisotropic gradients, so
    /// grazing views retain cross-wave detail while filtering only the
    /// foreshortened direction. RG and BA contain independent spectra.
    pub ocean_slope: Handle<Image>,
    /// Main-world entity of this body's udlod terrain (the one carrying
    /// `TileAtlas`), used to resolve the height-tile atlas + tile tree in
    /// `unprepared_bind_group`. `None` (or a stale entity after a terrain
    /// respawn, until the game's per-frame sky update refreshes it) binds
    /// fallbacks and force-disables the tile lookup in the uniform, so the
    /// shader falls back to the coast atlas. Sole writer: the game's
    /// `update_body_terrain_atmosphere`.
    pub terrain_entity: Option<Entity>,
    /// Cloud sun-transmittance cascade lookup frame (CLOUD-5 §3.5 atmosphere
    /// shafts): the atmosphere raymarch gates its per-sample sun term by this
    /// so the air under a cloud gap is a bright shaft and the air downwind of
    /// a cell is a dark one — the same field, and therefore the same weather,
    /// every surface receiver shades by. A default (zeroed) block stands the
    /// term down; only the active cloud body's sky carries a live one. Sole
    /// writer: the game's `update_body_terrain_atmosphere`.
    pub cloud_shadow: CloudShadowBlock,
    /// The cascade texture behind [`cloud_shadow`](Self::cloud_shadow)
    /// (`CloudShadowMap::handle`). An unresolvable handle binds a fallback and
    /// zeroes the block, so a missing cascade can never darken the sky.
    pub cloud_shadow_map: Handle<Image>,
}

impl AsBindGroup for BodySkyMaterial {
    type Data = ();
    type Param = (SRes<RenderAssets<GpuImage>>, SRes<FallbackImage>);

    fn label() -> &'static str {
        "body_sky_material"
    }

    fn bind_group_data(&self) -> Self::Data {}

    fn unprepared_bind_group(
        &self,
        _layout: &BindGroupLayout,
        render_device: &RenderDevice,
        (images, fallback): &mut SystemParamItem<'_, '_, Self::Param>,
        _force_no_bindless: bool,
    ) -> Result<UnpreparedBindGroup, AsBindGroupError> {
        let image = |handle: &Handle<Image>| -> Result<&GpuImage, AsBindGroupError> {
            images.get(handle).ok_or(AsBindGroupError::RetryNextUpdate)
        };

        let scene_depth = image(&self.scene_depth)?;
        let multi_scatter_lut = image(&self.multi_scatter_lut)?;
        let coast_atlas = image(&self.coast_atlas)?;
        let ocean_slope = image(&self.ocean_slope)?;
        // Optional by design: bodies without an active cloud cascade (or a
        // boot frame before the clouds plugin uploads it) bind a fallback and
        // zero the block — never RetryNextUpdate, which would hold the whole
        // sky hostage to a cloud resource.
        let cloud_shadow_image = images.get(&self.cloud_shadow_map);
        let mut cloud_shadow = self.cloud_shadow;
        if cloud_shadow_image.is_none() {
            cloud_shadow = CloudShadowBlock::default();
        }

        // The signed-sea-field height lookup (ADR-20260720T185958Z) used to
        // resolve this body's resident height atlas + tile tree out of udlod's
        // render-world registries. That binding is **gone** (2026-07-26): udlod
        // is off the default build, and it had already stopped resolving for
        // every tile-rendered body — no udlod terrain entity means no atlas, so
        // the default renderer has been taking the fallback below since tiles
        // became the default ground.
        //
        // Every body now takes it, so the ocean's coverage comes from the coarse
        // coast-atlas tail rather than the resident heightfield. The bindings
        // stay in the layout (the shader still declares them) bound to
        // fallbacks, with `tile_lookup.x = 0` gating the shader off them — the
        // same shape the `None` branch always had. Restoring close-range
        // coastline fidelity means giving this an equivalent source from the
        // tile renderer's height mirror, not resurrecting udlod.
        let tile_resources: Option<(
            bevy::render::render_resource::Texture,
            bevy::render::render_resource::Buffer,
            bevy::render::render_resource::Buffer,
        )> = None;

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
        let mut cloud_shadow_bytes = encase::UniformBuffer::new(Vec::new());
        cloud_shadow_bytes.write(&cloud_shadow).unwrap();
        let cloud_shadow_buffer = uniform(cloud_shadow_bytes.as_ref());

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
                (fallback.d2_array.texture_view.clone(), dummy.clone(), dummy)
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
                    TextureViewDimension::D2,
                    multi_scatter_lut.texture_view.clone(),
                ),
            ),
            (
                4,
                OwnedBindingResource::Sampler(
                    SamplerBindingType::Filtering,
                    multi_scatter_lut.sampler.clone(),
                ),
            ),
            (
                5,
                OwnedBindingResource::TextureView(
                    TextureViewDimension::Cube,
                    coast_atlas.texture_view.clone(),
                ),
            ),
            (
                6,
                OwnedBindingResource::Sampler(
                    SamplerBindingType::Filtering,
                    coast_atlas.sampler.clone(),
                ),
            ),
            (
                7,
                OwnedBindingResource::TextureView(TextureViewDimension::D2Array, tile_atlas_view),
            ),
            (
                8,
                OwnedBindingResource::Sampler(SamplerBindingType::Filtering, tile_sampler),
            ),
            (9, OwnedBindingResource::Buffer(tile_tree_buffer)),
            (10, OwnedBindingResource::Buffer(origins_buffer)),
            (
                11,
                OwnedBindingResource::TextureView(
                    TextureViewDimension::D2,
                    ocean_slope.texture_view.clone(),
                ),
            ),
            (
                12,
                OwnedBindingResource::Sampler(
                    SamplerBindingType::Filtering,
                    ocean_slope.sampler.clone(),
                ),
            ),
            (13, OwnedBindingResource::Buffer(cloud_shadow_buffer)),
            (
                14,
                OwnedBindingResource::TextureView(
                    TextureViewDimension::D2,
                    cloud_shadow_image
                        .map(|gpu| gpu.texture_view.clone())
                        .unwrap_or_else(|| fallback.d2.texture_view.clone()),
                ),
            ),
            (
                15,
                OwnedBindingResource::Sampler(
                    SamplerBindingType::Filtering,
                    cloud_shadow_image
                        .map(|gpu| gpu.sampler.clone())
                        .unwrap_or_else(|| fallback.d2.sampler.clone()),
                ),
            ),
        ];

        Ok(UnpreparedBindGroup {
            bindings: BindingResources(bindings),
        })
    }

    fn bind_group_layout_entries(
        _render_device: &RenderDevice,
        _force_no_bindless: bool,
    ) -> Vec<BindGroupLayoutEntry> {
        // Bindings 0–6 are the atmosphere/ocean material resources; 7–10 are
        // the ADR-20260720T185958Z height-tile lookup, and 11–12 are the
        // shared mipmapped ocean-slope field.
        BindGroupLayoutEntries::with_indices(
            ShaderStages::all(),
            (
                (0, uniform_buffer::<AtmosphereBlock>(false)),
                (1, uniform_buffer::<BodySkyExtra>(false)),
                (2, texture_depth_2d()),
                (3, texture_2d(TextureSampleType::Float { filterable: true })),
                (4, sampler(SamplerBindingType::Filtering)),
                (
                    5,
                    texture_cube(TextureSampleType::Float { filterable: true }),
                ),
                (6, sampler(SamplerBindingType::Filtering)),
                (
                    7,
                    texture_2d_array(TextureSampleType::Float { filterable: true }),
                ),
                (8, sampler(SamplerBindingType::Filtering)),
                (9, storage_buffer_read_only_sized(false, None)),
                (10, storage_buffer_read_only_sized(false, None)),
                (
                    11,
                    texture_2d(TextureSampleType::Float { filterable: true }),
                ),
                (12, sampler(SamplerBindingType::Filtering)),
                // 13–15: cloud sun-transmittance cascade (CLOUD-5 §3.5
                // atmosphere shafts) — lookup-frame block, map, sampler.
                (13, uniform_buffer::<CloudShadowBlock>(false)),
                (
                    14,
                    texture_2d(TextureSampleType::Float { filterable: true }),
                ),
                (15, sampler(SamplerBindingType::Filtering)),
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

    fn depth_bias(&self) -> f32 {
        crate::composite_order::ATMOSPHERE
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Fullscreen quad — no culling.
        descriptor.primitive.cull_mode = None;
        if let Some(fragment) = descriptor.fragment.as_mut() {
            // The analytic ocean is owned by `BodyOceanMaterial`; this material
            // is now exclusively the legacy atmosphere A/B.
            fragment.shader_defs.push("ATMOSPHERE_ONLY".into());
        }
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
