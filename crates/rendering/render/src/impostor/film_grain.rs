//! Film grain post-process pass.
//!
//! Runs after CAS (so sharpening doesn't amplify the noise) and before the
//! end of the main post stack. Animated, luma-weighted, monochromatic.

use bevy::core_pipeline::FullscreenShader;
// Bevy 0.19: the node-based render graph became systems in the `Core3d`
// schedule, ordered by `Core3dSystems` sets. Film grain runs after the whole
// `PostProcess` set and before `upscaling` (was: the `Node3d`
// CAS → FilmGrain → EndMainPassPostProcessing edges, whose CAS / end-marker
// nodes no longer exist).
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_resource::{
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedRenderPipelineId, ColorTargetState, ColorWrites, DynamicUniformBuffer, FilterMode,
        FragmentState, MipmapFilterMode, Operations, PipelineCache, RenderPassColorAttachment,
        RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType,
        SamplerDescriptor, ShaderStages, ShaderType, SpecializedRenderPipeline,
        SpecializedRenderPipelines, TextureFormat, TextureSampleType,
        binding_types::{sampler, texture_2d, uniform_buffer},
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    sync_component::SyncComponent,
    view::{ExtractedView, ViewTarget},
};
use bevy::shader::Shader;
use bevy::transform::TransformSystems;

/// Tunable film grain component. Attach to a camera alongside the rest of the
/// post stack. `FilmGrainState` is auto-added as a required component to
/// track stillness.
#[derive(Component, Clone, Copy, Debug)]
#[require(FilmGrainState)]
pub struct FilmGrain {
    /// Peak additive amplitude in linear-ish LDR units. 0.008 ≈ 0.8%.
    pub intensity: f32,
    /// Floor of grain retained in mid/high tones (0 = grain only in darks,
    /// 1 = flat grain everywhere).
    pub shadow_bias: f32,
    /// Fraction of peak grain retained after camera stays still long
    /// enough. 0.3 = grain settles to 30% of intensity.
    pub stillness_floor: f32,
    /// Seconds of camera-still time required to reach the floor.
    pub settle_seconds: f32,
}

impl Default for FilmGrain {
    fn default() -> Self {
        Self {
            intensity: 0.010,
            shadow_bias: 0.02,
            stillness_floor: 0.3,
            settle_seconds: 2.5,
        }
    }
}

/// Runtime state updated by `update_film_grain_stillness`. Tracks camera
/// forward vector across frames and smoothly fades grain toward the floor
/// when the view direction stops changing.
#[derive(Component, Clone, Copy, Debug)]
pub struct FilmGrainState {
    prev_forward: Vec3,
    still_elapsed: f32,
    factor: f32,
    elapsed_s: f32,
}

impl Default for FilmGrainState {
    fn default() -> Self {
        Self {
            prev_forward: Vec3::ZERO,
            still_elapsed: 0.0,
            factor: 1.0,
            elapsed_s: 0.0,
        }
    }
}

/// Extracted into the render world with `intensity` already scaled by
/// stillness so the render side stays unaware of main-world state.
#[derive(Component, Clone, Copy, Debug)]
pub struct ExtractedFilmGrain {
    intensity: f32,
    shadow_bias: f32,
    time: f32,
}

// 0.19: `ExtractComponent` now requires `SyncComponent` (the component must be
// synced to the render world). The derive macro emits this automatically; our
// hand-written `ExtractComponent` impl needs it spelled out. `Target = Self`
// matches the trait's intended default.
impl SyncComponent for FilmGrain {
    type Target = Self;
}

impl ExtractComponent for FilmGrain {
    type QueryData = (&'static FilmGrain, &'static FilmGrainState);
    type QueryFilter = ();
    type Out = ExtractedFilmGrain;

    fn extract_component((grain, state): QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        let scaled = grain.intensity * state.factor;
        if scaled > 0.0 {
            Some(ExtractedFilmGrain {
                intensity: scaled,
                shadow_bias: grain.shadow_bias,
                time: state.elapsed_s,
            })
        } else {
            None
        }
    }
}

#[derive(ShaderType, Clone, Copy)]
struct FilmGrainUniform {
    intensity: f32,
    time: f32,
    shadow_bias: f32,
    _pad: f32,
}

#[derive(Resource, Default)]
struct FilmGrainUniformBuffer(DynamicUniformBuffer<FilmGrainUniform>);

#[derive(Component)]
struct FilmGrainUniformOffset(u32);

#[derive(Resource)]
struct FilmGrainPipeline {
    layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    fullscreen_shader: FullscreenShader,
    fragment_shader: Handle<Shader>,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct FilmGrainPipelineKey {
    texture_format: TextureFormat,
}

#[derive(Component)]
struct FilmGrainPipelineId(CachedRenderPipelineId);

pub(crate) struct FilmGrainPlugin;

impl Plugin for FilmGrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<FilmGrain>::default())
            .add_systems(
                PostUpdate,
                update_film_grain_stillness.after(TransformSystems::Propagate),
            );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<FilmGrainUniformBuffer>()
            .init_resource::<SpecializedRenderPipelines<FilmGrainPipeline>>()
            .add_systems(RenderStartup, init_film_grain_pipeline)
            .add_systems(
                Render,
                (prepare_film_grain_pipelines, prepare_film_grain_uniforms)
                    .in_set(RenderSystems::Prepare),
            )
            // 0.19: render-pass system in the `Core3d` schedule. It must be the
            // LAST effect *inside* the `PostProcess` set: after the AA/effect
            // chain (chromatic-aberration → tonemapping → SMAA → `cas`, so
            // `.after(cas)` covers all of them — `cas` is `.after(smaa)`), so the
            // scene is fully processed before grain; but still *in* `PostProcess`,
            // because the UI overlay + `upscaling` are ordered `.after(PostProcess)`
            // and read the parity-tracked `ViewTarget` main texture. Ordering film
            // grain only `.after(tonemapping)` left it mid-chain (SMAA/CAS flipped
            // the ping-pong after it → the scene's two-state flicker); ordering it
            // `.after(PostProcess)` put its `post_process_write()` flip in the same
            // bucket as the UI/upscaling consumers → raced them (the UI/lens-flare
            // flicker). `.in_set(PostProcess).after(cas)` is the one position that
            // satisfies both — matching the 0.18 `CAS → FilmGrain → end` node order.
            .add_systems(
                Core3d,
                film_grain_pass
                    .in_set(Core3dSystems::PostProcess)
                    .after(bevy::anti_alias::contrast_adaptive_sharpening::cas),
            );
    }
}

fn init_film_grain_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "film_grain_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<FilmGrainUniform>(true),
            ),
        ),
    );

    let sampler = render_device.create_sampler(&SamplerDescriptor {
        // wgpu 29 split the mipmap filter into its own `MipmapFilterMode`.
        mipmap_filter: MipmapFilterMode::Linear,
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        ..default()
    });

    commands.insert_resource(FilmGrainPipeline {
        layout,
        sampler,
        fullscreen_shader: fullscreen_shader.clone(),
        fragment_shader: asset_server.load("shaders/film_grain.wgsl"),
    });
}

impl SpecializedRenderPipeline for FilmGrainPipeline {
    type Key = FilmGrainPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("film_grain".into()),
            layout: vec![self.layout.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format: key.texture_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

fn update_film_grain_stillness(
    time: Res<Time<Real>>,
    mut q: Query<(&GlobalTransform, &FilmGrain, &mut FilmGrainState)>,
) {
    let dt = time.delta_secs();
    for (xf, grain, mut state) in &mut q {
        state.elapsed_s += dt;
        let fwd = xf.forward().as_vec3();
        // Cosine distance between last and current forward vector. Rotation-
        // only: position drift from following an orbiting ship shouldn't
        // prevent settling.
        let rot_delta = 1.0 - fwd.dot(state.prev_forward).clamp(-1.0, 1.0);
        if rot_delta > 1.0e-6 {
            state.still_elapsed = 0.0;
        } else {
            state.still_elapsed += dt;
        }
        state.prev_forward = fwd;

        let t = (state.still_elapsed / grain.settle_seconds.max(1.0e-3)).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        state.factor = 1.0 - smooth * (1.0 - grain.stillness_floor);
    }
}

fn prepare_film_grain_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<FilmGrainPipeline>>,
    pipeline: Res<FilmGrainPipeline>,
    views: Query<(Entity, &ExtractedView), With<ExtractedFilmGrain>>,
) {
    for (entity, view) in views.iter() {
        let id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            FilmGrainPipelineKey {
                // 0.19: `ExtractedView::hdr` → `target_format` (HDR = Rgba16Float).
                texture_format: view.target_format,
            },
        );
        commands.entity(entity).insert(FilmGrainPipelineId(id));
    }
}

fn prepare_film_grain_uniforms(
    mut commands: Commands,
    mut buffer: ResMut<FilmGrainUniformBuffer>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    views: Query<(Entity, &ExtractedFilmGrain)>,
) {
    buffer.0.clear();
    for (entity, grain) in views.iter() {
        let offset = buffer.0.push(&FilmGrainUniform {
            intensity: grain.intensity,
            time: grain.time,
            shadow_bias: grain.shadow_bias,
            _pad: 0.0,
        });
        commands
            .entity(entity)
            .insert(FilmGrainUniformOffset(offset));
    }
    buffer.0.write_buffer(&render_device, &render_queue);
}

/// Film-grain post-process pass, ported from the former `FilmGrainNode`
/// (`ViewNode`) to a Bevy 0.19 render-pass **system**. The `ViewQuery` fetches
/// the current view's components and auto-skips views that lack them (the map
/// camera, light/shadow views), so the pass only runs on cameras carrying
/// `FilmGrain`.
fn film_grain_pass(
    view: ViewQuery<(
        &'static ViewTarget,
        &'static FilmGrainPipelineId,
        &'static ExtractedFilmGrain,
        &'static FilmGrainUniformOffset,
    )>,
    pipeline_cache: Res<PipelineCache>,
    pipeline_res: Res<FilmGrainPipeline>,
    uniforms: Res<FilmGrainUniformBuffer>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (view_target, pipeline_id, _grain, offset) = view.into_inner();

    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
        return;
    };
    let Some(binding) = uniforms.0.binding() else {
        return;
    };

    let post_process = view_target.post_process_write();

    let bind_group = render_device.create_bind_group(
        Some("film_grain_bind_group"),
        &pipeline_cache.get_bind_group_layout(&pipeline_res.layout),
        &BindGroupEntries::sequential((post_process.source, &pipeline_res.sampler, binding)),
    );

    let mut pass = ctx
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("film_grain_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            // Added by wgpu 29 / bevy 0.19.
            multiview_mask: None,
        });

    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[offset.0]);
    pass.draw(0..3, 0..1);
}
