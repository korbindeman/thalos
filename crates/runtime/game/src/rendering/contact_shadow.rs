//! Screen-space contact shadows (graphics-fidelity W18a).
//!
//! The **contact tier** of the three-tier shadow split
//! (ADR-20260722T111848Z-shadows-three-tier-not-virtual-shadow-maps). Cascade 0
//! is a 400 m half-extent at 4096² ≈ 0.2 m/texel — coarser than a landing-gear
//! strut, a tree trunk, or a building's ground seam — so the 0–50 m band is a
//! regime the cascade rig structurally cannot serve, and craft/gear/trunks read
//! as pasted onto the ground until something else covers it.
//!
//! This pass reads the current frame's Bevy depth prepass, runs in view space
//! so it is f32-safe under big_space, and writes immediately before opaque
//! shading. The default tile ground therefore receives the contact term in the
//! same frame as the depth that produced it.
//!
//! Two deliberate differences from SSAO:
//!
//! - **Full resolution.** AO is low-frequency and upsamples cleanly; a contact
//!   shadow is high-frequency, and its casters are a few pixels wide, so half-res
//!   would alias away the detail the pass exists to produce.
//! - **No blur.** Blurring a contact shadow re-softens the hard near-contact edge
//!   that is the entire point. The march is jittered per-pixel instead.
//!
//! The gate rides in [`ShadowCascadeBlock::gate`]`.z` (a slot the block already
//! reserved) rather than a per-material flag, so it reaches every one of the
//! rig's consumers through the binding they already carry.

use crate::camera::ShipCamera;
use bevy::asset::{Assets, Handle, RenderAssetUsages};
use bevy::camera::Camera;
use bevy::core_pipeline::core_3d::main_opaque_pass_3d;
use bevy::core_pipeline::prepass::ViewPrepassTextures;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::ecs::prelude::*;
use bevy::image::Image;
use bevy::math::{Mat4, Vec3, Vec4};
use bevy::prelude::*;
use bevy::render::{
    RenderApp, RenderStartup,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::{binding_types::*, encase, *},
    renderer::{RenderContext, RenderDevice, ViewQuery},
    texture::GpuImage,
    view::ExtractedView,
};

/// Contact-shadow tuning, extracted to the render world each frame. Like
/// [`super::ssao::SsaoConfig`], the defaults are a calibration anchor — this is
/// verified by eye (objects grounded, no self-shadow acne on flat ground, no
/// dark halo trailing away from casters).
#[derive(Resource, Clone, Copy, ExtractResource)]
pub struct ContactShadowConfig {
    pub enabled: bool,
    /// Diagnostic: paint the raw contact term on receivers instead of shading
    /// it in (`THALOS_CONTACT_SHADOW=show`). Splits "the pass is wrong" from
    /// "the receiver applies it wrong" in one screenshot.
    pub debug_show: bool,
    /// March length in view metres. Sub-metre by design: this is the band the
    /// cascades miss, not a general screen-space shadow.
    pub reach: f32,
    /// Occluder thickness in view metres. Depth is a heightfield, not a solid,
    /// so an occluder only counts when the ray passes just behind it — without
    /// this, background geometry shadows everything drawn in front of it.
    pub thickness: f32,
    /// Shadow strength in [0,1].
    pub strength: f32,
    /// Receiver normal offset in view metres, widened as the sun grazes.
    pub normal_bias: f32,
    /// View distance (metres) at which the term has fully faded out. Past this
    /// the reach is sub-pixel and the cascades own the shadow anyway.
    pub fade_end: f32,
}

impl Default for ContactShadowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            debug_show: false,
            reach: 0.6,
            thickness: 0.5,
            strength: 1.0,
            normal_bias: 0.02,
            fade_end: 120.0,
        }
    }
}

impl ContactShadowConfig {
    /// Read the `THALOS_CONTACT_SHADOW` session override: `off`/`0`/`false`
    /// disables the pass entirely, `show` paints the raw term on receivers,
    /// anything else / unset = on.
    fn from_env() -> Self {
        let mode = std::env::var("THALOS_CONTACT_SHADOW")
            .unwrap_or_default()
            .to_ascii_lowercase();
        match mode.as_str() {
            "off" | "0" | "false" => Self {
                enabled: false,
                ..Default::default()
            },
            "show" | "debug" => Self {
                debug_show: true,
                ..Default::default()
            },
            _ => Self::default(),
        }
    }

    /// Apply a capture-only diagnostic mode without rebuilding the renderer —
    /// the `shadow` comparison axis (BL-37) drives this between requests, the
    /// same way [`super::ssao::SsaoConfig::apply_capture_mode`] serves `ssao`.
    pub(crate) fn apply_capture_mode(&mut self, mode: Option<&str>) {
        let selected = match mode
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "off" | "0" | "false" => Self {
                enabled: false,
                ..Default::default()
            },
            "show" | "debug" => Self {
                debug_show: true,
                ..Default::default()
            },
            _ => Self::default(),
        };
        self.enabled = selected.enabled;
        self.debug_show = selected.debug_show;
    }

    /// The receiver gate carried in [`ShadowCascadeBlock::gate`]`.z`:
    /// 0 = skip, 1 = apply, 2 = paint raw (debug).
    pub fn shadow_gate(&self) -> f32 {
        if !self.enabled {
            0.0
        } else if self.debug_show {
            2.0
        } else {
            1.0
        }
    }
}

/// Full-resolution contact-shadow target (**R16Float**, 1 = fully lit). White
/// until the first pass runs, so it never over-darkens.
///
/// R16Float rather than R8Unorm for the same reason [`super::ssao::AoImage`]
/// uses it: the distance-softened falloff is a smooth gradient, and 8 bits
/// posterize it into visible steps across a receding ground plane.
#[derive(Resource, Clone, ExtractResource)]
pub struct ContactShadowImage {
    pub handle: Handle<Image>,
}

/// Render-space direction toward the sun, mirrored from the shadow rig so the
/// render world can rotate it into view space. Separate from
/// [`super::sun_shadow::SunShadowState`] because that resource is main-world
/// only (it carries `Handle`s and the full cascade block).
///
/// **Sole writer:** [`sync_contact_shadow_sun`].
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
pub struct ContactShadowSun {
    /// Normalized render-space direction toward the sun; zero when unavailable.
    pub dir: Vec3,
}

const CONTACT_FORMAT: TextureFormat = TextureFormat::R16Float;

/// Std140 uniform for `contact_shadow.wgsl`. Field order + widths must match.
#[derive(ShaderType)]
struct ContactUniform {
    view_from_clip: Mat4,
    clip_from_view: Mat4,
    /// Target size in px (xy); zw padding.
    target_res: Vec4,
    /// x = reach, y = thickness, z = strength, w = normal bias.
    params: Vec4,
    /// xyz = normalized view-space direction toward the sun, w = fade-out distance.
    sun_view: Vec4,
}

pub struct ContactShadowPlugin;

impl Plugin for ContactShadowPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ContactShadowConfig::from_env())
            .init_resource::<ContactShadowSun>()
            .add_plugins(ExtractResourcePlugin::<ContactShadowConfig>::default())
            .add_plugins(ExtractResourcePlugin::<ContactShadowImage>::default())
            .add_plugins(ExtractResourcePlugin::<ContactShadowSun>::default())
            .add_systems(Startup, setup_contact_shadow_image)
            .add_systems(
                Update,
                (resize_contact_shadow_image, sync_contact_shadow_sun),
            );

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_systems(RenderStartup, init_contact_shadow_pipeline)
                .add_systems(
                    Core3d,
                    // Current-frame contact: prepass depth is complete, and the
                    // tile receiver has not shaded yet.
                    compute_contact_shadow
                        .in_set(Core3dSystems::MainPass)
                        .after(Core3dSystems::Prepass)
                        .before(main_opaque_pass_3d),
                );
        }
    }
}

/// 1×1 white placeholder; `resize_contact_shadow_image` grows it to the full
/// viewport once the camera reports a size.
fn setup_contact_shadow_image(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    // f16 1.0 (LE) = white = fully lit.
    let mut image = Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![0x00, 0x3C],
        CONTACT_FORMAT,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
    let handle = images.add(image);
    commands.insert_resource(ContactShadowImage { handle });
}

/// Keep the target at the ship camera's **full** viewport (see the module note
/// on why this pass does not go half-res).
fn resize_contact_shadow_image(
    contact: Option<Res<ContactShadowImage>>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<&Camera, With<ShipCamera>>,
) {
    let Some(contact) = contact else {
        return;
    };
    let Ok(camera) = cameras.single() else {
        return;
    };
    let Some(viewport) = camera.physical_viewport_size() else {
        return;
    };
    if viewport.x == 0 || viewport.y == 0 {
        return;
    }
    let full = Extent3d {
        width: viewport.x.max(1),
        height: viewport.y.max(1),
        depth_or_array_layers: 1,
    };
    let Some(mut image) = images.get_mut(&contact.handle) else {
        return;
    };
    if image.texture_descriptor.size != full {
        image.resize(full);
    }
}

/// Mirror the shadow rig's sun direction into the extractable
/// [`ContactShadowSun`]. Reads the same `sun_dir` the cascade block publishes,
/// so the contact tier and the cascade tier can never disagree about where the
/// sun is.
fn sync_contact_shadow_sun(
    state: Option<Res<super::sun_shadow::SunShadowState>>,
    mut sun: ResMut<ContactShadowSun>,
) {
    let Some(state) = state else {
        return;
    };
    sun.dir = state.block.sun_dir.truncate();
}

/// Render-world pipeline for the contact-shadow fullscreen pass.
#[derive(Resource)]
struct ContactShadowPipeline {
    layouts: [BindGroupLayoutDescriptor; 2],
    pipeline_ids: [CachedRenderPipelineId; 2],
}

fn init_contact_shadow_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let single_layout = BindGroupLayoutDescriptor::new(
        "contact_shadow_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (texture_depth_2d(), uniform_buffer::<ContactUniform>(false)),
        ),
    );
    let msaa_layout = BindGroupLayoutDescriptor::new(
        "contact_shadow_msaa_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_depth_2d_multisampled(),
                uniform_buffer::<ContactUniform>(false),
            ),
        ),
    );
    let shader: Handle<Shader> = asset_server.load("shaders/contact_shadow.wgsl");
    let make_pipeline = |layout: &BindGroupLayoutDescriptor, msaa: bool| {
        let defs = if msaa {
            vec!["CONTACT_DEPTH_MSAA".into()]
        } else {
            vec![]
        };
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some(
                if msaa {
                    "contact_shadow_msaa_pipeline"
                } else {
                    "contact_shadow_pipeline"
                }
                .into(),
            ),
            layout: vec![layout.clone()],
            immediate_size: 0,
            vertex: VertexState {
                shader: shader.clone(),
                shader_defs: defs.clone(),
                entry_point: Some("vertex".into()),
                buffers: vec![],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader: shader.clone(),
                shader_defs: defs,
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: CONTACT_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            zero_initialize_workgroup_memory: false,
        })
    };
    let single_pipeline = make_pipeline(&single_layout, false);
    let msaa_pipeline = make_pipeline(&msaa_layout, true);

    commands.insert_resource(ContactShadowPipeline {
        layouts: [single_layout, msaa_layout],
        pipeline_ids: [single_pipeline, msaa_pipeline],
    });
}

/// March current prepass depth toward the sun and write the contact-shadow
/// factor. Runs once for the ship-camera view.
fn compute_contact_shadow(
    view: ViewQuery<(
        &'static ExtractedView,
        &'static ShipCamera,
        &'static ViewPrepassTextures,
    )>,
    config: Res<ContactShadowConfig>,
    sun: Res<ContactShadowSun>,
    contact: Option<Res<ContactShadowImage>>,
    pipeline: Option<Res<ContactShadowPipeline>>,
    render_assets: Res<RenderAssets<GpuImage>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    if !config.enabled {
        return;
    }
    // No sun direction yet (rig not started, or the body has no star in view):
    // leave the target at its last contents rather than marching toward zero.
    if sun.dir.length_squared() <= 0.0 {
        return;
    }
    let (extracted, _ship, prepass) = view.into_inner();
    let (Some(contact), Some(pipeline)) = (contact, pipeline) else {
        return;
    };
    let (Some(dest), Some(depth)) = (render_assets.get(&contact.handle), prepass.depth.as_ref())
    else {
        return;
    };
    let variant = usize::from(depth.texture.texture.sample_count() > 1);
    let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipeline.pipeline_ids[variant])
    else {
        return;
    };

    // Render-space → view-space is a pure rotation for a direction, so the
    // floating origin never enters (a direction is translation-invariant).
    let sun_view = (extracted
        .world_from_view
        .compute_transform()
        .rotation
        .inverse()
        * sun.dir)
        .normalize_or_zero();

    let clip_from_view = extracted.clip_from_view;
    let size = dest.texture.size();
    let uniform = ContactUniform {
        view_from_clip: clip_from_view.inverse(),
        clip_from_view,
        target_res: Vec4::new(size.width as f32, size.height as f32, 0.0, 0.0),
        params: Vec4::new(
            config.reach,
            config.thickness,
            config.strength,
            config.normal_bias,
        ),
        sun_view: Vec4::new(sun_view.x, sun_view.y, sun_view.z, config.fade_end),
    };
    let mut scratch = encase::UniformBuffer::new(Vec::<u8>::new());
    // Infallible for an in-memory Vec sink.
    let _ = scratch.write(&uniform);
    let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("contact_shadow_uniform"),
        usage: BufferUsages::UNIFORM,
        contents: scratch.as_ref(),
    });

    let bind_group = render_device.create_bind_group(
        Some("contact_shadow_bind_group"),
        &pipeline_cache.get_bind_group_layout(&pipeline.layouts[variant]),
        &BindGroupEntries::sequential((&depth.texture.default_view, buffer.as_entire_binding())),
    );

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("contact_shadow_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &dest.texture_view,
            depth_slice: None,
            resolve_target: None,
            // Fullscreen triangle overwrites every pixel; default (load) is fine.
            ops: Operations::default(),
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_render_pipeline(render_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);
}
