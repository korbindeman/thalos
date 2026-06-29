//! Cascaded sun-aligned shadow maps for ground vegetation + terrain.
//!
//! The UDLOD terrain pass is a custom pipeline and does **not** receive Bevy's
//! cascaded shadow maps — it shades in its own shader and historically received
//! shadows only through the analytic craft proxy (`BodyTerrainShadow`), which
//! can't represent thousands of scattered trees, so the forest read as a flat,
//! unshadowed carpet.
//!
//! This module renders a self-managed **cascaded** directional shadow map:
//!
//! 1. [`CASCADE_COUNT`] plain orthographic [`Camera3d`]s *outside* big_space
//!    (like the map camera), all on [`SHADOW_CASTER_LAYER`], aimed down the
//!    active body's sun direction over the **craft** at increasing half-extents
//!    (near = crisp, far = wide). Tree mesh tiles are tagged onto that layer too,
//!    so the same `TreeMaterial` draw (leaf alpha-discard) writes leaf-shaped
//!    depth into every cascade that contains them.
//! 2. A render-graph node copies each cascade camera's depth attachment into its
//!    OWN sample-able [`SunShadowImage`] depth map (the `scene_depth` copy
//!    pattern, one plain `texture_depth_2d` per cascade — deliberately NOT a
//!    depth array, which broke terrain rendering).
//! 3. `body_terrain.wgsl` / `tree.wgsl` bind the per-cascade maps + transforms
//!    ([`thalos_body_render::ShadowCascadeBlock`]) and, per fragment, walk the
//!    cascades near→far and darken the direct-sun term using the tightest one.
//!
//! Centring on the craft (near the ground) — not the camera — keeps the shadowed
//! area put as the view orbits, and keeps each cascade's orthographic depth range
//! shallow regardless of how high the view camera is. Gated off above a
//! camera-altitude limit so it costs nothing in orbit.

use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy::asset::{Assets, Handle, RenderAssetUsages};
use bevy::camera::{Camera, ClearColorConfig, ImageRenderTarget, RenderTarget, ScalingMode};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::{
    RenderApp,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_graph::{
        NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
    },
    render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    renderer::RenderContext,
    texture::GpuImage,
    view::ViewDepthTexture,
};

use thalos_body_render::{CASCADE_COUNT, ShadowCascadeBlock};
use thalos_world::BodyId;

use crate::SimStage;
use crate::camera::ShipCamera;
use crate::rendering::real_space::update_real_space_body_positions;
use crate::rendering::types::{PlayerShip, RealSpaceBody};
use crate::solar_system_state::{SimulationState, SolarSystemState, sync_solar_system_state};

/// Render layer the sun-shadow cameras render. Casters (tree mesh tiles, craft
/// parts) are made visible to them by adding this layer alongside `SHIP_LAYER`.
/// 6/7 are the impostor-bake layers; 8 is the first free index.
pub const SHADOW_CASTER_LAYER: usize = 8;

/// Per-cascade square resolution. 2048² at the extents below is ~0.4 m/texel for
/// the near cascade and ~3.9 m for the far one.
const SHADOW_MAP_SIZE: u32 = 2048;

/// Half-width (m) of each cascade's orthographic box, near→far. Centred on the
/// craft; cascade 0 is tight + crisp, the last reaches out to cover the whole
/// mesh-tree band (~2.2 km swap) with margin.
const CASCADE_HALF_EXTENTS_M: [f32; CASCADE_COUNT] = [400.0, 1500.0, 4000.0];

/// Per-cascade orthographic far plane (m). Only needs to bracket terrain relief +
/// tree height + the box's low-sun tilt (the centre sits near the ground).
/// Orthographic depth is linear, so clip-space bias = metres / `(far − near)`.
const CASCADE_FARS_M: [f32; CASCADE_COUNT] = [1500.0, 5000.0, 12000.0];

/// Per-cascade depth-compare bias in **metres**. Larger for coarser far cascades
/// to fight acne; small for the crisp near cascade so canopy detail survives.
const CASCADE_BIAS_M: [f32; CASCADE_COUNT] = [0.6, 2.5, 10.0];

/// How far back along the sun the ortho cameras sit above the region centre.
/// Irrelevant to an orthographic footprint, but it IS the distance
/// `tree.wgsl`'s scale-fade sees, so it's kept small.
const SHADOW_BACK_DISTANCE_M: f32 = 150.0;
const SHADOW_NEAR_M: f32 = 0.5;

/// Disable the whole rig above this camera altitude (AGL, m). Shadows are a
/// ground-level effect; rendering cascades from orbit is pure waste.
const SHADOW_MAX_ALTITUDE_M: f32 = 6000.0;

/// Default shadow darkening strength (0 = off, 1 = black). Tunable.
const SHADOW_STRENGTH: f32 = 0.7;

/// Handles to the per-cascade depth maps the cascade cameras' depth attachments
/// are copied into. Extracted to the render world for [`CopySunShadowDepthNode`];
/// the same handles are bound on every terrain + tree material via [`SunShadowState`].
#[derive(Resource, Clone, ExtractResource)]
pub struct SunShadowImage {
    pub handles: [Handle<Image>; CASCADE_COUNT],
}

/// Main-world shadow state read by the terrain + tree material drivers.
///
/// **Sole writer:** [`update_sun_shadow_camera`].
#[derive(Resource, Clone)]
pub struct SunShadowState {
    /// Per-cascade depth maps bound on materials (same handles as [`SunShadowImage`]).
    pub images: [Handle<Image>; CASCADE_COUNT],
    /// Per-cascade transforms + compare params + `config.x` strength gate.
    pub block: ShadowCascadeBlock,
}

/// Marker + cascade index on each orthographic sun-shadow camera. Extracted so
/// the copy node knows which per-cascade depth map to write.
#[derive(Component, Clone, Copy, ExtractComponent)]
pub struct SunShadowCascade {
    pub index: u32,
}

/// Frame counter for the copy node's rate-limited diagnostics. Render-world, so
/// it can't use a `Local`; an atomic keeps it lock-free across the pass.
static COPY_DIAG_FRAME: AtomicU64 = AtomicU64::new(0);

#[derive(RenderLabel, Hash, PartialEq, Eq, Debug, Clone)]
struct CopySunShadowDepth;

#[derive(Default)]
struct CopySunShadowDepthNode;

impl ViewNode for CopySunShadowDepthNode {
    type ViewQuery = (&'static ViewDepthTexture, &'static SunShadowCascade);

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (depth, cascade): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        // ~once/sec at 60 fps, cascade 0 only — diagnostics, never per-frame.
        let n = COPY_DIAG_FRAME.fetch_add(1, Ordering::Relaxed);
        let diag = |s: &str| {
            if cascade.index == 0 && n % (60 * CASCADE_COUNT as u64) == 0 {
                log_shadow_state(s);
            }
        };

        let Some(shadow) = world.get_resource::<SunShadowImage>() else {
            diag("{\"copy\":\"no_resource\"}");
            return Ok(());
        };
        let Some(handle) = shadow.handles.get(cascade.index as usize) else {
            return Ok(());
        };
        let render_assets = world.resource::<RenderAssets<GpuImage>>();
        let Some(dest) = render_assets.get(handle) else {
            diag("{\"copy\":\"no_dest_gpuimage\"}");
            return Ok(());
        };

        let src_size = depth.texture.size();
        let dst_size = dest.texture.size();
        if src_size.width != dst_size.width || src_size.height != dst_size.height {
            diag(&format!(
                "{{\"copy\":\"size_mismatch\",\"src\":[{},{}],\"dst\":[{},{}]}}",
                src_size.width, src_size.height, dst_size.width, dst_size.height,
            ));
            return Ok(());
        }
        if depth.texture.sample_count() != dest.texture.sample_count() {
            diag(&format!(
                "{{\"copy\":\"sample_mismatch\",\"src\":{},\"dst\":{}}}",
                depth.texture.sample_count(),
                dest.texture.sample_count(),
            ));
            return Ok(());
        }

        // Plain full-texture copy into this cascade's own depth map — the exact
        // known-good single-map copy, one per cascade.
        render_context.command_encoder().copy_texture_to_texture(
            depth.texture.as_image_copy(),
            dest.texture.as_image_copy(),
            src_size,
        );
        diag(&format!(
            "{{\"copy\":\"ok\",\"size\":[{},{}],\"cascades\":{}}}",
            src_size.width, src_size.height, CASCADE_COUNT,
        ));
        Ok(())
    }
}

pub struct SunShadowPlugin;

impl Plugin for SunShadowPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<SunShadowImage>::default())
            .add_plugins(ExtractComponentPlugin::<SunShadowCascade>::default())
            .add_systems(Startup, setup_sun_shadow)
            .add_systems(
                Update,
                update_sun_shadow_camera
                    .after(update_real_space_body_positions)
                    .after(sync_solar_system_state)
                    .in_set(SimStage::Sync),
            );

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_graph_node::<ViewNodeRunner<CopySunShadowDepthNode>>(
                    Core3d,
                    CopySunShadowDepth,
                )
                .add_render_graph_edges(
                    Core3d,
                    (
                        Node3d::MainOpaquePass,
                        CopySunShadowDepth,
                        Node3d::MainTransparentPass,
                    ),
                );
        }
    }
}

/// Create a plain 2-D depth target (one per cascade) the camera depth is copied
/// into — the known-good single-map image, replicated.
fn make_depth_image(images: &mut Assets<Image>) -> Handle<Image> {
    let mut depth = Image::new_uninit(
        Extent3d {
            width: SHADOW_MAP_SIZE,
            height: SHADOW_MAP_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Depth32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    depth.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    images.add(depth)
}

/// Create the per-cascade depth targets + colour targets and spawn the
/// (inactive) orthographic cascade cameras.
fn setup_sun_shadow(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let handles: [Handle<Image>; CASCADE_COUNT] =
        core::array::from_fn(|_| make_depth_image(&mut images));

    commands.insert_resource(SunShadowImage {
        handles: handles.clone(),
    });
    commands.insert_resource(SunShadowState {
        images: handles,
        block: ShadowCascadeBlock::default(),
    });

    for index in 0..CASCADE_COUNT {
        // Each camera needs a colour attachment (we only read the depth).
        let mut color = Image::new_uninit(
            Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::RENDER_WORLD,
        );
        color.texture_descriptor.usage =
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING;
        let color_handle = images.add(color);

        let half = CASCADE_HALF_EXTENTS_M[index];
        commands.spawn((
            Camera3d {
                // COPY_SRC so the node can copy this camera's depth into its map
                // (same flag the ship camera uses in `rendering::scene_depth`).
                depth_texture_usages: (TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC)
                    .into(),
                ..default()
            },
            Camera {
                // All cascades render before the main view (order 0); distinct
                // negative orders keep them unambiguous.
                order: -(1 + index as isize),
                is_active: false,
                clear_color: ClearColorConfig::Custom(Color::NONE),
                ..default()
            },
            RenderTarget::Image(ImageRenderTarget::from(color_handle)),
            Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::Fixed {
                    width: half * 2.0,
                    height: half * 2.0,
                },
                near: SHADOW_NEAR_M,
                far: CASCADE_FARS_M[index],
                ..OrthographicProjection::default_3d()
            }),
            Msaa::Off,
            bevy::camera::visibility::RenderLayers::layer(SHADOW_CASTER_LAYER),
            SunShadowCascade {
                index: index as u32,
            },
            Name::new(format!("Sun Shadow Cascade {index}")),
        ));
    }
}

/// Orthographic clip matrix matching Bevy's reverse-z convention for a cascade
/// of the given half-extent + far plane (`OrthographicProjection::get_clip_from_view`
/// swaps near/far). Built by hand so it stays in lockstep with the camera
/// regardless of when Bevy's projection-update system runs this frame.
fn cascade_clip_from_view(half_extent: f32, far: f32) -> Mat4 {
    Mat4::orthographic_rh(
        -half_extent,
        half_extent,
        -half_extent,
        half_extent,
        far,
        SHADOW_NEAR_M,
    )
}

/// Append one JSONL line of shadow-pass diagnostics, but only when
/// `THALOS_SHADOW_LOG` names a file. Mirrors the house JSONL style used by
/// `THALOS_PERF_LOG`. A single env read when unset, so it's safe to leave wired.
fn log_shadow_state(line: &str) {
    let Ok(path) = std::env::var("THALOS_SHADOW_LOG") else {
        return;
    };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{line}");
    }
}

/// Aim every cascade camera down the sun over the craft and publish their
/// transforms. Disables the rig away from a vegetated surface / from orbit.
#[allow(clippy::type_complexity)]
fn update_sun_shadow_camera(
    sim: Res<SimulationState>,
    cache: Res<SolarSystemState>,
    bodies: Query<(&RealSpaceBody, &GlobalTransform)>,
    ship_cam: Query<&GlobalTransform, With<ShipCamera>>,
    ship: Query<&GlobalTransform, With<PlayerShip>>,
    mut shadow_cams: Query<(&mut Transform, &mut Camera, &SunShadowCascade), Without<ShipCamera>>,
    mut state: ResMut<SunShadowState>,
    mut frame: Local<u64>,
) {
    *frame = frame.wrapping_add(1);
    let log_now = *frame % 15 == 0;

    let mut reason = "ok";
    let mut altitude_m = -1.0_f32;
    let mut body_dbg = String::from("none");
    let mut eye_dbg = Vec3::ZERO;
    let mut sun_dbg = Vec3::ZERO;
    let mut n_terrain_bodies = 0u32;

    'resolve: {
        let Some(states) = cache.states.as_deref() else {
            reason = "no_states";
            break 'resolve;
        };
        let Ok(cam_xform) = ship_cam.single() else {
            reason = "no_ship_cam";
            break 'resolve;
        };
        let cam_pos = cam_xform.translation();

        let mut best: Option<(BodyId, f32)> = None;
        for (b, xform) in &bodies {
            let Some(def) = sim.system.bodies.get(b.body_id) else {
                continue;
            };
            if !def.terrain.is_some() {
                continue;
            }
            n_terrain_bodies += 1;
            let altitude = (cam_pos - xform.translation()).length() - def.radius_m as f32;
            if best.is_none_or(|(_, a)| altitude < a) {
                best = Some((b.body_id, altitude));
            }
        }
        let Some((active_id, altitude)) = best else {
            reason = "no_terrain_body";
            break 'resolve;
        };
        altitude_m = altitude;
        body_dbg = format!("{active_id:?}");
        if altitude > SHADOW_MAX_ALTITUDE_M {
            reason = "too_high";
            break 'resolve;
        }
        let (Some(star), Some(body_state)) = (states.first(), states.get(active_id)) else {
            reason = "no_state";
            break 'resolve;
        };

        let offset = star.position - body_state.position;
        let sun_dir = if offset.length_squared() > 0.0 {
            offset.normalize().as_vec3()
        } else {
            Vec3::Y
        };

        // Centre on the craft (near the ground), not the camera: the shadowed
        // area stays put when the view orbits, and the ortho depth range stays
        // shallow regardless of camera altitude. Falls back to the camera for
        // EVA / freecam (no craft).
        let center = ship
            .iter()
            .next()
            .map(|gt| gt.translation())
            .unwrap_or(cam_pos);
        let eye = center + sun_dir * SHADOW_BACK_DISTANCE_M;
        let up = if sun_dir.dot(Vec3::Y).abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let look = Transform::from_translation(eye).looking_at(center, up);
        let view = look.to_matrix().inverse();
        eye_dbg = eye;
        sun_dbg = sun_dir;

        let mut block = ShadowCascadeBlock::default();
        for i in 0..CASCADE_COUNT {
            let half = CASCADE_HALF_EXTENTS_M[i];
            let far = CASCADE_FARS_M[i];
            block.view_proj[i] = cascade_clip_from_view(half, far) * view;
            // Orthographic z is linear → clip-space bias = metres / (far − near).
            block.params[i] = Vec4::new(CASCADE_BIAS_M[i] / (far - SHADOW_NEAR_M), 0.0, 0.0, 0.0);
        }
        block.gate = Vec4::new(SHADOW_STRENGTH, CASCADE_COUNT as f32, 0.0, 0.0);
        state.block = block;

        for (mut tf, mut cam, _cascade) in &mut shadow_cams {
            *tf = look;
            cam.is_active = true;
        }

        if log_now {
            log_shadow_state(&format!(
                "{{\"frame\":{},\"reason\":\"{}\",\"active\":true,\"alt_m\":{:.1},\
                 \"body\":\"{}\",\"n_terrain\":{},\"strength\":{:.3},\"cascades\":{},\
                 \"eye\":[{:.1},{:.1},{:.1}],\"sun\":[{:.3},{:.3},{:.3}]}}",
                *frame,
                reason,
                altitude_m,
                body_dbg,
                n_terrain_bodies,
                SHADOW_STRENGTH,
                CASCADE_COUNT,
                eye_dbg.x,
                eye_dbg.y,
                eye_dbg.z,
                sun_dbg.x,
                sun_dbg.y,
                sun_dbg.z,
            ));
        }
        return;
    }

    for (_tf, mut cam, _cascade) in &mut shadow_cams {
        cam.is_active = false;
    }
    state.block.gate.x = 0.0;
    if log_now {
        log_shadow_state(&format!(
            "{{\"frame\":{},\"reason\":\"{}\",\"active\":false,\"alt_m\":{:.1},\
             \"body\":\"{}\",\"n_terrain\":{},\"strength\":0.0}}",
            *frame, reason, altitude_m, body_dbg, n_terrain_bodies,
        ));
    }
}
