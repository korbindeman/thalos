//! Cascaded sun-aligned shadow maps for ground vegetation + terrain.
//!
//! The UDLOD terrain pass is a custom pipeline and does **not** receive Bevy's
//! cascaded shadow maps — it shades in its own shader. (It historically received
//! shadows only through an analytic craft proxy (`BodyTerrainShadow`), which
//! couldn't represent thousands of scattered trees; that proxy is now retired —
//! the craft casts into this rig like everything else, so its shadow has one
//! definition.)
//!
//! This module renders a self-managed **cascaded** directional shadow map:
//!
//! 1. [`CASCADE_COUNT`] plain orthographic [`Camera3d`]s *outside* big_space
//!    (like the map camera), all on [`SHADOW_CASTER_LAYER`], aimed down the
//!    active body's sun direction over the **ground below the view** at
//!    increasing half-extents (near = crisp, far = wide). Tree mesh tiles are
//!    tagged onto that layer too, so the same `TreeMaterial` draw (leaf
//!    alpha-discard) writes leaf-shaped depth into every cascade that contains
//!    them.
//! 2. A render-graph node copies each cascade camera's depth attachment into its
//!    OWN sample-able [`SunShadowImage`] depth map (the `scene_depth` copy
//!    pattern, one plain `texture_depth_2d` per cascade — deliberately NOT a
//!    depth array, which broke terrain rendering).
//! 3. `body_terrain.wgsl` / `tree.wgsl` bind the per-cascade maps + transforms
//!    ([`thalos_body_render::ShadowCascadeBlock`]) and, per fragment, walk the
//!    cascades near→far and darken the direct-sun term using the tightest one.
//!
//! Centring on the **ground below the
//! [`ViewAnchor`](crate::rendering::view_anchor::ViewAnchor)** — the render camera,
//! whatever is driving it — is what makes the shadowed area follow the view
//! (flight, god view, freecam) with no per-mode plumbing, while keeping each
//! cascade's orthographic depth range shallow regardless of how high the camera
//! is. Two frames of reference meet here and must not be confused: the anchor
//! is resolved in f64 WORLD space, and the cascade cameras live *outside*
//! big_space, so the projection into render space goes through
//! [`RealSpaceOrigin`](crate::rendering::real_space::RealSpaceOrigin) — the
//! floating origin's cell origin — never through `RenderOrigin` (the craft-
//! tracking map pivot). Above a camera-altitude limit the ground-projected set
//! collapses to a single craft-centred cascade (craft self-shadow in orbit —
//! stock Bevy CSM is off, this is the one shadow world).

use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy::asset::{Assets, Handle, RenderAssetUsages};
use bevy::camera::{Camera, ClearColorConfig, ImageRenderTarget, RenderTarget, ScalingMode};
// Bevy 0.19: render passes are systems in the `Core3d` schedule (was the
// `Node3d::MainOpaquePass → … → MainTransparentPass` graph edges).
use bevy::core_pipeline::core_3d::{main_opaque_pass_3d, main_transparent_pass_3d};
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::image::Image;
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::render::{
    RenderApp,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    renderer::{RenderContext, ViewQuery},
    texture::GpuImage,
    view::ViewDepthTexture,
};

use thalos_body_render::{CASCADE_COUNT, CraftShadowMaps, ShadowCascadeBlock};
use thalos_world::BodyId;

use crate::SimStage;
use crate::camera::ShipCamera;
use crate::rendering::real_space::update_real_space_body_positions;
use crate::rendering::types::RealSpaceBody;
use crate::solar_system_state::{SimulationState, SolarSystemState, sync_solar_system_state};

/// Render layer the sun-shadow cameras render. Casters (tree mesh tiles, craft
/// parts) are made visible to them by adding this layer alongside `SHIP_LAYER`.
/// 6/7 are the impostor-bake layers; 8 is the first free index.
pub const SHADOW_CASTER_LAYER: usize = 8;

/// Per-cascade square resolution. 4096² at the extents below is ~0.2 m/texel for
/// the near cascade and ~2.0 m for the far one — crisper cliff and ridge shadows.
const SHADOW_MAP_SIZE: u32 = 4096;

/// BASELINE half-width (m) of each cascade's orthographic box, near→far,
/// at/below [`SHADOW_REFERENCE_ALTITUDE_M`]. Centred on the craft; cascade 0 is
/// tight + crisp, the last reaches out to cover the whole mesh-tree band
/// (~2.2 km swap) with margin. Above the reference altitude the whole set
/// scales ∝ camera altitude (see [`SHADOW_MAX_FOOTPRINT_SCALE`]) so coverage
/// tracks the visible footprint.
const CASCADE_HALF_EXTENTS_M: [f32; CASCADE_COUNT] = [400.0, 1500.0, 4000.0];

/// Per-cascade orthographic far plane (m). Only needs to bracket terrain relief +
/// tree height + the box's low-sun tilt (the centre sits near the ground).
/// Orthographic depth is linear, so clip-space bias = metres / `(far − near)`.
const CASCADE_FARS_M: [f32; CASCADE_COUNT] = [1500.0, 5000.0, 12000.0];

// Depth bias / receiver offset are no longer authored here: the shared sampler
// (`thalos::shadow`) derives them per cascade from the texel size published in
// `params.y` — texel-proportional with a hard absolute cap (`BIAS_MAX_M` /
// `NORMAL_OFFSET_MAX_M` in `shadow.wgsl`). The cap is the load-bearing part: a
// bias larger than a caster's height along the light ERASES its shadow, and
// per-cascade hand constants (the old 10 m far-cascade bias vs ~10 m trees) —
// let alone footprint-scaled ones — did exactly that, which is why far/zoomed
// shadow coverage looked dead while near cascades were fine.

/// Base up-sun eye offset of the ortho cameras above the region centre. The
/// real offset adds the per-cascade ground slack (`half / tanθ` — see the
/// up-sun depth-slack note in `update_sun_shadow_camera`), without which the
/// near plane clips everything more than ~this far up-sun of the craft out of
/// the shadow world. (The caster shaders bypass their camera-anchored fades in
/// the ortho pass, so a large eye distance is harmless to them.)
const SHADOW_BACK_DISTANCE_M: f32 = 150.0;
const SHADOW_NEAR_M: f32 = 0.5;

/// Per-cascade MINIMUM half-extents (m), keyed to the vegetation CASTER band:
/// tree tiles cast into the rig only out to `TREE_SHADOW_CASTER_MAX_M` (6 km —
/// rings 0–1 in `rendering/vegetation.rs`; the coarse far rings are sub-pixel
/// and no longer cast), so shadows "running out" inside that band is a
/// coverage bug, while beyond it nothing exists to cast. Cascade 0 keeps its
/// small footprint-scaled box (crisp craft / near-field shadows); cascade 1
/// always spans the mesh-tree ring (2.4 km + fade); cascade 2 always spans the
/// whole caster band. The footprint scale still grows any of them further when
/// the vantage demands it. (Previously 6.5 / 23.5 km to chase the 22 km
/// impostor band — ~3.2 / 11.5 m per texel, which made every shadow past the
/// near cascade a coarse blob. The far field beyond the caster band belongs to
/// the heightfield horizon term (W12), not to stretched cascades — the
/// MSFS-style shadow-map / terrain-shadow split.)
const CASCADE_MIN_HALF_M: [f32; CASCADE_COUNT] = [0.0, 3_000.0, 6_500.0];

/// Depth margin (m) bracketing terrain relief above/below the centre's tangent
/// plane. With band-wide boxes, casters on hills inside the box sit well above
/// the plane and would otherwise fall in front of the near plane at high sun —
/// the vertical cousin of the up-sun ground slack.
const SHADOW_RELIEF_MARGIN_M: f32 = 4_000.0;

/// Lower clamp on sin(sun elevation) for the up-sun ground-slack term. Below
/// this (~4°) the along-sun ground reach diverges toward the horizon; shadows
/// there are horizon-length streaks and the terminator is about to end them
/// anyway.
const SHADOW_MIN_SUN_SIN: f32 = 0.07;

/// Hard cap (m) on the per-cascade up-sun ground slack, bounding the ortho
/// depth range at extreme footprint scales (Depth32Float is linear over
/// `far − near`; keep the range sane).
const SHADOW_SLACK_MAX_M: f32 = 80_000.0;

/// Above this camera altitude (AGL, m) the rig drops the ground-projected
/// cascade set and switches to **craft-local mode**: cascade 0 only, centred on
/// the craft, far cascades parked. With stock Bevy CSM disabled (F6 — one
/// shadow world) the craft must keep shadowing *itself* in orbit, so the rig
/// never fully turns off while a craft exists. High: by here the whole surface
/// scene is far sub-pixel; below it the FOOTPRINT SCALING keeps ground shadows
/// alive at any zoom (the old 6 km hard cut made every shadow in the world
/// vanish the moment the camera boomed out — the "shadows only at some
/// distances" bug).
const SHADOW_MAX_ALTITUDE_M: f32 = 50_000.0;

/// Cap for the view footprint scale. At 32× the far cascade reaches 128 km and
/// the near cascade's texel is ~6 m — coarse, but shadows that far from the
/// vantage are a few pixels tall anyway; beyond the cap they'd be sub-pixel.
const SHADOW_MAX_FOOTPRINT_SCALE: f32 = 32.0;

/// Hysteresis for the QUANTIZED footprint scale (power-of-two steps): grow
/// immediately (coverage is correctness), shrink only once the raw requirement
/// drops below this fraction of the current step (comfortably inside the next
/// step down, so a camera hovering at a boundary doesn't strobe texel sizes).
/// A continuously-varying footprint rescaled every cascade's texel grid every
/// frame the camera moved — re-rasterizing every shadow edge = global shimmer.
const SHADOW_FOOTPRINT_SHRINK_FRACTION: f32 = 0.42;

/// Cap on the footprint's look-reach term, in multiples of camera AGL. The
/// cascade covers the ground point the camera looks at out to this many × AGL
/// from the nadir; 4.0 fully covers the god view's shallowest pitch
/// (15° → reach ≈ 3.73 × AGL) while keeping a horizon-grazing flight camera
/// from demanding tens of km of shadow box.
const SHADOW_LOOK_REACH_MAX_AGL: f32 = 4.0;

/// Re-anchor distance for the texel-snap's body-fixed reference point. The
/// snap phase is computed relative to a point that CO-ROTATES with the body
/// (see the snap note in `update_sun_shadow_camera`); once the cascade centre
/// drifts this far from it, re-anchor (a one-frame sub-texel phase jump).
/// Keeps the f32 relative coordinates small and precise.
const SNAP_ANCHOR_REACH_M: f64 = 8_000.0;

/// Quantized sun stepping (stable CSM, part 2 — the rotational cousin of the
/// body-fixed texel snap). The snap stabilizes the cascade's *translation*
/// relative to the co-rotating ground, but the sun's direction over the site
/// still changed every simulated frame (planet rotation × warp) — the light
/// basis rotated relative to the casters, so every shadow edge re-rasterized
/// with a fresh sub-texel phase each frame anyway. The rig therefore HOLDS its
/// sun direction fixed IN THE BODY FRAME and only steps it once the true sun
/// has drifted past this angle: between steps the light co-rotates rigidly
/// with the ground, the light↔caster geometry is frame-to-frame constant, and
/// the rendered depth maps are stable. At 0.1° a 100 m shadow moves ~17 cm per
/// step (sub-texel for cascade 0); under high warp shadows advance in visible
/// discrete steps, which reads as intended time-lapse rather than shimmer.
/// Scene lighting (`update_sun_light`) keeps the continuous sun — only the
/// shadow rig quantizes; a ≤0.1° mismatch is imperceptible.
const SUN_LOCK_STEP_RAD: f64 = 0.1 * core::f64::consts::PI / 180.0;

/// Default shadow darkening strength (0 = off, 1 = black). Higher values give
/// hard cliff/ridge contrast; ambient fill keeps shadowed ground from going pure black.
const SHADOW_STRENGTH: f32 = 0.88;

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

/// Copy each shadow cascade's rendered depth into its own depth map. Ported
/// from the former `CopySunShadowDepthNode` (`ViewNode`) to a Bevy 0.19
/// render-pass **system**. The `ViewQuery` filters to cascade views via
/// `SunShadowCascade` and auto-skips the main ship-camera view.
fn copy_sun_shadow_depth(
    view: ViewQuery<(&'static ViewDepthTexture, &'static SunShadowCascade)>,
    shadow: Option<Res<SunShadowImage>>,
    render_assets: Res<RenderAssets<GpuImage>>,
    mut ctx: RenderContext,
) {
    let (depth, cascade) = view.into_inner();

    // ~once/sec at 60 fps, cascade 0 only — diagnostics, never per-frame.
    let n = COPY_DIAG_FRAME.fetch_add(1, Ordering::Relaxed);
    let diag = |s: &str| {
        if cascade.index == 0 && n % (60 * CASCADE_COUNT as u64) == 0 {
            log_shadow_state(s);
        }
    };

    let Some(shadow) = shadow else {
        diag("{\"copy\":\"no_resource\"}");
        return;
    };
    let Some(handle) = shadow.handles.get(cascade.index as usize) else {
        return;
    };
    let Some(dest) = render_assets.get(handle) else {
        diag("{\"copy\":\"no_dest_gpuimage\"}");
        return;
    };

    let src_size = depth.texture.size();
    let dst_size = dest.texture.size();
    if src_size.width != dst_size.width || src_size.height != dst_size.height {
        diag(&format!(
            "{{\"copy\":\"size_mismatch\",\"src\":[{},{}],\"dst\":[{},{}]}}",
            src_size.width, src_size.height, dst_size.width, dst_size.height,
        ));
        return;
    }
    if depth.texture.sample_count() != dest.texture.sample_count() {
        diag(&format!(
            "{{\"copy\":\"sample_mismatch\",\"src\":{},\"dst\":{}}}",
            depth.texture.sample_count(),
            dest.texture.sample_count(),
        ));
        return;
    }

    // Plain full-texture copy into this cascade's own depth map — the exact
    // known-good single-map copy, one per cascade.
    ctx.command_encoder().copy_texture_to_texture(
        depth.texture.as_image_copy(),
        dest.texture.as_image_copy(),
        src_size,
    );
    diag(&format!(
        "{{\"copy\":\"ok\",\"size\":[{},{}],\"cascades\":{}}}",
        src_size.width, src_size.height, CASCADE_COUNT,
    ));
}

/// Mirror the live sun-shadow cascade (`SunShadowState`, owned by the rig) into
/// the render crate's `CraftShadowMaps`, so the craft hull / gear — Bevy-PBR
/// `ShipPartMaterial` — RECEIVE the same cascade the terrain / trees cast into
/// (graphics-fidelity F6b). `apply_craft_shadow` (render crate, PostUpdate) fans
/// it onto the materials. No-op until the rig's state exists and the craft
/// material is registered.
fn sync_craft_shadow(state: Option<Res<SunShadowState>>, maps: Option<ResMut<CraftShadowMaps>>) {
    let (Some(state), Some(mut maps)) = (state, maps) else {
        return;
    };
    maps.images = state.images.clone();
    maps.block = state.block;
}

pub struct SunShadowPlugin;

impl Plugin for SunShadowPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<SunShadowImage>::default())
            .add_plugins(ExtractComponentPlugin::<SunShadowCascade>::default())
            .add_systems(Startup, setup_sun_shadow)
            .add_systems(
                Update,
                (
                    update_sun_shadow_camera
                        .after(update_real_space_body_positions)
                        .after(sync_solar_system_state)
                        .after(crate::rendering::real_space::update_real_space_origin),
                    // Mirror the live cascade onto the craft hull/gear so they
                    // RECEIVE it (the render crate's `apply_craft_shadow` fans
                    // `CraftShadowMaps` onto the materials in PostUpdate).
                    sync_craft_shadow.after(update_sun_shadow_camera),
                )
                    .in_set(SimStage::Sync),
            );

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                Core3d,
                copy_sun_shadow_depth
                    .in_set(Core3dSystems::MainPass)
                    .after(main_opaque_pass_3d)
                    .before(main_transparent_pass_3d),
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
    let Some(path) = crate::artifact_paths::jsonl_path_from_env("THALOS_SHADOW_LOG") else {
        return;
    };
    if let Ok(mut f) = crate::artifact_paths::open_jsonl_append(&path) {
        let _ = writeln!(f, "{line}");
    }
}

/// Aim every cascade camera down the sun over the craft and publish their
/// transforms. Near the surface: three ground-projected cascades. In orbit /
/// high flight: one craft-centred cascade (self-shadow only). Fully disabled
/// only when there is no terrain body / camera / states at all.
#[allow(clippy::type_complexity)]
fn update_sun_shadow_camera(
    sim: Res<SimulationState>,
    cache: Res<SolarSystemState>,
    bodies: Query<(&RealSpaceBody, &GlobalTransform)>,
    ship_cam: Query<&GlobalTransform, With<ShipCamera>>,
    origin: Res<crate::rendering::real_space::RealSpaceOrigin>,
    view_anchor: Res<crate::rendering::view_anchor::ViewAnchor>,
    height_sources: Res<thalos_physics_local::HeightSourceRegistry>,
    // Contact tier (W18a): its gate is published in `block.gate.z` so it reaches
    // every consumer through the binding they already carry.
    contact: Res<super::contact_shadow::ContactShadowConfig>,
    mut shadow_cams: Query<
        (
            &mut Transform,
            &mut Camera,
            &mut Projection,
            &SunShadowCascade,
        ),
        Without<ShipCamera>,
    >,
    mut state: ResMut<SunShadowState>,
    mut frame: Local<u64>,
    // Body-fixed texel-snap anchor (see the snap note below) + the current
    // quantized footprint step (power-of-two, with shrink hysteresis) + the
    // held body-fixed sun direction (see `SUN_LOCK_STEP_RAD`).
    mut snap_anchor: Local<Option<(BodyId, DVec3)>>,
    mut footprint_step: Local<f32>,
    mut sun_lock: Local<Option<(BodyId, DVec3)>>,
) {
    *frame = frame.wrapping_add(1);
    let log_now = *frame % 15 == 0;

    let mut reason = "ok";
    let mut altitude_m = -1.0_f32;
    let mut body_dbg = String::from("none");
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

        let mut best: Option<(BodyId, f32, f32)> = None;
        for (b, xform) in &bodies {
            let Some(def) = sim.system.bodies.get(b.body_id) else {
                continue;
            };
            if !def.terrain.is_some() {
                continue;
            }
            n_terrain_bodies += 1;
            let radius = def.radius_m as f32;
            let altitude = (cam_pos - xform.translation()).length() - radius;
            if best.is_none_or(|(_, a, _)| altitude < a) {
                best = Some((b.body_id, altitude, radius));
            }
        }
        let Some((active_id, altitude, body_radius_m)) = best else {
            reason = "no_terrain_body";
            break 'resolve;
        };
        altitude_m = altitude;
        body_dbg = format!("{active_id:?}");
        // Off-surface, don't disable — switch to craft-local self-shadow mode
        // (see `SHADOW_MAX_ALTITUDE_M`). The gate is the CAMERA's altitude, so
        // any near-surface view (flight, god view, freecam) keeps ground
        // shadows regardless of where the craft is.
        let craft_local = altitude > SHADOW_MAX_ALTITUDE_M;
        if craft_local {
            reason = "craft_local";
        }
        let (Some(star), Some(body_state)) = (states.first(), states.get(active_id)) else {
            reason = "no_state";
            break 'resolve;
        };

        let offset = star.position - body_state.position;
        let sun_dir = if offset.length_squared() <= 0.0 {
            Vec3::Y
        } else if craft_local {
            // Orbit / high flight: the caster is the freely-rotating craft, so
            // there is no body frame to hold the light in — use the true sun.
            *sun_lock = None;
            offset.normalize().as_vec3()
        } else {
            // Quantized sun stepping: hold the shadow sun fixed in the BODY
            // frame until the true sun drifts `SUN_LOCK_STEP_RAD` away (see the
            // constant's note). Between steps the light basis, the body-fixed
            // snap grid, and every caster co-rotate rigidly — the rendered
            // cascade content is frame-to-frame identical while the sim runs.
            let cur_bf = body_state.orientation.inverse() * offset.normalize();
            let locked_bf = match *sun_lock {
                Some((b, l)) if b == active_id && l.dot(cur_bf) > SUN_LOCK_STEP_RAD.cos() => l,
                _ => {
                    *sun_lock = Some((active_id, cur_bf));
                    cur_bf
                }
            };
            (body_state.orientation * locked_bf).normalize().as_vec3()
        };

        // Centre the cascades on the ground point BELOW THE VIEW, so the crisp
        // near cascade follows whatever the camera is doing — flight, god view,
        // freecam — with no craft anchoring. `up_radial` is the local vertical
        // (the direction of a huge vector, so f32-precise); `altitude` carries
        // the small big_space cancellation error, which only nudges the box
        // height.
        //
        // The centre must be a THIS-FRAME inertial point: the casters (tree
        // tiles) + receivers render at this-frame body orientation, and reading
        // the ShipCamera GlobalTransform directly (one frame stale, km-scale at
        // the surface's ~260 m/s co-rotation × warp) made the cascade crawl the
        // instant the sim ran. The `ViewAnchor` solves exactly this: the camera
        // pose resolved BODY-FIXED at a coherent epoch, re-projected here with
        // this frame's body state (see `rendering::view_anchor`). Craft-local
        // mode (high orbit) genuinely is about the craft-as-caster, so it keeps
        // the canonical craft state; the anchor also falls back to the craft
        // when it is unresolved or resolved against another body.
        let anchor_here = view_anchor
            .resolved
            .filter(|a| !craft_local && a.body == active_id);
        let player_inertial = anchor_here
            .map(|a| a.cam_world(states))
            .unwrap_or_else(|| sim.simulation.ship_state().position);
        let radial = player_inertial - body_state.position;
        let r = radial.length();
        let radial_dir = if r > 1.0e-3 { radial / r } else { DVec3::Y };
        let up_radial = radial_dir.as_vec3();
        // Project the centre to the TRUE ground below the craft, not the datum
        // sphere: `r − body_radius` is altitude above the REFERENCE radius, so
        // at an elevated site it buried the centre by the full terrain
        // elevation. Cascade 0's ±400 m box (in the near-vertical light plane
        // of a low sun) then missed the surface outright, and every
        // near-craft fragment fell through to the metres-per-texel far
        // cascades — the "pixelated shadow right at the craft" bug. A missing
        // height sample (cold tiles) falls back to the datum, which merely
        // reproduces the old centring until tiles stream in.
        // Height sources sample in the SURFACE body-fixed frame (the frame the
        // terrain renders in). The anchor's nadir is already that frame; the
        // craft fallback converts through the ephemeris orientation, which for
        // a tidally-locked moon is a different frame — acceptable there only
        // because the fallback is the craft path, and a datum-height miss just
        // reproduces the old centring (see the comment above).
        let dir_body = anchor_here
            .map(|a| a.cam_dir.as_vec3())
            .unwrap_or_else(|| (body_state.orientation.inverse() * radial_dir).as_vec3());
        let terrain_h = height_sources
            .get(active_id)
            .and_then(|hs| {
                hs.sample_height_m(dir_body, crate::local_physics::PHYSICS_QUERY_TILE_LOD_M)
            })
            .unwrap_or(0.0) as f64;
        let player_alt = (r - body_radius_m as f64 - terrain_h) as f32;
        // Render space here means the BIG_SPACE render frame — the floating
        // origin's cell origin (`RealSpaceOrigin`), which is what the casters'
        // `GlobalTransform`s are measured from. It is emphatically NOT
        // `RenderOrigin` (the camera FOCUS pivot, i.e. the craft in flight):
        // that put the whole cascade set `|camera − craft|` away from the world
        // it was meant to cover, so ground shadows survived only while the view
        // stayed within a cascade half-extent of the ship and died as freecam
        // flew off. See `real_space::RealSpaceOrigin`.
        let player_render = origin.to_render(player_inertial);
        // Craft-local mode centres the (single) cascade on the craft itself —
        // projecting to a ground point tens/hundreds of km below would throw
        // the box away from the only caster that matters up here.
        let (center, center_inertial) = if craft_local {
            (player_render, player_inertial)
        } else {
            (
                player_render - up_radial * player_alt,
                player_inertial - radial_dir * player_alt as f64,
            )
        };
        let up = if sun_dir.dot(Vec3::Y).abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        // View footprint scale: the cascade set is centred on the ground below
        // the camera, but it must COVER whatever ground the camera is *looking
        // at* from ANY vantage. Three view terms size it: any camera↔anchor
        // separation (≈0 now that the centre is the view anchor itself), the
        // ground reach of the camera's look direction (a boomed-out god view
        // at shallow pitch inspects ground several × AGL away from the nadir —
        // clamped, or a horizon-grazing look would demand tens of km), plus a
        // foreground extent that grows with camera altitude. Everything scales
        // together: extents, depth range, back distance (the sampler's
        // bias/offset derive from the published texel size and are hard-capped,
        // so a wider box coarsens shadows instead of erasing them). Craft-local
        // mode keeps the baseline — the craft is metres across regardless of
        // altitude.
        let footprint = if craft_local {
            1.0
        } else {
            let cam_dist = (cam_pos - player_render).length();
            let cam_agl = (altitude - terrain_h as f32).max(0.0);
            let fwd = cam_xform
                .affine()
                .transform_vector3(Vec3::NEG_Z)
                .normalize_or_zero();
            let down = (-fwd).dot(up_radial).max(0.0);
            let horiz = (1.0 - down * down).max(0.0).sqrt();
            let look_reach = if down > 1.0e-3 {
                (cam_agl * horiz / down).min(cam_agl * SHADOW_LOOK_REACH_MAX_AGL)
            } else {
                cam_agl * SHADOW_LOOK_REACH_MAX_AGL
            };
            let required_half_m = cam_dist + look_reach + cam_agl * 2.0;
            let raw = (required_half_m / CASCADE_HALF_EXTENTS_M[CASCADE_COUNT - 1])
                .clamp(1.0, SHADOW_MAX_FOOTPRINT_SCALE);
            // QUANTIZE to power-of-two steps with shrink hysteresis: a
            // continuously-varying footprint rescaled every cascade's texel
            // grid every frame the camera moved, re-rasterizing every shadow
            // edge (global shimmer). Grow immediately — coverage is
            // correctness; shrink only once raw demand is comfortably inside
            // the lower step.
            let quantized = raw
                .log2()
                .ceil()
                .exp2()
                .clamp(1.0, SHADOW_MAX_FOOTPRINT_SCALE);
            if quantized > *footprint_step
                || raw < *footprint_step * SHADOW_FOOTPRINT_SHRINK_FRACTION
            {
                *footprint_step = quantized;
            }
            *footprint_step
        };
        // ── Up-sun depth slack ────────────────────────────────────────────────
        // The cascade box is square in the LIGHT plane, but its intersection
        // with the GROUND stretches along the sun azimuth as the sun drops: a
        // ground point `d` up-sun of the centre sits at ray-depth
        // `back − d·cosθ`, so with a small fixed `back` everything beyond
        // ~`back/cosθ` up-sun of the craft fell in front of the NEAR PLANE —
        // clipped out of the depth map AND out of the receiver test. Result:
        // shadows existed only on the down-sun side of the craft (a hard
        // directional boundary through the base — "inconsistent by angle").
        // Push the eye up-sun by the box's ground reach along the azimuth
        // (`half/tanθ`, overbounded by dropping a cosθ) and extend the far
        // plane by twice that so the down-sun extreme + terrain relief still
        // fit. Clamped: below ~4° sun the reach diverges (and shadows are
        // horizon-length anyway), and a hard cap bounds the depth range at
        // extreme footprints.
        let sin_elev = sun_dir.dot(up_radial).clamp(SHADOW_MIN_SUN_SIN, 1.0);
        let ground_slack = |half: f32| -> f32 {
            let cos_elev = (1.0 - sin_elev * sin_elev).max(0.0).sqrt();
            (half * cos_elev / sin_elev).min(SHADOW_SLACK_MAX_M)
        };
        // Rotation is shared by every cascade (only translation differs); any
        // eye distance yields the same rotation.
        let base_look = Transform::from_translation(center + sun_dir * SHADOW_BACK_DISTANCE_M)
            .looking_at(center, up);
        let light_right = base_look.rotation * Vec3::X;
        let light_up = base_look.rotation * Vec3::Y;
        let eye_dbg = center + sun_dir * SHADOW_BACK_DISTANCE_M;
        let sun_dbg = sun_dir;

        // ── Texel-snap reference (stable CSM on a ROTATING planet) ──────────
        // Snapping the centre to a grid in RENDER space assumed a static
        // world. Here the whole shadow world (terrain, trees, structures, the
        // parked craft) CO-ROTATES with the body at hundreds of m/s, and the
        // floating origin moves with the camera — so a render-space grid slid
        // under the casters with a fresh sub-texel phase every simulated
        // frame, re-rasterizing every shadow edge (the "shadow flickers the
        // moment the sim unpauses" bug; paused, nothing moved, so it looked
        // stable). Compute the snap phase RELATIVE TO A BODY-FIXED ANCHOR
        // instead: the grid then translates with the rotating ground, casters
        // keep their rasterization phase frame-to-frame, and the only residual
        // drift is the (slow, physical) sun motion. Re-anchor on multi-km
        // drift — a one-frame sub-texel phase jump. Craft-local mode glues the
        // grid to the craft (snap_rel = 0): the hull is the only caster there.
        let snap_rel = if craft_local {
            Vec3::ZERO
        } else {
            let center_bf =
                body_state.orientation.inverse() * (center_inertial - body_state.position);
            let anchor_bf = match *snap_anchor {
                Some((b, a))
                    if b == active_id && (a - center_bf).length() < SNAP_ANCHOR_REACH_M =>
                {
                    a
                }
                _ => {
                    *snap_anchor = Some((active_id, center_bf));
                    center_bf
                }
            };
            let anchor_inertial = body_state.position + body_state.orientation * anchor_bf;
            (center_inertial - anchor_inertial).as_vec3()
        };

        // Craft-local mode runs only the crisp near cascade; the far cascades'
        // matrices are zeroed (the shader's `clip.w <= 0` skip sentinel) and
        // their cameras deactivated, so their stale depth maps are never read.
        let active_cascades = if craft_local { 1 } else { CASCADE_COUNT };
        let mut block = ShadowCascadeBlock::default();
        let mut looks = [Transform::IDENTITY; CASCADE_COUNT];
        let mut halves = [0.0_f32; CASCADE_COUNT];
        let mut fars = [0.0_f32; CASCADE_COUNT];
        for i in 0..CASCADE_COUNT {
            if i >= active_cascades {
                block.view_proj[i] = Mat4::ZERO;
                continue;
            }
            let half = (CASCADE_HALF_EXTENTS_M[i] * footprint).max(CASCADE_MIN_HALF_M[i]);
            // Up-sun eye offset + far plane bracket this cascade's whole
            // ground footprint along the sun azimuth (see the slack note) plus
            // terrain relief above/below the tangent plane.
            let slack = ground_slack(half) + SHADOW_RELIEF_MARGIN_M;
            let back = SHADOW_BACK_DISTANCE_M * footprint + slack;
            let far = CASCADE_FARS_M[i] * footprint + 2.0 * slack;
            halves[i] = half;
            fars[i] = far;
            // Texel-snap the cascade centre to ITS shadow-map grid in the light
            // plane, so the ortho frustum slides in whole-texel steps and shadow
            // edges stop crawling as the centre drifts (stable CSM). Each cascade
            // snaps to its own (coarser, near→far) grid. The phase comes from
            // `snap_rel` — the centre RELATIVE to a body-fixed anchor — so the
            // grid co-moves with the rotating ground (see the snap note above).
            let texel = (2.0 * half) / SHADOW_MAP_SIZE as f32;
            let cr = snap_rel.dot(light_right);
            let cu = snap_rel.dot(light_up);
            let snap = ((cr / texel).round() * texel - cr) * light_right
                + ((cu / texel).round() * texel - cu) * light_up;
            let center_i = center + snap;
            let eye_i = center_i + sun_dir * back;
            let look_i = Transform::from_translation(eye_i).looking_at(center_i, up);
            block.view_proj[i] = cascade_clip_from_view(half, far) * look_i.to_matrix().inverse();
            // x = clip units per metre of light-space depth (orthographic z is
            // linear), y = texel size in world metres — the shared sampler
            // derives its capped, texel-proportional bias + receiver offset
            // from these (see the bias model note in `shadow.wgsl`).
            block.params[i] = Vec4::new(1.0 / (far - SHADOW_NEAR_M), texel, 0.0, 0.0);
            looks[i] = look_i;
        }
        // z = the contact-shadow gate (W18a). Published from the rig rather than
        // per-material so every consumer of the block inherits it. Note it rides
        // *inside* the cascade gate: when `gate.x == 0` the samplers early-out
        // fully lit and the contact term is moot along with them — the rig-off
        // cases (orbital map terrain, inactive pass) want no shadow at all.
        block.gate = Vec4::new(
            SHADOW_STRENGTH,
            active_cascades as f32,
            contact.shadow_gate(),
            0.0,
        );
        // Sun direction (toward the sun) drives the sampler's slope-scaled bias.
        block.sun_dir = Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, 0.0);
        state.block = block;

        for (mut tf, mut cam, mut proj, cascade) in &mut shadow_cams {
            let idx = cascade.index as usize;
            let on = idx < active_cascades;
            if on {
                *tf = looks[idx];
                // Keep the LIVE camera projection in lockstep with the
                // hand-built `block.view_proj` — the spawn-time projection only
                // covers the unscaled baseline footprint.
                *proj = Projection::Orthographic(OrthographicProjection {
                    scaling_mode: ScalingMode::Fixed {
                        width: halves[idx] * 2.0,
                        height: halves[idx] * 2.0,
                    },
                    near: SHADOW_NEAR_M,
                    far: fars[idx],
                    ..OrthographicProjection::default_3d()
                });
            }
            cam.is_active = on;
        }

        if log_now {
            // `centre_off_m` is the decisive coverage signal: the cascade set is
            // centred on the ground under the camera, so in the correct render
            // frame it can only differ from the camera by the nadir drop
            // (≈ `alt_m`). Anything larger means the boxes are sitting somewhere
            // the view isn't — the failure mode `RealSpaceOrigin` exists to kill.
            let centre_off_m = (center - cam_pos).length();
            log_shadow_state(&format!(
                "{{\"frame\":{},\"reason\":\"{}\",\"active\":true,\"alt_m\":{:.1},\
                 \"body\":\"{}\",\"n_terrain\":{},\"strength\":{:.3},\"cascades\":{},\
                 \"footprint\":{:.1},\"centre_off_m\":{:.1},\
                 \"eye\":[{:.1},{:.1},{:.1}],\"sun\":[{:.3},{:.3},{:.3}]}}",
                *frame,
                reason,
                altitude_m,
                body_dbg,
                n_terrain_bodies,
                SHADOW_STRENGTH,
                CASCADE_COUNT,
                footprint,
                centre_off_m,
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

    for (_tf, mut cam, _proj, _cascade) in &mut shadow_cams {
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
