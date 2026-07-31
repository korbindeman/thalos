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

use bevy::asset::{Assets, Handle, RenderAssetUsages};
use bevy::camera::{
    Camera, CameraProjection, ClearColorConfig, ImageRenderTarget, RenderTarget, ScalingMode,
};
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
use bevy::transform::TransformSystems;
use big_space::prelude::{BigSpace, CellCoord, Grid};

use thalos_body_render::tiles::material::TileTerrainMaterial;
use thalos_body_render::{
    BodyTerrainMaterial, CASCADE_COUNT, CraftShadowMaps, GpuGrassMaterial, GrassMaterial,
    GroundPatchMaterial, RockMaterial, ShadowCascadeBlock, TreeMaterial,
};
use thalos_world::BodyId;

use crate::camera::ShipCamera;
use crate::solar_system_state::{SimulationState, SolarSystemState};

/// Render layer the sun-shadow cameras render. Casters (tree mesh tiles, craft
/// parts) are made visible to them by adding this layer alongside `SHIP_LAYER`.
/// 6/7 are the impostor-bake layers; 8 is the first free index.
pub const SHADOW_CASTER_LAYER: usize = 8;

/// Per-cascade square resolution. 4096² at the extents below is ~0.03 m/texel
/// for the near cascade and ~2.0 m for the far one.
const SHADOW_MAP_SIZE: u32 = 4096;

/// BASELINE half-width (m) of each cascade's box **on the ground**, near→far,
/// at/below [`SHADOW_REFERENCE_ALTITUDE_M`]. Centred on the craft; cascade 0 is
/// tight + crisp, the last reaches out to cover the whole mesh-tree band
/// (~2.2 km swap) with margin. Above the reference altitude the whole set
/// scales ∝ camera altitude (see [`SHADOW_MAX_FOOTPRINT_SCALE`]) so coverage
/// tracks the visible footprint.
///
/// **The 64 m entry is new (2026-07-31).** The old near cascade was 400 m —
/// ~0.2 m/texel — which is coarser than the landing gear, the wing edge, or a
/// hull panel, so everything the camera actually looks at rendered its shadow
/// on a grid too coarse to hold the silhouette. 64 m is ~0.03 m/texel and
/// covers the band a surface camera inspects; the other three are unchanged and
/// simply shift out one slot.
const CASCADE_HALF_EXTENTS_M: [f32; CASCADE_COUNT] = [64.0, 400.0, 1500.0, 4000.0];

/// Per-cascade orthographic far plane (m). Only needs to bracket terrain relief +
/// tree height + the box's low-sun tilt (the centre sits near the ground).
/// Orthographic depth is linear, so clip-space bias = metres / `(far − near)`.
const CASCADE_FARS_M: [f32; CASCADE_COUNT] = [400.0, 1500.0, 5000.0, 12000.0];

/// Tallest caster (m) each cascade must catch from OUTSIDE its own ground box.
///
/// A caster `h` tall throws its shadow `h / tan(elev)` down-sun, so covering the
/// receivers inside a cascade means also rasterizing casters up to that far
/// up-sun of it. In LIGHT space that up-sun margin is only `h · cos(elev)` —
/// bounded by the caster height, never by the cascade's own reach — which is
/// what makes the square-on-ground box affordable (see the box note in
/// `update_sun_shadow_camera`).
///
/// The old square-in-light-plane box had this budget implicitly, and absurdly:
/// its ground reach was `half / sin(elev)`, i.e. it rasterized casters up to
/// roughly `half` TALL — 4 km-tall casters for the outer cascade. Paying map
/// resolution for casters that do not exist is exactly the waste this pass
/// removes. Sized per cascade by what actually casts into it: craft + trees
/// near, structures and modest relief mid, real terrain relief far.
const CASCADE_MAX_CASTER_M: [f32; CASCADE_COUNT] = [60.0, 120.0, 400.0, 1200.0];

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
/// coverage bug, while beyond it nothing exists to cast. Cascades 0 and 1 keep
/// their small footprint-scaled boxes (crisp craft / near-field shadows);
/// cascade 2 always spans the mesh-tree ring (2.4 km + fade); cascade 3 always
/// spans the whole caster band. The footprint scale still grows any of them
/// further when
/// the vantage demands it. (Previously 6.5 / 23.5 km to chase the 22 km
/// impostor band — ~3.2 / 11.5 m per texel, which made every shadow past the
/// near cascade a coarse blob. The far field beyond the caster band belongs to
/// the heightfield horizon term (W12), not to stretched cascades — the
/// MSFS-style shadow-map / terrain-shadow split.)
const CASCADE_MIN_HALF_M: [f32; CASCADE_COUNT] = [0.0, 0.0, 3_000.0, 6_500.0];

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

/// Width of the night stand-down band, in sin(sun elevation) below the local
/// horizon (~3.4°). The rig runs at FULL strength for any sun above the
/// horizon — golden-hour shadows are the longest and most prominent, so this
/// must NOT be keyed to `SunDaylight`, whose ramp is fractional up to ~7°
/// elevation — and fades to zero across this band below it. Without the
/// stand-down the cascades kept rendering all night from a below-horizon sun,
/// and because the samplers gate the WHOLE direct term, they carved the
/// night's moonlight with shadow geometry belonging to a light that was off
/// (phantom dark patches tracking an invisible sun), while paying for three
/// depth renders per frame (reviews/20260730T011353Z §11).
const SHADOW_NIGHT_FADE_SIN: f32 = 0.06;

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

/// Altitude (AGL, m) at which craft-local mode is LEFT again — the low edge of a
/// hysteresis band on [`SHADOW_MAX_ALTITUDE_M`].
///
/// The mode switch is a cliff, not a fade: entering it parks cascades 1–2 with
/// zeroed matrices, so every ground shadow in the world turns off in one frame.
/// With a single hard threshold, a camera loitering near 50 km AGL — hovering,
/// or oscillating a few metres on a smoothed follow cam — re-crossed it
/// constantly and switched the entire ground shadow world on and off at frame
/// rate. A 10 km band means the round trip needs a deliberate 10 km descent,
/// which no jitter produces. The gauge tell was `active_cascades` alternating
/// 3 → 1 → 3 between one-second samples.
const SHADOW_CRAFT_LOCAL_EXIT_M: f32 = 40_000.0;

/// Ratio between the demanded and current footprint scale beyond which the
/// smoother is bypassed and the scale snaps.
///
/// This exists for genuine DISCONTINUITIES (a teleport, a viewpoint jump, the
/// first frame) where crawling in over `SHADOW_FOOTPRINT_SMOOTH_TAU_S` would
/// just be wrong. It was 2.0, which ordinary flight trips constantly — every
/// doubling of camera AGL, i.e. a routine climb — and each trip steps every
/// cascade's texel size, and with it the texel-proportional foliage depth bias,
/// by 2× in ONE frame. That is precisely the scale pop the smoother was
/// introduced to remove (see `SHADOW_FOOTPRINT_SMOOTH_TAU_S`), left reachable by
/// a threshold set too low. 8× is past anything continuous camera motion
/// produces at a plausible frame rate.
const SHADOW_FOOTPRINT_SNAP_RATIO: f32 = 8.0;

/// Cap for the view footprint scale. At 32× the far cascade reaches 128 km and
/// the near cascade's texel is ~6 m — coarse, but shadows that far from the
/// vantage are a few pixels tall anyway; beyond the cap they'd be sub-pixel.
const SHADOW_MAX_FOOTPRINT_SCALE: f32 = 32.0;

/// Wall-clock smoothing time for cascade footprint changes. The old power-of-two
/// quantizer held a stable grid, then changed every texel by 2× in one frame.
/// Texel snapping already stabilizes translation; smoothing keeps zoom/altitude
/// changes continuous without bringing that scale pop back.
const SHADOW_FOOTPRINT_SMOOTH_TAU_S: f32 = 0.30;
const SHADOW_FOOTPRINT_HEADROOM: f32 = 1.12;

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

    let Some(shadow) = shadow else {
        return;
    };
    let Some(handle) = shadow.handles.get(cascade.index as usize) else {
        return;
    };
    let Some(dest) = render_assets.get(handle) else {
        return;
    };

    let src_size = depth.texture.size();
    let dst_size = dest.texture.size();
    if src_size.width != dst_size.width || src_size.height != dst_size.height {
        return;
    }
    if depth.texture.sample_count() != dest.texture.sample_count() {
        return;
    }

    // Plain full-texture copy into this cascade's own depth map — the exact
    // known-good single-map copy, one per cascade.
    ctx.command_encoder().copy_texture_to_texture(
        depth.texture.as_image_copy(),
        dest.texture.as_image_copy(),
        src_size,
    );
}

/// Mirror the live sun-shadow cascade (`SunShadowState`, owned by the rig) into
/// the render crate's `CraftShadowMaps`, so the craft hull / gear — Bevy-PBR
/// `ShipPartMaterial` — RECEIVE the same cascade the terrain / trees cast into
/// (graphics-fidelity F6b). `apply_craft_shadow` (render crate, Last) fans
/// it onto the materials. No-op until the rig's state exists and the craft
/// material is registered.
fn sync_craft_shadow(state: Option<Res<SunShadowState>>, maps: Option<ResMut<CraftShadowMaps>>) {
    let (Some(state), Some(mut maps)) = (state, maps) else {
        return;
    };
    maps.images = state.images.clone();
    maps.block = state.block;
}

/// Final shadow-only fan-out for receivers owned by the runtime render drivers.
///
/// Their ordinary material updates run in `Update` and intentionally own wind,
/// exposure, fades, and body parameters. Shadow placement now runs after the
/// camera in `PostUpdate`, so copying shadow fields there would be a frame late.
/// This single `Last` pass overwrites only the shared shadow payload immediately
/// before material extraction, making the maps, matrices, and every receiver a
/// single frame-coherent transaction.
#[allow(clippy::too_many_arguments)]
fn sync_shadow_receivers(
    state: Res<SunShadowState>,
    contact: Option<Res<super::contact_shadow::ContactShadowImage>>,
    mut grass: Option<ResMut<Assets<GrassMaterial>>>,
    mut gpu_grass: Option<ResMut<Assets<GpuGrassMaterial>>>,
    mut rocks: Option<ResMut<Assets<RockMaterial>>>,
    mut trees: Option<ResMut<Assets<TreeMaterial>>>,
    mut impostors: Option<ResMut<Assets<thalos_body_render::ground::TreeImpostorMaterial>>>,
    mut patches: Option<ResMut<Assets<GroundPatchMaterial>>>,
    mut legacy_ground: Option<ResMut<Assets<BodyTerrainMaterial>>>,
    mut tiles: Option<ResMut<Assets<TileTerrainMaterial>>>,
) {
    macro_rules! sync_plain {
        ($assets:expr) => {
            if let Some(assets) = $assets.as_deref_mut() {
                for (_, material) in assets.iter_mut() {
                    material.shadow = state.block;
                    material.sun_shadow_map_0 = state.images[0].clone();
                    material.sun_shadow_map_1 = state.images[1].clone();
                    material.sun_shadow_map_2 = state.images[2].clone();
                    material.sun_shadow_map_3 = state.images[3].clone();
                }
            }
        };
    }
    sync_plain!(grass);
    sync_plain!(gpu_grass);
    sync_plain!(rocks);
    sync_plain!(patches);

    if let Some(assets) = trees.as_deref_mut() {
        for (_, material) in assets.iter_mut() {
            material.extension.shadow = state.block;
            material.extension.sun_shadow_map_0 = state.images[0].clone();
            material.extension.sun_shadow_map_1 = state.images[1].clone();
            material.extension.sun_shadow_map_2 = state.images[2].clone();
            material.extension.sun_shadow_map_3 = state.images[3].clone();
        }
    }
    if let Some(assets) = impostors.as_deref_mut() {
        for (_, material) in assets.iter_mut() {
            material.extension.shadow = state.block;
            material.extension.sun_shadow_map_0 = state.images[0].clone();
            material.extension.sun_shadow_map_1 = state.images[1].clone();
            material.extension.sun_shadow_map_2 = state.images[2].clone();
            material.extension.sun_shadow_map_3 = state.images[3].clone();
        }
    }
    if let Some(assets) = legacy_ground.as_deref_mut() {
        for (_, material) in assets.iter_mut() {
            material.extras.shadow = state.block;
            material.sun_shadow_map_0 = state.images[0].clone();
            material.sun_shadow_map_1 = state.images[1].clone();
            material.sun_shadow_map_2 = state.images[2].clone();
            material.sun_shadow_map_3 = state.images[3].clone();
        }
    }
    if let Some(assets) = tiles.as_deref_mut() {
        for (_, material) in assets.iter_mut() {
            material.extension.shadow = state.block;
            material.extension.sun_shadow_map_0 = state.images[0].clone();
            material.extension.sun_shadow_map_1 = state.images[1].clone();
            material.extension.sun_shadow_map_2 = state.images[2].clone();
            material.extension.sun_shadow_map_3 = state.images[3].clone();
            if let Some(contact) = contact.as_deref() {
                material.extension.contact_shadow_map = contact.handle.clone();
            }
        }
    }
}

pub struct SunShadowPlugin;

impl Plugin for SunShadowPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<SunShadowImage>::default())
            .add_plugins(ExtractComponentPlugin::<SunShadowCascade>::default())
            .add_systems(Startup, setup_sun_shadow)
            .add_systems(
                PostUpdate,
                (
                    crate::rendering::real_space::update_real_space_origin,
                    update_sun_shadow_camera,
                    sync_craft_shadow,
                )
                    .chain()
                    // MUST run after big_space has settled every `CellCoord` for
                    // this frame. `CellCoord::recenter_large_transforms` is
                    // registered PLAIN in `PostUpdate` by `BigSpaceCorePlugin` —
                    // it is NOT inside `TransformSystems::Propagate` — so
                    // `.before(TransformSystems::Propagate)` alone leaves it
                    // completely unordered against this chain, and the
                    // multithreaded executor is free to slot it *between*
                    // `update_real_space_origin` (which reads the floating
                    // origin's cell) and `update_sun_shadow_camera` (which reads
                    // the same cell again). On the frames where it did, the whole
                    // cascade rig was placed one 1 km grid cell away from the
                    // world it was meant to cover — cascade 0's half-extent is
                    // ~450 m, so every near-field receiver fell out of the crisp
                    // cascade and dropped to the coarse one, whose foliage bias
                    // erases tree/shrub shadows outright. Non-deterministic frame
                    // to frame ⇒ shadows flickering in and out while the camera
                    // moved (INC-20260730T223451Z; the tell is
                    // `origin_frame_error_m` landing on an exact multiple of the
                    // 1 km cell size, which `just diag` reads as
                    // `shadow_frame_desync`).
                    //
                    // Order against the SYSTEM, never the enclosing
                    // `BigSpaceSystems::RecenterLargeTransforms` set: that set
                    // also contains `BigSpace::find_floating_origin`, which *is*
                    // inside `TransformSystems::Propagate`, so an `.after(set)`
                    // here would form a cycle with the `.before` below.
                    .after(CellCoord::recenter_large_transforms)
                    .before(TransformSystems::Propagate),
            )
            .add_systems(Last, sync_shadow_receivers);

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

        // Spawn-time baseline only — `update_sun_shadow_camera` overwrites this
        // every frame with the live square-on-ground extents (U ≠ V).
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
/// of the given half-extents + far plane (`OrthographicProjection::get_clip_from_view`
/// swaps near/far). Built by hand so it stays in lockstep with the camera
/// regardless of when Bevy's projection-update system runs this frame.
///
/// Takes the two half-extents separately: the box is square on the GROUND, not
/// in the light plane, so U (cross-sun) and V (along-sun) differ (see the box
/// note in [`update_sun_shadow_camera`]). It stays SYMMETRIC about the camera —
/// the up-sun caster margin is applied by sliding the eye along the V axis, not
/// by an off-centre frustum, so the live [`OrthographicProjection`] (which is
/// centred by construction) keeps matching this matrix exactly.
fn cascade_clip_from_view(half_u: f32, half_v: f32, far: f32) -> Mat4 {
    Mat4::orthographic_rh(-half_u, half_u, -half_v, half_v, far, SHADOW_NEAR_M)
}

/// Everything [`update_sun_shadow_camera`] must remember between frames.
///
/// One `Local` rather than six: the system sits at Bevy's `SystemParam` tuple
/// ceiling, and these fields are one concept anyway — the state that keeps
/// cascade placement STABLE frame to frame. Every field here exists because
/// recomputing it from scratch each frame made something visibly jitter.
#[derive(Default)]
struct SunShadowMemory {
    /// Seconds accumulated since the last `stability_gauge` emission (1 Hz).
    diagnostic_elapsed_s: f32,
    /// Body-fixed texel-snap anchor — see the snap note in the system body.
    snap_anchor: Option<(BodyId, DVec3)>,
    /// Smoothed view-footprint scale (`SHADOW_FOOTPRINT_SMOOTH_TAU_S`).
    footprint_scale: f32,
    /// Previous frame's light basis, parallel-transported so the cascade box
    /// does not spin about the sun axis as the basis is rebuilt.
    previous_light_right: Option<Vec3>,
    /// Latched craft-local mode — hysteresis, see [`SHADOW_CRAFT_LOCAL_EXIT_M`].
    craft_local: bool,
    /// Last terrain height that actually resolved, and the body it belongs to.
    /// Held across misses so a cold height source cannot step the cascade
    /// centre and footprint (see where it is read).
    last_terrain_h: Option<(BodyId, f64)>,
}

/// Aim every cascade camera down the sun over the craft and publish their
/// transforms. Near the surface: three ground-projected cascades. In orbit /
/// high flight: one craft-centred cascade (self-shadow only). Fully disabled
/// only when there is no terrain body / camera / states at all.
#[allow(clippy::type_complexity)]
fn update_sun_shadow_camera(
    sim: Res<SimulationState>,
    cache: Res<SolarSystemState>,
    grid: Query<&Grid, With<BigSpace>>,
    ship_cam: Query<(&CellCoord, &Transform), With<ShipCamera>>,
    origin: Res<crate::rendering::real_space::RealSpaceOrigin>,
    view_anchor: Res<crate::rendering::view_anchor::ViewAnchor>,
    real_time: Res<Time<Real>>,
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
            Option<&bevy::camera::visibility::VisibleEntities>,
        ),
        Without<ShipCamera>,
    >,
    mut state: ResMut<SunShadowState>,
    mut memory: Local<SunShadowMemory>,
) {
    memory.diagnostic_elapsed_s += real_time.delta_secs();

    'resolve: {
        let Some(states) = cache.states.as_deref() else {
            break 'resolve;
        };
        let (Ok(root_grid), Ok((cam_cell, cam_xform))) = (grid.single(), ship_cam.single()) else {
            break 'resolve;
        };
        // Where is the view? The `ViewAnchor` is the one coherent answer
        // (body-fixed, matching epoch — see `view_anchor.rs`); the raw
        // ShipCamera pose is only the fallback for frames before the anchor
        // first resolves. The raw entity is NOT the view whenever a mode poses
        // the view elsewhere: the flight follow-cam writes it at the CRAFT
        // mid-frame before the god-view/capture drivers re-pose it, so this
        // system — sampling in PostUpdate, between those writers — read the
        // craft. With the hub's placeholder craft parked in a 200 km orbit,
        // the craft-local gate below latched self-shadow-only mode at a 600 m
        // god view, parking every ground cascade and erasing all structure and
        // tree shadows while the craft kept its own
        // (INC-20260731T004704Z-craft-local-gate-read-the-craft).
        let camera_world = match view_anchor.resolved.filter(|a| states.get(a.body).is_some()) {
            Some(anchor) => anchor.cam_world(states),
            None => root_grid.grid_position_double(cam_cell, cam_xform),
        };
        let cam_pos = origin.to_render(camera_world);

        let mut best: Option<(BodyId, f32, f32)> = None;
        for body_state in states {
            let Some(def) = sim.system.bodies.get(body_state.id) else {
                continue;
            };
            if !def.terrain.is_some() {
                continue;
            }
            let radius = def.radius_m as f32;
            let altitude = (camera_world - body_state.position).length() as f32 - radius;
            if best.is_none_or(|(_, a, _)| altitude < a) {
                best = Some((body_state.id, altitude, radius));
            }
        }
        let Some((active_id, altitude, body_radius_m)) = best else {
            break 'resolve;
        };
        // Off-surface, don't disable — switch to craft-local self-shadow mode
        // (see `SHADOW_MAX_ALTITUDE_M`). The gate is the CAMERA's altitude, so
        // any near-surface view (flight, god view, freecam) keeps ground
        // shadows regardless of where the craft is. Latched across a hysteresis
        // band, because the switch parks cascades 1–2 outright: a bare
        // threshold let a camera loitering at the boundary strobe every ground
        // shadow in the world (see `SHADOW_CRAFT_LOCAL_EXIT_M`).
        let craft_local = if memory.craft_local {
            altitude > SHADOW_CRAFT_LOCAL_EXIT_M
        } else {
            altitude > SHADOW_MAX_ALTITUDE_M
        };
        memory.craft_local = craft_local;
        let (Some(star), Some(body_state)) = (states.first(), states.get(active_id)) else {
            break 'resolve;
        };

        let offset = star.position - body_state.position;
        let sun_dir = if offset.length_squared() <= 0.0 {
            Vec3::Y
        } else {
            // Keep lighting and shadow geometry on the same continuous sun.
            // The old 0.1° hold moved every shadow edge in visible time steps.
            offset.normalize().as_vec3()
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
        let player_inertial = if craft_local {
            sim.simulation.ship_state().position
        } else {
            camera_world
        };
        let radial = player_inertial - body_state.position;
        let r = radial.length();
        let radial_dir = if r > 1.0e-3 { radial / r } else { DVec3::Y };
        let up_radial = radial_dir.as_vec3();
        // ── Night stand-down (surface mode only) ────────────────────────────
        // See `SHADOW_NIGHT_FADE_SIN`. Craft-local mode is exempt: at orbital
        // altitude the tangent-plane test is wrong (the true horizon is
        // depressed), and the single hull cascade is harmless while the hull
        // is unlit — whether the craft sits in the umbra is the lighting
        // system's question, not the rig's.
        let night_fade = if craft_local {
            1.0
        } else {
            let sun_sin_raw = sun_dir.dot(up_radial);
            ((sun_sin_raw + SHADOW_NIGHT_FADE_SIN) / SHADOW_NIGHT_FADE_SIN).clamp(0.0, 1.0)
        };
        if night_fade <= 0.0 {
            // Fully night: identical to the rig-off path — cameras deactivate
            // and `gate.x` zeroes, so every sampler early-outs fully lit and
            // the moon's direct term reaches receivers ungated.
            break 'resolve;
        }
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
        // A miss HOLDS THE LAST RESOLVED HEIGHT for this body rather than
        // snapping to the datum. The datum fallback is a step equal to the whole
        // site elevation, and `terrain_h` feeds both the cascade centre and
        // (through `cam_agl`) the footprint scale, where the demanded box moves
        // by up to 6× that step — so an intermittently-cold height source
        // shoved every cascade extent, texel size, and derived depth bias around
        // between neighbouring frames. Holding the last good sample is strictly
        // closer than 0.0 for a camera that has not teleported, and the source
        // resolves again within a few frames once tiles stream in. The datum is
        // still the floor before any sample has ever landed, which reproduces
        // the old centring exactly.
        let sampled_h = height_sources.get(active_id).and_then(|hs| {
            hs.sample_height_m(dir_body, crate::local_physics::PHYSICS_QUERY_TILE_LOD_M)
        });
        let terrain_h = match sampled_h {
            Some(h) => {
                let h = h as f64;
                memory.last_terrain_h = Some((active_id, h));
                h
            }
            None => memory
                .last_terrain_h
                .filter(|(body, _)| *body == active_id)
                .map(|(_, h)| h)
                .unwrap_or(0.0),
        };
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
            let fwd = cam_xform.rotation * Vec3::NEG_Z;
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
            let target = (raw * SHADOW_FOOTPRINT_HEADROOM).clamp(1.0, SHADOW_MAX_FOOTPRINT_SCALE);
            if memory.footprint_scale <= 0.0
                || target / memory.footprint_scale.max(1.0e-3) > SHADOW_FOOTPRINT_SNAP_RATIO
            {
                memory.footprint_scale = target;
            } else {
                let alpha = 1.0 - (-real_time.delta_secs() / SHADOW_FOOTPRINT_SMOOTH_TAU_S).exp();
                memory.footprint_scale += (target - memory.footprint_scale) * alpha;
            }
            memory.footprint_scale
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
        //
        // **The reach argument is now the GROUND half-extent, not the light-plane
        // one.** With the square-on-ground box the azimuth reach is `half`
        // outright, so the depth slack a cascade needs is `half · cos(elev)` —
        // the old `half · cos/sin` was bounding a ground reach the box no longer
        // has. At a 15° sun that is a ~3.9× shorter depth range for the same
        // coverage, which tightens Depth32Float precision and shrinks every
        // bias derived from `1/(far − near)`.
        let sin_elev = sun_dir.dot(up_radial).clamp(SHADOW_MIN_SUN_SIN, 1.0);
        let ground_slack_ground_reach = |half: f32| -> f32 {
            let cos_elev = (1.0 - sin_elev * sin_elev).max(0.0).sqrt();
            (half * cos_elev).min(SHADOW_SLACK_MAX_M)
        };
        // Rotation is shared by every cascade (only translation differs); any
        // eye distance yields the same rotation.
        //
        // ── Why the basis is AZIMUTH-ALIGNED, not parallel-transported ───────
        // The box below is square on the ground, which means its two light-plane
        // axes carry DIFFERENT world scales. That is only expressible if the
        // axes line up with the directions whose scales differ: the sun azimuth
        // (compressed by `sin(elev)` when the ground projects into the light
        // plane) and the cross-azimuth direction (1:1). `sun_dir × up_radial` is
        // exactly the cross-azimuth axis, so it becomes U and the azimuth
        // becomes V. A parallel-transported roll frame — which is what this was,
        // for continuity — would sit at an arbitrary angle to both, and an
        // axis-aligned rectangle in it cannot represent the anisotropy at all.
        //
        // Transport is kept for the DEGENERATE case only: within a hair of the
        // sun crossing the local zenith the cross product vanishes and its
        // direction is meaningless. That is also precisely where the anisotropy
        // vanishes (`sin(elev) → 1`, box → square), so the frame is free to spin
        // there without changing coverage — it only re-rasterizes, and the
        // smooth `aniso` blend below means extents stay continuous through it.
        let cross_azimuth = sun_dir.cross(up_radial);
        let light_right = if cross_azimuth.length_squared() > 1.0e-8 {
            cross_azimuth.normalize()
        } else {
            let transported = memory
                .previous_light_right
                .map(|right| right - sun_dir * right.dot(sun_dir))
                .unwrap_or(Vec3::ZERO);
            if transported.length_squared() > 1.0e-6 {
                transported.normalize()
            } else {
                sun_dir.cross(Vec3::X).normalize_or_zero()
            }
        };
        let light_up = sun_dir.cross(light_right).normalize_or_zero();
        memory.previous_light_right = Some(light_right);
        let light_rotation =
            Quat::from_mat3(&Mat3::from_cols(light_right, light_up, sun_dir)).normalize();

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
            let anchor_bf = match memory.snap_anchor {
                Some((b, a))
                    if b == active_id && (a - center_bf).length() < SNAP_ANCHOR_REACH_M =>
                {
                    a
                }
                _ => {
                    memory.snap_anchor = Some((active_id, center_bf));
                    center_bf
                }
            };
            let anchor_inertial = body_state.position + body_state.orientation * anchor_bf;
            (center_inertial - anchor_inertial).as_vec3()
        };

        // Craft-local mode runs the two near cascades; the far ones' matrices
        // are zeroed (the shader's `clip.w <= 0` skip sentinel) and their
        // cameras deactivated, so their stale depth maps are never read.
        // TWO, not one: cascade 0 is now a ±64 m box, which a tall launch stack
        // can overflow, and up here it is the ONLY caster there is (stock Bevy
        // CSM is off). Cascade 1's ±400 m brackets any craft this game can
        // build, so the hull keeps a complete self-shadow while cascade 0 gives
        // the near panels their detail.
        let active_cascades = if craft_local { 2 } else { CASCADE_COUNT };
        let mut block = ShadowCascadeBlock::default();
        let mut looks = [Transform::IDENTITY; CASCADE_COUNT];
        let mut half_us = [0.0_f32; CASCADE_COUNT];
        let mut half_vs = [0.0_f32; CASCADE_COUNT];
        let mut fars = [0.0_f32; CASCADE_COUNT];
        // ── Square on the GROUND, not in the light plane ─────────────────────
        //
        // The map is sampled by receivers standing on the ground, so what has to
        // be uniform is the texel's footprint THERE. Projecting ground → light
        // plane compresses the sun-azimuth direction by `sin(elev)` and leaves
        // the cross-azimuth direction alone. A box that is square in the light
        // plane therefore covers `half / sin(elev)` of ground along the azimuth
        // while spending the same 4096 texels on it — at a 15° sun that is a
        // ~3.9× coarser ground texel in one direction than the other, and it is
        // the whole reason near-field shadows read as an elongated staircase.
        //
        // So: shrink the V (along-sun) half-extent by `sin(elev)`, which makes
        // the ground footprint a SQUARE of half-width `half` and the ground
        // texel isotropic. This is not a coverage cut in any direction that was
        // being used — the old box's extra along-azimuth reach was an accident
        // of the projection, not a decision, and it bought ground nobody framed.
        //
        // The one thing genuinely lost is casters standing up-sun of the covered
        // ground, whose shadows fall INTO it. Those are bounded by caster height
        // (`CASCADE_MAX_CASTER_M`): a caster `h` tall reaches `h / tan(elev)`
        // down-sun, which is only `h · cos(elev)` of extra LIGHT-space margin.
        // That margin is added to V and paid for by sliding the eye up-sun by
        // half of it, keeping the frustum symmetric.
        //
        // **Craft-local mode opts out.** All of the above reasons about a GROUND
        // plane the receivers stand on. In orbit there is none — the receiver is
        // the hull, a 3-D object, and the light-plane box is already the right
        // shape for it. Compressing V by `sin(elev)` there would squash the box
        // along the azimuth and clip the craft out of its own shadow map, and
        // the up-sun caster margin has nothing to catch. `sin_e = 1, cos_e = 0`
        // makes the box square and the margin zero, i.e. exactly the behaviour
        // this mode had before.
        let (sin_e, cos_e) = if craft_local {
            (1.0, 0.0)
        } else {
            let s = sin_elev;
            (s, (1.0 - s * s).max(0.0).sqrt())
        };
        for i in 0..CASCADE_COUNT {
            if i >= active_cascades {
                block.view_proj[i] = Mat4::ZERO;
                continue;
            }
            let half = (CASCADE_HALF_EXTENTS_M[i] * footprint).max(CASCADE_MIN_HALF_M[i]);
            // U spans the cross-sun ground directly; V spans the same ground
            // distance once compressed, plus the up-sun caster margin.
            let caster_margin = CASCADE_MAX_CASTER_M[i] * footprint * cos_e;
            let half_u = half;
            let half_v = half * sin_e + 0.5 * caster_margin;
            // Up-sun eye offset + far plane bracket this cascade's whole ground
            // footprint along the sun azimuth (see the slack note) plus terrain
            // relief above/below the tangent plane. The ground reach along the
            // azimuth is now `half`, not `half / sin(elev)`, so the slack — and
            // with it the ortho depth range, and every bias derived from it —
            // collapses by that same factor at a low sun.
            let slack = ground_slack_ground_reach(half) + SHADOW_RELIEF_MARGIN_M;
            let back = SHADOW_BACK_DISTANCE_M * footprint + slack;
            let far = CASCADE_FARS_M[i] * footprint + 2.0 * slack;
            half_us[i] = half_u;
            half_vs[i] = half_v;
            fars[i] = far;
            // Texel-snap the cascade centre to ITS shadow-map grid in the light
            // plane, so the ortho frustum slides in whole-texel steps and shadow
            // edges stop crawling as the centre drifts (stable CSM). Each cascade
            // snaps to its own (coarser, near→far) grid — and now to its own
            // grid PER AXIS, since U and V texels differ. The phase comes from
            // `snap_rel` — the centre RELATIVE to a body-fixed anchor — so the
            // grid co-moves with the rotating ground (see the snap note above).
            let texel_u = (2.0 * half_u) / SHADOW_MAP_SIZE as f32;
            let texel_v = (2.0 * half_v) / SHADOW_MAP_SIZE as f32;
            let cr = snap_rel.dot(light_right);
            let cu = snap_rel.dot(light_up);
            let snap = ((cr / texel_u).round() * texel_u - cr) * light_right
                + ((cu / texel_v).round() * texel_v - cu) * light_up;
            // Slide the box up-sun by half the caster margin so the receiver
            // region stays centred while the margin lands entirely on the up-sun
            // side (where casters are) — QUANTIZED TO WHOLE V TEXELS. The margin
            // varies continuously with sun elevation and footprint, so shifting
            // by its raw value would hand every cascade a fresh sub-texel phase
            // every frame and re-rasterize every shadow edge — precisely the
            // crawl the snap above exists to prevent, re-entered by the back
            // door.
            let v_shift = (0.5 * caster_margin / texel_v).round() * texel_v;
            let center_i = center + snap + light_up * v_shift;
            let eye_i = center_i + sun_dir * back;
            let look_i = Transform::from_translation(eye_i).with_rotation(light_rotation);
            block.view_proj[i] =
                cascade_clip_from_view(half_u, half_v, far) * look_i.to_matrix().inverse();
            // x = clip units per metre of light-space depth (orthographic z is
            // linear), y/z = texel size in world metres on the U/V axes — the
            // shared sampler derives its capped, texel-proportional bias +
            // receiver offset from `y` and shapes its PCSS kernel from the pair
            // (see the bias model note in `shadow.wgsl`). `y` stays the U texel
            // specifically so the bias model keeps the exact meaning it was
            // calibrated against: U is the axis with no projective compression.
            block.params[i] = Vec4::new(1.0 / (far - SHADOW_NEAR_M), texel_u, texel_v, 0.0);
            looks[i] = look_i;
        }
        // z = the contact-shadow gate (W18a). Published from the rig rather than
        // per-material so every consumer of the block inherits it. Note it rides
        // *inside* the cascade gate: when `gate.x == 0` the samplers early-out
        // fully lit and the contact term is moot along with them — the rig-off
        // cases (orbital map terrain, inactive pass) want no shadow at all.
        block.gate = Vec4::new(
            SHADOW_STRENGTH * night_fade,
            active_cascades as f32,
            contact.shadow_gate(),
            0.0,
        );
        // Sun direction (toward the sun) drives the sampler's slope-scaled bias.
        block.sun_dir = Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, 0.0);
        state.block = block;

        let mut dbg_visible = [0usize; CASCADE_COUNT];
        for (mut tf, mut cam, mut proj, cascade, visible) in &mut shadow_cams {
            if let Some(visible) = visible {
                dbg_visible[cascade.index as usize] =
                    visible.len(core::any::TypeId::of::<bevy::mesh::Mesh3d>());
            }
            let idx = cascade.index as usize;
            let on = idx < active_cascades;
            if on {
                *tf = looks[idx];
                // Keep the LIVE camera projection in lockstep with the
                // hand-built `block.view_proj` — the spawn-time projection only
                // covers the unscaled baseline footprint.
                let mut ortho = OrthographicProjection {
                    // Width/height are the U/V half-extents doubled — the box is
                    // square on the ground, not in the light plane, so these
                    // differ. `ScalingMode::Fixed` maps them straight onto the
                    // square map, which IS the anisotropic sampling that makes
                    // the ground texel isotropic.
                    scaling_mode: ScalingMode::Fixed {
                        width: half_us[idx] * 2.0,
                        height: half_vs[idx] * 2.0,
                    },
                    near: SHADOW_NEAR_M,
                    far: fars[idx],
                    ..OrthographicProjection::default_3d()
                };
                // Seat `area` HERE, not later. `get_clip_from_view` (and with
                // it `update_frusta`'s culling frustum) reads `area`, which
                // `default_3d()` leaves at a ±1 m placeholder; only
                // `camera_system` recomputes it from the scaling mode, and
                // that system is UNORDERED against this one. On frames where
                // it ran first, `update_frusta` (post-Propagate, reading the
                // LIVE projection) built a two-metre frustum: km-scale
                // terrain-tile casters still intersected it, every small
                // caster (buildings, posts, rocks) was culled out of its own
                // shadow map, and rendering looked fine because extraction
                // uses the clip matrix `camera_system` cached the frame
                // before. `update()` with the map extent is exact for
                // `ScalingMode::Fixed` (the arguments are ignored) and makes
                // the write order-independent
                // (INC-20260731T011523Z-cascade-frustum-default-area).
                ortho.update(SHADOW_MAP_SIZE as f32, SHADOW_MAP_SIZE as f32);
                *proj = Projection::Orthographic(ortho);
            }
            cam.is_active = on;
        }

        if memory.diagnostic_elapsed_s >= 1.0 {
            memory.diagnostic_elapsed_s = 0.0;
            let expected_origin = root_grid.grid_position_double(cam_cell, &Transform::IDENTITY);
            info!(
                target: "thalos::diagnostic::shadow",
                event = "stability_gauge",
                body_id = active_id,
                origin_frame_error_m = origin.position.distance(expected_origin),
                footprint_scale = footprint,
                // Both axes: the box is square on the ground, so these diverge
                // with sun elevation and their RATIO is the tell that the
                // anisotropy is being applied at all (1.0 ⇒ overhead sun or a
                // regression back to a light-plane-square box).
                cascade0_texel_u_m = block.params[0].y,
                cascade0_texel_v_m = block.params[0].z,
                active_cascades = active_cascades,
                sun_sin_elev = sun_dir.dot(up_radial),
                night_fade = night_fade,
                // The altitude the craft-local gate actually saw this frame,
                // plus the latch it produced. `active_cascades` alone cannot
                // distinguish "the gate's camera altitude is wrong" from "the
                // hysteresis failed to unlatch" (INC pending: craft-local stuck
                // on at a surface god view, killing every non-craft shadow).
                gate_alt_m = altitude,
                craft_local = craft_local,
                // Meshes that passed culling for each cascade camera LAST
                // frame (`VisibleEntities` is written by check_visibility
                // after this system runs). Terrain caster twins alone put
                // this in the hundreds; a near-zero count with props in
                // frame means the casters are being CULLED, not mis-drawn.
                cascade0_visible = dbg_visible[0] as u32,
                cascade1_visible = dbg_visible[1] as u32,
                cascade2_visible = dbg_visible[2] as u32,
                cascade3_visible = dbg_visible[3] as u32,
                "shadow stability gauge"
            );
        }
        return;
    }

    for (_tf, mut cam, _proj, _cascade, _visible) in &mut shadow_cams {
        cam.is_active = false;
    }
    state.block.gate.x = 0.0;
}
