//! NTR-X1 driver — the standard-path tile renderer, in-game. **This is the
//! default ground renderer** (keystone ADR-20260723T142945Z); the udlod stack
//! it replaces is legacy, kept only as an A/B baseline behind
//! `THALOS_TILE_RENDERER=0` and slated for deletion once the remaining tile
//! rows close (`ntr §6`).
//!
//! When the tile path owns a body (the default):
//!
//! - legacy udlod ground terrain stands down (`terrain_residency::try_spawn`
//!   gate) for that body,
//! - the first `ViewAnchor`-resolved terrain body gets a
//!   [`TileTerrainRoot`] on its `RealSpaceBody` grid entity — tiles stream as
//!   ordinary `Mesh` + `StandardMaterial` children of the rotating body grid,
//!   fed by the body's canonical `Arc<dyn SurfaceQuery>`,
//! - the selection eye is republished every frame from `ViewAnchor`
//!   (body-fixed camera position — no per-mode camera plumbing).
//!
//! **Surface detail** (grass / trees / rocks, colliders, camera floor, HUD
//! altitude) rides along: on install the root's CPU
//! [`TileHeightMirror`](thalos_body_render::tiles::TileHeightMirror) is
//! published as the body's [`RenderedGround`], replacing udlod's atlas mirror
//! in the registry and behind the body's `HeightSource`. Scatter's residency
//! gate and its ground samples then read the *tiles that are actually drawn* —
//! the same contract it has under udlod, so nothing in the scatter drivers
//! knows which renderer is up.
//!
//! **Structure pads** level the tile ground the same way they level udlod's:
//! the provider samples the canonical surface through the body's shared
//! [`FlattenedSurface`] decorator, taken from the one
//! [`TerrainFlattenRegistry`] handle the runway / base editor write. Both
//! renderers therefore read the same pads from the same object — the flatten is
//! a property of the *body's ground*, not of whichever renderer is drawing it.
//! The handle is read per tile *bake*, so a pad installed while tiles are
//! already resident over it (the base editor's runtime flatten) needs those
//! tiles dropped to re-bake; the tile path does not consume
//! [`TerrainRebuildRequest`] for that yet (NTR-X2b). Boot is unaffected — the
//! runway's pad is installed before the view moves to the site, so the tiles
//! that stream in there bake level from the start.
//!
//! The same handle also drives [`publish_pad_refinement_sites`], which turns
//! every pad into a `RefinementSite` floor on the tile selector. Baking a tile
//! level is not enough on its own — a tile too coarse to resolve the footprint
//! bakes a mesh that cuts straight across it and puts the terrain the pad
//! removed back over the base (INC-20260725T184654Z).
//!
//! **Body coverage.** The root follows [`ViewAnchor`]: when the anchor settles
//! on a different body for [`HANDOFF_DWELL_S`], the resident tiles are released
//! and the root re-installs there. One root at a time — the residency budget is
//! sized for one body (INC-20260725T012104Z) and you can only be near one — with
//! distant bodies left to the impostor billboard, as they already were. This
//! replaced "first anchor body wins, udlod streams the rest", which was the only
//! reason the legacy ground had to stay wired.
//!
//! Remaining slice-1 limitation (NTR-X1's backlog row): impostor visibility is
//! left to the existing swap, so a sliver of billboard may peek past the tile
//! limb (tiles win depth where they cover).

use std::sync::{Arc, OnceLock};

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use thalos_body_render::tiles::material::{TileShadingParams, TileTerrainMaterial, tile_material};
use thalos_body_render::tiles::{
    RefinementSite, SurfaceQueryProvider, TileEye, TileEyeTarget, TileStreamSet, TileTerrainRoot,
};
use thalos_body_render::{CpuPipelineHeightSource, RenderedGround, RenderedGroundHeightSource};
use thalos_physics_local::HeightSourceRegistry;
use thalos_terrain::{FlattenedSurface, SurfaceQuery};
use thalos_world::BodyId;

use super::ground_terrain::TerrainFlattenRegistry;
use super::terrain_residency::TerrainRebuildRequest;
use super::types::RealSpaceBody;
use super::view_anchor::ViewAnchor;
use crate::SimulationState;
use crate::coords::SHIP_LAYER;
use crate::solar_system_state::SolarSystemState;
use crate::terrain_registry::{BodySurfaceRegistry, RenderedGroundRegistry};
use std::sync::Mutex;

/// Bodies currently owned by the tile renderer — read by
/// `terrain_residency::try_spawn` (which may run before this plugin's
/// systems), hence a process-global rather than a Bevy resource.
static TILE_RENDERED: Mutex<Vec<thalos_world::BodyId>> = Mutex::new(Vec::new());

/// Is `body_id` rendered by the tile path (legacy udlod stands down for it)?
pub fn tile_rendered(body_id: BodyId) -> bool {
    tile_renderer_enabled() && TILE_RENDERED.lock().is_ok_and(|v| v.contains(&body_id))
}

/// How long `tile_claim_pending` will hold the legacy near-tier spawn back.
///
/// Generous: the claim needs a resolved `ViewAnchor`, which lands ~1.2 s into a
/// boot. The cap exists only so a *never*-resolving anchor degrades to legacy
/// terrain instead of leaving the player on a body with no ground at all.
const TILE_CLAIM_HOLD_S: f32 = 10.0;

/// Is the tile renderer about to claim its body (so a legacy near-tier atlas
/// would be allocated and thrown away)?
///
/// `ensure_tile_root` installs on the first anchor-resolved body, but the anchor
/// resolves ~1.2 s after `terrain_residency` first runs — so udlod used to
/// allocate its ~890 MB near-tier atlas array for that body and drop it again
/// moments later via the rebuild request. Holding the legacy spawn while a claim
/// is pending means the allocation never happens
/// (INC-20260725T012104Z-tile-residency-had-no-budget).
///
/// Scoped to the *dominant* body by the caller: every other body keeps the
/// legacy path immediately, which is the arrangement the udlod carve-out needs.
pub fn tile_claim_pending() -> bool {
    if !tile_renderer_enabled() {
        return false;
    }
    if TILE_RENDERED.lock().is_ok_and(|v| !v.is_empty()) {
        return false;
    }
    static FIRST_ASK: OnceLock<std::time::Instant> = OnceLock::new();
    let waited = FIRST_ASK.get_or_init(std::time::Instant::now).elapsed();
    if waited.as_secs_f32() >= TILE_CLAIM_HOLD_S {
        static WARNED: OnceLock<()> = OnceLock::new();
        if WARNED.set(()).is_ok() {
            warn!(
                "tile terrain: no body claimed after {TILE_CLAIM_HOLD_S:.0}s \
                 (unresolved ViewAnchor?) — releasing the legacy udlod ground"
            );
        }
        return false;
    }
    true
}

/// One env check, cached. The tile renderer is the **default**;
/// `THALOS_TILE_RENDERER=0|false|off|no` drops the process back onto the
/// legacy udlod ground for every body — an A/B baseline (the `renderer`
/// compare axis), not a supported production mode.
///
/// Cached in a `OnceLock` because the choice is structural (it decides which
/// ground streams at all), so it must not change mid-process — which is also
/// why the capture host restarts when this key changes.
pub fn tile_renderer_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("THALOS_TILE_RENDERER")
            .map(|v| {
                !matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(true)
    })
}

/// Capture-only inspection mode for the tile material, read once from
/// `THALOS_TERRAIN_INSPECTION` — the same env key (and the same spellings)
/// udlod's ground reads, so `just compare <scene> terrain-lighting` now means
/// the same thing on both renderers instead of silently doing nothing on this
/// one. Never changes geometry, tile synthesis, or provider data.
fn inspection_mode() -> u32 {
    static MODE: OnceLock<u32> = OnceLock::new();
    *MODE.get_or_init(|| {
        let Ok(value) = std::env::var("THALOS_TERRAIN_INSPECTION") else {
            return 0;
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "lit" | "default" | "off" | "0" => 0,
            // Numeric spellings accepted deliberately: `=1`/`=2` is what a
            // udlod-era muscle-memory probe types, and rejecting them silently
            // rendered lit — which cost a whole debugging session
            // (BL-20260727T004857Z: "inspection does not reach the tile
            // material" was really this match arm warning and falling through).
            "fullbright" | "albedo" | "on" | "1" => 1,
            "geo-normal" | "geometric-normal" | "smooth-normal" | "2" => 2,
            // udlod's `legacy-regolith` has no tile-path meaning; render lit
            // rather than warn, so the shared axis stays usable.
            "legacy-regolith" | "unfiltered-regolith" => 0,
            other => {
                warn!("unknown THALOS_TERRAIN_INSPECTION={other:?}; using lit tile terrain");
                0
            }
        }
    })
}

/// Driver-side tag pairing the tile root with its body id.
#[derive(Component)]
pub struct TileTerrainBody {
    pub body_id: BodyId,
}

pub struct TileTerrainDriverPlugin;

impl Plugin for TileTerrainDriverPlugin {
    fn build(&self, app: &mut App) {
        if !tile_renderer_enabled() {
            warn!(
                "THALOS_TILE_RENDERER=0 — LEGACY udlod ground terrain active (A/B baseline only); \
                 unset the variable for the default standard-path tile renderer"
            );
            return;
        }
        info!("standard-path tile terrain active; legacy udlod ground terrain gated off");
        app.init_resource::<TileRootHandoff>();
        app.add_systems(
            Update,
            (
                ensure_tile_root,
                update_tile_eye,
                publish_pad_refinement_sites,
                update_tile_material_params,
            )
                .chain()
                // The eye carries the body's f64 pose and the streamer places
                // every tile from it, so this must sit exactly where
                // `update_runway_transform` sits — after the frame's body states
                // are resolved, and before the tiles are placed from them. A
                // frame of slip here slides the ground metres against the base
                // built on it (232 m/s of surface speed at Thalos's equator).
                .after(crate::solar_system_state::sync_solar_system_state)
                .before(TileStreamSet),
        );
    }
}

/// How long [`ViewAnchor`] must agree on a *different* body before the root
/// moves to it.
///
/// A handoff throws away the whole resident set and re-streams, so a boundary
/// that flickers between two bodies must not be able to drive it. The anchor is
/// stable in practice (it follows the dominant body, which has its own
/// hysteresis), so this is cheap insurance rather than a load-bearing filter.
const HANDOFF_DWELL_S: f32 = 1.0;

/// Pending body change for [`ensure_tile_root`]'s dwell guard.
#[derive(Resource, Default)]
pub struct TileRootHandoff {
    pending: Option<(BodyId, f32)>,
}

/// Install the tile terrain on the anchor-resolved body, and **follow the anchor
/// when it changes bodies**.
///
/// This used to install once, on the first anchor-resolved body, and stay there
/// for the session — every other body was ground-rendered by legacy udlod, which
/// is the only reason that path was still wired (CLAUDE.md's ground-renderer
/// invariant). With udlod moving behind an off-by-default feature there is no
/// second renderer to fall back to, so the root has to go where the view goes or
/// a transfer arrives at a planet with no terrain.
///
/// One root at a time, deliberately: the residency budget is sized for one body
/// (INC-20260725T012104Z), and you can only be near one at a time. Distant
/// bodies are the impostor's job, exactly as they already were.
#[allow(clippy::too_many_arguments)]
fn ensure_tile_root(
    anchor: Res<ViewAnchor>,
    time: Res<Time<Real>>,
    surfaces: Res<BodySurfaceRegistry>,
    sim: Res<crate::solar_system_state::SimulationState>,
    bodies: Query<(Entity, &RealSpaceBody)>,
    mut existing: Query<(Entity, &mut TileTerrainRoot, &TileTerrainBody)>,
    mut handoff: ResMut<TileRootHandoff>,
    mut materials: ResMut<Assets<TileTerrainMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut rebuild: ResMut<TerrainRebuildRequest>,
    mut rendered_ground: ResMut<RenderedGroundRegistry>,
    mut height_sources: ResMut<HeightSourceRegistry>,
    mut flatten_registry: ResMut<TerrainFlattenRegistry>,
    mut tile_cache: ResMut<super::tile_cache::TileCacheRegistry>,
    mut commands: Commands,
) {
    let Some(resolved) = anchor.resolved else {
        return;
    };

    if let Some((held_entity, mut root, held)) = existing.iter_mut().next() {
        if held.body_id == resolved.body {
            handoff.pending = None;
            return;
        }
        // Different body — start (or continue) the dwell before committing.
        let waited = match handoff.pending {
            Some((body, elapsed)) if body == resolved.body => elapsed + time.delta_secs(),
            _ => 0.0,
        };
        if waited < HANDOFF_DWELL_S {
            handoff.pending = Some((resolved.body, waited));
            return;
        }
        handoff.pending = None;

        // Commit the handoff. Order matters: release the tiles first (they are
        // children of the big_space root, so nothing else would ever despawn
        // them), then drop the body's claim so `terrain_residency` is free to
        // take it back if the legacy feature is compiled in.
        let released = root.release_all();
        let released_count = released.len();
        for tile in released {
            commands.entity(tile).despawn();
        }
        commands
            .entity(held_entity)
            .remove::<(TileTerrainRoot, TileTerrainBody)>();
        if let Ok(mut list) = TILE_RENDERED.lock() {
            list.retain(|b| *b != held.body_id);
        }
        // Leave the vacated body on the canonical surface rather than with no
        // height authority at all: the propagator, colliders and HUD altitude
        // all read through `HeightSourceRegistry`, and a missing entry is a
        // sharper failure than a coarse one.
        rendered_ground.remove(held.body_id);
        if let Some(surface) = surfaces.surface(held.body_id) {
            height_sources.insert(
                held.body_id,
                Arc::new(CpuPipelineHeightSource::new(Arc::new(
                    FlattenedSurface::new(surface, flatten_registry.handle(held.body_id)),
                ))),
            );
        }
        info!(
            target: "thalos::diagnostic::tile_terrain",
            event = "handoff",
            from = ?held.body_id,
            to = ?resolved.body,
            released_tiles = released_count,
            "tile terrain root followed the view anchor to another body"
        );
    }

    let Some(surface) = surfaces.surface(resolved.body) else {
        return;
    };
    let Some((entity, _)) = bodies.iter().find(|(_, rsb)| rsb.body_id == resolved.body) else {
        return;
    };
    // Vertex colors carry the surface's linear albedo; the base material
    // stays neutral. Airless bodies shade through the Hapke regolith branch
    // (tile_terrain.wgsl) so ground reconverges with the impostor's Hapke
    // look; atmosphere-bearing bodies keep stock PBR. Both branches receive
    // the shared `thalos::shadow` cascade via `sync_shadow_receivers`.
    let airless = sim
        .system
        .bodies
        .get(resolved.body)
        .is_none_or(|body| body.terrestrial_atmosphere.is_none());
    let mut params = if airless {
        TileShadingParams::hapke()
    } else {
        TileShadingParams::pbr()
    };
    params.inspect = inspection_mode();
    let radius_m = resolved.radius_m;
    // Every tile shares one material — levels differ only in geometry (the
    // per-level render lift the mesher bakes in, `tiles::LEVEL_RENDER_LIFT_M`).
    let tile_mat = materials.add(tile_material(
        StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.97,
            metallic: 0.0,
            ..default()
        },
        params,
    ));
    // Structure pads level the ground through the body's *shared* flatten
    // handle — the same object the runway and the base editor write and udlod
    // reads, so a pad installed by either route reaches whichever renderer owns
    // the body. Wrapping is a passthrough while the handle is empty.
    let flatten = flatten_registry.handle(resolved.body);
    let ground_surface: Arc<dyn SurfaceQuery> = Arc::new(FlattenedSurface::new(
        Arc::clone(&surface),
        Arc::clone(&flatten),
    ));
    // Memoize the provider. Field evaluation is the floor on a cold surface load
    // (~870 tiles/s is the machine ceiling for the synthesis shape, measured), so
    // the only way past it is not to evaluate the same tile twice — across
    // despawn/respawn churn, across a revisit, and across boots. The legacy udlod
    // ground has had this since it was the default renderer; it did not come
    // across with the handover because the wrappers are typed to udlod's own
    // provider trait. The namespace folds in the live flatten, so a pad installed
    // after spawn re-keys subsequent tiles instead of poisoning the cache.
    let provider = tile_cache.wrap_tile_provider(
        resolved.body,
        Arc::new(SurfaceQueryProvider {
            surface: Arc::clone(&ground_surface),
        }),
        Some(Arc::clone(&flatten)),
        surfaces.fingerprint(resolved.body).unwrap_or_default(),
    );
    let mut root = TileTerrainRoot::new(
        radius_m,
        provider,
        tile_mat,
        // Real-scale ground is ship-view only, exactly like udlod's terrain
        // root and every other real-space mesh (`rendering::spawn`). Without
        // this the tiles land on the default layer 0, which the map camera
        // also renders — drawing the 1:1 landscape over the far-scale map.
        RenderLayers::layer(SHIP_LAYER),
    );
    // Terrain casts into the shared sun-shadow cascade (NTR-X6's in-band half:
    // ridges shadow valleys, hills shadow the plain, terrain shadows land on
    // structures/craft/trees through the same maps everything already samples).
    // Each tile gets a depth-caster child on the cascade layer, drawing the
    // SAME mesh with this bare unlit material — the cascade cameras never pay
    // the tile layer-stack's fragment cost, and the shared mesh handle means
    // no new per-tile GPU resource for the residency budget to count
    // (INC-20260725T012104Z). Beyond the cascade band the far field still has
    // no terrain shadow mechanism — that stays NTR-X6's horizon-term half.
    //
    // BACK faces only (`cull_mode: Front`). The cascade bias model was
    // calibrated for casters that are never their own receivers ("terrain
    // never renders into the maps" — sun_shadow.rs); front-face terrain depth
    // breaks that assumption, and at the outer cascades' multi-metre texels
    // the bias caps cannot cover the per-texel depth error, so the whole
    // gently-rolling plain self-shadowed. The PCSS blocker filter then blurred
    // that acne into a smooth grey veil whose inner edge traced the finer
    // cascade's box — giant straight-edged light/dark regions on open ground
    // (user screenshots 2026-07-26). Storing the terrain's DOWN-SUN faces
    // instead puts the stored depth beyond every sunward receiver — acne is
    // impossible by construction at any texel size — while valley floors
    // behind a ridge still fall deeper than the ridge's leeward face and stay
    // properly shadowed (the classic heightfield-caster fix). The thin band
    // that loses its cast shadow — the leeward face itself — is exactly the
    // ground `n·l` already darkens.
    root.caster = Some((
        std_materials.add(StandardMaterial {
            base_color: Color::WHITE,
            unlit: true,
            cull_mode: Some(bevy::render::render_resource::Face::Front),
            ..default()
        }),
        RenderLayers::layer(super::sun_shadow::SHADOW_CASTER_LAYER),
    ));
    info!(
        target: "thalos::diagnostic::tile_terrain",
        event = "installed",
        body = ?resolved.body,
        radius_m,
        max_level = root.max_level,
        "tile terrain installed"
    );

    // Hand this body's ground authority to the tile renderer. Everything that
    // sits on the terrain — grass / tree / rock scatter and their residency
    // gate, the local-physics collider patch, the camera terrain floor, HUD
    // altitude — reads through these two registries, so re-pointing them is the
    // whole of "terrain detail works on the tile path". The canonical surface
    // stays the fallback for directions no tile covers yet.
    let ground = RenderedGround::Tiles(Arc::clone(&root.height_mirror));
    rendered_ground.insert(resolved.body, ground.clone());
    // The fallback for directions no tile covers yet must be the *flattened*
    // surface too, or the collider / camera floor would step off the pad
    // wherever residency lags the eye.
    height_sources.insert(
        resolved.body,
        Arc::new(RenderedGroundHeightSource::new(ground, ground_surface)),
    );

    commands.entity(entity).insert((
        root,
        TileTerrainBody {
            body_id: resolved.body,
        },
    ));
    if let Ok(mut list) = TILE_RENDERED.lock() {
        list.push(resolved.body);
    }
    // Boot race: residency may have spawned udlod for this body before the
    // anchor resolved. A rebuild request despawns it; the respawn declines
    // via the `tile_rendered` gate, leaving the tile path sole owner.
    rebuild.request(resolved.body);
}

/// Per-frame material-layer params for the NTR-X4 tile shader: the body's
/// world rotation (so the shader can classify slope / build detail normals in
/// the stable body-fixed frame), the radial up at the view anchor, and the
/// inputs to the shader's per-fragment ambient day/night gate (body centre in
/// render space, sun direction, night-floor fraction of the flat ambient). The
/// tile materials are already dirtied every frame by `sync_shadow_receivers`'
/// cascade fan-in, so this adds no new invalidation.
fn update_tile_material_params(
    anchor: Res<ViewAnchor>,
    cache: Res<SolarSystemState>,
    ambient: Res<GlobalAmbientLight>,
    sun_daylight: Res<super::lighting::SunDaylight>,
    roots: Query<(&TileTerrainRoot, &TileTerrainBody, &GlobalTransform)>,
    mut materials: ResMut<Assets<TileTerrainMaterial>>,
) {
    // What share of `GlobalAmbientLight` is the night floor. `update_sun_light`
    // owns the magnitude — it folds the sky/space fill and this floor into one
    // per-camera value using the sun elevation *at the craft*, which is right
    // for the near field and wrong for a globe seen from orbit. The shader
    // re-spreads it per fragment; this and the craft's own gate (below) are the
    // only extra numbers it needs. When the craft is already at night the two
    // are equal, the fraction is 1, and the gate degenerates to identity.
    let night_frac = if ambient.brightness > 0.0 {
        (super::lighting::AMBIENT_NIGHT_BRIGHTNESS / ambient.brightness).clamp(0.0, 1.0)
    } else {
        1.0
    };
    // The gate already baked into that magnitude, so the shader can divide it
    // out instead of stacking a second terminator ramp on top of it. Floored
    // well above zero: as it approaches zero the ambient IS the floor, the
    // fill it would scale is gone, and only the reciprocal's blow-up is left.
    let craft_daylight = sun_daylight.0.clamp(1.0e-3, 1.0);
    // Star is always index 0 (same convention as `update_sun_light`).
    let star_pos = cache
        .states
        .as_ref()
        .and_then(|states| states.first())
        .map(|s| s.position);

    for (root, body, global) in &roots {
        let orient = global.rotation();
        let up = anchor
            .resolved
            .filter(|resolved| resolved.body == body.body_id)
            .map(|resolved| resolved.cam_body.normalize().as_vec3());
        // Body centre in render space — the real-space grid transform, exactly
        // as `ground_terrain` reads it for the sky pass (1 render unit = 1 m).
        let center = global.translation();
        // Sun direction is a pure direction, so the render origin offset drops
        // out and the physics-space delta carries over unrotated.
        let sun_dir = star_pos
            .zip(
                cache
                    .states
                    .as_ref()
                    .and_then(|states| states.get(body.body_id))
                    .map(|s| s.position),
            )
            .map(|(star, body_pos)| (star - body_pos).normalize_or_zero().as_vec3())
            .filter(|d| *d != Vec3::ZERO);

        let Some(mut mat) = materials.get_mut(&root.material) else {
            continue;
        };
        let params = &mut mat.extension.params;
        params.orient = Vec4::new(orient.x, orient.y, orient.z, orient.w);
        if let Some(up) = up {
            params.up_body = up.extend(0.0);
        }
        params.center_ws = center.extend(craft_daylight);
        // Without a resolved sun direction the gate must not darken anything,
        // so leave the identity default in place.
        params.sun_night = match sun_dir {
            Some(dir) => dir.extend(night_frac),
            None => Vec4::new(0.0, 1.0, 0.0, 1.0),
        };
    }
}

/// Samples across a pad's blend ramp the tile mesh must carry.
///
/// The ramp is the narrowest gradient a flatten authors, so it sets the mesh
/// resolution the whole pad needs: resolve it and the levelled interior is exact
/// right up to its edge; miss it and the reconstructed surface cuts the corner
/// from natural terrain straight across the pad — 83 m of it at the spaceport,
/// against paving that stands 0.12 m proud.
///
/// Four samples is two past Nyquist on the ramp. With the runway basin's 500 m
/// `BASIN_RAMP_M` that asks for ≤125 m spacing, i.e. level 10 on Thalos (76 m) —
/// a handful of tiles plus the 2:1 balance cascade, against a budget denominated
/// in thousands.
const PAD_RAMP_SAMPLES: f64 = 4.0;

/// Publish this body's structure pads to the tile selector as refinement floors,
/// so no camera distance can coarsen the ground under a base below the
/// resolution its flat footprint needs.
///
/// Read from the body's shared [`TerrainFlattenRegistry`] handle — the same
/// object the pads are installed through — rather than from the structure
/// registry, so a pad reaches the selector by whichever route installed it
/// (boot runway, base editor) with no second registration to forget.
fn publish_pad_refinement_sites(
    flatten_registry: Res<TerrainFlattenRegistry>,
    mut roots: Query<(&mut TileTerrainRoot, &TileTerrainBody)>,
) {
    for (mut root, body) in &mut roots {
        // Read-only lookup: a body with no pads yet must not be given a handle
        // here — creating one is the installer's job, not the selector's.
        let Some(handle) = flatten_registry.get(body.body_id) else {
            root.set_refinement_sites(Vec::new());
            continue;
        };
        let Ok(regions) = handle.read() else {
            continue;
        };
        let sites = regions
            .iter()
            .map(|region| {
                let pad = &region.flatten;
                // Same reach the flatten's own angular reject uses: the offset
                // rectangle's half-diagonal plus the ramp, as an angle off the
                // pad centre.
                let along = pad.offset_along_m.abs() + pad.half_along_m;
                let across = pad.offset_across_m.abs() + pad.half_across_m;
                let reach = (along * along + across * across).sqrt() + pad.ramp_m;
                RefinementSite {
                    center_dir: pad.center_dir,
                    angular_radius: (reach / pad.radius_m.max(1.0)).atan(),
                    spacing_m: (pad.ramp_m / PAD_RAMP_SAMPLES).max(1.0),
                }
            })
            .collect();
        root.set_refinement_sites(sites);
    }
}

/// Republish the selection eye from `ViewAnchor` each frame.
fn update_tile_eye(
    anchor: Res<ViewAnchor>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    roots: Query<(Entity, &TileTerrainBody), With<TileTerrainRoot>>,
    mut eye: ResMut<TileEye>,
) {
    eye.target = None;
    let Some(resolved) = anchor.resolved else {
        return;
    };
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let Some(state) = states.get(resolved.body) else {
        return;
    };
    // The shared surface frame — the same authority the body grid, the height
    // sources, and the capture framings resolve, so the tile ground can't land
    // in a different frame than the things standing on it (the Mira tile-shell
    // 132° misplacement, INC-20260723T232652Z). Kept in f64 all the way to the
    // streamer: see `tiles::TileBodyOrigin` for what f32 costs here.
    let Some(orientation) =
        super::transforms::surface_orientation_authored(&sim.system.bodies, resolved.body, states)
    else {
        return;
    };
    for (entity, body) in &roots {
        if body.body_id == resolved.body {
            eye.target = Some(TileEyeTarget {
                root: entity,
                cam_body: resolved.cam_body,
                speed_m_s: resolved.speed_m_s,
                body_position: state.position,
                body_orientation: orientation,
            });
            return;
        }
    }
}
