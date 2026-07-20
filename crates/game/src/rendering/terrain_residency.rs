//! Screen-size + trajectory-driven UDLOD terrain residency.
//!
//! Each procedural body's ground-LOD terrain entity owns a
//! [`thalos_udlod::prelude::TileAtlas`]. The near-tier atlas is large
//! (~1.8 GB of GPU vRAM by the [`super::ground_terrain`] constants), so
//! spawning a near atlas for every body does not scale. This module decides
//! which bodies should have terrain spawned ("resident") and at which
//! [`TerrainTier`]:
//!
//! 1. The canonical SOI body
//!    ([`thalos_physics_canonical::simulation::Simulation::dominant_body`])
//!    — always resident, `Near`; the player is gravitationally bound to it.
//! 2. Predicted encounters in the flight plan — resident `Near` when
//!    `(closest_epoch - sim_time) < preload_lead_time_s`. Lead time sized so
//!    the initial tile burst finishes before the camera reaches the body.
//! 3. Any other terrain-bearing body whose ship-view projected size exceeds the
//!    icon dot — resident `Distant` (a tiny, cheap atlas, ~20 MB; see
//!    [`RESIDENT_SCREEN_MARGIN`]). This is what lets a body bigger than a dot of
//!    light render as real udlod terrain instead of a flat billboard, while
//!    many such bodies stay resident at once.
//!
//! When a body crosses the near/distant boundary its terrain entity is
//! despawned and respawned at the new tier (re-streaming from cold).
//!
//! A body falls out of residency
//! [`TerrainResidencyConfig::despawn_debounce_s`] seconds after the
//! wanted set stops including it. Debouncing prevents thrash when a
//! maneuver edit briefly drops a body or warp jumps the prediction
//! window.
//!
//! Bodies with no resident terrain entity render via the impostor / icon path;
//! the visibility-swap system [`super::ground_terrain::sync_body_render_lod`]
//! gates the swap on whether terrain is resident, so a non-resident body does
//! not silently disappear.
//!
//! # Dev teleport (future)
//!
//! When a debug teleport tool is added, it must promote the destination
//! body to resident *before* mutating the ship's position. Otherwise the
//! body will render as its low-detail impostor billboard for 1–2 s
//! after the teleport while the bake load + tile burst catch up. The
//! cleanest hook is a small helper next to [`apply_residency_changes`]
//! that calls [`try_spawn`] for the requested body and inserts the
//! result into [`BodyTerrainResidency`]. The helper isn't wired today
//! because there is no teleport tool to consume it.

use std::collections::HashMap;

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use thalos_body_render::AtmosphereBlock;
use thalos_body_render::BodyTerrainMaterial;
use thalos_body_render::udlod::prelude::{TerrainViewComponents, TileTree};
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use super::ground_terrain::{TerrainTier, spawn_body_terrain};
use super::screen_marker_radius;
use super::types::{RealSpaceBody, SimulationState};
use crate::camera::ShipCamera;
use crate::coords::SHIP_SCALE;
use crate::loading::LoadingTracker;

/// Tunables for the residency planner. Lives in a [`Resource`] so the
/// values can be tweaked (edit the defaults and rebuild; a future debug UI
/// could expose them).
#[derive(Resource, Clone, Copy, Debug)]
pub struct TerrainResidencyConfig {
    /// Promote a body to resident when a predicted encounter or close
    /// approach satisfies `closest_epoch - sim_time < preload_lead_time_s`.
    /// Initial 60 s comfortably covers a Thalos-size cold disk load
    /// (~1–2 s) plus the visible-tile bake burst (~1–2 s).
    pub preload_lead_time_s: f64,
    /// A resident body must stay continuously *unwanted* for this many
    /// seconds before it is despawned. Prevents thrash from brief
    /// maneuver edits or warp-jump prediction gaps.
    pub despawn_debounce_s: f64,
}

impl Default for TerrainResidencyConfig {
    fn default() -> Self {
        Self {
            preload_lead_time_s: 60.0,
            despawn_debounce_s: 30.0,
        }
    }
}

/// Currently-resident terrain entities, keyed by body id. Read by the
/// visibility-swap system, written by [`apply_residency_changes`].
#[derive(Resource, Default)]
pub struct BodyTerrainResidency {
    entries: HashMap<BodyId, ResidencyEntry>,
    /// Last `sim_time` at which each body appeared in the wanted set.
    /// Used to debounce despawns.
    last_wanted_at: HashMap<BodyId, f64>,
}

#[derive(Debug, Clone, Copy)]
struct ResidencyEntry {
    terrain: Entity,
    water: Option<Entity>,
    /// Which atlas footprint this entity was spawned at. When the residency
    /// planner re-classifies a body across the near/distant boundary, the
    /// terrain entity is despawned and respawned at the new tier.
    tier: TerrainTier,
}

impl BodyTerrainResidency {
    pub fn is_resident(&self, body: BodyId) -> bool {
        self.entries.contains_key(&body)
    }
}

/// Bodies whose resident terrain must be rebuilt from cold this frame because
/// their flatten set changed *after* tiles had already streamed in (the base
/// editor's flatten-confirm, a deleted pad). Written by gameplay; drained by
/// [`apply_terrain_rebuild_requests`].
///
/// **Why a full despawn/respawn rather than a per-tile re-bake:** UDLOD bakes
/// each resident tile exactly once and has no per-tile invalidation path, so a
/// [`thalos_terrain::TerrainFlatten`] written after a tile is resident stays
/// invisible until that tile is rebuilt. Dropping and respawning the body's
/// terrain entity re-streams every tile through the body's *persistent* flatten
/// handle (which already carries the new region), and the GPU-atlas height
/// mirror + surface-local collider follow via their existing revision chain
/// with no extra work. The event is rare (one confirm per site) and the world is
/// paused under the editor, so the ~1–2 s cold re-stream is acceptable.
#[derive(Resource, Default)]
pub struct TerrainRebuildRequest {
    bodies: std::collections::HashSet<BodyId>,
}

impl TerrainRebuildRequest {
    /// Queue `body_id` for a cold terrain rebuild. Idempotent within a frame.
    pub fn request(&mut self, body_id: BodyId) {
        self.bodies.insert(body_id);
    }
}

/// Wanted set produced by [`compute_wanted_residency`] each frame and
/// consumed by [`apply_residency_changes`]. Pulled out as a `Resource`
/// rather than a system-local so the loading-screen gate can read it.
///
/// The value is the [`TerrainTier`] the body should render at: `Near` for the
/// dominant SOI body and predicted encounters (gameplay-relevant), `Distant`
/// for bodies that are only big enough on screen to deserve real terrain.
#[derive(Resource, Default)]
struct WantedResidencySet(HashMap<BodyId, TerrainTier>);

pub struct TerrainResidencyPlugin;

impl Plugin for TerrainResidencyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainResidencyConfig>()
            .init_resource::<BodyTerrainResidency>()
            .init_resource::<WantedResidencySet>()
            .init_resource::<TerrainRebuildRequest>()
            .init_resource::<super::ground_terrain::TerrainFlattenRegistry>()
            .add_systems(
                Update,
                (
                    compute_wanted_residency,
                    apply_residency_changes.after(compute_wanted_residency),
                    initial_residency_loading_gate.after(apply_residency_changes),
                )
                    .in_set(crate::SimStage::Sync),
            )
            // Ungated (not in `SimStage::Sync`) so a flatten-confirm under the
            // base editor — which pauses the sim — still re-streams the terrain.
            // The UDLOD streaming + height-mirror sync it relies on also run
            // ungated in `Last`.
            .add_systems(Update, apply_terrain_rebuild_requests);
    }
}

/// A body becomes [`TerrainTier::Distant`]-resident once its ship-view rendered
/// radius reaches this fraction of the icon-dot radius
/// ([`super::screen_marker_radius`]). Below 1.0 so the (cheap) distant terrain
/// streams in slightly *before* the body grows past the dot, hiding the cold
/// tile-load latency. 0.5 ≈ a ~4 screen-pixel radius at the ship camera's
/// default ~45° vertical FOV on a 1080p viewport — "more than a dot of light".
const RESIDENT_SCREEN_MARGIN: f32 = 0.5;

/// Reads canonical sim state + flight plan and emits the wanted set, tagging
/// each wanted body with the [`TerrainTier`] it should render at.
///
/// Three rules, in priority order (a `Near` assignment always wins over a later
/// `Distant` one for the same body):
/// 1. The dominant SOI body → `Near` (the player is bound to it; landing/EVA
///    colliders live here).
/// 2. Predicted encounters / close approaches within the preload window →
///    `Near` (the player is heading there).
/// 3. Any *other* body whose ship-view projected size exceeds the icon dot →
///    `Distant` (it is big enough to deserve real terrain instead of a flat
///    billboard, but it is only scenery).
///
/// **Edge case**: when the craft is `BodyFixed` (landed) or in surface contact
/// under Avian, `Simulation::prediction()` returns `None`; only rules 1 and 3
/// drive residency then — which is the default EVA-on-Thalos spawn.
fn compute_wanted_residency(
    sim: Res<SimulationState>,
    config: Res<TerrainResidencyConfig>,
    ship_cam_q: Query<&GlobalTransform, With<ShipCamera>>,
    body_q: Query<(&RealSpaceBody, &GlobalTransform)>,
    mut wanted: ResMut<WantedResidencySet>,
) {
    wanted.0.clear();
    let now = sim.simulation.sim_time();

    // (1) Always-resident at full detail: the body the craft is gravitationally
    // bound to right now.
    wanted
        .0
        .insert(sim.simulation.dominant_body(), TerrainTier::Near);

    // (2) Predicted encounters + close approaches within the preload lead-time
    // window, at full detail. `prediction()` is `None` when the craft is landed
    // or in contact — that's fine, we still keep the dominant body resident.
    if let Some(plan) = sim.simulation.prediction() {
        for enc in plan.encounters() {
            if enc.closest_epoch - now < config.preload_lead_time_s {
                wanted.0.insert(enc.body, TerrainTier::Near);
            }
        }
        for ap in plan.approaches() {
            if ap.epoch - now < config.preload_lead_time_s {
                wanted.0.insert(ap.body, TerrainTier::Near);
            }
        }
    }

    // (3) Screen-size residency: any terrain-bearing body whose ship-view
    // rendered disc is bigger than the icon dot gets cheap distant terrain. The
    // ship view renders at SHIP_SCALE (1 render unit = 1 m), so the body's
    // rendered sphere radius is just `radius_m`, and the icon-dot radius is
    // `screen_marker_radius(body_pos, cam_pos)`. Uses each body's render-space
    // grid origin (`RealSpaceBody` GlobalTransform), matching the camera-to-body
    // distance the LOD-swap system keys off. `entry` only inserts `Distant`
    // where rules 1–2 haven't already claimed the body as `Near`.
    if let Ok(cam_xform) = ship_cam_q.single() {
        let cam_pos = cam_xform.translation();
        for (rsb, xform) in &body_q {
            let Some(body) = sim.system.bodies.get(rsb.body_id) else {
                continue;
            };
            if !body.terrain.is_some() {
                continue;
            }
            let body_pos = xform.translation();
            let rendered_radius = (body.radius_m * SHIP_SCALE) as f32;
            let dot_radius = screen_marker_radius(body_pos, cam_pos);
            if rendered_radius >= dot_radius * RESIDENT_SCREEN_MARGIN {
                wanted.0.entry(rsb.body_id).or_insert(TerrainTier::Distant);
            }
        }
    }
}

/// SystemParam bundle for the spawn / despawn path. Spawning needs the terrain
/// material registry, the tile-tree resource, the GPU-atlas height-mirror
/// registry (for the collider), the flatten registry, the body's real-space
/// parent entity, and the ship camera entity.
#[derive(SystemParam)]
struct ResidencySpawnParams<'w, 's> {
    body_terrain_materials: ResMut<'w, Assets<BodyTerrainMaterial>>,
    tile_trees: ResMut<'w, TerrainViewComponents<TileTree>>,
    height_sources: Res<'w, HeightSourceRegistry>,
    flatten: ResMut<'w, super::ground_terrain::TerrainFlattenRegistry>,
    /// Per-body tile caches. Held in a resource so retained tile payloads survive
    /// the despawn/respawn this module performs on every tier change.
    tile_cache: ResMut<'w, super::tile_cache::TileCacheRegistry>,
    bodies: Query<'w, 's, (&'static RealSpaceBody, Entity)>,
    ship_camera_q: Query<'w, 's, Entity, With<ShipCamera>>,
    sun_shadow: Res<'w, super::sun_shadow::SunShadowImage>,
}

fn apply_residency_changes(
    mut commands: Commands,
    wanted: Res<WantedResidencySet>,
    config: Res<TerrainResidencyConfig>,
    sim: Res<SimulationState>,
    mut residency: ResMut<BodyTerrainResidency>,
    mut params: ResidencySpawnParams,
) {
    let now = sim.simulation.sim_time();
    let ship_camera = params.ship_camera_q.single().ok();

    // Refresh the "last wanted at" stamp for every currently-wanted
    // body, including those already resident. This is the debounce
    // baseline.
    for body_id in wanted.0.keys() {
        residency.last_wanted_at.insert(*body_id, now);
    }

    // Promote new bodies and re-tier any whose classification changed across the
    // near/distant boundary. Collect first so we don't borrow `residency.entries`
    // while spawning. `try_spawn` returns `None` and is retried next frame if the
    // ship camera or the body's real-space entity isn't up yet.
    let wanted_entries: Vec<(BodyId, TerrainTier)> =
        wanted.0.iter().map(|(b, t)| (*b, *t)).collect();
    for (body_id, tier) in wanted_entries {
        match residency.entries.get(&body_id).copied() {
            // Already resident at the right tier — nothing to do.
            Some(entry) if entry.tier == tier => {}
            // Tier changed (e.g. a distant scenery body became a predicted
            // encounter, or the dominant body left for a coarser one): drop the
            // old atlas and rebuild at the new footprint. The new tiles re-stream
            // from cold, which is acceptable for an infrequent re-classification.
            Some(entry) => {
                despawn_entry(&mut commands, &mut params, entry, ship_camera);
                residency.entries.remove(&body_id);
                if let Some(new_entry) = try_spawn(body_id, tier, &sim, &mut params, &mut commands)
                {
                    residency.entries.insert(body_id, new_entry);
                    info!(
                        "re-tiered ground terrain for body_id {} -> {:?} tier",
                        body_id, tier
                    );
                }
            }
            None => {
                if let Some(entry) = try_spawn(body_id, tier, &sim, &mut params, &mut commands) {
                    residency.entries.insert(body_id, entry);
                }
            }
        }
    }

    // Demote: despawn bodies that have been continuously unwanted for
    // longer than `despawn_debounce_s`. Collect first to avoid mutating
    // the map while iterating.
    let mut to_despawn: Vec<(BodyId, ResidencyEntry)> = Vec::new();
    for (body_id, entry) in &residency.entries {
        if wanted.0.contains_key(body_id) {
            continue;
        }
        let last = residency
            .last_wanted_at
            .get(body_id)
            .copied()
            .unwrap_or(now);
        if now - last >= config.despawn_debounce_s {
            to_despawn.push((*body_id, *entry));
        }
    }
    for (body_id, entry) in to_despawn {
        residency.entries.remove(&body_id);
        residency.last_wanted_at.remove(&body_id);
        despawn_entry(&mut commands, &mut params, entry, ship_camera);
        info!(
            "despawned ground terrain for body_id {} (unwanted for ≥{:.0}s)",
            body_id, config.despawn_debounce_s
        );
    }
}

/// Consume [`TerrainRebuildRequest`]: despawn and respawn each requested body's
/// resident terrain at its current tier, so a flatten region installed after the
/// body's tiles streamed in takes effect (see the resource's doc for why a full
/// rebuild is necessary). Mirrors the re-tier branch of [`apply_residency_changes`]
/// but keeps the same tier. Bodies that aren't resident are dropped silently —
/// nothing has streamed yet, so the flatten applies when their tiles first bake.
///
/// **Ungated** so it runs while the base editor pauses the sim.
fn apply_terrain_rebuild_requests(
    mut commands: Commands,
    mut request: ResMut<TerrainRebuildRequest>,
    sim: Res<SimulationState>,
    mut residency: ResMut<BodyTerrainResidency>,
    mut params: ResidencySpawnParams,
) {
    if request.bodies.is_empty() {
        return;
    }
    let ship_camera = params.ship_camera_q.single().ok();
    let bodies: Vec<BodyId> = request.bodies.drain().collect();
    for body_id in bodies {
        let Some(entry) = residency.entries.get(&body_id).copied() else {
            continue;
        };
        despawn_entry(&mut commands, &mut params, entry, ship_camera);
        residency.entries.remove(&body_id);
        if let Some(new_entry) = try_spawn(body_id, entry.tier, &sim, &mut params, &mut commands) {
            residency.entries.insert(body_id, new_entry);
            info!(
                "rebuilt ground terrain for body_id {} (flatten changed)",
                body_id
            );
        }
    }
}

/// Despawn a resident terrain (+ water) entity and clear its tile-tree slot.
///
/// `TerrainViewComponents` is a per-`(terrain, view)`-pair map; clearing the
/// slot stops the dropped entity from lingering as a stale key. Without this the
/// map would grow unboundedly over a long session (each re-tier / despawn /
/// respawn cycle allocates a fresh entity id).
fn despawn_entry(
    commands: &mut Commands,
    params: &mut ResidencySpawnParams,
    entry: ResidencyEntry,
    ship_camera: Option<Entity>,
) {
    commands.entity(entry.terrain).despawn();
    if let Some(water) = entry.water {
        commands.entity(water).despawn();
    }
    if let Some(cam) = ship_camera {
        params.tile_trees.remove(&(entry.terrain, cam));
    }
}

/// Try to spawn terrain (+ water) for one body. Returns `None` if any
/// prerequisite is unavailable — most commonly the bake hasn't finished
/// async loading yet, in which case the caller should just try again
/// next frame.
fn try_spawn(
    body_id: BodyId,
    tier: TerrainTier,
    sim: &SimulationState,
    params: &mut ResidencySpawnParams,
    commands: &mut Commands,
) -> Option<ResidencyEntry> {
    let body = sim.system.bodies.get(body_id)?;

    // Bodies without authored terrain never get a ground-LOD entity.
    // (Stars, gas giants.) `dominant_body()` could in principle return one of
    // these, so we have to guard.
    if !body.terrain.is_some() {
        return None;
    }

    let ship_camera = params.ship_camera_q.single().ok()?;

    // The real-space body entity (the BigSpace grid origin) is the parent for
    // the terrain entity. `spawn.rs` creates one per procedural body at
    // startup, so this lookup is stable.
    let ship_parent_entity = params
        .bodies
        .iter()
        .find(|(rsb, _)| rsb.body_id == body_id)
        .map(|(_, e)| e)?;

    let atmosphere = body
        .terrestrial_atmosphere
        .as_ref()
        .map(|a| AtmosphereBlock::from_terrestrial(a, (1.0 / SHIP_SCALE) as f32))
        .unwrap_or_default();
    // The GPU-atlas height mirror feeds the collider / character controller /
    // HUD altitude — only meaningful on the near tier (the body the player can
    // touch). Distant scenery bodies skip it: nothing queries their height, and
    // their tiny low-res atlas would only waste readback work mirroring it.
    let height_mirror = match tier {
        TerrainTier::Near => params.height_sources.gpu_mirror(body_id),
        // `Distant` scenery and `Map` (orbital-map focus, spawned by its own
        // path and never routed here) carry no collider/HUD height queries.
        TerrainTier::Distant | TerrainTier::Map => None,
    };
    let flatten = params.flatten.handle(body_id);
    let sun_shadow_maps = params.sun_shadow.handles.clone();

    let terrain = spawn_body_terrain(
        commands,
        body,
        ship_parent_entity,
        &mut params.body_terrain_materials,
        &mut params.tile_trees,
        ship_camera,
        atmosphere,
        height_mirror,
        flatten,
        tier,
        sun_shadow_maps,
        &mut params.tile_cache,
    );

    // Ocean is no longer a terrain-parented mesh: it is rendered analytically as
    // a ray-traced sphere inside the body's `BodySky` fullscreen pass (see
    // `body_sky.wgsl`), which is smooth at every scale and reads the seabed depth
    // for shallow/deep colour. No per-body water entity is spawned here.
    Some(ResidencyEntry {
        terrain,
        water: None,
        tier,
    })
}

/// Loading-screen gate: once every body in the initial wanted set is
/// either resident (terrain entity spawned) or has no authored terrain
/// (e.g., the dominant body is a star), complete the tracker's
/// [`crate::loading::step::TERRAIN`] step.
///
/// The bake load itself counts via the tracker's bake-install step; this
/// adds an additional gate so the loading screen doesn't reveal before
/// the ground terrain under the player's feet exists. Tiles still bake
/// asynchronously inside the first 1–2 s after the reveal, matching the
/// previous startup behavior.
fn initial_residency_loading_gate(
    mut tracker: ResMut<LoadingTracker>,
    wanted: Res<WantedResidencySet>,
    residency: Res<BodyTerrainResidency>,
    sim: Res<SimulationState>,
) {
    // No-op once complete, and on runtime re-loads that don't register the
    // step (the start screen's runway pass — the world is already up).
    if !tracker.has_step(crate::loading::step::TERRAIN)
        || tracker.is_step_complete(crate::loading::step::TERRAIN)
    {
        return;
    }
    // Don't flip until every bake install has landed — residency can only
    // spawn terrain for installed bakes, and waiting keeps this gate from
    // passing trivially on frame 0 before the wanted set means anything.
    if !tracker.is_step_complete(crate::loading::step::BODIES) {
        return;
    }
    // Every *gameplay-relevant* (`Near`) body the residency planner wants must
    // be either (a) resident, or (b) without authored terrain. Cosmetic
    // `Distant` scenery bodies are intentionally not gated on — the loading
    // screen waits for the ground under the player's feet, not distant terrain
    // that streams in lazily behind a perfectly good icon dot.
    for (body_id, tier) in &wanted.0 {
        if *tier != TerrainTier::Near {
            continue;
        }
        let resident = residency.is_resident(*body_id);
        let no_terrain = sim
            .system
            .bodies
            .get(*body_id)
            .is_some_and(|b| !b.terrain.is_some());
        if !resident && !no_terrain {
            return;
        }
    }
    tracker.complete(crate::loading::step::TERRAIN);
}
