//! Trajectory-driven UDLOD terrain residency.
//!
//! Each procedural body's ground-LOD terrain entity owns a
//! [`thalos_udlod::prelude::TileAtlas`] that allocates ~192 MB of GPU vRAM.
//! Spawning terrain for every body at startup wastes most of that — the
//! player only sees one body's ground at a time. This module decides
//! which bodies should have terrain spawned ("resident") based on:
//!
//! 1. The canonical SOI body
//!    ([`thalos_physics_canonical::simulation::Simulation::dominant_body`])
//!    — always resident; the player is gravitationally bound to it.
//! 2. Predicted encounters in the flight plan — resident when
//!    `(closest_epoch - sim_time) < preload_lead_time_s`. Lead time
//!    sized so the bake load + initial tile burst finish before the
//!    camera reaches the `4 × radius` impostor-handoff threshold.
//!
//! A body falls out of residency
//! [`TerrainResidencyConfig::despawn_debounce_s`] seconds after the
//! wanted set stops including it. Debouncing prevents thrash when a
//! maneuver edit briefly drops a body or warp jumps the prediction
//! window.
//!
//! Non-resident bodies render as their impostor billboard (already up
//! from [`super::generation::install_baked_planet`]). The visibility-
//! swap system [`super::ground_terrain::sync_body_render_lod`] gates
//! impostor hides on whether terrain is resident, so a non-resident
//! body inside `4 × radius` does not silently disappear.
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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use thalos_body_render::AtmosphereBlock;
use thalos_body_render::udlod::prelude::{TerrainViewComponents, TileTree};
use thalos_body_render::{BodyTerrainMaterial, BodyWaterMaterial};
use thalos_physics_local::{HeightSourceRegistry, TerrainSurfaceRegistry};
use thalos_terrain::PlanetSurface;
use thalos_world::BodyId;

use super::ground_terrain::{spawn_body_terrain, spawn_body_water};
use super::types::{RealSpaceBody, SimulationState, SolarSystemState};
use crate::camera::ShipCamera;
use crate::coords::SHIP_SCALE;
use crate::loading::LoadingProgress;

/// Tunables for the residency planner. Lives in a [`Resource`] so the
/// values can be tweaked at runtime (via BRP, debug UI) without
/// recompiling.
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
}

impl BodyTerrainResidency {
    pub fn is_resident(&self, body: BodyId) -> bool {
        self.entries.contains_key(&body)
    }
}

/// Wanted set produced by [`compute_wanted_residency`] each frame and
/// consumed by [`apply_residency_changes`]. Pulled out as a `Resource`
/// rather than a system-local so the loading-screen gate can read it.
#[derive(Resource, Default)]
struct WantedResidencySet(HashSet<BodyId>);

pub struct TerrainResidencyPlugin;

impl Plugin for TerrainResidencyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainResidencyConfig>()
            .init_resource::<BodyTerrainResidency>()
            .init_resource::<WantedResidencySet>()
            .init_resource::<super::ground_terrain::TerrainFlattenRegistry>()
            .add_systems(
                Update,
                (
                    compute_wanted_residency,
                    apply_residency_changes.after(compute_wanted_residency),
                    initial_residency_loading_gate.after(apply_residency_changes),
                )
                    .in_set(crate::SimStage::Sync)
                    // Run after `poll_planet_install_tasks` so a newly-
                    // completed bake's `Arc<PlanetSurface>` is visible
                    // to the executor on the same frame.
                    .after(super::generation::poll_planet_install_tasks),
            );
    }
}

/// Reads canonical sim state + flight plan and emits the wanted set.
///
/// **Edge case**: when the craft is `BodyFixed` (landed) or in surface
/// contact under Avian, `Simulation::prediction()` returns `None`. Only
/// `dominant_body()` drives residency in that case — which is the
/// default EVA-on-Thalos spawn.
fn compute_wanted_residency(
    sim: Res<SimulationState>,
    config: Res<TerrainResidencyConfig>,
    mut wanted: ResMut<WantedResidencySet>,
) {
    wanted.0.clear();
    let now = sim.simulation.sim_time();

    // (1) Always-resident: the body the craft is gravitationally bound
    // to right now.
    wanted.0.insert(sim.simulation.dominant_body());

    // (2) Predicted encounters + close approaches within the preload
    // lead-time window. `prediction()` is `None` when the craft is
    // landed or in contact — that's fine, we still keep the dominant
    // body resident from step 1.
    if let Some(plan) = sim.simulation.prediction() {
        for enc in plan.encounters() {
            if enc.closest_epoch - now < config.preload_lead_time_s {
                wanted.0.insert(enc.body);
            }
        }
        for ap in plan.approaches() {
            if ap.epoch - now < config.preload_lead_time_s {
                wanted.0.insert(ap.body);
            }
        }
    }
}

/// SystemParam bundle for the spawn / despawn path. Spawning needs the
/// material asset registries, the tile-tree resource, the body's
/// real-space parent entity, the ship camera entity, and the body's
/// cached `Arc<PlanetSurface>` from the terrain registry.
#[derive(SystemParam)]
struct ResidencySpawnParams<'w, 's> {
    body_terrain_materials: ResMut<'w, Assets<BodyTerrainMaterial>>,
    body_water_materials: ResMut<'w, Assets<BodyWaterMaterial>>,
    meshes: ResMut<'w, Assets<Mesh>>,
    tile_trees: ResMut<'w, TerrainViewComponents<TileTree>>,
    surfaces: Res<'w, TerrainSurfaceRegistry>,
    height_sources: Res<'w, HeightSourceRegistry>,
    flatten: ResMut<'w, super::ground_terrain::TerrainFlattenRegistry>,
    solar_system: Res<'w, SolarSystemState>,
    bodies: Query<'w, 's, (&'static RealSpaceBody, Entity)>,
    ship_camera_q: Query<'w, 's, Entity, With<ShipCamera>>,
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

    // Refresh the "last wanted at" stamp for every currently-wanted
    // body, including those already resident. This is the debounce
    // baseline.
    for body_id in &wanted.0 {
        residency.last_wanted_at.insert(*body_id, now);
    }

    // Promote: spawn terrain (+ water) for any wanted body that isn't
    // already resident. Returns `None` and silently retries next frame
    // if the bake hasn't loaded yet or the ship camera isn't up.
    let wanted_to_spawn: Vec<BodyId> = wanted
        .0
        .iter()
        .copied()
        .filter(|b| !residency.entries.contains_key(b))
        .collect();
    for body_id in wanted_to_spawn {
        if let Some(entry) = try_spawn(body_id, &sim, &mut params, &mut commands) {
            residency.entries.insert(body_id, entry);
        }
    }

    // Demote: despawn bodies that have been continuously unwanted for
    // longer than `despawn_debounce_s`. Collect first to avoid mutating
    // the map while iterating.
    let mut to_despawn: Vec<(BodyId, ResidencyEntry)> = Vec::new();
    for (body_id, entry) in &residency.entries {
        if wanted.0.contains(body_id) {
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
    let ship_camera = params.ship_camera_q.single().ok();
    for (body_id, entry) in to_despawn {
        residency.entries.remove(&body_id);
        residency.last_wanted_at.remove(&body_id);
        commands.entity(entry.terrain).despawn();
        if let Some(water) = entry.water {
            commands.entity(water).despawn();
        }
        if let Some(cam) = ship_camera {
            // `TerrainViewComponents` is a per-`(terrain, view)`-pair
            // map; clear the slot so the dropped entity doesn't linger
            // as a stale key. Without this the next spawn would not
            // reuse the entity id, but the map would grow unboundedly
            // over a long session.
            params.tile_trees.remove(&(entry.terrain, cam));
        }
        info!(
            "despawned ground terrain for body_id {} (unwanted for ≥{:.0}s)",
            body_id, config.despawn_debounce_s
        );
    }
}

/// Try to spawn terrain (+ water) for one body. Returns `None` if any
/// prerequisite is unavailable — most commonly the bake hasn't finished
/// async loading yet, in which case the caller should just try again
/// next frame.
fn try_spawn(
    body_id: BodyId,
    sim: &SimulationState,
    params: &mut ResidencySpawnParams,
    commands: &mut Commands,
) -> Option<ResidencyEntry> {
    let body = sim.system.bodies.get(body_id)?;

    // Bodies without authored terrain never get a ground-LOD entity.
    // (Stars, gas giants, the placeholder rocky bodies that haven't
    // been given a `terrain: ...` block.) `dominant_body()` could in
    // principle return one of these, so we have to guard.
    if !body.terrain.is_some() {
        return None;
    }

    // The bake might still be loading on the AsyncCompute pool —
    // `install_baked_planet` writes the `Arc<PlanetSurface>` into this
    // registry when it completes.
    let surface: Arc<PlanetSurface> = params.surfaces.get(body_id)?;

    let ship_camera = params.ship_camera_q.single().ok()?;

    // The real-space body entity (the BigSpace grid origin) is the
    // parent for the terrain entity. `spawn.rs` creates one per
    // procedural body at startup, so this lookup is stable.
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
    let dynamic_state = params.solar_system.dynamic_surface_for(body_id, &surface);
    let height_mirror = params.height_sources.gpu_mirror(body_id);
    let flatten = params.flatten.handle(body_id);

    let terrain = spawn_body_terrain(
        commands,
        body,
        surface.clone(),
        ship_parent_entity,
        &mut params.body_terrain_materials,
        &mut params.tile_trees,
        ship_camera,
        atmosphere,
        dynamic_state,
        height_mirror,
        flatten,
    );

    let water = spawn_body_water(
        commands,
        body,
        &surface,
        ship_parent_entity,
        &mut params.meshes,
        &mut params.body_water_materials,
    );

    Some(ResidencyEntry { terrain, water })
}

/// Loading-screen gate: once every body in the initial wanted set is
/// either resident (terrain entity spawned) or has no authored terrain
/// (e.g., the dominant body is a star), flip
/// [`LoadingProgress::initial_terrain_done`].
///
/// The bake load itself counts via `LoadingProgress.completed`; this
/// adds an additional gate so the `AppState::Loading → Running`
/// transition doesn't fire before the ground terrain under the player's
/// feet exists. Tiles still bake asynchronously inside the first 1–2 s
/// of `Running`, matching the previous startup behavior.
fn initial_residency_loading_gate(
    mut progress: ResMut<LoadingProgress>,
    wanted: Res<WantedResidencySet>,
    residency: Res<BodyTerrainResidency>,
    sim: Res<SimulationState>,
) {
    if progress.initial_terrain_done {
        return;
    }
    // Don't flip until `spawn_bodies` has seeded the bake-task counter —
    // before that, the wanted set might be empty (no SimulationState
    // bodies installed yet) and would trivially pass.
    if !progress.seeded {
        return;
    }
    // Every body the residency planner currently wants must either be
    // (a) resident, or (b) have no authored terrain.
    for body_id in &wanted.0 {
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
    progress.initial_terrain_done = true;
}
