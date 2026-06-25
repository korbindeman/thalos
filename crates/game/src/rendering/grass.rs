//! Grass-blade decoration driver.
//!
//! Maintains a camera-local set of body-fixed grass tiles around the player on
//! the nearest vegetated body, using the tile builder + material in
//! `thalos_body_render::ground::vegetation`. The driver owns the tile
//! lifecycle only — which tiles exist, where they are anchored, what the wind
//! and sun are doing; all placement/geometry logic lives in the engine crate.
//!
//! Anchoring follows the runway pattern exactly (`runway::update_runway_transform`):
//! each tile is a **root-grid big_space child** whose position is recomputed
//! in f64 every frame from the body's state, so the f32 `Transform.rotation`
//! only ever acts on the tile's small (≤ ~20 m) vertex offsets and the grass
//! stays rock-steady under high warp.
//!
//! Tiles are built on `AsyncComputeTaskPool` against the body's
//! [`HeightSource`] (GPU-atlas mirror with CPU fallback — the same source the
//! terrain collider uses), so blades sit on the rendered ground. A periodic
//! revision check rebuilds tiles whose underlying height data shifted (a finer
//! LOD tile streamed in, or a terrain-flatten pad — e.g. the runway — was
//! installed after the grass was built).

use std::collections::HashMap;
use std::sync::Arc;

use bevy::camera::visibility::RenderLayers;
use bevy::light::NotShadowCaster;
use bevy::math::{DVec3, Vec4};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use big_space::prelude::{BigSpace, CellCoord, Grid};

use thalos_body_render::{
    AU_M, GRASS_TILE_SIZE_M, GrassBladeLod, GrassMaterial, GrassTileBuildInput, GrassTileKey,
    GrassTileMesh, LIGHT_AT_1AU, TerrainShadingStyle, build_grass_tile_mesh, grass_tile_frame,
    grass_tile_key, grass_tiles_per_side,
};
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::SimStage;
use crate::coords::SHIP_LAYER;
use crate::rendering::ground_terrain::{TerrainFlattenRegistry, terrain_shading_style_for};
use crate::rendering::real_space::{RealSpaceRoot, real_space_grid};
use crate::rendering::types::CameraExposure;
use crate::solar_system_state::{SimulationState, SolarSystemState, sync_solar_system_state};

// ── Tuning ───────────────────────────────────────────────────────────────────
/// Tiles whose centre is within this ground distance of the camera exist.
const GRASS_RADIUS_M: f64 = 180.0;
/// Hysteresis: tiles are despawned only past this distance.
const GRASS_DESPAWN_RADIUS_M: f64 = 220.0;
/// Dithered fade band (shader-side), metres from the camera.
const GRASS_FADE_START_M: f32 = 140.0;
const GRASS_FADE_END_M: f32 = 180.0;
/// Blade candidate density. Per-blade gates (grass mask, slope, altitude)
/// thin this out, so realized density is lower.
const GRASS_BLADES_PER_M2: f32 = 24.0;
/// Above this altitude over the local terrain no new tiles are built.
const GRASS_MAX_AGL_M: f64 = 400.0;
/// Above this altitude all tiles are despawned (e.g. after launch).
const GRASS_DESPAWN_AGL_M: f64 = 2000.0;
/// Maximum concurrent tile builds. Builds only dispatch for tiles whose terrain
/// is already resident at a fine LOD (cheap GPU-mirror height samples), so this
/// can run wider without hammering the CPU field.
const GRASS_MAX_IN_FLIGHT: usize = 8;
/// Don't build grass until the terrain under a tile is resident at this texel
/// size or finer. The pinned LOD-0 tile is always present but kilometres-coarse;
/// building against it puts blades hundreds of metres off the real surface (the
/// floating-carpet bug). At grass-tile scale (~25 m) this only needs the terrain
/// reasonably detailed; blades sample the finest resident height regardless.
const GRASS_MAX_TERRAIN_TEXEL_M: f32 = 8.0;
/// World seed for blade placement hashes.
const GRASS_SEED: u64 = 0x6772_6173_7321;
/// Wind sway amplitude at the blade tip, metres.
const GRASS_WIND_SWAY_M: f32 = 0.06;
/// Rebuild-staleness scan interval, seconds.
const GRASS_REBUILD_CHECK_S: f32 = 0.5;
/// Rebuild a stale tile only when its centre height moved more than this —
/// caps churn while tiles stream in around the player.
const GRASS_REBUILD_DELTA_M: f32 = 0.05;
/// Stale-tile rebuilds dispatched per scan tick.
const GRASS_MAX_REBUILDS_PER_TICK: usize = 2;

/// One finished tile. `entity: None` means the tile was built and came back
/// empty (water, rock, alpine, flattened pad) — recorded so it isn't rebuilt
/// every frame.
struct BuiltTile {
    entity: Option<Entity>,
    built_revision: u64,
    center_height_m: f32,
}

/// Driver state. **Sole writer:** the systems in this module (drive →
/// finalize → rebuild-check run sequentially via their `ResMut` access).
#[derive(Resource, Default)]
struct GrassTiles {
    body: Option<BodyId>,
    tiles_per_side: i64,
    tiles: HashMap<GrassTileKey, BuiltTile>,
    /// In-flight builds, with the source revision snapshotted at dispatch.
    in_flight: HashMap<GrassTileKey, (Task<Option<GrassTileMesh>>, u64)>,
    /// One shared material for every tile.
    material: Option<Handle<GrassMaterial>>,
    rebuild_timer: f32,
}

/// Marker on a spawned grass-tile entity.
#[derive(Component)]
struct GrassTileVisual {
    body_id: BodyId,
    /// Body-fixed position of the tile centre on the surface.
    center_surface_body: DVec3,
}

pub struct GrassRenderPlugin;

impl Plugin for GrassRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GrassTiles>().add_systems(
            Update,
            (
                check_grass_rebuilds,
                drive_grass_tiles.after(check_grass_rebuilds),
                finalize_grass_tiles.after(drive_grass_tiles),
                update_grass_transforms.after(finalize_grass_tiles),
                update_grass_material,
            )
                .in_set(SimStage::Sync)
                .after(sync_solar_system_state),
        );
    }
}

/// Pick the active grass body and keep the tile set around the player's
/// ground point: dispatch builds for missing near tiles (nearest first),
/// despawn tiles beyond the hysteresis radius.
///
/// Centered on the **canonical player state**, not the camera: the canonical
/// position is at the same epoch as `solar.states`, whereas the camera's
/// big_space cell lags a frame — at ~30 km/s orbital velocity that's a
/// kilometre-scale error against this frame's body position. (The camera
/// trails the player by metres, well inside the grass radius, and the
/// shader's distance fade uses the true camera anyway.)
#[allow(clippy::too_many_arguments)]
fn drive_grass_tiles(
    mut grass: ResMut<GrassTiles>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    mut flatten_registry: ResMut<TerrainFlattenRegistry>,
    mut commands: Commands,
) {
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let cam_pos = sim.simulation.ship_state().position;

    // Active body: the vegetated, terrain-backed body the camera is closest
    // to (by altitude above its reference sphere) — the clouds driver's
    // selection rule, narrowed to bodies that can grow grass.
    let mut best: Option<(BodyId, f64)> = None;
    for (id, body) in sim.system.bodies.iter().enumerate() {
        // Eligibility is now the live HeightSource (GPU-atlas mirror over the
        // runtime ProceduralSurface) — the dead PlanetSurface registry is gone.
        if terrain_shading_style_for(body) != TerrainShadingStyle::Vegetated
            || !height_sources.contains(id)
        {
            continue;
        }
        let Some(state) = states.get(id) else {
            continue;
        };
        let alt = (cam_pos - state.position).length() - body.radius_m;
        if best.is_none_or(|(_, best_alt)| alt < best_alt) {
            best = Some((id, alt));
        }
    }

    let despawn_all = |grass: &mut GrassTiles, commands: &mut Commands| {
        for (_, tile) in grass.tiles.drain() {
            if let Some(entity) = tile.entity {
                commands.entity(entity).despawn();
            }
        }
        grass.in_flight.clear();
    };

    let Some((body_id, _)) = best else {
        if grass.body.is_some() {
            despawn_all(&mut grass, &mut commands);
            grass.body = None;
        }
        return;
    };
    if grass.body != Some(body_id) {
        despawn_all(&mut grass, &mut commands);
        grass.body = Some(body_id);
        let radius_m = sim.system.bodies[body_id].radius_m;
        grass.tiles_per_side = grass_tiles_per_side(radius_m, GRASS_TILE_SIZE_M);
    }

    let body = &sim.system.bodies[body_id];
    let radius_m = body.radius_m;
    let state = &states[body_id];
    let Some(height_source) = height_sources.get(body_id) else {
        return;
    };
    // GPU height mirror for the resident-tile gate in the dispatch loop below.
    let mirror = height_sources.gpu_mirror(body_id);

    // Camera in the body-fixed frame; AGL over the local terrain.
    let cam_body = state.orientation.inverse() * (cam_pos - state.position);
    let cam_r = cam_body.length();
    if cam_r <= 0.0 {
        return;
    }
    let cam_dir = cam_body / cam_r;
    let ground_h = height_source
        .sample_height_m(cam_dir.as_vec3(), GRASS_TILE_SIZE_M as f32)
        .unwrap_or(0.0) as f64;
    let agl = cam_r - (radius_m + ground_h);
    if agl > GRASS_DESPAWN_AGL_M {
        if !grass.tiles.is_empty() || !grass.in_flight.is_empty() {
            despawn_all(&mut grass, &mut commands);
        }
        return;
    }

    let tiles_per_side = grass.tiles_per_side;
    let arc_dist =
        |center_dir: DVec3| -> f64 { center_dir.angle_between(cam_dir) * radius_m };

    // Despawn tiles past the hysteresis radius.
    let stale: Vec<GrassTileKey> = grass
        .tiles
        .keys()
        .filter(|key| {
            grass_tile_frame(**key, tiles_per_side)
                .is_none_or(|(center, _)| arc_dist(center) > GRASS_DESPAWN_RADIUS_M)
        })
        .copied()
        .collect();
    for key in stale {
        if let Some(tile) = grass.tiles.remove(&key)
            && let Some(entity) = tile.entity
        {
            commands.entity(entity).despawn();
        }
    }

    // No new builds while high above the grass shell (existing tiles persist
    // through brief hops; the AGL despawn above handles real ascents).
    if agl > GRASS_MAX_AGL_M {
        return;
    }

    let slots = GRASS_MAX_IN_FLIGHT.saturating_sub(grass.in_flight.len());
    if slots == 0 {
        return;
    }

    // Candidate window around the camera's tile. The lattice shrinks tiles
    // toward cube-face corners (down to ~half size), so the index window is
    // sized for the worst case and each candidate is distance-checked.
    let center_key = grass_tile_key(cam_dir, tiles_per_side);
    let window = (GRASS_RADIUS_M / (GRASS_TILE_SIZE_M * 0.5)).ceil() as i64;
    let mut candidates: Vec<(f64, GrassTileKey)> = Vec::new();
    for dy in -window..=window {
        for dx in -window..=window {
            let key = GrassTileKey {
                face: center_key.face,
                x: center_key.x + dx,
                y: center_key.y + dy,
            };
            if grass.tiles.contains_key(&key) || grass.in_flight.contains_key(&key) {
                continue;
            }
            // Off-face keys return None — a small grass gap at cube-face
            // seams is the accepted v1 trade-off.
            let Some((center, _)) = grass_tile_frame(key, tiles_per_side) else {
                continue;
            };
            let dist = arc_dist(center);
            if dist <= GRASS_RADIUS_M {
                candidates.push((dist, key));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Water is disabled until the generator grows a sea level (terrain rewrite
    // Slice 1), so there is no sea-level cull for now — grass grows wherever the
    // mask / slope / altitude gates allow.
    let sea_level_m = f32::MIN;
    let flatten_exclusion = flatten_registry
        .handle(body_id)
        .read()
        .ok()
        .and_then(|guard| *guard);

    // Only build a grass tile once the terrain tile beneath it is resident in
    // the GPU height mirror. That makes the per-blade height samples cheap
    // (mirror lookups, not the CPU field) and — crucially — places blades on the
    // streamed mesh height, so they're settled at load instead of floating /
    // buried while the terrain is still coarse. Non-resident candidates are left
    // for a later frame (the candidate set is rebuilt every frame).
    let mirror_guard = mirror.as_ref().and_then(|m| m.read().ok());
    let pool = AsyncComputeTaskPool::get();
    let mut dispatched = 0usize;
    for (_, key) in candidates {
        if dispatched >= slots {
            break;
        }
        if let Some(guard) = &mirror_guard {
            let Some((center, _)) = grass_tile_frame(key, tiles_per_side) else {
                continue;
            };
            // Gate on the finest resident *texel size*, not mere presence: the
            // pinned LOD-0 tile is always resident but kilometres-coarse, so
            // building against it floats blades hundreds of metres up. Wait
            // until the terrain here is genuinely detailed.
            match guard.best_resident_texel_m(center.as_vec3()) {
                Some(texel) if texel <= GRASS_MAX_TERRAIN_TEXEL_M => {}
                _ => continue, // terrain not detailed here yet — retry next frame
            }
        }
        let input = GrassTileBuildInput {
            key,
            tiles_per_side,
            height_source: Arc::clone(&height_source),
            radius_m,
            sea_level_m,
            blades_per_m2: GRASS_BLADES_PER_M2,
            // STOPGAP (lighting track): the foliage track's in-flight clipmap-LOD
            // refactor added these fields to `GrassTileBuildInput` but hasn't
            // wired this single dispatch site yet, leaving the workspace
            // non-compiling. These values reproduce the pre-refactor single-LOD
            // grass (full curved blade, no width/height scaling) so the build is
            // green for lighting verification — the foliage track owns replacing
            // this with per-ring LOD dispatch.
            blade_lod: GrassBladeLod::Full,
            width_scale: 1.0,
            height_scale: 1.0,
            seed: GRASS_SEED,
            flatten_exclusion,
        };
        let revision = height_source.revision();
        let task = pool.spawn(async move { build_grass_tile_mesh(&input) });
        grass.in_flight.insert(key, (task, revision));
        dispatched += 1;
    }
}

/// Poll in-flight builds; spawn finished tiles as root-grid big_space
/// children (the runway visual pattern).
fn finalize_grass_tiles(
    mut grass: ResMut<GrassTiles>,
    solar: Res<SolarSystemState>,
    root: Option<Res<RealSpaceRoot>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GrassMaterial>>,
    mut commands: Commands,
) {
    if grass.in_flight.is_empty() {
        return;
    }
    let (Some(states), Some(root), Some(body_id)) =
        (solar.states.as_deref(), root, grass.body)
    else {
        return;
    };
    let Some(body_state) = states.get(body_id) else {
        return;
    };

    let material = grass
        .material
        .get_or_insert_with(|| {
            materials.add(GrassMaterial {
                params: thalos_body_render::GrassParams {
                    time_fade: Vec4::new(0.0, GRASS_FADE_START_M, GRASS_FADE_END_M, 0.0),
                    ..default()
                },
            })
        })
        .clone();

    let mut finished: Vec<(GrassTileKey, u64, Option<GrassTileMesh>)> = Vec::new();
    grass.in_flight.retain(|key, (task, revision)| {
        match block_on(poll_once(task)) {
            Some(result) => {
                finished.push((*key, *revision, result));
                false
            }
            None => true,
        }
    });

    for (key, revision, result) in finished {
        let Some(tile) = result else {
            grass.tiles.insert(
                key,
                BuiltTile {
                    entity: None,
                    built_revision: revision,
                    center_height_m: 0.0,
                },
            );
            continue;
        };

        // First-frame anchor; `update_grass_transforms` re-derives it every
        // frame in f64 (the runway pattern).
        let orientation = body_state.orientation.normalize();
        let center_world = body_state.position + orientation * tile.center_surface_body_m;
        let (cell, local) = real_space_grid().translation_to_grid(center_world);
        let entity = commands
            .spawn((
                Mesh3d(meshes.add(tile.mesh)),
                MeshMaterial3d(material.clone()),
                Transform {
                    translation: local,
                    rotation: orientation.as_quat(),
                    scale: Vec3::ONE,
                },
                cell,
                Visibility::Inherited,
                RenderLayers::layer(SHIP_LAYER),
                NotShadowCaster,
                ChildOf(root.entity),
                GrassTileVisual {
                    body_id,
                    center_surface_body: tile.center_surface_body_m,
                },
                Name::new("Grass Tile"),
            ))
            .id();
        grass.tiles.insert(
            key,
            BuiltTile {
                entity: Some(entity),
                built_revision: tile.built_revision,
                center_height_m: tile.center_height_m,
            },
        );
    }
}

/// Re-anchor every grass tile in f64 each frame — verbatim runway math: the
/// multi-Mm body-fixed offset is rotated in f64 here, and the f32
/// `Transform.rotation` only acts on the small blade vertex offsets.
fn update_grass_transforms(
    solar: Res<SolarSystemState>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut tiles: Query<(&GrassTileVisual, &mut CellCoord, &mut Transform)>,
) {
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let Ok(grid) = root_grid.single() else {
        return;
    };
    for (tile, mut cell, mut transform) in &mut tiles {
        let Some(state) = states.get(tile.body_id) else {
            continue;
        };
        let orientation = state.orientation.normalize();
        let center_world = state.position + orientation * tile.center_surface_body;
        let (next_cell, local) = grid.translation_to_grid(center_world);
        *cell = next_cell;
        transform.translation = local;
        transform.rotation = orientation.as_quat();
    }
}

/// Per-frame shading parameters: sun direction + flux toward the star, the
/// radial up and Rayleigh τ_v feeding the shared `thalos::lighting` sky model
/// (so blades light exactly like the ground), wall-clock sway time, and a
/// slowly veering wind direction tangent to the surface under the camera.
fn update_grass_material(
    grass: Res<GrassTiles>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    time: Res<Time>,
    exposure: Res<CameraExposure>,
    mut materials: ResMut<Assets<GrassMaterial>>,
) {
    let (Some(handle), Some(body_id), Some(states)) =
        (grass.material.as_ref(), grass.body, solar.states.as_deref())
    else {
        return;
    };
    let Some(material) = materials.get_mut(handle) else {
        return;
    };
    let Some(body_state) = states.get(body_id) else {
        return;
    };

    let star_pos = states.first().map(|s| s.position).unwrap_or(DVec3::ZERO);
    let offset = star_pos - body_state.position;
    let sun_dir = offset.normalize_or_zero().as_vec3();
    // Sun flux in the same units the terrain `SceneLighting` carries (lux ×
    // exposure gain), so the shared sky model exposes grass identically.
    let au_over_d = (AU_M / offset.length().max(1.0)) as f32;
    let flux = LIGHT_AT_1AU * au_over_d * au_over_d * exposure.gain;
    material.params.sun_dir = Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, flux);

    // Wind: tangent to the surface under the camera, veering slowly. Render
    // space is inertial-axis-aligned, so the world-space tangent basis comes
    // straight from the camera's world up.
    let t = time.elapsed_secs();
    let up = (sim.simulation.ship_state().position - body_state.position)
        .normalize_or_zero()
        .as_vec3();
    let seed = if up.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let east = seed.cross(up).normalize_or_zero();
    let north = up.cross(east);
    let veer = t * 0.03;
    let wind = (east * veer.cos() + north * veer.sin()).normalize_or_zero();
    material.params.wind = Vec4::new(wind.x, wind.y, wind.z, GRASS_WIND_SWAY_M);
    material.params.time_fade = Vec4::new(t, GRASS_FADE_START_M, GRASS_FADE_END_M, 0.0);

    // Sky hemisphere inputs: the local radial up and the body's authored
    // Rayleigh τ_v + atmosphere strength. τ_v (= β_R · H_R) is the same value
    // the terrain shader recovers from its `AtmosphereBlock`, so grass and
    // ground derive one identical sky environment.
    material.params.sky_up = Vec4::new(up.x, up.y, up.z, 0.0);
    let (tau, strength) = sim
        .system
        .bodies
        .get(body_id)
        .and_then(|b| b.terrestrial_atmosphere.as_ref())
        .and_then(|a| a.scattering.as_ref())
        .map(|s| (Vec3::from_array(s.vertical_optical_depth), s.strength))
        .unwrap_or((Vec3::ZERO, 0.0));
    material.params.sky_tau = Vec4::new(tau.x, tau.y, tau.z, strength);
}

/// Periodically reconcile tiles with the height source: when the source
/// revision advances (a finer atlas tile streamed in, a flatten pad was
/// installed), re-sample each stale tile's centre and rebuild it if the
/// ground actually moved. Rebuild = despawn + forget; `drive_grass_tiles`
/// re-dispatches it on the same frame's pass.
fn check_grass_rebuilds(
    mut grass: ResMut<GrassTiles>,
    height_sources: Res<HeightSourceRegistry>,
    time: Res<Time>,
    mut commands: Commands,
) {
    grass.rebuild_timer += time.delta_secs();
    if grass.rebuild_timer < GRASS_REBUILD_CHECK_S {
        return;
    }
    grass.rebuild_timer = 0.0;

    let Some(body_id) = grass.body else {
        return;
    };
    let Some(source) = height_sources.get(body_id) else {
        return;
    };
    let revision = source.revision();
    let tiles_per_side = grass.tiles_per_side;

    let mut rebuilt = 0usize;
    let mut to_remove: Vec<GrassTileKey> = Vec::new();
    for (key, tile) in grass.tiles.iter_mut() {
        if tile.built_revision == revision {
            continue;
        }
        let Some((center_dir, _)) = grass_tile_frame(*key, tiles_per_side) else {
            continue;
        };
        let Some(h) = source.sample_height_m(center_dir.as_vec3(), 0.5) else {
            continue;
        };
        if tile.entity.is_some() && (h - tile.center_height_m).abs() > GRASS_REBUILD_DELTA_M {
            if rebuilt < GRASS_MAX_REBUILDS_PER_TICK {
                to_remove.push(*key);
                rebuilt += 1;
            }
        } else {
            // Ground under this tile didn't move (or the tile is empty):
            // adopt the new revision without a rebuild.
            tile.built_revision = revision;
            if tile.entity.is_some() {
                tile.center_height_m = h;
            }
        }
    }
    for key in to_remove {
        if let Some(tile) = grass.tiles.remove(&key)
            && let Some(entity) = tile.entity
        {
            commands.entity(entity).despawn();
        }
    }
}
