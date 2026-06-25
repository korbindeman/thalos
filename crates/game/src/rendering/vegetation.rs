//! Tree / shrub scatter driver.
//!
//! Maintains a camera-local set of body-fixed **scatter tiles** around the
//! player on the nearest vegetated body, using the placement + scatter system
//! in `thalos_body_render::ground::scatter`. Unlike grass (one batched blade
//! mesh per tile), each scatter tile resolves to *discrete instances* realized
//! as child entities that share a per-species `(Mesh, Material)` — so Bevy
//! auto-batches every instance of a species into instanced draws ("Option A" in
//! `docs/vegetation.md`).
//!
//! Anchoring is the grass / runway pattern: each tile is a **root-grid
//! big_space child** re-posed in f64 every frame, so the f32 `Transform`
//! rotation only acts on the small (≤ ~tile) instance offsets and the trees
//! stay rock-steady under high warp. Per-instance transforms are children of
//! that tile node, in the body-fixed frame.
//!
//! Builds run on `AsyncComputeTaskPool` against the body's [`HeightSource`]
//! (GPU-atlas mirror with CPU fallback), gated on terrain residency so trees
//! seat on the streamed mesh instead of floating. A periodic revision check
//! rebuilds tiles whose ground shifted.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::camera::visibility::RenderLayers;
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use big_space::prelude::{BigSpace, CellCoord, Grid};

use thalos_body_render::{
    TileKey, TileLattice, TreeMeshParams, VegInstance, VegLayer, VegScatterInput, VegScatterTile,
    VegSpeciesPlacement, build_scatter_tile, build_tree_mesh,
};
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::SimStage;
use crate::coords::SHIP_LAYER;
use crate::rendering::ground_terrain::{TerrainFlattenRegistry, terrain_shading_style_for};
use crate::rendering::real_space::{RealSpaceRoot, real_space_grid};
use crate::solar_system_state::{SimulationState, SolarSystemState, sync_solar_system_state};
use thalos_body_render::TerrainShadingStyle;

// ── Tuning ───────────────────────────────────────────────────────────────────
/// Metric side of a scatter tile at a cube-face centre. Coarser than grass —
/// trees are sparse, so a big tile still holds a useful clump and keeps the tile
/// (entity-parent) count bounded.
const TREE_TILE_SIZE_M: f64 = 200.0;
/// Tiles whose centre is within this ground distance of the camera exist.
const TREE_RADIUS_M: f64 = 500.0;
/// Hysteresis: tiles despawn only past this distance.
const TREE_DESPAWN_RADIUS_M: f64 = 620.0;
/// Candidate density per m² before gates (clumping, slope, altitude) thin it.
const TREE_DENSITY_PER_M2: f32 = 0.010;
/// Above this altitude over the local terrain no new tiles are built.
const TREE_MAX_AGL_M: f64 = 350.0;
/// Above this altitude all tiles are despawned (e.g. after launch).
const TREE_DESPAWN_AGL_M: f64 = 2500.0;
/// Maximum concurrent tile builds.
const TREE_MAX_IN_FLIGHT: usize = 4;
/// Don't build trees until the terrain under a tile is resident at this texel
/// size or finer, so roots seat on the streamed mesh instead of the coarse
/// pinned LOD-0 (the floating-tree bug; mirrors the grass residency gate).
const TREE_MAX_TERRAIN_TEXEL_M: f32 = 16.0;
/// World seed for placement hashes (distinct from grass).
const TREE_SEED: u64 = 0x7472_6565_7331;
/// Rebuild-staleness scan interval, seconds.
const TREE_REBUILD_CHECK_S: f32 = 0.75;
/// Rebuild a stale tile only when its centre height moved more than this.
const TREE_REBUILD_DELTA_M: f32 = 0.10;
/// Stale-tile rebuilds dispatched per scan tick.
const TREE_MAX_REBUILDS_PER_TICK: usize = 1;
/// LOD sample hint for the AGL ground probe.
const TREE_GROUND_LOD_M: f32 = 2.0;

/// One species' shared render assets, paired by index with
/// [`SpeciesLibrary::placement`].
struct SpeciesAssets {
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
}

/// The procedural species library, built once at startup. The placement params
/// are also held as an `Arc<[…]>` for the async scatter build (asset-handle
/// free).
#[derive(Resource)]
struct SpeciesLibrary {
    placement: Arc<[VegSpeciesPlacement]>,
    species: Vec<SpeciesAssets>,
}

/// One finished scatter tile. `entity: None` means the tile built empty
/// (clearing, water, rock, alpine) — recorded so it isn't rebuilt every frame.
struct BuiltTile {
    entity: Option<Entity>,
    built_revision: u64,
    center_height_m: f32,
}

/// Driver state. **Sole writer:** the systems in this module (drive → finalize
/// → rebuild run sequentially via their `ResMut` access).
#[derive(Resource, Default)]
struct VegTiles {
    body: Option<BodyId>,
    lattice: TileLattice,
    tiles: HashMap<TileKey, BuiltTile>,
    in_flight: HashMap<TileKey, (Task<Option<VegScatterTile>>, u64)>,
    rebuild_timer: f32,
}

/// Marker on a spawned scatter-tile parent entity.
#[derive(Component)]
struct VegTileVisual {
    body_id: BodyId,
    /// Body-fixed position of the tile centre on the surface.
    center_surface_body: DVec3,
}

pub struct VegetationRenderPlugin;

impl Plugin for VegetationRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VegTiles>()
            .add_systems(Startup, setup_species_library)
            .add_systems(
                Update,
                (
                    check_veg_rebuilds,
                    drive_veg_tiles.after(check_veg_rebuilds),
                    finalize_veg_tiles.after(drive_veg_tiles),
                    update_veg_transforms.after(finalize_veg_tiles),
                )
                    .in_set(SimStage::Sync)
                    .after(sync_solar_system_state),
            );
    }
}

/// Build the procedural species library once at startup: a broadleaf tree mesh
/// + its shared PBR material, plus the placement params handed to the scatter
/// build.
fn setup_species_library(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let trunk_color = Vec3::new(0.16, 0.090, 0.045);
    let canopy_color = Vec3::new(0.055, 0.115, 0.040);

    let mesh = meshes.add(build_tree_mesh(&TreeMeshParams {
        trunk_height_m: 4.8,
        trunk_radius_m: 0.30,
        canopy_radius_m: 2.8,
        canopy_height_m: 2.6,
        trunk_color,
        canopy_color,
        seed: 0xB1_05_50,
        lod: 0,
    }));

    // White base so the mesh's per-vertex trunk/canopy colours come through;
    // matte, two-sided so thin crown silhouettes don't drop out.
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.95,
        metallic: 0.0,
        double_sided: true,
        cull_mode: None,
        ..default()
    });

    let placement: Arc<[VegSpeciesPlacement]> = Arc::from(vec![VegSpeciesPlacement {
        layer: VegLayer::Tree,
        density_per_m2: TREE_DENSITY_PER_M2,
        scale_range: (0.8, 1.5),
        slope_limit: 0.36,
        altitude_band: (1800.0, 2900.0, 2400.0, 3100.0),
        clump_affinity: 0.75,
        min_grass_w: 0.25,
    }]);

    commands.insert_resource(SpeciesLibrary {
        placement,
        species: vec![SpeciesAssets { mesh, material }],
    });
}

/// Pick the active vegetated body and keep the scatter-tile set around the
/// player's ground point. Centred on the canonical player state (same epoch as
/// `solar.states`), like the grass driver.
#[allow(clippy::too_many_arguments)]
fn drive_veg_tiles(
    mut veg: ResMut<VegTiles>,
    library: Option<Res<SpeciesLibrary>>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    mut flatten_registry: ResMut<TerrainFlattenRegistry>,
    mut commands: Commands,
) {
    let Some(library) = library else {
        return;
    };
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let cam_pos = sim.simulation.ship_state().position;

    // Active body: nearest vegetated, terrain-backed body (grass driver's rule).
    let mut best: Option<(BodyId, f64)> = None;
    for (id, body) in sim.system.bodies.iter().enumerate() {
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

    let despawn_all = |veg: &mut VegTiles, commands: &mut Commands| {
        for (_, tile) in veg.tiles.drain() {
            if let Some(entity) = tile.entity {
                commands.entity(entity).despawn();
            }
        }
        veg.in_flight.clear();
    };

    let Some((body_id, _)) = best else {
        if veg.body.is_some() {
            despawn_all(&mut veg, &mut commands);
            veg.body = None;
        }
        return;
    };
    if veg.body != Some(body_id) {
        despawn_all(&mut veg, &mut commands);
        veg.body = Some(body_id);
        let radius_m = sim.system.bodies[body_id].radius_m;
        veg.lattice = TileLattice::for_body(radius_m, TREE_TILE_SIZE_M);
    }

    let body = &sim.system.bodies[body_id];
    let radius_m = body.radius_m;
    let state = &states[body_id];
    let Some(height_source) = height_sources.get(body_id) else {
        return;
    };
    let mirror = height_sources.gpu_mirror(body_id);

    let cam_body = state.orientation.inverse() * (cam_pos - state.position);
    let cam_r = cam_body.length();
    if cam_r <= 0.0 {
        return;
    }
    let cam_dir = cam_body / cam_r;
    let ground_h = height_source
        .sample_height_m(cam_dir.as_vec3(), TREE_GROUND_LOD_M)
        .unwrap_or(0.0) as f64;
    let agl = cam_r - (radius_m + ground_h);
    if agl > TREE_DESPAWN_AGL_M {
        if !veg.tiles.is_empty() || !veg.in_flight.is_empty() {
            despawn_all(&mut veg, &mut commands);
        }
        return;
    }

    let lattice = veg.lattice;
    let arc_dist = |center_dir: DVec3| -> f64 { center_dir.angle_between(cam_dir) * radius_m };

    // Despawn tiles past the hysteresis radius.
    let stale: Vec<TileKey> = veg
        .tiles
        .keys()
        .filter(|key| {
            lattice
                .frame(**key)
                .is_none_or(|(center, _)| arc_dist(center) > TREE_DESPAWN_RADIUS_M)
        })
        .copied()
        .collect();
    for key in stale {
        if let Some(tile) = veg.tiles.remove(&key)
            && let Some(entity) = tile.entity
        {
            commands.entity(entity).despawn();
        }
    }

    if agl > TREE_MAX_AGL_M {
        return;
    }

    let slots = TREE_MAX_IN_FLIGHT.saturating_sub(veg.in_flight.len());
    if slots == 0 {
        return;
    }

    // Candidate window around the camera's tile (nearest-first).
    let center_key = lattice.key_of(cam_dir);
    let window = (TREE_RADIUS_M / (TREE_TILE_SIZE_M * 0.5)).ceil() as i64;
    let mut candidates: Vec<(f64, TileKey)> = Vec::new();
    for dy in -window..=window {
        for dx in -window..=window {
            let key = TileKey {
                face: center_key.face,
                x: center_key.x + dx,
                y: center_key.y + dy,
            };
            if veg.tiles.contains_key(&key) || veg.in_flight.contains_key(&key) {
                continue;
            }
            let Some((center, _)) = lattice.frame(key) else {
                continue;
            };
            if arc_dist(center) <= TREE_RADIUS_M {
                candidates.push((arc_dist(center), key));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Water disabled until the generator grows a sea level (grass driver note).
    let sea_level_m = f32::MIN;
    let flatten_exclusion = flatten_registry
        .handle(body_id)
        .read()
        .ok()
        .and_then(|guard| *guard);

    let mirror_guard = mirror.as_ref().and_then(|m| m.read().ok());
    let pool = AsyncComputeTaskPool::get();
    let mut dispatched = 0usize;
    for (_, key) in candidates {
        if dispatched >= slots {
            break;
        }
        if let Some(guard) = &mirror_guard {
            let Some((center, _)) = lattice.frame(key) else {
                continue;
            };
            match guard.best_resident_texel_m(center.as_vec3()) {
                Some(texel) if texel <= TREE_MAX_TERRAIN_TEXEL_M => {}
                _ => continue, // terrain not detailed here yet — retry next frame
            }
        }
        let input = VegScatterInput {
            key,
            lattice,
            radius_m,
            height_source: Arc::clone(&height_source),
            species: Arc::clone(&library.placement),
            seed: TREE_SEED,
            sea_level_m,
            flatten_exclusion,
        };
        let revision = height_source.revision();
        let task = pool.spawn(async move { build_scatter_tile(&input) });
        veg.in_flight.insert(key, (task, revision));
        dispatched += 1;
    }
}

/// Poll in-flight scatter builds; spawn finished tiles as root-grid big_space
/// children with one child instance entity per placed plant.
fn finalize_veg_tiles(
    mut veg: ResMut<VegTiles>,
    library: Option<Res<SpeciesLibrary>>,
    solar: Res<SolarSystemState>,
    root: Option<Res<RealSpaceRoot>>,
    mut commands: Commands,
) {
    if veg.in_flight.is_empty() {
        return;
    }
    let (Some(library), Some(states), Some(root), Some(body_id)) = (
        library,
        solar.states.as_deref(),
        root,
        veg.body,
    ) else {
        return;
    };
    let Some(body_state) = states.get(body_id) else {
        return;
    };

    let mut finished: Vec<(TileKey, u64, Option<VegScatterTile>)> = Vec::new();
    veg.in_flight.retain(|key, (task, revision)| {
        match block_on(poll_once(task)) {
            Some(result) => {
                finished.push((*key, *revision, result));
                false
            }
            None => true,
        }
    });

    let orientation = body_state.orientation.normalize();
    for (key, revision, result) in finished {
        let Some(tile) = result else {
            veg.tiles.insert(
                key,
                BuiltTile {
                    entity: None,
                    built_revision: revision,
                    center_height_m: 0.0,
                },
            );
            continue;
        };

        let center_world = body_state.position + orientation * tile.center_surface_body_m;
        let (cell, local) = real_space_grid().translation_to_grid(center_world);

        let center = tile.center_surface_body_m;
        let instances = tile.instances;
        let entity = commands
            .spawn((
                Transform {
                    translation: local,
                    rotation: orientation.as_quat(),
                    scale: Vec3::ONE,
                },
                cell,
                Visibility::Inherited,
                ChildOf(root.entity),
                VegTileVisual {
                    body_id,
                    center_surface_body: center,
                },
                Name::new("Vegetation Tile"),
            ))
            .with_children(|parent| {
                for inst in &instances {
                    let Some(assets) = library.species.get(inst.species as usize) else {
                        continue;
                    };
                    parent.spawn((
                        Mesh3d(assets.mesh.clone()),
                        MeshMaterial3d(assets.material.clone()),
                        instance_transform(inst),
                        Visibility::Inherited,
                        RenderLayers::layer(SHIP_LAYER),
                    ));
                }
            })
            .id();

        veg.tiles.insert(
            key,
            BuiltTile {
                entity: Some(entity),
                built_revision: tile.built_revision,
                center_height_m: tile.center_height_m,
            },
        );
    }
}

/// Per-instance local transform in the tile's (body-fixed) frame: orient mesh
/// +Y to the terrain normal, yaw, a small lean, and uniform scale.
fn instance_transform(inst: &VegInstance) -> Transform {
    let up = inst.up_body.normalize_or(Vec3::Y);
    let rotation = Quat::from_rotation_arc(Vec3::Y, up)
        * Quat::from_rotation_y(inst.yaw)
        * Quat::from_rotation_x(inst.tilt);
    Transform {
        translation: inst.root_offset_body_m,
        rotation,
        scale: Vec3::splat(inst.scale),
    }
}

/// Re-anchor every scatter tile in f64 each frame (the grass / runway pattern):
/// the multi-Mm body-fixed offset is rotated in f64 here; the f32 transform only
/// acts on the small per-instance child offsets.
fn update_veg_transforms(
    solar: Res<SolarSystemState>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut tiles: Query<(&VegTileVisual, &mut CellCoord, &mut Transform)>,
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

/// Periodically rebuild tiles whose underlying height shifted (finer atlas tile
/// streamed in, or a flatten pad installed). Rebuild = despawn + forget;
/// `drive_veg_tiles` re-dispatches on a later pass.
fn check_veg_rebuilds(
    mut veg: ResMut<VegTiles>,
    height_sources: Res<HeightSourceRegistry>,
    time: Res<Time>,
    mut commands: Commands,
) {
    veg.rebuild_timer += time.delta_secs();
    if veg.rebuild_timer < TREE_REBUILD_CHECK_S {
        return;
    }
    veg.rebuild_timer = 0.0;

    let Some(body_id) = veg.body else {
        return;
    };
    let Some(source) = height_sources.get(body_id) else {
        return;
    };
    let revision = source.revision();
    let lattice = veg.lattice;

    let mut rebuilt = 0usize;
    let mut to_remove: Vec<TileKey> = Vec::new();
    for (key, tile) in veg.tiles.iter_mut() {
        if tile.built_revision == revision {
            continue;
        }
        let Some((center_dir, _)) = lattice.frame(*key) else {
            continue;
        };
        let Some(h) = source.sample_height_m(center_dir.as_vec3(), TREE_GROUND_LOD_M) else {
            continue;
        };
        if tile.entity.is_some()
            && (h - tile.center_height_m).abs() > TREE_REBUILD_DELTA_M
            && rebuilt < TREE_MAX_REBUILDS_PER_TICK
        {
            to_remove.push(*key);
            rebuilt += 1;
        } else {
            tile.built_revision = revision;
            if tile.entity.is_some() {
                tile.center_height_m = h;
            }
        }
    }
    for key in to_remove {
        if let Some(tile) = veg.tiles.remove(&key)
            && let Some(entity) = tile.entity
        {
            commands.entity(entity).despawn();
        }
    }
}
