//! Tree / shrub scatter driver.
//!
//! Maintains a camera-local set of body-fixed **scatter tiles** around the
//! player on the nearest vegetated body, using the placement + scatter system in
//! `thalos_body_render::ground::scatter`. Each tile's trees/shrubs are baked into
//! **one batched mesh per tile** (the same one-mesh-per-tile batching the grass
//! uses) — so there is *no per-tree ECS entity*, and forests scale to dense/far.
//! Per-tree variation (scale, rotation, tint, wind phase) and the seamless
//! per-tree scale-fade survive because each tree's base is baked into the mesh
//! UVs and read by `TreeMaterial`.
//!
//! Anchoring is the grass / runway pattern: each tile is a **root-grid big_space
//! child** re-posed in f64 every frame, so the f32 transform only acts on the
//! tile's small vertex offsets and trees stay rock-steady under high warp.
//!
//! Builds run on `AsyncComputeTaskPool` (scatter placement + mesh combine),
//! gated on terrain residency so plants seat on the streamed mesh. Tiles are
//! built a tile beyond the fade edge (invisible build → no pop-in), re-LOD'd by
//! rebuilding the tile mesh at the new LOD (old kept until the new is ready → no
//! vanish), and rebuilt when the height source revision advances.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::camera::visibility::RenderLayers;
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use big_space::prelude::{BigSpace, CellCoord, Grid};

use thalos_body_render::{
    AU_M, CanopyStyle, LIGHT_AT_1AU, TerrainShadingStyle, TileKey, TileLattice, TreeMaterial,
    TreeMeshData, TreeMeshParams, VegLayer, VegScatterInput, VegSpeciesPlacement, build_scatter_tile,
    build_tree_mesh_data, combine_tree_tile_mesh,
};
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::SimStage;
use crate::coords::SHIP_LAYER;
use crate::rendering::ground_terrain::{TerrainFlattenRegistry, terrain_shading_style_for};
use crate::rendering::real_space::{RealSpaceRoot, real_space_grid};
use crate::rendering::types::{CameraExposure, PlayerShip};
use crate::solar_system_state::{SimulationState, SolarSystemState, sync_solar_system_state};

// ── Tuning ───────────────────────────────────────────────────────────────────
/// Metric side of a scatter tile at a cube-face centre.
const TREE_TILE_SIZE_M: f64 = 200.0;
/// Build tiles out to here — beyond the fade-end by ~a tile, so a tile finishes
/// building while its nearest trees are scaled to ~0 (invisible build, no
/// pop-in); they grow in as the craft approaches.
const TREE_RADIUS_M: f64 = 1400.0;
/// Hysteresis: tiles despawn only past this distance.
const TREE_DESPAWN_RADIUS_M: f64 = 1560.0;
/// Scale-fade band (shader-side, metres, from the craft anchor): trees full
/// inside `start`, grown from zero out to `end`. Seamless — no dither, no pop.
const TREE_FADE_START_M: f32 = 1050.0;
const TREE_FADE_END_M: f32 = 1280.0;
/// Broadleaf candidate density per m² before gates (clumping, slope, altitude).
const TREE_DENSITY_PER_M2: f32 = 0.011;
/// Conifer candidate density per m² (mixed into the same tiles for variety).
const CONIFER_DENSITY_PER_M2: f32 = 0.006;
/// Shrub candidate density per m² (denser, but only realized in the near band).
const SHRUB_DENSITY_PER_M2: f32 = 0.030;
/// Above this altitude over the local terrain no new tiles are built.
const TREE_MAX_AGL_M: f64 = 500.0;
/// Above this altitude all tiles are despawned (e.g. after launch).
const TREE_DESPAWN_AGL_M: f64 = 3000.0;
/// Maximum concurrent tile builds.
const TREE_MAX_IN_FLIGHT: usize = 8;
/// Don't build until the terrain under a tile is resident at this texel size or
/// finer (mirrors the grass residency gate); scaled up for far tiles below.
const TREE_MAX_TERRAIN_TEXEL_M: f32 = 16.0;
/// World seed for placement hashes (distinct from grass).
const TREE_SEED: u64 = 0x7472_6565_7331;
/// Rebuild-staleness scan interval, seconds.
const TREE_REBUILD_CHECK_S: f32 = 0.75;
/// Rebuild a stale tile only when its centre height moved more than this.
const TREE_REBUILD_DELTA_M: f32 = 0.10;
/// Stale-tile rebuilds forgotten per scan tick (drive re-dispatches them).
const TREE_MAX_REBUILDS_PER_TICK: usize = 2;
/// LOD sample hint for the AGL ground probe.
const TREE_GROUND_LOD_M: f32 = 2.0;
/// Mesh-LOD band edges (ground distance, m): LOD0 < [0], LOD1 < [1], else LOD2.
const TREE_LOD_BANDS_M: [f64; 2] = [260.0, 620.0];
/// Canopy wind sway amplitude at full weight, metres.
const TREE_WIND_SWAY_M: f32 = 0.35;
/// Number of mesh LODs per species.
const TREE_LOD_COUNT: usize = 3;

/// Mesh LOD index for a tile at ground distance `d`.
fn lod_for_dist(d: f64) -> usize {
    if d < TREE_LOD_BANDS_M[0] {
        0
    } else if d < TREE_LOD_BANDS_M[1] {
        1
    } else {
        2
    }
}

/// The procedural species library, built once at startup. `placement` is also
/// held as an `Arc<[…]>` for the async build; `lod_data[species][lod]` is the raw
/// CPU mesh combined per tile. All species share one [`TreeMaterial`].
#[derive(Resource)]
struct SpeciesLibrary {
    placement: Arc<[VegSpeciesPlacement]>,
    lod_data: Vec<Vec<Arc<TreeMeshData>>>,
    material: Handle<TreeMaterial>,
}

/// What one async tile build produces: the combined mesh + anchor/staleness meta.
struct VegTileBuild {
    mesh: Mesh,
    center_surface_body_m: DVec3,
    built_revision: u64,
    center_height_m: f32,
}

/// One finished tile. `entity: None` means the tile built empty (clearing,
/// water, rock, alpine) — recorded so it isn't rebuilt every frame.
struct BuiltTile {
    entity: Option<Entity>,
    built_revision: u64,
    center_height_m: f32,
    /// Mesh LOD this tile was baked at, so the driver can re-LOD on approach.
    lod: usize,
}

/// Driver state. **Sole writer:** the systems in this module (run sequentially
/// via their `ResMut` access).
#[derive(Resource, Default)]
struct VegTiles {
    body: Option<BodyId>,
    lattice: TileLattice,
    tiles: HashMap<TileKey, BuiltTile>,
    /// In-flight builds: (task, source revision, target LOD).
    in_flight: HashMap<TileKey, (Task<Option<VegTileBuild>>, u64, usize)>,
    rebuild_timer: f32,
}

/// Marker on a spawned scatter-tile entity (one batched mesh).
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
                    update_tree_material,
                )
                    .in_set(SimStage::Sync)
                    .after(sync_solar_system_state),
            );
    }
}

/// Build the procedural species library once at startup: a broadleaf tree and a
/// low shrub, each with a mesh-LOD chain of raw `TreeMeshData`, plus one shared
/// `TreeMaterial` and the placement params handed to the async build.
fn setup_species_library(mut commands: Commands, mut materials: ResMut<Assets<TreeMaterial>>) {
    let mut lod_data: Vec<Vec<Arc<TreeMeshData>>> = Vec::new();
    let mut placement: Vec<VegSpeciesPlacement> = Vec::new();

    // --- Tree (broadleaf) ---
    let tree = TreeMeshParams {
        trunk_height_m: 5.2,
        trunk_radius_m: 0.32,
        canopy_radius_m: 3.0,
        canopy_height_m: 2.8,
        trunk_color: Vec3::new(0.16, 0.090, 0.045),
        canopy_color: Vec3::new(0.055, 0.115, 0.040),
        style: CanopyStyle::Round,
        seed: 0xB1_05_50,
        lod: 0,
    };
    lod_data.push(build_lod_chain(&tree));
    placement.push(VegSpeciesPlacement {
        layer: VegLayer::Tree,
        density_per_m2: TREE_DENSITY_PER_M2,
        scale_range: (0.8, 1.6),
        slope_limit: 0.40,
        altitude_band: (1800.0, 2900.0, 2400.0, 3100.0),
        // Mild clumping: groves, but trees still spread into the near ground
        // instead of clustering into a single distant band.
        clump_affinity: 0.40,
        min_grass_w: 0.22,
    });

    // --- Conifer (pine) — a second, taller, narrower silhouette for variety ---
    let conifer = TreeMeshParams {
        trunk_height_m: 7.0,
        trunk_radius_m: 0.26,
        canopy_radius_m: 1.9,
        canopy_height_m: 3.4,
        trunk_color: Vec3::new(0.13, 0.080, 0.045),
        canopy_color: Vec3::new(0.040, 0.090, 0.045),
        style: CanopyStyle::Conifer,
        seed: 0xC0_1F_E5,
        lod: 0,
    };
    lod_data.push(build_lod_chain(&conifer));
    placement.push(VegSpeciesPlacement {
        layer: VegLayer::Tree,
        density_per_m2: CONIFER_DENSITY_PER_M2,
        scale_range: (0.85, 1.7),
        slope_limit: 0.45,
        altitude_band: (1900.0, 3000.0, 2600.0, 3300.0),
        clump_affinity: 0.55,
        min_grass_w: 0.20,
    });

    // --- Shrub (low bush) ---
    let shrub = TreeMeshParams {
        trunk_height_m: 0.35,
        trunk_radius_m: 0.06,
        canopy_radius_m: 0.78,
        canopy_height_m: 0.62,
        trunk_color: Vec3::new(0.13, 0.085, 0.050),
        canopy_color: Vec3::new(0.062, 0.110, 0.044),
        style: CanopyStyle::Round,
        seed: 0x5_417,
        lod: 0,
    };
    lod_data.push(build_lod_chain(&shrub));
    placement.push(VegSpeciesPlacement {
        layer: VegLayer::Shrub,
        density_per_m2: SHRUB_DENSITY_PER_M2,
        scale_range: (0.6, 1.3),
        slope_limit: 0.46,
        altitude_band: (1600.0, 2700.0, 2300.0, 3000.0),
        clump_affinity: 0.45,
        min_grass_w: 0.28,
    });

    let material = materials.add(TreeMaterial::default());

    commands.insert_resource(SpeciesLibrary {
        placement: Arc::from(placement),
        lod_data,
        material,
    });
}

/// Build the LOD0..LOD2 raw-mesh chain for a species template.
fn build_lod_chain(base: &TreeMeshParams) -> Vec<Arc<TreeMeshData>> {
    (0..TREE_LOD_COUNT as u32)
        .map(|lod| Arc::new(build_tree_mesh_data(&TreeMeshParams { lod, ..*base })))
        .collect()
}

/// Per-species mesh data for a given tile LOD, with shrubs skipped outside the
/// nearest band (too small to read far; bounds their geometry).
fn species_lod_for(library: &SpeciesLibrary, lod: usize) -> Vec<Option<Arc<TreeMeshData>>> {
    library
        .placement
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            if p.layer == VegLayer::Shrub && lod > 0 {
                None
            } else {
                let chain = &library.lod_data[idx];
                chain.get(lod.min(chain.len().saturating_sub(1))).cloned()
            }
        })
        .collect()
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

    // Candidate window around the camera's tile (nearest-first). A tile is a
    // candidate when it's missing OR its baked LOD no longer matches its distance
    // (re-LOD) — the rebuild keeps the old mesh until the new is ready.
    let center_key = lattice.key_of(cam_dir);
    let window = (TREE_RADIUS_M / (TREE_TILE_SIZE_M * 0.5)).ceil() as i64;
    let mut candidates: Vec<(f64, TileKey, usize)> = Vec::new();
    for dy in -window..=window {
        for dx in -window..=window {
            let key = TileKey {
                face: center_key.face,
                x: center_key.x + dx,
                y: center_key.y + dy,
            };
            if veg.in_flight.contains_key(&key) {
                continue;
            }
            let Some((center, _)) = lattice.frame(key) else {
                continue;
            };
            let d = arc_dist(center);
            if d > TREE_RADIUS_M {
                continue;
            }
            let desired = lod_for_dist(d);
            match veg.tiles.get(&key) {
                Some(tile) if tile.lod == desired => continue, // up to date
                _ => candidates.push((d, key, desired)),
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
    for (_, key, desired) in candidates {
        if dispatched >= slots {
            break;
        }
        // Far rings tolerate coarser terrain (their trees are scaled small).
        let texel_limit = ((TREE_TILE_SIZE_M * 0.5) as f32).max(TREE_MAX_TERRAIN_TEXEL_M);
        if let Some(guard) = &mirror_guard {
            let Some((center, _)) = lattice.frame(key) else {
                continue;
            };
            match guard.best_resident_texel_m(center.as_vec3()) {
                Some(texel) if texel <= texel_limit => {}
                _ => continue,
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
        let species_lod = species_lod_for(&library, desired);
        let revision = height_source.revision();
        let task = pool.spawn(async move {
            let tile = build_scatter_tile(&input)?;
            let mesh = combine_tree_tile_mesh(&tile.instances, &species_lod)?;
            Some(VegTileBuild {
                mesh,
                center_surface_body_m: tile.center_surface_body_m,
                built_revision: tile.built_revision,
                center_height_m: tile.center_height_m,
            })
        });
        veg.in_flight.insert(key, (task, revision, desired));
        dispatched += 1;
    }
}

/// Poll in-flight builds; spawn each finished tile's batched mesh as a root-grid
/// big_space child. A rebuild/re-LOD spawns the new entity, then despawns the old
/// one (it stays visible until the new mesh is ready — no vanish).
fn finalize_veg_tiles(
    mut veg: ResMut<VegTiles>,
    solar: Res<SolarSystemState>,
    root: Option<Res<RealSpaceRoot>>,
    library: Option<Res<SpeciesLibrary>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut commands: Commands,
) {
    if veg.in_flight.is_empty() {
        return;
    }
    let (Some(library), Some(states), Some(root), Some(body_id)) =
        (library, solar.states.as_deref(), root, veg.body)
    else {
        return;
    };
    let Some(body_state) = states.get(body_id) else {
        return;
    };

    let mut finished: Vec<(TileKey, u64, usize, Option<VegTileBuild>)> = Vec::new();
    veg.in_flight
        .retain(|key, (task, revision, lod)| match block_on(poll_once(task)) {
            Some(result) => {
                finished.push((*key, *revision, *lod, result));
                false
            }
            None => true,
        });

    let orientation = body_state.orientation.normalize();
    for (key, revision, lod, result) in finished {
        let old_entity = veg.tiles.get(&key).and_then(|t| t.entity);

        let Some(build) = result else {
            if let Some(old) = old_entity {
                commands.entity(old).despawn();
            }
            veg.tiles.insert(
                key,
                BuiltTile {
                    entity: None,
                    built_revision: revision,
                    center_height_m: 0.0,
                    lod,
                },
            );
            continue;
        };

        let center = build.center_surface_body_m;
        let center_world = body_state.position + orientation * center;
        let (cell, local) = real_space_grid().translation_to_grid(center_world);
        let entity = commands
            .spawn((
                Mesh3d(meshes.add(build.mesh)),
                MeshMaterial3d(library.material.clone()),
                Transform {
                    translation: local,
                    rotation: orientation.as_quat(),
                    scale: Vec3::ONE,
                },
                cell,
                Visibility::Inherited,
                RenderLayers::layer(SHIP_LAYER),
                ChildOf(root.entity),
                VegTileVisual {
                    body_id,
                    center_surface_body: center,
                },
                Name::new("Vegetation Tile"),
            ))
            .id();
        // Replace the old (previous-LOD) entity only now that the new one exists.
        if let Some(old) = old_entity {
            commands.entity(old).despawn();
        }
        veg.tiles.insert(
            key,
            BuiltTile {
                entity: Some(entity),
                built_revision: build.built_revision,
                center_height_m: build.center_height_m,
                lod,
            },
        );
    }
}

/// Re-anchor every scatter tile in f64 each frame (the grass / runway pattern).
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

/// Per-frame tree/shrub shading on the single shared [`TreeMaterial`]: sun
/// direction + flux, a slowly veering wind (drives the canopy sway), the shared
/// `thalos::lighting` sky inputs, and the scale-fade band + craft anchor (so the
/// fade is zoom-independent). Mirrors `rendering::grass::update_grass_material`.
#[allow(clippy::too_many_arguments)]
fn update_tree_material(
    library: Option<Res<SpeciesLibrary>>,
    veg: Res<VegTiles>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    time: Res<Time>,
    exposure: Res<CameraExposure>,
    ship: Query<&GlobalTransform, With<PlayerShip>>,
    mut materials: ResMut<Assets<TreeMaterial>>,
) {
    let Some(library) = library else {
        return;
    };
    let (Some(body_id), Some(states)) = (veg.body, solar.states.as_deref()) else {
        return;
    };
    let Some(material) = materials.get_mut(&library.material) else {
        return;
    };
    let Some(body_state) = states.get(body_id) else {
        return;
    };

    let star_pos = states.first().map(|s| s.position).unwrap_or(DVec3::ZERO);
    let offset = star_pos - body_state.position;
    let sun_dir = offset.normalize_or_zero().as_vec3();
    let au_over_d = (AU_M / offset.length().max(1.0)) as f32;
    let flux = LIGHT_AT_1AU * au_over_d * au_over_d * exposure.gain;
    material.params.sun_dir = Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, flux);

    let t = time.elapsed_secs();
    let up = (sim.simulation.ship_state().position - body_state.position)
        .normalize_or_zero()
        .as_vec3();
    let seed = if up.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let east = seed.cross(up).normalize_or_zero();
    let north = up.cross(east);
    let veer = t * 0.025;
    let wind_dir = (east * veer.cos() + north * veer.sin()).normalize_or_zero();
    material.params.wind = Vec4::new(wind_dir.x, wind_dir.y, wind_dir.z, TREE_WIND_SWAY_M);
    // time_fade: x = time, y = fade start, z = fade end (the scale-fade band).
    material.params.time_fade = Vec4::new(t, TREE_FADE_START_M, TREE_FADE_END_M, 0.0);
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

    // Fade reference = the player craft's render-space position, so camera
    // zoom/orbit doesn't change what's drawn (EVA → camera fallback).
    material.params.anchor = match ship.iter().next() {
        Some(gt) => {
            let p = gt.translation();
            Vec4::new(p.x, p.y, p.z, 1.0)
        }
        None => Vec4::ZERO,
    };
}

/// Periodically rebuild tiles whose underlying height shifted (finer atlas tile
/// streamed in, or a flatten pad installed): forget the stale tile so
/// `drive_veg_tiles` re-dispatches it (the old mesh stays until the new is ready).
fn check_veg_rebuilds(
    mut veg: ResMut<VegTiles>,
    height_sources: Res<HeightSourceRegistry>,
    time: Res<Time>,
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
            // Force a re-dispatch by invalidating the recorded LOD; drive sees a
            // mismatch and rebuilds (old mesh kept until the new is ready).
            tile.lod = usize::MAX;
            rebuilt += 1;
        } else {
            tile.built_revision = revision;
            if tile.entity.is_some() {
                tile.center_height_m = h;
            }
        }
    }
}
