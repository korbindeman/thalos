//! Scattered pebble / rock decoration driver — a single near-only ring.
//!
//! Maintains a camera-local set of body-fixed **rock tiles** around the player
//! on the nearest vegetated body, using the shared placement + scatter system in
//! `thalos_body_render::ground::scatter` (the same Poisson-disk grid, residency
//! gate, and one-batched-mesh-per-tile combine the trees use). Each tile's
//! stones are baked into one [`RockMaterial`] mesh.
//!
//! Unlike the trees, rocks **resolve only up close**: there is no impostor band
//! and no clipmap — just one fine ring out to a short reach. And unlike grass,
//! rocks are placed **inversely** to grass density (`VegLayer::Rock`): they
//! gather on bare / rocky patches and thin to a floor density under thick grass,
//! which visually covers the smaller stones anyway. So you mostly see them on
//! the rockier ground between the grass tufts, exactly where real pebbles show.
//!
//! Anchoring + lifecycle follow the grass driver exactly: each tile is a
//! root-grid big_space child re-posed in f64 every frame (rock-steady under
//! warp), built off-thread against the body's [`HeightSource`] once the terrain
//! under it is resident, with a periodic revision check for flatten pads.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::camera::primitives::MeshAabb;
use bevy::camera::visibility::RenderLayers;
use bevy::light::NotShadowCaster;
use bevy::math::{DVec3, Vec3, Vec4};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use big_space::prelude::{BigSpace, CellCoord, Grid};

use thalos_body_render::{
    AU_M, GrassParams, LIGHT_AT_1AU, RockMaterial, RockMeshData, RockMeshParams, TerrainShadingStyle,
    TileKey, TileLattice, VegLayer, VegScatterInput, VegSpeciesPlacement, build_rock_mesh_data,
    build_scatter_tile, combine_rock_tile_mesh, fallback_shadow_map,
};
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::SimStage;
use crate::coords::SHIP_LAYER;
use crate::rendering::ground_terrain::{TerrainFlattenRegistry, terrain_shading_style_for};
use crate::rendering::real_space::{RealSpaceRoot, real_space_grid};
use crate::rendering::sun_shadow::SunShadowState;
use crate::rendering::types::CameraExposure;
use crate::rendering::view_anchor::ViewAnchor;
use crate::solar_system_state::{SimulationState, SolarSystemState, sync_solar_system_state};

// ── Tuning ───────────────────────────────────────────────────────────────────
/// Metric tile side at a cube-face centre.
const ROCK_TILE_SIZE_M: f64 = 32.0;
/// Ground distance (m) the rock ring reaches. Short — pebbles read only up
/// close, and beyond this the stones are sub-pixel, so there's no point paying
/// for them (no impostor / clipmap, unlike the trees).
const ROCK_REACH_M: f64 = 100.0;
/// Fade band half-width (m) for the near-ring scale-grow at the reach edge.
const ROCK_FADE_BAND_M: f32 = 12.0;
/// Above this AGL no new tiles are built (a little above the reach so stones are
/// already present when descending below it). Existing tiles persist.
const ROCK_BUILD_MAX_AGL_M: f64 = 130.0;
/// Above this AGL all live tiles are despawned (past the reach fade, so it's
/// invisible — every stone is faded to nothing by then).
const ROCK_DESPAWN_AGL_M: f64 = 260.0;
/// Maximum concurrent tile builds.
const ROCK_MAX_IN_FLIGHT: usize = 6;
/// Don't build until the terrain under a tile is resident at this texel size or
/// finer (the floating-rock gate, mirroring grass).
const ROCK_MAX_TERRAIN_TEXEL_M: f32 = 8.0;
/// World seed for placement hashes (distinct from grass / trees).
const ROCK_SEED: u64 = 0x726F_636B_7331;
/// Rebuild-staleness scan interval, seconds.
const ROCK_REBUILD_CHECK_S: f32 = 0.75;
/// Rebuild a stale tile only when its centre height moved more than this (clears
/// the GPU-mirror quantization noise floor; catches flatten pads — see grass).
const ROCK_REBUILD_DELTA_M: f32 = 0.5;
/// Stale-tile rebuilds dispatched per scan tick.
const ROCK_MAX_REBUILDS_PER_TICK: usize = 2;
/// LOD sample hint for the AGL ground probe.
const ROCK_GROUND_LOD_M: f32 = 2.0;

/// Near/far/band fade for the single ring: full to the camera (large-negative
/// near edge), fading out across the reach.
fn rock_fade() -> (f32, f32, f32) {
    (-1.0e9, ROCK_REACH_M as f32, ROCK_FADE_BAND_M)
}

/// What one async tile build produces.
struct RockTileBuild {
    mesh: Mesh,
    center_surface_body_m: DVec3,
    built_revision: u64,
    center_height_m: f32,
}

/// One finished tile. `entity: None` means the tile built empty (no stones
/// passed the gates — pure grassland, water, pad) — recorded so it isn't rebuilt
/// every frame.
struct BuiltTile {
    entity: Option<Entity>,
    built_revision: u64,
    center_height_m: f32,
}

/// The procedural rock species library, built once at startup. `placement` is
/// also held as an `Arc<[…]>` for the async build; `meshes[species]` is the raw
/// CPU mesh combined per tile. All species share one [`RockMaterial`].
#[derive(Resource)]
struct RockLibrary {
    placement: Arc<[VegSpeciesPlacement]>,
    meshes: Vec<Option<Arc<RockMeshData>>>,
    material: Handle<RockMaterial>,
}

/// Driver state. **Sole writer:** the systems in this module (run sequentially
/// via their `ResMut` access).
#[derive(Resource, Default)]
struct RockTiles {
    body: Option<BodyId>,
    lattice: Option<TileLattice>,
    tiles: HashMap<TileKey, BuiltTile>,
    in_flight: HashMap<TileKey, (Task<Option<RockTileBuild>>, u64)>,
    rebuild_timer: f32,
}

/// Marker on a spawned rock-tile entity (one batched mesh).
#[derive(Component)]
struct RockTileVisual {
    body_id: BodyId,
    /// Body-fixed position of the tile centre on the surface.
    center_surface_body: DVec3,
}

pub struct RockScatterPlugin;

impl Plugin for RockScatterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RockTiles>()
            .add_systems(Startup, setup_rock_library)
            .add_systems(
                Update,
                (
                    check_rock_rebuilds,
                    drive_rock_tiles.after(check_rock_rebuilds),
                    finalize_rock_tiles.after(drive_rock_tiles),
                    update_rock_transforms.after(finalize_rock_tiles),
                    update_rock_material,
                )
                    .in_set(SimStage::Sync)
                    .after(sync_solar_system_state),
            );
    }
}

/// Build the procedural rock species library: a handful of distinct pebble /
/// stone shapes (so a scatter reads varied), each a `RockMeshData`, plus one
/// shared [`RockMaterial`]. They all share one Poisson grid (the `Rock` layer),
/// so no two stones interpenetrate regardless of species.
fn setup_rock_library(
    mut commands: Commands,
    mut materials: ResMut<Assets<RockMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    // Shared placement params for the rock layer. All members share the widest
    // `min_spacing_m`, so the spacing below sets the layer's blue-noise grid.
    let placement_common = VegSpeciesPlacement {
        layer: VegLayer::Rock,
        min_spacing_m: ROCK_SPACING_M,
        mix_weight: 1.0,
        // Tighter than the trees: keeps the per-instance size spread modest so the
        // species' own radii dominate the look (no giant outliers from the small
        // species, no tiny ones from the boulder).
        scale_range: (0.65, 1.2),
        // Stones tolerate much steeper ground than plants (scree on slopes).
        slope_limit: 0.85,
        // Rocks everywhere up to the alpine zone — effectively no altitude fade
        // (scree persists high where plants give out).
        altitude_band: (0.0, 9000.0, 8000.0, 12000.0),
        // Loose scree clustering (see `rock_scatter_field`).
        clump_affinity: 0.55,
        // No grass minimum — rocks *want* the bare ground (inverse-grass accept).
        min_grass_w: 0.0,
    };

    // Species weighted heavily toward **small stylized pebbles** (a grass field is
    // mostly little stones); larger cobbles and the odd boulder are rare. Each is
    // a plane-cut faceted stone (see `rock_mesh`); they differ only in size, tone,
    // tessellation, and cut count. Colours are light, warm-to-cool natural stone
    // (limestone / granite), several ×brighter than soil so they read as rock on
    // grass. `radius_m` is the *mean*; `scale_range` spreads it (effective
    // half-widths ≈ 4–7 cm tiny, up to ~0.5 m for the rare boulder). Tiny stones
    // stay at the cheap LOD (most placements), big ones get more facets.
    let species_params = [
        // Tiny chips — the dominant population (cheap LOD, fewer cuts).
        (
            RockMeshParams {
                radius_m: 0.055,
                axes: Vec3::new(1.0, 0.60, 0.84),
                color: Vec3::new(0.50, 0.47, 0.42), // light warm limestone
                seed: 0x01,
                subdivisions: 1,
                cuts: 8,
                ..RockMeshParams::default()
            },
            3.6_f32, // mix weight
        ),
        // Small pebbles.
        (
            RockMeshParams {
                radius_m: 0.090,
                axes: Vec3::new(1.0, 0.58, 0.82),
                color: Vec3::new(0.52, 0.46, 0.37), // warm tan
                seed: 0x02,
                subdivisions: 2,
                ..RockMeshParams::default()
            },
            2.4,
        ),
        // Medium stones.
        (
            RockMeshParams {
                radius_m: 0.150,
                axes: Vec3::new(1.0, 0.60, 0.86),
                color: Vec3::new(0.44, 0.45, 0.47), // cool grey granite
                seed: 0x03,
                subdivisions: 2,
                ..RockMeshParams::default()
            },
            0.85,
        ),
        // Occasional angular cobble.
        (
            RockMeshParams {
                radius_m: 0.240,
                axes: Vec3::new(1.0, 0.70, 0.90),
                color: Vec3::new(0.48, 0.45, 0.41), // pale grey
                seed: 0x04,
                subdivisions: 2,
                cuts: 16,
                ..RockMeshParams::default()
            },
            0.30,
        ),
        // Rare boulder (more facets — it fills the view up close).
        (
            RockMeshParams {
                radius_m: 0.380,
                axes: Vec3::new(1.0, 0.72, 0.92),
                color: Vec3::new(0.46, 0.44, 0.42), // neutral grey
                seed: 0x05,
                subdivisions: 3,
                cuts: 18,
                ..RockMeshParams::default()
            },
            0.10,
        ),
    ];

    let mut placement: Vec<VegSpeciesPlacement> = Vec::new();
    let mut meshes: Vec<Option<Arc<RockMeshData>>> = Vec::new();
    for (params, weight) in species_params {
        placement.push(VegSpeciesPlacement {
            mix_weight: weight,
            ..placement_common
        });
        meshes.push(Some(Arc::new(build_rock_mesh_data(&params))));
    }

    let (near, far, band) = rock_fade();
    let material = materials.add(RockMaterial {
        params: GrassParams {
            time_fade: Vec4::new(0.0, near, far, band),
            ..default()
        },
        // Valid depth textures from the start; `update_rock_material` swaps in the
        // live per-cascade maps each frame (`shadow.gate.x` gates sampling).
        sun_shadow_map_0: images.add(fallback_shadow_map()),
        sun_shadow_map_1: images.add(fallback_shadow_map()),
        sun_shadow_map_2: images.add(fallback_shadow_map()),
        ..default()
    });

    commands.insert_resource(RockLibrary {
        placement: Arc::from(placement),
        meshes,
        material,
    });
}

/// Minimum stone spacing (m) — the rock layer's blue-noise grid. Below this two
/// stones never sit closer; density falls out of it (gates only thin further).
/// Tighter than before (1.6) so the now-smaller pebbles read as a scattering of
/// little stones rather than a sparse handful of boulders, while the tri budget
/// stays bounded (most stones are the cheap small/faceted LOD).
const ROCK_SPACING_M: f32 = 1.25;

/// Keep the near rock-tile set around the **view anchor** (the render camera,
/// resolved body-fixed — see [`crate::rendering::view_anchor`]), like grass.
#[allow(clippy::too_many_arguments)]
fn drive_rock_tiles(
    mut rocks: ResMut<RockTiles>,
    library: Option<Res<RockLibrary>>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    rendered_ground: Res<crate::terrain_registry::RenderedGroundRegistry>,
    mut flatten_registry: ResMut<TerrainFlattenRegistry>,
    anchor: Res<ViewAnchor>,
    mut commands: Commands,
) {
    let Some(library) = library else {
        return;
    };
    if solar.states.is_none() {
        return;
    }

    let despawn_all = |rocks: &mut RockTiles, commands: &mut Commands| {
        for (_, tile) in rocks.tiles.drain() {
            if let Some(entity) = tile.entity {
                commands.entity(entity).despawn();
            }
        }
        rocks.in_flight.clear();
    };

    // Leak-bisection kill switch (`THALOS_NO_SCATTER=1`): park the rock scatter
    // entirely (see `mem_diag::scatter_killed`).
    if crate::mem_diag::scatter_killed() {
        if rocks.body.is_some() {
            despawn_all(&mut rocks, &mut commands);
            rocks.body = None;
        }
        return;
    }

    // Active body: the view anchor's (nearest terrain-backed) body, when
    // vegetated — rocks follow the VIEW, see `rendering::view_anchor`.
    let anchored = anchor.resolved.filter(|a| {
        sim.system
            .bodies
            .get(a.body)
            .is_some_and(|b| terrain_shading_style_for(b) == TerrainShadingStyle::Vegetated)
    });
    let Some(view) = anchored else {
        if rocks.body.is_some() {
            despawn_all(&mut rocks, &mut commands);
            rocks.body = None;
        }
        return;
    };
    let body_id = view.body;
    if rocks.body != Some(body_id) {
        despawn_all(&mut rocks, &mut commands);
        rocks.body = Some(body_id);
        let radius_m = sim.system.bodies[body_id].radius_m;
        rocks.lattice = Some(TileLattice::for_body(radius_m, ROCK_TILE_SIZE_M));
    }

    let radius_m = view.radius_m;
    let Some(height_source) = height_sources.get(body_id) else {
        return;
    };
    let mirror = rendered_ground.get(body_id);
    let Some(lattice) = rocks.lattice else {
        return;
    };

    let cam_dir = view.cam_dir;
    let agl = view.agl_m;
    if agl > ROCK_DESPAWN_AGL_M {
        if !rocks.tiles.is_empty() || !rocks.in_flight.is_empty() {
            despawn_all(&mut rocks, &mut commands);
        }
        return;
    }

    let arc_dist = |center_dir: DVec3| -> f64 { center_dir.angle_between(cam_dir) * radius_m };
    let reach = ROCK_REACH_M + ROCK_FADE_BAND_M as f64 + ROCK_TILE_SIZE_M;

    // Despawn tiles past the reach (+ a fade band + a tile of slack).
    let stale: Vec<TileKey> = rocks
        .tiles
        .keys()
        .filter(|key| {
            lattice
                .frame(**key)
                .is_none_or(|(center, _)| arc_dist(center) > reach)
        })
        .copied()
        .collect();
    for key in stale {
        if let Some(tile) = rocks.tiles.remove(&key)
            && let Some(entity) = tile.entity
        {
            commands.entity(entity).despawn();
        }
    }

    // No new builds while high above the rock shell (existing tiles persist).
    if agl > ROCK_BUILD_MAX_AGL_M {
        return;
    }

    let slots = ROCK_MAX_IN_FLIGHT.saturating_sub(rocks.in_flight.len());
    if slots == 0 {
        return;
    }

    // Gather missing-tile candidates around the player, nearest first. Build a
    // tile beyond the reach so its stones finish building while still scaled ~0
    // (invisible build → no pop-in).
    let hi = ROCK_REACH_M + ROCK_TILE_SIZE_M;
    let center_key = lattice.key_of(cam_dir);
    let window = (hi / (ROCK_TILE_SIZE_M * 0.5)).ceil() as i64;
    let mut candidates: Vec<(f64, TileKey)> = Vec::new();
    for dy in -window..=window {
        for dx in -window..=window {
            let key = TileKey {
                face: center_key.face,
                x: center_key.x + dx,
                y: center_key.y + dy,
            };
            if rocks.tiles.contains_key(&key) || rocks.in_flight.contains_key(&key) {
                continue;
            }
            let Some((center, _)) = lattice.frame(key) else {
                continue;
            };
            let d = arc_dist(center);
            if d <= hi {
                candidates.push((d, key));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Sea level is the project datum (the constant 0 m). Rocks require
    // `height > sea_level + 1 m`, so the seabed stays bare.
    let sea_level_m = 0.0;
    let flatten_exclusion = flatten_registry
        .handle(body_id)
        .read()
        .ok()
        .and_then(|guard| thalos_terrain::nearest_flatten(&guard, cam_dir));

    let ground = mirror.as_ref();
    let pool = AsyncComputeTaskPool::get();
    let mut dispatched = 0usize;
    for (_, key) in candidates {
        if dispatched >= slots {
            break;
        }
        if let Some(ground) = ground {
            let Some((center, _)) = lattice.frame(key) else {
                continue;
            };
            match ground.best_resident_texel_m(center.as_vec3()) {
                Some(texel) if texel <= ROCK_MAX_TERRAIN_TEXEL_M => {}
                _ => continue, // terrain not detailed here yet — retry next frame
            }
        }
        let input = VegScatterInput {
            key,
            lattice,
            radius_m,
            height_source: Arc::clone(&height_source),
            species: Arc::clone(&library.placement),
            seed: ROCK_SEED,
            sea_level_m,
            flatten_exclusion,
            spacing_scale: 1.0,
            keep_fraction: 1.0,
        };
        let meshes = library.meshes.clone();
        let revision = height_source.revision();
        let task = pool.spawn(async move {
            let tile = build_scatter_tile(&input)?;
            let mesh = combine_rock_tile_mesh(&tile.instances, &meshes)?;
            Some(RockTileBuild {
                mesh,
                center_surface_body_m: tile.center_surface_body_m,
                built_revision: tile.built_revision,
                center_height_m: tile.center_height_m,
            })
        });
        rocks.in_flight.insert(key, (task, revision));
        dispatched += 1;
    }
}

/// Poll in-flight builds; spawn each finished tile's batched mesh as a root-grid
/// big_space child (the grass / runway pattern).
fn finalize_rock_tiles(
    mut rocks: ResMut<RockTiles>,
    solar: Res<SolarSystemState>,
    root: Option<Res<RealSpaceRoot>>,
    library: Option<Res<RockLibrary>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut commands: Commands,
) {
    if rocks.in_flight.is_empty() {
        return;
    }
    let (Some(library), Some(states), Some(root), Some(body_id)) =
        (library, solar.states.as_deref(), root, rocks.body)
    else {
        return;
    };
    let Some(body_state) = states.get(body_id) else {
        return;
    };

    let mut finished: Vec<(TileKey, u64, Option<RockTileBuild>)> = Vec::new();
    rocks
        .in_flight
        .retain(|key, (task, revision)| match block_on(poll_once(task)) {
            Some(result) => {
                finished.push((*key, *revision, result));
                false
            }
            None => true,
        });

    let orientation = body_state.orientation.normalize();
    for (key, revision, result) in finished {
        let Some(build) = result else {
            rocks.tiles.insert(
                key,
                BuiltTile {
                    entity: None,
                    built_revision: revision,
                    center_height_m: 0.0,
                },
            );
            continue;
        };

        let center = build.center_surface_body_m;
        let center_world = body_state.position + orientation * center;
        let (cell, local) = real_space_grid().translation_to_grid(center_world);
        // Explicit local-space AABB so the `RENDER_WORLD`-only mesh is
        // frustum-culled (Bevy never auto-computes one — see grass / vegetation).
        let aabb = build.mesh.compute_aabb();
        let mut tile_cmd = commands.spawn((
            Mesh3d(meshes.add(build.mesh)),
            MeshMaterial3d(library.material.clone()),
            Transform {
                translation: local,
                rotation: orientation.as_quat(),
                scale: Vec3::ONE,
            },
            cell,
            Visibility::Inherited,
            // Rocks cast into the sun-shadow cascade (and receive it), so a stone
            // grounds with its own little shadow like the trees.
            RenderLayers::from_layers(&[
                SHIP_LAYER,
                crate::rendering::sun_shadow::SHADOW_CASTER_LAYER,
            ]),
            NotShadowCaster,
            ChildOf(root.entity),
            RockTileVisual {
                body_id,
                center_surface_body: center,
            },
            Name::new("Rock Tile"),
        ));
        if let Some(aabb) = aabb {
            tile_cmd.insert(aabb);
        }
        let entity = tile_cmd.id();
        rocks.tiles.insert(
            key,
            BuiltTile {
                entity: Some(entity),
                built_revision: build.built_revision,
                center_height_m: build.center_height_m,
            },
        );
    }
}

/// Re-anchor every rock tile in f64 each frame (the grass / runway pattern).
fn update_rock_transforms(
    solar: Res<SolarSystemState>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut tiles: Query<(&RockTileVisual, &mut CellCoord, &mut Transform)>,
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

/// Per-frame rock shading on the single shared [`RockMaterial`]: sun direction +
/// flux, the shared `thalos::lighting` sky inputs, the live sun-shadow cascade,
/// and the view-anchored scale-fade. Mirrors `rendering::grass::update_grass_material`.
#[allow(clippy::too_many_arguments)]
fn update_rock_material(
    library: Option<Res<RockLibrary>>,
    rocks: Res<RockTiles>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    time: Res<Time>,
    exposure: Res<CameraExposure>,
    anchor: Res<ViewAnchor>,
    sun_shadow: Option<Res<SunShadowState>>,
    mut materials: ResMut<Assets<RockMaterial>>,
) {
    let Some(library) = library else {
        return;
    };
    let (Some(body_id), Some(states)) = (rocks.body, solar.states.as_deref()) else {
        return;
    };
    let Some(body_state) = states.get(body_id) else {
        return;
    };
    let Some(mut material) = materials.get_mut(&library.material) else {
        return;
    };

    let star_pos = states.first().map(|s| s.position).unwrap_or(DVec3::ZERO);
    let offset = star_pos - body_state.position;
    let sun_dir = offset.normalize_or_zero().as_vec3();
    let au_over_d = (AU_M / offset.length().max(1.0)) as f32;
    let flux = LIGHT_AT_1AU * au_over_d * au_over_d * exposure.gain;
    let sun = Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, flux);

    // Local vertical at the VIEW (the tiles exist around the view anchor).
    let up = anchor
        .resolved
        .filter(|a| a.body == body_id)
        .map(|a| (body_state.orientation * a.cam_dir).as_vec3())
        .unwrap_or_else(|| {
            (sim.simulation.ship_state().position - body_state.position)
                .normalize_or_zero()
                .as_vec3()
        });
    let sky_up = Vec4::new(up.x, up.y, up.z, 0.0);

    let (tau, strength) = sim
        .system
        .bodies
        .get(body_id)
        .and_then(|b| b.terrestrial_atmosphere.as_ref())
        .and_then(|a| a.scattering.as_ref())
        .map(|s| (Vec3::from_array(s.vertical_optical_depth), s.strength))
        .unwrap_or((Vec3::ZERO, 0.0));
    let sky_tau = Vec4::new(tau.x, tau.y, tau.z, strength);

    // Fade reference = the VIEW (`view.world_position` in the shader, offset 0):
    // the scale-fade is a per-instance LOD keyed by distance from the eye —
    // origin-invariant and this-frame-exact (same as `rendering::grass`).
    let anchor = Vec4::ZERO;

    let (near, far, band) = rock_fade();
    material.params.sun_dir = sun;
    material.params.time_fade = Vec4::new(time.elapsed_secs(), near, far, band);
    material.params.sky_up = sky_up;
    material.params.sky_tau = sky_tau;
    material.params.anchor = anchor;
    if let Some(sun_shadow) = sun_shadow.as_deref() {
        material.shadow = sun_shadow.block;
        material.sun_shadow_map_0 = sun_shadow.images[0].clone();
        material.sun_shadow_map_1 = sun_shadow.images[1].clone();
        material.sun_shadow_map_2 = sun_shadow.images[2].clone();
    }
}

/// Periodically rebuild tiles whose underlying height shifted (a finer atlas
/// tile streamed in, or a flatten pad installed): despawn + forget the stale
/// tile so `drive_rock_tiles` re-dispatches it.
fn check_rock_rebuilds(
    mut rocks: ResMut<RockTiles>,
    height_sources: Res<HeightSourceRegistry>,
    time: Res<Time>,
    mut commands: Commands,
) {
    rocks.rebuild_timer += time.delta_secs();
    if rocks.rebuild_timer < ROCK_REBUILD_CHECK_S {
        return;
    }
    rocks.rebuild_timer = 0.0;

    let Some(body_id) = rocks.body else {
        return;
    };
    let Some(source) = height_sources.get(body_id) else {
        return;
    };
    let Some(lattice) = rocks.lattice else {
        return;
    };
    let revision = source.revision();

    let mut rebuilt = 0usize;
    let mut to_remove: Vec<TileKey> = Vec::new();
    for (key, tile) in rocks.tiles.iter_mut() {
        if tile.built_revision == revision {
            continue;
        }
        let Some((center_dir, _)) = lattice.frame(*key) else {
            continue;
        };
        let Some(h) = source.sample_height_m(center_dir.as_vec3(), ROCK_GROUND_LOD_M) else {
            continue;
        };
        if tile.entity.is_some()
            && (h - tile.center_height_m).abs() > ROCK_REBUILD_DELTA_M
            && rebuilt < ROCK_MAX_REBUILDS_PER_TICK
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
        if let Some(tile) = rocks.tiles.remove(&key)
            && let Some(entity) = tile.entity
        {
            commands.entity(entity).despawn();
        }
    }
}
