//! Grass-blade decoration driver — clipmap rings.
//!
//! Maintains a camera-local set of body-fixed grass tiles around the player on
//! the nearest vegetated body, using the tile builder + material in
//! `thalos_body_render::ground::vegetation`. The driver owns the tile
//! lifecycle only — which tiles exist, where they are anchored, what the wind
//! and sun are doing; all placement/geometry logic lives in the engine crate.
//!
//! **Clipmap.** Grass reaches the horizon through concentric LOD rings: each
//! ring is a coarser cube-sphere lattice (tile size doubles outward), so each
//! is a thin annulus of a bounded number of tiles. Near rings use the full
//! curved blade at high density; far rings use a cheap wide "clump" blade at low
//! density but widened so ground *coverage* stays roughly constant (no bald
//! ground). Each ring has its own material carrying a near/far/band fade, so
//! adjacent rings cross-fade through their shared boundary (`grass.wgsl`).
//!
//! Anchoring follows the runway pattern exactly (`runway::update_runway_transform`):
//! each tile is a **root-grid big_space child** whose position is recomputed
//! in f64 every frame from the body's state, so the f32 `Transform.rotation`
//! only ever acts on the tile's small vertex offsets and the grass stays
//! rock-steady under high warp.
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
use bevy::math::{DVec3, Vec3, Vec4};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use big_space::prelude::{BigSpace, CellCoord, Grid};

use thalos_body_render::{
    AU_M, GrassBladeLod, GrassMaterial, GrassTileBuildInput, GrassTileKey, GrassTileMesh,
    LIGHT_AT_1AU, TerrainShadingStyle, build_grass_tile_mesh, grass_tile_frame, grass_tile_key,
    grass_tiles_per_side,
};
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::SimStage;
use crate::coords::SHIP_LAYER;
use crate::camera::ShipCamera;
use crate::freecam::{FreeCam, scatter_view_center};
use crate::rendering::ground_terrain::{TerrainFlattenRegistry, terrain_shading_style_for};
use crate::rendering::real_space::{RealSpaceRoot, real_space_grid};
use crate::rendering::types::{CameraExposure, PlayerShip};
use crate::solar_system_state::{SimulationState, SolarSystemState, sync_solar_system_state};

// ── Clipmap rings ─────────────────────────────────────────────────────────────
/// One LOD ring: a coarser lattice with its own density / blade size / fade.
struct GrassRing {
    /// Metric tile side at a cube-face centre.
    tile_size_m: f64,
    /// Inner ground distance the ring covers (= previous ring's outer).
    inner_m: f64,
    /// Outer ground distance the ring covers.
    outer_m: f64,
    /// Candidate (placement-point) density per m² before gates.
    density_per_m2: f32,
    /// Blades fanned per accepted point. Coverage = density × clump, but only
    /// `density` pays the placement gate — so near rings get a thick carpet
    /// cheaply, far rings stay at 1 (a single wide clump card).
    blades_per_clump: u32,
    /// Blade width multiplier (constant-coverage rule: density ↓ ⇒ width ↑).
    width_scale: f32,
    /// Blade height multiplier.
    height_scale: f32,
    blade_lod: GrassBladeLod,
    /// Forest-cull strength `[0, 1]`: how aggressively this ring thins grass
    /// under tree canopy (occluded → pure overdraw). Near rings keep all grass
    /// (`0`); far rings ramp up so distant grass survives only on open plains.
    forest_cull: f32,
}

/// The clipmap: near full-detail blades → far wide clumps → terrain albedo
/// carries the rest. Reaches ~1.5 km. Far rings are blade-count-capped per tile
/// (`MAX_BLADES_PER_TILE`), so they widen + heighten the clumps aggressively to
/// hold *coverage* — a distant grassfield reads from blade height occluding at
/// grazing angles, not footprint area (see `docs/vegetation.md`).
const GRASS_RINGS: [GrassRing; 5] = [
    GrassRing {
        tile_size_m: 25.0,
        inner_m: 0.0,
        outer_m: 55.0,
        density_per_m2: 24.0,
        blades_per_clump: 5,
        width_scale: 1.0,
        height_scale: 1.0,
        blade_lod: GrassBladeLod::Full,
        // Near band: keep all grass, including the forest floor between trunks.
        forest_cull: 0.0,
    },
    GrassRing {
        tile_size_m: 50.0,
        inner_m: 55.0,
        outer_m: 140.0,
        density_per_m2: 12.0,
        blades_per_clump: 4,
        width_scale: 1.8,
        height_scale: 1.1,
        blade_lod: GrassBladeLod::Wide,
        forest_cull: 0.0,
    },
    GrassRing {
        tile_size_m: 100.0,
        inner_m: 140.0,
        outer_m: 340.0,
        density_per_m2: 6.0,
        blades_per_clump: 3,
        width_scale: 3.2,
        height_scale: 1.3,
        blade_lod: GrassBladeLod::Wide,
        forest_cull: 0.55,
    },
    GrassRing {
        tile_size_m: 200.0,
        inner_m: 340.0,
        outer_m: 760.0,
        density_per_m2: 2.5,
        blades_per_clump: 2,
        width_scale: 7.0,
        height_scale: 1.8,
        blade_lod: GrassBladeLod::Wide,
        forest_cull: 0.85,
    },
    GrassRing {
        tile_size_m: 400.0,
        inner_m: 760.0,
        outer_m: 1500.0,
        density_per_m2: 1.0,
        blades_per_clump: 1,
        width_scale: 16.0,
        height_scale: 2.6,
        blade_lod: GrassBladeLod::Wide,
        forest_cull: 0.95,
    },
];

/// Fade band half-width for a ring's near/far cross-fade.
fn ring_band_m(ring: &GrassRing) -> f32 {
    (((ring.outer_m - ring.inner_m) as f32) * 0.12).clamp(3.0, 70.0)
}

/// Near/far/band fade parameters for a ring (packed into `GrassParams.time_fade`
/// yzw). The innermost ring uses a large-negative near edge so it never fades
/// in. The build/existence range extends one band beyond each edge so adjacent
/// rings overlap and cross-fade.
fn ring_fade(idx: usize) -> (f32, f32, f32) {
    let r = &GRASS_RINGS[idx];
    let band = ring_band_m(r);
    let near = if idx == 0 { -1.0e6 } else { r.inner_m as f32 };
    (near, r.outer_m as f32, band)
}

// ── Tuning ───────────────────────────────────────────────────────────────────
/// Above this altitude over the local terrain no new tiles are built.
const GRASS_MAX_AGL_M: f64 = 600.0;
/// Above this altitude all tiles are despawned (e.g. after launch).
const GRASS_DESPAWN_AGL_M: f64 = 2500.0;
/// Maximum concurrent tile builds across all rings. Builds only dispatch for
/// tiles whose terrain is resident at a fine LOD (cheap GPU-mirror samples).
const GRASS_MAX_IN_FLIGHT: usize = 8;
/// Don't build grass until the terrain under a tile is resident at this texel
/// size or finer (the floating-carpet gate).
const GRASS_MAX_TERRAIN_TEXEL_M: f32 = 8.0;
/// World seed for blade placement hashes.
const GRASS_SEED: u64 = 0x6772_6173_7321;
/// Wind sway amplitude at the blade tip, metres.
const GRASS_WIND_SWAY_M: f32 = 0.06;
/// Rebuild-staleness scan interval, seconds.
const GRASS_REBUILD_CHECK_S: f32 = 0.5;
/// Rebuild a stale tile only when its centre height moved more than this.
///
/// Must stay **above the height-sample noise floor**, or rotation-driven atlas
/// re-streaming triggers a constant despawn→rebuild churn that pops tiles in and
/// out (the despawn→async-rebuild gap is far more visible than the height change
/// it chases). The GPU height mirror stores R16-quantized heights, so re-serving
/// the same point from a different LOD/atlas slot shifts it by ~1 quantization
/// step (~0.05–0.10 m on Thalos, measured). Terrain is LOD-invariant by design
/// (see `docs/terrain.md`), so the only *real* change grass must chase is a
/// flatten-pad install (the runway — metres); 0.5 m clears the noise with margin
/// and still catches pads.
const GRASS_REBUILD_DELTA_M: f32 = 0.5;
/// Stale-tile rebuilds dispatched per scan tick.
const GRASS_MAX_REBUILDS_PER_TICK: usize = 2;

/// A tile key tagged with its clipmap ring (the same `(face,x,y)` means
/// different tiles at different ring resolutions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RingTileKey {
    ring: u8,
    key: GrassTileKey,
}

/// One finished tile. `entity: None` means the tile built empty (water, rock,
/// alpine, flattened pad) — recorded so it isn't rebuilt every frame.
struct BuiltTile {
    entity: Option<Entity>,
    built_revision: u64,
    center_height_m: f32,
}

/// Driver state. **Sole writer:** the systems in this module (drive → finalize
/// → rebuild-check run sequentially via their `ResMut` access).
#[derive(Resource, Default)]
struct GrassTiles {
    body: Option<BodyId>,
    /// `tiles_per_side` for each clipmap ring on the current body.
    ring_tiles_per_side: Vec<i64>,
    tiles: HashMap<RingTileKey, BuiltTile>,
    /// In-flight builds, with the source revision snapshotted at dispatch.
    in_flight: HashMap<RingTileKey, (Task<Option<GrassTileMesh>>, u64)>,
    /// One material per ring (carries that ring's fade parameters).
    materials: Vec<Handle<GrassMaterial>>,
    rebuild_timer: f32,
    /// Per-second churn counters (grass-flicker investigation; logged + reset
    /// by `log_grass_diagnostics`). Remove once the flicker cause is pinned.
    dbg: GrassDiag,
}

/// Diagnostic event counters accumulated over one second.
#[derive(Default)]
struct GrassDiag {
    reach_despawns: u32,
    rebuild_despawns: u32,
    dispatched: u32,
    empty: u32,
    /// Largest |Δheight| that triggered a rebuild this second — tests whether
    /// the sampled terrain height under a fixed tile actually wobbles.
    max_rebuild_dh_m: f32,
    log_timer: f32,
    last_revision: u64,
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
                log_grass_diagnostics.after(update_grass_transforms),
            )
                .in_set(SimStage::Sync)
                .after(sync_solar_system_state),
        );
    }
}

/// Pick the active grass body and keep the clipmap tile set around the player's
/// ground point: dispatch builds for missing tiles (nearest first across all
/// rings), despawn tiles beyond each ring's reach.
///
/// Centered on the **canonical player state**, not the camera (the canonical
/// position is at the same epoch as `solar.states`; the camera's big_space cell
/// lags a frame — kilometre-scale at orbital speed).
#[allow(clippy::too_many_arguments)]
fn drive_grass_tiles(
    mut grass: ResMut<GrassTiles>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    mut flatten_registry: ResMut<TerrainFlattenRegistry>,
    freecam: Res<FreeCam>,
    ship_cam_q: Query<(&CellCoord, &Transform), With<ShipCamera>>,
    mut commands: Commands,
) {
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let cam_pos = scatter_view_center(
        &freecam,
        ship_cam_q.single().ok(),
        sim.simulation.ship_state().position,
    );

    // Active body: nearest vegetated, terrain-backed body (the clouds driver's
    // selection rule, narrowed to bodies that can grow grass).
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
        grass.ring_tiles_per_side = GRASS_RINGS
            .iter()
            .map(|r| grass_tiles_per_side(radius_m, r.tile_size_m))
            .collect();
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
        .sample_height_m(cam_dir.as_vec3(), 25.0)
        .unwrap_or(0.0) as f64;
    let agl = cam_r - (radius_m + ground_h);
    if agl > GRASS_DESPAWN_AGL_M {
        if !grass.tiles.is_empty() || !grass.in_flight.is_empty() {
            despawn_all(&mut grass, &mut commands);
        }
        return;
    }

    let ring_tps = grass.ring_tiles_per_side.clone();
    if ring_tps.len() != GRASS_RINGS.len() {
        return;
    }
    let arc_dist = |center_dir: DVec3| -> f64 { center_dir.angle_between(cam_dir) * radius_m };

    // Despawn tiles past their ring's reach (outer edge + a fade band of slack).
    let stale: Vec<RingTileKey> = grass
        .tiles
        .keys()
        .filter(|rk| {
            let ring = &GRASS_RINGS[rk.ring as usize];
            let reach = ring.outer_m + ring_band_m(ring) as f64 + ring.tile_size_m;
            grass_tile_frame(rk.key, ring_tps[rk.ring as usize])
                .is_none_or(|(center, _)| arc_dist(center) > reach)
        })
        .copied()
        .collect();
    for rk in stale {
        if let Some(tile) = grass.tiles.remove(&rk)
            && let Some(entity) = tile.entity
        {
            commands.entity(entity).despawn();
            grass.dbg.reach_despawns += 1;
        }
    }

    // No new builds while high above the grass shell (existing tiles persist).
    if agl > GRASS_MAX_AGL_M {
        return;
    }

    let slots = GRASS_MAX_IN_FLIGHT.saturating_sub(grass.in_flight.len());
    if slots == 0 {
        return;
    }

    // Gather missing-tile candidates across every ring, nearest first. Each ring
    // overlaps its neighbour at the near edge (one fade band) so they cross-fade,
    // and extends a full tile *beyond* its outer (fade) edge so the outermost
    // tiles build while their blades are scaled to ~0 — the build is invisible
    // (no pop-in), and they grow in as the craft approaches.
    let mut candidates: Vec<(f64, RingTileKey)> = Vec::new();
    for (ring_idx, ring) in GRASS_RINGS.iter().enumerate() {
        let tps = ring_tps[ring_idx];
        let band = ring_band_m(ring) as f64;
        let lo = (ring.inner_m - band).max(0.0);
        let hi = ring.outer_m + ring.tile_size_m;
        let center_key = grass_tile_key(cam_dir, tps);
        let window = (hi / (ring.tile_size_m * 0.5)).ceil() as i64;
        for dy in -window..=window {
            for dx in -window..=window {
                let key = GrassTileKey {
                    face: center_key.face,
                    x: center_key.x + dx,
                    y: center_key.y + dy,
                };
                let rk = RingTileKey {
                    ring: ring_idx as u8,
                    key,
                };
                if grass.tiles.contains_key(&rk) || grass.in_flight.contains_key(&rk) {
                    continue;
                }
                let Some((center, _)) = grass_tile_frame(key, tps) else {
                    continue;
                };
                let d = arc_dist(center);
                if d >= lo && d <= hi {
                    candidates.push((d, rk));
                }
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Water disabled until the generator grows a sea level (terrain rewrite
    // Slice 1), so no sea-level cull for now.
    let sea_level_m = f32::MIN;
    let flatten_exclusion = flatten_registry
        .handle(body_id)
        .read()
        .ok()
        .and_then(|guard| *guard);

    let mirror_guard = mirror.as_ref().and_then(|m| m.read().ok());
    let pool = AsyncComputeTaskPool::get();
    let mut dispatched = 0usize;
    for (_, rk) in candidates {
        if dispatched >= slots {
            break;
        }
        let ring = &GRASS_RINGS[rk.ring as usize];
        let tps = ring_tps[rk.ring as usize];
        if let Some(guard) = &mirror_guard {
            let Some((center, _)) = grass_tile_frame(rk.key, tps) else {
                continue;
            };
            // Far rings tolerate coarser terrain — their clump blades are huge,
            // so the residency threshold scales with tile size. Without this the
            // distant rings (over coarse far terrain) never pass the gate and
            // grass never reaches the horizon.
            let texel_limit = ((ring.tile_size_m * 0.5) as f32).max(GRASS_MAX_TERRAIN_TEXEL_M);
            match guard.best_resident_texel_m(center.as_vec3()) {
                Some(texel) if texel <= texel_limit => {}
                _ => continue, // terrain not detailed here yet — retry next frame
            }
        }
        let input = GrassTileBuildInput {
            key: rk.key,
            tiles_per_side: tps,
            height_source: Arc::clone(&height_source),
            radius_m,
            sea_level_m,
            blades_per_m2: ring.density_per_m2,
            blades_per_clump: ring.blades_per_clump,
            blade_lod: ring.blade_lod,
            width_scale: ring.width_scale,
            height_scale: ring.height_scale,
            seed: GRASS_SEED,
            flatten_exclusion,
            forest_cull: ring.forest_cull,
        };
        let revision = height_source.revision();
        let task = pool.spawn(async move { build_grass_tile_mesh(&input) });
        grass.in_flight.insert(rk, (task, revision));
        dispatched += 1;
        grass.dbg.dispatched += 1;
    }
}

/// Poll in-flight builds; spawn finished tiles as root-grid big_space children
/// (the runway visual pattern), each on its ring's material.
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
    let (Some(states), Some(root), Some(body_id)) = (solar.states.as_deref(), root, grass.body)
    else {
        return;
    };
    let Some(body_state) = states.get(body_id) else {
        return;
    };

    ensure_ring_materials(&mut grass, &mut materials);
    let ring_materials = grass.materials.clone();

    let mut finished: Vec<(RingTileKey, u64, Option<GrassTileMesh>)> = Vec::new();
    grass
        .in_flight
        .retain(|rk, (task, revision)| match block_on(poll_once(task)) {
            Some(result) => {
                finished.push((*rk, *revision, result));
                false
            }
            None => true,
        });

    let orientation = body_state.orientation.normalize();
    for (rk, revision, result) in finished {
        let Some(tile) = result else {
            grass.tiles.insert(
                rk,
                BuiltTile {
                    entity: None,
                    built_revision: revision,
                    center_height_m: 0.0,
                },
            );
            grass.dbg.empty += 1;
            continue;
        };

        let center_world = body_state.position + orientation * tile.center_surface_body_m;
        let (cell, local) = real_space_grid().translation_to_grid(center_world);
        let material = ring_materials[rk.ring as usize].clone();
        let entity = commands
            .spawn((
                Mesh3d(meshes.add(tile.mesh)),
                MeshMaterial3d(material),
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
            rk,
            BuiltTile {
                entity: Some(entity),
                built_revision: tile.built_revision,
                center_height_m: tile.center_height_m,
            },
        );
    }
}

/// Lazily create one material per clipmap ring, each seeded with its fade band.
fn ensure_ring_materials(grass: &mut GrassTiles, materials: &mut Assets<GrassMaterial>) {
    if grass.materials.len() == GRASS_RINGS.len() {
        return;
    }
    grass.materials = (0..GRASS_RINGS.len())
        .map(|idx| {
            let (near, far, band) = ring_fade(idx);
            materials.add(GrassMaterial {
                params: thalos_body_render::GrassParams {
                    time_fade: Vec4::new(0.0, near, far, band),
                    ..default()
                },
            })
        })
        .collect();
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
/// slowly veering wind direction tangent to the surface under the camera. The
/// shared fields are written to every ring material; each keeps its own fade.
fn update_grass_material(
    grass: Res<GrassTiles>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    time: Res<Time>,
    exposure: Res<CameraExposure>,
    ship: Query<&GlobalTransform, With<PlayerShip>>,
    ship_cam: Query<&GlobalTransform, With<ShipCamera>>,
    freecam: Res<FreeCam>,
    mut materials: ResMut<Assets<GrassMaterial>>,
) {
    let (Some(body_id), Some(states)) = (grass.body, solar.states.as_deref()) else {
        return;
    };
    if grass.materials.is_empty() {
        return;
    }
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
    let sun = Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, flux);

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
    let wind_dir = (east * veer.cos() + north * veer.sin()).normalize_or_zero();
    let wind = Vec4::new(wind_dir.x, wind_dir.y, wind_dir.z, GRASS_WIND_SWAY_M);
    let sky_up = Vec4::new(up.x, up.y, up.z, 0.0);

    // Sky hemisphere inputs: the body's authored Rayleigh τ_v + strength (the
    // same value the terrain shader recovers from its `AtmosphereBlock`).
    let (tau, strength) = sim
        .system
        .bodies
        .get(body_id)
        .and_then(|b| b.terrestrial_atmosphere.as_ref())
        .and_then(|a| a.scattering.as_ref())
        .map(|s| (Vec3::from_array(s.vertical_optical_depth), s.strength))
        .unwrap_or((Vec3::ZERO, 0.0));
    let sky_tau = Vec4::new(tau.x, tau.y, tau.z, strength);

    // Fade reference = the player craft, passed as an OFFSET from the camera
    // (`ship − camera`) so the shader can rebuild it in the current frame's
    // render origin (`view.world_position + offset`). An absolute anchor is one
    // frame stale and breaks across big_space floating-origin recentres — it
    // jumps a whole cell while the parked craft co-rotates through space,
    // popping fade-band tiles in/out (see `grass.wgsl`). The offset is
    // origin-invariant, so the recentre cancels. EVA has no PlayerShip and
    // freecam flies free of the player → offset 0 = fade around the camera.
    let cam_render = ship_cam.iter().next().map(|gt| gt.translation());
    let anchor = match (freecam.active, ship.iter().next(), cam_render) {
        (false, Some(ship_gt), Some(cam)) => {
            let off = ship_gt.translation() - cam;
            Vec4::new(off.x, off.y, off.z, 1.0)
        }
        _ => Vec4::ZERO,
    };

    for (idx, handle) in grass.materials.iter().enumerate() {
        let Some(material) = materials.get_mut(handle) else {
            continue;
        };
        let (near, far, band) = ring_fade(idx);
        material.params.sun_dir = sun;
        material.params.wind = wind;
        material.params.time_fade = Vec4::new(t, near, far, band);
        material.params.sky_up = sky_up;
        material.params.sky_tau = sky_tau;
        material.params.anchor = anchor;
    }
}

/// Periodically reconcile tiles with the height source: when the source
/// revision advances (a finer atlas tile streamed in, a flatten pad was
/// installed), re-sample each stale tile's centre height and rebuild it if the
/// ground actually moved. Rebuild = despawn + forget; `drive_grass_tiles`
/// re-dispatches it on a later pass.
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
    if grass.ring_tiles_per_side.len() != GRASS_RINGS.len() {
        return;
    }
    let ring_tps = grass.ring_tiles_per_side.clone();

    let mut rebuilt = 0usize;
    let mut to_remove: Vec<RingTileKey> = Vec::new();
    let mut max_dh = 0.0f32;
    for (rk, tile) in grass.tiles.iter_mut() {
        if tile.built_revision == revision {
            continue;
        }
        let Some((center_dir, _)) = grass_tile_frame(rk.key, ring_tps[rk.ring as usize]) else {
            continue;
        };
        let Some(h) = source.sample_height_m(center_dir.as_vec3(), 0.5) else {
            continue;
        };
        if tile.entity.is_some()
            && (h - tile.center_height_m).abs() > GRASS_REBUILD_DELTA_M
            && rebuilt < GRASS_MAX_REBUILDS_PER_TICK
        {
            max_dh = max_dh.max((h - tile.center_height_m).abs());
            to_remove.push(*rk);
            rebuilt += 1;
        } else {
            tile.built_revision = revision;
            if tile.entity.is_some() {
                tile.center_height_m = h;
            }
        }
    }
    let removed = to_remove.len() as u32;
    for rk in to_remove {
        if let Some(tile) = grass.tiles.remove(&rk)
            && let Some(entity) = tile.entity
        {
            commands.entity(entity).despawn();
        }
    }
    grass.dbg.rebuild_despawns += removed;
    grass.dbg.max_rebuild_dh_m = grass.dbg.max_rebuild_dh_m.max(max_dh);
}

/// **Diagnostic only** (grass-flicker investigation): once per second, append a
/// JSON line of the tile-churn counters + the height-source revision delta to
/// the file named by `THALOS_GRASS_LOG` (falls back to an `info!` line if the
/// env var is unset). A non-zero `rev_delta` while parked means the terrain
/// atlas is re-streaming under rotation; `rebuild_despawns` / `reach_despawns`
/// say which path is popping tiles; `max_rebuild_dh_m` says whether the sampled
/// height actually wobbled. Remove once the cause is pinned.
fn log_grass_diagnostics(
    mut grass: ResMut<GrassTiles>,
    height_sources: Res<HeightSourceRegistry>,
    time: Res<Time>,
) {
    grass.dbg.log_timer += time.delta_secs();
    if grass.dbg.log_timer < 1.0 {
        return;
    }
    grass.dbg.log_timer = 0.0;

    let Some(body_id) = grass.body else {
        return;
    };
    let revision = height_sources.get(body_id).map(|s| s.revision()).unwrap_or(0);
    let rev_delta = revision.wrapping_sub(grass.dbg.last_revision);
    grass.dbg.last_revision = revision;

    let line = format!(
        "{{\"t_s\":{:.1},\"tiles\":{},\"in_flight\":{},\"revision\":{},\"rev_delta\":{},\
\"reach_despawns\":{},\"rebuild_despawns\":{},\"dispatched\":{},\"empty\":{},\
\"max_rebuild_dh_m\":{:.3}}}",
        time.elapsed_secs(),
        grass.tiles.len(),
        grass.in_flight.len(),
        revision,
        rev_delta,
        grass.dbg.reach_despawns,
        grass.dbg.rebuild_despawns,
        grass.dbg.dispatched,
        grass.dbg.empty,
        grass.dbg.max_rebuild_dh_m,
    );
    // Always write to a file (default at the game's cwd = repo root) so the
    // console slow-frame spam is irrelevant and there's nothing to set up.
    let path =
        std::env::var("THALOS_GRASS_LOG").unwrap_or_else(|_| "grass_churn.jsonl".to_string());
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{line}");
    }

    grass.dbg.reach_despawns = 0;
    grass.dbg.rebuild_despawns = 0;
    grass.dbg.dispatched = 0;
    grass.dbg.empty = 0;
    grass.dbg.max_rebuild_dh_m = 0.0;
}
