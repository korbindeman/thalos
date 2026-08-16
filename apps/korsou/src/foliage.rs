use std::{
    collections::{HashMap, HashSet},
    f32::consts::TAU,
    time::Duration,
};

use bevy::{
    asset::RenderAssetUsages,
    camera::{primitives::MeshAabb, visibility::RenderLayers},
    light::{NotShadowCaster, NotShadowReceiver},
    math::Vec3A,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use thalos_vegetation::{
    CanopyStyle, FoliageImpostorExtension, FoliageImpostorMaterial, FoliageImpostorMaterialPlugin,
    ImpostorAtlasLayout, ImpostorBakeConfig, ImpostorBakeRig, ImpostorInstance, ImpostorViewParams,
    TreeBakeMaterial, TreeMeshData, TreeMeshParams, build_foliage_atlas, build_tree_mesh_data,
    combine_impostor_mesh, despawn_impostor_bake_rig, foliage_impostor_material,
    spawn_impostor_bake_rig,
};

use crate::{
    camera::{TerrainCamera, TerrainCameraSet},
    terrain::{TerrainDataset, canopy_coverage, rendered_height},
};

const CELL_SIZE_M: f64 = 128.0;
const STREAM_INTERVAL_S: f32 = 0.10;
const MAX_CELL_SPAWNS_PER_TICK: usize = 20;
const MIN_VISIBLE_RADIUS_M: f64 = 900.0;
const MAX_VISIBLE_RADIUS_M: f64 = 1_450.0;
const MAX_VISIBLE_CLEARANCE_M: f64 = 3_000.0;
const SLOPE_SAMPLE_M: f64 = 10.0;
const VISUAL_TERRAIN_LEVEL: u8 = 6;
const IMPOSTOR_BAKE_FRAMES: u32 = 90;
const IMPOSTOR_BAKE_ALBEDO_LAYER: usize = 6;
const IMPOSTOR_BAKE_NORMAL_LAYER: usize = 7;
pub(crate) const FOLIAGE_SHADOW_LAYER: usize = 20;
const FOLIAGE_SHADOW_RADIUS_M: f64 = 760.0;

const SHRUB_LAYER: PlacementLayer = PlacementLayer {
    species: 0,
    spacing_m: 2.35,
    seed: 0x51_47,
    density_edges: (0.02, 0.62),
    density_scale: 1.0,
    scale_range: (0.72, 1.38),
};

const TREE_LAYER: PlacementLayer = PlacementLayer {
    species: 1,
    spacing_m: 5.5,
    seed: 0xb1_05_50,
    density_edges: (0.18, 0.70),
    density_scale: 0.72,
    scale_range: (0.78, 1.28),
};

pub struct FoliagePlugin;

impl Plugin for FoliagePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FoliageImpostorMaterialPlugin)
            .init_resource::<FoliageStreamState>()
            .init_resource::<FoliageBakeState>()
            .init_resource::<FoliageStats>()
            .add_systems(Startup, setup_foliage_assets)
            .add_systems(
                Update,
                (
                    tick_impostor_bake,
                    stream_foliage
                        .after(TerrainCameraSet::Movement)
                        .after(tick_impostor_bake),
                ),
            );
    }
}

#[derive(Resource)]
struct FoliageAssets {
    material: Handle<FoliageImpostorMaterial>,
    shadow_material: Handle<StandardMaterial>,
    max_extent_m: f32,
}

fn setup_foliage_assets(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<FoliageImpostorMaterial>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    mut bake_materials: ResMut<Assets<TreeBakeMaterial>>,
) {
    let foliage_atlas = images.add(build_foliage_atlas());
    let species = build_species_payloads();
    let species_refs = species.iter().collect::<Vec<_>>();
    let impostor_atlas = spawn_impostor_bake_rig(
        &mut commands,
        &mut images,
        &mut meshes,
        &mut bake_materials,
        &species_refs,
        foliage_atlas,
        ImpostorBakeConfig {
            layout: ImpostorAtlasLayout {
                cells: 8,
                cell_px: 128,
                species: species.len() as u32,
            },
            cell_fill: 0.84,
            alpha_cutoff: 0.35,
            albedo_layer: IMPOSTOR_BAKE_ALBEDO_LAYER,
            normal_layer: IMPOSTOR_BAKE_NORMAL_LAYER,
        },
    );
    let material = materials.add(foliage_impostor_material(FoliageImpostorExtension {
        view: ImpostorViewParams {
            fade: Vec4::new(-1.0e9, MAX_VISIBLE_RADIUS_M as f32, CELL_SIZE_M as f32, 1.0),
            anchor: Vec4::ZERO,
        },
        impostor: impostor_atlas.params,
        albedo: impostor_atlas.albedo,
        normal: impostor_atlas.normal,
    }));
    let shadow_material = standard_materials.add(StandardMaterial {
        base_color: Color::BLACK,
        perceptual_roughness: 1.0,
        ..default()
    });
    commands.insert_resource(FoliageAssets {
        material,
        shadow_material,
        max_extent_m: impostor_atlas.max_extent_m,
    });
}

#[derive(Resource, Default)]
struct FoliageBakeState {
    frames: u32,
    ready: bool,
}

fn tick_impostor_bake(
    mut commands: Commands,
    mut bake: ResMut<FoliageBakeState>,
    rig: Query<Entity, With<ImpostorBakeRig>>,
) {
    if bake.ready {
        return;
    }
    bake.frames += 1;
    if bake.frames >= IMPOSTOR_BAKE_FRAMES {
        despawn_impostor_bake_rig(&mut commands, &rig);
        bake.ready = true;
    }
}

#[derive(Component)]
struct FoliageCell;

#[derive(Resource)]
struct FoliageStreamState {
    resident: HashMap<IVec2, ResidentCell>,
    timer: Timer,
    first_tick: bool,
}

struct ResidentCell {
    entity: Entity,
    mesh: Option<Handle<Mesh>>,
    shadow_entity: Option<Entity>,
    shadow_mesh: Option<Handle<Mesh>>,
    shadow_triangles: usize,
    shadow_visible: bool,
    woody_plants: usize,
}

impl Default for FoliageStreamState {
    fn default() -> Self {
        Self {
            resident: HashMap::new(),
            timer: Timer::new(
                Duration::from_secs_f32(STREAM_INTERVAL_S),
                TimerMode::Repeating,
            ),
            first_tick: true,
        }
    }
}

#[derive(Resource, Default)]
pub struct FoliageStats {
    pub resident: usize,
    pub desired: usize,
    pub queued: usize,
    pub woody_plants: usize,
    pub impostor_vertices: usize,
    pub shadow_cells: usize,
    pub shadow_triangles: usize,
    pub bake_ready: bool,
}

#[allow(clippy::too_many_arguments)]
fn stream_foliage(
    mut commands: Commands,
    time: Res<Time>,
    graphics: Res<thalos_runtime::preferences::GraphicsPreferences>,
    dataset: Res<TerrainDataset>,
    assets: Res<FoliageAssets>,
    bake: Res<FoliageBakeState>,
    camera: Single<&TerrainCamera>,
    mut state: ResMut<FoliageStreamState>,
    mut stats: ResMut<FoliageStats>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    stats.bake_ready = bake.ready;
    if !graphics.foliage {
        clear_foliage(&mut commands, &mut state, &mut stats, &mut meshes);
        return;
    }
    if !bake.ready {
        return;
    }
    if !state.first_tick && !state.timer.tick(time.delta()).just_finished() {
        return;
    }
    state.first_tick = false;

    let ground = f64::from(dataset.dem_height(camera.position_m.x, camera.position_m.z));
    let clearance = (camera.position_m.y - ground).max(0.0);
    let radius = foliage_radius(clearance);
    let desired = desired_cells(
        camera.position_m.x,
        camera.position_m.z,
        radius,
        dataset.land_bounds_local_m(),
    );

    state.resident.retain(|key, cell| {
        if desired.contains(key) {
            true
        } else {
            commands.entity(cell.entity).despawn();
            if let Some(entity) = cell.shadow_entity {
                commands.entity(entity).despawn();
            }
            if let Some(mesh) = &cell.mesh {
                meshes.remove(mesh.id());
            }
            if let Some(mesh) = &cell.shadow_mesh {
                meshes.remove(mesh.id());
            }
            false
        }
    });

    for (key, cell) in &mut state.resident {
        let visible = foliage_shadow_visible(*key, camera.position_m.x, camera.position_m.z);
        if cell.shadow_visible != visible {
            if let Some(entity) = cell.shadow_entity {
                commands.entity(entity).insert(if visible {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                });
            }
            cell.shadow_visible = visible;
        }
    }

    let mut missing: Vec<_> = desired
        .iter()
        .copied()
        .filter(|key| !state.resident.contains_key(key))
        .collect();
    missing.sort_by(|a, b| {
        cell_distance_squared(*a, camera.position_m.x, camera.position_m.z).total_cmp(
            &cell_distance_squared(*b, camera.position_m.x, camera.position_m.z),
        )
    });

    for key in missing.iter().take(MAX_CELL_SPAWNS_PER_TICK).copied() {
        let placements = woody_placements(&dataset, key);
        let instances = placements.iter().map(impostor_instance).collect::<Vec<_>>();
        let mesh = combine_impostor_mesh(&instances);
        let shadow_mesh = build_shadow_proxy_mesh(&placements);
        let shadow_triangles = shadow_mesh
            .as_ref()
            .and_then(Mesh::indices)
            .map_or(0, |indices| indices.len() / 3);
        let woody_plants = placements.len();
        let aabb = mesh
            .as_ref()
            .and_then(MeshAabb::compute_aabb)
            .map(|mut aabb| {
                aabb.half_extents += Vec3A::splat(assets.max_extent_m * TREE_LAYER.scale_range.1);
                aabb
            });
        let mesh = mesh.map(|mesh| meshes.add(mesh));
        let shadow_mesh = shadow_mesh.map(|mesh| meshes.add(mesh));
        let min_x = f64::from(key.x) * CELL_SIZE_M;
        let min_z = f64::from(key.y) * CELL_SIZE_M;
        let mut entity = commands.spawn((
            FoliageCell,
            Transform::from_xyz(min_x as f32, 0.0, min_z as f32),
            Visibility::Inherited,
            Name::new(format!(
                "Foliage cell {},{} ({woody_plants} woody plants)",
                key.x, key.y
            )),
        ));
        if let Some(mesh) = &mesh {
            entity.insert((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(assets.material.clone()),
                NotShadowCaster,
            ));
            if let Some(aabb) = aabb {
                entity.insert(aabb);
            }
        }
        let entity = entity.id();
        let shadow_visible = foliage_shadow_visible(key, camera.position_m.x, camera.position_m.z);
        let shadow_entity = shadow_mesh.as_ref().map(|mesh| {
            commands
                .spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(assets.shadow_material.clone()),
                    Transform::from_xyz(min_x as f32, 0.0, min_z as f32),
                    if shadow_visible {
                        Visibility::Visible
                    } else {
                        Visibility::Hidden
                    },
                    RenderLayers::layer(FOLIAGE_SHADOW_LAYER),
                    NotShadowReceiver,
                    Name::new(format!("Foliage shadow proxy {},{}", key.x, key.y)),
                ))
                .id()
        });
        state.resident.insert(
            key,
            ResidentCell {
                entity,
                mesh,
                shadow_entity,
                shadow_mesh,
                shadow_triangles,
                shadow_visible,
                woody_plants,
            },
        );
    }

    stats.resident = state.resident.len();
    stats.desired = desired.len();
    stats.queued = missing.len().saturating_sub(MAX_CELL_SPAWNS_PER_TICK);
    stats.woody_plants = state.resident.values().map(|cell| cell.woody_plants).sum();
    stats.impostor_vertices = stats.woody_plants * 4;
    stats.shadow_cells = state
        .resident
        .values()
        .filter(|cell| cell.shadow_visible && cell.shadow_entity.is_some())
        .count();
    stats.shadow_triangles = state
        .resident
        .values()
        .filter(|cell| cell.shadow_visible)
        .map(|cell| cell.shadow_triangles)
        .sum();
    stats.bake_ready = true;
}

fn clear_foliage(
    commands: &mut Commands,
    state: &mut FoliageStreamState,
    stats: &mut FoliageStats,
    meshes: &mut Assets<Mesh>,
) {
    for (_, cell) in state.resident.drain() {
        commands.entity(cell.entity).despawn();
        if let Some(entity) = cell.shadow_entity {
            commands.entity(entity).despawn();
        }
        if let Some(mesh) = cell.mesh {
            meshes.remove(mesh.id());
        }
        if let Some(mesh) = cell.shadow_mesh {
            meshes.remove(mesh.id());
        }
    }
    state.timer.reset();
    state.first_tick = true;
    *stats = FoliageStats::default();
}

fn foliage_radius(clearance_m: f64) -> f64 {
    if clearance_m >= MAX_VISIBLE_CLEARANCE_M {
        return 0.0;
    }
    let altitude_weight = (clearance_m / 550.0).clamp(0.0, 1.0);
    MIN_VISIBLE_RADIUS_M + (MAX_VISIBLE_RADIUS_M - MIN_VISIBLE_RADIUS_M) * altitude_weight
}

fn desired_cells(x: f64, z: f64, radius: f64, land_bounds: [f64; 4]) -> HashSet<IVec2> {
    if radius <= 0.0 {
        return HashSet::new();
    }
    let min_x = ((x - radius) / CELL_SIZE_M).floor() as i32;
    let max_x = ((x + radius) / CELL_SIZE_M).floor() as i32;
    let min_z = ((z - radius) / CELL_SIZE_M).floor() as i32;
    let max_z = ((z + radius) / CELL_SIZE_M).floor() as i32;
    let radius_squared = radius * radius;
    let mut cells = HashSet::new();

    for cell_z in min_z..=max_z {
        for cell_x in min_x..=max_x {
            let key = IVec2::new(cell_x, cell_z);
            let cell_min_x = f64::from(cell_x) * CELL_SIZE_M;
            let cell_min_z = f64::from(cell_z) * CELL_SIZE_M;
            let overlaps_land = cell_min_x + CELL_SIZE_M >= land_bounds[0]
                && cell_min_z + CELL_SIZE_M >= land_bounds[1]
                && cell_min_x <= land_bounds[2]
                && cell_min_z <= land_bounds[3];
            if overlaps_land && cell_distance_squared(key, x, z) <= radius_squared {
                cells.insert(key);
            }
        }
    }
    cells
}

fn cell_distance_squared(key: IVec2, x: f64, z: f64) -> f64 {
    let min_x = f64::from(key.x) * CELL_SIZE_M;
    let min_z = f64::from(key.y) * CELL_SIZE_M;
    let dx = if x < min_x {
        min_x - x
    } else if x > min_x + CELL_SIZE_M {
        x - (min_x + CELL_SIZE_M)
    } else {
        0.0
    };
    let dz = if z < min_z {
        min_z - z
    } else if z > min_z + CELL_SIZE_M {
        z - (min_z + CELL_SIZE_M)
    } else {
        0.0
    };
    dx * dx + dz * dz
}

fn foliage_shadow_visible(key: IVec2, x: f64, z: f64) -> bool {
    cell_distance_squared(key, x, z) <= FOLIAGE_SHADOW_RADIUS_M * FOLIAGE_SHADOW_RADIUS_M
}

#[derive(Clone, Copy)]
struct PlacementLayer {
    species: usize,
    spacing_m: f64,
    seed: u32,
    density_edges: (f32, f32),
    density_scale: f32,
    scale_range: (f32, f32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct WoodyPlacement {
    local_position: Vec3,
    yaw: f32,
    scale: f32,
    tint: Vec3,
    species: usize,
}

fn build_species_payloads() -> Vec<TreeMeshData> {
    vec![
        build_tree_mesh_data(&TreeMeshParams {
            trunk_height_m: 0.30,
            trunk_radius_m: 0.055,
            canopy_radius_m: 1.05,
            canopy_height_m: 0.82,
            trunk_color: Vec3::new(0.13, 0.075, 0.035),
            canopy_color: Vec3::new(0.94, 1.0, 0.82),
            style: CanopyStyle::Round,
            seed: 0x51_47,
            lod: 0,
        }),
        build_tree_mesh_data(&TreeMeshParams {
            trunk_height_m: 3.2,
            trunk_radius_m: 0.22,
            canopy_radius_m: 2.25,
            canopy_height_m: 1.75,
            trunk_color: Vec3::new(0.15, 0.082, 0.038),
            canopy_color: Vec3::new(0.92, 1.0, 0.80),
            style: CanopyStyle::Broadleaf,
            seed: 0xb1_05_50,
            lod: 0,
        }),
    ]
}

#[cfg(test)]
fn build_cell_impostor_mesh(dataset: &TerrainDataset, key: IVec2) -> (Option<Mesh>, usize) {
    let placements = woody_placements(dataset, key);
    let instances = placements.iter().map(impostor_instance).collect::<Vec<_>>();
    (combine_impostor_mesh(&instances), placements.len())
}

fn impostor_instance(placement: &WoodyPlacement) -> ImpostorInstance {
    ImpostorInstance {
        position: placement.local_position,
        up: Vec3::Y,
        scale: placement.scale,
        species: placement.species as u32,
        tint: placement.tint,
        yaw: placement.yaw,
    }
}

/// Build coarse crown volumes for the stock Bevy shadow pass. The visible
/// octahedral cards stay `NotShadowCaster`: sending their opaque card quads to
/// the depth pass would produce rectangular shadows. These camera-invisible
/// octahedra approximate only the taller broadleaf crowns and are streamed in
/// a bounded radius, producing grounded foliage shadows at predictable cost.
fn build_shadow_proxy_mesh(placements: &[WoodyPlacement]) -> Option<Mesh> {
    let tree_count = placements
        .iter()
        .filter(|placement| placement.species == TREE_LAYER.species)
        .count();
    if tree_count == 0 {
        return None;
    }

    let mut positions = Vec::with_capacity(tree_count * 6);
    let mut normals = Vec::with_capacity(tree_count * 6);
    let mut indices = Vec::with_capacity(tree_count * 24);
    for placement in placements
        .iter()
        .filter(|placement| placement.species == TREE_LAYER.species)
    {
        let start = positions.len() as u32;
        let radius = 1.85 * placement.scale;
        let centre_y = placement.local_position.y + 3.8 * placement.scale;
        let half_height = 1.55 * placement.scale;
        let centre = Vec3::new(
            placement.local_position.x,
            centre_y,
            placement.local_position.z,
        );
        positions.extend_from_slice(&[
            (centre - Vec3::Y * half_height).to_array(),
            (centre + Vec3::Y * half_height).to_array(),
            (centre + Vec3::X * radius).to_array(),
            (centre + Vec3::Z * radius).to_array(),
            (centre - Vec3::X * radius).to_array(),
            (centre - Vec3::Z * radius).to_array(),
        ]);
        normals.extend_from_slice(&[[0.0, 1.0, 0.0]; 6]);
        indices.extend_from_slice(&[
            start + 1,
            start + 2,
            start + 3,
            start + 1,
            start + 3,
            start + 4,
            start + 1,
            start + 4,
            start + 5,
            start + 1,
            start + 5,
            start + 2,
            start,
            start + 3,
            start + 2,
            start,
            start + 4,
            start + 3,
            start,
            start + 5,
            start + 4,
            start,
            start + 2,
            start + 5,
        ]);
    }

    Some(
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_indices(Indices::U32(indices)),
    )
}

fn woody_placements(dataset: &TerrainDataset, key: IVec2) -> Vec<WoodyPlacement> {
    let mut placements = layer_placements(dataset, key, SHRUB_LAYER);
    placements.extend(layer_placements(dataset, key, TREE_LAYER));
    placements
}

fn layer_placements(
    dataset: &TerrainDataset,
    key: IVec2,
    layer: PlacementLayer,
) -> Vec<WoodyPlacement> {
    let min_x = f64::from(key.x) * CELL_SIZE_M;
    let min_z = f64::from(key.y) * CELL_SIZE_M;
    let max_x = min_x + CELL_SIZE_M;
    let max_z = min_z + CELL_SIZE_M;
    let first_x = (min_x / layer.spacing_m).floor() as i32 - 1;
    let last_x = (max_x / layer.spacing_m).ceil() as i32 + 1;
    let first_z = (min_z / layer.spacing_m).floor() as i32 - 1;
    let last_z = (max_z / layer.spacing_m).ceil() as i32 + 1;
    let mut placements = Vec::new();

    for grid_z in first_z..=last_z {
        for grid_x in first_x..=last_x {
            let (world_x, world_z) = candidate_position(grid_x, grid_z, layer);
            if world_x < min_x || world_x >= max_x || world_z < min_z || world_z >= max_z {
                continue;
            }
            if !candidate_survives(grid_x, grid_z, layer) {
                continue;
            }

            let shore_distance = dataset.shore_distance_m(world_x, world_z);
            let height_m = dataset.dem_height(world_x, world_z);
            let slope = terrain_slope(dataset, world_x, world_z);
            let coverage = canopy_coverage(world_x, world_z, height_m, shore_distance, slope);
            let density = smoothstep(layer.density_edges.0, layer.density_edges.1, coverage)
                * layer.density_scale;
            if hash01(grid_x, grid_z, layer.seed ^ 47) > density {
                continue;
            }

            let variation = hash01(grid_x, grid_z, layer.seed ^ 71);
            let dryness = ((1.0 - coverage) * 0.52
                + hash01(grid_x, grid_z, layer.seed ^ 109) * 0.22)
                .clamp(0.0, 1.0);
            let visual_height = rendered_height(dataset, world_x, world_z, VISUAL_TERRAIN_LEVEL);
            placements.push(WoodyPlacement {
                local_position: Vec3::new(
                    (world_x - min_x) as f32,
                    visual_height + 0.025,
                    (world_z - min_z) as f32,
                ),
                yaw: hash01(grid_x, grid_z, layer.seed ^ 89) * TAU,
                scale: layer.scale_range.0
                    + variation * (layer.scale_range.1 - layer.scale_range.0),
                tint: Vec3::new(
                    0.74 + dryness * 0.18,
                    0.94 - dryness * 0.12,
                    0.62 - dryness * 0.16,
                ),
                species: layer.species,
            });
        }
    }
    placements
}

fn candidate_position(grid_x: i32, grid_z: i32, layer: PlacementLayer) -> (f64, f64) {
    let jitter_x = (f64::from(hash01(grid_x, grid_z, layer.seed ^ 11)) - 0.5) * 0.82;
    let jitter_z = (f64::from(hash01(grid_x, grid_z, layer.seed ^ 29)) - 0.5) * 0.82;
    (
        (f64::from(grid_x) + 0.5 + jitter_x) * layer.spacing_m,
        (f64::from(grid_z) + 0.5 + jitter_z) * layer.spacing_m,
    )
}

fn candidate_survives(grid_x: i32, grid_z: i32, layer: PlacementLayer) -> bool {
    let (x, z) = candidate_position(grid_x, grid_z, layer);
    let priority = hash01(grid_x, grid_z, layer.seed ^ 131);
    for dz in -2..=2 {
        for dx in -2..=2 {
            if dx == 0 && dz == 0 {
                continue;
            }
            let neighbour_x = grid_x + dx;
            let neighbour_z = grid_z + dz;
            let neighbour_priority = hash01(neighbour_x, neighbour_z, layer.seed ^ 131);
            if neighbour_priority < priority
                || (neighbour_priority == priority && (neighbour_x, neighbour_z) < (grid_x, grid_z))
            {
                continue;
            }
            let (other_x, other_z) = candidate_position(neighbour_x, neighbour_z, layer);
            if (other_x - x).hypot(other_z - z) < layer.spacing_m {
                return false;
            }
        }
    }
    true
}

fn terrain_slope(dataset: &TerrainDataset, x: f64, z: f64) -> f32 {
    let west = dataset.dem_height(x - SLOPE_SAMPLE_M, z);
    let east = dataset.dem_height(x + SLOPE_SAMPLE_M, z);
    let north = dataset.dem_height(x, z - SLOPE_SAMPLE_M);
    let south = dataset.dem_height(x, z + SLOPE_SAMPLE_M);
    let dx = (east - west) / (SLOPE_SAMPLE_M as f32 * 2.0);
    let dz = (south - north) / (SLOPE_SAMPLE_M as f32 * 2.0);
    dx.hypot(dz)
}

fn hash01(x: i32, z: i32, seed: u32) -> f32 {
    let mut value = (x as u32).wrapping_mul(0x9E37_79B1)
        ^ (z as u32).wrapping_mul(0x85EB_CA77)
        ^ seed.wrapping_mul(0xC2B2_AE3D);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^= value >> 16;
    value as f32 / u32::MAX as f32
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use bevy::asset::RenderAssetUsages;
    use bevy::math::{DQuat, DVec3};
    use bevy::mesh::{PrimitiveTopology, VertexAttributeValues};

    use super::*;

    #[test]
    fn every_woody_root_has_one_four_vertex_impostor() {
        let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/terrain/curacao");
        let dataset = TerrainDataset::load(&asset_dir).unwrap();
        let key = IVec2::new(-120, -32);
        let (mesh, woody_plants) = build_cell_impostor_mesh(&dataset, key);
        let mesh = mesh.expect("test cell should contain foliage");
        let VertexAttributeValues::Float32x3(positions) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
        else {
            panic!("foliage positions must be Float32x3")
        };
        assert!(woody_plants > 20);
        assert_eq!(positions.len(), woody_plants * 4);
        assert_eq!(mesh.indices().unwrap().len(), woody_plants * 6);
        assert_eq!(
            mesh.attribute(Mesh::ATTRIBUTE_UV_0).unwrap().len(),
            positions.len()
        );
        assert_eq!(
            mesh.attribute(Mesh::ATTRIBUTE_COLOR).unwrap().len(),
            positions.len()
        );
    }

    #[test]
    fn shadow_proxy_uses_tree_crowns_not_impostor_cards() {
        let placements = [
            WoodyPlacement {
                local_position: Vec3::new(1.0, 2.0, 3.0),
                yaw: 0.0,
                scale: 1.0,
                tint: Vec3::ONE,
                species: SHRUB_LAYER.species,
            },
            WoodyPlacement {
                local_position: Vec3::new(4.0, 5.0, 6.0),
                yaw: 0.0,
                scale: 1.0,
                tint: Vec3::ONE,
                species: TREE_LAYER.species,
            },
        ];
        let mesh = build_shadow_proxy_mesh(&placements).expect("tree should build a proxy");

        assert_eq!(mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().len(), 6);
        assert_eq!(mesh.indices().unwrap().len(), 24);
    }

    #[test]
    fn foliage_shadow_reach_is_bounded() {
        let key = IVec2::ZERO;
        assert!(foliage_shadow_visible(key, 64.0, 64.0));
        assert!(!foliage_shadow_visible(
            key,
            FOLIAGE_SHADOW_RADIUS_M + CELL_SIZE_M * 2.0,
            64.0,
        ));
    }

    #[test]
    fn blue_noise_candidates_respect_minimum_spacing() {
        let mut points = Vec::new();
        for z in -20..=20 {
            for x in -20..=20 {
                if candidate_survives(x, z, SHRUB_LAYER) {
                    points.push(candidate_position(x, z, SHRUB_LAYER));
                }
            }
        }
        for (index, point) in points.iter().enumerate() {
            for other in &points[index + 1..] {
                assert!((other.0 - point.0).hypot(other.1 - point.1) >= SHRUB_LAYER.spacing_m);
            }
        }
    }

    #[test]
    fn placements_are_deterministic_and_stay_off_the_beach() {
        let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/terrain/curacao");
        let dataset = TerrainDataset::load(&asset_dir).unwrap();
        let key = IVec2::new(-120, -32);
        let first = woody_placements(&dataset, key);
        let second = woody_placements(&dataset, key);

        assert_eq!(first, second);
        assert!(!first.is_empty());
        for placement in first {
            let world_x = f64::from(key.x) * CELL_SIZE_M + f64::from(placement.local_position.x);
            let world_z = f64::from(key.y) * CELL_SIZE_M + f64::from(placement.local_position.z);
            assert!(dataset.shore_distance_m(world_x, world_z) > 95.0);
        }
    }

    #[test]
    fn foliage_streaming_stops_at_aerial_clearance() {
        assert!(foliage_radius(100.0) >= MIN_VISIBLE_RADIUS_M);
        assert_eq!(foliage_radius(MAX_VISIBLE_CLEARANCE_M), 0.0);
    }

    #[test]
    fn disabling_foliage_despawns_cells_and_releases_meshes() {
        let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/terrain/curacao");
        let dataset = TerrainDataset::load(&asset_dir).unwrap();
        let mut app = App::new();
        app.insert_resource(Time::<()>::default())
            .insert_resource(dataset)
            .insert_resource(thalos_runtime::preferences::GraphicsPreferences {
                foliage: false,
                ..default()
            })
            .insert_resource(FoliageAssets {
                material: Handle::default(),
                shadow_material: Handle::default(),
                max_extent_m: 3.0,
            })
            .insert_resource(FoliageBakeState {
                frames: IMPOSTOR_BAKE_FRAMES,
                ready: true,
            })
            .insert_resource(Assets::<Mesh>::default())
            .insert_resource(FoliageStreamState::default())
            .insert_resource(FoliageStats {
                resident: 1,
                desired: 2,
                queued: 1,
                woody_plants: 42,
                impostor_vertices: 168,
                shadow_cells: 1,
                shadow_triangles: 8,
                bake_ready: true,
            })
            .add_systems(Update, stream_foliage);

        app.world_mut().spawn(TerrainCamera {
            position_m: DVec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            rotation_local: DQuat::IDENTITY,
        });
        let entity = app.world_mut().spawn(FoliageCell).id();
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::RENDER_WORLD,
            ));
        app.world_mut()
            .resource_mut::<FoliageStreamState>()
            .resident
            .insert(
                IVec2::ZERO,
                ResidentCell {
                    entity,
                    mesh: Some(mesh.clone()),
                    shadow_entity: None,
                    shadow_mesh: None,
                    shadow_triangles: 0,
                    shadow_visible: false,
                    woody_plants: 42,
                },
            );

        app.update();

        assert!(app.world().get_entity(entity).is_err());
        assert!(app.world().resource::<Assets<Mesh>>().get(&mesh).is_none());
        assert!(
            app.world()
                .resource::<FoliageStreamState>()
                .resident
                .is_empty()
        );
        assert_eq!(app.world().resource::<FoliageStats>().resident, 0);
        assert!(app.world().resource::<FoliageStreamState>().first_tick);
    }

    #[test]
    fn species_payloads_include_shrubs_and_taller_broadleafs() {
        let species = build_species_payloads();
        assert_eq!(species.len(), 2);
        let height = |payload: &TreeMeshData| {
            payload
                .positions
                .iter()
                .map(|position| position[1])
                .fold(f32::NEG_INFINITY, f32::max)
        };
        assert!(height(&species[1]) > height(&species[0]) * 2.0);
    }
}
