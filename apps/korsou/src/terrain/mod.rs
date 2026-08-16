mod data;
mod mesh;
mod quadtree;
mod rtin;
mod surface;

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    time::Duration,
};

use bevy::{camera::Projection, mesh::VertexAttributeValues, prelude::*};

use crate::camera::TerrainCamera;
use crate::{cli::RunConfig, spatial::TerrainSpatialFrame};
pub use data::TerrainDataset;
pub(crate) use mesh::rendered_height;
use mesh::{build_terrain_detail_normal, build_tile_mesh, collapse_positions};
use quadtree::{EdgeStitch, TileKey, edge_stitch, select_leaves};
pub(crate) use surface::canopy_coverage;

const STREAM_INTERVAL_S: f32 = 0.10;
const MAX_TILE_BUILDS_PER_TICK: usize = 36;
const MORPH_DURATION_S: f32 = 0.32;
const LEVEL_RENDER_LIFT_M: f32 = 0.002;

pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/terrain/curacao");
        let dataset = TerrainDataset::load(&asset_dir).unwrap_or_else(|error| {
            panic!(
                "failed to load baked Curaçao terrain from {}: {error:#}\nrun `cargo run -p korsou_terrain_baker -- …` first",
                asset_dir.display()
            )
        });
        let spatial =
            TerrainSpatialFrame::new(&dataset, app.world().resource::<RunConfig>().spatial)
                .expect("validated Curaçao metadata must construct its selected spatial adapter");
        app.insert_resource(dataset)
            .insert_resource(spatial)
            .init_resource::<TerrainStreamState>()
            .init_resource::<TerrainStats>()
            .add_systems(Startup, setup_terrain_material)
            .add_systems(Update, (stream_terrain, update_morphs).chain());
    }
}

#[derive(Resource)]
struct TerrainMaterial(Handle<StandardMaterial>);

fn setup_terrain_material(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let detail_normal = build_terrain_detail_normal();
    commands.insert_resource(TerrainMaterial(materials.add(StandardMaterial {
        base_color: Color::WHITE,
        // Terrain colour is already nonrepeating vertex data. A tiled albedo
        // aliases beyond the sampler's anisotropic limit in low grazing views.
        normal_map_texture: Some(images.add(detail_normal)),
        perceptual_roughness: 0.98,
        reflectance: 0.16,
        cull_mode: None,
        ..default()
    })));
}

#[derive(Component)]
pub struct TerrainChunk;

struct ResidentTile {
    entity: Entity,
    mesh: Handle<Mesh>,
    stitch: EdgeStitch,
    high_positions: Vec<[f32; 3]>,
    low_positions: Vec<[f32; 3]>,
    source_positions_m: Vec<[f64; 2]>,
    collapse_level: u8,
    morph: f32,
    target_morph: f32,
    triangles: usize,
}

#[derive(Resource)]
struct TerrainStreamState {
    resident: HashMap<TileKey, ResidentTile>,
    desired: HashSet<TileKey>,
    timer: Timer,
    first_tick: bool,
}

impl Default for TerrainStreamState {
    fn default() -> Self {
        Self {
            resident: HashMap::new(),
            desired: HashSet::new(),
            timer: Timer::new(
                Duration::from_secs_f32(STREAM_INTERVAL_S),
                TimerMode::Repeating,
            ),
            first_tick: true,
        }
    }
}

#[derive(Resource, Default)]
pub struct TerrainStats {
    pub resident: usize,
    pub desired: usize,
    pub queued: usize,
    pub by_lod: [usize; 7],
    pub triangles: usize,
    pub dense_triangles: usize,
    pub transitioning: usize,
}

#[allow(clippy::too_many_arguments)]
fn stream_terrain(
    mut commands: Commands,
    time: Res<Time>,
    dataset: Res<TerrainDataset>,
    spatial: Res<TerrainSpatialFrame>,
    material: Option<Res<TerrainMaterial>>,
    camera: Single<(&TerrainCamera, &Projection, &Camera)>,
    mut state: ResMut<TerrainStreamState>,
    mut stats: ResMut<TerrainStats>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let ready = state.first_tick || state.timer.tick(time.delta()).just_finished();
    if !ready {
        return;
    }
    let Some(viewport_height) = camera.2.physical_viewport_size().map(|size| size.y) else {
        return;
    };
    state.first_tick = false;
    let Some(material) = material else {
        return;
    };
    let focal_length_px = match camera.1 {
        Projection::Perspective(projection) => {
            0.5 * f64::from(viewport_height) / (f64::from(projection.fov) * 0.5).tan()
        }
        _ => f64::from(viewport_height),
    };
    let desired = select_leaves(
        &dataset,
        camera.0.position_m,
        focal_length_px,
        &state.desired,
    );

    let mut actions = Vec::new();
    for key in desired.iter().copied() {
        let stitch = edge_stitch(key, &desired);
        if state
            .resident
            .get(&key)
            .is_none_or(|resident| resident.stitch != stitch)
        {
            let bounds = dataset.quadtree_bounds(key.level, key.x, key.z);
            let distance = distance_to_square(
                camera.0.position_m.x,
                camera.0.position_m.z,
                bounds[0],
                bounds[1],
                bounds[2] - bounds[0],
            );
            actions.push((distance, key, stitch));
        }
    }
    actions.sort_by(|a, b| a.0.total_cmp(&b.0));
    stats.queued = actions.len();

    for (_, key, stitch) in actions.into_iter().take(MAX_TILE_BUILDS_PER_TICK) {
        let mut built = build_tile_mesh(&dataset, &spatial, key, stitch);
        if let Some(resident) = state.resident.get_mut(&key) {
            apply_morph_to_mesh(
                &mut built.mesh,
                &built.high_positions,
                &built.parent_positions,
                resident.morph,
            );
            let new_mesh = meshes.add(built.mesh);
            commands
                .entity(resident.entity)
                .insert(Mesh3d(new_mesh.clone()));
            meshes.remove(resident.mesh.id());
            resident.mesh = new_mesh;
            resident.stitch = stitch;
            resident.high_positions = built.high_positions;
            resident.low_positions = built.parent_positions;
            resident.source_positions_m = built.source_positions_m;
            resident.collapse_level = key.level.saturating_sub(1);
            resident.triangles = built.triangles;
            continue;
        }

        let ancestor_level = nearest_resident_ancestor(key, &state.resident).map(|key| key.level);
        let morph = if ancestor_level.is_some() { 0.0 } else { 1.0 };
        let collapse_level = ancestor_level.unwrap_or_else(|| key.level.saturating_sub(1));
        let low_positions = if let Some(level) = ancestor_level {
            collapse_positions(&dataset, &spatial, key, &built.source_positions_m, level)
        } else {
            built.parent_positions
        };
        apply_morph_to_mesh(
            &mut built.mesh,
            &built.high_positions,
            &low_positions,
            morph,
        );
        let mesh = meshes.add(built.mesh);
        let mut transform = Transform::from_translation(built.origin_render_m.as_vec3());
        transform.translation.y += f32::from(key.level) * LEVEL_RENDER_LIFT_M;
        let entity = commands
            .spawn((
                TerrainChunk,
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.0.clone()),
                transform,
                Name::new(format!(
                    "RTIN terrain L{} {},{} ({} tris)",
                    key.level, key.x, key.z, built.triangles
                )),
            ))
            .id();
        state.resident.insert(
            key,
            ResidentTile {
                entity,
                mesh,
                stitch,
                high_positions: built.high_positions,
                low_positions,
                source_positions_m: built.source_positions_m,
                collapse_level,
                morph,
                target_morph: 1.0,
                triangles: built.triangles,
            },
        );
    }

    state.desired = desired;
    update_transition_targets(&dataset, &spatial, &mut state);
    remove_replaced_tiles(&dataset, &mut commands, &mut meshes, &mut state);
    update_stats(&dataset, &state, &mut stats);
}

fn update_transition_targets(
    dataset: &TerrainDataset,
    spatial: &TerrainSpatialFrame,
    state: &mut TerrainStreamState,
) {
    let resident_keys: Vec<_> = state.resident.keys().copied().collect();
    for key in resident_keys {
        if state.desired.contains(&key) {
            state.resident.get_mut(&key).unwrap().target_morph = 1.0;
            continue;
        }
        let desired_ancestor = state
            .desired
            .iter()
            .copied()
            .filter(|desired| desired.is_ancestor_of(key))
            .max_by_key(|desired| desired.level)
            .filter(|desired| state.resident.contains_key(desired));
        let Some(ancestor) = desired_ancestor else {
            state.resident.get_mut(&key).unwrap().target_morph = 1.0;
            continue;
        };
        let resident = state.resident.get_mut(&key).unwrap();
        if resident.collapse_level != ancestor.level {
            resident.low_positions = collapse_positions(
                dataset,
                spatial,
                key,
                &resident.source_positions_m,
                ancestor.level,
            );
            resident.collapse_level = ancestor.level;
        }
        resident.target_morph = 0.0;
    }
}

fn update_morphs(
    time: Res<Time>,
    mut state: ResMut<TerrainStreamState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut stats: ResMut<TerrainStats>,
) {
    let step = time.delta_secs() / MORPH_DURATION_S;
    let mut transitioning = 0;
    for resident in state.resident.values_mut() {
        let previous = resident.morph;
        if resident.morph < resident.target_morph {
            resident.morph = (resident.morph + step).min(resident.target_morph);
        } else if resident.morph > resident.target_morph {
            resident.morph = (resident.morph - step).max(resident.target_morph);
        }
        if (resident.morph - resident.target_morph).abs() > f32::EPSILON {
            transitioning += 1;
        }
        if resident.morph != previous
            && let Some(mut mesh) = meshes.get_mut(&resident.mesh)
        {
            apply_morph_to_mesh(
                &mut mesh,
                &resident.high_positions,
                &resident.low_positions,
                resident.morph,
            );
        }
    }
    stats.transitioning = transitioning;
}

fn remove_replaced_tiles(
    dataset: &TerrainDataset,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    state: &mut TerrainStreamState,
) {
    let stale: Vec<_> = state
        .resident
        .keys()
        .copied()
        .filter(|key| !state.desired.contains(key))
        .collect();
    let mut removals = Vec::new();
    for key in stale {
        let resident = &state.resident[&key];
        let merged = resident.target_morph == 0.0 && resident.morph == 0.0;
        let split = covered_by_ready_desired(dataset, key, &state.desired, &state.resident);
        if merged || split {
            removals.push(key);
        }
    }
    for key in removals {
        if let Some(resident) = state.resident.remove(&key) {
            meshes.remove(resident.mesh.id());
            commands.entity(resident.entity).despawn();
        }
    }
}

fn covered_by_ready_desired(
    dataset: &TerrainDataset,
    key: TileKey,
    desired: &HashSet<TileKey>,
    resident: &HashMap<TileKey, ResidentTile>,
) -> bool {
    if desired.contains(&key) {
        return resident.get(&key).is_some_and(|tile| tile.morph >= 1.0);
    }
    if key.level >= dataset.metadata.quadtree.visual_max_level {
        return false;
    }
    key.children().into_iter().all(|child| {
        if dataset.quadtree_coverage(child.level, child.x, child.z) & data::COVERAGE_LAND == 0 {
            true
        } else {
            covered_by_ready_desired(dataset, child, desired, resident)
        }
    })
}

fn nearest_resident_ancestor(
    mut key: TileKey,
    resident: &HashMap<TileKey, ResidentTile>,
) -> Option<TileKey> {
    while let Some(parent) = key.parent() {
        if resident.contains_key(&parent) {
            return Some(parent);
        }
        key = parent;
    }
    None
}

fn apply_morph_to_mesh(mesh: &mut Mesh, high: &[[f32; 3]], low: &[[f32; 3]], morph: f32) {
    debug_assert_eq!(high.len(), low.len());
    let positions: Vec<[f32; 3]> = high
        .iter()
        .zip(low)
        .map(|(high, low)| [high[0], low[1] + (high[1] - low[1]) * morph, high[2]])
        .collect();
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(positions),
    );
}

fn update_stats(dataset: &TerrainDataset, state: &TerrainStreamState, stats: &mut TerrainStats) {
    stats.resident = state.resident.len();
    stats.desired = state.desired.len();
    stats.by_lod = [0; 7];
    stats.triangles = 0;
    for (key, resident) in &state.resident {
        stats.by_lod[key.level.min(6) as usize] += 1;
        stats.triangles += resident.triangles;
    }
    stats.dense_triangles = state
        .resident
        .keys()
        .map(|key| {
            let cells = mesh::tile_cells_for_level(dataset, key.level);
            cells * cells * 2
        })
        .sum();
}

fn distance_to_square(point_x: f64, point_z: f64, min_x: f64, min_z: f64, size: f64) -> f64 {
    let dx = if point_x < min_x {
        min_x - point_x
    } else if point_x > min_x + size {
        point_x - (min_x + size)
    } else {
        0.0
    };
    let dz = if point_z < min_z {
        min_z - point_z
    } else if point_z > min_z + size {
        point_z - (min_z + size)
    } else {
        0.0
    };
    dx.hypot(dz)
}
