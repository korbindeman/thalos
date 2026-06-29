//! Map-view UDLOD terrain (Slice 6, "Approach A").
//!
//! The orbital map renders bodies as flat billboards on `MAP_LAYER` at
//! [`MAP_SCALE`] (1 render unit = 1000 km), outside the ship-view BigSpace.
//! UDLOD terrain, however, is *true-metre* geometry whose vertex precision
//! path is unconditionally `big_space` — a view must be a `FloatingOrigin`
//! camera and the terrain must hang under a `Grid`. This module bridges the
//! two so that zooming a body on the map shows real terrain instead of a
//! billboard, while reusing the *same* `BodyTerrainMaterial` / shaders as the
//! ship view (no `.wgsl` edits).
//!
//! # How the two coordinate systems are reconciled — a *non*-`BigSpace` grid
//!
//! UDLOD only needs two things from `big_space` to render: the *view* must
//! carry a `GridTransform` (`CellCoord` + `Transform`), and the terrain must
//! have a `Grid` ancestor (`Grids::parent_grid` just looks for a `Grid`
//! component — **no `BigSpace` required**). It does **not** need a
//! `FloatingOrigin`. So this module deliberately gives the map terrain a plain
//! `Grid` root (`MapGridRoot`) that is **not** a `BigSpace`:
//!
//! - The ship view's nine `Query<&Grid, With<BigSpace>>::single()` call sites
//!   (incl. the foliage track's `grass.rs`/`vegetation.rs`) never match it, so
//!   there are **zero collisions** — a second `BigSpace` would have broken every
//!   one of those `.single()`s.
//! - With no `BigSpace`, big_space's precision propagation (gated on
//!   `With<BigSpace>`) skips the map subtree; its `GlobalTransform`s come from
//!   the ordinary Bevy-style `propagate_parent_transforms` (which big_space
//!   re-adds and which has no `CellCoord` filter). With every map cell pinned to
//!   `(0,0,0)`, that yields `GlobalTransform == Transform == origin-relative map
//!   position` — exactly the space the billboards live in
//!   ([`super::transforms::update_body_positions`]). One consistent frame, one
//!   camera, terrain + billboards together.
//! - Precision is fine because the map already keeps the **focused** body near
//!   the render origin (`RenderOrigin`), and terrain is only ever spawned for
//!   the focused body, so its coordinates are small.
//! - The existing [`MapCamera`] just gets a `CellCoord::ZERO` added (so UDLOD's
//!   view query resolves) — no `FloatingOrigin`, no reparenting, **no
//!   `camera.rs` edit**. Its `Transform`, written each frame by
//!   `camera_transform_system`, is its origin-relative map position; with cell
//!   `ZERO` that is also its `GlobalTransform`.
//!
//! # Scope
//!
//! Only the **focused** body gets map terrain, and only while it is zoomed in
//! past the icon dot (same screen-size rule as the ship view). It uses the
//! cheap `Distant` atlas tier (~20 MB); since UDLOD cannot share one atlas
//! across two scales, this is a *second* atlas that re-streams on focus change.

use std::sync::Arc;

use bevy::camera::visibility::RenderLayers;
use bevy::light::NotShadowCaster;
use bevy::math::DVec3;
use bevy::prelude::*;
use big_space::grid::Grid;
use big_space::prelude::CellCoord;
use thalos_body_render::udlod::math::TerrainModel;
use thalos_body_render::udlod::prelude::{
    TerrainBundle, TerrainViewComponents, TileAtlas, TileProvider, TileTree,
};
use thalos_body_render::{
    AtmosphereBlock, BodyTerrainDebug, BodyTerrainExtras, BodyTerrainMaterial, BodyTerrainShadow,
    CASCADE_COUNT, PipelineTileProvider, SceneLighting, rendered_height_range,
};
use thalos_terrain::{ProceduralSurface, SurfaceQuery};
use thalos_world::BodyId;

use super::SCREEN_MARKER_RADIUS;
use super::ground_terrain::{
    TerrainTier, body_terrain_view_config, build_terrain_config, build_terrain_scene_lighting,
    pin_root_tiles, terrain_shading_style_for,
};
use super::transforms::surface_body_to_world_orientation_f64;
use super::types::{
    BodyMesh, CameraExposure, CelestialBody, SimulationState, SolarSystemState, TidallyLocked,
};
use crate::camera::{CameraFocus, CameraFocusTarget, MapCamera};
use crate::coords::{MAP_LAYER, MAP_SCALE};
use crate::view::ViewMode;

/// Cell size of the map terrain's (non-`BigSpace`) grid, in MAP_SCALE render
/// units.
///
/// Every map entity's cell is pinned to `(0,0,0)`, so UDLOD's
/// `position_double = cell * cell_size + translation` reduces to `translation`
/// and the cell size is effectively unused — it is kept enormous purely so any
/// stray position still lands in cell zero. (The whole system spans ~1e7 units
/// origin-relative; 1e9 is far beyond that.)
pub const MAP_TERRAIN_CELL_SIZE: f32 = 1.0e9;

/// The dedicated (non-`BigSpace`) `Grid` root that parents map-view terrain, so
/// `Grids::parent_grid` resolves a `Grid` for it without dragging in a second
/// `BigSpace`.
#[derive(Resource, Debug, Clone, Copy)]
struct MapGridRoot {
    entity: Entity,
}

/// Marker on the single map-view terrain entity, tagged with the body it
/// currently renders.
#[derive(Component, Debug)]
struct MapBodyTerrain {
    body_id: BodyId,
}

/// Tracks the single resident map-terrain entity and which body it renders, so
/// the planner can re-target it when the camera focus changes.
#[derive(Resource, Default)]
struct MapTerrainState {
    entity: Option<Entity>,
    body_id: Option<BodyId>,
}

pub struct MapTerrainPlugin;

impl Plugin for MapTerrainPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MapTerrainState>()
            .add_systems(
                Startup,
                (
                    setup_map_grid,
                    attach_map_camera
                        .after(setup_map_grid)
                        .after(crate::camera::spawn_camera),
                ),
            )
            .add_systems(
                Update,
                (
                    attach_map_camera,
                    manage_map_terrain,
                    update_map_terrain.after(manage_map_terrain),
                    hide_focused_map_impostor
                        .after(update_map_terrain)
                        .after(super::body_lod::sync_body_icons),
                )
                    .in_set(crate::SimStage::Sync),
            );
    }
}

/// Spawn the map terrain's dedicated plain `Grid` root (deliberately **not** a
/// `BigSpace` — see the module docs). `Transform` + `Visibility` pull in their
/// required components (`GlobalTransform`, `InheritedVisibility`,
/// `ViewVisibility`) so it is a valid render-hierarchy root.
fn setup_map_grid(mut commands: Commands) {
    let entity = commands
        .spawn((
            Grid::new(MAP_TERRAIN_CELL_SIZE, 0.0),
            Transform::default(),
            Visibility::Inherited,
            Name::new("Map Terrain Grid"),
        ))
        .id();
    commands.insert_resource(MapGridRoot { entity });
}

/// Give the [`MapCamera`] a `CellCoord::ZERO` so UDLOD's per-view query
/// (`GridTransformReadOnly`) resolves for it. Additive: no `FloatingOrigin`, no
/// reparenting, no `Transform` change — `camera_transform_system` keeps writing
/// the camera's origin-relative map position, which with cell `ZERO` is also its
/// `GlobalTransform`. Runs at Startup and once more in `Update` to cover any
/// camera respawn; the `Without<CellCoord>` filter makes it a no-op once done.
fn attach_map_camera(
    mut commands: Commands,
    cameras: Query<Entity, (With<MapCamera>, Without<CellCoord>)>,
) {
    for entity in &cameras {
        commands.entity(entity).insert(CellCoord::ZERO);
    }
}

/// The body the map should render terrain for this frame: the camera-focus body
/// when it is a procedural-terrain body zoomed in past the icon dot. `None`
/// outside map view, when nothing is focused, or when the body is still a dot.
fn wanted_map_body(view: &ViewMode, focus: &CameraFocus, sim: &SimulationState) -> Option<BodyId> {
    if *view != ViewMode::Map {
        return None;
    }
    let CameraFocusTarget::Body(body_id) = focus.target else {
        return None;
    };
    let body = sim.system.bodies.get(body_id)?;
    if !body.terrain.is_some() {
        return None;
    }
    // Same screen-size rule as the ship view: terrain replaces the billboard
    // once the body's rendered radius exceeds the icon dot. The MAP_SCALE
    // cancels (`radius_m * s > distance * s * SCREEN_MARKER_RADIUS`), so the
    // threshold is the metre orbit distance directly.
    let swap_distance = body.radius_m / SCREEN_MARKER_RADIUS as f64;
    (focus.distance < swap_distance).then_some(body_id)
}

/// SystemParam-free spawn/despawn planner: spawns one map terrain entity for the
/// focused body and despawns/re-targets it when the focus changes.
#[allow(clippy::too_many_arguments)]
fn manage_map_terrain(
    mut commands: Commands,
    view: Res<ViewMode>,
    focus: Res<CameraFocus>,
    sim: Res<SimulationState>,
    root: Option<Res<MapGridRoot>>,
    map_camera_q: Query<Entity, (With<MapCamera>, With<CellCoord>)>,
    mut materials: ResMut<Assets<BodyTerrainMaterial>>,
    mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
    mut state: ResMut<MapTerrainState>,
    sun_shadow: Res<super::sun_shadow::SunShadowImage>,
) {
    let wanted = wanted_map_body(&view, &focus, &sim);
    if wanted == state.body_id {
        return;
    }

    let Some(root) = root else {
        return;
    };
    // The map camera must already carry its `CellCoord` (attached at Startup)
    // before we hand UDLOD a TileTree keyed on it — `compute_requests` unwraps
    // the view's GridTransform. The `With<CellCoord>` filter guarantees it.
    let Ok(map_camera) = map_camera_q.single() else {
        return;
    };

    // Despawn / unregister the previous map terrain, if any.
    if let Some(entity) = state.entity.take() {
        tile_trees.remove(&(entity, map_camera));
        commands.entity(entity).despawn();
    }
    state.body_id = None;

    // Spawn the new one.
    if let Some(body_id) = wanted
        && let Some(entity) = spawn_map_terrain(
            body_id,
            &sim,
            root.entity,
            map_camera,
            &mut materials,
            &mut tile_trees,
            &mut commands,
            sun_shadow.handles.clone(),
        )
    {
        state.entity = Some(entity);
        state.body_id = Some(body_id);
    }
}

/// Spawn one map-scale UDLOD terrain entity for `body_id`, parented under the
/// map `Grid` root and registered as a view of the map camera. Returns `None`
/// if the body has no authored terrain.
#[allow(clippy::too_many_arguments)]
fn spawn_map_terrain(
    body_id: BodyId,
    sim: &SimulationState,
    map_root: Entity,
    map_camera: Entity,
    materials: &mut Assets<BodyTerrainMaterial>,
    tile_trees: &mut TerrainViewComponents<TileTree>,
    commands: &mut Commands,
    sun_shadow_maps: [Handle<Image>; CASCADE_COUNT],
) -> Option<Entity> {
    let body = sim.system.bodies.get(body_id)?;
    if !body.terrain.is_some() {
        return None;
    }

    // Same procedural surface as the ship-view terrain — but no flatten layer
    // (the runway pad is a ship-surface concern). The provider returns
    // normalized heights, so the same surface drives any scale; only the
    // `TerrainModel` min/max (set below in MAP_SCALE units) decode it.
    let surface: Arc<dyn SurfaceQuery> =
        Arc::new(ProceduralSurface::new(body.radius_m as f32, body.id as u32));
    let height_range_m = rendered_height_range(surface.as_ref());

    // MAP_SCALE geometry: radius and height range scaled to render units.
    let scale = MAP_SCALE as f32;
    let map_radius = body.radius_m * MAP_SCALE;
    let map_height_range = height_range_m * scale;
    let model = TerrainModel::sphere(DVec3::ZERO, map_radius, -map_height_range, map_height_range);

    let config = build_terrain_config(model, TerrainTier::Map);
    let provider: Box<dyn TileProvider> =
        Box::new(PipelineTileProvider::new(body.name.clone(), surface));
    let mut tile_atlas = TileAtlas::with_provider(&config, provider);
    pin_root_tiles(&mut tile_atlas);

    let mut view_config = body_terrain_view_config(map_radius);
    // The map renders the whole body from outside at MAP_SCALE, where f32 is
    // amply precise. UDLOD's high-precision Taylor path is only valid for
    // vertices near the view anchor on the near cube face — forced across the
    // full visible sphere (as it is here, since `body_terrain_view_config`'s
    // metre-tuned threshold becomes ~1e10 m once re-scaled into MAP_SCALE
    // units) it diverges and splays the silhouette/back faces into spikes.
    // Pin it to 0 so every map vertex takes the exact `position_local_to_world`
    // model-transform path (clean sphere).
    view_config.precision_threshold_distance = 0.0;
    let tile_tree = TileTree::new(&tile_atlas, &view_config);

    let map_grid = Grid::new(MAP_TERRAIN_CELL_SIZE, 0.0);
    let mut bundle = TerrainBundle::new(tile_atlas, &map_grid);
    // Hidden until the pinned root tiles stream in (avoids a one-frame void
    // sphere); `update_map_terrain` flips it on.
    bundle.visibility = Visibility::Hidden;

    // Ship-view atmosphere is at SHIP_SCALE; the map uses 1 unit = 1/MAP_SCALE
    // metres, mirroring how the impostor builds its map vs ship blocks.
    let atmosphere = body
        .terrestrial_atmosphere
        .as_ref()
        .map(|a| AtmosphereBlock::from_terrestrial(a, (1.0 / MAP_SCALE) as f32))
        .unwrap_or_default();
    let shading_style = terrain_shading_style_for(body);
    let material = BodyTerrainMaterial {
        atmosphere,
        scene: SceneLighting::default(),
        extras: BodyTerrainExtras {
            craft_shadow: BodyTerrainShadow::default(),
            debug: BodyTerrainDebug::default(),
            // z = 1.0 flags "distant schematic" so the terrain shader renders the
            // specular matte — at map distance the undersampled baked normal makes
            // the GGX highlight alias into a crawling gleam (see `body_terrain.wgsl`).
            inspection: Vec4::new(0.0, shading_style.shader_flag(), 1.0, 0.0),
            // Map terrain never samples the sun-shadow maps (config.x stays 0).
            ..Default::default()
        },
        sun_shadow_map_0: sun_shadow_maps[0].clone(),
        sun_shadow_map_1: sun_shadow_maps[1].clone(),
        sun_shadow_map_2: sun_shadow_maps[2].clone(),
    };

    let entity = commands
        .spawn((
            bundle,
            MeshMaterial3d(materials.add(material)),
            RenderLayers::layer(MAP_LAYER),
            NotShadowCaster,
            ChildOf(map_root),
            Name::new(format!("{} Map Terrain", body.name)),
            MapBodyTerrain { body_id },
        ))
        .id();

    tile_trees.insert((entity, map_camera), tile_tree);

    info!(
        "spawned map terrain for '{}' (map radius {:.3} units, MAP_SCALE)",
        body.name, map_radius
    );

    Some(entity)
}

/// Per-frame pose + lighting for the resident map terrain. Positions it to
/// coincide with the focused body's billboard (read straight off the
/// `CelestialBody` transform — already origin-relative at MAP_SCALE), spins it
/// to the body's surface orientation, refreshes its scene lighting, and reveals
/// it once the pinned root tiles have streamed.
#[allow(clippy::type_complexity)]
fn update_map_terrain(
    cache: Res<SolarSystemState>,
    exposure: Res<CameraExposure>,
    bodies: Query<(&CelestialBody, &Transform, Option<&TidallyLocked>), Without<MapBodyTerrain>>,
    mut terrain_q: Query<
        (
            &MapBodyTerrain,
            &mut Transform,
            &mut CellCoord,
            &mut Visibility,
            &TileAtlas,
            &MeshMaterial3d<BodyTerrainMaterial>,
        ),
        Without<CelestialBody>,
    >,
    mut materials: ResMut<Assets<BodyTerrainMaterial>>,
) {
    let Some(states) = cache.states.as_ref() else {
        return;
    };
    // The single resident `MapBodyTerrain` entity (one focused body at a time).
    let Ok((terrain, mut transform, mut cell, mut vis, atlas, mat_handle)) = terrain_q.single_mut()
    else {
        return;
    };

    // Position: coincide with the focused body's billboard so the terrain
    // sphere sits exactly where the map expects the body. The `CelestialBody`
    // transform is already the origin-relative MAP_SCALE position.
    let Some((_, body_tf, lock)) = bodies.iter().find(|(b, _, _)| b.body_id == terrain.body_id)
    else {
        return;
    };
    *cell = CellCoord::ZERO;
    transform.translation = body_tf.translation;
    let orientation = surface_body_to_world_orientation_f64(terrain.body_id, lock, states)
        .map(|q| q.as_quat())
        .unwrap_or(Quat::IDENTITY);
    transform.rotation = orientation;

    // Scene lighting: star direction + flux are scale-invariant; skip eclipse
    // occluders on the map schematic (empty list).
    if let Some(mat) = materials.get_mut(mat_handle) {
        mat.scene = build_terrain_scene_lighting(terrain.body_id, states, &[], exposure.gain);
    }

    // Reveal once the pinned root tiles are ready (no void-sphere flash).
    let want = if atlas.pinned_tiles_ready() {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    if *vis != want {
        *vis = want;
    }
}

/// Hide the focused body's map billboard while its map terrain is visible, so
/// the flat disc doesn't z-fight / show through the terrain sphere. Runs after
/// `sync_body_icons` (which otherwise shows the billboard at this zoom);
/// restoring it is automatic — once the map terrain despawns this no-ops and
/// `sync_body_icons` shows the billboard again next frame.
#[allow(clippy::type_complexity)]
fn hide_focused_map_impostor(
    state: Res<MapTerrainState>,
    terrain_q: Query<(&MapBodyTerrain, &TileAtlas)>,
    bodies: Query<(&CelestialBody, &Children)>,
    mut billboards: Query<&mut Visibility, (With<BodyMesh>, Without<CelestialBody>)>,
) {
    let Some(active_body) = state.body_id else {
        return;
    };
    // Only hide once the terrain is actually drawable, else the body vanishes
    // during the cold tile stream.
    let terrain_ready = terrain_q
        .iter()
        .any(|(t, atlas)| t.body_id == active_body && atlas.pinned_tiles_ready());
    if !terrain_ready {
        return;
    }
    let Some((_, children)) = bodies.iter().find(|(b, _)| b.body_id == active_body) else {
        return;
    };
    for child in children.iter() {
        if let Ok(mut vis) = billboards.get_mut(child)
            && *vis != Visibility::Hidden
        {
            *vis = Visibility::Hidden;
        }
    }
}
