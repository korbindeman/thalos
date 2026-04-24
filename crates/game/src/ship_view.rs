//! Player ship rendering — loads `ships/apollo.ron` on startup, spawns its
//! parts as children of a [`PlayerShip`] root, and keeps the root's world
//! position in sync with the physics ship state each frame.
//!
//! The root carries `Transform::from_scale(RENDER_SCALE)` so the parts,
//! authored in meters, compose into the solar-system-wide render-units
//! coordinate space at their real physical size. Part meshes and the
//! [`ShipPartMaterial`] uniforms are rebuilt whenever `AttachNodes`
//! changes and kept in sync via ported versions of the editor's systems.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::Mesh;
use bevy::prelude::*;
use thalos_ship_rendering::{
    ShipPartExtension, ShipPartMaterial, ShipPartParams, ShipRenderingPlugin, stainless_steel_base,
};
use thalos_shipyard::{
    Adapter, AttachNodes, Attachment, CommandPod, Decoupler, Engine, FuelTank, Part, PartMaterial,
    Ship, ShipBlueprint, ShipyardPlugin,
};

use crate::SimStage;
use crate::camera::CameraFocus;
use crate::rendering::{
    PlayerShip, RENDER_SCALE, RenderOrigin, SimulationState, to_render_pos,
};
use crate::view::{HideInMapView, ViewMode};

/// Radial segments for cylinder / frustum part meshes. Matches the ship
/// editor's value so the two look identical side-by-side.
const PART_RESOLUTION: u32 = 128;

/// Initial orbital distance (metres) when switching into ship view. The
/// camera snaps to this distance — close enough that a ~10 m ship fills
/// a reasonable fraction of the screen.
const SHIP_VIEW_INITIAL_DISTANCE_M: f64 = 30.0;

pub struct ShipViewPlugin;

impl Plugin for ShipViewPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ShipyardPlugin)
            .add_plugins(ShipRenderingPlugin)
            .add_systems(Startup, spawn_player_ship)
            .add_systems(
                Update,
                (
                    rebuild_ship_visuals,
                    update_ship_part_transforms.after(rebuild_ship_visuals),
                    update_ship_part_shader_params.after(rebuild_ship_visuals),
                    sync_view_mode_changed
                        .run_if(resource_changed::<ViewMode>)
                        .before(crate::SimStage::Physics),
                    update_player_ship_world_position
                        .in_set(SimStage::Sync)
                        .after(crate::rendering::update_render_origin),
                ),
            );
    }
}

#[derive(Component)]
struct PartVisual;

#[derive(Component, Clone)]
struct PartShaderHandle(Handle<ShipPartMaterial>);

/// Resolved on startup so systems can target the player's ship without
/// having to re-find it each frame.
#[derive(Resource)]
pub struct PlayerShipEntity(pub Entity);

fn spawn_player_ship(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    view: Res<ViewMode>,
    sim: Res<SimulationState>,
) {
    let ron_path = PathBuf::from("ships/apollo.ron");
    let text = match std::fs::read_to_string(&ron_path) {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to read {}: {}", ron_path.display(), e);
            return;
        }
    };
    let blueprint = match ShipBlueprint::from_ron(&text) {
        Ok(bp) => bp,
        Err(e) => {
            error!("Failed to parse {}: {}", ron_path.display(), e);
            return;
        }
    };

    let ship_entity = blueprint.spawn(&mut commands);
    info!(
        "spawned ship blueprint '{}' with {} parts",
        blueprint.name,
        blueprint.parts.len(),
    );

    let initial_pos = to_render_pos(sim.simulation.ship_state().position);
    // Visibility is driven by `apply_view_mode_visibility` via the
    // [`HideInMapView`] tag — start at whatever the tag implies for the
    // current view.
    let initial_visibility = match *view {
        ViewMode::Map => Visibility::Hidden,
        ViewMode::Ship => Visibility::Inherited,
    };

    let player_ship = commands
        .spawn((
            PlayerShip,
            HideInMapView,
            Transform {
                translation: initial_pos,
                rotation: Quat::IDENTITY,
                scale: Vec3::splat(RENDER_SCALE as f32),
            },
            initial_visibility,
            Name::new("PlayerShip"),
        ))
        .id();

    // DEBUG: bright emissive cube child at origin so we can see where
    // PlayerShip renders even if the ship meshes don't.
    let debug_cube = meshes.add(Cuboid::new(20.0, 20.0, 20.0));
    let debug_mat = mats.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.0, 1.0),
        emissive: LinearRgba::new(5.0, 0.0, 5.0, 1.0),
        unlit: true,
        ..default()
    });
    commands.spawn((
        Mesh3d(debug_cube),
        MeshMaterial3d(debug_mat),
        Transform::from_xyz(0.0, 20.0, 0.0),
        NoFrustumCulling,
        ChildOf(player_ship),
        Name::new("DEBUG PlayerShip marker"),
    ));

    commands.insert_resource(PlayerShipEntity(player_ship));

    // Reparent all parts owned by this ship into the PlayerShip hierarchy
    // so they inherit its scale + translation. Runs as a deferred command
    // so the `Ship` component is committed first.
    commands.queue(move |world: &mut World| {
        let root = world
            .get::<Ship>(ship_entity)
            .map(|s| s.root)
            .unwrap_or(ship_entity);
        let mut attachments: HashMap<Entity, Vec<Entity>> = HashMap::new();
        let mut att_query = world.query::<(Entity, &Attachment)>();
        for (e, att) in att_query.iter(world) {
            attachments.entry(att.parent).or_default().push(e);
        }
        let mut queue: VecDeque<Entity> = VecDeque::from([root]);
        let mut to_reparent: Vec<Entity> = Vec::new();
        while let Some(e) = queue.pop_front() {
            to_reparent.push(e);
            if let Some(kids) = attachments.get(&e) {
                queue.extend(kids.iter().copied());
            }
        }
        let count = to_reparent.len();
        for part in &to_reparent {
            world.entity_mut(player_ship).add_child(*part);
        }
        // Verify the hierarchy actually took — log the parent we read back.
        let mut actually_reparented = 0;
        for part in &to_reparent {
            if let Some(child_of) = world.get::<ChildOf>(*part)
                && child_of.0 == player_ship
            {
                actually_reparented += 1;
            }
        }
        info!(
            "reparented {} ship parts under PlayerShip (verified {})",
            count, actually_reparented
        );
    });
}

/// Sync the [`PlayerShip`] root's translation to the physics ship state each
/// frame, applying the same render-origin shift as celestial bodies.
fn update_player_ship_world_position(
    sim: Res<SimulationState>,
    origin: Res<RenderOrigin>,
    view: Res<ViewMode>,
    mut query: Query<&mut Transform, With<PlayerShip>>,
    mut frame: Local<u32>,
) {
    let Ok(mut transform) = query.single_mut() else {
        return;
    };
    transform.translation =
        to_render_pos(sim.simulation.ship_state().position - origin.position);
    *frame += 1;
    // Log on view change + for ~5 frames afterward so we can see if origin
    // catches up across frames.
    if view.is_changed() || *frame % 60 == 1 {
        info!(
            "[f{}] {:?} PlayerShip.translation={:?} origin={:?} ship_pos={:?}",
            *frame,
            *view,
            transform.translation,
            origin.position,
            sim.simulation.ship_state().position,
        );
    }
}

/// React to [`ViewMode`] changes: switch the camera focus, show/hide the
/// 3D ship, and swap the camera projection.
fn sync_view_mode_changed(
    view: Res<ViewMode>,
    player_ship: Option<Res<PlayerShipEntity>>,
    mut focus: ResMut<CameraFocus>,
    mut projection: Query<&mut Projection, With<crate::camera::OrbitCamera>>,
    bodies: Query<(Entity, &Name), With<crate::rendering::CelestialBody>>,
) {
    let Some(player_ship) = player_ship else {
        return;
    };

    match *view {
        ViewMode::Ship => {
            info!(
                "entering ship view, focus target = {:?}, current dist = {}",
                player_ship.0, focus.distance
            );
            focus.target = Some(player_ship.0);
            focus.target_distance = SHIP_VIEW_INITIAL_DISTANCE_M;
            focus.distance = focus.distance.min(SHIP_VIEW_INITIAL_DISTANCE_M * 10.0);
            if let Ok(mut proj) = projection.single_mut() {
                *proj = Projection::Perspective(PerspectiveProjection {
                    // Near clip in render units: 1 render unit = 1 000 km,
                    // so 5e-7 = 50 cm — close enough to put the camera a
                    // couple of metres off the hull without clipping.
                    near: 5.0e-7,
                    // Far clip at ~1e10 m (~0.07 AU). Distant bodies past
                    // that will clip; acceptable MVP.
                    far: 1.0e4,
                    ..default()
                });
            }
        }
        ViewMode::Map => {
            if let Ok(mut proj) = projection.single_mut() {
                *proj = Projection::Perspective(PerspectiveProjection::default());
            }
            // Snap focus back to the homeworld so the map view isn't
            // centred on a point-sized ship.
            if let Some((entity, _)) = bodies.iter().find(|(_, n)| n.as_str() == "Thalos") {
                focus.target = Some(entity);
                focus.target_distance = 2.0e7;
                focus.distance = focus.distance.max(2.0e7);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Part mesh + material rebuild (ported from ship_editor)
// ---------------------------------------------------------------------------

struct VisualSpec {
    mesh: Mesh,
    height: f32,
}

fn visual_spec(
    nodes: &AttachNodes,
    pod: Option<&CommandPod>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    tank: Option<&FuelTank>,
    engine: Option<&Engine>,
) -> Option<VisualSpec> {
    if let Some(p) = pod {
        let d = p.diameter;
        let h = d * 0.9;
        Some(VisualSpec {
            mesh: ConicalFrustum {
                radius_top: d * 0.3,
                radius_bottom: d * 0.5,
                height: h,
            }
            .mesh()
            .resolution(PART_RESOLUTION)
            .into(),
            height: h,
        })
    } else if dec.is_some() {
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let h = 0.2;
        Some(VisualSpec {
            mesh: Cylinder::new(d * 0.5, h)
                .mesh()
                .resolution(PART_RESOLUTION)
                .into(),
            height: h,
        })
    } else if let Some(a) = adapter {
        let top_d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let bot_d = a.target_diameter;
        let h = ((top_d + bot_d) * 0.5).max(0.4);
        Some(VisualSpec {
            mesh: ConicalFrustum {
                radius_top: top_d * 0.5,
                radius_bottom: bot_d * 0.5,
                height: h,
            }
            .mesh()
            .resolution(PART_RESOLUTION)
            .into(),
            height: h,
        })
    } else if let Some(t) = tank {
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let h = t.length;
        Some(VisualSpec {
            mesh: Cylinder::new(d * 0.5, h)
                .mesh()
                .resolution(PART_RESOLUTION)
                .into(),
            height: h,
        })
    } else if let Some(e) = engine {
        let d = e.diameter;
        let (r_top, r_bot, h) = (d * 0.35, d * 0.5, d * 0.9);
        Some(VisualSpec {
            mesh: ConicalFrustum {
                radius_top: r_top,
                radius_bottom: r_bot,
                height: h,
            }
            .mesh()
            .resolution(PART_RESOLUTION)
            .into(),
            height: h,
        })
    } else {
        None
    }
}

fn ship_part_params(
    nodes: &AttachNodes,
    tank: Option<&FuelTank>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    seed: u32,
) -> ShipPartParams {
    let top_r = nodes.get("top").map(|n| n.diameter * 0.5).unwrap_or(0.5);
    let (radius_top, radius_bottom, length) = if let Some(t) = tank {
        (top_r, top_r, t.length)
    } else if dec.is_some() {
        (top_r, top_r, 0.2)
    } else if let Some(a) = adapter {
        let bot_r = a.target_diameter * 0.5;
        let h = (top_r + bot_r).max(0.4);
        let dr = top_r - bot_r;
        let slant = (h * h + dr * dr).sqrt();
        (top_r, bot_r, slant)
    } else {
        (top_r, top_r, 1.0)
    };
    ShipPartParams {
        length,
        radius_top,
        radius_bottom,
        seed,
        ..default()
    }
}

type VisualQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AttachNodes,
        Option<&'static CommandPod>,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Engine>,
        Option<&'static Children>,
        Option<&'static PartShaderHandle>,
        Has<PartMaterial>,
    ),
    Or<(Added<Part>, Changed<AttachNodes>)>,
>;

/// Spawn (or respawn) the body mesh child for each part whose attach
/// layout just changed. Parts with [`PartMaterial`] use [`ShipPartMaterial`]
/// for the procedural stainless finish; the remainder fall back to a plain
/// `StandardMaterial`.
fn rebuild_ship_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    parts: VisualQuery,
    stale: Query<(), With<PartVisual>>,
) {
    let rebuild_count = parts.iter().count();
    if rebuild_count > 0 {
        info!("rebuild_ship_visuals: processing {} parts", rebuild_count);
    }
    for (e, nodes, pod, dec, adapter, tank, engine, children, part_shader, has_part_mat) in
        parts.iter()
    {
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }

        let Some(spec) = visual_spec(nodes, pod, dec, adapter, tank, engine) else {
            continue;
        };
        let mesh = meshes.add(spec.mesh);

        let body_id = if has_part_mat {
            let params = ship_part_params(nodes, tank, dec, adapter, e.index_u32());
            let handle = match part_shader {
                Some(h) => h.0.clone(),
                None => {
                    let h = ship_materials.add(ShipPartMaterial {
                        base: stainless_steel_base(),
                        extension: ShipPartExtension { params },
                    });
                    commands.entity(e).insert(PartShaderHandle(h.clone()));
                    h
                }
            };
            commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(handle),
                    Transform::from_xyz(0.0, -spec.height * 0.5, 0.0),
                    Visibility::default(),
                    NoFrustumCulling,
                    PartVisual,
                ))
                .id()
        } else {
            // CommandPod and Engine have no `PartMaterial` yet: plain PBR
            // metal until they get their own shader. Enough to tell parts
            // apart visually.
            let mat = std_materials.add(StandardMaterial {
                base_color: Color::srgb(0.6, 0.62, 0.65),
                perceptual_roughness: 0.35,
                metallic: 0.85,
                ..default()
            });
            commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(mat),
                    Transform::from_xyz(0.0, -spec.height * 0.5, 0.0),
                    Visibility::default(),
                    NoFrustumCulling,
                    PartVisual,
                ))
                .id()
        };
        commands.entity(e).add_child(body_id);
    }
}

/// BFS from the ship root, positioning each part's local Transform based
/// on its [`Attachment`]. Copied from the ship editor.
fn update_ship_part_transforms(
    ships: Query<&Ship>,
    attachments: Query<(Entity, &Attachment)>,
    nodes: Query<&AttachNodes>,
    mut transforms: Query<&mut Transform, With<Part>>,
) {
    let mut children_map: HashMap<Entity, Vec<(Entity, Attachment)>> = HashMap::new();
    for (e, att) in attachments.iter() {
        children_map
            .entry(att.parent)
            .or_default()
            .push((e, att.clone()));
    }

    for ship in ships.iter() {
        if let Ok(mut t) = transforms.get_mut(ship.root) {
            t.translation = Vec3::ZERO;
            t.rotation = Quat::IDENTITY;
        }
        let mut queue: VecDeque<Entity> = VecDeque::from([ship.root]);
        while let Some(parent) = queue.pop_front() {
            let parent_pos = transforms
                .get(parent)
                .map(|t| t.translation)
                .unwrap_or(Vec3::ZERO);
            let Ok(parent_nodes) = nodes.get(parent) else {
                continue;
            };
            let parent_pos_and_nodes: Vec<(Entity, Vec3)> = children_map
                .get(&parent)
                .map(|kids| {
                    kids.iter()
                        .filter_map(|(c, att)| {
                            let pn = parent_nodes.get(&att.parent_node)?;
                            let child_offset = nodes
                                .get(*c)
                                .ok()
                                .and_then(|cn| cn.get(&att.my_node))
                                .map(|n| n.offset)
                                .unwrap_or(Vec3::ZERO);
                            Some((*c, parent_pos + pn.offset - child_offset))
                        })
                        .collect()
                })
                .unwrap_or_default();
            for (child, pos) in parent_pos_and_nodes {
                if let Ok(mut ct) = transforms.get_mut(child) {
                    ct.translation = pos;
                    ct.rotation = Quat::IDENTITY;
                }
                queue.push_back(child);
            }
        }
    }
}

/// Keep [`ShipPartMaterial`] uniforms in sync with part dimensions.
fn update_ship_part_shader_params(
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    parts: Query<
        (
            &AttachNodes,
            &PartShaderHandle,
            Option<&FuelTank>,
            Option<&Decoupler>,
            Option<&Adapter>,
        ),
        Or<(
            Changed<FuelTank>,
            Changed<Decoupler>,
            Changed<Adapter>,
            Changed<AttachNodes>,
        )>,
    >,
) {
    for (nodes, handle, tank, dec, adapter) in parts.iter() {
        let Some(mat) = ship_materials.get_mut(&handle.0) else {
            continue;
        };
        let params = ship_part_params(nodes, tank, dec, adapter, mat.extension.params.seed);
        mat.extension.params.length = params.length;
        mat.extension.params.radius_top = params.radius_top;
        mat.extension.params.radius_bottom = params.radius_bottom;
    }
}
