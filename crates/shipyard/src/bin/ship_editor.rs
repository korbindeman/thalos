//! Interactive 3D ship editor for the shipyard crate. Temporary home — once
//! the workflow settles, this should probably move out to its own crate so
//! `thalos_shipyard` can stay a headless library.
//!
//! Workflow:
//! - Left panel: parts palette + file I/O. Clicking a part arms it as
//!   "pending" — a popup then lists free attach nodes on the existing ship
//!   to place the pending part at.
//! - Right panel: inspector for the selected part (editable params,
//!   resource pools, delete).
//! - Viewport: orbit camera (right-drag + scroll), gizmo spheres at each
//!   attach node, parts rendered as cylinders/frustums sized from their
//!   attach-node diameters.

use bevy::input::gestures::PinchGesture;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::picking::events::{Click, Pointer};
use bevy::picking::mesh_picking::ray_cast::RayCastVisibility;
use bevy::picking::mesh_picking::{MeshPickingPlugin, MeshPickingSettings};
use bevy::picking::Pickable;
use bevy::prelude::*;
use bevy_egui::{EguiContextSettings, EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;

use thalos_shipyard::sizing::propagate_node_sizes;
use thalos_shipyard::*;

const SHIPS_DIR: &str = "ships";

fn sanitize_name(name: &str) -> String {
    let s: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() { "unnamed".into() } else { s }
}

fn ship_path(name: &str) -> PathBuf {
    PathBuf::from(SHIPS_DIR).join(format!("{}.ron", sanitize_name(name)))
}

fn list_ships() -> Vec<String> {
    let dir = PathBuf::from(SHIPS_DIR);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("ron") {
                return None;
            }
            p.file_stem().and_then(|s| s.to_str()).map(|s| s.to_string())
        })
        .collect();
    out.sort();
    out
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Thalos Shipyard".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .add_plugins(MeshPickingPlugin)
        .insert_resource(MeshPickingSettings {
            require_markers: false,
            ray_cast_visibility: RayCastVisibility::Any,
        })
        .add_plugins(ShipyardPlugin)
        .init_resource::<EditorState>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                orbit_camera,
                process_commands,
                rebuild_visuals,
                update_part_transforms.after(propagate_node_sizes),
                update_node_pin_style,
                disable_egui_pointer_capture,
                sync_self_nodes,
            ),
        )
        .add_systems(EguiPrimaryContextPass, editor_ui)
        .run();
}

// ---------------------------------------------------------------------------
// Resources / components
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
struct EditorState {
    ship_root: Option<Entity>,
    ship_entity: Option<Entity>,
    selected: Option<Entity>,
    pending: Option<PartData>,
    place_at: Option<(Entity, String)>,
    delete_selected: bool,
    save_requested: bool,
    load_target: Option<String>,
    delete_file: Option<String>,
    refresh_list: bool,
    ship_list: Vec<String>,
    status: String,
}

#[derive(Resource)]
struct EditorAssets {
    part_material: Handle<StandardMaterial>,
    pending_node_material: Handle<StandardMaterial>,
    node_mesh: Handle<Mesh>,
}

#[derive(Component)]
struct PartVisual;

#[derive(Component)]
struct PartBody(Entity);

#[derive(Component)]
struct AttachNodePin {
    part: Entity,
    node_id: NodeId,
}


#[derive(Component)]
struct OrbitCamera {
    focus: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

fn setup(
    mut commands: Commands,
    mut mats: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut state: ResMut<EditorState>,
) {
    commands.insert_resource(EditorAssets {
        part_material: mats.add(StandardMaterial {
            base_color: Color::srgb(0.75, 0.78, 0.82),
            perceptual_roughness: 0.4,
            metallic: 0.6,
            ..default()
        }),
        pending_node_material: mats.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.9, 1.0),
            emissive: LinearRgba::rgb(0.1, 0.6, 0.9),
            ..default()
        }),
        node_mesh: meshes.add(Sphere::new(0.25).mesh()),
    });

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(8.0, 4.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitCamera {
            focus: Vec3::new(0.0, -2.0, 0.0),
            distance: 12.0,
            yaw: 0.8,
            pitch: 0.4,
        },
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        PointLight {
            intensity: 400_000.0,
            ..default()
        },
        Transform::from_xyz(-6.0, 4.0, -4.0),
    ));


    // Initial ship: single command pod.
    let pod_data = PartData::CommandPod {
        model: "Mk1".into(),
        diameter: 1.25,
        dry_mass: 840.0,
    };
    let pod = ShipBlueprint::spawn_part(&mut commands, &pod_data, HashMap::new());
    let ship = commands
        .spawn(Ship {
            name: "New Ship".into(),
            root: pod,
        })
        .id();
    state.ship_root = Some(pod);
    state.ship_entity = Some(ship);
    state.selected = Some(pod);
    state.ship_list = list_ships();
    state.status = "Ready".into();
}

// ---------------------------------------------------------------------------
// Visuals
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
            .into(),
            height: h,
        })
    } else if dec.is_some() {
        let d = nodes
            .get("top")
            .map(|n| n.diameter)
            .unwrap_or(1.0);
        let h = 0.2;
        Some(VisualSpec {
            mesh: Cylinder::new(d * 0.55, h).mesh().into(),
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
            .into(),
            height: h,
        })
    } else if let Some(t) = tank {
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let h = t.length;
        Some(VisualSpec {
            mesh: Cylinder::new(d * 0.5, h).mesh().into(),
            height: h,
        })
    } else if let Some(e) = engine {
        let d = e.diameter;
        let h = d * 0.9;
        Some(VisualSpec {
            mesh: ConicalFrustum {
                radius_top: d * 0.35,
                radius_bottom: d * 0.5,
                height: h,
            }
            .mesh()
            .into(),
            height: h,
        })
    } else {
        None
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
    ),
    Or<(Added<Part>, Changed<AttachNodes>)>,
>;

fn rebuild_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<EditorAssets>,
    parts: VisualQuery,
    stale: Query<(), Or<(With<PartVisual>, With<AttachNodePin>)>>,
) {
    for (e, nodes, pod, dec, adapter, tank, engine, children) in parts.iter() {
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }

        // ---- Body visual --------------------------------------------------
        if let Some(spec) = visual_spec(nodes, pod, dec, adapter, tank, engine) {
            let mesh = meshes.add(spec.mesh);
            let body = commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(assets.part_material.clone()),
                    Transform::from_xyz(0.0, -spec.height * 0.5, 0.0),
                    Visibility::default(),
                    PartVisual,
                    PartBody(e),
                    Pickable::default(),
                ))
                .observe(on_body_click)
                .id();
            commands.entity(e).add_child(body);
        }

        // ---- Attach node pins --------------------------------------------
        for (id, node) in &nodes.nodes {
            let pin = commands
                .spawn((
                    Mesh3d(assets.node_mesh.clone()),
                    MeshMaterial3d(assets.pending_node_material.clone()),
                    Transform::from_translation(node.offset),
                    Visibility::Hidden,
                    AttachNodePin {
                        part: e,
                        node_id: id.clone(),
                    },
                    Pickable::default(),
                ))
                .observe(on_pin_click)
                .id();
            commands.entity(e).add_child(pin);
        }
    }
}

fn update_part_transforms(
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
                            parent_nodes
                                .get(&att.parent_node)
                                .map(|n| (*c, parent_pos + n.offset))
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

fn update_node_pin_style(
    state: Res<EditorState>,
    assets: Res<EditorAssets>,
    attachments: Query<&Attachment>,
    mut pins: Query<(
        &AttachNodePin,
        &mut MeshMaterial3d<StandardMaterial>,
        &mut Visibility,
    )>,
) {
    let occupied: HashSet<(Entity, String)> = attachments
        .iter()
        .map(|a| (a.parent, a.parent_node.clone()))
        .collect();
    let pending = state.pending.is_some();

    for (pin, mut mat, mut vis) in pins.iter_mut() {
        let is_occupied = occupied.contains(&(pin.part, pin.node_id.clone()));

        // Pins only appear while a part is pending, and only on unoccupied
        // nodes.
        *vis = if pending && !is_occupied {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };

        mat.0 = assets.pending_node_material.clone();
    }
}

/// bevy_egui's default `capture_pointer_input` writes a fake top-priority
/// PointerHits for the egui context entity whenever egui wants pointer
/// input, which redirects every click away from our meshes. Disable it and
/// filter picks manually via `is_pointer_over_area` below.
fn disable_egui_pointer_capture(mut q: Query<&mut EguiContextSettings>) {
    for mut s in q.iter_mut() {
        if s.capture_pointer_input {
            s.capture_pointer_input = false;
        }
    }
}

/// Propagate a part's own kind-component values (e.g. `CommandPod.diameter`,
/// `FuelTank.length`, `Engine.diameter`) into its own AttachNodes, only
/// touching the component when a value actually differs. This way editor
/// sliders drive AttachNodes → rebuild_visuals deterministically, without
/// the kind component's spurious Changed signals causing per-frame respawns.
fn sync_self_nodes(
    mut q: Query<(
        &mut AttachNodes,
        Option<&CommandPod>,
        Option<&FuelTank>,
        Option<&Engine>,
    )>,
) {
    for (mut nodes, pod, tank, engine) in q.iter_mut() {
        let mut targets: Vec<(String, f32, Vec3)> = Vec::new();
        if let Some(p) = pod {
            let d = p.diameter;
            targets.push(("bottom".into(), d, Vec3::new(0.0, -d * 0.9, 0.0)));
        }
        if let Some(t) = tank {
            // keep bottom offset in sync with length regardless of parent
            let d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
            targets.push(("bottom".into(), d, Vec3::new(0.0, -t.length, 0.0)));
        }
        if let Some(e) = engine {
            targets.push(("top".into(), e.diameter, Vec3::ZERO));
        }
        let needs_update = targets.iter().any(|(id, d, off)| {
            nodes
                .get(id)
                .map(|n| {
                    (n.diameter - *d).abs() > f32::EPSILON
                        || n.offset.distance_squared(*off) > f32::EPSILON
                })
                .unwrap_or(false)
        });
        if !needs_update {
            continue;
        }
        for (id, d, off) in &targets {
            if let Some(n) = nodes.nodes.get_mut(id) {
                n.diameter = *d;
                n.offset = *off;
            }
        }
    }
}

fn pointer_over_egui(contexts: &mut EguiContexts) -> bool {
    contexts
        .ctx_mut()
        .map(|c| c.is_pointer_over_area())
        .unwrap_or(false)
}

fn on_body_click(
    click: On<Pointer<Click>>,
    bodies: Query<&PartBody>,
    mut state: ResMut<EditorState>,
    mut contexts: EguiContexts,
) {
    if pointer_over_egui(&mut contexts) {
        return;
    }
    if let Ok(body) = bodies.get(click.entity) {
        state.selected = Some(body.0);
    }
}

fn on_pin_click(
    click: On<Pointer<Click>>,
    pins: Query<&AttachNodePin>,
    mut state: ResMut<EditorState>,
    mut contexts: EguiContexts,
) {
    if pointer_over_egui(&mut contexts) {
        return;
    }
    if let Ok(pin) = pins.get(click.entity) {
        if state.pending.is_some() {
            state.place_at = Some((pin.part, pin.node_id.clone()));
        } else {
            state.selected = Some(pin.part);
        }
    }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

fn orbit_camera(
    mut cam: Query<(&mut Transform, &mut OrbitCamera)>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut pinch: MessageReader<PinchGesture>,
    mut contexts: EguiContexts,
    state: Res<EditorState>,
) {
    let pointer_over_egui = contexts
        .ctx_mut()
        .map(|c| c.is_pointer_over_area() || c.wants_pointer_input())
        .unwrap_or(false);

    let mut delta = Vec2::ZERO;
    for m in motion.read() {
        delta += m.delta;
    }
    // Normalize wheel deltas so trackpad pixel scrolling isn't 20x faster
    // than a real mouse wheel.
    let mut wheel_d: f32 = 0.0;
    for w in wheel.read() {
        let scale = match w.unit {
            bevy::input::mouse::MouseScrollUnit::Line => 1.0,
            bevy::input::mouse::MouseScrollUnit::Pixel => 1.0 / 40.0,
        };
        wheel_d += w.y * scale;
    }
    let mut pinch_d: f32 = 0.0;
    for p in pinch.read() {
        pinch_d += p.0;
    }

    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    // While a part is pending, suppress orbit so a press→release on a pin
    // stays hovered on that pin and picking actually fires a Click. Any
    // camera rotation between press and release breaks hover and drops the
    // event into the window fallback entity.
    let orbit_allowed = !pointer_over_egui && state.pending.is_none();

    for (mut t, mut orbit) in cam.iter_mut() {
        if orbit_allowed && mouse.pressed(MouseButton::Left) {
            orbit.yaw -= delta.x * 0.005;
            orbit.pitch = (orbit.pitch - delta.y * 0.005).clamp(-1.5, 1.5);
        }

        if !pointer_over_egui && wheel_d.abs() > 0.0 {
            if shift {
                orbit.distance =
                    (orbit.distance * (1.0 - wheel_d * 0.05)).clamp(2.0, 200.0);
            } else {
                orbit.focus.y += wheel_d * orbit.distance * 0.015;
            }
        }

        // Trackpad pinch zooms regardless of shift.
        if !pointer_over_egui && pinch_d.abs() > 0.0 {
            orbit.distance =
                (orbit.distance * (1.0 - pinch_d * 8.0)).clamp(2.0, 200.0);
        }

        let rot = Quat::from_euler(EulerRot::YXZ, orbit.yaw, -orbit.pitch, 0.0);
        let offset = rot * Vec3::new(0.0, 0.0, orbit.distance);
        t.translation = orbit.focus + offset;
        t.look_at(orbit.focus, Vec3::Y);
    }
}

// ---------------------------------------------------------------------------
// Blueprint <-> ECS
// ---------------------------------------------------------------------------

type CollectQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static PartResources,
        Option<&'static CommandPod>,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Engine>,
    ),
>;

fn collect_blueprint(
    ship: &Ship,
    parts: &CollectQuery,
    attachments: &Query<(Entity, &Attachment)>,
) -> Option<ShipBlueprint> {
    let mut child_map: HashMap<Entity, Vec<(Entity, Attachment)>> = HashMap::new();
    for (e, att) in attachments.iter() {
        child_map
            .entry(att.parent)
            .or_default()
            .push((e, att.clone()));
    }

    let mut ordered: Vec<Entity> = Vec::new();
    let mut queue: VecDeque<Entity> = VecDeque::from([ship.root]);
    while let Some(e) = queue.pop_front() {
        ordered.push(e);
        if let Some(kids) = child_map.get(&e) {
            for (c, _) in kids {
                queue.push_back(*c);
            }
        }
    }

    let idx: HashMap<Entity, usize> = ordered
        .iter()
        .enumerate()
        .map(|(i, e)| (*e, i))
        .collect();

    let mut part_blueprints = Vec::with_capacity(ordered.len());
    for e in &ordered {
        let (_, res, pod, dec, adapter, tank, engine) = parts.get(*e).ok()?;
        let data = if let Some(p) = pod {
            PartData::CommandPod {
                model: p.model.clone(),
                diameter: p.diameter,
                dry_mass: p.dry_mass,
            }
        } else if let Some(d) = dec {
            PartData::Decoupler {
                ejection_impulse: d.ejection_impulse,
            }
        } else if let Some(a) = adapter {
            PartData::Adapter {
                target_diameter: a.target_diameter,
            }
        } else if let Some(t) = tank {
            PartData::FuelTank {
                length: t.length,
                fuel_density: t.fuel_density,
            }
        } else if let Some(en) = engine {
            PartData::Engine {
                model: en.model.clone(),
                diameter: en.diameter,
                thrust: en.thrust,
                isp: en.isp,
            }
        } else {
            return None;
        };
        part_blueprints.push(PartBlueprint {
            data,
            resources: res.pools.clone(),
        });
    }

    let mut connections = Vec::new();
    for (e, att) in attachments.iter() {
        if let (Some(&ci), Some(&pi)) = (idx.get(&e), idx.get(&att.parent)) {
            connections.push(Connection {
                parent: pi,
                parent_node: att.parent_node.clone(),
                child: ci,
                child_node: att.my_node.clone(),
            });
        }
    }

    Some(ShipBlueprint {
        name: ship.name.clone(),
        root: 0,
        parts: part_blueprints,
        connections,
    })
}

// ---------------------------------------------------------------------------
// Command processing (save / load / delete / place)
// ---------------------------------------------------------------------------

fn process_commands(
    mut commands: Commands,
    mut state: ResMut<EditorState>,
    ships: Query<&Ship>,
    parts_q: CollectQuery,
    attachments: Query<(Entity, &Attachment)>,
    all_parts: Query<Entity, With<Part>>,
    all_ships: Query<Entity, With<Ship>>,
) {
    // ---- Save ---------------------------------------------------------
    if state.save_requested {
        state.save_requested = false;
        if let Some(ship_entity) = state.ship_entity {
            if let Ok(ship) = ships.get(ship_entity) {
                match collect_blueprint(ship, &parts_q, &attachments) {
                    Some(bp) => match bp.to_ron() {
                        Ok(text) => {
                            let path = ship_path(&bp.name);
                            if let Err(e) = std::fs::create_dir_all(SHIPS_DIR) {
                                state.status = format!("mkdir failed: {e}");
                            } else {
                                match std::fs::write(&path, text) {
                                    Ok(()) => {
                                        state.status = format!("Saved {}", path.display());
                                        state.refresh_list = true;
                                    }
                                    Err(e) => state.status = format!("Save failed: {e}"),
                                }
                            }
                        }
                        Err(e) => state.status = format!("Serialize failed: {e}"),
                    },
                    None => state.status = "Failed to collect blueprint".into(),
                }
            }
        }
    }

    // ---- Load ---------------------------------------------------------
    if let Some(name) = state.load_target.take() {
        let path = ship_path(&name);
        match std::fs::read_to_string(&path) {
            Ok(text) => match ShipBlueprint::from_ron(&text) {
                Ok(bp) => {
                    for e in all_parts.iter() {
                        commands.entity(e).despawn();
                    }
                    for e in all_ships.iter() {
                        commands.entity(e).despawn();
                    }
                    state.ship_root = None;
                    state.ship_entity = None;
                    state.selected = None;

                    let path_disp = path.display().to_string();
                    commands.queue(move |world: &mut World| {
                        let mut cmds = world.commands();
                        let ship_entity = bp.spawn(&mut cmds);
                        world.flush();
                        let root = world.get::<Ship>(ship_entity).map(|s| s.root);
                        let mut st = world.resource_mut::<EditorState>();
                        st.ship_entity = Some(ship_entity);
                        st.ship_root = root;
                        st.selected = root;
                        st.status = format!("Loaded {path_disp}");
                    });
                }
                Err(e) => state.status = format!("Parse failed: {e}"),
            },
            Err(e) => state.status = format!("Read failed: {e}"),
        }
    }

    // ---- Delete file --------------------------------------------------
    if let Some(name) = state.delete_file.take() {
        let path = ship_path(&name);
        match std::fs::remove_file(&path) {
            Ok(()) => {
                state.status = format!("Deleted {}", path.display());
                state.refresh_list = true;
            }
            Err(e) => state.status = format!("Delete failed: {e}"),
        }
    }

    // ---- Refresh list -------------------------------------------------
    if state.refresh_list {
        state.refresh_list = false;
        state.ship_list = list_ships();
    }

    // ---- Delete selected (cannot delete root) -------------------------
    if state.delete_selected {
        state.delete_selected = false;
        if let Some(sel) = state.selected {
            if Some(sel) != state.ship_root {
                // collect all descendants, despawn them
                let mut child_map: HashMap<Entity, Vec<Entity>> = HashMap::new();
                for (e, att) in attachments.iter() {
                    child_map.entry(att.parent).or_default().push(e);
                }
                let mut to_remove: Vec<Entity> = Vec::new();
                let mut stack = vec![sel];
                while let Some(e) = stack.pop() {
                    to_remove.push(e);
                    if let Some(kids) = child_map.get(&e) {
                        stack.extend(kids.iter().copied());
                    }
                }
                for e in to_remove {
                    commands.entity(e).despawn();
                }
                state.selected = state.ship_root;
                state.status = "Deleted selection".into();
            } else {
                state.status = "Cannot delete root pod".into();
            }
        }
    }

    // ---- Place pending part at a given (parent, node) -----------------
    if let Some((parent, node)) = state.place_at.take() {
        let Some(pending) = state.pending.take() else {
            return;
        };
        let child = ShipBlueprint::spawn_part(&mut commands, &pending, HashMap::new());
        commands.entity(child).insert(Attachment {
            parent,
            parent_node: node,
            my_node: "top".into(),
        });
        state.selected = Some(child);
        state.status = "Placed part".into();
    }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

type InspectorQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AttachNodes,
        Option<&'static mut CommandPod>,
        Option<&'static mut Decoupler>,
        Option<&'static mut Adapter>,
        Option<&'static mut FuelTank>,
        Option<&'static mut Engine>,
        Option<&'static mut PartResources>,
    ),
>;

fn editor_ui(
    mut contexts: EguiContexts,
    mut state: ResMut<EditorState>,
    mut parts: InspectorQuery,
    mut ships: Query<&mut Ship>,
    attachments: Query<(Entity, &Attachment)>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let ctx = ctx.clone();

    // -------- Left palette --------
    egui::SidePanel::left("palette")
        .default_width(180.0)
        .show(&ctx, |ui| {
            ui.heading("Parts");
            if ui.button("Command Pod").clicked() {
                state.pending = Some(PartData::CommandPod {
                    model: "Mk1".into(),
                    diameter: 1.25,
                    dry_mass: 840.0,
                });
            }
            if ui.button("Decoupler").clicked() {
                state.pending = Some(PartData::Decoupler {
                    ejection_impulse: 250.0,
                });
            }
            if ui.button("Adapter").clicked() {
                state.pending = Some(PartData::Adapter {
                    target_diameter: 2.5,
                });
            }
            if ui.button("Fuel Tank").clicked() {
                state.pending = Some(PartData::FuelTank {
                    length: 4.0,
                    fuel_density: 5.0,
                });
            }
            if ui.button("Engine (2.5m)").clicked() {
                state.pending = Some(PartData::Engine {
                    model: "LV-T45".into(),
                    diameter: 2.5,
                    thrust: 215_000.0,
                    isp: 320.0,
                });
            }
            if ui.button("Engine (1.25m)").clicked() {
                state.pending = Some(PartData::Engine {
                    model: "LV-909".into(),
                    diameter: 1.25,
                    thrust: 60_000.0,
                    isp: 345.0,
                });
            }

            ui.separator();
            ui.heading("Ship");
            if let Some(se) = state.ship_entity {
                if let Ok(mut ship) = ships.get_mut(se) {
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.text_edit_singleline(&mut ship.name);
                    });
                }
            }
            if ui.button("Save").clicked() {
                state.save_requested = true;
            }
            if ui.button("Refresh list").clicked() {
                state.refresh_list = true;
            }

            ui.separator();
            ui.heading("Saved ships");
            let ship_list = state.ship_list.clone();
            if ship_list.is_empty() {
                ui.label("(none)");
            }
            for name in ship_list {
                ui.horizontal(|ui| {
                    if ui.button("Load").clicked() {
                        state.load_target = Some(name.clone());
                    }
                    if ui.button("X").clicked() {
                        state.delete_file = Some(name.clone());
                    }
                    ui.label(&name);
                });
            }

            ui.separator();
            ui.label(format!("Status: {}", state.status));
            if state.pending.is_some() {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "Pending part — pick an attach node to place.",
                );
                if ui.button("Cancel pending").clicked() {
                    state.pending = None;
                }
            }
        });

    // -------- Right inspector --------
    egui::SidePanel::right("inspector")
        .default_width(260.0)
        .show(&ctx, |ui| {
            ui.heading("Inspector");
            let Some(sel) = state.selected else {
                ui.label("(no selection)");
                return;
            };
            let Ok((entity, nodes, mut pod, mut dec, mut adapter, mut tank, mut engine, mut res)) =
                parts.get_mut(sel)
            else {
                ui.label("(invalid selection)");
                return;
            };
            ui.label(format!("Entity: {entity:?}"));

            if let Some(p) = pod.as_deref_mut() {
                ui.label("Kind: CommandPod");
                ui.label(format!("Model: {}", p.model));
                ui.add(egui::Slider::new(&mut p.diameter, 0.5..=5.0).text("Diameter"));
                ui.add(egui::Slider::new(&mut p.dry_mass, 100.0..=5000.0).text("Dry mass"));
            } else if let Some(d) = dec.as_deref_mut() {
                ui.label("Kind: Decoupler");
                ui.add(
                    egui::Slider::new(&mut d.ejection_impulse, 0.0..=2000.0)
                        .text("Ejection impulse"),
                );
            } else if let Some(a) = adapter.as_deref_mut() {
                ui.label("Kind: Adapter");
                ui.add(
                    egui::Slider::new(&mut a.target_diameter, 0.3..=6.0).text("Target diameter"),
                );
            } else if let Some(t) = tank.as_deref_mut() {
                ui.label("Kind: Fuel Tank");
                ui.add(egui::Slider::new(&mut t.length, 0.5..=12.0).text("Length"));
                ui.add(
                    egui::Slider::new(&mut t.fuel_density, 0.5..=20.0).text("Fuel density"),
                );
            } else if let Some(e) = engine.as_deref_mut() {
                ui.label("Kind: Engine");
                ui.label(format!("Model: {}", e.model));
                ui.label(format!("Diameter: {:.2}m (fixed)", e.diameter));
                ui.add(egui::Slider::new(&mut e.thrust, 1_000.0..=2_000_000.0).text("Thrust (N)"));
                ui.add(egui::Slider::new(&mut e.isp, 100.0..=450.0).text("Isp (s)"));
            }

            ui.separator();
            ui.label("Attach nodes:");
            for (id, node) in &nodes.nodes {
                ui.label(format!("  {id}: Ø{:.2}m", node.diameter));
            }

            ui.separator();
            ui.label("Resources:");
            if let Some(r) = res.as_deref_mut() {
                for (name, pool) in r.pools.iter_mut() {
                    ui.label(format!(
                        "{name}: {:.0}/{:.0} (ρ={:.1})",
                        pool.amount, pool.capacity, pool.density
                    ));
                    ui.add(egui::Slider::new(&mut pool.amount, 0.0..=pool.capacity).text("amount"));
                }
                if r.pools.is_empty() {
                    ui.label("  (none)");
                }
            }

            ui.separator();
            if ui.button("Delete part").clicked() {
                state.delete_selected = true;
            }
        });

    // -------- Bottom: ship hierarchy & placement picker --------
    egui::TopBottomPanel::bottom("hierarchy")
        .default_height(180.0)
        .show(&ctx, |ui| {
            ui.horizontal(|ui| {
                // Hierarchy list
                ui.vertical(|ui| {
                    ui.heading("Ship");
                    let Some(root) = state.ship_root else {
                        return;
                    };
                    let mut child_map: HashMap<Entity, Vec<(Entity, Attachment)>> = HashMap::new();
                    for (e, att) in attachments.iter() {
                        child_map
                            .entry(att.parent)
                            .or_default()
                            .push((e, att.clone()));
                    }
                    draw_hierarchy(ui, root, &child_map, &mut state, 0);
                });

                ui.separator();

                // Placement picker: list free nodes on the ship
                if state.pending.is_some() {
                    ui.vertical(|ui| {
                        ui.heading("Place at…");
                        let occupied: std::collections::HashSet<(Entity, String)> = attachments
                            .iter()
                            .map(|(_, a)| (a.parent, a.parent_node.clone()))
                            .collect();
                        let mut rows: Vec<(Entity, String, f32)> = Vec::new();
                        for (e, nodes, _, _, _, _, _, _) in parts.iter() {
                            for (nid, node) in &nodes.nodes {
                                if occupied.contains(&(e, nid.clone())) {
                                    continue;
                                }
                                // Skip command pod's implicit "top" — but
                                // we don't store one, so nothing to skip.
                                rows.push((e, nid.clone(), node.diameter));
                            }
                        }
                        for (e, nid, d) in rows {
                            if ui
                                .button(format!("{e:?} / {nid} (Ø{d:.2}m)"))
                                .clicked()
                            {
                                state.place_at = Some((e, nid));
                            }
                        }
                    });
                }
            });
        });
}

fn draw_hierarchy(
    ui: &mut egui::Ui,
    entity: Entity,
    child_map: &HashMap<Entity, Vec<(Entity, Attachment)>>,
    state: &mut EditorState,
    depth: usize,
) {
    let indent = "  ".repeat(depth);
    let selected = state.selected == Some(entity);
    let label = format!("{indent}{entity:?}");
    if ui.selectable_label(selected, label).clicked() {
        state.selected = Some(entity);
    }
    if let Some(kids) = child_map.get(&entity) {
        for (c, _) in kids {
            draw_hierarchy(ui, *c, child_map, state, depth + 1);
        }
    }
}
