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

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::gestures::PinchGesture;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::mesh::{Indices, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::picking::events::{Click, DragEnd, DragStart, Pointer};
use bevy::picking::hover::HoverMap;
use bevy::picking::mesh_picking::ray_cast::RayCastVisibility;
use bevy::picking::mesh_picking::{MeshPickingPlugin, MeshPickingSettings};
use bevy::picking::Pickable;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;
use bevy::window::PrimaryWindow;
use bevy_egui::{EguiContextSettings, EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;

use thalos_celestial::Universe;
use thalos_celestial::generate::{DefaultGenParams, generate_default};
use thalos_ship_rendering::{
    ShipPartExtension, ShipPartMaterial, ShipPartParams, ShipRenderingPlugin, stainless_steel_base,
};
use thalos_shipyard::sizing::propagate_node_sizes;
use thalos_shipyard::Resource as ShipResource;
use thalos_shipyard::*;

const SHIPS_DIR: &str = "ships";

/// Radial segment count for cylindrical/frustum part meshes. Bevy's
/// default is 32, which leaves a visibly faceted silhouette at editor
/// zoom levels. Cost is negligible at the part counts we render.
const PART_RESOLUTION: u32 = 128;

/// Methalox reactant mix at O/F ≈ 3.6 (Raptor-ish).
fn methalox_reactants() -> Vec<ReactantRatio> {
    vec![
        ReactantRatio {
            resource: ShipResource::Methane,
            mass_fraction: 1.0 / 4.6,
        },
        ReactantRatio {
            resource: ShipResource::Lox,
            mass_fraction: 3.6 / 4.6,
        },
    ]
}

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
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Thalos Shipyard".into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(bevy::asset::AssetPlugin {
                    // Resolve shaders from the workspace-root `assets/` dir,
                    // matching `thalos_game` and `thalos_planet_editor`.
                    file_path: "../../assets".to_string(),
                    ..default()
                }),
        )
        .add_plugins(EguiPlugin::default())
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(MeshPickingPlugin)
        .insert_resource(MeshPickingSettings {
            require_markers: false,
            // `VisibleInView` so hidden handles (resize arrow, non-pending
            // pins) don't absorb clicks from the body behind them.
            ray_cast_visibility: RayCastVisibility::VisibleInView,
        })
        .add_plugins(ShipyardPlugin)
        .add_plugins(ShipRenderingPlugin)
        .add_plugins(SkyBackdropPlugin)
        .init_resource::<EditorState>()
        .init_resource::<TankResizeDrag>()
        .init_resource::<DeselectTracker>()
        .init_resource::<SkyBackdropEnabled>()
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
                spawn_tank_resize_arrow,
                update_tank_resize_arrow.after(update_part_transforms),
                update_tank_resize_drag,
                update_selection_highlight.after(rebuild_visuals),
                update_part_shader_params.after(rebuild_visuals),
                update_part_shader_highlight.after(rebuild_visuals),
                deselect_on_empty_click,
                propagate_coupled_material.after(rebuild_visuals),
                sync_shrouds.after(update_part_transforms),
                update_shroud_transparency.after(sync_shrouds),
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
    ship_name: String,
    selected: Option<Entity>,
    pending: Option<PartData>,
    place_at: Option<(Entity, String)>,
    delete_selected: bool,
    set_as_root: bool,
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
    hover_material: Handle<StandardMaterial>,
    selected_material: Handle<StandardMaterial>,
    pending_node_material: Handle<StandardMaterial>,
    node_mesh: Handle<Mesh>,
    resize_arrow_mesh: Handle<Mesh>,
    resize_arrow_material: Handle<StandardMaterial>,
}

#[derive(Component)]
struct PartVisual;

#[derive(Component)]
struct PartBody(Entity);

/// Per-part `ShipPartMaterial` asset handle, cached on the part entity
/// so it survives child rebuilds (e.g. resizing a tank despawns and
/// respawns the body, but the material asset — and its tint state — is
/// stable). Used by any part that carries [`PartMaterial`] — tanks and
/// decouplers today.
#[derive(Component, Clone)]
struct PartShaderHandle(Handle<ShipPartMaterial>);

#[derive(Component)]
struct AttachNodePin {
    part: Entity,
    node_id: NodeId,
}

#[derive(Component)]
struct TankResizeArrow {
    tank: Entity,
}

#[derive(Resource, Default)]
struct TankResizeDrag {
    active: Option<TankDragState>,
}

struct TankDragState {
    tank: Entity,
    start_length: f32,
    start_cursor: Vec2,
    screen_axis: Vec2,
    world_per_pixel: f32,
}

/// Tracks the cursor at mouse-down when the press landed on empty space,
/// so a release at near-the-same position clears the selection but a
/// press→drag→release (camera orbit) does not.
#[derive(Resource, Default)]
struct DeselectTracker {
    press_cursor: Option<Vec2>,
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
        hover_material: mats.add(StandardMaterial {
            base_color: Color::srgb(0.82, 0.85, 0.88),
            perceptual_roughness: 0.4,
            metallic: 0.6,
            emissive: LinearRgba::rgb(0.08, 0.08, 0.08),
            ..default()
        }),
        selected_material: mats.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.9, 1.0),
            perceptual_roughness: 0.4,
            metallic: 0.6,
            emissive: LinearRgba::rgb(0.15, 0.35, 0.7),
            ..default()
        }),
        pending_node_material: mats.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.9, 1.0),
            emissive: LinearRgba::rgb(0.1, 0.6, 0.9),
            ..default()
        }),
        node_mesh: meshes.add(Sphere::new(0.25).mesh()),
        resize_arrow_mesh: meshes.add(Cone::new(0.3, 0.8).mesh()),
        resize_arrow_material: mats.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.75, 0.2),
            emissive: LinearRgba::rgb(0.9, 0.5, 0.05),
            perceptual_roughness: 0.5,
            unlit: false,
            ..default()
        }),
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


    state.ship_name = "New Ship".into();
    state.ship_list = list_ships();
    state.status = "Click a part to begin".into();
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
            .resolution(PART_RESOLUTION)
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
        let (r_top, r_bot, h) = engine_visual_profile(e.diameter);
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

/// Engine body silhouette: `(radius_top, radius_bottom, height)` for a
/// given engine diameter. Single source for both the engine mesh and the
/// matching shroud geometry — drift between the two would leave the
/// shroud edge either floating off the engine or clipping into it.
fn engine_visual_profile(diameter: f32) -> (f32, f32, f32) {
    (diameter * 0.35, diameter * 0.5, diameter * 0.9)
}

/// Pick `ShipPartMaterial` uniforms for a given part. Length / radius
/// drive the procedural panel + rivet layout; each part picks its own
/// dimensions so the pattern reads consistently across tank–decoupler
/// boundaries without sharing an asset handle.
fn ship_part_params(
    nodes: &AttachNodes,
    tank: Option<&FuelTank>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    seed: u32,
) -> ShipPartParams {
    let top_r = nodes.get("top").map(|n| n.diameter * 0.5).unwrap_or(0.5);
    // Tanks and decouplers are cylinders; adapters are conical frustums
    // from `top_r` at the mesh's +Y end to `target_diameter / 2` at -Y.
    let (radius_top, radius_bottom, length) = if let Some(t) = tank {
        (top_r, top_r, t.length)
    } else if dec.is_some() {
        (top_r, top_r, 0.2)
    } else if let Some(a) = adapter {
        let bot_r = a.target_diameter * 0.5;
        let h = (top_r + bot_r).max(0.4); // same formula as `visual_spec`
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

fn rebuild_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    assets: Res<EditorAssets>,
    state: Res<EditorState>,
    parts: VisualQuery,
    stale: Query<(), Or<(With<PartVisual>, With<AttachNodePin>)>>,
) {
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

        // ---- Body visual --------------------------------------------------
        if let Some(spec) = visual_spec(nodes, pod, dec, adapter, tank, engine) {
            let mesh = meshes.add(spec.mesh);

            // Parts carrying `PartMaterial` render with `ShipPartMaterial`
            // (procedural stainless); others use the shared
            // `StandardMaterial`. The ship-material asset is created lazily
            // on first rebuild and cached on the part entity so resizing
            // doesn't churn assets or drop per-part state (seed/tint).
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
                        PartVisual,
                        PartBody(e),
                        Pickable::default(),
                    ))
                    .observe(on_body_click)
                    .id()
            } else {
                let initial_material = if Some(e) == state.selected {
                    assets.selected_material.clone()
                } else {
                    assets.part_material.clone()
                };
                commands
                    .spawn((
                        Mesh3d(mesh),
                        MeshMaterial3d(initial_material),
                        Transform::from_xyz(0.0, -spec.height * 0.5, 0.0),
                        Visibility::default(),
                        PartVisual,
                        PartBody(e),
                        Pickable::default(),
                    ))
                    .observe(on_body_click)
                    .id()
            };
            commands.entity(e).add_child(body_id);
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
///
/// For parametric radius parts (Decoupler/Adapter/FuelTank) the sync is
/// bidirectional by root state:
/// - **Root**: `self.diameter → nodes.top` so the Diameter slider drives
///   the part's visual size.
/// - **Child**: `nodes.top → self.diameter` so the diameter inherited via
///   `sizing::propagate_node_sizes` is mirrored onto the component. This
///   way a later re-root starts from the displayed size instead of
///   snapping back to the palette's placeholder.
fn sync_self_nodes(
    mut q: Query<(
        &mut AttachNodes,
        Option<&Attachment>,
        Option<&CommandPod>,
        Option<&mut Decoupler>,
        Option<&mut Adapter>,
        Option<&mut FuelTank>,
        Option<&Engine>,
    )>,
) {
    for (mut nodes, attachment, pod, mut dec, mut adapter, mut tank, engine) in q.iter_mut() {
        let is_root = attachment.is_none();
        let mut targets: Vec<(String, f32, Vec3)> = Vec::new();
        if let Some(p) = pod {
            let d = p.diameter;
            targets.push(("bottom".into(), d, Vec3::new(0.0, -d * 0.9, 0.0)));
        }
        // Read kind-component fields through `as_ref()` so the borrow only
        // goes through Bevy's `Mut::deref` (no Changed trigger). The write
        // path below reaches for `as_mut()` only when the value actually
        // needs to change.
        if let Some(d) = dec.as_ref() {
            let self_d = d.diameter;
            let top_d = if is_root {
                targets.push(("top".into(), self_d, Vec3::ZERO));
                self_d
            } else {
                let inherited = nodes.get("top").map(|n| n.diameter).unwrap_or(self_d);
                if (self_d - inherited).abs() > f32::EPSILON {
                    if let Some(m) = dec.as_mut() {
                        m.diameter = inherited;
                    }
                }
                inherited
            };
            targets.push(("bottom".into(), top_d, Vec3::new(0.0, -0.2, 0.0)));
        }
        if let Some(a) = adapter.as_ref() {
            let self_d = a.diameter;
            let target_d = a.target_diameter;
            let top_d = if is_root {
                targets.push(("top".into(), self_d, Vec3::ZERO));
                self_d
            } else {
                let inherited = nodes.get("top").map(|n| n.diameter).unwrap_or(self_d);
                if (self_d - inherited).abs() > f32::EPSILON {
                    if let Some(m) = adapter.as_mut() {
                        m.diameter = inherited;
                    }
                }
                inherited
            };
            let h = ((top_d + target_d) * 0.5).max(0.4);
            targets.push(("bottom".into(), target_d, Vec3::new(0.0, -h, 0.0)));
        }
        if let Some(t) = tank.as_ref() {
            let self_d = t.diameter;
            let length = t.length;
            let top_d = if is_root {
                targets.push(("top".into(), self_d, Vec3::ZERO));
                self_d
            } else {
                let inherited = nodes.get("top").map(|n| n.diameter).unwrap_or(self_d);
                if (self_d - inherited).abs() > f32::EPSILON {
                    if let Some(m) = tank.as_mut() {
                        m.diameter = inherited;
                    }
                }
                inherited
            };
            targets.push(("bottom".into(), top_d, Vec3::new(0.0, -length, 0.0)));
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

// ---------------------------------------------------------------------------
// Tank resize arrow (parametric handle)
// ---------------------------------------------------------------------------

/// Spawn a single resize arrow per fuel tank on creation. The arrow is a
/// child of the tank entity, hidden until the tank becomes the current
/// selection, and positioned each frame by `update_tank_resize_arrow`.
fn spawn_tank_resize_arrow(
    mut commands: Commands,
    assets: Res<EditorAssets>,
    new_tanks: Query<Entity, Added<FuelTank>>,
) {
    for tank in new_tanks.iter() {
        let arrow = commands
            .spawn((
                Mesh3d(assets.resize_arrow_mesh.clone()),
                MeshMaterial3d(assets.resize_arrow_material.clone()),
                Transform::default(),
                Visibility::Hidden,
                TankResizeArrow { tank },
                Pickable::default(),
            ))
            .observe(on_arrow_drag_start)
            .observe(on_arrow_drag_end)
            .id();
        commands.entity(tank).add_child(arrow);
    }
}

/// Show the arrow only while the owning tank is selected; each frame, place
/// it on the camera-facing side of the tank at mid-height with the tip
/// pointing down along the tank's growth axis.
fn update_tank_resize_arrow(
    state: Res<EditorState>,
    tanks: Query<(&FuelTank, &AttachNodes), Without<TankResizeArrow>>,
    cameras: Query<&Transform, (With<OrbitCamera>, Without<TankResizeArrow>, Without<FuelTank>)>,
    mut arrows: Query<(&TankResizeArrow, &mut Transform, &mut Visibility)>,
) {
    let Ok(cam_transform) = cameras.single() else {
        return;
    };

    for (arrow, mut transform, mut vis) in arrows.iter_mut() {
        let is_selected = state.selected == Some(arrow.tank);
        let Ok((tank, nodes)) = tanks.get(arrow.tank) else {
            if *vis != Visibility::Hidden {
                *vis = Visibility::Hidden;
            }
            continue;
        };

        if !is_selected {
            if *vis != Visibility::Hidden {
                *vis = Visibility::Hidden;
            }
            continue;
        }
        if *vis != Visibility::Inherited {
            *vis = Visibility::Inherited;
        }

        // Place the arrow on the camera's right so it doesn't occlude the
        // tank body. Parts never rotate in this editor, so world-space XZ
        // equals local-space XZ.
        let cam_right = cam_transform.right();
        let right_xz =
            Vec2::new(cam_right.x, cam_right.z).try_normalize().unwrap_or(Vec2::X);
        let radius = nodes.get("top").map(|n| n.diameter * 0.5).unwrap_or(0.5);
        let offset_r = radius + 0.55;
        transform.translation = Vec3::new(
            right_xz.x * offset_r,
            -tank.length * 0.5,
            right_xz.y * offset_r,
        );
        // Bevy's Cone has its tip at +Y and base at -Y; rotate PI around X
        // to point the tip down (i.e., the direction the tank grows).
        transform.rotation = Quat::from_rotation_x(std::f32::consts::PI);
    }
}

/// On drag start: snapshot the tank's current length, the cursor origin,
/// and project the world growth axis (-Y) into screen space. Subsequent
/// cursor motion is decomposed along that axis and rescaled to world units.
fn on_arrow_drag_start(
    trigger: On<Pointer<DragStart>>,
    arrows: Query<&TankResizeArrow>,
    tanks: Query<(&FuelTank, &Transform)>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    mut drag: ResMut<TankResizeDrag>,
) {
    let event = trigger.event();
    let Ok(arrow) = arrows.get(event.entity) else {
        return;
    };
    let Ok((tank, tank_transform)) = tanks.get(arrow.tank) else {
        return;
    };
    let Ok((camera, cam_transform)) = camera_q.single() else {
        return;
    };

    let origin_world = tank_transform.translation;
    // Tanks grow in the -Y direction (bottom node offset = -length * Y).
    // Flip this if that ever changes.
    let grow_world = origin_world + Vec3::NEG_Y;
    let Ok(origin_screen) = camera.world_to_viewport(cam_transform, origin_world) else {
        return;
    };
    let Ok(grow_screen) = camera.world_to_viewport(cam_transform, grow_world) else {
        return;
    };

    let axis = grow_screen - origin_screen;
    let axis_len = axis.length();
    if axis_len < 1e-3 {
        return;
    }

    drag.active = Some(TankDragState {
        tank: arrow.tank,
        start_length: tank.length,
        start_cursor: event.pointer_location.position,
        screen_axis: axis / axis_len,
        world_per_pixel: 1.0 / axis_len,
    });
}

fn on_arrow_drag_end(_trigger: On<Pointer<DragEnd>>, mut drag: ResMut<TankResizeDrag>) {
    drag.active = None;
}

/// Apply the active drag to the tank's length each frame. Bails (and
/// clears) if the button was released without a DragEnd — can happen when
/// the pointer leaves the window mid-drag.
fn update_tank_resize_drag(
    mut drag: ResMut<TankResizeDrag>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut tanks: Query<&mut FuelTank>,
) {
    let Some(state) = drag.active.as_ref() else {
        return;
    };
    if !mouse.pressed(MouseButton::Left) {
        drag.active = None;
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Some(cursor) = window.cursor_position() else {
        return;
    };

    let cursor_delta = cursor - state.start_cursor;
    let pixels_along = cursor_delta.dot(state.screen_axis);
    let world_growth = pixels_along * state.world_per_pixel;
    let raw_length = state.start_length + world_growth;
    // Magnetic snap: smooth drag in-between, stick to nearest 0.5 within
    // a small neighborhood so users can dial in round values without
    // losing fine control.
    const SNAP_GRID: f32 = 0.5;
    const SNAP_THRESHOLD: f32 = 0.06;
    let nearest = (raw_length / SNAP_GRID).round() * SNAP_GRID;
    let length = if (raw_length - nearest).abs() < SNAP_THRESHOLD {
        nearest
    } else {
        raw_length
    };
    let new_length = length.clamp(0.5, 12.0);

    if let Ok(mut tank) = tanks.get_mut(state.tank) {
        if (tank.length - new_length).abs() > f32::EPSILON {
            tank.length = new_length;
        }
    }
}

/// Clear selection when the user clicks on empty space. Tracks the
/// press cursor so a camera orbit (press → drag → release) doesn't
/// deselect at release.
fn deselect_on_empty_click(
    mut tracker: ResMut<DeselectTracker>,
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    hover_map: Res<HoverMap>,
    pickables: Query<(), Or<(With<PartBody>, With<AttachNodePin>, With<TankResizeArrow>)>>,
    mut state: ResMut<EditorState>,
    mut contexts: EguiContexts,
) {
    const CLICK_THRESHOLD_PX: f32 = 4.0;

    let Ok(window) = windows.single() else {
        return;
    };
    let cursor = window.cursor_position();

    if mouse.just_pressed(MouseButton::Left) {
        if pointer_over_egui(&mut contexts) {
            tracker.press_cursor = None;
        } else {
            let on_pickable = hover_map
                .0
                .values()
                .any(|hovers| hovers.keys().any(|e| pickables.get(*e).is_ok()));
            tracker.press_cursor = if on_pickable { None } else { cursor };
        }
    }

    if mouse.just_released(MouseButton::Left) {
        if let (Some(press), Some(current)) = (tracker.press_cursor.take(), cursor) {
            if (current - press).length() < CLICK_THRESHOLD_PX {
                state.selected = None;
            }
        }
    }
}

/// Keep `ShipPartMaterial` uniforms in sync with the part's dimensions
/// (tank length, decoupler/tank radius). Triggered whenever the
/// kind-component or attach nodes change, so slider and resize-drag
/// updates flow through to the panel / rivet layout live.
fn update_part_shader_params(
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

/// Selection / hover tint for parts rendering through `ShipPartMaterial`
/// (tanks, decouplers). Writes into the material's tint uniform rather
/// than swapping handles so each part keeps its procedural detail.
/// Shrouds are excluded — they manage their own hover feedback via
/// `update_shroud_transparency`.
fn update_part_shader_highlight(
    state: Res<EditorState>,
    hover_map: Res<HoverMap>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    bodies: Query<
        (Entity, &PartBody, &MeshMaterial3d<ShipPartMaterial>),
        Without<ShroudBody>,
    >,
) {
    let hovered: HashSet<Entity> = hover_map
        .0
        .values()
        .flat_map(|hovers| hovers.keys().copied())
        .collect();

    for (body_entity, body, mesh_mat) in bodies.iter() {
        let target = if Some(body.0) == state.selected {
            Vec3::new(0.88, 1.0, 1.35)
        } else if hovered.contains(&body_entity) {
            Vec3::new(1.08, 1.08, 1.12)
        } else {
            Vec3::ONE
        };
        if let Some(mat) = ship_materials.get_mut(&mesh_mat.0) {
            if (mat.extension.params.tint - target).length_squared() > 1.0e-6 {
                mat.extension.params.tint = target;
            }
        }
    }
}

/// Swap each part body's material based on selection and hover state.
/// Priority: selected > hovered > default.
fn update_selection_highlight(
    state: Res<EditorState>,
    assets: Res<EditorAssets>,
    hover_map: Res<HoverMap>,
    mut bodies: Query<(Entity, &PartBody, &mut MeshMaterial3d<StandardMaterial>)>,
) {
    let hovered: HashSet<Entity> = hover_map
        .0
        .values()
        .flat_map(|hovers| hovers.keys().copied())
        .collect();

    for (body_entity, body, mut mat) in bodies.iter_mut() {
        let target = if Some(body.0) == state.selected {
            &assets.selected_material
        } else if hovered.contains(&body_entity) {
            &assets.hover_material
        } else {
            &assets.part_material
        };
        if mat.0.id() != target.id() {
            mat.0 = target.clone();
        }
    }
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
    resize_drag: Res<TankResizeDrag>,
    hover_map: Res<HoverMap>,
    arrows: Query<(), With<TankResizeArrow>>,
) {
    let pointer_over_egui = contexts
        .ctx_mut()
        .map(|c| c.is_pointer_over_area() || c.wants_pointer_input())
        .unwrap_or(false);

    let pointer_on_arrow = hover_map
        .0
        .values()
        .any(|hovers| hovers.keys().any(|e| arrows.get(*e).is_ok()));

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
    // event into the window fallback entity. Also suppress while the
    // pointer is over a resize arrow (or actively dragging one) so the
    // camera doesn't twitch between mouse-down and DragStart firing.
    let orbit_allowed = !pointer_over_egui
        && state.pending.is_none()
        && resize_drag.active.is_none()
        && !pointer_on_arrow;

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
                diameter: d.diameter,
                ejection_impulse: d.ejection_impulse,
                dry_mass: d.dry_mass,
            }
        } else if let Some(a) = adapter {
            PartData::Adapter {
                diameter: a.diameter,
                target_diameter: a.target_diameter,
                dry_mass: a.dry_mass,
            }
        } else if let Some(t) = tank {
            PartData::FuelTank {
                diameter: t.diameter,
                length: t.length,
                dry_mass: t.dry_mass,
            }
        } else if let Some(en) = engine {
            PartData::Engine {
                model: en.model.clone(),
                diameter: en.diameter,
                thrust: en.thrust,
                isp: en.isp,
                dry_mass: en.dry_mass,
                reactants: en.reactants.clone(),
                power_draw_kw: en.power_draw_kw,
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
    mut ships: Query<&mut Ship>,
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
                        let (root, name) = world
                            .get::<Ship>(ship_entity)
                            .map(|s| (Some(s.root), s.name.clone()))
                            .unwrap_or((None, String::new()));
                        let mut st = world.resource_mut::<EditorState>();
                        st.ship_entity = Some(ship_entity);
                        st.ship_root = root;
                        st.selected = root;
                        st.ship_name = name;
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

    // ---- Delete selected ---------------------------------------------
    // Deleting the root clears the whole canvas (despawns ship + all
    // parts). Deleting a non-root part despawns its subtree.
    if state.delete_selected {
        state.delete_selected = false;
        if let Some(sel) = state.selected {
            if Some(sel) == state.ship_root {
                if let Some(se) = state.ship_entity {
                    if let Ok(ship) = ships.get(se) {
                        state.ship_name = ship.name.clone();
                    }
                }
                for e in all_parts.iter() {
                    commands.entity(e).despawn();
                }
                for e in all_ships.iter() {
                    commands.entity(e).despawn();
                }
                state.ship_root = None;
                state.ship_entity = None;
                state.selected = None;
                state.status = "Cleared canvas".into();
            } else {
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
            }
        }
    }

    // ---- Set selection as root ---------------------------------------
    // Walk from the selection up through Attachment components to the
    // current root; reverse each link by inserting an Attachment on the
    // former parent pointing at the former child, with parent_node /
    // my_node swapped. Parts off the chain keep their attachments, so
    // branches follow their original subtree.
    if state.set_as_root {
        state.set_as_root = false;
        if let Some(sel) = state.selected {
            if Some(sel) != state.ship_root {
                let att_map: HashMap<Entity, Attachment> = attachments
                    .iter()
                    .map(|(e, a)| (e, a.clone()))
                    .collect();
                let mut chain: Vec<(Entity, Attachment)> = Vec::new();
                let mut current = sel;
                while let Some(att) = att_map.get(&current) {
                    chain.push((current, att.clone()));
                    current = att.parent;
                }
                commands.entity(sel).remove::<Attachment>();
                for (entity, att) in chain {
                    commands.entity(att.parent).insert(Attachment {
                        parent: entity,
                        parent_node: att.my_node,
                        my_node: att.parent_node,
                    });
                }
                if let Some(ship_entity) = state.ship_entity {
                    if let Ok(mut ship) = ships.get_mut(ship_entity) {
                        ship.root = sel;
                    }
                }
                state.ship_root = Some(sel);
                state.status = "Re-rooted ship".into();
            }
        }
    }

    // ---- Place pending part at a given (parent, node) -----------------
    if let Some((parent, node)) = state.place_at.take() {
        let Some(pending) = state.pending.take() else {
            return;
        };
        let resources = blueprint::default_resources_for(&pending);
        let child = ShipBlueprint::spawn_part(&mut commands, &pending, resources);
        commands.entity(child).insert(Attachment {
            parent,
            parent_node: node,
            my_node: "top".into(),
        });
        state.selected = Some(child);
        state.status = "Placed part".into();
    }

    // ---- Auto-place pending as root on empty canvas ------------------
    if state.ship_root.is_none() && state.pending.is_some() {
        let pending = state.pending.take().unwrap();
        let resources = blueprint::default_resources_for(&pending);
        let part = ShipBlueprint::spawn_part(&mut commands, &pending, resources);
        let ship = commands
            .spawn(Ship {
                name: state.ship_name.clone(),
                root: part,
            })
            .id();
        state.ship_root = Some(part);
        state.ship_entity = Some(ship);
        state.selected = Some(part);
        state.status = "Placed root".into();
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
    mut sky: ResMut<SkyBackdropEnabled>,
    mut clear_color: ResMut<ClearColor>,
    diagnostics: Res<DiagnosticsStore>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let ctx = ctx.clone();

    // -------- Left palette --------
    egui::SidePanel::left("palette")
        .default_width(180.0)
        .show(&ctx, |ui| {
            let fps = diagnostics
                .get(&FrameTimeDiagnosticsPlugin::FPS)
                .and_then(|d| d.smoothed())
                .unwrap_or(0.0);
            ui.label(format!("FPS: {:.0}", fps));
            ui.separator();
            ui.heading("Parts");
            if ui.button("Command Pod (1.25m)").clicked() {
                state.pending = Some(PartData::CommandPod {
                    model: "Mk1".into(),
                    diameter: 1.25,
                    dry_mass: 840.0,
                });
            }
            if ui.button("Command Pod (2.5m)").clicked() {
                state.pending = Some(PartData::CommandPod {
                    model: "Mk1-3".into(),
                    diameter: 2.5,
                    dry_mass: 2720.0,
                });
            }
            if ui.button("Decoupler").clicked() {
                state.pending = Some(PartData::Decoupler {
                    diameter: 1.25,
                    ejection_impulse: 250.0,
                    dry_mass: 50.0,
                });
            }
            if ui.button("Adapter").clicked() {
                state.pending = Some(PartData::Adapter {
                    diameter: 1.25,
                    target_diameter: 2.5,
                    dry_mass: 100.0,
                });
            }
            if ui.button("Fuel Tank").clicked() {
                state.pending = Some(PartData::FuelTank {
                    diameter: 1.25,
                    length: 1.0,
                    dry_mass: 250.0,
                });
            }
            if ui.button("Engine (2.5m)").clicked() {
                state.pending = Some(PartData::Engine {
                    model: "Poodle".into(),
                    diameter: 2.5,
                    thrust: 250_000.0,
                    isp: 350.0,
                    dry_mass: 250.0,
                    reactants: methalox_reactants(),
                    power_draw_kw: 0.0,
                });
            }
            if ui.button("Engine (1.25m)").clicked() {
                state.pending = Some(PartData::Engine {
                    model: "LV-909".into(),
                    diameter: 1.25,
                    thrust: 60_000.0,
                    isp: 345.0,
                    dry_mass: 50.0,
                    reactants: methalox_reactants(),
                    power_draw_kw: 0.0,
                });
            }

            ui.separator();
            ui.heading("Ship");
            ui.horizontal(|ui| {
                ui.label("Name:");
                if let Some(se) = state.ship_entity {
                    if let Ok(mut ship) = ships.get_mut(se) {
                        ui.text_edit_singleline(&mut ship.name);
                    }
                } else {
                    ui.text_edit_singleline(&mut state.ship_name);
                }
            });
            ui.add_enabled_ui(state.ship_entity.is_some(), |ui| {
                if ui.button("Save").clicked() {
                    state.save_requested = true;
                }
            });
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
            ui.heading("View");
            if ui.checkbox(&mut sky.0, "Celestial backdrop").changed() {
                // Black clears behind the additively-blended stars so
                // they read as points of light; the default grey washes
                // them out.
                clear_color.0 = if sky.0 {
                    Color::BLACK
                } else {
                    ClearColor::default().0
                };
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
            let is_root = Some(sel) == state.ship_root;

            if let Some(p) = pod.as_deref_mut() {
                ui.label("Kind: CommandPod");
                ui.label(format!("Model: {}", p.model));
                ui.label(format!("Diameter: {:.2}m (fixed)", p.diameter));
                ui.label(format!("Dry mass: {:.0} kg (fixed)", p.dry_mass));
            } else if let Some(d) = dec.as_deref_mut() {
                ui.label("Kind: Decoupler");
                if is_root {
                    ui.add(
                        egui::Slider::new(&mut d.diameter, 0.3..=6.0).text("Diameter"),
                    );
                } else {
                    ui.label(format!("Diameter: {:.2}m (from parent)", d.diameter));
                }
                ui.add(
                    egui::Slider::new(&mut d.ejection_impulse, 0.0..=2000.0)
                        .text("Ejection impulse"),
                );
                ui.label(format!("Dry mass: {:.0} kg (fixed)", d.dry_mass));
            } else if let Some(a) = adapter.as_deref_mut() {
                ui.label("Kind: Adapter");
                if is_root {
                    ui.add(egui::Slider::new(&mut a.diameter, 0.3..=6.0).text("Diameter"));
                } else {
                    ui.label(format!("Diameter: {:.2}m (from parent)", a.diameter));
                }
                ui.add(
                    egui::Slider::new(&mut a.target_diameter, 0.3..=6.0).text("Target diameter"),
                );
                ui.label(format!("Dry mass: {:.0} kg (fixed)", a.dry_mass));
            } else if let Some(t) = tank.as_deref_mut() {
                ui.label("Kind: Fuel Tank");
                if is_root {
                    ui.add(egui::Slider::new(&mut t.diameter, 0.3..=6.0).text("Diameter"));
                } else {
                    ui.label(format!("Diameter: {:.2}m (from parent)", t.diameter));
                }
                ui.add(egui::Slider::new(&mut t.length, 0.5..=12.0).text("Length"));
                ui.label(format!("Dry mass: {:.0} kg (fixed)", t.dry_mass));
            } else if let Some(e) = engine.as_deref_mut() {
                ui.label("Kind: Engine (vacuum)");
                ui.label(format!("Model: {}", e.model));
                ui.label(format!("Diameter: {:.2}m (fixed)", e.diameter));
                ui.label(format!("Thrust: {:.1} kN (fixed)", e.thrust / 1000.0));
                ui.label(format!("Isp: {:.0} s (fixed)", e.isp));
                ui.label(format!("Dry mass: {:.0} kg (fixed)", e.dry_mass));
                if e.power_draw_kw > 0.0 {
                    ui.label(format!("Power draw: {:.1} kW (fixed)", e.power_draw_kw));
                }
                ui.label("Reactants:");
                for r in &e.reactants {
                    ui.label(format!(
                        "  {}: {:.1}%",
                        r.resource.display_name(),
                        r.mass_fraction * 100.0,
                    ));
                }
            }

            ui.separator();
            ui.label("Attach nodes:");
            for (id, node) in &nodes.nodes {
                ui.label(format!("  {id}: Ø{:.2}m", node.diameter));
            }

            ui.separator();
            ui.label("Resources:");
            if let Some(r) = res.as_deref_mut() {
                for (resource, pool) in r.pools.iter_mut() {
                    ui.label(format!(
                        "{}: {:.0}/{:.0} {}",
                        resource.display_name(),
                        pool.amount,
                        pool.capacity,
                        resource.unit_label(),
                    ));
                    ui.add(egui::Slider::new(&mut pool.amount, 0.0..=pool.capacity).text("amount"));
                }
                if r.pools.is_empty() {
                    ui.label("  (none)");
                }
            }

            ui.separator();
            ui.add_enabled_ui(!is_root, |ui| {
                if ui.button("Set as root").clicked() {
                    state.set_as_root = true;
                }
            });
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

// ---------------------------------------------------------------------------
// Celestial backdrop
// ---------------------------------------------------------------------------
//
// Duplicated from `thalos_game::sky_render` with the game-specific bits
// (CameraExposure, SimStage, OrbitCamera) stripped out. Keep until sky
// rendering is extracted into its own crate.

#[derive(Resource, Default)]
struct SkyBackdropEnabled(bool);

#[derive(Component)]
struct SkyBackdrop;

#[derive(Clone, Copy, ShaderType)]
struct StarsParams {
    pixel_radius: f32,
    brightness: f32,
    size_gamma: f32,
    _pad0: f32,
}

impl Default for StarsParams {
    fn default() -> Self {
        Self { pixel_radius: 4.0, brightness: 140.0, size_gamma: 0.50, _pad0: 0.0 }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct StarsMaterial {
    #[uniform(0)]
    params: StarsParams,
}

impl Material for StarsMaterial {
    fn vertex_shader() -> ShaderRef { "shaders/stars.wgsl".into() }
    fn fragment_shader() -> ShaderRef { "shaders/stars.wgsl".into() }
    fn prepass_vertex_shader() -> ShaderRef { "shaders/stars_prepass.wgsl".into() }
    fn prepass_fragment_shader() -> ShaderRef { "shaders/stars_prepass.wgsl".into() }
    fn alpha_mode(&self) -> AlphaMode { AlphaMode::Add }

    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(1),
            Mesh::ATTRIBUTE_COLOR.at_shader_location(2),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = false;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, ShaderType)]
struct GalaxyParams {
    pixel_radius_scale: f32,
    min_pixel_radius: f32,
    brightness: f32,
    _pad0: f32,
}

impl Default for GalaxyParams {
    fn default() -> Self {
        Self {
            pixel_radius_scale: 2000.0,
            min_pixel_radius: 1.2,
            brightness: 1_500.0,
            _pad0: 0.0,
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct GalaxyMaterial {
    #[uniform(0)]
    params: GalaxyParams,
}

impl Material for GalaxyMaterial {
    fn vertex_shader() -> ShaderRef { "shaders/galaxy.wgsl".into() }
    fn fragment_shader() -> ShaderRef { "shaders/galaxy.wgsl".into() }
    fn prepass_vertex_shader() -> ShaderRef { "shaders/galaxy_prepass.wgsl".into() }
    fn prepass_fragment_shader() -> ShaderRef { "shaders/galaxy_prepass.wgsl".into() }
    fn alpha_mode(&self) -> AlphaMode { AlphaMode::Add }

    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(1),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(2),
            Mesh::ATTRIBUTE_TANGENT.at_shader_location(3),
            Mesh::ATTRIBUTE_COLOR.at_shader_location(4),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = false;
        }
        Ok(())
    }
}

struct SkyBackdropPlugin;

impl Plugin for SkyBackdropPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<StarsMaterial>::default())
            .add_plugins(MaterialPlugin::<GalaxyMaterial>::default())
            .add_systems(Startup, spawn_sky_backdrop)
            .add_systems(Update, (update_sky_visibility, update_galaxy_uniform));
    }
}

fn spawn_sky_backdrop(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut stars_materials: ResMut<Assets<StarsMaterial>>,
    mut galaxy_materials: ResMut<Assets<GalaxyMaterial>>,
) {
    let universe = generate_default(&DefaultGenParams::default());

    commands.spawn((
        SkyBackdrop,
        Mesh3d(meshes.add(build_star_mesh(&universe))),
        MeshMaterial3d(stars_materials.add(StarsMaterial { params: StarsParams::default() })),
        Transform::IDENTITY,
        Visibility::Hidden,
        NoFrustumCulling,
    ));

    commands.spawn((
        SkyBackdrop,
        Mesh3d(meshes.add(build_galaxy_mesh(&universe))),
        MeshMaterial3d(galaxy_materials.add(GalaxyMaterial { params: GalaxyParams::default() })),
        Transform::IDENTITY,
        Visibility::Hidden,
        NoFrustumCulling,
    ));
}

fn update_sky_visibility(
    enabled: Res<SkyBackdropEnabled>,
    mut q: Query<&mut Visibility, With<SkyBackdrop>>,
) {
    let target = if enabled.0 { Visibility::Inherited } else { Visibility::Hidden };
    for mut v in q.iter_mut() {
        if *v != target {
            *v = target;
        }
    }
}

fn update_galaxy_uniform(
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<&Projection, With<Camera3d>>,
    handles: Query<&MeshMaterial3d<GalaxyMaterial>>,
    mut materials: ResMut<Assets<GalaxyMaterial>>,
) {
    let Ok(window) = windows.single() else { return };
    let Ok(projection) = cameras.single() else { return };
    let Projection::Perspective(p) = projection else { return };
    let px_per_rad = window.resolution.physical_height() as f32 / p.fov;

    for handle in &handles {
        if let Some(mat) = materials.get_mut(&handle.0) {
            mat.params.pixel_radius_scale = px_per_rad;
        }
    }
}

fn build_star_mesh(universe: &Universe) -> Mesh {
    let n = universe.stars.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 6);

    const CORNERS: [[f32; 2]; 4] =
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

    for (i, star) in universe.stars.iter().enumerate() {
        let dir = star.position.normalize();
        let rgb = star.linear_srgb();
        let flux = star.magnitude_flux();
        for corner in CORNERS {
            positions.push([dir.x, dir.y, dir.z]);
            uvs.push(corner);
            colors.push([rgb[0], rgb[1], rgb[2], flux]);
        }
        let base = (i * 4) as u32;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn build_galaxy_mesh(universe: &Universe) -> Mesh {
    let n = universe.galaxies.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n * 4);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut tangents: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 6);

    const CORNERS: [[f32; 2]; 4] =
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

    for (i, galaxy) in universe.galaxies.iter().enumerate() {
        let dir = galaxy.position.normalize();
        let rgb = galaxy.linear_srgb();
        let flux = galaxy.magnitude_flux();
        let (sin_pa, cos_pa) = galaxy.position_angle_rad.sin_cos();
        for corner in CORNERS {
            positions.push([dir.x, dir.y, dir.z]);
            uvs.push(corner);
            normals.push([galaxy.effective_radius_rad, galaxy.sersic_n, 0.0]);
            tangents.push([galaxy.axis_ratio, cos_pa, sin_pa, 0.0]);
            colors.push([rgb[0], rgb[1], rgb[2], flux]);
        }
        let base = (i * 4) as u32;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

// ---------------------------------------------------------------------------
// Shrouds (auto-generated cones wrapping a [`Shroudable`] above a
// [`ShroudProvider`])
// ---------------------------------------------------------------------------

/// Attached to a shroud entity — a mesh child of the provider
/// (e.g. a decoupler) that wraps the shrouded part above. Spawned and
/// reconciled by [`sync_shrouds`]; not part of the persisted blueprint
/// and not user-spawnable.
#[derive(Component, Debug, Clone, Copy)]
struct Shroud {
    provider: Entity,
    shrouded: Entity,
    // Cached spec, compared each frame so we only rebuild the mesh /
    // material when geometry actually changed.
    bottom_radius: f32,
    top_radius: f32,
    height: f32,
}

/// Marker on the shroud entity's body. Kept distinct from [`PartBody`] so
/// part-level highlight systems (tint, material swap) don't fire on
/// hovered shrouds — the shroud manages its own hover feedback
/// (transparency) in [`update_shroud_transparency`].
#[derive(Component, Debug, Clone, Copy)]
struct ShroudBody;

/// Expected geometry for a shroud covering a given attachment. `None`
/// when no shroud should exist for this pair (misconfigured attachment,
/// shrouded part missing [`Shroudable`], or provider not wider than the
/// shrouded's top — the cone would degenerate).
struct ShroudSpec {
    bottom_radius: f32,
    top_radius: f32,
    height: f32,
    shrouded: Entity,
}

fn compute_shroud_spec(
    attachment: &Attachment,
    provider_nodes: &AttachNodes,
    shroudables: &Query<(&Engine, Has<Shroudable>)>,
) -> Option<ShroudSpec> {
    // Only the canonical "provider sits below shroudable" orientation
    // gets a shroud: provider's `top` mates with shroudable's `bottom`.
    if attachment.my_node != "top" || attachment.parent_node != "bottom" {
        return None;
    }
    let (engine, is_shroudable) = shroudables.get(attachment.parent).ok()?;
    if !is_shroudable {
        return None;
    }
    let provider_top_d = provider_nodes.get("top")?.diameter;
    // Shroud top matches the shrouded part's *attach* diameter — the
    // interface the stage above would mate with. That sits outside the
    // engine's narrowing visual silhouette, so the shroud stays clear
    // of the engine body instead of hugging (and z-fighting) it.
    let bottom_r = provider_top_d * 0.5;
    let top_r = engine.diameter * 0.5;
    let (_, _, height) = engine_visual_profile(engine.diameter);
    // Only generate when the provider is at least as wide as the
    // shrouded part at its top — a narrower provider would invert the
    // cone. Equal diameter gives a clean cylindrical interstage.
    if bottom_r + 1.0e-4 < top_r {
        return None;
    }
    Some(ShroudSpec {
        bottom_radius: bottom_r,
        top_radius: top_r,
        height,
        shrouded: attachment.parent,
    })
}

fn spec_matches(s: &Shroud, spec: &ShroudSpec) -> bool {
    s.shrouded == spec.shrouded
        && (s.bottom_radius - spec.bottom_radius).abs() < 1.0e-4
        && (s.top_radius - spec.top_radius).abs() < 1.0e-4
        && (s.height - spec.height).abs() < 1.0e-4
}

/// Reconcile shroud entities against current attachment state: spawn
/// missing shrouds, update ones whose geometry changed, and despawn
/// orphans. Idempotent per frame; cheap when attachment is stable.
fn sync_shrouds(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    providers: Query<(Entity, &Attachment, &AttachNodes), With<ShroudProvider>>,
    shroudables: Query<(&Engine, Has<Shroudable>)>,
    existing: Query<(Entity, &Shroud)>,
) {
    // Map provider -> (shroud_entity, current Shroud component).
    let mut current_by_provider: HashMap<Entity, (Entity, Shroud)> = HashMap::new();
    for (entity, shroud) in existing.iter() {
        current_by_provider.insert(shroud.provider, (entity, *shroud));
    }

    let mut kept: HashSet<Entity> = HashSet::new();

    for (provider, attachment, provider_nodes) in providers.iter() {
        let Some(spec) = compute_shroud_spec(attachment, provider_nodes, &shroudables) else {
            continue;
        };
        kept.insert(provider);

        // Reuse in-place if the cached spec still matches.
        if let Some((_, current)) = current_by_provider.get(&provider) {
            if spec_matches(current, &spec) {
                continue;
            }
        }
        if let Some((old, _)) = current_by_provider.get(&provider) {
            commands.entity(*old).despawn();
        }

        let shroud_mesh: Mesh = ConicalFrustum {
            radius_top: spec.top_radius,
            radius_bottom: spec.bottom_radius,
            height: spec.height,
        }
        .mesh()
        .resolution(PART_RESOLUTION)
        .into();
        let mesh_handle = meshes.add(shroud_mesh);

        // Slant length — the actual surface distance v = 0 → v = 1.
        // Matches the vertical height only when the two radii agree.
        let dr = spec.bottom_radius - spec.top_radius;
        let slant_length = (spec.height * spec.height + dr * dr).sqrt();
        // Blend mode is set once here; we only vary base-color alpha
        // from the hover system so the pipeline stays hot.
        let material = ship_materials.add(ShipPartMaterial {
            base: StandardMaterial {
                alpha_mode: AlphaMode::Blend,
                ..stainless_steel_base()
            },
            extension: ShipPartExtension {
                params: ShipPartParams {
                    length: slant_length,
                    radius_top: spec.top_radius,
                    radius_bottom: spec.bottom_radius,
                    // Mix provider index with a fixed mask so shroud
                    // detail doesn't look identical to the decoupler's.
                    seed: provider.index_u32() ^ 0x5A5A_5A5A,
                    ..default()
                },
            },
        });

        // Shroud mesh center sits at +height/2 in the provider's local
        // frame, since the provider's "top" node (y = 0) meets the
        // shrouded's base and the shroud extends upward from there.
        let shroud_entity = commands
            .spawn((
                Mesh3d(mesh_handle),
                MeshMaterial3d(material),
                Transform::from_xyz(0.0, spec.height * 0.5, 0.0),
                Visibility::default(),
                Shroud {
                    provider,
                    shrouded: spec.shrouded,
                    bottom_radius: spec.bottom_radius,
                    top_radius: spec.top_radius,
                    height: spec.height,
                },
                ShroudBody,
                Pickable::default(),
            ))
            .observe(on_shroud_click)
            .id();
        commands.entity(provider).add_child(shroud_entity);
    }

    // Despawn shrouds whose provider no longer qualifies (detachment,
    // geometry change below threshold, shrouded part removed, etc.).
    for (provider, (entity, _)) in &current_by_provider {
        if !kept.contains(provider) {
            commands.entity(*entity).despawn();
        }
    }
}

/// Drive the shroud's base-color alpha from hover: opaque by default
/// (engine hidden inside), partial transparency while hovered so the
/// shrouded silhouette reads through.
fn update_shroud_transparency(
    hover_map: Res<HoverMap>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    shrouds: Query<(Entity, &MeshMaterial3d<ShipPartMaterial>), With<ShroudBody>>,
) {
    let hovered: HashSet<Entity> = hover_map
        .0
        .values()
        .flat_map(|hovers| hovers.keys().copied())
        .collect();

    for (entity, mesh_mat) in shrouds.iter() {
        let target_alpha: f32 = if hovered.contains(&entity) { 0.18 } else { 1.0 };
        let Some(mat) = ship_materials.get_mut(&mesh_mat.0) else {
            continue;
        };
        let srgba = mat.base.base_color.to_srgba();
        if (srgba.alpha - target_alpha).abs() > 1.0e-3 {
            mat.base.base_color = Color::srgba(srgba.red, srgba.green, srgba.blue, target_alpha);
        }
    }
}

/// Click on a shroud selects the provider that owns it — the shroud is
/// a visual extension of the decoupler, not an independent part.
fn on_shroud_click(
    click: On<Pointer<Click>>,
    shrouds: Query<&Shroud>,
    mut state: ResMut<EditorState>,
    mut contexts: EguiContexts,
) {
    if pointer_over_egui(&mut contexts) {
        return;
    }
    if let Ok(shroud) = shrouds.get(click.entity) {
        state.selected = Some(shroud.provider);
    }
}

/// Propagate the coupled neighbor's [`MaterialKind`] onto parts that
/// visually continue with whatever is attached to their `bottom` node —
/// currently [`Decoupler`] (so the decoupler + its shroud read as part
/// of the stage below on staging) and [`Adapter`] (so a diameter
/// transition inherits from the narrower stage it feeds into). Parts
/// with nothing attached below keep their default [`MaterialKind`].
fn propagate_coupled_material(
    attachments: Query<(Entity, &Attachment)>,
    mut params: ParamSet<(
        Query<(Entity, &PartMaterial)>,
        Query<
            (Entity, &mut PartMaterial),
            Or<(With<Decoupler>, With<Adapter>)>,
        >,
    )>,
) {
    // Build parent → bottom-attached-child entity map.
    let mut coupled: HashMap<Entity, Entity> = HashMap::new();
    for (child, att) in attachments.iter() {
        if att.parent_node == "bottom" {
            coupled.insert(att.parent, child);
        }
    }

    // Snapshot every part's current MaterialKind so read + write on
    // PartMaterial can both run in this system without conflicting
    // mutable borrows.
    let kinds: HashMap<Entity, MaterialKind> =
        params.p0().iter().map(|(e, m)| (e, m.kind)).collect();

    for (entity, mut my_mat) in params.p1().iter_mut() {
        let Some(coupled_entity) = coupled.get(&entity).copied() else {
            continue;
        };
        let Some(&kind) = kinds.get(&coupled_entity) else {
            continue;
        };
        if my_mat.kind != kind {
            my_mat.kind = kind;
        }
    }
}
