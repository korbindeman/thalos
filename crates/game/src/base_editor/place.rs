//! Structure placement (the `PlaceBuildings` phase): a ghost tracks the cursor
//! on the active site's flattened pad, snapped to a grid and rotatable with Q/E.
//! **Tab** toggles the pending kind between a **building** (box) and a
//! **launchpad** (circular slab). Left-click on empty pad places it; left-click
//! on an existing structure selects it; X / Delete removes the selected one;
//! `[ ]` / `- =` resize the pending footprint/height. With a launchpad selected,
//! **L** launches — places the player ship on the pad and closes the editor.
//!
//! A placed structure is a [`StructureKind`] record (draped on the pad, no
//! terrain modification of its own) plus a visual anchored every frame in the
//! body-fixed frame exactly like `runway::update_runway_transform` — a root-grid
//! big_space child re-placed in f64, so it stays rock-steady at high warp.

use bevy::camera::visibility::RenderLayers;
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_physics_canonical::body_fixed::{body_fixed_pose_from_inertial, body_fixed_surface_velocity};
use thalos_physics_canonical::canonical::{AuthorityMode, TranslationalState};
use thalos_physics_canonical::types::BodyState;
use thalos_physics_local::ActiveLocalBubble;
use thalos_world::{BodyId, StateVector};

use crate::camera::{ActiveCamera, ShipCamera};
use crate::coords::{SHIP_LAYER, SHIP_SCALE};
use crate::rendering::real_space::RealSpaceRoot;
use crate::rendering::{PlayerShip, RealSpaceBody, SimulationState, SolarSystemState};
use crate::runway::{craft_ground_clearance, level_heading_attitude};
use crate::structures::{StructureId, StructureKind, StructurePlacement, StructureRegistry};

use super::{BaseEditor, BaseEditorMode, base_editor_open, ray_vs_sphere_dir};

/// Grid step for snapping placement, metres.
const GRID_STEP_M: f64 = 2.0;
/// Rotation step per Q/E press, radians (15°).
const ROTATE_STEP: f32 = std::f32::consts::PI / 12.0;
/// Thickness of the launchpad slab, metres.
const LAUNCHPAD_SLAB_H: f32 = 0.5;
/// Margin the launched craft's lowest point clears the pad top by, metres.
const LAUNCH_REST_MARGIN_M: f64 = 0.05;

/// Footprint + height of a building. The default is a modest hab-block.
#[derive(Clone, Copy, Debug)]
pub struct BuildingDims {
    pub half_x_m: f32,
    pub half_z_m: f32,
    pub height_m: f32,
}

impl Default for BuildingDims {
    fn default() -> Self {
        Self {
            half_x_m: 6.0,
            half_z_m: 6.0,
            height_m: 8.0,
        }
    }
}

/// Which kind of structure the next click places.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum PendingKind {
    #[default]
    Building,
    Launchpad,
}

/// Placement state. Public so the editor UI/overlay can read the pending kind +
/// footprint and the current selection.
#[derive(Resource)]
pub struct BaseBuildState {
    pub pending: BuildingDims,
    pub pending_kind: PendingKind,
    pub pending_radius_m: f32,
    pub selected: Option<StructureId>,
    yaw: f32,
    hover: Option<HoverPad>,
    material: Option<Handle<StandardMaterial>>,
    pad_material: Option<Handle<StandardMaterial>>,
    ring_material: Option<Handle<StandardMaterial>>,
}

impl Default for BaseBuildState {
    fn default() -> Self {
        Self {
            pending: BuildingDims::default(),
            pending_kind: PendingKind::Building,
            pending_radius_m: 18.0,
            selected: None,
            yaw: 0.0,
            hover: None,
            material: None,
            pad_material: None,
            ring_material: None,
        }
    }
}

#[derive(Clone, Copy)]
struct HoverPad {
    /// Body-fixed unit direction to the (grid-snapped) placement point.
    building_dir: DVec3,
}

/// Visual + anchor data for a placed structure. Re-placed each frame in the
/// body-fixed frame by [`update_placed_transforms`]. Field visibility is
/// `pub(super)` so the connections layer can read structure positions.
#[derive(Component)]
pub(super) struct PlacedVisual {
    pub(super) structure_id: StructureId,
    pub(super) body_id: BodyId,
    /// Body-fixed position of the visual centre.
    pub(super) center_body: DVec3,
    /// Visual-local axes → body-fixed rotation.
    pub(super) basis_body: DQuat,
    pub(super) kind: StructureKind,
}

pub(super) struct BaseEditorPlacePlugin;

impl Plugin for BaseEditorPlacePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BaseBuildState>()
            .add_systems(
                Update,
                (update_structure_placement, launch_from_pad, draw_placement_ghost)
                    .chain()
                    .run_if(base_editor_open),
            )
            // Ungated so placed structures stay anchored in flight too (not just
            // while the editor is open).
            .add_systems(Update, update_placed_transforms);
    }
}

#[allow(clippy::too_many_arguments)]
fn update_structure_placement(
    mut editor: ResMut<BaseEditor>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    mut registry: ResMut<StructureRegistry>,
    mut state: ResMut<BaseBuildState>,
    ui_gate: Res<crate::hud::UiPointerGate>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    root: Res<RealSpaceRoot>,
    queries: (
        Query<&Window, With<PrimaryWindow>>,
        Query<(&Camera, &GlobalTransform), (With<ShipCamera>, With<ActiveCamera>)>,
        Query<(&RealSpaceBody, &GlobalTransform)>,
        Query<(Entity, &PlacedVisual)>,
    ),
) {
    if editor.mode != BaseEditorMode::PlaceBuildings {
        return;
    }
    let Some(site_id) = editor.active_site else {
        return;
    };
    let Some(site) = registry.get(site_id).copied() else {
        // Site vanished (e.g. deleted) — drop back to picking.
        editor.active_site = None;
        editor.mode = BaseEditorMode::PickSite;
        return;
    };
    let StructurePlacement::FlattenTo {
        elevation_m,
        half_along_m,
        half_across_m,
        ..
    } = site.placement
    else {
        return;
    };
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let body_id = site.body_id;
    let Some(body_state) = states.get(body_id) else {
        return;
    };
    let Some(body) = sim.system.bodies.get(body_id) else {
        return;
    };
    let radius_m = body.radius_m;
    let pad_r = radius_m + elevation_m;
    let center_dir = site.anchor_dir;
    let heading = site.heading_tangent;
    let across = center_dir.cross(heading).normalize();

    let (windows, cameras, bodies, placed_q) = &queries;

    // Tab toggles the pending kind.
    if keys.just_pressed(KeyCode::Tab) {
        state.pending_kind = match state.pending_kind {
            PendingKind::Building => PendingKind::Launchpad,
            PendingKind::Launchpad => PendingKind::Building,
        };
    }

    // Q/E rotate the pending heading (discrete steps).
    if keys.just_pressed(KeyCode::KeyQ) {
        state.yaw -= ROTATE_STEP;
    }
    if keys.just_pressed(KeyCode::KeyE) {
        state.yaw += ROTATE_STEP;
    }

    // [ ] / - = resize the pending structure. Footprint is bounded by the pad so
    // it always fits.
    let max_half = (half_along_m.min(half_across_m).max(GRID_STEP_M)) as f32;
    match state.pending_kind {
        PendingKind::Building => {
            if keys.just_pressed(KeyCode::BracketLeft) {
                state.pending.half_x_m = (state.pending.half_x_m - 1.0).max(1.0);
                state.pending.half_z_m = (state.pending.half_z_m - 1.0).max(1.0);
            }
            if keys.just_pressed(KeyCode::BracketRight) {
                state.pending.half_x_m = (state.pending.half_x_m + 1.0).min(max_half);
                state.pending.half_z_m = (state.pending.half_z_m + 1.0).min(max_half);
            }
            if keys.just_pressed(KeyCode::Minus) {
                state.pending.height_m = (state.pending.height_m - 1.0).max(2.0);
            }
            if keys.just_pressed(KeyCode::Equal) {
                state.pending.height_m = (state.pending.height_m + 1.0).min(60.0);
            }
        }
        PendingKind::Launchpad => {
            if keys.just_pressed(KeyCode::BracketLeft) {
                state.pending_radius_m = (state.pending_radius_m - 2.0).max(5.0);
            }
            if keys.just_pressed(KeyCode::BracketRight) {
                state.pending_radius_m = (state.pending_radius_m + 2.0).min(max_half);
            }
        }
    }

    // Cursor → grid-snapped pad point.
    let footprint_half = pending_footprint_half(&state);
    state.hover = compute_pad_hover(
        windows,
        cameras,
        bodies,
        body_id,
        body_state,
        center_dir,
        heading,
        across,
        pad_r,
        half_along_m,
        half_across_m,
        footprint_half,
    );

    // Delete the selected structure.
    if (keys.just_pressed(KeyCode::Delete) || keys.just_pressed(KeyCode::KeyX))
        && let Some(sel) = state.selected.take()
    {
        registry.remove(sel);
        for (entity, pv) in placed_q.iter() {
            if pv.structure_id == sel {
                commands.entity(entity).despawn();
            }
        }
        return;
    }

    if ui_gate.hovered || !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let Some(hover) = state.hover else {
        return;
    };

    // Click on an existing structure selects it; otherwise place a new one.
    if let Some(existing) = structure_under(placed_q, body_id, hover.building_dir, pad_r) {
        state.selected = Some(existing);
        return;
    }

    let new_id = spawn_structure(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut registry,
        &mut state,
        root.entity,
        site_id,
        body_id,
        hover.building_dir,
        heading,
        across,
        pad_r,
    );
    state.selected = Some(new_id);
}

/// Half-footprint of the pending structure (for cursor clamping), metres.
fn pending_footprint_half(state: &BaseBuildState) -> f64 {
    match state.pending_kind {
        PendingKind::Building => state.pending.half_x_m.max(state.pending.half_z_m) as f64,
        PendingKind::Launchpad => state.pending_radius_m as f64,
    }
}

/// Raycast the cursor against the pad sphere and return the grid-snapped,
/// pad-clamped placement direction.
#[allow(clippy::too_many_arguments)]
fn compute_pad_hover(
    windows: &Query<&Window, With<PrimaryWindow>>,
    cameras: &Query<(&Camera, &GlobalTransform), (With<ShipCamera>, With<ActiveCamera>)>,
    bodies: &Query<(&RealSpaceBody, &GlobalTransform)>,
    body_id: BodyId,
    body_state: &BodyState,
    center_dir: DVec3,
    heading: DVec3,
    across: DVec3,
    pad_r: f64,
    half_along_m: f64,
    half_across_m: f64,
    footprint_half_m: f64,
) -> Option<HoverPad> {
    let window = windows.single().ok()?;
    let cursor = window.cursor_position()?;
    let (camera, cam_gt) = cameras.single().ok()?;
    let (_, body_gt) = bodies.iter().find(|(rsb, _)| rsb.body_id == body_id)?;
    let center_render = body_gt.translation();
    let ray = camera.viewport_to_world(cam_gt, cursor).ok()?;
    let dir_render =
        ray_vs_sphere_dir(ray.origin - center_render, *ray.direction, (pad_r * SHIP_SCALE) as f32)?;
    let dir_body = (body_state.orientation.inverse() * dir_render.as_dvec3()).normalize();

    // Tangent-plane offset from the pad centre, in metres.
    let offset = (dir_body - center_dir) * pad_r;
    let snap = |v: f64| (v / GRID_STEP_M).round() * GRID_STEP_M;
    let lim_along = (half_along_m - footprint_half_m).max(0.0);
    let lim_across = (half_across_m - footprint_half_m).max(0.0);
    let along = snap(offset.dot(heading)).clamp(-lim_along, lim_along);
    let across_off = snap(offset.dot(across)).clamp(-lim_across, lim_across);

    let building_dir = (center_dir * pad_r + heading * along + across * across_off).normalize();
    Some(HoverPad { building_dir })
}

/// The structure (on `body_id`) whose footprint the pad point `dir` falls
/// within, nearest first. Approximate (ignores rotation), fine for selection.
fn structure_under(
    placed_q: &Query<(Entity, &PlacedVisual)>,
    body_id: BodyId,
    dir: DVec3,
    pad_r: f64,
) -> Option<StructureId> {
    let mut best: Option<(StructureId, f64)> = None;
    for (_, pv) in placed_q.iter() {
        if pv.body_id != body_id {
            continue;
        }
        let bdir = pv.center_body.normalize();
        let ang = bdir.dot(dir).clamp(-1.0, 1.0).acos();
        let dist_m = ang * pad_r;
        if dist_m <= kind_bounding_m(&pv.kind) && best.is_none_or(|(_, d)| dist_m < d) {
            best = Some((pv.structure_id, dist_m));
        }
    }
    best.map(|(id, _)| id)
}

/// Footprint bounding radius (m) for selection / approximate hit-testing.
pub(super) fn kind_bounding_m(kind: &StructureKind) -> f64 {
    match kind {
        StructureKind::Building {
            half_x_m, half_z_m, ..
        } => half_x_m.hypot(*half_z_m) as f64,
        StructureKind::Launchpad { radius_m } => *radius_m as f64,
        _ => 1.0,
    }
}

/// Body-fixed visual basis for a structure standing at `up` with heading
/// `heading` rotated by `yaw`. `center_height_m` lifts the visual centre above
/// the pad (box half-height, or slab half-thickness). Returns `(center_body,
/// basis_body)`.
fn placement_frame(
    up: DVec3,
    heading: DVec3,
    across: DVec3,
    yaw: f32,
    pad_r: f64,
    center_height_m: f32,
) -> (DVec3, DQuat) {
    let yaw = yaw as f64;
    let hb0 = heading * yaw.cos() + across * yaw.sin();
    // Re-project onto the tangent plane at `up` (curvature over the pad is tiny,
    // but keep the basis exactly orthonormal).
    let hb = (hb0 - up * hb0.dot(up)).normalize();
    let zb = hb.cross(up); // right-handed: X×Y = Z
    let basis_body = DQuat::from_mat3(&DMat3::from_cols(hb, up, zb));
    let center_body = up * (pad_r + center_height_m as f64);
    (center_body, basis_body)
}

#[allow(clippy::too_many_arguments)]
fn spawn_structure(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    registry: &mut StructureRegistry,
    state: &mut BaseBuildState,
    root: Entity,
    site_id: StructureId,
    body_id: BodyId,
    building_dir: DVec3,
    heading: DVec3,
    across: DVec3,
    pad_r: f64,
) -> StructureId {
    let heading_proj = (heading - building_dir * heading.dot(building_dir)).normalize();
    match state.pending_kind {
        PendingKind::Building => {
            let dims = state.pending;
            let (center_body, basis_body) =
                placement_frame(building_dir, heading, across, state.yaw, pad_r, dims.height_m * 0.5);
            let kind = StructureKind::Building {
                half_x_m: dims.half_x_m,
                half_z_m: dims.half_z_m,
                height_m: dims.height_m,
            };
            let id = registry.register(
                body_id,
                building_dir,
                heading_proj,
                StructurePlacement::Drape,
                kind,
                Some(site_id),
            );
            let material = state
                .material
                .get_or_insert_with(|| {
                    materials.add(StandardMaterial {
                        base_color: Color::srgb(0.62, 0.64, 0.68),
                        perceptual_roughness: 0.85,
                        metallic: 0.0,
                        ..default()
                    })
                })
                .clone();
            let mesh = meshes.add(Cuboid::new(dims.half_x_m * 2.0, dims.height_m, dims.half_z_m * 2.0));
            commands.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::default(),
                Visibility::Inherited,
                CellCoord::ZERO,
                ChildOf(root),
                RenderLayers::layer(SHIP_LAYER),
                PlacedVisual {
                    structure_id: id,
                    body_id,
                    center_body,
                    basis_body,
                    kind,
                },
                Name::new("Base Building"),
            ));
            id
        }
        PendingKind::Launchpad => {
            let radius_m = state.pending_radius_m;
            let (center_body, basis_body) =
                placement_frame(building_dir, heading, across, state.yaw, pad_r, LAUNCHPAD_SLAB_H * 0.5);
            let kind = StructureKind::Launchpad { radius_m };
            let id = registry.register(
                body_id,
                building_dir,
                heading_proj,
                StructurePlacement::Drape,
                kind,
                Some(site_id),
            );
            let pad_material = state
                .pad_material
                .get_or_insert_with(|| {
                    materials.add(StandardMaterial {
                        base_color: Color::srgb(0.10, 0.10, 0.12),
                        perceptual_roughness: 0.9,
                        metallic: 0.0,
                        ..default()
                    })
                })
                .clone();
            let ring_material = state
                .ring_material
                .get_or_insert_with(|| {
                    materials.add(StandardMaterial {
                        base_color: Color::srgb(0.95, 0.78, 0.15),
                        perceptual_roughness: 0.6,
                        metallic: 0.0,
                        ..default()
                    })
                })
                .clone();
            let slab = meshes.add(Cylinder::new(radius_m, LAUNCHPAD_SLAB_H));
            let ring_outer = radius_m * 0.85;
            let ring = meshes.add(Torus::new((ring_outer - 0.6).max(0.1), ring_outer));
            let pad_entity = commands
                .spawn((
                    Mesh3d(slab),
                    MeshMaterial3d(pad_material),
                    Transform::default(),
                    Visibility::Inherited,
                    CellCoord::ZERO,
                    ChildOf(root),
                    RenderLayers::layer(SHIP_LAYER),
                    PlacedVisual {
                        structure_id: id,
                        body_id,
                        center_body,
                        basis_body,
                        kind,
                    },
                    Name::new("Launchpad"),
                ))
                .id();
            // Ring marking on top of the slab (own RenderLayers — layers don't
            // inherit through the hierarchy).
            commands.spawn((
                Mesh3d(ring),
                MeshMaterial3d(ring_material),
                Transform::from_xyz(0.0, LAUNCHPAD_SLAB_H * 0.5 + 0.02, 0.0),
                Visibility::Inherited,
                RenderLayers::layer(SHIP_LAYER),
                ChildOf(pad_entity),
                Name::new("Launchpad Ring"),
            ));
            id
        }
    }
}

/// **L**: with a launchpad selected, place the player ship at rest on it and
/// close the editor. Mirrors `runway::place_parked` — sets canonical state,
/// a frozen `BodyFixed` authority, zeroes throttle, and tears down the Avian
/// bubble so it rebuilds from the placed pose. Runs while the editor is open.
#[allow(clippy::too_many_arguments)]
fn launch_from_pad(
    keys: Res<ButtonInput<KeyCode>>,
    mut editor: ResMut<BaseEditor>,
    build: Res<BaseBuildState>,
    registry: Res<StructureRegistry>,
    mut sim: ResMut<SimulationState>,
    solar: Res<SolarSystemState>,
    mut active_bubble: ResMut<ActiveLocalBubble>,
    mut commands: Commands,
    ship_q: Query<(Entity, &GlobalTransform), With<PlayerShip>>,
    children_q: Query<&Children>,
    mesh_q: Query<(&GlobalTransform, &Mesh3d)>,
    meshes: Res<Assets<Mesh>>,
) {
    if !keys.just_pressed(KeyCode::KeyL) {
        return;
    }
    let Some(sel) = build.selected else {
        return;
    };
    let Some(pad) = registry.get(sel).copied() else {
        return;
    };
    if !matches!(pad.kind, StructureKind::Launchpad { .. }) {
        return;
    }
    // Elevation comes from the parent site's flatten.
    let elevation_m = pad
        .parent_site
        .and_then(|p| registry.get(p))
        .and_then(|site| match site.placement {
            StructurePlacement::FlattenTo { elevation_m, .. } => Some(elevation_m),
            StructurePlacement::Drape => None,
        })
        .unwrap_or(0.0);

    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let body_id = pad.body_id;
    let Some(body_state) = states.get(body_id) else {
        return;
    };
    let Some(radius_m) = sim.system.bodies.get(body_id).map(|b| b.radius_m) else {
        return;
    };
    let Ok((ship_entity, ship_gt)) = ship_q.single() else {
        return;
    };
    // How far the craft's lowest point sits below its origin — lift it that much
    // so it rests on the pad top. None ⇒ meshes not ready; retry next press.
    let Some(clearance_m) =
        craft_ground_clearance(ship_entity, ship_gt, &children_q, &mesh_q, &meshes)
    else {
        return;
    };

    let up = pad.anchor_dir;
    let heading = pad.heading_tangent;
    let pad_top_r = radius_m + elevation_m + LAUNCHPAD_SLAB_H as f64;
    let position_body = up * (pad_top_r + clearance_m + LAUNCH_REST_MARGIN_M);
    let position = body_state.position + body_state.orientation * position_body;
    let velocity = body_fixed_surface_velocity(body_state, position_body);
    let state = StateVector { position, velocity };
    let attitude = level_heading_attitude(body_state, up, heading);

    sim.simulation.set_ship_state(state);
    sim.simulation.set_attitude(attitude);
    let pose = body_fixed_pose_from_inertial(body_state, TranslationalState::from(state), attitude);
    sim.simulation
        .transition_authority(AuthorityMode::BodyFixed {
            body: body_id,
            pose,
        });
    sim.simulation.set_throttle(0.0);
    sim.simulation.set_target_body(Some(body_id));
    sim.simulation.warp.reset();

    // Tear down the live Avian bubble so the rebuild seeds from the placed pose.
    crate::scenario_menu::clear_bubble(&mut commands, &mut active_bubble);

    editor.open = false;
    info!("launched player ship onto launchpad {:?}", sel);
}

/// Re-place every structure in the body-fixed frame each frame (ungated, like
/// `runway::update_runway_transform`): a root-grid big_space child posed in f64.
fn update_placed_transforms(
    solar: Res<SolarSystemState>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut placed: Query<(&PlacedVisual, &mut CellCoord, &mut Transform)>,
) {
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let Ok(grid) = root_grid.single() else {
        return;
    };
    for (pv, mut cell, mut transform) in &mut placed {
        let Some(state) = states.get(pv.body_id) else {
            continue;
        };
        let orientation = state.orientation.normalize();
        let center_world = state.position + orientation * pv.center_body;
        let (next_cell, local) = grid.translation_to_grid(center_world);
        *cell = next_cell;
        transform.translation = local;
        transform.rotation = (orientation * pv.basis_body).as_quat();
    }
}

/// Draw the placement ghost (in `PlaceBuildings`) and a highlight around the
/// selected structure.
fn draw_placement_ghost(
    editor: Res<BaseEditor>,
    state: Res<BaseBuildState>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    registry: Res<StructureRegistry>,
    bodies: Query<(&RealSpaceBody, &GlobalTransform)>,
    placed: Query<&PlacedVisual>,
    mut gizmos: Gizmos,
) {
    if editor.mode != BaseEditorMode::PlaceBuildings {
        return;
    }
    let Some(site_id) = editor.active_site else {
        return;
    };
    let Some(site) = registry.get(site_id) else {
        return;
    };
    let StructurePlacement::FlattenTo { elevation_m, .. } = site.placement else {
        return;
    };
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let body_id = site.body_id;
    let Some(body_state) = states.get(body_id) else {
        return;
    };
    let Some(body) = sim.system.bodies.get(body_id) else {
        return;
    };
    let Some((_, body_gt)) = bodies.iter().find(|(rsb, _)| rsb.body_id == body_id) else {
        return;
    };
    let center_render = body_gt.translation();
    let orientation = body_state.orientation.normalize();
    let pad_r = body.radius_m + elevation_m;

    // Selected-structure highlight.
    if let Some(sel) = state.selected {
        for pv in placed.iter() {
            if pv.structure_id != sel {
                continue;
            }
            let center = box_center_render(center_render, orientation, pv.center_body);
            let rot = (orientation * pv.basis_body).as_quat();
            draw_kind_outline(&mut gizmos, &pv.kind, center, rot, Color::srgb(1.0, 0.85, 0.3));
        }
    }

    // Placement ghost.
    let Some(hover) = state.hover else {
        return;
    };
    let across = site.anchor_dir.cross(site.heading_tangent).normalize();
    let ghost_color = Color::srgb(0.3, 1.0, 0.6);
    match state.pending_kind {
        PendingKind::Building => {
            let dims = state.pending;
            let (center_body, basis_body) = placement_frame(
                hover.building_dir,
                site.heading_tangent,
                across,
                state.yaw,
                pad_r,
                dims.height_m * 0.5,
            );
            draw_box(
                &mut gizmos,
                box_center_render(center_render, orientation, center_body),
                (orientation * basis_body).as_quat(),
                Vec3::new(dims.half_x_m, dims.height_m * 0.5, dims.half_z_m),
                ghost_color,
            );
        }
        PendingKind::Launchpad => {
            let (center_body, basis_body) = placement_frame(
                hover.building_dir,
                site.heading_tangent,
                across,
                state.yaw,
                pad_r,
                LAUNCHPAD_SLAB_H * 0.5,
            );
            draw_ring(
                &mut gizmos,
                box_center_render(center_render, orientation, center_body),
                (orientation * basis_body).as_quat(),
                state.pending_radius_m,
                ghost_color,
            );
        }
    }
}

/// Outline for the selected structure, by kind.
fn draw_kind_outline(gizmos: &mut Gizmos, kind: &StructureKind, center: Vec3, rot: Quat, color: Color) {
    match kind {
        StructureKind::Building {
            half_x_m,
            half_z_m,
            height_m,
        } => draw_box(
            gizmos,
            center,
            rot,
            Vec3::new(*half_x_m, height_m * 0.5, *half_z_m),
            color,
        ),
        StructureKind::Launchpad { radius_m } => draw_ring(gizmos, center, rot, *radius_m, color),
        _ => {}
    }
}

/// Render-space position of a body-fixed point. Large-minus-large in f32, so
/// good to ~decimetre — fine for a gizmo preview (the committed entity uses the
/// precise big_space transform path instead).
fn box_center_render(center_render: Vec3, orientation: DQuat, point_body: DVec3) -> Vec3 {
    center_render + (orientation * point_body).as_vec3() * SHIP_SCALE as f32
}

/// Wireframe box: `half` extents along the rotated local axes about `center`.
fn draw_box(gizmos: &mut Gizmos, center: Vec3, rot: Quat, half: Vec3, color: Color) {
    let signs = [
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
    ];
    let c: [Vec3; 8] = std::array::from_fn(|i| center + rot * (half * signs[i]));
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    for (a, b) in edges {
        gizmos.line(c[a], c[b], color);
    }
}

/// Circle of `radius` in the rotated local XZ plane (Y = local up) about `center`.
fn draw_ring(gizmos: &mut Gizmos, center: Vec3, rot: Quat, radius: f32, color: Color) {
    const SEGS: usize = 32;
    let mut prev = center + rot * Vec3::new(radius, 0.0, 0.0);
    for i in 1..=SEGS {
        let a = i as f32 / SEGS as f32 * std::f32::consts::TAU;
        let p = center + rot * Vec3::new(radius * a.cos(), 0.0, radius * a.sin());
        gizmos.line(prev, p, color);
        prev = p;
    }
}
