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
use thalos_body_render::{ShadowedStandardMaterial, shadowed};
use thalos_physics_canonical::body_fixed::{
    body_fixed_pose_from_inertial, body_fixed_surface_velocity,
};
use thalos_physics_canonical::canonical::{AuthorityMode, TranslationalState};
use thalos_physics_canonical::types::{AttitudeState, BodyState};
use thalos_physics_local::{ActiveLocalBubble, LocalPrimitiveShape, spawn_structure_collider};
use thalos_world::{BodyId, StateVector};

use crate::SimStage;
use crate::camera::{ActiveCamera, ShipCamera};
use crate::coords::{SHIP_LAYER, SHIP_SCALE};
use crate::game_context::{ContextHistory, GameContext};
use crate::rendering::real_space::RealSpaceRoot;
use crate::rendering::sun_shadow::SHADOW_CASTER_LAYER;
use crate::rendering::{PlayerShip, RealSpaceBody, SimulationState, SolarSystemState};
use crate::runway::craft_extent_below;
use crate::solar_system_state::sync_solar_system_state;
use crate::spawn::{CraftPlacement, place_craft};
use crate::structures::{StructureId, StructureKind, StructurePlacement, StructureRegistry};

use super::{BaseEditor, BaseEditorMode, base_editor_open, cursor_body_dir};

/// Grid step for snapping placement, metres.
const GRID_STEP_M: f64 = 2.0;
/// Rotation step per Q/E press, radians (15°).
const ROTATE_STEP: f32 = std::f32::consts::PI / 12.0;
/// Thickness of the launchpad slab, metres.
const LAUNCHPAD_SLAB_H: f32 = 0.5;
/// Depth (m) of the launchpad's kinematic collider, extending *down* from the
/// slab top. Deeper than the thin visual slab so the `BodyFixed → bubble`
/// handoff can't drop a released craft through it, mirroring the runway slab's
/// generous thickness (the pad is the sole ground once a `RunwaySite` exists —
/// the generic terrain patch is skipped on that body, see
/// `local_physics::terrain_patch`).
const LAUNCHPAD_COLLIDER_H_M: f64 = 4.0;
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

/// Which kind of structure a placement makes.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum PendingKind {
    #[default]
    Building,
    Launchpad,
}

/// The active editing tool. **Select** is the default — clicking picks a
/// structure to move (drag) or delete, and never places. **Place** is armed by
/// picking an item from the palette; then clicks place that item (right-click
/// returns to Select).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum Tool {
    #[default]
    Select,
    Place,
}

/// Placement state. Public so the editor UI/palette can read/write the tool,
/// pending footprint, and selection.
#[derive(Resource)]
pub struct BaseBuildState {
    pub tool: Tool,
    pub pending: BuildingDims,
    pub pending_kind: PendingKind,
    pub pending_radius_m: f32,
    pub selected: Option<StructureId>,
    /// Bumped on any structural change (place / delete / move-commit) so the
    /// connection layer knows to rebuild.
    pub(super) structures_rev: u32,
    yaw: f32,
    /// Structure being dragged in Select mode (between press and release).
    dragging: Option<StructureId>,
    hover: Option<HoverPad>,
}

impl Default for BaseBuildState {
    fn default() -> Self {
        Self {
            tool: Tool::Select,
            pending: BuildingDims::default(),
            pending_kind: PendingKind::Building,
            pending_radius_m: 18.0,
            selected: None,
            structures_rev: 0,
            yaw: 0.0,
            dragging: None,
            hover: None,
        }
    }
}

/// Structure materials, created once at startup and shared by the editor and
/// the authored default base (so they look identical and there's one source).
/// [`ShadowedStandardMaterial`] so every structure RECEIVES the shared
/// sun-shadow cascade (F6 — the hangar darkens under the craft's shadow and
/// vice versa); they all cast via `SHADOW_CASTER_LAYER` at spawn.
#[derive(Resource)]
pub(super) struct BaseMaterials {
    building: Handle<ShadowedStandardMaterial>,
    pad: Handle<ShadowedStandardMaterial>,
    ring: Handle<ShadowedStandardMaterial>,
    tank: Handle<ShadowedStandardMaterial>,
    /// Dark asphalt — taxiways and aprons.
    pub(super) tarmac: Handle<ShadowedStandardMaterial>,
    /// Light concrete — landside service roads.
    pub(super) road: Handle<ShadowedStandardMaterial>,
    /// Neutral gravel — the VAB→pad crawlerway.
    pub(super) crawlerway: Handle<ShadowedStandardMaterial>,
}

impl BaseMaterials {
    /// Build the shared structure materials. Used by the startup init and by the
    /// authored default base ([`super::spawn_default_base`]).
    pub(super) fn create(materials: &mut Assets<ShadowedStandardMaterial>) -> Self {
        Self {
            building: materials.add(shadowed(StandardMaterial {
                base_color: Color::srgb(0.62, 0.64, 0.68),
                perceptual_roughness: 0.85,
                metallic: 0.0,
                ..default()
            })),
            pad: materials.add(shadowed(StandardMaterial {
                base_color: Color::srgb(0.10, 0.10, 0.12),
                perceptual_roughness: 0.9,
                metallic: 0.0,
                ..default()
            })),
            ring: materials.add(shadowed(StandardMaterial {
                base_color: Color::srgb(0.95, 0.78, 0.15),
                perceptual_roughness: 0.6,
                metallic: 0.0,
                ..default()
            })),
            tank: materials.add(shadowed(StandardMaterial {
                base_color: Color::srgb(0.80, 0.82, 0.85),
                perceptual_roughness: 0.35,
                metallic: 0.5,
                ..default()
            })),
            tarmac: materials.add(shadowed(StandardMaterial {
                base_color: Color::srgb(0.14, 0.14, 0.16),
                perceptual_roughness: 0.92,
                metallic: 0.0,
                ..default()
            })),
            road: materials.add(shadowed(StandardMaterial {
                base_color: Color::srgb(0.34, 0.34, 0.36),
                perceptual_roughness: 0.95,
                metallic: 0.0,
                ..default()
            })),
            crawlerway: materials.add(shadowed(StandardMaterial {
                base_color: Color::srgb(0.33, 0.30, 0.26),
                perceptual_roughness: 0.98,
                metallic: 0.0,
                ..default()
            })),
        }
    }
}

fn init_base_materials(
    mut commands: Commands,
    mut materials: ResMut<Assets<ShadowedStandardMaterial>>,
) {
    let mats = BaseMaterials::create(&mut materials);
    commands.insert_resource(mats);
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
    /// Heading rotation relative to the site, kept so a move re-derives the
    /// frame without changing the structure's orientation.
    yaw: f32,
}

pub(super) struct BaseEditorPlacePlugin;

impl Plugin for BaseEditorPlacePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BaseBuildState>()
            .add_systems(Startup, init_base_materials)
            .add_systems(
                Update,
                (
                    reset_build_state_on_open,
                    update_structure_placement,
                    launch_from_pad,
                    draw_placement_ghost,
                )
                    .chain()
                    // Pick after the god-view camera moves so the placement
                    // raycast reads this frame's camera pose (see `cursor_body_dir`).
                    .after(crate::god_view::GodViewCameraSet)
                    .run_if(base_editor_open),
            )
            // Anchored every frame like the runway (`update_runway_transform`).
            // Must run in `SimStage::Sync` strictly after `sync_solar_system_state`
            // so the body orientation it reads is the *current* frame's — the same
            // snapshot the body mesh, runway, terrain, and camera/floating-origin
            // all use. As a bare unordered `Update` system it read a body pose that
            // was one frame stale (and, under the nondeterministic executor order,
            // inconsistent frame-to-frame), so at warp > 1× the structures slewed
            // off the pad and jittered relative to the correctly-synced ground —
            // invisible while paused, since the body pose is then constant.
            // `SimStage::Sync` still runs while the base editor is open (only the
            // `Camera` set is gated for it), so structures stay anchored both in
            // flight and while editing.
            .add_systems(
                Update,
                update_placed_transforms
                    .in_set(SimStage::Sync)
                    .after(sync_solar_system_state),
            );
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
    mats: Res<BaseMaterials>,
    root: Res<RealSpaceRoot>,
    mut placed_q: Query<(Entity, &mut PlacedVisual)>,
    queries: (
        Query<&Window, With<PrimaryWindow>>,
        Query<(&Camera, &CellCoord, &Transform), (With<ShipCamera>, With<ActiveCamera>)>,
        Query<&Grid, With<BigSpace>>,
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

    let (windows, cameras, root_grid) = &queries;

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
        root_grid,
        body_state,
        center_dir,
        heading,
        across,
        pad_r,
        half_along_m,
        half_across_m,
        footprint_half,
    );

    // Delete the selected structure (either tool).
    if (keys.just_pressed(KeyCode::Delete) || keys.just_pressed(KeyCode::KeyX))
        && let Some(sel) = state.selected.take()
    {
        registry.remove(sel);
        for (entity, pv) in placed_q.iter() {
            if pv.structure_id == sel {
                commands.entity(entity).despawn();
            }
        }
        state.dragging = None;
        state.structures_rev = state.structures_rev.wrapping_add(1);
        return;
    }

    // Right-click cancels placement → back to the Select tool.
    if mouse.just_pressed(MouseButton::Right) && state.tool == Tool::Place {
        state.tool = Tool::Select;
        return;
    }

    let over_ui = ui_gate.hovered;
    let Some(hover) = state.hover else {
        state.dragging = None;
        return;
    };

    match state.tool {
        Tool::Place => {
            if !over_ui && mouse.just_pressed(MouseButton::Left) {
                let new_id = spawn_structure(
                    &mut commands,
                    &mut meshes,
                    &mats,
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
                state.structures_rev = state.structures_rev.wrapping_add(1);
            }
        }
        Tool::Select => {
            // Press picks the structure under the cursor (or clears selection);
            // a press over a structure also begins a drag-move.
            if !over_ui && mouse.just_pressed(MouseButton::Left) {
                let hit = structure_under(&placed_q, body_id, hover.building_dir, pad_r);
                state.selected = hit;
                state.dragging = hit;
            }
            // Drag the held structure to the cursor.
            if mouse.pressed(MouseButton::Left)
                && !over_ui
                && let Some(id) = state.dragging
            {
                reposition_structure(
                    id,
                    hover.building_dir,
                    heading,
                    across,
                    pad_r,
                    &mut registry,
                    &mut placed_q,
                );
            }
            // Release commits the move.
            if mouse.just_released(MouseButton::Left) && state.dragging.take().is_some() {
                state.structures_rev = state.structures_rev.wrapping_add(1);
            }
        }
    }
}

/// Reset the editing tool to Select (and clear selection/drag) each time the
/// editor opens, so a fresh session never starts mid-placement.
fn reset_build_state_on_open(editor: Res<BaseEditor>, mut state: ResMut<BaseBuildState>) {
    if editor.is_changed() && editor.open {
        state.tool = Tool::Select;
        state.selected = None;
        state.dragging = None;
    }
}

/// Move a placed structure to `new_dir` on the pad, keeping its size + heading.
fn reposition_structure(
    id: StructureId,
    new_dir: DVec3,
    heading: DVec3,
    across: DVec3,
    pad_r: f64,
    registry: &mut StructureRegistry,
    placed_q: &mut Query<(Entity, &mut PlacedVisual)>,
) {
    for (_, mut pv) in placed_q.iter_mut() {
        if pv.structure_id != id {
            continue;
        }
        let center_height = match pv.kind {
            StructureKind::Building { height_m, .. } => height_m * 0.5,
            StructureKind::Launchpad { .. } => LAUNCHPAD_SLAB_H * 0.5,
            StructureKind::Tank { height_m, .. } => height_m * 0.5,
            _ => 0.0,
        };
        let (center_body, basis_body) =
            placement_frame(new_dir, heading, across, pv.yaw, pad_r, center_height);
        pv.center_body = center_body;
        pv.basis_body = basis_body;
        let heading_proj = (heading - new_dir * heading.dot(new_dir)).normalize();
        registry.update(id, |s| {
            s.anchor_dir = new_dir;
            s.heading_tangent = heading_proj;
        });
        break;
    }
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
    cameras: &Query<(&Camera, &CellCoord, &Transform), (With<ShipCamera>, With<ActiveCamera>)>,
    root_grid: &Query<&Grid, With<BigSpace>>,
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
    let (camera, cam_cell, cam_transform) = cameras.single().ok()?;
    let root_grid = root_grid.single().ok()?;
    let dir_body = cursor_body_dir(
        camera,
        cam_cell,
        cam_transform,
        root_grid,
        cursor,
        body_state.position,
        body_state.orientation,
        pad_r,
    )?;

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
    placed_q: &Query<(Entity, &mut PlacedVisual)>,
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
        StructureKind::Tank { radius_m, .. } => *radius_m as f64,
        // A runway is a long strip, not a disc; its across half-width is the
        // right inset for a taxiway meeting its side. Connection endpoints on a
        // runway are projected onto the nearest point of the strip separately
        // (see `connections`), so this is only a fallback.
        StructureKind::Runway { half_width_m, .. } => *half_width_m as f64,
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

/// Attitude for a craft standing vertically on a pad: nose (`+Y`) along local
/// up, dorsal (`+Z`) toward `heading_body`. Mirror of
/// `runway::level_heading_attitude` with up/heading swapped.
fn vertical_attitude(body_state: &BodyState, up_body: DVec3, heading_body: DVec3) -> AttitudeState {
    let nose = up_body.normalize();
    let dorsal = (heading_body - nose * heading_body.dot(nose))
        .try_normalize()
        .unwrap_or_else(|| {
            let seed = if nose.x.abs() < 0.9 {
                DVec3::X
            } else {
                DVec3::Z
            };
            (seed - nose * seed.dot(nose)).normalize()
        });
    let right = nose.cross(dorsal).normalize();
    let craft_to_body = DMat3::from_cols(right, nose, dorsal);
    AttitudeState {
        orientation: (body_state.orientation * DQuat::from_mat3(&craft_to_body)).normalize(),
        angular_velocity: DVec3::ZERO,
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_structure(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &BaseMaterials,
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
    let kind = match state.pending_kind {
        PendingKind::Building => StructureKind::Building {
            half_x_m: state.pending.half_x_m,
            half_z_m: state.pending.half_z_m,
            height_m: state.pending.height_m,
        },
        PendingKind::Launchpad => StructureKind::Launchpad {
            radius_m: state.pending_radius_m,
        },
    };
    place_structure(
        commands,
        meshes,
        mats,
        registry,
        root,
        body_id,
        Some(site_id),
        building_dir,
        heading,
        across,
        pad_r,
        kind,
        state.yaw,
    )
}

/// Register a structure (`Drape` on its parent site) and spawn its visual.
/// Shared by the editor's click-place and the authored default base
/// (`super::spawn_default_base`). Returns the new structure id.
#[allow(clippy::too_many_arguments)]
pub(super) fn place_structure(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &BaseMaterials,
    registry: &mut StructureRegistry,
    root: Entity,
    body_id: BodyId,
    parent_site: Option<StructureId>,
    anchor_dir: DVec3,
    heading: DVec3,
    across: DVec3,
    pad_r: f64,
    kind: StructureKind,
    yaw: f32,
) -> StructureId {
    let center_height = match kind {
        StructureKind::Building { height_m, .. } => height_m * 0.5,
        StructureKind::Launchpad { .. } => LAUNCHPAD_SLAB_H * 0.5,
        StructureKind::Tank { height_m, .. } => height_m * 0.5,
        _ => 0.0,
    };
    let (center_body, basis_body) =
        placement_frame(anchor_dir, heading, across, yaw, pad_r, center_height);
    let heading_proj = (heading - anchor_dir * heading.dot(anchor_dir)).normalize();
    let id = registry.register(
        body_id,
        anchor_dir,
        heading_proj,
        StructurePlacement::Drape,
        kind,
        parent_site,
    );
    spawn_structure_entity(
        commands,
        meshes,
        mats,
        root,
        id,
        body_id,
        center_body,
        basis_body,
        kind,
        yaw,
    );
    id
}

/// Spawn the visual entity for a placed structure (cuboid building, or cylinder
/// launchpad + ring). The big_space anchor is set each frame by
/// [`update_placed_transforms`].
#[allow(clippy::too_many_arguments)]
fn spawn_structure_entity(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &BaseMaterials,
    root: Entity,
    id: StructureId,
    body_id: BodyId,
    center_body: DVec3,
    basis_body: DQuat,
    kind: StructureKind,
    yaw: f32,
) {
    let placed = PlacedVisual {
        structure_id: id,
        body_id,
        center_body,
        basis_body,
        kind,
        yaw,
    };
    match kind {
        StructureKind::Building {
            half_x_m,
            half_z_m,
            height_m,
        } => {
            let mesh = meshes.add(Cuboid::new(half_x_m * 2.0, height_m, half_z_m * 2.0));
            commands.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(mats.building.clone()),
                Transform::default(),
                Visibility::Inherited,
                CellCoord::ZERO,
                ChildOf(root),
                // Cast into the custom sun-shadow cascade so the building grounds
                // with a shadow on the terrain, like the craft + trees (F6b).
                RenderLayers::from_layers(&[SHIP_LAYER, SHADOW_CASTER_LAYER]),
                placed,
                Name::new("Base Building"),
            ));
        }
        StructureKind::Launchpad { radius_m } => {
            let slab = meshes.add(Cylinder::new(radius_m, LAUNCHPAD_SLAB_H));
            let ring_outer = radius_m * 0.85;
            let ring = meshes.add(Torus::new((ring_outer - 0.6).max(0.1), ring_outer));
            let pad_entity = commands
                .spawn((
                    Mesh3d(slab),
                    MeshMaterial3d(mats.pad.clone()),
                    Transform::default(),
                    Visibility::Inherited,
                    CellCoord::ZERO,
                    ChildOf(root),
                    // Receive-only, like the runway paving and tarmac: a flat
                    // ground-flush slab that also casts self-shadow-acnes at
                    // grazing sun (its top is its own nearest caster), and its
                    // sub-metre rim shadow is sub-texel in every cascade.
                    RenderLayers::layer(SHIP_LAYER),
                    placed,
                    Name::new("Launchpad"),
                ))
                .id();
            // Ring marking on top of the slab (own RenderLayers — layers don't
            // inherit through the hierarchy).
            commands.spawn((
                Mesh3d(ring),
                MeshMaterial3d(mats.ring.clone()),
                Transform::from_xyz(0.0, LAUNCHPAD_SLAB_H * 0.5 + 0.02, 0.0),
                Visibility::Inherited,
                RenderLayers::layer(SHIP_LAYER),
                ChildOf(pad_entity),
                Name::new("Launchpad Ring"),
            ));
            // Solid kinematic collider so a craft placed on / launched from the
            // pad has real ground under it. Without this the pad is visual-only,
            // and once a `RunwaySite` exists on the body the generic terrain
            // patch is skipped there (`local_physics::terrain_patch`), so a
            // pad-launched craft would fall through the ground. A Y-axis cylinder
            // matches the visual slab (`basis_body` has local Y = up); its top
            // face coincides with the slab top and it extends `LAUNCHPAD_COLLIDER_H_M`
            // down. Posed each frame by the executor's `sync_structure_collider_pose`.
            let up = center_body.normalize();
            let slab_top = center_body + up * (LAUNCHPAD_SLAB_H as f64 * 0.5);
            let collider_center = slab_top - up * (LAUNCHPAD_COLLIDER_H_M * 0.5);
            spawn_structure_collider(
                commands,
                body_id,
                LocalPrimitiveShape::Cylinder {
                    radius: radius_m as f64,
                    height: LAUNCHPAD_COLLIDER_H_M,
                },
                collider_center,
                basis_body,
                "Launchpad collider slab",
            );
        }
        StructureKind::Tank { radius_m, height_m } => {
            let mesh = meshes.add(Cylinder::new(radius_m, height_m));
            commands.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(mats.tank.clone()),
                Transform::default(),
                Visibility::Inherited,
                CellCoord::ZERO,
                ChildOf(root),
                RenderLayers::from_layers(&[SHIP_LAYER, SHADOW_CASTER_LAYER]),
                placed,
                Name::new("Storage Tank"),
            ));
        }
        _ => {}
    }
}

/// **L**: with a launchpad selected, place the player ship at rest on it and
/// close the editor. Mirrors `runway::place_parked` — sets canonical state,
/// a frozen `BodyFixed` authority, zeroes throttle, and tears down the Avian
/// bubble so it rebuilds from the placed pose. Runs while the editor is open.
#[allow(clippy::too_many_arguments)]
fn launch_from_pad(
    keys: Res<ButtonInput<KeyCode>>,
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
    mut next_ctx: Option<ResMut<NextState<GameContext>>>,
    mut history: ResMut<ContextHistory>,
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
    // Spawn vertical (rocket nose-up). Rest on the craft's lower end, which for a
    // vertical craft is its `-Y` extent (the engine bell), not its belly `-Z`.
    // None ⇒ meshes not ready; retry next press.
    let Some(clearance_m) = craft_extent_below(
        ship_entity,
        ship_gt,
        &children_q,
        &mesh_q,
        &meshes,
        Vec3::NEG_Y,
    ) else {
        return;
    };

    let up = pad.anchor_dir;
    let heading = pad.heading_tangent;
    place_on_launchpad(
        &mut sim,
        body_state,
        body_id,
        up,
        heading,
        radius_m,
        elevation_m,
        clearance_m,
        &mut commands,
        &mut active_bubble,
    );
    // The L-launch flies immediately, so resume to 1× (the placement core is
    // warp-neutral). `spawn::apply_initial_warp` doesn't run here — this is a
    // runtime teleport, not a `Loading → Running` edge.
    sim.simulation.warp.reset();

    // The L-launch flies immediately: clear the return stack and drop to Flight.
    history.0.clear();
    if let Some(next) = next_ctx.as_mut() {
        next.set(GameContext::Flight);
    }
    info!("launched player ship onto launchpad {:?}", sel);
}

/// Place a craft **vertically** at rest on a launchpad and tear down the live
/// Avian bubble so it rebuilds from the placed pose. Warp-neutral — the caller
/// sets the time-warp level. `up`/`heading` are the pad's body-fixed
/// anchor/takeoff tangent, `clearance_m` the craft's `-Y` (engine-end) extent,
/// `elevation_m` the parent site's flatten. The vertical mirror of
/// [`crate::runway::place_on_runway`]; shared by the base editor's L-launch and
/// the launch-select flow.
#[allow(clippy::too_many_arguments)]
pub(crate) fn place_on_launchpad(
    sim: &mut SimulationState,
    body_state: &BodyState,
    body_id: BodyId,
    up: DVec3,
    heading: DVec3,
    radius_m: f64,
    elevation_m: f64,
    clearance_m: f64,
    commands: &mut Commands,
    active_bubble: &mut ActiveLocalBubble,
) {
    let pad_top_r = radius_m + elevation_m + LAUNCHPAD_SLAB_H as f64;
    let position_body = up * (pad_top_r + clearance_m + LAUNCH_REST_MARGIN_M);
    let position = body_state.position + body_state.orientation * position_body;
    let velocity = body_fixed_surface_velocity(body_state, position_body);
    let state = StateVector { position, velocity };
    let attitude = vertical_attitude(body_state, up, heading);

    let pose = body_fixed_pose_from_inertial(body_state, TranslationalState::from(state), attitude);
    // Tear the live Avian bubble down first, then seat the placed pose, so the
    // rebuild seeds from it rather than fighting the pre-teleport state (see
    // `spawn::place_craft`).
    place_craft(
        sim,
        CraftPlacement {
            state,
            attitude,
            authority: AuthorityMode::BodyFixed {
                body: body_id,
                pose,
            },
        },
        Some((commands, active_bubble)),
    );
    sim.simulation.set_throttle(0.0);
    sim.simulation.set_target_body(Some(body_id));
}

/// Re-place every structure in the body-fixed frame each frame, exactly like
/// `runway::update_runway_transform`: a root-grid big_space child posed in f64,
/// scheduled in `SimStage::Sync` after `sync_solar_system_state`.
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
    mut gizmos: Gizmos<crate::god_view::GodViewGizmos>,
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
            draw_kind_outline(
                &mut gizmos,
                &pv.kind,
                center,
                rot,
                Color::srgb(1.0, 0.85, 0.3),
            );
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
fn draw_kind_outline(
    gizmos: &mut Gizmos<crate::god_view::GodViewGizmos>,
    kind: &StructureKind,
    center: Vec3,
    rot: Quat,
    color: Color,
) {
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
        StructureKind::Tank { radius_m, .. } => draw_ring(gizmos, center, rot, *radius_m, color),
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
fn draw_box(
    gizmos: &mut Gizmos<crate::god_view::GodViewGizmos>,
    center: Vec3,
    rot: Quat,
    half: Vec3,
    color: Color,
) {
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
fn draw_ring(
    gizmos: &mut Gizmos<crate::god_view::GodViewGizmos>,
    center: Vec3,
    rot: Quat,
    radius: f32,
    color: Color,
) {
    const SEGS: usize = 32;
    let mut prev = center + rot * Vec3::new(radius, 0.0, 0.0);
    for i in 1..=SEGS {
        let a = i as f32 / SEGS as f32 * std::f32::consts::TAU;
        let p = center + rot * Vec3::new(radius * a.cos(), 0.0, radius * a.sin());
        gizmos.line(prev, p, color);
        prev = p;
    }
}
