//! View mode: map (far-scale orbit view) vs. ship (1:1 scale ship view).
//!
//! M toggles between the two. Each view has its own camera entity with a
//! fixed `RenderLayers` set; switching views flips which camera is
//! [`ActiveCamera`](crate::camera::ActiveCamera) (and `Camera::is_active`).
//!
//! Mesh-based overlays opt in by carrying [`HideInShipView`] or
//! [`HideInMapView`]. Observers in this module forward those tags onto
//! the appropriate `RenderLayers` so the inactive camera physically
//! cannot draw them — no per-frame visibility flipping needed.
//!
//! Gizmo systems (trajectories, orbit lines) can't be hidden by
//! visibility; configure their gizmo group's `render_layers` so they
//! draw to one camera only.

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, PrimaryEguiContext};

use crate::camera::{ActiveCamera, MapCamera, ShipCamera};
use crate::coords::{MAP_LAYER, SHIP_LAYER};

#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    Map,
    #[default]
    Ship,
}

/// Marker: entities with this component are hidden while the view is
/// [`ViewMode::Ship`]. Attach to overlays that only make sense in the
/// far-scale map view (planet icons, impostor billboards, maneuver arrows,
/// ghost bodies, the flat ship marker).
#[derive(Component)]
pub struct HideInShipView;

/// Marker: entities with this component are hidden while the view is
/// [`ViewMode::Map`]. Attach to the 3D ship mesh and the real-scale body
/// spheres that only make sense up close.
#[derive(Component)]
pub struct HideInMapView;

pub struct ViewPlugin;

impl Plugin for ViewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ViewMode>()
            .init_resource::<crate::coords::WorldScale>()
            .add_observer(attach_map_layer_for_hide_in_ship)
            .add_observer(attach_ship_layer_for_hide_in_map)
            .add_systems(Startup, spawn_ui_camera)
            .add_systems(Update, toggle_view_input)
            .add_systems(
                Update,
                apply_active_camera
                    .after(toggle_view_input)
                    .after(crate::SimStage::Sync)
                    .before(crate::SimStage::Camera),
            )
            .add_systems(PostUpdate, propagate_view_render_layers);
    }
}

/// Layer index reserved for the UI overlay camera. No game entity is
/// placed on this layer, so the camera's 3D main pass renders nothing.
/// We use a real (but empty) layer rather than `RenderLayers::none()`
/// because the latter short-circuits the camera's render graph and
/// takes the egui sub-graph node down with it.
const UI_LAYER: usize = 31;

/// Dedicated overlay camera that owns the primary egui context.
///
/// `bevy_egui` skips inactive cameras when extracting render output, so
/// pinning the egui context to either [`MapCamera`] or [`ShipCamera`]
/// would make the UI vanish whenever the other view is active. This
/// camera is always active and sits above the scene cameras via a
/// higher `order` so the egui pass composites over whichever scene
/// camera is currently rendering. The alpha-blended output mode mirrors
/// `bevy_egui`'s `split_screen` example — without it, the camera's
/// transparent intermediate texture would overwrite the scene cameras'
/// output instead of compositing over it.
fn spawn_ui_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 10,
            clear_color: ClearColorConfig::Custom(Color::NONE),
            output_mode: bevy::camera::CameraOutputMode::Write {
                blend_state: Some(bevy::render::render_resource::BlendState::ALPHA_BLENDING),
                clear_color: ClearColorConfig::None,
            },
            ..default()
        },
        RenderLayers::layer(UI_LAYER),
        PrimaryEguiContext,
        Name::new("UiCamera"),
    ));
}

/// On insertion of [`HideInShipView`], attach `RenderLayers(MAP_LAYER)`
/// so only the map camera can see the entity.
fn attach_map_layer_for_hide_in_ship(
    trigger: On<Add, HideInShipView>,
    mut commands: Commands,
    existing: Query<&RenderLayers>,
) {
    let entity = trigger.entity;
    let layers = match existing.get(entity) {
        Ok(rl) => rl.clone().with(MAP_LAYER).without(SHIP_LAYER).without(0),
        Err(_) => RenderLayers::layer(MAP_LAYER),
    };
    commands.entity(entity).insert(layers);
}

/// On insertion of [`HideInMapView`], attach `RenderLayers(SHIP_LAYER)`
/// so only the ship camera can see the entity.
fn attach_ship_layer_for_hide_in_map(
    trigger: On<Add, HideInMapView>,
    mut commands: Commands,
    existing: Query<&RenderLayers>,
) {
    let entity = trigger.entity;
    let layers = match existing.get(entity) {
        Ok(rl) => rl.clone().with(SHIP_LAYER).without(MAP_LAYER).without(0),
        Err(_) => RenderLayers::layer(SHIP_LAYER),
    };
    commands.entity(entity).insert(layers);
}

/// Propagate the `RenderLayers` of any entity carrying [`HideInShipView`]
/// or [`HideInMapView`] down its full descendant tree.
///
/// `RenderLayers` does not propagate through Bevy's hierarchy on its
/// own, but the natural mental model for callers is "tag the root, the
/// whole vehicle disappears in the other view." This system reasserts
/// the layer on every descendant each frame so reparenting, late-spawned
/// children (ship parts that load after the root), and rebuilt mesh
/// children all stay tied to the right view.
fn propagate_view_render_layers(
    mut commands: Commands,
    roots: Query<
        (Entity, Has<HideInShipView>, Has<HideInMapView>),
        Or<(With<HideInShipView>, With<HideInMapView>)>,
    >,
    children_q: Query<&Children>,
    layers_q: Query<&RenderLayers>,
) {
    for (root, hide_ship, hide_map) in &roots {
        let target = if hide_ship {
            RenderLayers::layer(MAP_LAYER)
        } else if hide_map {
            RenderLayers::layer(SHIP_LAYER)
        } else {
            continue;
        };
        let mut stack: Vec<Entity> = Vec::new();
        if let Ok(c) = children_q.get(root) {
            stack.extend(c.iter());
        }
        while let Some(e) = stack.pop() {
            let needs = layers_q.get(e).map(|rl| rl != &target).unwrap_or(true);
            if needs {
                commands.entity(e).insert(target.clone());
            }
            if let Ok(c) = children_q.get(e) {
                stack.extend(c.iter());
            }
        }
    }
}

/// Flip [`ActiveCamera`] + `Camera::is_active` to track the current
/// [`ViewMode`]. Replaces the per-frame visibility-flip mechanism.
///
/// Filtered to scene cameras only — the UI overlay camera in
/// [`spawn_ui_camera`] must stay active across both views, otherwise
/// `bevy_egui` skips extracting its render output and the UI vanishes.
fn apply_active_camera(
    view: Res<ViewMode>,
    mut commands: Commands,
    mut cameras: Query<
        (
            Entity,
            &mut Camera,
            Option<&MapCamera>,
            Option<&ShipCamera>,
            Has<ActiveCamera>,
        ),
        Or<(With<MapCamera>, With<ShipCamera>)>,
    >,
) {
    if !view.is_changed() {
        return;
    }
    for (entity, mut camera, map, ship, active) in &mut cameras {
        let should_be_active = match *view {
            ViewMode::Map => map.is_some(),
            ViewMode::Ship => ship.is_some(),
        };
        if camera.is_active != should_be_active {
            camera.is_active = should_be_active;
        }
        if should_be_active && !active {
            commands
                .entity(entity)
                .insert((ActiveCamera, IsDefaultUiCamera));
        } else if !should_be_active && active {
            commands
                .entity(entity)
                .remove::<(ActiveCamera, IsDefaultUiCamera)>();
        }
    }
}

pub fn in_map_view(view: Res<ViewMode>) -> bool {
    *view == ViewMode::Map
}

#[allow(dead_code)]
pub fn in_ship_view(view: Res<ViewMode>) -> bool {
    *view == ViewMode::Ship
}

fn toggle_view_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut contexts: EguiContexts,
    mut view: ResMut<ViewMode>,
) {
    if !keys.just_pressed(KeyCode::KeyM) {
        return;
    }
    if let Ok(ctx) = contexts.ctx_mut()
        && ctx.wants_keyboard_input()
    {
        return;
    }
    *view = match *view {
        ViewMode::Map => ViewMode::Ship,
        ViewMode::Ship => ViewMode::Map,
    };
}
