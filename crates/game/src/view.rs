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
use bevy_egui::EguiContexts;

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
            .add_systems(Update, toggle_view_input)
            .add_systems(
                Update,
                (sync_world_scale_to_view, apply_active_camera)
                    .after(toggle_view_input)
                    .after(crate::SimStage::Sync)
                    .before(crate::SimStage::Camera),
            )
            .add_systems(PostUpdate, propagate_view_render_layers);
    }
}

/// Keep [`WorldScale`](crate::coords::WorldScale) in sync with the
/// current view: map → `1e-6`, ship → `1.0`.
///
/// Used only by body-rendering systems that still share entities across
/// both views. Per-view systems should read [`crate::coords::MAP_SCALE`]
/// or [`crate::coords::SHIP_SCALE`] directly.
fn sync_world_scale_to_view(
    view: Res<ViewMode>,
    mut scale: ResMut<crate::coords::WorldScale>,
) {
    if !view.is_changed() {
        return;
    }
    let new_scale = match *view {
        ViewMode::Map => crate::coords::MAP_SCALE,
        ViewMode::Ship => crate::coords::SHIP_SCALE,
    };
    if (scale.0 - new_scale).abs() > f64::EPSILON {
        scale.0 = new_scale;
    }
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
fn apply_active_camera(
    view: Res<ViewMode>,
    mut commands: Commands,
    mut cameras: Query<(
        Entity,
        &mut Camera,
        Option<&MapCamera>,
        Option<&ShipCamera>,
        Has<ActiveCamera>,
    )>,
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
            commands.entity(entity).insert(ActiveCamera);
        } else if !should_be_active && active {
            commands.entity(entity).remove::<ActiveCamera>();
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

