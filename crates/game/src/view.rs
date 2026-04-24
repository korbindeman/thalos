//! View mode: map (far-scale orbit view) vs. ship (1:1 scale ship view).
//!
//! M toggles between the two. Systems that are only meaningful in one
//! view gate on [`in_map_view`] / [`in_ship_view`]. The camera projection
//! and min-distance swap per view so the ship (meter-scale) is viewable in
//! ship mode and solar-system-scale bodies stay viewable in map mode.
//!
//! Mesh-based overlays opt in by carrying [`HideInShipView`] or
//! [`HideInMapView`]; visibility flips whenever [`ViewMode`] changes.
//! Gizmo systems (trajectories, orbit lines, maneuver arrows) can't be
//! hidden by visibility — gate their run condition on [`in_map_view`].

use bevy::prelude::*;
use bevy_egui::EguiContexts;

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
            .add_systems(Update, toggle_view_input)
            .add_systems(
                Update,
                apply_view_mode_visibility
                    .after(toggle_view_input)
                    .after(crate::SimStage::Sync),
            );
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

/// Drive visibility for view-tagged entities. Flips everything on view
/// change and catches newly-spawned tagged entities each frame so they
/// don't pop into the wrong view for one frame.
fn apply_view_mode_visibility(
    view: Res<ViewMode>,
    mut hide_ship: Query<
        &mut Visibility,
        (With<HideInShipView>, Without<HideInMapView>),
    >,
    mut hide_map: Query<
        &mut Visibility,
        (With<HideInMapView>, Without<HideInShipView>),
    >,
    new_hide_ship: Query<Entity, Added<HideInShipView>>,
    new_hide_map: Query<Entity, Added<HideInMapView>>,
) {
    if view.is_changed() {
        let ship_vis = match *view {
            ViewMode::Ship => Visibility::Hidden,
            ViewMode::Map => Visibility::Inherited,
        };
        let mut ship_count = 0;
        for mut vis in &mut hide_ship {
            *vis = ship_vis;
            ship_count += 1;
        }
        let map_vis = match *view {
            ViewMode::Map => Visibility::Hidden,
            ViewMode::Ship => Visibility::Inherited,
        };
        let mut map_count = 0;
        for mut vis in &mut hide_map {
            *vis = map_vis;
            map_count += 1;
        }
        info!(
            "apply_view_mode_visibility: view={:?}, hide_ship set {:?} on {} entities, hide_map set {:?} on {} entities",
            *view, ship_vis, ship_count, map_vis, map_count
        );
    } else {
        // Pick up entities that were just tagged so they don't pop through
        // the wrong view for one frame.
        match *view {
            ViewMode::Ship => {
                for e in &new_hide_ship {
                    if let Ok(mut vis) = hide_ship.get_mut(e) {
                        *vis = Visibility::Hidden;
                    }
                }
            }
            ViewMode::Map => {
                for e in &new_hide_map {
                    if let Ok(mut vis) = hide_map.get_mut(e) {
                        *vis = Visibility::Hidden;
                    }
                }
            }
        }
    }
}
