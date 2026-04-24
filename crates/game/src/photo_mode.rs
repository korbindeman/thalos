//! Photo mode: P toggles a "clean scene" view that hides all UI and
//! gizmo overlays (HUD, orbits, trajectory, maneuver UI, ghost bodies) so
//! the user can frame and capture the scene.
//!
//! Mesh-based overlay entities opt in by carrying [`HideInPhotoMode`];
//! their `Visibility` is flipped whenever [`PhotoMode`] changes. Gizmo and
//! egui systems opt in by gating their run on [`not_in_photo_mode`].
//!
//! Future work: extend [`PhotoMode`] with camera parameters (focal length,
//! aperture, exposure) and a dedicated photo-mode panel.

use bevy::prelude::*;
use bevy_egui::EguiContexts;

/// Global photo-mode state.
#[derive(Resource, Default, Debug)]
pub struct PhotoMode {
    pub active: bool,
}

/// Marker: entities with this component are hidden while photo mode is active.
/// Attach to the root of any overlay mesh (children inherit visibility).
#[derive(Component)]
pub struct HideInPhotoMode;

pub struct PhotoModePlugin;

impl Plugin for PhotoModePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PhotoMode>()
            .add_systems(Update, toggle_photo_mode_input)
            // Runs after everything in `Sync` so newly spawned tagged entities
            // (e.g. ghost bodies, maneuver handles) are caught the same frame.
            .add_systems(
                Update,
                apply_photo_mode_visibility.after(crate::SimStage::Sync),
            );
    }
}

/// Run condition: true when photo mode is inactive. Use to gate HUD panels,
/// gizmo draws, and the per-frame update systems of overlay mesh entities.
pub fn not_in_photo_mode(photo_mode: Res<PhotoMode>) -> bool {
    !photo_mode.active
}

fn toggle_photo_mode_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut contexts: EguiContexts,
    mut photo_mode: ResMut<PhotoMode>,
) {
    if !keys.just_pressed(KeyCode::KeyP) {
        return;
    }
    // Don't steal P while the user is typing into an egui widget.
    if let Ok(ctx) = contexts.ctx_mut()
        && ctx.wants_keyboard_input()
    {
        return;
    }
    photo_mode.active = !photo_mode.active;
}

fn apply_photo_mode_visibility(
    photo_mode: Res<PhotoMode>,
    mut all: Query<&mut Visibility, With<HideInPhotoMode>>,
    newly_added: Query<Entity, Added<HideInPhotoMode>>,
) {
    if photo_mode.is_changed() {
        // Mode toggled: flip every tagged entity.
        let target = if photo_mode.active {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        for mut vis in &mut all {
            *vis = target;
        }
    } else if photo_mode.active {
        // Mode unchanged: only hide entities that spawned this frame, so
        // e.g. a ghost body freshly spawned while in photo mode doesn't pop
        // into view for one frame.
        for entity in &newly_added {
            if let Ok(mut vis) = all.get_mut(entity) {
                *vis = Visibility::Hidden;
            }
        }
    }
}
