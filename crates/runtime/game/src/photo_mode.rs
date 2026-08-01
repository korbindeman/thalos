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
use thalos_input::game::GameInputIntent;

// Moved to `thalos_game_state::ui` (Phase 5b); the toggle system stays here.
pub use thalos_game_state::ui::{HideInPhotoMode, PhotoMode, not_in_photo_mode};

pub struct PhotoModePlugin;

impl Plugin for PhotoModePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PhotoMode>()
            .add_systems(
                Update,
                toggle_photo_mode_input.run_if(crate::pause_menu::not_game_paused),
            )
            // Runs after everything in `Sync` so newly spawned tagged entities
            // (e.g. ghost bodies, maneuver handles) are caught the same frame.
            .add_systems(
                Update,
                apply_photo_mode_visibility.after(crate::SimStage::Sync),
            );
    }
}


fn toggle_photo_mode_input(
    input: Res<GameInputIntent>,
    ui_keyboard: Res<crate::hud::UiKeyboardGate>,
    mut photo_mode: ResMut<PhotoMode>,
) {
    if !input.toggle_photo_mode {
        return;
    }
    // Don't steal P while the user is typing into a text field.
    if ui_keyboard.text_entry() {
        return;
    }
    photo_mode.active = !photo_mode.active;
}

fn apply_photo_mode_visibility(
    photo_mode: Res<PhotoMode>,
    newly_added: Query<Entity, Added<HideInPhotoMode>>,
    mut visibility: ParamSet<(
        Query<&mut Visibility, With<HideInPhotoMode>>,
        Query<&mut Visibility, With<thalos_ui::ToastArea>>,
    )>,
) {
    if photo_mode.is_changed() {
        // Mode toggled: flip every tagged overlay, including the shared toast
        // container. Photo mode must keep the viewport clean even when a
        // capture finishes after F1 was pressed.
        let target = if photo_mode.active {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        for mut vis in visibility.p0().iter_mut() {
            *vis = target;
        }
        for mut vis in visibility.p1().iter_mut() {
            *vis = target;
        }
    } else if photo_mode.active {
        // Mode unchanged: only hide entities that spawned this frame, so
        // e.g. a ghost body freshly spawned while in photo mode doesn't pop
        // into view for one frame.
        for entity in &newly_added {
            if let Ok(mut vis) = visibility.p0().get_mut(entity) {
                *vis = Visibility::Hidden;
            }
        }
    }
}
