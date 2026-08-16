//! Game input adapter for shared photo mode. F1/P toggles a clean scene view
//! that hides all UI and
//! gizmo overlays (HUD, orbits, trajectory, maneuver UI, ghost bodies) so
//! the user can frame and capture the scene.
//!
//! Shared state and visibility arbitration live in `thalos_photo_mode` so
//! Kòrsou and the game cannot drift. Gizmo and egui systems opt in by gating
//! their run on [`not_in_photo_mode`].
//!
//! Future work: extend [`PhotoMode`] with camera parameters (focal length,
//! aperture, exposure) and a dedicated photo-mode panel.

use bevy::prelude::*;
use thalos_input::game::GameInputIntent;

pub use thalos_photo_mode::{HideInPhotoMode, PhotoMode, not_in_photo_mode};

pub struct PhotoModePlugin;

impl Plugin for PhotoModePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(thalos_photo_mode::PhotoModePlugin)
            .add_systems(
                Update,
                toggle_photo_mode_input.run_if(crate::pause_menu::not_game_paused),
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
    photo_mode.toggle();
}
