//! Kòrsou input adapter for the shared F1 clean-view mode.

use bevy::prelude::*;
use thalos_runtime::{
    photo_mode::{PhotoMode, PhotoModePlugin},
    preferences::SettingsMenu,
    ui::TextFieldFocus,
    viewer::ViewpointUiState,
};

pub struct KorsouPhotoModePlugin;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct KorsouPhotoModeInput;

impl Plugin for KorsouPhotoModePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(PhotoModePlugin)
            .add_systems(Update, toggle_photo_mode.in_set(KorsouPhotoModeInput));
    }
}

fn toggle_photo_mode(
    keys: Res<ButtonInput<KeyCode>>,
    settings: Res<SettingsMenu>,
    viewpoints: Res<ViewpointUiState>,
    text_focus: Res<TextFieldFocus>,
    mut photo_mode: ResMut<PhotoMode>,
) {
    if keys.just_pressed(KeyCode::F1)
        && !settings.open
        && !viewpoints.is_open()
        && !text_focus.is_focused()
    {
        photo_mode.toggle();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f1_toggles_photo_mode() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, KorsouPhotoModePlugin))
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<SettingsMenu>()
            .init_resource::<ViewpointUiState>()
            .init_resource::<TextFieldFocus>();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::F1);

        app.update();

        assert!(app.world().resource::<PhotoMode>().active);
    }

    #[test]
    fn f1_does_not_escape_an_open_modal() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, KorsouPhotoModePlugin))
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<SettingsMenu>()
            .init_resource::<ViewpointUiState>()
            .init_resource::<TextFieldFocus>();
        app.world_mut().resource_mut::<SettingsMenu>().open = true;
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::F1);

        app.update();

        assert!(!app.world().resource::<PhotoMode>().active);
    }
}
