//! `bevy_ui` overlay node displaying the navball render target.
//!
//! Anchored to the bottom-left of the screen. Will later be the parent of
//! direction-marker overlay nodes (prograde, retrograde, etc.).

use bevy::prelude::*;

use crate::navball::render::NavballRenderTarget;
use crate::photo_mode::HideInPhotoMode;

/// Marker for the navball UI root node.
#[derive(Component)]
pub struct NavballUiRoot;

pub fn setup_navball_ui(mut commands: Commands, target: Res<NavballRenderTarget>) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(40.0),
            bottom: Val::Px(40.0),
            width: Val::Px(256.0),
            height: Val::Px(256.0),
            ..default()
        },
        ImageNode::new(target.image.clone()),
        NavballUiRoot,
        HideInPhotoMode,
        Name::new("NavballUiRoot"),
    ));
}
