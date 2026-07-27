//! `bevy_ui` overlay node displaying the navball render target.
//!
//! Anchored to the bottom-left of the screen. The render target sits inside
//! a circular HUD frame, while direction-marker overlay nodes (prograde,
//! retrograde, etc.) parent to the image node so their [`NAVBALL_SIZE_PX`]
//! coordinate system stays stable.

use bevy::prelude::*;

use crate::hud::theme::HudTheme;
use crate::navball::render::NavballRenderTarget;
use crate::photo_mode::HideInPhotoMode;

pub const NAVBALL_LEFT_PX: f32 = 28.0;
pub const NAVBALL_BOTTOM_PX: f32 = 28.0;
pub const NAVBALL_SIZE_PX: f32 = 224.0;
pub const FRAME_PADDING_PX: f32 = 8.0;
pub const FRAME_SIZE_PX: f32 = NAVBALL_SIZE_PX + FRAME_PADDING_PX * 2.0;

/// Marker for the circular navball frame node. HUD adornments that need to
/// wrap the actual frame should parent to this instead of copying screen-space
/// frame coordinates.
#[derive(Component)]
pub struct NavballFrameRoot;

/// Marker for the navball UI root node.
#[derive(Component)]
pub struct NavballUiRoot;

pub fn setup_navball_ui(
    mut commands: Commands,
    target: Res<NavballRenderTarget>,
    theme: Res<HudTheme>,
) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(NAVBALL_LEFT_PX - FRAME_PADDING_PX),
                bottom: Val::Px(NAVBALL_BOTTOM_PX - FRAME_PADDING_PX),
                width: Val::Px(FRAME_SIZE_PX),
                height: Val::Px(FRAME_SIZE_PX),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Percent(50.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            NavballFrameRoot,
            HideInPhotoMode,
            Name::new("NavballFrame"),
        ))
        .with_children(|p| {
            p.spawn((
                Node {
                    width: Val::Px(NAVBALL_SIZE_PX),
                    height: Val::Px(NAVBALL_SIZE_PX),
                    ..default()
                },
                ImageNode::new(target.image.clone()),
                NavballUiRoot,
                Name::new("NavballUiRoot"),
            ));
        });
}
