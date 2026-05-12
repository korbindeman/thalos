//! Shared visual style for the bevy_ui HUD.
//!
//! The font asset is the same JetBrains Mono that the navball texture
//! uses (already in `assets/fonts/`). All HUD colours and the panel
//! frame look live here so a future styling pass touches a single file.

use bevy::prelude::*;

#[derive(Resource, Clone)]
pub struct HudTheme {
    pub font: Handle<Font>,
    pub panel_bg: Color,
    pub panel_border: Color,
    pub text_primary: Color,
    pub text_dim: Color,
    pub text_accent: Color,
    pub text_warn: Color,
    /// Red-orange used for AP/PE labels in the orbital info panel.
    pub text_label_alt: Color,
    /// Cyan subtitle colour used for panel sub-labels (e.g. "ORBITAL").
    pub text_subtitle: Color,
}

pub fn init_theme(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(HudTheme {
        font: asset_server.load("fonts/JetBrainsMono-Regular.ttf"),
        panel_bg: Color::srgba(0.04, 0.06, 0.10, 0.85),
        panel_border: Color::srgba(0.30, 0.42, 0.58, 0.65),
        text_primary: Color::srgba(0.90, 0.93, 0.97, 1.0),
        text_dim: Color::srgba(0.55, 0.62, 0.72, 1.0),
        text_accent: Color::srgba(1.0, 0.82, 0.30, 1.0),
        text_warn: Color::srgba(0.95, 0.45, 0.30, 1.0),
        text_label_alt: Color::srgba(0.92, 0.45, 0.32, 1.0),
        text_subtitle: Color::srgba(0.35, 0.78, 0.92, 1.0),
    });
}

/// Standard panel frame components: dark translucent fill + subtle
/// border colour. In Bevy 0.18 `BackgroundColor` and `BorderColor` are
/// separate components required by `Node`. `border_radius` is a `Node`
/// field, set by [`panel_node`].
pub fn panel_frame(theme: &HudTheme) -> (BackgroundColor, BorderColor) {
    (
        BackgroundColor(theme.panel_bg),
        BorderColor::all(theme.panel_border),
    )
}

/// Build a `Node` pre-configured as a HUD panel — absolute positioning,
/// 1px border, rounded corners, comfortable padding, column flex with
/// small row gaps.
pub fn panel_node() -> Node {
    Node {
        position_type: PositionType::Absolute,
        border: UiRect::all(Val::Px(1.0)),
        border_radius: BorderRadius::all(Val::Px(4.0)),
        padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
        flex_direction: FlexDirection::Column,
        row_gap: Val::Px(4.0),
        ..default()
    }
}

/// Single-line text bundle styled for HUD body text.
pub fn text(theme: &HudTheme, content: impl Into<String>) -> impl Bundle {
    (
        Text::new(content),
        TextFont {
            font: theme.font.clone(),
            font_size: 14.0,
            ..default()
        },
        TextColor(theme.text_primary),
    )
}

/// Dim label-style text (for "ALT", "AP", "PE" headers).
pub fn label(theme: &HudTheme, content: impl Into<String>) -> impl Bundle {
    (
        Text::new(content),
        TextFont {
            font: theme.font.clone(),
            font_size: 11.0,
            ..default()
        },
        TextColor(theme.text_dim),
    )
}

/// Larger emphasis text for primary readouts (mission time, altitude).
pub fn emphasis(theme: &HudTheme, content: impl Into<String>) -> impl Bundle {
    (
        Text::new(content),
        TextFont {
            font: theme.font.clone(),
            font_size: 18.0,
            ..default()
        },
        TextColor(theme.text_primary),
    )
}
