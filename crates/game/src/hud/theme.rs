//! Shared visual style for the bevy_ui HUD.
//!
//! The font asset is the same Fira Code that the navball texture
//! uses (bundled in `assets/fonts/`). All HUD colours and the panel
//! frame look live here so a future styling pass touches a single file.

use bevy::prelude::*;

#[derive(Resource, Clone)]
pub struct HudTheme {
    /// 0.19: `TextFont.font` is a `FontSource` (cosmic-text → parley); the
    /// shared HUD font is stored as `FontSource::Handle` so every panel's
    /// `font: theme.font.clone()` keeps working.
    pub font: FontSource,
    pub panel_bg: Color,
    pub panel_bg_alt: Color,
    pub panel_border: Color,
    pub text_primary: Color,
    pub text_dim: Color,
    pub text_accent: Color,
    pub text_warn: Color,
    /// Red-orange used for AP/PE labels in the orbital info panel.
    pub text_label_alt: Color,
    /// Muted metal subtitle colour used for panel sub-labels (e.g. "ORBITAL").
    pub text_subtitle: Color,
    /// Cool blue used for the SEA altitude datum label.
    pub text_datum_sea: Color,
    /// Warm earth tone used for the GND altitude datum label.
    pub text_datum_gnd: Color,
}

pub fn init_theme(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(HudTheme {
        font: FontSource::Handle(asset_server.load("fonts/FiraCode-Regular.ttf")),
        panel_bg: Color::srgba(0.055, 0.055, 0.050, 0.86),
        panel_bg_alt: Color::srgba(0.085, 0.080, 0.070, 0.84),
        panel_border: Color::srgba(0.46, 0.43, 0.36, 0.66),
        text_primary: Color::srgba(0.92, 0.91, 0.86, 1.0),
        text_dim: Color::srgba(0.62, 0.60, 0.53, 1.0),
        text_accent: Color::srgba(0.95, 0.70, 0.28, 1.0),
        text_warn: Color::srgba(0.95, 0.42, 0.26, 1.0),
        text_label_alt: Color::srgba(0.88, 0.43, 0.30, 1.0),
        text_subtitle: Color::srgba(0.66, 0.68, 0.60, 1.0),
        text_datum_sea: Color::srgba(0.42, 0.74, 0.88, 1.0),
        text_datum_gnd: Color::srgba(0.86, 0.62, 0.32, 1.0),
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

/// Dim label-style text (for "ALT", "AP", "PE" headers).
pub fn label(theme: &HudTheme, content: impl Into<String>) -> impl Bundle {
    (
        Text::new(content),
        TextFont {
            font: theme.font.clone(),
            font_size: FontSize::Px(11.0),
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
            font_size: FontSize::Px(18.0),
            ..default()
        },
        TextColor(theme.text_primary),
    )
}
