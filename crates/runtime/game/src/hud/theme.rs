//! Shared visual style for the bevy_ui HUD.
//!
//! The font asset is the same Fira Code that the navball texture
//! uses (bundled in `assets/fonts/`). All HUD colours and the panel
//! frame look live here so a future styling pass touches a single file.

use bevy::prelude::*;
use bevy::ui_render::prelude::MaterialNode;
use thalos_ui::GlassMaterial;

#[derive(Resource, Clone)]
pub struct HudTheme {
    /// 0.19: `TextFont.font` is a `FontSource` (cosmic-text → parley); the
    /// shared HUD font is stored as `FontSource::Handle` so every panel's
    /// `font: theme.font.clone()` keeps working.
    pub font: FontSource,
    /// The shared frosted-glass panel material (same asset as
    /// `UiTheme::glass_regular`), so HUD panels are the same surface as the
    /// menu/editor screens.
    pub glass: Handle<GlassMaterial>,
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

/// Ordered `.after(thalos_ui::init_ui_theme)` (see `hud/mod.rs`) so the
/// shared glass material handle exists.
pub fn init_theme(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    ui_theme: Res<thalos_ui::UiTheme>,
) {
    // The palette derives from the shared design tokens (`thalos_ui::tokens`)
    // so the flight HUD reads as the same family as the menu/editor screens;
    // only the HUD-specific datum colours are local. Fira Code stays as the
    // HUD face — flight readouts want tabular mono digits.
    use thalos_ui::tokens;
    commands.insert_resource(HudTheme {
        font: FontSource::Handle(asset_server.load("fonts/FiraCode-Regular.ttf")),
        glass: ui_theme.glass_regular.clone(),
        panel_bg: Color::srgba(0.024, 0.030, 0.038, 0.78),
        panel_bg_alt: Color::srgba(1.0, 1.0, 1.0, 0.07),
        panel_border: tokens::STROKE,
        text_primary: tokens::TEXT_PRIMARY,
        text_dim: tokens::TEXT_DIM,
        text_accent: tokens::ACCENT,
        text_warn: tokens::DANGER,
        text_label_alt: Color::srgba(0.88, 0.43, 0.30, 1.0),
        text_subtitle: tokens::TEXT_FAINT,
        text_datum_sea: Color::srgba(0.42, 0.74, 0.88, 1.0),
        text_datum_gnd: Color::srgba(0.86, 0.62, 0.32, 1.0),
    });
}

/// Standard panel surface: the shared frosted glass + floating-sheet shadow
/// (the material draws its own hairline stroke, so the node border stays
/// colourless). Returned as a pair so the many existing
/// `let (bg, border) = panel_frame(&theme)` call sites keep working unchanged
/// — `bg` is simply a nested bundle now.
pub fn panel_frame(theme: &HudTheme) -> ((MaterialNode<GlassMaterial>, BoxShadow), BorderColor) {
    (
        (
            MaterialNode(theme.glass.clone()),
            thalos_ui::widgets::panel::panel_shadow(),
        ),
        BorderColor::all(Color::NONE),
    )
}

/// Build a `Node` pre-configured as a HUD panel — absolute positioning,
/// rounded corners (the radius flows into the glass shader), comfortable
/// padding, column flex with small row gaps.
pub fn panel_node() -> Node {
    Node {
        position_type: PositionType::Absolute,
        border: UiRect::all(Val::Px(1.0)),
        border_radius: BorderRadius::all(Val::Px(7.0)),
        padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
        flex_direction: FlexDirection::Column,
        row_gap: Val::Px(3.0),
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
            font_size: FontSize::Px(16.0),
            ..default()
        },
        TextColor(theme.text_primary),
    )
}

/// Spawn a compact mono-labelled HUD button. Interaction visuals come from the
/// shared kit (`thalos_ui::style_buttons` via [`thalos_ui::UiButton`]); only
/// the label face is HUD-specific (Fira Code, matching the readouts around it).
pub fn hud_button(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    binding: impl Bundle,
    label: &str,
    font_size: f32,
    height_px: f32,
) -> Entity {
    parent
        .spawn((
            Button,
            Node {
                height: Val::Px(height_px),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(thalos_ui::RADIUS_CTRL)),
                padding: UiRect::axes(Val::Px(8.0), Val::Px(2.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::NONE),
            BorderColor::all(thalos_ui::tokens::STROKE),
            Interaction::None,
            thalos_ui::UiButton::default(),
            binding,
        ))
        .with_children(|c| {
            c.spawn((
                Text::new(label),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(font_size),
                    ..default()
                },
                TextColor(theme.text_primary),
                thalos_ui::ButtonLabel,
            ));
        })
        .id()
}
