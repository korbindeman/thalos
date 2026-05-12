//! Shared visual style for the bevy_ui HUD.
//!
//! The font asset is the same JetBrains Mono that the navball texture
//! uses (already in `assets/fonts/`). All HUD colours and the panel
//! frame look live here so a future styling pass touches a single file.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

#[derive(Resource, Clone)]
pub struct HudTheme {
    pub font: Handle<Font>,
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
}

pub fn init_theme(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(HudTheme {
        font: asset_server.load("fonts/JetBrainsMono-Regular.ttf"),
        panel_bg: Color::srgba(0.055, 0.055, 0.050, 0.86),
        panel_bg_alt: Color::srgba(0.085, 0.080, 0.070, 0.84),
        panel_border: Color::srgba(0.46, 0.43, 0.36, 0.66),
        text_primary: Color::srgba(0.92, 0.91, 0.86, 1.0),
        text_dim: Color::srgba(0.62, 0.60, 0.53, 1.0),
        text_accent: Color::srgba(0.95, 0.70, 0.28, 1.0),
        text_warn: Color::srgba(0.95, 0.42, 0.26, 1.0),
        text_label_alt: Color::srgba(0.88, 0.43, 0.30, 1.0),
        text_subtitle: Color::srgba(0.66, 0.68, 0.60, 1.0),
    });
}

pub fn apply_egui_theme(mut contexts: EguiContexts, mut applied: Local<bool>) {
    if *applied {
        return;
    }

    let Ok(ctx) = contexts.ctx_mut() else { return };

    let panel = egui::Color32::from_rgb(15, 15, 14);
    let panel_alt = egui::Color32::from_rgb(24, 23, 20);
    let recessed = egui::Color32::from_rgb(10, 10, 9);
    let border = egui::Color32::from_rgb(118, 110, 94);
    let border_soft = egui::Color32::from_rgb(86, 80, 68);
    let text = egui::Color32::from_rgb(235, 232, 220);
    let text_dim = egui::Color32::from_rgb(160, 154, 136);
    let accent = egui::Color32::from_rgb(212, 166, 76);

    let mut style = (*ctx.style()).clone();
    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = panel;
    visuals.panel_fill = panel;
    visuals.window_stroke = egui::Stroke::new(1.0, border_soft);
    visuals.faint_bg_color = egui::Color32::from_rgb(23, 22, 20);
    visuals.extreme_bg_color = recessed;
    visuals.code_bg_color = egui::Color32::from_rgb(38, 35, 30);
    visuals.weak_text_color = Some(text_dim);
    visuals.hyperlink_color = egui::Color32::from_rgb(190, 168, 116);
    visuals.warn_fg_color = egui::Color32::from_rgb(238, 148, 64);
    visuals.error_fg_color = egui::Color32::from_rgb(230, 82, 54);
    visuals.selection.bg_fill = egui::Color32::from_rgb(112, 88, 48);
    visuals.selection.stroke = egui::Stroke::new(1.0, text);

    visuals.widgets.noninteractive.weak_bg_fill = panel;
    visuals.widgets.noninteractive.bg_fill = panel;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, border_soft);
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, text_dim);

    visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(42, 39, 34);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(46, 43, 37);
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, border_soft);
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, text);

    visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(58, 53, 43);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(62, 56, 45);
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, accent);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.5, text);

    visuals.widgets.active.weak_bg_fill = egui::Color32::from_rgb(72, 63, 46);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(78, 68, 50);
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, accent);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(2.0, text);

    visuals.widgets.open.weak_bg_fill = panel_alt;
    visuals.widgets.open.bg_fill = panel;
    visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, border);
    visuals.widgets.open.fg_stroke = egui::Stroke::new(1.0, text);

    style.visuals = visuals;
    ctx.set_style(style);
    *applied = true;
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
