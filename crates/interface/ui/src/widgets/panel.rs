//! Panels, headings, dividers, key hints — the static composition pieces.

use bevy::prelude::*;
use bevy::ui_render::prelude::MaterialNode;

use crate::UiTheme;
use crate::glass::GlassMaterial;
use crate::tokens::*;

/// A `Node` pre-configured as a floating panel: rounded, padded, column flex.
/// Compose with [`UiTheme::glass`] for the frosted surface.
pub fn panel_node() -> Node {
    Node {
        border_radius: BorderRadius::all(Val::Px(RADIUS_PANEL)),
        padding: UiRect::axes(Val::Px(SPACE_LG), Val::Px(SPACE_MD)),
        flex_direction: FlexDirection::Column,
        row_gap: Val::Px(SPACE_SM),
        ..Default::default()
    }
}

/// [`panel_node`] with absolute positioning (HUD-corner style panels).
pub fn floating_panel_node() -> Node {
    Node {
        position_type: PositionType::Absolute,
        ..panel_node()
    }
}

/// The floating-sheet drop shadow shared by every glass panel.
pub fn panel_shadow() -> BoxShadow {
    BoxShadow::new(
        PANEL_SHADOW,
        Val::Px(0.0),
        Val::Px(10.0),
        Val::Px(-6.0),
        Val::Px(28.0),
    )
}

impl UiTheme {
    /// The shared frosted surface for regular panels (+ its drop shadow).
    pub fn glass(&self) -> (MaterialNode<GlassMaterial>, BoxShadow) {
        (MaterialNode(self.glass_regular.clone()), panel_shadow())
    }

    /// The stronger frosted surface for dominant overlays (dialogs).
    pub fn glass_heavy(&self) -> (MaterialNode<GlassMaterial>, BoxShadow) {
        (MaterialNode(self.glass_strong.clone()), panel_shadow())
    }
}

/// Spawn a section heading (caps by convention): faint semibold text with
/// breathing room above when `spaced`.
pub fn spawn_heading(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    label: &str,
    spaced: bool,
) {
    parent.spawn((
        theme.heading(label),
        Node {
            margin: UiRect::top(Val::Px(if spaced { SPACE_SM } else { 0.0 })),
            ..Default::default()
        },
    ));
}

/// A 1px hairline divider.
pub fn spawn_divider(parent: &mut ChildSpawnerCommands<'_>) {
    parent.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Px(1.0),
            margin: UiRect::vertical(Val::Px(SPACE_XS)),
            ..Default::default()
        },
        BackgroundColor(STROKE),
    ));
}

/// A small bordered key-hint chip ("Esc", "F9").
pub fn spawn_key_hint(parent: &mut ChildSpawnerCommands<'_>, theme: &UiTheme, key: &str) {
    parent
        .spawn((
            Node {
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                padding: UiRect::axes(Val::Px(5.0), Val::Px(1.0)),
                align_items: AlignItems::Center,
                ..Default::default()
            },
            BorderColor::all(STROKE),
            BackgroundColor(FILL_HOVER),
        ))
        .with_children(|c| {
            let mut text = theme.mono_dim(key);
            text.1.font_size = FontSize::Px(10.0);
            c.spawn(text);
        });
}

/// A label→value readout row (label dim left, mono value right).
pub fn spawn_value_row(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    label: &str,
    value: &str,
    value_marker: impl Bundle,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            ..Default::default()
        })
        .with_children(|row| {
            row.spawn(theme.small(label));
            row.spawn((theme.mono(value), value_marker));
        });
}
