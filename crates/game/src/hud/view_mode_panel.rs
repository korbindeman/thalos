//! Top-left HUD panel: ship-view / map-view selector.
//!
//! Sits to the right of the warp/time panel. Two buttons, mutually
//! exclusive, drive the global [`ViewMode`] resource. The keyboard `M`
//! toggle in `view::toggle_view_input` keeps working — this panel is
//! just a clickable surface for the same state.

use bevy::prelude::*;

use crate::hud::HudPanel;
use crate::hud::TopLeftRowAnchor;
use crate::hud::theme::{HudTheme, panel_frame, panel_node};
use crate::view::ViewMode;

#[derive(Component, Clone, Copy)]
pub(super) struct ViewModeButton {
    pub target: ViewMode,
}

const BUTTON_WIDTH: f32 = 56.0;
const BUTTON_HEIGHT: f32 = 32.0;

pub fn setup(mut commands: Commands, theme: Res<HudTheme>, anchor: Res<TopLeftRowAnchor>) {
    let mut root = panel_node();
    root.position_type = PositionType::Relative;
    root.padding = UiRect::axes(Val::Px(10.0), Val::Px(8.0));
    root.row_gap = Val::Px(6.0);

    let (bg, border) = panel_frame(&theme);

    commands.entity(anchor.0).with_children(|row_parent| {
        row_parent
            .spawn((root, bg, border, HudPanel, Name::new("HudViewMode")))
            .with_children(|p| {
                p.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(6.0),
                    ..default()
                })
                .with_children(|row| {
                    spawn_button(row, &theme, ViewMode::Ship, "SHIP");
                    spawn_button(row, &theme, ViewMode::Map, "MAP");
                });
                p.spawn(subtitle(&theme, "VIEW"));
            });
    });
}

fn spawn_button(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    target: ViewMode,
    label: &str,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Px(BUTTON_WIDTH),
                height: Val::Px(BUTTON_HEIGHT),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            ViewModeButton { target },
            Name::new(format!("ViewMode_{}", label)),
        ))
        .with_children(|c| {
            c.spawn((
                Text::new(label),
                TextFont {
                    font: theme.font.clone(),
                    font_size: 13.0,
                    ..default()
                },
                TextColor(theme.text_dim),
            ));
        });
}

fn subtitle(theme: &HudTheme, content: impl Into<String>) -> impl Bundle {
    (
        Text::new(content),
        TextFont {
            font: theme.font.clone(),
            font_size: 11.0,
            ..default()
        },
        TextColor(theme.text_subtitle),
    )
}

pub fn handle_clicks(
    interactions: Query<(&Interaction, &ViewModeButton), Changed<Interaction>>,
    mut view: ResMut<ViewMode>,
) {
    for (interaction, button) in &interactions {
        if matches!(interaction, Interaction::Pressed) && *view != button.target {
            *view = button.target;
        }
    }
}

pub fn update_button_visuals(
    view: Res<ViewMode>,
    theme: Res<HudTheme>,
    mut buttons: Query<(
        &ViewModeButton,
        &Interaction,
        &mut BorderColor,
        &mut BackgroundColor,
        &Children,
    )>,
    mut text_q: Query<&mut TextColor>,
) {
    for (button, interaction, mut border, mut bg, children) in &mut buttons {
        let active = *view == button.target;
        let (border_color, bg_color) = match (active, interaction) {
            (true, _) => (theme.text_accent, theme.panel_bg),
            (false, Interaction::Pressed) => (theme.text_primary, theme.panel_border),
            (false, Interaction::Hovered) => (theme.text_primary, theme.panel_bg),
            (false, Interaction::None) => (theme.panel_border, theme.panel_bg),
        };
        let new_border = BorderColor::all(border_color);
        if border.top != new_border.top {
            *border = new_border;
        }
        if bg.0 != bg_color {
            bg.0 = bg_color;
        }
        let label_color = if active {
            theme.text_accent
        } else {
            theme.text_dim
        };
        if let Some(&child) = children.first()
            && let Ok(mut tc) = text_q.get_mut(child)
            && tc.0 != label_color
        {
            tc.0 = label_color;
        }
    }
}
