//! Labelled boolean toggle.

use bevy::prelude::*;

use crate::UiTheme;
use crate::tokens::*;

/// A labelled boolean toggle. The whole row is the button; clicking flips
/// `checked`. Consumers react to `Changed<UiCheckbox>`.
#[derive(Component, Debug, Clone, Copy)]
pub struct UiCheckbox {
    pub checked: bool,
}

/// Marker on the filled square inside a checkbox row.
#[derive(Component)]
pub struct CheckboxBox;

/// Spawn a checkbox row: `[x] Label`. Returns the row entity (carries
/// [`UiCheckbox`] + `binding`).
pub fn spawn_checkbox_row(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    label: &str,
    checked: bool,
    binding: impl Bundle,
) -> Entity {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(CTRL_H_SM),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(SPACE_SM),
                ..Default::default()
            },
            Interaction::None,
            UiCheckbox { checked },
            binding,
        ))
        .with_children(|row| {
            row.spawn((
                Node {
                    width: Val::Px(14.0),
                    height: Val::Px(14.0),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    flex_shrink: 0.0,
                    ..Default::default()
                },
                BackgroundColor(if checked { ACCENT } else { Color::NONE }),
                BorderColor::all(STROKE),
                CheckboxBox,
            ));
            row.spawn(theme.body(label));
        })
        .id()
}

/// Flip a checkbox on press.
pub fn drive_checkboxes(mut boxes: Query<(&Interaction, &mut UiCheckbox), Changed<Interaction>>) {
    for (interaction, mut checkbox) in &mut boxes {
        if matches!(interaction, Interaction::Pressed) {
            checkbox.checked = !checkbox.checked;
        }
    }
}

/// Fill / border tint for each checkbox from its state + hover.
pub fn update_checkbox_visuals(
    boxes: Query<(&UiCheckbox, &Interaction, &Children)>,
    mut squares: Query<(&mut BackgroundColor, &mut BorderColor), With<CheckboxBox>>,
) {
    for (checkbox, interaction, children) in &boxes {
        let Some(&child) = children.first() else {
            continue;
        };
        let Ok((mut bg, mut border)) = squares.get_mut(child) else {
            continue;
        };
        let fill = if checkbox.checked {
            ACCENT
        } else {
            Color::NONE
        };
        let border_color = if matches!(interaction, Interaction::Hovered | Interaction::Pressed) {
            STROKE_BRIGHT
        } else {
            STROKE
        };
        if bg.0 != fill {
            bg.0 = fill;
        }
        let new_border = BorderColor::all(border_color);
        if border.top != new_border.top {
            *border = new_border;
        }
    }
}
