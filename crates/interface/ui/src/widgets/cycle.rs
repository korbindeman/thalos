//! `[‹] value [›]` one-of-N picker — a dropdown without a popup, which sorts
//! reliably under Bevy UI's z-order.

use bevy::prelude::*;

use crate::UiTheme;
use crate::tokens::*;
use crate::widgets::button::{ButtonVariant, spawn_button};

/// A one-of-N picker. Holds its own option labels so it renders the current
/// value without help; consumers react to `Changed<UiCycle>` and map `index`
/// onto their real value.
#[derive(Component, Debug, Clone)]
pub struct UiCycle {
    pub index: usize,
    pub options: Vec<String>,
}

impl UiCycle {
    fn len(&self) -> usize {
        self.options.len()
    }
}

/// Prev/next arrow inside a cycle. Its parent ([`ChildOf`]) is the [`UiCycle`]
/// track, so the driver needs no entity back-reference.
#[derive(Component)]
pub struct CycleArrow {
    delta: i32,
}

/// Value label; its parent ([`ChildOf`]) is the [`UiCycle`] track.
#[derive(Component)]
pub struct CycleValueText;

/// Spawn a labelled cycle row: `LABEL  ‹ value ›`. Returns the track entity
/// (carries [`UiCycle`] + `binding`).
pub fn spawn_cycle_row(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    label: &str,
    options: Vec<String>,
    index: usize,
    binding: impl Bundle,
) -> Entity {
    let index = index.min(options.len().saturating_sub(1));
    let current = options.get(index).cloned().unwrap_or_default();
    let mut track_entity = Entity::PLACEHOLDER;
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Px(CTRL_H),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(SPACE_SM),
            ..Default::default()
        })
        .with_children(|row| {
            row.spawn((
                theme.small(label),
                Node {
                    width: Val::Px(120.0),
                    flex_shrink: 0.0,
                    ..Default::default()
                },
            ));
            track_entity = row
                .spawn((
                    Node {
                        flex_grow: 1.0,
                        height: Val::Px(CTRL_H_SM),
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        column_gap: Val::Px(SPACE_XS + 2.0),
                        ..Default::default()
                    },
                    UiCycle { index, options },
                ))
                .insert(binding)
                .with_children(|track| {
                    spawn_cycle_arrow(track, theme, -1, "‹");
                    track.spawn((
                        theme.body(current),
                        Node {
                            flex_grow: 1.0,
                            justify_content: JustifyContent::Center,
                            ..Default::default()
                        },
                        CycleValueText,
                    ));
                    spawn_cycle_arrow(track, theme, 1, "›");
                })
                .id();
        });
    track_entity
}

fn spawn_cycle_arrow(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    delta: i32,
    glyph: &str,
) {
    let arrow = spawn_button(
        parent,
        theme,
        CycleArrow { delta },
        glyph,
        ButtonVariant::Ghost,
        20.0,
    );
    parent
        .commands_mut()
        .entity(arrow)
        .entry::<Node>()
        .and_modify(|mut node| {
            node.width = Val::Px(20.0);
            node.padding = UiRect::ZERO;
            node.flex_shrink = 0.0;
        });
}

/// Advance a cycle's index when its arrows are pressed (wrapping).
pub fn drive_cycles(
    arrows: Query<(&Interaction, &ChildOf, &CycleArrow), Changed<Interaction>>,
    mut cycles: Query<&mut UiCycle>,
) {
    for (interaction, child_of, arrow) in &arrows {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        let Ok(mut cycle) = cycles.get_mut(child_of.0) else {
            continue;
        };
        let len = cycle.len();
        if len == 0 {
            continue;
        }
        let next = (cycle.index as i32 + arrow.delta).rem_euclid(len as i32) as usize;
        if next != cycle.index {
            cycle.index = next;
        }
    }
}

/// Render each cycle's current option into its value label.
pub fn update_cycle_visuals(
    cycles: Query<&UiCycle>,
    mut labels: Query<(&ChildOf, &mut Text), With<CycleValueText>>,
) {
    for (child_of, mut text) in &mut labels {
        let Ok(cycle) = cycles.get(child_of.0) else {
            continue;
        };
        let value = cycle.options.get(cycle.index).cloned().unwrap_or_default();
        if **text != value {
            **text = value;
        }
    }
}
