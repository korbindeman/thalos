//! Interplanetary widget (stub).
//!
//! Registered against the MFD seam for transfer-window / heliocentric
//! plotting. `relevance` returns `None` for now (the `on_escape` context
//! signal is reserved but not yet populated), so it only appears when pinned,
//! showing a placeholder.

use bevy::prelude::*;

use crate::hud::theme::{HudTheme, label};

use super::super::{FlightContext, MfdWidgetRoot, WidgetKind};

pub(crate) fn relevance(_ctx: &FlightContext) -> Option<i32> {
    None
}

pub(crate) fn build(area: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    area.spawn((
        Node {
            width: Val::Px(200.0),
            min_height: Val::Px(140.0),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            row_gap: Val::Px(8.0),
            ..default()
        },
        Visibility::Hidden,
        MfdWidgetRoot {
            kind: WidgetKind::Interplanetary,
        },
        Name::new("MfdInterplanetary"),
    ))
    .with_children(|p| {
        p.spawn(label(theme, "INTERPLANETARY"));
        p.spawn((
            Text::new("NO DATA"),
            TextFont {
                font: theme.font.clone(),
                font_size: 14.0,
                ..default()
            },
            TextColor(theme.text_dim),
        ));
    });
}
