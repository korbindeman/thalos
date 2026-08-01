//! Docking widget (stub).
//!
//! Registered against the MFD seam so it can be pinned and so the alignment
//! display drops in later. Targets today are bodies, not craft/ports
//! ([`thalos_game_state::nav::TargetBody`]), so `relevance` returns `None` and the
//! widget shows a placeholder until a dock/port target model exists.

use bevy::prelude::*;

use crate::theme::{HudTheme, label};

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
            kind: WidgetKind::Docking,
        },
        Name::new("MfdDocking"),
    ))
    .with_children(|p| {
        p.spawn(label(theme, "DOCKING"));
        p.spawn((
            Text::new("NO DATA"),
            TextFont {
                font: theme.font.clone(),
                font_size: FontSize::Px(14.0),
                ..default()
            },
            TextColor(theme.text_dim),
        ));
    });
}
