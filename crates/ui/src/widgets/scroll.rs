//! Wheel-scrollable columns.

use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;

/// Marker for wheel-scrollable columns. Spawn with `Overflow::scroll_y()`,
/// `ScrollPosition`, `RelativeCursorPosition`, and `Interaction`.
#[derive(Component)]
pub struct ScrollableColumn;

/// A `Node` + companions bundle pre-configured as a scrollable column.
pub fn scroll_column_node() -> impl Bundle {
    (
        Node {
            flex_direction: FlexDirection::Column,
            overflow: Overflow::scroll_y(),
            flex_grow: 1.0,
            ..Default::default()
        },
        ScrollPosition::default(),
        RelativeCursorPosition::default(),
        Interaction::None,
        ScrollableColumn,
    )
}

/// Wheel-scroll whichever scrollable column the cursor is over.
pub fn scroll_scrollables(
    mut wheel: MessageReader<MouseWheel>,
    mut scrollables: Query<(&RelativeCursorPosition, &mut ScrollPosition), With<ScrollableColumn>>,
) {
    for event in wheel.read() {
        let dy = match event.unit {
            MouseScrollUnit::Line => event.y * 28.0,
            MouseScrollUnit::Pixel => event.y,
        };
        if dy == 0.0 {
            continue;
        }
        for (rel, mut scroll) in &mut scrollables {
            if rel.cursor_over() {
                scroll.y -= dy;
            }
        }
    }
}
