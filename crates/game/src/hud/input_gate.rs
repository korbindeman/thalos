//! UI pointer gating — prevents camera/scene input when the cursor is
//! over a HUD interactive element.
//!
//! Bevy's `Interaction` component sees `Hovered` whenever the pointer is
//! over a UI node. We aggregate that into a single resource that the
//! camera (and any other scene-input system) can read.

use bevy::prelude::*;

/// Updated every frame from [`update_ui_pointer_gate`].
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct UiPointerGate {
    /// True when the pointer is hovering over (or clicking) any
    /// interactive UI element.
    pub hovered: bool,
}

pub fn update_ui_pointer_gate(interactions: Query<&Interaction>, mut gate: ResMut<UiPointerGate>) {
    let hovered = interactions.iter().any(|i| !matches!(i, Interaction::None));
    if gate.hovered != hovered {
        gate.hovered = hovered;
    }
}
