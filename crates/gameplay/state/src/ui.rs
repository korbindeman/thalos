//! Shared game-specific UI-facing state and markers read across feature crates.
//! Cross-application photo-mode state lives in `thalos_photo_mode`; the
//! per-device input gates and HUD-panel marker remain on the game blackboard.

use bevy::prelude::*;

/// Marker for every HUD root container so the HUD's `hide_in_photo_mode` (and
/// the editors, which must blank the flight UI) can toggle them all.
#[derive(Component)]
pub struct HudPanel;

/// True while a UI surface owns the pointer: the cursor is over (or dragging
/// with) an interactive Bevy-UI element or an egui area.
///
/// **Sole writer:** the HUD's `update_ui_input_gates` — one answer per device
/// to "does a UI surface own this input right now?", so scene systems never
/// have to know *which* UI is up.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct UiPointerGate {
    pub hovered: bool,
}

/// True while a text-entry surface owns the keyboard: a focused
/// `thalos_ui::UiTextField` or an egui widget taking keystrokes.
///
/// Every gameplay keyboard consumer gates on this — the enhanced-input
/// keyboard source and the few systems that read `ButtonInput<KeyCode>` raw
/// (freecam, god-view pan) — so a typed character can never double as a
/// flight control.
///
/// **Sole writer:** the HUD's `update_ui_input_gates`.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct UiKeyboardGate {
    pub text_entry: bool,
}

impl UiKeyboardGate {
    /// Is a text field (native or egui) eating keystrokes this frame?
    pub fn text_entry(&self) -> bool {
        self.text_entry
    }
}

/// Marker: entities with this component are hidden while the view is
/// [`ViewMode::Ship`]. Attach to overlays that only make sense in the
/// far-scale map view (planet icons, impostor billboards, maneuver arrows,
/// ghost bodies, the flat ship marker).
#[derive(Component)]
pub struct HideInShipView;

#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct GamePause {
    pub active: bool,
}

pub fn not_game_paused(pause: Res<GamePause>, scenario: Res<ScenarioMenu>) -> bool {
    !pause.active && !scenario.open
}

/// Whether the destruction scenario picker is shown (and the game halted).
///
/// Mirrors `Simulation::is_destroyed()` via [`sync_menu_to_destruction`]; read
/// as a pause source by [`crate::pause_menu`]. Default `false`; the resource is
/// inserted at plugin build so the `not_game_paused` run condition can read it
/// from the first frame.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct ScenarioMenu {
    pub open: bool,
}
