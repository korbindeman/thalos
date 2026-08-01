//! UI input gating — one answer per device to "does a UI surface own this
//! input right now?", so scene systems never have to know *which* UI is up.
//!
//! Thalos draws two interfaces over the same 3-D view: native Bevy UI (the
//! HUD, `thalos_ui` panels and text fields) and the egui F8 viewpoint
//! manager. Both can take the pointer, and both can take the keyboard while a
//! name is being typed. A reader that consults only one of them leaks the
//! other — typing a viewpoint name into the egui manager used to fly the
//! freecam and pan the god-view, because the raw key readers knew about
//! `thalos_ui::TextFieldFocus` and nothing else.
//!
//! So: **one resource per device, one writer, every consumer reads it.** Add a
//! third UI system and it gets wired in here, not hunted through every
//! keyboard reader.

use bevy::prelude::*;
use bevy_egui::input::EguiWantsInput;

// The gate resources moved to `thalos_game_state::ui` (Phase 5b); the
// sole writer below stays here with the egui/native fold logic.
pub use thalos_game_state::ui::{UiKeyboardGate, UiPointerGate};

/// Fold both UI systems into [`UiPointerGate`] and [`UiKeyboardGate`].
///
/// `EguiWantsInput` is absent in the headless capture app, which builds no
/// egui context; the native half stands alone there.
pub fn update_ui_input_gates(
    interactions: Query<&Interaction>,
    text_focus: Res<thalos_ui::TextFieldFocus>,
    egui: Option<Res<EguiWantsInput>>,
    mut pointer: ResMut<UiPointerGate>,
    mut keyboard: ResMut<UiKeyboardGate>,
) {
    let egui = egui.as_deref();
    let hovered = egui.is_some_and(EguiWantsInput::wants_any_pointer_input)
        || interactions.iter().any(|i| !matches!(i, Interaction::None));
    let text_entry =
        text_focus.is_focused() || egui.is_some_and(EguiWantsInput::wants_any_keyboard_input);

    if pointer.hovered != hovered {
        pointer.hovered = hovered;
    }
    if keyboard.text_entry != text_entry {
        keyboard.text_entry = text_entry;
    }
}
