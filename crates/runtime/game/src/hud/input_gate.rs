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

/// True while a UI surface owns the pointer: the cursor is over (or dragging
/// with) an interactive Bevy-UI element or an egui area.
///
/// **Sole writer:** [`update_ui_input_gates`].
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct UiPointerGate {
    pub hovered: bool,
}

/// True while a text-entry surface owns the keyboard: a focused
/// `thalos_ui::UiTextField` or an egui widget taking keystrokes.
///
/// Every gameplay keyboard consumer gates on this — the enhanced-input
/// keyboard source (`crate::input::gate_enhanced_input_sources`) and the few
/// systems that read `ButtonInput<KeyCode>` raw (freecam, god-view pan) — so a
/// typed character can never double as a flight control.
///
/// **Sole writer:** [`update_ui_input_gates`].
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
