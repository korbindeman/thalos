//! Reusable widgets, all styled from the tokens in [`crate::tokens`].

pub mod button;
pub mod checkbox;
pub mod cycle;
pub mod panel;
pub mod scroll;
pub mod slider;
pub mod text_field;
pub mod toast;

pub use button::{
    ButtonDesc, ButtonLabel, ButtonVariant, UiButton, spawn_button, spawn_menu_row, style_buttons,
};
pub use checkbox::{UiCheckbox, drive_checkboxes, spawn_checkbox_row, update_checkbox_visuals};
pub use cycle::{UiCycle, drive_cycles, spawn_cycle_row, update_cycle_visuals};
pub use panel::{
    floating_panel_node, panel_node, spawn_divider, spawn_heading, spawn_key_hint, spawn_value_row,
};
pub use scroll::{ScrollableColumn, scroll_column_node, scroll_scrollables};
pub use slider::{
    SliderFill, SliderFormat, SliderValueText, UiSlider, drive_sliders, spawn_slider_row,
    update_slider_visuals,
};
pub use text_field::{
    TextFieldFocus, TextFieldText, UiTextField, apply_text_field_input, focus_text_fields,
    spawn_text_field, update_text_field_visuals,
};
pub use toast::{Toast, ToastArea, ToastKind, spawn_toast, update_toasts};
