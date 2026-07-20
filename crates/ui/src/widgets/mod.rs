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
    spawn_button, spawn_menu_row, style_buttons, ButtonDesc, ButtonLabel, ButtonVariant, UiButton,
};
pub use checkbox::{drive_checkboxes, spawn_checkbox_row, update_checkbox_visuals, UiCheckbox};
pub use cycle::{drive_cycles, spawn_cycle_row, update_cycle_visuals, UiCycle};
pub use panel::{
    floating_panel_node, panel_node, spawn_divider, spawn_heading, spawn_key_hint,
    spawn_value_row,
};
pub use scroll::{scroll_column_node, scroll_scrollables, ScrollableColumn};
pub use slider::{
    drive_sliders, spawn_slider_row, update_slider_visuals, SliderFill, SliderFormat,
    SliderValueText, UiSlider,
};
pub use text_field::{
    apply_text_field_input, focus_text_fields, spawn_text_field, update_text_field_visuals,
    TextFieldFocus, TextFieldText, UiTextField,
};
pub use toast::{spawn_toast, update_toasts, Toast, ToastArea, ToastKind};
