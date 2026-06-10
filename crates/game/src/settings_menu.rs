//! Input settings overlay.
//!
//! Opens from the pause menu via the "SETTINGS" button. Shows all keyboard /
//! mouse / controller bindings (read-only in this version) and a live-editable
//! HOTAS axis configuration — HOTAS changes take effect immediately because the
//! runtime reader polls `InputSettings` every frame.
//!
//! **Escape priority:** the escape-priority chain in `pause_menu` checks
//! `SettingsMenu::open` before `GamePause::active`, so Escape closes this
//! panel while leaving the pause backdrop up.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use thalos_input::settings::{AxisSpec, BindingSection, BindingSpec, HotasDeviceSelector, InputSettings};

// ── Resource ──────────────────────────────────────────────────────────────────

#[derive(Resource, Default)]
pub struct SettingsMenu {
    pub open: bool,
    tab: Tab,
}

#[derive(Default, PartialEq, Clone, Copy)]
enum Tab {
    #[default]
    Keyboard,
    Mouse,
    Controller,
    Hotas,
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct SettingsMenuPlugin;

impl Plugin for SettingsMenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SettingsMenu>().add_systems(
            bevy_egui::EguiPrimaryContextPass,
            settings_ui.after(crate::hud::theme::apply_egui_theme),
        );
    }
}

// ── Main UI system ─────────────────────────────────────────────────────────────

fn settings_ui(
    mut contexts: EguiContexts,
    mut settings_menu: ResMut<SettingsMenu>,
    mut input_settings: ResMut<InputSettings>,
) {
    if !settings_menu.open {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };

    let mut open = true;
    egui::Window::new("Input Settings")
        .open(&mut open)
        .resizable(false)
        .default_width(540.0)
        .collapsible(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                for (t, label) in [
                    (Tab::Keyboard, "Keyboard"),
                    (Tab::Mouse, "Mouse"),
                    (Tab::Controller, "Controller"),
                    (Tab::Hotas, "HOTAS"),
                ] {
                    if ui
                        .selectable_label(settings_menu.tab == t, label)
                        .clicked()
                    {
                        settings_menu.tab = t;
                    }
                }
            });
            ui.separator();

            match settings_menu.tab {
                Tab::Keyboard => show_keyboard_tab(ui, &input_settings),
                Tab::Mouse => show_mouse_tab(ui, &input_settings),
                Tab::Controller => show_controller_tab(ui, &input_settings),
                Tab::Hotas => show_hotas_tab(ui, &mut input_settings),
            }
        });

    if !open {
        settings_menu.open = false;
    }
}

// ── Keyboard tab ──────────────────────────────────────────────────────────────

fn show_keyboard_tab(ui: &mut egui::Ui, settings: &InputSettings) {
    egui::ScrollArea::vertical()
        .id_salt("kb_scroll")
        .max_height(420.0)
        .show(ui, |ui| {
            let sections: &[(&str, &BindingSection)] = &[
                ("Flight", &settings.game.flight),
                ("Warp", &settings.game.warp),
                ("View", &settings.game.view),
                ("EVA", &settings.game.eva),
                ("EVA Move", &settings.game.eva_move),
                ("Maneuver", &settings.game.maneuver),
                ("Maneuver Precision", &settings.game.maneuver_precision),
                ("System", &settings.game.system),
            ];
            for (title, section) in sections {
                show_binding_group(ui, title, section, is_keyboard_spec);
            }
        });
}

fn is_keyboard_spec(spec: &BindingSpec) -> bool {
    matches!(spec, BindingSpec::Key(_))
}

// ── Mouse tab ─────────────────────────────────────────────────────────────────

fn show_mouse_tab(ui: &mut egui::Ui, settings: &InputSettings) {
    egui::ScrollArea::vertical()
        .id_salt("mouse_scroll")
        .max_height(420.0)
        .show(ui, |ui| {
            let sections: &[(&str, &BindingSection)] = &[
                ("Camera", &settings.game.camera),
                ("Flight", &settings.game.flight),
                ("Warp", &settings.game.warp),
                ("Maneuver", &settings.game.maneuver),
            ];
            let any = sections
                .iter()
                .any(|(_, s)| section_has_spec(s, is_mouse_spec));
            if any {
                for (title, section) in sections {
                    show_binding_group(ui, title, section, is_mouse_spec);
                }
            } else {
                empty_hint(ui, "No mouse bindings configured.");
            }
        });
}

fn is_mouse_spec(spec: &BindingSpec) -> bool {
    matches!(
        spec,
        BindingSpec::MouseButton(_) | BindingSpec::MouseMotion | BindingSpec::MouseWheel
    )
}

// ── Controller tab ────────────────────────────────────────────────────────────

fn show_controller_tab(ui: &mut egui::Ui, settings: &InputSettings) {
    egui::ScrollArea::vertical()
        .id_salt("ctrl_scroll")
        .max_height(420.0)
        .show(ui, |ui| {
            let sections: &[(&str, &BindingSection)] = &[
                ("Flight", &settings.game.flight),
                ("Warp", &settings.game.warp),
                ("View", &settings.game.view),
                ("EVA", &settings.game.eva),
                ("EVA Move", &settings.game.eva_move),
                ("Maneuver", &settings.game.maneuver),
                ("System", &settings.game.system),
            ];
            let any = sections
                .iter()
                .any(|(_, s)| section_has_spec(s, is_gamepad_spec));
            if any {
                for (title, section) in sections {
                    show_binding_group(ui, title, section, is_gamepad_spec);
                }
            } else {
                empty_hint(ui, "No controller bindings configured.");
                ui.add_space(4.0);
                ui.weak("Add GamepadButton(…) entries to assets/input.ron.");
            }
        });
}

fn is_gamepad_spec(spec: &BindingSpec) -> bool {
    matches!(
        spec,
        BindingSpec::GamepadButton(_) | BindingSpec::GamepadAxis(_)
    )
}

// ── HOTAS tab ─────────────────────────────────────────────────────────────────

fn show_hotas_tab(ui: &mut egui::Ui, settings: &mut InputSettings) {
    let hotas = &mut settings.game.hotas;

    ui.horizontal(|ui| {
        ui.checkbox(&mut hotas.enabled, "HOTAS enabled");
    });

    ui.add_space(6.0);

    // Device selector
    ui.horizontal(|ui| {
        ui.label("Device:");
        let is_any = matches!(hotas.device, HotasDeviceSelector::Any);
        let is_name = matches!(hotas.device, HotasDeviceSelector::NameContains(_));

        if ui.selectable_label(is_any, "Any").clicked() && !is_any {
            hotas.device = HotasDeviceSelector::Any;
        }
        if ui.selectable_label(is_name, "Name contains").clicked() && !is_name {
            hotas.device = HotasDeviceSelector::NameContains(String::new());
        }
        if let HotasDeviceSelector::NameContains(ref mut name) = hotas.device {
            ui.text_edit_singleline(name);
        }
    });

    ui.add_space(8.0);
    ui.separator();
    ui.add_space(4.0);

    let axes_order = ["pitch", "yaw", "roll", "throttle"];

    ui.add_enabled_ui(hotas.enabled, |ui| {
        egui::Grid::new("hotas_axes_grid")
            .num_columns(5)
            .spacing([8.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.strong("Axis");
                ui.strong("Raw code");
                ui.strong("Invert");
                ui.strong("Deadzone");
                ui.strong("");
                ui.end_row();

                for axis_name in axes_order.iter() {
                    let has_binding = hotas.axes.contains_key(*axis_name);

                    ui.label(*axis_name);

                    if has_binding {
                        // Clone for safe independent mutation
                        let mut binding = hotas.axes.get(*axis_name).unwrap().clone();
                        let mut remove = false;

                        // Raw platform axis code (gilrs Code::into_u32). Discover
                        // values via `cargo run -p thalos_input --example
                        // gamepad_axes`; codes are platform-specific.
                        ui.add(
                            egui::DragValue::new(&mut binding.code)
                                .speed(1.0)
                                .prefix("code "),
                        );

                        ui.checkbox(&mut binding.invert, "");
                        ui.add(
                            egui::Slider::new(&mut binding.deadzone, 0.0..=0.9)
                                .step_by(0.01)
                                .fixed_decimals(2),
                        );
                        if ui.small_button("✕").clicked() {
                            remove = true;
                        }

                        if remove {
                            hotas.axes.remove(*axis_name);
                        } else {
                            hotas.axes.insert(axis_name.to_string(), binding);
                        }
                    } else {
                        ui.weak("—");
                        ui.label("");
                        ui.label("");
                        if ui.small_button("Add").clicked() {
                            hotas.axes.insert(
                                axis_name.to_string(),
                                thalos_input::settings::HotasAxisBinding {
                                    code: 0,
                                    device: None,
                                    invert: false,
                                    deadzone: 0.05,
                                    min: -1.0,
                                    max: 1.0,
                                },
                            );
                        }
                    }

                    ui.end_row();
                }
            });
    });

    ui.add_space(8.0);
    ui.weak(
        "Raw codes are platform-specific — find yours with the gamepad_axes \
         example. Changes apply immediately; edit assets/input.ron to persist.",
    );
}

// ── Shared helpers ─────────────────────────────────────────────────────────────

fn section_has_spec(section: &BindingSection, filter: fn(&BindingSpec) -> bool) -> bool {
    section.bindings.values().any(|v| v.iter().any(filter))
        || section.axes.values().any(|a: &AxisSpec| {
            a.positive.iter().any(filter) || a.negative.iter().any(filter)
        })
}

fn show_binding_group(
    ui: &mut egui::Ui,
    title: &str,
    section: &BindingSection,
    filter: fn(&BindingSpec) -> bool,
) {
    if !section_has_spec(section, filter) {
        return;
    }

    ui.add_space(4.0);
    ui.label(egui::RichText::new(title.to_uppercase()).strong().size(10.0));

    egui::Grid::new(format!("bindings_{title}"))
        .num_columns(2)
        .spacing([16.0, 2.0])
        .show(ui, |ui| {
            for (action, specs) in &section.bindings {
                let filtered: Vec<String> = specs
                    .iter()
                    .filter(|s| filter(s))
                    .map(format_binding)
                    .collect();
                if filtered.is_empty() {
                    continue;
                }
                ui.weak(format_action(action));
                ui.label(filtered.join(" / "));
                ui.end_row();
            }

            for (axis_name, axis) in &section.axes {
                let pos: Vec<String> = axis
                    .positive
                    .iter()
                    .filter(|s| filter(s))
                    .map(format_binding)
                    .collect();
                let neg: Vec<String> = axis
                    .negative
                    .iter()
                    .filter(|s| filter(s))
                    .map(format_binding)
                    .collect();
                if pos.is_empty() && neg.is_empty() {
                    continue;
                }
                ui.weak(format!("{} +/−", format_action(axis_name)));
                let label = match (pos.is_empty(), neg.is_empty()) {
                    (false, false) => format!("{} / {}", pos.join(", "), neg.join(", ")),
                    (false, true) => pos.join(", "),
                    (true, false) => neg.join(", "),
                    (true, true) => unreachable!(),
                };
                ui.label(label);
                ui.end_row();
            }
        });
}

fn empty_hint(ui: &mut egui::Ui, msg: &str) {
    ui.vertical_centered(|ui| {
        ui.add_space(20.0);
        ui.weak(msg);
    });
}

// ── Formatting helpers ─────────────────────────────────────────────────────────

fn format_binding(spec: &BindingSpec) -> String {
    match spec {
        BindingSpec::Key(key) => format_keycode(*key),
        BindingSpec::MouseButton(btn) => format_mouse_button(*btn),
        BindingSpec::MouseMotion => "Mouse Move".into(),
        BindingSpec::MouseWheel => "Scroll Wheel".into(),
        BindingSpec::GamepadButton(btn) => format_gamepad_button(*btn).into(),
        BindingSpec::GamepadAxis(axis) => format_gamepad_axis(*axis).into(),
    }
}

fn format_keycode(key: KeyCode) -> String {
    let s: &str = match key {
        KeyCode::KeyA => "A",
        KeyCode::KeyB => "B",
        KeyCode::KeyC => "C",
        KeyCode::KeyD => "D",
        KeyCode::KeyE => "E",
        KeyCode::KeyF => "F",
        KeyCode::KeyG => "G",
        KeyCode::KeyH => "H",
        KeyCode::KeyI => "I",
        KeyCode::KeyJ => "J",
        KeyCode::KeyK => "K",
        KeyCode::KeyL => "L",
        KeyCode::KeyM => "M",
        KeyCode::KeyN => "N",
        KeyCode::KeyO => "O",
        KeyCode::KeyP => "P",
        KeyCode::KeyQ => "Q",
        KeyCode::KeyR => "R",
        KeyCode::KeyS => "S",
        KeyCode::KeyT => "T",
        KeyCode::KeyU => "U",
        KeyCode::KeyV => "V",
        KeyCode::KeyW => "W",
        KeyCode::KeyX => "X",
        KeyCode::KeyY => "Y",
        KeyCode::KeyZ => "Z",
        KeyCode::Digit0 => "0",
        KeyCode::Digit1 => "1",
        KeyCode::Digit2 => "2",
        KeyCode::Digit3 => "3",
        KeyCode::Digit4 => "4",
        KeyCode::Digit5 => "5",
        KeyCode::Digit6 => "6",
        KeyCode::Digit7 => "7",
        KeyCode::Digit8 => "8",
        KeyCode::Digit9 => "9",
        KeyCode::F1 => "F1",
        KeyCode::F2 => "F2",
        KeyCode::F3 => "F3",
        KeyCode::F4 => "F4",
        KeyCode::F5 => "F5",
        KeyCode::F6 => "F6",
        KeyCode::F7 => "F7",
        KeyCode::F8 => "F8",
        KeyCode::F9 => "F9",
        KeyCode::F10 => "F10",
        KeyCode::F11 => "F11",
        KeyCode::F12 => "F12",
        KeyCode::Space => "Space",
        KeyCode::Escape => "Esc",
        KeyCode::Enter => "Enter",
        KeyCode::NumpadEnter => "Num Enter",
        KeyCode::Tab => "Tab",
        KeyCode::Backspace => "Bksp",
        KeyCode::Delete => "Del",
        KeyCode::Insert => "Ins",
        KeyCode::Home => "Home",
        KeyCode::End => "End",
        KeyCode::PageUp => "PgUp",
        KeyCode::PageDown => "PgDn",
        KeyCode::ArrowUp => "↑",
        KeyCode::ArrowDown => "↓",
        KeyCode::ArrowLeft => "←",
        KeyCode::ArrowRight => "→",
        KeyCode::ShiftLeft => "Shift",
        KeyCode::ShiftRight => "Shift",
        KeyCode::ControlLeft => "Ctrl",
        KeyCode::ControlRight => "Ctrl",
        KeyCode::AltLeft => "Alt",
        KeyCode::AltRight => "Alt",
        KeyCode::SuperLeft => "Super",
        KeyCode::SuperRight => "Super",
        KeyCode::CapsLock => "CapsLock",
        KeyCode::Period => ".",
        KeyCode::Comma => ",",
        KeyCode::Slash => "/",
        KeyCode::Backslash => "\\",
        KeyCode::Semicolon => ";",
        KeyCode::Quote => "'",
        KeyCode::BracketLeft => "[",
        KeyCode::BracketRight => "]",
        KeyCode::Minus => "-",
        KeyCode::Equal => "=",
        KeyCode::Backquote => "`",
        _ => return format!("{key:?}"),
    };
    s.to_string()
}

fn format_mouse_button(btn: MouseButton) -> String {
    match btn {
        MouseButton::Left => "LMB".into(),
        MouseButton::Right => "RMB".into(),
        MouseButton::Middle => "MMB".into(),
        MouseButton::Other(n) => format!("Mouse {n}"),
        _ => format!("{btn:?}"),
    }
}

fn format_gamepad_axis(axis: GamepadAxis) -> &'static str {
    match axis {
        GamepadAxis::LeftStickX => "L Stick X",
        GamepadAxis::LeftStickY => "L Stick Y",
        GamepadAxis::LeftZ => "L Z / Trigger",
        GamepadAxis::RightStickX => "R Stick X",
        GamepadAxis::RightStickY => "R Stick Y",
        GamepadAxis::RightZ => "R Z / Trigger",
        _ => "Other",
    }
}

fn format_gamepad_button(btn: GamepadButton) -> &'static str {
    match btn {
        GamepadButton::South => "A (South)",
        GamepadButton::East => "B (East)",
        GamepadButton::West => "X (West)",
        GamepadButton::North => "Y (North)",
        GamepadButton::LeftTrigger => "L Trigger",
        GamepadButton::LeftTrigger2 => "LT2",
        GamepadButton::RightTrigger => "R Trigger",
        GamepadButton::RightTrigger2 => "RT2",
        GamepadButton::Select => "Select",
        GamepadButton::Start => "Start",
        GamepadButton::LeftThumb => "L Stick",
        GamepadButton::RightThumb => "R Stick",
        GamepadButton::DPadUp => "D-Pad ↑",
        GamepadButton::DPadDown => "D-Pad ↓",
        GamepadButton::DPadLeft => "D-Pad ←",
        GamepadButton::DPadRight => "D-Pad →",
        GamepadButton::Mode => "Mode",
        GamepadButton::C => "C",
        GamepadButton::Z => "Z",
        _ => "Other",
    }
}

fn format_action(name: &str) -> String {
    const ACRONYMS: &[&str] = &["sas", "eva", "hud", "rcs", "ap", "pe", "asl"];
    name.split('_')
        .map(|word| {
            if ACRONYMS.contains(&word) {
                word.to_uppercase()
            } else {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => {
                        let upper: String = first.to_uppercase().collect();
                        upper + chars.as_str()
                    }
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
