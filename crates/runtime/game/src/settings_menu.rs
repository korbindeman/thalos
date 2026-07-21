//! Settings overlay (native Bevy UI).
//!
//! Opens from the pause menu / start screen via the "SETTINGS" button. A
//! centred modal with a tab strip; the body is rebuilt from the current tab +
//! model whenever the tab changes, the menu opens, or a structural edit bumps
//! [`SettingsMenu::rebuild`] (HOTAS add/remove, device-mode switch). Interactive
//! widgets come from [`thalos_ui`]; per-tab apply systems read
//! `Changed<Widget>` and write the backing resource (value-compared, so an open
//! tab never churns change detection).
//!
//! - **Window** — live-edits the persisted [`WindowSettings`].
//! - **Graphics** — live-edits the persisted [`GraphicsSettings`].
//! - **Keyboard / Mouse / Controller** — read-only binding lists.
//! - **HOTAS** — live-editable axis configuration (`InputSettings`); the runtime
//!   reader polls `InputSettings` each frame, so changes apply immediately.
//!
//! **Escape priority:** the chain in `pause_menu` checks `SettingsMenu::open`
//! before `GamePause::active`, so Escape closes this panel while leaving the
//! pause backdrop up.

use bevy::picking::Pickable;
use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;
use bevy::window::{Monitor, PrimaryMonitor};
use thalos_input::settings::{
    AxisSpec, BindingSection, BindingSpec, HotasAxisBinding, HotasDeviceSelector, InputSettings,
};

use thalos_ui::{
    self as ui, ButtonVariant, ScrollableColumn, SliderFormat, UiButton, UiCheckbox, UiCycle,
    UiSlider, UiTextField, UiTheme, spawn_button, spawn_checkbox_row, spawn_cycle_row,
    spawn_divider, spawn_slider_row, spawn_text_field, tokens,
};

use crate::graphics_settings::{GraphicsSettings, MsaaSetting};
use crate::units_settings::{UnitSystem, UnitsSettings};
use crate::window_settings::{
    MonitorChoice, RESOLUTION_PRESETS, UI_SCALE_MAX, UI_SCALE_MIN, WindowModeSetting,
    WindowSettings, WindowSettingsOverrides,
};

// ── Resource ──────────────────────────────────────────────────────────────────

#[derive(Resource, Default)]
pub struct SettingsMenu {
    pub open: bool,
    tab: Tab,
    /// Bumped to force a tab-body rebuild after a structural change (HOTAS
    /// add/remove, device-mode switch, reset-to-defaults).
    rebuild: u32,
}

impl SettingsMenu {
    fn dirty(&mut self) {
        self.rebuild = self.rebuild.wrapping_add(1);
    }
}

#[derive(Default, PartialEq, Eq, Clone, Copy)]
enum Tab {
    #[default]
    Window,
    Graphics,
    Units,
    Keyboard,
    Mouse,
    Controller,
    Hotas,
}

impl Tab {
    const ALL: [Tab; 7] = [
        Tab::Window,
        Tab::Graphics,
        Tab::Units,
        Tab::Keyboard,
        Tab::Mouse,
        Tab::Controller,
        Tab::Hotas,
    ];

    fn label(self) -> &'static str {
        match self {
            Tab::Window => "Window",
            Tab::Graphics => "Graphics",
            Tab::Units => "Units",
            Tab::Keyboard => "Keyboard",
            Tab::Mouse => "Mouse",
            Tab::Controller => "Controller",
            Tab::Hotas => "HOTAS",
        }
    }
}

const HOTAS_AXES: [&str; 4] = ["pitch", "yaw", "roll", "throttle"];

// ── Markers ─────────────────────────────────────────────────────────────────

#[derive(Component)]
struct SettingsRoot;

#[derive(Component)]
struct SettingsTabBody;

#[derive(Component, Clone, Copy)]
struct TabButton(Tab);

#[derive(Component)]
struct CloseButton;

// Window tab
#[derive(Component)]
struct WindowModeControl;
#[derive(Component)]
struct ResolutionControl {
    values: Vec<(u32, u32)>,
}
#[derive(Component)]
struct MonitorControl {
    names: Vec<Option<String>>,
}
#[derive(Component)]
struct VsyncControl;
#[derive(Component)]
struct UiScaleControl;
#[derive(Component)]
struct ResetWindowControl;

// Graphics tab
#[derive(Component)]
struct CloudsControl;
#[derive(Component)]
struct GrassControl;
#[derive(Component)]
struct GpuGrassControl;
#[derive(Component)]
struct MsaaControl;
#[derive(Component)]
struct ResetGraphicsControl;

// Units tab
#[derive(Component)]
struct UnitsControl;

// HOTAS tab
#[derive(Component)]
struct HotasEnabledControl;
#[derive(Component)]
struct HotasDeviceModeControl;
#[derive(Component)]
struct HotasDeviceNameControl;
#[derive(Component)]
struct HotasCodeControl {
    axis: &'static str,
}
#[derive(Component)]
struct HotasInvertControl {
    axis: &'static str,
}
#[derive(Component)]
struct HotasDeadzoneControl {
    axis: &'static str,
}
#[derive(Component)]
struct HotasAddControl {
    axis: &'static str,
}
#[derive(Component)]
struct HotasRemoveControl {
    axis: &'static str,
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct SettingsMenuPlugin;

impl Plugin for SettingsMenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SettingsMenu>()
            .add_systems(Startup, setup_ui.after(thalos_ui::init_ui_theme))
            .add_systems(
                Update,
                (
                    sync_visibility,
                    handle_close_click,
                    handle_tab_clicks,
                    update_tab_latches,
                    rebuild_tab_body,
                    apply_window_controls,
                    apply_graphics_controls,
                    apply_units_controls,
                    apply_hotas_controls,
                ),
            );
    }
}

// ── Setup ─────────────────────────────────────────────────────────────────────

fn setup_ui(mut commands: Commands, theme: Res<UiTheme>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.45)),
            // Above the pause menu backdrop (z 100) so settings stacks over it.
            GlobalZIndex(110),
            Pickable {
                is_hoverable: false,
                should_block_lower: true,
            },
            Visibility::Hidden,
            SettingsRoot,
            Name::new("SettingsMenu"),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: Val::Px(560.0),
                    height: Val::Px(540.0),
                    max_height: Val::Percent(92.0),
                    ..ui::panel_node()
                },
                theme.glass_heavy(),
                Name::new("SettingsPanel"),
            ))
            .with_children(|panel| {
                // Title row + close.
                panel
                    .spawn(Node {
                        width: Val::Percent(100.0),
                        flex_direction: FlexDirection::Row,
                        justify_content: JustifyContent::SpaceBetween,
                        align_items: AlignItems::Center,
                        ..default()
                    })
                    .with_children(|row| {
                        row.spawn(theme.title("SETTINGS"));
                        spawn_button(row, &theme, CloseButton, "×", ButtonVariant::Bare, 24.0);
                    });

                spawn_divider(panel);

                // Tab strip.
                panel
                    .spawn(Node {
                        width: Val::Percent(100.0),
                        flex_direction: FlexDirection::Row,
                        column_gap: Val::Px(4.0),
                        flex_wrap: FlexWrap::Wrap,
                        row_gap: Val::Px(4.0),
                        ..default()
                    })
                    .with_children(|strip| {
                        for tab in Tab::ALL {
                            spawn_button(
                                strip,
                                &theme,
                                TabButton(tab),
                                tab.label(),
                                ButtonVariant::Ghost,
                                24.0,
                            );
                        }
                    });

                spawn_divider(panel);

                // Scrollable body (children rebuilt per tab).
                panel.spawn((
                    Node {
                        width: Val::Percent(100.0),
                        flex_grow: 1.0,
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(6.0),
                        overflow: Overflow::scroll_y(),
                        ..default()
                    },
                    ScrollPosition::default(),
                    RelativeCursorPosition::default(),
                    Interaction::None,
                    ScrollableColumn,
                    SettingsTabBody,
                    Name::new("SettingsTabBody"),
                ));
            });
        });
}

// ── Chrome systems ──────────────────────────────────────────────────────────

fn sync_visibility(menu: Res<SettingsMenu>, mut roots: Query<&mut Visibility, With<SettingsRoot>>) {
    if !menu.is_changed() {
        return;
    }
    let target = if menu.open {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut vis in &mut roots {
        if *vis != target {
            *vis = target;
        }
    }
}

fn handle_close_click(
    interactions: Query<&Interaction, (Changed<Interaction>, With<CloseButton>)>,
    mut menu: ResMut<SettingsMenu>,
) {
    for interaction in &interactions {
        if matches!(interaction, Interaction::Pressed) {
            menu.open = false;
        }
    }
}

fn handle_tab_clicks(
    interactions: Query<(&Interaction, &TabButton), Changed<Interaction>>,
    mut menu: ResMut<SettingsMenu>,
) {
    for (interaction, tab) in &interactions {
        if matches!(interaction, Interaction::Pressed) && menu.tab != tab.0 {
            menu.tab = tab.0;
        }
    }
}

fn update_tab_latches(menu: Res<SettingsMenu>, mut tabs: Query<(&TabButton, &mut UiButton)>) {
    for (tab, mut button) in &mut tabs {
        let latched = tab.0 == menu.tab;
        if button.latched != latched {
            button.latched = latched;
        }
    }
}

// ── Tab body rebuild ──────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn rebuild_tab_body(
    mut commands: Commands,
    menu: Res<SettingsMenu>,
    theme: Res<UiTheme>,
    window: Res<WindowSettings>,
    overrides: Res<WindowSettingsOverrides>,
    graphics: Res<GraphicsSettings>,
    units: Res<UnitsSettings>,
    input: Res<InputSettings>,
    monitors: Query<(&Monitor, Has<PrimaryMonitor>)>,
    body: Query<(Entity, Option<&Children>), With<SettingsTabBody>>,
    mut shown: Local<Option<(bool, Tab, u32)>>,
) {
    let key = (menu.open, menu.tab, menu.rebuild);
    if *shown == Some(key) {
        return;
    }
    *shown = Some(key);

    let Ok((body_entity, children)) = body.single() else {
        return;
    };
    if let Some(children) = children {
        for child in children.iter() {
            commands.entity(child).despawn();
        }
    }
    if !menu.open {
        return;
    }

    let theme = theme.clone();
    commands
        .entity(body_entity)
        .with_children(|b| match menu.tab {
            Tab::Window => build_window_tab(b, &theme, &window, &overrides, &monitors),
            Tab::Graphics => build_graphics_tab(b, &theme, &graphics),
            Tab::Units => build_units_tab(b, &theme, &units),
            Tab::Keyboard => build_binding_tab(b, &theme, &input, BindingKind::Keyboard),
            Tab::Mouse => build_binding_tab(b, &theme, &input, BindingKind::Mouse),
            Tab::Controller => build_binding_tab(b, &theme, &input, BindingKind::Controller),
            Tab::Hotas => build_hotas_tab(b, &theme, &input),
        });
}

// ── Window tab ────────────────────────────────────────────────────────────────

fn build_window_tab(
    b: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    settings: &WindowSettings,
    overrides: &WindowSettingsOverrides,
    monitors: &Query<(&Monitor, Has<PrimaryMonitor>)>,
) {
    // Mode.
    if let Some(mode) = overrides.mode {
        pinned_row(b, theme, "Mode", mode_label(mode), "THALOS_WINDOW_MODE");
    } else {
        let index = match settings.mode {
            WindowModeSetting::Windowed => 0,
            WindowModeSetting::Borderless => 1,
            WindowModeSetting::Exclusive => 2,
        };
        spawn_cycle_row(
            b,
            theme,
            "Mode",
            vec!["Windowed".into(), "Borderless".into(), "Fullscreen".into()],
            index,
            WindowModeControl,
        );
    }

    // Resolution (windowed).
    if let Some((w, h)) = overrides.resolution {
        pinned_row(
            b,
            theme,
            "Resolution",
            format!("{w} × {h}"),
            "THALOS_WINDOW_SIZE",
        );
    } else {
        let mut values: Vec<(u32, u32)> = RESOLUTION_PRESETS.to_vec();
        if !values.contains(&settings.resolution) {
            values.insert(0, settings.resolution);
        }
        let index = values
            .iter()
            .position(|&r| r == settings.resolution)
            .unwrap_or(0);
        let options = values.iter().map(|(w, h)| format!("{w} × {h}")).collect();
        spawn_cycle_row(
            b,
            theme,
            "Resolution",
            options,
            index,
            ResolutionControl { values },
        );
        note(
            b,
            theme,
            "Applies in windowed mode; drag-resizing updates it too.",
        );
    }

    // Monitor.
    let mut choices: Vec<MonitorChoice> = monitors
        .iter()
        .filter_map(|(monitor, primary)| {
            let name = monitor.name.clone()?;
            let label = format!(
                "{name} — {}×{}{}",
                monitor.physical_width,
                monitor.physical_height,
                if primary { " (primary)" } else { "" },
            );
            Some(MonitorChoice { name, label })
        })
        .collect();
    choices.sort_by(|a, b| a.name.cmp(&b.name));

    let mut options = vec!["Primary".to_string()];
    let mut names: Vec<Option<String>> = vec![None];
    for choice in &choices {
        options.push(choice.label.clone());
        names.push(Some(choice.name.clone()));
    }
    let mut index = match settings.monitor.as_deref() {
        None => 0,
        Some(wanted) => names
            .iter()
            .position(|n| n.as_deref() == Some(wanted))
            .unwrap_or(usize::MAX),
    };
    // Persisted-but-unplugged monitor: keep it selectable so it round-trips.
    if index == usize::MAX
        && let Some(wanted) = settings.monitor.as_deref()
    {
        options.push(format!("{wanted} (not connected)"));
        names.push(Some(wanted.to_string()));
        index = names.len() - 1;
    }
    spawn_cycle_row(
        b,
        theme,
        "Monitor",
        options,
        index,
        MonitorControl { names },
    );
    note(b, theme, "Used by the fullscreen modes.");

    // VSync.
    if let Some(vsync) = overrides.vsync {
        let label = if vsync { "On" } else { "Off" };
        pinned_row(b, theme, "VSync", label.to_string(), "THALOS_VSYNC");
    } else {
        spawn_checkbox_row(b, theme, "VSync", settings.vsync, VsyncControl);
    }

    // UI scale.
    spawn_slider_row(
        b,
        theme,
        "UI scale",
        UiSlider {
            min: UI_SCALE_MIN,
            max: UI_SCALE_MAX,
            value: settings.ui_scale,
            step: 0.05,
            format: SliderFormat::Scale2,
        },
        UiScaleControl,
    );

    spacer(b);
    spawn_button(
        b,
        theme,
        ResetWindowControl,
        "Reset to defaults",
        ButtonVariant::Ghost,
        26.0,
    );
    note(
        b,
        theme,
        "Saved to user/settings.ron. THALOS_WINDOW_MODE / _SIZE / _VSYNC override for one session.",
    );
}

// ── Graphics tab ────────────────────────────────────────────────────────────────

fn build_graphics_tab(
    b: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    settings: &GraphicsSettings,
) {
    spawn_checkbox_row(
        b,
        theme,
        "Volumetric clouds",
        settings.clouds,
        CloudsControl,
    );
    note(
        b,
        theme,
        "Off parks the cloud raymarch (no GPU cost); the sky renders clear.",
    );

    spacer(b);

    spawn_checkbox_row(b, theme, "Grass blades", settings.grass, GrassControl);
    note(
        b,
        theme,
        "Off parks the grass clipmap (no blades built); the ground still reads green from the terrain.",
    );

    spacer(b);

    spawn_checkbox_row(
        b,
        theme,
        "GPU grass generation",
        settings.gpu_grass,
        GpuGrassControl,
    );
    note(
        b,
        theme,
        "Generates near/mid blades on the GPU each frame (no stored grass geometry). \
         Off falls back to the CPU-built blade tiles.",
    );

    spacer(b);

    let index = MsaaSetting::ALL
        .iter()
        .position(|m| *m == settings.msaa)
        .unwrap_or(0);
    let options = MsaaSetting::ALL
        .iter()
        .map(|m| m.label().to_string())
        .collect();
    spawn_cycle_row(b, theme, "Anti-aliasing", options, index, MsaaControl);
    note(
        b,
        theme,
        "MSAA smooths geometry edges and (via alpha-to-coverage) tree-leaf edges; \
         any level replaces the SMAA post pass.",
    );

    spacer(b);
    spawn_button(
        b,
        theme,
        ResetGraphicsControl,
        "Reset to defaults",
        ButtonVariant::Ghost,
        26.0,
    );
    note(b, theme, "Saved to user/graphics.ron.");
}

// ── Units tab ─────────────────────────────────────────────────────────────────────

fn build_units_tab(b: &mut ChildSpawnerCommands<'_>, theme: &UiTheme, settings: &UnitsSettings) {
    let index = UnitSystem::ALL
        .iter()
        .position(|s| *s == settings.system)
        .unwrap_or(0);
    let options = UnitSystem::ALL
        .iter()
        .map(|s| s.label().to_string())
        .collect();
    spawn_cycle_row(b, theme, "Measurement", options, index, UnitsControl);
    note(
        b,
        theme,
        "Imperial shows altitude in feet, speed in knots, vertical speed in ft/min, \
         and mass in pounds. Internal physics stays SI; this only affects the readouts.",
    );
    note(b, theme, "Saved to user/units.ron.");
}

// ── Binding-list tabs ───────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum BindingKind {
    Keyboard,
    Mouse,
    Controller,
}

fn build_binding_tab(
    b: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    settings: &InputSettings,
    kind: BindingKind,
) {
    let game = &settings.game;
    let sections: &[(&str, &BindingSection)] = match kind {
        BindingKind::Keyboard | BindingKind::Controller => &[
            ("Flight", &game.flight),
            ("Warp", &game.warp),
            ("View", &game.view),
            ("EVA", &game.eva),
            ("EVA Move", &game.eva_move),
            ("Maneuver", &game.maneuver),
            ("Maneuver Precision", &game.maneuver_precision),
            ("System", &game.system),
        ],
        BindingKind::Mouse => &[
            ("Camera", &game.camera),
            ("Flight", &game.flight),
            ("Warp", &game.warp),
            ("Maneuver", &game.maneuver),
        ],
    };
    let filter: fn(&BindingSpec) -> bool = match kind {
        BindingKind::Keyboard => is_keyboard_spec,
        BindingKind::Mouse => is_mouse_spec,
        BindingKind::Controller => is_gamepad_spec,
    };

    let any = sections.iter().any(|(_, s)| section_has_spec(s, filter));
    if !any {
        let msg = match kind {
            BindingKind::Mouse => "No mouse bindings configured.",
            BindingKind::Controller => "No controller bindings configured.",
            BindingKind::Keyboard => "No keyboard bindings configured.",
        };
        note(b, theme, msg);
        if matches!(kind, BindingKind::Controller) {
            note(
                b,
                theme,
                "Add GamepadButton(…) entries to assets/input.ron.",
            );
        }
        return;
    }

    for (title, section) in sections {
        build_binding_group(b, theme, title, section, filter);
    }
}

fn build_binding_group(
    b: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    title: &str,
    section: &BindingSection,
    filter: fn(&BindingSpec) -> bool,
) {
    if !section_has_spec(section, filter) {
        return;
    }
    section_header(b, theme, &title.to_uppercase());

    for (action, specs) in &section.bindings {
        let bound: Vec<String> = specs
            .iter()
            .filter(|s| filter(s))
            .map(format_binding)
            .collect();
        if bound.is_empty() {
            continue;
        }
        binding_row(b, theme, &format_action(action), &bound.join(" / "));
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
        let value = match (pos.is_empty(), neg.is_empty()) {
            (false, false) => format!("{} / {}", pos.join(", "), neg.join(", ")),
            (false, true) => pos.join(", "),
            (true, false) => neg.join(", "),
            (true, true) => String::new(),
        };
        binding_row(
            b,
            theme,
            &format!("{} +/−", format_action(axis_name)),
            &value,
        );
    }
}

fn binding_row(b: &mut ChildSpawnerCommands<'_>, theme: &UiTheme, action: &str, value: &str) {
    b.spawn(Node {
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Row,
        justify_content: JustifyContent::SpaceBetween,
        column_gap: Val::Px(16.0),
        ..default()
    })
    .with_children(|row| {
        row.spawn((
            Text::new(action.to_string()),
            TextFont {
                font: theme.font_ui.clone(),
                font_size: FontSize::Px(10.0),
                ..default()
            },
            TextColor(tokens::TEXT_DIM),
        ));
        row.spawn((
            Text::new(value.to_string()),
            TextFont {
                font: theme.font_ui.clone(),
                font_size: FontSize::Px(10.0),
                ..default()
            },
            TextColor(tokens::TEXT_PRIMARY),
        ));
    });
}

// ── HOTAS tab ───────────────────────────────────────────────────────────────────

fn build_hotas_tab(b: &mut ChildSpawnerCommands<'_>, theme: &UiTheme, settings: &InputSettings) {
    let hotas = &settings.game.hotas;
    spawn_checkbox_row(
        b,
        theme,
        "HOTAS enabled",
        hotas.enabled,
        HotasEnabledControl,
    );

    spacer(b);

    // Device selector: Any / Name contains (+ name field).
    let is_name = matches!(hotas.device, HotasDeviceSelector::NameContains(_));
    spawn_cycle_row(
        b,
        theme,
        "Device",
        vec!["Any".into(), "Name contains".into()],
        if is_name { 1 } else { 0 },
        HotasDeviceModeControl,
    );
    if let HotasDeviceSelector::NameContains(name) = &hotas.device {
        b.spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new("Name:"),
                TextFont {
                    font: theme.font_ui.clone(),
                    font_size: FontSize::Px(11.0),
                    ..default()
                },
                TextColor(tokens::TEXT_DIM),
                Node {
                    width: Val::Px(120.0),
                    ..default()
                },
            ));
            spawn_text_field(
                row,
                theme,
                UiTextField::new(name, "device name"),
                Val::Px(220.0),
                HotasDeviceNameControl,
            );
        });
    }

    spacer(b);

    if !hotas.enabled {
        note(b, theme, "Enable HOTAS to configure axes.");
        return;
    }

    section_header(b, theme, "AXES");
    for axis in HOTAS_AXES {
        match hotas.axes.get(axis) {
            Some(binding) => build_hotas_axis_row(b, theme, axis, binding),
            None => build_hotas_axis_empty(b, theme, axis),
        }
    }
    note(
        b,
        theme,
        "Raw codes are platform-specific — find yours with the gamepad_axes example. \
         Edit assets/input.ron to persist.",
    );
}

fn build_hotas_axis_row(
    b: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    axis: &'static str,
    binding: &HotasAxisBinding,
) {
    b.spawn(Node {
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Row,
        align_items: AlignItems::Center,
        column_gap: Val::Px(8.0),
        ..default()
    })
    .with_children(|row| {
        row.spawn((
            Text::new(axis.to_string()),
            TextFont {
                font: theme.font_ui.clone(),
                font_size: FontSize::Px(11.0),
                ..default()
            },
            TextColor(tokens::TEXT_PRIMARY),
            Node {
                width: Val::Px(70.0),
                flex_shrink: 0.0,
                ..default()
            },
        ));
        row.spawn((
            Text::new("code"),
            TextFont {
                font: theme.font_ui.clone(),
                font_size: FontSize::Px(10.0),
                ..default()
            },
            TextColor(tokens::TEXT_DIM),
        ));
        spawn_text_field(
            row,
            theme,
            UiTextField::new(binding.code.to_string(), "code"),
            Val::Px(56.0),
            HotasCodeControl { axis },
        );
        spawn_button(
            row,
            theme,
            HotasRemoveControl { axis },
            "✕",
            ButtonVariant::Ghost,
            20.0,
        );
    });

    // Invert + deadzone, each on its own full-width row (indented under the axis).
    indented(b, |row| {
        spawn_checkbox_row(
            row,
            theme,
            "invert",
            binding.invert,
            HotasInvertControl { axis },
        );
    });
    indented(b, |row| {
        spawn_slider_row(
            row,
            theme,
            "deadzone",
            UiSlider {
                min: 0.0,
                max: 0.9,
                value: binding.deadzone,
                step: 0.01,
                format: SliderFormat::Plain2,
            },
            HotasDeadzoneControl { axis },
        );
    });
}

/// Spawn a full-width row indented under a HOTAS axis header, calling `build`
/// to populate it.
fn indented(b: &mut ChildSpawnerCommands<'_>, build: impl FnOnce(&mut ChildSpawnerCommands<'_>)) {
    b.spawn(Node {
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Row,
        align_items: AlignItems::Center,
        padding: UiRect::left(Val::Px(70.0)),
        ..default()
    })
    .with_children(|row| build(row));
}

fn build_hotas_axis_empty(b: &mut ChildSpawnerCommands<'_>, theme: &UiTheme, axis: &'static str) {
    b.spawn(Node {
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Row,
        align_items: AlignItems::Center,
        column_gap: Val::Px(8.0),
        ..default()
    })
    .with_children(|row| {
        row.spawn((
            Text::new(axis.to_string()),
            TextFont {
                font: theme.font_ui.clone(),
                font_size: FontSize::Px(11.0),
                ..default()
            },
            TextColor(tokens::TEXT_DIM),
            Node {
                width: Val::Px(70.0),
                flex_shrink: 0.0,
                ..default()
            },
        ));
        spawn_button(
            row,
            theme,
            HotasAddControl { axis },
            "Add",
            ButtonVariant::Ghost,
            20.0,
        );
    });
}

// ── Apply systems ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn apply_window_controls(
    mut settings: ResMut<WindowSettings>,
    mut menu: ResMut<SettingsMenu>,
    mode_q: Query<&UiCycle, (Changed<UiCycle>, With<WindowModeControl>)>,
    res_q: Query<(&UiCycle, &ResolutionControl), Changed<UiCycle>>,
    monitor_q: Query<(&UiCycle, &MonitorControl), Changed<UiCycle>>,
    vsync_q: Query<&UiCheckbox, (Changed<UiCheckbox>, With<VsyncControl>)>,
    scale_q: Query<&UiSlider, (Changed<UiSlider>, With<UiScaleControl>)>,
    reset_q: Query<&Interaction, (Changed<Interaction>, With<ResetWindowControl>)>,
) {
    for cycle in &mode_q {
        let mode = match cycle.index {
            0 => WindowModeSetting::Windowed,
            1 => WindowModeSetting::Borderless,
            _ => WindowModeSetting::Exclusive,
        };
        if settings.mode != mode {
            settings.mode = mode;
        }
    }
    for (cycle, control) in &res_q {
        if let Some(&value) = control.values.get(cycle.index)
            && settings.resolution != value
        {
            settings.resolution = value;
        }
    }
    for (cycle, control) in &monitor_q {
        if let Some(name) = control.names.get(cycle.index)
            && settings.monitor != *name
        {
            settings.monitor = name.clone();
        }
    }
    for checkbox in &vsync_q {
        if settings.vsync != checkbox.checked {
            settings.vsync = checkbox.checked;
        }
    }
    for slider in &scale_q {
        if (settings.ui_scale - slider.value).abs() > 1.0e-4 {
            settings.ui_scale = slider.value;
        }
    }
    for interaction in &reset_q {
        if matches!(interaction, Interaction::Pressed) {
            *settings = WindowSettings::default();
            menu.dirty();
        }
    }
}

fn apply_graphics_controls(
    mut settings: ResMut<GraphicsSettings>,
    mut menu: ResMut<SettingsMenu>,
    clouds_q: Query<&UiCheckbox, (Changed<UiCheckbox>, With<CloudsControl>)>,
    grass_q: Query<&UiCheckbox, (Changed<UiCheckbox>, With<GrassControl>)>,
    gpu_grass_q: Query<&UiCheckbox, (Changed<UiCheckbox>, With<GpuGrassControl>)>,
    msaa_q: Query<&UiCycle, (Changed<UiCycle>, With<MsaaControl>)>,
    reset_q: Query<&Interaction, (Changed<Interaction>, With<ResetGraphicsControl>)>,
) {
    for checkbox in &clouds_q {
        if settings.clouds != checkbox.checked {
            settings.clouds = checkbox.checked;
        }
    }
    for checkbox in &grass_q {
        if settings.grass != checkbox.checked {
            settings.grass = checkbox.checked;
        }
    }
    for checkbox in &gpu_grass_q {
        if settings.gpu_grass != checkbox.checked {
            settings.gpu_grass = checkbox.checked;
        }
    }
    for cycle in &msaa_q {
        if let Some(&msaa) = MsaaSetting::ALL.get(cycle.index)
            && settings.msaa != msaa
        {
            settings.msaa = msaa;
        }
    }
    for interaction in &reset_q {
        if matches!(interaction, Interaction::Pressed) {
            *settings = GraphicsSettings::default();
            menu.dirty();
        }
    }
}

fn apply_units_controls(
    mut settings: ResMut<UnitsSettings>,
    units_q: Query<&UiCycle, (Changed<UiCycle>, With<UnitsControl>)>,
) {
    for cycle in &units_q {
        if let Some(&system) = UnitSystem::ALL.get(cycle.index)
            && settings.system != system
        {
            settings.system = system;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_hotas_controls(
    mut settings: ResMut<InputSettings>,
    mut menu: ResMut<SettingsMenu>,
    enabled_q: Query<&UiCheckbox, (Changed<UiCheckbox>, With<HotasEnabledControl>)>,
    mode_q: Query<&UiCycle, (Changed<UiCycle>, With<HotasDeviceModeControl>)>,
    name_q: Query<&UiTextField, (Changed<UiTextField>, With<HotasDeviceNameControl>)>,
    code_q: Query<(&UiTextField, &HotasCodeControl), Changed<UiTextField>>,
    invert_q: Query<(&UiCheckbox, &HotasInvertControl), Changed<UiCheckbox>>,
    deadzone_q: Query<(&UiSlider, &HotasDeadzoneControl), Changed<UiSlider>>,
    add_q: Query<(&Interaction, &HotasAddControl), Changed<Interaction>>,
    remove_q: Query<(&Interaction, &HotasRemoveControl), Changed<Interaction>>,
) {
    let hotas = &mut settings.game.hotas;

    for checkbox in &enabled_q {
        if hotas.enabled != checkbox.checked {
            hotas.enabled = checkbox.checked;
            menu.dirty(); // enabling shows/hides the axis rows
        }
    }
    for cycle in &mode_q {
        let want_name = cycle.index == 1;
        let is_name = matches!(hotas.device, HotasDeviceSelector::NameContains(_));
        if want_name != is_name {
            hotas.device = if want_name {
                HotasDeviceSelector::NameContains(String::new())
            } else {
                HotasDeviceSelector::Any
            };
            menu.dirty(); // shows/hides the name field
        }
    }
    for field in &name_q {
        if let HotasDeviceSelector::NameContains(name) = &mut hotas.device
            && *name != field.value
        {
            *name = field.value.clone();
        }
    }
    for (field, control) in &code_q {
        if let Some(binding) = hotas.axes.get_mut(control.axis)
            && let Ok(code) = field.value.trim().parse::<u32>()
            && binding.code != code
        {
            binding.code = code;
        }
    }
    for (checkbox, control) in &invert_q {
        if let Some(binding) = hotas.axes.get_mut(control.axis)
            && binding.invert != checkbox.checked
        {
            binding.invert = checkbox.checked;
        }
    }
    for (slider, control) in &deadzone_q {
        if let Some(binding) = hotas.axes.get_mut(control.axis)
            && (binding.deadzone - slider.value).abs() > 1.0e-4
        {
            binding.deadzone = slider.value;
        }
    }
    for (interaction, control) in &add_q {
        if matches!(interaction, Interaction::Pressed) {
            hotas.axes.insert(
                control.axis.to_string(),
                HotasAxisBinding {
                    code: 0,
                    device: None,
                    invert: false,
                    deadzone: 0.05,
                    min: -1.0,
                    max: 1.0,
                },
            );
            menu.dirty();
        }
    }
    for (interaction, control) in &remove_q {
        if matches!(interaction, Interaction::Pressed) {
            hotas.axes.remove(control.axis);
            menu.dirty();
        }
    }
}

// ── Small layout helpers ──────────────────────────────────────────────────────

fn spacer(b: &mut ChildSpawnerCommands<'_>) {
    b.spawn(Node {
        height: Val::Px(6.0),
        ..default()
    });
}

fn section_header(b: &mut ChildSpawnerCommands<'_>, theme: &UiTheme, text: &str) {
    b.spawn((
        Node {
            margin: UiRect::top(Val::Px(4.0)),
            ..default()
        },
        Text::new(text.to_string()),
        TextFont {
            font: theme.font_ui.clone(),
            font_size: FontSize::Px(10.0),
            ..default()
        },
        TextColor(tokens::TEXT_FAINT),
    ));
}

fn note(b: &mut ChildSpawnerCommands<'_>, theme: &UiTheme, text: &str) {
    b.spawn((
        Text::new(text.to_string()),
        TextFont {
            font: theme.font_ui.clone(),
            font_size: FontSize::Px(9.0),
            ..default()
        },
        TextColor(tokens::TEXT_DIM),
    ));
}

fn pinned_row(
    b: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    label: &str,
    value: String,
    env_var: &str,
) {
    b.spawn(Node {
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Row,
        column_gap: Val::Px(8.0),
        align_items: AlignItems::Center,
        ..default()
    })
    .with_children(|row| {
        row.spawn((
            Text::new(label.to_string()),
            TextFont {
                font: theme.font_ui.clone(),
                font_size: FontSize::Px(11.0),
                ..default()
            },
            TextColor(tokens::TEXT_DIM),
            Node {
                width: Val::Px(120.0),
                ..default()
            },
        ));
        row.spawn((
            Text::new(format!("{value}  (pinned by {env_var})")),
            TextFont {
                font: theme.font_ui.clone(),
                font_size: FontSize::Px(11.0),
                ..default()
            },
            TextColor(tokens::TEXT_DIM),
        ));
    });
}

fn mode_label(mode: WindowModeSetting) -> String {
    match mode {
        WindowModeSetting::Windowed => "Windowed",
        WindowModeSetting::Borderless => "Borderless",
        WindowModeSetting::Exclusive => "Fullscreen",
    }
    .to_string()
}

// ── Binding spec filters / formatting ─────────────────────────────────────────

fn section_has_spec(section: &BindingSection, filter: fn(&BindingSpec) -> bool) -> bool {
    section.bindings.values().any(|v| v.iter().any(filter))
        || section
            .axes
            .values()
            .any(|a: &AxisSpec| a.positive.iter().any(filter) || a.negative.iter().any(filter))
}

fn is_keyboard_spec(spec: &BindingSpec) -> bool {
    matches!(spec, BindingSpec::Key(_))
}

fn is_mouse_spec(spec: &BindingSpec) -> bool {
    matches!(
        spec,
        BindingSpec::MouseButton(_) | BindingSpec::MouseMotion | BindingSpec::MouseWheel
    )
}

fn is_gamepad_spec(spec: &BindingSpec) -> bool {
    matches!(
        spec,
        BindingSpec::GamepadButton(_) | BindingSpec::GamepadAxis(_)
    )
}

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
