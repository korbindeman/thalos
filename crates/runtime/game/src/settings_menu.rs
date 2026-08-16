//! Game-specific contributors to the shared application settings modal.
//!
//! Window behavior and anti-aliasing live in `thalos_preferences`. This module
//! adds only settings that belong to the full game: simulation rendering,
//! measurement units, input binding reference pages, and HOTAS controls.

use bevy::prelude::*;
use thalos_input::settings::{
    AxisSpec, BindingSection, BindingSpec, HotasAxisBinding, HotasDeviceSelector, InputSettings,
};
pub use thalos_preferences::SettingsMenu;
use thalos_preferences::{
    SettingsMenuSet, SettingsPage, SettingsPageBuild, register_settings_page,
};
use thalos_ui::{
    ButtonVariant, SliderFormat, UiCheckbox, UiCycle, UiSlider, UiTextField, UiTheme, spawn_button,
    spawn_checkbox_row, spawn_cycle_row, spawn_slider_row, spawn_text_field, tokens,
};

use crate::graphics_settings::GraphicsSettings;
use crate::units_settings::{AviationUnits, UnitSystem, UnitsSettings};

const HOTAS_AXES: [&str; 4] = ["pitch", "yaw", "roll", "throttle"];

#[derive(Component)]
struct CloudsControl;
#[derive(Component)]
struct GrassControl;
#[derive(Component)]
struct GpuGrassControl;
#[derive(Component)]
struct TerrainLodControl;
#[derive(Component)]
struct ShadowCascadesControl;
#[derive(Component)]
struct ResetGraphicsControl;

#[derive(Component)]
struct UnitsControl;
#[derive(Component)]
struct AviationUnitsControl;

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

/// Adds the full game's sections to the common settings host.
pub struct SettingsMenuPlugin;

impl Plugin for SettingsMenuPlugin {
    fn build(&self, app: &mut App) {
        for page in [
            SettingsPage {
                id: "graphics",
                label: "Graphics",
                order: 10,
            },
            SettingsPage {
                id: "units",
                label: "Units",
                order: 20,
            },
            SettingsPage {
                id: "keyboard",
                label: "Keyboard",
                order: 30,
            },
            SettingsPage {
                id: "mouse",
                label: "Mouse",
                order: 40,
            },
            SettingsPage {
                id: "controller",
                label: "Controller",
                order: 50,
            },
            SettingsPage {
                id: "hotas",
                label: "HOTAS",
                order: 60,
            },
        ] {
            register_settings_page(app, page);
        }

        app.add_systems(
            Update,
            build_game_sections.in_set(SettingsMenuSet::BuildSections),
        )
        .add_systems(
            Update,
            (
                apply_graphics_controls,
                apply_units_controls,
                apply_hotas_controls,
            )
                .in_set(SettingsMenuSet::Apply),
        );
    }
}

fn build_game_sections(
    mut commands: Commands,
    mut builds: MessageReader<SettingsPageBuild>,
    theme: Res<UiTheme>,
    graphics: Res<GraphicsSettings>,
    units: Res<UnitsSettings>,
    input: Res<InputSettings>,
) {
    for build in builds.read() {
        commands
            .entity(build.body)
            .with_children(|body| match build.id {
                "graphics" => build_graphics_tab(body, &theme, &graphics),
                "units" => build_units_tab(body, &theme, &units),
                "keyboard" => build_binding_tab(body, &theme, &input, BindingKind::Keyboard),
                "mouse" => build_binding_tab(body, &theme, &input, BindingKind::Mouse),
                "controller" => build_binding_tab(body, &theme, &input, BindingKind::Controller),
                "hotas" => build_hotas_tab(body, &theme, &input),
                _ => {}
            });
    }
}

// ── Graphics tab ────────────────────────────────────────────────────────────────

fn build_graphics_tab(
    b: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    settings: &GraphicsSettings,
) {
    section_header(b, theme, "GAME RENDERING");
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
    spawn_slider_row(
        b,
        theme,
        "Terrain detail",
        UiSlider {
            min: GraphicsSettings::TERRAIN_LOD_MIN,
            max: GraphicsSettings::TERRAIN_LOD_MAX,
            value: settings.terrain_lod,
            step: 0.05,
            format: SliderFormat::Scale2,
        },
        TerrainLodControl,
    );
    note(
        b,
        theme,
        "Coarsens streamed terrain. 1.00× is Showcase; 0.50× is the Laptop default.",
    );

    spacer(b);
    let shadow_index = settings.shadow_cascades as usize;
    let shadow_options = (0..=GraphicsSettings::SHADOW_CASCADES_MAX)
        .map(|count| {
            if count == 0 {
                "Off".to_string()
            } else {
                format!("{count}")
            }
        })
        .collect();
    spawn_cycle_row(
        b,
        theme,
        "Shadow cascades",
        shadow_options,
        shadow_index,
        ShadowCascadesControl,
    );
    note(
        b,
        theme,
        "Each cascade is a 4096² depth pass. Laptop uses 2. THALOS_SHADOW_CASCADES still pins a session.",
    );

    spacer(b);
    spawn_button(
        b,
        theme,
        ResetGraphicsControl,
        "Reset to Showcase",
        ButtonVariant::Ghost,
        26.0,
    );
    note(b, theme, "Saved to settings.ron.");
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

    spacer(b);
    let aviation_index = AviationUnits::ALL
        .iter()
        .position(|a| *a == settings.aviation)
        .unwrap_or(0);
    let aviation_options = AviationUnits::ALL
        .iter()
        .map(|a| a.label().to_string())
        .collect();
    spawn_cycle_row(
        b,
        theme,
        "Flight instruments",
        aviation_options,
        aviation_index,
        AviationUnitsControl,
    );
    note(
        b,
        theme,
        "Aviation keeps feet, knots, ft/min, and nautical miles on the flight \
         instruments — the PFD tapes, the atmospheric TAS/q/Mach readout, and the \
         MFD navigation display — even when Measurement is metric, which is how \
         real cockpits are marked. Orbital altitude, \u{394}v, staging masses, and \
         map scales still follow Measurement.",
    );
    note(b, theme, "Saved to settings.ron.");
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

fn apply_graphics_controls(
    mut settings: ResMut<GraphicsSettings>,
    mut menu: ResMut<SettingsMenu>,
    clouds_q: Query<&UiCheckbox, (Changed<UiCheckbox>, With<CloudsControl>)>,
    grass_q: Query<&UiCheckbox, (Changed<UiCheckbox>, With<GrassControl>)>,
    gpu_grass_q: Query<&UiCheckbox, (Changed<UiCheckbox>, With<GpuGrassControl>)>,
    terrain_q: Query<&UiSlider, (Changed<UiSlider>, With<TerrainLodControl>)>,
    shadows_q: Query<&UiCycle, (Changed<UiCycle>, With<ShadowCascadesControl>)>,
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
    for slider in &terrain_q {
        if (settings.terrain_lod - slider.value).abs() > 1.0e-4 {
            settings.terrain_lod = slider.value;
        }
    }
    for cycle in &shadows_q {
        let value = cycle.index as u8;
        if value <= GraphicsSettings::SHADOW_CASCADES_MAX && settings.shadow_cascades != value {
            settings.shadow_cascades = value;
        }
    }
    for interaction in &reset_q {
        if matches!(interaction, Interaction::Pressed) {
            *settings = GraphicsSettings::showcase();
            menu.dirty();
        }
    }
}

fn apply_units_controls(
    mut settings: ResMut<UnitsSettings>,
    units_q: Query<&UiCycle, (Changed<UiCycle>, With<UnitsControl>)>,
    aviation_q: Query<&UiCycle, (Changed<UiCycle>, With<AviationUnitsControl>)>,
) {
    for cycle in &units_q {
        if let Some(&system) = UnitSystem::ALL.get(cycle.index)
            && settings.system != system
        {
            settings.system = system;
        }
    }
    for cycle in &aviation_q {
        if let Some(&aviation) = AviationUnits::ALL.get(cycle.index)
            && settings.aviation != aviation
        {
            settings.aviation = aviation;
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
