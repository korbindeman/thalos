//! Shared settings modal and the common Window / Graphics sections.

use bevy::picking::Pickable;
use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;
use bevy::window::{Monitor, PrimaryMonitor};
use thalos_ui::{
    self as ui, ButtonVariant, ScrollableColumn, SliderFormat, TextFieldFocus, UiButton,
    UiCheckbox, UiCycle, UiSlider, UiTheme, spawn_button, spawn_checkbox_row, spawn_cycle_row,
    spawn_divider, spawn_slider_row, tokens,
};

use crate::graphics::{
    FRAME_CAP_CHOICES, GraphicsPreferenceCapabilities, GraphicsPreferences, MsaaSetting,
    QualityOverrides, QualityPreset, RENDER_SCALE_MAX, RENDER_SCALE_MIN,
};
use crate::window::{
    RESOLUTION_PRESETS, UI_SCALE_MAX, UI_SCALE_MIN, WindowModeSetting, WindowSettings,
    WindowSettingsOverrides,
};

#[derive(Resource, Default)]
pub struct SettingsMenu {
    pub open: bool,
    active_page: Option<&'static str>,
    rebuild: u32,
}

impl SettingsMenu {
    pub fn dirty(&mut self) {
        self.rebuild = self.rebuild.wrapping_add(1);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SettingsPage {
    pub id: &'static str,
    pub label: &'static str,
    pub order: u16,
}

#[derive(Resource, Default)]
struct SettingsPageRegistry(Vec<SettingsPage>);

/// Register one tab in the shared modal. Multiple plugins may append sections
/// to the tab by reading [`SettingsPageBuild`] for the same `id`.
pub fn register_settings_page(app: &mut App, page: SettingsPage) {
    app.init_resource::<SettingsPageRegistry>();
    let mut pages = app.world_mut().resource_mut::<SettingsPageRegistry>();
    if let Some(existing) = pages.0.iter().find(|existing| existing.id == page.id) {
        assert_eq!(
            existing.label, page.label,
            "settings page {} registered with two labels",
            page.id
        );
        return;
    }
    pages.0.push(page);
    pages.0.sort_by_key(|page| page.order);
}

/// Requests every contributor to append its section to the selected tab body.
#[derive(Message, Clone, Copy, Debug)]
pub struct SettingsPageBuild {
    pub id: &'static str,
    pub body: Entity,
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SettingsMenuSet {
    Chrome,
    Rebuild,
    BuildSections,
    Apply,
}

#[derive(Component)]
struct SettingsRoot;

#[derive(Component)]
struct SettingsTabBody;

#[derive(Component, Clone, Copy)]
struct TabButton(&'static str);

#[derive(Component)]
struct CloseButton;

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

#[derive(Component)]
struct QualityPresetControl {
    values: Vec<QualityPreset>,
}

#[derive(Component)]
struct RenderScaleControl;

#[derive(Component)]
struct FrameCapControl {
    values: Vec<u32>,
}

#[derive(Component)]
struct MsaaControl;

#[derive(Component)]
struct FoliageControl;

#[derive(Component)]
struct ResetGraphicsControl {
    foliage: bool,
}

pub struct SettingsMenuPlugin;

impl Plugin for SettingsMenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SettingsMenu>()
            .init_resource::<SettingsPageRegistry>()
            .add_message::<SettingsPageBuild>()
            .configure_sets(
                Update,
                (
                    SettingsMenuSet::Chrome,
                    SettingsMenuSet::Rebuild,
                    SettingsMenuSet::BuildSections,
                    SettingsMenuSet::Apply,
                )
                    .chain(),
            )
            .add_systems(Startup, setup_ui.after(thalos_ui::init_ui_theme))
            .add_systems(
                Update,
                (
                    sync_visibility,
                    handle_close_click,
                    handle_tab_clicks,
                    update_tab_latches,
                    close_with_escape,
                    toggle_shortcut,
                )
                    .in_set(SettingsMenuSet::Chrome),
            )
            .add_systems(Update, rebuild_tab_body.in_set(SettingsMenuSet::Rebuild))
            .add_systems(
                Update,
                (build_window_section, build_graphics_section)
                    .in_set(SettingsMenuSet::BuildSections),
            )
            .add_systems(
                Update,
                (apply_window_controls, apply_graphics_controls).in_set(SettingsMenuSet::Apply),
            );
    }
}

fn setup_ui(
    mut commands: Commands,
    theme: Res<UiTheme>,
    pages: Res<SettingsPageRegistry>,
    mut menu: ResMut<SettingsMenu>,
) {
    if menu.active_page.is_none() {
        menu.active_page = pages.0.first().map(|page| page.id);
    }

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
                        for page in &pages.0 {
                            spawn_button(
                                strip,
                                &theme,
                                TabButton(page.id),
                                page.label,
                                ButtonVariant::Ghost,
                                24.0,
                            );
                        }
                    });

                spawn_divider(panel);
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

fn sync_visibility(menu: Res<SettingsMenu>, mut roots: Query<&mut Visibility, With<SettingsRoot>>) {
    if !menu.is_changed() {
        return;
    }
    let target = if menu.open {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut visibility in &mut roots {
        if *visibility != target {
            *visibility = target;
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
        if matches!(interaction, Interaction::Pressed) && menu.active_page != Some(tab.0) {
            menu.active_page = Some(tab.0);
        }
    }
}

fn update_tab_latches(menu: Res<SettingsMenu>, mut tabs: Query<(&TabButton, &mut UiButton)>) {
    for (tab, mut button) in &mut tabs {
        let latched = menu.active_page == Some(tab.0);
        if button.latched != latched {
            button.latched = latched;
        }
    }
}

fn toggle_shortcut(
    keys: Res<ButtonInput<KeyCode>>,
    focus: Res<TextFieldFocus>,
    mut menu: ResMut<SettingsMenu>,
) {
    if !focus.is_focused() && keys.just_pressed(KeyCode::F10) {
        menu.open = !menu.open;
    }
}

fn close_with_escape(keys: Res<ButtonInput<KeyCode>>, mut menu: ResMut<SettingsMenu>) {
    if menu.open && keys.just_pressed(KeyCode::Escape) {
        menu.open = false;
    }
}

fn rebuild_tab_body(
    mut commands: Commands,
    menu: Res<SettingsMenu>,
    body: Query<(Entity, Option<&Children>), With<SettingsTabBody>>,
    mut builds: MessageWriter<SettingsPageBuild>,
    mut shown: Local<Option<(bool, Option<&'static str>, u32)>>,
) {
    let key = (menu.open, menu.active_page, menu.rebuild);
    if *shown == Some(key) {
        return;
    }
    *shown = Some(key);

    let Ok((body, children)) = body.single() else {
        return;
    };
    if let Some(children) = children {
        for child in children.iter() {
            commands.entity(child).despawn();
        }
    }
    if menu.open
        && let Some(id) = menu.active_page
    {
        builds.write(SettingsPageBuild { id, body });
    }
}

fn build_window_section(
    mut commands: Commands,
    mut builds: MessageReader<SettingsPageBuild>,
    theme: Res<UiTheme>,
    settings: Res<WindowSettings>,
    overrides: Res<WindowSettingsOverrides>,
    monitors: Query<(&Monitor, Has<PrimaryMonitor>)>,
) {
    let choices = monitor_choices(&monitors);
    for build in builds.read() {
        if build.id != "window" {
            continue;
        }
        commands.entity(build.body).with_children(|body| {
            build_window_controls(body, &theme, &settings, &overrides, &choices)
        });
    }
}

fn monitor_choices(monitors: &Query<(&Monitor, Has<PrimaryMonitor>)>) -> Vec<MonitorChoice> {
    let mut choices: Vec<_> = monitors
        .iter()
        .filter_map(|(monitor, primary)| {
            let name = monitor.name.clone()?;
            Some(MonitorChoice {
                label: format!(
                    "{name} — {}×{}{}",
                    monitor.physical_width,
                    monitor.physical_height,
                    if primary { " (primary)" } else { "" }
                ),
                name,
            })
        })
        .collect();
    choices.sort_by(|left, right| left.name.cmp(&right.name));
    choices
}

struct MonitorChoice {
    name: String,
    label: String,
}

fn build_window_controls(
    body: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    settings: &WindowSettings,
    overrides: &WindowSettingsOverrides,
    choices: &[MonitorChoice],
) {
    if let Some(mode) = overrides.mode {
        pinned_row(body, theme, "Mode", mode_label(mode), "THALOS_WINDOW_MODE");
    } else {
        let index = match settings.mode {
            WindowModeSetting::Windowed => 0,
            WindowModeSetting::Borderless => 1,
            WindowModeSetting::Exclusive => 2,
        };
        spawn_cycle_row(
            body,
            theme,
            "Mode",
            vec!["Windowed".into(), "Borderless".into(), "Fullscreen".into()],
            index,
            WindowModeControl,
        );
    }

    if let Some((width, height)) = overrides.resolution {
        pinned_row(
            body,
            theme,
            "Resolution",
            format!("{width} × {height}"),
            "THALOS_WINDOW_SIZE",
        );
    } else {
        let mut values = RESOLUTION_PRESETS.to_vec();
        if !values.contains(&settings.resolution) {
            values.insert(0, settings.resolution);
        }
        let index = values
            .iter()
            .position(|value| *value == settings.resolution)
            .unwrap_or(0);
        let options = values
            .iter()
            .map(|(width, height)| format!("{width} × {height}"))
            .collect();
        spawn_cycle_row(
            body,
            theme,
            "Resolution",
            options,
            index,
            ResolutionControl { values },
        );
        note(
            body,
            theme,
            "Applies in windowed mode; drag-resizing updates it too.",
        );
    }

    let mut options = vec!["Primary".to_string()];
    let mut names = vec![None];
    for choice in choices {
        options.push(choice.label.clone());
        names.push(Some(choice.name.clone()));
    }
    let mut index = match settings.monitor.as_deref() {
        None => 0,
        Some(wanted) => names
            .iter()
            .position(|name| name.as_deref() == Some(wanted))
            .unwrap_or(usize::MAX),
    };
    if index == usize::MAX
        && let Some(wanted) = settings.monitor.as_deref()
    {
        options.push(format!("{wanted} (not connected)"));
        names.push(Some(wanted.to_string()));
        index = names.len() - 1;
    }
    spawn_cycle_row(
        body,
        theme,
        "Monitor",
        options,
        index,
        MonitorControl { names },
    );
    note(body, theme, "Used by the fullscreen modes.");

    if let Some(vsync) = overrides.vsync {
        pinned_row(
            body,
            theme,
            "VSync",
            if vsync { "On" } else { "Off" }.to_string(),
            "THALOS_VSYNC",
        );
    } else {
        spawn_checkbox_row(body, theme, "VSync", settings.vsync, VsyncControl);
    }

    spawn_slider_row(
        body,
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
    spacer(body);
    spawn_button(
        body,
        theme,
        ResetWindowControl,
        "Reset to defaults",
        ButtonVariant::Ghost,
        26.0,
    );
    note(
        body,
        theme,
        "Saved to preferences.ron. THALOS_WINDOW_MODE / _SIZE / _VSYNC pin one session.",
    );
}

fn build_graphics_section(
    mut commands: Commands,
    mut builds: MessageReader<SettingsPageBuild>,
    theme: Res<UiTheme>,
    settings: Res<GraphicsPreferences>,
    overrides: Res<QualityOverrides>,
    capabilities: Res<GraphicsPreferenceCapabilities>,
) {
    for build in builds.read() {
        if build.id != "graphics" {
            continue;
        }
        commands.entity(build.body).with_children(|body| {
            section_header(body, &theme, "QUALITY");
            if let Some(preset) = overrides.preset {
                pinned_row(
                    body,
                    &theme,
                    "Preset",
                    preset.label().to_string(),
                    "THALOS_QUALITY",
                );
            } else {
                let mut values = QualityPreset::SELECTABLE.to_vec();
                if settings.preset == QualityPreset::Custom {
                    values.insert(0, QualityPreset::Custom);
                }
                let index = values
                    .iter()
                    .position(|preset| *preset == settings.preset)
                    .unwrap_or(0);
                let options = values.iter().map(|preset| preset.label().to_string()).collect();
                spawn_cycle_row(
                    body,
                    &theme,
                    "Preset",
                    options,
                    index,
                    QualityPresetControl { values },
                );
            }
            note(
                body,
                &theme,
                "Showcase is the canonical look. Laptop is the developer profile for Mac and other constrained machines. Editing a knob below becomes Custom.",
            );

            spacer(body);
            spawn_slider_row(
                body,
                &theme,
                "Render scale",
                UiSlider {
                    min: RENDER_SCALE_MIN,
                    max: RENDER_SCALE_MAX,
                    value: settings.render_scale,
                    step: 0.05,
                    format: SliderFormat::Scale2,
                },
                RenderScaleControl,
            );
            note(
                body,
                &theme,
                "Does not change UI density. OS HiDPI stays. THALOS_SCALE still pins one session.",
            );

            spacer(body);
            let cap_index = FRAME_CAP_CHOICES
                .iter()
                .position(|value| *value == settings.frame_cap_hz)
                .unwrap_or(0);
            let cap_options = FRAME_CAP_CHOICES
                .iter()
                .map(|value| {
                    if *value == 0 {
                        "Off".to_string()
                    } else {
                        format!("{value} Hz")
                    }
                })
                .collect();
            spawn_cycle_row(
                body,
                &theme,
                "Frame cap",
                cap_options,
                cap_index,
                FrameCapControl {
                    values: FRAME_CAP_CHOICES.to_vec(),
                },
            );
            note(
                body,
                &theme,
                "30 Hz is the battery play. VSync can still floor an uncapped rate.",
            );

            spacer(body);
            section_header(body, &theme, "ANTI-ALIASING");
            let index = MsaaSetting::ALL
                .iter()
                .position(|level| *level == settings.msaa)
                .unwrap_or(0);
            let options = MsaaSetting::ALL
                .iter()
                .map(|level| level.label().to_string())
                .collect();
            spawn_cycle_row(body, &theme, "Anti-aliasing", options, index, MsaaControl);
            note(
                body,
                &theme,
                "MSAA smooths geometry edges. Off restores this application's post-process AA.",
            );
            if capabilities.foliage {
                spacer(body);
                section_header(body, &theme, "WORLD DETAIL");
                spawn_checkbox_row(body, &theme, "Foliage", settings.foliage, FoliageControl);
                note(
                    body,
                    &theme,
                    "Off removes trees and shrubs and stops their cell or tile builds.",
                );
            }
            spacer(body);
            spawn_button(
                body,
                &theme,
                ResetGraphicsControl {
                    foliage: capabilities.foliage,
                },
                "Reset to Showcase",
                ButtonVariant::Ghost,
                26.0,
            );
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_window_controls(
    mut settings: ResMut<WindowSettings>,
    mut menu: ResMut<SettingsMenu>,
    mode: Query<&UiCycle, (Changed<UiCycle>, With<WindowModeControl>)>,
    resolution: Query<(&UiCycle, &ResolutionControl), Changed<UiCycle>>,
    monitor: Query<(&UiCycle, &MonitorControl), Changed<UiCycle>>,
    vsync: Query<&UiCheckbox, (Changed<UiCheckbox>, With<VsyncControl>)>,
    scale: Query<&UiSlider, (Changed<UiSlider>, With<UiScaleControl>)>,
    reset: Query<&Interaction, (Changed<Interaction>, With<ResetWindowControl>)>,
) {
    for cycle in &mode {
        let value = match cycle.index {
            0 => WindowModeSetting::Windowed,
            1 => WindowModeSetting::Borderless,
            _ => WindowModeSetting::Exclusive,
        };
        if settings.mode != value {
            settings.mode = value;
        }
    }
    for (cycle, control) in &resolution {
        if let Some(&value) = control.values.get(cycle.index)
            && settings.resolution != value
        {
            settings.resolution = value;
        }
    }
    for (cycle, control) in &monitor {
        if let Some(value) = control.names.get(cycle.index)
            && settings.monitor != *value
        {
            settings.monitor = value.clone();
        }
    }
    for checkbox in &vsync {
        if settings.vsync != checkbox.checked {
            settings.vsync = checkbox.checked;
        }
    }
    for slider in &scale {
        if (settings.ui_scale - slider.value).abs() > 1.0e-4 {
            settings.ui_scale = slider.value;
        }
    }
    for interaction in &reset {
        if matches!(interaction, Interaction::Pressed) {
            *settings = WindowSettings::default();
            menu.dirty();
        }
    }
}

fn apply_graphics_controls(
    mut settings: ResMut<GraphicsPreferences>,
    mut menu: ResMut<SettingsMenu>,
    presets: Query<(&UiCycle, &QualityPresetControl), Changed<UiCycle>>,
    render_scale: Query<&UiSlider, (Changed<UiSlider>, With<RenderScaleControl>)>,
    frame_cap: Query<(&UiCycle, &FrameCapControl), Changed<UiCycle>>,
    cycles: Query<&UiCycle, (Changed<UiCycle>, With<MsaaControl>)>,
    foliage: Query<&UiCheckbox, (Changed<UiCheckbox>, With<FoliageControl>)>,
    reset: Query<(&Interaction, &ResetGraphicsControl), Changed<Interaction>>,
) {
    for (cycle, control) in &presets {
        if let Some(&value) = control.values.get(cycle.index)
            && value != QualityPreset::Custom
            && settings.preset != value
        {
            settings.apply_preset(value);
            menu.dirty();
        }
    }
    for slider in &render_scale {
        if (settings.render_scale - slider.value).abs() > 1.0e-4 {
            settings.render_scale = slider.value;
            mark_custom_and_refresh(&mut settings, &mut menu);
        }
    }
    for (cycle, control) in &frame_cap {
        if let Some(&value) = control.values.get(cycle.index)
            && settings.frame_cap_hz != value
        {
            settings.frame_cap_hz = value;
            mark_custom_and_refresh(&mut settings, &mut menu);
        }
    }
    for cycle in &cycles {
        if let Some(&value) = MsaaSetting::ALL.get(cycle.index)
            && settings.msaa != value
        {
            settings.msaa = value;
            mark_custom_and_refresh(&mut settings, &mut menu);
        }
    }
    for checkbox in &foliage {
        if settings.foliage != checkbox.checked {
            settings.foliage = checkbox.checked;
            mark_custom_and_refresh(&mut settings, &mut menu);
        }
    }
    for (interaction, control) in &reset {
        if matches!(interaction, Interaction::Pressed) {
            let defaults = GraphicsPreferences::showcase();
            settings.preset = defaults.preset;
            settings.msaa = defaults.msaa;
            settings.render_scale = defaults.render_scale;
            settings.frame_cap_hz = defaults.frame_cap_hz;
            if control.foliage {
                settings.foliage = defaults.foliage;
            } else {
                settings.mark_custom_if_knobs_changed();
            }
            menu.dirty();
        }
    }
}

fn mark_custom_and_refresh(settings: &mut GraphicsPreferences, menu: &mut SettingsMenu) {
    let was = settings.preset;
    settings.mark_custom_if_knobs_changed();
    if settings.preset != was {
        menu.dirty();
    }
}

fn spacer(body: &mut ChildSpawnerCommands<'_>) {
    body.spawn(Node {
        height: Val::Px(6.0),
        ..default()
    });
}

fn section_header(body: &mut ChildSpawnerCommands<'_>, theme: &UiTheme, text: &str) {
    body.spawn((
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

fn note(body: &mut ChildSpawnerCommands<'_>, theme: &UiTheme, text: &str) {
    body.spawn((
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
    body: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    label: &str,
    value: String,
    env_var: &str,
) {
    body.spawn(Node {
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
