//! Game-level pause menu.
//!
//! The menu itself owns only the Escape-modal state (`GamePause`) and its UI.
//! Simulation pause aggregation lives in [`crate::sim_clock`], which folds the
//! menu, destruction scenario picker, freecam, and warp pause into an explicit
//! simulation clock. Bevy's default `Time`/`Time<Virtual>` remains an app clock
//! so presentation effects can keep animating while canonical/local simulation
//! is paused.

use bevy::app::AppExit;
use bevy::picking::prelude::Pickable;
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, WindowCloseRequested};
use thalos_input::game::GameInputIntent;

use crate::hud::theme::{HudTheme, panel_frame};
use crate::maneuver::InteractionMode;
use crate::settings_menu::SettingsMenu;
use crate::target::TargetBody;

#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct GamePause {
    pub active: bool,
}

#[derive(Component)]
struct PauseMenuRoot;

#[derive(Component, Clone, Copy)]
enum PauseMenuAction {
    Resume,
    Shipyard,
    BaseEditor,
    Settings,
    Quit,
}

pub struct PauseMenuPlugin;

impl Plugin for PauseMenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GamePause>()
            .add_systems(Startup, setup.after(crate::hud::theme::init_theme))
            // Gated to `Running`: during loading there is nothing to pause,
            // and on the start screen Escape is owned by `crate::main_menu`.
            .add_systems(
                Update,
                handle_escape_input
                    .run_if(in_state(crate::loading::AppState::Running))
                    .before(crate::SimStage::Physics),
            )
            .add_systems(
                Update,
                (
                    handle_button_clicks,
                    update_visibility,
                    update_button_visuals,
                )
                    .chain(),
            );
    }
}

pub fn not_game_paused(
    pause: Res<GamePause>,
    scenario: Res<crate::scenario_menu::ScenarioMenu>,
) -> bool {
    !pause.active && !scenario.open
}

fn setup(mut commands: Commands, theme: Res<HudTheme>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.36)),
            GlobalZIndex(100),
            Pickable {
                is_hoverable: false,
                should_block_lower: true,
            },
            Visibility::Hidden,
            PauseMenuRoot,
            Name::new("PauseMenu"),
        ))
        .with_children(|root| {
            let (bg, border) = panel_frame(&theme);
            root.spawn((
                Node {
                    width: Val::Px(246.0),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    padding: UiRect::axes(Val::Px(16.0), Val::Px(14.0)),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Stretch,
                    row_gap: Val::Px(10.0),
                    ..default()
                },
                bg,
                border,
                Name::new("PauseMenuPanel"),
            ))
            .with_children(|panel| {
                panel.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(0.0),
                        top: Val::Px(0.0),
                        bottom: Val::Px(0.0),
                        width: Val::Px(2.0),
                        ..default()
                    },
                    BackgroundColor(theme.text_subtitle),
                    Name::new("PauseMenuAccent"),
                ));
                panel
                    .spawn((
                        Node {
                            width: Val::Percent(100.0),
                            flex_direction: FlexDirection::Row,
                            justify_content: JustifyContent::SpaceBetween,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        Name::new("PauseMenuHeader"),
                    ))
                    .with_children(|header| {
                        header.spawn((
                            Text::new("PAUSED"),
                            TextFont {
                                font: theme.font.clone(),
                                font_size: FontSize::Px(15.0),
                                ..default()
                            },
                            TextColor(theme.text_accent),
                            Name::new("PauseMenuTitle"),
                        ));
                        header.spawn((
                            Text::new("THALOS v0"),
                            TextFont {
                                font: theme.font.clone(),
                                font_size: FontSize::Px(10.0),
                                ..default()
                            },
                            TextColor(theme.text_dim),
                            Name::new("PauseMenuVersion"),
                        ));
                    });
                panel.spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(1.0),
                        ..default()
                    },
                    BackgroundColor(theme.panel_border),
                    Name::new("PauseMenuDivider"),
                ));
                panel
                    .spawn(Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(6.0),
                        align_items: AlignItems::Stretch,
                        ..default()
                    })
                    .with_children(|buttons| {
                        spawn_menu_button(buttons, &theme, PauseMenuAction::Resume, "RESUME");
                        spawn_menu_button(buttons, &theme, PauseMenuAction::Shipyard, "SHIPYARD");
                        spawn_menu_button(
                            buttons,
                            &theme,
                            PauseMenuAction::BaseEditor,
                            "SURFACE BASE",
                        );
                        spawn_menu_button(buttons, &theme, PauseMenuAction::Settings, "SETTINGS");
                        spawn_menu_button(buttons, &theme, PauseMenuAction::Quit, "QUIT");
                    });
            });
        });
}

fn spawn_menu_button(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    action: PauseMenuAction,
    label: &str,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(30.0),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                padding: UiRect::left(Val::Px(12.0)),
                justify_content: JustifyContent::FlexStart,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            action,
            Name::new(format!("PauseMenu{label}Button")),
        ))
        .with_children(|c| {
            c.spawn((
                Text::new(label),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(theme.text_primary),
            ));
        });
}

pub(crate) fn handle_escape_input(
    intent: Res<GameInputIntent>,
    scenario: Res<crate::scenario_menu::ScenarioMenu>,
    mut pause: ResMut<GamePause>,
    mut settings_menu: ResMut<SettingsMenu>,
    shipyard: Option<ResMut<crate::shipyard_editor::ShipyardEditor>>,
    base_editor: Option<ResMut<crate::base_editor::BaseEditor>>,
    mode: Option<ResMut<InteractionMode>>,
    target: Option<ResMut<TargetBody>>,
) {
    if !intent.escape {
        return;
    }

    // The destruction scenario picker is a forced modal: Escape must not
    // dismiss it or stack the pause menu on top. See `crate::scenario_menu`.
    if scenario.open {
        return;
    }

    // Settings overlay closes before the pause menu backdrop.
    if settings_menu.open {
        settings_menu.open = false;
        return;
    }

    // The shipyard editor closes before the pause menu opens. (While its
    // ship-name field is focused, the editor's own text-input system eats
    // Escape first by disabling the keyboard action source.)
    if let Some(mut shipyard) = shipyard
        && shipyard.open
    {
        shipyard.open = false;
        return;
    }

    // The base editor closes before the pause menu opens, same as the shipyard.
    if let Some(mut base_editor) = base_editor
        && base_editor.open
    {
        base_editor.open = false;
        return;
    }

    if pause.active {
        pause.active = false;
        return;
    }

    if let Some(mut mode) = mode
        && !matches!(*mode, InteractionMode::Idle)
    {
        *mode = InteractionMode::Idle;
        return;
    }

    if let Some(mut target) = target
        && target.target.is_some()
    {
        target.target = None;
        target.set_changed();
        return;
    }

    pause.active = true;
}

fn handle_button_clicks(
    interactions: Query<(&Interaction, &PauseMenuAction), Changed<Interaction>>,
    primary_window: Query<Entity, With<PrimaryWindow>>,
    mut pause: ResMut<GamePause>,
    mut settings_menu: ResMut<SettingsMenu>,
    mut shipyard: ResMut<crate::shipyard_editor::ShipyardEditor>,
    mut base_editor: ResMut<crate::base_editor::BaseEditor>,
    mut close_requested: MessageWriter<WindowCloseRequested>,
    mut app_exit: MessageWriter<AppExit>,
) {
    for (interaction, action) in &interactions {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        match action {
            PauseMenuAction::Resume => pause.active = false,
            PauseMenuAction::Shipyard => {
                pause.active = false;
                shipyard.open = true;
            }
            PauseMenuAction::BaseEditor => {
                pause.active = false;
                base_editor.open = true;
                base_editor.mode = crate::base_editor::BaseEditorMode::PickSite;
                base_editor.active_site = None;
            }
            PauseMenuAction::Settings => settings_menu.open = true,
            PauseMenuAction::Quit => {
                pause.active = false;
                if let Ok(window) = primary_window.single() {
                    close_requested.write(WindowCloseRequested { window });
                } else {
                    app_exit.write(AppExit::Success);
                }
            }
        }
    }
}

fn update_visibility(
    pause: Res<GamePause>,
    mut roots: Query<&mut Visibility, With<PauseMenuRoot>>,
) {
    if !pause.is_changed() {
        return;
    }
    let target = if pause.active {
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

fn update_button_visuals(
    theme: Res<HudTheme>,
    mut buttons: Query<
        (
            &Interaction,
            &mut BorderColor,
            &mut BackgroundColor,
            &Children,
        ),
        With<PauseMenuAction>,
    >,
    mut text_q: Query<&mut TextColor>,
) {
    for (interaction, mut border, mut bg, children) in &mut buttons {
        let (border_color, bg_color, label_color) = match interaction {
            Interaction::Pressed => (theme.text_primary, theme.panel_border, theme.text_primary),
            Interaction::Hovered => (theme.text_accent, theme.panel_bg, theme.text_accent),
            Interaction::None => (theme.panel_border, theme.panel_bg, theme.text_primary),
        };
        let new_border = BorderColor::all(border_color);
        if border.top != new_border.top {
            *border = new_border;
        }
        if bg.0 != bg_color {
            bg.0 = bg_color;
        }
        if let Some(&child) = children.first()
            && let Ok(mut tc) = text_q.get_mut(child)
            && tc.0 != label_color
        {
            tc.0 = label_color;
        }
    }
}
