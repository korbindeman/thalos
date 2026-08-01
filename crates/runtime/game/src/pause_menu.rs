//! Game-level pause menu.
//!
//! The menu itself owns only the Escape-modal state (`GamePause`) and its UI.
//! Simulation pause aggregation lives in [`crate::sim_clock`], which folds the
//! menu, destruction scenario picker, freecam (when not warp-eligible on enter),
//! and warp pause into an explicit
//! simulation clock. Bevy's default `Time`/`Time<Virtual>` remains an app clock
//! so presentation effects can keep animating while canonical/local simulation
//! is paused.

use bevy::app::AppExit;
use bevy::picking::prelude::Pickable;
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, WindowCloseRequested};
use thalos_input::game::GameInputIntent;

use thalos_ui::{self as ui, SPACE_XS, UiTheme, spawn_divider, spawn_menu_row};

use crate::game_context::{ContextHistory, GameContext, back_out, enter_context};
use crate::maneuver::InteractionMode;
use crate::settings_menu::SettingsMenu;
use crate::target::TargetBody;

pub use thalos_game_state::ui::{GamePause, not_game_paused};

#[derive(Component)]
struct PauseMenuRoot;

#[derive(Component, Clone, Copy)]
enum PauseMenuAction {
    Resume,
    SpaceCenter,
    Shipyard,
    BaseEditor,
    Settings,
    /// Leave the flight and return to the start screen (`AppState::MainMenu`) —
    /// the only direct flight→menu route (previously reachable only via the hub).
    MainMenu,
    Quit,
}

pub struct PauseMenuPlugin;

impl Plugin for PauseMenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GamePause>()
            .add_systems(Startup, setup.after(thalos_ui::init_ui_theme))
            // Gated to `Running`: during loading there is nothing to pause,
            // and on the start screen Escape is owned by `crate::main_menu`.
            .add_systems(
                Update,
                handle_escape_input
                    .run_if(in_state(crate::loading::AppState::Running))
                    .before(crate::SimStage::Physics),
            )
            .add_systems(Update, (handle_button_clicks, update_visibility).chain());
    }
}

fn setup(mut commands: Commands, theme: Res<UiTheme>) {
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
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.30)),
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
            root.spawn((
                Node {
                    width: Val::Px(260.0),
                    align_items: AlignItems::Stretch,
                    ..ui::panel_node()
                },
                theme.glass_heavy(),
                Name::new("PauseMenuPanel"),
            ))
            .with_children(|panel| {
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
                        header.spawn((theme.title("PAUSED"), Name::new("PauseMenuTitle")));
                        header.spawn((theme.faint("THALOS v0"), Name::new("PauseMenuVersion")));
                    });
                spawn_divider(panel);
                panel
                    .spawn(Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(SPACE_XS + 2.0),
                        align_items: AlignItems::Stretch,
                        ..default()
                    })
                    .with_children(|buttons| {
                        spawn_menu_row(buttons, &theme, PauseMenuAction::Resume, "RESUME", "");
                        spawn_menu_row(
                            buttons,
                            &theme,
                            PauseMenuAction::SpaceCenter,
                            "SPACE CENTER",
                            "",
                        );
                        spawn_menu_row(buttons, &theme, PauseMenuAction::Shipyard, "SHIPYARD", "");
                        spawn_menu_row(
                            buttons,
                            &theme,
                            PauseMenuAction::BaseEditor,
                            "SURFACE BASE",
                            "",
                        );
                        spawn_menu_row(buttons, &theme, PauseMenuAction::Settings, "SETTINGS", "");
                        spawn_menu_row(buttons, &theme, PauseMenuAction::MainMenu, "MAIN MENU", "");
                        spawn_menu_row(buttons, &theme, PauseMenuAction::Quit, "QUIT", "");
                    });
            });
        });
}

pub(crate) fn handle_escape_input(
    intent: Res<GameInputIntent>,
    scenario: Res<crate::scenario_menu::ScenarioMenu>,
    mut pause: ResMut<GamePause>,
    mut settings_menu: ResMut<SettingsMenu>,
    ctx: Option<Res<State<GameContext>>>,
    next_ctx: Option<ResMut<NextState<GameContext>>>,
    mut history: ResMut<ContextHistory>,
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

    // The pause menu (which can sit over a root context, e.g. the PLAY hub)
    // closes before we back out of a context.
    if pause.active {
        pause.active = false;
        return;
    }

    // Context back-out: in a non-Flight mode (hub / VAB / base editor), Escape
    // pops one level toward the parent it was opened from. At the **root** (an
    // empty stack — the PLAY-rooted hub, or a `just game shipyard` VAB) Escape
    // opens the pause menu instead of leaving the game; the pause menu's MAIN
    // MENU button is the sole deliberate exit to the start screen (the user's
    // rule). A focused editor text field eats Escape upstream by disabling the
    // keyboard action source, so this never fires mid-rename.
    if let (Some(ctx), Some(mut next)) = (ctx, next_ctx)
        && !matches!(*ctx.get(), GameContext::Flight)
    {
        if back_out(&mut next, &mut history).is_none() {
            pause.active = true;
        }
        return;
    }

    // Flight sub-modals.
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
    mut base_editor: ResMut<crate::base_editor::BaseEditor>,
    ctx: Option<Res<State<GameContext>>>,
    mut next_ctx: Option<ResMut<NextState<GameContext>>>,
    mut history: ResMut<ContextHistory>,
    mut next_state: ResMut<NextState<crate::loading::AppState>>,
    mut close_requested: MessageWriter<WindowCloseRequested>,
    mut app_exit: MessageWriter<AppExit>,
) {
    // Where the pause menu was opened from (usually Flight; SpaceCenter when
    // paused over the PLAY-rooted hub). Entering a mode remembers it so Escape
    // backs out here.
    let current = ctx
        .as_ref()
        .map(|c| *c.get())
        .unwrap_or(GameContext::Flight);
    for (interaction, action) in &interactions {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        match action {
            PauseMenuAction::Resume => pause.active = false,
            PauseMenuAction::SpaceCenter => {
                pause.active = false;
                if let Some(next) = next_ctx.as_mut() {
                    enter_context(next, &mut history, current, GameContext::SpaceCenter);
                }
            }
            PauseMenuAction::Shipyard => {
                pause.active = false;
                if let Some(next) = next_ctx.as_mut() {
                    enter_context(next, &mut history, current, GameContext::Vab);
                }
            }
            PauseMenuAction::BaseEditor => {
                pause.active = false;
                base_editor.mode = crate::base_editor::BaseEditorMode::PickSite;
                base_editor.active_site = None;
                if let Some(next) = next_ctx.as_mut() {
                    enter_context(next, &mut history, current, GameContext::BaseEditor);
                }
            }
            PauseMenuAction::Settings => settings_menu.open = true,
            PauseMenuAction::MainMenu => {
                // Return to the start screen — the direct flight→menu route.
                // Mirrors the hub's EXIT-to-menu path: setting the app state is
                // enough; `OnEnter(MainMenu)` shows the menu and `sim_clock`
                // freezes the still-loaded flight world behind it.
                pause.active = false;
                next_state.set(crate::loading::AppState::MainMenu);
            }
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
