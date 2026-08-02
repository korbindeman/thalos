//! Start screen (main menu).
//!
//! Shown when the game is launched without a scenario (`just game`, or
//! `THALOS_SPAWN=menu`). A bare menu boot **defers the world entirely**
//! ([`crate::loading::WorldState::Absent`]): no bodies, player ship, or sky
//! are spawned, no terrain streams, and the loading pass registers zero
//! steps — the menu appears near-instantly as a static UI over an empty
//! scene, and the frame loop is throttled to reactive updates while it is
//! up. Launching with an explicit scenario (`just game runway`) or with
//! `THALOS_AUTO_RUN` set skips the menu (and boots the world immediately) —
//! agents and scripted runs keep their one-shot flow. See `main.rs` for the
//! boot routing.
//!
//! PLAY and every developer shortcut submit a
//! [`SessionLoadRequest`](crate::session_loading::SessionLoadRequest). The
//! session loader is the only code that decides how a source is validated and
//! projected, so a cold boot and a live-world replacement cannot grow separate
//! gameplay setup paths here.
//!
//! While the menu is up the sim clock is paused ([`crate::sim_clock`]) and
//! gameplay input contexts are deactivated ([`crate::input`]); Escape only
//! closes the settings overlay (the pause menu is gated to `Running`).

use std::time::Duration;

use bevy::app::AppExit;
use bevy::picking::prelude::Pickable;
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, WindowCloseRequested};
use bevy::winit::{UpdateMode, WinitSettings};
use thalos_input::game::GameInputIntent;

use thalos_ui::{self as ui, SPACE_SM, SPACE_XS, UiTheme, spawn_divider, spawn_menu_row, tokens};

use crate::loading::AppState;
use crate::session_loading::{ScenarioFixture, SessionLoadRequest, SessionSource};
use crate::settings_menu::SettingsMenu;
use crate::spawn::SpawnSituation;

#[derive(Component)]
struct MainMenuRoot;

/// The collapsible "Quick start / Dev" container (the direct-scenario shortcuts,
/// tucked below the primary PLAY / SETTINGS / QUIT).
#[derive(Component)]
struct DevSection;

/// Whether the Dev / Quick-start submenu is expanded. **Sole writer:**
/// [`apply_menu_action`] (via the toggle button).
#[derive(Resource, Default)]
struct DevMenuExpanded(bool);

/// What a menu button does. Scenario buttons carry the situation they start.
#[derive(Component, Debug, Clone, Copy)]
enum MenuAction {
    /// The primary entry: load the spaceport world and land in the space center.
    Play,
    /// Expand / collapse the Dev / Quick-start scenario submenu.
    ToggleDev,
    Scenario(SpawnSituation),
    Shipyard,
    Settings,
    Quit,
}

/// Click → action hand-off so the (param-heavy) applier stays a single
/// consumer. **Sole writer:** [`collect_menu_clicks`].
#[derive(Resource, Default)]
struct PendingMenuAction(Option<MenuAction>);

/// Ordering seam consumed by the canonical session loader. All menu click
/// collection/application finishes before it looks for a request.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MainMenuActionSet;

pub struct MainMenuPlugin;

impl Plugin for MainMenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PendingMenuAction>()
            .init_resource::<DevMenuExpanded>()
            .add_systems(
                OnEnter(AppState::MainMenu),
                (spawn_menu, throttle_menu_updates),
            )
            .add_systems(
                OnExit(AppState::MainMenu),
                (despawn_menu, restore_update_rate),
            )
            .add_systems(
                Update,
                (
                    handle_menu_escape,
                    collect_menu_clicks,
                    apply_menu_action,
                    update_dev_visibility,
                )
                    .chain()
                    .in_set(MainMenuActionSet)
                    .run_if(in_state(AppState::MainMenu)),
            );
    }
}

/// Throttle the frame loop while the menu is up: the menu is a static UI over
/// an opaque backdrop, so there is nothing to render at full tick. Reactive
/// mode still wakes immediately on input (hover/click stay crisp) and
/// otherwise idles at ~30 Hz focused / 4 Hz unfocused instead of spinning the
/// GPU at monitor refresh.
fn throttle_menu_updates(mut commands: Commands) {
    commands.insert_resource(WinitSettings {
        focused_mode: UpdateMode::reactive(Duration::from_millis(33)),
        unfocused_mode: UpdateMode::reactive_low_power(Duration::from_millis(250)),
    });
}

/// Back to the continuous game loop the moment the menu is left (Loading and
/// Running both need full-rate updates — loading drives async work per frame,
/// and the sim renders every frame).
fn restore_update_rate(mut commands: Commands) {
    commands.insert_resource(WinitSettings::game());
}

// Opaque backdrop matching the loading screen, so the boot placeholder world
// (and its HUD) never shows through. A live-world backdrop is a later polish
// pass — it needs the HUD hidden per app state first.
const SCREEN_BG: Color = tokens::SCREEN_BG;

const MENU_WIDTH: f32 = 380.0;

const SCENARIOS: &[(SpawnSituation, &str, &str)] = &[
    (
        SpawnSituation::ShipOrbit,
        "ORBIT",
        "low Thalos parking orbit",
    ),
    (
        SpawnSituation::PolarOrbit,
        "POLAR ORBIT",
        "low polar parking orbit",
    ),
    (
        SpawnSituation::Landing,
        "LANDING APPROACH",
        "powered descent from 25 km",
    ),
    (
        SpawnSituation::FinalApproach,
        "FINAL APPROACH",
        "low and slow over flat ground",
    ),
    (
        SpawnSituation::Cruise,
        "CRUISE",
        "Meridian level at 15,000 ft",
    ),
    (
        SpawnSituation::Launch,
        "LAUNCH",
        "Saturn standing on the launchpad",
    ),
    (
        SpawnSituation::Runway,
        "RUNWAY",
        "Meridian parked for takeoff",
    ),
    (
        SpawnSituation::RunwayApproach,
        "RUNWAY APPROACH",
        "Meridian on short final",
    ),
    (SpawnSituation::Eva, "EVA", "on foot on the surface"),
];

fn spawn_menu(mut commands: Commands, theme: Res<UiTheme>, mut dev: ResMut<DevMenuExpanded>) {
    // Start collapsed each time the menu is shown (keeps the section's spawned
    // `Hidden` state and the resource in agreement on re-entry).
    dev.0 = false;
    commands
        .spawn((
            MainMenuRoot,
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
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(26.0),
                ..default()
            },
            BackgroundColor(SCREEN_BG),
            // Above the HUD and pause menu (100), below the loading screen
            // (1000) so a runway re-load cleanly covers any teardown frame.
            GlobalZIndex(900),
            Pickable {
                is_hoverable: false,
                should_block_lower: true,
            },
            Name::new("MainMenu"),
        ))
        .with_children(|root| {
            // Logotype + version, mirroring the loading screen.
            root.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    row_gap: Val::Px(4.0),
                    ..default()
                },
                Name::new("MainMenuHeader"),
            ))
            .with_children(|header| {
                let mut title = theme.display("THALOS");
                title.2 = TextColor(tokens::ACCENT);
                header.spawn((title, Name::new("MainMenuTitle")));
                header.spawn((
                    theme.faint(concat!("pre-alpha  v", env!("CARGO_PKG_VERSION"))),
                    Name::new("MainMenuVersion"),
                ));
            });

            root.spawn((
                Node {
                    width: Val::Px(MENU_WIDTH),
                    align_items: AlignItems::Stretch,
                    row_gap: Val::Px(SPACE_SM),
                    ..ui::panel_node()
                },
                theme.glass(),
                Name::new("MainMenuPanel"),
            ))
            .with_children(|panel| {
                // Primary: the usual PLAY / SETTINGS / QUIT.
                spawn_menu_row(
                    panel,
                    &theme,
                    MenuAction::Play,
                    "PLAY",
                    "enter the space center",
                );
                spawn_menu_row(
                    panel,
                    &theme,
                    MenuAction::Settings,
                    "SETTINGS",
                    "window & input",
                );
                spawn_menu_row(panel, &theme, MenuAction::Quit, "QUIT", "");

                spawn_divider(panel);

                // Secondary: the direct-scenario shortcuts, collapsed by default
                // (toggled by the button; visibility driven by `update_dev_visibility`).
                spawn_menu_row(
                    panel,
                    &theme,
                    MenuAction::ToggleDev,
                    "QUICK START / DEV",
                    "jump straight to a scenario",
                );
                panel
                    .spawn((
                        DevSection,
                        Node {
                            flex_direction: FlexDirection::Column,
                            align_items: AlignItems::Stretch,
                            row_gap: Val::Px(SPACE_XS + 2.0),
                            margin: UiRect::top(Val::Px(SPACE_XS + 2.0)),
                            ..default()
                        },
                        Visibility::Hidden,
                        Name::new("MainMenuDevSection"),
                    ))
                    .with_children(|dev| {
                        for &(situation, label, desc) in SCENARIOS {
                            spawn_menu_row(
                                dev,
                                &theme,
                                MenuAction::Scenario(situation),
                                label,
                                desc,
                            );
                        }
                        spawn_menu_row(
                            dev,
                            &theme,
                            MenuAction::Shipyard,
                            "SHIPYARD",
                            "design a craft",
                        );
                    });
            });
        });
}

fn despawn_menu(mut commands: Commands, roots: Query<Entity, With<MainMenuRoot>>) {
    for entity in &roots {
        commands.entity(entity).despawn();
    }
}

/// Escape on the start screen only closes the settings overlay; the pause
/// menu's Escape chain is gated to `Running`.
fn handle_menu_escape(intent: Res<GameInputIntent>, mut settings: ResMut<SettingsMenu>) {
    if intent.escape && settings.open {
        settings.open = false;
    }
}

fn collect_menu_clicks(
    interactions: Query<(&Interaction, &MenuAction), Changed<Interaction>>,
    mut pending: ResMut<PendingMenuAction>,
) {
    for (interaction, action) in &interactions {
        if matches!(interaction, Interaction::Pressed) {
            pending.0 = Some(*action);
            break;
        }
    }
}

/// Consume UI-only actions locally and submit playable sources to the one
/// session loader. No campaign object is spawned from this module.
fn apply_menu_action(
    mut pending: ResMut<PendingMenuAction>,
    mut requests: ResMut<SessionLoadRequest>,
    mut settings: ResMut<SettingsMenu>,
    mut dev: ResMut<DevMenuExpanded>,
    exit: (
        Query<Entity, With<PrimaryWindow>>,
        MessageWriter<WindowCloseRequested>,
        MessageWriter<AppExit>,
    ),
) {
    let Some(action) = pending.0.take() else {
        return;
    };
    let (primary_window, mut close_requested, mut app_exit) = exit;

    let source = match action {
        MenuAction::ToggleDev => {
            dev.0 = !dev.0;
            return;
        }
        MenuAction::Play => SessionSource::NewCampaign,
        MenuAction::Settings => {
            settings.open = true;
            return;
        }
        MenuAction::Quit => {
            // Same path as the pause menu's QUIT: request a window close so
            // any close handlers run, with a bare exit as the fallback.
            if let Ok(window) = primary_window.single() {
                close_requested.write(WindowCloseRequested { window });
            } else {
                app_exit.write(AppExit::Success);
            }
            return;
        }
        MenuAction::Shipyard => SessionSource::Fixture(ScenarioFixture::Shipyard),
        MenuAction::Scenario(situation) => {
            SessionSource::Fixture(ScenarioFixture::Flight(situation))
        }
    };
    let generation = requests.request(source);
    info!(
        "start screen: requested session generation {} from {:?}",
        generation.0, source
    );
}

/// Show/hide the collapsible Dev / Quick-start section on toggle.
fn update_dev_visibility(
    dev: Res<DevMenuExpanded>,
    mut sections: Query<&mut Visibility, With<DevSection>>,
) {
    if !dev.is_changed() {
        return;
    }
    let target = if dev.0 {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut vis in &mut sections {
        if *vis != target {
            *vis = target;
        }
    }
}
