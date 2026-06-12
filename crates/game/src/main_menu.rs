//! Start screen (main menu).
//!
//! Shown when the game is launched without a scenario (`just game`, or
//! `THALOS_SPAWN=menu`): the boot load runs behind the loading screen with
//! the placeholder parking-orbit world, then [`AppState::MainMenu`] presents
//! the scenario list plus SHIPYARD / SETTINGS / QUIT. Launching with an
//! explicit scenario (`just game runway`) or with `THALOS_AUTO_RUN` set
//! skips the menu entirely — agents and scripted runs keep their one-shot
//! flow. See `main.rs` for the boot routing.
//!
//! Scenario starts reuse the existing in-place machinery rather than
//! restarting the process:
//!
//! - **Same-craft scenarios** (orbit, the two descents, EVA) go through
//!   [`crate::scenario_menu::respawn_into`] — the destruction picker's
//!   respawn path — and transition straight to `Running`.
//! - **Craft-swapping scenarios** (cruise + the runway pair fly the
//!   Meridian, not the boot placeholder's rocket) queue a
//!   [`crate::relaunch::RelaunchRequest`], the shipyard Launch path.
//! - **Runway scenarios** additionally re-arm the deferred placement
//!   ([`crate::runway::RunwayPlacement`]) and the tile-settle gate, register
//!   a fresh loading pass ([`crate::loading::LoadingTracker::begin`]), and
//!   re-enter [`AppState::Loading`] so the site build + park + settle happen
//!   behind the loading screen exactly like a `just game runway` boot.
//!
//! While the menu is up the sim clock is paused ([`crate::sim_clock`]) and
//! gameplay input contexts are deactivated ([`crate::input`]); Escape only
//! closes the settings overlay (the pause menu is gated to `Running`).

use bevy::app::AppExit;
use bevy::picking::prelude::Pickable;
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, WindowCloseRequested};
use thalos_input::game::GameInputIntent;
use thalos_physics_local::{ActiveLocalBubble, HeightSourceRegistry, TerrainSurfaceRegistry};

use crate::hud::theme::{HudTheme, panel_frame};
use crate::loading::{AppState, LoadDestination, LoadingTracker, steps_for};
use crate::maneuver::{ManeuverPlan, SelectedNode};
use crate::player_controller::EvaMode;
use crate::relaunch::{RelaunchRequest, RelaunchSpec};
use crate::rendering::{PlayerShip, SimulationState};
use crate::runway::RunwayPlacement;
use crate::scenario_menu::respawn_into;
use crate::settings_menu::SettingsMenu;
use crate::shipyard_editor::OpenShipyardOnStart;
use crate::spawn::{Homeworld, SpawnSituation};
use crate::surface_settle::SurfaceSettle;

#[derive(Component)]
struct MainMenuRoot;

/// What a menu button does. Scenario buttons carry the situation they start.
#[derive(Component, Debug, Clone, Copy)]
enum MenuAction {
    Scenario(SpawnSituation),
    Shipyard,
    Settings,
    Quit,
}

/// Click → action hand-off so the (param-heavy) applier stays a single
/// consumer. **Sole writer:** [`collect_menu_clicks`].
#[derive(Resource, Default)]
struct PendingMenuAction(Option<MenuAction>);

pub struct MainMenuPlugin;

impl Plugin for MainMenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PendingMenuAction>()
            .add_systems(OnEnter(AppState::MainMenu), spawn_menu)
            .add_systems(OnExit(AppState::MainMenu), despawn_menu)
            .add_systems(
                Update,
                (
                    handle_menu_escape,
                    collect_menu_clicks,
                    apply_menu_action,
                    update_button_visuals,
                )
                    .chain()
                    .run_if(in_state(AppState::MainMenu)),
            );
    }
}

// Opaque backdrop matching the loading screen, so the boot placeholder world
// (and its HUD) never shows through. A live-world backdrop is a later polish
// pass — it needs the HUD hidden per app state first.
const SCREEN_BG: Color = Color::srgb(0.040, 0.038, 0.034);

const MENU_WIDTH: f32 = 380.0;

const SCENARIOS: &[(SpawnSituation, &str, &str)] = &[
    (SpawnSituation::ShipOrbit, "ORBIT", "low Thalos parking orbit"),
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

fn spawn_menu(mut commands: Commands, theme: Res<HudTheme>) {
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
                header.spawn((
                    Text::new("THALOS"),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: 56.0,
                        ..default()
                    },
                    TextColor(theme.text_accent),
                    Name::new("MainMenuTitle"),
                ));
                header.spawn((
                    Text::new(concat!("pre-alpha  v", env!("CARGO_PKG_VERSION"))),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: 11.0,
                        ..default()
                    },
                    TextColor(theme.text_dim),
                    Name::new("MainMenuVersion"),
                ));
            });

            let (bg, border) = panel_frame(&theme);
            root.spawn((
                Node {
                    width: Val::Px(MENU_WIDTH),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    padding: UiRect::axes(Val::Px(18.0), Val::Px(16.0)),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Stretch,
                    row_gap: Val::Px(6.0),
                    ..default()
                },
                bg,
                border,
                Name::new("MainMenuPanel"),
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new("START SCENARIO"),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: 11.0,
                        ..default()
                    },
                    TextColor(theme.text_dim),
                    Name::new("MainMenuScenarioHeading"),
                ));
                for &(situation, label, desc) in SCENARIOS {
                    spawn_menu_button(panel, &theme, MenuAction::Scenario(situation), label, desc);
                }
                panel.spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(1.0),
                        margin: UiRect::vertical(Val::Px(6.0)),
                        ..default()
                    },
                    BackgroundColor(theme.panel_border),
                    Name::new("MainMenuDivider"),
                ));
                spawn_menu_button(panel, &theme, MenuAction::Shipyard, "SHIPYARD", "design a craft");
                spawn_menu_button(panel, &theme, MenuAction::Settings, "SETTINGS", "input bindings");
                spawn_menu_button(panel, &theme, MenuAction::Quit, "QUIT", "");
            });
        });
}

fn spawn_menu_button(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    action: MenuAction,
    label: &str,
    desc: &str,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(30.0),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                padding: UiRect::horizontal(Val::Px(12.0)),
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            action,
            Name::new(format!("MainMenu{label}Button")),
        ))
        .with_children(|c| {
            c.spawn((
                Text::new(label),
                TextFont {
                    font: theme.font.clone(),
                    font_size: 12.0,
                    ..default()
                },
                TextColor(theme.text_primary),
            ));
            if !desc.is_empty() {
                c.spawn((
                    Text::new(desc),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: 10.0,
                        ..default()
                    },
                    TextColor(theme.text_dim),
                ));
            }
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

/// Consume the clicked action. Scenario starts mutate [`SpawnSituation`] (the
/// per-frame scenario consumers — engine lighting, runway systems — read it
/// live) and route per the module docs.
#[allow(clippy::too_many_arguments)]
fn apply_menu_action(
    mut pending: ResMut<PendingMenuAction>,
    mut commands: Commands,
    mut sim: ResMut<SimulationState>,
    mut situation: ResMut<SpawnSituation>,
    // `respawn_into` inputs, bundled to stay within the 16-param limit.
    respawn: (
        ResMut<ActiveLocalBubble>,
        Res<HeightSourceRegistry>,
        Res<TerrainSurfaceRegistry>,
        ResMut<EvaMode>,
        ResMut<ManeuverPlan>,
        ResMut<SelectedNode>,
        Res<Homeworld>,
    ),
    player_ship: Query<Entity, With<PlayerShip>>,
    // Loading-pass plumbing for the runway scenarios.
    load: (
        ResMut<LoadingTracker>,
        ResMut<LoadDestination>,
        ResMut<SurfaceSettle>,
        ResMut<RunwayPlacement>,
        ResMut<RelaunchRequest>,
    ),
    mut next_state: ResMut<NextState<AppState>>,
    ui: (ResMut<SettingsMenu>, ResMut<OpenShipyardOnStart>),
    exit: (
        Query<Entity, With<PrimaryWindow>>,
        MessageWriter<WindowCloseRequested>,
        MessageWriter<AppExit>,
    ),
) {
    let Some(action) = pending.0.take() else {
        return;
    };
    let (
        mut active,
        height_sources,
        surfaces,
        mut eva_mode,
        mut plan,
        mut selected,
        homeworld,
    ) = respawn;
    let (mut tracker, mut dest, mut settle, mut runway_placement, mut relaunch) = load;
    let (mut settings, mut open_shipyard) = ui;
    let (primary_window, mut close_requested, mut app_exit) = exit;

    let start = match action {
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
        MenuAction::Shipyard => {
            // The editor opens on entry to `Running` (never during a load) —
            // the same deferred-open used by `just game shipyard`.
            open_shipyard.0 = true;
            SpawnSituation::ShipOrbit
        }
        MenuAction::Scenario(s) => s,
    };

    *situation = start;
    match start {
        // Same-craft starts: seat the boot placeholder craft (or the EVA
        // capsule) into the scenario in place and reveal immediately.
        SpawnSituation::ShipOrbit
        | SpawnSituation::Landing
        | SpawnSituation::FinalApproach
        | SpawnSituation::Eva => {
            respawn_into(
                start,
                &mut commands,
                &mut sim,
                &mut active,
                &height_sources,
                &surfaces,
                &mut eva_mode,
                &player_ship,
                &mut plan,
                &mut selected,
                homeworld.0,
            );
            next_state.set(AppState::Running);
        }
        // Craft swap, airborne placement handled inside the relaunch flow.
        SpawnSituation::Cruise => {
            let Some(blueprint) =
                crate::ship_view::load_blueprint_from_path(start.ship_blueprint_path())
            else {
                error!("start screen: cruise blueprint failed to load; staying on menu");
                return;
            };
            relaunch.0 = Some(RelaunchSpec {
                blueprint,
                situation: start,
            });
            next_state.set(AppState::Running);
        }
        // Craft swap + deferred terrain-aware placement: run a fresh loading
        // pass so the site build, park, and tile settle stay behind the
        // loading screen, exactly like a runway boot.
        SpawnSituation::Runway | SpawnSituation::RunwayApproach => {
            let Some(blueprint) =
                crate::ship_view::load_blueprint_from_path(start.ship_blueprint_path())
            else {
                error!("start screen: runway blueprint failed to load; staying on menu");
                return;
            };
            relaunch.0 = Some(RelaunchSpec {
                blueprint,
                situation: start,
            });
            runway_placement.pending = true;
            settle.arm(matches!(start, SpawnSituation::Runway), false);
            tracker.begin(steps_for(start, false));
            dest.0 = AppState::Running;
            next_state.set(AppState::Loading);
        }
    }
    info!("start screen: launching {start:?}");
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
        With<MenuAction>,
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
