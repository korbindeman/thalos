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
//! Scenario starts route two ways, on [`crate::loading::WorldState`]:
//!
//! - **World absent** (first start after a bare menu boot): every start is
//!   literally a *boot* triggered at runtime. `apply_menu_action` seats the
//!   sim / arms the boot deferred-placement flags
//!   ([`crate::spawn::DescentPlacement`], [`crate::runway::RunwayPlacement`]),
//!   registers the boot step set (world load + placement), flips the world
//!   [`Live`](crate::loading::WorldState::Live) — which fires the
//!   `OnEnter(WorldState::Live)` world-spawn systems next frame — and
//!   re-enters [`AppState::Loading`]. `ship_view::spawn_player_ship` builds
//!   the chosen scenario's own blueprint, so no craft swap is needed.
//! - **World live** (menu re-entered from flight): the existing in-place
//!   machinery. Same-craft scenarios (orbit, the two descents, EVA) go
//!   through [`crate::scenario_menu::respawn_into`] and transition straight
//!   to `Running`; craft-swapping scenarios (cruise + the runway pair fly
//!   the Meridian) queue a [`crate::relaunch::RelaunchRequest`]; runway
//!   scenarios additionally re-arm the deferred placement + settle gate and
//!   re-enter `Loading`.
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
use thalos_physics_local::{ActiveLocalBubble, HeightSourceRegistry};

use crate::hud::theme::{HudTheme, panel_frame};
use crate::loading::{
    AppState, LoadDestination, LoadingTracker, StepDesc, WorldState, step, steps_for,
    world_load_steps,
};
use crate::maneuver::{ManeuverPlan, SelectedNode};
use crate::player_controller::EvaMode;
use crate::relaunch::{RelaunchRequest, RelaunchSpec};
use crate::rendering::{PlayerShip, SimulationState};
use crate::runway::{RunwayPlacement, RunwaySite};
use crate::scenario_menu::respawn_into;
use crate::settings_menu::SettingsMenu;
use crate::shipyard_editor::OpenShipyardOnStart;
use crate::space_center::{HubSpaceportBuild, OpenSpaceCenterOnStart};
use crate::spawn::{DescentPlacement, Homeworld, SpawnSituation};
use crate::surface_settle::SurfaceSettle;

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

pub struct MainMenuPlugin;

impl Plugin for MainMenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PendingMenuAction>()
            .init_resource::<DevMenuExpanded>()
            .add_systems(OnEnter(AppState::MainMenu), (spawn_menu, throttle_menu_updates))
            .add_systems(OnExit(AppState::MainMenu), (despawn_menu, restore_update_rate))
            .add_systems(
                Update,
                (
                    handle_menu_escape,
                    collect_menu_clicks,
                    apply_menu_action,
                    update_dev_visibility,
                    update_button_visuals,
                )
                    .chain()
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
const SCREEN_BG: Color = Color::srgb(0.040, 0.038, 0.034);

const MENU_WIDTH: f32 = 380.0;

const SCENARIOS: &[(SpawnSituation, &str, &str)] = &[
    (
        SpawnSituation::ShipOrbit,
        "ORBIT",
        "low Thalos parking orbit",
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

fn spawn_menu(mut commands: Commands, theme: Res<HudTheme>, mut dev: ResMut<DevMenuExpanded>) {
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
                header.spawn((
                    Text::new("THALOS"),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(56.0),
                        ..default()
                    },
                    TextColor(theme.text_accent),
                    Name::new("MainMenuTitle"),
                ));
                header.spawn((
                    Text::new(concat!("pre-alpha  v", env!("CARGO_PKG_VERSION"))),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(11.0),
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
                // Primary: the usual PLAY / SETTINGS / QUIT.
                spawn_menu_button(
                    panel,
                    &theme,
                    MenuAction::Play,
                    "PLAY",
                    "enter the space center",
                );
                spawn_menu_button(
                    panel,
                    &theme,
                    MenuAction::Settings,
                    "SETTINGS",
                    "window & input",
                );
                spawn_menu_button(panel, &theme, MenuAction::Quit, "QUIT", "");

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

                // Secondary: the direct-scenario shortcuts, collapsed by default
                // (toggled by the button; visibility driven by `update_dev_visibility`).
                spawn_menu_button(
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
                            row_gap: Val::Px(6.0),
                            margin: UiRect::top(Val::Px(6.0)),
                            ..default()
                        },
                        Visibility::Hidden,
                        Name::new("MainMenuDevSection"),
                    ))
                    .with_children(|dev| {
                        for &(situation, label, desc) in SCENARIOS {
                            spawn_menu_button(
                                dev,
                                &theme,
                                MenuAction::Scenario(situation),
                                label,
                                desc,
                            );
                        }
                        spawn_menu_button(
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
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(theme.text_primary),
            ));
            if !desc.is_empty() {
                c.spawn((
                    Text::new(desc),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(10.0),
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
        ResMut<DescentPlacement>,
    ),
    // Deferred-world boot: `Absent` after a bare menu boot, until the first
    // start flips it `Live` (see `loading::WorldState`).
    world: (Res<State<WorldState>>, ResMut<NextState<WorldState>>),
    mut next_state: ResMut<NextState<AppState>>,
    runway_site: Option<Res<RunwaySite>>,
    ui: (
        ResMut<SettingsMenu>,
        ResMut<OpenShipyardOnStart>,
        ResMut<OpenSpaceCenterOnStart>,
        ResMut<HubSpaceportBuild>,
        ResMut<DevMenuExpanded>,
    ),
    exit: (
        Query<Entity, With<PrimaryWindow>>,
        MessageWriter<WindowCloseRequested>,
        MessageWriter<AppExit>,
    ),
) {
    let Some(action) = pending.0.take() else {
        return;
    };
    let (mut active, height_sources, mut eva_mode, mut plan, mut selected, homeworld) = respawn;
    let (mut tracker, mut dest, mut settle, mut runway_placement, mut relaunch, mut descent) =
        load;
    let (world_state, mut next_world) = world;
    let (mut settings, mut open_shipyard, mut open_space_center, mut hub_build, mut dev) = ui;
    let (primary_window, mut close_requested, mut app_exit) = exit;
    let world_absent = *world_state.get() == WorldState::Absent;

    let start = match action {
        MenuAction::ToggleDev => {
            dev.0 = !dev.0;
            return;
        }
        MenuAction::Play => {
            // Clean start: build the spaceport (base only — **no craft parked**)
            // behind the loading screen and reveal into the space-center hub. The
            // player launches a ship themselves from the VAB; nothing is loaded on
            // the pad/runway. If the spaceport already exists this session, skip
            // straight to the hub.
            open_space_center.0 = true;
            if runway_site.is_some() {
                next_state.set(AppState::Running);
            } else {
                hub_build.pending = true;
                // After a deferred menu boot the world itself still needs to
                // spawn: prepend the world-load steps and flip it live. The
                // spaceport build self-gates on terrain residency, so it
                // simply waits its turn within the same loading pass.
                let mut steps = Vec::new();
                if world_absent {
                    steps.extend(world_load_steps());
                    next_world.set(WorldState::Live);
                }
                steps.push(StepDesc::new(step::PLACEMENT, "Building spaceport", 1.0));
                tracker.begin(steps);
                dest.0 = AppState::Running;
                next_state.set(AppState::Loading);
            }
            return;
        }
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

    // Deferred-world start (first start after a bare menu boot): the world
    // was never spawned, so every scenario is literally a *boot* triggered at
    // runtime — no craft exists yet to swap or reseat. Arm the same
    // deferred-placement flags the boot arming systems set, register the boot
    // step set (world load + placement), and flip the world live: the
    // `OnEnter(WorldState::Live)` chain spawns bodies / ship / sky next frame
    // behind the loading pass, and `spawn_player_ship` reads the situation
    // just written above, so it builds the scenario's own blueprint directly
    // (meridian for the aircraft starts) — the relaunch craft-swap the
    // live-world paths below need is unnecessary here.
    if world_absent {
        match start {
            // Seat the sim itself into the scenario now (orbit state reset /
            // EVA vessel-kind swap — neither needs terrain, and the EVA
            // branch's ship-despawn loop just sees an empty query).
            SpawnSituation::ShipOrbit | SpawnSituation::Eva => {
                respawn_into(
                    start,
                    &mut commands,
                    &mut sim,
                    &mut active,
                    &height_sources,
                    &mut eva_mode,
                    &player_ship,
                    &mut plan,
                    &mut selected,
                    homeworld.0,
                );
            }
            // The descent family boots from the placeholder parking orbit and
            // is placed by `spawn::refine_descent_spawn` once terrain is
            // resident — `respawn_into` here would find no height source and
            // silently fall back to a bare orbit.
            SpawnSituation::Landing | SpawnSituation::FinalApproach | SpawnSituation::Cruise => {
                descent.pending = true;
            }
            SpawnSituation::Runway | SpawnSituation::RunwayApproach => {
                runway_placement.pending = true;
                settle.arm(matches!(start, SpawnSituation::Runway), false);
            }
        }
        tracker.begin(steps_for(start, true));
        dest.0 = AppState::Running;
        next_world.set(WorldState::Live);
        next_state.set(AppState::Loading);
        info!("start screen: launching {start:?} (world boot)");
        return;
    }

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
