//! Post-destruction scenario picker.
//!
//! When the player craft is destroyed (`Simulation::is_destroyed`), the game is
//! force-paused and this modal overlay offers the same four start scenarios the
//! game boots into (`just game [mode]`): ship in orbit, landing approach, final
//! approach, or on-foot EVA. Picking one repairs the craft and respawns the
//! player **in place** — no process relaunch — and unpauses.
//!
//! Pause coupling lives in [`crate::sim_clock`] and [`crate::pause_menu`]:
//! `ScenarioMenu::open` is folded into `SimClock` (freezes canonical/local sim
//! time) and `not_game_paused` (gates the `SimStage` system sets and every
//! other `not_game_paused` system), so while the picker is up the whole game
//! halts. `open` simply mirrors
//! `is_destroyed()` ([`sync_menu_to_destruction`]), so any repair path — a
//! button here, or a debug teleport — closes it.
//!
//! Respawn relies on the existing on-demand body spawner: tearing down the
//! wrecked craft's Avian bubble (`ActiveLocalBubble`) lets
//! `local_physics::spawn_player_avian_body` build a fresh body for the chosen
//! vessel kind on the next physics frame. The three ship scenarios reuse
//! `spawn::orbit_parking_state` / `spawn::compute_descent_state`; EVA performs
//! the one runtime vessel-kind swap (Ship → on-foot). Going the other way
//! (re-boarding a ship from EVA) is not wired — but EVA can't be destroyed, so
//! the picker only ever opens on a wrecked ship.

use bevy::picking::prelude::Pickable;
use bevy::prelude::*;

use thalos_physics_canonical::canonical::AuthorityMode;
use thalos_physics_canonical::types::{ShipParameters, VesselKind};
use thalos_physics_local::{ActiveLocalBubble, HeightSourceRegistry};
use thalos_world::BodyId;

use crate::hud::theme::{HudTheme, panel_frame};
use crate::maneuver::{ManeuverPlan, SelectedNode};
use crate::player_controller::EvaMode;
use crate::rendering::{PlayerShip, SimulationState};
use crate::spawn::{Homeworld, SpawnSituation, compute_descent_state, orbit_respawn_state};

/// Whether the destruction scenario picker is shown (and the game halted).
///
/// Mirrors `Simulation::is_destroyed()` via [`sync_menu_to_destruction`]; read
/// as a pause source by [`crate::pause_menu`]. Default `false`; the resource is
/// inserted at plugin build so the `not_game_paused` run condition can read it
/// from the first frame.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct ScenarioMenu {
    pub open: bool,
}

#[derive(Component)]
struct ScenarioMenuRoot;

/// The sub-line text node carrying the impact speed.
#[derive(Component)]
struct ScenarioMenuDetail;

/// Tags each button with the scenario it respawns into.
#[derive(Component, Clone, Copy)]
struct ScenarioButton(SpawnSituation);

pub struct ScenarioMenuPlugin;

impl Plugin for ScenarioMenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScenarioMenu>()
            .add_systems(Startup, setup.after(crate::hud::theme::init_theme))
            // Open/close tracks destruction. Runs before the physics gate so the
            // pause run-conditions see a coherent value within the frame.
            .add_systems(
                Update,
                sync_menu_to_destruction.before(crate::SimStage::Physics),
            )
            // Button handling + visuals. Not gated by `not_game_paused` (the
            // whole point is to run while the game is halted) — Bevy UI focus
            // updates `Interaction` on wall-clock time regardless.
            .add_systems(
                Update,
                (
                    handle_button_clicks,
                    update_visibility,
                    update_detail,
                    update_button_visuals,
                )
                    .chain(),
            );
    }
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
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            // Above the escape pause menu (z 100); the two never co-show, but a
            // crash mid-frame should never sit behind it.
            GlobalZIndex(110),
            Pickable {
                is_hoverable: false,
                should_block_lower: true,
            },
            Visibility::Hidden,
            ScenarioMenuRoot,
            Name::new("ScenarioMenu"),
        ))
        .with_children(|root| {
            // Panel background from the shared frame; the border is overridden
            // warn-tinted below so this reads as a failure, not a pause.
            let (bg, _border) = panel_frame(&theme);
            root.spawn((
                Node {
                    width: Val::Px(300.0),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    padding: UiRect::axes(Val::Px(18.0), Val::Px(16.0)),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Stretch,
                    row_gap: Val::Px(10.0),
                    ..default()
                },
                bg,
                // Warn-tinted edge so it reads as a failure state, not the
                // ordinary pause menu.
                BorderColor::all(theme.text_warn),
                Name::new("ScenarioMenuPanel"),
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new("VESSEL DESTROYED"),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: 18.0,
                        ..default()
                    },
                    TextColor(theme.text_warn),
                    Name::new("ScenarioMenuTitle"),
                ));
                panel.spawn((
                    Text::new(""),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: 12.0,
                        ..default()
                    },
                    TextColor(theme.text_dim),
                    ScenarioMenuDetail,
                    Name::new("ScenarioMenuDetail"),
                ));
                panel.spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(1.0),
                        ..default()
                    },
                    BackgroundColor(theme.panel_border),
                    Name::new("ScenarioMenuDivider"),
                ));
                panel
                    .spawn(Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(6.0),
                        align_items: AlignItems::Stretch,
                        ..default()
                    })
                    .with_children(|buttons| {
                        spawn_scenario_button(
                            buttons,
                            &theme,
                            SpawnSituation::ShipOrbit,
                            "RELAUNCH — SHIP IN ORBIT",
                        );
                        spawn_scenario_button(
                            buttons,
                            &theme,
                            SpawnSituation::Landing,
                            "RELAUNCH — LANDING APPROACH",
                        );
                        spawn_scenario_button(
                            buttons,
                            &theme,
                            SpawnSituation::FinalApproach,
                            "RELAUNCH — FINAL APPROACH",
                        );
                        spawn_scenario_button(
                            buttons,
                            &theme,
                            SpawnSituation::Cruise,
                            "RELAUNCH — CRUISE (15,000 FT)",
                        );
                        spawn_scenario_button(
                            buttons,
                            &theme,
                            SpawnSituation::Eva,
                            "DISEMBARK — ON FOOT (EVA)",
                        );
                    });
            });
        });
}

fn spawn_scenario_button(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    situation: SpawnSituation,
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
            ScenarioButton(situation),
            Name::new(format!("ScenarioButton{label}")),
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
        });
}

/// Sole writer of `ScenarioMenu::open`: it tracks `is_destroyed()`, so the menu
/// opens on a crash and closes the moment any path repairs the craft.
pub(crate) fn sync_menu_to_destruction(sim: Res<SimulationState>, mut menu: ResMut<ScenarioMenu>) {
    let destroyed = sim.simulation.is_destroyed();
    if menu.open != destroyed {
        menu.open = destroyed;
    }
}

fn update_visibility(
    menu: Res<ScenarioMenu>,
    mut roots: Query<&mut Visibility, With<ScenarioMenuRoot>>,
) {
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

fn update_detail(
    menu: Res<ScenarioMenu>,
    sim: Res<SimulationState>,
    mut detail_q: Query<&mut Text, With<ScenarioMenuDetail>>,
) {
    if !menu.open {
        return;
    }
    if let Ok(mut text) = detail_q.single_mut() {
        let new_value = format!("impact {:.0} m/s", sim.simulation.last_impact_speed_m_s());
        if text.0 != new_value {
            text.0 = new_value;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_button_clicks(
    menu: Res<ScenarioMenu>,
    interactions: Query<(&Interaction, &ScenarioButton), Changed<Interaction>>,
    mut commands: Commands,
    mut sim: ResMut<SimulationState>,
    mut active: ResMut<ActiveLocalBubble>,
    height_sources: Res<HeightSourceRegistry>,
    mut eva_mode: ResMut<EvaMode>,
    player_ship: Query<Entity, With<PlayerShip>>,
    mut plan: ResMut<ManeuverPlan>,
    mut selected: ResMut<SelectedNode>,
    homeworld: Res<Homeworld>,
) {
    if !menu.open {
        return;
    }
    for (interaction, button) in &interactions {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        respawn_into(
            button.0,
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
        // One respawn per frame; `is_destroyed` is now clear, so the menu
        // closes next frame via `sync_menu_to_destruction`.
        break;
    }
}

/// Reset the craft and seat it into `situation` in place. Shared by the
/// destruction picker's buttons and the start screen's same-craft scenario
/// starts ([`crate::main_menu`]); craft-swapping starts (cruise, runway) go
/// through [`crate::relaunch`] instead.
#[allow(clippy::too_many_arguments)]
pub(crate) fn respawn_into(
    situation: SpawnSituation,
    commands: &mut Commands,
    sim: &mut SimulationState,
    active: &mut ActiveLocalBubble,
    height_sources: &HeightSourceRegistry,
    eva_mode: &mut EvaMode,
    player_ship: &Query<Entity, With<PlayerShip>>,
    plan: &mut ManeuverPlan,
    selected: &mut SelectedNode,
    homeworld: BodyId,
) {
    // Fresh-craft reset shared by every scenario: clear structural failure,
    // drop back to 1×, discard the wreck's flight plan, and tear down its Avian
    // bubble. `spawn_player_avian_body` rebuilds a clean body for the (possibly
    // new) vessel kind next physics frame. The `transition_authority` calls
    // below leave any landed `BodyFixed` state by switching to `OnRails`.
    sim.simulation.repair();
    sim.simulation.warp.reset();
    if !plan.nodes.is_empty() {
        plan.nodes.clear();
        plan.dirty = true;
    }
    selected.id = None;
    clear_bubble(commands, active);

    match situation {
        SpawnSituation::Eva => {
            // Ship → on-foot. Swap the vessel kind, drop the rocket visuals, and
            // let `spawn_player_avian_body` plant the EVA capsule at the
            // sub-stellar daylight point next frame (it refines the EVA pose
            // itself), so no canonical state is set here.
            sim.simulation.set_vessel_kind(VesselKind::Eva);
            sim.simulation.set_ship_params(ShipParameters::eva());
            for entity in player_ship.iter() {
                commands.entity(entity).despawn();
            }
            *eva_mode = EvaMode::Grounded;
            sim.simulation
                .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
        }
        SpawnSituation::ShipOrbit => {
            let (state, attitude) = orbit_respawn_state(sim, homeworld);
            sim.simulation.set_ship_state(state);
            sim.simulation.set_attitude(attitude);
            sim.simulation
                .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
        }
        SpawnSituation::Landing | SpawnSituation::FinalApproach | SpawnSituation::Cruise => {
            // Terrain is resident (we just crashed on it), so this resolves on
            // the first try; the parking-orbit fallback is belt-and-braces for a
            // somehow-missing height source.
            let (state, attitude) = compute_descent_state(situation, sim, height_sources)
                .unwrap_or_else(|| {
                    warn!("respawn: terrain not resident for {situation:?}; using orbit");
                    orbit_respawn_state(sim, homeworld)
                });
            sim.simulation.set_ship_state(state);
            sim.simulation.set_attitude(attitude);
            sim.simulation
                .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
        }
        SpawnSituation::Runway | SpawnSituation::RunwayApproach => {
            warn!("respawn: runway scenarios are one-shot at startup; using orbit");
            let (state, attitude) = orbit_respawn_state(sim, homeworld);
            sim.simulation.set_ship_state(state);
            sim.simulation.set_attitude(attitude);
            sim.simulation
                .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
        }
    }

    info!("respawned into {situation:?}");
}

/// Despawn the active bubble's craft body (and any attached terrain patch) and
/// clear the slot, so the on-demand spawner makes a fresh body next frame.
/// Shared with the editor's Launch relaunch ([`crate::relaunch`]).
pub(crate) fn clear_bubble(commands: &mut Commands, active: &mut ActiveLocalBubble) {
    let Some(bubble) = active.bubble.take() else {
        return;
    };
    commands.entity(bubble.craft_entity).despawn();
    if let Some(terrain_entity) = bubble.terrain_entity {
        commands.entity(terrain_entity).despawn();
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
        With<ScenarioButton>,
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
