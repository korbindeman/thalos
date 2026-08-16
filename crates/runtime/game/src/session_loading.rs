//! Canonical runtime-session load coordinator.
//!
//! Menus and future persistence adapters submit [`SessionLoadRequest`]. This
//! module is the sole runtime consumer: it validates source assets before
//! mutation, arms the existing projection workers, and publishes the new
//! generation only when `Running` is entered.

use bevy::prelude::*;
use thalos_game_state::{ActiveCraft, ActiveCraftMut};
use thalos_physics_local::{ActiveLocalBubble, HeightSourceRegistry};

use crate::content::ContentRoot;
use crate::game_context::{ContextHistory, GameContext, InitialContext};
use crate::loading::{
    AppState, LoadDestination, LoadingTracker, StepDesc, WorldState, step, steps_for,
    world_load_steps,
};
use crate::maneuver::{ManeuverPlan, SelectedNode};
use crate::relaunch::{RelaunchRequest, RelaunchSpec};
use crate::rendering::SimulationState;
use crate::runway::RunwayPlacement;
use crate::scenario_menu::respawn_into;
use crate::space_center::HubSpaceportBuild;
use crate::spawn::{DescentPlacement, Homeworld, SpawnSituation};
use crate::surface_settle::SurfaceSettle;

pub use thalos_game_state::session::{
    ActiveSession, PendingSessionLoad, ScenarioFixture, SessionGeneration, SessionLoadRequest,
    SessionSource,
};

/// Load accepted and awaiting its `Loading/MainMenu → Running` projection
/// reveal. Kept separate from [`ActiveSession`] so a failed validation leaves
/// the former session authoritative.
#[derive(Resource, Default)]
struct SessionLoadInFlight(Option<PendingSessionLoad>);

pub struct SessionLoadingPlugin;

impl Plugin for SessionLoadingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveSession>()
            .init_resource::<SessionLoadRequest>()
            .init_resource::<SessionLoadInFlight>()
            .add_systems(
                Update,
                apply_session_load_request
                    .after(crate::main_menu::MainMenuActionSet)
                    .run_if(crate::relaunch::relaunch_idle),
            )
            .add_systems(
                Update,
                complete_live_session_projection
                    .run_if(in_state(AppState::Loading))
                    .run_if(crate::relaunch::relaunch_idle),
            )
            .add_systems(
                Update,
                publish_loaded_session
                    .run_if(in_state(AppState::Running))
                    .run_if(crate::relaunch::relaunch_idle),
            );
    }
}

/// Validate and arm one session projection. Existing projection workers remain
/// implementation adapters during the complete-snapshot migration; only this
/// system decides which combination constitutes a load.
#[allow(clippy::too_many_arguments)]
fn apply_session_load_request(
    mut request: ResMut<SessionLoadRequest>,
    mut in_flight: ResMut<SessionLoadInFlight>,
    mut commands: Commands,
    mut sim: ResMut<SimulationState>,
    mut situation: ResMut<SpawnSituation>,
    content: Res<ContentRoot>,
    respawn: (
        ResMut<ActiveLocalBubble>,
        Res<HeightSourceRegistry>,
        ActiveCraftMut<ManeuverPlan>,
        ResMut<SelectedNode>,
        Res<Homeworld>,
        Res<ActiveCraft>,
    ),
    load: (
        ResMut<LoadingTracker>,
        ResMut<LoadDestination>,
        ResMut<SurfaceSettle>,
        ResMut<RunwayPlacement>,
        ResMut<RelaunchRequest>,
        ResMut<DescentPlacement>,
    ),
    world: (Res<State<WorldState>>, ResMut<NextState<WorldState>>),
    mut next_state: ResMut<NextState<AppState>>,
    session_ui: (
        ResMut<InitialContext>,
        ResMut<ContextHistory>,
        ResMut<HubSpaceportBuild>,
    ),
) {
    let Some(pending) = request.take() else {
        return;
    };
    let plan = pending.source.plan();
    let world_absent = *world.0.get() == WorldState::Absent;

    // Validate the craft fixture before changing any session state. Cold boot
    // still lets `spawn_player_ship` build it, while a live projection consumes
    // this same parsed blueprint through `RelaunchRequest`.
    let blueprint = if plan.entry_context != GameContext::SpaceCenter
        && plan.situation != SpawnSituation::Eva
    {
        let Some(blueprint) = crate::ship_view::load_blueprint_from_path(
            &content,
            plan.situation.ship_blueprint_path(),
        ) else {
            error!(
                "session load generation {}: fixture craft failed validation; old session retained",
                pending.generation.0
            );
            return;
        };
        Some(blueprint)
    } else {
        None
    };

    // Fixtures are absolute situations, not teleports within the former
    // session. Untimed fixtures start at epoch zero; spaceport materialization
    // installs its authored morning epoch before surface placement.
    sim.simulation
        .set_sim_time(crate::runway::canonical_epoch_s(plan.situation).unwrap_or(0.0));

    let (mut active, height_sources, mut maneuver, mut selected, homeworld, active_craft) = respawn;
    let (mut tracker, mut destination, mut settle, mut runway, mut relaunch, mut descent) = load;
    let (_world_state, mut next_world) = world;
    let (mut initial_context, mut history, mut hub_build) = session_ui;

    // Clear every transient request owned by the former projection before
    // arming the new one. They are not campaign state and may not leak across a
    // generation boundary.
    runway.pending = false;
    descent.pending = false;
    hub_build.pending = false;
    relaunch.0 = None;
    settle.arm(false, true);
    history.0.clear();
    initial_context.0 = Some(plan.entry_context);
    *situation = plan.situation;
    in_flight.0 = Some(pending);

    // New Campaign and the hub fixture have the same materialization; only the
    // campaign adapter's durability differs. Always pass through Loading so
    // stable base identity is reconciled even when the `RunwaySite` cache is
    // missing. An existing base completes this pass without another spawn.
    if plan.entry_context == GameContext::SpaceCenter {
        hub_build.pending = true;
        let mut steps = Vec::new();
        if world_absent {
            steps.extend(world_load_steps());
            next_world.set(WorldState::Live);
        }
        steps.push(StepDesc::new(step::PLACEMENT, "Building spaceport", 1.0));
        tracker.begin(steps);
        destination.0 = AppState::Running;
        next_state.set(AppState::Loading);
        info!(
            "session load generation {}: {:?}",
            pending.generation.0, pending.source
        );
        return;
    }

    if world_absent {
        match plan.situation {
            SpawnSituation::ShipOrbit | SpawnSituation::PolarOrbit | SpawnSituation::Eva => {
                respawn_into(
                    plan.situation,
                    &mut commands,
                    &mut sim,
                    &mut active,
                    &height_sources,
                    active_craft.0,
                    maneuver.get_mut().as_deref_mut(),
                    &mut selected,
                    homeworld.0,
                );
            }
            SpawnSituation::Landing | SpawnSituation::FinalApproach | SpawnSituation::Cruise => {
                descent.pending = true;
            }
            SpawnSituation::Runway | SpawnSituation::RunwayApproach | SpawnSituation::Launch => {
                runway.pending = true;
                settle.arm(
                    matches!(
                        plan.situation,
                        SpawnSituation::Runway | SpawnSituation::Launch
                    ),
                    false,
                );
            }
        }
        tracker.begin(steps_for(plan.situation, true));
        destination.0 = AppState::Running;
        next_world.set(WorldState::Live);
        next_state.set(AppState::Loading);
    } else if plan.situation == SpawnSituation::Eva {
        respawn_into(
            plan.situation,
            &mut commands,
            &mut sim,
            &mut active,
            &height_sources,
            active_craft.0,
            maneuver.get_mut().as_deref_mut(),
            &mut selected,
            homeworld.0,
        );
        tracker.begin([StepDesc::new(step::SESSION, "Projecting session", 1.0)]);
        destination.0 = AppState::Running;
        next_state.set(AppState::Loading);
    } else {
        // Every ship fixture replaces the craft, even when the outgoing craft
        // happens to use the same blueprint. This makes EVA→orbit and all other
        // live starts converge with a cold fixture boot.
        relaunch.0 = Some(RelaunchSpec {
            blueprint: blueprint.expect("non-EVA fixture was validated above"),
            situation: plan.situation,
        });
        if plan.situation.is_spaceport() {
            runway.pending = true;
            settle.arm(
                matches!(
                    plan.situation,
                    SpawnSituation::Runway | SpawnSituation::Launch
                ),
                false,
            );
            tracker.begin(steps_for(plan.situation, false));
            destination.0 = AppState::Running;
            next_state.set(AppState::Loading);
        } else {
            tracker.begin([StepDesc::new(step::SESSION, "Projecting session", 1.0)]);
            destination.0 = AppState::Running;
            next_state.set(AppState::Loading);
        }
    }

    info!(
        "session load generation {}: {:?}",
        pending.generation.0, pending.source
    );
}

/// Release the load gate only after the live-world craft adapter has finished
/// replacing the outgoing projection. Spaceport loads use their stricter
/// placement/settle steps instead; cold boots use world-load completion.
fn complete_live_session_projection(mut tracker: ResMut<LoadingTracker>) {
    if tracker.has_step(step::SESSION) {
        tracker.complete(step::SESSION);
    }
}

fn publish_loaded_session(
    mut in_flight: ResMut<SessionLoadInFlight>,
    mut active: ResMut<ActiveSession>,
) {
    let Some(loaded) = in_flight.0.take() else {
        return;
    };
    *active = ActiveSession::projected(loaded.generation, loaded.source);
    info!(
        "session generation {} projected from {:?}",
        loaded.generation.0, loaded.source
    );
}
