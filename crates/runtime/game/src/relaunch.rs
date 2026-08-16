//! Relaunch the flight craft from an arbitrary blueprint — the editor's
//! **Launch** path.
//!
//! Startup builds the flight ship once from a hardcoded scenario blueprint
//! (`ship_view::spawn_player_ship`). The in-game shipyard editor lets you
//! design a *different* craft; Launch swaps the flying vessel for it without
//! relaunching the process.
//!
//! This extends the destruction-respawn machinery
//! ([`crate::scenario_menu`]) — which repairs the *same* craft in place — to
//! the harder case of replacing it: the blueprint, and therefore the part
//! tree, visuals, `ShipParameters`, aero config, staging plan, and Avian
//! bubble, all change. The flow is two-phase so the rebuild never races the
//! teardown:
//!
//! 1. [`begin_relaunch`] consumes a [`RelaunchRequest`]: despawn the old
//!    `PlayerShip` part tree + map billboard, tear down the Avian bubble,
//!    reset the sim to a clean live craft, and place it into the chosen
//!    scenario (orbit, or cruise for an aircraft). The blueprint is parked in
//!    [`RelaunchInFlight`].
//! 2. [`finish_relaunch`] waits until the old craft has actually despawned
//!    (commands are deferred), then builds the new craft from the parked
//!    blueprint via [`crate::ship_view::build_player_ship`]. The fresh
//!    `PlayerShip` has no `StagingPlan` and the bubble slot is empty, so the
//!    existing `build_staging_plan` / `spawn_player_avian_body` systems
//!    rebuild both on the following frames.
//!
//! The launched craft is a plain blueprint spawn — its parts carry **no**
//! [`crate::shipyard_editor::core::EditorPart`] marker, so they enter the flight
//! aggregations the editor's persistent build is filtered out of. The
//! editor's build is left untouched (hidden), so the design survives flying.

use bevy::prelude::*;
use thalos_game_state::{ActiveCraft, ActiveCraftMut, CraftRoot};

use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::{ActiveLocalBubble, HeightSourceRegistry};

use crate::SimStage;
use crate::maneuver::{ManeuverPlan, SelectedNode};
use crate::rendering::SimulationState;
use crate::scenario_menu::clear_bubble;
use crate::spawn::{
    Homeworld, SpawnSituation, coast_placement, compute_descent_state, orbit_respawn_state,
    place_craft, polar_orbit_respawn_state,
};
use crate::view::ViewMode;

// Moved to `thalos_game_state::relaunch` (Phase 5b); `begin_relaunch`
// (the consumer) stays below.
pub use thalos_game_state::relaunch::{RelaunchRequest, RelaunchSpec};

/// Internal hand-off between the two relaunch phases: holds the blueprint
/// after teardown until the old craft has despawned and the new one can build.
/// (`pub(crate)` only so [`relaunch_idle`] can appear in run conditions; no
/// other module touches it.)
#[derive(Resource, Default)]
pub(crate) struct RelaunchInFlight(Option<PendingRelaunch>);

struct PendingRelaunch {
    spec: RelaunchSpec,
    outgoing_root: Option<Entity>,
}

/// Run condition: no relaunch teardown/rebuild is in flight. Systems that
/// measure or place the player craft (the deferred runway placement) gate on
/// this so they never act on the outgoing craft during the swap window.
pub(crate) fn relaunch_idle(
    request: Res<RelaunchRequest>,
    in_flight: Res<RelaunchInFlight>,
) -> bool {
    request.0.is_none() && in_flight.0.is_none()
}

pub struct RelaunchPlugin;

impl Plugin for RelaunchPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RelaunchRequest>()
            .init_resource::<RelaunchInFlight>()
            .add_systems(
                Update,
                (begin_relaunch, finish_relaunch)
                    .chain()
                    // Before the physics chain so the cleared bubble + fresh
                    // canonical state are coherent when it runs this frame.
                    .before(SimStage::Physics),
            );
    }
}

/// Phase 1: tear down the old flight craft and seat the sim into the chosen
/// scenario. Mirrors [`crate::scenario_menu`]'s respawn reset, plus the
/// despawn of the `PlayerShip` tree (the respawn path keeps the same craft;
/// we are replacing it).
#[allow(clippy::too_many_arguments)]
fn begin_relaunch(
    mut commands: Commands,
    mut request: ResMut<RelaunchRequest>,
    mut in_flight: ResMut<RelaunchInFlight>,
    mut sim: ResMut<SimulationState>,
    mut active: ResMut<ActiveLocalBubble>,
    height_sources: Res<HeightSourceRegistry>,
    homeworld: Res<Homeworld>,
    active_craft: Res<ActiveCraft>,
    mut plan: ActiveCraftMut<ManeuverPlan>,
    mut selected: ResMut<SelectedNode>,
) {
    let Some(spec) = request.0.take() else {
        return;
    };

    // Drop only the selected craft root. Other craft roots and their markers
    // remain intact; marker reconciliation removes this root's billboard.
    let outgoing_root = active_craft.0;
    // Tear down the wreck/old bubble so `spawn_player_avian_body` builds a fresh
    // one for the new craft once it exists.
    let bubble_root = clear_bubble(&mut commands, &mut active);
    if let Some(entity) = outgoing_root
        && Some(entity) != bubble_root
    {
        commands.entity(entity).despawn();
    }

    // Fresh-craft reset, mirroring `scenario_menu::respawn_into`: clear any
    // structural-failure flag, drop to 1×, discard the old flight plan.
    sim.simulation.repair();
    sim.simulation.warp.reset();
    if let Some(mut plan) = plan.get_mut()
        && !plan.nodes.is_empty()
    {
        plan.nodes.clear();
        plan.dirty = true;
    }
    selected.id = None;
    sim.simulation.set_vessel_kind(VesselKind::Ship);

    // Place the craft into the fixture situation. This is the live-world
    // adapter used for every ship fixture, so changing scenarios after EVA or
    // after a different blueprint converges with a cold fixture boot.
    let (state, attitude) = match spec.situation {
        SpawnSituation::PolarOrbit => polar_orbit_respawn_state(&sim, homeworld.0),
        situation if situation.is_descent() => {
            compute_descent_state(situation, &sim, &height_sources).unwrap_or_else(|| {
                warn!("relaunch: terrain not resident for {situation:?}; using orbit");
                orbit_respawn_state(&sim, homeworld.0)
            })
        }
        _ => orbit_respawn_state(&sim, homeworld.0),
    };
    // Bubble already torn down above; seat the placed pose. (`spawn::place_craft`
    // is the shared canonical-state core.)
    place_craft(&mut sim, coast_placement(state, attitude), None);

    in_flight.0 = Some(PendingRelaunch {
        spec,
        outgoing_root,
    });
}

/// Phase 2: once the old `PlayerShip` has actually despawned, build the new
/// craft from the parked blueprint. Guarding on an empty `PlayerShip` query
/// makes the build race-free regardless of when the phase-1 despawns flush.
fn finish_relaunch(
    mut commands: Commands,
    mut in_flight: ResMut<RelaunchInFlight>,
    view: Res<ViewMode>,
    mut sim: ResMut<SimulationState>,
    catalog: Res<thalos_shipyard::PartCatalog>,
    craft_roots: Query<(), With<CraftRoot>>,
) {
    let Some(pending) = in_flight.0.as_ref() else {
        return;
    };
    // Wait only for the outgoing root. Unrelated roots must not block a launch.
    if pending
        .outgoing_root
        .is_some_and(|entity| craft_roots.get(entity).is_ok())
    {
        return;
    }
    let spec = in_flight.0.take().unwrap().spec;
    crate::ship_view::build_player_ship(&mut commands, &view, &spec.blueprint, &mut sim, &catalog);
    info!(
        "relaunch: built '{}' ({} parts) into {:?}",
        spec.blueprint.name,
        spec.blueprint.parts.len(),
        spec.situation,
    );
}
