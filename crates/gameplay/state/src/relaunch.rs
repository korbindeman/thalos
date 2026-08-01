//! Cross-feature launch/relaunch request slots: the shipyard editor (and the
//! base editor's launch flow) write them; the runtime's `relaunch` /
//! `base_editor::launch_select` systems consume them. Requests through the
//! blackboard are how feature crates talk without depending on each other.

use bevy::prelude::*;
use thalos_shipyard::ShipBlueprint;

use crate::scenario::SpawnSituation;

/// A pending relaunch: the design to fly and the scenario to drop it into.
pub struct RelaunchSpec {
    pub blueprint: ShipBlueprint,
    /// Scenario placement. Today: [`SpawnSituation::Cruise`] for aircraft
    /// (flown airborne over land) or [`SpawnSituation::ShipOrbit`] for
    /// everything else. Other situations fall back to orbit.
    pub situation: SpawnSituation,
}

/// Editor → relaunch request slot. The shipyard editor's Launch button writes
/// a [`RelaunchSpec`] here; the runtime's `begin_relaunch` consumes it.
///
/// **Sole writer:** `thalos_shipyard_editor`'s top bar (`handle_actions`).
#[derive(Resource, Default)]
pub struct RelaunchRequest(pub Option<RelaunchSpec>);

/// Editor → launch-select flow arming slot. Consumed by the runtime's
/// `begin_launch_flow`.
///
/// **Sole writer:** `thalos_shipyard_editor`'s top bar (the LAUNCH button).
#[derive(Resource, Default)]
pub struct SpaceportLaunchRequest {
    pub arm: bool,
}
