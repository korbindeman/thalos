//! Campaign-session loading vocabulary.
//!
//! A process may keep renderer/content services alive while replacing the
//! playable session many times. Entry points submit a [`SessionLoadRequest`];
//! the runtime is the sole consumer and projects the resulting session. This
//! prevents menus, developer fixtures, and future persistence adapters from
//! growing independent spawn paths.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::context::GameContext;
use crate::scenario::SpawnSituation;

/// Current in-memory session schema. Disk revisions and bundled fixtures must
/// migrate to this version before they can be committed by the loader.
pub const SESSION_SCHEMA_VERSION: u32 = 1;

/// Monotonically increasing identity for one projection of campaign state into
/// the running process. Async work stamped with an older generation must not
/// write into the current session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SessionGeneration(pub u64);

/// Whether revisions created while this session is active belong to a durable
/// player campaign or to a discard-by-default developer fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionDurability {
    Campaign,
    Ephemeral,
}

/// Bundled development situations. These are fixture identities, not gameplay
/// modes: their plan enters the same session loader as [`SessionSource::NewCampaign`]
/// and, later, disk-backed campaign revisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScenarioFixture {
    SpaceCenter,
    Shipyard,
    Flight(SpawnSituation),
}

/// Source selected by an entry point. Persistence adds a revision source here;
/// consumers continue to receive the same validated session plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionSource {
    NewCampaign,
    Fixture(ScenarioFixture),
}

/// Compatibility materialization plan for the current runtime. As singleton
/// spawn resources move into complete snapshot records, this type shrinks to
/// operator-entry policy; gameplay systems must never branch on fixture origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionPlan {
    pub situation: SpawnSituation,
    pub entry_context: GameContext,
    pub requires_space_center: bool,
    pub durability: SessionDurability,
}

impl SessionSource {
    pub fn plan(self) -> SessionPlan {
        match self {
            Self::NewCampaign => SessionPlan {
                // The current simulation kernel requires an active vessel. It
                // remains an internal orbit placeholder while the campaign
                // opens at a craft-less space center; the complete snapshot
                // slice removes that compatibility detail.
                situation: SpawnSituation::ShipOrbit,
                entry_context: GameContext::SpaceCenter,
                requires_space_center: true,
                durability: SessionDurability::Campaign,
            },
            Self::Fixture(ScenarioFixture::SpaceCenter) => SessionPlan {
                situation: SpawnSituation::ShipOrbit,
                entry_context: GameContext::SpaceCenter,
                requires_space_center: true,
                durability: SessionDurability::Ephemeral,
            },
            Self::Fixture(ScenarioFixture::Shipyard) => SessionPlan {
                situation: SpawnSituation::ShipOrbit,
                entry_context: GameContext::Vab,
                requires_space_center: false,
                durability: SessionDurability::Ephemeral,
            },
            Self::Fixture(ScenarioFixture::Flight(situation)) => SessionPlan {
                situation,
                entry_context: GameContext::Flight,
                requires_space_center: situation.is_spaceport(),
                durability: SessionDurability::Ephemeral,
            },
        }
    }
}

/// One queued load, assigned an identity before any mutation begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingSessionLoad {
    pub generation: SessionGeneration,
    pub source: SessionSource,
}

/// Entry-point → runtime handoff. The runtime is the sole consumer. A newer
/// request supersedes an older request that has not started; generations are
/// never reused.
#[derive(Resource, Debug)]
pub struct SessionLoadRequest {
    next_generation: u64,
    pending: Option<PendingSessionLoad>,
}

impl Default for SessionLoadRequest {
    fn default() -> Self {
        Self::after(SessionGeneration(0))
    }
}

impl SessionLoadRequest {
    pub fn after(active: SessionGeneration) -> Self {
        Self {
            next_generation: active
                .0
                .checked_add(1)
                .expect("session generation space exhausted"),
            pending: None,
        }
    }

    pub fn request(&mut self, source: SessionSource) -> SessionGeneration {
        let generation = SessionGeneration(self.next_generation);
        self.next_generation = self
            .next_generation
            .checked_add(1)
            .expect("session generation space exhausted");
        self.pending = Some(PendingSessionLoad { generation, source });
        generation
    }

    pub fn take(&mut self) -> Option<PendingSessionLoad> {
        self.pending.take()
    }

    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }
}

/// The source currently projected by the process. This is runtime identity and
/// observability metadata, not the campaign's revision graph.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveSession {
    pub schema_version: u32,
    pub generation: SessionGeneration,
    pub source: Option<SessionSource>,
}

impl Default for ActiveSession {
    fn default() -> Self {
        Self {
            schema_version: SESSION_SCHEMA_VERSION,
            generation: SessionGeneration(0),
            source: None,
        }
    }
}

impl ActiveSession {
    pub fn projected(generation: SessionGeneration, source: SessionSource) -> Self {
        Self {
            schema_version: SESSION_SCHEMA_VERSION,
            generation,
            source: Some(source),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_campaign_and_space_center_fixture_share_materialization() {
        let campaign = SessionSource::NewCampaign.plan();
        let fixture = SessionSource::Fixture(ScenarioFixture::SpaceCenter).plan();

        assert_eq!(campaign.situation, fixture.situation);
        assert_eq!(campaign.entry_context, fixture.entry_context);
        assert_eq!(
            campaign.requires_space_center,
            fixture.requires_space_center
        );
        assert_eq!(campaign.durability, SessionDurability::Campaign);
        assert_eq!(fixture.durability, SessionDurability::Ephemeral);
    }

    #[test]
    fn every_flight_fixture_uses_its_situation_without_origin_policy() {
        let situations = [
            SpawnSituation::ShipOrbit,
            SpawnSituation::PolarOrbit,
            SpawnSituation::Eva,
            SpawnSituation::Landing,
            SpawnSituation::FinalApproach,
            SpawnSituation::Runway,
            SpawnSituation::RunwayApproach,
            SpawnSituation::Launch,
            SpawnSituation::Cruise,
        ];

        for situation in situations {
            let plan = SessionSource::Fixture(ScenarioFixture::Flight(situation)).plan();
            assert_eq!(plan.situation, situation);
            assert_eq!(plan.entry_context, GameContext::Flight);
            assert_eq!(plan.requires_space_center, situation.is_spaceport());
            assert_eq!(plan.durability, SessionDurability::Ephemeral);
        }
    }

    #[test]
    fn generations_are_monotonic_when_pending_request_is_replaced() {
        let mut requests = SessionLoadRequest::after(SessionGeneration(41));
        assert_eq!(
            requests.request(SessionSource::NewCampaign),
            SessionGeneration(42)
        );
        assert_eq!(
            requests.request(SessionSource::Fixture(ScenarioFixture::Flight(
                SpawnSituation::Eva
            ))),
            SessionGeneration(43)
        );
        assert_eq!(requests.take().unwrap().generation, SessionGeneration(43));
        assert!(!requests.is_pending());
    }
}
