use glam::{DMat3, DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::types::{BodyId, StateVector};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Epoch(pub f64);

impl Epoch {
    pub const ZERO: Self = Self(0.0);
}

impl Default for Epoch {
    fn default() -> Self {
        Self::ZERO
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DurationS(pub f64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorldPreset {
    Classic,
    Realistic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GravityBackendConfig {
    PatchedConics,
    NBodyEphemeris,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderPolicyRef {
    pub id: String,
    pub version: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldPhysicsConfig {
    pub preset: WorldPreset,
    pub gravity_backend: GravityBackendConfig,
    pub provider_policy: ProviderPolicyRef,
    pub epoch: Epoch,
}

impl WorldPhysicsConfig {
    pub fn classic() -> Self {
        Self {
            preset: WorldPreset::Classic,
            gravity_backend: GravityBackendConfig::PatchedConics,
            provider_policy: ProviderPolicyRef {
                id: "ClassicPolicyV1".to_string(),
                version: 1,
            },
            epoch: Epoch::ZERO,
        }
    }

    pub fn validate_supported(&self) -> Result<(), UnsupportedWorldPreset> {
        match self.preset {
            WorldPreset::Classic => Ok(()),
            WorldPreset::Realistic => Err(UnsupportedWorldPreset {
                preset: self.preset,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedWorldPreset {
    pub preset: WorldPreset,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TimeMode {
    Realtime,
    PhysicsWarp,
    RailsWarp,
    PlanningPreview,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SimClock {
    pub current: Epoch,
    pub scale: f64,
    pub paused: bool,
    pub mode: TimeMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TranslationalState {
    pub position: DVec3,
    pub velocity: DVec3,
}

impl From<StateVector> for TranslationalState {
    fn from(value: StateVector) -> Self {
        Self {
            position: value.position,
            velocity: value.velocity,
        }
    }
}

impl From<TranslationalState> for StateVector {
    fn from(value: TranslationalState) -> Self {
        Self {
            position: value.position,
            velocity: value.velocity,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MassState {
    pub wet_mass_kg: f64,
    pub dry_mass_kg: f64,
    pub inertia_body_kg_m2: DMat3,
    pub center_of_mass_body_m: DVec3,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ResourceState;

pub type CraftId = u64;
pub type TrajectoryId = u64;
pub type WarpIntegratorId = u64;
pub type LocalBubbleId = u64;
pub type AssemblyId = u64;
pub type DockingPortId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityRef(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyFixedPose {
    pub position_body_m: DVec3,
    pub orientation_body: DQuat,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AuthorityMode {
    OnRails {
        trajectory: TrajectoryId,
    },
    WarpIntegrated {
        integrator: WarpIntegratorId,
    },
    LocalRigidBody {
        bubble: LocalBubbleId,
        root_entity: EntityRef,
    },
    BodyFixed {
        body: BodyId,
        pose: BodyFixedPose,
    },
    Docked {
        assembly: AssemblyId,
        port: DockingPortId,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CraftState {
    pub id: CraftId,
    pub epoch: Epoch,
    pub translation: TranslationalState,
    pub attitude: crate::types::AttitudeState,
    pub mass: MassState,
    pub resources: ResourceState,
    pub authority: AuthorityMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthorityChanged {
    pub craft: CraftId,
    pub from: AuthorityMode,
    pub to: AuthorityMode,
    pub epoch: Epoch,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CraftAuthorityBook {
    pub craft: CraftId,
    pub mode: AuthorityMode,
    pub log: Vec<AuthorityChanged>,
}

impl CraftAuthorityBook {
    pub fn new(craft: CraftId, mode: AuthorityMode) -> Self {
        Self {
            craft,
            mode,
            log: Vec::new(),
        }
    }

    pub fn transition_to(&mut self, epoch: Epoch, next: AuthorityMode) {
        if self.mode == next {
            return;
        }
        let from = self.mode;
        self.log.push(AuthorityChanged {
            craft: self.craft,
            from,
            to: next,
            epoch,
        });
        self.mode = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translational_state_round_trips_state_vector() {
        let state = StateVector {
            position: DVec3::new(10.0, -20.0, 30.0),
            velocity: DVec3::new(1.0, 2.0, -3.0),
        };

        let canonical = TranslationalState::from(state);
        let round_trip = StateVector::from(canonical);

        assert_eq!(round_trip, state);
    }

    #[test]
    fn authority_transition_bookkeeping_logs_only_real_changes() {
        let mut book = CraftAuthorityBook::new(42, AuthorityMode::OnRails { trajectory: 7 });

        book.transition_to(Epoch(10.0), AuthorityMode::OnRails { trajectory: 7 });
        assert!(book.log.is_empty());

        book.transition_to(Epoch(11.0), AuthorityMode::WarpIntegrated { integrator: 3 });

        assert_eq!(book.mode, AuthorityMode::WarpIntegrated { integrator: 3 });
        assert_eq!(book.log.len(), 1);
        assert_eq!(book.log[0].craft, 42);
        assert_eq!(book.log[0].from, AuthorityMode::OnRails { trajectory: 7 });
        assert_eq!(
            book.log[0].to,
            AuthorityMode::WarpIntegrated { integrator: 3 }
        );
    }

    #[test]
    fn realistic_preset_is_defined_but_not_supported_yet() {
        let mut config = WorldPhysicsConfig::classic();
        config.preset = WorldPreset::Realistic;
        config.gravity_backend = GravityBackendConfig::NBodyEphemeris;

        let err = config.validate_supported().unwrap_err();

        assert_eq!(err.preset, WorldPreset::Realistic);
    }
}
