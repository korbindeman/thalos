//! Swappable gravity strategies.
//!
//! [`GravityMode`] picks one named gravity model (today: only patched conics);
//! [`GravityImpls`] is the concrete pair of trait objects the rest of the
//! engine consumes — a [`BodyStateProvider`] for body motion and a
//! [`ShipPropagator`] for ship motion. Construction-time only: the runtime
//! holds the trait objects, and the discriminant is meant to live alongside
//! save data so a savegame can pin the gravity model it was authored under.
//!
//! Adding a new mode is a single `match` arm in [`GravityMode::build`] plus a
//! new variant on the enum. The trait surface stays untouched until the SOI
//! rename in the future N-body PR.

use std::sync::Arc;

use crate::body_state_provider::BodyStateProvider;
use crate::patched_conics::PatchedConics;
use crate::ship_propagator::{KeplerianPropagator, ShipPropagator};
use crate::types::SolarSystemDefinition;

/// Named gravity strategies. Pinned per simulation (and, eventually, per
/// savegame) — not swappable mid-flight.
///
/// `Copy` is intentionally not derived: future variants (e.g. an N-body mode
/// carrying an ephemeris path) will hold owned data, and dropping `Copy` now
/// keeps the audit surface minimal when that variant lands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GravityMode {
    /// Bodies on Keplerian rails, ship under patched-conics SOI propagation.
    PatchedConics,
}

/// Concrete trait objects produced by [`GravityMode::build`]. Owned by the
/// simulation; cloned (`Arc`-share) into the prediction request so live and
/// predicted ship motion route through the same propagator.
#[derive(Clone)]
pub struct GravityImpls {
    pub body_state: Arc<dyn BodyStateProvider>,
    pub ship_propagator: Arc<dyn ShipPropagator>,
}

impl GravityMode {
    pub fn build(&self, system: &SolarSystemDefinition, time_span: f64) -> GravityImpls {
        match self {
            Self::PatchedConics => GravityImpls {
                body_state: Arc::new(PatchedConics::new(system, time_span)),
                ship_propagator: Arc::new(KeplerianPropagator::default()),
            },
        }
    }
}
