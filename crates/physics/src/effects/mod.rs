//! Effect system: independent acceleration contributions that sum into the
//! total force on the ship at a given state and time.
//!
//! Gravity is distinguished because its analysis (dominant body, perturbation
//! ratio) is consumed by the integrator mode switch and rendering metadata —
//! it lives in its own slot on the registry. All other contributions (thrust,
//! drag, SRP, modder extensions) implement [`Effect`] and live in
//! [`EffectRegistry::effects`].
//!
//! Each effect is a pure function of `(state, time, body_states)` so trajectory
//! prediction and live stepping share the same code path and produce identical
//! results given identical inputs.

pub mod gravity;
pub mod thrust;

pub use gravity::{GravityModel, GravityResult, NewtonianGravity};
pub use thrust::ManeuverThrustEffect;

use std::sync::Arc;

use glam::DVec3;

use crate::types::BodyStates;

/// Runtime inputs to an [`Effect`] at a single integrator substep.
///
/// Lifetimes borrow the body-state buffer and any future shared data — the
/// context is constructed fresh each call, never stored.
pub struct EffectContext<'a> {
    pub position: DVec3,
    pub velocity: DVec3,
    pub time: f64,
    pub bodies: &'a BodyStates,
}

/// A non-gravity acceleration source. Implementors are shared via `Arc` so the
/// registry can be cloned cheaply for prediction workers.
pub trait Effect: Send + Sync {
    /// Acceleration contribution in world frame, m/s². Must be a pure function
    /// of `ctx` — no hidden state mutation, no reliance on wall-clock time.
    fn accelerate(&self, ctx: &EffectContext) -> DVec3;
}

/// Full set of forces acting on the ship. Gravity is always present; other
/// effects are an open list.
#[derive(Clone)]
pub struct EffectRegistry {
    pub gravity: Arc<dyn GravityModel>,
    pub effects: Vec<Arc<dyn Effect>>,
}

impl EffectRegistry {
    /// Registry with Newtonian gravity and no other effects.
    pub fn newtonian() -> Self {
        Self {
            gravity: Arc::new(NewtonianGravity),
            effects: Vec::new(),
        }
    }

    /// Newtonian gravity + the given effects.
    pub fn with_effects(effects: Vec<Arc<dyn Effect>>) -> Self {
        Self {
            gravity: Arc::new(NewtonianGravity),
            effects,
        }
    }

    pub fn push(&mut self, effect: Arc<dyn Effect>) {
        self.effects.push(effect);
    }

    pub fn clear_effects(&mut self) {
        self.effects.clear();
    }
}

impl Default for EffectRegistry {
    fn default() -> Self {
        Self::newtonian()
    }
}

/// Compute the total acceleration on the ship at the given context.
///
/// Returns `(total, gravity_result)` so callers (e.g. the integrator) can read
/// gravity-specific analysis (dominant body, perturbation ratio) without a
/// redundant gravity pass.
#[inline]
pub fn compute_total_acceleration(
    ctx: &EffectContext,
    registry: &EffectRegistry,
) -> (DVec3, GravityResult) {
    let gravity = registry.gravity.compute(ctx.position, ctx.bodies);
    let mut total = gravity.acceleration;
    for effect in &registry.effects {
        total += effect.accelerate(ctx);
    }
    (total, gravity)
}
