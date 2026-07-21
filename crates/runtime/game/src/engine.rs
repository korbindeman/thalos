//! Per-engine runtime thrust.
//!
//! [`update_engine_thrust`] writes the gated effective throttle into each
//! enabled [`Engine`]'s [`EngineThrust`] component as
//! `engine.thrust * throttle.effective` newtons. Disabled engines are forced to
//! zero. This is the plumbing the visual effects consume: the plume system
//! (`crate::rendering::plume`) reads `EngineThrust.current_n` rather than
//! rederiving it from `ThrottleState` + ship config, so it stays in sync with
//! whatever gating the bridge applies (fuel-out, auto-burn vs manual,
//! warp-disabled, etc.).

use crate::shipyard_editor::core::EditorPart;
use bevy::prelude::*;
use thalos_shipyard::{Engine, EngineActivation, EngineThrust};

use crate::SimStage;
use crate::fuel::{ActivePropulsion, ThrottleState};

pub struct EnginePlugin;

impl Plugin for EnginePlugin {
    fn build(&self, app: &mut App) {
        // Run after the fuel system has gated the throttle — that's
        // what we read — and inside the same physics-stage chain so
        // the engine state is fresh by the time anything renders.
        app.add_systems(
            Update,
            update_engine_thrust
                .in_set(SimStage::Physics)
                .after(crate::bridge::advance_simulation),
        );
    }
}

/// Compute per-engine current thrust from the gated effective throttle
/// and write it back to each engine's [`EngineThrust`] component.
pub(crate) fn update_engine_thrust(
    throttle: Res<ThrottleState>,
    active: Res<ActivePropulsion>,
    mut engines: Query<
        (
            Entity,
            &Engine,
            Option<&EngineActivation>,
            &mut EngineThrust,
        ),
        Without<EditorPart>,
    >,
) {
    let throttle_eff = throttle.effective.clamp(0.0, 1.0) as f32;
    for (entity, engine, activation, mut thrust) in engines.iter_mut() {
        let enabled = activation.map(|a| a.enabled).unwrap_or(true);
        let active_flow = active.engines.iter().find(|e| e.entity == entity);
        let active = enabled && active_flow.is_some();
        let thrust_scale = active_flow.map(|flow| flow.thrust_scale).unwrap_or(0.0) as f32;
        thrust.current_n = if active {
            engine.thrust * throttle_eff * thrust_scale
        } else {
            0.0
        };
    }
}
