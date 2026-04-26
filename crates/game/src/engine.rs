//! Per-engine runtime thrust + temporary visual feedback.
//!
//! Two systems live here:
//!
//! 1. [`update_engine_thrust`] writes the gated effective throttle into
//!    each [`Engine`]'s [`EngineThrust`] component as `engine.thrust *
//!    throttle.effective` newtons. v1 has no per-engine throttling, so
//!    every engine on the ship reads the same throttle. This is the
//!    plumbing future visual effects (particles, plumes, light) will
//!    consume — anyone wanting to know "how hard is this engine
//!    firing" reads `EngineThrust.current_n` instead of rederiving it
//!    from `ThrottleState` + ship config.
//!
//! 2. [`update_engine_tint`] is the *temporary* visual: it lerps each
//!    engine's [`StandardMaterial`] base color from neutral gray to
//!    full red proportional to `current_n / engine.thrust`. Replace
//!    this fn with a real plume/particle effect when ready — the
//!    plumbing in (1) stays.

use bevy::prelude::*;
use thalos_shipyard::{Engine, EngineThrust};

use crate::SimStage;
use crate::fuel::ThrottleState;
use crate::ship_view::PartVisual;

/// Default engine mesh tint when idle. Matches the placeholder
/// `StandardMaterial` constructed in
/// [`crate::ship_view::rebuild_ship_visuals`] for parts without their
/// own procedural shader, so the idle look stays consistent with what
/// the ship view spawns.
const ENGINE_IDLE_TINT: LinearRgba = LinearRgba::new(0.6, 0.62, 0.65, 1.0);

/// Full-throttle tint. Pure red so the gradient is unambiguous; the
/// future replacement (plume, emission glow, particle stream) will
/// reuse the same `EngineThrust.current_n` signal but render
/// something less placeholder-shaped.
const ENGINE_HOT_TINT: LinearRgba = LinearRgba::new(1.0, 0.0, 0.0, 1.0);

pub struct EnginePlugin;

impl Plugin for EnginePlugin {
    fn build(&self, app: &mut App) {
        // Run after the fuel system has gated the throttle — that's
        // what we read — and inside the same physics-stage chain so
        // the engine state is fresh by the time anything renders.
        app.add_systems(
            Update,
            (update_engine_thrust, update_engine_tint)
                .chain()
                .in_set(SimStage::Physics)
                .after(crate::bridge::advance_simulation),
        );
    }
}

/// Compute per-engine current thrust from the gated effective throttle
/// and write it back to each engine's [`EngineThrust`] component.
fn update_engine_thrust(
    throttle: Res<ThrottleState>,
    mut engines: Query<(&Engine, &mut EngineThrust)>,
) {
    let throttle_eff = throttle.effective.clamp(0.0, 1.0) as f32;
    for (engine, mut thrust) in engines.iter_mut() {
        thrust.current_n = engine.thrust * throttle_eff;
    }
}

/// Temporary placeholder visual: tint the engine's body mesh from
/// gray to red proportional to thrust. Replace with a real effect.
fn update_engine_tint(
    engines: Query<(&Engine, &EngineThrust, &Children)>,
    visuals: Query<&MeshMaterial3d<StandardMaterial>, With<PartVisual>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (engine, thrust, children) in engines.iter() {
        let max = engine.thrust.max(1.0);
        let ratio = (thrust.current_n / max).clamp(0.0, 1.0);
        let tint = lerp_linear(ENGINE_IDLE_TINT, ENGINE_HOT_TINT, ratio);
        let color = Color::LinearRgba(tint);
        for child in children.iter() {
            let Ok(handle) = visuals.get(child) else {
                continue;
            };
            let Some(mat) = materials.get_mut(&handle.0) else {
                continue;
            };
            // Skip the write when the color is already there so we
            // don't churn `Changed<StandardMaterial>` queries every
            // idle frame.
            if mat.base_color != color {
                mat.base_color = color;
            }
        }
    }
}

fn lerp_linear(a: LinearRgba, b: LinearRgba, t: f32) -> LinearRgba {
    LinearRgba::new(
        a.red + (b.red - a.red) * t,
        a.green + (b.green - a.green) * t,
        a.blue + (b.blue - a.blue) * t,
        a.alpha + (b.alpha - a.alpha) * t,
    )
}
