//! Engine throttle input + per-frame fuel consumption.
//!
//! The simulation knows nothing about reactant pools — its
//! [`ControlInput::throttle`] is a 0..1 multiplier on `thrust_n / mass`,
//! and it tracks ship mass internally. This module is the bridge
//! between that abstract throttle and the ECS-side per-tank reactant
//! state in [`PartResources`]:
//!
//! 1. Sample keyboard (Shift/Ctrl/Z/X, KSP convention) into
//!    [`ThrottleState::commanded`]. The burn-execution autopilot
//!    (`crates/game/src/autopilot.rs`) writes the same field while
//!    flying a scheduled maneuver — programmatic and manual control
//!    share the throttle surface.
//! 2. Cap the throttle to 0 when any required reactant runs out and
//!    push the gated value into the simulation via
//!    [`Simulation::set_throttle`]. Engine fires only at 1× warp;
//!    every other warp level forces the gated throttle to 0.
//! 3. Reconcile tank pools to whatever mass the simulation actually
//!    burned last frame. The simulation is the **single source of
//!    truth** for ship mass — its propagator integrates burn duration,
//!    SOI clipping, and dry-mass cutoff exactly. We read back the mass
//!    delta and drain tank pools by that amount, split across the
//!    engine's reactant fractions, so the per-tank UI matches the
//!    physics regardless of short burns inside the frame.
//!
//! Mass flow + reactant fractions + dry mass live in [`ShipFuelParams`],
//! populated once at ship spawn from [`ShipStats`]. v1 has no staging,
//! so they never change after spawn.

use std::collections::HashMap;

use bevy::prelude::*;
use thalos_shipyard::{FuelTank, PartResources, Resource};

use crate::SimStage;
use crate::rendering::SimulationState;

/// Engine + ship-mass constants the throttle/drain/mass-push systems
/// need each frame — derived from [`thalos_shipyard::ShipStats`] at
/// spawn.
#[derive(Resource, Debug, Clone, Default)]
pub struct ShipFuelParams {
    /// Sum of every part's `dry_mass`, kg. Combined with the running
    /// tank totals to push the simulation's current `ship_mass_kg` each
    /// frame.
    pub dry_mass_kg: f64,
    /// Total mass flow at full throttle across all engines, kg/s.
    pub mass_flow_kg_per_s: f64,
    /// What fraction of the total mass flow each reactant accounts
    /// for. Sums to 1 when any engine is present; empty when no
    /// engine is on the ship (in which case the drain system is a
    /// no-op).
    pub reactant_fractions: HashMap<Resource, f64>,
}

/// Player throttle state, persisted across frames so Shift/Ctrl can
/// ramp continuously.
///
/// `commanded` is what the player asked for; `effective` is what was
/// actually applied this frame after the fuel-availability cap (the
/// two diverge only when a tank is running dry). HUD reads both.
#[derive(Resource, Debug, Default)]
pub struct ThrottleState {
    pub commanded: f64,
    pub effective: f64,
}

pub struct FuelPlugin;

impl Plugin for FuelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShipFuelParams>()
            .init_resource::<ThrottleState>()
            .add_systems(
                Update,
                (
                    // Sample the keyboard early so the burn-execution
                    // autopilot in `crates/game/src/autopilot.rs` can
                    // run between this and `handle_attitude_controls`,
                    // overriding `ThrottleState::commanded` for the
                    // upcoming gate step. Detached from the
                    // reconcile/gate chain because it has no data
                    // dependency on attitude.
                    handle_throttle_input
                        .in_set(SimStage::Physics)
                        .before(crate::bridge::handle_attitude_controls),
                    // Reconcile first: at this point the simulation's
                    // `ship_mass_kg` reflects the previous frame's step,
                    // so any burn that fired during that step is now
                    // accounted for in the sim mass. Subtract the delta
                    // from the tanks before we gate the upcoming throttle
                    // so the gate sees the post-drain availability.
                    (
                        reconcile_tanks_from_sim_drain,
                        gate_throttle_on_fuel_availability,
                    )
                        .chain()
                        .in_set(SimStage::Physics)
                        .after(crate::bridge::handle_attitude_controls)
                        .before(crate::bridge::advance_simulation),
                ),
            );
    }
}

/// How fast the throttle ramps when Shift/Ctrl is held, in fraction
/// per second. KSP uses something close to this — full ramp takes
/// ~2 s, fast enough for combat-style adjustments and slow enough to
/// hit a target setting without overshoot.
const THROTTLE_RAMP_RATE: f64 = 0.5;

/// State shared across the throttle/drain systems for prediction
/// management. Tracks the previous frame's effective throttle so the
/// engine-cut edge can dirty the prediction one final time after the
/// burn ends. Held as a [`Local`] so it's per-system and survives
/// across frames.
#[derive(Default)]
pub struct PredictionRefresh {
    prev_effective: f64,
}

/// Push the effective throttle into both the bridge state resource
/// and the simulation, and dirty the prediction every frame the
/// engine fires (plus once on the engine-cut edge to catch the
/// final post-burn state).
///
/// `apply_live_thrust` in the simulation deliberately doesn't dirty
/// at all; this fn owns the policy so the bridge can tune it
/// without touching physics.
///
/// Per-frame dirtying matches KSP's live-trail behaviour. CLAUDE.md
/// notes the prediction is designed to run in-line each frame —
/// `propagate_flight_plan` terminates early (stable orbit, fade
/// horizon) so the typical pass is sub-frame even at 60 Hz. If a
/// future scenario makes the recompute too expensive (deep encounter
/// chains, dense plans), the right knob is a short-horizon "live
/// preview" prediction variant, not throttling the dirty edge here —
/// that just trades smoothness for choppiness.
fn finish_with_throttle(
    effective: f64,
    throttle: &mut ThrottleState,
    sim: &mut SimulationState,
    refresh: &mut PredictionRefresh,
) {
    throttle.effective = effective;
    sim.simulation.set_throttle(effective);

    let active = effective > 0.0;
    let cut_edge = refresh.prev_effective > 0.0 && !active;
    if active || cut_edge {
        sim.simulation.prediction_state.mark_dirty();
    }

    refresh.prev_effective = effective;
}

/// Sample keyboard → [`ThrottleState::commanded`].
///
/// - `Z` snaps to full
/// - `X` snaps to zero
/// - `Shift` (held) ramps up
/// - `Ctrl` (held) ramps down
pub fn handle_throttle_input(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut throttle: ResMut<ThrottleState>,
) {
    if keys.just_pressed(KeyCode::KeyZ) {
        throttle.commanded = 1.0;
        return;
    }
    if keys.just_pressed(KeyCode::KeyX) {
        throttle.commanded = 0.0;
        return;
    }

    let dt = time.delta_secs_f64();
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let ctrl = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    if shift {
        throttle.commanded = (throttle.commanded + THROTTLE_RAMP_RATE * dt).min(1.0);
    }
    if ctrl {
        throttle.commanded = (throttle.commanded - THROTTLE_RAMP_RATE * dt).max(0.0);
    }
}

/// Drain tank reactant pools by however much mass the simulation burned
/// since this system last ran. The simulation's `ship_mass_kg` is the
/// authoritative record of "current ship mass" — it integrates short
/// burns inside long warp ticks correctly, while a naive
/// `mass_flow · real_dt · warp` overdrains the tanks. Reading the sim's
/// actual delta avoids that whole class of bug.
///
/// Drain is split across [`ShipFuelParams::reactant_fractions`] and
/// distributed inside each resource's pools by tank capacity (so
/// already-empty tanks don't pin the whole engine).
///
/// First-call behaviour: records the spawn-time ship mass and skips
/// drain. Without this, the very first frame would treat the ship's
/// entire propellant load as a one-frame burn.
fn reconcile_tanks_from_sim_drain(
    sim: Res<SimulationState>,
    fuel_params: Res<ShipFuelParams>,
    mut tanks: Query<&mut PartResources, With<FuelTank>>,
    mut prev_sim_mass: Local<Option<f64>>,
) {
    let current = sim.simulation.ship_mass_kg();
    let Some(prev) = *prev_sim_mass else {
        // First run after spawn — no drain yet, just take the baseline.
        *prev_sim_mass = Some(current);
        return;
    };
    *prev_sim_mass = Some(current);

    let drained_total_kg = (prev - current).max(0.0);
    if drained_total_kg <= 0.0 || fuel_params.reactant_fractions.is_empty() {
        return;
    }

    for (&res, &frac) in &fuel_params.reactant_fractions {
        if frac <= 0.0 {
            continue;
        }
        let drain_kg = drained_total_kg * frac;
        if drain_kg <= 0.0 {
            continue;
        }

        let total_capacity_kg: f64 = tanks
            .iter()
            .filter_map(|t| t.pools.get(&res).map(|p| p.capacity_mass_kg(res)))
            .sum();
        if total_capacity_kg <= 0.0 {
            continue;
        }

        let density = res.density_kg_per_unit();
        if density <= 0.0 {
            continue;
        }

        for mut tank in tanks.iter_mut() {
            let Some(pool) = tank.pools.get_mut(&res) else {
                continue;
            };
            let weight = pool.capacity_mass_kg(res) / total_capacity_kg;
            let take_kg = drain_kg * weight;
            let take_units = take_kg / density;
            pool.amount = ((pool.amount as f64 - take_units).max(0.0)) as f32;
        }
    }
}

/// Cap throttle on fuel shortage and push the gated value into the
/// simulation. No drain happens here — the simulation drains its own
/// `ship_mass_kg` during its step, and [`reconcile_tanks_from_sim_drain`]
/// drains the tanks by that delta on the next frame.
///
/// Engine fires only at 1× warp. Any other warp level (including
/// pause) gates the throttle to 0; the burn-execution autopilot is
/// responsible for stepping warp down to 1× before opening the
/// throttle on a scheduled maneuver.
pub fn gate_throttle_on_fuel_availability(
    time: Res<Time>,
    fuel_params: Res<ShipFuelParams>,
    mut sim: ResMut<SimulationState>,
    mut throttle: ResMut<ThrottleState>,
    tanks: Query<&PartResources, With<FuelTank>>,
    mut refresh: Local<PredictionRefresh>,
) {
    let warp = sim.simulation.warp.speed();
    let real_dt = time.delta_secs_f64();

    if (warp - 1.0).abs() > f64::EPSILON {
        // Any warp level other than 1× → engine off entirely.
        finish_with_throttle(0.0, &mut throttle, &mut sim, &mut refresh);
        return;
    }

    let throttle_in_use = throttle.commanded;
    let drain_dt = real_dt;

    if throttle_in_use <= 0.0
        || drain_dt <= 0.0
        || fuel_params.mass_flow_kg_per_s <= 0.0
        || fuel_params.reactant_fractions.is_empty()
    {
        finish_with_throttle(throttle_in_use, &mut throttle, &mut sim, &mut refresh);
        return;
    }

    let requested_total_kg = throttle_in_use * fuel_params.mass_flow_kg_per_s * drain_dt;

    // Sum each reactant's available mass across every tank that holds
    // some, so a depleted tank in a multi-tank ship doesn't pin the
    // whole engine.
    let mut available_per_res: HashMap<Resource, f64> = HashMap::new();
    for tank in tanks.iter() {
        for (&res, pool) in &tank.pools {
            *available_per_res.entry(res).or_insert(0.0) += pool.mass_kg(res);
        }
    }

    // Cap throttle to the bottleneck reactant's availability ratio.
    let mut feasibility = 1.0_f64;
    for (&res, &frac) in &fuel_params.reactant_fractions {
        if frac <= 0.0 {
            continue;
        }
        let req = requested_total_kg * frac;
        if req <= 0.0 {
            continue;
        }
        let avail = available_per_res.get(&res).copied().unwrap_or(0.0);
        if req > avail {
            feasibility = feasibility.min(avail / req);
        }
    }
    let throttle_effective = (throttle_in_use * feasibility).max(0.0);
    finish_with_throttle(throttle_effective, &mut throttle, &mut sim, &mut refresh);
}
