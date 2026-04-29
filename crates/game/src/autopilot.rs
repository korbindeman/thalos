//! Burn-execution autopilot — drives the ship's primitives (warp,
//! attitude target, throttle) to fly scheduled maneuver nodes through
//! the same control surface a human player uses. Three states:
//!
//! - [`BurnAutopilot::Idle`] — no node pending, autopilot doesn't act.
//! - [`BurnAutopilot::Pending`] — a node is upcoming. Each frame, drop
//!   warp by one or more levels if continuing at the current warp would
//!   overshoot the burn-start window; while at 1× warp let the
//!   `ManeuverNode` autopilot point the ship; transition to
//!   [`BurnAutopilot::Burn`] when `now ≥ node.time − duration/2`.
//! - [`BurnAutopilot::Burn`] — throttle held at 1.0 each frame,
//!   integrating `delivered_dv` until magnitude meets the planned Δv;
//!   then the node is consumed and the autopilot returns to `Idle`.
//!
//! Burns are centered on `node.time`: the throttle opens at
//! `node.time − duration/2`. The trajectory predictor in
//! [`thalos_physics::trajectory`] uses the same convention so the
//! displayed flight plan matches what the autopilot flies.
//!
//! Disengage path: if the user toggles the auto-maneuvers checkbox
//! mid-burn, the autopilot writes throttle = 0 once and returns to
//! `Idle`. Without the explicit zero-write the previously-asserted
//! `commanded = 1.0` would persist and the engine would keep firing
//! after the autopilot stopped owning the throttle.

use bevy::prelude::*;

use crate::SimStage;
use crate::fuel::{ThrottleState, gate_throttle_on_fuel_availability, handle_throttle_input};
use crate::maneuver::ManeuverPlan;
use crate::navigation::{AUTOPILOT_SETTLE_S, NavigationMode, NavigationState};
use crate::rendering::SimulationState;

#[derive(Resource, Debug, Default, Clone, Copy)]
pub enum BurnAutopilot {
    #[default]
    Idle,
    Pending {
        node_id: u64,
    },
    Burn {
        node_id: u64,
        /// Magnitude of the planned Δv, m/s.
        planned_dv: f64,
        /// Value of [`thalos_physics::simulation::Simulation::delivered_dv`]
        /// captured at the moment the burn started; subtracted from the
        /// live value each frame to get "Δv delivered since burn start."
        anchor_delivered_dv: f64,
    },
}

pub struct BurnAutopilotPlugin;

impl Plugin for BurnAutopilotPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BurnAutopilot>().add_systems(
            Update,
            burn_autopilot_system
                .in_set(SimStage::Physics)
                .after(crate::bridge::handle_warp_controls)
                .after(handle_throttle_input)
                .before(crate::bridge::handle_attitude_controls)
                .before(gate_throttle_on_fuel_availability)
                .before(crate::bridge::advance_simulation),
        );
    }
}

/// How many real seconds before `node.time` the autopilot needs to be
/// at 1× warp. Sized to cover one PD settling cycle (with safety
/// margin) plus the half-burn that precedes `node.time`.
fn lead_seconds_for(burn_duration_s: f64) -> f64 {
    AUTOPILOT_SETTLE_S * 1.5 + burn_duration_s / 2.0 + 0.5
}

/// Worst-case real-frame budget — must match
/// [`thalos_physics::simulation::SimulationConfig::max_real_delta`] so
/// our look-ahead captures the largest sim-time advance the upcoming
/// step might apply.
const FRAME_DT_BUDGET_S: f64 = 0.1;

fn burn_autopilot_system(
    mut state: ResMut<BurnAutopilot>,
    mut sim: ResMut<SimulationState>,
    mut nav: ResMut<NavigationState>,
    mut throttle: ResMut<ThrottleState>,
    plan: Res<ManeuverPlan>,
) {
    if !sim.simulation.auto_maneuvers_enabled() {
        if matches!(*state, BurnAutopilot::Burn { .. }) {
            throttle.commanded = 0.0;
        }
        *state = BurnAutopilot::Idle;
        return;
    }

    match *state {
        BurnAutopilot::Idle => {
            let Some(node) = plan.nodes.first() else {
                return;
            };
            let dv_mag = node.delta_v.length();
            if dv_mag <= 0.0 {
                return;
            }
            if sim.simulation.estimated_burn_duration(dv_mag) <= 0.0 {
                return;
            }
            *state = BurnAutopilot::Pending { node_id: node.id.0 };
            nav.mode = Some(NavigationMode::ManeuverNode);
        }

        BurnAutopilot::Pending { node_id } => {
            let Some(node) = plan.nodes.iter().find(|n| n.id.0 == node_id) else {
                *state = BurnAutopilot::Idle;
                return;
            };
            let dv_mag = node.delta_v.length();
            let duration = sim.simulation.estimated_burn_duration(dv_mag);
            if dv_mag <= 0.0 || duration <= 0.0 {
                *state = BurnAutopilot::Idle;
                return;
            }

            let now = sim.simulation.sim_time();
            let burn_start = node.time - duration / 2.0;
            let lead = lead_seconds_for(duration);
            // We need to be at 1× warp by this sim-time. Floored at
            // `now` so a node placed inside the lead window doesn't
            // produce a target in the past.
            let safe_target = (node.time - lead).max(now);

            // Drop warp as many levels as needed to keep one worst-case
            // frame from overshooting `safe_target`. Naturally cascades
            // from any starting warp down to 1× as the node approaches.
            while sim.simulation.warp.speed() > 1.0 {
                let warp = sim.simulation.warp.speed();
                if now + warp * FRAME_DT_BUDGET_S > safe_target {
                    sim.simulation.warp.decrease();
                } else {
                    break;
                }
            }

            let warp = sim.simulation.warp.speed();
            let at_1x = (warp - 1.0).abs() < f64::EPSILON;
            if at_1x && now >= burn_start {
                *state = BurnAutopilot::Burn {
                    node_id,
                    planned_dv: dv_mag,
                    anchor_delivered_dv: sim.simulation.delivered_dv(),
                };
                throttle.commanded = 1.0;
            }
        }

        BurnAutopilot::Burn {
            node_id,
            planned_dv,
            anchor_delivered_dv,
        } => {
            let delivered = sim.simulation.delivered_dv() - anchor_delivered_dv;
            if delivered >= planned_dv {
                throttle.commanded = 0.0;
                sim.simulation.consume_maneuver_node(node_id);
                *state = BurnAutopilot::Idle;
            } else {
                throttle.commanded = 1.0;
            }
        }
    }
}
