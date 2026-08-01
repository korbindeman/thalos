//! Craft flight configuration: the flap lever and the brakes-driven spoilers.
//!
//! Deliberately shallow (KSP-style) — two controls, no trim management:
//!
//! - **Flap lever** (`F` extend / `R` retract): three detents — UP, TAKEOFF,
//!   LANDING. The aero model scales flap lift linearly with the setting and
//!   flap drag quadratically, so the half detent is the high-lift/low-drag
//!   takeoff configuration and full is the draggy landing one. The per-craft
//!   force increments derive from the authored `Flap` windows on the wings
//!   (see [`crate::aero::build_ship_aero_config`]).
//! - **Brakes** (`B`, the existing latched toggle): wheel brakes on the
//!   ground *and* spoilers in the air. One key to shed speed — tap it on
//!   descent to deploy the spoilers, and it is already latched for the
//!   rollout when you touch down. The spoiler half is **airspeed-gated**
//!   ([`SPOILER_DEPLOY_AIRSPEED_M_S`]): parked or taxiing with the parking
//!   brake latched keeps the panels stowed (a parked aircraft with its
//!   spoilers standing up made no sense), they auto-deploy for the rollout
//!   lift dump while fast, and auto-stow as the rollout decays to taxi speed.
//!
//! The resource carries both the lever setting and the *actual* actuator
//! positions (flaps travel slowly, spoilers snap faster). The aero model and
//! the control-surface visuals both consume the actual positions, so
//! deployment forces build smoothly and the meshes move at the same rate the
//! forces do.

use bevy::prelude::*;
use thalos_input::game::GameInputIntent;

use crate::SimStage;
use crate::local_physics::ParkingBrake;
use crate::sim_clock::SimClock;

pub use thalos_game_state::flight::{FLAP_DETENTS, FlightConfig};
#[cfg(test)]
use thalos_game_state::flight::TAKEOFF_FLAP_DETENT;

/// Full flap travel time (UP → LANDING), seconds.
const FLAP_TRAVEL_S: f64 = 6.0;
/// Spoiler travel time, seconds.
const SPOILER_TRAVEL_S: f64 = 0.8;
/// Airspeed above which the brakes latch deploys the spoilers (m/s). Below
/// flying speed the latch means *wheel brakes only* — a parked or taxiing
/// aircraft never stands its spoiler panels up. Set below any plausible
/// approach speed so a brakes-latched touchdown still gets its lift dump
/// from the first frame of the rollout.
const SPOILER_DEPLOY_AIRSPEED_M_S: f64 = 30.0;
/// Airspeed below which deployed spoilers auto-stow (m/s). Hysteresis under
/// [`SPOILER_DEPLOY_AIRSPEED_M_S`] so the panels don't flap around one
/// threshold as the rollout decays through it.
const SPOILER_STOW_AIRSPEED_M_S: f64 = 20.0;



pub struct FlightConfigPlugin;

impl Plugin for FlightConfigPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FlightConfig>()
            .register_type::<FlightConfig>()
            .add_systems(
                Update,
                update_flight_config
                    .in_set(SimStage::Physics)
                    .after(crate::control_bus::realize_control)
                    .before(crate::bridge::advance_simulation),
            );
    }
}

/// Step the flap lever on the input edges and chase the actuator positions.
/// Actuators run on [`SimClock`] so a paused sim freezes them with the rest
/// of the craft.
fn update_flight_config(
    clock: Res<SimClock>,
    intent: Res<GameInputIntent>,
    brake: Res<ParkingBrake>,
    ground_control: Res<crate::control_bus::ResolvedGroundControl>,
    kin: Res<thalos_physics_local::LocalCraftKinematics>,
    mut config: ResMut<FlightConfig>,
    mut fast_enough: Local<bool>,
) {
    if intent.flaps_extend {
        config.flap_setting = (config.flap_setting + 1).min(FLAP_DETENTS);
    }
    if intent.flaps_retract {
        config.flap_setting = config.flap_setting.saturating_sub(1);
    }

    let dt = clock.delta_secs_f64();
    let flap_target = config.flap_setting as f64 / FLAP_DETENTS as f64;
    config.flap_fraction = approach(config.flap_fraction, flap_target, dt / FLAP_TRAVEL_S);
    // Spoilers deploy only when the brakes latch is on AND the craft is at
    // flying speed (with hysteresis) — see the module doc. The SLF velocity is
    // air-relative (co-rotating frame, wind = 0), so it is the airspeed.
    let airspeed = if kin.valid {
        kin.slf_linear_velocity_m_s.length()
    } else {
        0.0
    };
    if airspeed >= SPOILER_DEPLOY_AIRSPEED_M_S {
        *fast_enough = true;
    } else if airspeed < SPOILER_STOW_AIRSPEED_M_S {
        *fast_enough = false;
    }
    let braking = brake.engaged || ground_control.brake > 0.05;
    let spoiler_target = if braking && *fast_enough { 1.0 } else { 0.0 };
    config.spoiler_fraction = approach(
        config.spoiler_fraction,
        spoiler_target,
        dt / SPOILER_TRAVEL_S,
    );
}

/// Move `current` toward `target` by at most `max_step`.
fn approach(current: f64, target: f64, max_step: f64) -> f64 {
    current + (target - current).clamp(-max_step.abs(), max_step.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runway_takeoff_starts_at_the_takeoff_detent() {
        let config = FlightConfig::runway_takeoff();

        assert_eq!(config.flap_setting, TAKEOFF_FLAP_DETENT);
        assert_eq!(config.flap_fraction, 0.5);
        assert_eq!(config.spoiler_fraction, 0.0);
    }
}
