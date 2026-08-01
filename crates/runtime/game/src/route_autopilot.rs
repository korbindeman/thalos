//! Runway-destination autoflight.
//!
//! LAND is deliberately a control-demand producer, not a second effector
//! path. It reads the route service, advances one explicit phase machine, and
//! publishes attitude/throttle/ground demands for `control_bus` to arbitrate.
//! The mode remains engaged through touchdown and rollout; completion means a
//! stable stop with the parking brake holding.

use bevy::prelude::*;
use thalos_control::{AttitudeDemand, ControlDemand, PlaneHoldTarget};
use thalos_input::game::GameInputIntent;
use thalos_navigation::{ApproachPhase, GS_FULL_SCALE_RAD, LOC_FULL_SCALE_RAD};
use thalos_physics_local::LocalCraftKinematics;

use crate::SimStage;
use crate::flight_config::{FLAP_DETENTS, FlightConfig};
use crate::fuel::{PilotThrottleInput, ThrottleState};
use crate::local_physics::{GearState, ParkingBrake, WeightOnWheels};
use crate::rendering::SimulationState;
use crate::route::RouteState;
use crate::sim_clock::SimClock;

pub use thalos_game_state::nav::{LandAutopilot, LandPhase};

const PILOT_OVERRIDE_DEADZONE_SQ: f32 = 0.05 * 0.05;
const FLARE_HEIGHT_M: f64 = 18.0;
const TOUCHDOWN_CONFIRM_S: f64 = 0.25;
const BOUNCE_CONFIRM_S: f64 = 0.35;
const STOP_SPEED_M_S: f64 = 0.5;
const STOP_CONFIRM_S: f64 = 1.0;
const MAX_GO_AROUNDS: u8 = 3;
const APPROACH_ALPHA_BIAS_RAD: f64 = 4.0_f64.to_radians();
const FLARE_PITCH_RAD: f64 = 7.0_f64.to_radians();
const GO_AROUND_PITCH_RAD: f64 = 10.0_f64.to_radians();





pub struct RouteAutopilotPlugin;

impl Plugin for RouteAutopilotPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LandAutopilot>().add_systems(
            Update,
            update_land_autopilot
                .in_set(SimStage::Physics)
                .after(crate::fuel::handle_throttle_input)
                .before(crate::controls::update_control_locks)
                .before(crate::control_bus::realize_control),
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_land_autopilot(
    mut land: ResMut<LandAutopilot>,
    mut route: ResMut<RouteState>,
    input: Res<GameInputIntent>,
    pilot_throttle: Res<PilotThrottleInput>,
    kin: Res<LocalCraftKinematics>,
    wow: Res<WeightOnWheels>,
    mut gear: ResMut<GearState>,
    mut parking_brake: ResMut<ParkingBrake>,
    mut config: ResMut<FlightConfig>,
    mut throttle: ResMut<ThrottleState>,
    sim: Res<SimulationState>,
    clock: Res<SimClock>,
) {
    // The program's own selection flag, not a shared mode slot.
    if !land.engaged {
        land.demand = ControlDemand::NONE;
        if !matches!(land.phase, LandPhase::Stopped | LandPhase::Unable) {
            land.phase = LandPhase::Off;
        }
        return;
    }

    // Raw input remains sampled while LAND owns the channels. A real stick or
    // throttle movement is the unambiguous takeover gesture.
    if input.attitude.length_squared() > PILOT_OVERRIDE_DEADZONE_SQ
        || pilot_throttle.moved
        || input.parking_brake_toggle
    {
        info!(
            target: "thalos::diagnostic::approach_ap",
            event = "land_disengaged",
            reason = "pilot_override",
            phase = ?land.phase,
            "LAND disengaged by pilot"
        );
        land.engaged = false;
        land.phase = LandPhase::Off;
        land.demand = ControlDemand::NONE;
        return;
    }

    if sim.simulation.is_destroyed() {
        land.engaged = false;
        land.phase = LandPhase::Unable;
        land.demand = ControlDemand::NONE;
        return;
    }

    if matches!(
        land.phase,
        LandPhase::Off | LandPhase::Stopped | LandPhase::Unable
    ) {
        let initial = if route.destination_guidance.is_some() {
            LandPhase::Enroute
        } else if route.guidance.is_some() {
            LandPhase::TerminalCapture
        } else {
            info!(
            target: "thalos::diagnostic::approach_ap",
                event = "land_disengaged",
                reason = "no_runway_guidance",
                "LAND unavailable"
            );
            land.engaged = false;
            land.phase = LandPhase::Unable;
            return;
        };
        land.reset_for_engagement(initial);
        parking_brake.engaged = false;
        info!(
            target: "thalos::diagnostic::approach_ap",
            event = "land_engaged",
            phase = ?initial,
            "LAND engaged"
        );
    }

    let dt = clock.delta_secs_f64().clamp(0.0, 0.1);
    let airspeed = if kin.valid {
        kin.slf_linear_velocity_m_s.length()
    } else {
        0.0
    };

    // Touchdown confirmation is intentionally independent of route phase:
    // wheel contact is the physical boundary between flight and rollout.
    if wow.grounded {
        land.contact_s += dt;
        land.airborne_s = 0.0;
    } else {
        land.contact_s = 0.0;
        if land.phase == LandPhase::Rollout {
            land.airborne_s += dt;
        }
    }

    let bounced = land.phase == LandPhase::Rollout
        && land.airborne_s >= BOUNCE_CONFIRM_S
        && airspeed > STOP_SPEED_M_S;
    let unstable_approach = if matches!(
        land.phase,
        LandPhase::Final | LandPhase::Flare | LandPhase::TerminalCapture
    ) {
        route
            .guidance
            .zip(route.plan.as_ref())
            .is_some_and(|(guidance, plan)| {
                let height_m = guidance.altitude_m - plan.frame.origin_altitude_m;
                let sink_rate_m_s = guidance.fpa_rad.sin() * airspeed;
                guidance.phase != ApproachPhase::Transition
                    && height_m < 250.0
                    && (guidance.loc_deviation_rad.abs() > 1.5 * LOC_FULL_SCALE_RAD
                        || guidance.gs_deviation_rad.abs() > 1.5 * GS_FULL_SCALE_RAD
                        || sink_rate_m_s < -8.0)
            })
    } else {
        false
    };
    if bounced || unstable_approach {
        if land.go_arounds >= MAX_GO_AROUNDS {
            info!(
                target: "thalos::diagnostic::approach_ap",
                event = "land_disengaged",
                reason = "go_around_limit",
                phase = ?land.phase,
                "LAND unable to recover"
            );
            land.engaged = false;
            land.set_phase(LandPhase::Unable);
            land.demand = ControlDemand::NONE;
            return;
        }
        land.go_arounds += 1;
        land.go_around_s = 0.0;
        parking_brake.engaged = false;
        info!(
            target: "thalos::diagnostic::approach_ap",
            event = "land_go_around",
            reason = if bounced { "bounce" } else { "unstable_approach" },
            retry_count = land.go_arounds,
            "LAND going around"
        );
        route.recover_to_destination_ingress();
        land.set_phase(LandPhase::GoAround);
    }

    if land.contact_s >= TOUCHDOWN_CONFIRM_S
        && matches!(
            land.phase,
            LandPhase::TerminalCapture | LandPhase::Final | LandPhase::Flare
        )
    {
        land.set_phase(LandPhase::Rollout);
    }

    match land.phase {
        LandPhase::Enroute => {
            let Some(guidance) = route.destination_guidance else {
                land.set_phase(LandPhase::TerminalCapture);
                // Hold the last ingress command for this single phase-handoff
                // frame; terminal guidance is already available on the next
                // physics pass.
                return;
            };
            if !wow.grounded {
                gear.down = false;
            }
            config.flap_setting = 0;
            parking_brake.engaged = false;
            let pitch = pitch_for_vertical_speed(
                guidance.vertical_speed_command_m_s,
                airspeed,
                APPROACH_ALPHA_BIAS_RAD,
            );
            let throttle_command = speed_throttle(
                airspeed,
                guidance.target_speed_m_s,
                dt,
                &mut land.speed_integral,
            );
            land.demand = flight_demand(pitch, guidance.bank_command_rad, throttle_command);
        }
        LandPhase::TerminalCapture | LandPhase::Final | LandPhase::Flare => {
            let Some(guidance) = route.guidance else {
                land.demand = ControlDemand::NONE;
                return;
            };
            let Some(plan) = route.plan.as_ref() else {
                land.demand = ControlDemand::NONE;
                return;
            };

            let height_m = guidance.altitude_m - plan.frame.origin_altitude_m;
            if guidance.dtg_m < 12_000.0 || guidance.phase != ApproachPhase::Transition {
                gear.down = true;
                config.flap_setting = FLAP_DETENTS;
            }
            let next_phase =
                if height_m <= FLARE_HEIGHT_M || guidance.phase == ApproachPhase::Touchdown {
                    LandPhase::Flare
                } else if guidance.phase == ApproachPhase::Final {
                    LandPhase::Final
                } else {
                    LandPhase::TerminalCapture
                };
            land.set_phase(next_phase);

            let target_speed = guidance
                .target_speed_m_s
                .unwrap_or(route.approach_speed_m_s.max(1.0));
            let (pitch, throttle_command) = if land.phase == LandPhase::Flare {
                let blend = (1.0 - height_m / FLARE_HEIGHT_M).clamp(0.0, 1.0);
                let approach_pitch = guidance.fpa_command_rad + APPROACH_ALPHA_BIAS_RAD;
                (
                    approach_pitch + (FLARE_PITCH_RAD - approach_pitch) * blend,
                    0.0,
                )
            } else {
                (
                    guidance.fpa_command_rad + APPROACH_ALPHA_BIAS_RAD,
                    speed_throttle(airspeed, target_speed, dt, &mut land.speed_integral),
                )
            };
            let bank = if land.phase == LandPhase::Flare {
                guidance.bank_command_rad * 0.35
            } else {
                guidance.bank_command_rad
            };
            land.demand = flight_demand(pitch, bank, throttle_command);
        }
        LandPhase::Rollout => {
            gear.down = true;
            config.flap_setting = FLAP_DETENTS;
            let steer = route
                .guidance
                .map(|g| {
                    let runway_heading = route
                        .plan
                        .as_ref()
                        .map(|plan| plan.landing_heading_rad)
                        .unwrap_or(g.course_heading_rad);
                    let heading_command =
                        thalos_navigation::wrap_angle(runway_heading - g.track_heading_rad)
                            / 15.0_f64.to_radians();
                    (heading_command - 0.7 * g.loc_deflection()).clamp(-1.0, 1.0)
                })
                .unwrap_or(0.0);
            land.demand = rollout_demand(steer);

            if wow.grounded && airspeed <= STOP_SPEED_M_S {
                land.stopped_s += dt;
            } else {
                land.stopped_s = 0.0;
            }
            if land.stopped_s >= STOP_CONFIRM_S {
                parking_brake.engaged = true;
                throttle.hold_idle_until_pilot_move = true;
                land.set_phase(LandPhase::Stopped);
                land.demand = rollout_demand(0.0);
                info!(
                    target: "thalos::diagnostic::approach_ap",
                    event = "land_completed",
                    speed_m_s = airspeed,
                    go_arounds = land.go_arounds,
                    "LAND complete: aircraft stopped"
                );
                land.engaged = false;
            }
        }
        LandPhase::GoAround => {
            parking_brake.engaged = false;
            gear.down = true;
            config.flap_setting = 1;
            land.demand = flight_demand(GO_AROUND_PITCH_RAD, 0.0, 1.0);
            if !wow.grounded {
                land.go_around_s += dt;
            }
            if land.go_around_s >= 5.0 {
                land.speed_integral = 0.0;
                land.set_phase(if route.destination_guidance.is_some() {
                    LandPhase::Enroute
                } else {
                    LandPhase::TerminalCapture
                });
            }
        }
        LandPhase::Stopped | LandPhase::Unable | LandPhase::Off => {
            land.demand = ControlDemand::NONE;
        }
    }

    land.diagnostic_s += dt;
    if land.diagnostic_s >= 1.0 && land.active() {
        land.diagnostic_s = 0.0;
        let (
            dtg_m,
            cross_track_m,
            altitude_error_m,
            target_speed_m_s,
            loc_dev_rad,
            gs_dev_rad,
            height_over_rwy_m,
        ) = if let Some(guidance) = route.guidance {
            (
                guidance.dtg_m,
                guidance.cross_track_m,
                guidance.altitude_error_m,
                guidance
                    .target_speed_m_s
                    .unwrap_or(route.approach_speed_m_s),
                guidance.loc_deviation_rad,
                guidance.gs_deviation_rad,
                route
                    .plan
                    .as_ref()
                    .map(|plan| guidance.altitude_m - plan.frame.origin_altitude_m)
                    .unwrap_or(0.0),
            )
        } else if let Some(guidance) = route.destination_guidance {
            (
                guidance.distance_to_arrival_m,
                0.0,
                0.0,
                guidance.target_speed_m_s,
                0.0,
                0.0,
                0.0,
            )
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        };
        let (pitch_cmd_rad, bank_cmd_rad) = match land.demand.attitude {
            AttitudeDemand::FlightPath(target) => (target.pitch_rad, target.bank_rad),
            _ => (0.0, 0.0),
        };
        info!(
            target: "thalos::diagnostic::approach_ap",
            event = "appr_frame",
            phase = ?land.phase,
            dtg_m,
            cross_track_m,
            altitude_error_m,
            loc_dev_rad,
            gs_dev_rad,
            pitch_cmd_rad,
            bank_cmd_rad,
            height_over_rwy_m,
            airspeed_m_s = airspeed,
            ground_speed_m_s = airspeed,
            target_speed_m_s,
            throttle_cmd = land.demand.throttle.unwrap_or(0.0),
            ground_steer_cmd = land.demand.ground_steer.unwrap_or(0.0),
            brake_cmd = land.demand.wheel_brake.unwrap_or(0.0),
            retry_count = land.go_arounds,
            weight_on_wheels = wow.grounded,
            "LAND guidance frame"
        );
    }
}

fn flight_demand(pitch_rad: f64, bank_rad: f64, throttle: f64) -> ControlDemand {
    ControlDemand::autoflight(
        AttitudeDemand::FlightPath(PlaneHoldTarget {
            pitch_rad: pitch_rad.clamp(-12.0_f64.to_radians(), 18.0_f64.to_radians()),
            bank_rad,
        }),
        Some(throttle),
        None,
        None,
    )
}

fn rollout_demand(steer: f64) -> ControlDemand {
    ControlDemand::autoflight(
        // The same right-positive steering command reaches the rudder at
        // rollout speed and the nosewheel as its speed fade hands authority
        // down. Body +Z torque yaws left, hence the sign conversion.
        AttitudeDemand::Rate(Vec3::new(0.0, 0.0, -steer as f32).as_dvec3()),
        Some(0.0),
        Some(steer),
        Some(1.0),
    )
}

fn pitch_for_vertical_speed(vertical_speed_m_s: f64, airspeed_m_s: f64, alpha_bias: f64) -> f64 {
    (vertical_speed_m_s / airspeed_m_s.max(20.0)).atan() + alpha_bias
}

/// PI autothrottle with conditional integration. The integrator is held when
/// the output is saturated in the same direction as the speed error, avoiding
/// a long throttle hangover after a large descent-speed transient.
fn speed_throttle(speed_m_s: f64, target_m_s: f64, dt: f64, integral: &mut f64) -> f64 {
    const TRIM: f64 = 0.48;
    const KP: f64 = 0.012;
    const KI: f64 = 0.003;
    let error = target_m_s - speed_m_s;
    let candidate = (*integral + error * dt).clamp(-100.0, 100.0);
    let raw = TRIM + KP * error + KI * candidate;
    if (raw < 1.0 || error < 0.0) && (raw > 0.0 || error > 0.0) {
        *integral = candidate;
    }
    (TRIM + KP * error + KI * *integral).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autothrottle_opens_below_target_and_closes_above_it() {
        let mut integral = 0.0;
        let slow = speed_throttle(80.0, 100.0, 0.02, &mut integral);
        integral = 0.0;
        let fast = speed_throttle(120.0, 100.0, 0.02, &mut integral);
        assert!(slow > 0.48);
        assert!(fast < 0.48);
    }

    #[test]
    fn vertical_speed_pitch_includes_angle_of_attack_bias() {
        let level = pitch_for_vertical_speed(0.0, 100.0, APPROACH_ALPHA_BIAS_RAD);
        let descent = pitch_for_vertical_speed(-5.0, 100.0, APPROACH_ALPHA_BIAS_RAD);
        assert_eq!(level, APPROACH_ALPHA_BIAS_RAD);
        assert!(descent < level);
    }
}
