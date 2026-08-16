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
use thalos_navigation::ApproachPhase;
use thalos_physics_local::{ActiveLocalBubble, LocalCraftKinematics};

use crate::SimStage;
use crate::flight_config::{FLAP_DETENTS, FlightConfig};
use crate::fuel::{PilotThrottleInput, ThrottleState};
use crate::local_physics::{GearState, ParkingBrake, WeightOnWheels};
use crate::rendering::SimulationState;
use crate::route::RouteState;
use crate::sim_clock::SimClock;

pub use thalos_game_state::nav::{LandAutopilot, LandNotice, LandPhase};

const PILOT_OVERRIDE_DEADZONE_SQ: f32 = 0.05 * 0.05;
const FLARE_HEIGHT_M: f64 = 18.0;
const TOUCHDOWN_CONFIRM_S: f64 = 0.25;
/// Time continuously airborne after a confirmed touchdown before LAND treats
/// the separation as an escaped bounce rather than suspension rebound (s).
///
/// The recorded normal touchdown unloaded the wheels for 0.37 s while the
/// craft was still aligned and only ~5 m above runway reference. A 0.35 s gate
/// therefore converted the first oleo rebound directly into full-power flight.
/// Two seconds leaves the idle-throttle flare law time to settle an ordinary
/// rebound, while a craft that is genuinely flying away still recovers.
const BOUNCE_ESCAPE_CONFIRM_S: f64 = 2.0;
const STOP_SPEED_M_S: f64 = 0.5;
const STOP_CONFIRM_S: f64 = 1.0;
const MAX_GO_AROUNDS: u8 = 3;
const APPROACH_ALPHA_BIAS_RAD: f64 = 4.0_f64.to_radians();
/// Vertical speed LAND aims to carry through main-wheel contact. Airliner
/// autoland is normally firm enough to guarantee contact, but nowhere near the
/// recorded -2.8 m/s arrival that exhausted the Meridian's suspension stroke.
const TOUCHDOWN_SINK_RATE_M_S: f64 = -0.75;
/// Sink-error feedback in the flare. The route's flight-path command is only a
/// feed-forward attitude; this term is what actually arrests a descent that is
/// not following it because of aircraft response lag.
const FLARE_SINK_GAIN_RAD_PER_M_S: f64 = 4.0_f64.to_radians();
const FLARE_SINK_CORRECTION_DOWN_RAD: f64 = 3.0_f64.to_radians();
const FLARE_SINK_CORRECTION_UP_RAD: f64 = 8.0_f64.to_radians();
/// Keep approach power through the high flare, then retard progressively over
/// the last few metres. Cutting to idle at 18 m / 60 ft made the sink build
/// just as the pitch loop was trying to arrest it.
const FLARE_RETARD_START_HEIGHT_M: f64 = 9.0;
const FLARE_IDLE_HEIGHT_M: f64 = 3.0;
const GO_AROUND_PITCH_RAD: f64 = 10.0_f64.to_radians();

/// How long the approach must stay out of tolerance before LAND gives up on it
/// (s).
///
/// A dwell, not an instantaneous test, because the deviations legitimately
/// spike for a frame — a re-plan, a projection settling onto a new leg, the
/// craft rolling out of the join. The old code sampled one frame and could
/// throw away a recoverable approach.
///
/// Sized as "long enough to roll out of the intercept and settle, short enough
/// that the go-around still happens with the runway ahead". At the final
/// approach point (9 km out, ~85 m/s) this fires with more than 8 km to run,
/// where a reposition is cheap — versus the recorded flight, which flew an
/// approach 1.8 km off the centreline for another 44 s before objecting at
/// 250 m over the threshold.
const UNSTABLE_DWELL_S: f64 = 5.0;
/// Sink rate (m/s, negative down) beyond which the approach is unflyable
/// regardless of where the needles sit.
const MAX_APPROACH_SINK_M_S: f64 = -8.0;
/// How long a notice stays on the annunciator (s). Long enough to read after
/// looking back from outside, short enough not to shadow the next one.
const NOTICE_HOLD_S: f64 = 12.0;

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
    bubble: Res<ActiveLocalBubble>,
    wow: Res<WeightOnWheels>,
    mut gear: thalos_game_state::ActiveCraftMut<GearState>,
    mut parking_brake: thalos_game_state::ActiveCraftMut<ParkingBrake>,
    mut config: ResMut<FlightConfig>,
    mut throttle: ResMut<ThrottleState>,
    sim: Res<SimulationState>,
    clock: Res<SimClock>,
) {
    let (Some(mut gear), Some(mut parking_brake)) = (gear.get_mut(), parking_brake.get_mut())
    else {
        return;
    };
    // The notice ages on real time whether or not LAND is engaged — the player
    // most needs to read "GO-AROUND: not lined up" in the seconds *after* it
    // happened, which is often after a disengagement.
    let notice_dt = clock.delta_secs_f64().clamp(0.0, 0.1);
    if land.notice.is_some() {
        land.notice_age_s += notice_dt;
        if land.notice_age_s >= NOTICE_HOLD_S && !land.active() {
            land.notice = None;
        }
    }

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
        land.notify(LandNotice::DisengagedByPilot);
        info!(
            target: "thalos::diagnostic::approach_ap",
            event = "land_disengaged",
            reason = LandNotice::DisengagedByPilot.diagnostic_reason(),
            phase = ?land.phase,
            "LAND disengaged by pilot"
        );
        land.engaged = false;
        land.phase = LandPhase::Off;
        land.demand = ControlDemand::NONE;
        return;
    }

    if sim.simulation.is_destroyed() {
        land.notify(LandNotice::UnableDestroyed);
        land.engaged = false;
        land.phase = LandPhase::Unable;
        land.demand = ControlDemand::NONE;
        return;
    }

    // The runway can be deselected, deleted, or lost with the body underneath a
    // craft that is still flying an approach to it. Saying so is the difference
    // between "UNABLE: runway selection lost" and an autopilot that appears to
    // forget what it was doing.
    if land.active()
        && route.guidance.is_none()
        && route.destination_guidance.is_none()
        && !matches!(land.phase, LandPhase::Rollout | LandPhase::GoAround)
    {
        land.notify(LandNotice::UnableLostRunway);
        info!(
            target: "thalos::diagnostic::approach_ap",
            event = "land_disengaged",
            reason = LandNotice::UnableLostRunway.diagnostic_reason(),
            phase = ?land.phase,
            "LAND lost its destination"
        );
        land.engaged = false;
        land.set_phase(LandPhase::Unable);
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
            land.notify(LandNotice::UnableNoGuidance);
            info!(
                target: "thalos::diagnostic::approach_ap",
                event = "land_disengaged",
                reason = LandNotice::UnableNoGuidance.diagnostic_reason(),
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

    let bounced = bounce_requires_go_around(land.phase, land.airborne_s, airspeed);

    // --- The stabilisation gate.
    //
    // Only on final and in the flare. The localizer and glideslope are measured
    // against the runway centreline, so on a base leg they describe a beam the
    // craft is not supposed to be on yet — `thalos_navigation::guidance` says so
    // in as many words, and reading them during the join is how a normal
    // intercept looked like a 3.5x full-scale failure.
    //
    // `route.established` is the honest test (both needles inside full scale),
    // and it is live rather than latched, so an approach that goes out of
    // tolerance again is caught rather than grandfathered.
    let sink_rate_m_s = route
        .guidance
        .map(|g| g.fpa_rad.sin() * airspeed)
        .unwrap_or(0.0);
    let out_of_tolerance = approach_out_of_tolerance(
        land.phase,
        route.guidance.map(|g| g.phase),
        route.established,
        sink_rate_m_s,
    );
    if out_of_tolerance.is_some() {
        land.unstable_s += dt;
    } else {
        land.unstable_s = 0.0;
    }
    let unstable_approach = land.unstable_s >= UNSTABLE_DWELL_S;

    if bounced || unstable_approach {
        let cause = if bounced {
            LandNotice::GoAroundBounce
        } else {
            out_of_tolerance.unwrap_or(LandNotice::GoAroundUnstable)
        };
        if land.go_arounds >= MAX_GO_AROUNDS {
            land.notify(LandNotice::UnableGoAroundLimit);
            info!(
                target: "thalos::diagnostic::approach_ap",
                event = "land_disengaged",
                reason = LandNotice::UnableGoAroundLimit.diagnostic_reason(),
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
        land.unstable_s = 0.0;
        parking_brake.engaged = false;
        land.notify(cause);
        info!(
            target: "thalos::diagnostic::approach_ap",
            event = "land_go_around",
            reason = cause.diagnostic_reason(),
            retry_count = land.go_arounds,
            loc_dev_rad = route.guidance.map(|g| g.loc_deviation_rad).unwrap_or(0.0),
            gs_dev_rad = route.guidance.map(|g| g.gs_deviation_rad).unwrap_or(0.0),
            cross_track_m = route.guidance.map(|g| g.cross_track_m).unwrap_or(0.0),
            dtg_m = route.guidance.map(|g| g.dtg_m).unwrap_or(0.0),
            sink_rate_m_s,
            post_touchdown_airborne_s = land.airborne_s,
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
                let approach_throttle =
                    speed_throttle(airspeed, target_speed, dt, &mut land.speed_integral);
                let command = flare_command(
                    height_m,
                    guidance.fpa_command_rad,
                    sink_rate_m_s,
                    airspeed,
                    approach_throttle,
                );
                (command.pitch_rad, command.throttle)
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
            // A confirmed touchdown is a commitment to land. If the suspension
            // briefly unloads, keep idle throttle, spoilers/brakes and the flare
            // attitude instead of snapping to level pitch or immediately adding
            // go-around power. A sustained separation is still caught above.
            land.demand = rollout_demand(steer, !wow.grounded, airspeed);

            if wow.grounded && airspeed <= STOP_SPEED_M_S {
                land.stopped_s += dt;
            } else {
                land.stopped_s = 0.0;
            }
            if land.stopped_s >= STOP_CONFIRM_S {
                parking_brake.engaged = true;
                throttle.commanded = 0.0;
                land.set_phase(LandPhase::Stopped);
                land.notify(LandNotice::Completed);
                land.demand = rollout_demand(0.0, false, airspeed);
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
        // Commanded *and* achieved, because they answer different questions and
        // the log could not tell them apart. A bank pegged at the limit for
        // 50 s while cross-track ran out to 2 km is either an aircraft that
        // will not roll or a plan that is steering the wrong way, and one
        // number cannot say which.
        // `slf_position_m` is an offset inside the surface-local frame, not a
        // radial vector from the body centre — normalising it yields a direction
        // that is not "up" and made the first version of this instrument report
        // 50° of bank against a 12° command. The frame knows the real radial;
        // ask it, exactly as `control_bus` does when it builds `FlightState`.
        let (bank_rad, fpa_rad) = match bubble.bubble.as_ref().filter(|_| kin.valid) {
            Some(bubble) => {
                let up = thalos_physics_canonical::surface_local::radial_up(
                    &bubble.frame,
                    kin.slf_position_m,
                );
                let up_body = kin.orientation.inverse() * up;
                let speed = kin.slf_linear_velocity_m_s.length();
                let vertical = kin.slf_linear_velocity_m_s.dot(up);
                (
                    (-up_body.x).atan2(up_body.z),
                    if speed > 1.0 {
                        (vertical / speed).clamp(-1.0, 1.0).asin()
                    } else {
                        0.0
                    },
                )
            }
            None => (0.0, 0.0),
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
            bank_rad,
            fpa_rad,
            sink_rate_m_s,
            target_sink_rate_m_s = route
                .guidance
                .map(|guidance| flare_target_sink_rate(
                    height_over_rwy_m,
                    guidance.fpa_command_rad,
                    airspeed,
                ))
                .unwrap_or(0.0),
            height_over_rwy_m,
            airspeed_m_s = airspeed,
            ground_speed_m_s = airspeed,
            target_speed_m_s,
            throttle_cmd = land.demand.throttle.unwrap_or(0.0),
            ground_steer_cmd = land.demand.ground_steer.unwrap_or(0.0),
            brake_cmd = land.demand.wheel_brake.unwrap_or(0.0),
            retry_count = land.go_arounds,
            unstable_s = land.unstable_s,
            touchdown_contact_s = land.contact_s,
            post_touchdown_airborne_s = land.airborne_s,
            established = route.established,
            plan_frozen = route.plan_frozen,
            weight_on_wheels = wow.grounded,
            "LAND guidance frame"
        );
    }
}

/// Is this frame of the approach outside what LAND will accept? `None` means
/// acceptable; the caller applies [`UNSTABLE_DWELL_S`] before acting, so a
/// single frame here is never a decision.
///
/// The phase gate is the substance. The localizer and glideslope are measured
/// against the runway centreline, so during the join they describe a beam the
/// craft is not on yet and is not meant to be — reading them there turns a
/// normal intercept into a 3.5x full-scale "failure". They are only meaningful
/// once the craft is on final, which is exactly where a stabilisation check
/// belongs anyway.
fn approach_out_of_tolerance(
    phase: LandPhase,
    guidance_phase: Option<ApproachPhase>,
    established: bool,
    sink_rate_m_s: f64,
) -> Option<LandNotice> {
    if !matches!(phase, LandPhase::Final | LandPhase::Flare) {
        return None;
    }
    if sink_rate_m_s < MAX_APPROACH_SINK_M_S {
        return Some(LandNotice::GoAroundSinkRate);
    }
    // No guidance at all is a lost destination, handled separately; it must not
    // read here as "not established" and trigger a go-around toward nothing.
    let on_final = guidance_phase.is_some_and(|p| p != ApproachPhase::Transition);
    if on_final && !established {
        return Some(LandNotice::GoAroundUnstable);
    }
    None
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

fn rollout_demand(steer: f64, airborne_after_touchdown: bool, airspeed_m_s: f64) -> ControlDemand {
    let attitude = if airborne_after_touchdown {
        AttitudeDemand::FlightPath(PlaneHoldTarget {
            // Do not turn a suspension rebound into a commanded climb. Hold
            // the same shallow descending attitude the flare targets at wheel
            // contact until the wheels reload.
            pitch_rad: pitch_for_vertical_speed(
                TOUCHDOWN_SINK_RATE_M_S,
                airspeed_m_s,
                APPROACH_ALPHA_BIAS_RAD,
            ),
            bank_rad: 0.0,
        })
    } else {
        // The same right-positive steering command reaches the rudder at
        // rollout speed and the nosewheel as its speed fade hands authority
        // down. Body +Z torque yaws left, hence the sign conversion.
        AttitudeDemand::Rate(Vec3::new(0.0, 0.0, -steer as f32).as_dvec3())
    };
    ControlDemand::autoflight(attitude, Some(0.0), Some(steer), Some(1.0))
}

#[derive(Debug, Clone, Copy)]
struct FlareCommand {
    pitch_rad: f64,
    throttle: f64,
}

fn flare_target_sink_rate(height_m: f64, approach_fpa_rad: f64, airspeed_m_s: f64) -> f64 {
    let approach_sink_rate = approach_fpa_rad.sin() * airspeed_m_s;
    let high_flare_fraction = (height_m / FLARE_HEIGHT_M).clamp(0.0, 1.0);
    TOUCHDOWN_SINK_RATE_M_S + (approach_sink_rate - TOUCHDOWN_SINK_RATE_M_S) * high_flare_fraction
}

/// Closed-loop flare command: schedule a progressively gentler vertical speed,
/// then add pitch when the aircraft is descending faster than that schedule.
/// This is deliberately a pure law so recorded flight states can pin it without
/// booting Bevy or relying on a subjective play pass.
fn flare_command(
    height_m: f64,
    approach_fpa_rad: f64,
    actual_sink_rate_m_s: f64,
    airspeed_m_s: f64,
    approach_throttle: f64,
) -> FlareCommand {
    let target_sink_rate_m_s = flare_target_sink_rate(height_m, approach_fpa_rad, airspeed_m_s);
    let pitch_feed_forward =
        pitch_for_vertical_speed(target_sink_rate_m_s, airspeed_m_s, APPROACH_ALPHA_BIAS_RAD);
    let sink_correction =
        ((target_sink_rate_m_s - actual_sink_rate_m_s) * FLARE_SINK_GAIN_RAD_PER_M_S).clamp(
            -FLARE_SINK_CORRECTION_DOWN_RAD,
            FLARE_SINK_CORRECTION_UP_RAD,
        );
    let throttle_fraction = ((height_m - FLARE_IDLE_HEIGHT_M)
        / (FLARE_RETARD_START_HEIGHT_M - FLARE_IDLE_HEIGHT_M))
        .clamp(0.0, 1.0);
    FlareCommand {
        pitch_rad: pitch_feed_forward + sink_correction,
        throttle: approach_throttle.clamp(0.0, 1.0) * throttle_fraction,
    }
}

fn bounce_requires_go_around(phase: LandPhase, airborne_s: f64, airspeed_m_s: f64) -> bool {
    phase == LandPhase::Rollout
        && airborne_s >= BOUNCE_ESCAPE_CONFIRM_S
        && airspeed_m_s > STOP_SPEED_M_S
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
    fn the_recorded_touchdown_rebound_stays_committed_to_rollout() {
        // Session 10360-1785559021717: rollout began at 1785559481710 and the
        // old 0.35 s gate fired at 1785559482081, despite a clean centreline
        // touchdown and an imminent re-contact.
        assert!(!bounce_requires_go_around(LandPhase::Rollout, 0.371, 63.0));

        let demand = rollout_demand(0.2, true, 63.0);
        assert_eq!(demand.throttle, Some(0.0));
        assert_eq!(demand.wheel_brake, Some(1.0));
        assert_eq!(demand.ground_steer, Some(0.2));
        assert_eq!(
            demand.attitude,
            AttitudeDemand::FlightPath(PlaneHoldTarget {
                pitch_rad: pitch_for_vertical_speed(
                    TOUCHDOWN_SINK_RATE_M_S,
                    63.0,
                    APPROACH_ALPHA_BIAS_RAD,
                ),
                bank_rad: 0.0,
            })
        );
    }

    #[test]
    fn flare_preserves_the_final_command_at_entry() {
        let airspeed = 65.0;
        let approach_fpa = -3.0_f64.to_radians();
        let on_profile_sink = approach_fpa.sin() * airspeed;
        let command = flare_command(
            FLARE_HEIGHT_M,
            approach_fpa,
            on_profile_sink,
            airspeed,
            0.42,
        );
        assert!(
            (command.pitch_rad - (approach_fpa + APPROACH_ALPHA_BIAS_RAD)).abs()
                < 0.01_f64.to_radians()
        );
        assert!((command.throttle - 0.42).abs() < 1.0e-12);
    }

    #[test]
    fn recorded_hard_landing_state_commands_an_actual_arrest() {
        // Session 28324-1785560800045, one second before first contact: the old
        // height-only blend asked for ~5.1 degrees and idle power while the
        // Meridian was still descending at -2.82 m/s, then hit full stroke.
        let command = flare_command(5.92, -3.0_f64.to_radians(), -2.82, 64.3, 0.42);
        assert!(
            command.pitch_rad >= 7.0_f64.to_radians(),
            "pitch {} deg does not arrest the recorded sink",
            command.pitch_rad.to_degrees()
        );
        assert!(command.throttle > 0.08, "power was retarded too early");
        assert!(command.throttle < 0.42, "power must already be retarding");
        assert!(
            flare_target_sink_rate(5.92, -3.0_f64.to_radians(), 64.3) > -1.7,
            "scheduled sink must be materially gentler than the recorded arrival"
        );
    }

    #[test]
    fn flare_reaches_idle_and_a_shallow_descent_at_touchdown() {
        let airspeed = 64.0;
        let command = flare_command(
            0.0,
            -3.0_f64.to_radians(),
            TOUCHDOWN_SINK_RATE_M_S,
            airspeed,
            0.5,
        );
        assert_eq!(command.throttle, 0.0);
        assert_eq!(
            command.pitch_rad,
            pitch_for_vertical_speed(TOUCHDOWN_SINK_RATE_M_S, airspeed, APPROACH_ALPHA_BIAS_RAD,)
        );
    }

    #[test]
    fn sustained_post_touchdown_separation_still_goes_around() {
        assert!(!bounce_requires_go_around(
            LandPhase::Rollout,
            BOUNCE_ESCAPE_CONFIRM_S - 0.01,
            63.0
        ));
        assert!(bounce_requires_go_around(
            LandPhase::Rollout,
            BOUNCE_ESCAPE_CONFIRM_S,
            63.0
        ));
        assert!(!bounce_requires_go_around(
            LandPhase::Flare,
            BOUNCE_ESCAPE_CONFIRM_S,
            63.0
        ));
    }

    #[test]
    fn the_join_is_never_judged_against_the_runway_centreline() {
        // The recorded defect: a normal bank-limited intercept sits far off the
        // extended centreline by construction, so consulting the localizer
        // during the join calls every approach unstable.
        for phase in [
            LandPhase::Enroute,
            LandPhase::TerminalCapture,
            LandPhase::GoAround,
            LandPhase::Rollout,
        ] {
            assert_eq!(
                approach_out_of_tolerance(phase, Some(ApproachPhase::Transition), false, -2.0),
                None,
                "{phase:?} must not be judged against the beam"
            );
        }
    }

    #[test]
    fn on_final_and_off_the_beam_is_a_go_around() {
        assert_eq!(
            approach_out_of_tolerance(LandPhase::Final, Some(ApproachPhase::Final), false, -4.0),
            Some(LandNotice::GoAroundUnstable)
        );
        // Established on the same final is fine.
        assert_eq!(
            approach_out_of_tolerance(LandPhase::Final, Some(ApproachPhase::Final), true, -4.0),
            None
        );
    }

    #[test]
    fn an_excessive_sink_rate_is_a_go_around_even_on_the_beam() {
        assert_eq!(
            approach_out_of_tolerance(LandPhase::Final, Some(ApproachPhase::Final), true, -12.0),
            Some(LandNotice::GoAroundSinkRate)
        );
    }

    #[test]
    fn losing_guidance_does_not_read_as_an_unstable_approach() {
        // A vanished destination is UNABLE, not a go-around toward nothing —
        // and it must not be reported as the wrong reason.
        assert_eq!(
            approach_out_of_tolerance(LandPhase::Final, None, false, -4.0),
            None
        );
    }

    #[test]
    fn every_notice_says_something_specific() {
        // A reason the player cannot act on is the same as no reason, which is
        // the failure this whole channel exists to fix.
        for notice in [
            LandNotice::GoAroundUnstable,
            LandNotice::GoAroundSinkRate,
            LandNotice::GoAroundBounce,
            LandNotice::UnableGoAroundLimit,
            LandNotice::UnableNoGuidance,
            LandNotice::UnableLostRunway,
            LandNotice::UnableDestroyed,
            LandNotice::DisengagedByPilot,
            LandNotice::Completed,
        ] {
            assert!(!notice.label().is_empty());
            assert!(!notice.detail().is_empty());
            assert!(!notice.diagnostic_reason().is_empty());
            assert!(
                !notice.diagnostic_reason().contains(' '),
                "{:?} must be a snake_case lane key",
                notice
            );
        }
    }

    #[test]
    fn vertical_speed_pitch_includes_angle_of_attack_bias() {
        let level = pitch_for_vertical_speed(0.0, 100.0, APPROACH_ALPHA_BIAS_RAD);
        let descent = pitch_for_vertical_speed(-5.0, 100.0, APPROACH_ALPHA_BIAS_RAD);
        assert_eq!(level, APPROACH_ALPHA_BIAS_RAD);
        assert!(descent < level);
    }
}
