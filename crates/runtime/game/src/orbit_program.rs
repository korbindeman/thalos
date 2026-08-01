//! ORBIT target state and program ownership.
//!
//! The top-centre orbit widget publishes [`OrbitTargetRequest`] messages.
//! This module is their sole consumer and the sole writer of the selected
//! target, generated-node provenance, and ORBIT phase.

use bevy::{math::DVec3, prelude::*};
use thalos_control::{AttitudeDemand, ControlDemand};
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::canonical::AuthorityMode;
use thalos_physics_canonical::orbit_planner::{
    OrbitDirection, OrbitPlan, OrbitPlanError, OrbitPlanRequest,
    plan_target_orbit,
};
use thalos_world::StateVector;

use crate::fuel::{ActivePropulsion, PilotThrottleInput, ThrottleState};
use crate::maneuver::{GameNode, ManeuverPlan, NodeBurnPhase, NodeSource};
use crate::rendering::{SimulationState, SolarSystemState};
use crate::sim_clock::SimClock;
use crate::stage_sequencer::{StageCommand, StageSequencer, StageSequencerInput};
use crate::staging::{StageDemand, StagingSummaries};

pub use thalos_game_state::autoflight::SequenceEvent;
pub use thalos_game_state::nav::{
    MIN_ORBIT_ALTITUDE_M,
    OrbitDraft, OrbitPlanSummary, OrbitPlaneChoice, OrbitProgram, OrbitProgramPhase,
    OrbitShape, OrbitTargetRequest,
};

const ALTITUDE_STEP_M: f64 = 10_000.0;
const INCLINATION_STEP_RAD: f64 = 5.0_f64.to_radians();
const PILOT_OVERRIDE_DEADZONE_SQ: f32 = 0.05 * 0.05;
const MIN_LAUNCH_TWR: f64 = 1.05;
const RISE_HEIGHT_M: f64 = 500.0;
const TURN_END_HEIGHT_M: f64 = 15_000.0;
const MAX_DYNAMIC_PRESSURE_PA: f64 = 35_000.0;
const MAX_ASCENT_ACCELERATION_M_S2: f64 = 4.0 * 9.806_65;
const MECO_APOAPSIS_MARGIN_M: f64 = 2_000.0;
const PREFLIGHT_POINTING_COS: f64 = 0.984_807_753; // 10°
const PREFLIGHT_POINTING_TIMEOUT_S: f64 = 5.0;
/// Reference normal used by `orbital_math::cartesian_to_elements`.
///
/// This is deliberately inertial +Y, not the body's tilted local pole:
/// inclination is measured in the canonical XZ reference plane.
const ORBIT_REFERENCE_NORMAL: DVec3 = DVec3::Y;













pub struct OrbitProgramPlugin;

impl Plugin for OrbitProgramPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<OrbitProgram>()
            .add_message::<OrbitTargetRequest>()
            .add_systems(
                Update,
                handle_orbit_target_requests.before(crate::SimStage::Physics),
            )
            .add_systems(
                Update,
                (
                    update_surface_orbit_program
                        .after(crate::fuel::handle_throttle_input)
                        .before(crate::staging::activate_stage)
                        .before(crate::controls::update_control_locks)
                        .before(crate::control_bus::realize_control),
                    monitor_orbit_maneuver_program.after(crate::autopilot::autopilot_system),
                    apply_orbit_idle_handoff
                        .after(update_surface_orbit_program)
                        .after(monitor_orbit_maneuver_program)
                        .before(crate::controls::update_control_locks),
                )
                    .in_set(crate::SimStage::Physics),
            );
    }
}

fn handle_orbit_target_requests(
    mut requests: MessageReader<OrbitTargetRequest>,
    mut program: ResMut<OrbitProgram>,
    mut maneuver_plan: ResMut<ManeuverPlan>,
    mut stage_demand: ResMut<StageDemand>,
    mut sequencer: ResMut<StageSequencer>,
    mut throttle: ResMut<ThrottleState>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
) {
    for request in requests.read().copied() {
        let was_active = program.active();
        sequencer.cancel(&mut stage_demand);
        match request {
            OrbitTargetRequest::ToggleShape => {
                program.draft.shape = match program.draft.shape {
                    OrbitShape::Circular => OrbitShape::Elliptical,
                    OrbitShape::Elliptical => OrbitShape::Circular,
                };
                invalidate_preview(&mut program, &mut maneuver_plan);
            }
            OrbitTargetRequest::AdjustPeriapsis(direction) => {
                program.draft.periapsis_altitude_m += f64::from(direction) * ALTITUDE_STEP_M;
                if program.draft.shape == OrbitShape::Circular {
                    program.draft.apoapsis_altitude_m = program.draft.periapsis_altitude_m;
                }
                program.draft.normalize();
                invalidate_preview(&mut program, &mut maneuver_plan);
            }
            OrbitTargetRequest::AdjustApoapsis(direction) => {
                program.draft.apoapsis_altitude_m += f64::from(direction) * ALTITUDE_STEP_M;
                if program.draft.shape == OrbitShape::Circular {
                    program.draft.periapsis_altitude_m = program.draft.apoapsis_altitude_m;
                }
                program.draft.normalize();
                invalidate_preview(&mut program, &mut maneuver_plan);
            }
            OrbitTargetRequest::AdjustInclination(direction) => {
                program.draft.plane = OrbitPlaneChoice::Nearest;
                program.draft.inclination_rad += f64::from(direction) * INCLINATION_STEP_RAD;
                program.draft.normalize();
                invalidate_preview(&mut program, &mut maneuver_plan);
            }
            OrbitTargetRequest::ToggleDirection => {
                program.draft.plane = OrbitPlaneChoice::Nearest;
                program.draft.direction = match program.draft.direction {
                    OrbitDirection::Prograde => OrbitDirection::Retrograde,
                    OrbitDirection::Retrograde => OrbitDirection::Prograde,
                };
                invalidate_preview(&mut program, &mut maneuver_plan);
            }
            OrbitTargetRequest::TogglePlane => {
                program.draft.plane = match program.draft.plane {
                    OrbitPlaneChoice::Auto => OrbitPlaneChoice::Nearest,
                    OrbitPlaneChoice::Nearest => OrbitPlaneChoice::Preserve,
                    OrbitPlaneChoice::Preserve => OrbitPlaneChoice::Auto,
                };
                invalidate_preview(&mut program, &mut maneuver_plan);
            }
            OrbitTargetRequest::Plan => {
                plan_current_target(
                    &mut program,
                    &mut maneuver_plan,
                    &sim,
                    &solar,
                    false,
                );
            }
            OrbitTargetRequest::Execute => {
                plan_current_target(
                    &mut program,
                    &mut maneuver_plan,
                    &sim,
                    &solar,
                    true,
                );
            }
            OrbitTargetRequest::Cancel => {
                clear_program_nodes(&program, &mut maneuver_plan);
                program.phase = OrbitProgramPhase::Idle;
                program.summary = None;
                program.error = None;
                program.surface_program = false;
                program.demand = ControlDemand::NONE;
                program.sequence = SequenceEvent::None;
                program.within_tolerance_s = 0.0;
                program.target_plane_normal = DVec3::ZERO;
                program.program_id = program.program_id.wrapping_add(1).max(1);
            }
        }
        if was_active && !program.active() {
            throttle.selected = 0.0;
            throttle.hold_idle_until_pilot_move = true;
            program.idle_handoff_pending = false;
        }
    }
}

fn invalidate_preview(program: &mut OrbitProgram, maneuver_plan: &mut ManeuverPlan) {
    clear_program_nodes(program, maneuver_plan);
    program.phase = OrbitProgramPhase::Idle;
    program.summary = None;
    program.error = None;
    program.surface_program = false;
    program.demand = ControlDemand::NONE;
    program.sequence = SequenceEvent::None;
    program.within_tolerance_s = 0.0;
    program.target_plane_normal = DVec3::ZERO;
}

fn plan_current_target(
    program: &mut OrbitProgram,
    maneuver_plan: &mut ManeuverPlan,
    sim: &SimulationState,
    solar: &SolarSystemState,
    execute: bool,
) {
    clear_program_nodes(program, maneuver_plan);
    program.idle_handoff_pending = false;
    program.error = None;
    program.summary = None;
    program.draft.normalize();

    let body_id = sim.simulation.dominant_body();
    let Some(body) = sim.system.bodies.get(body_id) else {
        fail_program(program, "dominant body is unavailable");
        return;
    };
    let Some(body_state) = solar
        .states
        .as_deref()
        .and_then(|states| states.get(body_id))
    else {
        fail_program(program, "body state is unavailable");
        return;
    };
    let ship = sim.simulation.ship_state();
    let relative = StateVector {
        position: ship.position - body_state.position,
        velocity: ship.velocity - body_state.velocity,
    };
    let altitude_m = relative.position.length() - body.radius_m;
    let atmosphere_top_m = body
        .terrestrial_atmosphere
        .as_ref()
        .map_or(0.0, |atmosphere| atmosphere.karman_line_m as f64);
    let surface_program = matches!(sim.simulation.authority(), AuthorityMode::BodyFixed { .. })
        || altitude_m <= atmosphere_top_m.max(5_000.0);

    program.target_body = Some(body_id);
    program.surface_program = surface_program;
    if surface_program {
        let launch_inclination_rad = launch_inclination(program.draft, relative.position);
        let launch_direction = launch_direction(program.draft);
        program.launch_altitude_m = altitude_m;
        program.phase_started_s = sim.simulation.sim_time();
        program.sequence = SequenceEvent::None;
        program.diagnostic_s = 0.0;
        program.within_tolerance_s = 0.0;
        program.target_plane_normal = DVec3::ZERO;
        program.phase = if execute {
            OrbitProgramPhase::Preflight
        } else {
            OrbitProgramPhase::Planned
        };
        program.summary = Some(OrbitPlanSummary {
            node_count: 0,
            total_delta_v_m_s: estimate_surface_delta_v(body.gm, body.radius_m, program.draft),
            predicted_periapsis_altitude_m: program.draft.periapsis_altitude_m,
            predicted_apoapsis_altitude_m: program.draft.apoapsis_altitude_m,
            predicted_inclination_rad: resolved_inclination(
                launch_inclination_rad,
                launch_direction,
            ),
        });
        return;
    }

    match plan_target_orbit(OrbitPlanRequest {
        state: relative,
        epoch_s: sim.simulation.sim_time(),
        mu: body.gm,
        body_radius_m: body.radius_m,
        target: program.draft.target(body_id),
    }) {
        Ok(plan) => install_orbit_plan(
            program,
            maneuver_plan,
            body.radius_m,
            plan,
            execute,
        ),
        Err(error) => {
            fail_program(program, &plan_error_label(&error));
            info!(
                target: "thalos::diagnostic::orbit_autoflight",
                event = "orbit_plan_result",
                outcome = "error",
                error = ?error,
                body_id,
                "ORBIT plan rejected"
            );
        }
    }
}

fn install_orbit_plan(
    program: &mut OrbitProgram,
    maneuver_plan: &mut ManeuverPlan,
    body_radius_m: f64,
    plan: OrbitPlan,
    execute: bool,
) {
    for node in &plan.nodes {
        let id = maneuver_plan.next_node_id();
        maneuver_plan.nodes.push(GameNode {
            id,
            time: node.time,
            delta_v: node.delta_v,
            reference_body: node.reference_body,
            phase: NodeBurnPhase::Planned,
            source: NodeSource::OrbitProgram(program.program_id),
        });
    }
    maneuver_plan.dirty = true;
    program.summary = Some(OrbitPlanSummary {
        node_count: plan.nodes.len(),
        total_delta_v_m_s: plan.total_delta_v_m_s,
        predicted_periapsis_altitude_m: plan.predicted_elements.periapsis_m - body_radius_m,
        predicted_apoapsis_altitude_m: plan.predicted_elements.apoapsis_m - body_radius_m,
        predicted_inclination_rad: plan.predicted_elements.inclination_rad,
    });
    program.phase = if execute {
        OrbitProgramPhase::Coast
    } else {
        OrbitProgramPhase::Planned
    };
    info!(
        target: "thalos::diagnostic::orbit_autoflight",
        event = "orbit_plan_result",
        outcome = "ok",
        node_count = plan.nodes.len(),
        total_delta_v_m_s = plan.total_delta_v_m_s,
        "ORBIT plan generated"
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_surface_orbit_program(
    mut program: ResMut<OrbitProgram>,
    mut maneuver_plan: ResMut<ManeuverPlan>,
    mut stage_demand: ResMut<StageDemand>,
    mut sequencer: ResMut<StageSequencer>,
    propulsion: Res<ActivePropulsion>,
    staging: Res<StagingSummaries>,
    input: Res<GameInputIntent>,
    pilot_throttle: Res<PilotThrottleInput>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    clock: Res<SimClock>,
) {
    if !program.guidance_active() {
        program.demand = ControlDemand::NONE;
        program.sequence = SequenceEvent::None;
        return;
    }
    if input.attitude.length_squared() > PILOT_OVERRIDE_DEADZONE_SQ || pilot_throttle.moved {
        sequencer.cancel(&mut stage_demand);
        abort_surface_program(&mut program, "pilot_override");
        return;
    }
    if sim.simulation.is_destroyed() {
        sequencer.cancel(&mut stage_demand);
        abort_surface_program(&mut program, "craft_destroyed");
        return;
    }

    let Some(body_id) = program.target_body else {
        sequencer.cancel(&mut stage_demand);
        abort_surface_program(&mut program, "target_body_unavailable");
        return;
    };
    let Some(body) = sim.system.bodies.get(body_id) else {
        sequencer.cancel(&mut stage_demand);
        abort_surface_program(&mut program, "target_body_unavailable");
        return;
    };
    let Some(body_state) = solar
        .states
        .as_deref()
        .and_then(|states| states.get(body_id))
    else {
        program.demand = ControlDemand::NONE;
        return;
    };
    let ship = sim.simulation.ship_state();
    let relative_position = ship.position - body_state.position;
    let radius_m = relative_position.length();
    if radius_m <= body.radius_m * 0.5 {
        abort_surface_program(&mut program, "invalid_surface_position");
        return;
    }
    let up = relative_position / radius_m;
    let altitude_m = radius_m - body.radius_m;
    let surface_gravity_m_s2 = body.gm / radius_m.powi(2);
    let latitude_rad = orbital_latitude(relative_position);
    let launch_inclination_rad = launch_inclination(program.draft, relative_position);
    let launch_direction = launch_direction(program.draft);

    // Everything knowable without lighting an engine is checked first. A
    // cold staged vessel reports no ActivePropulsion by design, so the staged
    // summary is the pre-ignition source for TWR and delta-v.
    if program.phase == OrbitProgramPhase::Preflight {
        let atmosphere_top_m = body
            .terrestrial_atmosphere
            .as_ref()
            .map_or(0.0, |atmosphere| atmosphere.karman_line_m as f64);
        if program.draft.periapsis_altitude_m <= atmosphere_top_m.max(MIN_ORBIT_ALTITUDE_M) {
            abort_surface_program(&mut program, "target_orbit_not_safe");
            return;
        }
        if launch_inclination_rad + 1.0e-6 < latitude_rad.abs() {
            warn!(
                target: "thalos::diagnostic::orbit_autoflight",
                event = "orbit_launch_geometry",
                outcome = "unreachable",
                launch_latitude_rad = latitude_rad,
                requested_inclination_rad = launch_inclination_rad,
                plane_policy = ?program.draft.plane,
                "ORBIT launch plane is unreachable"
            );
            abort_surface_program(
                &mut program,
                "inclination_unreachable_from_launch_site",
            );
            return;
        }
        if staging.0.is_empty() {
            if sim.simulation.sim_time() - program.phase_started_s > 1.0 {
                abort_surface_program(&mut program, "no_staging_plan");
            } else {
                program.demand = ControlDemand::NONE;
            }
            return;
        }
        let Some(first_powered_stage) = staging.0.iter().find(|stage| stage.has_engine) else {
            abort_surface_program(&mut program, "no_powered_stage");
            return;
        };
        let staged_twr = first_powered_stage.thrust_n
            / (first_powered_stage.initial_mass_kg.max(1.0) * surface_gravity_m_s2.max(1.0e-6));
        if staged_twr < MIN_LAUNCH_TWR {
            abort_surface_program(
                &mut program,
                "insufficient_thrust_to_weight",
            );
            return;
        }
        let available_delta_v_m_s: f64 = staging.0.iter().map(|stage| stage.delta_v_m_s).sum();
        let required_delta_v_m_s = estimate_surface_delta_v(body.gm, body.radius_m, program.draft);
        if available_delta_v_m_s < required_delta_v_m_s * 0.9 {
            abort_surface_program(&mut program, "insufficient_delta_v");
            return;
        }
        let params = sim.simulation.ship_params();
        let pitch_authority = params.max_torque.x + params.gimbal_torque_full.x;
        let yaw_authority = params.max_torque.z + params.gimbal_torque_full.z;
        if pitch_authority <= 0.0 || yaw_authority <= 0.0 {
            if sim.simulation.sim_time() - program.phase_started_s > 1.0 {
                abort_surface_program(&mut program, "no_attitude_authority");
            } else {
                program.demand = ControlDemand::NONE;
            }
            return;
        }
        let nose = sim.simulation.attitude().orientation * DVec3::Y;
        if nose.dot(up) < PREFLIGHT_POINTING_COS {
            program.demand =
                ControlDemand::autoflight(AttitudeDemand::PointNose(up), Some(0.0), None, None);
            if sim.simulation.sim_time() - program.phase_started_s > PREFLIGHT_POINTING_TIMEOUT_S {
                abort_surface_program(&mut program, "unable_to_align_for_launch");
            }
            return;
        }
    }

    // --- Staging sequence ---
    //
    // Advanced every guidance frame, and it only ever takes *throttle*.
    // Guidance keeps steering through cutoff, separation, and ignition,
    // which is the substantive change from the path this replaces: that one
    // waited for thrust to collapse, then pointed the vehicle at local up
    // and held zero throttle until an acknowledgement arrived, throwing the
    // ascent off its gravity turn at every staging event.
    //
    // Burnout is predicted from the throttle guidance commanded *last*
    // frame. That is the honest available value here — this frame's throttle
    // is computed below, from the atmosphere and the apoapsis error — and at
    // 60 Hz a one-frame-old throttle tracks a changing setting far inside
    // the cutoff lead.
    let commanded_throttle = program.demand.throttle.unwrap_or(0.0);
    let active_stage_fuel_kg = staging
        .0
        .iter()
        .find(|stage| stage.active)
        .map_or(0.0, |stage| stage.fuel_kg);
    let stage_command = sequencer.update(
        StageSequencerInput {
            now_s: sim.simulation.sim_time(),
            active_stage_fuel_kg,
            mass_flow_full_kg_per_s: propulsion.mass_flow_kg_per_s,
            commanded_throttle,
            total_thrust_n: propulsion.total_thrust_n,
            angular_rate_rad_s: sim.simulation.attitude().angular_velocity.length(),
            stage_available: staging.0.iter().any(|stage| !stage.active),
        },
        &mut stage_demand,
    );
    if stage_command == StageCommand::Exhausted {
        abort_surface_program(&mut program, "no_stage_available");
        return;
    }
    program.sequence = match sequencer.armed_in_s(sim.simulation.sim_time()) {
        Some(in_s) => SequenceEvent::Staging {
            stage_index: sequencer.completed_events as usize + 1,
            in_s,
        },
        None => SequenceEvent::None,
    };

    if launch_inclination_rad + 1.0e-6 < latitude_rad.abs() {
        abort_surface_program(&mut program, "inclination_unreachable_from_launch_site");
        return;
    }
    let Some(launch_heading) = launch_heading(
        up,
        ORBIT_REFERENCE_NORMAL,
        latitude_rad,
        launch_inclination_rad,
        launch_direction,
    ) else {
        abort_surface_program(&mut program, "launch_heading_undefined");
        return;
    };
    if program.target_plane_normal.length_squared() <= 1.0e-12 {
        program.target_plane_normal = up.cross(launch_heading).normalize();
    }
    let plane_heading = program
        .target_plane_normal
        .cross(up)
        .try_normalize()
        .unwrap_or(launch_heading);

    if program.phase == OrbitProgramPhase::Preflight {
        let live_twr = propulsion.total_thrust_n
            / (propulsion.wet_mass_kg.max(1.0) * surface_gravity_m_s2.max(1.0e-6));
        if live_twr < MIN_LAUNCH_TWR {
            abort_surface_program(
                &mut program,
                "insufficient_live_thrust_to_weight",
            );
            return;
        }
        set_orbit_phase(
            &mut program,
            OrbitProgramPhase::Rise,
            sim.simulation.sim_time(),
        );
    }

    let relative_velocity = ship.velocity - body_state.velocity;
    let surface_relative_velocity =
        relative_velocity - body_state.angular_velocity.cross(relative_position);
    let horizontal_velocity = relative_velocity - up * relative_velocity.dot(up);
    let target_apoapsis_altitude_m = program.draft.apoapsis_altitude_m;
    let elements = thalos_physics_canonical::orbital_math::cartesian_to_elements(
        StateVector {
            position: relative_position,
            velocity: relative_velocity,
        },
        body.gm,
    );
    let current_apoapsis_altitude_m = elements
        .filter(|elements| elements.eccentricity < 1.0)
        .map(|elements| elements.apoapsis_m - body.radius_m);
    if current_apoapsis_altitude_m
        .is_some_and(|apoapsis| apoapsis >= target_apoapsis_altitude_m - MECO_APOAPSIS_MARGIN_M)
        && altitude_m > program.launch_altitude_m + RISE_HEIGHT_M
    {
        set_orbit_phase(
            &mut program,
            OrbitProgramPhase::MainEngineCutoff,
            sim.simulation.sim_time(),
        );
        program.demand = ControlDemand::autoflight(
            AttitudeDemand::PointNose(plane_heading),
            Some(0.0),
            None,
            None,
        );
        let request = OrbitPlanRequest {
            state: StateVector {
                position: relative_position,
                velocity: relative_velocity,
            },
            epoch_s: sim.simulation.sim_time(),
            mu: body.gm,
            body_radius_m: body.radius_m,
            target: program.draft.target(body_id),
        };
        match plan_target_orbit(request) {
            Ok(plan) => {
                install_orbit_plan(
                    &mut program,
                    &mut maneuver_plan,
                    body.radius_m,
                    plan,
                    true,
                );
                program.surface_program = false;
                program.demand = ControlDemand::NONE;
            }
            Err(error) => {
                abort_surface_program(
                    &mut program,
                    &format!("circularization_plan_failed:{}", plan_error_label(&error)),
                );
            }
        }
        return;
    }

    let height_since_launch_m = altitude_m - program.launch_altitude_m;
    let (attitude_target, phase) = if height_since_launch_m < RISE_HEIGHT_M {
        (up, OrbitProgramPhase::Rise)
    } else if height_since_launch_m < TURN_END_HEIGHT_M {
        let fraction = ((height_since_launch_m - RISE_HEIGHT_M)
            / (TURN_END_HEIGHT_M - RISE_HEIGHT_M))
            .clamp(0.0, 1.0);
        let smooth = fraction * fraction * (3.0 - 2.0 * fraction);
        (
            (up * (1.0 - smooth) + plane_heading * smooth).normalize(),
            OrbitProgramPhase::Turn,
        )
    } else {
        let horizontal = if horizontal_velocity.length() > 50.0 {
            (plane_heading * 0.75 + horizontal_velocity.normalize() * 0.25).normalize()
        } else {
            plane_heading
        };
        let apoapsis_error_fraction = current_apoapsis_altitude_m
            .map(|apoapsis| {
                ((target_apoapsis_altitude_m - apoapsis) / target_apoapsis_altitude_m.max(1.0))
                    .clamp(0.0, 1.0)
            })
            .unwrap_or(1.0);
        let pitch_rad = 3.0_f64.to_radians() + apoapsis_error_fraction * 22.0_f64.to_radians();
        (
            (horizontal * pitch_rad.cos() + up * pitch_rad.sin()).normalize(),
            OrbitProgramPhase::Ascent,
        )
    };
    set_orbit_phase(&mut program, phase, sim.simulation.sim_time());

    let density_kg_m3 = body
        .terrestrial_atmosphere
        .as_ref()
        .map(|atmosphere| {
            atmosphere
                .sample_at_altitude_m(altitude_m, body.surface_pressure_pa(), surface_gravity_m_s2)
                .density_kg_m3
        })
        .unwrap_or(0.0);
    let dynamic_pressure_pa = 0.5 * density_kg_m3 * surface_relative_velocity.length_squared();
    let full_acceleration_m_s2 = propulsion.total_thrust_n / propulsion.wet_mass_kg.max(1.0);
    let guidance_throttle = ascent_throttle(dynamic_pressure_pa, full_acceleration_m_s2);
    // The staging sequence takes throttle and nothing else. Attitude stays
    // with guidance throughout, so the vehicle holds its pitch program
    // across cutoff, separation, and ignition instead of pitching to local
    // up and losing the turn.
    let throttle = match stage_command {
        StageCommand::HoldThrottleClosed => 0.0,
        StageCommand::Free | StageCommand::Exhausted => guidance_throttle,
    };
    program.demand = ControlDemand::autoflight(
        AttitudeDemand::PointNose(attitude_target),
        Some(throttle),
        None,
        None,
    );

    program.diagnostic_s += clock.delta_secs_f64();
    if program.diagnostic_s >= 1.0 {
        program.diagnostic_s = 0.0;
        let orbital_normal = relative_position
            .cross(relative_velocity)
            .try_normalize()
            .unwrap_or(program.target_plane_normal);
        let plane_error_rad = orbital_normal
            .dot(program.target_plane_normal)
            .clamp(-1.0, 1.0)
            .acos();
        info!(
            target: "thalos::diagnostic::orbit_autoflight",
            event = "orbit_autoflight_guidance",
            phase = ?program.phase,
            altitude_m,
            apoapsis_altitude_m = current_apoapsis_altitude_m.unwrap_or(f64::NAN),
            apoapsis_error_m = current_apoapsis_altitude_m
                .map_or(f64::NAN, |apoapsis| target_apoapsis_altitude_m - apoapsis),
            plane_error_rad,
            dynamic_pressure_pa,
            dynamic_pressure_limit_pa = MAX_DYNAMIC_PRESSURE_PA,
            acceleration_m_s2 = full_acceleration_m_s2 * throttle,
            acceleration_limit_m_s2 = MAX_ASCENT_ACCELERATION_M_S2,
            throttle,
            twr = propulsion.total_thrust_n
                / (propulsion.wet_mass_kg.max(1.0) * surface_gravity_m_s2.max(1.0e-6)),
            "ORBIT ascent state"
        );
    }
}

fn set_orbit_phase(program: &mut OrbitProgram, next: OrbitProgramPhase, now_s: f64) {
    if program.phase == next {
        return;
    }
    info!(
        target: "thalos::diagnostic::orbit_autoflight",
        event = "orbit_autoflight_transition",
        from = ?program.phase,
        to = ?next,
        "ORBIT phase transition"
    );
    program.phase = next;
    program.phase_started_s = now_s;
}

fn launch_heading(
    up: DVec3,
    spin_axis: DVec3,
    latitude_rad: f64,
    inclination_rad: f64,
    direction: OrbitDirection,
) -> Option<DVec3> {
    if inclination_rad + 1.0e-6 < latitude_rad.abs() {
        return None;
    }
    let north = (spin_axis - up * spin_axis.dot(up)).normalize_or_zero();
    if north.length_squared() <= 1.0e-12 {
        return None;
    }
    // `cartesian_to_elements` defines a zero-degree prograde orbit with
    // angular momentum along -Y. At up=+X that requires velocity=+Z, hence
    // east is `up × north`.
    let east = up.cross(north).normalize();
    let east_component =
        (inclination_rad.cos() / latitude_rad.cos().abs().max(1.0e-6)).clamp(-1.0, 1.0);
    let north_component = (1.0 - east_component * east_component).sqrt();
    let direction_sign = match direction {
        OrbitDirection::Prograde => 1.0,
        OrbitDirection::Retrograde => -1.0,
    };
    Some((north * north_component + east * east_component * direction_sign).normalize())
}

fn ascent_throttle(dynamic_pressure_pa: f64, full_acceleration_m_s2: f64) -> f64 {
    let q_throttle = if dynamic_pressure_pa > MAX_DYNAMIC_PRESSURE_PA {
        (MAX_DYNAMIC_PRESSURE_PA / dynamic_pressure_pa).clamp(0.2, 1.0)
    } else {
        1.0
    };
    let acceleration_throttle = if full_acceleration_m_s2 > MAX_ASCENT_ACCELERATION_M_S2 {
        MAX_ASCENT_ACCELERATION_M_S2 / full_acceleration_m_s2
    } else {
        1.0
    };
    q_throttle.min(acceleration_throttle).clamp(0.0, 1.0)
}

fn abort_surface_program(program: &mut OrbitProgram, reason: &str) {
    program.phase = OrbitProgramPhase::Abort;
    program.error = Some(reason.replace('_', " "));
    program.demand = ControlDemand::NONE;
    program.sequence = SequenceEvent::None;
    program.idle_handoff_pending = true;
    warn!(
        target: "thalos::diagnostic::orbit_autoflight",
        event = "orbit_autoflight_abort",
        reason,
        "ORBIT ascent aborted"
    );
}

fn apply_orbit_idle_handoff(
    mut program: ResMut<OrbitProgram>,
    mut throttle: ResMut<ThrottleState>,
) {
    if !program.idle_handoff_pending {
        return;
    }
    throttle.selected = 0.0;
    throttle.hold_idle_until_pilot_move = true;
    program.idle_handoff_pending = false;
}

fn monitor_orbit_maneuver_program(
    mut program: ResMut<OrbitProgram>,
    maneuver_plan: Res<ManeuverPlan>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    clock: Res<SimClock>,
) {
    // Gated on the program's own phase, not on a selected mode. The mode
    // check this replaces early-returned whenever anything else touched the
    // autopilot selection - leaving the program stranded in COAST forever
    // while its widget still claimed to be running.
    if !program.active() || program.surface_program {
        return;
    }
    let relevant: Vec<_> = maneuver_plan
        .nodes
        .iter()
        .filter(|node| node.source == NodeSource::OrbitProgram(program.program_id))
        .collect();
    if relevant
        .iter()
        .any(|node| !matches!(node.phase, NodeBurnPhase::Executed))
    {
        program.phase = if relevant
            .iter()
            .any(|node| matches!(node.phase, NodeBurnPhase::Executing))
        {
            OrbitProgramPhase::Circularize
        } else {
            OrbitProgramPhase::Coast
        };
        return;
    }
    match achieved_target_errors(&program, &sim, &solar) {
        Some((pe_error_m, ap_error_m, inclination_error_rad))
            if pe_error_m <= 2_000.0 && ap_error_m <= 2_000.0 && inclination_error_rad <= 0.01 =>
        {
            program.phase = OrbitProgramPhase::Trim;
            program.within_tolerance_s += clock.delta_secs_f64().clamp(0.0, 0.1);
            if program.within_tolerance_s < 1.0 {
                return;
            }
            program.phase = OrbitProgramPhase::Complete;
            program.error = None;
            program.idle_handoff_pending = true;
            info!(
                target: "thalos::diagnostic::orbit_autoflight",
                event = "orbit_autoflight_complete",
                periapsis_error_m = pe_error_m,
                apoapsis_error_m = ap_error_m,
                inclination_error_rad,
                "ORBIT complete"
            );
        }
        Some((pe_error_m, ap_error_m, inclination_error_rad)) => {
            program.within_tolerance_s = 0.0;
            program.phase = OrbitProgramPhase::Abort;
            program.error = Some("final orbit is outside tolerance".to_string());
            program.idle_handoff_pending = true;
            warn!(
                target: "thalos::diagnostic::orbit_autoflight",
                event = "orbit_autoflight_abort",
                reason = "final_orbit_outside_tolerance",
                periapsis_error_m = pe_error_m,
                apoapsis_error_m = ap_error_m,
                inclination_error_rad,
                "ORBIT final validation failed"
            );
        }
        None => {}
    }
}

fn achieved_target_errors(
    program: &OrbitProgram,
    sim: &SimulationState,
    solar: &SolarSystemState,
) -> Option<(f64, f64, f64)> {
    let body_id = program.target_body?;
    let body = sim.system.bodies.get(body_id)?;
    let body_state = solar.states.as_deref()?.get(body_id)?;
    let ship = sim.simulation.ship_state();
    let elements = thalos_physics_canonical::orbital_math::cartesian_to_elements(
        StateVector {
            position: ship.position - body_state.position,
            velocity: ship.velocity - body_state.velocity,
        },
        body.gm,
    )?;
    let target_inclination_rad = program
        .summary
        .as_ref()
        .map_or(elements.inclination_rad, |summary| {
            summary.predicted_inclination_rad
        });
    Some((
        (elements.periapsis_m - body.radius_m - program.draft.periapsis_altitude_m).abs(),
        (elements.apoapsis_m - body.radius_m - program.draft.apoapsis_altitude_m).abs(),
        (elements.inclination_rad - target_inclination_rad).abs(),
    ))
}

fn clear_program_nodes(program: &OrbitProgram, maneuver_plan: &mut ManeuverPlan) {
    let old_len = maneuver_plan.nodes.len();
    maneuver_plan.nodes.retain(|node| {
        node.source != NodeSource::OrbitProgram(program.program_id)
            || matches!(node.phase, NodeBurnPhase::Executed)
    });
    if maneuver_plan.nodes.len() != old_len {
        maneuver_plan.dirty = true;
    }
}

fn fail_program(program: &mut OrbitProgram, reason: &str) {
    program.phase = OrbitProgramPhase::Abort;
    program.error = Some(reason.to_string());
    program.demand = ControlDemand::NONE;
}

fn orbital_latitude(position: DVec3) -> f64 {
    position
        .normalize_or_zero()
        .dot(ORBIT_REFERENCE_NORMAL)
        .clamp(-1.0, 1.0)
        .asin()
}

fn launch_inclination(draft: OrbitDraft, position: DVec3) -> f64 {
    match draft.plane {
        OrbitPlaneChoice::Auto | OrbitPlaneChoice::Preserve => orbital_latitude(position).abs(),
        OrbitPlaneChoice::Nearest => draft.inclination_rad,
    }
}

fn launch_direction(draft: OrbitDraft) -> OrbitDirection {
    match draft.plane {
        OrbitPlaneChoice::Auto => OrbitDirection::Prograde,
        OrbitPlaneChoice::Preserve | OrbitPlaneChoice::Nearest => draft.direction,
    }
}

fn resolved_inclination(inclination_rad: f64, direction: OrbitDirection) -> f64 {
    match direction {
        OrbitDirection::Prograde => inclination_rad,
        OrbitDirection::Retrograde => std::f64::consts::PI - inclination_rad,
    }
}

fn estimate_surface_delta_v(mu: f64, body_radius_m: f64, draft: OrbitDraft) -> f64 {
    let target_radius_m = body_radius_m + draft.apoapsis_altitude_m;
    let orbital_speed_m_s = (mu / target_radius_m).sqrt();
    let gravity_drag_reserve_m_s = 1_800.0;
    orbital_speed_m_s + gravity_drag_reserve_m_s
}

fn plan_error_label(error: &OrbitPlanError) -> String {
    match error {
        OrbitPlanError::InvalidGravity => "invalid body gravity",
        OrbitPlanError::InvalidBodyRadius => "invalid body radius",
        OrbitPlanError::InvalidTargetAltitude => "invalid target altitude",
        OrbitPlanError::PeriapsisAboveApoapsis => "periapsis is above apoapsis",
        OrbitPlanError::TargetIntersectsBody => "target orbit intersects the body",
        OrbitPlanError::DegenerateState => "current orbit cannot be solved",
        OrbitPlanError::CurrentOrbitNotBound => "current trajectory is not a bound orbit",
        OrbitPlanError::InvalidInclination => "inclination must be between 0° and 90°",
        OrbitPlanError::FixedPlaneUnsupported => "fixed-plane targeting is not available yet",
        OrbitPlanError::NoSafeTransfer => "no safe transfer was found",
        OrbitPlanError::ValidationFailed { .. } => "generated plan failed validation",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circular_adjustment_keeps_apsides_equal() {
        let mut draft = OrbitDraft::default();
        draft.apoapsis_altitude_m += ALTITUDE_STEP_M;
        draft.periapsis_altitude_m = draft.apoapsis_altitude_m;
        draft.normalize();
        assert_eq!(draft.periapsis_altitude_m, draft.apoapsis_altitude_m);
    }

    #[test]
    fn elliptical_draft_orders_apsides() {
        let mut draft = OrbitDraft {
            shape: OrbitShape::Elliptical,
            periapsis_altitude_m: 500_000.0,
            apoapsis_altitude_m: 200_000.0,
            ..default()
        };
        draft.normalize();
        assert_eq!(draft.periapsis_altitude_m, 200_000.0);
        assert_eq!(draft.apoapsis_altitude_m, 500_000.0);
    }

    #[test]
    fn equatorial_launch_heading_tracks_direction() {
        let up = DVec3::X;
        let spin = DVec3::Y;
        let prograde = launch_heading(up, spin, 0.0, 0.0, OrbitDirection::Prograde).unwrap();
        let retrograde = launch_heading(up, spin, 0.0, 0.0, OrbitDirection::Retrograde).unwrap();

        assert!((prograde - DVec3::Z).length() < 1.0e-12);
        assert!((retrograde + DVec3::Z).length() < 1.0e-12);
        assert!(DVec3::X.cross(prograde).dot(-DVec3::Y) > 1.0 - 1.0e-12);
        assert!(DVec3::X.cross(retrograde).dot(DVec3::Y) > 1.0 - 1.0e-12);
    }

    #[test]
    fn auto_selects_the_minimum_reachable_prograde_plane() {
        let latitude = 7.6_f64.to_radians();
        let position = DVec3::new(latitude.cos(), latitude.sin(), 0.0);
        let mut draft = OrbitDraft {
            direction: OrbitDirection::Retrograde,
            ..default()
        };

        assert!((launch_inclination(draft, position) - latitude).abs() < 1.0e-12);
        assert_eq!(launch_direction(draft), OrbitDirection::Prograde);

        draft.plane = OrbitPlaneChoice::Nearest;
        draft.inclination_rad = 0.0;
        assert!(launch_inclination(draft, position) < latitude);
    }

    #[test]
    fn launch_heading_rejects_inclination_below_latitude() {
        let latitude = 35.0_f64.to_radians();
        assert!(
            launch_heading(
                DVec3::X,
                DVec3::Y,
                latitude,
                20.0_f64.to_radians(),
                OrbitDirection::Prograde,
            )
            .is_none()
        );
    }

    #[test]
    fn ascent_throttle_respects_q_and_acceleration_limits() {
        assert_eq!(ascent_throttle(0.0, 9.0), 1.0);
        assert!(ascent_throttle(MAX_DYNAMIC_PRESSURE_PA * 2.0, 9.0) <= 0.5 + f64::EPSILON);
        assert!(ascent_throttle(0.0, MAX_ASCENT_ACCELERATION_M_S2 * 2.0) <= 0.5 + f64::EPSILON);
    }
}
