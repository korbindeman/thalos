//! Where the strategic and tactical autoflight layers meet.
//!
//! Three responsibilities, deliberately together because they are the same
//! decision viewed from three sides:
//!
//! 1. [`update_flight_program`] — the sole writer of [`FlightProgram`],
//!    derived from the programs' own phases so the selection can never
//!    disagree with what is actually running. It also owns the *arming
//!    policy*: which program implies which [`BurnArm`].
//! 2. [`resolve_autoflight`] — the pure function that decides which single
//!    source fills the `DemandSource::Autopilot` slot this frame, and what
//!    to annunciate. Replaces a match arm in the control bus that hand-
//!    rolled a two-level priority inside one `AutoflightMode` variant.
//! 3. [`request_program_override`] — the one call site that consults
//!    [`ProgramOverridePolicy`] when a player action would contradict the
//!    engaged program.
//!
//! The invariant this module exists to hold: **an illegal combination of
//! program and tactical mode is unrepresentable, not merely detected.**
//! Before the split, `nav_panel` could call `toggle_mode(Maneuver)` while
//! the ascent program held authority — leaving the ascent program's monitor
//! early-returning forever on a mode check while its widget still
//! annunciated a live program, with the vehicle in neither state.

use bevy::prelude::*;
use thalos_control::ControlDemand;

use crate::SimStage;
use crate::autopilot::{Autopilot, AutopilotState, autopilot_system};
use crate::orbit_program::OrbitProgram;
use crate::rendering::SimulationState;
use crate::route_autopilot::{LandAutopilot, update_land_autopilot};

pub use thalos_game_state::autoflight::{
    AttitudeChannel, AutoflightAnnunciation, AutoflightPolicy, AutoflightRequest, BurnArm,
    FlightProgram, OverrideOutcome, SequenceEvent, ThrottleChannel,
};

pub struct AutoflightPlugin;

impl Plugin for AutoflightPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FlightProgram>()
            .init_resource::<AutoflightPolicy>()
            .init_resource::<AutoflightAnnunciation>()
            .init_resource::<crate::stage_sequencer::StageSequencer>()
            .add_message::<AutoflightRequest>()
            .add_systems(
                Update,
                handle_autoflight_requests
                    .in_set(SimStage::Physics)
                    // Before the executor and the programs read their own
                    // state this frame, so a press takes effect immediately
                    // rather than a frame late.
                    .before(autopilot_system)
                    .before(update_land_autopilot)
                    .before(crate::orbit_program::update_surface_orbit_program),
            )
            .add_systems(
                Update,
                update_flight_program
                    .in_set(SimStage::Physics)
                    // Both programs must have settled this frame's phase
                    // before we derive the selection from them, and the
                    // selection must land before the lock table and the
                    // control bus read it.
                    .after(autopilot_system)
                    .after(update_land_autopilot)
                    .after(crate::orbit_program::update_surface_orbit_program)
                    .before(crate::controls::update_control_locks)
                    .before(crate::control_bus::realize_control),
            );
    }
}

/// Apply player autoflight requests, subject to the override policy.
///
/// The **one** place a panel's intent becomes a state change, and therefore
/// the one place the program-conflict interlock lives. Before the split,
/// `nav_panel` called `toggle_mode` directly on the shared mode enum, so
/// pressing `MNVR` during an ascent both stole the slot and left the ascent
/// program's monitor early-returning forever on a mode check it could no
/// longer satisfy.
pub(crate) fn handle_autoflight_requests(
    mut requests: MessageReader<AutoflightRequest>,
    mut policy: ResMut<AutoflightPolicy>,
    mut autopilot: ResMut<Autopilot>,
    mut land: ResMut<LandAutopilot>,
    mut orbit: ResMut<OrbitProgram>,
    mut sequencer: ResMut<crate::stage_sequencer::StageSequencer>,
    mut stage_demand: ResMut<crate::staging::StageDemand>,
    program: Res<FlightProgram>,
    sim: Res<SimulationState>,
) {
    let now_s = sim.simulation.sim_time();
    for request in requests.read().copied() {
        // Engaging the landing program is itself a program change, so it
        // consults the same interlock as any other override.
        match request_program_override(&mut policy, *program, now_s) {
            OverrideOutcome::Proceed => {}
            OverrideOutcome::Refused | OverrideOutcome::ConfirmationPending => continue,
            OverrideOutcome::Disconnect => {
                disconnect_program(
                    *program,
                    &mut orbit,
                    &mut land,
                    &mut sequencer,
                    &mut stage_demand,
                );
            }
        }
        match request {
            AutoflightRequest::ToggleBurnArm => {
                let next = if autopilot.arm().armed() {
                    BurnArm::Off
                } else {
                    BurnArm::Pilot
                };
                autopilot.set_arm(next);
            }
            AutoflightRequest::ToggleLanding => {
                // Toggling off is a disengage; toggling on hands the program
                // to `update_land_autopilot`, which picks the entry phase
                // from the guidance it can actually see.
                land.engaged = !land.engaged;
                if !land.engaged {
                    land.phase = thalos_game_state::nav::LandPhase::Off;
                    land.demand = ControlDemand::NONE;
                }
            }
        }
    }
}

/// Tear down the engaged program through its own cancel path.
///
/// One way a program stops, so a pilot disconnect and a program's own abort
/// leave identical state — including the pending staging request, which
/// would otherwise be acknowledged into a program that no longer exists.
fn disconnect_program(
    program: FlightProgram,
    orbit: &mut OrbitProgram,
    land: &mut LandAutopilot,
    sequencer: &mut crate::stage_sequencer::StageSequencer,
    stage_demand: &mut crate::staging::StageDemand,
) {
    match program {
        FlightProgram::None => {}
        FlightProgram::Ascent => {
            sequencer.cancel(stage_demand);
            orbit.phase = thalos_game_state::nav::OrbitProgramPhase::Abort;
            orbit.error = Some("disconnected by pilot".to_string());
            orbit.demand = ControlDemand::NONE;
            orbit.sequence = SequenceEvent::None;
            orbit.surface_program = false;
            orbit.idle_handoff_pending = true;
        }
        FlightProgram::Landing => {
            land.engaged = false;
            land.phase = thalos_game_state::nav::LandPhase::Off;
            land.demand = ControlDemand::NONE;
        }
    }
    warn!(
        target: "thalos::diagnostic::autoflight",
        event = "program_disconnected",
        program = ?program,
        "flight program disconnected by pilot override"
    );
}

/// Derive the engaged program from the programs' own phases, and apply the
/// arming policy on transition.
///
/// Derived rather than independently written so "which program is engaged"
/// cannot drift from "which program is running" — the drift that let the
/// panel annunciate `MNVR` over a live ascent.
pub(crate) fn update_flight_program(
    orbit: Res<OrbitProgram>,
    land: Res<LandAutopilot>,
    sim: Res<SimulationState>,
    mut program: ResMut<FlightProgram>,
    mut autopilot: ResMut<Autopilot>,
    mut policy: ResMut<AutoflightPolicy>,
    mut sequencer: ResMut<crate::stage_sequencer::StageSequencer>,
    mut previous: Local<Option<FlightProgram>>,
) {
    policy.expire_pending(sim.simulation.sim_time());

    let next = resolve_engaged_program(orbit.active(), land.active());
    *program = next;

    // Arming is applied on *transition* only, so a player who deliberately
    // disarms the executor mid-coast keeps it disarmed until the next
    // program boundary. Applying it every frame would silently undo them.
    if previous.replace(next) != Some(next) {
        autopilot.set_arm(arm_for_program(next));
        // A program boundary is the staging sequence's lifecycle
        // boundary too: counters from the last ascent must not be
        // read as this one's.
        sequencer.reset();
        info!(
            target: "thalos::diagnostic::autoflight",
            event = "flight_program",
            program = ?next,
            burn_arm = ?autopilot.arm(),
            "flight program transition"
        );
    }
}

/// Ascent outranks landing if both somehow claim to be active.
///
/// They are mutually exclusive by construction — engaging one disengages
/// the other at the request sites — so this is a backstop, not a policy.
fn resolve_engaged_program(ascent_active: bool, landing_active: bool) -> FlightProgram {
    match (ascent_active, landing_active) {
        (true, _) => FlightProgram::Ascent,
        (false, true) => FlightProgram::Landing,
        (false, false) => FlightProgram::None,
    }
}

/// Which arming state each program implies for the shared burn executor.
///
/// This is the "can't be configured broken" rule in one place: an approach
/// must never fly a leftover maneuver node, and an ascent must fly the
/// circularisation nodes it installed itself without the player having to
/// arm anything.
fn arm_for_program(program: FlightProgram) -> BurnArm {
    match program {
        // Hand-placed nodes, executed on the player's behalf. The
        // historical default.
        FlightProgram::None => BurnArm::Pilot,
        // Flies the circularisation nodes it installs at MECO.
        FlightProgram::Ascent => BurnArm::Program,
        // An approach profile never executes maneuver nodes; a stale node
        // firing on short final is exactly the failure this prevents.
        FlightProgram::Landing => BurnArm::Off,
    }
}

/// Coarse burn-executor status, as the resolver needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnStatus {
    /// Idle or merely armed — asserting nothing.
    Passive,
    /// Pointing at the burn vector, throttle held closed.
    Engaging,
    /// Engine firing.
    Burning,
}

impl BurnStatus {
    pub fn of(autopilot: &Autopilot) -> Self {
        if !autopilot.arm().armed() {
            return Self::Passive;
        }
        match autopilot.state() {
            AutopilotState::Engaging { .. } => Self::Engaging,
            AutopilotState::Burn { .. } => Self::Burning,
            AutopilotState::Idle | AutopilotState::Armed { .. } => Self::Passive,
        }
    }

    fn asserting(self) -> bool {
        !matches!(self, Self::Passive)
    }
}

/// The single source that fills the autopilot demand slot this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoflightResolution {
    pub demand: ControlDemand,
    pub attitude: AttitudeChannel,
    pub throttle: ThrottleChannel,
}

impl AutoflightResolution {
    pub const NONE: Self = Self {
        demand: ControlDemand::NONE,
        attitude: AttitudeChannel::Free,
        throttle: ThrottleChannel::Pilot,
    };
}

/// Resolve the autopilot slot.
///
/// Priority, highest first:
///
/// 1. **A program's continuous guidance law.** A program that is actively
///    flying knows whether it wants anything else to happen; nothing may
///    preempt an ascent pitch program or an approach profile. (The ascent
///    program drops its guidance at MECO precisely so the executor can take
///    over for circularisation — the handoff is expressed by yielding, not
///    by a mode change.)
/// 2. **The scheduled-burn executor**, when engaging or burning.
/// 3. **An engaged program with neither** — a ballistic coast. It still
///    commands throttle idle, so a stale pilot setpoint cannot leak through
///    the arbiter's fallback and light the engine mid-coast.
/// 4. Nothing: the slot is empty and lower-priority sources (nav modes,
///    SAS) own attitude.
///
/// Pure and Bevy-free so the priority can be tested directly; it is the
/// rule that used to live as an `if`-chain inside a match arm.
pub fn resolve_autoflight(
    program: FlightProgram,
    burn: BurnStatus,
    burn_demand: ControlDemand,
    guidance: Option<ControlDemand>,
) -> AutoflightResolution {
    if let Some(demand) = guidance {
        return AutoflightResolution {
            demand,
            attitude: if demand.attitude.is_active() {
                AttitudeChannel::Guidance
            } else {
                AttitudeChannel::Free
            },
            throttle: if demand.throttle.is_some() {
                ThrottleChannel::Guidance
            } else {
                ThrottleChannel::Pilot
            },
        };
    }

    if burn.asserting() {
        return AutoflightResolution {
            demand: burn_demand,
            attitude: AttitudeChannel::NodeBurn,
            throttle: match burn {
                BurnStatus::Burning => ThrottleChannel::Burn,
                _ => ThrottleChannel::Idle,
            },
        };
    }

    if program.engaged() {
        return AutoflightResolution {
            demand: ControlDemand::throttle(0.0),
            attitude: AttitudeChannel::Free,
            throttle: ThrottleChannel::Idle,
        };
    }

    AutoflightResolution::NONE
}

/// The next armed sequencing event, for the annunciator.
///
/// Programs publish their own; the burn executor's armed directive is the
/// fallback so a hand-placed node still annunciates a countdown.
pub fn armed_sequence_event(
    orbit: &OrbitProgram,
    autopilot: &Autopilot,
    schedule: &crate::autopilot::AutopilotBurnSchedule,
    now_s: f64,
) -> SequenceEvent {
    if orbit.sequence != SequenceEvent::None {
        return orbit.sequence;
    }
    if !autopilot.arm().armed() {
        return SequenceEvent::None;
    }
    match schedule.next() {
        Some(directive) => SequenceEvent::Burn {
            in_s: (directive.burn_start() - now_s).max(0.0),
        },
        None => SequenceEvent::None,
    }
}

/// Resolve a player action that would contradict the engaged program.
///
/// The one consulting call site for [`ProgramOverridePolicy`]. Returns
/// `true` when the caller should go ahead with the action; on
/// `Disconnect` the engaged program is torn down first, through the same
/// path its own cancel request uses, so there is exactly one way a program
/// stops.
pub fn request_program_override(
    policy: &mut AutoflightPolicy,
    program: FlightProgram,
    now_s: f64,
) -> OverrideOutcome {
    let outcome = policy.request_override(program.engaged(), now_s);
    if !matches!(outcome, OverrideOutcome::Proceed) {
        info!(
            target: "thalos::diagnostic::autoflight",
            event = "program_override",
            program = ?program,
            outcome = ?outcome,
            "pilot override of an engaged flight program"
        );
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::DVec3;
    use thalos_control::AttitudeDemand;

    fn guidance_demand() -> ControlDemand {
        ControlDemand::autoflight(AttitudeDemand::PointNose(DVec3::Y), Some(0.85), None, None)
    }

    fn burn_demand() -> ControlDemand {
        ControlDemand::autoflight(AttitudeDemand::PointNose(DVec3::X), Some(1.0), None, None)
    }

    #[test]
    fn guidance_outranks_the_burn_executor() {
        // A node armed mid-gravity-turn must not preempt the pitch program.
        let out = resolve_autoflight(
            FlightProgram::Ascent,
            BurnStatus::Burning,
            burn_demand(),
            Some(guidance_demand()),
        );
        assert_eq!(out.demand, guidance_demand());
        assert_eq!(out.attitude, AttitudeChannel::Guidance);
        assert_eq!(out.throttle, ThrottleChannel::Guidance);
    }

    #[test]
    fn executor_takes_over_once_guidance_yields() {
        // This is the MECO handoff: guidance drops to None, the executor
        // flies the circularisation node. No mode change is involved.
        let out = resolve_autoflight(
            FlightProgram::Ascent,
            BurnStatus::Burning,
            burn_demand(),
            None,
        );
        assert_eq!(out.demand, burn_demand());
        assert_eq!(out.attitude, AttitudeChannel::NodeBurn);
        assert_eq!(out.throttle, ThrottleChannel::Burn);
    }

    #[test]
    fn engaging_annunciates_idle_throttle_not_burn() {
        let out = resolve_autoflight(
            FlightProgram::None,
            BurnStatus::Engaging,
            ControlDemand::autoflight(AttitudeDemand::PointNose(DVec3::X), Some(0.0), None, None),
            None,
        );
        assert_eq!(out.attitude, AttitudeChannel::NodeBurn);
        assert_eq!(out.throttle, ThrottleChannel::Idle);
    }

    #[test]
    fn coasting_program_holds_throttle_closed() {
        // The ascent coast: no guidance, no burn, but a stale pilot
        // setpoint must not reach the engine.
        let out = resolve_autoflight(
            FlightProgram::Ascent,
            BurnStatus::Passive,
            ControlDemand::NONE,
            None,
        );
        assert_eq!(out.demand.throttle, Some(0.0));
        assert_eq!(out.throttle, ThrottleChannel::Idle);
        assert_eq!(out.attitude, AttitudeChannel::Free);
    }

    #[test]
    fn no_program_and_passive_executor_yields_the_slot() {
        let out = resolve_autoflight(
            FlightProgram::None,
            BurnStatus::Passive,
            ControlDemand::NONE,
            None,
        );
        assert_eq!(out, AutoflightResolution::NONE);
        assert_eq!(out.demand, ControlDemand::NONE);
    }

    #[test]
    fn landing_disarms_the_executor_and_ascent_arms_it() {
        assert_eq!(arm_for_program(FlightProgram::Landing), BurnArm::Off);
        assert_eq!(arm_for_program(FlightProgram::Ascent), BurnArm::Program);
        assert_eq!(arm_for_program(FlightProgram::None), BurnArm::Pilot);
    }

    #[test]
    fn engaged_program_is_derived_and_ascent_wins_ties() {
        assert_eq!(resolve_engaged_program(false, false), FlightProgram::None);
        assert_eq!(resolve_engaged_program(true, false), FlightProgram::Ascent);
        assert_eq!(resolve_engaged_program(false, true), FlightProgram::Landing);
        assert_eq!(resolve_engaged_program(true, true), FlightProgram::Ascent);
    }
}
