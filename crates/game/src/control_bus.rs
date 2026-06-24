//! The game-side fly-by-wire control bus.
//!
//! This is the Bevy glue around [`thalos_control`]. Every per-frame attitude
//! command source — pilot stick, the free-flight SAS hold, the directional
//! navigation modes, and the scheduled-burn autopilot — is collected here as
//! a tagged [`ControlDemand`], arbitrated once by priority, run through the
//! single [`AttitudeController`], and allocated to *every* attitude effector
//! the craft has (reaction wheels via `ControlInput::torque_command`, aero
//! control surfaces via [`RealizedControl::aero`]). Nothing else may write an
//! attitude effector directly.
//!
//! This replaces the old fragmented paths: `navigation::compute_attitude_control`
//! (reaction-wheel torque + an `sas_enabled` flag that drove a per-frame
//! deadbeat damper) and the aero system reading the raw stick straight into
//! `evaluate_aero`. The deadbeat damper was the jitter source — it tried to
//! annihilate all angular velocity every frame and limit-cycled against the
//! continuous aero moments, because the two effectors were uncoordinated.
//! Here one controller commands one torque that both effectors execute.
//!
//! **Scope:** attitude. Throttle remains on its existing setpoint path
//! (`ThrottleState::commanded`, with the autopilot overriding it directly and
//! [`crate::controls::ControlLocks`] gating the player); folding the throttle
//! *setpoint* into the bus is the next step (see `docs/control.md`). Warp,
//! EVA, and RCS are likewise documented extension points: a new source is a
//! new [`DemandSource`]; a new effector is a new branch in
//! [`thalos_control::allocate`].

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_control::{
    AssistStatus, AttitudeController, AttitudeDemand, ControlDemand, DemandSource, FlightState,
    allocate, arbitrate,
};
use thalos_physics_canonical::aero::{ControlInputs, control_authority};
use thalos_physics_canonical::types::ControlInput;
use thalos_physics_local::avian::{LinearVelocity, Position, Rotation};
use thalos_physics_local::{ActiveLocalBubble, LocalCraftBody};
use thalos_input::game::GameInputIntent;

use crate::SimStage;
use crate::aero::{AeroTuning, ShipAero, resolved_aero_config};
use crate::autopilot::{Autopilot, autopilot_system};
use crate::controls::ControlLocks;
use crate::navigation::{NavigationMode, NavigationState, nav_attitude_demand};
use crate::rendering::SimulationState;
use crate::sim_clock::SimClock;
use crate::target::TargetBody;
use crate::maneuver::ManeuverPlan;
use crate::velocity_frame::VelocityFrameState;

/// Query for the player craft's aero state, used to size the dynamic-pressure
/// authority split and build the flight-assist state in [`realize_control`].
type CraftAeroQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static Position,
        &'static Rotation,
        &'static LinearVelocity,
        &'static ShipAero,
    ),
    With<LocalCraftBody>,
>;

/// Below this stick magnitude the pilot is "hands off" and the lower-priority
/// holds (SAS / nav mode) own attitude.
const STICK_DEADZONE_SQ: f64 = 1.0e-6;

/// Airspeed floor for the flight assist: below this the control surfaces are
/// mush and the angle math is degenerate (taxi / parked), so SAS falls back to
/// the plain hold. Above it a winged craft in atmosphere flies fly-by-wire.
const ASSIST_MIN_AIRSPEED_M_S: f64 = 15.0;

/// Free-flight SAS toggle state (the `T` key / the HUD SAS button). When
/// enabled and nothing higher-priority is engaged, the controller holds the
/// current attitude — the "centered stick = hold current attitude" behaviour,
/// and the arming switch for the plane fly-by-wire assist.
///
/// **Defaults on**: every craft spawns with SAS engaged (spaceships hold
/// attitude, planes fly FBW with auto-trim + stall protection), and the flag
/// survives destruction/respawn. Toggling off is the deliberate act.
#[derive(Resource, Debug, Clone, Copy)]
pub struct SasState {
    pub enabled: bool,
}

impl Default for SasState {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// The stateful attitude controller (holds the captured SAS target).
#[derive(Resource, Debug, Default)]
pub struct AttitudeControllerState(pub AttitudeController);

/// The realized control-surface command published each frame by
/// [`realize_control`].
///
/// **Sole writer:** [`realize_control`]. Read by the aero force system
/// ([`crate::aero::apply_aero_forces`]) for control-surface deflections. The
/// matching reaction-wheel command lands directly in the simulation's
/// `ControlInput::torque_command` (consumed by `apply_local_forces`), so it is
/// not mirrored here.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct RealizedControl {
    /// Aero control-surface deflections fed to `evaluate_aero`.
    pub aero: ControlInputs,
    /// The controller's normalized attitude command this frame, body frame
    /// `[-1, 1]` (`x` pitch, `y` roll, `z` yaw) — the arbitrated pilot / SAS /
    /// nav / autopilot effort *before* the reaction-wheel↔aero split. This is
    /// what the control-surface visuals deflect to: it shows commanded control
    /// effort at full scale, independent of how the allocator happens to divide
    /// the torque (the allocated `aero` fraction collapses toward zero when aero
    /// authority dwarfs the reaction-wheel torque, so it is not a usable visual
    /// signal).
    pub command: DVec3,
    /// Flight-assist status this frame: whether the plane fly-by-wire law is
    /// engaged and whether stall protection is actively clamping the pitch
    /// command. Read by the HUD's SAS button.
    pub assist: AssistStatus,
}

pub struct ControlBusPlugin;

impl Plugin for ControlBusPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SasState>()
            .init_resource::<AttitudeControllerState>()
            .init_resource::<RealizedControl>()
            .add_systems(
                Update,
                realize_control
                    .in_set(SimStage::Physics)
                    // Pilot setpoints, the autopilot state machine, and warp
                    // gating must settle before we arbitrate; the realized
                    // command must land before the canonical step and the
                    // effector systems that read it.
                    .after(autopilot_system)
                    .after(crate::bridge::handle_warp_controls)
                    .after(crate::fuel::handle_throttle_input)
                    .before(crate::bridge::advance_simulation),
            );
    }
}

/// Collect this frame's attitude demands, arbitrate, run the controller, and
/// allocate the result across the craft's effectors.
pub fn realize_control(
    input: Res<GameInputIntent>,
    nav: Res<NavigationState>,
    locks: Res<ControlLocks>,
    target: Res<TargetBody>,
    plan: Res<ManeuverPlan>,
    velocity_frame: Res<VelocityFrameState>,
    autopilot: Res<Autopilot>,
    active: Res<ActiveLocalBubble>,
    tuning: Res<AeroTuning>,
    clock: Res<SimClock>,
    craft_aero: CraftAeroQuery,
    mut sim: ResMut<SimulationState>,
    mut controller: ResMut<AttitudeControllerState>,
    mut sas: ResMut<SasState>,
    mut realized: ResMut<RealizedControl>,
) {
    // A destroyed craft accepts no attitude command: clear the hold target
    // and emit inert control so the wreck tumbles freely. `SasState` itself
    // is deliberately left alone — SAS is on by default, so the respawned
    // craft comes back with it armed rather than silently disarmed.
    if sim.simulation.is_destroyed() {
        controller.0 = AttitudeController::new();
        sim.simulation.set_control(ControlInput::default());
        *realized = RealizedControl::default();
        return;
    }

    if input.toggle_sas {
        sas.enabled = !sas.enabled;
    }

    // --- Collect demands, one per source. ---
    let mut demands: [(DemandSource, ControlDemand); 4] = [
        (DemandSource::Sas, ControlDemand::NONE),
        (DemandSource::NavMode, ControlDemand::NONE),
        (DemandSource::Autopilot, ControlDemand::NONE),
        (DemandSource::Pilot, ControlDemand::NONE),
    ];

    // SAS free-flight hold (lowest priority).
    if sas.enabled {
        demands[0].1 = ControlDemand::attitude(AttitudeDemand::Hold);
    }

    // Directional navigation modes.
    demands[1].1 = ControlDemand::attitude(nav_attitude_demand(
        nav.mode,
        velocity_frame.active,
        &target,
        &plan,
        &sim.simulation,
    ));

    // Scheduled-burn autopilot.
    demands[2].1 = ControlDemand::attitude(autopilot.attitude_demand());

    // Pilot stick (highest priority) — unless a programmatic system holds the
    // attitude lock, in which case the player can't fight it (KSP behaviour).
    if !locks.attitude {
        // Yaw convention: the pilot's +z means "nose right". In the body frame
        // (X=right, Y=nose, Z=up) nose-right is a *negative* torque about +Z, so
        // the effector chain (controller → allocator → aero / reaction wheels)
        // realizes nose-right only at command.z < 0. Negate the pilot's yaw here
        // so right stick / right twist (and keyboard D) yaw the nose right in the
        // air and in orbit, matching the ground nosewheel steering — which reads
        // intent.attitude.z directly and is already nose-right. We negate at this
        // pilot→demand boundary, not in the effectors, because the SAS / nav /
        // autopilot closed loops depend on +command.z → +τz for stability.
        let stick = DVec3::new(
            input.attitude.x as f64,
            input.attitude.y as f64,
            -(input.attitude.z as f64),
        );
        if stick.length_squared() > STICK_DEADZONE_SQ {
            demands[3].1 = ControlDemand::attitude(AttitudeDemand::Rate(stick));
        }
    }

    let arb = arbitrate(&demands);

    // --- One controller, one torque. ---
    // The controller normalizes its PD output by the *total* available
    // authority (reaction wheels + aero surfaces at the current dynamic
    // pressure) so that driving both effectors at the resulting fraction
    // realizes exactly the PD's intended torque — no over-actuation in thick
    // air (which showed up as a yaw oscillation under SAS), and full pilot
    // deflection still maps to full surface throw (real roll authority).
    //
    // The flight assist arms only while SAS is engaged (the `T` toggle or the
    // HUD's Stability mode): a `Some` flight state switches the SAS hold to
    // the plane fly-by-wire law (pitch/bank hold + auto-trim) and clamps every
    // pitch command — the pilot's included — to the AoA envelope. SAS off is
    // fully manual, KSP-style; spaceships and vacuum never get a flight state.
    let assist_armed = sas.enabled || nav.mode == Some(NavigationMode::Stability);
    let (aero_authority, flight) =
        player_aero_environment(&sim, &active, &tuning, &craft_aero, assist_armed);
    let attitude = *sim.simulation.attitude();
    let params = *sim.simulation.ship_params();
    let torque = controller.0.update(
        arb.attitude,
        &attitude,
        &params,
        aero_authority,
        flight.as_ref(),
        clock.delta_secs_f64(),
    );

    // --- Allocate to every effector. ---
    let alloc = allocate(torque);

    // Reaction wheels: write the torque command, preserving the (downstream-
    // unused) throttle field. `sas_enabled` is now meaningless — the controller
    // does the damping — so it is always false.
    let throttle = sim.simulation.control().throttle;
    sim.simulation.set_control(ControlInput {
        torque_command: alloc.reaction_wheel,
        sas_enabled: false,
        throttle,
    });

    realized.aero = alloc.aero;
    realized.command = torque;
    realized.assist = controller.0.assist_status();
}

/// Per-axis aero control-moment authority (N·m) for the player craft at the
/// current dynamic pressure — zero when there is no flying aero (no bubble,
/// not a ship, in vacuum, or below the airspeed floor) — plus, when
/// `assist_armed` and the craft is a winged vessel actually flying, the
/// body-frame [`FlightState`] the controller's fly-by-wire law reads.
/// Mirrors the density / airspeed / config resolve that
/// [`crate::aero::apply_aero_forces`] uses so the authority the allocator
/// splits against equals what the evaluator applies.
fn player_aero_environment(
    sim: &SimulationState,
    active: &ActiveLocalBubble,
    tuning: &AeroTuning,
    craft_aero: &CraftAeroQuery,
    assist_armed: bool,
) -> (DVec3, Option<FlightState>) {
    let Some(bubble) = active.bubble.as_ref() else {
        return (DVec3::ZERO, None);
    };
    let Ok((position, rotation, lin_vel, ship_aero)) = craft_aero.get(bubble.craft_entity) else {
        return (DVec3::ZERO, None);
    };
    let Some(body) = sim.system.bodies.get(bubble.body_id) else {
        return (DVec3::ZERO, None);
    };
    let density = match body.terrestrial_atmosphere.as_ref() {
        Some(atmosphere) => {
            // Mirrors `apply_aero_forces`: Avian `Position` is surface-local.
            let altitude_m =
                thalos_physics_canonical::surface_local::altitude_asl_m(&bubble.frame, position.0);
            atmosphere
                .sample_at_altitude_m(
                    altitude_m,
                    body.surface_pressure_pa(),
                    body.surface_gravity_m_s2(),
                )
                .density_kg_m3
        }
        None => 0.0,
    };
    let config = resolved_aero_config(ship_aero.config, tuning);
    let airspeed = lin_vel.0.length();
    let authority = control_authority(&config, density, airspeed);

    // The bubble integrates in the surface-local (co-rotating) frame, so
    // `LinearVelocity` is already air-relative (wind = 0) and the local
    // radial up comes straight from the SLF; both rotate into the body frame
    // with the craft's Avian rotation.
    let flight = (assist_armed
        && config.lift_slope > 0.0
        && density > 0.0
        && airspeed >= ASSIST_MIN_AIRSPEED_M_S)
        .then(|| {
            let to_body = rotation.0.inverse();
            FlightState {
                up_body: to_body
                    * thalos_physics_canonical::surface_local::radial_up(&bubble.frame, position.0),
                vel_body: to_body * lin_vel.0,
                stall_alpha: config.stall_alpha,
            }
        });
    (authority, flight)
}
