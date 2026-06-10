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
    AttitudeController, AttitudeDemand, ControlDemand, DemandSource, allocate, arbitrate,
};
use thalos_physics_canonical::aero::{ControlInputs, control_authority};
use thalos_physics_canonical::types::ControlInput;
use thalos_physics_local::avian::{LinearVelocity, Position};
use thalos_physics_local::{ActiveLocalBubble, LocalCraftBody};
use thalos_input::game::GameInputIntent;

use crate::SimStage;
use crate::aero::{AeroTuning, ShipAero, resolved_aero_config};
use crate::autopilot::{Autopilot, autopilot_system};
use crate::controls::ControlLocks;
use crate::navigation::{NavigationState, nav_attitude_demand};
use crate::rendering::SimulationState;
use crate::target::TargetBody;
use crate::maneuver::ManeuverPlan;
use crate::velocity_frame::VelocityFrameState;

/// Query for the player craft's aero state, used to size the dynamic-pressure
/// authority split in [`realize_control`].
type CraftAeroQuery<'w, 's> = Query<'w, 's, (&'static Position, &'static LinearVelocity, &'static ShipAero), With<LocalCraftBody>>;

/// Below this stick magnitude the pilot is "hands off" and the lower-priority
/// holds (SAS / nav mode) own attitude.
const STICK_DEADZONE_SQ: f64 = 1.0e-6;

/// Free-flight SAS toggle state (the `T` key). When enabled and nothing
/// higher-priority is engaged, the controller holds the current attitude.
/// This is the "centered stick = hold current attitude" behaviour.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct SasState {
    pub enabled: bool,
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
    craft_aero: CraftAeroQuery,
    mut sim: ResMut<SimulationState>,
    mut controller: ResMut<AttitudeControllerState>,
    mut sas: ResMut<SasState>,
    mut realized: ResMut<RealizedControl>,
) {
    // A destroyed craft accepts no attitude command: drop SAS, clear the hold
    // target, and emit inert control so the wreck tumbles freely.
    if sim.simulation.is_destroyed() {
        sas.enabled = false;
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
        let stick = DVec3::new(
            input.attitude.x as f64,
            input.attitude.y as f64,
            input.attitude.z as f64,
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
    let aero_authority = player_aero_authority(&sim, &active, &tuning, &craft_aero);
    let attitude = *sim.simulation.attitude();
    let params = *sim.simulation.ship_params();
    let torque = controller
        .0
        .update(arb.attitude, &attitude, &params, aero_authority);

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
}

/// Per-axis aero control-moment authority (N·m) for the player craft at the
/// current dynamic pressure, or zero when there is no flying aero (no bubble,
/// not a ship, in vacuum, or below the airspeed floor). Mirrors the density /
/// airspeed / config resolve that [`crate::aero::apply_aero_forces`] uses so the
/// authority the allocator splits against equals what the evaluator applies.
fn player_aero_authority(
    sim: &SimulationState,
    active: &ActiveLocalBubble,
    tuning: &AeroTuning,
    craft_aero: &CraftAeroQuery,
) -> DVec3 {
    let Some(bubble) = active.bubble.as_ref() else {
        return DVec3::ZERO;
    };
    let Ok((position, lin_vel, ship_aero)) = craft_aero.get(bubble.craft_entity) else {
        return DVec3::ZERO;
    };
    let Some(body) = sim.system.bodies.get(bubble.body_id) else {
        return DVec3::ZERO;
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
    control_authority(&config, density, lin_vel.0.length())
}
