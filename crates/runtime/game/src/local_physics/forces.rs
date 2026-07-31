//! Per-frame force application: gravity/thrust accumulators and fly-by-wire torque.
//!
//! Split out of the former monolithic `local_physics.rs` (Phase B, `docs/simulation/regimes.md`).

#[allow(unused_imports)]
use super::*;

use bevy::math::{DQuat, DVec3};
use thalos_physics_canonical::surface_local::surface_local_acceleration;
use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::avian::{
    AngularVelocity, ConstantAngularAcceleration, ConstantLinearAcceleration, LinearVelocity,
    Position, Rotation,
};
use thalos_physics_local::{ActiveLocalBubble, LocalCraftBody};

use crate::fuel::ThrottleState;
use crate::rendering::SimulationState;
use crate::sim_clock::SimClock;

/// Write Avian's per-frame `ConstantLinearAcceleration` and
/// `ConstantAngularAcceleration` accumulators.
///
/// Two paths through the function, by Avian role:
/// - **`AttitudeOnly`**: write `angular_accel` from player + SAS torque
///   (so rotation integrates correctly while coasting), and write
///   `linear_accel = 0` so a stale `gravity + thrust` value from a
///   previous `Full` frame doesn't drive Avian's translation through
///   Kepler's authoritative pos/vel.
/// - **`Full`**: write both — `linear_accel = gravity + thrust` and
///   `angular_accel` from torque. Avian owns translation here so the
///   gravity term is what actually moves the ship.
///
/// In `Paused` we skip entirely; the snap zeroes both accumulators on
/// the way out anyway.
pub(crate) fn apply_local_forces(
    clock: Res<SimClock>,
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    weight_on_wheels: Res<WeightOnWheels>,
    hull_ground: Res<HullGroundContact>,
    mut sim: ResMut<SimulationState>,
    throttle: Res<ThrottleState>,
    mut craft_q: Query<(
        &Position,
        &Rotation,
        &LinearVelocity,
        &AngularVelocity,
        &mut ConstantLinearAcceleration,
        &mut ConstantAngularAcceleration,
        &LocalCraftBody,
    )>,
) {
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    // EVA owns its own force application via `player_controller` — both
    // gravity (from `apply_player_controller_motion`) and the walking
    // velocity targeting. Reaction-wheel torque and thrust don't apply.
    if sim.simulation.vessel_kind() == VesselKind::Eva {
        return;
    }
    if !authority.integrator_active() {
        // Avian's clock is paused; the snap will zero accel on its way out.
        return;
    }
    let Ok((
        position,
        rotation,
        linear_velocity,
        angular_velocity,
        mut linear_accel,
        mut angular_accel,
        _,
    )) = craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };
    let mut params = *sim.simulation.ship_params();
    // Ground control regime: with weight on the wheels the reaction wheels
    // lose roll/yaw (keep pitch for rotation); on the hull (tipped / belly)
    // they lose everything — see `wheel_torque_ground_mask`.
    // The controller (`control_bus::realize_control`) normalized against the
    // same masked authority, so commanded and realized torque stay equal.
    params.max_torque *= wheel_torque_ground_mask(weight_on_wheels.grounded, hull_ground.grounded);
    // A destroyed craft is inert debris: gravity still acts (so it falls and
    // settles), but thrust and reaction-wheel torque are cut. See
    // `docs/simulation/surface.md`.
    let destroyed = sim.simulation.is_destroyed();

    // Thrust-vectoring (engine gimbal) authority available this frame: the
    // full-thrust geometry term scaled by the fraction of thrust actually
    // firing, so it vanishes at zero throttle / out of fuel / coast — and a
    // destroyed craft can't gimbal. The same value the fly-by-wire controller
    // folded into its authority (`control_bus`), so what we realize here equals
    // what it commanded. Added onto `max_torque` in `compute_angular_acceleration`.
    let gimbal_effective = if destroyed {
        DVec3::ZERO
    } else {
        params.gimbal_torque_full
            * crate::fuel::active_thrust_fraction(
                &params,
                sim.simulation.ship_mass_kg(),
                throttle.effective,
            )
    };

    // Linear: gravity + thrust only when Avian owns translation. Otherwise
    // explicitly zero so a stale value from the previous `Full` frame
    // doesn't drift Avian's pos/vel away from Kepler's authoritative state.
    if authority.owns_translation() {
        let body = &sim.system.bodies[bubble.body_id];
        // Ships integrate in the **surface-local frame**, so `position.0` and
        // `linear_velocity.0` are anchor-relative SLF quantities. The exact
        // radial gravity plus the rotating-frame centrifugal and Coriolis
        // terms come from one canonical helper (unit-tested against an
        // inertial integration). At Thalos' spin the fictitious terms are
        // ~0.02 m/s², but they keep an orbital burn correct and a parked
        // craft from creeping.
        let mut accel =
            surface_local_acceleration(body.gm, &bubble.frame, position.0, linear_velocity.0);
        let throttle_eff = throttle.effective.clamp(0.0, 1.0);
        let mass = sim.simulation.ship_mass_kg();
        if !destroyed && throttle_eff > 0.0 && params.thrust_n > 0.0 && mass > params.dry_mass_kg {
            // `rotation.0` is the craft orientation in the SLF, so the nose
            // direction is already in frame axes.
            let nose_frame = rotation.0 * DVec3::Y;
            accel += nose_frame * (params.thrust_n / mass) * throttle_eff;
            sim.simulation
                .apply_external_mass_flow(throttle_eff, clock.delta_secs_f64());
        }
        linear_accel.0 = accel;
    } else {
        linear_accel.0 = DVec3::ZERO;
    }

    // Angular accel always written when the integrator is active, in both
    // `AttitudeOnly` and `Full`. This is the system that lets the player
    // rotate the ship while coasting. A destroyed craft gets zero torque —
    // no player input, no SAS damping — so it tumbles freely as debris.
    angular_accel.0 = if destroyed {
        DVec3::ZERO
    } else {
        compute_angular_acceleration(
            sim.simulation.control(),
            &params,
            gimbal_effective,
            rotation.0,
            angular_velocity.0,
            clock.delta_secs_f64(),
        )
    };
}

/// Convert the realized attitude command into a world-space angular
/// acceleration for the Avian rigid body.
///
/// `control.torque_command` is the *output of the fly-by-wire attitude
/// controller* ([`crate::control_bus`]) — pointing, hold, or raw rate — a
/// normalized `[-1, 1]` fraction. It is realized against the craft's total
/// attitude authority: the reaction wheels (`max_torque`) plus the engine
/// gimbal (`gimbal_effective`, the throttle-scaled thrust-vectoring torque —
/// pitch/yaw only, zero at coast). Both are driven at the same fraction, which
/// is exactly what the controller normalized its PD output against, so the
/// realized torque equals the intended torque. The former per-frame deadbeat
/// SAS damper (`−I·ω/dt`) lived here; SAS is now a proper controller upstream,
/// so `sas_enabled` no longer does anything.
pub(crate) fn compute_angular_acceleration(
    control: &thalos_physics_canonical::types::ControlInput,
    params: &thalos_physics_canonical::types::ShipParameters,
    gimbal_effective: DVec3,
    rotation: DQuat,
    _angular_velocity_world: DVec3,
    _dt: f64,
) -> DVec3 {
    let inertia_body = params.moment_of_inertia;
    let max_torque = params.max_torque + gimbal_effective;
    let cmd = control
        .torque_command
        .clamp(DVec3::splat(-1.0), DVec3::splat(1.0));

    let torque_body = cmd * max_torque;

    let inv_i = DVec3::new(
        if inertia_body.x > 0.0 {
            1.0 / inertia_body.x
        } else {
            0.0
        },
        if inertia_body.y > 0.0 {
            1.0 / inertia_body.y
        } else {
            0.0
        },
        if inertia_body.z > 0.0 {
            1.0 / inertia_body.z
        } else {
            0.0
        },
    );
    let accel_body = torque_body * inv_i;
    rotation * accel_body
}
