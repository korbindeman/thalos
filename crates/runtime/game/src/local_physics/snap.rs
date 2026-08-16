//! Canonical<->Avian state flow: the role-conditional snap, readback, and SLF re-anchoring.
//!
//! Split out of the former monolithic `local_physics.rs` (Phase B, `docs/simulation/regimes.md`).

#[allow(unused_imports)]
use super::*;

use bevy::math::DVec3;
use thalos_game_state::ActiveCraftRef;
use thalos_physics_canonical::surface_local::{
    SurfaceAnchor, SurfaceLocalFrame, SurfaceLocalState, reanchor,
};
use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::avian::{
    AngularVelocity, ConstantAngularAcceleration, ConstantLinearAcceleration, LinearVelocity,
    Position, Rotation,
};
use thalos_physics_local::{ActiveLocalBubble, HeightSourceRegistry, LocalCraftBody};

use crate::player_controller::EvaMode;
use crate::rendering::SimulationState;

/// Push canonical state into Avian's components, with what we push
/// depending on Avian's current role:
///
/// - **`Paused`** (warp ≠ 1× or `BodyFixed`): full snap every frame.
///   Canonical owns everything; Avian's components mirror it so render
///   and contact queries stay coherent without an integrator race.
/// - **`AttitudeOnly`** (1× coast): snap pos/vel from canonical every
///   frame (Kepler is propagating canonical translation; Avian's pos/vel
///   would otherwise drift kinematically by `velocity · dt` per frame).
///   Leave rotation/angular_velocity alone — Avian is integrating those
///   under player attitude commands and SAS damping.
/// - **`Full`** (1× thrust/contact): snap nothing on regular frames
///   (Avian owns both translation and rotation). On the one frame the
///   role transitions to `Full` from another role, do a full snap so
///   Avian starts the burn from canonical's freshest Kepler-evolved
///   state.
///
/// The handoff-frame snap is critical: snap and readback must convert
/// using the *same* `body_state` for the round-trip to be exact. Without
/// it, the last snap in frame K−1 used `body_state(K−1)`, the first
/// readback in frame K uses `body_state(K)`, and inertial canonical
/// jumps by `relative_velocity · sim_dt` (~117 m of apo/peri shift at
/// Thalos LEO) at every authority handoff. `just_took_translation`
/// reruns the full snap with `body_state(K)`, so readback's conversion
/// cancels exactly.
///
/// At warp > 1× the angular velocity is forced to zero (matching the old
/// `Simulation::integrate_attitude` behaviour). The original comment was
/// explicit: "allowing ω to persist would let a ship spin up at warp
/// entry and keep tumbling out of warp." SAS-off players who tap rotation
/// keys right before warp would otherwise emerge spinning.
pub(crate) fn snap_avian_from_canonical(
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    eva_mode: ActiveCraftRef<EvaMode>,
    mut sim: ResMut<SimulationState>,
    mut craft_q: Query<
        (
            &mut Position,
            &mut Rotation,
            &mut LinearVelocity,
            &mut AngularVelocity,
            &mut ConstantLinearAcceleration,
            &mut ConstantAngularAcceleration,
        ),
        With<LocalCraftBody>,
    >,
) {
    let Some(eva_mode) = eva_mode.get() else {
        return;
    };
    // KSP-style EVA *while grounded*: the player owns Avian state outright via
    // `player_controller`'s motion + terrain-clamp systems. Snapping
    // canonical → Avian here would fight those writes (canonical is
    // refreshed from Avian by `readback_local_craft`, so the snap would
    // either no-op or revert a frame of input). Skip entirely.
    //
    // Airborne (coasting) EVA is the mirror image: Kepler owns canonical
    // translation, the walk controller stands down, and the snap below drives
    // the capsule — exactly like a ship coasting in vacuum. So fall through.
    if sim.simulation.vessel_kind() == VesselKind::Eva && eva_mode.is_grounded() {
        return;
    }
    // `Full` mid-burn: Avian owns everything. No snap.
    if matches!(authority.role, AvianRole::Full) && !authority.just_took_translation() {
        return;
    }
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let high_warp = sim.simulation.warp.speed() > 1.0 + f64::EPSILON;
    if high_warp {
        // Zero canonical ω so prediction / map view see a non-tumbling
        // ship the moment warp engages, and the next snap doesn't push a
        // stale ω into Avian.
        let mut attitude = *sim.simulation.attitude();
        if attitude.angular_velocity.length_squared() > 0.0 {
            attitude.angular_velocity = DVec3::ZERO;
            sim.simulation.set_attitude(attitude);
        }
    }
    let Ok((
        mut position,
        mut rotation,
        mut linear_velocity,
        mut angular_velocity,
        mut linear_accel,
        mut angular_accel,
    )) = craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };
    let body_state = body_state_for(&sim, bubble.body_id);
    let craft = sim.simulation.craft_state();
    let frame = inertial_to_craft_frame(
        sim.simulation.vessel_kind(),
        &body_state,
        &bubble.frame,
        craft.translation,
        craft.attitude,
    );

    // Pos/vel are always snapped from canonical when this system runs.
    // (The early return above means we only get here in `Paused`,
    // `AttitudeOnly`, or the just-took-`Full` handoff frame — in all of
    // those, canonical is the source of truth for translation.)
    position.0 = frame.position_m;
    linear_velocity.0 = frame.linear_velocity_m_s;
    // Always zero the linear accel accumulator. In `AttitudeOnly` Avian's
    // step still runs, and a stale `gravity + thrust` value from a prior
    // `Full` frame would otherwise drive the ship through Kepler-managed
    // pos/vel. In `Paused`/handoff cases this is just the existing reset.
    linear_accel.0 = DVec3::ZERO;

    // Rotation handling depends on the role:
    // - `AttitudeOnly`: Avian is integrating rotation under player +
    //   SAS torque; overwriting it would erase the player's input each
    //   frame. Leave rotation, angular_velocity, and angular_accel
    //   alone — `apply_local_forces` writes angular_accel each frame.
    // - `Paused` / `Full` handoff: full snap from canonical; zero the
    //   torque accumulator so we don't double-apply at handoff.
    if !matches!(authority.role, AvianRole::AttitudeOnly) {
        rotation.0 = frame.rotation;
        angular_velocity.0 = if high_warp {
            DVec3::ZERO
        } else {
            frame.angular_velocity_rad_s
        };
        angular_accel.0 = DVec3::ZERO;
    }
}

/// Pull Avian's integrated state back into canonical, with what we install
/// depending on Avian's role:
///
/// - **`Paused`**: skip entirely; canonical owns everything and Avian's
///   pos/vel/rot are just snapped mirrors of canonical.
/// - **`AttitudeOnly`** (1× coast): install attitude only. Translation
///   stays Kepler-driven — Avian's pos/vel kinematically drift inside the
///   frame (zero linear_accel, but velocity carries position by `v · dt`)
///   and would otherwise corrupt canonical.
/// - **`Full`**: install both translation and attitude.
pub(crate) fn readback_local_craft(
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    eva_mode: ActiveCraftRef<EvaMode>,
    mut sim: ResMut<SimulationState>,
    mut diagnostics: ResMut<AvianHandoffDiagnostics>,
    craft_q: Query<(
        &Position,
        &Rotation,
        &LinearVelocity,
        &AngularVelocity,
        &LocalCraftBody,
    )>,
) {
    let Some(eva_mode) = eva_mode.get() else {
        return;
    };
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    // Grounded EVA owns canonical translation outright (see
    // `snap_avian_from_canonical`); airborne EVA coasts like a ship and falls
    // through to the role-driven split below.
    let eva_grounded = sim.simulation.vessel_kind() == VesselKind::Eva && eva_mode.is_grounded();
    if !eva_grounded && !authority.integrator_active() {
        return;
    }
    let Ok((position, rotation, linear_velocity, angular_velocity, _)) =
        craft_q.get(bubble.craft_entity)
    else {
        return;
    };
    let body_state = body_state_for(&sim, bubble.body_id);
    let (translation, attitude) = craft_frame_to_inertial(
        sim.simulation.vessel_kind(),
        &body_state,
        &bubble.frame,
        position.0,
        rotation.0,
        linear_velocity.0,
        angular_velocity.0,
    );
    // Grounded EVA always owns the canonical position outright — see the
    // comment in `snap_avian_from_canonical`. Ships (and airborne EVA) fall
    // back to the role-driven split.
    if eva_grounded || authority.owns_translation() {
        // On the take-translation frame, measure the gap between canonical's
        // pre-handoff state and Avian's converted state. The snap re-ran with
        // this frame's `body_state`, so a coherent handoff leaves this near
        // zero; a large value means snap and readback disagreed on the body
        // frame (the discontinuity the snap exists to prevent).
        if !eva_grounded && authority.just_took_translation() {
            let canonical = sim.simulation.craft_state().translation;
            let position_residual_m = (translation.position - canonical.position).length();
            let velocity_residual_m_s = (translation.velocity - canonical.velocity).length();
            // A residual above tolerance means snap and readback disagreed on
            // the body frame at the take — conversion drift, a stale
            // `body_state`, an SOI race, or (legitimately) a *landed* craft
            // released onto a ballistic `OnRails` arc that built up real
            // surface-relative velocity before the backend took translation.
            // That last case is a benign one-frame reconciliation: the
            // read-back state installs normally below and the sim recovers next
            // frame. This is observability, not an invariant, so surface it
            // loudly but never crash the session. (Previously a `debug_assert!`
            // — it killed the whole debug build on the edge case of throttling
            // up a craft dropped onto raw terrain; release builds, where the
            // assert compiled out, already survived it.)
            if position_residual_m >= HANDOFF_RESIDUAL_TOLERANCE_M {
                warn!(
                    "Avian↔canonical take-translation handoff position discontinuity: \
                     {position_residual_m:.3} m (tolerance {HANDOFF_RESIDUAL_TOLERANCE_M} m) — \
                     snap/readback body-frame skew, SOI race, or ballistic-release velocity"
                );
            }
            diagnostics.last_handoff_kind = "TookTranslation".to_string();
            diagnostics.last_handoff_sim_time_s = sim.simulation.sim_time();
            diagnostics.position_residual_m = position_residual_m;
            diagnostics.velocity_residual_m_s = velocity_residual_m_s;
        }
        sim.simulation
            .install_local_rigid_body_state(translation, attitude);
    } else {
        // AttitudeOnly: attitude flows back, translation stays Kepler-owned.
        sim.simulation.set_attitude(attitude);
    }
}

/// Horizontal drift from the anchor that triggers a re-anchor. Keeps the
/// craft's SLF coordinates small near the ground; each re-anchor is an exact
/// f64 state translation (a handful of quaternion ops), so even the orbital
/// AttitudeOnly regime crossing the surface at ~2 km/s re-anchors cheaply.
pub(crate) const REANCHOR_HORIZONTAL_M: f64 = 1500.0;

/// Move the surface-local frame's anchor back under the craft when it has
/// drifted too far horizontally. The state translation is exact
/// ([`thalos_physics_canonical::surface_local::reanchor`] — no inertial round
/// trip), so canonical state is untouched. Runs after [`readback_local_craft`]
/// and before [`maintain_terrain_patch`] / [`sync_terrain_collider_pose`], so
/// the collider systems immediately re-pose the static ground geometry in the
/// new frame within the same chain.
pub(crate) fn reanchor_surface_frame(
    height_sources: Res<HeightSourceRegistry>,
    sim: Res<SimulationState>,
    mut active: ResMut<ActiveLocalBubble>,
    mut craft_q: Query<
        (
            &mut Position,
            &mut Rotation,
            &mut LinearVelocity,
            &mut AngularVelocity,
        ),
        With<LocalCraftBody>,
    >,
) {
    let Some(bubble) = active.bubble.as_mut() else {
        return;
    };
    // EVA keeps the body-centered seam until its SLF fold-in; its frame is
    // refreshed only on explicit teleports.
    if sim.simulation.vessel_kind() == VesselKind::Eva {
        return;
    }
    let Ok((mut position, mut rotation, mut linear_velocity, mut angular_velocity)) =
        craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };
    let horizontal = DVec3::new(position.0.x, 0.0, position.0.z).length();
    if horizontal <= REANCHOR_HORIZONTAL_M {
        return;
    }
    let body_state = body_state_for(&sim, bubble.body_id);
    // New anchor directly under the craft, in body-fixed coordinates.
    let dir_body = (bubble.frame.rotation_body_to_frame.inverse()
        * bubble.frame.body_center_offset(position.0))
    .normalize_or_zero();
    if dir_body == DVec3::ZERO {
        return;
    }
    let elevation_m = height_sources
        .get(bubble.body_id)
        .and_then(|hs| hs.sample_height_m(dir_body.as_vec3(), PHYSICS_QUERY_TILE_LOD_M))
        .map(|h| h as f64)
        .unwrap_or(0.0);
    let new_frame = SurfaceLocalFrame::new(
        &body_state,
        SurfaceAnchor {
            dir_body,
            elevation_m,
        },
    );
    let orientation = rotation.0.normalize();
    let moved = reanchor(
        &bubble.frame,
        &new_frame,
        SurfaceLocalState {
            position_m: position.0,
            velocity_m_s: linear_velocity.0,
            orientation_frame: orientation,
            angular_velocity_body: orientation.inverse() * angular_velocity.0,
        },
    );
    position.0 = moved.position_m;
    linear_velocity.0 = moved.velocity_m_s;
    rotation.0 = moved.orientation_frame;
    angular_velocity.0 = moved.orientation_frame * moved.angular_velocity_body;
    bubble.frame = new_frame;
}
