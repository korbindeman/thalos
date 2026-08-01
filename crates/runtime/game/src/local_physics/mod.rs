//! Game-side orchestration for the M5 aggregate local-physics bubble.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_body_render::HeightSource;
use thalos_physics_canonical::types::BodyState;
use thalos_physics_local::avian::{Physics, PhysicsTime};
use thalos_physics_local::{
    ActiveLocalBubble, LocalCraftBody, LocalPhysicsPlugin, publish_local_craft_kinematics,
    sync_structure_collider_pose,
};
use thalos_world::BodyId;

use crate::SimStage;
use crate::player_controller::PlayerControllerState;
use crate::rendering::SimulationState;
use crate::sim_clock::SimClock;

pub use thalos_game_state::flight::PHYSICS_QUERY_TILE_LOD_M;

const THALOS_NAME: &str = "Thalos";
const DEBUG_DROP_KEY: KeyCode = KeyCode::F9;

/// Apply an inertial-frame velocity kick to the active local rigid body.
///
/// Stage separation owns the canonical momentum transaction, but only this
/// executor boundary may name Avian components. Mirroring the parent's
/// reaction impulse here prevents the next local-physics readback from
/// overwriting the canonical velocity change.
pub(crate) fn apply_inertial_delta_v(world: &mut World, delta_v: DVec3) {
    if delta_v.length_squared() <= f64::EPSILON {
        return;
    }
    let Some((body_id, craft_entity, frame)) = world
        .get_resource::<ActiveLocalBubble>()
        .and_then(|active| active.bubble.as_ref())
        .map(|bubble| (bubble.body_id, bubble.craft_entity, bubble.frame.clone()))
    else {
        return;
    };
    let Some(sim) = world.get_resource::<SimulationState>() else {
        return;
    };
    let body = sim.ephemeris.state(
        body_id,
        thalos_physics_canonical::canonical::Epoch(sim.simulation.sim_time()),
    );
    let body_fixed_delta = body.orientation.inverse() * delta_v;
    let surface_local_delta = frame.rotation_body_to_frame * body_fixed_delta;
    if let Some(mut velocity) =
        world.get_mut::<thalos_physics_local::avian::LinearVelocity>(craft_entity)
    {
        velocity.0 += surface_local_delta;
    }
}

/// Position discontinuity above which a take-translation handoff is logged as
/// anomalous (see [`snap::readback_local_craft`]). A healthy handoff residual
/// is the distance Avian's integrator drifts from the snap source in one step
/// (`~|accel|·dt²`, sub-centimetre). The frame-skew / SOI-race failure the snap
/// guards against produces `~|relative_velocity|·dt` (~100 m at Thalos LEO), so
/// 2 m cleanly separates the two regimes. Non-fatal: a *landed* craft released
/// onto a ballistic `OnRails` arc can legitimately land in between (it gains
/// real surface-relative velocity before the backend takes translation), so
/// exceeding this only emits a warning — the read-back state installs anyway.
const HANDOFF_RESIDUAL_TOLERANCE_M: f64 = 2.0;

pub struct GameLocalPhysicsPlugin;

impl Plugin for GameLocalPhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(LocalPhysicsPlugin)
            .init_resource::<AvianAuthority>()
            .init_resource::<AvianHandoffDiagnostics>()
            .init_resource::<GearTuning>()
            .init_resource::<GearState>()
            .init_resource::<ParkingBrake>()
            .init_resource::<WeightOnWheels>()
            .init_resource::<GearVisualCompression>()
            .init_resource::<SurfaceFriction>()
            .init_resource::<HullGroundContact>()
            .init_resource::<TerrainFloorBackstop>()
            .register_type::<TerrainFloorBackstop>()
            .register_type::<AvianRole>()
            .register_type::<AvianAuthority>()
            .register_type::<AvianHandoffDiagnostics>()
            .register_type::<GearTuning>()
            .register_type::<GearState>()
            .register_type::<ParkingBrake>()
            .register_type::<WeightOnWheels>()
            .register_type::<SurfaceFriction>()
            .register_type::<HullGroundContact>()
            .register_type::<Wheel>()
            .register_type::<WheelSet>()
            .add_systems(
                Update,
                hard_pause_avian_time
                    .after(crate::sim_clock::sync_sim_clock)
                    .before(SimStage::Physics),
            )
            .add_systems(
                Update,
                (
                    debug_surface_drop,
                    spawn_player_avian_body,
                    rebase_bubble_to_dominant_body,
                    attach_terrain_patch_when_close,
                    detach_terrain_patch_when_far,
                    compute_avian_authority,
                    // Phase A3 port #1: the single authority executor,
                    // applying `CraftRegimeState::expected_authority`. It
                    // replaced `manage_authority`,
                    // `release_landed_ship_on_throttle`, and
                    // `collapse_or_constrain_warp` (see `docs/simulation/regimes.md`
                    // and `crate::regime::apply_regime_authority`).
                    crate::regime::apply_regime_authority,
                    sync_avian_time,
                    snap_avian_from_canonical,
                    apply_local_forces,
                    toggle_parking_brake,
                    toggle_gear,
                    apply_landing_gear_forces,
                    terrain_floor_backstop,
                    apply_surface_friction,
                )
                    .chain()
                    .in_set(SimStage::Physics)
                    .after(crate::bridge::advance_simulation),
            )
            // Second half of the per-frame chain (Bevy's `.chain()` tuple caps
            // at 20 systems): readback + frame/collider maintenance, strictly
            // after the force/contact half above.
            .add_systems(
                Update,
                (
                    readback_local_craft,
                    detect_terrain_impact,
                    reanchor_surface_frame,
                    maintain_terrain_patch,
                    sync_terrain_collider_pose,
                    // Pose terrain-anchored structure colliders (runway slab)
                    // static in the SLF — generic executor home for what was
                    // runway.rs's `sync_runway_collider_pose` (Phase 0 seam).
                    sync_structure_collider_pose,
                    // Last: snapshot the post-re-anchor craft SLF state into the
                    // Avian-free `LocalCraftKinematics` readout for next frame's
                    // non-executor readers (control bus). See `docs/simulation/physics.md`.
                    publish_local_craft_kinematics,
                )
                    .chain()
                    .in_set(SimStage::Physics)
                    .after(apply_surface_friction),
            );
    }
}

// `AvianRole` / `AvianAuthority` moved to `thalos_game_state::regime`
// (Phase 5a); the projection writer (`compute_avian_authority`) stays in
// this module.
pub use thalos_game_state::regime::{AvianAuthority, AvianRole};

/// Observability for the canonical↔Avian translation handoff. Recorded at
/// each authority transition by the authority executor
/// ([`crate::regime::apply_regime_authority`], direction + time) and
/// [`readback_local_craft`] (the residual measured when Avian first takes
/// translation). Reflect-registered (for a future debug overlay) so it can be
/// read without attaching a debugger.
///
/// `position_residual_m` / `velocity_residual_m_s` describe the **most recent
/// take-translation** handoff: the gap between canonical's pre-handoff state
/// and the state read back from Avian after its first integration step.
/// Healthy values are sub-centimetre (`~|accel|·dt²`); a value approaching
/// `|relative_velocity|·dt` (~100 m at Thalos LEO) means snap and readback
/// disagreed on the body frame at the take (conversion drift, a stale
/// `body_state` between the two systems, or mid-frame state mutation).
///
/// Scope caveat: recording is gated on the same `just_took_translation`
/// predicate the snap uses to do its fresh re-sync, so this can't catch a
/// regression that makes *that predicate itself* go false (snap and recording
/// would skip together). It verifies the handoff snap was coherent, not that
/// the handoff was triggered. The residuals persist unchanged across a release
/// handoff, so `last_handoff_kind` may read `"ReleasedTranslation"` while they
/// still describe the prior take.
#[derive(Resource, Default, Debug, Clone, Reflect)]
#[reflect(Resource)]
pub struct AvianHandoffDiagnostics {
    /// `"TookTranslation"` or `"ReleasedTranslation"`; empty before the
    /// first handoff.
    pub last_handoff_kind: String,
    pub last_handoff_sim_time_s: f64,
    pub position_residual_m: f64,
    pub velocity_residual_m_s: f64,
}

/// Project the per-craft `CraftRegime` record onto [`AvianAuthority`]
/// (A3 port #2, `docs/simulation/regimes.md`).
///
/// The role *classification* lives in the resolver
/// (`thalos_physics_canonical::regime::resolve` — warp/BodyFixed →
/// clock off, thrust/terrain-collider/atmosphere → Backend translation,
/// vacuum coast → Backend rotation only); this system only maps the
/// record's owner/clock fields onto the legacy three-way role and rolls
/// the `previous_role` edge. Before the first record exists (the frames
/// between bubble spawn and the resolver's component insert) the role
/// holds at the `Paused` default — every scenario spawns warp-paused, so
/// the classifications agree.
fn compute_avian_authority(
    active: Res<ActiveLocalBubble>,
    mut authority: ResMut<AvianAuthority>,
    craft_q: Query<&crate::regime::CraftRegimeState, With<LocalCraftBody>>,
) {
    let role = active
        .bubble
        .as_ref()
        .and_then(|bubble| craft_q.get(bubble.craft_entity).ok())
        .map(|state| crate::regime::legacy_avian_role(&state.regime))
        .unwrap_or(AvianRole::Paused);
    authority.previous_role = authority.role;
    authority.role = role;
}

fn hard_pause_avian_time(clock: Res<SimClock>, mut physics_time: ResMut<Time<Physics>>) {
    if clock.is_paused() {
        physics_time.pause();
    }
}

fn sync_avian_time(
    clock: Res<SimClock>,
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    sim: Res<SimulationState>,
    player: Option<Res<PlayerControllerState>>,
    mut physics_time: ResMut<Time<Physics>>,
) {
    // Avian's clock runs both for `Full` (translation+rotation+contact) and
    // `AttitudeOnly` (rotation+contact while Kepler owns translation).
    // `SimClock` is a hard pause over that role classifier so menu/freecam-freeze/warp
    // pause stops local and canonical physics together.
    if clock.is_paused() {
        physics_time.pause();
        return;
    }

    // Never step Avian under time-warp. At warp ≠ 1× the role is `Paused`
    // (`integrator_active()` false), so the only thing that would otherwise
    // keep the integrator alive is `player_active` (grounded EVA). But Avian
    // integrating the EVA capsule — which carries the body's surface
    // co-rotation velocity (several km/s) — over the warp-scaled timestep
    // explodes its position by tens of km per frame; `step_eva_controller`
    // re-plants it analytically each frame, but the rendered/Avian state in
    // between is garbage and crashes the UDLOD tile streamer. The grounded EVA
    // controller writes `Position` directly and needs no integrator, so pausing
    // here is safe and keeps surface time-warp stable.
    let warping = (sim.simulation.warp.speed() - 1.0).abs() > f64::EPSILON;
    let player_active = !warping
        && player
            .as_deref()
            .map(|state| state.is_active())
            .unwrap_or(false);
    if active.bubble.is_some() && (authority.integrator_active() || player_active) {
        physics_time.unpause();
    } else {
        physics_time.pause();
    }
}

fn thalos_body_id(sim: &SimulationState) -> Option<BodyId> {
    sim.system.name_to_id.get(THALOS_NAME).copied()
}

fn body_state_for(sim: &SimulationState, body_id: BodyId) -> BodyState {
    sim.ephemeris.state(
        body_id,
        thalos_physics_canonical::canonical::Epoch(sim.simulation.sim_time()),
    )
}

fn agl_above_rendered_surface(
    body: &thalos_world::BodyDefinition,
    body_state: &BodyState,
    height_source: &dyn HeightSource,
    ship_position: DVec3,
) -> Option<(f64, DVec3, DVec3)> {
    let position_body = body_state.orientation.inverse() * (ship_position - body_state.position);
    let dir = position_body.try_normalize()?;
    let height = height_source.sample_height_m(dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)? as f64;
    let radius = body.radius_m + height;
    Some((position_body.length() - radius, dir, position_body))
}

// Submodules of the local-physics layer (Phase B split; see docs/simulation/regimes.md).
mod colliders;
mod forces;
mod frames;
mod gear;
mod ground;
mod snap;
mod spawn;
mod terrain_patch;
pub(crate) use colliders::*;
pub(crate) use forces::*;
pub(crate) use frames::*;
pub(crate) use gear::*;
pub(crate) use ground::*;
pub(crate) use snap::*;
pub(crate) use spawn::*;
pub(crate) use terrain_patch::*;

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::DQuat;
    use thalos_body_render::{TerrainPatchBasis, TerrainPatchMesh};
    use thalos_physics_canonical::canonical::{Epoch, TranslationalState};
    use thalos_physics_canonical::surface_local::{SurfaceAnchor, SurfaceLocalFrame};
    use thalos_physics_canonical::types::{AttitudeState, BodyState};
    use thalos_physics_local::{LocalBubble, LocalPrimitiveShape};
    use thalos_shipyard::{AttachNodes, CommandPod, Engine, EngineGeometry};

    #[test]
    fn bubble_frame_round_trip_preserves_aggregate_state() {
        let basis = TerrainPatchBasis::from_normal(DVec3::Y);
        let patch = TerrainPatchMesh {
            vertices_body_m: Vec::new(),
            indices: Vec::new(),
            center_surface_body_m: DVec3::Y * 1000.0,
            basis,
            half_extent_m: 0.0,
        };
        let body = BodyState {
            id: 0,
            epoch: Epoch(0.0),
            position: DVec3::new(10.0, 20.0, 30.0),
            velocity: DVec3::new(5.0, 0.0, -1.0),
            orientation: DQuat::from_rotation_z(0.25),
            angular_velocity: DVec3::Y * 0.1,
            mass_kg: 1.0e20,
            gm: 1.0,
            radius_m: 1000.0,
        };
        let bubble = LocalBubble {
            id: 1,
            body_id: 0,
            craft_entity: Entity::PLACEHOLDER,
            frame: SurfaceLocalFrame::new(
                &body,
                SurfaceAnchor {
                    dir_body: DVec3::Y,
                    elevation_m: 0.0,
                },
            ),
            terrain_entity: Some(Entity::PLACEHOLDER),
            center_dir_body: DVec3::Y,
            center_surface_body_m: patch.center_surface_body_m,
            basis,
            patch_half_extent_m: 0.0,
            terrain_built_at_revision: 0,
        };
        let local_position = DVec3::new(12.0, 3.0, -7.0);
        let local_rotation = DQuat::from_rotation_x(0.2) * DQuat::from_rotation_y(-0.1);
        let local_velocity = DVec3::new(0.3, -0.4, 0.5);
        let local_angular_velocity = DVec3::new(0.01, 0.02, -0.03);

        let _ = bubble;
        let _ = patch;
        let (translation, attitude) = bubble_frame_to_inertial(
            &body,
            local_position,
            local_rotation,
            local_velocity,
            local_angular_velocity,
        );
        let round_trip = inertial_to_bubble_frame(&body, translation, attitude);

        assert!((round_trip.position_m - local_position).length() < 1e-9);
        assert!((round_trip.linear_velocity_m_s - local_velocity).length() < 1e-9);
        assert!(round_trip.rotation.angle_between(local_rotation) < 1e-9);
        assert!((round_trip.angular_velocity_rad_s - local_angular_velocity).length() < 1e-9);
    }

    #[test]
    fn handoff_round_trip_preserves_canonical_state() {
        // The take-translation handoff direction: the snap converts canonical
        // inertial → Avian bubble frame, the readback converts back. They must
        // compose to identity at orbital magnitudes, otherwise a handoff
        // injects a position/attitude jump (the `HANDOFF_RESIDUAL_TOLERANCE_M`
        // assertion in `readback_local_craft` would fire). This covers the
        // inertial→bubble→inertial direction; the test above covers the other.
        let body = BodyState {
            id: 0,
            epoch: Epoch(0.0),
            position: DVec3::new(-4.0e6, 1.2e6, 8.0e5),
            velocity: DVec3::new(120.0, -30.0, 7.0),
            orientation: DQuat::from_rotation_y(0.6) * DQuat::from_rotation_x(-0.2),
            angular_velocity: DVec3::new(0.0, 0.0, 7.292e-5),
            mass_kg: 5.0e22,
            gm: 3.3e12,
            radius_m: 1.6e6,
        };
        let translation = TranslationalState {
            position: body.position + DVec3::new(1.0e5, -2.0e5, 5.0e4),
            velocity: body.velocity + DVec3::new(-40.0, 60.0, -10.0),
        };
        let attitude = AttitudeState {
            orientation: DQuat::from_rotation_z(0.9) * DQuat::from_rotation_x(0.3),
            angular_velocity: DVec3::new(0.02, -0.01, 0.005),
        };

        let frame = inertial_to_bubble_frame(&body, translation, attitude);
        let (rt_translation, rt_attitude) = bubble_frame_to_inertial(
            &body,
            frame.position_m,
            frame.rotation,
            frame.linear_velocity_m_s,
            frame.angular_velocity_rad_s,
        );

        assert!((rt_translation.position - translation.position).length() < 1e-6);
        assert!((rt_translation.velocity - translation.velocity).length() < 1e-9);
        assert!(rt_attitude.orientation.angle_between(attitude.orientation) < 1e-9);
        assert!((rt_attitude.angular_velocity - attitude.angular_velocity).length() < 1e-9);
    }

    // The role *classification* tests (vacuum coast -> AttitudeOnly,
    // thrust/terrain/atmosphere -> Full, warp/BodyFixed -> Paused) live in
    // `thalos_physics_canonical::regime` since the A3 port — the resolver
    // is the classifier; only the projection lives here.

    #[test]
    fn integrator_is_active_in_attitude_only_and_full() {
        // The regression we're guarding against: pausing Avian in coast
        // mode killed rotation. `integrator_active` must be true for
        // both AttitudeOnly (so player rotation works) and Full.
        let attitude_only = AvianAuthority {
            role: AvianRole::AttitudeOnly,
            previous_role: AvianRole::AttitudeOnly,
        };
        let full = AvianAuthority {
            role: AvianRole::Full,
            previous_role: AvianRole::Full,
        };
        let paused = AvianAuthority {
            role: AvianRole::Paused,
            previous_role: AvianRole::Paused,
        };
        assert!(attitude_only.integrator_active());
        assert!(full.integrator_active());
        assert!(!paused.integrator_active());
    }

    #[test]
    fn owns_translation_is_only_full() {
        // Full is the only role where Avian's pos/vel are authoritative —
        // AttitudeOnly leaves translation to Kepler, Paused has nothing
        // running.
        let attitude_only = AvianAuthority {
            role: AvianRole::AttitudeOnly,
            previous_role: AvianRole::AttitudeOnly,
        };
        let full = AvianAuthority {
            role: AvianRole::Full,
            previous_role: AvianRole::Full,
        };
        let paused = AvianAuthority {
            role: AvianRole::Paused,
            previous_role: AvianRole::Paused,
        };
        assert!(!attitude_only.owns_translation());
        assert!(full.owns_translation());
        assert!(!paused.owns_translation());
    }

    #[test]
    fn just_took_translation_fires_only_on_transition_into_full() {
        // The handoff snap fires once when Avian takes translation
        // ownership, regardless of whether the previous role was
        // AttitudeOnly (typical thrust-on case) or Paused
        // (warp-down-with-throttle-on, landed-ship release).
        let cases = [
            (AvianRole::Paused, AvianRole::Full, true), // warp-down/landed release
            (AvianRole::AttitudeOnly, AvianRole::Full, true), // thrust-on
            (AvianRole::Full, AvianRole::Full, false),  // mid-burn
            (AvianRole::Full, AvianRole::AttitudeOnly, false), // burn-end
            (AvianRole::Full, AvianRole::Paused, false), // warp-up
            (AvianRole::AttitudeOnly, AvianRole::AttitudeOnly, false),
            (AvianRole::Paused, AvianRole::Paused, false),
        ];
        for (previous, current, want) in cases {
            let auth = AvianAuthority {
                role: current,
                previous_role: previous,
            };
            assert_eq!(
                auth.just_took_translation(),
                want,
                "previous={previous:?} current={current:?}"
            );
        }
    }

    #[test]
    fn pod_collider_uses_full_radius_cylinder() {
        let nodes = AttachNodes::default();
        let pod = CommandPod {
            model: "test".to_string(),
            geometry: Default::default(),
            diameter: 2.0,
            dry_mass: 0.0,
            reaction_wheel_torque: 0.0,
        };

        let (shape, offset) =
            part_collider_shape(&nodes, Some(&pod), None, None, None, None, None).unwrap();

        let LocalPrimitiveShape::Cylinder { radius, height } = shape else {
            panic!("pod collider should be a cylinder");
        };
        assert!((radius - 1.0).abs() < 1e-12);
        assert!((height - 1.8).abs() < 1e-6);
        assert!((offset - DVec3::Y * -(height * 0.5)).length() < 1e-12);
    }

    #[test]
    fn engine_collider_uses_full_radius_cylinder() {
        let nodes = AttachNodes::default();
        let engine = Engine {
            model: "test".to_string(),
            geometry: EngineGeometry::RocketBell,
            optimized_for: Default::default(),
            requires_atmosphere: false,
            intake_requirement: None,
            builtin_intake: None,
            diameter: 2.0,
            thrust: 0.0,
            isp: 0.0,
            sea_level_isp: None,
            dry_mass: 0.0,
            reactants: Vec::new(),
            power_draw_kw: 0.0,
            gimbal_range_deg: 0.0,
        };

        let (shape, offset) =
            part_collider_shape(&nodes, None, None, None, None, Some(&engine), None).unwrap();

        let LocalPrimitiveShape::Cylinder { radius, height } = shape else {
            panic!("engine collider should be a cylinder");
        };
        assert!((radius - 1.0).abs() < 1e-12);
        assert!((height - 1.8).abs() < 1e-6);
        assert!((offset - DVec3::Y * -(height * 0.5)).length() < 1e-12);
    }
}
