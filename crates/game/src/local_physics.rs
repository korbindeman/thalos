//! Game-side orchestration for the M5 aggregate local-physics bubble.

use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use thalos_local_physics::avian::{
    AngularVelocity, ConstantAngularAcceleration, ConstantLinearAcceleration, ContactGraph,
    LinearVelocity, Physics, PhysicsTime, Position, Rotation,
};
use thalos_local_physics::{
    ActiveLocalBubble, LocalBubble, LocalBubbleConfig, LocalCraftBody, LocalCraftSpawn,
    LocalPhysicsPlugin, LocalPrimitiveCollider, LocalPrimitiveShape, TerrainColliderPatch,
    TerrainSurfaceRegistry, craft_contacts_terrain, spawn_local_craft_body,
    spawn_terrain_collider_patch, stable_contact_reached,
};
use thalos_physics::body_centered::{
    BodyCenteredState, body_centered_to_inertial, inertial_to_body_centered,
};
use thalos_physics::body_fixed::body_fixed_pose_from_inertial;
use thalos_physics::canonical::{AuthorityMode, BodyFixedPose, EntityRef, TranslationalState};
use thalos_physics::types::{AttitudeState, BodyId, BodyState};
use thalos_shipyard::{Adapter, AttachNodes, CommandPod, Decoupler, Engine, FuelTank, Part};
use thalos_terrain::rendered_height_m;

use crate::SimStage;
use crate::debug::{DebugLaunchMount, DebugMode};
use crate::fuel::ThrottleState;
use crate::player_controller::{PlayerControllerBody, PlayerControllerState};
use crate::rendering::{PlayerShip, SimulationState};
use crate::view::ViewMode;

const THALOS_NAME: &str = "Thalos";
const DEBUG_DROP_KEY: KeyCode = KeyCode::F9;
const DEBUG_LAUNCH_MOUNT_RELEASE_THROTTLE: f64 = 0.001;

pub struct GameLocalPhysicsPlugin;

impl Plugin for GameLocalPhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(LocalPhysicsPlugin)
            .init_resource::<AvianAuthority>()
            .add_systems(
                Update,
                (
                    debug_surface_drop,
                    release_debug_launch_mount,
                    spawn_player_avian_body,
                    rebase_bubble_to_dominant_body,
                    attach_terrain_patch_when_close,
                    detach_terrain_patch_when_far,
                    compute_avian_authority,
                    manage_authority,
                    sync_avian_time,
                    snap_avian_from_canonical,
                    apply_local_forces,
                    readback_local_craft,
                    maintain_terrain_patch,
                    sync_terrain_collider_pose,
                    collapse_or_constrain_warp,
                    debug_log_body_fixed_state,
                )
                    .chain()
                    .in_set(SimStage::Physics)
                    .after(crate::bridge::advance_simulation),
            );
    }
}

/// What role does Avian play this frame?
///
/// Three roles, corresponding to three regimes of canonical/Avian
/// authority. The split exists because two distinct questions need
/// independent answers:
///
/// 1. *Should Avian's PhysicsSchedule step at all?* — needed for rotation
///    integration (player attitude commands, SAS damping) and for contact
///    detection. False under warp (numerical integration explodes at large
///    `dt`) and under `BodyFixed` (landed pose is analytic).
/// 2. *Should Avian's translation be authoritative?* — only when there is
///    a non-gravity force to integrate (thrust, contact). Otherwise
///    canonical Kepler owns translation, and AP/PE do not drift even when
///    Avian's clock keeps stepping for rotation.
///
/// Conflating the two — pausing Avian whenever it didn't own translation —
/// also paused rotation integration, which broke player rotation while
/// coasting. The split here keeps Avian's clock alive for rotation/contact
/// in coast mode while leaving translation to Kepler.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvianRole {
    /// Avian's clock is paused; canonical owns everything (translation,
    /// rotation, pose). Used at non-1× warp and under `BodyFixed`. The
    /// snap writes canonical state into Avian's components each frame so
    /// render and contact queries stay coherent without an integrator
    /// race.
    #[default]
    Paused,
    /// Avian's clock runs to integrate rotation under player/SAS torque
    /// and to keep the contact graph live, but Kepler owns translation.
    /// Used at 1× warp when the ship is coasting in vacuum (no thrust,
    /// no terrain collider attached). The snap writes canonical pos/vel
    /// into Avian each frame; rotation is left alone for Avian to
    /// integrate.
    AttitudeOnly,
    /// Avian owns both rotation and translation. Used at 1× warp when
    /// there is a non-gravity force to integrate (throttle active or
    /// terrain collider attached so contact resolution may need to fire).
    Full,
}

/// Per-frame Avian role + previous-frame role for edge detection.
///
/// Computed once at the top of the local-physics chain by
/// [`compute_avian_authority`] from canonical state + throttle + terrain
/// collider presence, so every downstream system reads a single
/// authoritative value instead of recomputing the predicate.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct AvianAuthority {
    pub role: AvianRole,
    pub previous_role: AvianRole,
}

impl AvianAuthority {
    /// True when Avian's `PhysicsSchedule` should step this frame —
    /// either coasting (rotation only) or full ownership.
    pub fn integrator_active(self) -> bool {
        !matches!(self.role, AvianRole::Paused)
    }

    /// True when Avian's translation (`Position`, `LinearVelocity`) is
    /// the authoritative source for canonical translation.
    pub fn owns_translation(self) -> bool {
        matches!(self.role, AvianRole::Full)
    }

    /// True on the single frame Avian transitions from not owning
    /// translation to owning it (Paused/AttitudeOnly → Full). The snap
    /// uses this to do a one-shot full-state push at the handoff so
    /// readback's conversion cancels exactly.
    pub fn just_took_translation(self) -> bool {
        matches!(self.role, AvianRole::Full) && !matches!(self.previous_role, AvianRole::Full)
    }
}

/// Resource-aware shim that unpacks ECS state and delegates to
/// [`avian_role_from_inputs`]. Split for unit-testability.
fn avian_role_for(
    sim: &SimulationState,
    throttle: &ThrottleState,
    active: &ActiveLocalBubble,
) -> AvianRole {
    let terrain_attached = active
        .bubble
        .as_ref()
        .and_then(|b| b.terrain_entity)
        .is_some();
    avian_role_from_inputs(
        sim.simulation.warp.speed(),
        sim.simulation.authority(),
        throttle.effective,
        terrain_attached,
    )
}

/// Pure predicate: classify Avian's role from raw inputs.
///
/// - **Warp ≠ 1×** → `Paused`. Time-stepped integration of central-force
///   gravity blows up at large `dt`, and we don't run rotation under warp
///   either (the existing convention zeroes ω at warp entry to avoid a
///   tap-rotate-then-warp leaving the ship spinning out).
/// - **`BodyFixed` authority** → `Paused`. Landed pose is analytic.
/// - **Throttle active OR terrain collider attached** → `Full`. We need
///   Avian to integrate the non-gravity force (thrust, contact) plus
///   gravity. Terrain-collider presence is the "contact is physically
///   possible here" signal — the collider only spawns inside the AGL
///   handoff band, so its existence flags us being close enough that
///   contact resolution must be live.
/// - **Otherwise (coasting in vacuum at 1× warp)** → `AttitudeOnly`. Avian
///   keeps integrating rotation and contact; Kepler owns translation, so
///   AP/PE don't drift across pause/unpause cycles.
fn avian_role_from_inputs(
    warp_speed: f64,
    authority: AuthorityMode,
    throttle_effective: f64,
    terrain_attached: bool,
) -> AvianRole {
    let near_one_x = (warp_speed - 1.0).abs() <= f64::EPSILON;
    if !near_one_x {
        return AvianRole::Paused;
    }
    if matches!(authority, AuthorityMode::BodyFixed { .. }) {
        return AvianRole::Paused;
    }
    let thrust_active = throttle_effective > 0.0;
    if thrust_active || terrain_attached {
        AvianRole::Full
    } else {
        AvianRole::AttitudeOnly
    }
}

fn compute_avian_authority(
    sim: Res<SimulationState>,
    throttle: Res<ThrottleState>,
    active: Res<ActiveLocalBubble>,
    mut authority: ResMut<AvianAuthority>,
) {
    authority.previous_role = authority.role;
    authority.role = avian_role_for(&sim, &throttle, &active);
}

fn sync_avian_time(
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    player: Option<Res<PlayerControllerState>>,
    mut physics_time: ResMut<Time<Physics>>,
) {
    // Avian's clock runs both for `Full` (translation+rotation+contact) and
    // `AttitudeOnly` (rotation+contact while Kepler owns translation).
    // Only `Paused` halts the integrator entirely.
    let player_active = player
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
        thalos_physics::canonical::Epoch(sim.simulation.sim_time()),
    )
}

fn agl_above_rendered_surface(
    body: &thalos_physics::types::BodyDefinition,
    body_state: &BodyState,
    surface: &thalos_terrain_gen::StaticSurfaceData,
    ship_position: DVec3,
) -> Option<(f64, DVec3, DVec3)> {
    let position_body = body_state.orientation.inverse() * (ship_position - body_state.position);
    let dir = position_body.try_normalize()?;
    let height = rendered_height_m(surface, dir.as_vec3()) as f64;
    let radius = body.radius_m + height;
    Some((position_body.length() - radius, dir, position_body))
}

/// Spawn the player ship's Avian rigid body the first time the simulation
/// is ready to host it (ship params populated, [`PlayerShip`] present, sane
/// dominant body). The body lives in body-centered inertial coordinates —
/// origin at the dominant body's centre, axes are the parent inertial axes
/// (no rotation). Gravity is a clean `−μr/r³` with no fictitious forces;
/// the terrain collider (when attached later) carries `Rotation =
/// body.orientation` so its body-fixed vertices land in the right place.
///
/// Avian owns rotation and live thrust in every regime — this system is the
/// single spawn point. Re-runs after [`clear_active_local_bubble`] tears the
/// bubble down on a debug teleport, so cmd-click / cmd-shift-click cleanly
/// rebuild around the new canonical state.
fn spawn_player_avian_body(
    mut commands: Commands,
    view: Res<ViewMode>,
    mut active: ResMut<ActiveLocalBubble>,
    sim: Res<SimulationState>,
    player_ship: Query<&GlobalTransform, With<PlayerShip>>,
    parts: PartColliderQuery,
) {
    if active.bubble.is_some() || *view != ViewMode::Ship {
        return;
    }
    if player_ship.iter().next().is_none() {
        return;
    }
    let params = *sim.simulation.ship_params();
    if params.moment_of_inertia.length_squared() <= 0.0 {
        // Ship spawn hasn't pushed real params yet.
        return;
    }
    let body_id = sim.simulation.dominant_body();
    let body_state = body_state_for(&sim, body_id);
    let craft = sim.simulation.craft_state();
    let frame = inertial_to_bubble_frame(&body_state, craft.translation, craft.attitude);
    let collider_primitives = build_ship_collider_primitives(&player_ship, &parts);
    let craft_entity = spawn_local_craft_body(
        &mut commands,
        LocalCraftSpawn {
            craft_id: craft.id,
            position_m: frame.position_m,
            rotation: frame.rotation,
            linear_velocity_m_s: frame.linear_velocity_m_s,
            angular_velocity_rad_s: frame.angular_velocity_rad_s,
            mass_kg: craft.mass.wet_mass_kg,
            angular_inertia_kg_m2: params.moment_of_inertia,
            collider_primitives,
        },
    );
    let bubble_id = active.allocate_id();
    active.bubble = Some(LocalBubble {
        id: bubble_id,
        body_id,
        craft_entity,
        terrain_entity: None,
        center_dir_body: DVec3::Y,
        center_surface_body_m: DVec3::ZERO,
        basis: thalos_terrain::TerrainPatchBasis::from_normal(DVec3::Y),
        stable_contact_s: 0.0,
        stable_landed: false,
    });
    // Authority is left at whatever Simulation::new chose (`OnRails`) on
    // spawn. `manage_authority` drives the LocalRigidBody transition next
    // frame if there's actually a non-gravity force to integrate (thrust,
    // contact). Defaulting to OnRails means a ship that spawns coasting in
    // orbit stays Kepler-driven instead of accumulating Avian integration
    // drift from frame one.
    info!(
        "spawned player Avian body bubble={} body_id={}",
        bubble_id, body_id
    );
}

/// Re-project the Avian rigid body onto the new dominant body's
/// body-centered inertial frame when the ship transits an SOI boundary.
/// `apply_local_forces` computes gravity against `bubble.body_id`, so a
/// stale value would pull the ship toward the body it just left.
///
/// Runs every frame but does work only when the dominant body actually
/// changes — cheap in the common case. The transformation is mediated
/// through canonical inertial state so we don't need to compute
/// body-to-body change-of-basis directly. Any attached terrain patch
/// belongs to the old body and is despawned;
/// `attach_terrain_patch_when_close` re-spawns over the new body on a
/// subsequent frame if the ship is close enough.
fn rebase_bubble_to_dominant_body(
    mut commands: Commands,
    mut active: ResMut<ActiveLocalBubble>,
    sim: Res<SimulationState>,
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
    let new_body_id = sim.simulation.dominant_body();
    if new_body_id == bubble.body_id {
        return;
    }
    let Ok((mut position, mut rotation, mut linear_velocity, mut angular_velocity)) =
        craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };
    let old_body_state = body_state_for(&sim, bubble.body_id);
    let (translation, attitude) = bubble_frame_to_inertial(
        &old_body_state,
        position.0,
        rotation.0,
        linear_velocity.0,
        angular_velocity.0,
    );
    let new_body_state = body_state_for(&sim, new_body_id);
    let frame = inertial_to_bubble_frame(&new_body_state, translation, attitude);
    position.0 = frame.position_m;
    rotation.0 = frame.rotation;
    linear_velocity.0 = frame.linear_velocity_m_s;
    angular_velocity.0 = frame.angular_velocity_rad_s;

    if let Some(terrain_entity) = bubble.terrain_entity.take() {
        commands.entity(terrain_entity).despawn();
    }
    bubble.center_dir_body = DVec3::Y;
    bubble.center_surface_body_m = DVec3::ZERO;
    bubble.basis = thalos_terrain::TerrainPatchBasis::from_normal(DVec3::Y);
    bubble.stable_contact_s = 0.0;
    bubble.stable_landed = false;
    let old_body_id = bubble.body_id;
    bubble.body_id = new_body_id;
    info!(
        "rebased local bubble across SOI transit: body_id {} -> {}",
        old_body_id, new_body_id
    );
}

/// Keep `AuthorityMode` aligned with [`AvianAuthority::owns_translation`].
///
/// `OnRails` is the default; we transition to `LocalRigidBody` only when
/// the role is [`AvianRole::Full`] — i.e., a non-gravity force is in play
/// (thrust, terrain contact possible) at 1× warp. Coasting in vacuum stays
/// `OnRails` so the Kepler propagator owns translation and AP/PE do not
/// drift across pause/unpause cycles or just from elapsed sim time. Note
/// that [`AvianRole::AttitudeOnly`] also stays `OnRails` — Avian is still
/// integrating rotation, but translation is canonical's responsibility.
///
/// `BodyFixed`, `WarpIntegrated`, and `Docked` are owned by other systems
/// (landed-pose evaluation, warp integrators, docking) and left alone.
///
/// Previously this function gated solely on warp level: at 1× warp, Avian
/// always owned; warping up handed translation back to Kepler. The result
/// was visible orbital drift any time the player paused/unpaused
/// mid-orbit, because Avian was integrating central-force gravity for a
/// ship that wasn't actually doing anything that needed an integrator.
fn manage_authority(
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    mut sim: ResMut<SimulationState>,
) {
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    match (sim.simulation.authority(), authority.owns_translation()) {
        (AuthorityMode::LocalRigidBody { .. }, false) => {
            sim.simulation
                .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
        }
        (AuthorityMode::OnRails { .. }, true) => {
            sim.simulation
                .transition_authority(AuthorityMode::LocalRigidBody {
                    bubble: bubble.id,
                    root_entity: EntityRef(bubble.craft_entity.to_bits()),
                });
        }
        _ => {}
    }
}

/// Attach a terrain collider patch when the ship enters the AGL handoff
/// band over a body whose surface is registered. The collider is
/// [`RigidBody::Kinematic`] with `Rotation = body.orientation`, so its
/// body-fixed vertices land at their inertial positions in the Avian rigid
/// body's frame. [`sync_terrain_collider_pose`] re-poses it each frame as
/// the body rotates.
fn attach_terrain_patch_when_close(
    mut commands: Commands,
    surfaces: Res<TerrainSurfaceRegistry>,
    config: Res<LocalBubbleConfig>,
    mut active: ResMut<ActiveLocalBubble>,
    sim: Res<SimulationState>,
) {
    let Some(bubble) = active.bubble.as_mut() else {
        return;
    };
    if bubble.terrain_entity.is_some() {
        return;
    }
    let Some(body_id) = thalos_body_id(&sim) else {
        return;
    };
    if bubble.body_id != body_id {
        return;
    }
    let Some(surface) = surfaces.get(body_id) else {
        return;
    };
    let body = &sim.system.bodies[body_id];
    let body_state = body_state_for(&sim, body_id);
    let craft = sim.simulation.craft_state();
    let Some((agl_m, center_dir, _)) = agl_above_rendered_surface(
        body,
        &body_state,
        &surface.static_surface,
        craft.translation.position,
    ) else {
        return;
    };
    if agl_m > config.handoff_agl_m {
        return;
    }
    let patch = spawn_terrain_collider_patch(
        &mut commands,
        body_id,
        &surface.static_surface,
        body.radius_m,
        center_dir,
        body_state.orientation,
        body_state.angular_velocity,
        &config,
    );
    bubble.terrain_entity = Some(patch.entity);
    bubble.center_dir_body = center_dir;
    bubble.center_surface_body_m = patch.mesh.center_surface_body_m;
    bubble.basis = patch.mesh.basis;
    info!(
        "attached terrain collider patch over {} at AGL {:.0} m",
        body.name, agl_m
    );
}

/// Despawn the terrain collider patch when the ship climbs back above the
/// handoff band (with hysteresis so we don't churn on the boundary).
fn detach_terrain_patch_when_far(
    mut commands: Commands,
    surfaces: Res<TerrainSurfaceRegistry>,
    config: Res<LocalBubbleConfig>,
    mut active: ResMut<ActiveLocalBubble>,
    sim: Res<SimulationState>,
) {
    let Some(bubble) = active.bubble.as_mut() else {
        return;
    };
    let Some(terrain_entity) = bubble.terrain_entity else {
        return;
    };
    let Some(surface) = surfaces.get(bubble.body_id) else {
        return;
    };
    let body = &sim.system.bodies[bubble.body_id];
    let body_state = body_state_for(&sim, bubble.body_id);
    let craft = sim.simulation.craft_state();
    let Some((agl_m, _, _)) = agl_above_rendered_surface(
        body,
        &body_state,
        &surface.static_surface,
        craft.translation.position,
    ) else {
        return;
    };
    // Hysteresis: detach at 1.5× the attach threshold.
    if agl_m <= config.handoff_agl_m * 1.5 {
        return;
    }
    commands.entity(terrain_entity).despawn();
    bubble.terrain_entity = None;
    bubble.center_dir_body = DVec3::Y;
    bubble.center_surface_body_m = DVec3::ZERO;
    bubble.basis = thalos_terrain::TerrainPatchBasis::from_normal(DVec3::Y);
    info!(
        "detached terrain collider patch from {} at AGL {:.0} m",
        body.name, agl_m
    );
}

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
fn snap_avian_from_canonical(
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
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
    let frame = inertial_to_bubble_frame(&body_state, craft.translation, craft.attitude);

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

fn debug_surface_drop(
    keys: Res<ButtonInput<KeyCode>>,
    debug: Option<Res<DebugMode>>,
    surfaces: Res<TerrainSurfaceRegistry>,
    config: Res<LocalBubbleConfig>,
    mut active: ResMut<ActiveLocalBubble>,
    mut launch_mount: ResMut<DebugLaunchMount>,
    mut sim: ResMut<SimulationState>,
) {
    if !keys.just_pressed(DEBUG_DROP_KEY) || !debug.as_deref().map(|d| d.enabled).unwrap_or(false) {
        return;
    }
    let Some(body_id) = thalos_body_id(&sim) else {
        return;
    };
    let Some(surface) = surfaces.get(body_id) else {
        warn!("debug surface drop requested before Thalos surface is available");
        return;
    };
    if let Some(bubble) = active.bubble.take() {
        warn!(
            "debug surface drop requested while local bubble {} is active; keeping current bubble",
            bubble.id
        );
        active.bubble = Some(bubble);
        return;
    }

    let body = sim.system.bodies[body_id].clone();
    let body_state = body_state_for(&sim, body_id);
    let dir = DVec3::new(0.271, 0.893, -0.361).normalize();
    let height = rendered_height_m(&surface.static_surface, dir.as_vec3()) as f64;
    let position_body = dir * (body.radius_m + height + config.debug_drop_height_m);
    let surface_velocity = body_state.velocity
        + body_state
            .angular_velocity
            .cross(body_state.orientation * position_body);
    let velocity = surface_velocity + body_state.orientation * (-dir * config.debug_drop_speed_m_s);
    let translation = TranslationalState {
        position: body_state.position + body_state.orientation * position_body,
        velocity,
    };
    let attitude = AttitudeState {
        orientation: level_attitude_for_body_dir(body_state.orientation, dir),
        angular_velocity: DVec3::ZERO,
    };
    sim.simulation
        .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
    sim.simulation
        .install_local_rigid_body_state(translation, attitude);
    sim.simulation.warp.reset();
    launch_mount.active = None;
    info!(
        "debug surface drop placed craft {:.0} m above rendered {} terrain",
        config.debug_drop_height_m, body.name
    );
}

fn release_debug_launch_mount(
    throttle: Res<ThrottleState>,
    mut launch_mount: ResMut<DebugLaunchMount>,
    mut sim: ResMut<SimulationState>,
) {
    if throttle.commanded <= DEBUG_LAUNCH_MOUNT_RELEASE_THROTTLE {
        return;
    }
    let Some(mount) = launch_mount.active else {
        return;
    };
    let AuthorityMode::BodyFixed { body, pose } = sim.simulation.authority() else {
        launch_mount.active = None;
        return;
    };
    if body != mount.body_id || pose != mount.pose {
        launch_mount.active = None;
        return;
    }

    // Debug-only launch clamp release. This is a temporary staging substitute:
    // remove it when real staging/launch-clamp parts own attach and release.
    sim.simulation
        .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
    sim.simulation.warp.reset_immediate();
    launch_mount.active = None;
    let body = &sim.system.bodies[mount.body_id];
    let gravity_m_s2 = body.gm / body.radius_m.powi(2);
    let mass_kg = sim.simulation.ship_mass_kg();
    let thrust_n = sim.simulation.ship_params().thrust_n;
    let twr = if mass_kg > 0.0 && gravity_m_s2 > 0.0 {
        thrust_n / (mass_kg * gravity_m_s2)
    } else {
        0.0
    };
    info!(
        "released debug launch mount on commanded throttle {:.2}; thrust={:.0} N mass={:.0} kg local TWR={:.2}",
        throttle.commanded, thrust_n, mass_kg, twr
    );
}

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
fn apply_local_forces(
    time: Res<Time>,
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    mut sim: ResMut<SimulationState>,
    throttle: Res<ThrottleState>,
    mut craft_q: Query<(
        &Position,
        &Rotation,
        &AngularVelocity,
        &mut ConstantLinearAcceleration,
        &mut ConstantAngularAcceleration,
        &LocalCraftBody,
    )>,
) {
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    if !authority.integrator_active() {
        // Avian's clock is paused; the snap will zero accel on its way out.
        return;
    }
    let Ok((position, rotation, angular_velocity, mut linear_accel, mut angular_accel, _)) =
        craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };
    let params = *sim.simulation.ship_params();

    // Linear: gravity + thrust only when Avian owns translation. Otherwise
    // explicitly zero so a stale value from the previous `Full` frame
    // doesn't drift Avian's pos/vel away from Kepler's authoritative state.
    if authority.owns_translation() {
        let body = &sim.system.bodies[bubble.body_id];
        // Avian's body lives in body-centered inertial coordinates: the
        // dominant body's centre is the origin and the axes don't rotate. So
        // `position.0` is the inertial offset from the body's centre and
        // gravity is the textbook two-body `−μr/r³` — no Coriolis or
        // centrifugal needed.
        let body_pos = position.0;
        if body_pos.length_squared() > 0.0 {
            let gravity_accel = -body.gm * body_pos / body_pos.length().powi(3);
            let mut accel = gravity_accel;
            let throttle_eff = throttle.effective.clamp(0.0, 1.0);
            let mass = sim.simulation.ship_mass_kg();
            if throttle_eff > 0.0 && params.thrust_n > 0.0 && mass > params.dry_mass_kg {
                let nose_world = rotation.0 * DVec3::Y;
                accel += nose_world * (params.thrust_n / mass) * throttle_eff;
                sim.simulation
                    .apply_external_mass_flow(throttle_eff, time.delta_secs_f64());
            }
            linear_accel.0 = accel;
        } else {
            linear_accel.0 = DVec3::ZERO;
        }
    } else {
        linear_accel.0 = DVec3::ZERO;
    }

    // Angular accel always written when the integrator is active, in both
    // `AttitudeOnly` and `Full`. This is the system that lets the player
    // rotate the ship while coasting.
    angular_accel.0 = compute_angular_acceleration(
        sim.simulation.control(),
        &params,
        rotation.0,
        angular_velocity.0,
        time.delta_secs_f64(),
    );
}

/// Convert player attitude command + SAS damping into a world-space angular
/// acceleration for the Avian rigid body. Matches `Simulation::integrate_attitude`
/// (now removed) so the rotational feel is identical whether the ship is in
/// deep space or on a surface — Avian is the integrator in both cases.
fn compute_angular_acceleration(
    control: &thalos_physics::types::ControlInput,
    params: &thalos_physics::types::ShipParameters,
    rotation: DQuat,
    angular_velocity_world: DVec3,
    dt: f64,
) -> DVec3 {
    let inertia_body = params.moment_of_inertia;
    let max_torque = params.max_torque;
    let cmd = control
        .torque_command
        .clamp(DVec3::splat(-1.0), DVec3::splat(1.0));
    let no_input = cmd.length_squared() < 1e-6;

    let torque_body = if control.sas_enabled && no_input {
        if dt <= 0.0 {
            DVec3::ZERO
        } else {
            let omega_body = rotation.inverse() * angular_velocity_world;
            (-inertia_body * omega_body / dt).clamp(-max_torque, max_torque)
        }
    } else {
        cmd * max_torque
    };

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
fn readback_local_craft(
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    mut sim: ResMut<SimulationState>,
    craft_q: Query<(
        &Position,
        &Rotation,
        &LinearVelocity,
        &AngularVelocity,
        &LocalCraftBody,
    )>,
) {
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    if !authority.integrator_active() {
        return;
    }
    let Ok((position, rotation, linear_velocity, angular_velocity, _)) =
        craft_q.get(bubble.craft_entity)
    else {
        return;
    };
    let body_state = body_state_for(&sim, bubble.body_id);
    let (translation, attitude) = bubble_frame_to_inertial(
        &body_state,
        position.0,
        rotation.0,
        linear_velocity.0,
        angular_velocity.0,
    );
    if authority.owns_translation() {
        sim.simulation
            .install_local_rigid_body_state(translation, attitude);
    } else {
        // AttitudeOnly: attitude flows back, translation stays Kepler-owned.
        sim.simulation.set_attitude(attitude);
    }
}

fn maintain_terrain_patch(
    mut commands: Commands,
    surfaces: Res<TerrainSurfaceRegistry>,
    config: Res<LocalBubbleConfig>,
    mut active: ResMut<ActiveLocalBubble>,
    sim: Res<SimulationState>,
    craft_q: Query<&Position, With<LocalCraftBody>>,
    player: Option<Res<PlayerControllerState>>,
    player_q: Query<&Position, With<PlayerControllerBody>>,
) {
    let Some(current) = active.bubble.clone() else {
        return;
    };
    if current.terrain_entity.is_none() {
        return;
    }
    let player_position = player
        .as_deref()
        .and_then(|state| state.is_active().then_some(()))
        .and_then(|_| player_q.iter().next());
    let position = if let Some(position) = player_position {
        position
    } else {
        let Ok(position) = craft_q.get(current.craft_entity) else {
            return;
        };
        position
    };
    // Avian's body is in body-centered inertial; the patch metadata
    // (`center_surface_body_m`, `center_dir_body`) is in body-fixed.
    // Rotate the craft position into body-fixed before the lateral check.
    let body_state = body_state_for(&sim, current.body_id);
    let craft_body_fixed = body_state.orientation.inverse() * position.0;
    let delta = craft_body_fixed - current.center_surface_body_m;
    let along = delta.dot(current.center_dir_body);
    let lateral = (delta - along * current.center_dir_body).length();
    if lateral <= config.patch_rebuild_distance_m {
        return;
    }
    let Some(surface) = surfaces.get(current.body_id) else {
        return;
    };
    let body = &sim.system.bodies[current.body_id];
    let center_dir = craft_body_fixed.normalize_or_zero();
    if center_dir == DVec3::ZERO {
        return;
    }
    if let Some(terrain_entity) = current.terrain_entity {
        commands.entity(terrain_entity).despawn();
    }
    let patch = spawn_terrain_collider_patch(
        &mut commands,
        current.body_id,
        &surface.static_surface,
        body.radius_m,
        center_dir,
        body_state.orientation,
        body_state.angular_velocity,
        &config,
    );
    active.bubble = Some(LocalBubble {
        terrain_entity: Some(patch.entity),
        center_dir_body: center_dir,
        center_surface_body_m: patch.mesh.center_surface_body_m,
        basis: patch.mesh.basis,
        stable_contact_s: current.stable_contact_s,
        stable_landed: current.stable_landed,
        ..current
    });
}

/// Pose the kinematic terrain collider to match the dominant body's current
/// orientation and angular velocity. The collider's local vertices are
/// body-fixed; `Rotation = body.orientation` carries them into body-centered
/// inertial each frame, and `AngularVelocity = body.angular_velocity` gives
/// contact resolution the correct surface velocity at the contact point so
/// a craft sitting on the spinning surface feels itself co-rotate.
fn sync_terrain_collider_pose(
    active: Res<ActiveLocalBubble>,
    sim: Res<SimulationState>,
    mut terrain_q: Query<
        (&mut Rotation, &mut AngularVelocity),
        (With<TerrainColliderPatch>, Without<LocalCraftBody>),
    >,
) {
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Some(terrain_entity) = bubble.terrain_entity else {
        return;
    };
    let Ok((mut rotation, mut angular_velocity)) = terrain_q.get_mut(terrain_entity) else {
        return;
    };
    let body_state = body_state_for(&sim, bubble.body_id);
    rotation.0 = body_state.orientation;
    angular_velocity.0 = body_state.angular_velocity;
}

/// Track stable-contact landing and collapse to `BodyFixed` when the ship
/// settles. Warp gating now lives in [`manage_authority`], which derives
/// authority from [`AvianOwnership`] (warp ≠ 1× falls back to `OnRails`
/// regardless of contact). Warping no longer requires tearing down the
/// bubble.
fn collapse_or_constrain_warp(
    mut commands: Commands,
    time: Res<Time>,
    contact_graph: Res<ContactGraph>,
    config: Res<LocalBubbleConfig>,
    throttle: Res<ThrottleState>,
    mut active: ResMut<ActiveLocalBubble>,
    mut sim: ResMut<SimulationState>,
    craft_q: Query<(&LinearVelocity, &AngularVelocity), With<LocalCraftBody>>,
) {
    let Some(mut bubble) = active.bubble.clone() else {
        return;
    };
    if matches!(sim.simulation.authority(), AuthorityMode::BodyFixed { .. }) {
        bubble.stable_contact_s = 0.0;
        bubble.stable_landed = false;
        active.bubble = Some(bubble);
        return;
    }
    // Only track stable contact while a terrain patch is actually attached;
    // contact graph queries against `None` would be vacuously false anyway,
    // but skipping early keeps the timer from accumulating in deep space.
    let Some(terrain_entity) = bubble.terrain_entity else {
        bubble.stable_contact_s = 0.0;
        bubble.stable_landed = false;
        active.bubble = Some(bubble);
        return;
    };
    let Ok((linear_velocity, angular_velocity)) = craft_q.get(bubble.craft_entity) else {
        return;
    };
    // Contact graph is the source of truth — a terrain-aware altitude
    // fallback would need to sample the rendered cubemap at the craft's
    // body-fixed direction (radius + terrain_height varies by km), so we
    // skip the half-correct radial shortcut the previous body-fixed
    // implementation had. If F9 surface-drop ever fails to register
    // contact, add a proper AGL fallback via `agl_above_rendered_surface`.
    let contact = craft_contacts_terrain(&contact_graph, bubble.craft_entity, terrain_entity);
    bubble.stable_landed = stable_contact_reached(
        &mut bubble.stable_contact_s,
        time.delta_secs_f64(),
        contact,
        linear_velocity.length(),
        angular_velocity.length(),
        throttle.effective,
        &config,
    );

    if bubble.stable_landed {
        collapse_to_body_fixed(&mut sim, &bubble);
        // Avian body persists; reset stability tracking and let
        // snap_avian_from_canonical keep the rigid body aligned with the
        // BodyFixed pose until throttle-up releases the clamp.
        bubble.stable_contact_s = 0.0;
        bubble.stable_landed = false;
    }
    let _ = &mut commands;
    active.bubble = Some(bubble);
}

/// Temporary diagnostic: log canonical attitude + Avian state while
/// [`AuthorityMode::BodyFixed`] is active, for ~60 frames per entry into
/// the mode. Helps catch teleport-clamp drift without spamming the log
/// once a session settles in.
fn debug_log_body_fixed_state(
    sim: Res<SimulationState>,
    active: Res<ActiveLocalBubble>,
    craft_q: Query<(&Position, &Rotation, &AngularVelocity), With<LocalCraftBody>>,
    mut frames_in_mode: Local<u32>,
    mut was_body_fixed: Local<bool>,
) {
    let is_body_fixed = matches!(sim.simulation.authority(), AuthorityMode::BodyFixed { .. });
    if !is_body_fixed {
        *frames_in_mode = 0;
        *was_body_fixed = false;
        return;
    }
    if !*was_body_fixed {
        *frames_in_mode = 0;
        *was_body_fixed = true;
    }
    *frames_in_mode += 1;
    if *frames_in_mode > 60 {
        return;
    }
    let attitude = sim.simulation.attitude();
    let body_id = sim.simulation.dominant_body();
    let body_state = body_state_for(&sim, body_id);
    let avian_state = active.bubble.as_ref().and_then(|b| {
        craft_q
            .get(b.craft_entity)
            .ok()
            .map(|(p, r, av)| (p.0, r.0, av.0))
    });
    if let Some((pos, rot, av)) = avian_state {
        info!(
            "BodyFixed frame {}: canon ori.w={:.5} canon ω=({:.2e},{:.2e},{:.2e}) | avian pos.y={:.1} rot.w={:.5} ω=({:.2e},{:.2e},{:.2e}) | body.ori.w={:.5} body.ω=({:.2e},{:.2e},{:.2e})",
            *frames_in_mode,
            attitude.orientation.w,
            attitude.angular_velocity.x,
            attitude.angular_velocity.y,
            attitude.angular_velocity.z,
            pos.y,
            rot.w,
            av.x,
            av.y,
            av.z,
            body_state.orientation.w,
            body_state.angular_velocity.x,
            body_state.angular_velocity.y,
            body_state.angular_velocity.z,
        );
    } else {
        info!(
            "BodyFixed frame {}: canon ori.w={:.5} canon ω=({:.2e},{:.2e},{:.2e}) | no avian | body.ori.w={:.5}",
            *frames_in_mode,
            attitude.orientation.w,
            attitude.angular_velocity.x,
            attitude.angular_velocity.y,
            attitude.angular_velocity.z,
            body_state.orientation.w,
        );
    }
}

/// Transition canonical authority to `BodyFixed` once the ship has settled
/// onto the terrain. The Avian body itself stays alive — Avian remains the
/// universal rigid-body integrator, and [`snap_avian_from_canonical`] holds
/// it on the body-fixed pose until throttle-up triggers
/// [`release_debug_launch_mount`].
fn collapse_to_body_fixed(sim: &mut SimulationState, bubble: &LocalBubble) {
    let body_state = body_state_for(sim, bubble.body_id);
    let craft = sim.simulation.craft_state();
    let pose: BodyFixedPose =
        body_fixed_pose_from_inertial(&body_state, craft.translation, craft.attitude);
    sim.simulation
        .transition_authority(AuthorityMode::BodyFixed {
            body: bubble.body_id,
            pose,
        });
    info!("collapsed stable landed craft to BodyFixed authority");
}

struct BubbleFrame {
    position_m: DVec3,
    rotation: DQuat,
    linear_velocity_m_s: DVec3,
    angular_velocity_rad_s: DVec3,
}

/// Convert canonical inertial state into the Avian rigid body's frame.
///
/// The Avian body lives in **body-centered inertial** coordinates: the origin
/// tracks the dominant body's centre but the axes are the parent inertial
/// axes (no rotation). Position and velocity are simple offsets from the
/// body's centre, and the craft's attitude / angular velocity are expressed
/// in inertial axes — matching how Avian treats the frame it integrates in.
///
/// Avian's `AngularVelocity` lives in the rigid body's surrounding frame
/// (here, inertial), while [`AttitudeState::angular_velocity`] is expressed
/// in the craft body frame, so we rotate by `orientation`.
fn inertial_to_bubble_frame(
    body_state: &BodyState,
    translation: TranslationalState,
    attitude: AttitudeState,
) -> BubbleFrame {
    let state = inertial_to_body_centered(body_state, translation, attitude);
    BubbleFrame {
        position_m: state.translation_bc.position,
        rotation: state.attitude.orientation.normalize(),
        linear_velocity_m_s: state.translation_bc.velocity,
        angular_velocity_rad_s: state.attitude.orientation * state.attitude.angular_velocity,
    }
}

fn bubble_frame_to_inertial(
    body_state: &BodyState,
    position_m: DVec3,
    rotation: DQuat,
    linear_velocity_m_s: DVec3,
    angular_velocity_rad_s: DVec3,
) -> (TranslationalState, AttitudeState) {
    let orientation = rotation.normalize();
    let state = BodyCenteredState {
        translation_bc: TranslationalState {
            position: position_m,
            velocity: linear_velocity_m_s,
        },
        attitude: AttitudeState {
            orientation,
            angular_velocity: orientation.inverse() * angular_velocity_rad_s,
        },
    };
    body_centered_to_inertial(body_state, state)
}

fn level_attitude_for_body_dir(body_orientation: DQuat, up_body: DVec3) -> DQuat {
    let basis = thalos_terrain::TerrainPatchBasis::from_normal(up_body);
    let nose_body = basis.tangent_z;
    let dorsal_body = up_body.normalize();
    let right_body = nose_body.cross(dorsal_body).normalize();
    let craft_to_body = DMat3::from_cols(right_body, nose_body, dorsal_body);
    (body_orientation * DQuat::from_mat3(&craft_to_body)).normalize()
}

type PartColliderQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static GlobalTransform,
        &'static AttachNodes,
        Option<&'static CommandPod>,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Engine>,
    ),
    With<Part>,
>;

fn build_ship_collider_primitives(
    player_ship: &Query<&GlobalTransform, With<PlayerShip>>,
    parts: &PartColliderQuery,
) -> Vec<LocalPrimitiveCollider> {
    let Ok(root) = player_ship.single() else {
        return vec![fallback_collider()];
    };
    let root_inv = root.affine().inverse();
    let mut primitives = Vec::new();
    for (global, nodes, pod, dec, adapter, tank, engine) in parts.iter() {
        let part_affine = root_inv * global.affine();
        let (_, rotation, translation) = part_affine.to_scale_rotation_translation();
        let rotation = rotation.as_dquat();
        let Some((shape, local_offset)) = part_shape(nodes, pod, dec, adapter, tank, engine) else {
            continue;
        };
        primitives.push(LocalPrimitiveCollider {
            offset_m: translation.as_dvec3() + rotation * local_offset,
            rotation,
            shape,
        });
    }
    if primitives.is_empty() {
        primitives.push(fallback_collider());
    }
    primitives
}

fn fallback_collider() -> LocalPrimitiveCollider {
    LocalPrimitiveCollider {
        offset_m: DVec3::ZERO,
        rotation: DQuat::IDENTITY,
        shape: LocalPrimitiveShape::Cuboid {
            x: 2.0,
            y: 6.0,
            z: 2.0,
        },
    }
}

fn part_shape(
    nodes: &AttachNodes,
    pod: Option<&CommandPod>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    tank: Option<&FuelTank>,
    engine: Option<&Engine>,
) -> Option<(LocalPrimitiveShape, DVec3)> {
    if let Some(pod) = pod {
        let height = pod.diameter * 0.9;
        Some((
            LocalPrimitiveShape::Cone {
                radius: (pod.diameter * 0.5) as f64,
                height: height as f64,
            },
            DVec3::Y * -(height as f64 * 0.5),
        ))
    } else if dec.is_some() {
        let diameter = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let height = 0.2;
        Some((
            LocalPrimitiveShape::Cylinder {
                radius: (diameter * 0.5) as f64,
                height,
            },
            DVec3::Y * -(height * 0.5),
        ))
    } else if let Some(adapter) = adapter {
        let top_d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let bot_d = adapter.target_diameter;
        let height = ((top_d + bot_d) * 0.5).max(0.4);
        Some((
            LocalPrimitiveShape::Cylinder {
                radius: (top_d.max(bot_d) * 0.5) as f64,
                height: height as f64,
            },
            DVec3::Y * -(height as f64 * 0.5),
        ))
    } else if let Some(tank) = tank {
        let diameter = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        Some((
            LocalPrimitiveShape::Cylinder {
                radius: (diameter * 0.5) as f64,
                height: tank.length as f64,
            },
            DVec3::Y * -(tank.length as f64 * 0.5),
        ))
    } else if let Some(engine) = engine {
        let height = engine.diameter * 0.9;
        Some((
            LocalPrimitiveShape::Cone {
                radius: (engine.diameter * 0.5) as f64,
                height: height as f64,
            },
            DVec3::Y * -(height as f64 * 0.5),
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thalos_physics::canonical::Epoch;
    use thalos_physics::types::BodyState;
    use thalos_terrain::{TerrainPatchBasis, TerrainPatchMesh};

    #[test]
    fn bubble_frame_round_trip_preserves_aggregate_state() {
        let basis = TerrainPatchBasis::from_normal(DVec3::Y);
        let patch = TerrainPatchMesh {
            vertices_body_m: Vec::new(),
            indices: Vec::new(),
            center_surface_body_m: DVec3::Y * 1000.0,
            basis,
        };
        let bubble = LocalBubble {
            id: 1,
            body_id: 0,
            craft_entity: Entity::PLACEHOLDER,
            terrain_entity: Some(Entity::PLACEHOLDER),
            center_dir_body: DVec3::Y,
            center_surface_body_m: patch.center_surface_body_m,
            basis,
            stable_contact_s: 0.0,
            stable_landed: false,
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

    fn on_rails() -> AuthorityMode {
        AuthorityMode::OnRails { trajectory: 0 }
    }

    fn body_fixed() -> AuthorityMode {
        AuthorityMode::BodyFixed {
            body: 0,
            pose: thalos_physics::canonical::BodyFixedPose {
                position_body_m: DVec3::Y * 1000.0,
                orientation_body: DQuat::IDENTITY,
            },
        }
    }

    #[test]
    fn coasting_in_vacuum_at_one_x_uses_attitude_only() {
        // The bug we're fixing: at 1× warp with no thrust and no contact,
        // Avian was integrating gravity and producing visible AP/PE drift.
        // The role here must be `AttitudeOnly` so Kepler owns translation
        // (no drift) while Avian's clock keeps stepping for rotation —
        // otherwise the player can't rotate the ship while coasting.
        assert_eq!(
            avian_role_from_inputs(1.0, on_rails(), 0.0, false),
            AvianRole::AttitudeOnly
        );
    }

    #[test]
    fn thrust_at_one_x_takes_full_ownership() {
        assert_eq!(
            avian_role_from_inputs(1.0, on_rails(), 0.5, false),
            AvianRole::Full
        );
    }

    #[test]
    fn terrain_collider_attached_at_one_x_takes_full_ownership() {
        // Inside the AGL handoff band the terrain collider is present;
        // Avian needs to own translation so contact resolution can fire.
        assert_eq!(
            avian_role_from_inputs(1.0, on_rails(), 0.0, true),
            AvianRole::Full
        );
    }

    #[test]
    fn high_warp_pauses_avian_entirely() {
        // Avian's integrator can't take warp-sized timesteps; the
        // physical-state triggers do not override the warp guard.
        // Rotation also stops integrating (matching the existing
        // "ω is zeroed at warp entry" convention).
        assert_eq!(
            avian_role_from_inputs(10.0, on_rails(), 1.0, true),
            AvianRole::Paused
        );
        assert_eq!(
            avian_role_from_inputs(1_000_000.0, on_rails(), 0.5, false),
            AvianRole::Paused
        );
    }

    #[test]
    fn body_fixed_authority_pauses_avian() {
        // Landed pose is evaluated analytically from the body's rotation;
        // Avian holds the rigid body in place but does not integrate. This
        // must hold even with thrust applied — `release_debug_launch_mount`
        // releases the clamp by transitioning out of BodyFixed first.
        assert_eq!(
            avian_role_from_inputs(1.0, body_fixed(), 0.0, false),
            AvianRole::Paused
        );
        assert_eq!(
            avian_role_from_inputs(1.0, body_fixed(), 0.9, true),
            AvianRole::Paused
        );
    }

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
        // (warp-down-with-throttle-on, launch-clamp release).
        let cases = [
            (AvianRole::Paused, AvianRole::Full, true), // warp-down/clamp release
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
}
