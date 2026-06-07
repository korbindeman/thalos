//! Game-side orchestration for the M5 aggregate local-physics bubble.

use std::collections::{HashMap, VecDeque};

use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use thalos_physics_canonical::body_centered::{
    BodyCenteredState, body_centered_to_inertial, inertial_to_body_centered,
};
use thalos_physics_canonical::body_fixed::body_fixed_pose_from_inertial;
use thalos_physics_canonical::canonical::{
    AuthorityMode, BodyFixedPose, EntityRef, TranslationalState,
};
use thalos_world::BodyId;
use thalos_physics_canonical::types::{AttitudeState, BodyState, VesselKind};
use thalos_physics_local::avian::{
    AngularVelocity, Collider, ConstantAngularAcceleration, ConstantLinearAcceleration,
    ContactGraph, CustomPositionIntegration, LinearVelocity, LockedAxes, Physics, PhysicsTime,
    Position, RigidBody, Rotation,
};
use thalos_physics_local::{
    ActiveLocalBubble, HeightSourceRegistry, LocalBubble, LocalBubbleConfig, LocalCraftBody,
    LocalCraftSpawn, LocalPhysicsPlugin, LocalPrimitiveCollider, LocalPrimitiveShape,
    TerrainColliderPatch, craft_contacts_terrain, spawn_local_craft_body,
    spawn_terrain_collider_patch, stable_contact_reached, terrain_patch_pose,
};
use thalos_shipyard::{
    Adapter, AttachNodes, Attachment, CommandPod, Decoupler, Engine, FuelTank, Part,
};
use thalos_body_render::HeightSource;

use crate::SimStage;
use crate::bridge::WarpLimits;
use crate::debug::{DebugLaunchMount, DebugMode};
use crate::fuel::ThrottleState;
use crate::player_controller::{EvaMode, PlayerControllerBody, PlayerControllerState};
use crate::rendering::{PlayerShip, SimulationState};
use crate::sim_clock::SimClock;
use crate::view::ViewMode;

/// `tile_lod_m` hint for queries that want the finest CPU-synthesizable
/// terrain detail. GPU-backed height sources prefer the resident atlas
/// when populated; when they fall back to the CPU pipeline this hint
/// drives `compute_detail_height` to its full octave count.
pub const PHYSICS_QUERY_TILE_LOD_M: f32 = 0.5;

const THALOS_NAME: &str = "Thalos";
const DEBUG_DROP_KEY: KeyCode = KeyCode::F9;
const DEBUG_LAUNCH_MOUNT_RELEASE_THROTTLE: f64 = 0.001;

/// Position discontinuity above which a take-translation handoff is treated
/// as a bug in debug builds. A healthy handoff residual is the distance
/// Avian's integrator drifts from the snap source in one step (`~|accel|·dt²`,
/// sub-centimetre). The frame-skew / SOI-race failure the snap guards against
/// produces `~|relative_velocity|·dt` (~100 m at Thalos LEO), so 2 m cleanly
/// separates the two regimes.
const HANDOFF_RESIDUAL_TOLERANCE_M: f64 = 2.0;

pub struct GameLocalPhysicsPlugin;

impl Plugin for GameLocalPhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(LocalPhysicsPlugin)
            .init_resource::<AvianAuthority>()
            .init_resource::<AvianHandoffDiagnostics>()
            .register_type::<AvianRole>()
            .register_type::<AvianAuthority>()
            .register_type::<AvianHandoffDiagnostics>()
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
                    detect_terrain_impact,
                    maintain_terrain_patch,
                    sync_terrain_collider_pose,
                    collapse_or_constrain_warp,
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
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
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
#[derive(Resource, Default, Debug, Clone, Copy, Reflect)]
#[reflect(Resource)]
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

/// Observability for the canonical↔Avian translation handoff. Recorded at
/// each authority transition by [`manage_authority`] (direction + time) and
/// [`readback_local_craft`] (the residual measured when Avian first takes
/// translation). Reflect-registered so an agent can read it over BRP without
/// attaching a debugger.
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

/// Terrain colliders are the expensive near-ground contact path, not the
/// general "local bubble exists" signal. Build and refresh them only once the
/// altitude warp gate has forced the craft into the live 1x zone.
fn terrain_colliders_allowed_by_warp(sim: &SimulationState, limits: &WarpLimits) -> bool {
    let warp = &sim.simulation.warp;
    terrain_colliders_allowed_by_warp_inputs(warp.speed(), warp.levels(), limits.max_level)
}

fn terrain_colliders_allowed_by_warp_inputs(
    warp_speed: f64,
    warp_levels: &[f64],
    max_level: usize,
) -> bool {
    let Some(one_x_index) = warp_levels
        .iter()
        .position(|&speed| (speed - 1.0).abs() <= f64::EPSILON)
    else {
        return false;
    };
    (warp_speed - 1.0).abs() <= f64::EPSILON && max_level <= one_x_index
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
    // `SimClock` is a hard pause over that role classifier so menu/freecam/warp
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

/// Spawn the player's Avian rigid body the first time the simulation is
/// ready to host it. The body lives in body-centered inertial coordinates
/// — origin at the dominant body's centre, axes are the parent inertial
/// axes (no rotation). Gravity is a clean `−μr/r³` with no fictitious
/// forces; the terrain collider (when attached later) is centered on its
/// patch and carries `Rotation = body.orientation` so its body-fixed
/// vertex offsets land in the right place.
///
/// Two vessel kinds spawn through this single seam, KSP-style:
/// - `VesselKind::Ship`: waits for `PlayerShip` + ship params, then
///   spawns a compound collider built from the rendered ship parts.
/// - `VesselKind::Eva`: spawns a 1.8 m capsule with rotation locked and
///   walking-friendly friction. The same entity carries
///   `PlayerControllerBody` so the EVA controller's systems find it.
///
/// Avian owns rotation and live thrust for ships; for EVA, rotation is
/// fully locked and the controller drives translation directly.
#[allow(clippy::too_many_arguments)]
fn spawn_player_avian_body(
    mut commands: Commands,
    view: Res<ViewMode>,
    mut active: ResMut<ActiveLocalBubble>,
    mut sim: ResMut<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    player_ship: Query<&GlobalTransform, With<PlayerShip>>,
    parts: PartColliderQuery,
) {
    if active.bubble.is_some() || *view != ViewMode::Ship {
        return;
    }
    let vessel_kind = sim.simulation.vessel_kind();
    let params = *sim.simulation.ship_params();
    if params.moment_of_inertia.length_squared() <= 0.0 {
        return;
    }
    let body_id = sim.simulation.dominant_body();
    let body_state = body_state_for(&sim, body_id);

    // EVA refines its canonical spawn pose to sit just above the
    // rendered terrain at the sub-stellar point (daylight) before the
    // Avian body is created. main.rs only knows the body radius, so it
    // seeds the rough 12 km drop; once the height source exists, we can
    // plant the player at the actual terrain.
    if vessel_kind == VesselKind::Eva {
        let Some(height_source) = height_sources.get(body_id) else {
            return;
        };
        let body = &sim.system.bodies[body_id];
        let sun_dir_inertial = (-body_state.position).normalize_or_zero();
        let mut dir_body_fixed = if sun_dir_inertial == DVec3::ZERO {
            DVec3::Y
        } else {
            (body_state.orientation.inverse() * sun_dir_inertial).normalize()
        };
        // EVA drop-site selection, searching the daylight hemisphere near the
        // sub-stellar point:
        //   default / `plain` → flattest usable plain (the intended on-foot start),
        //   `relief`          → highest-relief hill site (terrain inspection),
        //   `substellar`      → exact sub-stellar point (legacy behaviour).
        let eva_site = std::env::var("THALOS_EVA_SITE").ok();
        if eva_site.as_deref() != Some("substellar") {
            let seek_hills = eva_site.as_deref() == Some("relief");
            let up = if dir_body_fixed.y.abs() > 0.99 {
                DVec3::X
            } else {
                DVec3::Y
            };
            let east = up.cross(dir_body_fixed).normalize();
            let north = dir_body_fixed.cross(east);
            let probe = 3_000.0 / body.radius_m; // ~3 km cross, in radians
            let h = |d: DVec3| {
                height_source
                    .sample_height_m(d.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
                    .unwrap_or(0.0) as f64
            };
            let n = 24i32;
            let mut best_dir = dir_body_fixed;
            let mut best_score = f64::NEG_INFINITY;
            let mut best_relief = 0.0f64;
            for iy in -n..=n {
                for ix in -n..=n {
                    // Offsets within ~50° of the sub-stellar point keep the site lit.
                    let ax = (ix as f64 / n as f64) * 0.9;
                    let ay = (iy as f64 / n as f64) * 0.9;
                    let cand = (dir_body_fixed + east * ax + north * ay).normalize();
                    let relief = (h((cand + east * probe).normalize())
                        - h((cand - east * probe).normalize()))
                    .abs()
                        + (h((cand + north * probe).normalize())
                            - h((cand - north * probe).normalize()))
                        .abs();
                    // Maximise relief for hills, minimise it for a usable plain.
                    let score = if seek_hills { relief } else { -relief };
                    if score > best_score {
                        best_score = score;
                        best_relief = relief;
                        best_dir = cand;
                    }
                }
            }
            dir_body_fixed = best_dir;
            let kind = if seek_hills { "hill" } else { "plain" };
            eprintln!("EVA {kind} site selected (relief proxy {best_relief:.0} m)");
        }
        let terrain_h = height_source
            .sample_height_m(dir_body_fixed.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
            .unwrap_or(0.0) as f64;
        let stand_clearance_m = 1.0;
        let position_body = dir_body_fixed * (body.radius_m + terrain_h + stand_clearance_m);
        let position_inertial = body_state.position + body_state.orientation * position_body;
        let velocity_inertial = body_state.velocity
            + body_state
                .angular_velocity
                .cross(body_state.orientation * position_body);
        let translation = TranslationalState {
            position: position_inertial,
            velocity: velocity_inertial,
        };
        let attitude = AttitudeState {
            orientation: level_attitude_for_body_dir(body_state.orientation, dir_body_fixed),
            angular_velocity: DVec3::ZERO,
        };
        sim.simulation
            .install_local_rigid_body_state(translation, attitude);
    }

    let craft = sim.simulation.craft_state();
    let frame = inertial_to_bubble_frame(&body_state, craft.translation, craft.attitude);

    let craft_entity = match vessel_kind {
        VesselKind::Ship => {
            if player_ship.iter().next().is_none() {
                return;
            }
            let collider_primitives = build_ship_collider_primitives(&parts);
            spawn_local_craft_body(
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
            )
        }
        VesselKind::Eva => {
            // KSP-on-foot is a kinematic capsule whose position is set
            // each frame by `step_eva_controller` from direct terrain
            // heightmap queries — no Avian contact resolution. Spawn a
            // placeholder cuboid so `spawn_local_craft_body` is happy
            // (it falls back to a 1 m cube if the list is empty, which
            // is fine but loud in inspectors); then immediately remove
            // the `Collider` entirely so writeback_solver_bodies has
            // nothing to integrate, which was producing the visible
            // sliding (delta_position from kinematic↔kinematic contacts
            // was being applied on top of our terrain-clamped writes).
            let entity = spawn_local_craft_body(
                &mut commands,
                LocalCraftSpawn {
                    craft_id: craft.id,
                    position_m: frame.position_m,
                    rotation: frame.rotation,
                    linear_velocity_m_s: frame.linear_velocity_m_s,
                    angular_velocity_rad_s: DVec3::ZERO,
                    mass_kg: craft.mass.wet_mass_kg.max(params.dry_mass_kg),
                    angular_inertia_kg_m2: params.moment_of_inertia,
                    collider_primitives: vec![LocalPrimitiveCollider {
                        offset_m: DVec3::ZERO,
                        rotation: DQuat::IDENTITY,
                        shape: LocalPrimitiveShape::Capsule {
                            radius: 0.32,
                            length: 1.8 - 0.64,
                        },
                    }],
                },
            );
            commands.entity(entity).remove::<Collider>().insert((
                RigidBody::Kinematic,
                CustomPositionIntegration,
                LockedAxes::ROTATION_LOCKED,
                PlayerControllerBody,
                Name::new("EVA player vessel"),
            ));
            entity
        }
    };

    let bubble_id = active.allocate_id();
    active.bubble = Some(LocalBubble {
        id: bubble_id,
        body_id,
        craft_entity,
        terrain_entity: None,
        center_dir_body: DVec3::Y,
        center_surface_body_m: DVec3::ZERO,
        basis: thalos_body_render::TerrainPatchBasis::from_normal(DVec3::Y),
        patch_half_extent_m: 0.0,
        stable_contact_s: 0.0,
        stable_landed: false,
        terrain_built_at_revision: 0,
    });
    info!(
        "spawned player vessel bubble={} body_id={} kind={:?}",
        bubble_id, body_id, vessel_kind,
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
    bubble.basis = thalos_body_render::TerrainPatchBasis::from_normal(DVec3::Y);
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
/// Grounded EVA is special-cased ahead of the match: it is pinned to
/// `LocalRigidBody` so warping on foot doesn't release it to `OnRails` and
/// Kepler-coast its surface state into the ground (which tripped the
/// collision warp-reset). See the inline comment for the failure mode.
///
/// Previously this function gated solely on warp level: at 1× warp, Avian
/// always owned; warping up handed translation back to Kepler. The result
/// was visible orbital drift any time the player paused/unpaused
/// mid-orbit, because Avian was integrating central-force gravity for a
/// ship that wasn't actually doing anything that needed an integrator.
fn manage_authority(
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    eva_mode: Res<EvaMode>,
    contact_graph: Res<ContactGraph>,
    config: Res<LocalBubbleConfig>,
    throttle: Res<ThrottleState>,
    mut sim: ResMut<SimulationState>,
    mut diagnostics: ResMut<AvianHandoffDiagnostics>,
    craft_q: Query<(&LinearVelocity, &AngularVelocity), With<LocalCraftBody>>,
) {
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };

    // Grounded EVA co-rotates with the surface, so it must never be
    // Kepler-coasted. Pin it to `LocalRigidBody`, whose `Simulation::step` arm
    // only advances sim-time (no coast, no surface-collision warp reset) and
    // leaves translation to `step_eva_controller` + `readback_local_craft`.
    //
    // Without this pin, time-warp breaks on foot: warping flips the Avian role
    // to `Paused`, the match below releases translation to `OnRails`, and the
    // next `step()` coasts the player's slow surface-velocity state — a
    // sub-surface trajectory — until `coast_segment` reports a collision and
    // `warp.reset_immediate()` snaps warp back to 1×. (The altitude gate skips
    // grounded EVA, so unlike a ship it isn't capped near terrain; nothing else
    // stops the coast.) `LocalRigidBody` rather than `BodyFixed` because EVA can
    // walk, which would leave an analytic `BodyFixed` pose stale.
    if sim.simulation.vessel_kind() == VesselKind::Eva && eva_mode.is_grounded() {
        if !matches!(
            sim.simulation.authority(),
            AuthorityMode::LocalRigidBody { .. }
        ) {
            sim.simulation
                .transition_authority(AuthorityMode::LocalRigidBody {
                    bubble: bubble.id,
                    root_entity: EntityRef(bubble.craft_entity.to_bits()),
                });
        }
        return;
    }

    // Landed ships should behave like landed EVA under time warp: fixed to the
    // rotating surface, not released to an inertial Kepler coast. If the player
    // requests warp while the ship is already in quiet terrain contact, collapse
    // to the same analytic `BodyFixed` authority before the generic
    // `LocalRigidBody -> OnRails` release below can run. In freefall (no
    // terrain contact, moving too fast, or throttle active) ships and EVA stay
    // ordinary ballistic craft.
    if sim.simulation.vessel_kind() == VesselKind::Ship
        && sim.simulation.warp.target_speed() > 1.0
        && matches!(
            sim.simulation.authority(),
            AuthorityMode::LocalRigidBody { .. }
        )
        && let Some(terrain_entity) = bubble.terrain_entity
        && craft_contacts_terrain(&contact_graph, bubble.craft_entity, terrain_entity)
        && let Ok((linear_velocity, angular_velocity)) = craft_q.get(bubble.craft_entity)
        && linear_velocity.length() < config.max_stable_speed_m_s
        && angular_velocity.length() < config.max_stable_angular_speed_rad_s
        && throttle.effective <= 1.0e-3
    {
        collapse_to_body_fixed(&mut sim, bubble);
        return;
    }

    match (sim.simulation.authority(), authority.owns_translation()) {
        (AuthorityMode::LocalRigidBody { .. }, false) => {
            sim.simulation
                .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
            diagnostics.last_handoff_kind = "ReleasedTranslation".to_string();
            diagnostics.last_handoff_sim_time_s = sim.simulation.sim_time();
        }
        (AuthorityMode::OnRails { .. }, true) => {
            sim.simulation
                .transition_authority(AuthorityMode::LocalRigidBody {
                    bubble: bubble.id,
                    root_entity: EntityRef(bubble.craft_entity.to_bits()),
                });
            // `last_handoff_kind` / time and the residual for this take are
            // finalized in `readback_local_craft` once Avian's converted
            // state is available; recording here would only capture the
            // pre-step canonical position.
        }
        _ => {}
    }
}

/// Attach a terrain collider patch when the ship enters the AGL handoff
/// band over a body whose surface is registered. The collider is
/// [`RigidBody::Kinematic`] centered on the patch's surface point. Its
/// local vertices are body-fixed offsets from that center, so
/// `Position + Rotation * local_vertex` lands at the correct
/// body-centered-inertial position while the narrow phase solves against
/// small local coordinates. [`sync_terrain_collider_pose`] re-poses it
/// each frame as the body rotates.
fn attach_terrain_patch_when_close(
    mut commands: Commands,
    height_sources: Res<HeightSourceRegistry>,
    config: Res<LocalBubbleConfig>,
    limits: Res<WarpLimits>,
    mut active: ResMut<ActiveLocalBubble>,
    sim: Res<SimulationState>,
) {
    let Some(bubble) = active.bubble.as_mut() else {
        return;
    };
    if bubble.terrain_entity.is_some() {
        return;
    }
    if !terrain_colliders_allowed_by_warp(&sim, &limits) {
        return;
    }
    // EVA never collides: its capsule's `Collider` is removed at spawn
    // (`spawn_player_avian_body`) and `step_eva_controller` plants it
    // kinematically from direct height queries. A terrain collider patch would
    // therefore collide with nothing — yet once attached, `maintain_terrain_patch`
    // rebuilds its trimesh every frame as the GPU-atlas height-source
    // `revision()` churns with tile streaming (the player co-rotates with the
    // planet, so the streamer constantly loads/evicts tiles). That rebuild was
    // ~11% of surface frame time and the cause of the EVA "unplayable stutter".
    // Skip the patch entirely for EVA; ships still get it for real contact.
    if sim.simulation.vessel_kind() == VesselKind::Eva {
        return;
    }
    let Some(body_id) = thalos_body_id(&sim) else {
        return;
    };
    if bubble.body_id != body_id {
        return;
    }
    let Some(height_source) = height_sources.get(body_id) else {
        return;
    };
    let body = &sim.system.bodies[body_id];
    let body_state = body_state_for(&sim, body_id);
    let craft = sim.simulation.craft_state();
    let Some((agl_m, center_dir, _)) = agl_above_rendered_surface(
        body,
        &body_state,
        height_source.as_ref(),
        craft.translation.position,
    ) else {
        return;
    };
    if agl_m > config.handoff_agl_m {
        return;
    }
    let built_revision = height_source.revision();
    let patch = spawn_terrain_collider_patch(
        &mut commands,
        body_id,
        height_source.as_ref(),
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
    bubble.patch_half_extent_m = patch.mesh.half_extent_m;
    bubble.terrain_built_at_revision = built_revision;
    info!(
        "attached terrain collider patch over {} at AGL {:.0} m (height-source revision {})",
        body.name, agl_m, built_revision,
    );
}

/// Despawn the terrain collider patch when the ship climbs back above the
/// handoff band (with hysteresis so we don't churn on the boundary).
fn detach_terrain_patch_when_far(
    mut commands: Commands,
    height_sources: Res<HeightSourceRegistry>,
    config: Res<LocalBubbleConfig>,
    limits: Res<WarpLimits>,
    contact_graph: Res<ContactGraph>,
    mut active: ResMut<ActiveLocalBubble>,
    sim: Res<SimulationState>,
) {
    let Some(bubble) = active.bubble.as_mut() else {
        return;
    };
    let Some(terrain_entity) = bubble.terrain_entity else {
        return;
    };
    if matches!(sim.simulation.authority(), AuthorityMode::BodyFixed { .. }) {
        clear_terrain_patch(&mut commands, bubble);
        info!("detached terrain collider patch from BodyFixed craft");
        return;
    }
    if !terrain_colliders_allowed_by_warp(&sim, &limits)
        && !craft_contacts_terrain(&contact_graph, bubble.craft_entity, terrain_entity)
    {
        clear_terrain_patch(&mut commands, bubble);
        info!("detached terrain collider patch outside the 1x warp-lock zone");
        return;
    }
    let Some(height_source) = height_sources.get(bubble.body_id) else {
        return;
    };
    let body = &sim.system.bodies[bubble.body_id];
    let body_state = body_state_for(&sim, bubble.body_id);
    let craft = sim.simulation.craft_state();
    let Some((agl_m, _, _)) = agl_above_rendered_surface(
        body,
        &body_state,
        height_source.as_ref(),
        craft.translation.position,
    ) else {
        return;
    };
    // Hysteresis: detach at 1.5× the attach threshold.
    if agl_m <= config.handoff_agl_m * 1.5 {
        return;
    }
    clear_terrain_patch(&mut commands, bubble);
    info!(
        "detached terrain collider patch from {} at AGL {:.0} m",
        body.name, agl_m
    );
}

fn clear_terrain_patch(commands: &mut Commands, bubble: &mut LocalBubble) {
    if let Some(terrain_entity) = bubble.terrain_entity.take() {
        commands.entity(terrain_entity).despawn();
    }
    bubble.center_dir_body = DVec3::Y;
    bubble.center_surface_body_m = DVec3::ZERO;
    bubble.basis = thalos_body_render::TerrainPatchBasis::from_normal(DVec3::Y);
    bubble.patch_half_extent_m = 0.0;
    bubble.terrain_built_at_revision = 0;
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
    eva_mode: Res<EvaMode>,
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
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    debug: Option<Res<DebugMode>>,
    height_sources: Res<HeightSourceRegistry>,
    config: Res<LocalBubbleConfig>,
    mut active: ResMut<ActiveLocalBubble>,
    mut launch_mount: ResMut<DebugLaunchMount>,
    mut eva_mode: ResMut<EvaMode>,
    mut sim: ResMut<SimulationState>,
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
    if !keys.just_pressed(DEBUG_DROP_KEY) || !debug.as_deref().map(|d| d.enabled).unwrap_or(false) {
        return;
    }
    let Some(body_id) = thalos_body_id(&sim) else {
        return;
    };
    let Some(height_source) = height_sources.get(body_id) else {
        warn!("debug surface drop requested before Thalos height source is available");
        return;
    };

    let is_eva = sim.simulation.vessel_kind() == VesselKind::Eva;
    // For ships the bubble is teardown-and-respawn territory: a teleport
    // while it exists would skew the contact graph and Avian's internal
    // state. EVA spawns its bubble at startup and never tears it down,
    // so we teleport in place and let `maintain_terrain_patch` (or our
    // explicit terrain despawn below) rebuild the surface mesh around
    // the new position.
    if !is_eva && let Some(bubble) = active.bubble.take() {
        warn!(
            "debug surface drop requested while local bubble {} is active; keeping current bubble",
            bubble.id
        );
        active.bubble = Some(bubble);
        return;
    }

    let body = sim.system.bodies[body_id].clone();
    let body_state = body_state_for(&sim, body_id);
    // Body-fixed direction toward the star at the current sim time.
    // Pyros sits at the heliocentric origin, so the sun direction in
    // body-centered inertial coordinates is `-body_state.position`;
    // rotating by `orientation.inverse()` puts it in body-fixed coords
    // so the spawn rotates with the planet (always day-side).
    let sun_dir_inertial = (-body_state.position).normalize_or_zero();
    let dir = if sun_dir_inertial == DVec3::ZERO {
        // Star is at the body's centre — degenerate, fall back to a
        // fixed body-fixed heading rather than dividing by zero.
        DVec3::new(0.271, 0.893, -0.361).normalize()
    } else {
        (body_state.orientation.inverse() * sun_dir_inertial).normalize()
    };
    let height = height_source
        .sample_height_m(dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
        .unwrap_or(0.0) as f64;
    let position_body = dir * (body.radius_m + height + config.debug_drop_height_m);
    let surface_velocity = body_state.velocity
        + body_state
            .angular_velocity
            .cross(body_state.orientation * position_body);
    // Ships get a small downward kick so they land instead of hover;
    // EVA arrives at rest (the controller's gravity will pull it down).
    let velocity = if is_eva {
        surface_velocity
    } else {
        surface_velocity + body_state.orientation * (-dir * config.debug_drop_speed_m_s)
    };
    let translation = TranslationalState {
        position: body_state.position + body_state.orientation * position_body,
        velocity,
    };
    let attitude = AttitudeState {
        orientation: level_attitude_for_body_dir(body_state.orientation, dir),
        angular_velocity: DVec3::ZERO,
    };

    if is_eva {
        // Grounded EVA owns its Avian capsule directly (the canonical→Avian
        // snap is short-circuited), so the shared helper writes canonical,
        // marks the player grounded, and plants the capsule in one place.
        if let Some(bubble) = active.bubble.as_mut()
            && let Ok((mut position, mut rotation, mut linear_velocity, mut angular_velocity)) =
                craft_q.get_mut(bubble.craft_entity)
        {
            place_eva_on_surface(
                &mut commands,
                &mut sim,
                &mut eva_mode,
                bubble,
                (
                    &mut position,
                    &mut rotation,
                    &mut linear_velocity,
                    &mut angular_velocity,
                ),
                body_id,
                translation,
                attitude,
            );
        }
    } else {
        sim.simulation
            .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
        sim.simulation
            .install_local_rigid_body_state(translation, attitude);
        sim.simulation.warp.reset();
    }
    // A fresh drop hands back a flyable craft — clear any structural failure.
    sim.simulation.repair();
    launch_mount.active = None;

    info!(
        "debug surface drop placed {:?} {:.0} m above rendered {} terrain (day-side)",
        sim.simulation.vessel_kind(),
        config.debug_drop_height_m,
        body.name,
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
    clock: Res<SimClock>,
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
    let Ok((position, rotation, angular_velocity, mut linear_accel, mut angular_accel, _)) =
        craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };
    let params = *sim.simulation.ship_params();
    // A destroyed craft is inert debris: gravity still acts (so it falls and
    // settles), but thrust and reaction-wheel torque are cut. See
    // `docs/landing.md`.
    let destroyed = sim.simulation.is_destroyed();

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
            if !destroyed
                && throttle_eff > 0.0
                && params.thrust_n > 0.0
                && mass > params.dry_mass_kg
            {
                let nose_world = rotation.0 * DVec3::Y;
                accel += nose_world * (params.thrust_n / mass) * throttle_eff;
                sim.simulation
                    .apply_external_mass_flow(throttle_eff, clock.delta_secs_f64());
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
    // rotate the ship while coasting. A destroyed craft gets zero torque —
    // no player input, no SAS damping — so it tumbles freely as debris.
    angular_accel.0 = if destroyed {
        DVec3::ZERO
    } else {
        compute_angular_acceleration(
            sim.simulation.control(),
            &params,
            rotation.0,
            angular_velocity.0,
            clock.delta_secs_f64(),
        )
    };
}

/// Convert player attitude command + SAS damping into a world-space angular
/// acceleration for the Avian rigid body. Matches `Simulation::integrate_attitude`
/// (now removed) so the rotational feel is identical whether the ship is in
/// deep space or on a surface — Avian is the integrator in both cases.
fn compute_angular_acceleration(
    control: &thalos_physics_canonical::types::ControlInput,
    params: &thalos_physics_canonical::types::ShipParameters,
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
    eva_mode: Res<EvaMode>,
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
    let (translation, attitude) = bubble_frame_to_inertial(
        &body_state,
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
            debug_assert!(
                position_residual_m < HANDOFF_RESIDUAL_TOLERANCE_M,
                "Avian↔canonical take-translation handoff position discontinuity: \
                 {position_residual_m:.3} m (tolerance {HANDOFF_RESIDUAL_TOLERANCE_M} m) — \
                 snap/readback body-frame skew or SOI race"
            );
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

/// Short ring buffer of recent surface-relative approach speeds. The
/// impact detector reads its **peak** at contact onset rather than the
/// instantaneous speed because `SweptCcd` books the velocity arrest a frame
/// or two after the geometric sweep, and speculative collision can shave
/// the final approach frame — by contact-start the instantaneous speed is
/// already damped, but the peak across the last ~8 frames is still the true
/// approach speed (gravity changes it by only ~1 m/s over that window).
#[derive(Default)]
struct ImpactSpeedWindow {
    samples: [f64; Self::LEN],
    idx: usize,
}

impl ImpactSpeedWindow {
    const LEN: usize = 8;

    fn push(&mut self, speed_m_s: f64) {
        self.samples[self.idx] = speed_m_s;
        self.idx = (self.idx + 1) % Self::LEN;
    }

    fn peak(&self) -> f64 {
        self.samples.iter().copied().fold(0.0, f64::max)
    }

    fn clear(&mut self) {
        self.samples = [0.0; Self::LEN];
        self.idx = 0;
    }
}

/// Detect a destroying terrain impact and mark the craft destroyed.
///
/// Only meaningful while Avian owns translation (`AvianRole::Full`), which
/// is exactly when a terrain patch is attached and contacts are being
/// solved. Each frame we record the craft's surface-relative approach speed
/// (`v − ω × r`, the speed the co-rotating terrain collider actually sees);
/// on the **rising edge** of contact with the terrain patch we compare the
/// windowed peak approach speed against [`ShipParameters::impact_tolerance_m_s`]
/// and destroy the craft if it was coming in too hard.
///
/// EVA is exempt (the capsule has no collider, so no contacts). A craft that
/// is already destroyed short-circuits so debris settling on the ground
/// doesn't re-trigger. See `docs/landing.md`.
fn detect_terrain_impact(
    contact_graph: Res<ContactGraph>,
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    mut sim: ResMut<SimulationState>,
    craft_q: Query<(&LinearVelocity, &Position), With<LocalCraftBody>>,
    mut speed_window: Local<ImpactSpeedWindow>,
    mut was_touching: Local<bool>,
) {
    // Only Full integrates contacts and owns the craft's velocity. Outside
    // it (coast, warp/Paused, BodyFixed) there is nothing to detect, and the
    // snapped canonical velocity would read as a false high-speed approach.
    let owns_translation = authority.owns_translation();
    let Some(bubble) = active.bubble.as_ref() else {
        *was_touching = false;
        speed_window.clear();
        return;
    };
    if !owns_translation
        || sim.simulation.vessel_kind() == VesselKind::Eva
        || sim.simulation.is_destroyed()
    {
        *was_touching = false;
        speed_window.clear();
        return;
    }
    let Some(terrain_entity) = bubble.terrain_entity else {
        *was_touching = false;
        speed_window.clear();
        return;
    };
    let Ok((linear_velocity, position)) = craft_q.get(bubble.craft_entity) else {
        return;
    };

    // Surface-relative approach speed in body-centered inertial. The terrain
    // collider is centered on its local patch, but its velocity field is still
    // the body's rotation field, so the surface point under the craft moves at
    // ω × r. Subtracting it gives the relative speed the contact resolves
    // against (a craft resting on the spinning surface reads ~0, not the
    // surface's inertial speed).
    let body_state = body_state_for(&sim, bubble.body_id);
    let surface_velocity = body_state.angular_velocity.cross(position.0);
    let approach_speed = (linear_velocity.0 - surface_velocity).length();
    speed_window.push(approach_speed);

    let touching = craft_contacts_terrain(&contact_graph, bubble.craft_entity, terrain_entity);
    let contact_started = touching && !*was_touching;
    *was_touching = touching;
    if !contact_started {
        return;
    }

    let impact_speed = speed_window.peak();
    let tolerance = sim.simulation.ship_params().impact_tolerance_m_s;
    if impact_speed > tolerance {
        warn!(
            "VESSEL DESTROYED: terrain impact at {:.1} m/s (tolerance {:.1} m/s)",
            impact_speed, tolerance
        );
        sim.simulation.mark_destroyed(impact_speed);
    }
}

fn maintain_terrain_patch(
    mut commands: Commands,
    height_sources: Res<HeightSourceRegistry>,
    config: Res<LocalBubbleConfig>,
    limits: Res<WarpLimits>,
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
    if !terrain_colliders_allowed_by_warp(&sim, &limits) {
        return;
    }
    let Some(height_source) = height_sources.get(current.body_id) else {
        return;
    };
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
    let current_revision = height_source.revision();
    // Re-center before the craft drifts off the patch edge. The tile-based
    // collider window (docs/landing.md §3.6) is only tens of metres, so cap the
    // global drift distance by a fraction of the patch's own half-extent; the
    // coarse tangent-grid fallback (km-scale half-extent) keeps the global
    // distance. `patch_half_extent_m` is 0 only with no patch attached, which
    // the early return above already excludes.
    let rebuild_distance_m = if current.patch_half_extent_m > 0.0 {
        config
            .patch_rebuild_distance_m
            .min(0.45 * current.patch_half_extent_m)
    } else {
        config.patch_rebuild_distance_m
    };
    let lateral_stale = lateral > rebuild_distance_m;
    let source_stale = current_revision != current.terrain_built_at_revision;
    if !lateral_stale && !source_stale {
        return;
    }
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
        height_source.as_ref(),
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
        patch_half_extent_m: patch.mesh.half_extent_m,
        stable_contact_s: current.stable_contact_s,
        stable_landed: current.stable_landed,
        terrain_built_at_revision: current_revision,
        ..current
    });
}

/// Pose the kinematic terrain collider to match the dominant body's current
/// orientation and angular velocity. The collider's body origin sits at the
/// patch center so its local mesh stays near zero; `LinearVelocity =
/// ω × origin` plus `AngularVelocity = ω` gives every contact point the
/// correct rotating-surface velocity.
fn sync_terrain_collider_pose(
    active: Res<ActiveLocalBubble>,
    sim: Res<SimulationState>,
    mut terrain_q: Query<
        (
            &mut Position,
            &mut Rotation,
            &mut LinearVelocity,
            &mut AngularVelocity,
        ),
        (With<TerrainColliderPatch>, Without<LocalCraftBody>),
    >,
) {
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Some(terrain_entity) = bubble.terrain_entity else {
        return;
    };
    let Ok((mut position, mut rotation, mut linear_velocity, mut angular_velocity)) =
        terrain_q.get_mut(terrain_entity)
    else {
        return;
    };
    let body_state = body_state_for(&sim, bubble.body_id);
    let (patch_position, patch_velocity) = terrain_patch_pose(
        bubble.center_surface_body_m,
        body_state.orientation,
        body_state.angular_velocity,
    );
    position.0 = patch_position;
    rotation.0 = body_state.orientation;
    linear_velocity.0 = patch_velocity;
    angular_velocity.0 = body_state.angular_velocity;
}

/// Track stable-contact landing and collapse to `BodyFixed` when the ship
/// settles. Warp gating now lives in [`manage_authority`], which derives
/// authority from [`AvianOwnership`] (warp ≠ 1× falls back to `OnRails`
/// regardless of contact). Warping no longer requires tearing down the
/// bubble.
fn collapse_or_constrain_warp(
    mut commands: Commands,
    clock: Res<SimClock>,
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
        clock.delta_secs_f64(),
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

/// Plant a grounded EVA player at a surface pose, in place.
///
/// EVA keeps its persistent bubble across teleports (KSP-on-foot never tears
/// the capsule down), so a surface teleport is a rewrite rather than a
/// respawn: set canonical, mark the EVA grounded, move the bubble onto the
/// target body, drop the old terrain patch, and plant the Avian capsule. The
/// grounded canonical→Avian snap is short-circuited, so this is the only
/// thing that moves the capsule; [`crate::player_controller::step_eva_controller`]
/// takes over next frame and glues it to the rendered surface.
///
/// Shared by the F9 sub-stellar drop and the map-cursor surface teleport so
/// both place EVA the same way.
#[allow(clippy::too_many_arguments)]
pub(crate) fn place_eva_on_surface(
    commands: &mut Commands,
    sim: &mut SimulationState,
    eva_mode: &mut EvaMode,
    bubble: &mut LocalBubble,
    avian: (
        &mut Position,
        &mut Rotation,
        &mut LinearVelocity,
        &mut AngularVelocity,
    ),
    body_id: BodyId,
    translation: TranslationalState,
    attitude: AttitudeState,
) {
    let (position, rotation, linear_velocity, angular_velocity) = avian;
    let body_state = body_state_for(sim, body_id);

    sim.simulation
        .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
    sim.simulation
        .install_local_rigid_body_state(translation, attitude);
    sim.simulation.warp.reset();
    *eva_mode = EvaMode::Grounded;

    // Move the bubble onto the target body and drop the old terrain patch so
    // `attach_terrain_patch_when_close` rebuilds it around the new spot. When
    // the body is unchanged (the common Thalos→Thalos case) these are no-ops.
    if let Some(terrain_entity) = bubble.terrain_entity.take() {
        commands.entity(terrain_entity).despawn();
    }
    bubble.body_id = body_id;
    bubble.center_dir_body = DVec3::Y;
    bubble.center_surface_body_m = DVec3::ZERO;
    bubble.basis = thalos_body_render::TerrainPatchBasis::from_normal(DVec3::Y);
    bubble.stable_contact_s = 0.0;
    bubble.stable_landed = false;

    let frame = inertial_to_bubble_frame(&body_state, translation, attitude);
    position.0 = frame.position_m;
    rotation.0 = frame.rotation;
    linear_velocity.0 = frame.linear_velocity_m_s;
    angular_velocity.0 = DVec3::ZERO;
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
    let basis = thalos_body_render::TerrainPatchBasis::from_normal(up_body);
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
        Entity,
        &'static AttachNodes,
        Option<&'static Attachment>,
        Option<&'static CommandPod>,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Engine>,
    ),
    With<Part>,
>;

fn build_ship_collider_primitives(parts: &PartColliderQuery) -> Vec<LocalPrimitiveCollider> {
    let part_positions = compute_part_collider_positions(parts);
    let mut primitives = Vec::new();
    for (entity, nodes, _, pod, dec, adapter, tank, engine) in parts.iter() {
        let Some(part_position) = part_positions.get(&entity).copied() else {
            continue;
        };
        let Some((shape, local_offset)) =
            part_collider_shape(nodes, pod, dec, adapter, tank, engine)
        else {
            continue;
        };
        primitives.push(LocalPrimitiveCollider {
            offset_m: part_position + local_offset,
            rotation: DQuat::IDENTITY,
            shape,
        });
    }
    if primitives.is_empty() {
        primitives.push(fallback_collider());
    }
    primitives
}

fn compute_part_collider_positions(parts: &PartColliderQuery) -> HashMap<Entity, DVec3> {
    let mut nodes_by_entity: HashMap<Entity, &AttachNodes> = HashMap::new();
    let mut children_by_parent: HashMap<Entity, Vec<(Entity, Attachment)>> = HashMap::new();
    let mut roots = Vec::new();

    for (entity, nodes, attachment, ..) in parts.iter() {
        nodes_by_entity.insert(entity, nodes);
        if let Some(attachment) = attachment {
            children_by_parent
                .entry(attachment.parent)
                .or_default()
                .push((entity, attachment.clone()));
        } else {
            roots.push(entity);
        }
    }

    let mut positions = HashMap::new();
    let mut queue = VecDeque::new();
    for root in roots {
        positions.insert(root, DVec3::ZERO);
        queue.push_back(root);
    }

    while let Some(parent) = queue.pop_front() {
        let Some(parent_position) = positions.get(&parent).copied() else {
            continue;
        };
        let Some(parent_nodes) = nodes_by_entity.get(&parent).copied() else {
            continue;
        };
        let Some(children) = children_by_parent.get(&parent) else {
            continue;
        };
        for (child, attachment) in children {
            let Some(parent_node) = parent_nodes.get(&attachment.parent_node) else {
                continue;
            };
            let child_offset = nodes_by_entity
                .get(child)
                .and_then(|nodes| nodes.get(&attachment.my_node))
                .map(|node| node.offset)
                .unwrap_or(Vec3::ZERO);
            let child_position = parent_position + (parent_node.offset - child_offset).as_dvec3();
            positions.insert(*child, child_position);
            queue.push_back(*child);
        }
    }

    positions
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

fn part_collider_shape(
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
            LocalPrimitiveShape::Cylinder {
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
            LocalPrimitiveShape::Cylinder {
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
    use thalos_physics_canonical::canonical::Epoch;
    use thalos_physics_canonical::types::BodyState;
    use thalos_body_render::{TerrainPatchBasis, TerrainPatchMesh};

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
        let bubble = LocalBubble {
            id: 1,
            body_id: 0,
            craft_entity: Entity::PLACEHOLDER,
            terrain_entity: Some(Entity::PLACEHOLDER),
            center_dir_body: DVec3::Y,
            center_surface_body_m: patch.center_surface_body_m,
            basis,
            patch_half_extent_m: 0.0,
            stable_contact_s: 0.0,
            stable_landed: false,
            terrain_built_at_revision: 0,
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

    fn on_rails() -> AuthorityMode {
        AuthorityMode::OnRails { trajectory: 0 }
    }

    fn body_fixed() -> AuthorityMode {
        AuthorityMode::BodyFixed {
            body: 0,
            pose: thalos_physics_canonical::canonical::BodyFixedPose {
                position_body_m: DVec3::Y * 1000.0,
                orientation_body: DQuat::IDENTITY,
            },
        }
    }

    #[test]
    fn terrain_colliders_wait_until_warp_gate_locks_to_one_x() {
        let levels = [0.0, 1.0, 10.0, 100.0];

        assert!(!terrain_colliders_allowed_by_warp_inputs(1.0, &levels, 2));
        assert!(terrain_colliders_allowed_by_warp_inputs(1.0, &levels, 1));
    }

    #[test]
    fn terrain_colliders_do_not_build_while_paused_or_high_warp() {
        let levels = [0.0, 1.0, 10.0, 100.0];

        assert!(!terrain_colliders_allowed_by_warp_inputs(0.0, &levels, 1));
        assert!(!terrain_colliders_allowed_by_warp_inputs(10.0, &levels, 1));
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

    #[test]
    fn pod_collider_uses_full_radius_cylinder() {
        let nodes = AttachNodes::default();
        let pod = CommandPod {
            model: "test".to_string(),
            diameter: 2.0,
            dry_mass: 0.0,
            reaction_wheel_torque: 0.0,
        };

        let (shape, offset) =
            part_collider_shape(&nodes, Some(&pod), None, None, None, None).unwrap();

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
            diameter: 2.0,
            thrust: 0.0,
            isp: 0.0,
            dry_mass: 0.0,
            reactants: Vec::new(),
            power_draw_kw: 0.0,
        };

        let (shape, offset) =
            part_collider_shape(&nodes, None, None, None, None, Some(&engine)).unwrap();

        let LocalPrimitiveShape::Cylinder { radius, height } = shape else {
            panic!("engine collider should be a cylinder");
        };
        assert!((radius - 1.0).abs() < 1e-12);
        assert!((height - 1.8).abs() < 1e-6);
        assert!((offset - DVec3::Y * -(height * 0.5)).length() < 1e-12);
    }
}
