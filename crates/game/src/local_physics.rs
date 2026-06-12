//! Game-side orchestration for the M5 aggregate local-physics bubble.

use std::collections::{HashMap, VecDeque};

use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use thalos_body_render::HeightSource;
use thalos_physics_canonical::body_centered::{
    BodyCenteredState, body_centered_to_inertial, inertial_to_body_centered,
};
use thalos_physics_canonical::body_fixed::body_fixed_pose_from_inertial;
use thalos_physics_canonical::canonical::{
    AuthorityMode, BodyFixedPose, EntityRef, TranslationalState,
};
use thalos_physics_canonical::surface_local::{
    SurfaceAnchor, SurfaceLocalFrame, SurfaceLocalState, inertial_to_surface_local, reanchor,
    surface_local_acceleration, surface_local_to_inertial,
};
use thalos_physics_canonical::types::{AttitudeState, BodyState, VesselKind};
use thalos_physics_local::avian::{
    AngularVelocity, CenterOfMass, Collider, ConstantAngularAcceleration, ConstantLinearAcceleration,
    ContactGraph, CustomPositionIntegration, LinearVelocity, LockedAxes, NoAutoCenterOfMass,
    Physics, PhysicsTime, Position, RigidBody, Rotation, SpatialQuery, SpatialQueryFilter,
};
use thalos_physics_local::{
    ActiveLocalBubble, HeightSourceRegistry, LocalBubble, LocalBubbleConfig, LocalCraftBody,
    LocalCraftColliderPrimitives, LocalCraftSpawn, LocalPhysicsPlugin, LocalPrimitiveCollider,
    LocalPrimitiveShape,
    TerrainColliderPatch, craft_contacts_terrain, spawn_local_craft_body,
    spawn_terrain_collider_patch, stable_contact_reached,
};
use thalos_shipyard::{
    Adapter, AirIntake, AttachNodes, Attachment, CommandPod, Decoupler, Engine, EngineGeometry,
    FuelTank, Gear, Part, SurfaceMount, SurfaceMountKind, Wing, gear_leg_frames, wing_panel_frame,
};
use thalos_input::game::GameInputIntent;
use thalos_world::BodyId;

use crate::SimStage;
use crate::bridge::WarpLimits;
use crate::debug::DebugMode;
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
/// Commanded-throttle threshold above which a landed (`BodyFixed`) ship is
/// released back to live physics by [`release_landed_ship_on_throttle`].
const LANDED_THROTTLE_RELEASE: f64 = 0.001;

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
            .init_resource::<GearTuning>()
            .init_resource::<ParkingBrake>()
            .init_resource::<WeightOnWheels>()
            .init_resource::<SurfaceFriction>()
            .init_resource::<TerrainFloorBackstop>()
            .register_type::<TerrainFloorBackstop>()
            .register_type::<AvianRole>()
            .register_type::<AvianAuthority>()
            .register_type::<AvianHandoffDiagnostics>()
            .register_type::<GearTuning>()
            .register_type::<ParkingBrake>()
            .register_type::<WeightOnWheels>()
            .register_type::<SurfaceFriction>()
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
                    release_landed_ship_on_throttle,
                    spawn_player_avian_body,
                    rebase_bubble_to_dominant_body,
                    attach_terrain_patch_when_close,
                    detach_terrain_patch_when_far,
                    compute_avian_authority,
                    manage_authority,
                    sync_avian_time,
                    snap_avian_from_canonical,
                    apply_local_forces,
                    toggle_parking_brake,
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
                    collapse_or_constrain_warp,
                )
                    .chain()
                    .in_set(SimStage::Physics)
                    .after(apply_surface_friction),
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
        craft_in_atmosphere(sim),
    )
}

/// True when the craft sits below the dominant body's Kármán line — i.e. inside
/// the atmosphere shell, where aerodynamic forces act.
///
/// Uses the cheap mean-radius altitude (`|r| − radius`); the Kármán line sits
/// far above terrain relief, so a per-pixel terrain-height query is unnecessary
/// here. Returns false for airless bodies (no `terrestrial_atmosphere`, or a
/// zero Kármán line).
fn craft_in_atmosphere(sim: &SimulationState) -> bool {
    let body_id = sim.simulation.dominant_body();
    let body = &sim.system.bodies[body_id];
    let Some(atmosphere) = body.terrestrial_atmosphere.as_ref() else {
        return false;
    };
    if atmosphere.karman_line_m <= 0.0 {
        return false;
    }
    let body_state = body_state_for(sim, body_id);
    let craft = sim.simulation.craft_state();
    let altitude_m = (craft.translation.position - body_state.position).length() - body.radius_m;
    altitude_m < atmosphere.karman_line_m as f64
}

/// Pure predicate: classify Avian's role from raw inputs.
///
/// - **Warp ≠ 1×** → `Paused`. Time-stepped integration of central-force
///   gravity blows up at large `dt`, and we don't run rotation under warp
///   either (the existing convention zeroes ω at warp entry to avoid a
///   tap-rotate-then-warp leaving the ship spinning out).
/// - **`BodyFixed` authority** → `Paused`. Landed pose is analytic.
/// - **Throttle active, terrain collider attached, OR inside the atmosphere
///   shell** → `Full`. We need Avian to integrate the non-gravity force
///   (thrust, contact, aerodynamic drag/lift) plus gravity. Terrain-collider
///   presence flags "contact is physically possible here"; `in_atmosphere`
///   flags "aero forces act here" — and crucially makes Avian own translation
///   across the *whole* atmospheric column (Kármán line down), not only inside
///   the ~20 km terrain-collider band, so reentry drag is applied the entire
///   way down instead of the ship Kepler-coasting drag-free through the upper
///   atmosphere.
/// - **Otherwise (coasting in vacuum at 1× warp)** → `AttitudeOnly`. Avian
///   keeps integrating rotation and contact; Kepler owns translation, so
///   AP/PE don't drift across pause/unpause cycles.
fn avian_role_from_inputs(
    warp_speed: f64,
    authority: AuthorityMode,
    throttle_effective: f64,
    terrain_attached: bool,
    in_atmosphere: bool,
) -> AvianRole {
    let near_one_x = (warp_speed - 1.0).abs() <= f64::EPSILON;
    if !near_one_x {
        return AvianRole::Paused;
    }
    if matches!(authority, AuthorityMode::BodyFixed { .. }) {
        return AvianRole::Paused;
    }
    let thrust_active = throttle_effective > 0.0;
    if thrust_active || terrain_attached || in_atmosphere {
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
/// ready to host it. Ships live in the **surface-local frame** (a body-fixed
/// tangent frame anchored under the craft, Y-up, small coordinates — see
/// `docs/surface_local.md`); gravity plus the rotating-frame terms come from
/// `surface_local_acceleration`, and ground colliders are static in the
/// frame. The EVA capsule still lives in body-centered inertial coordinates
/// until its SLF fold-in.
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
    gear_q: Query<(&Gear, &SurfaceMount), (With<Part>, Without<thalos_shipyard::editor::EditorPart>)>,
    host_nodes: Query<&AttachNodes>,
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
    // Anchor the surface-local frame under the craft's surface projection.
    // The height source may not be registered yet (bakes still loading) —
    // a reference-radius anchor is exact regardless, and the re-anchor
    // system refreshes the elevation as the craft moves.
    let height_source = height_sources.get(body_id);
    let slf = SurfaceLocalFrame::new(
        &body_state,
        surface_anchor_under(
            &body_state,
            height_source.as_deref(),
            craft.translation.position,
        ),
    );
    let frame = inertial_to_craft_frame(
        vessel_kind,
        &body_state,
        &slf,
        craft.translation,
        craft.attitude,
    );

    let craft_entity = match vessel_kind {
        VesselKind::Ship => {
            if player_ship.iter().next().is_none() {
                return;
            }
            let collider_primitives = build_ship_collider_primitives(&parts);
            let part_positions = compute_part_collider_positions(&parts);
            let wheels = build_wheel_set(&gear_q, &host_nodes, &part_positions);
            let entity = spawn_local_craft_body(
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
            // Pin the rigid body's rotation pivot to the craft's *real* CoM for
            // every ship, not just gear ships. Two systems depend on it:
            //   - aero: the native aero model (`thalos_physics_canonical::aero`)
            //     takes each surface's moment about the CoM. With Avian's auto
            //     CoM (the collider centroid) the static margin — hence pitch/yaw
            //     stability — is accidental, which is the wingless-craft tumble.
            //   - gear: upward wheel forces aft of the nose origin need the pivot
            //     at the CoM they straddle, or they tip the craft over.
            // `NoAutoCenterOfMass` stops the compound collider from overwriting
            // it; Position still tracks the root origin, so snap/readback are
            // unaffected.
            let com = params.center_of_mass.as_vec3();
            commands
                .entity(entity)
                .insert((CenterOfMass(com), NoAutoCenterOfMass));
            if !wheels.is_empty() {
                info!(
                    "landing gear: {} wheel(s) on player ship, CoM = ({:.2}, {:.2}, {:.2}) m",
                    wheels.len(),
                    com.x,
                    com.y,
                    com.z,
                );
                // Gear is the sole ground interface for a wheeled craft: filter
                // the hull compound collider out of solver contact with the
                // ground so it can't fight the raycast suspension (which flung
                // the craft on its gear). The gear raycast is a SpatialQuery,
                // unaffected by these layers; crash detection switches to the
                // weight-on-wheels signal. See `docs/surface_local.md`.
                commands.entity(entity).insert((
                    WheelSet { wheels },
                    thalos_physics_local::wheeled_craft_collision_layers(),
                ));
            }
            entity
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
        frame: slf,
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
    height_sources: Res<HeightSourceRegistry>,
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
    let kind = sim.simulation.vessel_kind();
    let old_body_state = body_state_for(&sim, bubble.body_id);
    let (translation, attitude) = craft_frame_to_inertial(
        kind,
        &old_body_state,
        &bubble.frame,
        position.0,
        rotation.0,
        linear_velocity.0,
        angular_velocity.0,
    );
    let new_body_state = body_state_for(&sim, new_body_id);
    // Fresh surface-local frame anchored under the craft on the *new* body.
    let height_source = height_sources.get(new_body_id);
    let new_frame = SurfaceLocalFrame::new(
        &new_body_state,
        surface_anchor_under(
            &new_body_state,
            height_source.as_deref(),
            translation.position,
        ),
    );
    let frame = inertial_to_craft_frame(kind, &new_body_state, &new_frame, translation, attitude);
    bubble.frame = new_frame;
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
    runway: Option<Res<crate::runway::RunwaySite>>,
    mut active: ResMut<ActiveLocalBubble>,
    sim: Res<SimulationState>,
) {
    let Some(bubble) = active.bubble.as_mut() else {
        return;
    };
    if bubble.terrain_entity.is_some() {
        return;
    }
    // In a runway scenario the purpose-built flat `RunwayCollider` already backs
    // the surface under the craft. Attaching the generic terrain patch on top
    // would put a *second* flat kinematic collider at the same elevation, so the
    // craft's compound collider resolves contacts against both at once — a
    // double penetration-recovery push that launches it off its gear. Skip the
    // patch on that body and let the runway collider be the sole ground.
    if runway.as_deref().is_some_and(|r| r.body_id == bubble.body_id) {
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
    let slf = bubble.frame;
    let patch = spawn_terrain_collider_patch(
        &mut commands,
        body_id,
        height_source.as_ref(),
        body.radius_m,
        center_dir,
        &config,
        &slf,
    );
    bubble.terrain_entity = Some(patch.entity);
    bubble.center_dir_body = center_dir;
    bubble.center_surface_body_m = patch.center_surface_body_m;
    bubble.basis = patch.basis;
    bubble.patch_half_extent_m = patch.half_extent_m;
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

fn debug_surface_drop(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    debug: Option<Res<DebugMode>>,
    height_sources: Res<HeightSourceRegistry>,
    config: Res<LocalBubbleConfig>,
    mut active: ResMut<ActiveLocalBubble>,
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

    info!(
        "debug surface drop placed {:?} {:.0} m above rendered {} terrain (day-side)",
        sim.simulation.vessel_kind(),
        config.debug_drop_height_m,
        body.name,
    );
}

/// Release a landed ship from `BodyFixed` when the pilot applies throttle.
///
/// A landed/parked ship sits in analytic `BodyFixed` authority — set at a runway
/// spawn ([`crate::runway`]), reached via the stable-landing collapse
/// ([`collapse_or_constrain_warp`]), or dropped there by a debug surface
/// teleport. Advancing the throttle is the pilot's "fly" command: drop warp to
/// 1× and hand translation back to the live regimes (`OnRails` →
/// [`manage_authority`] → `LocalRigidBody`/`Full`), where thrust and the landing
/// gear take over from the equilibrium pose. Because the parked pose already is
/// the gear equilibrium, the handoff is jump-free.
///
/// Ships only. Grounded EVA is pinned to `LocalRigidBody` (never `BodyFixed`) by
/// [`manage_authority`], so it is unaffected. This is the single landed→flying
/// transition, replacing the former debug-only `DebugLaunchMount` launch-clamp
/// release; a real staging / launch-clamp part can supersede it later.
fn release_landed_ship_on_throttle(throttle: Res<ThrottleState>, mut sim: ResMut<SimulationState>) {
    if throttle.commanded <= LANDED_THROTTLE_RELEASE {
        return;
    }
    if !matches!(sim.simulation.authority(), AuthorityMode::BodyFixed { .. })
        || sim.simulation.vessel_kind() != VesselKind::Ship
    {
        return;
    }
    sim.simulation
        .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
    sim.simulation.warp.reset_immediate();
    info!(
        "released landed ship on commanded throttle {:.2}",
        throttle.commanded
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
    let params = *sim.simulation.ship_params();
    // A destroyed craft is inert debris: gravity still acts (so it falls and
    // settles), but thrust and reaction-wheel torque are cut. See
    // `docs/surface.md`.
    let destroyed = sim.simulation.is_destroyed();

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
            rotation.0,
            angular_velocity.0,
            clock.delta_secs_f64(),
        )
    };
}

/// Convert the realized reaction-wheel torque command into a world-space
/// angular acceleration for the Avian rigid body.
///
/// `control.torque_command` is now the *output of the fly-by-wire attitude
/// controller* ([`crate::control_bus`]) — pointing, hold, or raw rate — so
/// this just scales it by `max_torque` and divides by inertia. The former
/// per-frame deadbeat SAS damper (`−I·ω/dt` when `sas_enabled`) lived here;
/// it annihilated all angular velocity every frame and limit-cycled against
/// continuous aero moments. SAS is now a proper critically-damped controller
/// upstream, so `sas_enabled` no longer does anything here.
fn compute_angular_acceleration(
    control: &thalos_physics_canonical::types::ControlInput,
    params: &thalos_physics_canonical::types::ShipParameters,
    rotation: DQuat,
    _angular_velocity_world: DVec3,
    _dt: f64,
) -> DVec3 {
    let inertia_body = params.moment_of_inertia;
    let max_torque = params.max_torque;
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

/// One landing-gear wheel as a **raycast suspension**, in the craft body frame.
///
/// All directions/points are craft-local (`X=right, Y=nose, Z=dorsal`) — the
/// same frame `gear_mesh` authors in — so `Rotation.0 * p` maps them into the
/// body-centered inertial frame the Avian rigid body lives in. Built once at
/// spawn from the gear parts ([`build_wheel_set`]) and cached so the per-frame
/// system does no part-tree walking.
#[derive(Clone, Copy, Debug, Reflect)]
pub struct Wheel {
    /// Strut top at the host skin — the suspension ray origin.
    pub strut_top_local: DVec3,
    /// Suspension axis (belly-ward `r̂`): the ray direction and spring line.
    pub susp_dir_local: DVec3,
    /// Roll axis (`fore`): brake / rolling resistance act along this.
    pub roll_dir_local: DVec3,
    /// Axle axis (`lateral`): lateral grip resists slip along this.
    pub axle_dir_local: DVec3,
    pub strut_length: f64,
    pub wheel_radius: f64,
    /// Nose (single-leg) gear steers; main pairs do not.
    pub steerable: bool,
}

/// Every wheel on a craft, attached to its Avian rigid body so
/// [`apply_landing_gear_forces`] can find them.
#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct WheelSet {
    pub wheels: Vec<Wheel>,
}

/// Tunable landing-gear suspension/grip coefficients. Reflect-registered so an
/// agent can live-tune them over BRP (`world_mutate_resources`) while taxiing,
/// rather than recompiling. Forces are computed per wheel and summed into the
/// craft's acceleration accumulators.
#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct GearTuning {
    /// Spring stiffness, N per metre of compression.
    pub k_spring: f64,
    /// Suspension damping ratio ζ. The actual damper coefficient is derived
    /// per frame from the craft's real per-wheel mass —
    /// `c = 2·ζ·√(k·m_per_wheel)` — so a wheel is near-critically damped (no
    /// bounce, no spin-pumping) regardless of craft mass. ζ = 1 is critical;
    /// slightly above gives a dead-beat settle.
    pub damping_ratio: f64,
    /// Friction-circle limit: max horizontal force as a multiple of normal load.
    pub mu: f64,
    /// Lateral grip stiffness, N per (m/s) of sideways slip (clamped by `mu·N`).
    pub k_lat: f64,
    /// Free-rolling resistance coefficient: the Coulomb cap on fore/aft wheel
    /// force as a fraction of normal load (`μ_roll·N`). A *constant* opposing
    /// force, not viscous — so a coasting craft decelerates linearly to a true
    /// stop and then holds, instead of asymptotically creeping forever. Small,
    /// so wheels stay low-resistance under thrust and roll on gentle slopes.
    pub rolling_mu: f64,
    /// Hold stiffness for the rolling-resistance Coulomb term: N per (m/s) of
    /// roll speed, clamped to `rolling_mu·N`. High enough that the breakaway
    /// speed (`cap/stiffness`) is a few cm/s, so below it the wheel is pinned
    /// (static) and above it the force saturates to the constant Coulomb cap.
    pub rolling_hold_stiffness: f64,
    /// Parking-brake hold stiffness: N per (m/s) of fore/aft creep, clamped to
    /// the friction circle `mu·N`. High so even a tiny creep is opposed
    /// near-maximally — the craft stays put when the brake is engaged.
    pub parking_brake_stiffness: f64,
    /// Max nosewheel steer angle at full yaw input, radians. This is the
    /// *taxi* (tiller) authority; it fades with ground speed — see
    /// [`GearTuning::steer_fade_speed_m_s`].
    pub max_steer_rad: f64,
    /// Ground speed (m/s) at which nosewheel steering authority has faded to
    /// half its taxi value (`scale = 1 / (1 + (v/v_fade)²)`). Full tiller
    /// throw at taxi speed would trip the craft over its main gear at takeoff
    /// speed, so steering blends out as the aero rudder blends in — the
    /// real-world tiller→pedals split.
    pub steer_fade_speed_m_s: f64,
    /// Max suspension travel as a fraction of strut length.
    pub max_travel_fraction: f64,
    /// Extra ray length past the rest length so a wheel just off the ground or
    /// over a slope edge still finds the surface, metres.
    pub skin_margin: f64,
}

impl Default for GearTuning {
    fn default() -> Self {
        // Sized for the ~20–40 t demo aircraft on Thalos surface gravity. The
        // craft settles to a static squat of `(m·g/N)/k_spring`; these put that
        // in the tens-of-cm range with near-critical damping. Live-tune via BRP.
        Self {
            // Stiff enough that the static sag `m·g/(n·k)` is ~cm-scale for the
            // demo aircraft (so the rigid wheel meshes don't visibly clip the
            // ground), but not so stiff it rings at the 60 Hz step — the
            // near-critical `damping_ratio` keeps it dead-beat.
            k_spring: 800_000.0,
            damping_ratio: 1.2,
            // Dry-tire grip. Deliberately below ~1.0: the lateral force a
            // skidding tire can transmit is what rolls a craft over its gear,
            // and a real tire slides at ~0.8 before it can generate a
            // tipping moment that large.
            mu: 0.8,
            k_lat: 40_000.0,
            rolling_mu: 0.02,
            rolling_hold_stiffness: 60_000.0,
            parking_brake_stiffness: 60_000.0,
            max_steer_rad: 0.5,
            steer_fade_speed_m_s: 12.0,
            max_travel_fraction: 0.8,
            skin_margin: 0.5,
        }
    }
}

/// Latched brakes (KSP-style, the B key). When engaged,
/// [`apply_landing_gear_forces`] replaces free rolling with a high-gain
/// fore/aft hold (clamped to the tyre friction circle), so the craft stays
/// put under gravity, slopes, and the residual settle — though full takeoff
/// thrust still overpowers it — and the spoilers deploy
/// ([`crate::flight_config`]), so the same latch is the in-air speedbrake
/// and the rollout lift dump.
///
/// Defaults **off** (most spawns are airborne and must not start with
/// spoilers out); the parked runway placement engages it explicitly so a
/// freshly-spawned aircraft holds on the strip
/// (`runway::finish_runway_spawn`). Reflect-registered so it's
/// visible/toggleable over BRP.
#[derive(Resource, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Resource)]
pub struct ParkingBrake {
    pub engaged: bool,
}

/// Whether any landing-gear wheel is currently bearing load on the ground
/// ("weight on wheels"). Set each frame by [`apply_landing_gear_forces`] from
/// its per-wheel suspension raycast, and read in the aero pass
/// ([`crate::aero::apply_aero_forces`]) to drop all aero on a grounded craft
/// below the taxi airspeed floor, where the AoA is degenerate (the velocity is
/// suspension settle, not flow). Above that floor a grounded craft flies the
/// full aero model — rotation authority and ground-roll damping are real
/// aerodynamics. Reflect-registered for BRP inspection.
#[derive(Resource, Default, Debug, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub struct WeightOnWheels {
    pub grounded: bool,
}

/// Coulomb friction tuning for a ship resting/sliding on its **hull** — gearless
/// craft (landers, rockets) or a craft on its belly. Wheeled craft get their
/// tangential ground reaction from the landing-gear model instead. Stick/slip
/// with a static and a kinetic coefficient, so a landed craft comes to a true
/// rest in finite time rather than the indefinite frictionless slide it had
/// before (the only ground force was the floor backstop, which removes the
/// into-surface velocity component only). Reflect-registered for BRP tuning.
#[derive(Resource, Clone, Copy, Debug, Reflect)]
#[reflect(Resource)]
pub struct SurfaceFriction {
    /// Static coefficient: a craft whose per-frame tangential slip is below
    /// `μ_static · g · dt` sticks (its surface-parallel velocity is zeroed).
    pub mu_static: f64,
    /// Kinetic coefficient (≤ static): a faster-sliding craft decelerates at
    /// `μ_kinetic · g` along its slip direction until it drops to the stick band.
    pub mu_kinetic: f64,
    /// How close the deepest hull point must sit to the surface (metres) to
    /// count as in contact — a small band above the floor backstop's lift so
    /// a craft held exactly at the surface still reads as grounded.
    pub contact_margin_m: f64,
}

impl Default for SurfaceFriction {
    fn default() -> Self {
        // Metal/composite hull on rock/regolith: high grip, true stop. A small
        // static>kinetic gap gives the usual break-free-then-slide feel.
        Self {
            mu_static: 0.8,
            mu_kinetic: 0.6,
            contact_margin_m: 0.3,
        }
    }
}

/// Flip the parking brake on the toggle edge (B). Runs before the gear forces.
fn toggle_parking_brake(intent: Res<GameInputIntent>, mut brake: ResMut<ParkingBrake>) {
    if intent.parking_brake_toggle {
        brake.engaged = !brake.engaged;
    }
}

/// Build the cached [`WheelSet`] for a craft from its landing-gear parts,
/// reusing [`gear_leg_frames`] (the same per-leg geometry the visual mesh
/// draws) so collider wheels sit exactly under the rendered ones. `positions`
/// is the part-tree translation map ([`compute_part_collider_positions`]); the
/// gear's mount point on the host axis mirrors the `BodySkin` station offset
/// that map applies.
/// Landing-gear contact geometry for parked placement: the lowest wheel-bottom
/// depth below the craft origin along the ventral (−Z) axis, plus the wheel
/// count. Derived from the **gear contact geometry** ([`build_wheel_set`], the
/// same data the runtime suspension uses), *not* visual meshes: at parked-spawn
/// time the gear's visual meshes may not be spawned yet, so a visual-extent
/// measurement would only see the fuselage and bury the gear. Returns `None` for
/// a craft with no landing gear (the caller falls back to the visual-mesh
/// clearance and rests it on its belly).
///
/// The depth is the *zero-compression* rest height (wheels just touching). The
/// caller subtracts the static suspension sag so the craft spawns with its gear
/// already loaded — see [`crate::runway`].
pub(crate) fn gear_contact_geometry(
    parts: &PartColliderQuery,
    gear_q: &Query<(&Gear, &SurfaceMount), (With<Part>, Without<thalos_shipyard::editor::EditorPart>)>,
    host_nodes: &Query<&AttachNodes>,
) -> Option<(f64, usize)> {
    let positions = compute_part_collider_positions(parts);
    let wheels = build_wheel_set(gear_q, host_nodes, &positions);
    if wheels.is_empty() {
        return None;
    }
    let lowest = wheels.iter().fold(f64::INFINITY, |acc, w| {
        let bottom = w.strut_top_local + w.susp_dir_local * (w.strut_length + w.wheel_radius);
        acc.min(bottom.z)
    });
    (lowest.is_finite() && lowest < 0.0).then_some((-lowest, wheels.len()))
}

pub(crate) fn build_wheel_set(
    gear_q: &Query<(&Gear, &SurfaceMount), (With<Part>, Without<thalos_shipyard::editor::EditorPart>)>,
    host_nodes: &Query<&AttachNodes>,
    positions: &HashMap<Entity, DVec3>,
) -> Vec<Wheel> {
    let mut wheels = Vec::new();
    for (gear, mount) in gear_q.iter() {
        let Ok(nodes) = host_nodes.get(mount.parent) else {
            continue;
        };
        let parent_radius = nodes
            .get("top")
            .map(|n| n.diameter * 0.5)
            .unwrap_or(1.0)
            .max(0.01);
        let host_pos = positions.get(&mount.parent).copied().unwrap_or(DVec3::ZERO);
        // The gear's mount origin on the host axis at its station — mirrors the
        // `SurfaceMountKind` branch in `compute_part_collider_positions`.
        let mount_axis = match mount.kind {
            SurfaceMountKind::BodySkin => {
                let host_height = nodes.get("bottom").map(|n| -n.offset.y).unwrap_or(0.0) as f64;
                host_pos + DVec3::new(0.0, -(mount.station as f64) * host_height, 0.0)
            }
            SurfaceMountKind::WingPylon => host_pos,
        };
        for leg in gear_leg_frames(gear, mount.angle, parent_radius) {
            wheels.push(Wheel {
                strut_top_local: mount_axis + leg.strut_top.as_dvec3(),
                susp_dir_local: leg.susp_dir.as_dvec3(),
                roll_dir_local: leg.roll_dir.as_dvec3(),
                axle_dir_local: leg.axle_dir.as_dvec3(),
                strut_length: gear.strut_length as f64,
                wheel_radius: gear.wheel_radius as f64,
                steerable: gear.legs() == 1,
            });
        }
    }
    wheels
}

/// Carry the craft on its wheels: a raycast spring/damper per wheel, plus
/// lateral grip, rolling resistance, brake, and emergent nosewheel-steer yaw.
///
/// Forces are summed into the craft's acceleration accumulators *after*
/// [`apply_local_forces`] has written gravity + thrust + reaction-wheel torque,
/// so this is a parallel channel on top of them. Runs only when Avian owns
/// translation ([`AvianRole::Full`]) for a live Ship — exactly when there is a
/// ground collider (runway slab or terrain patch) under the craft. The
/// craft-excluded downward raycast is itself the "is there ground here" test:
/// no hit → that wheel is airborne and contributes nothing. See `docs/surface.md`.
#[allow(clippy::too_many_arguments)]
fn apply_landing_gear_forces(
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    tuning: Res<GearTuning>,
    parking_brake: Res<ParkingBrake>,
    intent: Res<GameInputIntent>,
    spatial: SpatialQuery,
    sim: Res<SimulationState>,
    mut weight_on_wheels: ResMut<WeightOnWheels>,
    wheels_q: Query<&WheelSet>,
    mut craft_q: Query<
        (
            &Position,
            &Rotation,
            &LinearVelocity,
            &AngularVelocity,
            &mut ConstantLinearAcceleration,
            &mut ConstantAngularAcceleration,
        ),
        With<LocalCraftBody>,
    >,
) {
    // Default to airborne; any loaded wheel below flips this true. Cleared up
    // front so every early-return path (not owning translation, no gear, etc.)
    // correctly reports "no weight on wheels".
    weight_on_wheels.grounded = false;
    // Full only: Avian must own translation for the integrated force to mean
    // anything. Ships only — EVA has no gear — and never a destroyed wreck.
    if !authority.owns_translation()
        || sim.simulation.vessel_kind() != VesselKind::Ship
        || sim.simulation.is_destroyed()
    {
        return;
    }
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Ok(wheelset) = wheels_q.get(bubble.craft_entity) else {
        return;
    };
    if wheelset.wheels.is_empty() {
        return;
    }
    let Ok((position, rotation, linear_velocity, angular_velocity, mut linear_accel, mut angular_accel)) =
        craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };

    let rot = rotation.0;
    // Ships integrate in the body-fixed (rotating) frame and the ground
    // collider is static there, so the surface under every wheel reads ~0
    // velocity — `v_rel` is just the contact point's body-fixed velocity, with
    // no `ω × r` co-rotation term to subtract (and no phantom slip to pump a
    // spin). This is the payoff of the body-fixed frame.
    let mass = sim.simulation.ship_mass_kg().max(1.0);
    let inertia_body = sim.simulation.ship_params().moment_of_inertia;
    // Wheel torque is about the craft CoM (the Avian rotation pivot, set to
    // this same point at spawn), not the root origin — otherwise the upward
    // forces from gear that sit aft of the nose origin have no balancing
    // torque and flip the craft.
    let com_local = sim.simulation.ship_params().center_of_mass;
    // Near-critical damper derived from the *actual* per-wheel mass, so the
    // suspension settles dead-beat on any craft instead of bouncing.
    let m_per_wheel = mass / wheelset.wheels.len().max(1) as f64;
    let c_damp = 2.0 * tuning.damping_ratio * (tuning.k_spring * m_per_wheel).sqrt();
    // Note on ride height: the suspension finds its own torque-balanced
    // equilibrium (loaded wheels compress more), which is what keeps the craft
    // upright — do NOT preload the spring uniformly to cancel the sag, that
    // unbalances the torque and tips the craft over. Instead `k_spring` is sized
    // stiff enough that the static sag `m·g/(n·k)` is small (a couple cm), so the
    // rigid wheel meshes barely dip below the surface.

    // Nosewheel steering: full tiller throw at taxi speed, fading toward zero
    // with ground speed (the aero rudder takes over) so a hard yaw input at
    // takeoff speed can't generate the lateral grip that trips the craft over
    // its main gear.
    let ground_speed = linear_velocity.0.length();
    let steer_scale = 1.0 / (1.0 + (ground_speed / tuning.steer_fade_speed_m_s).powi(2));
    let steer = (intent.attitude.z as f64).clamp(-1.0, 1.0) * tuning.max_steer_rad * steer_scale;
    let filter = SpatialQueryFilter::default().with_excluded_entities([bubble.craft_entity]);

    let mut net_force = DVec3::ZERO;
    let mut net_torque = DVec3::ZERO;
    for wheel in &wheelset.wheels {
        let origin = position.0 + rot * wheel.strut_top_local;
        let down = rot * wheel.susp_dir_local;
        let Ok(dir) = Dir3::new(down.as_vec3()) else {
            continue;
        };
        // Geometric rest = wheel bottom on the surface (compression 0). The
        // craft sinks by the small static sag until the spring balances its load.
        let rest_len = wheel.strut_length + wheel.wheel_radius;
        let max_len = rest_len + tuning.skin_margin;
        let Some(hit) = spatial.cast_ray(origin, dir, max_len, true, &filter) else {
            continue;
        };
        let max_travel = tuning.max_travel_fraction * wheel.strut_length;
        let compression = (rest_len - hit.distance).clamp(0.0, max_travel);
        if compression <= 0.0 {
            continue;
        }
        // A wheel is bearing load: the craft has weight on its wheels, so the
        // aero pass should treat it as grounded and suppress tipping moments.
        weight_on_wheels.grounded = true;
        let up = -down;
        // Contact point relative to the craft CoM: the arm that turns wheel
        // force into torque about the rotation pivot (lets steered front-wheel
        // grip yaw the craft, and lets nose/main share the load in pitch).
        let contact_local = wheel.strut_top_local + wheel.susp_dir_local * hit.distance;
        let r_arm = rot * (contact_local - com_local);
        // Avian's LinearVelocity is the CoM velocity; the contact moves at
        // v_com + ω_craft × arm.
        // Ground is static in the body-fixed frame, so the contact's body-fixed
        // velocity is the slip directly.
        let v_rel = linear_velocity.0 + angular_velocity.0.cross(r_arm);

        // One-way spring + damper along the suspension axis (never pulls down).
        let compress_rate = -v_rel.dot(up);
        let normal_n = (tuning.k_spring * compression + c_damp * compress_rate).max(0.0);
        if normal_n <= 0.0 {
            continue;
        }
        let mu_n = tuning.mu * normal_n;

        // Steer the nose wheel by rotating its roll/axle dirs about the strut.
        let (roll_local, axle_local) = if wheel.steerable && steer.abs() > 1.0e-6 {
            let q = DQuat::from_axis_angle(wheel.susp_dir_local.normalize_or_zero(), steer);
            (q * wheel.roll_dir_local, q * wheel.axle_dir_local)
        } else {
            (wheel.roll_dir_local, wheel.axle_dir_local)
        };
        let axle_w = (rot * axle_local).normalize_or_zero();
        let roll_w = (rot * roll_local).normalize_or_zero();

        // Lateral grip resists sideways slip; longitudinal resists roll. Both
        // clamped to the friction circle so they only ever remove ground-relative
        // speed, never propel.
        let f_lat = -axle_w * (tuning.k_lat * v_rel.dot(axle_w)).clamp(-mu_n, mu_n);
        let roll_speed = v_rel.dot(roll_w);
        // Parking brake engaged → high-gain fore/aft hold (pins the craft);
        // released → free rolling resistance only.
        let f_roll = if parking_brake.engaged {
            -roll_w * (tuning.parking_brake_stiffness * roll_speed).clamp(-mu_n, mu_n)
        } else {
            // Coulomb rolling resistance: a stiff hold clamped to a small
            // `μ_roll·N` cap. The constant (non-viscous) cap means a coasting
            // craft loses speed linearly and reaches a true stop in finite time,
            // then the stiff term holds it within the cap — instead of the old
            // `∝ v` law that decayed exponentially and crept forever.
            let roll_cap = (tuning.rolling_mu * normal_n).min(mu_n);
            -roll_w * (tuning.rolling_hold_stiffness * roll_speed).clamp(-roll_cap, roll_cap)
        };

        let f = up * normal_n + f_lat + f_roll;
        net_force += f;
        net_torque += r_arm.cross(f);
    }

    if net_force == DVec3::ZERO && net_torque == DVec3::ZERO {
        return;
    }
    // Inertia-relative safety clamp, mirroring the aero force model
    // (`crate::aero`): a real undercarriage imparts at most a few g and a few
    // rad/s², so bounding the per-frame gear acceleration to the craft's own
    // mass/MOI makes a stiff-spring numerical blow-up impossible — a single bad
    // frame (or a discrete-step pumping cycle) can no longer spike the craft to
    // hundreds of rad/s and fling it off the runway — while leaving normal
    // taxi/landing loads (well under these limits) untouched. Without this the
    // gear was the one unclamped force path; see `docs/surface_local.md`.
    const GEAR_MAX_LIN_ACCEL_M_S2: f64 = 50.0; // ~5 g
    const GEAR_MAX_ANG_ACCEL_RAD_S2: f64 = 4.0;
    let lin_accel = net_force / mass;
    let lin_len = lin_accel.length();
    let lin_accel = if lin_len > GEAR_MAX_LIN_ACCEL_M_S2 {
        lin_accel * (GEAR_MAX_LIN_ACCEL_M_S2 / lin_len)
    } else {
        lin_accel
    };
    linear_accel.0 += lin_accel;

    let torque_body = rot.inverse() * net_torque;
    let inv_i = DVec3::new(
        if inertia_body.x > 0.0 { 1.0 / inertia_body.x } else { 0.0 },
        if inertia_body.y > 0.0 { 1.0 / inertia_body.y } else { 0.0 },
        if inertia_body.z > 0.0 { 1.0 / inertia_body.z } else { 0.0 },
    );
    let ang_accel = torque_body * inv_i;
    let ang_len = ang_accel.length();
    let ang_accel = if ang_len > GEAR_MAX_ANG_ACCEL_RAD_S2 {
        ang_accel * (GEAR_MAX_ANG_ACCEL_RAD_S2 / ang_len)
    } else {
        ang_accel
    };
    angular_accel.0 += rot * ang_accel;
}

/// Analytic ground backstop — a deterministic safety net that guarantees the
/// craft hull can never tunnel through the terrain, independent of the collision
/// mesh.
///
/// The terrain collider patch + `SweptCcd` are the *primary* contact layer, but
/// any mesh-based contact is probabilistic: a fast enough descent, an
/// edge-of-patch / not-yet-streamed tile, or a single missed sweep can let the
/// hull cross the surface — and the patch is a one-sided trimesh with no
/// "inside" to push back out of, so one missed frame becomes a permanent
/// fall-through. This system is the deterministic backstop. It samples terrain
/// height analytically (the same [`HeightSource`] the renderer and collider
/// read) directly under the craft and lifts any penetrating hull point back to
/// the surface, killing the into-surface velocity component. Because it is a
/// closed-form height query evaluated every frame — not a swept intersection —
/// it has no tunneling failure mode at any speed.
///
/// Ships only, and only while Avian owns translation ([`AvianRole::Full`]) —
/// the sole regime where Avian-integrated motion can drive the hull into the
/// ground. Under Kepler/OnRails coast the craft is far above the handoff band;
/// under warp / `BodyFixed` the pose is analytic and pinned. EVA is exempt (no
/// hull collider; the grounded controller owns its pose and clamps its own
/// terrain height). Runs after the force systems and just before
/// [`readback_local_craft`], so the corrected pose is what flows into canonical;
/// in `Full` the snap does not overwrite `Position`, so the correction persists
/// as the start state for Avian's next integration.
fn terrain_floor_backstop(
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    backstop: Res<TerrainFloorBackstop>,
    height_sources: Res<HeightSourceRegistry>,
    sim: Res<SimulationState>,
    mut craft_q: Query<
        (
            &mut Position,
            &Rotation,
            &mut LinearVelocity,
            &LocalCraftColliderPrimitives,
        ),
        With<LocalCraftBody>,
    >,
) {
    if !backstop.enabled
        || !authority.owns_translation()
        || sim.simulation.vessel_kind() != VesselKind::Ship
        || sim.simulation.is_destroyed()
    {
        return;
    }
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Some(height_source) = height_sources.get(bubble.body_id) else {
        return;
    };
    let body = &sim.system.bodies[bubble.body_id];
    let Ok((mut position, rotation, mut linear_velocity, primitives)) =
        craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };
    // Ship Avian state lives in the surface-local frame: `Position` is
    // anchor-relative (small), so recover the body-center offset for the
    // radial direction, and convert to body-fixed axes for the height query.
    // `LinearVelocity` is surface-relative (a parked craft reads ~0).
    let r_center = bubble.frame.body_center_offset(position.0);
    let Some(dir) = r_center.try_normalize() else {
        return;
    };
    let dir_body = bubble.frame.rotation_body_to_frame.inverse() * dir;
    let Some(height) = height_source.sample_height_m(dir_body.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
    else {
        return;
    };
    let surface_radius = body.radius_m + height as f64;

    // Deepest hull point along the radial — the lowest the hull reaches toward
    // the surface, measured from the body centre. See [`deepest_hull_radial`].
    let deepest = deepest_hull_radial(r_center, rotation.0, primitives, dir);
    if !deepest.is_finite() {
        return;
    }

    // Only the depth *past the skin* is corrected — see
    // [`TerrainFloorBackstop::skin_m`]. Resting on the collider (penetration
    // shallower than the skin) leaves this ≤ 0 and the backstop stands down, so
    // it never becomes a second hard floor fighting the soft contact solver.
    let excess = (surface_radius - deepest) - backstop.skin_m;
    if excess <= 0.0 {
        return;
    }
    if excess > 1.0 {
        // A correction this deep means the primary contact layer let the hull
        // sink a metre+ past the skin — exactly the tunnelling the backstop
        // exists to catch. Surface it so the mesh layer's gap can be investigated.
        debug!(
            "terrain floor backstop caught a {:.2} m hull penetration (past {:.2} m skin) over {}",
            excess + backstop.skin_m,
            backstop.skin_m,
            body.name
        );
    }
    // Lift the hull out to skin depth and remove only the into-surface (negative
    // radial) velocity, so tangential motion (taxi / slide) is untouched.
    position.0 += dir * excess;
    let radial_speed = linear_velocity.0.dot(dir);
    if radial_speed < 0.0 {
        linear_velocity.0 -= dir * radial_speed;
    }
}

/// Tuning + kill-switch for [`terrain_floor_backstop`]. Reflect-registered so it
/// can be toggled / tuned live over BRP (`world_mutate_resources`) while
/// diagnosing ground contact.
#[derive(Resource, Debug, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub struct TerrainFloorBackstop {
    /// Master enable. Off → the backstop never moves the craft (diagnostic
    /// isolation lever).
    pub enabled: bool,
    /// Allowed penetration skin, metres.
    ///
    /// The backstop is a *deep-penetration safety net*, **not** a zero-tolerance
    /// surface clamp. The primary contact layer — the terrain/runway collider,
    /// gear suspension, and hull friction — is a soft (XPBD) solver that tolerates
    /// small per-frame penetration and resolves it over substeps. A backstop that
    /// clamps at zero depth becomes a *second, stiffer* hard floor that disagrees
    /// with the collider by sub-metre amounts and fights it every frame →
    /// uncontrollable jitter. So the backstop ignores penetration shallower than
    /// this skin and corrects only the excess below it: normal resting contact is
    /// left entirely to the solver, while genuine tunnelling (metres deep) is
    /// still caught. The craft can never end up more than `skin_m` below the
    /// surface, and can never pass through.
    pub skin_m: f64,
}

impl Default for TerrainFloorBackstop {
    fn default() -> Self {
        Self {
            enabled: true,
            skin_m: 0.5,
        }
    }
}

/// Coulomb surface friction for a ship resting/sliding on its **hull** (no
/// weight on wheels). Velocity-level stick/slip on the tangential
/// (surface-parallel) component of the craft's body-fixed velocity, applied the
/// same frame [`terrain_floor_backstop`] removes the into-surface component and
/// just before [`readback_local_craft`] flows the corrected velocity into
/// canonical. Brings a landed gearless craft to a true rest in finite time
/// instead of the indefinite slide it had before — the only ground force was the
/// backstop, which touches the normal direction only.
///
/// Done at the velocity level (like the backstop), not as a force into the
/// acceleration accumulator: a velocity-level stick/slip cancels exactly within
/// the friction budget each frame, so it reaches a true stop regardless of step
/// size, whereas an `∝ v` force law only ever decays asymptotically (the bug
/// this replaces). Gravity here is central (radial), so it has no tangential
/// component — friction only has to remove residual slip, and the normal load
/// per unit mass is just `g = μ/r²`.
///
/// Wheeled craft are skipped: when any wheel bears load the landing-gear model
/// owns the tangential ground reaction (lateral grip + Coulomb rolling) and the
/// suspension holds the hull clear of the surface.
fn apply_surface_friction(
    clock: Res<SimClock>,
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    height_sources: Res<HeightSourceRegistry>,
    weight_on_wheels: Res<WeightOnWheels>,
    tuning: Res<SurfaceFriction>,
    sim: Res<SimulationState>,
    mut craft_q: Query<
        (
            &Position,
            &Rotation,
            &mut LinearVelocity,
            &LocalCraftColliderPrimitives,
        ),
        With<LocalCraftBody>,
    >,
) {
    if !authority.owns_translation()
        || sim.simulation.vessel_kind() != VesselKind::Ship
        || sim.simulation.is_destroyed()
        || weight_on_wheels.grounded
    {
        return;
    }
    let dt = clock.delta_secs_f64();
    if dt <= 0.0 {
        return;
    }
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Some(height_source) = height_sources.get(bubble.body_id) else {
        return;
    };
    let body = &sim.system.bodies[bubble.body_id];
    let Ok((position, rotation, mut linear_velocity, primitives)) =
        craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };
    // Surface-local frame: recover the body-center offset for the radial
    // direction (local up), body-fixed axes for the height query.
    // `LinearVelocity` is surface-relative (the ground is static here, so
    // tangential velocity is the slip directly).
    let r_center = bubble.frame.body_center_offset(position.0);
    let Some(dir) = r_center.try_normalize() else {
        return;
    };
    let dir_body = bubble.frame.rotation_body_to_frame.inverse() * dir;
    let Some(height) = height_source.sample_height_m(dir_body.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
    else {
        return;
    };
    let surface_radius = body.radius_m + height as f64;

    // In hull contact? The backstop holds the deepest hull point ~at the surface;
    // a small margin keeps a resting craft reading grounded across sample noise.
    let deepest = deepest_hull_radial(r_center, rotation.0, primitives, dir);
    if !deepest.is_finite() || deepest > surface_radius + tuning.contact_margin_m {
        return;
    }

    let v = linear_velocity.0;
    let v_tan = v - dir * v.dot(dir);
    let speed = v_tan.length();
    if speed < 1.0e-9 {
        return;
    }
    // Normal load per unit mass = gravity into the (near-radial) surface. Mass
    // cancels in the velocity-level form, so it never enters.
    let g = body.gm / r_center.length_squared().max(1.0);
    let static_budget = tuning.mu_static * g * dt;
    if speed <= static_budget {
        // Stick: remove all tangential motion, keep the radial component.
        linear_velocity.0 = v - v_tan;
    } else {
        // Slip: kinetic friction opposes the slip at `μ_kinetic · g`.
        linear_velocity.0 = v - (v_tan / speed) * (tuning.mu_kinetic * g * dt);
    }
}

/// Deepest hull point along `dir` (local up): the minimum radial coordinate
/// (projection onto `dir`) over every collider primitive's true support point in
/// the `-dir` direction. Each primitive lives in the craft body frame, so its
/// world (body-fixed) centre is `position + R_craft · offset`. `dir` must be
/// unit-length. Shared by [`terrain_floor_backstop`] (penetration → lift) and
/// [`apply_surface_friction`] (contact test).
fn deepest_hull_radial(
    position: DVec3,
    rotation: DQuat,
    primitives: &LocalCraftColliderPrimitives,
    dir: DVec3,
) -> f64 {
    let mut deepest = f64::INFINITY;
    for prim in &primitives.0 {
        let center = position + rotation * prim.offset_m;
        // `a = (R_craft · R_prim)^T · dir` is unit-length (rotations preserve
        // norm, `dir` is unit); `shape_min_support` returns the shape's signed
        // support depth along `-a`, i.e. how far below its centre the hull
        // reaches along the radial.
        let a = (rotation * prim.rotation).inverse() * dir;
        deepest = deepest.min(center.dot(dir) + shape_min_support(prim.shape, a));
    }
    deepest
}

/// Minimum of `a · p` over the points `p` of a centred primitive shape — the
/// signed depth of the shape's support point in the `-a` direction. `a` must be
/// unit-length; the result is ≤ 0. Exact for cuboid/sphere/capsule/cylinder;
/// the cone is bounded conservatively by its enclosing cylinder (a backstop
/// erring toward catching the hull slightly early is safe).
fn shape_min_support(shape: LocalPrimitiveShape, a: DVec3) -> f64 {
    match shape {
        // `Collider::cuboid` takes full side lengths; support uses half-extents.
        LocalPrimitiveShape::Cuboid { x, y, z } => {
            -(a.x.abs() * x + a.y.abs() * y + a.z.abs() * z) * 0.5
        }
        LocalPrimitiveShape::Sphere { radius } => -radius,
        // Parry capsule/cylinder/cone principal axis is local Y. `length` is the
        // capsule's segment length (between hemisphere centres); `height` is the
        // full cylinder/cone height.
        LocalPrimitiveShape::Capsule { radius, length } => -(a.y.abs() * length * 0.5) - radius,
        LocalPrimitiveShape::Cylinder { radius, height }
        | LocalPrimitiveShape::Cone { radius, height } => {
            -(a.y.abs() * height * 0.5) - radius * (a.x * a.x + a.z * a.z).sqrt()
        }
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
/// doesn't re-trigger. See `docs/surface.md`.
fn detect_terrain_impact(
    contact_graph: Res<ContactGraph>,
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    weight_on_wheels: Res<WeightOnWheels>,
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
    let Ok((linear_velocity, _position)) = craft_q.get(bubble.craft_entity) else {
        return;
    };

    // Ships integrate in the surface-local frame and the ground collider is
    // static there, so the craft's SLF velocity is already the surface-relative
    // approach speed (a craft resting on the surface reads ~0). No `ω × r`
    // subtraction needed.
    let approach_speed = linear_velocity.0.length();
    speed_window.push(approach_speed);

    // Ground contact onset. A wheeled craft's hull is filtered out of solver
    // contact with the ground (gear is its sole interface), so use the gear's
    // weight-on-wheels signal for it; a gearless craft still contacts the
    // terrain heightfield directly, so fall back to the contact graph.
    let hull_touches = bubble
        .terrain_entity
        .is_some_and(|t| craft_contacts_terrain(&contact_graph, bubble.craft_entity, t));
    let touching = weight_on_wheels.grounded || hull_touches;
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

/// Horizontal drift from the anchor that triggers a re-anchor. Keeps the
/// craft's SLF coordinates small near the ground; each re-anchor is an exact
/// f64 state translation (a handful of quaternion ops), so even the orbital
/// AttitudeOnly regime crossing the surface at ~2 km/s re-anchors cheaply.
const REANCHOR_HORIZONTAL_M: f64 = 1500.0;

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
    // Terrain patches only exist for ships, whose Avian body is in the
    // surface-local frame — recover the body-fixed body-centered position the
    // patch metadata (`center_surface_body_m`, `center_dir_body`) is expressed
    // in. (EVA never attaches a patch, so it never reaches here.)
    let craft_body_fixed = current.frame.rotation_body_to_frame.inverse()
        * current.frame.body_center_offset(position.0);
    let delta = craft_body_fixed - current.center_surface_body_m;
    let along = delta.dot(current.center_dir_body);
    let lateral = (delta - along * current.center_dir_body).length();
    let current_revision = height_source.revision();
    // Re-center before the craft drifts off the patch edge. The tile-based
    // collider window (docs/surface.md §3.6) is only tens of metres, so cap the
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
        &config,
        &current.frame,
    );
    active.bubble = Some(LocalBubble {
        terrain_entity: Some(patch.entity),
        center_dir_body: center_dir,
        center_surface_body_m: patch.center_surface_body_m,
        basis: patch.basis,
        patch_half_extent_m: patch.half_extent_m,
        stable_contact_s: current.stable_contact_s,
        stable_landed: current.stable_landed,
        terrain_built_at_revision: current_revision,
        ..current
    });
}

/// Hold the kinematic terrain collider **static in the surface-local frame**:
/// its mesh vertices are body-fixed offsets from `center_surface_body_m`, so
/// with `Position` = the patch centre in SLF coordinates and `Rotation` = the
/// constant body-fixed→SLF rotation, every contact point sits exactly where
/// the rotating surface is — with zero velocity, genuinely static geometry.
/// The pose is constant between re-anchors (this is a cheap idempotent write
/// that guarantees consistency after [`reanchor_surface_frame`] swaps the
/// frame), so the contact solver sees a floor that never moves.
fn sync_terrain_collider_pose(
    active: Res<ActiveLocalBubble>,
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
    // The heightfield is authored in the patch-tangent frame (height along its
    // local +Y = the patch up-normal), so its SLF rotation composes the
    // body-fixed→SLF rotation with the patch-basis rotation.
    position.0 = bubble.frame.rotation_body_to_frame
        * (bubble.center_surface_body_m - bubble.frame.anchor_point_body_m);
    rotation.0 = bubble.frame.rotation_body_to_frame
        * thalos_physics_local::patch_basis_rotation(&bubble.basis);
    linear_velocity.0 = DVec3::ZERO;
    angular_velocity.0 = DVec3::ZERO;
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
    weight_on_wheels: Res<WeightOnWheels>,
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
    let Ok((linear_velocity, angular_velocity)) = craft_q.get(bubble.craft_entity) else {
        return;
    };
    // Ground contact: a wheeled craft's hull is filtered out of solver contact
    // (gear is its sole ground interface), so weight-on-wheels is its landed
    // signal; a gearless craft contacts the terrain heightfield directly via the
    // contact graph. Either way, with no contact the timer resets and the craft
    // stays a ballistic / coasting body.
    let hull_touches = bubble
        .terrain_entity
        .is_some_and(|t| craft_contacts_terrain(&contact_graph, bubble.craft_entity, t));
    let contact = weight_on_wheels.grounded || hull_touches;
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
        // BodyFixed pose until throttle-up releases it back to live physics.
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
/// [`release_landed_ship_on_throttle`].
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
    // Keep the (ship-frame) SLF coherent with the new body even though the EVA
    // seam doesn't read it — ship-only systems consult `bubble.frame` by body.
    bubble.frame = SurfaceLocalFrame::new(
        &body_state,
        surface_anchor_under(&body_state, None, translation.position),
    );

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

/// Convert canonical inertial state into the **ship** Avian body's frame.
///
/// Ships use the **surface-local frame (SLF)**: a body-fixed tangent frame
/// anchored at a surface point, Y-up, small coordinates near the anchor
/// (`thalos_physics_canonical::surface_local`, `docs/surface_local.md`). The
/// frame co-rotates with the body, so a craft parked on or taxiing across the
/// surface is ~stationary instead of translating at the surface co-rotation
/// speed (`ω×r`, hundreds of m/s), and the ground colliders are genuinely
/// static geometry (see [`sync_terrain_collider_pose`]). Frame velocity is
/// airspeed (the atmosphere co-rotates).
///
/// EVA keeps the body-centered [`inertial_to_bubble_frame`] seam — its capsule
/// is owned directly by the on-foot controller and never touches a runway.
fn inertial_to_ship_frame(
    body_state: &BodyState,
    frame: &SurfaceLocalFrame,
    translation: TranslationalState,
    attitude: AttitudeState,
) -> BubbleFrame {
    let state = inertial_to_surface_local(body_state, frame, translation, attitude);
    BubbleFrame {
        position_m: state.position_m,
        rotation: state.orientation_frame.normalize(),
        linear_velocity_m_s: state.velocity_m_s,
        // Avian's `AngularVelocity` lives in the surrounding (SLF) axes;
        // `SurfaceLocalState` carries it in the craft body frame.
        angular_velocity_rad_s: state.orientation_frame * state.angular_velocity_body,
    }
}

fn ship_frame_to_inertial(
    body_state: &BodyState,
    frame: &SurfaceLocalFrame,
    position_m: DVec3,
    rotation: DQuat,
    linear_velocity_m_s: DVec3,
    angular_velocity_rad_s: DVec3,
) -> (TranslationalState, AttitudeState) {
    let orientation = rotation.normalize();
    let state = SurfaceLocalState {
        position_m,
        velocity_m_s: linear_velocity_m_s,
        orientation_frame: orientation,
        angular_velocity_body: orientation.inverse() * angular_velocity_rad_s,
    };
    surface_local_to_inertial(body_state, frame, state)
}

/// Pick the inertial→Avian conversion for the craft's kind: ships are
/// surface-local (body-fixed tangent frame), EVA is body-centered inertial.
fn inertial_to_craft_frame(
    kind: VesselKind,
    body_state: &BodyState,
    frame: &SurfaceLocalFrame,
    translation: TranslationalState,
    attitude: AttitudeState,
) -> BubbleFrame {
    match kind {
        VesselKind::Ship => inertial_to_ship_frame(body_state, frame, translation, attitude),
        VesselKind::Eva => inertial_to_bubble_frame(body_state, translation, attitude),
    }
}

/// Inverse of [`inertial_to_craft_frame`].
fn craft_frame_to_inertial(
    kind: VesselKind,
    body_state: &BodyState,
    frame: &SurfaceLocalFrame,
    position_m: DVec3,
    rotation: DQuat,
    linear_velocity_m_s: DVec3,
    angular_velocity_rad_s: DVec3,
) -> (TranslationalState, AttitudeState) {
    match kind {
        VesselKind::Ship => ship_frame_to_inertial(
            body_state,
            frame,
            position_m,
            rotation,
            linear_velocity_m_s,
            angular_velocity_rad_s,
        ),
        VesselKind::Eva => bubble_frame_to_inertial(
            body_state,
            position_m,
            rotation,
            linear_velocity_m_s,
            angular_velocity_rad_s,
        ),
    }
}

/// Build a [`SurfaceAnchor`] at the surface projection of an inertial
/// position, sampling terrain elevation when a height source is available
/// (the anchor elevation only places the frame origin — conversions are
/// exact regardless, so a missing source degrades to reference-radius
/// origin, not to incorrectness).
fn surface_anchor_under(
    body_state: &BodyState,
    height_source: Option<&dyn HeightSource>,
    position_inertial: DVec3,
) -> SurfaceAnchor {
    let dir_body = (body_state.orientation.inverse() * (position_inertial - body_state.position))
        .normalize_or_zero();
    let dir_body = if dir_body == DVec3::ZERO {
        DVec3::Y
    } else {
        dir_body
    };
    let elevation_m = height_source
        .and_then(|hs| hs.sample_height_m(dir_body.as_vec3(), PHYSICS_QUERY_TILE_LOD_M))
        .map(|h| h as f64)
        .unwrap_or(0.0);
    SurfaceAnchor {
        dir_body,
        elevation_m,
    }
}

fn level_attitude_for_body_dir(body_orientation: DQuat, up_body: DVec3) -> DQuat {
    let basis = thalos_body_render::TerrainPatchBasis::from_normal(up_body);
    let nose_body = basis.tangent_z;
    let dorsal_body = up_body.normalize();
    let right_body = nose_body.cross(dorsal_body).normalize();
    let craft_to_body = DMat3::from_cols(right_body, nose_body, dorsal_body);
    (body_orientation * DQuat::from_mat3(&craft_to_body)).normalize()
}

pub(crate) type PartColliderQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AttachNodes,
        Option<&'static Attachment>,
        Option<&'static SurfaceMount>,
        Option<&'static CommandPod>,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Engine>,
        Option<&'static AirIntake>,
        Option<&'static Wing>,
    ),
    // The in-game shipyard editor's build shares these components; it must
    // never contribute colliders, wheels, or clearance to the flight craft.
    (With<Part>, Without<thalos_shipyard::editor::EditorPart>),
>;

fn build_ship_collider_primitives(parts: &PartColliderQuery) -> Vec<LocalPrimitiveCollider> {
    let part_positions = compute_part_collider_positions(parts);
    let nodes_by_entity: HashMap<Entity, &AttachNodes> =
        parts.iter().map(|(e, nodes, ..)| (e, nodes)).collect();
    let mut primitives = Vec::new();
    for (entity, nodes, _, surface_mount, pod, dec, adapter, tank, engine, intake, wing) in
        parts.iter()
    {
        let Some(part_position) = part_positions.get(&entity).copied() else {
            continue;
        };
        // Wings are thin lifting surfaces, not body-axis solids: give each a
        // thin angled slab matching its planform so a wingtip catches the
        // ground (e.g. on an over-banked landing) instead of passing through.
        if let (Some(wing), Some(mount)) = (wing, surface_mount) {
            let parent_radius = nodes_by_entity
                .get(&mount.parent)
                .and_then(|n| n.get("top"))
                .map(|node| node.diameter * 0.5)
                .unwrap_or(1.0);
            primitives.push(wing_collider_primitive(wing, mount, parent_radius, part_position));
            continue;
        }
        if matches!(
            (engine, surface_mount.map(|m| m.kind)),
            (Some(engine), Some(SurfaceMountKind::WingPylon))
                if engine.geometry == EngineGeometry::JetNacelle
        ) {
            continue;
        }
        let Some((shape, local_offset)) =
            part_collider_shape(nodes, pod, dec, adapter, tank, engine, intake)
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

/// A thin oriented-cuboid collider matching a wing panel's planform, in the
/// craft body frame. Reuses [`wing_panel_frame`] — the same geometry the wing
/// mesh draws — so the collider tracks the rendered wing. `host_axis_pos` is the
/// wing's mount point on the host axis (from [`compute_part_collider_positions`]).
fn wing_collider_primitive(
    wing: &Wing,
    mount: &SurfaceMount,
    parent_radius: f32,
    host_axis_pos: DVec3,
) -> LocalPrimitiveCollider {
    let frame = wing_panel_frame(wing, mount.angle, parent_radius);
    let center_local = (frame.root_center + frame.tip_center) * 0.5;
    let span_len = (frame.tip_center - frame.root_center).length().max(0.1);
    let chord = wing.root_chord.max(wing.tip_chord).max(0.1);
    let thickness = (wing.root_chord * wing.thickness).max(0.05);
    // Orthonormal slab axes: span (y) and thickness (z) are perpendicular by
    // construction (`thick = span × fore`); recover a clean chord axis as
    // `span × thick` so the basis is a valid rotation even though `fore_dir`
    // itself is tilted by incidence and not exactly ⊥ to span.
    let span_dir = frame.span_dir.as_dvec3().normalize_or(DVec3::X);
    let thick_dir = frame.thick_dir.as_dvec3().normalize_or(DVec3::Z);
    let chord_dir = span_dir.cross(thick_dir).normalize_or(DVec3::Y);
    let basis = DMat3::from_cols(chord_dir, span_dir, thick_dir);
    LocalPrimitiveCollider {
        offset_m: host_axis_pos + center_local.as_dvec3(),
        rotation: DQuat::from_mat3(&basis).normalize(),
        shape: LocalPrimitiveShape::Cuboid {
            x: chord as f64,
            y: span_len as f64,
            z: thickness as f64,
        },
    }
}

pub(crate) fn compute_part_collider_positions(parts: &PartColliderQuery) -> HashMap<Entity, DVec3> {
    let mut nodes_by_entity: HashMap<Entity, &AttachNodes> = HashMap::new();
    let mut children_by_parent: HashMap<Entity, Vec<(Entity, Attachment)>> = HashMap::new();
    let mut surface_children_by_parent: HashMap<Entity, Vec<(Entity, SurfaceMount)>> =
        HashMap::new();
    let mut roots = Vec::new();

    for (entity, nodes, attachment, surface_mount, ..) in parts.iter() {
        nodes_by_entity.insert(entity, nodes);
        if let Some(attachment) = attachment {
            children_by_parent
                .entry(attachment.parent)
                .or_default()
                .push((entity, attachment.clone()));
        } else if let Some(surface_mount) = surface_mount {
            surface_children_by_parent
                .entry(surface_mount.parent)
                .or_default()
                .push((entity, *surface_mount));
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
        if let Some(children) = children_by_parent.get(&parent) {
            for (child, attachment) in children {
                let Some(parent_node) = parent_nodes.get(&attachment.parent_node) else {
                    continue;
                };
                let child_offset = nodes_by_entity
                    .get(child)
                    .and_then(|nodes| nodes.get(&attachment.my_node))
                    .map(|node| node.offset)
                    .unwrap_or(Vec3::ZERO);
                let child_position =
                    parent_position + (parent_node.offset - child_offset).as_dvec3();
                positions.insert(*child, child_position);
                queue.push_back(*child);
            }
        }

        if let Some(children) = surface_children_by_parent.get(&parent) {
            for (child, mount) in children {
                let local_offset = match mount.kind {
                    SurfaceMountKind::BodySkin => {
                        let host_height = parent_nodes
                            .get("bottom")
                            .map(|node| -node.offset.y)
                            .unwrap_or(0.0);
                        DVec3::new(0.0, -(mount.station as f64) * host_height as f64, 0.0)
                    }
                    SurfaceMountKind::WingPylon => DVec3::ZERO,
                };
                positions.insert(*child, parent_position + local_offset);
                queue.push_back(*child);
            }
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
    intake: Option<&AirIntake>,
) -> Option<(LocalPrimitiveShape, DVec3)> {
    if let Some(pod) = pod {
        let height = pod.diameter * pod.geometry.length_factor();
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
        let height = match engine.geometry {
            EngineGeometry::RocketBell => engine.diameter * 0.9,
            EngineGeometry::JetNacelle => thalos_shipyard::jet_nacelle_length(engine),
        };
        Some((
            LocalPrimitiveShape::Cylinder {
                radius: (engine.diameter * 0.5) as f64,
                height: height as f64,
            },
            DVec3::Y * -(height as f64 * 0.5),
        ))
    } else if let Some(intake) = intake {
        Some((
            LocalPrimitiveShape::Cylinder {
                radius: (intake.diameter * 0.5) as f64,
                height: intake.length as f64,
            },
            DVec3::Y * -(intake.length as f64 * 0.5),
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thalos_body_render::{TerrainPatchBasis, TerrainPatchMesh};
    use thalos_physics_canonical::canonical::Epoch;
    use thalos_physics_canonical::types::BodyState;

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
            stable_contact_s: 0.0,
            stable_landed: false,
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
            avian_role_from_inputs(1.0, on_rails(), 0.0, false, false),
            AvianRole::AttitudeOnly
        );
    }

    #[test]
    fn thrust_at_one_x_takes_full_ownership() {
        assert_eq!(
            avian_role_from_inputs(1.0, on_rails(), 0.5, false, false),
            AvianRole::Full
        );
    }

    #[test]
    fn terrain_collider_attached_at_one_x_takes_full_ownership() {
        // Inside the AGL handoff band the terrain collider is present;
        // Avian needs to own translation so contact resolution can fire.
        assert_eq!(
            avian_role_from_inputs(1.0, on_rails(), 0.0, true, false),
            AvianRole::Full
        );
    }

    #[test]
    fn inside_atmosphere_at_one_x_takes_full_ownership() {
        // Below the Kármán line, aerodynamic forces act, so Avian must own
        // translation across the whole column (not just the terrain band) —
        // otherwise reentry drag would be skipped above the handoff altitude.
        assert_eq!(
            avian_role_from_inputs(1.0, on_rails(), 0.0, false, true),
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
            avian_role_from_inputs(10.0, on_rails(), 1.0, true, true),
            AvianRole::Paused
        );
        assert_eq!(
            avian_role_from_inputs(1_000_000.0, on_rails(), 0.5, false, true),
            AvianRole::Paused
        );
    }

    #[test]
    fn body_fixed_authority_pauses_avian() {
        // Landed pose is evaluated analytically from the body's rotation;
        // Avian holds the rigid body in place but does not integrate. This
        // must hold even with thrust applied — `release_landed_ship_on_throttle`
        // releases a landed ship by transitioning out of BodyFixed first.
        assert_eq!(
            avian_role_from_inputs(1.0, body_fixed(), 0.0, false, false),
            AvianRole::Paused
        );
        assert_eq!(
            avian_role_from_inputs(1.0, body_fixed(), 0.9, true, true),
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
            requires_atmosphere: false,
            intake_requirement: None,
            builtin_intake: None,
            diameter: 2.0,
            thrust: 0.0,
            isp: 0.0,
            dry_mass: 0.0,
            reactants: Vec::new(),
            power_draw_kw: 0.0,
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
