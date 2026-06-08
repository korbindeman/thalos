//! EVA-style player controller for walking on rendered terrain.
//!
//! The Avian rigid body (a 1.8 m capsule) is spawned by
//! `local_physics::spawn_player_avian_body` when `VesselKind::Eva` is
//! active — the same seam that builds the multi-part compound collider
//! for a `Ship`. The entity carries both `LocalCraftBody` (so canonical
//! readback / authority / terrain-collider attachment all flow through
//! the existing local-physics machinery) and `PlayerControllerBody` (so
//! the systems in this module find it).
//!
//! # Frame
//!
//! The character is simulated in the body's **body-fixed (rotating)
//! frame**: a position `pos_bf` relative to the body centre, expressed in
//! axes that co-rotate with the surface. In that frame the ground is
//! stationary, gravity is a near-constant radial "down", and surface
//! velocity is the player's own walking/jumping speed (m/s) — not the
//! body-centred-inertial co-rotation drag (ω×r, hundreds of m/s to
//! km/s). This is the same trick KSP uses near the ground (Krakensbane),
//! and it lets the controller run real character physics — gravity, a
//! grounded/airborne state machine, jumping, falling off ledges, landing
//! — without the inertial-frame velocity exploding under time warp.
//!
//! Each frame the body-fixed state is converted to the body-centred
//! inertial offset Avian's kinematic `Position` holds via
//! `position = body.orientation * pos_bf`; under warp this co-rotation is
//! exact and cheap regardless of warp level.
//!
//! This module owns:
//! - the visible mesh (`PlayerControllerVisual`), a BigSpace entity
//!   synced from physics each frame so render transforms stay in
//!   metre-scale cells instead of inheriting body-centred-inertial
//!   coordinates;
//! - the EVA character physics (the only force writer for EVA —
//!   `apply_local_forces` early-returns for `VesselKind::Eva`);
//! - terrain snap, rest detection, and camera focus.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::canonical::{Epoch, TranslationalState};
use thalos_physics_canonical::types::{AttitudeState, BodyState};
use thalos_physics_local::avian::{AngularVelocity, LinearVelocity, Position, Rotation};
use thalos_physics_local::{ActiveLocalBubble, HeightSourceRegistry, LocalCraftBody};
use thalos_world::BodyId;

use crate::SimStage;
use crate::camera::{ActiveCamera, CameraFocus, CameraFocusTarget, OrbitCamera};
use crate::local_physics::PHYSICS_QUERY_TILE_LOD_M;
use crate::rendering::real_space::RealSpaceRoot;
use crate::rendering::{SimulationState, SolarSystemState};
use crate::sim_clock::SimClock;
use crate::view::{HideInMapView, ViewMode};

const PLAYER_HEIGHT_M: f64 = 1.8;
const PLAYER_RADIUS_M: f64 = 0.32;
const PLAYER_CAPSULE_SEGMENT_M: f64 = PLAYER_HEIGHT_M - PLAYER_RADIUS_M * 2.0;
const PLAYER_HALF_HEIGHT_M: f64 = PLAYER_HEIGHT_M * 0.5;
// The EVA mesh is a round-bottom capsule with no feet. A tangent placement
// reads as hovering in third-person, so sink it a hair into the terrain like
// most game character controllers do.
const PLAYER_FOOT_CLEARANCE_M: f64 = -0.04;
const PLAYER_WALK_SPEED_M_S: f64 = 1.6;
const PLAYER_RUN_SPEED_M_S: f64 = 5.5;
const PLAYER_CAMERA_DISTANCE_M: f64 = 6.0;

/// Target apex height of a standing jump, in metres. Converted to a launch
/// speed against the local surface gravity, so jumps feel floatier on
/// low-gravity bodies (KSP-style) without retuning per body.
const PLAYER_JUMP_HEIGHT_M: f64 = 1.1;
/// Largest ground step (up or down) the grounded controller will follow in a
/// single contact resolve. Beyond this the player has walked off a ledge and
/// transitions to the airborne (falling) state instead of snapping down.
const GROUND_SNAP_M: f64 = 0.6;
/// How quickly airborne horizontal velocity steers toward the input direction
/// (fraction per second). Lower than ground control so jumps commit.
const AIR_CONTROL_PER_S: f64 = 3.0;
/// Surface-relative speed below which the player counts as stationary.
const REST_SPEED_EPS_M_S: f64 = 0.05;
/// How long the player must be stationary before counting as "at rest"
/// (warp-eligible). Short enough to feel instant, long enough to debounce.
const REST_TIME_S: f64 = 0.2;
/// Maximum turn rate of the character's facing toward its movement direction
/// (radians/second). Gives a smooth third-person pivot rather than a snap.
const TURN_RATE_RAD_S: f64 = 14.0;
/// Discrepancy between the stored body-fixed pose and Avian's `Position` that
/// counts as an external teleport (F9 drop / map-cursor plant), triggering a
/// re-seed of the character state from the rigid body.
const TELEPORT_RESEED_M: f64 = 1.0;

pub struct PlayerControllerPlugin;

impl Plugin for PlayerControllerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlayerControllerState>()
            .init_resource::<EvaMode>()
            .register_type::<EvaMode>()
            .add_systems(
                Update,
                (
                    register_eva_visual,
                    step_eva_controller,
                    refresh_player_controller_state,
                )
                    .chain()
                    .in_set(SimStage::Physics)
                    .after(crate::bridge::advance_simulation),
            )
            .add_systems(
                Update,
                (
                    sync_player_controller_visual,
                    sync_player_controller_camera_focus,
                )
                    .chain()
                    .in_set(SimStage::Sync)
                    .after(crate::solar_system_state::sync_solar_system_state),
            );
    }
}

#[derive(Resource, Default, Debug, Clone)]
pub struct PlayerControllerState {
    active: Option<ActivePlayerController>,
}

impl PlayerControllerState {
    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }

    pub fn active_position_m(&self) -> Option<DVec3> {
        self.active.map(|active| active.inertial_position_m)
    }

    /// Whether the on-foot player is standing on the surface (vs airborne /
    /// falling). `false` when there is no active EVA player.
    pub fn is_grounded(&self) -> bool {
        self.active.map(|a| a.grounded).unwrap_or(false)
    }

    /// Whether the on-foot player has been stationary on the surface long
    /// enough to be warp-eligible. `false` when there is no active EVA player.
    pub fn is_at_rest(&self) -> bool {
        self.active.map(|a| a.at_rest).unwrap_or(false)
    }

    /// Surface-relative speed (m/s) of the on-foot player — walking + vertical,
    /// excluding the body's co-rotation. `0.0` when there is no active player.
    pub fn surface_speed_m_s(&self) -> f64 {
        self.active.map(|a| a.surface_speed_m_s).unwrap_or(0.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct ActivePlayerController {
    body_entity: Entity,
    visual_entity: Entity,
    body_id: BodyId,
    inertial_position_m: DVec3,
    /// Player position relative to the body centre, in the **body-fixed**
    /// (co-rotating) frame. `inertial_offset = body.orientation * pos_bf`.
    /// `ZERO` is the "uninitialised" sentinel — re-seeded from the rigid body
    /// on the first grounded frame and after any teleport.
    pos_bf: DVec3,
    /// Surface-relative velocity in the body-fixed frame (walking + vertical).
    vel_bf: DVec3,
    /// Horizontal facing direction in the body-fixed frame, slewed toward the
    /// movement direction for a smooth third-person pivot.
    facing_bf: DVec3,
    /// The body-centred inertial offset this controller last wrote to Avian's
    /// `Position`. Used to detect an *external* teleport (F9 drop / map plant)
    /// — `Position` changing to something the controller didn't write — without
    /// mistaking the body's normal per-frame co-rotation for one.
    last_avian_offset: DVec3,
    grounded: bool,
    at_rest: bool,
    rest_timer_s: f64,
    surface_speed_m_s: f64,
}

#[derive(Component)]
pub struct PlayerControllerBody;

#[derive(Component)]
pub struct PlayerControllerVisual;

/// Whether the EVA player is walking on terrain or coasting like a craft.
///
/// EVA is a full craft (KSP-style): it can stand on a surface or sit in
/// orbit. The two regimes need opposite state flow, so this flag picks one:
///
/// - `Grounded`: [`step_eva_controller`] owns the capsule pose, running the
///   body-fixed character physics, and the canonical→Avian snap stands down.
/// - `Airborne`: Kepler owns canonical translation and the snap drives the
///   capsule from canonical (exactly like a ship coasting in vacuum); the
///   character controller stands down.
///
/// Set explicitly by the EVA teleport actions — surface teleports ground it,
/// orbit teleports make it airborne. (Suborbital ballistic flight — jumping,
/// walking off a cliff — stays *within* the grounded regime; this flag is the
/// coarse surface↔orbit switch, not the per-frame grounded/airborne state,
/// which lives in [`ActivePlayerController::grounded`].)
/// Defaults to `Grounded` to match the startup surface spawn.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default, Reflect)]
#[reflect(Resource)]
pub enum EvaMode {
    #[default]
    Grounded,
    Airborne,
}

impl EvaMode {
    pub fn is_grounded(self) -> bool {
        matches!(self, EvaMode::Grounded)
    }
}

fn body_state_for(sim: &SimulationState, body_id: BodyId) -> BodyState {
    sim.ephemeris
        .state(body_id, Epoch(sim.simulation.sim_time()))
}

/// Attach the BigSpace visual mesh + register `PlayerControllerState`
/// once the EVA body has been spawned by `local_physics`. The body
/// itself is created in `local_physics::spawn_player_avian_body` for
/// `VesselKind::Eva`; this system just finds it and pairs the visual.
fn register_eva_visual(
    mut commands: Commands,
    mut state: ResMut<PlayerControllerState>,
    active_bubble: Res<ActiveLocalBubble>,
    sim: Res<SimulationState>,
    real_root: Option<Res<RealSpaceRoot>>,
    grid: Query<&Grid, With<BigSpace>>,
    body_q: Query<
        (Entity, &Position, &Rotation),
        (With<PlayerControllerBody>, With<LocalCraftBody>),
    >,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut focus: ResMut<CameraFocus>,
) {
    if state.active.is_some() {
        return;
    }
    let Some(bubble) = active_bubble.bubble.as_ref() else {
        return;
    };
    let Ok((body_entity, position, rotation)) = body_q.single() else {
        return;
    };
    let Some(real_root) = real_root.as_deref() else {
        return;
    };
    let Ok(root_grid) = grid.single() else {
        return;
    };

    let body_state = body_state_for(&sim, bubble.body_id);
    let inertial_position = body_state.position + position.0;
    let (cell, local) = root_grid.translation_to_grid(inertial_position);
    let mesh = meshes.add(Capsule3d::new(
        PLAYER_RADIUS_M as f32,
        PLAYER_CAPSULE_SEGMENT_M as f32,
    ));
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.84, 0.64),
        perceptual_roughness: 0.8,
        metallic: 0.0,
        ..default()
    });
    let visual_entity = commands
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_translation(local).with_rotation(rotation.0.as_quat()),
            Visibility::Inherited,
            cell,
            ChildOf(real_root.entity),
            HideInMapView,
            NoFrustumCulling,
            PlayerControllerVisual,
            Name::new("EVA player visual"),
        ))
        .id();

    state.active = Some(ActivePlayerController {
        body_entity,
        visual_entity,
        body_id: bubble.body_id,
        inertial_position_m: inertial_position,
        // ZERO sentinel: re-seeded from the rigid body on the first grounded
        // frame in `step_eva_controller`.
        pos_bf: DVec3::ZERO,
        vel_bf: DVec3::ZERO,
        facing_bf: DVec3::ZERO,
        last_avian_offset: position.0,
        grounded: true,
        at_rest: false,
        rest_timer_s: 0.0,
        surface_speed_m_s: 0.0,
    });

    focus.target = CameraFocusTarget::PlayerController;
    focus.target_distance = PLAYER_CAMERA_DISTANCE_M;
    focus.distance = focus.distance.min(PLAYER_CAMERA_DISTANCE_M * 2.0);
    focus.azimuth = std::f32::consts::PI;
    focus.elevation = 0.2;

    info!(
        "registered EVA visual for player controller on body {}",
        sim.system.bodies[bubble.body_id].name,
    );
}

fn refresh_player_controller_state(
    mut state: ResMut<PlayerControllerState>,
    sim: Res<SimulationState>,
    active_bubble: Res<ActiveLocalBubble>,
    bodies: Query<&Position, With<PlayerControllerBody>>,
) {
    let Some(mut active) = state.active else {
        return;
    };
    let Ok(position) = bodies.get(active.body_entity) else {
        state.active = None;
        return;
    };
    // Follow the bubble across SOI rebases and surface teleports so the
    // height query and body-state lookup track the player to a new body.
    if let Some(bubble) = active_bubble.bubble.as_ref() {
        active.body_id = bubble.body_id;
    }
    let body_state = body_state_for(&sim, active.body_id);
    active.inertial_position_m = body_state.position + position.0;
    state.active = Some(active);
}

/// EVA character controller — a kinematic capsule simulated in the body-fixed
/// (rotating) frame. See the module docs for the frame rationale.
///
/// The capsule is `RigidBody::Kinematic` with `CustomPositionIntegration`, so
/// this system owns `Position`/`Rotation`/`LinearVelocity` outright — no
/// dynamic contact resolution, no force-based gravity, no second-pass snap.
/// Each frame, in the body-fixed frame:
///
/// 1. Re-seed from the rigid body on first run / after a teleport.
/// 2. Under time warp, freeze the player at rest (warp is gated on rest) and
///    only re-derive the inertial offset from the body's current orientation.
/// 3. At 1×, run the grounded/airborne state machine:
///    - **Grounded:** move horizontally by the (camera-relative) walk input,
///      follow the terrain for steps up to `GROUND_SNAP_M`, walk off larger
///      drops into the airborne state, and launch on jump.
///    - **Airborne:** integrate radial gravity with limited air control, and
///      land when the capsule reaches the terrain.
/// 4. Convert the body-fixed pose back to the body-centred inertial offset
///    Avian holds, publish the surface-relative velocity for the canonical
///    readback, and update rest state for the warp gate + HUD.
#[allow(clippy::too_many_arguments)]
fn step_eva_controller(
    clock: Res<SimClock>,
    input: Res<GameInputIntent>,
    view: Res<ViewMode>,
    eva_mode: Res<EvaMode>,
    mut state: ResMut<PlayerControllerState>,
    mut sim: ResMut<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    camera: Query<&Transform, (With<ActiveCamera>, With<OrbitCamera>)>,
    mut bodies: Query<
        (
            &mut Position,
            &mut Rotation,
            &mut LinearVelocity,
            &mut AngularVelocity,
        ),
        With<PlayerControllerBody>,
    >,
) {
    // Airborne (orbiting) EVA coasts on rails — `snap_avian_from_canonical`
    // owns the capsule. Only the grounded controller runs character physics.
    if !eva_mode.is_grounded() {
        return;
    }
    let Some(mut active) = state.active else {
        return;
    };
    let Some(height_source) = height_sources.get(active.body_id) else {
        return;
    };
    let Ok((mut position, mut rotation, mut linear_velocity, mut angular_velocity)) =
        bodies.get_mut(active.body_entity)
    else {
        warn!(
            "step_eva_controller: query missed body {:?} — components likely stripped",
            active.body_entity
        );
        return;
    };

    let body = &sim.system.bodies[active.body_id];
    let body_state = body_state_for(&sim, active.body_id);
    let dt = clock.delta_secs_f64();

    // --- Re-seed from the rigid body on first run / after an external
    // teleport (F9 drop, map-cursor plant), which writes Avian's `Position`
    // directly and leaves `pos_bf` stale. Detect that by comparing against the
    // offset the controller itself last wrote — NOT against
    // `orientation * pos_bf`, which differs by the body's per-frame rotation
    // (several metres at planet scale) and would spuriously re-seed every
    // frame, pinning the player in inertial space so the surface slides out
    // from under it. ---
    let avian_offset = position.0;
    if active.pos_bf == DVec3::ZERO
        || (avian_offset - active.last_avian_offset).length() > TELEPORT_RESEED_M
    {
        active.pos_bf = body_state.orientation.inverse() * avian_offset;
        active.vel_bf = DVec3::ZERO;
        active.facing_bf = DVec3::ZERO;
        active.grounded = true;
    }

    // Body-fixed ground radius (centre → capsule centre when standing) at a
    // body-fixed direction. `None` only when the height source has no data and
    // no CPU fallback — callers then hold altitude rather than snapping.
    let target_radius = |dir: DVec3| -> Option<f64> {
        let dir = dir.try_normalize()?;
        let h = height_source.sample_height_m(dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)? as f64;
        Some(body.radius_m + h + PLAYER_HALF_HEIGHT_M + PLAYER_FOOT_CLEARANCE_M)
    };

    let up = active.pos_bf.try_normalize().unwrap_or(DVec3::Y);
    let warp_speed = sim.simulation.warp.speed();
    let warping = (warp_speed - 1.0).abs() > 1.0e-6;

    if warping {
        // Time warp is gated on the player being at rest (see
        // `bridge::enforce_warp_altitude_limits`), so under warp we hold the
        // body-fixed pose and let the analytic co-rotation
        // (`orientation(t) * pos_bf`) sweep the inertial offset. No
        // integration — `dt` here is the warp-scaled sim delta and must never
        // multiply a velocity. Gentle ground re-snap only, to track tiles
        // streaming in without ever producing a large jump.
        active.vel_bf = DVec3::ZERO;
        active.grounded = true;
        active.at_rest = true;
        active.rest_timer_s = REST_TIME_S;
        active.surface_speed_m_s = 0.0;
        if let Some(tr) = target_radius(active.pos_bf) {
            let alt = active.pos_bf.length();
            if (alt - tr).abs() < GROUND_SNAP_M {
                active.pos_bf = up * tr;
            }
        }
    } else {
        // --- 1× gameplay: full body-fixed character physics ---
        let inertial_up = (body_state.orientation * up)
            .try_normalize()
            .unwrap_or(DVec3::Y);
        let move_input = if *view == ViewMode::Ship {
            input.player_move
        } else {
            Vec2::ZERO
        };
        let moving = move_input.length_squared() > 1.0e-4;
        let speed = if input.player_sprint {
            PLAYER_RUN_SPEED_M_S
        } else {
            PLAYER_WALK_SPEED_M_S
        };
        let camera_transform = camera.single().ok();
        let walk_dir_inertial = movement_direction(move_input, inertial_up, camera_transform);
        // Walk direction in the body-fixed frame, re-projected to the tangent
        // plane at the local up so it carries no radial component.
        let walk_dir_bf = if walk_dir_inertial == DVec3::ZERO {
            DVec3::ZERO
        } else {
            let d = body_state.orientation.inverse() * walk_dir_inertial;
            (d - up * d.dot(up)).normalize_or_zero()
        };

        // Surface gravity at the current radius: g = μ / r².
        let r2 = active.pos_bf.length_squared().max(1.0);
        let gravity = body.gm / r2;

        if active.grounded {
            let horiz = walk_dir_bf * speed;
            if input.player_jump {
                // Launch: vertical speed for the target apex height, keeping
                // current horizontal motion.
                let jump_speed = (2.0 * gravity * PLAYER_JUMP_HEIGHT_M).sqrt();
                active.vel_bf = horiz + up * jump_speed;
                active.pos_bf += active.vel_bf * dt;
                active.grounded = false;
            } else {
                let new_pos = active.pos_bf + horiz * dt;
                match target_radius(new_pos) {
                    Some(tr) => {
                        let above = new_pos.length() - tr;
                        // Only treat a drop as "walked off a ledge" while
                        // actually moving — a standing player must never go
                        // airborne from frame-to-frame height-sample noise
                        // (GPU-tile vs CPU-fallback can disagree by metres as
                        // tiles stream), which would otherwise prevent
                        // `at_rest` from ever latching and block surface warp.
                        if moving && above > GROUND_SNAP_M {
                            // Walked off a ledge — start falling from here.
                            active.pos_bf = new_pos;
                            active.vel_bf = horiz;
                            active.grounded = false;
                        } else {
                            // Follow the terrain (step up/down, absorbing noise).
                            active.pos_bf = new_pos.normalize_or(up) * tr;
                            active.vel_bf = horiz;
                        }
                    }
                    None => {
                        // Terrain not resident: move horizontally, hold altitude.
                        active.pos_bf = new_pos;
                        active.vel_bf = horiz;
                    }
                }
            }
        } else {
            // Airborne: limited air control on the horizontal, gravity on the
            // radial, then integrate and test for landing.
            let mut v = active.vel_bf;
            let radial = v.dot(up);
            let cur_h = v - up * radial;
            let want_h = walk_dir_bf * speed;
            let blend = (AIR_CONTROL_PER_S * dt).clamp(0.0, 1.0);
            let new_h = cur_h + (want_h - cur_h) * blend;
            v = new_h + up * radial;
            v -= up * gravity * dt;
            let new_pos = active.pos_bf + v * dt;
            match target_radius(new_pos) {
                Some(tr) if new_pos.length() <= tr => {
                    // Landed: clamp to the surface, keep horizontal motion.
                    let dir = new_pos.normalize_or(up);
                    active.pos_bf = dir * tr;
                    active.vel_bf = v - dir * v.dot(dir);
                    active.grounded = true;
                }
                _ => {
                    active.pos_bf = new_pos;
                    active.vel_bf = v;
                }
            }
        }

        // Smoothly pivot the facing toward the movement direction.
        let new_up = active.pos_bf.try_normalize().unwrap_or(up);
        let desired_facing = if walk_dir_bf != DVec3::ZERO {
            (walk_dir_bf - new_up * walk_dir_bf.dot(new_up)).normalize_or_zero()
        } else {
            active.facing_bf
        };
        if active.facing_bf == DVec3::ZERO {
            active.facing_bf = if desired_facing != DVec3::ZERO {
                desired_facing
            } else {
                tangent_pair(new_up).1
            };
        } else if desired_facing != DVec3::ZERO {
            let t = (TURN_RATE_RAD_S * dt).clamp(0.0, 1.0);
            let blended = active.facing_bf.lerp(desired_facing, t);
            active.facing_bf =
                (blended - new_up * blended.dot(new_up)).normalize_or(desired_facing);
        }

        // Rest detection for the warp gate + HUD.
        active.surface_speed_m_s = active.vel_bf.length();
        let resting_now = active.grounded
            && !moving
            && !input.player_jump
            && active.surface_speed_m_s < REST_SPEED_EPS_M_S;
        if resting_now {
            active.rest_timer_s += dt;
        } else {
            active.rest_timer_s = 0.0;
        }
        active.at_rest = active.rest_timer_s >= REST_TIME_S;
    }

    // --- Publish to the rigid body (body-centred inertial offset + velocity)
    // and orient the capsule. ---
    let final_up = active.pos_bf.try_normalize().unwrap_or(up);
    let new_position = body_state.orientation * active.pos_bf;
    position.0 = new_position;
    // Remember what we wrote so next frame can tell our own co-rotation from an
    // external teleport (see the re-seed guard above).
    active.last_avian_offset = new_position;

    // Inertial velocity = body co-rotation drag (ω×r) + surface-relative
    // velocity rotated into inertial axes. `readback_local_craft` converts
    // this back to canonical so HUD orbital readings stay truthful.
    linear_velocity.0 =
        body_state.angular_velocity.cross(new_position) + body_state.orientation * active.vel_bf;
    angular_velocity.0 = DVec3::ZERO;

    let inertial_up = (body_state.orientation * final_up)
        .try_normalize()
        .unwrap_or(DVec3::Y);
    let inertial_forward = if active.facing_bf != DVec3::ZERO {
        body_state.orientation * active.facing_bf
    } else {
        rotation.0 * DVec3::Z
    };
    rotation.0 = level_orientation(inertial_up, inertial_forward);

    // Grounded EVA is conceptually body-fixed while standing on the surface,
    // even though we keep `AuthorityMode::LocalRigidBody` so the player can
    // resume walking at 1× without rebuilding authority. Publish the analytic
    // body-fixed pose to canonical immediately after the controller writes the
    // capsule so every downstream system in this frame (camera focus, BRP,
    // HUD/map snapshots, and the next warp tick) sees a craft stationary on
    // the rotating planet instead of a one-frame-old inertial position.
    sim.simulation.install_local_rigid_body_state(
        TranslationalState {
            position: body_state.position + position.0,
            velocity: body_state.velocity + linear_velocity.0,
        },
        AttitudeState {
            orientation: rotation.0,
            angular_velocity: rotation.0.inverse() * angular_velocity.0,
        },
    );

    state.active = Some(active);
}

fn sync_player_controller_visual(
    state: Res<PlayerControllerState>,
    sim: Res<SimulationState>,
    body_states: Res<SolarSystemState>,
    grid: Query<&Grid, With<BigSpace>>,
    bodies: Query<(&Position, &Rotation), With<PlayerControllerBody>>,
    mut visuals: Query<(&mut CellCoord, &mut Transform), With<PlayerControllerVisual>>,
) {
    let Some(active) = state.active else {
        return;
    };
    let Ok(root_grid) = grid.single() else {
        return;
    };
    let Ok((position, rotation)) = bodies.get(active.body_entity) else {
        return;
    };
    let Ok((mut cell, mut transform)) = visuals.get_mut(active.visual_entity) else {
        return;
    };

    let body_state = body_states
        .states
        .as_deref()
        .and_then(|states| states.get(active.body_id))
        .cloned()
        .unwrap_or_else(|| body_state_for(&sim, active.body_id));
    let inertial_position = body_state.position + position.0;
    let (next_cell, local) = root_grid.translation_to_grid(inertial_position);
    *cell = next_cell;
    transform.translation = local;
    transform.rotation = rotation.0.as_quat();
}

fn sync_player_controller_camera_focus(
    state: Res<PlayerControllerState>,
    view: Res<ViewMode>,
    mut focus: ResMut<CameraFocus>,
) {
    if *view != ViewMode::Ship {
        return;
    }
    if state.is_active() {
        if focus.target != CameraFocusTarget::PlayerController {
            focus.target = CameraFocusTarget::PlayerController;
            focus.target_distance = focus.target_distance.min(PLAYER_CAMERA_DISTANCE_M);
        }
    } else if focus.target == CameraFocusTarget::PlayerController {
        focus.target = CameraFocusTarget::Ship;
    }
}

fn movement_direction(input: Vec2, up: DVec3, camera: Option<&Transform>) -> DVec3 {
    if input.length_squared() <= f32::EPSILON {
        return DVec3::ZERO;
    }
    let (right, forward) = camera
        .and_then(|camera| {
            let forward_seed = (camera.rotation * Vec3::NEG_Z).as_dvec3();
            let right_seed = (camera.rotation * Vec3::X).as_dvec3();
            let forward = project_tangent(forward_seed, up)?;
            let right = project_tangent(right_seed, up).unwrap_or_else(|| up.cross(forward));
            Some((right.normalize(), forward.normalize()))
        })
        .unwrap_or_else(|| tangent_pair(up));

    (right * input.x as f64 + forward * input.y as f64).normalize_or_zero()
}

fn level_orientation(up: DVec3, forward_seed: DVec3) -> DQuat {
    let up = up.normalize_or_zero();
    if up == DVec3::ZERO {
        return DQuat::IDENTITY;
    }
    let forward = project_tangent(forward_seed, up).unwrap_or_else(|| tangent_pair(up).1);
    let right = up.cross(forward).normalize_or_zero();
    if right == DVec3::ZERO {
        return DQuat::IDENTITY;
    }
    let forward = right.cross(up).normalize_or_zero();
    DQuat::from_mat3(&DMat3::from_cols(right, up, forward))
}

fn project_tangent(v: DVec3, up: DVec3) -> Option<DVec3> {
    let tangent = v - up * v.dot(up);
    (tangent.length_squared() > 1.0e-8).then(|| tangent.normalize())
}

fn tangent_pair(up: DVec3) -> (DVec3, DVec3) {
    let seed = if up.y.abs() < 0.9 { DVec3::Y } else { DVec3::X };
    let forward = project_tangent(seed, up).unwrap_or(DVec3::Z);
    let right = up.cross(forward).normalize_or_zero();
    (right, forward)
}
