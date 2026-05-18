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
//! This module owns:
//! - the visible mesh (`PlayerControllerVisual`), a BigSpace entity
//!   synced from physics each frame so render transforms stay in
//!   metre-scale cells instead of inheriting body-centred-inertial
//!   coordinates;
//! - walking-velocity targeting and gravity (the only force writer for
//!   EVA — `apply_local_forces` early-returns for `VesselKind::Eva`);
//! - terrain snap and camera focus.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::canonical::Epoch;
use thalos_physics_canonical::types::{BodyId, BodyState};
use thalos_physics_local::avian::{AngularVelocity, LinearVelocity, Position, Rotation};
use thalos_physics_local::{ActiveLocalBubble, HeightSourceRegistry, LocalCraftBody};

use crate::SimStage;
use crate::camera::{ActiveCamera, CameraFocus, CameraFocusTarget, OrbitCamera};
use crate::local_physics::PHYSICS_QUERY_TILE_LOD_M;
use crate::rendering::real_space::RealSpaceRoot;
use crate::rendering::{SimulationState, SolarSystemState};
use crate::view::{HideInMapView, ViewMode};

const PLAYER_HEIGHT_M: f64 = 1.8;
const PLAYER_RADIUS_M: f64 = 0.32;
const PLAYER_CAPSULE_SEGMENT_M: f64 = PLAYER_HEIGHT_M - PLAYER_RADIUS_M * 2.0;
const PLAYER_HALF_HEIGHT_M: f64 = PLAYER_HEIGHT_M * 0.5;
const PLAYER_FOOT_CLEARANCE_M: f64 = 0.08;
const PLAYER_WALK_SPEED_M_S: f64 = 1.4;
const PLAYER_CAMERA_DISTANCE_M: f64 = 6.0;

pub struct PlayerControllerPlugin;

impl Plugin for PlayerControllerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlayerControllerState>()
            .add_systems(
                Update,
                (
                    register_eva_visual,
                    walk_eva_on_terrain,
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
}

#[derive(Debug, Clone, Copy)]
struct ActivePlayerController {
    body_entity: Entity,
    visual_entity: Entity,
    body_id: BodyId,
    inertial_position_m: DVec3,
}

#[derive(Component)]
pub struct PlayerControllerBody;

#[derive(Component)]
pub struct PlayerControllerVisual;

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
    bodies: Query<&Position, With<PlayerControllerBody>>,
) {
    let Some(mut active) = state.active else {
        return;
    };
    let Ok(position) = bodies.get(active.body_entity) else {
        state.active = None;
        return;
    };
    let body_state = body_state_for(&sim, active.body_id);
    active.inertial_position_m = body_state.position + position.0;
    state.active = Some(active);
}

/// Pure terrain-following EVA movement.
///
/// The capsule is kinematic with `CustomPositionIntegration`, so this
/// system owns `Position`/`Rotation`/`LinearVelocity` outright — no
/// dynamic contact resolution, no force-based gravity, no second-pass
/// snap. Each frame we:
///
/// 1. Translate walking input into a tangent direction at the current
///    "up" (radial-out from the body's centre);
/// 2. Step the position forward by `surface_velocity + walking * dt`
///    in body-centred inertial coordinates;
/// 3. Convert the resulting direction to body-fixed and read the
///    rendered terrain height there;
/// 4. Glue the position altitude to `body.radius + terrain_h +
///    capsule_half_height + foot_clearance` so the player follows the
///    terrain exactly, with no clipping or wobble;
/// 5. Refresh `LinearVelocity` (for `readback_local_craft` →
///    canonical) and orient the capsule upright facing the walk
///    direction.
///
/// Always-glued is intentional for this iteration: walking off a cliff
/// teleports down the drop instead of arcing through the air. Realistic
/// for an arcade-y first pass; the ballistic path (grounded marker +
/// `Grounded`-component-style detection + integrated gravity when
/// airborne) is the natural follow-up when jumping or surface-to-orbit
/// transitions need it.
#[allow(clippy::too_many_arguments)]
fn walk_eva_on_terrain(
    time: Res<Time>,
    input: Res<GameInputIntent>,
    view: Res<ViewMode>,
    state: Res<PlayerControllerState>,
    sim: Res<SimulationState>,
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
    let Some(active) = state.active else {
        return;
    };
    let Some(height_source) = height_sources.get(active.body_id) else {
        return;
    };
    let Ok((mut position, mut rotation, mut linear_velocity, mut angular_velocity)) =
        bodies.get_mut(active.body_entity)
    else {
        warn!(
            "walk_eva_on_terrain: query missed body {:?} — components likely stripped",
            active.body_entity
        );
        return;
    };
    let Some(current_up) = position.0.try_normalize() else {
        return;
    };

    let body = &sim.system.bodies[active.body_id];
    let body_state = body_state_for(&sim, active.body_id);
    // Two distinct time steps drive position changes:
    //
    // * **Co-rotation with the surface** uses **sim time**. The body's
    //   `orientation` only evolves with sim time, so stepping co-rotation
    //   by anything else would drift the player across a frozen world.
    // * **Walking** uses **real time**, so at warp > 1× the player still
    //   walks at a human pace even though the world is spinning faster.
    //
    // Pause handling is structural: `pause_menu::sync_virtual_time_pause`
    // pauses `Time<Virtual>` whenever the player or warp asks for pause,
    // so `real_dt` is zero and both terms below collapse to zero. No
    // explicit per-system guard.
    let real_dt = time.delta_secs_f64();
    let sim_dt = real_dt * sim.simulation.warp.speed();

    let move_input = if *view == ViewMode::Ship {
        input.player_move
    } else {
        Vec2::ZERO
    };
    let camera_transform = camera.single().ok();
    let walk_dir = movement_direction(move_input, current_up, camera_transform);
    let surface_velocity = body_state.angular_velocity.cross(position.0);
    let walking_velocity = walk_dir * PLAYER_WALK_SPEED_M_S;

    // Step in inertial coords, then look up terrain at the new direction.
    // Doing the height query at the *new* direction (not the current
    // one) means the player crests bumps smoothly instead of skipping
    // upward a frame late.
    let stepped = position.0 + surface_velocity * sim_dt + walking_velocity * real_dt;
    let new_dir_inertial = stepped.try_normalize().unwrap_or(current_up);
    let new_dir_body_fixed = (body_state.orientation.inverse() * new_dir_inertial).normalize();
    let terrain_h = height_source
        .sample_height_m(new_dir_body_fixed.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
        .unwrap_or(0.0) as f64;
    let target_altitude =
        body.radius_m + terrain_h + PLAYER_HALF_HEIGHT_M + PLAYER_FOOT_CLEARANCE_M;
    let new_position = body_state.orientation * new_dir_body_fixed * target_altitude;
    position.0 = new_position;

    // Propagate the actual horizontal+co-rotation velocity for the
    // canonical readback chain so HUD readings (orbital velocity etc.)
    // remain truthful.
    linear_velocity.0 = surface_velocity + walking_velocity;

    // Kinematic body with locked rotation; clear residual angular
    // velocity so it can't leak into rotation. Acceleration
    // accumulators are irrelevant for a kinematic body with
    // `CustomPositionIntegration` (no integrator step touches them).
    angular_velocity.0 = DVec3::ZERO;

    let new_up = new_position.normalize_or(current_up);
    let forward_seed = walk_dir
        .try_normalize()
        .unwrap_or_else(|| rotation.0 * DVec3::Z);
    rotation.0 = level_orientation(new_up, forward_seed);
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
