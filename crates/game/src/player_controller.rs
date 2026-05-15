//! EVA-style player controller for walking on rendered terrain.
//!
//! The controller is a local Avian capsule that lives in the same
//! body-centered inertial frame as the ship's aggregate rigid body. Its
//! visible mesh is a separate BigSpace entity, synced from physics each frame
//! so render transforms stay in metre-scale cells instead of inheriting the
//! multi-megametre physics coordinates.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_input::game::GameInputIntent;
use thalos_local_physics::avian::{
    AngularVelocity, CoefficientCombine, Collider, ConstantAngularAcceleration,
    ConstantLinearAcceleration, Friction, LinearVelocity, LockedAxes, Mass, Position, Restitution,
    RigidBody, Rotation, SleepingDisabled,
};
use thalos_local_physics::{ActiveLocalBubble, TerrainSurfaceRegistry};
use thalos_physics::canonical::Epoch;
use thalos_physics::types::{BodyId, BodyState};
use thalos_terrain::rendered_height_m;

use crate::SimStage;
use crate::camera::{ActiveCamera, CameraFocus, CameraFocusTarget, OrbitCamera};
use crate::rendering::real_space::RealSpaceRoot;
use crate::rendering::{SimulationState, SolarSystemState};
use crate::view::{HideInMapView, ViewMode};

const PLAYER_HEIGHT_M: f64 = 1.8;
const PLAYER_RADIUS_M: f64 = 0.32;
const PLAYER_CAPSULE_SEGMENT_M: f64 = PLAYER_HEIGHT_M - PLAYER_RADIUS_M * 2.0;
const PLAYER_HALF_HEIGHT_M: f64 = PLAYER_HEIGHT_M * 0.5;
const PLAYER_FOOT_CLEARANCE_M: f64 = 0.08;
const PLAYER_MASS_KG: f32 = 90.0;
const PLAYER_WALK_SPEED_M_S: f64 = 1.4;
const PLAYER_VELOCITY_RESPONSE: f64 = 14.0;
const PLAYER_CAMERA_DISTANCE_M: f64 = 6.0;
const PLAYER_EXIT_SIDE_OFFSET_M: f64 = 2.0;
const PLAYER_GROUND_SNAP_M: f64 = 0.75;

pub struct PlayerControllerPlugin;

impl Plugin for PlayerControllerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlayerControllerState>()
            .add_systems(
                Update,
                (
                    toggle_player_controller,
                    apply_player_controller_motion,
                    constrain_player_to_terrain,
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
                    .after(crate::rendering::cache_body_states),
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

fn toggle_player_controller(
    mut commands: Commands,
    input: Res<GameInputIntent>,
    view: Res<ViewMode>,
    mut state: ResMut<PlayerControllerState>,
    active_bubble: Res<ActiveLocalBubble>,
    surfaces: Res<TerrainSurfaceRegistry>,
    sim: Res<SimulationState>,
    real_root: Option<Res<RealSpaceRoot>>,
    grid: Query<&Grid, With<BigSpace>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut focus: ResMut<CameraFocus>,
) {
    if !input.toggle_player_controller {
        return;
    }
    if *view != ViewMode::Ship {
        return;
    }

    if let Some(active) = state.active.take() {
        commands.entity(active.body_entity).despawn();
        commands.entity(active.visual_entity).despawn();
        focus.target = CameraFocusTarget::Ship;
        focus.target_distance = focus.target_distance.max(30.0);
        info!("boarded ship; EVA player controller despawned");
        return;
    }

    let Some(bubble) = active_bubble.bubble.as_ref() else {
        warn!("cannot spawn EVA controller before the local physics bubble exists");
        return;
    };
    let Some(terrain_entity) = bubble.terrain_entity else {
        warn!("cannot spawn EVA controller until a terrain collider patch is attached");
        return;
    };
    let Some(surface) = surfaces.get(bubble.body_id) else {
        warn!("cannot spawn EVA controller before the body's terrain surface is available");
        return;
    };
    let Some(real_root) = real_root.as_deref() else {
        warn!("cannot spawn EVA controller before BigSpace is ready");
        return;
    };
    let Ok(root_grid) = grid.single() else {
        warn!("cannot spawn EVA controller without a BigSpace grid");
        return;
    };

    let body = &sim.system.bodies[bubble.body_id];
    let body_state = body_state_for(&sim, bubble.body_id);
    let craft = sim.simulation.craft_state();
    let ship_body_centered = craft.translation.position - body_state.position;
    let up = ship_body_centered.normalize_or_zero();
    if up == DVec3::ZERO {
        warn!("cannot spawn EVA controller at the dominant body's centre");
        return;
    }

    let side_seed = sim.simulation.attitude().orientation * DVec3::X;
    let side = project_tangent(side_seed, up).unwrap_or_else(|| tangent_pair(up).0);
    let candidate_body_centered = ship_body_centered + side * PLAYER_EXIT_SIDE_OFFSET_M;
    let dir_body_fixed =
        (body_state.orientation.inverse() * candidate_body_centered).normalize_or_zero();
    if dir_body_fixed == DVec3::ZERO {
        warn!("cannot spawn EVA controller from a zero-length surface direction");
        return;
    }

    let terrain_height_m = rendered_height_m(&surface.static_surface, dir_body_fixed.as_vec3());
    let spawn_body_centered = body_state.orientation
        * dir_body_fixed
        * (body.radius_m
            + terrain_height_m as f64
            + PLAYER_HALF_HEIGHT_M
            + PLAYER_FOOT_CLEARANCE_M);
    let spawn_up = spawn_body_centered.normalize_or_zero();
    let forward_seed = sim.simulation.attitude().orientation * DVec3::Y;
    let spawn_rotation = level_orientation(spawn_up, forward_seed);
    let surface_velocity = body_state.angular_velocity.cross(spawn_body_centered);
    let inertial_position = body_state.position + spawn_body_centered;

    let body_entity = commands
        .spawn((
            RigidBody::Dynamic,
            Collider::capsule(PLAYER_RADIUS_M, PLAYER_CAPSULE_SEGMENT_M),
            Position(spawn_body_centered),
            Rotation(spawn_rotation),
            LinearVelocity(surface_velocity),
            AngularVelocity(DVec3::ZERO),
            ConstantLinearAcceleration(DVec3::ZERO),
            ConstantAngularAcceleration(DVec3::ZERO),
            Mass(PLAYER_MASS_KG),
            LockedAxes::ROTATION_LOCKED,
            Friction::new(0.9).with_combine_rule(CoefficientCombine::Max),
            Restitution::ZERO.with_combine_rule(CoefficientCombine::Min),
            SleepingDisabled,
            PlayerControllerBody,
            Name::new("EVA player controller body"),
        ))
        .id();

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
            Transform::from_translation(local).with_rotation(spawn_rotation.as_quat()),
            Visibility::Inherited,
            cell,
            ChildOf(real_root.entity),
            HideInMapView,
            NoFrustumCulling,
            PlayerControllerVisual,
            Name::new("EVA player controller"),
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
        "spawned EVA player controller on {} using terrain patch {:?}",
        body.name, terrain_entity
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

fn apply_player_controller_motion(
    time: Res<Time>,
    input: Res<GameInputIntent>,
    view: Res<ViewMode>,
    state: Res<PlayerControllerState>,
    sim: Res<SimulationState>,
    camera: Query<&Transform, (With<ActiveCamera>, With<OrbitCamera>)>,
    mut bodies: Query<
        (
            &mut Position,
            &mut Rotation,
            &mut LinearVelocity,
            &mut AngularVelocity,
            &mut ConstantLinearAcceleration,
            &mut ConstantAngularAcceleration,
        ),
        With<PlayerControllerBody>,
    >,
) {
    let Some(active) = state.active else {
        return;
    };
    let Ok((
        position,
        mut rotation,
        mut linear_velocity,
        mut angular_velocity,
        mut linear_acceleration,
        mut angular_acceleration,
    )) = bodies.get_mut(active.body_entity)
    else {
        return;
    };

    let radius_sq = position.0.length_squared();
    if radius_sq <= f64::EPSILON {
        linear_acceleration.0 = DVec3::ZERO;
        return;
    }

    let body = &sim.system.bodies[active.body_id];
    let body_state = body_state_for(&sim, active.body_id);
    let up = position.0.normalize();
    let gravity = -body.gm * position.0 / radius_sq.sqrt().powi(3);
    linear_acceleration.0 = gravity;
    angular_acceleration.0 = DVec3::ZERO;
    angular_velocity.0 = DVec3::ZERO;

    let move_input = if *view == ViewMode::Ship {
        input.player_move
    } else {
        Vec2::ZERO
    };
    let camera_transform = camera.single().ok();
    let desired_direction = movement_direction(move_input, up, camera_transform);
    let surface_velocity = body_state.angular_velocity.cross(position.0);
    let relative_velocity = linear_velocity.0 - surface_velocity;
    let vertical_velocity = up * relative_velocity.dot(up);
    let target_velocity =
        surface_velocity + desired_direction * PLAYER_WALK_SPEED_M_S + vertical_velocity;

    let response =
        (1.0 - (-PLAYER_VELOCITY_RESPONSE * time.delta_secs_f64()).exp()).clamp(0.0, 1.0);
    linear_velocity.0 = linear_velocity.0.lerp(target_velocity, response);

    let forward_seed = desired_direction
        .try_normalize()
        .unwrap_or_else(|| rotation.0 * DVec3::Z);
    rotation.0 = level_orientation(up, forward_seed);
}

fn constrain_player_to_terrain(
    surfaces: Res<TerrainSurfaceRegistry>,
    state: Res<PlayerControllerState>,
    sim: Res<SimulationState>,
    mut bodies: Query<
        (
            &mut Position,
            &mut Rotation,
            &mut LinearVelocity,
            &mut ConstantLinearAcceleration,
        ),
        With<PlayerControllerBody>,
    >,
) {
    let Some(active) = state.active else {
        return;
    };
    let Some(surface) = surfaces.get(active.body_id) else {
        return;
    };
    let Ok((mut position, mut rotation, mut linear_velocity, mut linear_acceleration)) =
        bodies.get_mut(active.body_entity)
    else {
        return;
    };

    let body = &sim.system.bodies[active.body_id];
    let body_state = body_state_for(&sim, active.body_id);
    let Some((up, floor_center_radius_m)) =
        player_floor_center_radius(body, &body_state, &surface.static_surface, position.0)
    else {
        return;
    };

    let clearance_m = position.0.length() - floor_center_radius_m;
    if clearance_m > PLAYER_GROUND_SNAP_M {
        return;
    }

    position.0 = up * floor_center_radius_m;
    let surface_velocity = body_state.angular_velocity.cross(position.0);
    let relative_velocity = linear_velocity.0 - surface_velocity;
    let tangent_velocity = relative_velocity - up * relative_velocity.dot(up);
    linear_velocity.0 = surface_velocity + tangent_velocity;
    linear_acceleration.0 = DVec3::ZERO;

    let forward_seed = rotation.0 * DVec3::Z;
    rotation.0 = level_orientation(up, forward_seed);
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

fn player_floor_center_radius(
    body: &thalos_physics::types::BodyDefinition,
    body_state: &BodyState,
    surface: &thalos_terrain_gen::StaticSurfaceData,
    position_body_centered_m: DVec3,
) -> Option<(DVec3, f64)> {
    let up = position_body_centered_m.try_normalize()?;
    let dir_body_fixed = (body_state.orientation.inverse() * up)
        .as_vec3()
        .normalize_or_zero();
    if dir_body_fixed == Vec3::ZERO {
        return None;
    }
    let height_m = rendered_height_m(surface, dir_body_fixed) as f64;
    Some((
        up,
        body.radius_m + height_m + PLAYER_HALF_HEIGHT_M + PLAYER_FOOT_CLEARANCE_M,
    ))
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
