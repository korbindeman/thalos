//! Debug utilities. Hardcoded on for now; later this becomes an
//! in-game settings toggle.

use bevy::gizmos::prelude::{GizmoConfigGroup, GizmoConfigStore, GizmoPrimitive3d};
use bevy::math::primitives::{Capsule3d, Cone, Cuboid, Cylinder, Sphere};
use bevy::math::{DMat3, DQuat, DVec3, Isometry3d, Quat, Vec3};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::EguiContexts;
use thalos_body_render::rendered_height_m;
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::{
    body_fixed::body_fixed_pose_from_inertial,
    body_fixed::body_fixed_surface_velocity,
    canonical::{AuthorityMode, BodyFixedPose, TranslationalState},
    debug_orbits::debug_parking_orbit_state,
    types::{AttitudeState, BodyState, VesselKind},
};
use thalos_physics_local::avian::{AngularVelocity, LinearVelocity, Position, Rotation};
use thalos_physics_local::{
    ActiveLocalBubble, LocalCraftBody, LocalCraftColliderPrimitives, LocalPrimitiveCollider,
    LocalPrimitiveShape, TerrainSurfaceRegistry,
};
use thalos_world::{BodyDefinition, BodyId, BodyKind, StateVector};

use crate::camera::{ActiveCamera, MapCamera};
use crate::coords::{MAP_SCALE, SHIP_LAYER};
use crate::fuel::ThrottleState;
use crate::local_physics::place_eva_on_surface;
use crate::maneuver::{ManeuverPlan, SelectedNode};
use crate::navigation::SHIP_NOSE_BODY;
use crate::pause_menu::not_game_paused;
use crate::photo_mode::not_in_photo_mode;
use crate::player_controller::EvaMode;
use crate::rendering::{CelestialBody, PlayerShip, SimulationState, SolarSystemState};
use crate::target::TargetBody;
use crate::view::{ViewMode, in_map_view};

/// Debug surface drops park the craft slightly above terrain, then hold it in
/// body-fixed authority until the player throttles up.
pub const DEBUG_LAUNCH_MOUNT_HEIGHT_M: f64 = 18.0;

/// EVA surface teleports plant the capsule a couple of metres above the
/// rendered terrain; `step_eva_controller` re-seeds and snaps it onto the
/// surface on the next frame, so this is just a safe initial clearance.
const EVA_SURFACE_CLEARANCE_M: f64 = 2.0;
const DEBUG_CRAFT_COLLIDERS_KEY: KeyCode = KeyCode::F8;

#[derive(Resource, Debug, Clone, Copy)]
pub struct DebugMode {
    pub enabled: bool,
    pub show_craft_colliders: bool,
}

/// Temporary debug-only launch clamp used by command-shift body-tree surface
/// spawns. It keeps the craft in a stable body-fixed pose above terrain until
/// the player applies throttle, at which point game-side local physics releases
/// it. Remove this once real staging/launch clamps exist.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct DebugLaunchMount {
    pub active: Option<DebugLaunchMountState>,
}

#[derive(Debug, Clone, Copy)]
pub struct DebugLaunchMountState {
    pub body_id: BodyId,
    pub pose: BodyFixedPose,
}

/// Explicit map-view debug teleport mode.
///
/// A body-tree `drop` button arms this resource. While armed, the map cursor
/// raycasts against that body's visible disc, draws a small cursor on the
/// corresponding terrain direction, and left-clicking mounts the craft at the
/// rendered height under the cursor.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct DebugSurfaceTeleport {
    pub armed_body: Option<BodyId>,
    hover: Option<DebugSurfaceTeleportHit>,
}

impl DebugSurfaceTeleport {
    pub fn arm(&mut self, body_id: BodyId) {
        self.armed_body = Some(body_id);
        self.hover = None;
    }

    pub fn cancel(&mut self) {
        *self = Self::default();
    }
}

#[derive(Debug, Clone, Copy)]
struct DebugSurfaceTeleportHit {
    body_id: BodyId,
    dir_body: DVec3,
    surface_height_m: f64,
    render_pos: Vec3,
    normal_render: Vec3,
    used_rendered_surface: bool,
}

pub struct DebugPlugin;

impl Plugin for DebugPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DebugMode {
            enabled: true,
            show_craft_colliders: false,
        })
        .init_gizmo_group::<CraftColliderGizmos>()
        .init_resource::<DebugLaunchMount>()
        .init_resource::<DebugSurfaceTeleport>()
        .add_systems(Startup, configure_craft_collider_gizmos)
        .add_systems(
            Update,
            (
                toggle_debug_craft_colliders.run_if(not_game_paused.and(not_in_photo_mode)),
                draw_debug_craft_colliders
                    .run_if(not_game_paused.and(not_in_photo_mode))
                    .after(crate::SimStage::Camera),
            ),
        )
        .add_systems(
            Update,
            (
                update_debug_surface_teleport_cursor,
                commit_debug_surface_teleport.after(update_debug_surface_teleport_cursor),
            )
                .run_if(not_game_paused.and(not_in_photo_mode).and(in_map_view))
                .after(crate::SimStage::Camera),
        );
    }
}

#[derive(Default, Reflect, GizmoConfigGroup)]
#[reflect(Default)]
struct CraftColliderGizmos;

fn configure_craft_collider_gizmos(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<CraftColliderGizmos>();
    config.line.width = 4.0;
    config.depth_bias = -1.0;
    config.render_layers = bevy::camera::visibility::RenderLayers::layer(SHIP_LAYER);
}

fn toggle_debug_craft_colliders(keys: Res<ButtonInput<KeyCode>>, mut debug: ResMut<DebugMode>) {
    if !debug.enabled || !keys.just_pressed(DEBUG_CRAFT_COLLIDERS_KEY) {
        return;
    }
    debug.show_craft_colliders = !debug.show_craft_colliders;
    let state = if debug.show_craft_colliders {
        "enabled"
    } else {
        "disabled"
    };
    info!("craft collider debug {state}");
}

fn draw_debug_craft_colliders(
    debug: Res<DebugMode>,
    view: Res<ViewMode>,
    active: Res<ActiveLocalBubble>,
    ship_q: Query<&GlobalTransform, With<PlayerShip>>,
    craft_q: Query<&LocalCraftColliderPrimitives, With<LocalCraftBody>>,
    mut gizmos: Gizmos<CraftColliderGizmos>,
) {
    if !debug.enabled || !debug.show_craft_colliders || *view != ViewMode::Ship {
        return;
    }
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Ok(primitives) = craft_q.get(bubble.craft_entity) else {
        return;
    };
    let Ok(root) = ship_q.single() else {
        return;
    };
    let (_, root_rotation, root_translation) = root.affine().to_scale_rotation_translation();
    let color = Color::srgba(0.0, 1.0, 0.75, 0.9);

    for primitive in &primitives.0 {
        draw_collider_primitive(
            &mut gizmos,
            root_translation,
            root_rotation,
            primitive,
            color,
        );
    }
}

fn draw_collider_primitive(
    gizmos: &mut Gizmos<CraftColliderGizmos>,
    root_translation: Vec3,
    root_rotation: Quat,
    primitive: &LocalPrimitiveCollider,
    color: Color,
) {
    let offset = primitive.offset_m.as_vec3();
    let rotation = root_rotation * primitive.rotation.as_quat();
    let isometry = Isometry3d::new(root_translation + root_rotation * offset, rotation);
    match primitive.shape {
        LocalPrimitiveShape::Cuboid { x, y, z } => {
            gizmos.primitive_3d(&Cuboid::new(x as f32, y as f32, z as f32), isometry, color);
        }
        LocalPrimitiveShape::Cylinder { radius, height } => {
            gizmos
                .primitive_3d(
                    &Cylinder::new(radius as f32, height as f32),
                    isometry,
                    color,
                )
                .resolution(24);
        }
        LocalPrimitiveShape::Cone { radius, height } => {
            gizmos
                .primitive_3d(&Cone::new(radius as f32, height as f32), isometry, color)
                .resolution(24);
        }
        LocalPrimitiveShape::Sphere { radius } => {
            gizmos
                .primitive_3d(&Sphere::new(radius as f32), isometry, color)
                .resolution(24);
        }
        LocalPrimitiveShape::Capsule { radius, length } => {
            gizmos
                .primitive_3d(
                    &Capsule3d::new(radius as f32, length as f32),
                    isometry,
                    color,
                )
                .resolution(24);
        }
    }
}

/// Compute a near-circular low-orbit state vector around `body` at the given
/// `body_state` (the body's heliocentric state at the current sim_time).
///
/// Uses the same 200 km debug parking-orbit helper as initial ship spawn,
/// capped so small-body teleports stay inside the body's SOI.
///
/// Returns the heliocentric state plus a body→world attitude that points
/// the ship's nose along its prograde velocity.
pub fn low_orbit_state(
    body: &BodyDefinition,
    body_state: &BodyState,
) -> (StateVector, AttitudeState) {
    let state = debug_parking_orbit_state(body, body_state);
    let rel_vel = state.velocity - body_state.velocity;
    let attitude = AttitudeState {
        orientation: DQuat::from_rotation_arc(SHIP_NOSE_BODY, rel_vel.normalize()),
        angular_velocity: DVec3::ZERO,
    };
    (state, attitude)
}

/// Compute a surface-aligned debug spawn state for `body`.
///
/// `dir_body` is a body-fixed unit direction, and `surface_height_m` is the
/// rendered-terrain height at that direction. The returned craft is stationary
/// relative to the rotating surface and upright for a rocket launch: ship
/// nose +Y points along local up, with roll chosen from the local east tangent
/// when the body has a spin axis.
pub fn surface_spawn_state(
    body: &BodyDefinition,
    body_state: &BodyState,
    dir_body: DVec3,
    surface_height_m: f64,
    clearance_m: f64,
) -> (StateVector, AttitudeState) {
    let up_body = dir_body.normalize();
    let position_body = up_body * (body.radius_m + surface_height_m + clearance_m);
    let state = StateVector {
        position: body_state.position + body_state.orientation * position_body,
        velocity: body_fixed_surface_velocity(body_state, position_body),
    };
    let attitude = AttitudeState {
        orientation: level_surface_attitude(body_state, up_body),
        angular_velocity: DVec3::ZERO,
    };
    (state, attitude)
}

fn level_surface_attitude(body_state: &BodyState, up_body: DVec3) -> DQuat {
    let nose_body = up_body.normalize();
    let spin_body = body_state.orientation.inverse() * body_state.angular_velocity;
    let mut dorsal_body = spin_body.cross(nose_body);
    if dorsal_body.length_squared() < 1.0e-18 {
        let reference = if nose_body.y.abs() < 0.9 {
            DVec3::Y
        } else {
            DVec3::X
        };
        dorsal_body = (reference - nose_body * reference.dot(nose_body)).normalize();
    } else {
        dorsal_body = dorsal_body.normalize();
    }
    let right_body = nose_body.cross(dorsal_body).normalize();
    let craft_to_body = DMat3::from_cols(right_body, nose_body, dorsal_body);
    (body_state.orientation * DQuat::from_mat3(&craft_to_body)).normalize()
}

fn update_debug_surface_teleport_cursor(
    debug: Res<DebugMode>,
    mut teleport: ResMut<DebugSurfaceTeleport>,
    sim: Res<SimulationState>,
    body_states: Res<SolarSystemState>,
    surfaces: Res<TerrainSurfaceRegistry>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), (With<ActiveCamera>, With<MapCamera>)>,
    bodies: Query<(&CelestialBody, &Transform)>,
    mut gizmos: Gizmos,
) {
    if !debug.enabled || teleport.armed_body.is_none() {
        teleport.hover = None;
        return;
    }

    teleport.hover = raycast_debug_surface_cursor(
        &teleport,
        &sim,
        &body_states,
        &surfaces,
        &windows,
        &cameras,
        &bodies,
    );

    let Some(hit) = teleport.hover else {
        return;
    };
    let Ok((_, cam_transform)) = cameras.single() else {
        return;
    };
    draw_debug_surface_cursor(&mut gizmos, hit, cam_transform);
}

fn commit_debug_surface_teleport(
    mut commands: Commands,
    debug: Res<DebugMode>,
    input: Res<GameInputIntent>,
    mut contexts: EguiContexts,
    mut teleport: ResMut<DebugSurfaceTeleport>,
    mut active_bubble: Option<ResMut<ActiveLocalBubble>>,
    mut sim: ResMut<SimulationState>,
    mut launch_mount: ResMut<DebugLaunchMount>,
    mut eva_mode: ResMut<EvaMode>,
    mut plan: ResMut<ManeuverPlan>,
    mut selected: ResMut<SelectedNode>,
    mut target: ResMut<TargetBody>,
    mut view: ResMut<ViewMode>,
    mut throttle: ResMut<ThrottleState>,
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
    if !debug.enabled || teleport.armed_body.is_none() {
        return;
    }
    if input.escape {
        teleport.cancel();
        return;
    }
    if !input.primary_started {
        return;
    }
    let egui_pointer_busy = contexts
        .ctx_mut()
        .map(|ctx| ctx.wants_pointer_input())
        .unwrap_or(false);
    if egui_pointer_busy {
        return;
    }

    let Some(hit) = teleport.hover else {
        return;
    };
    if Some(hit.body_id) != teleport.armed_body {
        return;
    }
    let Some(body) = sim.system.bodies.get(hit.body_id).cloned() else {
        teleport.cancel();
        return;
    };
    if matches!(body.kind, BodyKind::Star) {
        warn!("surface drop ignored for star {}", body.name);
        teleport.cancel();
        return;
    }

    let sim_time = sim.simulation.sim_time();
    let body_state = sim.ephemeris.state(
        hit.body_id,
        thalos_physics_canonical::canonical::Epoch(sim_time),
    );

    if sim.simulation.vessel_kind() == VesselKind::Eva {
        // EVA keeps its persistent bubble: rewrite the capsule in place,
        // ground it, and let `step_eva_controller` snap it onto the surface.
        let (state, attitude) = surface_spawn_state(
            &body,
            &body_state,
            hit.dir_body,
            hit.surface_height_m,
            EVA_SURFACE_CLEARANCE_M,
        );
        if let Some(active) = active_bubble.as_mut()
            && let Some(bubble) = active.bubble.as_mut()
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
                hit.body_id,
                TranslationalState::from(state),
                attitude,
            );
        }
        launch_mount.active = None;
    } else {
        // Ships drop onto a launch clamp in body-fixed authority, respawning
        // their bubble from scratch.
        clear_active_local_bubble(&mut commands, &mut active_bubble);
        let (state, attitude) = surface_spawn_state(
            &body,
            &body_state,
            hit.dir_body,
            hit.surface_height_m,
            DEBUG_LAUNCH_MOUNT_HEIGHT_M,
        );
        let pose =
            body_fixed_pose_from_inertial(&body_state, TranslationalState::from(state), attitude);
        sim.simulation
            .transition_authority(AuthorityMode::BodyFixed {
                body: hit.body_id,
                pose,
            });
        sim.simulation.set_ship_state(state);
        sim.simulation.set_attitude(attitude);
        sim.simulation.warp.reset();
        launch_mount.active = Some(DebugLaunchMountState {
            body_id: hit.body_id,
            pose,
        });
    }

    sim.simulation.set_target_body(Some(hit.body_id));
    sim.simulation.set_throttle(0.0);
    target.target = Some(hit.body_id);
    throttle.commanded = 0.0;
    throttle.effective = 0.0;
    clear_debug_teleport_maneuvers(&mut plan, &mut selected);
    *view = ViewMode::Ship;
    teleport.cancel();

    let surface_label = if hit.used_rendered_surface {
        "rendered surface"
    } else {
        "spherical surface"
    };
    if !hit.used_rendered_surface {
        warn!(
            "surface drop for {} used spherical radius; rendered terrain is not available",
            body.name
        );
    }
    info!(
        "mounted craft {:.0} m above {} {} via cursor (dir_body=({:.3},{:.3},{:.3}) h={:.1}m)",
        DEBUG_LAUNCH_MOUNT_HEIGHT_M,
        body.name,
        surface_label,
        hit.dir_body.x,
        hit.dir_body.y,
        hit.dir_body.z,
        hit.surface_height_m,
    );
}

fn raycast_debug_surface_cursor(
    teleport: &DebugSurfaceTeleport,
    sim: &SimulationState,
    body_states: &SolarSystemState,
    surfaces: &TerrainSurfaceRegistry,
    windows: &Query<&Window, With<PrimaryWindow>>,
    cameras: &Query<(&Camera, &GlobalTransform), (With<ActiveCamera>, With<MapCamera>)>,
    bodies: &Query<(&CelestialBody, &Transform)>,
) -> Option<DebugSurfaceTeleportHit> {
    let body_id = teleport.armed_body?;
    let body = sim.system.bodies.get(body_id)?;
    if matches!(body.kind, BodyKind::Star) {
        return None;
    }
    let states = body_states.states.as_deref()?;
    let body_state = states.get(body_id)?;
    let window = windows.single().ok()?;
    let cursor = window.cursor_position()?;
    let (camera, cam_transform) = cameras.single().ok()?;
    let ray = camera.viewport_to_world(cam_transform, cursor).ok()?;
    let (render_body, body_transform) = bodies.iter().find(|(body, _)| body.body_id == body_id)?;
    let center = body_transform.translation;
    let dir_render = ray_vs_sphere_dir(
        ray.origin - center,
        *ray.direction,
        render_body.render_radius,
    )?;

    let dir_world = dir_render.as_dvec3().normalize();
    let dir_body = (body_state.orientation.inverse() * dir_world).normalize();
    let (surface_height_m, used_rendered_surface) = if let Some(surface) = surfaces.get(body_id) {
        let dynamic_state = body_states.dynamic_surface_for(body_id, &surface);
        let query = thalos_terrain::BakedSurface::new(surface.clone(), dynamic_state);
        (
            rendered_height_m(&query, dir_body.as_vec3(), 1.0) as f64,
            true,
        )
    } else {
        (0.0, false)
    };
    let normal_render = (body_state.orientation * dir_body)
        .as_vec3()
        .normalize_or_zero();
    let render_pos =
        center + normal_render * ((body.radius_m + surface_height_m) * MAP_SCALE) as f32;

    Some(DebugSurfaceTeleportHit {
        body_id,
        dir_body,
        surface_height_m,
        render_pos,
        normal_render,
        used_rendered_surface,
    })
}

/// Ray-vs-sphere intersection for a sphere centered at the origin. Returns
/// the hit direction on the sphere, or `None` when the ray misses.
fn ray_vs_sphere_dir(origin: Vec3, dir: Vec3, radius: f32) -> Option<Vec3> {
    let b = origin.dot(dir);
    let c = origin.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    let near = -b - root;
    let far = -b + root;
    let t = if near >= 0.0 {
        near
    } else if far >= 0.0 {
        far
    } else {
        return None;
    };
    Some((origin + dir * t).normalize_or_zero())
}

fn draw_debug_surface_cursor(
    gizmos: &mut Gizmos,
    hit: DebugSurfaceTeleportHit,
    cam_transform: &GlobalTransform,
) {
    let cam = cam_transform.compute_transform();
    let right = cam.rotation * Vec3::X;
    let up = cam.rotation * Vec3::Y;
    let distance = (cam.translation - hit.render_pos).length().max(1.0);
    let size = (distance * 0.015).clamp(0.015, 0.4);
    let color = Color::srgb(0.15, 0.95, 1.0);
    let normal_color = Color::srgb(1.0, 0.95, 0.35);

    gizmos.line(
        hit.render_pos - right * size,
        hit.render_pos + right * size,
        color,
    );
    gizmos.line(
        hit.render_pos - up * size,
        hit.render_pos + up * size,
        color,
    );
    gizmos.line(
        hit.render_pos,
        hit.render_pos + hit.normal_render * size * 1.25,
        normal_color,
    );
}

fn clear_active_local_bubble(
    commands: &mut Commands,
    active_bubble: &mut Option<ResMut<ActiveLocalBubble>>,
) {
    let Some(active) = active_bubble.as_mut() else {
        return;
    };
    let Some(bubble) = active.bubble.take() else {
        return;
    };
    commands.entity(bubble.craft_entity).despawn();
    if let Some(terrain_entity) = bubble.terrain_entity {
        commands.entity(terrain_entity).despawn();
    }
}

fn clear_debug_teleport_maneuvers(plan: &mut ManeuverPlan, selected: &mut SelectedNode) {
    if !plan.nodes.is_empty() {
        plan.nodes.clear();
        plan.dirty = true;
    }
    selected.id = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use thalos_physics_canonical::canonical::Epoch;
    use thalos_world::BodyKind;

    fn body_definition() -> BodyDefinition {
        BodyDefinition {
            id: 1,
            name: "Test".to_string(),
            kind: BodyKind::Planet,
            parent: None,
            mass_kg: 1.0e20,
            radius_m: 1000.0,
            color: [1.0, 1.0, 1.0],
            rotation_period_s: 86_400.0,
            axial_tilt_rad: 0.0,
            gm: 1.0,
            soi_radius_m: 100_000.0,
            orbital_elements: None,
            terrain: thalos_terrain::TerrainConfig::None,
            tectonics: None,
            atmosphere: None,
            terrestrial_atmosphere: None,
            rings: None,
            surface_frame_ceiling_m: None,
        }
    }

    fn body_state() -> BodyState {
        BodyState {
            id: 1,
            epoch: Epoch(0.0),
            position: DVec3::new(100.0, 20.0, -30.0),
            velocity: DVec3::new(5.0, 0.0, -2.0),
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::Y * 0.1,
            mass_kg: 1.0e20,
            gm: 1.0,
            radius_m: 1000.0,
        }
    }

    #[test]
    fn surface_spawn_matches_rotating_surface_velocity_and_up_attitude() {
        let body = body_definition();
        let body_state = body_state();
        let dir_body = DVec3::Z;
        let (state, attitude) = surface_spawn_state(&body, &body_state, dir_body, 12.0, 8.0);
        let position_body = dir_body * 1020.0;

        assert!((state.position - (body_state.position + position_body)).length() < 1.0e-9);
        assert!(
            (state.velocity - body_fixed_surface_velocity(&body_state, position_body)).length()
                < 1.0e-9
        );
        assert!(((attitude.orientation * DVec3::Y) - dir_body).length() < 1.0e-9);
        assert!((attitude.orientation * DVec3::Z).dot(DVec3::X) > 0.999);
    }
}
