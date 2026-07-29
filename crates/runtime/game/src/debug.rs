//! Debug utilities. Hardcoded on for now; later this becomes an
//! in-game settings toggle.

use bevy::gizmos::prelude::{GizmoConfigGroup, GizmoConfigStore, GizmoPrimitive3d};
use bevy::math::primitives::{Capsule3d, Cone, Cuboid, Cylinder, Sphere};
use bevy::math::{DMat3, DQuat, DVec3, Isometry3d, Quat, Vec3};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use thalos_body_render::TerrainPatchBasis;
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::{
    body_fixed::body_fixed_pose_from_inertial,
    body_fixed::body_fixed_surface_velocity,
    canonical::{AuthorityMode, TranslationalState},
    debug_orbits::debug_parking_orbit_state,
    types::{AttitudeState, BodyState, VesselKind},
};
use thalos_physics_local::avian::{AngularVelocity, LinearVelocity, Position, Rotation};
use thalos_physics_local::{
    ActiveLocalBubble, HeightSourceRegistry, LocalCraftBody, LocalCraftColliderPrimitives,
    LocalPrimitiveCollider, LocalPrimitiveShape,
};
use thalos_world::{BodyDefinition, BodyId, BodyKind, StateVector};

use crate::camera::{ActiveCamera, MapCamera};
use crate::coords::{MAP_SCALE, SHIP_LAYER};
use crate::fuel::ThrottleState;
use crate::local_physics::{PHYSICS_QUERY_TILE_LOD_M, WheelSet, place_eva_on_surface};
use crate::maneuver::{ManeuverPlan, SelectedNode};
use crate::navigation::SHIP_NOSE_BODY;
use crate::pause_menu::not_game_paused;
use crate::photo_mode::not_in_photo_mode;
use crate::player_controller::EvaMode;
use crate::rendering::{CelestialBody, PlayerShip, SimulationState, SolarSystemState};
use crate::target::TargetBody;
use crate::view::{ViewMode, in_map_view};

/// Debug surface drops place the craft this far above the terrain in a landed
/// `BodyFixed` pose; the player throttles up to fly it off
/// ([`crate::regime::apply_regime_authority`], the landed throttle release).
pub const DEBUG_SURFACE_DROP_HEIGHT_M: f64 = 18.0;

/// EVA surface teleports plant the capsule a couple of metres above the
/// rendered terrain; `step_eva_controller` re-seeds and snaps it onto the
/// surface on the next frame, so this is just a safe initial clearance.
const EVA_SURFACE_CLEARANCE_M: f64 = 2.0;

#[derive(Resource, Debug, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub struct DebugMode {
    pub enabled: bool,
    /// Draw physics hitboxes (craft colliders + gear contact + ground surface).
    /// Toggled by F3; see [`draw_debug_hitboxes`].
    pub show_hitboxes: bool,
    /// Debug hack: let air-breathing engines produce rated sea-level thrust
    /// regardless of atmosphere — fire in vacuum, no density lapse — so
    /// aircraft can taxi/fly on airless bodies for ground/wheel testing.
    /// Contradicts the atmosphere model; **off by default** now that Thalos
    /// has air (leaving it on pinned every jet at rated thrust and defeated
    /// the thrust lapse / transonic wall). Edit the default and rebuild to
    /// toggle it (Reflect-registered for a future debug UI).
    pub jets_in_vacuum: bool,
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
            show_hitboxes: false,
            // Off: Thalos has an atmosphere now, so the runway scenarios fly
            // on real air. Flip the default on for airless-body ground testing.
            jets_in_vacuum: false,
        })
        .register_type::<DebugMode>()
        .init_gizmo_group::<CraftColliderGizmos>()
        .init_resource::<DebugSurfaceTeleport>()
        .add_systems(Startup, configure_craft_collider_gizmos)
        // PostUpdate after big_space's transform propagation: this system
        // anchors gizmos to the PlayerShip `GlobalTransform`, which under
        // big_space is recomputed (floating-origin-relative) only in
        // `TransformSystems::Propagate`. Reading it in Update used the
        // *previous* frame's value while the gizmo lines render with this
        // frame's camera — and a parked craft still sweeps the root grid at
        // heliocentric speed (~500 m/frame), so the overlay jittered by
        // hundreds of metres whenever the sim ran. Same staleness fix as
        // `draw_aero_debug` and `update_body_terrain_atmosphere`.
        .add_systems(
            bevy::app::PostUpdate,
            draw_debug_hitboxes
                .run_if(not_game_paused.and_then(not_in_photo_mode))
                .after(bevy::transform::TransformSystems::Propagate),
        )
        .add_systems(
            Update,
            (
                update_debug_surface_teleport_cursor,
                commit_debug_surface_teleport.after(update_debug_surface_teleport_cursor),
            )
                .run_if(
                    not_game_paused
                        .and_then(not_in_photo_mode)
                        .and_then(in_map_view),
                )
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

// The F3 toggle for `show_hitboxes` lives in `perf::overlay::toggle_debug_view`:
// one press flips the debug view, these hitboxes, and the aero gizmos together.

/// Half-width of the ground-collider grid drawn under the craft, in metres.
const GROUND_GRID_HALF_M: f64 = 50.0;
/// Cells per side of the ground-collider grid.
const GROUND_GRID_STEPS: usize = 16;

/// **F3 hitbox overlay.** Draws, at the rendered ship pose, three things the
/// surface-interaction physics actually uses:
///
/// - **Craft colliders** (cyan) — the compound rigid-body primitives.
/// - **Landing-gear contact geometry** (yellow) — the suspension travel line
///   and wheel rim. Wheels are raycast springs, *not* colliders, so they never
///   show up in the craft compound or the analytic backstop's primitive set —
///   yet they are exactly what touches (and can sink into) the ground. Drawing
///   them is what makes gear-vs-ground clipping visible.
/// - **Ground collider surface** (orange grid) — sampled from the *same*
///   [`HeightSource`] the terrain collider patch and the floor backstop read, so
///   this is the true contact surface. Compare it against the rendered terrain
///   to spot collider/visual mismatch, and against the gear rims to spot
///   penetration.
///
/// Avian's built-in `PhysicsGizmos` can't be used here: under `big_space` the
/// colliders live at body-fixed coordinates (~planet radius), which don't map to
/// render space. Everything below is placed by recovering the body-fixed→render
/// isometry from the ship's pose.
fn draw_debug_hitboxes(
    debug: Res<DebugMode>,
    view: Res<ViewMode>,
    active: Res<ActiveLocalBubble>,
    height_sources: Res<HeightSourceRegistry>,
    sim: Res<SimulationState>,
    ship_q: Query<&GlobalTransform, With<PlayerShip>>,
    craft_q: Query<
        (
            &LocalCraftColliderPrimitives,
            &Position,
            &Rotation,
            Option<&WheelSet>,
        ),
        With<LocalCraftBody>,
    >,
    mut gizmos: Gizmos<CraftColliderGizmos>,
) {
    if !debug.enabled || !debug.show_hitboxes || *view != ViewMode::Ship {
        return;
    }
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Ok((primitives, position, rotation, wheels)) = craft_q.get(bubble.craft_entity) else {
        return;
    };
    let Ok(root) = ship_q.single() else {
        return;
    };
    let (_, root_rotation, root_translation) = root.affine().to_scale_rotation_translation();

    // Craft-local point → render: the rendered `GlobalTransform` *is* the craft
    // pose in render space, so a craft-local vector rotates by `root_rotation`.
    let craft_to_render = |q: DVec3| root_translation + root_rotation * q.as_vec3();

    // Body-fixed point → render. The craft's Avian body lives in the
    // surface-local frame; composing its craft→SLF `Rotation` with the
    // constant body-fixed→SLF frame rotation recovers the craft→body-fixed
    // pose, which paired with the rendered pose gives the body-fixed→render
    // isometry — placing the ground samples (body-fixed frame) without
    // touching big_space. Differences `p − P_craft` are local (metres), so
    // the f64→f32 narrowing is exact here.
    let frame_rot_inv = bubble.frame.rotation_body_to_frame.inverse();
    let p_craft = frame_rot_inv * bubble.frame.body_center_offset(position.0);
    let rot_craft_to_body_fixed = frame_rot_inv * rotation.0;
    let r_map = root_rotation * rot_craft_to_body_fixed.as_quat().inverse();
    let body_to_render = |p: DVec3| root_translation + r_map * (p - p_craft).as_vec3();

    // --- Craft collider primitives (cyan) ---------------------------------
    let craft_color = Color::srgba(0.0, 1.0, 0.75, 0.9);
    for primitive in &primitives.0 {
        draw_collider_primitive(
            &mut gizmos,
            root_translation,
            root_rotation,
            primitive,
            craft_color,
        );
    }

    // --- Landing-gear contact geometry (yellow) ---------------------------
    if let Some(wheels) = wheels {
        let gear_color = Color::srgba(1.0, 0.85, 0.2, 0.95);
        for wheel in &wheels.wheels {
            let top = craft_to_render(wheel.strut_top_local);
            let axle = wheel.strut_top_local + wheel.susp_dir_local * wheel.strut_length;
            let bottom = craft_to_render(axle + wheel.susp_dir_local * wheel.wheel_radius);
            // Suspension travel line, strut top down to the contact patch.
            gizmos.line(top, bottom, gear_color);
            // Wheel rim: a circle of `wheel_radius` about the axle, normal along
            // the axle axis so it reads as the side profile of the tyre.
            let rim_normal = (root_rotation * wheel.axle_dir_local.as_vec3()).normalize_or_zero();
            if rim_normal != Vec3::ZERO {
                gizmos
                    .circle(
                        Isometry3d::new(
                            craft_to_render(axle),
                            Quat::from_rotation_arc(Vec3::Z, rim_normal),
                        ),
                        wheel.wheel_radius as f32,
                        gear_color,
                    )
                    .resolution(20);
            }
        }
    }

    // --- Ground collider surface (orange grid) ----------------------------
    if let Some(height_source) = height_sources.get(bubble.body_id)
        && let Some(dir_craft) = p_craft.try_normalize()
    {
        let body = &sim.system.bodies[bubble.body_id];
        let basis = TerrainPatchBasis::from_normal(dir_craft);
        let ground_color = Color::srgba(1.0, 0.5, 0.0, 0.7);
        let cell =
            |i: usize| (i as f64 / GROUND_GRID_STEPS as f64 * 2.0 - 1.0) * GROUND_GRID_HALF_M;
        let mut grid: Vec<Vec<Vec3>> = Vec::with_capacity(GROUND_GRID_STEPS + 1);
        for iz in 0..=GROUND_GRID_STEPS {
            let z = cell(iz);
            let mut row = Vec::with_capacity(GROUND_GRID_STEPS + 1);
            for ix in 0..=GROUND_GRID_STEPS {
                let x = cell(ix);
                let dir = (dir_craft * body.radius_m + basis.tangent_x * x + basis.tangent_z * z)
                    .normalize();
                let h = height_source
                    .sample_height_m(dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
                    .unwrap_or(0.0) as f64;
                row.push(body_to_render(dir * (body.radius_m + h)));
            }
            grid.push(row);
        }
        for iz in 0..=GROUND_GRID_STEPS {
            for ix in 0..=GROUND_GRID_STEPS {
                if ix < GROUND_GRID_STEPS {
                    gizmos.line(grid[iz][ix], grid[iz][ix + 1], ground_color);
                }
                if iz < GROUND_GRID_STEPS {
                    gizmos.line(grid[iz][ix], grid[iz + 1][ix], ground_color);
                }
            }
        }
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
    height_sources: Res<HeightSourceRegistry>,
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
        &height_sources,
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
    ui_pointer_gate: Res<crate::hud::UiPointerGate>,
    mut teleport: ResMut<DebugSurfaceTeleport>,
    mut active_bubble: Option<ResMut<ActiveLocalBubble>>,
    mut sim: ResMut<SimulationState>,
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
    if ui_pointer_gate.hovered {
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
    } else {
        // Ships drop in landed `BodyFixed` authority just above the terrain,
        // respawning their bubble from scratch; the player throttles up to fly
        // off (the authority executor's landed throttle release).
        clear_active_local_bubble(&mut commands, &mut active_bubble);
        let (state, attitude) = surface_spawn_state(
            &body,
            &body_state,
            hit.dir_body,
            hit.surface_height_m,
            DEBUG_SURFACE_DROP_HEIGHT_M,
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
        "dropped craft {:.0} m above {} {} via cursor (dir_body=({:.3},{:.3},{:.3}) h={:.1}m)",
        DEBUG_SURFACE_DROP_HEIGHT_M,
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
    height_sources: &HeightSourceRegistry,
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
    let (surface_height_m, used_rendered_surface) = match height_sources.get(body_id) {
        Some(hs) => (
            hs.sample_height_m(dir_body.as_vec3(), 1.0).unwrap_or(0.0) as f64,
            true,
        ),
        None => (0.0, false),
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
            ocean: None,
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
