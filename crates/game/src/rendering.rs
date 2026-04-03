//! Rendering module: celestial bodies, orbit lines, and ship marker.
//!
//! # Coordinate system
//! The physics simulation uses a heliocentric inertial frame with the ecliptic
//! as the XZ plane (Y up). All positions from the ephemeris are in metres.
//! We apply `RENDER_SCALE` to convert metres to render units so Bevy's f32
//! transforms don't lose precision on solar-system distances.
//!
//! 1 render unit = 1 / RENDER_SCALE metres = 1,000 km.

use std::sync::Arc;

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use thalos_physics::{
    ephemeris::Ephemeris,
    simulation::Simulation,
    types::{BodyKind, BodyStates, SolarSystemDefinition},
};

use crate::camera::{CameraFocus, OrbitCamera};
use crate::SimStage;

// ---------------------------------------------------------------------------
// Scale
// ---------------------------------------------------------------------------

/// Metres → render units.  1 render unit = 1,000 km.
pub const RENDER_SCALE: f64 = 1e-6;

/// The physics-space position (metres, f64) that maps to the render-space
/// origin.  Updated every frame to the camera focus body's position so that
/// objects near the camera always have small render-space coordinates,
/// preserving f32 precision at any zoom level.
#[derive(Resource, Default)]
pub struct RenderOrigin {
    pub position: bevy::math::DVec3,
}

/// Radius of screen-stable body icon markers as a fraction of camera distance
/// (in render units). Bodies whose rendered sphere is smaller than this get
/// replaced by a fixed-size circle billboard.
const MARKER_RADIUS: f32 = 0.006;

/// Convert a physics DVec3 (metres, f64) to a Bevy Vec3 (render units, f32).
#[inline]
fn to_render_pos(v: bevy::math::DVec3) -> Vec3 {
    (v * RENDER_SCALE).as_vec3()
}

/// Points sampled along each orbit for gizmo line drawing.
const ORBIT_SAMPLES: usize = 256;

// ---------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------

/// Central simulation state.
#[derive(Resource)]
pub struct SimulationState {
    pub simulation: Simulation,
    pub system: SolarSystemDefinition,
    pub ephemeris: Arc<Ephemeris>,
}

/// Per-frame cache of all body states at the current sim time. Populated once
/// per frame by `cache_body_states` and read by multiple rendering systems to
/// avoid redundant ephemeris queries.
#[derive(Resource, Default)]
pub struct FrameBodyStates {
    pub states: Option<BodyStates>,
    pub time: f64,
}

fn cache_body_states(sim: Res<SimulationState>, mut cache: ResMut<FrameBodyStates>) {
    let t = sim.simulation.sim_time();
    cache.states = Some(sim.ephemeris.query(t));
    cache.time = t;
}

/// Convert an sRGB component (0..1) to linear light.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Distance (in render units) the focus body must move from the last-set
/// origin before `update_render_origin` actually updates.  Prevents jitter
/// from tiny frame-to-frame position changes.
const ORIGIN_UPDATE_THRESHOLD: f64 = 1000.0; // ~1 000 000 km

/// Tracks which entity the origin was last locked to, so a focus-body change
/// always triggers an immediate origin update.
#[derive(Resource, Default)]
struct PreviousFocusEntity {
    entity: Option<Entity>,
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

#[derive(Component)]
pub struct CelestialBody {
    pub body_id: usize,
    pub is_star: bool,
    pub render_radius: f32,
    /// True physical radius in metres (not clamped like render_radius).
    pub radius_m: f64,
}

#[derive(Component)]
pub struct ShipMarker;

/// Marker for the 3D sphere mesh child of a celestial body.
#[derive(Component)]
struct BodyMesh;

/// Marker for the flat circle icon child of a celestial body.
#[derive(Component)]
struct BodyIcon;

// ---------------------------------------------------------------------------
// Precomputed orbit line data (computed once at startup)
// ---------------------------------------------------------------------------

/// Stores precomputed orbit line points for each body. Computed at startup
/// from the ephemeris and reused every frame to avoid per-frame queries.
#[derive(Resource)]
struct OrbitLines {
    /// One entry per body. `None` for the star (no orbit). Each entry is a
    /// vec of render-space points forming a closed loop.
    lines: Vec<Option<OrbitLine>>,
}

struct OrbitLine {
    points: Vec<Vec3>,
    color: Color,
    /// Parent body whose position is subtracted from the orbit points.
    /// The line must be translated by the parent's current render position
    /// each frame so the orbit tracks the moving parent.
    parent_id: usize,
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

/// Tracks the last left-click time and screen position for double-click detection.
#[derive(Resource, Default)]
struct LastClick {
    time: f64,
    position: Vec2,
}

const DOUBLE_CLICK_THRESHOLD: f64 = 0.4; // seconds
const DOUBLE_CLICK_RADIUS: f32 = 10.0;   // pixels — tolerance for cursor drift between clicks

pub struct RenderingPlugin;

impl Plugin for RenderingPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(LastClick::default())
            .insert_resource(RenderOrigin::default())
            .insert_resource(FrameBodyStates::default())
            .insert_resource(PreviousFocusEntity::default())
            .add_systems(Startup, (configure_gizmos, spawn_bodies, precompute_orbit_lines, focus_camera_on_homeworld.after(spawn_bodies)))
            .add_systems(Update, (
                cache_body_states,
                update_render_origin.after(cache_body_states),
                update_body_positions.after(update_render_origin),
                update_ship_position.after(update_render_origin),
                draw_orbits.after(update_render_origin),
                sync_body_icons,
                double_click_focus_system,
            ).in_set(SimStage::Sync));
    }
}

fn configure_gizmos(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.line.width = 2.0;
}

// ---------------------------------------------------------------------------
// Startup: spawn body meshes and ship marker
// ---------------------------------------------------------------------------

fn spawn_bodies(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sim: Res<SimulationState>,
) {
    let bodies = &sim.system.bodies;
    let initial_states = sim.ephemeris.query(0.0);

    // Shared icon mesh: unit circle, scaled per frame.
    let icon_mesh = meshes.add(Circle::new(1.0));

    for body in bodies {
        let state = &initial_states[body.id];
        let pos = to_render_pos(state.position);

        let radius_render = (body.radius_m * RENDER_SCALE) as f32;
        let render_radius = radius_render.max(0.005);

        let sphere_mesh = meshes.add(Sphere::new(render_radius).mesh().ico(3).unwrap());

        let [r, g, b] = body.color;
        let base_color = Color::srgb(r, g, b);
        let is_star = body.kind == BodyKind::Star;

        let sphere_material = if is_star {
            materials.add(StandardMaterial {
                base_color,
                emissive: LinearRgba::new(r, g, b, 1.0) * 3.0,
                ..default()
            })
        } else {
            materials.add(StandardMaterial {
                base_color,
                perceptual_roughness: 0.8,
                metallic: 0.0,
                ..default()
            })
        };

        // Icon material: unlit, emissive, double-sided flat circle.
        let icon_material = materials.add(StandardMaterial {
            base_color,
            emissive: LinearRgba::new(r, g, b, 1.0) * 2.0,
            unlit: true,
            double_sided: true,
            ..default()
        });

        commands
            .spawn((
                Transform::from_translation(pos),
                Visibility::Inherited,
                CelestialBody {
                    body_id: body.id,
                    is_star,
                    render_radius,
                    radius_m: body.radius_m,
                },
                Name::new(body.name.clone()),
            ))
            .with_child((
                Mesh3d(sphere_mesh),
                MeshMaterial3d(sphere_material),
                BodyMesh,
            ))
            .with_child((
                Mesh3d(icon_mesh.clone()),
                MeshMaterial3d(icon_material),
                Transform::default(),
                Visibility::Hidden,
                BodyIcon,
            ));
    }

    // Ship marker: screen-stable billboard circle, white.
    let ship_pos = to_render_pos(sim.simulation.ship_state().position);
    let ship_icon = meshes.add(Circle::new(1.0));
    let ship_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        emissive: LinearRgba::WHITE * 2.0,
        unlit: true,
        double_sided: true,
        ..default()
    });

    commands.spawn((
        Mesh3d(ship_icon),
        MeshMaterial3d(ship_material),
        Transform::from_translation(ship_pos),
        ShipMarker,
        Name::new("Ship"),
    ));

    // Point light at the star so the scene is lit.
    commands.spawn((
        PointLight {
            intensity: 1_000_000.0,
            range: 10_000.0,
            color: Color::WHITE,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_translation(Vec3::ZERO),
    ));
}

// ---------------------------------------------------------------------------
// Startup: precompute orbit lines from ephemeris (runs once)
// ---------------------------------------------------------------------------

fn precompute_orbit_lines(mut commands: Commands, sim: Res<SimulationState>) {
    let bodies = &sim.system.bodies;
    let mut lines = Vec::with_capacity(bodies.len());

    for body in bodies {
        if body.parent.is_none() {
            lines.push(None);
            continue;
        }

        let period_s = match &body.orbital_elements {
            Some(el) => {
                let parent_id = body.parent.unwrap();
                let parent_gm = bodies[parent_id].gm;
                let a = el.semi_major_axis_m;
                2.0 * std::f64::consts::PI * (a * a * a / parent_gm).sqrt()
            }
            None => sim.ephemeris.time_span() * 0.01,
        };

        let parent_id = body.parent.unwrap();
        let points: Vec<Vec3> = (0..=ORBIT_SAMPLES)
            .map(|i| {
                let t = (i as f64 / ORBIT_SAMPLES as f64) * period_s;
                let body_state = sim.ephemeris.query_body(body.id, t);
                let parent_state = sim.ephemeris.query_body(parent_id, t);
                to_render_pos(body_state.position - parent_state.position)
            })
            .collect();

        let [r, g, b] = body.color;
        let orbit_color = Color::linear_rgba(
            srgb_to_linear(r) * 0.4,
            srgb_to_linear(g) * 0.4,
            srgb_to_linear(b) * 0.4,
            0.6,
        );

        lines.push(Some(OrbitLine {
            points,
            color: orbit_color,
            parent_id,
        }));
    }

    commands.insert_resource(OrbitLines { lines });
}

// ---------------------------------------------------------------------------
// Per-frame: floating origin & transform updates
// ---------------------------------------------------------------------------

/// Sets the render origin to the camera focus body's position so that nearby
/// objects always have small render-space coordinates (full f32 precision).
/// Applies hysteresis: only updates when the focus body has moved more than
/// `ORIGIN_UPDATE_THRESHOLD` render units from the current origin, or when
/// the focus entity changes.
fn update_render_origin(
    cache: Res<FrameBodyStates>,
    focus: Res<CameraFocus>,
    bodies: Query<&CelestialBody>,
    mut origin: ResMut<RenderOrigin>,
    mut prev_focus: ResMut<PreviousFocusEntity>,
) {
    let Some(ref states) = cache.states else { return };

    let candidate = focus
        .target
        .and_then(|e| bodies.get(e).ok())
        .and_then(|b| states.get(b.body_id))
        .map(|s| s.position)
        .unwrap_or(bevy::math::DVec3::ZERO);

    let focus_changed = focus.target != prev_focus.entity;
    if focus_changed {
        prev_focus.entity = focus.target;
    }

    let delta_render = (candidate - origin.position) * RENDER_SCALE;
    if focus_changed || delta_render.length() > ORIGIN_UPDATE_THRESHOLD {
        origin.position = candidate;
    }
}

fn update_body_positions(
    cache: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    mut query: Query<(&CelestialBody, &mut Transform)>,
) {
    let Some(ref states) = cache.states else { return };

    for (body, mut transform) in &mut query {
        if let Some(state) = states.get(body.body_id) {
            transform.translation = to_render_pos(state.position - origin.position);
        }
    }
}

fn update_ship_position(
    sim: Res<SimulationState>,
    origin: Res<RenderOrigin>,
    focus: Res<CameraFocus>,
    camera_query: Query<&Transform, With<OrbitCamera>>,
    mut query: Query<&mut Transform, (With<ShipMarker>, Without<OrbitCamera>)>,
) {
    let Ok(cam_tf) = camera_query.single() else { return };
    let cam_render_dist = (focus.distance * RENDER_SCALE) as f32;
    let icon_radius = cam_render_dist * MARKER_RADIUS;

    for mut transform in &mut query {
        transform.translation = to_render_pos(sim.simulation.ship_state().position - origin.position);
        transform.rotation = cam_tf.rotation;
        transform.scale = Vec3::splat(icon_radius);
    }
}

// ---------------------------------------------------------------------------
// Per-frame: draw precomputed orbit lines with Gizmos
// ---------------------------------------------------------------------------

/// Distance-to-parent / camera-distance ratio at which orbit trails start fading.
const ORBIT_FADE_START: f64 = 20.0;
/// Ratio at which orbit trails are fully hidden.
const ORBIT_FADE_END: f64 = 100.0;

fn draw_orbits(
    mut gizmos: Gizmos,
    orbit_lines: Option<Res<OrbitLines>>,
    cache: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    focus: Res<CameraFocus>,
    bodies: Query<&CelestialBody>,
) {
    let Some(orbit_lines) = orbit_lines else {
        return;
    };
    let Some(ref states) = cache.states else {
        return;
    };

    // Determine the camera focus position in metres.
    let focus_pos = focus
        .target
        .and_then(|e| bodies.get(e).ok())
        .and_then(|b| states.get(b.body_id))
        .map(|s| s.position)
        .unwrap_or(bevy::math::DVec3::ZERO);

    let cam_dist = focus.distance;

    for line in orbit_lines.lines.iter().flatten() {
        let parent_pos_m = states.get(line.parent_id)
            .map(|s| s.position)
            .unwrap_or(bevy::math::DVec3::ZERO);
        let dist_to_parent = (parent_pos_m - focus_pos).length();
        let ratio = dist_to_parent / cam_dist;

        if ratio > ORBIT_FADE_END {
            continue;
        }

        let parent_render_pos = to_render_pos(parent_pos_m - origin.position);

        if ratio > ORBIT_FADE_START {
            // Smooth fade between ORBIT_FADE_START and ORBIT_FADE_END.
            let t = (ratio - ORBIT_FADE_START) / (ORBIT_FADE_END - ORBIT_FADE_START);
            let alpha = (1.0 - t) as f32;
            let faded = line.color.with_alpha(line.color.alpha() * alpha);
            gizmos.linestrip(
                line.points.iter().map(|p| *p + parent_render_pos),
                faded,
            );
        } else {
            gizmos.linestrip(
                line.points.iter().map(|p| *p + parent_render_pos),
                line.color,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame: toggle body mesh vs icon based on screen-space size
// ---------------------------------------------------------------------------

type IconFilter = (With<BodyIcon>, Without<CelestialBody>, Without<OrbitCamera>);
type MeshFilter = (With<BodyMesh>, Without<BodyIcon>, Without<CelestialBody>, Without<OrbitCamera>);

fn sync_body_icons(
    bodies: Query<(&CelestialBody, &Transform, &Children)>,
    focus: Res<CameraFocus>,
    camera_query: Query<&Transform, With<OrbitCamera>>,
    mut icons: Query<(&mut Transform, &mut Visibility), IconFilter>,
    mut meshes: Query<&mut Visibility, MeshFilter>,
) {
    let Ok(cam_tf) = camera_query.single() else {
        return;
    };

    let cam_rotation = cam_tf.rotation;
    let cam_render_dist = (focus.distance * RENDER_SCALE) as f32;
    let icon_radius = cam_render_dist * MARKER_RADIUS;

    for (body, _body_tf, children) in &bodies {
        let use_icon = !body.is_star && body.render_radius < icon_radius;

        for child in children.iter() {
            if let Ok((mut icon_tf, mut icon_vis)) = icons.get_mut(child) {
                if use_icon {
                    *icon_vis = Visibility::Inherited;
                    icon_tf.rotation = cam_rotation;
                    icon_tf.scale = Vec3::splat(icon_radius);
                } else {
                    *icon_vis = Visibility::Hidden;
                }
            }
            if let Ok(mut mesh_vis) = meshes.get_mut(child) {
                *mesh_vis = if use_icon {
                    Visibility::Hidden
                } else {
                    Visibility::Inherited
                };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Startup: focus camera on homeworld
// ---------------------------------------------------------------------------

fn focus_camera_on_homeworld(
    mut focus: ResMut<CameraFocus>,
    bodies: Query<(Entity, &CelestialBody, &Name)>,
) {
    // Find "Thalos" (homeworld) entity and focus the camera on it.
    for (entity, _body, name) in &bodies {
        if name.as_str() == "Thalos" {
            focus.target = Some(entity);
            focus.distance = 2e7; // 20,000 km — close enough to see the planet
            focus.target_distance = 2e7;
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Double-click body to focus camera
// ---------------------------------------------------------------------------

/// Detects double-clicks and focuses the camera on the nearest body by
/// projecting every body's world position to screen space and picking the
/// closest one to the cursor.  Works for both 3D sphere meshes and billboard
/// icons because we test the parent entity's transform, not the mesh child.
fn double_click_focus_system(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), With<OrbitCamera>>,
    bodies: Query<(Entity, &CelestialBody, &Transform)>,
    mut focus: ResMut<CameraFocus>,
    mut last_click: ResMut<LastClick>,
) {
    if !mouse_buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = windows.single() else { return };
    let Some(cursor_pos) = window.cursor_position() else { return };
    let Ok((camera, cam_gt)) = camera_q.single() else { return };

    let now = time.elapsed_secs_f64();
    let is_double = (now - last_click.time) < DOUBLE_CLICK_THRESHOLD
        && cursor_pos.distance(last_click.position) < DOUBLE_CLICK_RADIUS;

    last_click.time = now;
    last_click.position = cursor_pos;

    if !is_double {
        return;
    }

    // Reset so a third click doesn't trigger another double-click.
    last_click.time = 0.0;

    // Find the body whose screen-space projection is closest to the cursor.
    let mut best: Option<(Entity, f32)> = None;
    for (entity, body, transform) in &bodies {
        let Ok(screen) = camera.world_to_viewport(cam_gt, transform.translation) else {
            continue;
        };
        // Hit radius: whichever is larger — the projected sphere or the icon.
        let cam_dist = cam_gt.translation().distance(transform.translation);
        let projected_radius = if cam_dist > 0.0 {
            let viewport_height = window.height();
            let fov = std::f32::consts::FRAC_PI_4; // Bevy default
            let pixels_per_unit = viewport_height / (2.0 * (fov / 2.0).tan() * cam_dist);
            (body.render_radius * pixels_per_unit).max(MARKER_RADIUS * cam_dist * pixels_per_unit)
        } else {
            20.0
        };
        let hit_radius = projected_radius.max(12.0); // minimum 12px so tiny dots are clickable

        let dist = screen.distance(cursor_pos);
        if dist > hit_radius {
            continue;
        }
        if best.is_none() || dist < best.unwrap().1 {
            best = Some((entity, dist));
        }
    }

    let Some((target_entity, _)) = best else { return };

    // Compute smooth transition offset.
    let old_pos = focus
        .target
        .and_then(|e| bodies.get(e).ok())
        .map(|(_, _, t)| t.translation)
        .unwrap_or(Vec3::ZERO)
        + focus.focus_offset;

    let new_pos = bodies
        .get(target_entity)
        .map(|(_, _, t)| t.translation)
        .unwrap_or(Vec3::ZERO);

    focus.focus_offset = old_pos - new_pos;
    focus.target = Some(target_entity);
}
