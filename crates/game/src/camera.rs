use bevy::camera::visibility::RenderLayers;
use bevy::input::mouse::{AccumulatedMouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy_egui::EguiContexts;
use thalos_physics::types::{BodyDefinition, BodyState};
use thalos_planet_rendering::space_camera_post_stack;

use crate::coords::{MAP_LAYER, SHIP_LAYER};
use crate::rendering::{FrameBodyStates, SimulationState};
use crate::view::ViewMode;

/// Plugin that registers the orbit camera systems and spawns the camera entity.
pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BlockCameraInput>()
            .init_resource::<ShipCameraMode>()
            .insert_resource(CameraFocus::default())
            // Background is pure black — the forward-rendered
            // `SkyRenderPlugin` draws stars additively on top.
            .insert_resource(ClearColor(Color::BLACK))
            .add_systems(Startup, spawn_camera)
            .add_systems(
                Update,
                (
                    camera_min_distance_system,
                    ship_camera_mode_input,
                    camera_input_system,
                    camera_zoom_interpolation_system,
                    camera_focus_offset_decay_system,
                    camera_transform_system,
                )
                    .chain()
                    .in_set(crate::SimStage::Camera),
            );
    }
}

/// Marker component placed on every orbit camera entity (one per view).
/// Both cameras carry it; consumers that need *the active* camera should
/// query [`ActiveCamera`] instead.
#[derive(Component)]
pub struct OrbitCamera;

/// Marker for the map-view camera (renders [`MAP_LAYER`]).
#[derive(Component)]
pub struct MapCamera;

/// Marker for the ship-view camera (renders [`SHIP_LAYER`]).
#[derive(Component)]
pub struct ShipCamera;

/// Marker placed on whichever orbit camera is currently driving the
/// rendered view. Flipped between the two cameras when [`ViewMode`]
/// changes (see [`apply_active_camera`] in `view.rs`). Use this filter
/// in queries that need the camera the user is actually looking through
/// (billboard alignment, picking, screen-space sizing).
#[derive(Component)]
pub struct ActiveCamera;

/// Per-target offset, in the target's local frame, that the camera should
/// pivot around instead of the entity's transform translation. Used by the
/// player ship to centre the camera on the mass-weighted CoM of all parts
/// (matching KSP's vessel camera behaviour) — celestial bodies don't need
/// this and simply omit the component.
#[derive(Component, Default, Debug, Clone, Copy)]
pub struct CameraTargetOffset(pub Vec3);

/// Set to true by the maneuver plugin when the pointer is over a maneuver
/// element (arrow, slide sphere) or an active drag/placement is in progress.
/// Camera rotation is suppressed while this is set.
#[derive(Resource, Default)]
pub struct BlockCameraInput(pub bool);

/// KSP-style camera modes for ship view. `V` cycles between them.
///
/// - **Free**: camera "up" is gravity-up (radial out from the dominant body),
///   "forward" is the horizon-projected prograde direction. As the ship orbits,
///   the planet stays "down" in the view.
/// - **Orbital**: camera "up" is the orbital plane normal, "forward" is the
///   prograde direction. The orbit appears edge-on, and the camera frame
///   rotates with the ship around the orbit.
///
/// In map view this resource is ignored — that view always uses world-Y up.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShipCameraMode {
    #[default]
    Free,
    Orbital,
}

impl ShipCameraMode {
    fn cycle(self) -> Self {
        match self {
            Self::Free => Self::Orbital,
            Self::Orbital => Self::Free,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Free => "Free",
            Self::Orbital => "Orbital",
        }
    }
}

/// The camera orbits around `target` (if Some) using spherical coordinates.
///
/// Distance is in metres, stored as f64 to cover the full range from
/// 100 km (low orbit) to ~67 AU without precision loss.
/// Azimuth and elevation are in radians.
///
/// Zoom is smoothed: scroll input sets `target_distance` and each frame
/// `distance` interpolates toward it in log-space for scale-independent feel.
#[derive(Resource)]
pub struct CameraFocus {
    /// Entity to orbit around. When None the camera sits at the world origin.
    pub target: Option<Entity>,
    /// Current radial distance from the target, in metres (interpolated each frame).
    pub distance: f64,
    /// Desired radial distance — scroll input drives this, `distance` chases it.
    pub target_distance: f64,
    /// Horizontal angle around the target, in radians.
    pub azimuth: f32,
    /// Vertical angle from the equatorial plane, clamped to ±89°, in radians.
    pub elevation: f32,
    /// Minimum distance in metres — set to the focused body's surface radius.
    pub min_distance: f64,
    /// Render-space offset from the true target position, used to smoothly
    /// interpolate the camera's look-at point when switching focus targets.
    /// Set to `old_pos - new_pos` on switch, then decayed to zero each frame.
    pub focus_offset: Vec3,
}

impl Default for CameraFocus {
    fn default() -> Self {
        Self {
            target: None,
            distance: 5e11, // ~3.3 AU, sees inner system
            target_distance: 5e11,
            azimuth: 0.0,
            elevation: 0.3, // slight downward tilt so the horizon is visible
            min_distance: DISTANCE_MIN_DEFAULT,
            focus_offset: Vec3::ZERO,
        }
    }
}

const DISTANCE_MIN_DEFAULT: f64 = 1e5; // 100 km
const DISTANCE_MAX: f64 = 1e13; // ~67 AU
/// Camera stops at 3× the body's radius (comfortable viewing distance).
const SURFACE_MARGIN: f64 = 3.0;
/// Closest the camera may zoom to the player ship in ship view (metres).
/// Small enough to put the camera a few metres off the hull.
const SHIP_MIN_DISTANCE_M: f64 = 5.0;

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

fn spawn_camera(mut commands: Commands, view: Res<ViewMode>) {
    let map_active = matches!(*view, ViewMode::Map);

    let mut map_cam = commands.spawn((
        Camera3d::default(),
        Camera {
            is_active: map_active,
            order: 0,
            ..default()
        },
        // At MAP_SCALE (1 unit = 1000 km), the system fits in ~1e7 units;
        // the Bevy default perspective projection covers that range.
        Projection::Perspective(PerspectiveProjection::default()),
        space_camera_post_stack(),
        OrbitCamera,
        MapCamera,
        bevy::picking::mesh_picking::MeshPickingCamera,
        // Layer 0 (default) covers entities visible in both views (bodies,
        // sky); MAP_LAYER covers map-only overlays.
        RenderLayers::from_layers(&[0, MAP_LAYER]),
        // Transform is overwritten every frame by camera_transform_system.
        // We set a sane default so the first frame renders something.
        Transform::from_xyz(0.0, 0.0, 5e6).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    if map_active {
        map_cam.insert(ActiveCamera);
    }

    let mut ship_cam = commands.spawn((
        Camera3d::default(),
        Camera {
            is_active: !map_active,
            order: 0,
            ..default()
        },
        // At SHIP_SCALE (1 unit = 1 m), near 0.5 m puts the camera a few cm
        // off the hull; far 1e11 m (~0.67 AU) covers the nearest bodies and
        // the system's star with f32 precision.
        Projection::Perspective(PerspectiveProjection {
            near: 0.5,
            far: 1.0e11,
            ..default()
        }),
        space_camera_post_stack(),
        OrbitCamera,
        ShipCamera,
        bevy::picking::mesh_picking::MeshPickingCamera,
        // Layer 0 (default) covers entities visible in both views (bodies,
        // sky); SHIP_LAYER covers ship-only entities (ship parts, etc.).
        RenderLayers::from_layers(&[0, SHIP_LAYER]),
        Transform::from_xyz(0.0, 0.0, 5e6).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    if !map_active {
        ship_cam.insert(ActiveCamera);
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// `V` cycles ship-view camera mode (Free ↔ Orbital). Suppressed in map view
/// and while egui is consuming keyboard input (e.g. text fields).
fn ship_camera_mode_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut contexts: EguiContexts,
    view: Res<ViewMode>,
    mut mode: ResMut<ShipCameraMode>,
) {
    if *view != ViewMode::Ship || !keys.just_pressed(KeyCode::KeyV) {
        return;
    }
    if let Ok(ctx) = contexts.ctx_mut()
        && ctx.wants_keyboard_input()
    {
        return;
    }
    *mode = mode.cycle();
}

/// Reads mouse input and updates [`CameraFocus`].
///
/// - Left-button drag  → rotate (azimuth / elevation)
/// - Scroll wheel      → sets `target_distance` (actual zoom is interpolated by `camera_zoom_interpolation_system`)
pub fn camera_input_system(
    block: Res<BlockCameraInput>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mut scroll_events: MessageReader<MouseWheel>,
    mut focus: ResMut<CameraFocus>,
) {
    const ROTATION_SENSITIVITY: f32 = 0.005; // rad per pixel
    const ZOOM_FACTOR_MIN: f64 = 0.005; // near surface / local system
    const ZOOM_FACTOR_MAX: f64 = 0.01; // interplanetary scale
    const ELEVATION_MAX: f32 = 89.0_f32.to_radians();

    // --- Rotation -----------------------------------------------------------
    // Suppressed while a maneuver element is hovered or being dragged.
    if mouse_buttons.pressed(MouseButton::Left) && !block.0 {
        let delta = mouse_motion.delta;
        if delta != Vec2::ZERO {
            focus.azimuth += delta.x * ROTATION_SENSITIVITY;
            focus.elevation -= delta.y * ROTATION_SENSITIVITY;
            focus.elevation = focus.elevation.clamp(-ELEVATION_MAX, ELEVATION_MAX);
        }
    }

    // --- Zoom ---------------------------------------------------------------
    // Zoom factor scales with distance: gentle near the surface, faster at
    // interplanetary range. We lerp in log-space between min and max distance.
    let log_min = focus.min_distance.ln();
    let log_max = DISTANCE_MAX.ln();
    let log_cur = focus.target_distance.ln();
    let t = ((log_cur - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
    let zoom_factor = ZOOM_FACTOR_MIN + (ZOOM_FACTOR_MAX - ZOOM_FACTOR_MIN) * t;

    for event in scroll_events.read() {
        let raw = event.y as f64;
        let ticks = match event.unit {
            MouseScrollUnit::Line => raw * 25.0,
            MouseScrollUnit::Pixel => raw,
        };
        let multiplier = (1.0 - zoom_factor * ticks).max(0.01);
        focus.target_distance =
            (focus.target_distance * multiplier).clamp(focus.min_distance, DISTANCE_MAX);
    }
}

/// Smoothly interpolates `distance` toward `target_distance` in log-space.
///
/// Log-space interpolation means the same lerp factor produces equal *proportional*
/// change at every scale — zooming from 1 AU to 0.5 AU feels the same as
/// zooming from 1000 km to 500 km.
fn camera_zoom_interpolation_system(time: Res<Time>, mut focus: ResMut<CameraFocus>) {
    const SMOOTHING_SPEED: f64 = 10.0; // higher = snappier

    let dt = time.delta_secs_f64();
    let t = (1.0 - (-SMOOTHING_SPEED * dt).exp()).clamp(0.0, 1.0);

    let log_current = focus.distance.ln();
    let log_target = focus.target_distance.ln();
    let log_new = log_current + (log_target - log_current) * t;
    focus.distance = log_new.exp().clamp(focus.min_distance, DISTANCE_MAX);
}

/// Updates `min_distance` based on the focused body's radius so the camera
/// cannot zoom inside the body's surface.
fn camera_min_distance_system(
    mut focus: ResMut<CameraFocus>,
    bodies: Query<&crate::rendering::CelestialBody>,
    ghosts: Query<&crate::flight_plan_view::GhostBody>,
    ships: Query<(), With<crate::rendering::PlayerShip>>,
) {
    let min = if let Some(target) = focus.target {
        if let Ok(body) = bodies.get(target) {
            (body.radius_m * SURFACE_MARGIN).max(DISTANCE_MIN_DEFAULT)
        } else if let Ok(ghost) = ghosts.get(target) {
            (ghost.radius_m * SURFACE_MARGIN).max(DISTANCE_MIN_DEFAULT)
        } else if ships.get(target).is_ok() {
            SHIP_MIN_DISTANCE_M
        } else {
            DISTANCE_MIN_DEFAULT
        }
    } else {
        DISTANCE_MIN_DEFAULT
    };
    focus.min_distance = min;
    if focus.target_distance < min {
        focus.target_distance = min;
    }
}

/// Smoothly decays `focus_offset` toward zero so the camera glides to the new
/// target after a focus switch.
fn camera_focus_offset_decay_system(time: Res<Time>, mut focus: ResMut<CameraFocus>) {
    const TRANSITION_SPEED: f64 = 6.0; // higher = faster snap

    if focus.focus_offset == Vec3::ZERO {
        return;
    }

    let dt = time.delta_secs_f64();
    let t = (1.0 - (-TRANSITION_SPEED * dt).exp()) as f32;
    focus.focus_offset *= 1.0 - t;

    // Snap to zero when close enough to avoid perpetual micro-drift.
    if focus.focus_offset.length_squared() < 1e-12 {
        focus.focus_offset = Vec3::ZERO;
    }
}

/// Computes the camera [`Transform`] from [`CameraFocus`] and the target's world position.
///
/// In **ship view**, the camera builds a local frame `(right, up, forward)`
/// from the ship state and its dominant body, so that rotation feels natural
/// regardless of where the ship is in its orbit:
/// - **Free**: `up = radial_out`, `forward = horizon-projected prograde`
/// - **Orbital**: `up = orbital plane normal`, `forward = prograde`
///
/// In **map view** (and any other case where the ship state isn't available),
/// the basis falls back to world axes: `up = +Y`, `forward = +Z`,
/// `right = +X`, which preserves the original spherical orbit behaviour.
///
/// In all cases the offset is `cos(el)·sin(az)·right + sin(el)·up + cos(el)·cos(az)·forward`,
/// scaled to render units, then `looking_at(target)` orients the camera with
/// the chosen `up` as the world-up reference.
pub fn camera_transform_system(
    focus: Res<CameraFocus>,
    view: Res<ViewMode>,
    mode: Res<ShipCameraMode>,
    sim: Option<Res<SimulationState>>,
    body_states: Res<FrameBodyStates>,
    target_query: Query<(&Transform, Option<&CameraTargetOffset>), Without<OrbitCamera>>,
    mut camera_query: Query<&mut Transform, (With<OrbitCamera>, With<ActiveCamera>)>,
) {
    let Ok(mut camera_transform) = camera_query.single_mut() else {
        return;
    };

    let scale = match *view {
        ViewMode::Map => crate::coords::MAP_SCALE,
        ViewMode::Ship => crate::coords::SHIP_SCALE,
    };

    // Resolve the target's pivot in world space. If the target carries a
    // `CameraTargetOffset` (e.g. the player ship's CoM), apply it through the
    // entity's rotation so the pivot tracks the entity's orientation.
    let target_pos: Vec3 = focus
        .target
        .and_then(|entity| target_query.get(entity).ok())
        .map(|(t, offset)| {
            let local = offset.copied().unwrap_or_default().0;
            t.translation + t.rotation * local
        })
        .unwrap_or(Vec3::ZERO)
        + focus.focus_offset;

    // Pick a local basis. In ship view we derive it from the ship's gravity
    // frame so the planet stays "down" as the ship orbits. Otherwise fall
    // back to world axes (the original behaviour, used by the map view).
    let basis = if *view == ViewMode::Ship
        && let Some(sim) = sim.as_deref()
        && let Some(states) = body_states.states.as_deref()
    {
        let ship_state = sim.simulation.ship_state();
        let bodies = sim.simulation.bodies();
        let ref_id = find_reference_body(ship_state.position, bodies, states);
        let body = &states[ref_id];
        ship_camera_basis(
            *mode,
            ship_state.position - body.position,
            ship_state.velocity - body.velocity,
        )
    } else {
        CameraBasis {
            right: Vec3::X,
            up: Vec3::Y,
            forward: Vec3::Z,
        }
    };

    let distance = (focus.distance * scale) as f32;
    let cos_el = focus.elevation.cos();
    let local = Vec3::new(
        cos_el * focus.azimuth.sin(),
        focus.elevation.sin(),
        cos_el * focus.azimuth.cos(),
    );
    let offset =
        (basis.right * local.x + basis.up * local.y + basis.forward * local.z) * distance;

    let camera_pos = target_pos + offset;
    *camera_transform =
        Transform::from_translation(camera_pos).looking_at(target_pos, basis.up);
}

/// Local camera basis. `right × up = forward` (right-handed), so at
/// `azimuth = 0, elevation = 0` the camera sits at `target + forward * distance`.
struct CameraBasis {
    right: Vec3,
    up: Vec3,
    forward: Vec3,
}

/// Build the ship-view local basis from the ship's body-relative state.
///
/// `r` and `v_rel` are body-relative position and velocity in physics units;
/// only their directions matter, so `f64 → f32` cast is safe after normalization.
fn ship_camera_basis(
    mode: ShipCameraMode,
    r: bevy::math::DVec3,
    v_rel: bevy::math::DVec3,
) -> CameraBasis {
    let radial = r.normalize().as_vec3();
    let h = r.cross(v_rel);

    match mode {
        ShipCameraMode::Free => {
            // Forward = prograde projected onto the horizon plane (radial-perpendicular).
            // Falls back to an arbitrary perpendicular of `up` when velocity is purely
            // radial — rare in practice but possible at periapsis of a radial trajectory.
            let v = v_rel.as_vec3();
            let proj = v - radial * v.dot(radial);
            let forward = if proj.length_squared() > 1e-6 {
                proj.normalize()
            } else {
                radial.any_orthonormal_pair().0
            };
            // Right-handed: right × up = forward, so right = up × forward.
            let right = radial.cross(forward).normalize();
            CameraBasis {
                right,
                up: radial,
                forward,
            }
        }
        ShipCameraMode::Orbital => {
            let up = if h.length_squared() > 1e-6 {
                h.normalize().as_vec3()
            } else {
                radial
            };
            let v = v_rel.as_vec3();
            let forward = if v.length_squared() > 1e-6 {
                let proj = v - up * v.dot(up);
                if proj.length_squared() > 1e-6 {
                    proj.normalize()
                } else {
                    up.any_orthonormal_pair().0
                }
            } else {
                up.any_orthonormal_pair().0
            };
            let right = up.cross(forward).normalize();
            CameraBasis { right, up, forward }
        }
    }
}

/// Find the body whose sphere of influence contains `ship_pos` and is
/// smallest among such bodies — the same rule the patched-conics propagator
/// uses to pick an anchor. The star (infinite SOI) is the fallback.
fn find_reference_body(
    ship_pos: bevy::math::DVec3,
    bodies: &[BodyDefinition],
    states: &[BodyState],
) -> usize {
    let mut best: Option<(usize, f64)> = None;
    for body in bodies {
        let dist_sq = (ship_pos - states[body.id].position).length_squared();
        if dist_sq < body.soi_radius_m * body.soi_radius_m {
            match best {
                None => best = Some((body.id, body.soi_radius_m)),
                Some((_, soi)) if body.soi_radius_m < soi => {
                    best = Some((body.id, body.soi_radius_m));
                }
                _ => {}
            }
        }
    }
    // Fallback: the star (infinite SOI) is always a match, but be defensive
    // in case the body list is empty for any reason.
    best.map(|(id, _)| id).unwrap_or(0)
}
