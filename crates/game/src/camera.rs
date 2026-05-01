use bevy::camera::visibility::RenderLayers;
use bevy::input::mouse::{AccumulatedMouseMotion, MouseScrollUnit, MouseWheel};
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy_egui::EguiContexts;
use thalos_physics::types::{BodyDefinition, BodyId, BodyState};
use thalos_planet_rendering::space_camera_post_stack;

use crate::coords::{MAP_LAYER, RenderGhostFocus, SHIP_LAYER};
use crate::rendering::{CelestialBody, FrameBodyStates, PlayerShip, SimulationState};
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
                    camera_focus_transition_system,
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

/// Semantic camera focus shared across map and ship views.
///
/// This deliberately does not use body or ship ECS entities as the shared
/// identity. Map-view proxies and ship-view real entities are different
/// worlds; systems resolve this target into their own local entity/transform.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CameraFocusTarget {
    #[default]
    None,
    Body(BodyId),
    Ship,
    /// Map-only transient focus for future encounter projections.
    Ghost(RenderGhostFocus),
}

/// The camera orbits around `target` using spherical coordinates.
///
/// Distance is in metres, stored as f64 to cover the full range from
/// 100 km (low orbit) to ~67 AU without precision loss.
/// Azimuth and elevation are in radians.
///
/// Zoom is smoothed: scroll input sets `target_distance` and each frame
/// `distance` interpolates toward it in log-space for scale-independent feel.
#[derive(Resource)]
pub struct CameraFocus {
    /// Semantic target to orbit around.
    pub target: CameraFocusTarget,
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
    /// Physics-space (heliocentric, metres, f64) position the
    /// [`RenderOrigin`](crate::coords::RenderOrigin) sat at when the
    /// current focus transition began. While `Some`, the origin
    /// interpolates in f64 from this point to the new focus target's
    /// physics position over [`FOCUS_TRANSITION_DURATION_S`]. `None`
    /// when no transition is active — origin tracks the focus target
    /// directly.
    ///
    /// Stored in physics space (rather than as a render-space `Vec3`
    /// offset) so the camera never sits at large render-unit
    /// coordinates mid-switch — at MAP_SCALE the old visual position
    /// of a distant body can be 1e6+ RU, which collapses
    /// `looking_at`'s `(target − camera).normalize()` to f32 ulp
    /// noise. With the origin interpolating in f64, both the camera
    /// and its target stay near render-space (0,0,0) throughout.
    pub transition_origin_start: Option<DVec3>,
    /// Azimuth at the moment the current transition began. The renderer
    /// reads `azimuth` interpolated from this value toward the field's
    /// current value across the transition, so a focus pick that also
    /// retargets the camera (e.g. body-tree pick → Sun-side aim) pans
    /// smoothly instead of snapping. Only valid while `transition_origin_start`
    /// is `Some`.
    pub transition_azimuth_start: f32,
    /// Elevation at the moment the current transition began. See
    /// [`Self::transition_azimuth_start`] for the rationale.
    pub transition_elevation_start: f32,
    /// Seconds elapsed since the current transition began. Reset on each
    /// focus switch.
    pub transition_elapsed_s: f64,
}

impl Default for CameraFocus {
    fn default() -> Self {
        Self {
            target: CameraFocusTarget::None,
            distance: 5e11, // ~3.3 AU, sees inner system
            target_distance: 5e11,
            azimuth: 0.0,
            elevation: 0.3, // slight downward tilt so the horizon is visible
            min_distance: DISTANCE_MIN_DEFAULT,
            transition_origin_start: None,
            transition_azimuth_start: 0.0,
            transition_elevation_start: 0.0,
            transition_elapsed_s: 0.0,
        }
    }
}

impl CameraFocus {
    /// Begin a smooth transition to `target`. `current_origin` is the
    /// physics-space position the [`RenderOrigin`](crate::coords::RenderOrigin)
    /// sits at right now (typically the previous focus body's heliocentric
    /// position, possibly already mid-interpolation if the user retargets
    /// during a transition). The origin will interpolate in f64 from this
    /// point to the new target's physics position over
    /// [`FOCUS_TRANSITION_DURATION_S`] seconds regardless of distance, so
    /// the camera never sits at large render-unit coordinates during the
    /// switch.
    ///
    /// Preserves the current zoom (`target_distance`). Callers that want
    /// to also frame the new body to a comparable on-screen size should
    /// follow up with [`Self::frame_for_radius`].
    pub fn focus_on(&mut self, target: CameraFocusTarget, current_origin: DVec3) {
        // Capture *effective* (mid-transition) az/el so a retarget while
        // a transition is already in flight continues smoothly from where
        // the camera currently appears, not from the previous target's
        // stored values.
        let start_az = self.effective_azimuth();
        let start_el = self.effective_elevation();
        self.transition_origin_start = Some(current_origin);
        self.transition_azimuth_start = start_az;
        self.transition_elevation_start = start_el;
        self.transition_elapsed_s = 0.0;
        self.target = target;
    }

    pub fn focus_on_body(&mut self, body_id: BodyId, current_origin: DVec3) {
        self.focus_on(CameraFocusTarget::Body(body_id), current_origin);
    }

    pub fn focus_on_ship(&mut self, current_origin: DVec3) {
        self.focus_on(CameraFocusTarget::Ship, current_origin);
    }

    /// Set `target_distance` to a body-sized framing distance — bodies
    /// sharing a radius land at the same zoom, so on-screen size stays
    /// comparable across the system. Body-tree picks call this; passive
    /// refocus events (double-click, ghost retirement) do not, so they
    /// preserve whatever zoom the user had.
    pub fn frame_for_radius(&mut self, radius_m: f64) {
        self.target_distance = (radius_m * FOCUS_FRAMING_RADII).max(DISTANCE_MIN_DEFAULT);
    }

    /// Set [`azimuth`](Self::azimuth) and [`elevation`](Self::elevation)
    /// so the camera-to-target offset points along `world_dir` — i.e. the
    /// camera ends up sitting at `target + world_dir * distance`. Used to
    /// place the camera on the lit side of a body (Sun-direction) when the
    /// user picks it from the body tree.
    ///
    /// Only meaningful in map view, where the camera basis is the world
    /// axes. Ship view uses a gravity-aligned basis that this helper
    /// doesn't translate to.
    pub fn aim_from(&mut self, world_dir: Vec3) {
        let dir = world_dir.normalize_or_zero();
        if dir == Vec3::ZERO {
            return;
        }
        self.elevation = dir.y.asin();
        self.azimuth = dir.x.atan2(dir.z);
    }

    /// Azimuth as it appears this frame. While a focus transition is
    /// active, lerps from [`Self::transition_azimuth_start`] toward
    /// [`Self::azimuth`] using the same eased curve as the origin lerp;
    /// otherwise returns [`Self::azimuth`] directly. Shortest-arc wrapped
    /// so a 350°→10° transition pans 20° forward, not 340° back.
    pub fn effective_azimuth(&self) -> f32 {
        if self.transition_origin_start.is_none() {
            return self.azimuth;
        }
        let t = focus_transition_progress(self) as f32;
        let delta = wrap_pi(self.azimuth - self.transition_azimuth_start);
        self.transition_azimuth_start + delta * t
    }

    /// Elevation as it appears this frame — see [`Self::effective_azimuth`].
    /// No wrap needed: elevation is clamped to ±89°.
    pub fn effective_elevation(&self) -> f32 {
        if self.transition_origin_start.is_none() {
            return self.elevation;
        }
        let t = focus_transition_progress(self) as f32;
        self.transition_elevation_start + (self.elevation - self.transition_elevation_start) * t
    }
}

/// Wrap `angle` to `(-π, π]` for shortest-arc azimuth interpolation.
fn wrap_pi(angle: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let mut a = angle % TAU;
    if a > PI {
        a -= TAU;
    } else if a < -PI {
        a += TAU;
    }
    a
}

const DISTANCE_MIN_DEFAULT: f64 = 1e5; // 100 km
const MAP_DISTANCE_MAX: f64 = 1e13; // ~67 AU
/// Farthest the ship-view chase camera may pull back from the vessel.
/// Map view handles orbital/system-scale framing; ship view stays local.
const SHIP_VIEW_MAX_DISTANCE_M: f64 = 5_000.0;
/// Camera stops at 3× the body's radius (comfortable viewing distance).
const SURFACE_MARGIN: f64 = 3.0;
/// Closest the camera may zoom to the player ship in ship view (metres).
/// Small enough to put the camera a few metres off the hull.
const SHIP_MIN_DISTANCE_M: f64 = 5.0;
/// Closest the map camera may zoom to the player ship (metres).
/// The ship is represented as a screen-stable marker in map view, so the
/// ship-view hull clamp is far too close for the orbit-scale camera.
const SHIP_MAP_MIN_DISTANCE_M: f64 = DISTANCE_MIN_DEFAULT;
/// Duration of a focus-switch transition, regardless of distance between
/// bodies. Tuned for snappy-but-not-jarring camera handoff.
pub const FOCUS_TRANSITION_DURATION_S: f64 = 0.8;
/// Multiple of body radius used as the framing distance when switching
/// focus. ~10× gives an establishing-shot view — body clearly visible in
/// frame without dominating it. Must stay above [`SURFACE_MARGIN`] so
/// `camera_min_distance_system` doesn't clamp the framing back up.
const FOCUS_FRAMING_RADII: f64 = 10.0;

fn max_distance_for_view(view: ViewMode) -> f64 {
    match view {
        ViewMode::Map => MAP_DISTANCE_MAX,
        ViewMode::Ship => SHIP_VIEW_MAX_DISTANCE_M,
    }
}

fn distance_bounds_for_view(view: ViewMode, min_distance: f64) -> (f64, f64) {
    let min = min_distance.max(f64::MIN_POSITIVE);
    let max = max_distance_for_view(view).max(min);
    (min, max)
}

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
        map_cam.insert((ActiveCamera, IsDefaultUiCamera));
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
        ship_cam.insert((ActiveCamera, IsDefaultUiCamera));
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
///
/// Suppressed while egui is consuming pointer input — without this guard,
/// dragging an egui window would simultaneously rotate the camera, and
/// scrolling over a window would zoom both.
pub fn camera_input_system(
    block: Res<BlockCameraInput>,
    mut contexts: EguiContexts,
    view: Res<ViewMode>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mut scroll_events: MessageReader<MouseWheel>,
    mut focus: ResMut<CameraFocus>,
) {
    const ROTATION_SENSITIVITY: f32 = 0.005; // rad per pixel
    const ZOOM_FACTOR_MIN: f64 = 0.005; // near surface / local system
    const ZOOM_FACTOR_MAX: f64 = 0.01; // interplanetary scale
    const ELEVATION_MAX: f32 = 89.0_f32.to_radians();

    let egui_wants_pointer = contexts
        .ctx_mut()
        .map(|ctx| ctx.wants_pointer_input())
        .unwrap_or(false);

    // --- Rotation -----------------------------------------------------------
    // Suppressed while a maneuver element is hovered or being dragged, or
    // while egui is handling the pointer (e.g. dragging a panel).
    if mouse_buttons.pressed(MouseButton::Left) && !block.0 && !egui_wants_pointer {
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
    let (min_distance, max_distance) = distance_bounds_for_view(*view, focus.min_distance);
    let log_min = min_distance.ln();
    let log_max = max_distance.ln();
    let log_cur = focus.target_distance.ln();
    let t = if log_max > log_min {
        ((log_cur - log_min) / (log_max - log_min)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let zoom_factor = ZOOM_FACTOR_MIN + (ZOOM_FACTOR_MAX - ZOOM_FACTOR_MIN) * t;

    // Drain scroll events even when blocked so they don't carry into the
    // next frame and cause a delayed zoom after the cursor leaves egui.
    for event in scroll_events.read() {
        if egui_wants_pointer {
            continue;
        }
        let raw = event.y as f64;
        let ticks = match event.unit {
            MouseScrollUnit::Line => raw * 25.0,
            MouseScrollUnit::Pixel => raw,
        };
        let multiplier = (1.0 - zoom_factor * ticks).max(0.01);
        focus.target_distance =
            (focus.target_distance * multiplier).clamp(min_distance, max_distance);
    }
}

/// Smoothly interpolates `distance` toward `target_distance` in log-space.
///
/// Log-space interpolation means the same lerp factor produces equal *proportional*
/// change at every scale — zooming from 1 AU to 0.5 AU feels the same as
/// zooming from 1000 km to 500 km.
fn camera_zoom_interpolation_system(
    time: Res<Time>,
    view: Res<ViewMode>,
    mut focus: ResMut<CameraFocus>,
) {
    const SMOOTHING_SPEED: f64 = 10.0; // higher = snappier

    let dt = time.delta_secs_f64();
    let t = (1.0 - (-SMOOTHING_SPEED * dt).exp()).clamp(0.0, 1.0);

    let log_current = focus.distance.ln();
    let log_target = focus.target_distance.ln();
    let log_new = log_current + (log_target - log_current) * t;
    let (min_distance, max_distance) = distance_bounds_for_view(*view, focus.min_distance);
    focus.distance = log_new.exp().clamp(min_distance, max_distance);
}

/// Updates `min_distance` based on the focused body's radius so the camera
/// cannot zoom inside the body's surface.
fn camera_min_distance_system(
    mut focus: ResMut<CameraFocus>,
    view: Res<ViewMode>,
    bodies: Query<&crate::rendering::CelestialBody>,
    ghosts: Query<&crate::flight_plan_view::GhostBody>,
) {
    let min = match focus.target {
        CameraFocusTarget::Body(body_id) => bodies
            .iter()
            .find(|body| body.body_id == body_id)
            .map(|body| (body.radius_m * SURFACE_MARGIN).max(DISTANCE_MIN_DEFAULT))
            .unwrap_or(DISTANCE_MIN_DEFAULT),
        CameraFocusTarget::Ghost(ghost_focus) => ghosts
            .iter()
            .find(|ghost| ghost_focus.matches(ghost.body_id, ghost.encounter_epoch))
            .map(|ghost| (ghost.radius_m * SURFACE_MARGIN).max(DISTANCE_MIN_DEFAULT))
            .unwrap_or(DISTANCE_MIN_DEFAULT),
        CameraFocusTarget::Ship => match *view {
            ViewMode::Map => SHIP_MAP_MIN_DISTANCE_M,
            ViewMode::Ship => SHIP_MIN_DISTANCE_M,
        },
        CameraFocusTarget::None => DISTANCE_MIN_DEFAULT,
    };
    focus.min_distance = min;
    let (min_distance, max_distance) = distance_bounds_for_view(*view, min);
    focus.target_distance = focus.target_distance.clamp(min_distance, max_distance);
}

/// Advances the focus-transition timer and clears
/// [`CameraFocus::transition_origin_start`] when the transition is
/// complete. The origin's actual interpolation is driven by
/// `update_render_origin` in `rendering.rs`, which reads this timer
/// each frame — keeping the lerp in physics space (DVec3) rather than
/// re-deriving a render-space `focus_offset` here, so the camera
/// never sits at large render-unit coordinates during the switch.
///
/// Fixed duration (rather than an exponential decay) means near and
/// distant focus switches feel equally responsive: a Sun↔Acheron jump
/// completes in the same 0.8 s as a Moon↔Earth jump.
fn camera_focus_transition_system(time: Res<Time>, mut focus: ResMut<CameraFocus>) {
    if focus.transition_origin_start.is_none() {
        return;
    }

    focus.transition_elapsed_s += time.delta_secs_f64();
    if focus.transition_elapsed_s >= FOCUS_TRANSITION_DURATION_S {
        focus.transition_origin_start = None;
        focus.transition_elapsed_s = 0.0;
    }
}

/// Eased progress of the active focus transition in `[0.0, 1.0]`.
/// Returns `1.0` when no transition is active so `update_render_origin`
/// lerps directly to the focus target. Ease-out cubic — most of the
/// visual movement lands in the first ~30 % of the duration, the last
/// fraction settles gently.
pub fn focus_transition_progress(focus: &CameraFocus) -> f64 {
    if focus.transition_origin_start.is_none() {
        return 1.0;
    }
    let t = (focus.transition_elapsed_s / FOCUS_TRANSITION_DURATION_S).clamp(0.0, 1.0);
    1.0 - (1.0 - t).powi(3)
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
    body_targets: Query<(&CelestialBody, &Transform), Without<OrbitCamera>>,
    ship_targets: Query<
        (&Transform, Option<&CameraTargetOffset>),
        (With<PlayerShip>, Without<OrbitCamera>),
    >,
    ghost_targets: Query<
        (&crate::flight_plan_view::GhostBody, &Transform),
        (
            With<crate::flight_plan_view::GhostBody>,
            Without<OrbitCamera>,
        ),
    >,
    mut camera_query: Query<&mut Transform, (With<OrbitCamera>, With<ActiveCamera>)>,
) {
    let Ok(mut camera_transform) = camera_query.single_mut() else {
        return;
    };

    let scale = match *view {
        ViewMode::Map => crate::coords::MAP_SCALE,
        ViewMode::Ship => crate::coords::SHIP_SCALE,
    };

    // Resolve the target's pivot in world space.
    //
    // - When a focus transition is active, `RenderOrigin` is mid-lerp
    //   between the old and new focus positions in physics space. The
    //   camera follows that moving origin, which sits at `Vec3::ZERO`
    //   in render space by definition; bodies (including the focus
    //   target) slide past as the origin sweeps. We deliberately
    //   ignore the focus entity's transform here so the camera never
    //   anchors to its non-zero render-space position mid-switch —
    //   that's exactly the failure mode the structural fix prevents.
    // - When settled, the focus entity's transform sits at the
    //   render-space origin (origin tracks it directly), so we read
    //   it normally and apply any per-target pivot offset (e.g. the
    //   player ship's mass-weighted CoM).
    let target_pos: Vec3 = if focus.transition_origin_start.is_some() {
        Vec3::ZERO
    } else {
        match focus.target {
            CameraFocusTarget::Body(body_id) => body_targets
                .iter()
                .find(|(body, _)| body.body_id == body_id)
                .map(|(_, t)| t.translation)
                .unwrap_or(Vec3::ZERO),
            CameraFocusTarget::Ship => {
                if *view == ViewMode::Ship {
                    ship_targets
                        .single()
                        .ok()
                        .map(|(t, offset)| {
                            let local = offset.copied().unwrap_or_default().0;
                            t.translation + t.rotation * local
                        })
                        .unwrap_or(Vec3::ZERO)
                } else {
                    Vec3::ZERO
                }
            }
            CameraFocusTarget::Ghost(ghost_focus) => ghost_targets
                .iter()
                .find(|(ghost, _)| ghost_focus.matches(ghost.body_id, ghost.encounter_epoch))
                .map(|(_, t)| t.translation)
                .unwrap_or(Vec3::ZERO),
            CameraFocusTarget::None => Vec3::ZERO,
        }
    };

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
    let azimuth = focus.effective_azimuth();
    let elevation = focus.effective_elevation();
    let cos_el = elevation.cos();
    let local = Vec3::new(
        cos_el * azimuth.sin(),
        elevation.sin(),
        cos_el * azimuth.cos(),
    );
    let offset = (basis.right * local.x + basis.up * local.y + basis.forward * local.z) * distance;

    let camera_pos = target_pos + offset;
    *camera_transform = Transform::from_translation(camera_pos).looking_at(target_pos, basis.up);
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
pub(crate) fn find_reference_body(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ship_view_uses_local_zoom_cap() {
        assert_eq!(
            max_distance_for_view(ViewMode::Ship),
            SHIP_VIEW_MAX_DISTANCE_M
        );
        assert!(max_distance_for_view(ViewMode::Ship) < max_distance_for_view(ViewMode::Map));
    }

    #[test]
    fn distance_bounds_never_invert() {
        let (min, max) = distance_bounds_for_view(ViewMode::Ship, DISTANCE_MIN_DEFAULT);
        assert_eq!(min, DISTANCE_MIN_DEFAULT);
        assert_eq!(max, DISTANCE_MIN_DEFAULT);
    }
}
