use bevy::camera::visibility::RenderLayers;
use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy_egui::EguiContexts;
use big_space::prelude::CellCoord;
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::types::{BodyDefinition, BodyId, BodyState};
use thalos_physics_local::TerrainSurfaceRegistry;
use thalos_planet_rendering::space_camera_post_stack;
use thalos_terrain::{DynamicSurfaceState, PlanetSurface};
use thalos_terrain_render::rendered_height_m;

/// `tile_lod_m` passed to `rendered_height_m` for the camera boom's
/// ray-vs-terrain check. The boom rarely runs in tight sub-metre proximity
/// to the ground, so 1 m is a sensible LOD floor — it engages the same
/// 5-octave detail as the renderer at fine LOD without paying for sub-metre
/// over-sampling.
const CAMERA_HEIGHT_QUERY_TILE_LOD_M: f32 = 1.0;

use crate::coords::{MAP_LAYER, RenderGhostFocus, SHIP_LAYER};
use crate::freecam::FreeCam;
use crate::rendering::{CelestialBody, PlayerShip, SimulationState, SolarSystemState};
use crate::view::ViewMode;

/// Plugin that registers the orbit camera systems and spawns the camera entity.
pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<ShipCamera>::default())
            .init_resource::<BlockCameraInput>()
            .init_resource::<ShipCameraMode>()
            .init_resource::<CameraCollisionState>()
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
///
/// Extracted to the render world so the scene-depth-copy node
/// (`rendering::scene_depth::CopySceneDepthNode`) can filter its
/// `ViewQuery` to only the ship-view, skipping the map camera.
#[derive(Component, Clone, ExtractComponent)]
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

#[derive(Resource, Default)]
pub(crate) struct CameraCollisionState {
    ship_boom_m: Option<f64>,
    target: CameraFocusTarget,
    view: ViewMode,
}

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
    PlayerController,
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
const CAMERA_ELEVATION_MAX: f32 = 89.0_f32.to_radians();
/// Duration of a focus-switch transition, regardless of distance between
/// bodies. Tuned for snappy-but-not-jarring camera handoff.
pub const FOCUS_TRANSITION_DURATION_S: f64 = 0.8;
/// Multiple of body radius used as the framing distance when switching
/// focus. ~10× gives an establishing-shot view — body clearly visible in
/// frame without dominating it. Must stay above [`SURFACE_MARGIN`] so
/// `camera_min_distance_system` doesn't clamp the framing back up.
const FOCUS_FRAMING_RADII: f64 = 10.0;
/// Log-distance change per logical scroll line.
///
/// This is deliberately a log-space step instead of a per-frame multiplier:
/// wheel / trackpad deltas are frame-accumulated input, so applying them in
/// log space makes the result independent of how many frames the OS batches
/// the same physical scroll across.
const ZOOM_LOG_STEP_PER_LINE: f64 = 0.26;
/// Base log-space catch-up speed for small zoom gaps.
const ZOOM_SMOOTHING_SPEED_MIN: f64 = 28.0;
/// Catch-up speed for large zoom gaps. This makes sustained scrolls feel
/// punchy instead of dragging through a uniform interpolation tail.
const ZOOM_SMOOTHING_SPEED_MAX: f64 = 90.0;
/// Log-distance error that reaches the fast smoothing speed. `ln(2)` means
/// a target one octave away or more closes at the aggressive rate.
const ZOOM_FAST_ERROR: f64 = std::f64::consts::LN_2;

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

fn zoom_target_after_scroll(
    target_distance: f64,
    min_distance: f64,
    max_distance: f64,
    scroll_lines: f64,
) -> f64 {
    let log_min = min_distance.ln();
    let log_max = max_distance.ln();
    let log_new = target_distance.ln() - ZOOM_LOG_STEP_PER_LINE * scroll_lines;
    if log_new <= log_min {
        min_distance
    } else if log_new >= log_max {
        max_distance
    } else {
        log_new.exp()
    }
}

fn zoom_smoothing_speed(log_error: f64) -> f64 {
    let t = (log_error.abs() / ZOOM_FAST_ERROR).clamp(0.0, 1.0);
    ZOOM_SMOOTHING_SPEED_MIN + (ZOOM_SMOOTHING_SPEED_MAX - ZOOM_SMOOTHING_SPEED_MIN) * t.sqrt()
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

pub(crate) fn spawn_camera(mut commands: Commands, view: Res<ViewMode>) {
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
        Camera3d {
            // Add COPY_SRC so the scene-depth-copy render-graph node
            // (`CopySceneDepthNode` in `rendering::scene_depth`) can copy
            // the main depth attachment into our sampleable depth Image
            // each frame.
            depth_texture_usages: (bevy::render::render_resource::TextureUsages::RENDER_ATTACHMENT
                | bevy::render::render_resource::TextureUsages::COPY_SRC)
                .into(),
            ..default()
        },
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
    input: Res<GameInputIntent>,
    mut contexts: EguiContexts,
    view: Res<ViewMode>,
    mut mode: ResMut<ShipCameraMode>,
) {
    if *view != ViewMode::Ship || !input.cycle_ship_camera {
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
    ui_pointer_gate: Res<crate::hud::UiPointerGate>,
    view: Res<ViewMode>,
    input: Res<GameInputIntent>,
    freecam: Res<FreeCam>,
    debug_surface_teleport: Option<Res<crate::debug::DebugSurfaceTeleport>>,
    mut focus: ResMut<CameraFocus>,
) {
    if freecam.active {
        return;
    }
    const ROTATION_SENSITIVITY: f32 = 0.005; // rad per pixel

    let egui_wants_pointer = contexts
        .ctx_mut()
        .map(|ctx| ctx.wants_pointer_input())
        .unwrap_or(false);
    // Bevy-UI interactive HUD elements (the new HUD nav buttons, etc.)
    // also block the camera so clicking a button doesn't double up as a
    // camera drag/zoom.
    let ui_pointer_busy = egui_wants_pointer || ui_pointer_gate.hovered;
    let debug_surface_armed = debug_surface_teleport
        .as_deref()
        .and_then(|teleport| teleport.armed_body)
        .is_some();

    // --- Rotation -----------------------------------------------------------
    // Suppressed while a maneuver element is hovered or being dragged, or
    // while egui is handling the pointer (e.g. dragging a panel).
    if input.primary_pressed && !block.0 && !ui_pointer_busy && !debug_surface_armed {
        let delta = input.camera_motion;
        if delta != Vec2::ZERO {
            focus.azimuth += delta.x * ROTATION_SENSITIVITY;
            focus.elevation -= delta.y * ROTATION_SENSITIVITY;
            focus.elevation = focus
                .elevation
                .clamp(-CAMERA_ELEVATION_MAX, CAMERA_ELEVATION_MAX);
        }
    }

    // --- Zoom ---------------------------------------------------------------
    let (min_distance, max_distance) = distance_bounds_for_view(*view, focus.min_distance);
    if !ui_pointer_busy && input.camera_wheel.y != 0.0 {
        focus.target_distance = zoom_target_after_scroll(
            focus.target_distance,
            min_distance,
            max_distance,
            input.camera_wheel.y as f64,
        );
    }
}

/// Smoothly interpolates `distance` toward `target_distance` in log-space.
///
/// Log-space interpolation means the same lerp factor produces equal *proportional*
/// change at every scale — zooming from 1 AU to 0.5 AU feels the same as
/// zooming from 1000 km to 500 km.
fn camera_zoom_interpolation_system(
    time: Res<Time<Real>>,
    view: Res<ViewMode>,
    mut focus: ResMut<CameraFocus>,
) {
    // Wall-clock so smoothing continues under sim pause (Time<Virtual>
    // pause); the camera is a view affordance, not part of the sim.
    let dt = time.delta_secs_f64();
    let log_current = focus.distance.ln();
    let log_target = focus.target_distance.ln();
    let speed = zoom_smoothing_speed(log_target - log_current);
    let t = (1.0 - (-speed * dt).exp()).clamp(0.0, 1.0);
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
        CameraFocusTarget::PlayerController => match *view {
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
fn camera_focus_transition_system(time: Res<Time<Real>>, mut focus: ResMut<CameraFocus>) {
    if focus.transition_origin_start.is_none() {
        return;
    }

    // Wall-clock so the focus lerp completes during sim pause.
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

/// Metres the camera spring arm stays away from the blocking surface.
/// Keep this small; large probe radii turn shallow ground hits into violent
/// zoom-ins instead of a modest line-of-sight correction.
const CAMERA_TERRAIN_MARGIN_M: f64 = 8.0;
/// Cinemachine-style "minimum distance from target": do not let near-target
/// terrain collapse the spring arm into a useless extreme close-up.
const CAMERA_COLLISION_MIN_TARGET_DISTANCE_M: f64 = 180.0;
/// How quickly the collision arm pulls inward when the view is obstructed.
const CAMERA_COLLISION_OCCLUDED_SPEED: f64 = 18.0;
/// How quickly the arm returns to the user's requested zoom once clear.
const CAMERA_COLLISION_RECOVERY_SPEED: f64 = 6.0;

/// Coarse-pass samples along the target→camera ray. Combined with
/// `CAMERA_RAY_REFINEMENT_STEPS` binary halvings, the worst-case stop
/// resolution is `ray_length / (CAMERA_RAY_SAMPLES · 2^CAMERA_RAY_REFINEMENT_STEPS)`.
const CAMERA_RAY_SAMPLES: usize = 16;
const CAMERA_RAY_REFINEMENT_STEPS: usize = 6;

#[derive(Debug, Clone, Copy)]
struct TerrainClearance {
    agl_m: f64,
    radial_inertial: DVec3,
    surface_radius_m: f64,
}

fn blocking_surface_height_m(
    surface: &PlanetSurface,
    dynamic_state: &DynamicSurfaceState,
    dir_body: Vec3,
) -> f64 {
    let terrain_height_m = rendered_height_m(
        surface,
        dynamic_state,
        dir_body,
        CAMERA_HEIGHT_QUERY_TILE_LOD_M,
    ) as f64;
    surface
        .static_surface
        .sea_level_m
        .map(|sea_level_m| terrain_height_m.max(sea_level_m as f64))
        .unwrap_or(terrain_height_m)
}

fn terrain_clearance_at_physics_pos(
    body_position: DVec3,
    body_inv_orientation: DQuat,
    body_radius_m: f64,
    surface: Option<(&PlanetSurface, &DynamicSurfaceState)>,
    position: DVec3,
) -> Option<TerrainClearance> {
    let from_body = position - body_position;
    let distance = from_body.length();
    if distance < 1.0 {
        return None;
    }

    let radial_inertial = from_body / distance;
    let dir_body = (body_inv_orientation * radial_inertial)
        .as_vec3()
        .normalize_or_zero();
    if dir_body == Vec3::ZERO {
        return None;
    }

    let surface_height_m = surface
        .map(|(surface, dynamic_state)| blocking_surface_height_m(surface, dynamic_state, dir_body))
        .unwrap_or(0.0);
    let surface_radius_m = body_radius_m + surface_height_m;
    Some(TerrainClearance {
        agl_m: distance - surface_radius_m,
        radial_inertial,
        surface_radius_m,
    })
}

fn clamp_physics_pos_above_terrain(
    body_position: DVec3,
    body_inv_orientation: DQuat,
    body_radius_m: f64,
    surface: Option<(&PlanetSurface, &DynamicSurfaceState)>,
    position: DVec3,
) -> Option<DVec3> {
    let clearance = terrain_clearance_at_physics_pos(
        body_position,
        body_inv_orientation,
        body_radius_m,
        surface,
        position,
    )?;
    if clearance.agl_m >= CAMERA_TERRAIN_MARGIN_M {
        return None;
    }

    Some(
        body_position
            + clearance.radial_inertial * (clearance.surface_radius_m + CAMERA_TERRAIN_MARGIN_M),
    )
}

/// Final surface floor for the camera point itself. This catches body-focused
/// cameras and any ship-camera case where the desired endpoint is below the
/// heightfield even after the boom pass has shortened the line of sight. If a
/// rendered surface is not registered yet, falls back to the authored sphere
/// so the camera still cannot enter the body.
fn clamp_camera_position_above_body_terrain(
    camera_pos: Vec3,
    target_pos: Vec3,
    target_physics_pos: DVec3,
    render_scale: f64,
    body_id: BodyId,
    body_states: &SolarSystemState,
    sim: &SimulationState,
    surfaces: &TerrainSurfaceRegistry,
) -> Option<Vec3> {
    if render_scale <= 0.0 {
        return None;
    }

    let states = body_states.states.as_deref()?;
    let surface = surfaces.get(body_id);
    let dynamic_state = surface
        .as_ref()
        .map(|surface| body_states.dynamic_surface_for(body_id, surface));
    let body = &sim.system.bodies[body_id];
    let body_state = states.get(body_id)?;

    let offset_m = (camera_pos - target_pos).as_dvec3() / render_scale;
    let camera_physics_pos = target_physics_pos + offset_m;
    let corrected_physics_pos = clamp_physics_pos_above_terrain(
        body_state.position,
        body_state.orientation.inverse(),
        body.radius_m,
        surface.as_deref().zip(dynamic_state.as_ref()),
        camera_physics_pos,
    )?;

    Some(target_pos + ((corrected_physics_pos - target_physics_pos) * render_scale).as_vec3())
}

/// Spring-arm collision length against the rendered heightfield, or the
/// authored sphere before a surface is available: cast from `target_pos`
/// toward `camera_pos`, find the first point where the ray dips within
/// `CAMERA_TERRAIN_MARGIN_M` of the surface, and return the corrected boom
/// length. Returns `None` when the boom is clear, the offset is sub-metre, or
/// the body state is not available.
///
/// This is the same idea as KSP's vessel camera and the standard third-person
/// orbit camera in most engines: the camera always retains line-of-sight to
/// the target by shortening the boom, so it correctly dodges mountains and
/// ridges between target and viewer — not just the radial column directly
/// under the camera.
///
/// The render positions are converted back to metres via `render_scale` so
/// this can be reused by any metre-derived view.
fn camera_boom_collision_length_m(
    camera_pos: Vec3,
    target_pos: Vec3,
    target_physics_pos: DVec3,
    render_scale: f64,
    body_id: BodyId,
    body_states: &SolarSystemState,
    sim: &SimulationState,
    surfaces: &TerrainSurfaceRegistry,
) -> Option<f64> {
    if render_scale <= 0.0 {
        return None;
    }

    let states = body_states.states.as_deref()?;
    let surface = surfaces.get(body_id);
    let dynamic_state = surface
        .as_ref()
        .map(|surface| body_states.dynamic_surface_for(body_id, surface));
    let body = &sim.system.bodies[body_id];
    let body_state = states.get(body_id)?;

    let camera_offset = (camera_pos - target_pos).as_dvec3() / render_scale;
    if camera_offset.length_squared() < 1.0 {
        return None;
    }
    let body_inv = body_state.orientation.inverse();

    let surface_with_state = surface.as_deref().zip(dynamic_state.as_ref());

    // AGL of the point `t` of the way from target to camera. Negative = inside terrain.
    let agl = |t: f64| -> f64 {
        terrain_clearance_at_physics_pos(
            body_state.position,
            body_inv,
            body.radius_m,
            surface_with_state,
            target_physics_pos + camera_offset * t,
        )
        .map(|clearance| clearance.agl_m)
        .unwrap_or(f64::MAX)
    };

    let boom_length_m = camera_offset.length();

    let find_safe_t = |margin_m: f64| -> Option<f64> {
        let min_t = (CAMERA_COLLISION_MIN_TARGET_DISTANCE_M / boom_length_m).clamp(0.0, 1.0);
        if min_t >= 1.0 {
            return None;
        }
        if agl(min_t) < margin_m {
            return Some(min_t);
        }

        // Coarse linear pass: find the first sample inside the requested
        // surface margin. `t_safe` tracks the most recent above-margin sample;
        // once we find a blocked one, refine between the two.
        let n = CAMERA_RAY_SAMPLES;
        let mut t_safe = min_t;
        let mut t_block: Option<f64> = None;
        for i in 1..=n {
            let t = (i as f64 / n as f64).max(min_t);
            if agl(t) < margin_m {
                t_block = Some(t);
                break;
            }
            t_safe = t;
        }
        let mut t_high = t_block?;

        // Binary refinement. Invariant: agl(t_safe) ≥ margin > agl(t_high).
        // Holds even when t_safe = 0 is itself below margin — the bisection just
        // converges to 0, parking the camera on top of the target. That degenerate
        // case (ship buried in terrain) has no good camera placement anyway.
        for _ in 0..CAMERA_RAY_REFINEMENT_STEPS {
            let t_mid = (t_safe + t_high) * 0.5;
            if agl(t_mid) < margin_m {
                t_high = t_mid;
            } else {
                t_safe = t_mid;
            }
        }

        Some(t_safe)
    };

    let t_safe = find_safe_t(CAMERA_TERRAIN_MARGIN_M)?;
    Some(t_safe * boom_length_m)
}

fn damped_camera_collision_boom(current_m: f64, target_m: f64, dt_s: f64) -> f64 {
    let speed = if target_m < current_m {
        CAMERA_COLLISION_OCCLUDED_SPEED
    } else {
        CAMERA_COLLISION_RECOVERY_SPEED
    };
    let t = (1.0 - (-speed * dt_s.max(0.0)).exp()).clamp(0.0, 1.0);
    current_m + (target_m - current_m) * t
}

fn clamp_ship_camera_against_terrain(
    camera_pos: Vec3,
    target_pos: Vec3,
    scale: f64,
    body_states: &SolarSystemState,
    sim: &SimulationState,
    surfaces: &TerrainSurfaceRegistry,
    time: &Time<Real>,
    collision: &mut CameraCollisionState,
) -> Vec3 {
    let body_id = sim.simulation.dominant_body();
    let ship_pos = sim.simulation.ship_state().position;
    let desired_offset_m = (camera_pos - target_pos).as_dvec3() / scale;
    let desired_boom_m = desired_offset_m.length();
    if desired_boom_m < 1.0 {
        collision.ship_boom_m = None;
        return camera_pos;
    }

    let target_boom_m = camera_boom_collision_length_m(
        camera_pos,
        target_pos,
        ship_pos,
        scale,
        body_id,
        body_states,
        sim,
        surfaces,
    )
    .unwrap_or(desired_boom_m)
    .clamp(0.0, desired_boom_m);

    let current_boom_m = collision
        .ship_boom_m
        .unwrap_or(desired_boom_m)
        .min(desired_boom_m);
    let next_boom_m =
        damped_camera_collision_boom(current_boom_m, target_boom_m, time.delta_secs_f64())
            .clamp(0.0, desired_boom_m);
    collision.ship_boom_m = Some(next_boom_m);

    let dir = desired_offset_m / desired_boom_m;
    let corrected = target_pos + (dir * next_boom_m * scale).as_vec3();

    // Damping avoids visual snaps. This final endpoint floor keeps the
    // camera's own position outside terrain even while the boom length is
    // easing toward the collision-corrected value.
    clamp_camera_position_above_body_terrain(
        corrected,
        target_pos,
        ship_pos,
        scale,
        body_id,
        body_states,
        sim,
        surfaces,
    )
    .unwrap_or(corrected)
}

fn clamp_body_focus_camera_against_terrain(
    camera_pos: Vec3,
    target_pos: Vec3,
    scale: f64,
    body_id: BodyId,
    body_states: &SolarSystemState,
    sim: &SimulationState,
    surfaces: &TerrainSurfaceRegistry,
) -> Vec3 {
    let Some(states) = body_states.states.as_deref() else {
        return camera_pos;
    };
    let Some(body_state) = states.get(body_id) else {
        return camera_pos;
    };

    clamp_camera_position_above_body_terrain(
        camera_pos,
        target_pos,
        body_state.position,
        scale,
        body_id,
        body_states,
        sim,
        surfaces,
    )
    .unwrap_or(camera_pos)
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
    // Wall-clock so the collision-boom damper keeps smoothing during
    // sim pause; this system only consumes `delta` for that damper, so a
    // single `Time<Real>` covers it.
    time: Res<Time<Real>>,
    freecam: Res<FreeCam>,
    sim: Option<Res<SimulationState>>,
    body_states: Res<SolarSystemState>,
    surfaces: Res<TerrainSurfaceRegistry>,
    mut collision: ResMut<CameraCollisionState>,
    body_targets: Query<(&CelestialBody, &Transform), Without<OrbitCamera>>,
    ship_targets: Query<
        (&Transform, Option<&CameraTargetOffset>, Option<&CellCoord>),
        (With<PlayerShip>, Without<OrbitCamera>),
    >,
    player_targets: Query<
        (&Transform, Option<&CellCoord>),
        (
            With<crate::player_controller::PlayerControllerVisual>,
            Without<OrbitCamera>,
        ),
    >,
    ghost_targets: Query<
        (&crate::flight_plan_view::GhostBody, &Transform),
        (
            With<crate::flight_plan_view::GhostBody>,
            Without<OrbitCamera>,
        ),
    >,
    mut camera_query: Query<
        (&mut Transform, Option<&mut CellCoord>),
        (With<OrbitCamera>, With<ActiveCamera>),
    >,
) {
    if freecam.active {
        return;
    }
    let Ok((mut camera_transform, camera_cell)) = camera_query.single_mut() else {
        return;
    };

    if collision.target != focus.target || collision.view != *view {
        collision.ship_boom_m = None;
        collision.target = focus.target;
        collision.view = *view;
    }

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
    let mut target_cell: Option<CellCoord> = None;
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
                        .map(|(t, offset, cell)| {
                            target_cell = cell.copied();
                            let local = offset.copied().unwrap_or_default().0;
                            t.translation + t.rotation * local
                        })
                        .unwrap_or(Vec3::ZERO)
                } else {
                    Vec3::ZERO
                }
            }
            CameraFocusTarget::PlayerController => {
                if *view == ViewMode::Ship {
                    player_targets
                        .single()
                        .ok()
                        .map(|(t, cell)| {
                            target_cell = cell.copied();
                            t.translation
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

    let mut camera_pos = target_pos + offset;

    // Once the target is settled in render space, keep cameras outside the
    // rendered heightfield. Ship focus gets a KSP-style spring arm plus a
    // final endpoint floor; body focus gets the endpoint floor so terrain
    // close-ups cannot place the camera inside a mountain or crater wall.
    if focus.transition_origin_start.is_none()
        && let Some(sim_ref) = sim.as_deref()
    {
        camera_pos = match focus.target {
            CameraFocusTarget::Ship if *view == ViewMode::Ship => {
                clamp_ship_camera_against_terrain(
                    camera_pos,
                    target_pos,
                    scale,
                    &body_states,
                    sim_ref,
                    &surfaces,
                    &time,
                    &mut collision,
                )
            }
            CameraFocusTarget::Body(body_id) => clamp_body_focus_camera_against_terrain(
                camera_pos,
                target_pos,
                scale,
                body_id,
                &body_states,
                sim_ref,
                &surfaces,
            ),
            _ => camera_pos,
        };
    }

    if *view == ViewMode::Ship
        && let (Some(mut camera_cell), Some(target_cell)) = (camera_cell, target_cell)
    {
        *camera_cell = target_cell;
    }
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

    #[test]
    fn scroll_zoom_is_independent_of_frame_batching() {
        let min = DISTANCE_MIN_DEFAULT;
        let max = MAP_DISTANCE_MAX;
        let start = 1.0e9;

        let batched = zoom_target_after_scroll(start, min, max, 2.0);
        let split = zoom_target_after_scroll(
            zoom_target_after_scroll(start, min, max, 1.0),
            min,
            max,
            1.0,
        );

        assert!((batched - split).abs() / start <= 1.0e-12);
    }

    #[test]
    fn scroll_zoom_clamps_to_view_bounds() {
        let min = DISTANCE_MIN_DEFAULT;
        let max = MAP_DISTANCE_MAX;

        assert_eq!(zoom_target_after_scroll(min, min, max, 1000.0), min);
        assert_eq!(zoom_target_after_scroll(max, min, max, -1000.0), max);
    }

    #[test]
    fn zoom_smoothing_accelerates_for_large_errors() {
        assert_eq!(zoom_smoothing_speed(0.0), ZOOM_SMOOTHING_SPEED_MIN);
        assert_eq!(
            zoom_smoothing_speed(ZOOM_FAST_ERROR),
            ZOOM_SMOOTHING_SPEED_MAX
        );
        assert!(zoom_smoothing_speed(ZOOM_FAST_ERROR * 0.25) > ZOOM_SMOOTHING_SPEED_MIN);
    }

    #[test]
    fn collision_boom_damps_instead_of_snapping() {
        let next = damped_camera_collision_boom(1_000.0, 180.0, 1.0 / 60.0);
        assert!(next < 1_000.0);
        assert!(next > 180.0);
    }

    #[test]
    fn collision_boom_recovers_slowly() {
        let next = damped_camera_collision_boom(180.0, 1_000.0, 1.0 / 60.0);
        assert!(next > 180.0);
        assert!(next < 1_000.0);
        assert!(next < damped_camera_collision_boom(1_000.0, 180.0, 1.0 / 60.0));
    }
}
