//! Debug-only free-fly camera (ship view).
//!
//! F4 toggles the camera off the orbit-focus pipeline. Bindings:
//!
//! | Key            | Action                              |
//! |----------------|-------------------------------------|
//! | W / S          | Forward / back                      |
//! | A / D          | Strafe left / right                 |
//! | R / F          | Up / down (along the local vertical) |
//! | Q / E          | Roll left / right (**weak/no level lock only**) |
//! | L              | Toggle level-to-planet-up           |
//! | C              | Toggle the ground floor (clip stop)  |
//! | Z (hold)       | Spring zoom (4× focal length)       |
//! | LMB-drag       | Yaw + pitch (mouse look)            |
//! | Shift / Ctrl   | 5× / 0.2× speed                     |
//! | Scroll wheel   | Adjust base cruise speed            |
//!
//! [`panel`] renders the same speed / level / ground-floor state as an on-screen
//! control surface, so none of it is keyboard-only trivia.
//!
//! Gated on [`DebugMode`] so it cannot be activated by accident in
//! non-debug builds when the toggle eventually becomes runtime-conditional.
//!
//! While active:
//! - [`camera::camera_input_system`] and [`camera::camera_transform_system`]
//!   bail early, leaving the camera transform under freecam control.
//! - [`GameFlightContext`] is suspended so movement keys don't simultaneously
//!   pitch/yaw/roll the ship.
//! - Sim time follows the craft's warp eligibility **snapshotted on enter**:
//!   if the vehicle could time-warp (>1×) at that moment, freecam is *not* a
//!   pause source — warp keys keep driving `SimClock` as in normal flight.
//!   If warp was capped at ≤1× (atmosphere, walking while moving, …), freecam
//!   freezes sim time so framing a dynamic surface flight doesn't advance the
//!   craft. Camera motion always uses wall-clock time either way.
//! - On enter, the camera latches the terrain-backed body selected by
//!   [`ViewAnchor`] and stores its complete pose in that body's rotating frame.
//!   Every frame reprojects the pose through the current body state, so both
//!   position and heading remain fixed relative to the world under warp. The
//!   body is never re-selected while freecam is active; a view with no resolved
//!   terrain body deliberately remains heliocentric-inertial.
//! - **Level lock** ([`FreeCam::level_to_up`], default on) constrains the pose to
//!   the local vertical at the camera's *current* position: yaw turns about that
//!   vertical, pitch is clamped short of the poles, and roll is zero. Because the
//!   constraint is re-applied every frame against the up direction where the
//!   camera now is, flying a quarter of the way around a body keeps the horizon
//!   level instead of slowly tipping — and there is no accumulated roll to
//!   hand-correct.
//!
//!   Its strength is an *authority* in `0..=1` ([`level_lock_authority`]) driven
//!   by how large the body looks from here — its apparent angular diameter, not
//!   an authored altitude — so it is rigid while the world fills the view,
//!   fades out as the world becomes an object you look at, and never switches
//!   state discontinuously. Q/E roll comes back as the authority falls.
//! - **Ground floor** ([`FreeCam::ground_collision`], default on) clamps the
//!   camera's radius to the terrain height under it plus
//!   [`FREECAM_GROUND_CLEARANCE_M`]. It is a *floor*, not a swept collision: it
//!   stops the camera sinking through the surface it is parked on, but a single
//!   fast frame can still cross a ridge. Turn it off to fly through a planet.
//! - Both constraints only touch a pose **freecam produced**, or the flight-camera
//!   pose it inherited on entry. A pose *handed* to it —
//!   [`FreeCam::activate_at_world_pose`], i.e. an applied viewpoint or a headless
//!   capture framing — is reproduced exactly until the user moves the camera, so
//!   authored roll and authored framing survive replay and capture baselines
//!   don't shift under them.

use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_capture_protocol::CameraOptics as CameraOpticsSpec;
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::types::BodyState;
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::bridge::WarpLimits;
use crate::camera::{ActiveCamera, OrbitCamera, ShipCamera};
use crate::camera_optics::CameraOptics;
use crate::debug::DebugMode;
use crate::hud::{UiKeyboardGate, UiPointerGate};
use crate::rendering::transforms::surface_orientation_authored;
use crate::rendering::{SimulationState, SolarSystemState, view_anchor::ViewAnchor};
use crate::view::ViewMode;

pub mod panel;

#[derive(Resource, Debug)]
pub struct FreeCam {
    pub active: bool,
    /// When `active`, whether freecam should leave sim time under warp control
    /// instead of freezing [`crate::sim_clock::SimClock`]. Set only on enter
    /// from [`craft_allows_time_warp`]; cleared on exit.
    pub allow_sim_time: bool,
    /// Base translation speed in m/s before Shift/Ctrl multipliers.
    /// Scroll adjusts this in log-space so the same wheel input feels
    /// proportional at every scale.
    ///
    /// One of the three **settings** fields (with [`Self::level_to_up`] and
    /// [`Self::ground_collision`]) that both [`freecam_drive_system`] and
    /// [`panel`] write — they are user preferences with two equal input
    /// surfaces, keyboard/wheel and the on-screen panel, which is why they are
    /// public while `reference_frame` stays private.
    pub base_speed_m_s: f64,
    /// Constrain the pose to the local vertical: no roll, clamped pitch, yaw
    /// about the body-radial up at the camera's current position. See the
    /// module docs.
    pub level_to_up: bool,
    /// Clamp the camera's radius to the terrain under it (plus
    /// [`FREECAM_GROUND_CLEARANCE_M`]) instead of letting it sink through.
    pub ground_collision: bool,
    /// Reference frame selected once per activation. Kept private so no other
    /// system can silently retarget an active freecam.
    reference_frame: FreeCamReferenceFrame,
    /// Optics owned by the camera rig that handed control to freecam. Restored
    /// on exit so framing edits remain local to the photographic mode.
    return_optics: Option<CameraOpticsSpec>,
    /// True for the exit frame after freecam releases the camera transform.
    /// The normal rig must rebuild its pose immediately, but must not consume
    /// the final freecam mouse delta as orbit input during that handoff.
    flight_input_handoff_pending: bool,
    /// Whether freecam may constrain the current pose — see the module docs on
    /// authored poses. Set once freecam has produced or inherited a pose of its
    /// own, and *kept* set afterwards: the level lock eases toward level over
    /// several frames, so it has to keep running on frames with no input or a
    /// half-corrected horizon would freeze the moment the user let go.
    pose_is_freecam_owned: bool,
}

impl Default for FreeCam {
    fn default() -> Self {
        Self {
            active: false,
            allow_sim_time: false,
            base_speed_m_s: FREECAM_DEFAULT_SPEED_M_S,
            level_to_up: true,
            ground_collision: true,
            reference_frame: FreeCamReferenceFrame::Inertial,
            return_optics: None,
            flight_input_handoff_pending: false,
            pose_is_freecam_owned: false,
        }
    }
}

impl FreeCam {
    /// Whether the freecam should suppress the normal flight-camera transform
    /// writer this frame. A just-activated (`Pending`) freecam deliberately
    /// lets that writer produce one final pose at the current simulation epoch
    /// before [`freecam_drive_system`] captures it.
    pub(crate) fn owns_camera_transform(&self) -> bool {
        self.active && !matches!(self.reference_frame, FreeCamReferenceFrame::Pending)
    }

    /// Whether the normal orbit rig must ignore mouse/scroll input this frame.
    ///
    /// On entry, `active` blocks it. On exit, the one-frame handoff flag keeps
    /// the last freecam drag delta from being reinterpreted as orbit input while
    /// still allowing [`crate::camera::camera_transform_system`] to restore the
    /// normal rig's own pose immediately.
    pub(crate) fn blocks_flight_camera_input(&self) -> bool {
        self.active || self.flight_input_handoff_pending
    }

    /// The body this session is anchored to, if any. `None` for a deep-space
    /// (inertial) session — both level lock and the ground floor need a body
    /// and stand down without one.
    pub fn anchor_body(&self) -> Option<BodyId> {
        match self.reference_frame {
            FreeCamReferenceFrame::BodyFixed(pose) => Some(pose.body),
            FreeCamReferenceFrame::Pending | FreeCamReferenceFrame::Inertial => None,
        }
    }

    /// Enter freecam at an already-resolved camera pose without giving the
    /// normal orbit camera one frame in which to overwrite it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn activate_at_world_pose(
        &mut self,
        body: BodyId,
        body_state: &BodyState,
        camera_world: DVec3,
        camera_rotation_world: DQuat,
        sim: &SimulationState,
        warp_limits: &WarpLimits,
        return_optics: CameraOpticsSpec,
    ) {
        self.allow_sim_time = craft_allows_time_warp(sim, warp_limits);
        self.return_optics = Some(return_optics);
        self.reference_frame = FreeCamReferenceFrame::BodyFixed(BodyFixedCameraPose::capture(
            body,
            body_state,
            camera_world,
            camera_rotation_world,
        ));
        self.flight_input_handoff_pending = false;
        // An authored pose: reproduced exactly until the user moves the camera.
        self.pose_is_freecam_owned = false;
        self.active = true;
    }

    fn begin_flight_camera_handoff(&mut self) {
        self.active = false;
        self.allow_sim_time = false;
        self.reference_frame = FreeCamReferenceFrame::Inertial;
        self.flight_input_handoff_pending = true;
        self.pose_is_freecam_owned = false;
    }

    fn finish_flight_camera_handoff(&mut self) {
        self.flight_input_handoff_pending = false;
    }
}

/// The freecam's reference frame is chosen exactly once on entry.
///
/// `Pending` lets the toggle run before the normal flight-camera driver and
/// capture the final inherited camera pose afterward. If the canonical
/// [`ViewAnchor`] has no terrain-backed body that frame, `Inertial` preserves
/// the old deep-space behaviour without attaching to an arbitrary distant
/// world.
#[derive(Debug, Clone, Copy)]
enum FreeCamReferenceFrame {
    Pending,
    Inertial,
    BodyFixed(BodyFixedCameraPose),
}

/// Complete camera pose in one body's rotating frame.
///
/// Position-only anchoring is insufficient: the camera would follow the body
/// centre but its view direction would remain inertial, so the local horizon
/// would still rotate away at high warp.
#[derive(Debug, Clone, Copy)]
struct BodyFixedCameraPose {
    body: BodyId,
    position_body: DVec3,
    rotation_body: DQuat,
}

impl BodyFixedCameraPose {
    fn capture(
        body: BodyId,
        body_state: &BodyState,
        camera_world: DVec3,
        camera_rotation_world: DQuat,
    ) -> Self {
        let world_to_body = body_state.orientation.inverse();
        Self {
            body,
            position_body: world_to_body * (camera_world - body_state.position),
            rotation_body: (world_to_body * camera_rotation_world).normalize(),
        }
    }

    fn world_pose(self, body_state: &BodyState) -> (DVec3, DQuat) {
        (
            body_state.position + body_state.orientation * self.position_body,
            (body_state.orientation * self.rotation_body).normalize(),
        )
    }
}

/// True when the craft's current warp cap admits any ladder rung above 1×.
///
/// Matches "allowed to time warp" as players mean it: not merely unpausing to
/// 1×, but accelerating sim time. Uses [`WarpLimits`] (regime policy applied
/// this frame) against the live warp ladder.
fn craft_allows_time_warp(sim: &SimulationState, limits: &WarpLimits) -> bool {
    let cap = limits.max_level;
    sim.simulation
        .warp
        .levels()
        .iter()
        .enumerate()
        .any(|(i, &speed)| i <= cap && speed > 1.0)
}

/// How hard the horizon constraint holds at the camera's current position.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LevelLock {
    /// The local vertical at the camera, in world/render axes. big_space grid
    /// cells are pure translations, so a world direction *is* a
    /// camera-`Transform`-space direction — no conversion needed.
    up: Vec3,
    /// Constraint strength in `0..=1`. See [`level_lock_authority`].
    authority: f32,
}

/// Resolve the level lock against the anchored body. `None` for a deep-space
/// session, an unresolvable body, or a camera far enough out that the body no
/// longer defines a horizon.
///
/// Body-fixed anchoring remains active regardless, for warp stability; only the
/// ground-camera leveling constraint fades.
fn local_up_world(
    reference_frame: FreeCamReferenceFrame,
    sim: &SimulationState,
    states: Option<&[BodyState]>,
    grid: &Grid,
    cell: &CellCoord,
    transform: &Transform,
) -> Option<LevelLock> {
    let FreeCamReferenceFrame::BodyFixed(anchor) = reference_frame else {
        return None;
    };
    let state = states.and_then(|states| states.get(anchor.body))?;
    let body = sim.system.bodies.get(anchor.body)?;
    let camera_world = grid.grid_position_double(cell, transform);
    surface_flight_level_lock(camera_world, state.position, body.radius_m)
}

fn surface_flight_level_lock(
    camera_world: DVec3,
    body_position: DVec3,
    body_radius_m: f64,
) -> Option<LevelLock> {
    let radial = camera_world - body_position;
    let up = radial.try_normalize()?.as_vec3();
    let authority = level_lock_authority(radial.length(), body_radius_m);
    (authority > 0.0).then_some(LevelLock { up, authority })
}

/// Level-lock authority from the body's **apparent size** — the angular
/// diameter `2·asin(R/r)` it subtends at the camera — smoothstepped from
/// [`LEVEL_LOCK_FULL_ANGLE_RAD`] down to [`LEVEL_LOCK_RELEASE_ANGLE_RAD`].
///
/// Deliberately independent of where the camera looks and of the lens: it
/// answers "how much of *everything* is planet", not "how much of the frame is
/// planet", so panning the view or racking the zoom can never change the flight
/// model under the user's hands.
///
/// Being a pure function of `r/R` is what lets one rule fit a 190 km moonlet and
/// a 3186 km planet — the horizon is worth levelling to exactly while the body
/// still dominates the surroundings, which is a statement about apparent size,
/// never about metres. The old rule (authored Kármán line, or 5 % of radius
/// capped at 100 km) released far too low and switched state on a threshold.
fn level_lock_authority(radius_m: f64, body_radius_m: f64) -> f32 {
    // Written positively so a NaN radius — a degenerate body state, not a
    // camera the user flew somewhere — falls out as "no lock" rather than
    // propagating into the pose.
    let usable = radius_m.is_finite() && radius_m > 0.0 && body_radius_m > 0.0;
    if !usable {
        return 0.0;
    }
    // At or below the surface the whole sky is body; `min` also keeps `asin`
    // out of its NaN domain there.
    let sin_angular_radius = (body_radius_m / radius_m).min(1.0);
    let angular_diameter_rad = 2.0 * sin_angular_radius.asin();
    let t = ((angular_diameter_rad - LEVEL_LOCK_RELEASE_ANGLE_RAD)
        / (LEVEL_LOCK_FULL_ANGLE_RAD - LEVEL_LOCK_RELEASE_ANGLE_RAD))
        .clamp(0.0, 1.0);
    // Smoothstep, not a linear ramp: authority then has zero slope at both ends
    // of the band, so climbing out has no perceptible moment where the assist
    // starts or stops — which is the whole point of the band.
    (t * t * (3.0 - 2.0 * t)) as f32
}

/// Per-frame slerp weight toward the levelled pose.
///
/// The effective rate is `LEVEL_LOCK_RATE_HZ · a/(1−a)`, which diverges as
/// authority approaches 1 and vanishes as it approaches 0. That is what makes a
/// full-authority lock *rigid* (indistinguishable from the old hard constraint,
/// including its pitch clamp) while keeping the correction continuous in
/// between — mid-band the horizon returns over about half a second, near the top
/// of the band it is a barely-perceptible drift.
fn level_lock_weight(authority: f32, dt_s: f32) -> f32 {
    if authority >= 1.0 {
        return 1.0;
    }
    if authority <= 0.0 || dt_s <= 0.0 {
        return 0.0;
    }
    let rate_hz = LEVEL_LOCK_RATE_HZ * authority / (1.0 - authority);
    1.0 - (-rate_hz * dt_s).exp()
}

/// The reference direction the camera treats as "up" for look and climb input:
/// the local vertical at full authority, the camera's own up with none, and the
/// normalized blend in between — so the *control* feel crosses the band as
/// continuously as the pose constraint does.
fn control_up(transform: &Transform, lock: Option<LevelLock>) -> Vec3 {
    let camera_up = *transform.up();
    let Some(lock) = lock else { return camera_up };
    camera_up
        .lerp(lock.up, lock.authority)
        // Only degenerate when the camera is exactly inverted relative to the
        // local vertical at half authority; the local vertical is the answer
        // the constraint is heading for anyway.
        .try_normalize()
        .unwrap_or(lock.up)
}

/// Ease the camera rotation toward the roll-free, pitch-clamped look about `up`
/// that matches its current heading, by `weight` of the way.
///
/// **Idempotent** — applying it to an already-level pose reproduces that pose,
/// at any weight — which is what lets it run every frame without drifting the
/// view.
fn apply_level_lock(transform: &mut Transform, up: Vec3, weight: f32) {
    let weight = weight.clamp(0.0, 1.0);
    if weight <= 0.0 {
        return;
    }
    let forward = *transform.forward();
    let sin_pitch = forward
        .dot(up)
        .clamp(-FREECAM_LEVEL_MAX_SIN_PITCH, FREECAM_LEVEL_MAX_SIN_PITCH);
    let horizontal = (forward - up * forward.dot(up))
        .try_normalize()
        // Looking straight up or down: `forward` carries no heading. The
        // camera's right is still horizontal under level lock, and
        // `up × right` is the forward that goes with it.
        .or_else(|| up.cross(*transform.right()).try_normalize());
    let Some(horizontal) = horizontal else {
        return;
    };
    let cos_pitch = (1.0 - sin_pitch * sin_pitch).max(0.0).sqrt();
    let mut levelled = *transform;
    levelled.look_to(horizontal * cos_pitch + up * sin_pitch, up);
    transform.rotation = transform
        .rotation
        .slerp(levelled.rotation, weight)
        .normalize();
}

/// Body-centred radius the camera may not descend below: terrain height under
/// the camera plus [`FREECAM_GROUND_CLEARANCE_M`].
///
/// `None` means *don't clamp* — too high for it to matter, no height source, or
/// a surface too cold to answer. Falling back to the datum instead would push a
/// camera parked in a valley up through the terrain.
fn ground_floor_radius_m(
    body: BodyId,
    sim: &SimulationState,
    states: &[BodyState],
    height_sources: &HeightSourceRegistry,
    camera_world: DVec3,
) -> Option<f64> {
    let definition = sim.system.bodies.get(body)?;
    let state = states.get(body)?;
    let radial = camera_world - state.position;
    let radius = radial.length();
    if radius - definition.radius_m > FREECAM_GROUND_PROBE_MAX_ALT_M {
        return None;
    }
    let source = height_sources.get(body)?;
    // The **surface** body-fixed frame — the one the height sources and terrain
    // renderers share. The raw ephemeris orientation is a different frame for a
    // tidally-locked moon, and would sample the wrong side of it.
    let orientation = surface_orientation_authored(&sim.system.bodies, body, states)
        .unwrap_or_else(|| state.orientation.normalize());
    let direction = (orientation.inverse() * radial).try_normalize()?;
    let height_m = source.sample_height_m(direction.as_vec3(), FREECAM_GROUND_PROBE_LOD_M)?;
    Some(definition.radius_m + height_m as f64 + FREECAM_GROUND_CLEARANCE_M)
}

/// A familiar real-world speed this cruise setting is about as fast as, so the
/// number on the panel means something without mental arithmetic.
pub fn speed_reference(speed_m_s: f64) -> &'static str {
    match speed_m_s {
        v if v < 2.0 => "a slow walk",
        v if v < 6.0 => "a brisk walk",
        v if v < 12.0 => "a sprint",
        v if v < 35.0 => "city traffic",
        v if v < 90.0 => "highway traffic",
        v if v < 200.0 => "a race car",
        v if v < 340.0 => "an airliner",
        v if v < 1_000.0 => "supersonic",
        v if v < 3_000.0 => "a re-entry capsule",
        v if v < 12_000.0 => "orbital velocity",
        v if v < 100_000.0 => "an interplanetary transfer",
        v if v < 3.0e6 => "a solar-wind gust",
        _ => "1 % of light speed",
    }
}

/// The cruise speed as the panel prints it: metres per second up to a km/s, then
/// km/s — a five-digit m/s number reads as noise.
pub fn format_speed(speed_m_s: f64) -> String {
    if speed_m_s < 1_000.0 {
        format!("{speed_m_s:.0} m/s")
    } else {
        format!("{:.1} km/s", speed_m_s / 1_000.0)
    }
}

const FREECAM_DEFAULT_SPEED_M_S: f64 = 100.0;
pub const FREECAM_MIN_SPEED_M_S: f64 = 1.0;
pub const FREECAM_MAX_SPEED_M_S: f64 = 1.0e7;
const FREECAM_SHIFT_MULT: f64 = 5.0;
const FREECAM_CTRL_MULT: f64 = 0.2;
const FREECAM_LOOK_SENSITIVITY: f32 = 0.0025;
const FREECAM_SCROLL_LOG_STEP: f64 = 0.20;
/// Roll rate in rad/s while a roll key is held. ~86°/s — fast enough to
/// re-level from inverted in a couple of seconds without making fine
/// roll adjustments impossible.
const FREECAM_ROLL_RATE_RAD_S: f32 = 1.5;
/// Telephoto zoom factor while Z is held. This multiplies focal length, which
/// is the photographic operation the control claims to perform.
const FREECAM_ZOOM_FACTOR: f32 = 4.0;
/// Exponential smoothing rate (1/s) for easing the freecam lens toward its zoom
/// target, so zoom in/out reads like racking a lens rather than a hard snap.
const FREECAM_ZOOM_LERP_RATE: f32 = 12.0;
/// Pitch limit under level lock, as `sin(pitch)`. Just short of the pole: at
/// exactly ±90° the heading is undefined and `look_to` has no valid basis.
const FREECAM_LEVEL_MAX_SIN_PITCH: f32 = 0.9998; // ≈ 88.9°
/// Apparent angular diameter at and above which the horizon constraint is
/// rigid. 120° is more of the surroundings than any normal lens frames, so
/// "down" is unambiguous and a tilted horizon reads as a mistake. That is
/// `r = 1.155 R` — ~493 km over Thalos and ~134 km over Mira, against the 80 km
/// Kármán line and 43 km airless ceiling the old rule released at.
const LEVEL_LOCK_FULL_ANGLE_RAD: f64 = 120.0 * std::f64::consts::PI / 180.0;
/// Apparent angular diameter at and below which the camera is fully 6-DOF. 45°
/// is where the body has become an object you look *at* rather than a world you
/// are over — `r = 2.61 R`, ~5100 km over Thalos.
const LEVEL_LOCK_RELEASE_ANGLE_RAD: f64 = 45.0 * std::f64::consts::PI / 180.0;
/// Base rate (1/s) of the pull toward level, at half authority — see
/// [`level_lock_weight`] for why it is scaled by `a/(1−a)` rather than used
/// directly.
const LEVEL_LOCK_RATE_HZ: f32 = 2.0;
/// Authority above which Q/E roll stands down, since the constraint would win
/// against it anyway. Below it the two coexist: roll freely, and the horizon
/// creeps back at whatever rate the altitude earns.
const LEVEL_LOCK_ROLL_SUPPRESS_AUTHORITY: f32 = 0.5;
/// How far above the terrain the ground floor holds the camera. Small enough
/// to park the lens on the deck, large enough that a metre of terrain LOD
/// disagreement doesn't put the near plane inside a hill.
pub const FREECAM_GROUND_CLEARANCE_M: f64 = 2.0;
/// Skip the ground probe above this altitude. The clamp can't trigger up there
/// anyway (no body has 100 km of relief), so orbital freecam pays nothing.
const FREECAM_GROUND_PROBE_MAX_ALT_M: f64 = 100_000.0;
/// `tile_lod_m` hint for the ground probe — matches the view anchor's nadir
/// probe, so the floor agrees with the altitude the HUD reports.
const FREECAM_GROUND_PROBE_LOD_M: f32 = 2.0;

pub struct FreeCamPlugin;

impl Plugin for FreeCamPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FreeCam>()
            .add_systems(
                Update,
                (
                    // Toggle before the normal camera chain so its input/driver
                    // sees the new ownership in the same frame. Drive afterward
                    // so activation captures the final inherited camera pose and
                    // deactivation lets the flight camera snap back immediately.
                    toggle_freecam_system.before(crate::camera::camera_input_system),
                    freecam_drive_system.after(crate::camera::camera_transform_system),
                    freecam_zoom_system,
                    crate::camera_optics::sync_camera_optics_projection,
                )
                    .chain()
                    .in_set(crate::SimStage::Camera),
            )
            .add_plugins(panel::FreeCamPanelPlugin);
    }
}

fn toggle_freecam_system(
    debug: Option<Res<DebugMode>>,
    view: Res<ViewMode>,
    input: Res<GameInputIntent>,
    shipyard: Option<Res<crate::shipyard_editor::ShipyardEditor>>,
    sim: Res<SimulationState>,
    warp_limits: Res<WarpLimits>,
    mut camera: Query<(&Projection, &mut CameraOptics), (With<OrbitCamera>, With<ShipCamera>)>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut freecam: ResMut<FreeCam>,
    ui_keyboard: Res<UiKeyboardGate>,
) {
    // Auto-disable when leaving ship view — map view has no freecam analogue
    // and we don't want input gating to drift while the user can't recover.
    // [`gate_enhanced_input_sources`] picks up the change next PreUpdate.
    if freecam.active && *view != ViewMode::Ship {
        freecam.begin_flight_camera_handoff();
        if let Ok((_, mut optics)) = camera.single_mut() {
            restore_return_optics(&mut freecam, &mut optics);
        }
        return;
    }

    if !input.toggle_free_cam {
        return;
    }
    // The shipyard editor owns the screen; freecam has no scene to fly.
    if shipyard.as_deref().map(|s| s.open).unwrap_or(false) {
        return;
    }
    if ui_keyboard.text_entry() {
        return;
    }
    if *view != ViewMode::Ship {
        return;
    }
    if !debug.as_deref().map(|d| d.enabled).unwrap_or(false) {
        return;
    }

    if freecam.active {
        freecam.begin_flight_camera_handoff();
        if let Ok((_, mut optics)) = camera.single_mut() {
            restore_return_optics(&mut freecam, &mut optics);
        }
    } else {
        freecam.flight_input_handoff_pending = false;
        freecam.allow_sim_time = craft_allows_time_warp(&sim, &warp_limits);
        if let Ok((Projection::Perspective(perspective), mut optics)) = camera.single_mut() {
            let aspect = windows
                .single()
                .map(|window| {
                    [
                        window.resolution.physical_width().max(1),
                        window.resolution.physical_height().max(1),
                    ]
                })
                .unwrap_or([16, 9]);
            if let Ok(spec) = CameraOpticsSpec::from_vertical_fov(perspective.fov, aspect) {
                let _ = optics.set_spec(spec);
                freecam.return_optics = Some(spec);
            }
        }
        freecam.reference_frame = FreeCamReferenceFrame::Pending;
        freecam.pose_is_freecam_owned = false;
        freecam.active = true;
    }
}

fn restore_return_optics(freecam: &mut FreeCam, optics: &mut CameraOptics) {
    if let Some(spec) = freecam.return_optics.take() {
        let _ = optics.set_spec(spec);
    } else {
        optics.set_zoom_multiplier(1.0);
    }
}

fn freecam_drive_system(
    time: Res<Time<Real>>,
    keys: Res<ButtonInput<KeyCode>>,
    input: Res<GameInputIntent>,
    ui_gate: Res<UiPointerGate>,
    ui_keyboard: Res<UiKeyboardGate>,
    mut freecam: ResMut<FreeCam>,
    view_anchor: Res<ViewAnchor>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    height_sources: Res<HeightSourceRegistry>,
    grid_q: Query<&Grid, With<BigSpace>>,
    mut cam_q: Query<
        (&mut Transform, &mut CellCoord),
        (With<OrbitCamera>, With<ActiveCamera>, With<ShipCamera>),
    >,
) {
    if !freecam.active {
        freecam.finish_flight_camera_handoff();
        return;
    }
    let Ok(grid) = grid_q.single() else { return };
    let Ok((mut transform, mut cell)) = cam_q.single_mut() else {
        return;
    };

    let states = solar.states.as_deref();

    // `Pending` means this frame inherits the *flight camera's* pose — a pose
    // freecam is free to sanitise (see the level/ground gate below). A session
    // entered through `activate_at_world_pose` never passes through `Pending`,
    // which is exactly what distinguishes an authored pose from an inherited
    // one.
    let inherited_pose = matches!(freecam.reference_frame, FreeCamReferenceFrame::Pending);

    // Resolve the reference frame once, after inheriting the normal flight
    // camera's final pose. `ViewAnchor` is the canonical answer to which
    // terrain-backed body the render view belongs to; latching its BodyId
    // avoids discontinuous nearest-body switches while freecam is active.
    if inherited_pose {
        let camera_world = grid.grid_position_double(&cell, &transform);
        freecam.reference_frame = view_anchor
            .resolved
            .and_then(|anchor| {
                states.and_then(|states| {
                    states.get(anchor.body).map(|body_state| {
                        FreeCamReferenceFrame::BodyFixed(BodyFixedCameraPose::capture(
                            anchor.body,
                            body_state,
                            camera_world,
                            transform.rotation.as_dquat(),
                        ))
                    })
                })
            })
            .unwrap_or(FreeCamReferenceFrame::Inertial);
    }

    // Carry a body-fixed camera pose to the current simulation epoch before
    // applying wall-clock view input. This executes in `SimStage::Camera`,
    // after `SolarSystemState` was refreshed in `SimStage::Sync`.
    if let FreeCamReferenceFrame::BodyFixed(anchor) = freecam.reference_frame
        && let Some(body_state) = states.and_then(|states| states.get(anchor.body))
    {
        let (camera_world, camera_rotation_world) = anchor.world_pose(body_state);
        let (next_cell, local) = grid.translation_to_grid(camera_world);
        *cell = next_cell;
        transform.translation = local;
        transform.rotation = camera_rotation_world.as_quat();
    }

    // The pointer belongs to the freecam panel (or any other HUD control) when
    // it is over one: without this, dragging the speed slider would also
    // mouse-look and scrolling over the panel would move the same setting
    // twice. Mirrors `camera::camera_input_system`'s guard.
    let ui_pointer_busy = ui_gate.hovered;

    // Scroll adjusts cruise speed in log-space.
    if !ui_pointer_busy && input.camera_wheel.y != 0.0 {
        let log =
            freecam.base_speed_m_s.ln() + (input.camera_wheel.y as f64) * FREECAM_SCROLL_LOG_STEP;
        freecam.base_speed_m_s = log
            .exp()
            .clamp(FREECAM_MIN_SPEED_M_S, FREECAM_MAX_SPEED_M_S);
    }

    // Every key read below is raw: the flight context is suspended while
    // freecam owns the camera, so the enhanced-input gate can't stand in for
    // it. That also means the text-entry gate has to be applied here by hand —
    // typing a viewpoint name (native F9 prompt or egui F8 manager) must not
    // fly the camera away from the very view being saved. The pose carry above
    // still runs while typing; only the input reads stand down.
    let typing = ui_keyboard.text_entry();
    let pressed = |key| !typing && keys.pressed(key);
    let just_pressed = |key| !typing && keys.just_pressed(key);

    // Mode toggles.
    if just_pressed(KeyCode::KeyL) {
        freecam.level_to_up = !freecam.level_to_up;
    }
    if just_pressed(KeyCode::KeyC) {
        freecam.ground_collision = !freecam.ground_collision;
    }

    let reference_frame = freecam.reference_frame;
    // Level lock needs a local vertical and a body big enough in the view to
    // define a horizon. Where it doesn't apply the camera keeps 6-DOF behaviour
    // without changing the user's remembered `level_to_up` setting.
    let lock = freecam
        .level_to_up
        .then(|| local_up_world(reference_frame, &sim, states, grid, &cell, &transform))
        .flatten();

    let mut pose_changed = false;

    // Mouse-look while LMB held. Pitch is always about the camera's own right;
    // only the yaw axis moves with authority, from the camera's own up to the
    // local vertical:
    //
    // - **Level lock**: yaw about the *local vertical*, pitch about a right
    //   that stays horizontal — the familiar ground-camera control.
    //   `apply_level_lock` below clamps the pitch and removes any roll, so the
    //   horizon can never end up tilted no matter how the drag curved.
    // - **Free look**: yaw about the camera's own up is exactly intrinsic yaw,
    //   because pre-multiplying by a rotation about `R·Y` equals
    //   post-multiplying by one about `Y`. Not world-Y: the ship-view orbit
    //   basis is radial-up, so freecam activating over a planet inherits a roll
    //   relative to world-Y, and yawing about world-Y there arcs and tumbles
    //   instead of panning. Roll then accumulates passively from circular
    //   drags, matching spacecraft-cam convention — which is exactly what level
    //   lock exists to opt out of.
    //
    // Writing both as one axis-blended operation is what keeps the *control*
    // continuous across the band; two branches would swap feel on a threshold
    // even though the pose itself eased.
    if input.primary_pressed && !ui_pointer_busy {
        let delta = input.camera_motion;
        if delta != Vec2::ZERO {
            let yaw_axis = control_up(&transform, lock);
            let yaw = Quat::from_axis_angle(yaw_axis, -delta.x * FREECAM_LOOK_SENSITIVITY);
            let pitch =
                Quat::from_axis_angle(*transform.right(), -delta.y * FREECAM_LOOK_SENSITIVITY);
            transform.rotation = (pitch * yaw * transform.rotation).normalize();
            pose_changed = true;
        }
    }

    // Roll on Q/E around camera-local Z. Positive Z rotation rotates +X (right)
    // toward +Y (up), which from the pilot's POV looking down -Z is a
    // counter-clockwise tilt — i.e. "roll left." E gets the negated angle so
    // it tilts the camera clockwise (right wing down) as conventional.
    // Suppressed under a strong level lock, whose whole contract is zero roll;
    // as authority fades the keys come back rather than switching on at a
    // boundary the user can feel.
    let mut roll_input = 0.0_f32;
    if !lock.is_some_and(|lock| lock.authority >= LEVEL_LOCK_ROLL_SUPPRESS_AUTHORITY) {
        if pressed(KeyCode::KeyQ) {
            roll_input += 1.0;
        }
        if pressed(KeyCode::KeyE) {
            roll_input -= 1.0;
        }
    }
    // Freecam is a view/debug affordance, so it keeps moving while sim-time
    // is paused. The escape pause menu still gates the whole camera stage.
    let dt_f32 = time.delta_secs();
    if roll_input != 0.0 {
        let roll = Quat::from_rotation_z(roll_input * FREECAM_ROLL_RATE_RAD_S * dt_f32);
        transform.rotation *= roll;
        pose_changed = true;
    }

    // Translation keys. Under level lock R/F climb along the *local vertical*,
    // not the camera's up — "up" has to mean up, or holding R while pitched
    // down flies you into the ground. Same blended reference as mouse-look.
    let vertical = control_up(&transform, lock);
    let mut dir = Vec3::ZERO;
    if pressed(KeyCode::KeyW) {
        dir += *transform.forward();
    }
    if pressed(KeyCode::KeyS) {
        dir -= *transform.forward();
    }
    if pressed(KeyCode::KeyD) {
        dir += *transform.right();
    }
    if pressed(KeyCode::KeyA) {
        dir -= *transform.right();
    }
    if pressed(KeyCode::KeyR) {
        dir += vertical;
    }
    if pressed(KeyCode::KeyF) {
        dir -= vertical;
    }

    if dir != Vec3::ZERO {
        let speed_mult = if pressed(KeyCode::ShiftLeft) || pressed(KeyCode::ShiftRight) {
            FREECAM_SHIFT_MULT
        } else if pressed(KeyCode::ControlLeft) || pressed(KeyCode::ControlRight) {
            FREECAM_CTRL_MULT
        } else {
            1.0
        };
        let speed = freecam.base_speed_m_s * speed_mult;
        let dt = time.delta_secs_f64();
        let step: DVec3 = dir.normalize().as_dvec3() * speed * dt;

        let world = grid.grid_position_double(&cell, &transform) + step;
        let (next_cell, local) = grid.translation_to_grid(world);
        *cell = next_cell;
        transform.translation = local;
        pose_changed = true;
    }

    // Both constraints below only touch a pose **freecam itself produced** this
    // frame, or the flight-camera pose it just inherited. A pose that was handed
    // to it — a saved viewpoint applied from F8, a headless capture framing —
    // is reproduced exactly until the user moves the camera. Without this an
    // authored shot with deliberate roll, or one framed under an overhang,
    // would be silently rewritten on replay, and every capture baseline with it.
    //
    // Latched, not re-derived per frame: the level lock eases over several
    // frames, so it has to keep running after the user stops moving or a
    // half-corrected horizon would freeze the instant they let go of the mouse.
    let sanitize_pose = freecam.pose_is_freecam_owned || pose_changed || inherited_pose;
    freecam.pose_is_freecam_owned = sanitize_pose;

    // Ground floor. `ground_floor_radius_m` returns `None` while the surface is
    // cold rather than falling back to the datum — clamping to sea level would
    // eject a camera parked in a valley.
    if sanitize_pose
        && freecam.ground_collision
        && let FreeCamReferenceFrame::BodyFixed(anchor) = reference_frame
        && let Some(body_state) = states.and_then(|states| states.get(anchor.body))
    {
        let camera_world = grid.grid_position_double(&cell, &transform);
        let radial = camera_world - body_state.position;
        let r = radial.length();
        if let Some(floor_r) = ground_floor_radius_m(
            anchor.body,
            &sim,
            states.unwrap_or_default(),
            &height_sources,
            camera_world,
        ) && r > 0.0
            && r < floor_r
        {
            let lifted = body_state.position + radial * (floor_r / r);
            let (next_cell, local) = grid.translation_to_grid(lifted);
            *cell = next_cell;
            transform.translation = local;
            pose_changed = true;
        }
    }

    // Re-derive the level pose against the vertical *where the camera now is*.
    // Doing this last (after translation and the ground clamp) is what makes
    // the horizon hold while flying across a body: the vertical rotates under
    // the camera and the constraint follows it, instead of a one-shot level
    // that decays with every kilometre travelled.
    let final_lock = freecam
        .level_to_up
        .then(|| local_up_world(reference_frame, &sim, states, grid, &cell, &transform))
        .flatten();
    if sanitize_pose && let Some(lock) = final_lock {
        let before = transform.rotation;
        apply_level_lock(
            &mut transform,
            lock.up,
            level_lock_weight(lock.authority, dt_f32),
        );
        // The entry frame has no other reason to persist, and the levelled pose
        // must reach the anchor — otherwise next frame's reprojection restores
        // the roll it just removed.
        if transform.rotation.dot(before).abs() < 1.0 - 1.0e-7 {
            pose_changed = true;
        }
    }

    // Persist mouse-look, roll, and translation back into the latched body's
    // frame. A no-input frame deliberately keeps the original f64 anchor;
    // round-tripping it through the camera's f32 `Transform` every frame would
    // accumulate quantisation drift of its own.
    if pose_changed
        && let FreeCamReferenceFrame::BodyFixed(anchor) = freecam.reference_frame
        && let Some(body_state) = states.and_then(|states| states.get(anchor.body))
    {
        let camera_world = grid.grid_position_double(&cell, &transform);
        freecam.reference_frame = FreeCamReferenceFrame::BodyFixed(BodyFixedCameraPose::capture(
            anchor.body,
            body_state,
            camera_world,
            transform.rotation.as_dquat(),
        ));
    }
}

/// Hold Z to multiply the freecam's effective focal length; release to return
/// to the base lens.
///
/// Runs regardless of `freecam.active` so that toggling freecam off — or
/// switching out of ship view — while still zoomed eases the lens back to
/// normal instead of stranding the camera at a magnified projection.
fn freecam_zoom_system(
    time: Res<Time<Real>>,
    keys: Res<ButtonInput<KeyCode>>,
    ui_keyboard: Res<UiKeyboardGate>,
    freecam: Res<FreeCam>,
    mut cam_q: Query<&mut CameraOptics, (With<OrbitCamera>, With<ActiveCamera>, With<ShipCamera>)>,
) {
    let Ok(mut optics) = cam_q.single_mut() else {
        return;
    };
    // The `z` in a typed viewpoint name is a character, not a zoom.
    let target = if freecam.active && !ui_keyboard.text_entry() && keys.pressed(KeyCode::KeyZ) {
        FREECAM_ZOOM_FACTOR
    } else {
        1.0
    };

    let current = optics.zoom_multiplier();
    if (current - target).abs() < 1.0e-4 {
        optics.set_zoom_multiplier(target);
        return;
    }

    let smoothing = 1.0 - (-FREECAM_ZOOM_LERP_RATE * time.delta_secs()).exp();
    let next = current + (target - current) * smoothing;
    optics.set_zoom_multiplier(next);
}

#[cfg(test)]
mod tests {
    use super::*;
    use thalos_physics_canonical::canonical::Epoch;

    fn body_state(position: DVec3, orientation: DQuat) -> BodyState {
        BodyState {
            id: 3,
            epoch: Epoch(0.0),
            position,
            velocity: DVec3::ZERO,
            orientation,
            angular_velocity: DVec3::ZERO,
            mass_kg: 1.0,
            gm: 1.0,
            radius_m: 1.0,
        }
    }

    fn assert_vec3_close(actual: DVec3, expected: DVec3) {
        assert!(
            actual.abs_diff_eq(expected, 1.0e-5),
            "actual={actual:?}, expected={expected:?}"
        );
    }

    fn assert_quat_close(actual: DQuat, expected: DQuat) {
        // q and -q encode the same rotation.
        assert!(
            actual.dot(expected).abs() > 1.0 - 1.0e-12,
            "actual={actual:?}, expected={expected:?}"
        );
    }

    #[test]
    fn pending_activation_leaves_one_final_pose_to_flight_camera() {
        let mut freecam = FreeCam {
            active: true,
            reference_frame: FreeCamReferenceFrame::Pending,
            ..Default::default()
        };
        assert!(!freecam.owns_camera_transform());

        freecam.reference_frame = FreeCamReferenceFrame::Inertial;
        assert!(freecam.owns_camera_transform());
    }

    #[test]
    fn exit_handoff_restores_pose_without_consuming_freecam_input() {
        let mut freecam = FreeCam {
            active: true,
            reference_frame: FreeCamReferenceFrame::Inertial,
            ..Default::default()
        };

        freecam.begin_flight_camera_handoff();

        assert!(!freecam.owns_camera_transform());
        assert!(freecam.blocks_flight_camera_input());

        freecam.finish_flight_camera_handoff();
        assert!(!freecam.blocks_flight_camera_input());
    }

    #[test]
    fn body_fixed_pose_round_trips_world_pose() {
        let state = body_state(
            DVec3::new(8.0e8, -4.0e8, 2.0e8),
            DQuat::from_rotation_y(0.7) * DQuat::from_rotation_x(-0.2),
        );
        let camera_world = state.position + DVec3::new(2.0e6, 3.0e6, -4.0e6);
        let camera_rotation =
            (DQuat::from_rotation_z(0.4) * DQuat::from_rotation_x(-0.3)).normalize();

        let anchor = BodyFixedCameraPose::capture(state.id, &state, camera_world, camera_rotation);
        let (round_trip_position, round_trip_rotation) = anchor.world_pose(&state);

        assert_vec3_close(round_trip_position, camera_world);
        assert_quat_close(round_trip_rotation, camera_rotation);
    }

    #[test]
    fn body_fixed_pose_follows_body_translation_and_rotation() {
        let initial = body_state(
            DVec3::new(1.0e9, 2.0e9, -3.0e9),
            DQuat::from_rotation_z(0.1),
        );
        let position_body = DVec3::new(3.2e6, 8_000.0, -12_000.0);
        let rotation_body =
            (DQuat::from_rotation_y(-0.5) * DQuat::from_rotation_x(0.25)).normalize();
        let camera_world = initial.position + initial.orientation * position_body;
        let camera_rotation = (initial.orientation * rotation_body).normalize();
        let anchor =
            BodyFixedCameraPose::capture(initial.id, &initial, camera_world, camera_rotation);

        let advanced = body_state(
            DVec3::new(-4.0e9, 7.0e9, 9.0e9),
            DQuat::from_rotation_z(2.1) * DQuat::from_rotation_y(-0.4),
        );
        let (advanced_position, advanced_rotation) = anchor.world_pose(&advanced);

        assert_vec3_close(
            advanced.orientation.inverse() * (advanced_position - advanced.position),
            position_body,
        );
        assert_quat_close(
            advanced.orientation.inverse() * advanced_rotation,
            rotation_body,
        );
    }

    /// An arbitrary local vertical, so the tests can't pass by accidentally
    /// agreeing with a world axis.
    fn tilted_up() -> Vec3 {
        Vec3::new(0.3, 0.9, -0.32).normalize()
    }

    /// The radius at which a body of radius `r` subtends `degrees` of angular
    /// diameter, i.e. the inverse of the authority curve's input.
    fn radius_for_apparent_diameter(body_radius_m: f64, degrees: f64) -> f64 {
        body_radius_m / (degrees.to_radians() / 2.0).sin()
    }

    /// The whole point of the rule: authority is a function of *apparent size*,
    /// so two bodies three orders of magnitude apart in radius behave
    /// identically at the same apparent size — and neither one's behaviour can
    /// be read off an altitude in metres.
    #[test]
    fn level_lock_authority_tracks_apparent_size_across_body_scales() {
        let planet_m = 3_186_000.0;
        let moonlet_m = 190_000.0;

        for degrees in [150.0, 120.0, 90.0, 60.0, 45.0, 20.0] {
            let planet =
                level_lock_authority(radius_for_apparent_diameter(planet_m, degrees), planet_m);
            let moonlet =
                level_lock_authority(radius_for_apparent_diameter(moonlet_m, degrees), moonlet_m);
            assert!(
                (planet - moonlet).abs() < 1.0e-4,
                "{degrees}° disagrees across scales: planet={planet}, moonlet={moonlet}"
            );
        }
    }

    #[test]
    fn level_lock_authority_is_full_inside_the_band_and_zero_outside_it() {
        let radius_m = 3_186_000.0;
        let full_deg = LEVEL_LOCK_FULL_ANGLE_RAD.to_degrees();
        let release_deg = LEVEL_LOCK_RELEASE_ANGLE_RAD.to_degrees();

        // On and under the surface, and anywhere the body is bigger than the
        // full-authority angle: rigid.
        assert_eq!(level_lock_authority(radius_m * 0.5, radius_m), 1.0);
        assert_eq!(level_lock_authority(radius_m, radius_m), 1.0);
        assert_eq!(
            level_lock_authority(radius_for_apparent_diameter(radius_m, full_deg), radius_m),
            1.0
        );
        // Below the release angle: fully 6-DOF, and `surface_flight_level_lock`
        // reports no lock at all rather than a zero-strength one.
        assert_eq!(
            level_lock_authority(
                radius_for_apparent_diameter(radius_m, release_deg),
                radius_m
            ),
            0.0
        );
        assert_eq!(
            surface_flight_level_lock(
                DVec3::Y * radius_for_apparent_diameter(radius_m, release_deg * 0.5),
                DVec3::ZERO,
                radius_m,
            ),
            None
        );
    }

    /// A user climbing out must never cross a step: authority is continuous and
    /// monotone in altitude, with no slope discontinuity at either end.
    #[test]
    fn level_lock_authority_decreases_smoothly_with_altitude() {
        let radius_m = 3_186_000.0;
        let mut previous = 1.0_f32;
        let mut steps = 0;
        // Sweep the whole band and a margin past both ends.
        for i in 0..=400 {
            let degrees = 170.0 - (i as f64) * 0.4;
            let authority =
                level_lock_authority(radius_for_apparent_diameter(radius_m, degrees), radius_m);
            assert!(
                authority <= previous + 1.0e-6,
                "authority rose with altitude at {degrees}°: {previous} → {authority}"
            );
            assert!(
                (previous - authority) < 0.05,
                "authority stepped at {degrees}°: {previous} → {authority}"
            );
            if authority > 0.0 && authority < 1.0 {
                steps += 1;
            }
            previous = authority;
        }
        assert!(steps > 50, "band is too narrow to be a transition: {steps}");
    }

    /// Mid-band the constraint pulls toward level over time rather than
    /// snapping, and full authority still lands rigidly in one frame.
    #[test]
    fn level_lock_eases_at_partial_authority_and_snaps_at_full() {
        let dt = 1.0 / 60.0;
        assert_eq!(level_lock_weight(1.0, dt), 1.0);
        assert_eq!(level_lock_weight(0.0, dt), 0.0);

        let half = level_lock_weight(0.5, dt);
        assert!(
            half > 0.0 && half < 0.2,
            "half authority should be a gentle pull, got {half}"
        );
        // Frame-rate independence is the property that matters: stepping the
        // per-frame weight for one second must leave the residual the
        // continuous law predicts, `exp(-LEVEL_LOCK_RATE_HZ)` at half
        // authority — a ~0.5 s time constant — no matter how it is subdivided.
        let expected = (-LEVEL_LOCK_RATE_HZ).exp();
        for (frames, dt) in [(60, 1.0 / 60.0), (144, 1.0 / 144.0), (20, 1.0 / 20.0)] {
            let step = level_lock_weight(0.5, dt);
            let mut roll_left = 1.0_f32;
            for _ in 0..frames {
                roll_left *= 1.0 - step;
            }
            assert!(
                (roll_left - expected).abs() < 1.0e-4,
                "{frames} fps disagrees with the continuous law: {roll_left} vs {expected}"
            );
        }

        // Rate rises with authority, so the band reads as "tighter the lower
        // you are" rather than a single soft setting.
        assert!(level_lock_weight(0.9, dt) > level_lock_weight(0.5, dt));
        assert!(level_lock_weight(0.5, dt) > level_lock_weight(0.1, dt));
    }

    /// A partial weight moves the pose toward level without arriving, which is
    /// what makes the boundary crossing invisible.
    #[test]
    fn partial_level_lock_reduces_roll_without_removing_it() {
        let up = tilted_up();
        let mut transform = Transform::from_translation(Vec3::ZERO);
        transform.look_to(Vec3::new(1.0, 0.1, 0.4), up);
        transform.rotation *= Quat::from_rotation_z(0.8);
        let before = transform.right().dot(up).abs();

        apply_level_lock(&mut transform, up, 0.25);

        let after = transform.right().dot(up).abs();
        assert!(after < before, "roll did not shrink: {before} → {after}");
        assert!(after > 1.0e-3, "partial weight snapped to level: {after}");
    }

    #[test]
    fn level_lock_removes_roll() {
        let up = tilted_up();
        let mut transform = Transform::from_translation(Vec3::ZERO);
        transform.look_to(Vec3::new(1.0, 0.1, 0.4), up);
        // Bank the camera hard, the way a few circular mouse drags do.
        transform.rotation *= Quat::from_rotation_z(0.8);
        assert!(transform.right().dot(up).abs() > 0.1, "test did not bank");

        apply_level_lock(&mut transform, up, 1.0);

        assert!(
            transform.right().dot(up).abs() < 1.0e-5,
            "right is not horizontal: {:?}",
            transform.right()
        );
    }

    /// The constraint runs every frame, so a level pose must be a fixed point —
    /// otherwise the view would creep while the camera sits still. That has to
    /// hold at *every* weight, or a camera parked mid-band would drift.
    #[test]
    fn level_lock_is_idempotent_at_any_weight() {
        let up = tilted_up();
        let mut transform = Transform::from_translation(Vec3::ZERO);
        transform.look_to(Vec3::new(-0.6, 0.25, 0.9), up);
        apply_level_lock(&mut transform, up, 1.0);
        let once = transform.rotation;

        for weight in [1.0, 0.5, 0.05] {
            apply_level_lock(&mut transform, up, weight);
            assert!(
                transform.rotation.dot(once).abs() > 1.0 - 1.0e-6,
                "weight {weight} moved a level pose: {:?} vs {once:?}",
                transform.rotation
            );
        }
    }

    /// The blended control reference is what keeps mouse-look and R/F from
    /// switching feel on a threshold.
    #[test]
    fn control_up_interpolates_between_camera_and_local_vertical() {
        let up = tilted_up();
        let mut transform = Transform::from_translation(Vec3::ZERO);
        transform.look_to(Vec3::new(1.0, 0.1, 0.4), up);
        transform.rotation *= Quat::from_rotation_z(0.9);
        let camera_up = *transform.up();
        assert!(camera_up.dot(up) < 0.95, "test did not bank");

        assert_eq!(control_up(&transform, None), camera_up);
        let full = control_up(&transform, Some(LevelLock { up, authority: 1.0 }));
        assert!(
            full.dot(up) > 1.0 - 1.0e-5,
            "full authority is not local up"
        );

        let half = control_up(&transform, Some(LevelLock { up, authority: 0.5 }));
        assert!(half.is_normalized());
        assert!(
            half.dot(up) > camera_up.dot(up) && half.dot(up) < full.dot(up),
            "half authority is not between the two references"
        );
    }

    #[test]
    fn level_lock_holds_heading_and_clamps_pitch_short_of_the_pole() {
        let up = tilted_up();
        let heading =
            (Vec3::new(1.0, 0.0, 0.2) - up * Vec3::new(1.0, 0.0, 0.2).dot(up)).normalize();
        let mut transform = Transform::from_translation(Vec3::ZERO);
        // Straight up — past the clamp, and a direction that carries no heading
        // of its own.
        transform.look_to(heading * 0.2 + up * 0.98, up);

        apply_level_lock(&mut transform, up, 1.0);

        let forward = *transform.forward();
        assert!(
            forward.dot(up) <= FREECAM_LEVEL_MAX_SIN_PITCH + 1.0e-5,
            "pitch not clamped: {}",
            forward.dot(up)
        );
        let horizontal = (forward - up * forward.dot(up)).normalize();
        assert!(
            horizontal.dot(heading) > 0.99,
            "heading drifted: {horizontal:?} vs {heading:?}"
        );
    }
}
