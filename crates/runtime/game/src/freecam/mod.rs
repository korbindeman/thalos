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
//! `thalos_viewer` renders the same speed / level / ground-floor state as an
//! on-screen control surface, so none of it is keyboard-only trivia.
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
//! - **Level lock** ([`ViewerPreferences::level_to_up`], default on) constrains the pose to
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
//! - **Ground floor** ([`ViewerPreferences::ground_collision`], default on) clamps the
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
use thalos_viewer::{
    LevelLock, ViewerIntent, ViewerPose, ViewerPreferences, ViewerStatus, drive_motion,
    level_lock_authority, settle_level_lock, update_spring_zoom,
};
use thalos_world::BodyId;

use crate::bridge::WarpLimits;
use crate::camera::{ActiveCamera, OrbitCamera, ShipCamera};
use crate::camera_optics::CameraOptics;
use crate::debug::DebugMode;
use crate::hud::{UiKeyboardGate, UiPointerGate};
use crate::pause_menu::GamePause;
use crate::photo_mode::PhotoMode;
use crate::rendering::transforms::surface_orientation_authored;
use crate::rendering::{SimulationState, SolarSystemState, view_anchor::ViewAnchor};
use crate::settings_menu::SettingsMenu;
use crate::view::ViewMode;

#[derive(Resource, Debug)]
pub struct FreeCam {
    pub active: bool,
    /// When `active`, whether freecam should leave sim time under warp control
    /// instead of freezing [`crate::sim_clock::SimClock`]. Set only on enter
    /// from [`craft_allows_time_warp`]; cleared on exit.
    pub allow_sim_time: bool,
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
    let authority = level_lock_authority(radial.length(), body_radius_m);
    (authority > 0.0)
        .then(|| LevelLock::new(radial.try_normalize()?, authority))
        .flatten()
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
                )
                    .chain()
                    .in_set(crate::SimStage::Camera),
            )
            .add_systems(
                Update,
                project_viewer_status.before(thalos_viewer::ViewpointSet::Input),
            );
    }
}

fn project_viewer_status(
    freecam: Res<FreeCam>,
    photo: Res<PhotoMode>,
    pause: Res<GamePause>,
    settings: Res<SettingsMenu>,
    viewpoints: Option<Res<thalos_viewer::ViewpointUiState>>,
    app_state: Res<State<crate::loading::AppState>>,
    view: Res<ViewMode>,
    shipyard: Option<Res<crate::shipyard_editor::ShipyardEditor>>,
    base_editor: Option<Res<crate::base_editor::BaseEditor>>,
    space_center: Option<Res<crate::space_center::SpaceCenter>>,
    anchor: Res<ViewAnchor>,
    sim: Res<SimulationState>,
    mut status: ResMut<ViewerStatus>,
) {
    let application_modal = *app_state.get() != crate::loading::AppState::Running
        || *view != ViewMode::Ship
        || shipyard.as_deref().is_some_and(|editor| editor.open)
        || base_editor.as_deref().is_some_and(|editor| editor.open)
        || space_center.as_deref().is_some_and(|hub| hub.open);
    status.active = freecam.active;
    status.panel_visible = !photo.active
        && !pause.active
        && !settings.open
        && !viewpoints.as_deref().is_some_and(|state| state.is_open())
        && !application_modal;
    status.interaction_blocked = photo.active || pause.active || settings.open || application_modal;
    let anchor_label = freecam.anchor_body().map_or("inertial", |body| {
        sim.system
            .bodies
            .get(body)
            .map(|definition| definition.name.as_str())
            .unwrap_or("—")
    });
    if status.anchor_label != anchor_label {
        status.anchor_label.clear();
        status.anchor_label.push_str(anchor_label);
    }
    status.altitude_agl_m = anchor
        .resolved
        .filter(|resolved| Some(resolved.body) == freecam.anchor_body())
        .map(|resolved| resolved.agl_m);
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
    settings: Res<SettingsMenu>,
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
    if ui_keyboard.text_entry() || settings.open {
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
    settings: Res<SettingsMenu>,
    mut freecam: ResMut<FreeCam>,
    mut viewer: ResMut<ViewerPreferences>,
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

    // Game input is projected into the shared semantic viewer vocabulary.
    // Reference-frame carry, terrain probing, and big-space projection stay in
    // this adapter because they are planetary facts.
    let ui_pointer_busy = ui_gate.hovered;
    let typing = ui_keyboard.text_entry() || settings.open;
    let pressed = |key| !typing && keys.pressed(key);
    let reference_frame = freecam.reference_frame;
    let lock = viewer
        .level_to_up
        .then(|| local_up_world(reference_frame, &sim, states, grid, &cell, &transform))
        .flatten();

    let axis = |positive, negative| {
        f32::from(u8::from(pressed(positive))) - f32::from(u8::from(pressed(negative)))
    };
    let movement = Vec3::new(
        axis(KeyCode::KeyD, KeyCode::KeyA),
        axis(KeyCode::KeyR, KeyCode::KeyF),
        axis(KeyCode::KeyW, KeyCode::KeyS),
    );
    let mut roll_axis = 0.0;
    if pressed(KeyCode::KeyQ) {
        roll_axis += 1.0;
    }
    if pressed(KeyCode::KeyE) {
        roll_axis -= 1.0;
    }
    let intent = ViewerIntent {
        look_delta: input.camera_motion,
        look_active: !typing && !ui_pointer_busy && input.primary_pressed,
        movement,
        roll_axis,
        speed_scroll_lines: if ui_pointer_busy {
            0.0
        } else {
            input.freecam_speed_scroll_lines
        },
        fast: pressed(KeyCode::ShiftLeft) || pressed(KeyCode::ShiftRight),
        slow: pressed(KeyCode::ControlLeft) || pressed(KeyCode::ControlRight),
        toggle_level: !typing && keys.just_pressed(KeyCode::KeyL),
        toggle_ground: !typing && keys.just_pressed(KeyCode::KeyC),
        spring_zoom: !typing && keys.pressed(KeyCode::KeyZ),
    };
    let mut pose = ViewerPose {
        position: grid.grid_position_double(&cell, &transform),
        rotation: transform.rotation.as_dquat(),
    };
    let mut pose_changed =
        drive_motion(&mut pose, &mut viewer, intent, time.delta_secs_f64(), lock);
    if pose_changed {
        let (next_cell, local) = grid.translation_to_grid(pose.position);
        *cell = next_cell;
        transform.translation = local;
        transform.rotation = pose.rotation.as_quat();
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
        && viewer.ground_collision
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
    let final_lock = viewer
        .level_to_up
        .then(|| local_up_world(reference_frame, &sim, states, grid, &cell, &transform))
        .flatten();
    if sanitize_pose && let Some(lock) = final_lock {
        let mut pose = ViewerPose {
            position: grid.grid_position_double(&cell, &transform),
            rotation: transform.rotation.as_dquat(),
        };
        if settle_level_lock(&mut pose, lock, time.delta_secs_f64()) {
            let (next_cell, local) = grid.translation_to_grid(pose.position);
            *cell = next_cell;
            transform.translation = local;
            transform.rotation = pose.rotation.as_quat();
            // The entry frame has no other reason to persist, and the levelled
            // pose must reach the body-fixed anchor.
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
    settings: Res<SettingsMenu>,
    freecam: Res<FreeCam>,
    mut cam_q: Query<&mut CameraOptics, (With<OrbitCamera>, With<ActiveCamera>, With<ShipCamera>)>,
) {
    let Ok(mut optics) = cam_q.single_mut() else {
        return;
    };
    // The `z` in a typed viewpoint name is a character, not a zoom.
    update_spring_zoom(
        &mut optics,
        freecam.active
            && !ui_keyboard.text_entry()
            && !settings.open
            && keys.pressed(KeyCode::KeyZ),
        time.delta_secs(),
    );
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
}
