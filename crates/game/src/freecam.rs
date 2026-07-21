//! Debug-only free-fly camera (ship view).
//!
//! F4 toggles the camera off the orbit-focus pipeline. Bindings:
//!
//! | Key            | Action                       |
//! |----------------|------------------------------|
//! | W / S          | Forward / back               |
//! | A / D          | Strafe left / right          |
//! | R / F          | Up / down (raise / fall)     |
//! | Q / E          | Roll left / right            |
//! | Z (hold)       | Zoom in (telephoto FOV)      |
//! | LMB-drag       | Yaw + pitch (mouse look)     |
//! | Shift / Ctrl   | 5× / 0.2× speed              |
//! | Scroll wheel   | Adjust base cruise speed     |
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
//! - No terrain collision — flying through a planet is the point.

use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::types::BodyState;
use thalos_world::BodyId;

use crate::bridge::WarpLimits;
use crate::camera::{ActiveCamera, OrbitCamera, ShipCamera};
use crate::debug::DebugMode;
use crate::rendering::{SimulationState, SolarSystemState, view_anchor::ViewAnchor};
use crate::view::ViewMode;

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
    pub base_speed_m_s: f64,
    /// Reference frame selected once per activation. Kept private so no other
    /// system can silently retarget an active freecam.
    reference_frame: FreeCamReferenceFrame,
}

impl Default for FreeCam {
    fn default() -> Self {
        Self {
            active: false,
            allow_sim_time: false,
            base_speed_m_s: FREECAM_DEFAULT_SPEED_M_S,
            reference_frame: FreeCamReferenceFrame::Inertial,
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

const FREECAM_DEFAULT_SPEED_M_S: f64 = 100.0;
const FREECAM_MIN_SPEED_M_S: f64 = 1.0;
const FREECAM_MAX_SPEED_M_S: f64 = 1.0e7;
const FREECAM_SHIFT_MULT: f64 = 5.0;
const FREECAM_CTRL_MULT: f64 = 0.2;
const FREECAM_LOOK_SENSITIVITY: f32 = 0.0025;
const FREECAM_SCROLL_LOG_STEP: f64 = 0.20;
/// Roll rate in rad/s while a roll key is held. ~86°/s — fast enough to
/// re-level from inverted in a couple of seconds without making fine
/// roll adjustments impossible.
const FREECAM_ROLL_RATE_RAD_S: f32 = 1.5;
/// Telephoto zoom factor while Z is held: the field of view narrows to 1/this
/// of the normal FOV (≈45° → ≈11°), magnifying the view ~4×.
const FREECAM_ZOOM_FACTOR: f32 = 4.0;
/// Exponential smoothing rate (1/s) for easing the freecam FOV toward its zoom
/// target, so zoom in/out reads like racking a lens rather than a hard snap.
const FREECAM_ZOOM_LERP_RATE: f32 = 12.0;

pub struct FreeCamPlugin;

impl Plugin for FreeCamPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FreeCam>().add_systems(
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
        );
    }
}

fn toggle_freecam_system(
    debug: Option<Res<DebugMode>>,
    view: Res<ViewMode>,
    input: Res<GameInputIntent>,
    shipyard: Option<Res<crate::shipyard_editor::ShipyardEditor>>,
    sim: Res<SimulationState>,
    warp_limits: Res<WarpLimits>,
    mut freecam: ResMut<FreeCam>,
    ui_text: Res<thalos_ui::TextFieldFocus>,
) {
    // Auto-disable when leaving ship view — map view has no freecam analogue
    // and we don't want input gating to drift while the user can't recover.
    // [`gate_enhanced_input_sources`] picks up the change next PreUpdate.
    if freecam.active && *view != ViewMode::Ship {
        freecam.active = false;
        freecam.allow_sim_time = false;
        freecam.reference_frame = FreeCamReferenceFrame::Inertial;
        return;
    }

    if !input.toggle_free_cam {
        return;
    }
    // The shipyard editor owns the screen; freecam has no scene to fly.
    if shipyard.as_deref().map(|s| s.open).unwrap_or(false) {
        return;
    }
    if ui_text.is_focused() {
        return;
    }
    if *view != ViewMode::Ship {
        return;
    }
    if !debug.as_deref().map(|d| d.enabled).unwrap_or(false) {
        return;
    }

    if freecam.active {
        freecam.active = false;
        freecam.allow_sim_time = false;
        freecam.reference_frame = FreeCamReferenceFrame::Inertial;
    } else {
        freecam.allow_sim_time = craft_allows_time_warp(&sim, &warp_limits);
        freecam.reference_frame = FreeCamReferenceFrame::Pending;
        freecam.active = true;
    }
}

fn freecam_drive_system(
    time: Res<Time<Real>>,
    keys: Res<ButtonInput<KeyCode>>,
    input: Res<GameInputIntent>,
    mut freecam: ResMut<FreeCam>,
    view_anchor: Res<ViewAnchor>,
    solar: Res<SolarSystemState>,
    grid_q: Query<&Grid, With<BigSpace>>,
    mut cam_q: Query<
        (&mut Transform, &mut CellCoord),
        (With<OrbitCamera>, With<ActiveCamera>, With<ShipCamera>),
    >,
) {
    if !freecam.active {
        return;
    }
    let Ok(grid) = grid_q.single() else { return };
    let Ok((mut transform, mut cell)) = cam_q.single_mut() else {
        return;
    };

    let states = solar.states.as_deref();

    // Resolve the reference frame once, after inheriting the normal flight
    // camera's final pose. `ViewAnchor` is the canonical answer to which
    // terrain-backed body the render view belongs to; latching its BodyId
    // avoids discontinuous nearest-body switches while freecam is active.
    if matches!(freecam.reference_frame, FreeCamReferenceFrame::Pending) {
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

    // Scroll adjusts cruise speed in log-space.
    if input.camera_wheel.y != 0.0 {
        let log =
            freecam.base_speed_m_s.ln() + (input.camera_wheel.y as f64) * FREECAM_SCROLL_LOG_STEP;
        freecam.base_speed_m_s = log
            .exp()
            .clamp(FREECAM_MIN_SPEED_M_S, FREECAM_MAX_SPEED_M_S);
    }

    let mut pose_changed = false;

    // Mouse-look while LMB held. Yaw/pitch around the camera's own up/right
    // (intrinsic rotation in camera-local frame) rather than world axes:
    //
    // - The ship-view orbit basis is radial-up, so when freecam activates over
    //   a planet the camera transform inherits a roll relative to world-Y.
    //   Yawing around world-Y in that state rotates the view around a tilted
    //   axis — the camera arcs and tumbles instead of panning. Yawing around
    //   the camera's own up gives a clean "turn your head" feel regardless of
    //   how the camera was oriented on entry.
    // - No roll input: only horizontal yaw and vertical pitch. Roll
    //   accumulates passively only if the user makes circular drag motions,
    //   matching spacecraft-cam convention.
    if input.primary_pressed {
        let delta = input.camera_motion;
        if delta != Vec2::ZERO {
            let yaw = Quat::from_rotation_y(-delta.x * FREECAM_LOOK_SENSITIVITY);
            let pitch = Quat::from_rotation_x(-delta.y * FREECAM_LOOK_SENSITIVITY);
            transform.rotation = transform.rotation * yaw * pitch;
            pose_changed = true;
        }
    }

    // Roll on Q/E around camera-local Z. Positive Z rotation rotates +X (right)
    // toward +Y (up), which from the pilot's POV looking down -Z is a
    // counter-clockwise tilt — i.e. "roll left." E gets the negated angle so
    // it tilts the camera clockwise (right wing down) as conventional.
    let mut roll_input = 0.0_f32;
    if keys.pressed(KeyCode::KeyQ) {
        roll_input += 1.0;
    }
    if keys.pressed(KeyCode::KeyE) {
        roll_input -= 1.0;
    }
    // Freecam is a view/debug affordance, so it keeps moving while sim-time
    // is paused. The escape pause menu still gates the whole camera stage.
    let dt_f32 = time.delta_secs();
    if roll_input != 0.0 {
        let roll = Quat::from_rotation_z(roll_input * FREECAM_ROLL_RATE_RAD_S * dt_f32);
        transform.rotation *= roll;
        pose_changed = true;
    }

    // Translation keys. Read directly: this is a debug tool, and the flight
    // context is suspended while freecam is active so the same keys can't
    // simultaneously drive the ship.
    let mut dir = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        dir += *transform.forward();
    }
    if keys.pressed(KeyCode::KeyS) {
        dir -= *transform.forward();
    }
    if keys.pressed(KeyCode::KeyD) {
        dir += *transform.right();
    }
    if keys.pressed(KeyCode::KeyA) {
        dir -= *transform.right();
    }
    if keys.pressed(KeyCode::KeyR) {
        dir += *transform.up();
    }
    if keys.pressed(KeyCode::KeyF) {
        dir -= *transform.up();
    }

    if dir != Vec3::ZERO {
        let speed_mult = if keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
            FREECAM_SHIFT_MULT
        } else if keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]) {
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

/// Hold Z to zoom the freecam view in (narrow the field of view) like a
/// telephoto lens; release to return to the normal FOV.
///
/// Runs regardless of `freecam.active` so that toggling freecam off — or
/// switching out of ship view — while still zoomed eases the FOV back to
/// normal instead of stranding the camera at a magnified projection.
fn freecam_zoom_system(
    time: Res<Time<Real>>,
    keys: Res<ButtonInput<KeyCode>>,
    freecam: Res<FreeCam>,
    mut cam_q: Query<&mut Projection, (With<OrbitCamera>, With<ActiveCamera>, With<ShipCamera>)>,
) {
    let Ok(mut projection) = cam_q.single_mut() else {
        return;
    };
    // Read the current FOV without yet triggering change detection — most
    // frames the projection is already settled and shouldn't be re-marked.
    let Projection::Perspective(perspective) = projection.as_ref() else {
        return;
    };
    let current = perspective.fov;

    let base_fov = PerspectiveProjection::default().fov;
    let target = if freecam.active && keys.pressed(KeyCode::KeyZ) {
        base_fov / FREECAM_ZOOM_FACTOR
    } else {
        base_fov
    };

    if (current - target).abs() < 1.0e-4 {
        return;
    }

    let smoothing = 1.0 - (-FREECAM_ZOOM_LERP_RATE * time.delta_secs()).exp();
    let next = current + (target - current) * smoothing;
    if let Projection::Perspective(perspective) = projection.as_mut() {
        perspective.fov = next;
    }
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
