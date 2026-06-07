//! Debug-only free-fly camera (ship view).
//!
//! F3 toggles the camera off the orbit-focus pipeline. Bindings:
//!
//! | Key            | Action                       |
//! |----------------|------------------------------|
//! | W / S          | Forward / back               |
//! | A / D          | Strafe left / right          |
//! | R / F          | Up / down (raise / fall)     |
//! | Q / E          | Roll left / right            |
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
//! - Sim time is paused through `sim_clock::SimClock`, while camera motion
//!   continues on wall-clock time.
//! - No terrain collision — flying through a planet is the point.

use bevy::math::DVec3;
use bevy::prelude::*;
use bevy_egui::EguiContexts;
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_input::game::GameInputIntent;

use crate::camera::{ActiveCamera, OrbitCamera, ShipCamera};
use crate::debug::DebugMode;
use crate::view::ViewMode;

#[derive(Resource, Debug)]
pub struct FreeCam {
    pub active: bool,
    /// Base translation speed in m/s before Shift/Ctrl multipliers.
    /// Scroll adjusts this in log-space so the same wheel input feels
    /// proportional at every scale.
    pub base_speed_m_s: f64,
}

impl Default for FreeCam {
    fn default() -> Self {
        Self {
            active: false,
            base_speed_m_s: FREECAM_DEFAULT_SPEED_M_S,
        }
    }
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

pub struct FreeCamPlugin;

impl Plugin for FreeCamPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FreeCam>().add_systems(
            Update,
            (toggle_freecam_system, freecam_drive_system)
                .chain()
                .in_set(crate::SimStage::Camera),
        );
    }
}

fn toggle_freecam_system(
    debug: Option<Res<DebugMode>>,
    view: Res<ViewMode>,
    input: Res<GameInputIntent>,
    mut freecam: ResMut<FreeCam>,
    mut egui: EguiContexts,
) {
    // Auto-disable when leaving ship view — map view has no freecam analogue
    // and we don't want input gating to drift while the user can't recover.
    // [`gate_enhanced_input_sources`] picks up the change next PreUpdate.
    if freecam.active && *view != ViewMode::Ship {
        freecam.active = false;
        return;
    }

    if !input.toggle_free_cam {
        return;
    }
    if let Ok(ctx) = egui.ctx_mut()
        && ctx.wants_keyboard_input()
    {
        return;
    }
    if *view != ViewMode::Ship {
        return;
    }
    if !debug.as_deref().map(|d| d.enabled).unwrap_or(false) {
        return;
    }

    freecam.active = !freecam.active;
}

fn freecam_drive_system(
    time: Res<Time<Real>>,
    keys: Res<ButtonInput<KeyCode>>,
    input: Res<GameInputIntent>,
    mut freecam: ResMut<FreeCam>,
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

    // Scroll adjusts cruise speed in log-space.
    if input.camera_wheel.y != 0.0 {
        let log =
            freecam.base_speed_m_s.ln() + (input.camera_wheel.y as f64) * FREECAM_SCROLL_LOG_STEP;
        freecam.base_speed_m_s = log
            .exp()
            .clamp(FREECAM_MIN_SPEED_M_S, FREECAM_MAX_SPEED_M_S);
    }

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

    if dir == Vec3::ZERO {
        return;
    }

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
}
