//! God-view camera for the base editor.
//!
//! Rather than spawning a second camera, this repositions the existing
//! [`ShipCamera`] (which carries the `FloatingOrigin`) to a Cities:Skylines-style
//! 3/4 view over the build site, exactly like `runway::update_runway_transform`
//! places the runway: compute the heliocentric world position, convert it to a
//! big_space `(CellCoord, local)` via the root grid, and set the camera
//! transform. The flight camera systems (`SimStage::Camera`) are gated off while
//! the editor is open, so they don't fight this; they resume — snapping back to
//! the ship — the moment the editor closes.
//!
//! Controls (raw input, since gameplay contexts are suppressed): right-drag
//! orbits, scroll zooms. The focus point comes from [`super::compute_focus`].

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::math::DVec3;
use bevy::prelude::*;
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_physics_local::HeightSourceRegistry;

use crate::camera::ShipCamera;
use crate::rendering::{SimulationState, SolarSystemState};
use crate::structures::StructureRegistry;

use super::{BaseEditor, base_editor_closed, base_editor_open, compute_focus};

const DEFAULT_DISTANCE_M: f32 = 500.0;
const DEFAULT_PITCH: f32 = 0.9; // ~51° — a comfortable 3/4 establishing view
const MIN_PITCH: f32 = 0.26; // ~15°, near-horizon
const MAX_PITCH: f32 = 1.2; // ~69°; below 90° so the look-up stays well-defined
const MIN_DISTANCE_M: f32 = 30.0;
const MAX_DISTANCE_M: f32 = 6000.0;
const ORBIT_SENSITIVITY: f32 = 0.005; // rad per pixel
const ZOOM_SENSITIVITY: f32 = 0.1; // fraction per scroll line
/// WASD pan speed as a fraction of the boom distance per second.
const PAN_SPEED_FACTOR: f32 = 0.9;

/// Orbit state for the base-editor god-view: yaw/pitch around the focus, the
/// boom distance, and a WASD-driven pan offset of the focus across the ground.
/// Reset to a default establishing view each time the editor opens.
#[derive(Resource)]
pub(super) struct BaseCameraOrbit {
    yaw: f32,
    pitch: f32,
    distance: f32,
    /// World-space (tangent) pan offset added to the focus point.
    pan_world: DVec3,
}

impl Default for BaseCameraOrbit {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: DEFAULT_PITCH,
            distance: DEFAULT_DISTANCE_M,
            pan_world: DVec3::ZERO,
        }
    }
}

pub(super) struct BaseEditorCameraPlugin;

impl Plugin for BaseEditorCameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BaseCameraOrbit>()
            .add_systems(
                Update,
                (reset_orbit_on_open, drive_base_editor_camera)
                    .chain()
                    .run_if(base_editor_open),
            )
            // While closed, the sun-shadow cascade reverts to craft-centred + the
            // altitude gate (see `ShadowFocusOverride`).
            .add_systems(Update, clear_shadow_focus.run_if(base_editor_closed));
    }
}

/// Drop the cascade-centre override when the editor is closed.
fn clear_shadow_focus(
    mut shadow_focus: ResMut<crate::rendering::sun_shadow::ShadowFocusOverride>,
) {
    if shadow_focus.center_world.is_some() {
        shadow_focus.center_world = None;
    }
}

/// Fresh establishing view whenever the editor opens.
fn reset_orbit_on_open(editor: Res<BaseEditor>, mut orbit: ResMut<BaseCameraOrbit>) {
    if editor.is_changed() && editor.open {
        *orbit = BaseCameraOrbit::default();
    }
}

/// Place the ship camera at the god-view defined by [`BaseCameraOrbit`] around
/// the editor's focus point. Ungated by `SimStage` so it runs while the editor
/// pauses the sim.
#[allow(clippy::too_many_arguments)]
fn drive_base_editor_camera(
    editor: Res<BaseEditor>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    height_sources: Res<HeightSourceRegistry>,
    registry: Res<StructureRegistry>,
    ui_gate: Res<crate::hud::UiPointerGate>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time<Real>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut orbit: ResMut<BaseCameraOrbit>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut camera: Query<(&mut Transform, &mut CellCoord), With<ShipCamera>>,
    mut shadow_focus: ResMut<crate::rendering::sun_shadow::ShadowFocusOverride>,
) {
    // Always drain the input streams so they don't pile up across frames.
    let drag: Vec2 = motion.read().map(|m| m.delta).sum();
    let scroll: f32 = wheel.read().map(|w| w.y).sum();

    let Some(focus) = compute_focus(&editor, &sim, &solar, &height_sources, &registry) else {
        return;
    };
    let Ok(root_grid) = root_grid.single() else {
        return;
    };
    let Ok((mut transform, mut cell)) = camera.single_mut() else {
        return;
    };

    let over_ui = ui_gate.hovered;
    if !over_ui && mouse_buttons.pressed(MouseButton::Right) && drag != Vec2::ZERO {
        orbit.yaw -= drag.x * ORBIT_SENSITIVITY;
        orbit.pitch = (orbit.pitch + drag.y * ORBIT_SENSITIVITY).clamp(MIN_PITCH, MAX_PITCH);
    }
    if !over_ui && scroll != 0.0 {
        orbit.distance =
            (orbit.distance * (1.0 - scroll * ZOOM_SENSITIVITY)).clamp(MIN_DISTANCE_M, MAX_DISTANCE_M);
    }

    // Build a tangent basis at the focus (east/north on the local horizon).
    let up = focus.up_world;
    let seed = if up.dot(DVec3::Y).abs() < 0.99 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let east = seed.cross(up).normalize();
    let north = up.cross(east).normalize();

    let (yaw, pitch) = (orbit.yaw as f64, orbit.pitch as f64);
    let horiz = east * yaw.cos() + north * yaw.sin();

    // WASD pans the focus across the ground, relative to the camera facing. Pan
    // speed scales with zoom so it feels constant on screen.
    if !over_ui {
        let speed = (orbit.distance * PAN_SPEED_FACTOR * time.delta_secs()) as f64;
        let forward = -horiz; // into the screen, along the ground
        let right = forward.cross(up).normalize();
        if keys.pressed(KeyCode::KeyW) {
            orbit.pan_world += forward * speed;
        }
        if keys.pressed(KeyCode::KeyS) {
            orbit.pan_world -= forward * speed;
        }
        if keys.pressed(KeyCode::KeyD) {
            orbit.pan_world += right * speed;
        }
        if keys.pressed(KeyCode::KeyA) {
            orbit.pan_world -= right * speed;
        }
    }

    let focus_center = focus.center_world + orbit.pan_world;
    // Drive the sun-shadow cascade to follow the god-view focus (not the parked
    // craft), so building shadows render across the whole base + persist when the
    // camera booms out (`ShadowFocusOverride` also bypasses the altitude gate).
    shadow_focus.center_world = Some(focus_center);
    let offset_dir = horiz * pitch.cos() + up * pitch.sin();
    let camera_pos_world = focus_center + offset_dir * orbit.distance as f64;
    let to_focus = (focus_center - camera_pos_world).normalize();

    let (next_cell, local) = root_grid.translation_to_grid(camera_pos_world);
    *cell = next_cell;
    *transform = Transform::from_translation(local).looking_to(to_focus.as_vec3(), up.as_vec3());
}
