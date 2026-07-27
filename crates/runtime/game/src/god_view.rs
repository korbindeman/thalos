//! Shared Cities:Skylines-style god-view orbit camera.
//!
//! Both the base editor and the space-center hub look down at the surface with
//! the same 3/4 establishing view: rather than spawning a second camera, they
//! reposition the existing [`ShipCamera`](crate::camera::ShipCamera) (which
//! carries the `FloatingOrigin`) around a surface focus point, exactly like
//! `runway::update_runway_transform` places the runway — compute the heliocentric
//! world position, convert it to a big_space `(CellCoord, local)` via the root
//! grid, and set the camera transform. The flight camera systems
//! (`SimStage::Camera`) are gated off while either mode is open, so they don't
//! fight this; they resume — snapping back to the ship — the moment the mode
//! closes.
//!
//! This module owns the orbit state ([`GodViewOrbit`], shared because the two
//! modes are mutually exclusive) and the pure per-frame driver
//! ([`drive_god_view`]); each consumer reads its own raw input, resolves its own
//! focus point, and hands both to the driver.

use bevy::gizmos::prelude::{GizmoConfigGroup, GizmoConfigStore};
use bevy::math::DVec3;
use bevy::prelude::*;
use big_space::prelude::{CellCoord, Grid};

/// Gizmo group for the god-view overlays (launch-point highlights, base-editor
/// placement ghosts, space-center hover outlines). The **default** gizmo group
/// is restricted to `MAP_LAYER` (see `rendering::configure_gizmos`), so any
/// overlay meant for the ship camera's god view must draw through this group
/// instead — a default-`Gizmos` drawer is silently invisible in the god view.
#[derive(Default, Reflect, GizmoConfigGroup)]
#[reflect(Default)]
pub struct GodViewGizmos;

/// System set owning the per-frame god-view camera drive (the base editor and
/// space-center hub both place the `ShipCamera` through [`drive_god_view`]).
///
/// Every god-view **cursor picker** — site pick, building placement, launch
/// point, hub building select — must run `.after(GodViewCameraSet)`. The drive
/// writes the camera `Transform` in `Update`; the pickers read that fresh pose
/// (via [`crate::base_editor::cursor_body_dir`], not the one-frame-stale
/// `GlobalTransform`) to stay in sync with the rendered scene when the camera
/// pans fast. Without the ordering a picker might read the `Transform` before the
/// drive writes it, reintroducing the lag.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct GodViewCameraSet;

fn configure_god_view_gizmos(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<GodViewGizmos>();
    config.line.width = 3.0;
    // Overlay affordances must never be swallowed by the pad paving they sit
    // flush on (or the buildings they outline) — draw on top, like the collider
    // debug group.
    config.depth_bias = -1.0;
    config.render_layers = bevy::camera::visibility::RenderLayers::layer(crate::coords::SHIP_LAYER);
}

const DEFAULT_DISTANCE_M: f32 = 500.0;
/// The single canonical god-view boom distance for framing a whole **base**
/// (metres). The kilometre-scale spaceport (a 5 km runway basin) needs a wide
/// 3/4 establishing shot; the close [`DEFAULT_DISTANCE_M`] (~500 m) frames only a
/// corner of it, deep inside the tree-cleared basin. Every god-view opened *over
/// a base* — the space-center hub, the base editor placing buildings, and the
/// launch-point picker — boots at this one distance (via
/// [`GodViewOrbit::reset_over_base`]), so there is **one default god-view per
/// base**, not one per mode. Matches the headless spaceport-aerial / hub
/// screenshot framing.
pub const BASE_ESTABLISHING_DISTANCE_M: f32 = 4000.0;
const DEFAULT_PITCH: f32 = 0.9; // ~51° — a comfortable 3/4 establishing view
const MIN_PITCH: f32 = 0.26; // ~15°, near-horizon
const MAX_PITCH: f32 = 1.2; // ~69°; below 90° so the look-up stays well-defined
const MIN_DISTANCE_M: f32 = 30.0;
const MAX_DISTANCE_M: f32 = 6000.0;
const ORBIT_SENSITIVITY: f32 = 0.005; // rad per pixel
const ZOOM_SENSITIVITY: f32 = 0.1; // fraction per scroll line
/// WASD pan speed as a fraction of the boom distance per second.
const PAN_SPEED_FACTOR: f32 = 0.9;

/// The world-space frame the god-view is looking at: a point on a body's surface
/// plus its local vertical. All positions are heliocentric metres (the big_space
/// absolute frame — see `rendering::real_space`).
#[derive(Clone, Copy)]
pub struct GodViewFocus {
    /// Focus point in heliocentric metres.
    pub center_world: DVec3,
    /// Local vertical at the focus (world-space, unit).
    pub up_world: DVec3,
}

/// Orbit state for the god-view: yaw/pitch around the focus, the boom distance,
/// and a WASD-driven pan offset of the focus across the ground. Shared by the
/// base editor and the space-center hub (mutually exclusive), and reset to a
/// default establishing view each time a mode opens (via [`Self::reset`]).
#[derive(Resource)]
pub struct GodViewOrbit {
    yaw: f32,
    pitch: f32,
    distance: f32,
    /// World-space (tangent) pan offset added to the focus point.
    pan_world: DVec3,
}

impl Default for GodViewOrbit {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: DEFAULT_PITCH,
            distance: DEFAULT_DISTANCE_M,
            pan_world: DVec3::ZERO,
        }
    }
}

impl GodViewOrbit {
    /// Fresh **close** establishing view — for a god-view over open terrain (the
    /// base editor picking a *new* site, where there is no base to frame). For a
    /// view over an existing base use [`Self::reset_over_base`].
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Fresh establishing view framed to take in a whole **base** — the single
    /// default god-view every base-focused mode (hub / place-buildings /
    /// launch-select) opens at (see [`BASE_ESTABLISHING_DISTANCE_M`]).
    pub fn reset_over_base(&mut self) {
        self.reset();
        self.distance = BASE_ESTABLISHING_DISTANCE_M.clamp(MIN_DISTANCE_M, MAX_DISTANCE_M);
    }
}

/// Raw per-frame input for the god-view, gathered by each consumer.
pub struct GodViewInput {
    /// Whether the cursor is over a UI panel (suppresses camera control).
    pub over_ui: bool,
    /// Whether a text field owns the keyboard (suppresses the WASD pan) —
    /// `crate::hud::UiKeyboardGate`, which covers both the native fields and
    /// the egui viewpoint manager.
    pub text_entry: bool,
    /// Whether right mouse is held (orbit drag).
    pub orbit_held: bool,
    /// Accumulated mouse-motion delta this frame.
    pub drag: Vec2,
    /// Accumulated scroll delta this frame.
    pub scroll: f32,
    /// Seconds since last frame (`Time<Real>`), for WASD pan.
    pub dt: f32,
}

/// Place the ship camera at the god-view defined by `orbit` around `focus`.
/// Right-drag orbits, scroll zooms, WASD pans the focus across the ground.
/// Ungated by `SimStage` — the caller runs it while the mode pauses the sim.
/// Every view-dependent detail system (scatter, shadow cascade) follows the
/// camera by itself via [`crate::rendering::view_anchor`] — no per-mode focus
/// plumbing.
#[allow(clippy::too_many_arguments)]
pub fn drive_god_view(
    focus: GodViewFocus,
    orbit: &mut GodViewOrbit,
    keys: &ButtonInput<KeyCode>,
    input: GodViewInput,
    root_grid: &Grid,
    transform: &mut Transform,
    cell: &mut CellCoord,
) {
    if !input.over_ui && input.orbit_held && input.drag != Vec2::ZERO {
        orbit.yaw -= input.drag.x * ORBIT_SENSITIVITY;
        orbit.pitch = (orbit.pitch + input.drag.y * ORBIT_SENSITIVITY).clamp(MIN_PITCH, MAX_PITCH);
    }
    if !input.over_ui && input.scroll != 0.0 {
        orbit.distance = (orbit.distance * (1.0 - input.scroll * ZOOM_SENSITIVITY))
            .clamp(MIN_DISTANCE_M, MAX_DISTANCE_M);
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
    // speed scales with zoom so it feels constant on screen. Read raw, so the
    // text-entry gate has to be applied here: typing a name over the hub must
    // not slide the view out from under it.
    if !input.over_ui && !input.text_entry {
        let speed = (orbit.distance * PAN_SPEED_FACTOR * input.dt) as f64;
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
    let offset_dir = horiz * pitch.cos() + up * pitch.sin();
    let camera_pos_world = focus_center + offset_dir * orbit.distance as f64;
    let to_focus = (focus_center - camera_pos_world).normalize();

    let (next_cell, local) = root_grid.translation_to_grid(camera_pos_world);
    *cell = next_cell;
    *transform = Transform::from_translation(local).looking_to(to_focus.as_vec3(), up.as_vec3());
}

pub struct GodViewPlugin;

impl Plugin for GodViewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GodViewOrbit>()
            .init_gizmo_group::<GodViewGizmos>()
            .add_systems(Startup, configure_god_view_gizmos);
    }
}
