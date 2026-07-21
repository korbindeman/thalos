//! The editor's 3D hangar scene: a dedicated camera on
//! [`crate::coords::EDITOR_LAYER`] with its own clear colour, key/fill
//! lights, a faint floor disc for spatial grounding, and the orbit-camera
//! controls (ported from the standalone editor binary, gated on the Bevy-UI
//! pointer instead of egui).

use bevy::camera::visibility::RenderLayers;
use bevy::input::gestures::PinchGesture;
use bevy::light::cascade::CascadeShadowConfigBuilder;
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::shipyard_editor::core::{
    BuildOrientation, CLICK_THRESHOLD_PX, EditorState, EditorUiGate, EditorViewCamera,
    TankResizeArrow, TankResizeDrag,
};
use thalos_input::shipyard::ShipyardInputIntent;

use crate::coords::EDITOR_LAYER;

/// Marker for the editor's scene camera.
#[derive(Component)]
pub struct EditorCamera;

/// Marker on hangar dressing (lights, floor) so the mode switch can hide it
/// and the layer propagation can claim it.
#[derive(Component)]
pub struct EditorSceneEntity;

/// Orbit state for the editor camera — yaw/pitch around a focus point.
#[derive(Component)]
pub struct EditorOrbit {
    pub focus: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

pub struct EditorScenePlugin;

impl Plugin for EditorScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_editor_scene).add_systems(
            Update,
            (orbit_editor_camera, recenter_on_orientation_change).run_if(super::editor_open),
        );
    }
}

fn setup_editor_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let layer = RenderLayers::layer(EDITOR_LAYER);

    commands.spawn((
        Camera3d::default(),
        Camera {
            is_active: false,
            // Hangar void: a desaturated warm dark, distinct from the
            // space-black flight clear colour, so the editor reads as its
            // own place.
            clear_color: ClearColorConfig::Custom(Color::srgb(0.040, 0.042, 0.048)),
            ..default()
        },
        layer.clone(),
        // The game runs mesh picking with `require_markers: true` (main.rs),
        // so a camera without this marker casts no picking rays — the editor's
        // part meshes could never be clicked or hovered. The flight cameras
        // carry it too (camera.rs); the standalone editor instead runs with
        // `require_markers: false`.
        bevy::picking::mesh_picking::MeshPickingCamera,
        Transform::from_xyz(8.0, 4.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        EditorOrbit {
            focus: Vec3::new(0.0, -2.0, 0.0),
            distance: 12.0,
            yaw: 0.8,
            pitch: 0.4,
        },
        EditorCamera,
        EditorViewCamera,
        // Frosted panels blur this camera's output while the editor is open
        // (the ship camera deactivates, so exactly one marked camera renders).
        thalos_ui::UiBackdropSource,
        Name::new("ShipyardEditorCamera"),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        // Match the world `SunLight`'s cascade count. A bare `DirectionalLight`
        // would default to 4 cascades, and Bevy 0.19's
        // `check_dir_light_mesh_visibility` shares one thread-queue across
        // lights/frames — a 4-vs-2 disagreement over-indexes it and panics
        // (this was the shipyard-open UI-drag crash). See
        // `crate::rendering::SHADOW_CASCADE_COUNT`.
        CascadeShadowConfigBuilder {
            num_cascades: crate::rendering::SHADOW_CASCADE_COUNT,
            ..default()
        }
        .build(),
        layer.clone(),
        Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        EditorSceneEntity,
        Visibility::Hidden,
        Name::new("ShipyardKeyLight"),
    ));
    commands.spawn((
        PointLight {
            intensity: 400_000.0,
            ..default()
        },
        layer.clone(),
        Transform::from_xyz(-6.0, 4.0, -4.0),
        EditorSceneEntity,
        Visibility::Hidden,
        Name::new("ShipyardFillLight"),
    ));

    // Faint floor disc far below the build origin (rockets grow downward
    // from y = 0). Pure spatial grounding — the craft never rests on it.
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(400.0).mesh().resolution(96))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.085, 0.09, 0.10),
            perceptual_roughness: 0.95,
            metallic: 0.0,
            ..default()
        })),
        layer,
        Transform::from_xyz(0.0, -80.0, 0.0)
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
        EditorSceneEntity,
        Visibility::Hidden,
        Name::new("ShipyardFloor"),
    ));
}

/// Re-centre the orbit camera when the build layout flips, so the craft
/// stays framed (it moves from a tall upright stack to a level fuselage).
fn recenter_on_orientation_change(
    orientation: Res<BuildOrientation>,
    mut cam: Query<&mut EditorOrbit>,
) {
    if !orientation.is_changed() {
        return;
    }
    for mut c in cam.iter_mut() {
        c.focus = if orientation.horizontal {
            Vec3::ZERO
        } else {
            Vec3::new(0.0, -2.0, 0.0)
        };
    }
}

/// Orbit / pan / zoom for the editor camera. Same interaction grammar as
/// the standalone editor: drag orbits (held back while a pending part
/// protects its placement click), scroll pans along the build axis, Shift +
/// scroll and pinch zoom.
fn orbit_editor_camera(
    mut cam: Query<(&mut Transform, &mut EditorOrbit)>,
    input: Res<ShipyardInputIntent>,
    mut pinch: MessageReader<PinchGesture>,
    ui_gate: Res<EditorUiGate>,
    state: Res<EditorState>,
    resize_drag: Res<TankResizeDrag>,
    hover_map: Res<HoverMap>,
    orientation: Res<BuildOrientation>,
    arrows: Query<(), With<TankResizeArrow>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut press_cursor: Local<Option<Vec2>>,
    mut orbit_active: Local<bool>,
) {
    let pointer_over_ui = ui_gate.pointer_busy;

    let pointer_on_arrow = hover_map
        .0
        .values()
        .any(|hovers| hovers.keys().any(|e| arrows.get(*e).is_ok()));

    let delta = input.camera_motion;
    let wheel = input.camera_wheel;
    let mut pinch_d: f32 = 0.0;
    for p in pinch.read() {
        pinch_d += p.0;
    }

    let shift = input.precision_slow;

    // Click/drag arbitration for LMB — see the standalone editor's
    // `orbit_camera` for the full rationale: while a part is pending, hold
    // orbit until the cursor moves past the click threshold so the
    // press→release lands as a placement `Pointer<Click>`.
    let cursor = windows.single().ok().and_then(|w| w.cursor_position());
    if input.primary_started {
        *press_cursor = cursor;
        *orbit_active = false;
    }
    if input.primary_released {
        *press_cursor = None;
        *orbit_active = false;
    }
    if !*orbit_active
        && let (Some(press), Some(current)) = (*press_cursor, cursor)
        && (current - press).length() >= CLICK_THRESHOLD_PX
    {
        *orbit_active = true;
    }

    let orbit_allowed = !pointer_over_ui
        && resize_drag.active.is_none()
        && !pointer_on_arrow
        && (state.pending.is_none() || *orbit_active);

    for (mut t, mut orbit) in cam.iter_mut() {
        if orbit_allowed && input.primary_pressed {
            orbit.yaw -= delta.x * 0.005;
            orbit.pitch = (orbit.pitch - delta.y * 0.005).clamp(-1.5, 1.5);
        }

        if !pointer_over_ui && (wheel.x.abs() > 0.0 || wheel.y.abs() > 0.0) {
            if shift {
                orbit.distance = (orbit.distance * (1.0 - wheel.y * 0.05)).clamp(2.0, 200.0);
            } else {
                // Vertical scroll: pan along the build's long axis; the
                // horizontal layout lays that axis down, so use the rotated
                // direction. Horizontal scroll pans screen-left/right.
                if wheel.y.abs() > 0.0 {
                    let pan = wheel.y * orbit.distance * 0.015;
                    orbit.focus += orientation.rotation() * Vec3::Y * pan;
                }
                if wheel.x.abs() > 0.0 {
                    let cam_right = Quat::from_rotation_y(orbit.yaw) * Vec3::X;
                    let pan = wheel.x * orbit.distance * 0.015;
                    orbit.focus += cam_right * pan;
                }
            }
        }

        if !pointer_over_ui && pinch_d.abs() > 0.0 {
            orbit.distance = (orbit.distance * (1.0 - pinch_d * 8.0)).clamp(2.0, 200.0);
        }

        let rot = Quat::from_euler(EulerRot::YXZ, orbit.yaw, -orbit.pitch, 0.0);
        let offset = rot * Vec3::new(0.0, 0.0, orbit.distance);
        t.translation = orbit.focus + offset;
        t.look_at(orbit.focus, Vec3::Y);
    }
}
