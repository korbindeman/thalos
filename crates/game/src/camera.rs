use bevy::core_pipeline::Skybox;
use bevy::input::mouse::{AccumulatedMouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::render::render_resource::{TextureViewDescriptor, TextureViewDimension};

/// Plugin that registers the orbit camera systems and spawns the camera entity.
pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BlockCameraInput>()
            .insert_resource(CameraFocus::default())
            .add_systems(Startup, spawn_camera)
            .add_systems(
                Update,
                (
                    camera_min_distance_system,
                    camera_input_system,
                    camera_zoom_interpolation_system,
                    camera_focus_offset_decay_system,
                    camera_transform_system,
                    setup_skybox_cubemap,
                )
                    .chain()
                    .in_set(crate::SimStage::Camera),
            );
    }
}

/// Holds the skybox image handle so we can reinterpret it as a cubemap once loaded.
#[derive(Resource)]
struct SkyboxImage(Handle<Image>);

/// Marker component placed on the orbit camera entity.
#[derive(Component)]
pub struct OrbitCamera;

/// Set to true by the maneuver plugin when the pointer is over a maneuver
/// element (arrow, slide sphere) or an active drag/placement is in progress.
/// Camera rotation is suppressed while this is set.
#[derive(Resource, Default)]
pub struct BlockCameraInput(pub bool);

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

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

fn spawn_camera(mut commands: Commands, asset_server: Res<AssetServer>) {
    let skybox_handle: Handle<Image> = asset_server.load("skybox.png");

    commands.spawn((
        Camera3d::default(),
        OrbitCamera,
        Skybox {
            image: skybox_handle.clone(),
            brightness: 500.0,
            rotation: Quat::IDENTITY,
        },
        // Transform is overwritten every frame by camera_transform_system.
        // We set a sane default so the first frame renders something.
        Transform::from_xyz(0.0, 0.0, 5e6).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.insert_resource(SkyboxImage(skybox_handle));
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Reinterprets the skybox image from a vertical strip (6 stacked faces) into a
/// cubemap texture. Runs once after the image finishes loading.
fn setup_skybox_cubemap(
    skybox: Option<Res<SkyboxImage>>,
    mut images: ResMut<Assets<Image>>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    let Some(skybox) = skybox else { return };
    let Some(image) = images.get_mut(&skybox.0) else {
        return;
    };

    let _ = image.reinterpret_stacked_2d_as_array(image.height() / image.width());
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });

    *done = true;
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
        let ticks = event.y as f64;
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
) {
    let min = match focus.target.and_then(|e| bodies.get(e).ok()) {
        Some(body) => (body.radius_m * SURFACE_MARGIN).max(DISTANCE_MIN_DEFAULT),
        None => DISTANCE_MIN_DEFAULT,
    };
    focus.min_distance = min;
    // If target_distance is already below the new min, clamp it.
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
/// Spherical → Cartesian:
/// ```text
///   x = distance * cos(elevation) * sin(azimuth)
///   y = distance * sin(elevation)
///   z = distance * cos(elevation) * cos(azimuth)
/// ```
/// The camera then looks toward the target position.
pub fn camera_transform_system(
    focus: Res<CameraFocus>,
    target_query: Query<&Transform, Without<OrbitCamera>>,
    mut camera_query: Query<&mut Transform, With<OrbitCamera>>,
) {
    let Ok(mut camera_transform) = camera_query.single_mut() else {
        return;
    };

    // Resolve the target's world position (default to origin when unset),
    // then apply the focus offset for smooth transitions between targets.
    let target_pos: Vec3 = focus
        .target
        .and_then(|entity| target_query.get(entity).ok())
        .map(|t| t.translation)
        .unwrap_or(Vec3::ZERO)
        + focus.focus_offset;

    // Spherical → Cartesian offset.
    // `focus.distance` is in metres; convert to render units (1 unit = 1000 km)
    // so the offset is in the same coordinate system as `target_pos`.
    let distance = focus.distance * crate::rendering::RENDER_SCALE;
    let az = focus.azimuth as f64;
    let el = focus.elevation as f64;

    let cos_el = el.cos();
    let offset = Vec3::new(
        (cos_el * az.sin() * distance) as f32,
        (el.sin() * distance) as f32,
        (cos_el * az.cos() * distance) as f32,
    );

    let camera_pos = target_pos + offset;

    // Orient the camera toward the target, keeping +Y as the world up axis.
    // `looking_at` handles the degenerate case gracefully when the camera is
    // directly above/below (elevation ≈ ±90°), which we already prevent by
    // clamping to ±89°.
    *camera_transform = Transform::from_translation(camera_pos).looking_at(target_pos, Vec3::Y);
}
