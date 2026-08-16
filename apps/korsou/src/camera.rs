use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    app::AppExit,
    asset::RenderAssetUsages,
    camera::{Exposure, PerspectiveProjection, Projection, RenderTarget},
    core_pipeline::tonemapping::Tonemapping,
    image::Image,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
    light::AtmosphereEnvironmentMapLight,
    math::{DQuat, DVec3},
    pbr::AtmosphereSettings,
    post_process::bloom::Bloom,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};

use crate::{
    cli::{RunConfig, ViewpointSelection},
    places::{PlaceCatalog, PlaceSet, PlaceState},
    spatial::TerrainSpatialFrame,
    terrain::TerrainDataset,
    viewpoint::{ViewpointLibrary, ViewpointSet, ViewpointStartupSet, ViewpointUiState},
};

const MAX_CAMERA_ALTITUDE_M: f64 = 60_000.0;
const MIN_SURFACE_CLEARANCE_M: f64 = 2.0;

pub struct TerrainCameraPlugin;

impl Plugin for TerrainCameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraRenderTarget>()
            .add_systems(
                Startup,
                (spawn_camera, project_camera)
                    .chain()
                    .in_set(TerrainCameraSet::Spawn)
                    .after(ViewpointStartupSet::Load),
            )
            .add_systems(
                Update,
                (apply_camera_presets, move_camera)
                    .chain()
                    .in_set(TerrainCameraSet::Movement)
                    .after(ViewpointSet::Input)
                    .before(ViewpointSet::Apply),
            )
            .add_systems(
                Update,
                project_camera
                    .in_set(TerrainCameraSet::Projection)
                    .after(ViewpointSet::Apply),
            )
            .add_systems(
                Update,
                project_viewer_interaction
                    .after(thalos_runtime::preferences::SettingsMenuSet::Chrome)
                    .before(ViewpointSet::Input),
            )
            .add_systems(
                Update,
                project_viewer_display
                    .after(PlaceSet::Locate)
                    .before(ViewpointSet::Ui),
            );
    }
}

fn project_camera(
    spatial: Res<TerrainSpatialFrame>,
    mut camera: Single<(&TerrainCamera, &mut Transform)>,
) {
    let controller = camera.0;
    let forward = controller.rotation_local * DVec3::NEG_Z;
    let up = controller.rotation_local * DVec3::Y;
    let render_position = spatial.project(controller.position_m);
    let render_forward = spatial.project_direction(controller.position_m, forward);
    let render_up = spatial.project_direction(controller.position_m, up);
    *camera.1 = Transform::from_translation(render_position.as_vec3())
        .looking_to(render_forward.as_vec3(), render_up.as_vec3());
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TerrainCameraSet {
    Spawn,
    Movement,
    Projection,
}

#[derive(Default, Resource)]
pub struct CameraRenderTarget(pub Option<Handle<Image>>);

#[derive(Component)]
pub struct TerrainCamera {
    pub position_m: DVec3,
    pub yaw: f64,
    pub pitch: f64,
    pub rotation_local: DQuat,
}

fn spawn_camera(
    mut commands: Commands,
    dataset: Res<TerrainDataset>,
    config: Res<RunConfig>,
    viewpoints: Res<ViewpointLibrary>,
    mut images: ResMut<Assets<Image>>,
    mut render_target: ResMut<CameraRenderTarget>,
    mut exit: MessageWriter<AppExit>,
) {
    if config.is_headless()
        && let Some(error) = viewpoints.load_error()
    {
        error!("headless capture cancelled: {error}");
        exit.write(AppExit::error());
    }

    let pose = resolve_initial_pose(&dataset, &viewpoints, &config.initial_viewpoint, &mut exit);
    let target = config.capture.as_ref().map(|capture| {
        let mut image = Image::new_uninit(
            Extent3d {
                width: capture.width,
                height: capture.height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD,
        );
        image.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT;
        let handle = images.add(image);
        render_target.0 = Some(handle.clone());
        RenderTarget::Image(handle.into())
    });

    let mut entity = commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            near: 0.5,
            far: 220_000.0,
            ..default()
        }),
        Transform::from_translation(pose.position.as_vec3()).with_rotation(pose.rotation),
        TerrainCamera {
            position_m: pose.position,
            yaw: pose.yaw,
            pitch: pose.pitch,
            rotation_local: pose.rotation.as_dquat(),
        },
        AtmosphereSettings {
            aerial_view_lut_max_distance: 120_000.0,
            ..default()
        },
        AtmosphereEnvironmentMapLight::default(),
        Exposure { ev100: 14.5 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        Msaa::Off,
        TemporalAntiAliasing::default(),
        thalos_runtime::preferences::PreferencesCamera::taa(),
        thalos_runtime::preferences::UiBackdropSource,
        Name::new("Terrain camera"),
    ));
    entity.insert((
        thalos_runtime::viewer::ViewerCamera,
        thalos_runtime::viewer::CameraOptics::default(),
    ));
    if let Some(target) = target {
        entity.insert(target);
    }
}

struct CameraPose {
    position: DVec3,
    yaw: f64,
    pitch: f64,
    rotation: Quat,
}

fn resolve_initial_pose(
    dataset: &TerrainDataset,
    viewpoints: &ViewpointLibrary,
    selection: &ViewpointSelection,
    exit: &mut MessageWriter<AppExit>,
) -> CameraPose {
    match selection {
        ViewpointSelection::Default => viewpoints
            .entries()
            .first()
            .and_then(|viewpoint| viewpoint_pose(viewpoint).ok())
            .unwrap_or_else(|| preset_camera_pose(dataset, 1)),
        ViewpointSelection::Preset(preset) => preset_camera_pose(dataset, *preset),
        ViewpointSelection::Named(name) => match viewpoints.find(name) {
            Some(viewpoint) => match viewpoint_pose(viewpoint) {
                Ok(pose) => pose,
                Err(error) => {
                    error!("cannot apply viewpoint {name:?}: {error}");
                    exit.write(AppExit::error());
                    preset_camera_pose(dataset, 1)
                }
            },
            None => {
                let available = viewpoints
                    .entries()
                    .iter()
                    .map(|viewpoint| viewpoint.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                error!("unknown viewpoint `{name}`; available viewpoints: {available}");
                exit.write(AppExit::error());
                preset_camera_pose(dataset, 1)
            }
        },
        ViewpointSelection::Arbitrary {
            position_m,
            target_m,
        } => {
            let (yaw, pitch, rotation) = look_rotation(*position_m, *target_m);
            CameraPose {
                position: *position_m,
                yaw,
                pitch,
                rotation,
            }
        }
    }
}

fn viewpoint_pose(viewpoint: &crate::viewpoint::Viewpoint) -> Result<CameraPose, String> {
    match &viewpoint.frame {
        thalos_runtime::viewer::ViewpointFrame::ProjectedLocal { .. } => {}
        thalos_runtime::viewer::ViewpointFrame::AuthoredBodyFixed { .. } => {
            return Err("body-fixed viewpoints belong to the planetary game".into());
        }
    }
    let rotation = DQuat::from_array(viewpoint.camera_rotation_xyzw).normalize();
    let (yaw, pitch, _) = rotation.to_euler(EulerRot::YXZ);
    Ok(CameraPose {
        position: DVec3::from_array(viewpoint.camera_position_m),
        yaw,
        pitch,
        rotation: rotation.as_quat(),
    })
}

fn preset_camera_pose(dataset: &TerrainDataset, preset: u8) -> CameraPose {
    let (position, target) = preset_pose(dataset, preset);
    let (yaw, pitch, rotation) = look_rotation(position, target);
    CameraPose {
        position,
        yaw,
        pitch,
        rotation,
    }
}

fn apply_camera_presets(
    keys: Res<ButtonInput<KeyCode>>,
    dataset: Res<TerrainDataset>,
    viewpoint_ui: Res<ViewpointUiState>,
    settings: Res<thalos_runtime::preferences::SettingsMenu>,
    mut camera: Single<(&mut TerrainCamera, &mut Transform)>,
) {
    if viewpoint_ui.is_open() || settings.open {
        return;
    }
    let preset = if keys.just_pressed(KeyCode::Digit1) {
        Some(1)
    } else if keys.just_pressed(KeyCode::Digit2) {
        Some(2)
    } else if keys.just_pressed(KeyCode::Digit3) {
        Some(3)
    } else {
        None
    };
    let Some(preset) = preset else {
        return;
    };
    let (position, target) = preset_pose(&dataset, preset);
    let (yaw, pitch, rotation) = look_rotation(position, target);
    camera.0.position_m = position;
    camera.0.yaw = yaw;
    camera.0.pitch = pitch;
    camera.0.rotation_local = rotation.as_dquat();
    camera.1.translation = position.as_vec3();
    camera.1.rotation = rotation;
}

pub(crate) fn preset_pose(dataset: &TerrainDataset, preset: u8) -> (DVec3, DVec3) {
    let bounds = dataset.land_bounds_local_m();
    let width = bounds[2] - bounds[0];
    let depth = bounds[3] - bounds[1];
    let scale = width.max(depth);
    let center_x = (bounds[0] + bounds[2]) * 0.5;
    let center_z = (bounds[1] + bounds[3]) * 0.5;
    match preset {
        1 => (
            DVec3::new(center_x - scale * 0.65, scale, center_z + scale),
            DVec3::new(center_x, 70.0, center_z),
        ),
        2 => {
            let (x, z) =
                nearest_land_point(dataset, bounds[0] + width * 0.18, center_z - depth * 0.08);
            let (target_x, target_z) = nearest_land_point(dataset, x + 5_000.0, z + 500.0);
            (
                DVec3::new(x, f64::from(dataset.dem_height(x, z)) + 620.0, z),
                DVec3::new(
                    target_x,
                    f64::from(dataset.dem_height(target_x, target_z)) + 105.0,
                    target_z,
                ),
            )
        }
        _ => {
            let (x, z) =
                nearest_land_point(dataset, bounds[0] + width * 0.82, center_z + depth * 0.08);
            let (target_x, target_z) = nearest_land_point(dataset, x - 5_000.0, z - 500.0);
            (
                DVec3::new(x, f64::from(dataset.dem_height(x, z)) + 760.0, z),
                DVec3::new(
                    target_x,
                    f64::from(dataset.dem_height(target_x, target_z)) + 95.0,
                    target_z,
                ),
            )
        }
    }
}

fn nearest_land_point(dataset: &TerrainDataset, preferred_x: f64, preferred_z: f64) -> (f64, f64) {
    let bounds = dataset.land_bounds_local_m();
    let mut best = None;
    let step = 240.0;
    let mut z = bounds[1];
    while z <= bounds[3] {
        let mut x = bounds[0];
        while x <= bounds[2] {
            if dataset.is_land(x, z) {
                let distance = (x - preferred_x).hypot(z - preferred_z);
                if best.is_none_or(|(_, _, best_distance)| distance < best_distance) {
                    best = Some((x, z, distance));
                }
            }
            x += step;
        }
        z += step;
    }
    best.map_or((preferred_x, preferred_z), |(x, z, _)| (x, z))
}

#[allow(clippy::too_many_arguments)]
fn move_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    dataset: Res<TerrainDataset>,
    viewpoint_ui: Res<ViewpointUiState>,
    settings: Res<thalos_runtime::preferences::SettingsMenu>,
    capture: Res<thalos_runtime::viewer::ViewerUiCapture>,
    mut preferences: ResMut<thalos_runtime::viewer::ViewerPreferences>,
    mut camera: Single<(
        &mut TerrainCamera,
        &mut thalos_runtime::viewer::CameraOptics,
    )>,
) {
    let blocked = viewpoint_ui.is_open() || settings.open;
    thalos_runtime::viewer::update_spring_zoom(
        &mut camera.1,
        !blocked && keys.pressed(KeyCode::KeyZ),
        time.delta_secs(),
    );
    if blocked {
        return;
    }

    let pressed = |key| keys.pressed(key);
    let axis = |positive, negative| {
        f32::from(u8::from(pressed(positive))) - f32::from(u8::from(pressed(negative)))
    };
    let scroll_lines = freecam_speed_scroll_lines(mouse_scroll.delta.y, mouse_scroll.unit);
    let intent = thalos_runtime::viewer::ViewerIntent {
        look_delta: mouse_motion.delta,
        look_active: buttons.pressed(MouseButton::Left) && !capture.pointer_busy,
        movement: Vec3::new(
            axis(KeyCode::KeyD, KeyCode::KeyA),
            axis(KeyCode::KeyR, KeyCode::KeyF),
            axis(KeyCode::KeyW, KeyCode::KeyS),
        ),
        roll_axis: axis(KeyCode::KeyQ, KeyCode::KeyE),
        speed_scroll_lines: if capture.pointer_busy {
            0.0
        } else {
            scroll_lines
        },
        fast: pressed(KeyCode::ShiftLeft) || pressed(KeyCode::ShiftRight),
        slow: pressed(KeyCode::ControlLeft) || pressed(KeyCode::ControlRight),
        toggle_level: keys.just_pressed(KeyCode::KeyL),
        toggle_ground: keys.just_pressed(KeyCode::KeyC),
        spring_zoom: keys.pressed(KeyCode::KeyZ),
    };
    let level = thalos_runtime::viewer::LevelLock::new(DVec3::Y, 1.0);
    let mut pose = thalos_runtime::viewer::ViewerPose {
        position: camera.0.position_m,
        rotation: camera.0.rotation_local,
    };
    thalos_runtime::viewer::drive_motion(
        &mut pose,
        &mut preferences,
        intent,
        time.delta_secs_f64(),
        level,
    );

    let bounds = dataset.metadata.quadtree.domain_bounds_local_m;
    let clamped_x = pose.position.x.clamp(bounds[0], bounds[2]);
    let clamped_z = pose.position.z.clamp(bounds[1], bounds[3]);
    let ground = dataset.dem_height(clamped_x, clamped_z) as f64;
    pose.position = constrain_to_map(pose.position, bounds, ground, preferences.ground_collision);
    if preferences.level_to_up
        && let Some(level) = level
    {
        thalos_runtime::viewer::settle_level_lock(&mut pose, level, time.delta_secs_f64());
    }

    camera.0.position_m = pose.position;
    camera.0.rotation_local = pose.rotation;
    let (yaw, pitch, _) = pose.rotation.to_euler(EulerRot::YXZ);
    camera.0.yaw = yaw;
    camera.0.pitch = pitch;
}

fn project_viewer_interaction(
    settings: Res<thalos_runtime::preferences::SettingsMenu>,
    mut status: ResMut<thalos_runtime::viewer::ViewerStatus>,
) {
    status.interaction_blocked = settings.open;
}

fn project_viewer_display(
    camera: Single<&TerrainCamera>,
    dataset: Res<TerrainDataset>,
    places: Res<PlaceCatalog>,
    place_state: Res<PlaceState>,
    viewpoint_ui: Res<ViewpointUiState>,
    settings: Res<thalos_runtime::preferences::SettingsMenu>,
    mut status: ResMut<thalos_runtime::viewer::ViewerStatus>,
) {
    let ground = dataset.dem_height(camera.position_m.x, camera.position_m.z) as f64;
    status.active = true;
    status.panel_visible = !viewpoint_ui.is_open() && !settings.open;
    let location = place_state
        .current_area(&places)
        .map_or("Curaçao", |place| place.name.as_str());
    if status.anchor_label != location {
        status.anchor_label.clear();
        status.anchor_label.push_str(location);
    }
    status.altitude_agl_m = Some(camera.position_m.y - ground);
}

fn constrain_to_map(
    mut position: DVec3,
    bounds: [f64; 4],
    ground: f64,
    ground_collision: bool,
) -> DVec3 {
    position.x = position.x.clamp(bounds[0], bounds[2]);
    position.z = position.z.clamp(bounds[1], bounds[3]);
    let minimum = if ground_collision {
        ground + MIN_SURFACE_CLEARANCE_M
    } else {
        -MAX_CAMERA_ALTITUDE_M
    };
    position.y = position.y.clamp(minimum, MAX_CAMERA_ALTITUDE_M);
    position
}

/// Match Thalos freecam semantics without changing ordinary wheel direction.
fn freecam_speed_scroll_lines(delta_y: f32, unit: MouseScrollUnit) -> f32 {
    match unit {
        MouseScrollUnit::Line => delta_y,
        MouseScrollUnit::Pixel => -delta_y / 35.0,
    }
}

fn look_rotation(position: DVec3, target: DVec3) -> (f64, f64, Quat) {
    let transform =
        Transform::from_translation(position.as_vec3()).looking_at(target.as_vec3(), Vec3::Y);
    let (yaw, pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
    (yaw as f64, pitch as f64, transform.rotation)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn locate_after_camera_movement() {}

    #[test]
    fn camera_and_viewpoint_schedule_is_acyclic() {
        let mut app = App::new();
        app.add_plugins(thalos_runtime::viewer::ViewpointPlugin::new(
            PathBuf::from("unused-viewpoints.json"),
            false,
        ))
        .add_plugins(TerrainCameraPlugin)
        .add_systems(
            Update,
            locate_after_camera_movement
                .in_set(PlaceSet::Locate)
                .after(TerrainCameraSet::Movement),
        );

        let mut update = app
            .world_mut()
            .resource_mut::<Schedules>()
            .remove(Update)
            .expect("Update schedule exists");

        assert!(
            update.initialize(app.world_mut()).is_ok(),
            "Korsou camera/viewpoint ordering must not form a cycle"
        );
    }

    #[test]
    fn camera_stays_inside_the_playable_volume() {
        let bounds = [-30_720.0, -30_720.0, 30_720.0, 30_720.0];

        let below_and_outside =
            constrain_to_map(DVec3::new(-40_000.0, -10.0, 50_000.0), bounds, 25.0, true);
        assert_eq!(below_and_outside, DVec3::new(-30_720.0, 27.0, 30_720.0));

        let above = constrain_to_map(DVec3::new(100.0, 80_000.0, -200.0), bounds, 0.0, true);
        assert_eq!(above, DVec3::new(100.0, MAX_CAMERA_ALTITUDE_M, -200.0));

        let underground = constrain_to_map(DVec3::new(0.0, -500.0, 0.0), bounds, 25.0, false);
        assert_eq!(underground.y, -500.0);
    }

    #[test]
    fn freecam_speed_reverses_trackpad_but_not_wheel_scroll() {
        assert_eq!(
            freecam_speed_scroll_lines(-70.0, MouseScrollUnit::Pixel),
            2.0
        );
        assert_eq!(
            freecam_speed_scroll_lines(-2.0, MouseScrollUnit::Line),
            -2.0
        );
    }
}
