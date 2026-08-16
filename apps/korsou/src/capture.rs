use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use bevy::{
    app::AppExit,
    ecs::observer::On,
    prelude::*,
    render::view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk},
};

use crate::{
    camera::CameraRenderTarget, cli::RunConfig, foliage::FoliageStats, terrain::TerrainStats,
};

const SCREENSHOT_DIRECTORY: &str = "artifacts/korsou/screenshots";

pub struct CapturePlugin;

impl Plugin for CapturePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CaptureState {
            settled_frames: 0,
            requested: false,
            manual: ManualCapture::Idle,
        })
        .add_systems(First, restore_ui_after_capture)
        .add_systems(Update, begin_manual_capture)
        .add_systems(Last, (capture_when_settled, capture_without_ui));
    }
}

#[derive(Resource)]
struct CaptureState {
    settled_frames: u32,
    requested: bool,
    manual: ManualCapture,
}

enum ManualCapture {
    Idle,
    Requested(PathBuf),
    RestoreUi,
}

fn capture_when_settled(
    mut commands: Commands,
    stats: Res<TerrainStats>,
    foliage: Option<Res<FoliageStats>>,
    config: Res<RunConfig>,
    render_target: Res<CameraRenderTarget>,
    mut state: ResMut<CaptureState>,
) {
    let Some(capture) = config.capture.as_ref() else {
        return;
    };
    if state.requested {
        return;
    }
    let foliage_settled = foliage.as_deref().is_none_or(|foliage| {
        foliage.bake_ready && foliage.resident == foliage.desired && foliage.queued == 0
    });
    if stats.resident > 0
        && stats.resident == stats.desired
        && stats.queued == 0
        && stats.transitioning == 0
        && foliage_settled
    {
        state.settled_frames += 1;
    } else {
        state.settled_frames = 0;
    }
    if state.settled_frames < 45 {
        return;
    }
    let output = capture.output.clone();
    if let Some(parent) = output.parent()
        && let Err(error) = fs::create_dir_all(parent)
    {
        error!(
            "cannot create capture directory {}: {error}",
            parent.display()
        );
        commands.write_message(AppExit::error());
        return;
    }
    state.requested = true;
    let screenshot = render_target
        .0
        .as_ref()
        .map_or_else(Screenshot::primary_window, |target| {
            Screenshot::image(target.clone())
        });
    commands
        .spawn(screenshot)
        .observe(save_to_disk(output))
        .observe(exit_after_capture);
}

fn begin_manual_capture(
    keys: Res<ButtonInput<KeyCode>>,
    settings: Res<thalos_runtime::preferences::SettingsMenu>,
    mut state: ResMut<CaptureState>,
    mut ui: Query<&mut Visibility, With<thalos_runtime::viewer::ViewerUiRoot>>,
) {
    if settings.open
        || !keys.just_pressed(KeyCode::F2)
        || !matches!(state.manual, ManualCapture::Idle)
    {
        return;
    }

    if let Err(error) = fs::create_dir_all(SCREENSHOT_DIRECTORY) {
        error!("cannot create screenshot directory {SCREENSHOT_DIRECTORY}: {error}");
        return;
    }

    for mut visibility in &mut ui {
        *visibility = Visibility::Hidden;
    }
    state.manual = ManualCapture::Requested(manual_screenshot_path());
}

fn capture_without_ui(mut commands: Commands, mut state: ResMut<CaptureState>) {
    let ManualCapture::Requested(output) = &state.manual else {
        return;
    };
    let output = output.clone();

    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(output));
    state.manual = ManualCapture::RestoreUi;
}

fn restore_ui_after_capture(
    mut state: ResMut<CaptureState>,
    photo_mode: Option<Res<thalos_runtime::photo_mode::PhotoMode>>,
    mut ui: Query<&mut Visibility, With<thalos_runtime::viewer::ViewerUiRoot>>,
) {
    if !matches!(state.manual, ManualCapture::RestoreUi) {
        return;
    }

    let target = if photo_mode.is_some_and(|mode| mode.active) {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    for mut visibility in &mut ui {
        *visibility = target;
    }
    state.manual = ManualCapture::Idle;
}

fn manual_screenshot_path() -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    Path::new(SCREENSHOT_DIRECTORY).join(format!("korsou-{timestamp}.png"))
}

fn exit_after_capture(_capture: On<ScreenshotCaptured>, mut exit: MessageWriter<AppExit>) {
    exit.write(AppExit::Success);
}
