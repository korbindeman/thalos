//! Headless loading-screen preview: renders the real loading screen — logotype,
//! progress bar, status line, and the device/memory readout — to one PNG
//! (`just loading-preview`).
//!
//! # Why this exists
//!
//! The loading screen is the only UI in the game that **no capture preset can
//! reach**. It despawns the instant the last load step completes, which is
//! strictly before the capture host takes its shot; holding a real load open
//! long enough to shoot it would mean screenshotting a race whose contents
//! depend on how far streaming happened to get. So the screen was, until this
//! example, the one surface changed blind.
//!
//! # What it is evidence of, and what it is not
//!
//! [`LoadingScreenPreviewPlugin`] spawns the screen with the game's own
//! `spawn_loading_screen` and drives it with the game's own
//! `update_loading_progress_ui` / `update_loading_diagnostics`, so the layout, the label
//! column alignment, and the number formatting are all the real ones. The load
//! is seeded part-way through a surface scenario, so the bar, the status line,
//! and every gauge column show representative content rather than zeros.
//!
//! It is **not** evidence about the load itself: no step here is driven by a
//! real producer, so step ordering, weights, and the `Loading → next`
//! transition still need an in-game check. The GPU and VRAM rows read this
//! machine live, so they differ between runs by design.

use std::time::Duration;

use bevy::app::{AppExit, ScheduleRunnerPlugin};
use bevy::asset::{AssetPlugin, RenderAssetUsages};
use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::ui::IsDefaultUiCamera;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;

use thalos_game_runtime::loading_preview::{
    LoadingScreenPreviewPlugin, LoadingTracker, PerfSamples, StepDesc, VramBarPlugin, step,
};

const OUT_PATH: &str = "artifacts/visual/latest/loading_preview.png";
const WIDTH: u32 = 900;
const HEIGHT: u32 = 620;
/// Frames before the capture: pipeline compile + font atlas fill. The readout
/// refreshes at 4 Hz and the VRAM poller publishes its first sample within
/// ~500 ms, so this window also covers "the numbers have arrived".
const WARMUP_FRAMES: u32 = 120;
const TAIL_FRAMES: u32 = 16;

/// Gauge values for the seeded pane. Representative of a surface scenario
/// mid-load on the development machine — big enough that every column exercises
/// its GiB/MiB switch, not so big that they look like a fault.
const SEED_MAIN_MESHES: u32 = 4_210;
const SEED_MAIN_IMAGES: u32 = 186;
const SEED_TILE_RESIDENT: u32 = 1_284;
const SEED_TILE_MIB: f32 = 512.0;
const SEED_SLAB_MIB: f32 = 1_536.0;
const SEED_TEXTURE_MIB: f32 = 1_120.0;
const SEED_RSS_MIB: f32 = 3_280.0;
const SEED_MESH_CPU_MIB: f32 = 820.0;
const SEED_IMAGE_CPU_MIB: f32 = 244.0;

fn main() {
    std::fs::create_dir_all("artifacts/visual/latest").ok();

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                close_when_requested: false,
                ..default()
            })
            .set(AssetPlugin {
                // Relative to CARGO_MANIFEST_DIR (crates/runtime/game).
                file_path: "../../../assets".to_string(),
                ..default()
            })
            .disable::<WinitPlugin>(),
    )
    .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
        1.0 / 60.0,
    )))
    .add_plugins((LoadingScreenPreviewPlugin, VramBarPlugin))
    .add_systems(Startup, setup)
    .add_systems(Update, drive_capture)
    .run();
}

#[derive(Resource)]
struct CaptureTarget(Handle<Image>);

#[derive(Resource, Default)]
struct CaptureState {
    frames: u32,
    captured: bool,
    tail: u32,
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut tracker: ResMut<LoadingTracker>,
) {
    let mut target = Image::new_fill(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[4, 5, 7, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    target.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT;
    let target = images.add(target);
    commands.spawn((
        Camera2d,
        RenderTarget::Image(ImageRenderTarget::from(target.clone())),
        IsDefaultUiCamera,
    ));
    commands.insert_resource(CaptureTarget(target));
    commands.init_resource::<CaptureState>();

    // A surface scenario's step set, part-way through: bodies installed,
    // terrain streaming. This is what the status line and `step i/N` counter
    // read, and it puts the progress bar somewhere other than 0 % or 100 %.
    tracker.begin([
        StepDesc::new(step::BODIES, "Celestial bodies", 3.0),
        StepDesc::new(step::TERRAIN, "Surface terrain", 1.0),
        StepDesc::new(step::SETTLE, "Settling terrain", 2.0),
    ]);
    tracker.set_total(step::BODIES, 9);
    tracker.advance(step::BODIES, 9);
    tracker.set_detail(step::TERRAIN, "Thalos");
    tracker.set_fraction(step::TERRAIN, 0.45);

    let mut samples = PerfSamples::default();
    samples.seed_gauges(
        SEED_MAIN_MESHES,
        SEED_MAIN_IMAGES,
        SEED_TILE_RESIDENT,
        SEED_TILE_MIB,
        SEED_SLAB_MIB,
        SEED_TEXTURE_MIB,
        SEED_RSS_MIB,
        SEED_MESH_CPU_MIB,
        SEED_IMAGE_CPU_MIB,
    );
    commands.insert_resource(samples);
}

fn drive_capture(
    mut state: ResMut<CaptureState>,
    target: Res<CaptureTarget>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    if state.captured {
        state.tail += 1;
        if state.tail >= TAIL_FRAMES {
            println!("loading-screen preview written to {OUT_PATH}");
            exit.write(AppExit::Success);
        }
        return;
    }
    state.frames += 1;
    if state.frames < WARMUP_FRAMES {
        return;
    }
    commands
        .spawn(Screenshot::image(target.0.clone()))
        .observe(save_to_disk(OUT_PATH));
    state.captured = true;
}
