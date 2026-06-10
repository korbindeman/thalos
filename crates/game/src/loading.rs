//! Startup loading screen — fullscreen bevy_ui overlay shown from the
//! very first frame until every procedural body's bake is installed.
//!
//! Acts as a gate, not just a decoration: the [`AppState::Loading`] state
//! is the default; [`AppState::Running`] is entered exactly once, when
//! `spawn_bodies` has seeded the bake-task totals and every task has
//! finished its main-thread install (GPU upload + entity spawn). Covering
//! the very first frame also masks the brief swapchain-uninitialised
//! magenta flash some Vulkan / Metal drivers emit before the first real
//! render — keeping the loading screen up across the transition means the
//! user never sees an unpainted swapchain.
//!
//! Visual style follows `hud::theme` (same Fira Code, accent gold,
//! warm-black panel fill) so the screen feels like part of the same UI
//! the HUD will draw once gameplay starts.

use bevy::prelude::*;

/// Top-level app state. Starts in [`Loading`] so the very first frame is
/// covered by the loading screen. Transitions to [`Running`] once
/// [`LoadingProgress`] reports `completed >= total` and the totals have
/// been seeded.
#[derive(States, Default, Clone, Eq, PartialEq, Debug, Hash)]
pub enum AppState {
    #[default]
    Loading,
    Running,
}

/// Shared counter driving the progress bar. `total` is seeded by
/// `rendering::spawn::spawn_bodies` once it has dispatched one async
/// bake-load task per procedural body and set `seeded = true`.
/// `completed` is incremented by `rendering::generation::
/// poll_planet_install_tasks` each time an install finishes on the main
/// thread.
///
/// The transition to `AppState::Running` also waits on
/// `initial_terrain_done`. Bake installation only spawns each body's
/// impostor; the ground-LOD terrain entity (the thing the player
/// actually stands on) is spawned lazily by
/// `rendering::terrain_residency`. Holding the loading screen until
/// that residency has fired ensures the player's first frame in
/// `Running` already has a `BodyTerrain` under their feet, so the
/// visibility-swap system at `4 × radius` does not briefly fall back
/// to the flat impostor billboard.
#[derive(Resource, Default)]
pub struct LoadingProgress {
    pub total: usize,
    pub completed: usize,
    /// `true` once `spawn_bodies` has counted every procedural body. The
    /// state transition needs this so it does not fire on frame 0 when
    /// `total == completed == 0` is the trivially-satisfied initial state.
    pub seeded: bool,
    /// `true` once the initial-wanted bodies in `BodyTerrainResidency`
    /// have terrain entities spawned (or have no authored terrain).
    /// Flipped by `terrain_residency::initial_residency_loading_gate`.
    pub initial_terrain_done: bool,
    /// Human-readable line under the progress bar — the most recently
    /// installed body's name.
    pub label: String,
}

#[derive(Component)]
struct LoadingScreenRoot;

#[derive(Component)]
struct LoadingProgressBarFill;

#[derive(Component)]
struct LoadingStatusText;

pub struct LoadingScreenPlugin;

impl Plugin for LoadingScreenPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<AppState>()
            .init_resource::<LoadingProgress>()
            .add_systems(Startup, spawn_loading_screen)
            .add_systems(
                Update,
                (update_loading_progress_ui, advance_to_running)
                    .chain()
                    .run_if(in_state(AppState::Loading)),
            )
            .add_systems(OnExit(AppState::Loading), despawn_loading_screen);
    }
}

// Visual palette — kept local to this module so the loading screen can
// render on frame 1 (before `hud::theme::init_theme` has populated the
// `HudTheme` resource). Values mirror `HudTheme` so the loading screen
// and the in-game HUD feel like the same UI.
const SCREEN_BG: Color = Color::srgb(0.040, 0.038, 0.034);
const TRACK_BG: Color = Color::srgba(0.085, 0.080, 0.070, 1.0);
const TRACK_BORDER: Color = Color::srgba(0.46, 0.43, 0.36, 0.66);
const ACCENT: Color = Color::srgba(0.95, 0.70, 0.28, 1.0);
const TEXT_DIM: Color = Color::srgba(0.62, 0.60, 0.53, 1.0);

const PROGRESS_BAR_WIDTH: f32 = 360.0;
const PROGRESS_BAR_HEIGHT: f32 = 14.0;

fn spawn_loading_screen(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font: Handle<Font> = asset_server.load("fonts/FiraCode-Regular.ttf");

    commands
        .spawn((
            LoadingScreenRoot,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(28.0),
                ..default()
            },
            BackgroundColor(SCREEN_BG),
            // Sits above every other UI element. Pause menu uses 100,
            // HUD panels use the default (0); 1000 is well clear of both
            // so even a misbehaving overlay can't poke through.
            GlobalZIndex(1000),
            Name::new("LoadingScreen"),
        ))
        .with_children(|root| {
            // Title — large, accent-gold, monospace. Letter-spacing widens
            // it a touch so the all-caps reads as a logotype.
            root.spawn((
                Text::new("THALOS"),
                TextFont {
                    font: font.clone(),
                    font_size: 56.0,
                    ..default()
                },
                TextColor(ACCENT),
                TextLayout::new_with_justify(Justify::Center),
                Name::new("LoadingTitle"),
            ));

            // Progress bar track + fill. Two-node nested layout:
            // the outer node is the static "track" with border + clip;
            // the inner node is the gold fill whose `width: Val::Percent`
            // is updated every frame.
            root.spawn((
                Node {
                    width: Val::Px(PROGRESS_BAR_WIDTH),
                    height: Val::Px(PROGRESS_BAR_HEIGHT),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Stretch,
                    overflow: Overflow::clip(),
                    ..default()
                },
                BackgroundColor(TRACK_BG),
                BorderColor::all(TRACK_BORDER),
                Name::new("LoadingProgressTrack"),
            ))
            .with_children(|track| {
                track.spawn((
                    LoadingProgressBarFill,
                    Node {
                        width: Val::Percent(0.0),
                        height: Val::Percent(100.0),
                        ..default()
                    },
                    BackgroundColor(ACCENT),
                    Name::new("LoadingProgressFill"),
                ));
            });

            // Status text under the bar — body name being installed, or
            // a count fallback while no body has finished yet.
            root.spawn((
                LoadingStatusText,
                Text::new("Loading…"),
                TextFont {
                    font,
                    font_size: 13.0,
                    ..default()
                },
                TextColor(TEXT_DIM),
                TextLayout::new_with_justify(Justify::Center),
                Name::new("LoadingStatusText"),
            ));
        });
}

fn update_loading_progress_ui(
    progress: Res<LoadingProgress>,
    mut fill_q: Query<&mut Node, With<LoadingProgressBarFill>>,
    mut text_q: Query<&mut Text, With<LoadingStatusText>>,
) {
    if let Ok(mut node) = fill_q.single_mut() {
        let ratio = if progress.total == 0 {
            0.0
        } else {
            (progress.completed as f32 / progress.total as f32).clamp(0.0, 1.0)
        };
        node.width = Val::Percent(ratio * 100.0);
    }
    if let Ok(mut text) = text_q.single_mut() {
        let new_text = if !progress.seeded {
            "Preparing…".to_string()
        } else if progress.label.is_empty() {
            format!("Loading {} / {}", progress.completed, progress.total)
        } else {
            format!(
                "{} / {}  ·  {}",
                progress.completed, progress.total, progress.label
            )
        };
        if text.0 != new_text {
            **text = new_text;
        }
    }
}

fn advance_to_running(
    progress: Res<LoadingProgress>,
    settle: Res<crate::surface_settle::SurfaceSettle>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    // Wait for `spawn_bodies` to have seeded the totals — otherwise the
    // first frame's trivially-satisfied `0 >= 0` would fire the
    // transition before any work has started. Also wait for the initial
    // residency planner pass to spawn the ground-LOD terrain entity for
    // the body the player is starting on; otherwise the first `Running`
    // frame falls back to the flat impostor billboard until the lazy
    // residency executor catches up.
    //
    // For near-surface spawns (runway, descents, EVA) also wait for the
    // tile streamer to settle the ground at the site — otherwise the first
    // visible frame shows tiles popping in and the runway pad heaving up to
    // the strip. See `crate::surface_settle`.
    if progress.seeded
        && progress.completed >= progress.total
        && progress.initial_terrain_done
        && settle.ready()
    {
        next_state.set(AppState::Running);
    }
}

fn despawn_loading_screen(mut commands: Commands, roots: Query<Entity, With<LoadingScreenRoot>>) {
    for entity in &roots {
        commands.entity(entity).despawn();
    }
}
