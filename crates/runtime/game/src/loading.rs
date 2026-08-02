//! Startup loading screen + the flexible loading-step tracker.
//!
//! Acts as a gate, not just a decoration: the [`AppState::Loading`] state
//! is the default; the screen stays up until every step registered in
//! [`LoadingTracker`] reports complete, then transitions to the
//! [`LoadDestination`] (the start screen for a bare launch, straight into
//! [`AppState::Running`] for `just game <scenario>`). Covering the very
//! first frame also masks the brief swapchain-uninitialised magenta flash
//! some Vulkan / Metal drivers emit before the first real render.
//!
//! # The tracker
//!
//! Loading work is declared as **steps** ([`LoadingTracker::begin`]): each
//! has an id, a display label, a weight (its share of the progress bar),
//! and either counted (`n / total`) or binary completion. Producer systems
//! update *their* step by id and never see each other:
//!
//! - [`step::BODIES`] — per-body bake install. Total seeded by
//!   `rendering::spawn::spawn_bodies`, advanced by
//!   `rendering::generation::poll_planet_install_tasks` (detail = the body
//!   name just installed).
//! - [`step::TERRAIN`] — ground-LOD terrain for the initially-wanted
//!   bodies. Completed by
//!   `rendering::terrain_residency::initial_residency_loading_gate`.
//! - [`step::PLACEMENT`] — deferred terrain-aware craft placement
//!   (descents via `spawn::refine_descent_spawn`, runway scenarios via
//!   `runway::finish_runway_spawn`). Registered only for scenarios with a
//!   deferred placement; the reveal now waits for it, so the first visible
//!   frame has the craft in its real scenario state.
//! - [`step::SETTLE`] — tile streaming settled under the view
//!   (`crate::surface_settle`). Registered only for the parked runway
//!   start.
//!
//! Updates to an unregistered step are no-ops, so producers don't need to
//! know which scenario is loading. A load can be re-armed at runtime
//! (`begin` again + `NextState(AppState::Loading)`) — the session loader does
//! this for every in-process replacement.
//!
//! Visual style follows `hud::theme` (same Fira Code, accent gold,
//! warm-black panel fill) so the screen feels like part of the same UI
//! the HUD will draw once gameplay starts.

use bevy::prelude::*;

use crate::spawn::SpawnSituation;

// `AppState` / `WorldState` / `LoadDestination` moved to the game-state
// blackboard (Phase 5a, ADR-20260731T024003Z); re-exported so every
// existing path keeps working.
pub use thalos_game_state::app::{AppState, LoadDestination, WorldState};

/// Well-known step ids. Plain strings so future systems can add their own
/// steps without touching this module.
pub mod step {
    /// Session-level craft/state projection for a live-world replacement.
    pub const SESSION: &str = "session";
    /// Per-body bake load + install (counted).
    pub const BODIES: &str = "bodies";
    /// Ground-LOD terrain spawned for the initially-wanted bodies.
    pub const TERRAIN: &str = "terrain";
    /// Deferred terrain-aware craft placement (descents, runway).
    pub const PLACEMENT: &str = "placement";
    /// Tile streaming settled under the view (parked runway).
    pub const SETTLE: &str = "settle";
}

/// Descriptor passed to [`LoadingTracker::begin`].
pub struct StepDesc {
    pub id: &'static str,
    pub label: &'static str,
    /// Relative share of the overall progress bar.
    pub weight: f32,
}

impl StepDesc {
    pub const fn new(id: &'static str, label: &'static str, weight: f32) -> Self {
        Self { id, label, weight }
    }
}

/// One registered unit of loading work.
pub struct LoadStep {
    pub id: &'static str,
    pub label: &'static str,
    /// Mutable sub-label shown after the step label (body name, LOD…).
    pub detail: String,
    weight: f32,
    /// Counted steps: progress is `done / total`. `total` stays `None`
    /// until the producer seeds it, so an unseeded counter is never
    /// trivially complete (`0 >= 0`).
    done: usize,
    total: Option<usize>,
    /// Smooth 0..1 progress for binary steps that can estimate it
    /// (settle). Counted steps derive it from `done / total`.
    fraction: f32,
    complete: bool,
}

impl LoadStep {
    fn progress(&self) -> f32 {
        if self.complete {
            return 1.0;
        }
        match self.total {
            Some(total) if total > 0 => (self.done as f32 / total as f32).clamp(0.0, 1.0),
            _ => self.fraction.clamp(0.0, 1.0),
        }
    }

    /// `"3/9"`-style counter for counted steps, empty otherwise.
    fn counter(&self) -> Option<String> {
        self.total.map(|total| format!("{}/{}", self.done, total))
    }
}

/// Declarative loading-step registry driving the loading screen and the
/// `Loading → next` transition.
///
/// **Sole registrar:** [`register_boot_steps`] at startup and the session-load
/// coordinator ([`crate::session_loading`]) at runtime; both go through
/// [`begin`](Self::begin). Producer systems only update steps.
#[derive(Resource, Default)]
pub struct LoadingTracker {
    steps: Vec<LoadStep>,
    /// `true` once `begin` has registered the step set for this load —
    /// guards the empty-on-frame-0 tracker from reading as complete.
    sealed: bool,
}

impl LoadingTracker {
    /// Reset the tracker and register the full step set for one load.
    pub fn begin(&mut self, steps: impl IntoIterator<Item = StepDesc>) {
        self.steps = steps
            .into_iter()
            .map(|desc| LoadStep {
                id: desc.id,
                label: desc.label,
                detail: String::new(),
                weight: desc.weight.max(f32::EPSILON),
                done: 0,
                total: None,
                fraction: 0.0,
                complete: false,
            })
            .collect();
        self.sealed = true;
    }

    fn get_mut(&mut self, id: &str) -> Option<&mut LoadStep> {
        self.steps.iter_mut().find(|s| s.id == id)
    }

    pub fn has_step(&self, id: &str) -> bool {
        self.steps.iter().any(|s| s.id == id)
    }

    pub fn is_step_complete(&self, id: &str) -> bool {
        self.steps.iter().any(|s| s.id == id && s.complete)
    }

    /// Seed a counted step's total. Completes immediately when `done`
    /// already covers it (including `total == 0`).
    pub fn set_total(&mut self, id: &str, total: usize) {
        if let Some(s) = self.get_mut(id) {
            s.total = Some(total);
            s.complete = s.done >= total;
        }
    }

    /// Advance a counted step; completes it when `done` reaches the total.
    /// (Counted-step incrementing isn't wired to a producer yet — the only
    /// counted step today is seeded with `set_total(.., 0)` and completes
    /// immediately — but it's the documented complement of `set_total`.)
    #[allow(dead_code)]
    pub fn advance(&mut self, id: &str, n: usize) {
        if let Some(s) = self.get_mut(id) {
            s.done += n;
            if let Some(total) = s.total {
                s.complete = s.done >= total;
            }
        }
    }

    pub fn set_detail(&mut self, id: &str, detail: impl Into<String>) {
        if let Some(s) = self.get_mut(id) {
            s.detail = detail.into();
        }
    }

    /// Smooth progress estimate for a binary step (no completion side
    /// effect — call [`complete`](Self::complete) for that).
    pub fn set_fraction(&mut self, id: &str, fraction: f32) {
        if let Some(s) = self.get_mut(id) {
            s.fraction = fraction.clamp(0.0, 1.0);
        }
    }

    pub fn complete(&mut self, id: &str) {
        if let Some(s) = self.get_mut(id) {
            s.complete = true;
        }
    }

    /// Weighted overall progress in 0..1.
    pub fn overall(&self) -> f32 {
        let total_weight: f32 = self.steps.iter().map(|s| s.weight).sum();
        if total_weight <= 0.0 {
            return if self.sealed { 1.0 } else { 0.0 };
        }
        self.steps
            .iter()
            .map(|s| s.weight * s.progress())
            .sum::<f32>()
            / total_weight
    }

    /// First incomplete step, in registration order.
    pub fn active_step(&self) -> Option<&LoadStep> {
        self.steps.iter().find(|s| !s.complete)
    }

    /// `(1-based index of the active step, step count)` for the
    /// `step i/N` readout.
    pub fn step_position(&self) -> (usize, usize) {
        let total = self.steps.len();
        let index = self
            .steps
            .iter()
            .position(|s| !s.complete)
            .map(|i| i + 1)
            .unwrap_or(total);
        (index, total)
    }

    pub fn is_complete(&self) -> bool {
        self.sealed && self.steps.iter().all(|s| s.complete)
    }
}

/// The once-per-process world-load steps (bake installs + initial terrain).
/// Registered by the first loading pass that spawns the world: a scenario
/// boot's [`steps_for`]`(…, boot: true)`, or the start screen's first start
/// after a deferred (world-[`Absent`](WorldState::Absent)) menu boot.
pub fn world_load_steps() -> [StepDesc; 2] {
    [
        StepDesc::new(step::BODIES, "Celestial bodies", 3.0),
        StepDesc::new(step::TERRAIN, "Surface terrain", 1.0),
    ]
}

/// Build the step set for loading into `situation`. `boot` includes the
/// world-load steps ([`world_load_steps`]), which only happen once per
/// process; a runtime re-load (start screen → runway with a live world)
/// passes `boot = false` and gets only the scenario steps.
pub fn steps_for(situation: SpawnSituation, boot: bool) -> Vec<StepDesc> {
    let mut steps = Vec::new();
    if boot {
        steps.extend(world_load_steps());
    }
    if situation.has_deferred_placement() {
        steps.push(StepDesc::new(step::PLACEMENT, "Placing craft", 1.0));
    }
    if matches!(situation, SpawnSituation::Runway | SpawnSituation::Launch) {
        steps.push(StepDesc::new(step::SETTLE, "Settling terrain", 2.0));
    }
    steps
}

/// Backstop on the whole loading pass, measured per `Loading` entry. The
/// per-gate timeouts in `surface_settle` cover the settle wait; this
/// covers everything else (a stalled placement, a bake task that never
/// completes) so the screen reveals with a warning rather than hanging.
const LOADING_HARD_TIMEOUT_S: f64 = 120.0;

#[derive(Component)]
struct LoadingScreenRoot;

#[derive(Component)]
struct LoadingProgressBarFill;

#[derive(Component)]
struct LoadingStatusText;

#[derive(Component)]
struct LoadingStepCounterText;

pub struct LoadingScreenPlugin;

impl Plugin for LoadingScreenPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<AppState>()
            .init_resource::<LoadingTracker>()
            .init_resource::<LoadDestination>()
            // Startup (not just OnEnter) so the screen exists on the very
            // first rendered frame; OnEnter covers runtime re-loads.
            .add_systems(Startup, (register_boot_steps, spawn_loading_screen))
            .add_systems(OnEnter(AppState::Loading), spawn_loading_screen)
            .add_systems(
                Update,
                (update_loading_progress_ui, finish_loading)
                    .chain()
                    .run_if(in_state(AppState::Loading)),
            )
            .add_systems(OnExit(AppState::Loading), despawn_loading_screen);
    }
}

/// The loading screen without its gate, for the headless preview
/// (`examples/loading_preview.rs`, `just loading-preview`).
///
/// This screen is the one UI in the game no capture preset can reach: it
/// despawns the instant loading completes, which is strictly before the capture
/// host takes its shot, and holding a real load still would mean screenshotting
/// a race. So the preview drives the *real* [`spawn_loading_screen`],
/// [`update_loading_progress_ui`] and the shared VRAM bar against
/// seeded state — genuine evidence about layout, alignment, and formatting,
/// and none about whether the steps themselves complete.
pub struct LoadingScreenPreviewPlugin;

impl Plugin for LoadingScreenPreviewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LoadingTracker>()
            .add_systems(Startup, spawn_loading_screen)
            .add_systems(Update, update_loading_progress_ui);
    }
}

/// Register the boot load's steps from the startup [`SpawnSituation`].
///
/// A bare menu boot ([`LoadDestination`] = [`AppState::MainMenu`]) defers the
/// world ([`WorldState::Absent`]) — no bodies bake, no terrain streams — so it
/// registers **no steps** and the screen reveals into the menu on the first
/// update. The world-load steps run later, when the menu starts a scenario
/// (`session_loading::apply_session_load_request` begins the boot step set).
fn register_boot_steps(
    situation: Res<SpawnSituation>,
    dest: Res<LoadDestination>,
    hub_build: Res<crate::space_center::HubSpaceportBuild>,
    mut tracker: ResMut<LoadingTracker>,
) {
    if dest.0 == AppState::MainMenu {
        tracker.begin(Vec::new());
    } else {
        let mut steps = steps_for(*situation, true);
        // A hub boot (`just game hub` / the headless hub preset) builds the
        // spaceport behind this same pass (`space_center::finish_hub_spaceport`
        // completes PLACEMENT), mirroring the start screen's PLAY.
        if hub_build.pending {
            steps.push(StepDesc::new(step::PLACEMENT, "Building spaceport", 1.0));
        }
        tracker.begin(steps);
    }
}

// Colours come straight from the shared token consts (`thalos_ui::tokens`) —
// no theme *resource* dependency, so the loading screen can render on frame 1
// (before `thalos_ui::init_ui_theme` has run); only the font asset loads
// asynchronously.
use thalos_ui::tokens::{ACCENT, SCREEN_BG, TEXT_DIM, TEXT_FAINT};

const TRACK_BG: Color = Color::srgba(1.0, 1.0, 1.0, 0.10);

const PROGRESS_BAR_WIDTH: f32 = 360.0;
const PROGRESS_BAR_HEIGHT: f32 = 4.0;

fn spawn_loading_screen(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    existing: Query<(), With<LoadingScreenRoot>>,
) {
    // Idempotent: Startup and OnEnter(Loading) both fire on boot.
    if !existing.is_empty() {
        return;
    }
    let display_font = FontSource::Handle(asset_server.load::<Font>("fonts/Inter-Light.ttf"));
    let font = FontSource::Handle(asset_server.load::<Font>("fonts/Inter-Regular.ttf"));
    // The diagnostics block is numbers in columns; a proportional face would
    // make the digits jitter as they change.
    let mono_font = FontSource::Handle(asset_server.load::<Font>("fonts/FiraCode-Regular.ttf"));

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
            // Sits above every other UI element. Pause menu uses 100, the
            // start screen 900, HUD panels the default (0); 1000 is well
            // clear of all of them so even a misbehaving overlay can't
            // poke through.
            GlobalZIndex(1000),
            Name::new("LoadingScreen"),
        ))
        .with_children(|root| {
            // Title — the logotype, in the display face.
            root.spawn((
                Text::new("THALOS"),
                TextFont {
                    font: display_font,
                    font_size: FontSize::Px(52.0),
                    ..default()
                },
                TextColor(ACCENT),
                TextLayout::justify(Justify::Center),
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
                    border_radius: BorderRadius::all(Val::Px(2.0)),
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Stretch,
                    overflow: Overflow::clip(),
                    ..default()
                },
                BackgroundColor(TRACK_BG),
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

            // Status text under the bar — the active step's label + detail.
            root.spawn((
                LoadingStatusText,
                Text::new("Preparing…"),
                TextFont {
                    font: font.clone(),
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                TextColor(TEXT_DIM),
                TextLayout::justify(Justify::Center),
                Name::new("LoadingStatusText"),
            ));

            // Faint `step i/N` counter under the status line.
            root.spawn((
                LoadingStepCounterText,
                Text::new(""),
                TextFont {
                    font,
                    font_size: FontSize::Px(11.0),
                    ..default()
                },
                TextColor(TEXT_FAINT),
                TextLayout::justify(Justify::Center),
                Name::new("LoadingStepCounter"),
            ));

            // VRAM, and only VRAM. A load is the one moment where the card's
            // state is both interesting and unobstructed, but it is also a
            // moment the player is *waiting through* — so this is a glance, not
            // a report. What is going wrong with memory is legible from a bar
            // (how full, and which band grew); the numbers that only answer
            // follow-up questions live in the F3 view, where someone has
            // actually asked.
            crate::vram_bar::spawn_vram_bar(
                root,
                &crate::vram_bar::VramBarStyle {
                    width: Val::Px(PROGRESS_BAR_WIDTH),
                    bar_height: 6.0,
                    font: mono_font,
                    font_size: 10.0,
                    label_color: TEXT_FAINT,
                    value_color: TEXT_DIM,
                    caption: "VRAM",
                },
            );
        });
}

fn update_loading_progress_ui(
    tracker: Res<LoadingTracker>,
    mut fill_q: Query<&mut Node, With<LoadingProgressBarFill>>,
    mut text_q: Query<&mut Text, (With<LoadingStatusText>, Without<LoadingStepCounterText>)>,
    mut counter_q: Query<&mut Text, (With<LoadingStepCounterText>, Without<LoadingStatusText>)>,
) {
    if let Ok(mut node) = fill_q.single_mut() {
        node.width = Val::Percent(tracker.overall().clamp(0.0, 1.0) * 100.0);
    }
    if let Ok(mut text) = text_q.single_mut() {
        let new_text = match tracker.active_step() {
            Some(step) => {
                let mut line = step.label.to_string();
                if let Some(counter) = step.counter() {
                    line.push_str(&format!("  {counter}"));
                }
                if !step.detail.is_empty() {
                    line.push_str(&format!("  ·  {}", step.detail));
                }
                line
            }
            None => "Preparing…".to_string(),
        };
        if text.0 != new_text {
            **text = new_text;
        }
    }
    if let Ok(mut text) = counter_q.single_mut() {
        let (index, total) = tracker.step_position();
        let new_text = if total > 1 {
            format!("step {index} / {total}")
        } else {
            String::new()
        };
        if text.0 != new_text {
            **text = new_text;
        }
    }
}

/// Transition out of `Loading` once every registered step completes (or
/// the hard timeout fires), into the configured [`LoadDestination`].
fn finish_loading(
    time: Res<Time<Real>>,
    mut elapsed_s: Local<f64>,
    tracker: Res<LoadingTracker>,
    dest: Res<LoadDestination>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    *elapsed_s += time.delta_secs_f64();
    let timed_out = *elapsed_s >= LOADING_HARD_TIMEOUT_S;
    if !tracker.is_complete() && !timed_out {
        return;
    }
    if timed_out && !tracker.is_complete() {
        let stuck = tracker
            .active_step()
            .map(|s| s.label)
            .unwrap_or("(unregistered)");
        warn!(
            "loading hard-timeout after {:.0} s — revealing with step '{}' incomplete",
            *elapsed_s, stuck
        );
    }
    // Reset for a potential later re-entry into `Loading` (the Local
    // persists across state transitions).
    *elapsed_s = 0.0;
    next_state.set(dest.0);
}

fn despawn_loading_screen(mut commands: Commands, roots: Query<Entity, With<LoadingScreenRoot>>) {
    for entity in &roots {
        commands.entity(entity).despawn();
    }
}
