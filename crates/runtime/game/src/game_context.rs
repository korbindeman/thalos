//! `GameContext` — the single in-`Running` mode authority (in migration).
//!
//! The outer shell ([`AppState`]) stays `Loading | MainMenu | Running`; this is
//! the sub-state that says *which mode you are in* while running: flying, in the
//! space-center hub, the VAB, or the base editor. It replaces the loose bag of
//! cross-referencing `.open` booleans (`SpaceCenter`/`ShipyardEditor`/
//! `BaseEditor`) as the source of truth. See `docs/gameplay/ui_flow.md` for the full
//! model, the per-context camera/HUD/pause authority, and the migration phases.
//!
//! **Migration status: Phase 3 (ownership inverted).** `GameContext` is now the
//! **authority**: the button / Escape / facility handlers set
//! `NextState<GameContext>` directly and navigate a [`ContextHistory`] return
//! stack, and boot routing goes through [`InitialContext`]. The legacy `.open`
//! booleans survive only as **derived mirrors** — [`mirror_context_to_booleans`]
//! is their sole writer — so every `.open` *reader* (input gating, each modal's
//! `apply_open_state`, run conditions) keeps working untouched. Because a state
//! change set in `Update` applies at the next `StateTransition`, entering a mode
//! still carries a **one-frame lag**. Mirrors `docs/simulation/regimes.md` A2→A3.

use bevy::prelude::*;

use crate::base_editor::BaseEditor;
use crate::loading::AppState;
use crate::shipyard_editor::ShipyardEditor;
use crate::space_center::SpaceCenter;

/// Which in-game mode is active. A Bevy [`SubStates`] existing only while
/// [`AppState::Running`]; the ship/map [`ViewMode`](crate::view::ViewMode) nests
/// inside [`Flight`](GameContext::Flight).
///
/// **Sole authority (Phase 3):** the button / Escape / facility handlers set
/// this via `NextState<GameContext>`; the `.open` booleans are derived from it
/// by [`mirror_context_to_booleans`].
#[derive(SubStates, Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[source(AppState = AppState::Running)]
pub enum GameContext {
    /// Flying an active craft — ship or map view (`ViewMode`).
    #[default]
    Flight,
    /// The space-center hub: a god-view over a base.
    SpaceCenter,
    /// The VAB / shipyard editor — an isolated full-freeze scene.
    Vab,
    /// The in-world surface base editor — a god-view overlay.
    BaseEditor,
    // TrackingStation — deferred: the ship-less map. See `docs/gameplay/ui_flow.md`.
}

/// The return stack for in-`Running` mode navigation. Entering a context pushes
/// where you came from; Escape / EXIT / back-out pops it. Popping an **empty**
/// stack means "at the session root" — the caller decides what that means (open
/// the pause menu, or leave to the main menu). Replaces the old `return_to_menu`
/// / `ReturnToSpaceCenter` bools + the `Local<bool>` edge latches. See
/// `docs/gameplay/ui_flow.md` *Return-stack*.
#[derive(Resource, Default)]
pub struct ContextHistory(pub Vec<GameContext>);

/// Boot routing: the context to enter on the next `OnEnter(AppState::Running)`.
/// `None` (the default) leaves the [`GameContext::Flight`] default; `Some(x)` is
/// consumed (taken) once on reveal. Set by `main.rs` (`just game shipyard` /
/// `hub`) and the start screen's PLAY. Replaces `OpenShipyardOnStart` /
/// `OpenSpaceCenterOnStart`.
#[derive(Resource, Default)]
pub struct InitialContext(pub Option<GameContext>);

/// Enter `target` from `current`, remembering `current` on the return stack.
pub fn enter_context(
    next: &mut NextState<GameContext>,
    history: &mut ContextHistory,
    current: GameContext,
    target: GameContext,
) {
    history.0.push(current);
    next.set(target);
}

/// Back out one level toward the parent context. Returns the context we popped
/// to, or `None` if the stack was empty (we are at the session root — the caller
/// decides what leaving the root does: pause menu, or main menu).
pub fn back_out(
    next: &mut NextState<GameContext>,
    history: &mut ContextHistory,
) -> Option<GameContext> {
    let parent = history.0.pop()?;
    next.set(parent);
    Some(parent)
}

// --- Run conditions (Phase-2 consumers read `GameContext`, not the booleans) ---
//
// All treat an **absent** sub-state (outside `Running` — Loading / MainMenu) the
// same way the legacy `*_closed` conditions did: as "no modal open", so
// world-sync and the camera keep running behind the loading screen.

/// `true` unless in the isolated-scene VAB (absent → `true`). The VAB is the
/// only context that freezes *all* `SimStage`s; the hub / base editor keep
/// `Sync` streaming, so they are **not** excluded here.
pub fn not_vab(ctx: Option<Res<State<GameContext>>>) -> bool {
    ctx.map(|c| !matches!(*c.get(), GameContext::Vab))
        .unwrap_or(true)
}

/// `true` only in [`GameContext::Flight`], or when the sub-state is absent
/// (Loading / MainMenu) so flight systems that must still run outside `Running`
/// — the camera during a deferred-placement settle — keep running. The hub /
/// VAB / base-editor contexts return `false` (their god-view / editor camera
/// owns the view instead).
pub fn flight_or_no_context(ctx: Option<Res<State<GameContext>>>) -> bool {
    ctx.map(|c| matches!(*c.get(), GameContext::Flight))
        .unwrap_or(true)
}

/// `true` while a modal context (hub / VAB / base editor) freezes the sim.
/// Absent sub-state → `false` (Loading is not paused; MainMenu pauses via its
/// own `AppState` check in `sim_clock`).
pub fn context_freezes_sim(ctx: Option<&State<GameContext>>) -> bool {
    ctx.map(|c| !matches!(*c.get(), GameContext::Flight))
        .unwrap_or(false)
}

pub struct GameContextPlugin;

impl Plugin for GameContextPlugin {
    fn build(&self, app: &mut App) {
        app.add_sub_state::<GameContext>()
            .init_resource::<ContextHistory>()
            .init_resource::<InitialContext>()
            .add_systems(OnEnter(AppState::Running), apply_initial_context)
            .add_systems(OnExit(AppState::Running), leave_running)
            .add_systems(
                Update,
                (mirror_context_to_booleans, log_context_transitions)
                    .chain()
                    .run_if(in_state(AppState::Running)),
            );
    }
}

/// Boot routing: on entry to `Running`, switch to the armed [`InitialContext`]
/// (PLAY → hub, `just game shipyard` → VAB), else leave the `GameContext::Flight`
/// default. One-shot (`take`), so later loading passes (runway starts, the
/// launch-flow spaceport build) reveal into Flight and let their own systems
/// drive the context. Seeds an empty return stack — the initial context is the
/// session root.
fn apply_initial_context(
    mut initial: ResMut<InitialContext>,
    mut history: ResMut<ContextHistory>,
    next: Option<ResMut<NextState<GameContext>>>,
) {
    let Some(target) = initial.0.take() else {
        return;
    };
    history.0.clear();
    if let Some(mut next) = next {
        next.set(target);
    }
}

/// Leaving `Running` (→ MainMenu): reset the derived mirrors and the return
/// stack so a later re-entry starts clean. `GameContext` itself is removed by
/// Bevy when the source state exits, so the mirror stops running; without this
/// the last-open booleans would linger into the menu.
fn leave_running(
    mut shipyard: ResMut<ShipyardEditor>,
    mut base: ResMut<BaseEditor>,
    mut hub: ResMut<SpaceCenter>,
    mut history: ResMut<ContextHistory>,
) {
    shipyard.open = false;
    base.open = false;
    hub.open = false;
    history.0.clear();
}

/// **Phase 3 authority mirror.** `GameContext` is the authority; this derives the
/// legacy `.open` booleans from it (its sole writer), so every `.open` *reader*
/// — input gating, each modal's `apply_open_state`, the run conditions — is
/// untouched by the ownership inversion. Change-guarded so the modals' edge
/// detection sees one clean transition. Only `open` is mirrored: the
/// modal-internal fields (`mode`, `active_site`, `hovered`) stay caller-owned.
fn mirror_context_to_booleans(
    ctx: Option<Res<State<GameContext>>>,
    mut shipyard: ResMut<ShipyardEditor>,
    mut base: ResMut<BaseEditor>,
    mut hub: ResMut<SpaceCenter>,
) {
    let Some(ctx) = ctx else {
        return; // sub-state not active this frame
    };
    let ctx = *ctx.get();
    let want_vab = matches!(ctx, GameContext::Vab);
    let want_base = matches!(ctx, GameContext::BaseEditor);
    let want_hub = matches!(ctx, GameContext::SpaceCenter);
    if shipyard.open != want_vab {
        shipyard.open = want_vab;
    }
    if base.open != want_base {
        base.open = want_base;
    }
    if hub.open != want_hub {
        hub.open = want_hub;
    }
}

/// Log every [`GameContext`] change (target `thalos::ui`) so the shadow resolver
/// can be verified by navigating the modes and watching the console.
fn log_context_transitions(
    current: Option<Res<State<GameContext>>>,
    mut last: Local<Option<GameContext>>,
) {
    let Some(current) = current else { return };
    let now = *current.get();
    if *last != Some(now) {
        info!(target: "thalos::ui", "GameContext -> {now:?}");
        *last = Some(now);
    }
}
