//! `GameContext` — the single in-`Running` mode authority.
//!
//! The outer shell ([`AppState`]) stays `Loading | MainMenu | Running`; this is
//! the sub-state that says *which mode you are in* while running: flying, in the
//! space-center hub, the VAB, or the base editor. It replaces the loose bag of
//! cross-referencing `.open` booleans as the source of truth. See
//! `docs/gameplay/ui_flow.md` for the full model, the per-context
//! camera/HUD/pause authority, and the migration phases.
//!
//! The runtime's `game_context` module owns the plugin and the systems (boot
//! routing, the derived `.open` mirrors, transition logging); this module owns
//! the state vocabulary and the navigation helpers.

use bevy::prelude::*;

use crate::app::AppState;

/// Which in-game mode is active. A Bevy [`SubStates`] existing only while
/// [`AppState::Running`]; the ship/map `ViewMode` nests inside
/// [`Flight`](GameContext::Flight).
///
/// **Sole authority (ui_flow Phase 3):** the button / Escape / facility
/// handlers set this via `NextState<GameContext>`; the legacy `.open` booleans
/// are derived mirrors (sole writer: the runtime's
/// `mirror_context_to_booleans`).
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
/// consumed (taken) once on reveal. Set by the launcher (`just game shipyard` /
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

// --- Run conditions (consumers read `GameContext`, not the booleans) ---
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
/// own `AppState` check in the runtime's `sim_clock`).
pub fn context_freezes_sim(ctx: Option<&State<GameContext>>) -> bool {
    ctx.map(|c| !matches!(*c.get(), GameContext::Flight))
        .unwrap_or(false)
}
