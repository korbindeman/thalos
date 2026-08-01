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

// The state vocabulary and navigation helpers moved to the game-state
// blackboard (Phase 5a); this module keeps the plugin and the systems.
pub use thalos_game_state::context::{
    ContextHistory, GameContext, InitialContext, back_out, context_freezes_sim, enter_context,
    flight_or_no_context, not_vab,
};

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
