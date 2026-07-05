//! `GameContext` — the single in-`Running` mode authority (in migration).
//!
//! The outer shell ([`AppState`]) stays `Loading | MainMenu | Running`; this is
//! the sub-state that says *which mode you are in* while running: flying, in the
//! space-center hub, the VAB, or the base editor. It replaces the loose bag of
//! cross-referencing `.open` booleans (`SpaceCenter`/`ShipyardEditor`/
//! `BaseEditor`) as the source of truth. See `docs/ui_flow.md` for the full
//! model, the per-context camera/HUD/pause authority, and the migration phases.
//!
//! **Migration status: Phase 2 (consumers flipped).** [`resolve_game_context`]
//! still *derives* the sub-state from the legacy `.open` booleans (they remain
//! the writers until Phase 3), but the **cross-cutting consumers now read
//! `GameContext`**: the sim-clock pause ([`context_freezes_sim`]), the
//! `SimStage` gates ([`not_vab`] / [`flight_or_no_context`] in `main.rs`), the
//! single camera authority (`view::apply_active_camera`), and the HUD-update
//! gate. Because the resolver writes `NextState` in `Update` (applied next
//! `StateTransition`), there is a **one-frame lag** entering a mode — a brief
//! flight-world flash / sim-tick on VAB open, gone in Phase 3 when the buttons
//! set `NextState<GameContext>` directly. Mirrors `docs/regimes.md` A2→A3.

use bevy::prelude::*;

use crate::base_editor::BaseEditor;
use crate::loading::AppState;
use crate::shipyard_editor::ShipyardEditor;
use crate::space_center::SpaceCenter;

/// Which in-game mode is active. A Bevy [`SubStates`] existing only while
/// [`AppState::Running`]; the ship/map [`ViewMode`](crate::view::ViewMode) nests
/// inside [`Flight`](GameContext::Flight).
///
/// **Sole writer (Phase 1):** [`resolve_game_context`] (shadow-derived). Phase 3
/// inverts this so button/Escape handlers set it directly.
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
    // TrackingStation — deferred: the ship-less map. See `docs/ui_flow.md`.
}

/// Pure classifier: the context implied by the legacy modal booleans. The
/// priority order matters only if two are somehow open at once (they shouldn't
/// be): the isolated-scene VAB wins, then the base editor, then the hub, else
/// flight.
pub fn expected_context(shipyard_open: bool, base_open: bool, hub_open: bool) -> GameContext {
    if shipyard_open {
        GameContext::Vab
    } else if base_open {
        GameContext::BaseEditor
    } else if hub_open {
        GameContext::SpaceCenter
    } else {
        GameContext::Flight
    }
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

/// `true` in the two god-view overlay contexts — the space-center hub and the
/// in-world base editor — where the camera booms around a surface focus decoupled
/// from any craft. Surface scatter (grass / trees / rocks) reads this to follow
/// the view instead of the (possibly parked, possibly orbiting-placeholder)
/// craft. Absent sub-state (Loading / MainMenu) → `false`.
pub fn god_view_active(ctx: Option<&State<GameContext>>) -> bool {
    ctx.map(|c| matches!(*c.get(), GameContext::SpaceCenter | GameContext::BaseEditor))
        .unwrap_or(false)
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
        app.add_sub_state::<GameContext>().add_systems(
            Update,
            (resolve_game_context, log_context_transitions)
                .chain()
                .run_if(in_state(AppState::Running)),
        );
    }
}

/// Phase-1 shadow resolver: keep [`GameContext`] tracking the legacy `.open`
/// booleans. Reads the three modal resources, sets the sub-state when it
/// differs. `Option` params guard the one-frame window around a source-state
/// transition where the sub-state resources may not yet be present.
fn resolve_game_context(
    shipyard: Res<ShipyardEditor>,
    base: Res<BaseEditor>,
    hub: Res<SpaceCenter>,
    current: Option<Res<State<GameContext>>>,
    next: Option<ResMut<NextState<GameContext>>>,
) {
    let (Some(current), Some(mut next)) = (current, next) else {
        return; // sub-state not active this frame
    };
    let expected = expected_context(shipyard.open, base.open, hub.open);
    if *current.get() != expected {
        next.set(expected);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_priority() {
        assert_eq!(expected_context(false, false, false), GameContext::Flight);
        assert_eq!(
            expected_context(false, false, true),
            GameContext::SpaceCenter
        );
        assert_eq!(expected_context(false, true, false), GameContext::BaseEditor);
        assert_eq!(expected_context(true, false, false), GameContext::Vab);
        // The isolated-scene VAB wins any overlap.
        assert_eq!(expected_context(true, true, true), GameContext::Vab);
        assert_eq!(expected_context(false, true, true), GameContext::BaseEditor);
    }
}
