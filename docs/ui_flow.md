# UI flow — screens, modes, and transitions

Status: **migration complete** (Phases 1–4 landed; Phase 3 — ownership inverted —
landed 2026-07-05, compile+clippy-clean, game-UNVERIFIED). `GameContext` is now
the sole in-`Running` mode authority; the `.open` booleans survive only as derived
mirrors. This doc is the design of record for the game's screen/mode flow.

## The problem this fixes

The game's outer shell is a clean state machine —
[`AppState`](../crates/runtime/game/src/loading.rs) = `Loading | MainMenu | Running`.
But **everything inside `Running`** — the space-center hub, the VAB (shipyard
editor), the base editor, and the ship/map views — was modelled as a loose bag
of boolean `.open` resources (`SpaceCenter::open`, `ShipyardEditor::open`,
`BaseEditor::open`) plus a `ViewMode` enum, each of which has to *know about all
the others*. Consistency was maintained by:

- a hand-ordered Escape ladder of `if open { close; return }` across five
  resources (`pause_menu::handle_escape_input`);
- edge-latch `Local<bool>`s reconstructing transitions (`restore_after_facility`,
  each modal's `apply_open_state`);
- run-condition gating scattered across `main.rs` (`editor_closed`,
  `base_editor_closed`, `space_center_closed`);
- and **three separate systems writing `Camera::is_active`**
  (`view::apply_active_camera`, `shipyard_editor::apply_open_state`, the
  `god_view` path).

There is no single answer to "what mode am I in", so every mode is coupled to
every other mode, and adding one (a tracking station, a second base) means
editing the Escape chain, the pause sources, the camera gates, and the
return-latches all at once.

## The model

Two levels, each a **real** state machine.

```
AppState (shell)     Loading │ MainMenu │ Running
                                            └── GameContext  (Bevy SubStates, exists only in Running)
GameContext          SpaceCenter │ Vab │ BaseEditor │ Flight    [ + TrackingStation, later ]
                                                       └── ViewMode  Ship │ Map   (Flight-only)
```

`GameContext` is a Bevy [`SubStates`] sourced on `AppState::Running`, so it only
exists while in-game and gives us `OnEnter/OnExit(GameContext::X)` scheduling for
the per-context setup/teardown. `ViewMode` (Ship vs. Map camera) becomes a
purely **Flight-internal** concern: `M` only toggles inside `Flight`, and the
god-view contexts own their camera directly instead of forcing `ViewMode::Ship`.
`TrackingStation` (map with no active craft) is a deferred sibling context — the
map camera active with no craft — not a special case of Flight.

### Per-context authority

One authority (`OnEnter/OnExit(GameContext)` handlers + a camera function of
`(GameContext, ViewMode)`) sets all of this, replacing the scattered writers:

| Context | Camera | Sim stages | HUD | Escape pops to |
|---|---|---|---|---|
| `Flight` | Ship or Map (`ViewMode`) | all run | shown | pause menu (then sub-modals) |
| `SpaceCenter` | Ship cam, `god_view` drives it | Physics paused, **Sync streams** | hidden | previous (Flight, or → MainMenu if root) |
| `BaseEditor` | Ship cam, `god_view` drives it | Physics paused, **Sync streams** | hidden | previous (SpaceCenter or Flight) |
| `Vab` | Editor cam (layer 5) | **all frozen** (isolated scene) | hidden | previous (SpaceCenter or Flight) |

**`Vab` deliberately stays a full-freeze isolated scene** (all `SimStage` sets
off, dedicated `EDITOR_LAYER` camera) — the flight world is suspended, not just
hidden. The hub and base editor are *in-world overlays*: the planet stays
visible and frozen (Physics off) but `Sync` keeps streaming terrain, and the
shared god-view camera repositions the `ShipCamera` in place.

### Return-stack

A single `ContextHistory` (a small stack of `GameContext`) replaces the
`return_to_menu` and `ReturnToSpaceCenter` bools and the `Local<bool>` latches.
Entering a context pushes where you came from; Escape / EXIT pops. Popping the
**root** (Flight, or SpaceCenter when it is the session root after PLAY)
transitions `AppState → MainMenu`. A VAB **Launch** clears the stack and drops to
Flight (it queues a relaunch), so "launched to fly" is distinguished from
"escaped out" without a special flag.

### Escape

**Escape never exits to the start screen** — once in the game the pause menu's
**MAIN MENU** button is the *sole* deliberate exit (the user's rule). Escape's
behaviour is a function of the current `GameContext`, with the genuinely-nested
sub-modals checked first, in order: forced modal (destruction picker) → Settings
overlay → text-field focus → target deselect → interaction mode → pause menu →
context back-out. In a non-Flight context Escape backs out one level toward its
parent (VAB/base editor → the hub or flight they were opened from); at the **root**
context (nothing below it — the PLAY-rooted hub, or Flight) Escape opens the
**pause menu** instead of leaving the game. This collapses the 9-rung ladder in
`pause_menu::handle_escape_input`.

Escape's priority order (Phase 3): forced modal (destruction picker) → Settings
overlay → pause menu (it can sit over a root context) → context back-out (pop the
return stack; empty stack at the root → open the pause menu) → Flight sub-modals
(interaction mode, target deselect) → open the pause menu. A focused editor text
field eats Escape upstream by disabling the keyboard action source.

### Boot routing

An `InitialContext` resource read on `OnEnter(Running)` sets the starting
context, retiring `OpenShipyardOnStart` / `OpenSpaceCenterOnStart`:

- scenario launch (`just game orbit`, runway, …) → `Flight`
- start-screen **PLAY** → `SpaceCenter`
- `just game shipyard` / start-screen **SHIPYARD** → `Vab`

### Entry points (transitions)

| From | Action | To |
|---|---|---|
| MainMenu | PLAY | Running / `SpaceCenter` |
| MainMenu | scenario (dev submenu) | Running / `Flight` |
| MainMenu | SHIPYARD | Running / `Vab` |
| Flight | pause menu → SPACE CENTER | `SpaceCenter` |
| Flight | pause menu → SHIPYARD | `Vab` |
| Flight | pause menu → SURFACE BASE | `BaseEditor` |
| Flight | pause menu → **MAIN MENU** *(new)* | MainMenu |
| Flight | `M` | Flight (`ViewMode::Map`) |
| SpaceCenter | click VAB / EDIT BASE | `Vab` / `BaseEditor` |
| SpaceCenter | Escape | back to Flight (if opened from flight), else pause menu (root) — **never** MainMenu |
| SpaceCenter | EXIT button | back to Flight, or MainMenu if root |
| any context | pause menu → MAIN MENU | MainMenu (the sole deliberate exit) |
| Vab / BaseEditor | Escape | pop → SpaceCenter or Flight |
| Vab | Launch | Flight (relaunch; stack cleared) |

## "Base" is data, not a context

`GameContext::SpaceCenter` references *a* base by `StructureId`. Multiple bases +
a default is a **data** extension (e.g. a `default_base` marker on
`StructureSite`), not a new context or new plumbing. Kept as a seam for now.

## Migration phases

Compile-clean and game-runnable at each step, mirroring the `docs/regimes.md`
A2→A3 strangler pattern (introduce a shadow authority, flip consumers onto it,
then invert ownership).

1. **Shadow** *(landed 2026-07-04)* — `GameContext` sub-state +
   `game_context::resolve_game_context` deriving it from today's `.open`
   booleans + a transition log (`thalos::ui`). **Nothing consumes it yet.**
   Verify it tracks every mode as you navigate.
2. **Flip consumers** *(landed 2026-07-04)* — the cross-cutting consumers now
   read `GameContext` instead of the `.open` booleans: `sim_clock` pause
   (`context_freezes_sim`), the `SimStage::Physics/Sync/Camera` gates in
   `main.rs` (`not_vab` / `flight_or_no_context`, replacing `editor_closed` /
   `base_editor_closed` / `space_center_closed`), the **single camera authority**
   `view::apply_active_camera` (now a pure function of `(GameContext, ViewMode)`
   owning ship/map **and** editor cameras — the three `is_active` writers
   collapsed to one; `shipyard_editor::apply_open_state` no longer touches
   cameras), and the HUD-update gate. The resolver still shadow-derives the
   sub-state from the booleans (Phase 3 inverts that), so there is a **one-frame
   lag** on mode entry — a brief flight-world flash + one sim-tick when opening
   the VAB, gone in Phase 3. HUD-visibility *ownership* stays per-modal for now
   (moved to `OnEnter/OnExit` in Phase 3, bundled with deleting the booleans;
   deferred here to avoid editing files under unrelated WIP).
3. **Invert ownership** *(landed 2026-07-05)* — button clicks + Escape + the
   facility flows set `NextState<GameContext>` directly and navigate a
   `ContextHistory` return stack (`game_context::{enter_context, back_out}`);
   `return_to_menu`, `ReturnToSpaceCenter`, `restore_after_facility`, and the two
   `Open*OnStart` flags are deleted; `InitialContext` (consumed on
   `OnEnter(Running)`) is the boot route. The `.open` bools are kept as **derived
   mirrors** (`game_context::mirror_context_to_booleans`, the sole writer) so all
   `.open` *readers* — input gating, each modal's `apply_open_state`, the run
   conditions — are untouched. `OnExit(Running)` resets the mirrors + stack. The
   Escape ladder collapsed into a context-aware back-out (root context → pause
   menu, never the start screen). A VAB **Launch** / launchpad L-launch clears the
   stack and drops to Flight; the launch-select picker re-enters `BaseEditor`
   parented to Flight, so place/cancel both land in flight.
4. **Main Menu exit** *(landed 2026-07-04, partial)* — the pause menu gained a
   **MAIN MENU** button (`pause_menu.rs`) → the direct flight→`AppState::MainMenu`
   route (previously only reachable via the hub). Collapsing the Escape ladder
   into a per-context handler is deferred with Phase 3 (it needs `GameContext` to
   be settable).

Follow-ups the state machine makes cheap: transition fades ("good transitions"),
multi-base + default, and the **Tracking Station** context (ship-less map).

## Invariants

- **One mode authority.** `GameContext` is the single source of truth for which
  in-`Running` mode is active. Do not reintroduce cross-referencing `.open`
  booleans as the authority.
- **Camera is a function of `(GameContext, ViewMode)`**, written in one place.
- **`ViewMode` is Flight-only.** The god-view contexts own their camera directly.
- **Return routing is a stack**, not per-pair bools.

[`SubStates`]: https://docs.rs/bevy/latest/bevy/state/state/trait.SubStates.html
