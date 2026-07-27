# Architecture & code quality cleanup

**Status:** plan drafted 2026-07-05 from a three-track codebase audit (craft
lifecycle, mode/state machinery, duplication/dead-code inventory). This doc is
rationale and sequencing only — what has actually landed is in
[`docs/backlog.md`](../backlog.md).

## 1. Why

The feature push (surface bases, space-center hub, launch flows, GPU grass,
shadow unification) delivered features faster than it delivered *systems*. The
result is a steady stream of bugs whose root cause is structural, not local:

- **Hand-assembled sequences.** Placing/teleporting a craft is a five-step
  ritual (reset sim → clear Avian bubble → set canonical state → set attitude →
  set authority) that each of ~7 entry points re-assembles by hand. Every
  recent placement bug (phantom bubble velocity, the big_space root-seating
  race, stale `ReturnToSpaceCenter`) is a missed or misordered step in one
  copy of that ritual.
- **Parallel mechanisms for the same job.** In-game modes are governed by five
  `.open` booleans *and* the shadow `GameContext` sub-state; three modals each
  carry a near-identical `apply_open_state` + HUD hide/restore + edge-latch
  stack.
- **Single-instance assumptions.** ~15 resources hard-code "the one craft"
  (`ActiveLocalBubble`'s single slot, `ManeuverPlan`, `RealizedControl`,
  `GearState`, `EvaMode`, `CameraFocus`, `PlayerShip` `.single()` queries…),
  and the runway grew from "the one strip" into a god module.
- **Dead mass.** ~11 kLOC of the superseded bake pipeline still sits in
  `crates/domain/terrain` + `body_render`, confusing every search and every new
  agent.

The sprint goal: one canonical path per operation, N-by-default, and the
in-flight unifications (`GameContext`, `CraftRegime`) finished rather than
worked around.

## 2. What the audit found (baseline, 2026-07-05)

Good news first — the hygiene floor is high: no TODO/FIXME/HACK markers,
near-zero `unwrap()` in the game crate, single-writer resources hold, run
conditions are consistent pure functions, and `SimClock` already folds all six
pause sources in one place. The debt is *architectural seams*, not sloppy
lines.

### 2.1 Craft lifecycle

Seven spawn/placement paths (boot orbit/EVA, deferred descent ×3, deferred
runway ×2, destruction respawn, editor relaunch, launch-select pad/runway
click, EVA drop/teleport). Shared cores exist (`orbit_respawn_state`,
`compute_descent_state`, `build_player_ship`, `place_on_runway`,
`clear_bubble`) but the **orchestration** around them is copy-pasted:

- The clear-bubble + place + rebuild sequence appears independently at
  `runway.rs` (dev park + launch-select), `base_editor/place.rs` (launchpad),
  and `relaunch.rs` — forgetting `clear_bubble()` on any future path leaves
  phantom velocity (this class of bug has already shipped twice).
- Order-sensitive invariants live only in comments / folklore: bubble teardown
  before placement, big_space root `CellCoord` seated before children attach
  (`ship_view.rs` build closure), canonical state set before authority,
  spaceport flatten installed before craft placement.
- Missing helpers the audit named: a canonical
  `place_craft(state, attitude, authority, body)` core, a unified
  clear-and-rebuild, a shared cursor→pad raycast (duplicated between
  `launch_select.rs` and `base_editor/place.rs`), a scenario→authority picker,
  and one craft-clearance measurer (`measure_runway_clearance` vs the
  launchpad's extent probe).

### 2.2 Single-craft assumptions (the N-craft blocker list)

| Assumption | Where | N-craft shape |
|---|---|---|
| One canonical craft state | `Simulation` (`ship_state`/`set_ship_state`) | **resolved + landed 2026-07-25:** ordered `CraftId` → `VesselRecord` registry; existing methods are active-craft wrappers and new mutation seams take an id |
| One bubble slot | `ActiveLocalBubble.bubble: Option<…>` | per-craft component or map |
| One authority answer/frame | `AvianAuthority` resource | already derived from per-craft `CraftRegimeState` — finish the projection |
| One control pipeline | `ControlDemand`/`RealizedControl` resources | per-craft components |
| One flight plan | `ManeuverPlan` | per-craft |
| One gear/brake/EVA-mode/flap state | `GearState`, `ParkingBrake`, `EvaMode`, `FlightConfig` | per-craft components |
| One focus/billboard/HUD subject | `CameraFocus`, `ShipMarker`, HUD panels | active-craft indirection |
| `.single()` on `PlayerShip` | queries throughout | `ActiveCraft` resource naming an entity |

`CraftRegimeState` (per-bubble-entity, from the regimes A3 port) is the proof
that per-craft components work here — it is the **template** for migrating the
rest. Canonical N-craft is no longer deferred. Runtime migration remains
requirements-driven by physical staging: (a) no new single-craft resources —
new per-craft state goes on the craft root; (b) active-only reads route through
`ActiveCraft`; (c) world simulation/render/map paths iterate canonical ids; and
(d) every knowingly-kept singleton stays in the ledger below.

### 2.3 Mode/state machinery

`GameContext` (docs/gameplay/ui_flow.md) Phase 1 (shadow) and Phase 2 (flip consumers:
sim-clock, SimStage gates, the single camera authority
`view::apply_active_camera`, HUD) have landed. **Phase 3 (invert ownership) is
no longer blocked** — the base-editor WIP that stalled it now compiles
(`cargo check -p thalos_game` clean, verified 2026-07-05; the memory/spec
claim of a `build_parallel_taxiway` blocker is stale). Debt Phase 3 removes:

- Five `.open` booleans with 3–4 writers each vs the one derived enum
  (one-frame lag on every mode entry; VAB open flashes the flight world).
- `ReturnToSpaceCenter` + `Local<bool>` edge latches
  (`restore_after_facility`, `facility_was_open`) — already caused the
  stuck-in-hub-after-launch race → replaced by a `ContextHistory` stack.
- Three near-identical `apply_open_state` implementations + three independent
  HUD hide/restore systems (defended today by a per-frame `enforce_hud_hidden`
  loop) → one per-context `OnEnter`/`OnExit` pair.
- The 9-rung Escape ladder in `pause_menu::handle_escape_input` → per-context
  handlers.

### 2.4 Duplication / dead code / god modules

- **Dead bake pipeline, ~11 kLOC**: `crates/domain/terrain/src/{feature_compiler,
  cache, cold_desert_field, aging_oceanic_field, generic_terrestrial_field,
  surface_color}.rs`, `body_render/src/impostor/bake.rs`, plus the vestigial
  bridge in `rendering/generation.rs`. Audit found no live callers
  (`PlanetSurface` reachable only from tests/dead-end imports) — **re-verify
  with `cargo check --workspace` after each deletion**, and remember the
  planet-generation-test moratorium (deleting their tests is fine).
- **`runway.rs` is a god module (~2,040 LOC, 6 concerns)**: runway geometry +
  designator rasterization, collider sync, f64 anchoring, deferred placement,
  spaceport construction, flatten integration. It overlaps `structures.rs`
  (which was *built* to be the generic structure layer) and
  `base_editor/place.rs`.
- **Connections aren't structures**: taxiways/aprons are `ConnectionVisual`
  meshes outside `StructureRegistry`, so scatter clearing and any future
  per-structure behaviour skip them (already the co-cause of the grass-under-
  taxiway artifact).
- **f64 body-fixed anchoring** re-implemented at ~16 sites
  (`grid.translation_to_grid` + cell/transform write) with no shared utility.
- **Deferred placement flags** (`DescentPlacement`, `RunwayPlacement`,
  `LaunchSpaceportBuild`) are three copies of one armed-gate pattern.
- Minor: ~~duplicated menu-button builders~~ (→ package G, landed), ~~dead
  `format_thrust` / `format_duration_s`~~ (deleted 2026-07-05), terrain-height
  three-mirror design is *intentional* but undocumented at the code site.

## 3. The plan

Ordered work packages. Each lands compile-clean + clippy-clean, updates the
relevant `docs/` spec, and ends with a user runtime-verification checklist
(agents can't run the game). Do them roughly in order — A and B are
prerequisites that shrink the surface everything else touches.

### A. Delete the dead bake pipeline — RESCOPED 2026-07-05 ☐ (deferred)

**The audit's premise was wrong.** An execution pass proved the "~11 kLOC of
file-deletable dead code" does not exist as separable files:

- The **live `SurfaceQuery` seam** (`query.rs`, `sample.rs`, `BakedSurface`,
  `surface_height_m`) compiles *against* the supposedly-dead
  `feature_compositor`, `generic_terrestrial_field`, `static_surface`
  (`PlanetSurface`/`StaticSurfaceData`), `crater_profile`, `spatial_index`,
  `tectonics` — they are transitively live.
- `feature_compiler.rs` / `cold_desert_field.rs` hold **live authored-RON
  schema** types (`BodyArchetype`, `FeatureTerrainConfig`, `CompositionClass`,
  `AtmosphereSpec`, `IceInventory`, `TerrainIntent`, …) deserialized from
  `assets/bodies/*.ron` and read live in `ground_terrain.rs`.
- `impostor/bake.rs` previously mixed the dead bake path with reference-cloud
  helpers. CLOUD-1 removed those helpers and the body-name-selected
  `reference_clouds.rs` authority; terrestrial clouds now project the canonical
  per-body `CloudWeatherField` (ADR-20260720T212214Z-one-weather-field-many-cloud-projections).

The genuinely-dead code is the `compile_terrain_config` /
`compile_initial_static_surface` / `compile_manifest_to_static_surface` /
`dynamic_surface_layers_for` function chain — reachable only from within the
terrain crate's own dead chain. Removing it is **function-level surgery inside
~20 mutually-referencing live modules**, inside a system CLAUDE.md marks as
under active iteration — high effort, modest payoff, real risk to the live
schema + SurfaceQuery. **Deferred** as not worth it now; the biggest wins are
B/C. Revisit only if the terrain generator is reworked (a natural time to drop
the compile chain wholesale).

**Landed anyway (compiler-verified-safe small cleanups):** deleted
`crates/domain/terrain/examples/oceanic_bake_timing.rs` (dead bake-timing harness);
removed uncalled `install_dynamic_surface_state` + `dynamic_surface_for` from
`solar_system_state.rs` (dropped the now-unused `PlanetSurface` import, kept the
`DynamicSurfaceState` seam); removed uncalled `from_static_surface` builders +
the `StaticSurfaceData` import from `impostor/material.rs`. Tree still compiles.

### B. GameContext Phase 3 — LANDED 2026-07-05 ☑ (game-UNVERIFIED)

`NextState<GameContext>` is now the **only** mode writer (pause-menu buttons,
Escape, the hub facility flows, the shipyard Launch/Exit, the launch-select
picker, the L-launch); a `ContextHistory` return stack replaces
`ReturnToSpaceCenter` + `return_to_menu` + `restore_after_facility`'s edge
latch; `InitialContext` (consumed on `OnEnter(Running)`) replaces the two
`Open*OnStart` flags. The `.open` booleans are kept as **derived mirrors**
(`mirror_context_to_booleans`, sole writer; `OnExit(Running)` resets them) so
every `.open` reader — input gating, the three `apply_open_state`, run
conditions — is untouched. The 9-rung Escape ladder collapsed into a
context-aware back-out. Compile + clippy clean (game crate). Deleted dead
`expected_context` + its test and `RelaunchInFlight::active` on contact.

**Deferred to a B2 follow-up (optional):** moving the three `apply_open_state`
HUD hide/restore into `OnEnter`/`OnExit(GameContext)` and deleting the `.open`
mirrors once no reader needs them. Kept as mirrors here for minimal blast
radius — the sanctioned ui_flow.md option.

*Verify (user):* menu→hub→VAB→(Exit)→hub→(Escape)→pause; menu→hub→EDIT BASE→
Escape→hub; flight→pause→SPACE CENTER→Escape→flight; flight→pause→SHIPYARD→
Exit→flight; VAB→LAUNCH→pick runway/pad→fly; VAB→LAUNCH→Escape-cancel→flight
(craft in orbit); `just game hub` / `shipyard` boot into the right mode;
pause→MAIN MENU from each; no HUD flash on facility→hub handoff.

### C. One craft-placement core — CORE LANDED 2026-07-05 ☑ (game-UNVERIFIED)

The canonical-state ritual (`clear bubble → set_ship_state → set_attitude →
transition_authority`, previously re-assembled — and occasionally mis-ordered —
at ~9 sites) is now **one core** in `spawn.rs`:

```rust
struct CraftPlacement { state, attitude, authority }
place_craft(sim, placement, teardown: Option<(&mut Commands, &mut ActiveLocalBubble)>)
coast_placement(state, attitude) -> CraftPlacement   // the common OnRails case
```

`place_craft` encodes the ordering invariant in one place, with the bubble
teardown (`teardown = Some(..)`) up front so `spawn_player_avian_body` reseeds
from the placed pose — the documented fix for the "buzzing/jitter after
teleport" bug class. Routed through it (behavior-neutral): `spawn.rs`
(`refine_descent_spawn`), `scenario_menu::respawn_into` (all 3 ship branches),
`runway::{place_on_runway, place_approach}`, `place::place_on_launchpad` (its
separate `clear_bubble` folded into the `teardown`), `relaunch`. Compile +
clippy clean. Site-specific extras (throttle, target body, engine relight,
gear/brake) stay at the call site.

**Deliberately not routed (behavior-risk / low value):**
- `body_tree_panel` (map cmd-click teleport) and `debug` teleport use the
  **opposite order** (`transition_authority` *before* `set_ship_state`); routing
  them would reorder, which needs runtime verification of whether
  `transition_authority` snapshots state. **Candidate bug flagged:** the map
  teleport does **not** clear the Avian bubble (violates "every craft teleport
  clears the bubble") — routing it through `place_craft` with `teardown` would
  fix it, but confirm the reorder is safe first.
- The remaining §2.1 helper dedups (shared cursor→pad raycast in
  `launch_select`/`place`, `measure_runway_clearance` vs `craft_extent_below`)
  and the generic deferred-placement gate (`DescentPlacement` /
  `RunwayPlacement` / `LaunchSpaceportBuild` — the audit deemed unifying these
  *optional*) are left as C-follow-ups.

*Verify (user):* every spawn/respawn/relaunch/launch path still seats the craft
correctly — orbit/descent/final/cruise boot, destruction respawn into each,
VAB→LAUNCH onto runway + onto pad, base-editor L-launch, runway/approach boot.
Watch for any teleport jitter (the bubble-teardown ordering).

> **In-flight (user, 2026-07-05):** `rendering/view_anchor.rs` introduces
> `ViewAnchor` — "the one per-frame answer to *where is the view?*" — a
> single-writer body-fixed resource that surface scatter (grass/trees/rocks),
> the sun-shadow cascade centre, and future clipmap layers all read instead of
> anchoring to the player craft or a per-mode focus override. This is a
> package-D-shaped unification (one canonical anchor, N camera modes correct by
> construction) and it retires the `scatter_view_center` fallback chain +
> `god_view_active` plumbing (the [[god-view-scatter-centering]] hack). The
> shared f64 `snap_to_body_surface` utility below should build on / align with
> `AnchorBody::cam_world` / `ground_world` rather than duplicate them.

### D. Structures become the one placement layer ☐

Split `runway.rs`: geometry/designators → `structures/runway_geometry.rs`,
placement → package C's module, collider sync + f64 anchoring + spaceport
building → `structures/`. Promote **connections into `StructureRegistry`**
(fixes scatter clearing under taxiways structurally). Extract the shared
f64 body-fixed anchoring utility (`snap_to_body_surface`) and adopt it at the
runway/structure/EVA/god-view sites. `structures.rs` + registry become the
single "thing anchored to terrain" system — runways, pads, buildings, tanks,
connections are all just `StructureKind`s with N-instance support (runways
already are; make the rest uniform).
*Verify:* spaceport looks unchanged (`just screenshot` spaceport-aerial),
z-fighting/grass-clearing checks, base editor place/edit/remove round-trip.

### E. N-craft groundwork (posture, not multiplayer) — SEAM + LEDGER 2026-07-05 ◐

**Scoping decision (2026-07-05):** the full re-homing the audit imagined —
migrating ~13 `PlayerShip.single()` sites to an accessor *and* moving
`GearState` / `ParkingBrake` / `EvaMode` / `RealizedControl` / `ManeuverPlan`
(the last used at **23 sites**) onto the craft entity — is **pure churn with
zero benefit while there is one craft**, verifiable only as "nothing broke", and
risks baking in the wrong abstraction speculatively. The durable, low-risk value
of E is the **posture**, not the mass migration. So E delivers:

1. **`ActiveCraft(Option<Entity>)`** (`rendering/types.rs`) — the declared
   N-craft **accessor seam**, kept mirroring the active `PlayerShip` each frame
   by its sole writer `track_active_craft` (`SimStage::Sync`). It is the single
   sanctioned answer to "which craft is active"; consumers take the craft by id
   (`None` in the respawn/relaunch window) instead of `q.single()` (which
   *panics* the moment a second craft exists). *(Code written 2026-07-05,
   UNVERIFIED — workspace build blocked by an unrelated in-progress `thalos_ui`
   crate.)*
2. **The convention** (also in CLAUDE.md's sprint rules): new/reworked per-craft
   state is a **component on the craft entity**, never a new global resource; new
   craft-entity reads go through `ActiveCraft`. Existing `.single()` sites and the
   five global per-craft resources migrate **incrementally** (mixed idiom is
   fine, per the plan) — ideally when a second craft is actually added, so the
   change is requirements-driven and runtime-verifiable, not speculative.
3. **Kept-singleton ledger** (below) — the single-instance assumptions that stay,
   knowingly, with their accessor boundary.

**Direction update (2026-07-24):** real multi-craft is now a product
requirement, led by persistent physical stage separation. The 2026-07-05
"posture only" limit remains useful history but no longer governs scope.
ADR-20260724T230226Z resolves the canonical shape as a deterministic
`CraftId`-keyed vessel registry with active craft as a separate selection.
The executable migration and acceptance contract is
`docs/simulation/vessels.md`; backlog rows now sequence the fleet kernel,
runtime identity, shared local scene, physical graph cut, and switching /
persistence.

**Fleet kernel landed 2026-07-25:** `Simulation` now owns the deterministic
registry and per-vessel bookkeeping, id-addressed mutation seams, and
active-selection compatibility wrappers. CL-E2 is unblocked; the remaining
singletons in this section are runtime/ECS concerns.

**Kept singletons (deliberate, N-craft seams recorded):**

| Singleton | Where | Accessor / boundary | N-craft path |
|---|---|---|---|
| One canonical craft state + authority | `Simulation` (`ship_state`/`set_ship_state`/`transition_authority`) | the `place_craft` core (§C) is the sole seat-the-state path | **resolved:** deterministic `CraftId`-keyed vessel registry; current methods become active-craft wrappers |
| One Avian bubble slot | `ActiveLocalBubble.bubble: Option<…>` | `clear_bubble` / `spawn_player_avian_body` | **resolved direction:** one dominant-body local scene with N vessel rigid bodies, not one bubble per craft |
| The active craft entity | `PlayerShip` (+ `.single()` sites) | **`ActiveCraft`** (new) | picks the active craft among many |
| Per-craft global resources | `GearState`, `ParkingBrake`, `EvaMode`, `RealizedControl`, `ManeuverPlan` | still global resources today | move onto each `CraftRoot` as components in CL-E2 |
| Per-craft regime record | `CraftRegimeState` | **already per-craft component** (the template) | — none, already N-safe |

*Verify (once the workspace builds):* full scenario matrix behaves identically —
`ActiveCraft` is a pure add (mirrors the one craft), so nothing observable should
change; destruction respawn / relaunch clears it to `None` during the rebuild
window and repopulates it.

### F. Small unifications (batch as touched) ☐

Document the terrain-height three-mirror design at `terrain_registry.rs`;
camera.rs submodule split (only when camera work next happens — don't do it
cold). *(The "one menu-button builder" item was subsumed by package G.)*

**BL-25 (2026-07-21):** freecam now projects one latched `ViewAnchor` body into
a complete body-fixed camera pose (position + orientation) while warp advances,
with an inertial fallback when no terrain-backed view body exists. The body ID
is captured once on entry rather than recomputed by another nearest-body path;
this keeps `ViewAnchor` the canonical selector and prevents frame-switch jumps.

*Verify (user):* in `just game orbit`, enter freecam (F4), raise warp, and
confirm a parked surface feature and the camera heading stay fixed. Translate
or rotate the camera, then repeat to confirm the updated pose stays fixed too.

### G. UI kit consolidation — LANDED 2026-07-05 ☑ (game-UNVERIFIED)

The "duplicated menu-button builders" finding, solved structurally: a new
**`thalos_ui`** crate (see `docs/gameplay/ui.md`) owns design tokens, a frosted-glass
panel material (+ scene-copy backdrop pass), and the whole widget library.
Deleted on contact: the game's `ui_widgets.rs`, the shipyard editor's private
`ui/widgets.rs`, and four per-screen `update_button_visuals` clones. Every
menu/editor screen (main/pause/settings/scenario/hub/base-editor/shipyard +
loading) now composes the kit; `HudTheme` re-points at the shared tokens.
Shipyard UX pass in the same change: status bar → toasts + pending pill,
HANGAR craft load/save overlay, kit text field for the ship name.
Verified: compile+clippy clean, kitchen-sink PNG (`just ui-preview`),
headless hub screenshot. **Runtime interaction pass still needed** (hover /
drag / typing / hangar flows).

## 4. Rules of engagement

- **No new parallel mechanisms** while this sprint runs: new modes go through
  `GameContext`, new placements through the package-C core (once it exists),
  new per-craft state on the craft entity, new terrain-anchored things through
  `structures.rs`.
- **Verify audit claims before acting on them.** The §2 baseline came from an
  automated sweep; line numbers and "never called" claims must be re-checked
  at edit time (the Phase-3 "blocked" claim was already stale).
- **Delete, don't deprecate**, except where a one-package transition mirror is
  explicitly called out (package E).
- Status for these packages lives in `docs/backlog.md`, not here. When a package
  lands, flip its backlog row and update the spec it changed (`ui_flow.md`,
  `regimes.md`, `base_building.md`, `boot.md`) — this doc keeps the rationale.
