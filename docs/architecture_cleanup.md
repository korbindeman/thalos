# Architecture & code quality cleanup

**Status:** plan drafted 2026-07-05 from a three-track codebase audit (craft
lifecycle, mode/state machinery, duplication/dead-code inventory). Nothing in
this doc has been executed yet unless its checkbox says so.

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
  `crates/terrain` + `body_render`, confusing every search and every new
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
| One canonical craft state | `Simulation` (`ship_state`/`set_ship_state`) | multi-craft API or per-craft records — **decision needed, defer implementation** |
| One bubble slot | `ActiveLocalBubble.bubble: Option<…>` | per-craft component or map |
| One authority answer/frame | `AvianAuthority` resource | already derived from per-craft `CraftRegimeState` — finish the projection |
| One control pipeline | `ControlDemand`/`RealizedControl` resources | per-craft components |
| One flight plan | `ManeuverPlan` | per-craft |
| One gear/brake/EVA-mode/flap state | `GearState`, `ParkingBrake`, `EvaMode`, `FlightConfig` | per-craft components |
| One focus/billboard/HUD subject | `CameraFocus`, `ShipMarker`, HUD panels | active-craft indirection |
| `.single()` on `PlayerShip` | queries throughout | `ActiveCraft` resource naming an entity |

`CraftRegimeState` (per-bubble-entity, from the regimes A3 port) is the proof
that per-craft components work here — it is the **template** for migrating the
rest. The realistic near-term posture: do **not** build multiplayer-grade
N-craft now, but (a) stop adding new single-craft *resources* — new per-craft
state goes on the craft entity; (b) route reads through an `ActiveCraft`
indirection instead of `.single()`; (c) record every knowingly-kept
single-instance assumption in this doc.

### 2.3 Mode/state machinery

`GameContext` (docs/ui_flow.md) Phase 1 (shadow) and Phase 2 (flip consumers:
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

- **Dead bake pipeline, ~11 kLOC**: `crates/terrain/src/{feature_compiler,
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
- Minor: duplicated menu-button builders, dead `format_thrust` /
  `format_duration_s` (compiler already warns), terrain-height three-mirror
  design is *intentional* but undocumented at the code site.

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
- `impostor/bake.rs` mixes a dead bake path with **live cloud-cover helpers**
  (`blank_cloud_cover_image`, `equirect_to_cloud_cover_image_with_rotation`)
  called via `reference_clouds.rs` from `rendering/spawn.rs`.

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
`crates/terrain/examples/oceanic_bake_timing.rs` (dead bake-timing harness);
removed uncalled `install_dynamic_surface_state` + `dynamic_surface_for` from
`solar_system_state.rs` (dropped the now-unused `PlanetSurface` import, kept the
`DynamicSurfaceState` seam); removed uncalled `from_static_surface` builders +
the `StaticSurfaceData` import from `impostor/material.rs`. Tree still compiles.

### B. Finish GameContext Phase 3 (docs/ui_flow.md) ☐

Now unblocked. `NextState<GameContext>` becomes the **only** mode writer
(buttons, Escape, facility flows); add the `ContextHistory` return stack
(deleting `ReturnToSpaceCenter` + the edge latches); collapse the three
`apply_open_state` + HUD hide/restore stacks into per-context
`OnEnter`/`OnExit` systems; per-context Escape handling; `.open` booleans
become read-only mirrors, then die. This is the biggest bug-class kill in the
plan (every mode-transition race traces here).
*Verify:* the ui_flow.md Phase-3 checklist — menu→hub→VAB→launch→flight→pause
→hub→flight round-trips, Escape at every level, no HUD flash, no one-frame
world flash on VAB open.

### C. One craft-lifecycle module ☐

Create `crates/game/src/craft/` (or `spawn/` grown properly) owning a single
canonical placement core:

```
PlacementSpec { state, attitude, authority, body, engines: EngineState,
                flatten_prereq: Option<…> }
place_craft(&mut sim, &mut bubble, spec)   // encodes: clear bubble →
                                           // set state → set attitude →
                                           // set authority, in that order
```

Every entry point — boot scenarios, `respawn_into`, `begin_relaunch`,
`apply_launch_placement`, dev runway park, EVA teleports (bubble-preserving
variant) — routes through it. Fold in the missing helpers from §2.1 (shared
raycast, clearance measurement, scenario→authority). Encode the big_space
root-seating invariant in `build_player_ship` itself (already done — keep it
there, add a doc-comment naming the invariant). Deferred placement gets one
generic armed-gate (`DeferredPlacement` with a reason enum) replacing the
three copies.
*Verify:* all seven spawn paths, plus the two historical regressions (VAB
launch buzzing, runway teleport without bubble clear) as explicit user checks.

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

### E. N-craft groundwork (posture, not multiplayer) ☐

Apply §2.2's rules: introduce `ActiveCraft` (entity id) and migrate
`.single()` consumers to it; move per-craft state that is currently a global
resource (`GearState`, `ParkingBrake`, `EvaMode`, `RealizedControl`,
`ManeuverPlan`) onto the craft entity as components, with the resource kept
briefly as a deprecated mirror if needed; document the knowingly-kept
singletons (`Simulation`'s one canonical state, one bubble slot) here with
their accessor boundaries. No gameplay change — this package is pure
re-homing.
*Verify:* full scenario matrix behaves identically; destruction respawn keeps
per-craft state coherent.

### F. Small unifications (batch as touched) ☐

One `HudTheme` menu-button builder; document the terrain-height three-mirror
design at `terrain_registry.rs`; camera.rs submodule split (only when camera
work next happens — don't do it cold).

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
- Update this doc's checkboxes + the relevant spec (`ui_flow.md`,
  `regimes.md`, `base_building.md`, `boot.md`) in the same change that lands a
  package.
