# Boot, loading, and the start screen

**Status: implemented (2026-06-12).** This doc specifies the app-state
graph, the loading-step tracker, and the start screen — the systems that
take the process from `main()` to a ready, revealed scenario. It replaced
the ad-hoc `LoadingProgress` counter (four hard-coded gates poked directly
by producer systems, with no way to boot into anything but the CLI-chosen
scenario).

Related: `docs/simulation/simulation.md` (authority + warp the scenario starts drive),
`docs/simulation/surface_local.md` (the runway placement this gates on),
CLAUDE.md *Invariants* (the paused-on-spawn / `THALOS_AUTO_RUN` contract).

## 1. App states

`crate::loading::AppState` (game crate; `crates/runtime/game/src/loading.rs`):

```
Loading ──(all tracker steps complete)──▶ LoadDestination
                                            ├─ MainMenu   (bare launch)
                                            └─ Running    (direct fixture boot)
MainMenu ──(SessionLoadRequest accepted)──▶ Loading? ──▶ Running
```

- `Loading` is the **default state** and covers the very first frame (also
  masking the swapchain-uninitialised flash). The loading screen UI spawns
  at `Startup` *and* `OnEnter(Loading)` (idempotent), so runtime re-entries
  get the screen back.
- `LoadDestination` is a resource set once in `main.rs` (start screen for a
  bare launch, `Running` otherwise); the start screen rewrites it to
  `Running` before re-entering `Loading` for a runway start.
- The **sim is never paused by `Loading`** — deferred placements settle the
  craft and stream tiles behind the screen (see `docs/` notes in
  `crate::surface_settle`). `MainMenu` *is* a sim-clock pause source, and
  gameplay input contexts deactivate there (`crate::input`), exactly like
  the shipyard editor's frame ownership.
- `OnEnter(Running)` hooks fire on **every** entry (menu starts included):
  `spawn::apply_initial_warp` (paused-on-spawn policy, `THALOS_AUTO_RUN`
  override) and `shipyard_editor::open_on_start` (one-shot — it clears its
  own flag).

## 2. The loading tracker

`crate::loading::LoadingTracker` is a declarative step registry. A load is
declared **up front** — `tracker.begin(steps)` registers the full step set
(id, label, bar weight) — and producer systems update only their own step
by id. Updates to unregistered steps are no-ops, so producers are oblivious
to which scenario is loading. `is_complete()` requires the set to have been
registered (`begin` seals it), which kills the frame-0 "0/0 is trivially
done" class of bug the old counter needed a `seeded` flag for.

Steps are built by `loading::steps_for(situation, boot)`:

| id | label | registered | completed by |
|---|---|---|---|
| `session` | Projecting session | live-world non-spaceport session replacement | `session_loading::complete_live_session_projection`, after relaunch becomes idle |
| `bodies` | Celestial bodies | boot only | total seeded by `rendering::spawn::spawn_bodies`; advanced (+ body-name detail) by `rendering::generation::poll_planet_install_tasks` |
| `terrain` | Surface terrain | boot only | `rendering::terrain_residency::initial_residency_loading_gate`, gated on `bodies` completing first |
| `placement` | Placing craft | scenarios with a deferred placement | `spawn::refine_descent_spawn` or `runway::finish_runway_spawn` |
| `settle` | Settling terrain | parked `Runway` only | `surface_settle::update_surface_settle` (publishes a live m/texel detail + progress estimate) |

The reveal now waits on `placement` — previously the descent/runway state
could install *after* the reveal. The loading screen renders the weighted
overall bar, the active step's label/detail/count, and a `step i/N` line.

Backstops: `surface_settle` keeps its own settle/placement timeouts, and
`loading::finish_loading` has a 120 s hard timeout that reveals with a
warning rather than hanging (a stalled bake task or placement).

**Adding a loading step** (any future system): include a `StepDesc` in
`steps_for` (or whatever calls `begin`), then `tracker.complete(id)` /
`advance` / `set_detail` from the producer. Nothing else to wire.

## 3. The start screen (`crates/runtime/game/src/main_menu.rs`)

Shown when no scenario is named: bare `just game` (the justfile default
mode is now `menu`), `THALOS_SPAWN=menu`, or a bare `cargo run`. Skipped
by: an explicit scenario, `just game shipyard`, `just game hub` (the PLAY
route without the menu: orbit placeholder + `HubSpaceportBuild` +
`InitialContext(Some(SpaceCenter))` armed at build, and `register_boot_steps`
appends the PLACEMENT step — the headless `hub` screenshot preset rides this), and
**`THALOS_AUTO_RUN`** (truthy) — agents keep a one-shot launch into orbit.

**A bare menu boot defers the world** (`loading::WorldState`, default
`Absent`): no celestial bodies, player ship, or star-field are spawned, no
terrain streams, and `register_boot_steps` registers an **empty** step set,
so the menu reveals on the first update — a static UI over an empty scene.
The world-spawn systems (`rendering::spawn_bodies` +
`focus_camera_on_homeworld`, `ship_view::spawn_player_ship`,
`sky_render::dispatch_sky_generation`) are registered on
`OnEnter(WorldState::Live)` instead of `Startup`; a `just game <scenario>`
boot **queues** `Live` from a `Startup` system (never `insert_state(Live)`
at build — Bevy fires the *initial* `StateTransition` before `PreStartup`,
which would run the world-spawn chain before `Startup`'s resources like
`RealSpaceRoot` exist and panic every spawn system), so the same chain
fires at the first regular state transition (same frame, after `Startup`,
still behind the loading screen) and the scenario boot is unchanged. While the menu is up, `WinitSettings` is
swapped to reactive mode (~30 Hz idle / instant on input), so an idle menu
costs near-zero CPU/GPU; the continuous game loop is restored on exit.
`WorldState` currently records whether process-level world projections have
ever been created. It may remain `Live` across a session replacement so content
and renderer services can be reused; it is **not** campaign authority and must
not be used for uniqueness or save decisions.

PLAY and every developer shortcut submit one generation-stamped
`SessionLoadRequest`:

- **PLAY** selects `SessionSource::NewCampaign`: a durable campaign adapter,
  default space-center situation, and `GameContext::SpaceCenter`.
- **Quick Start / Dev** selects a versioned `ScenarioFixture` hosted by an
  ephemeral campaign adapter. Flight, space-center, and shipyard fixtures all
  enter the same coordinator.
- `session_loading::apply_session_load_request` validates the fixture craft
  before mutation, clears former transient requests, assigns the situation and
  entry context, and arms the projection workers. It is the sole consumer.
- With an absent process world, it also registers world-load steps and flips
  `WorldState::Live`. With a live world, every non-EVA flight fixture rebuilds
  its declared blueprint through `RelaunchRequest`; orbit/landing may not retain
  a coincidentally compatible outgoing craft. This keeps EVA→orbit and all
  craft appearances/behavior equal to a cold fixture boot.
- Spaceport fixtures and New Campaign reconcile the authored `BaseId` through
  `runway::ensure_spaceport`. `RunwaySite` is only a cache; repeated requests
  cannot append a second base or second set of geometry.
- The new `ActiveSession` generation is published only once the app is
  `Running` and craft replacement is idle. Validation failure leaves the old
  active session untouched.

SETTINGS and QUIT remain process-shell actions and do not enter the loader.

Escape on the menu only closes the settings overlay; the pause-menu Escape
chain is gated to `Running`.

Returning to the main menu and starting another fixture is supported. Complete
authoritative resource replacement (rather than the current compatibility
adapters over singleton runtime state) is tracked by the campaign snapshot
slice in `docs/gameplay/campaigns.md` §8.

## 4. Runtime-scenario invariants

`SpawnSituation` is temporary fixture-adapter vocabulary and remains mutable at
runtime. Gameplay must not branch on fixture origin. Consequences already
handled during the complete-snapshot migration:

- Deferred placements are **explicitly armed** (`DescentPlacement`,
  `RunwayPlacement` resources), not keyed off the situation resource with
  a `Local<bool>` — a situation switch must not retrigger them, and
  runtime starts place synchronously instead (descents) or re-arm
  deliberately (runway).
- The runway/cruise engine lighting is keyed **per ship root** and gated
  on the `StagingPlan` existing, so a runtime craft swap lights the new
  craft after staging's disable-at-spawn pass — never the outgoing craft.
- `runway::finish_runway_spawn` runs under `relaunch::relaunch_idle` so it
  never measures/parks the outgoing craft mid-swap, and after placing it
  **tears down any live Avian bubble** (`scenario_menu::clear_bubble`) —
  a bubble seeded from the pre-placement placeholder orbit must be
  rebuilt from the placed state, or the rendered craft diverges from the
  canonical one.
- **Runtime-built craft must be seated into BigSpace.**
  `rendering::real_space::attach_player_ship_to_big_space` runs every
  frame (filtered on `Without<CellCoord>`, so it's a no-op once attached),
  not just at startup: `ship_view::build_player_ship` spawns a bare root,
  and an unattached root silently misses the canonical→render transform
  sync — the craft freezes in the inertial frame while the planet sails
  away at orbital speed (the "parked at 5 m AGL but floating in space"
  bug).
- The runway site is a **fixed body-fixed location** (constant lat/lon +
  heading via `runway::fixed_runway_site`, overridable with
  `THALOS_RUNWAY_SITE="lat,lon[,heading]"`), not an auto-chosen scan. This
  replaced the former daylight-first flat/dry scan, whose night-side
  fallback could reveal a parked spawn in the dark; the pad is flattened
  under the footprint regardless, so the coordinates only need to be dry
  land sunlit at the spawn epoch (a below-sea-level sample is warned in the
  log, not corrected). The settle gate logs a bounded diagnostic line every
  ~5 s (resident LOD + the tile tree's view radius) — if a settle ever
  times out, that view radius says immediately whether the streamer's view
  actually reached the site.
- The destruction picker (`scenario_menu`) deliberately does *not* write
  `SpawnSituation` — respawns keep the boot scenario's per-frame consumers
  unchanged.
