# CLAUDE.md

Operating manual for agents working in this repository. It carries only what an
agent needs **in context every session**: current direction, how to verify work,
and the invariants that are expensive to rediscover. Everything else lives in
`docs/` — **`docs/README.md` is the canonical documentation map**. Cursor loads
specialist rules from `.cursor/rules/` when matching files are in play.

## Project purpose

**Thalos** currently names three things; keep the boundary explicit:

- the **Thalos game** is the product: a spaceflight simulator intended for
  release, not a renderer showcase;
- the **world foundation** is the internal engine for representing and rendering
  natural worlds (planar maps through solar systems);
- **Kòrsou** is a real secondary project and focused laboratory — not a demo and
  not a scenario inside the game.

The game is the integration anchor. Focused apps may deepen a system; promote
into the foundation only when the mechanism genuinely matches, then bring it
back. Share meaning, not topology: planar / cube-sphere / analytic-body /
far-body / ellipsoid stay in explicit adapters. Full purpose: `docs/purpose.md`.

Planetary exploration / orbital-mechanics sandbox in Rust (edition 2024, Bevy
0.19, glam 0.32), **pre-alpha**. Tear down infrastructure that is lacking and
replace it — leave a trail in this file plus the relevant `docs/` spec. No
silent rewrites.

## Current focus

**Keystone** (`docs/roadmap/neural_terrain_renderer.md`, `ntr §N`;
ADR-20260723T142945Z): *make Thalos look good*.

- **Neural terrain** is the default height backing (`DiffusionSurface`;
  `THALOS_TERRAIN=procedural` is the session A/B), fine-tuned offline behind the
  terrain-package boundary. Thalos/earth-like first.
- **Tiles are the default ground** (`thalos_body_render::tiles`): ordinary
  `Mesh` + `StandardMaterial`/`ExtendedMaterial` on Bevy's lighting path.
  `thalos_udlod` and the old terrain WGSL stack are sealed legacy. Spine-port
  graphics items (F4r/F5r/F7/F8/F9/W12r/TM1) stay frozen; composites (clouds,
  atmosphere, ocean, plumes, celestial sky, capture) continue. MIRA-1 paused
  after L2.

What's next is `docs/backlog.md`, not this paragraph. Background sprints:
`docs/roadmap/architecture_cleanup.md` (`clean`),
`docs/roadmap/application_runtime.md` (`app`),
`docs/roadmap/graphics_fidelity.md` (`gfx` — one-world lighting still holds).

## Steering & memory

**Default: do the work.** Writing is the exception (ADR-20260724T223339Z).

- **`docs/backlog.md`** — the only status authority (`next` / `wip` /
  **`verify`** / `blocked` / `done` / `later`). Rows are for work that outlives
  the session. Plan docs hold rationale, never a parallel checkbox.
- **`steer` skill** — "what's next?" → propose, then stop for a go; "add X /
  fix Y" → just do it (a row only if it won't finish now); vision → update the
  plan doc, then decompose into rows.
- **`docs/adr/`** — expensive-to-reverse decisions. `rg '<topic>' docs/adr`
  before reopening; supersede, don't rewrite.
- **`docs/reviews/`** — `expert-review` survivors, not tracked work. Check
  `docs/reviews/dismissed.md` before re-filing a settled defect.
- **`docs/incidents/`** — non-obvious bugs: symptom, mechanism, recurrence tell.
  Written in the same change as the fix.
- **IDs are chronological**: `<KIND>-<YYYYMMDDTHHMMSSZ>-<kebab-slug>`. Unix:
  `date -u '+%Y%m%dT%H%M%SZ'`. PowerShell:
  `[DateTime]::UtcNow.ToString("yyyyMMdd'T'HHmmss'Z'")`. Never allocate "the
  next number" (ADR-20260722T170714Z).

## Standing quality bar

- **One canonical path per operation.** A new entry point parameterizes the
  core; it does not fork it.
- **N by default.** No new system assumes "the one craft / runway / base".
- **Finish in-flight unifications first** — `GameContext`
  (`docs/gameplay/ui_flow.md`) and `CraftRegime` (`docs/simulation/regimes.md`)
  outrank new features.
- **Delete dead code on contact.**
- **Crates split on payoff** (ADR-20260731T024003Z): cheaper edit loop,
  compiler-enforced dependency, standalone preview, or agent isolation. Feature
  crates depend only downward (domain + `thalos_game_state`), never on each
  other. Don't split what's scheduled for demolition (`thalos_udlod`, the
  procedural terrain chain). Layers: `docs/architecture.md`.
- **The shared runtime is light by default.** `thalos_runtime` is an
  empty-default capability facade (ADR-20260809T201216Z). A disabled capability
  must be absent from the dependency graph, not merely inactive.
- **Size work in LLM tokens, not days.** "~200k tokens, one session" — not
  "half a day". Wall-clock still counts for builds, capture latency, and play.

## Bug fixing

1. Hypothesis set from the symptom — don't jump to the first plausible fix.
2. Rule candidates out with targeted, falsifiable tests.
3. Fix the cause, structurally.

A silent symptom-disappearance is not a fix. `rg '<symptom>' docs/incidents/`
before re-deriving. Non-obvious diagnosis → post-mortem in the same change;
typo-grade fixes need none.

## Commands

```bash
just game [mode]            # USER-RUN ONLY (owns the machine-wide renderer lease)
just check [package]        # fast type-check (default thalos_game)
just screenshot <name>      # headless still → artifacts/visual/latest/
just capture <name>...      # batch through one host
just compare <preset> <axis>
just diag [hours]           # what crossed a threshold in the runtime + tool lanes
just map                    # whole-planet biome + relief maps
just preview / ui-preview / loading-preview
```

The justfile is the catalog; `docs/development/tooling.md` holds contracts.
New agent-facing entry points go there, not here.

## Verification

**Do not launch the game.** `just game` is the user's job. Headless capture is
yours: `just screenshot` / `capture` / `compare` / `preview` / `ui-preview` /
`loading-preview` / `just map`. Read the PNGs directly. Framing knobs without a
recompile: `THALOS_SCREENSHOT_{AZIMUTH,ELEVATION,DISTANCE,SIZE,OUT,WARMUP,HUD}`.

**Check provenance** in the neighboring `.capture.json` (`workspace_matches`).
If false, the image includes an older source floor — recapture only when a later
edit is itself under verification.

A terrain / biome / scatter / visual change that compiles but hasn't been
screenshotted is `verify`, not `done`. Add a `ScreenshotPreset` if none frames
it.

**A dirty worktree is not your problem.** Confirm breakage is outside your
edits, finish, and say what blocked verification.

**Visual complaints: capture first, then stop for a yes.** Say back in plain
language what you see, what you believe the problem is, and what you will touch
versus leave alone. If the capture doesn't show the symptom, say that.

**Comparison loop** (full: `docs/development/visual_testing.md`): one typed
axis; inspect stderr → `manifest.json` → contact sheet → full frames; reject
the run if any variant logged a shader/pipeline error (**BL-20**: Bevy can
fatal-log and still exit 0). Pin the cause before patching.

Player handoff: **F9** saves the view; `just screenshot <slug>` replays it
(`latest` = newest). **F8** manages `assets/viewpoints.json`.

Hand work back in chat. Show PNGs when they are evidence. No HTML report unless
the user asks.

## Observability

An agent diagnoses from data already in the tree. When in doubt, emit the
number. Full contract: `docs/development/tooling.md`.

- One lane: `info!(target: "thalos::diagnostic::<subsystem>", event = "…",
  field = …)` → `artifacts/diagnostics/runtime.jsonl`. Console is for humans.
  Tools: `thalos_diagnostics::install_tool_lane()` → `tools.jsonl`.
- Events are data (`snake_case` noun, unit-suffixed scalars). Gauges carry
  their denominator. A diagnostic that cannot falsify a hypothesis is not one.
- Instrument first, then patch. Every event needs a `just diag` check
  (`tools/diag/src/checks.rs`) or it does not ship.
- Capture slowness and silent-wrong output are first-class defects. Read
  `just diag`, the `.capture.json` receipt, and the JSONL before asking the
  user to reproduce.

## Build & iteration

Rust 1.97.0 (`rust-toolchain.toml`). Policy: `docs/development/build_speed.md`.

- **One Cargo command at a time** against workspace `target/`. Parallel agents
  need separate worktrees (`scripts/setup-build-env.{sh,ps1}`).
- Screenshot loop: WGSL **hot-reloads** (~3 s); Rust/manifest **restarts the
  host** (~1.5–2.5 min). No in-process Rust reload (ADR-20260724T153619Z).
- **Never** `cargo clean -p <subset>` — partial clean poisons `bevy_dylib`
  (INC-20260724T182642Z). Capture client self-heals once; then `just build-reset`.
- Every dev renderer lane stays on the one `dev-renderer` fingerprint.
- No unstable `-Zthreads` (INC-0006). No compiler cache (ADR-20260723T222214Z).

## Codebase map

Crate/module anatomy: `docs/architecture.md`. Behaviour: specs in
`docs/gameplay/`, `docs/simulation/`, `docs/world/`, `docs/rendering/`.

Sim order is **Physics → Sync → Camera**; input intent runs in `PreUpdate`
before them. Feature crates order against `thalos_game_state::sched`, never
against each other.

## Invariants

**Crate boundaries**

- Pure crates have no Bevy, even transitively: `thalos_render_model`,
  `thalos_world`, `thalos_physics_canonical`, `thalos_terrain`,
  `thalos_celestial`. (`bevy_erosion_filter` only with `default-features = false`.)
- Avian lives behind `thalos_physics_local`; never add it to
  `physics_canonical`. Don't derive `Reflect` there — mirror into
  `CraftStateMirror` at the bridge.
- `thalos_body_render` is the sole consumer of `thalos_udlod`.
- `thalos_render_foundation` owns GPU resources and pass ordering only. It may
  depend on Bevy and `thalos_diagnostics`, never on a world, spatial adapter,
  gameplay crate, or application composition.

**Ground renderer**

- `thalos_body_render::tiles` is the ground. New terrain / shading / scatter /
  LOD work goes here, driven per frame from `ViewAnchor`.
- `thalos_udlod` + `body_render::ground`'s terrain half + `body_terrain.wgsl`
  are sealed legacy: defect-driven fixes only, behind `legacy-udlod`.
- `THALOS_TILE_RENDERER=0` is a feature-only A/B baseline, not a supported
  mode. Unset is the tile path.
- `BodySky`, `BodyOcean`, and the impostor handoff in `body_render::ground`
  are **not** legacy.

**One authority per concern**

- One `ShipPropagator` for live stepping and prediction. Body positions through
  `BodyTrajectoryProvider`.
- Presentation reads snapshots or accessors, never parallel transform-owned
  craft state.
- `SolarSystemState` is the frame-local source for body + environment; render,
  map, impostor, terrain, and materials are projections of it.
- Single-writer resources: the writer is named in the resource's doc comment.
  Known: `SolarSystemState` ← `sync_solar_system_state`, `MapSnapshot` ←
  `update_map_snapshot`, `CraftStateMirror` ← `refresh_craft_state_mirror`,
  `CraftRegimeState` ← `regime::resolve_regime`, `AvianAuthority` ←
  `compute_avian_authority`, `ViewAnchor` ← `update_view_anchor`,
  `FlightProgram` ← `autoflight::update_flight_program`,
  `AutoflightAnnunciation` ← `control_bus::realize_control`.
- Autoflight is two layers (ADR-20260731T232619Z): strategic `FlightProgram`
  vs tactical channels `thalos_control::arbitrate` resolves. A new source
  declares `required_locks()`; panels emit `AutoflightRequest`.
- One height authority: `BodySurfaceRegistry` builds one
  `Arc<dyn SurfaceQuery>` per body. Sea level is the constant **0 m**.
- One `ViewAnchor` for every view-dependent detail system. Never anchor detail
  to the player craft or a mode flag.
- One sim-pause: `sim_clock::SimClock` → zero sim delta. Do not pause
  `Time<Virtual>`/`Res<Time>`; presentation uses `Time<Real>`.
- Real-space rendering lives under BigSpace; the active ship camera owns
  `FloatingOrigin`. Map systems read `MapSnapshot`.
- Semantic input from `thalos_input` intent resources; raw Bevy input only for
  cursor, picking, and UI internals.

**Traps**

- Every spawn starts paused (warp 0×). `spawn::apply_initial_warp` is the
  source of truth, gated by `spawn::AutoRun` (`THALOS_AUTO_RUN=1`).
- `EditorPart` is the only marker separating the build world from the flying
  craft. Flight aggregators must filter `Without<EditorPart>`.
- Tile-cache staleness is silent. New synthesis inputs go in the cache
  namespace; bump `thalos_terrain::GENERATOR_VERSION` when output changes.
  `THALOS_TILE_CACHE=0` disables the disk tier.
- Per-body failures stay per-body (`BodySurfaceRegistry::degraded`). Capture
  refuses a shot whose *target* body is degraded (INC-20260724T182643Z).
- Detached processes inherit nothing from the caller's console/pipes. On
  Windows, launch via `Start-Process` with redirection (INC-20260724T185500Z).
- No planet/terrain generation tests. Verify with `just map` +
  `just screenshot`. `crates/domain/terrain` is the source of truth.

## Learned constraints

- Primary machine is a Mac: FPS and battery are first-class. This checkout
  also runs on Windows — the handle-inheritance trap above still applies.
- 3D render scale must leave UI at OS HiDPI. Draw the 3D target at a fraction
  and upscale; `scale_factor_override` shrinks the HUD and is the wrong lever.
- Quality / Laptop must not change window mode (default borderless fullscreen).
  Laptop keeps clouds and woody foliage; levers are 0.50× 3D scale and a 30 Hz
  cap. Capture stays on Showcase. `THALOS_QUALITY` / `just … quality=` pins one
  session without writing back; capture ignores the pin.
- `thalos_clouds` owns the shared compute marcher and weather cubes;
  `thalos_body_render` keeps the planetary BodySky composite. Kòrsou consumes
  the same marcher through a local Earth-shell adapter — do not fork a second
  cloud system.
