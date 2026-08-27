# CLAUDE.md

Operating manual for agents working in this repository. It carries only what an
agent needs **before knowing the task**: current direction, how to verify work,
and the invariants that are expensive to rediscover. Fetch everything else.
Adding a paragraph here is a claim it belongs in *every* session
(ADR-20260724T222409Z). Map: `docs/README.md`.

## Project purpose

The name **Thalos** covers three related things; keep the boundary explicit:

- the **Thalos game** is the primary product: a spaceflight simulator intended
  for release, not a renderer showcase;
- the **world foundation** is the internal engine for representing, emulating,
  and rendering natural worlds;
- **Kòrsou** is a real secondary project and focused laboratory — neither a
  demo nor a scenario inside the game.

The game is the integration anchor. Focused apps may deepen a system; promote
into the foundation only when the mechanism matches, then bring it back.
Share meaning, not topology: planar, cube-sphere, analytic-body, far-body, and
geodetic-ellipsoid concerns stay in explicit adapters. Full rules:
`docs/purpose.md`.

Rust edition 2024, Bevy 0.19, glam 0.32, **pre-alpha**. Tear down
infrastructure that is lacking and replace it — leave a trail (what, why) and
update this file plus the owning `docs/` spec in the same change. No silent
rewrites.

## Current focus

**Keystone (primary; ADR-20260723T142945Z)** — *make Thalos look good*:

- **Neural terrain** — default height backing (`DiffusionSurface`;
  `THALOS_TERRAIN=procedural` is the session A/B), fine-tuned offline behind
  the terrain-package boundary; Thalos/earth-like first. Plan: `ntr §N`.
- **Renderer on Bevy's standard path** — terrain and every opaque surface as
  ordinary `Mesh` + `StandardMaterial`/`ExtendedMaterial`. Volumetrics/sky
  (BodySky, clouds, analytic ocean, celestial sky) stay custom composites.

In this repo: **`thalos_body_render::tiles` is the default ground.**
`thalos_udlod` and the terrain WGSL stack are sealed legacy (defect-driven
fixes only). Spine-port graphics items are frozen; composites and the capture
harness continue.

**Background** — architecture cleanup (`clean §N`), capability-selected
runtime (`app §N`), graphics composites (`gfx §N`: every surface obeys the
same light, shadows, occlusion, and air).

## Where to look

| Need | Open |
|------|------|
| What's next | `just queue`, then `just queue <id>` |
| Done / later history | `docs/backlog.jsonl` (`rg`; title-only lines, not the queue) |
| Sprint strategy | `docs/roadmap/<sprint>.md` (`ntr`, `clean`, `gfx`, `stab`, …) |
| Crate / module anatomy | `docs/architecture.md` |
| Why a choice was made | `rg '<topic>' docs/adr` — do not read the directory |
| Bug forensics | `rg '<symptom>' docs/incidents` |
| Capture / visual loop | `docs/development/visual_testing.md` |
| Diagnostics / tool lane | `docs/development/tooling.md` · `just diag` |
| Build / linker / worktrees | `docs/development/build_speed.md` |
| Subsystem behaviour | `docs/gameplay/`, `simulation/`, `world/`, `rendering/` |
| Docs map | `docs/README.md` |

## Steering & memory

**Default: do the work.** Writing is the exception (ADR-20260724T223339Z).

- **`just queue`** — live statuses `next` / `wip` / `blocked` in
  `docs/backlog.jsonl`. Landed work is `just backlog done <id>` (strips the
  note); user silence is acceptance (ADR-20260819T065009Z). Later
  disagreement is a new row. File a row only if the work outlives this
  session. **Note:** title + done-criteria, max 360 characters, not a
  session diary.
- **`steer` skill** — "what's next?" → `just queue`, propose, stop for a go;
  "add X / fix Y" → do it; vision → plan doc, then decompose.
- **`docs/adr/`** — decisions expensive to reverse. Ordinary calls go in the
  commit. Immutable; supersede, don't rewrite.
- **`docs/incidents/`** — non-obvious diagnoses, written with the fix. Short.
- **IDs** — `<KIND>-<YYYYMMDDTHHMMSSZ>-<kebab-slug>` from
  `date -u '+%Y%m%dT%H%M%SZ'`. Never allocate "the next number".

## Standing quality bar

- **One canonical path per operation.** A new entry point parameterizes the
  core; it does not fork it.
- **N by default.** No new system assumes "the one craft / runway / base".
- **Finish in-flight unifications first** — `GameContext`
  (`docs/gameplay/ui_flow.md`) and `CraftRegime`
  (`docs/simulation/regimes.md`) outrank new features.
- **Delete dead code on contact.**
- **Crates split on payoff** (ADR-20260731T024003Z). Feature crates depend
  only downward — domain crates and `thalos_game_state` — never on each
  other. `thalos_runtime` accepts composition, sim-coupled drivers, and glue.
  Don't split what's scheduled for demolition (`thalos_udlod`, the procedural
  terrain chain). Layers: `docs/architecture.md`.
- **The shared runtime is light by default.** Capabilities are selected at
  composition; a disabled capability is absent from the dependency graph,
  not merely inactive (ADR-20260809T201216Z).
- **Size work in LLM tokens, not days.** "~200k tokens, one session" — not
  "half a day".

## Bug fixing

1. Hypothesis set from the symptom — don't jump to the first plausible fix.
2. Rule candidates out with falsifiable tests that distinguish them.
3. Fix the cause, structurally — not the symptom.

A change that makes the symptom vanish without *why* is not a fix. Search
`docs/incidents/` first. Non-obvious diagnosis → post-mortem in the same
change; typo-grade needs none.

## Commands

```bash
just queue              # live backlog (what's next). just queue -- --json
just backlog done <id>  # close a row (strips the note)
just check [package]    # fast type-check (default thalos_game)
just screenshot <name>  # headless still → artifacts/visual/latest/<name>.png
just capture / compare  # batch stills / one-axis matrix
just map                # whole-planet biome + relief maps
just diag [hours]       # what in the diagnostic lane crossed a threshold
just game [mode]        # USER-RUN ONLY — never launch this yourself
```

Full catalog: `justfile`. Don't copy it here. A new agent-facing entry point
goes in the justfile and `docs/development/tooling.md`; promote to this file
only if every session must know it before the task is known.

## Verification

**Do not launch the game.** The user has the display. Headless capture is
yours: `just screenshot` / `capture` / `compare` / `preview` / `ui-preview` /
`loading-preview` / `map`. Read the PNG. Framing knobs:
`THALOS_SCREENSHOT_{AZIMUTH,ELEVATION,DISTANCE,SIZE,OUT,WARMUP,HUD}`.

Every PNG has a `<name>.capture.json` receipt. If `workspace_matches` is
false, the image is the capture's source floor — recapture only when a later
edit is itself under verification.

**Screenshot visual work when you can** (terrain, biome, scatter, lighting)
and read the PNG. No preset → add a `ScreenshotPreset`. Capture is how you
know, not a backlog gate: `just backlog done <id>`. Capture blocked (dirty
tree, no GPU) → say so in the report, still `done`. User silence is
acceptance (ADR-20260819T065009Z).

**A dirty worktree is not your problem.** Don't chase other agents' breakage;
don't stall on a green build. Finish your change.

**User-reported visual issue:** capture first, then say in plain language
what you see, what you think is wrong, and what you will and won't touch.
Stop for a yes. That is the one place asking beats doing.

Typed A/B is `just compare` — one axis, never a split viewport
(ADR-20260721T192218Z). Full loop: `docs/development/visual_testing.md`.
Reject a comparison if any variant logged a shader/pipeline error (**BL-20**:
PNGs can exist under a fatal pipeline error and exit 0).

Write runtime numbers to a file; never ask the user to read a console.
**F9** saves the current view; `just screenshot <slug>` replays it
(ADR-20260724T211627Z). Hand work back in chat; no HTML report unless asked.

## Observability

You diagnose as well as the data already in the tree. Contract:
`docs/development/tooling.md`. Load-bearing:

- One lane, one file: `info!(target: "thalos::diagnostic::<subsystem>", event = "…", field = …)` → `artifacts/diagnostics/runtime.jsonl`. Tools use `install_tool_lane()` → `tools.jsonl`. Don't invent a second JSONL to avoid naming an event.
- Events are data (`event = "snake_case_noun"`, scalars with units in the name). Console is for humans.
- Every diagnostic owes a reader: `just diag`. No check → don't add the event.
- Capture slowness, contention, and silently-wrong output are defects of the highest class.

## Build & iteration

`docs/development/build_speed.md`. Load-bearing:

- **One Cargo command at a time** against workspace `target/`. Parallel
  agents need separate worktrees (`scripts/setup-build-env.{sh,ps1}`).
- Screenshot loop: WGSL hot-reloads (~3 s); Rust/manifest restarts the host
  (~1.5–2.5 min). No in-process Rust reload (ADR-20260724T153619Z).
- **Never** `cargo clean -p <subset>` — `bevy_dylib` is one artifact set
  (INC-20260724T182642Z). Capture self-heals once; then `just build-reset`.
- One `dev-renderer` fingerprint for every dev renderer lane.
- No `-Zthreads` (INC-0006). No sccache (ADR-20260723T222214Z).

## Invariants

**Crate boundaries**

- **The pure crates have no Bevy**, even transitively: `thalos_render_model`,
  `thalos_world`, `thalos_physics_canonical`, `thalos_terrain`,
  `thalos_celestial`. CI-guarded. (`bevy_erosion_filter` only via
  `default-features = false`.)
- Avian lives behind `thalos_physics_local`; never add it to
  `physics_canonical`. Don't derive `Reflect` there — mirror into
  `CraftStateMirror` at the bridge.
- `thalos_body_render` is the **sole** consumer of `thalos_udlod`.
- `thalos_render_foundation` owns GPU resources and pass ordering only. It
  may depend on Bevy and `thalos_diagnostics`, never on a world, spatial
  adapter, gameplay crate, or application composition.

**Ground renderer: tiles by default, udlod is legacy**

- **`thalos_body_render::tiles` is the ground renderer**, driven from
  `ViewAnchor` by `rendering::tile_terrain`. New terrain/shading/scatter/LOD
  work goes here.
- **`thalos_udlod` + `body_render::ground`'s terrain half + the terrain WGSL
  stack are sealed legacy**: defect-driven fixes only, behind `legacy-udlod`.
- **`THALOS_TILE_RENDERER=0` is a feature-only A/B**, not a supported mode.
- BodySky, BodyOcean, and the impostor handoff in `body_render::ground` are
  **not** legacy.

**One authority per concern**

- **One propagator** — live stepping and prediction both through
  `ShipPropagator`. Bodies through `BodyTrajectoryProvider`.
- **One craft state.** Presentation reads snapshots or accessors.
- **One solar-system state.** `SolarSystemState` is the frame-local source;
  render/map/impostor/terrain/material are projections.
- **Single-writer resources.** The sole writer is named in the doc comment.
  Don't add a second; route through an accessor.
- **Autoflight is two layers** (ADR-20260731T232619Z). Strategic
  `FlightProgram` vs tactical channels `thalos_control::arbitrate` resolves.
  A new automation source declares `required_locks()` — never a lock table
  keyed on a mode enum. Panels emit `AutoflightRequest`.
- **One height authority.** `BodySurfaceRegistry` builds one
  `Arc<dyn SurfaceQuery>` per body. Sea level is the constant **0 m**.
- **One view anchor.** `ViewAnchor` answers "where is the view?" for every
  view-dependent detail system. Never the player craft or a mode flag.
- **One sim-pause boundary.** `sim_clock::SimClock` folds pause into a zero
  sim delta. Presentation uses `Time<Real>`.
- Real-space rendering lives under BigSpace; the active ship camera owns
  `FloatingOrigin`. Map systems read `MapSnapshot`.
- Semantic input comes from `thalos_input`; raw Bevy input only for cursor,
  picking, and UI internals.
- Systems run in `SimStage` order **Physics → Sync → Camera**; input intent
  collection runs in `PreUpdate` before them.

**Traps**

- **Every spawn scenario starts paused** (warp 0×).
  `spawn::apply_initial_warp` is the source of truth (`THALOS_AUTO_RUN=1`
  resumes). Deferred placement must not reset warp.
- **`EditorPart`** is the only thing separating the build world from the
  flying craft. Flight aggregations filter `Without<EditorPart>`.
- **Tile-cache staleness is silent.** New synthesis inputs go in the cache
  namespace; bump `thalos_terrain::GENERATOR_VERSION` when output changes.
  `THALOS_TILE_CACHE=0` disables the disk tier.
- **Per-body failures stay per-body.** Capture refuses a shot whose *target*
  body is degraded (INC-20260724T182643Z).
- **A detached process inherits nothing** from the caller's console/pipes
  (INC-20260724T185500Z).
- **No planet/terrain generation tests.** `just map` + `just screenshot`.
  `crates/domain/terrain` is the source of truth.

## Shared shader library

Surface materials import from the shared libraries — never re-implement
lighting/palette locally. WGSL/naga pitfalls:
`.claude/skills/wgsl-bevy/SKILL.md`.

| Import | Key exports |
|--------|-------------|
| `thalos::lighting` | `shade_surface`, `shade_foliage`, `compute_surface_sky`, `moonlight_radiance`, `object_aerial_recession`, `sun_daylight` |
| `thalos::shadow` | `ShadowCascadeBlock`, `sun_shadow_factor` |
| `thalos::landcover` | `vegetation_color`, `vegetation_understory_color`, `forest_coverage` |
| `thalos::foliage` | foliage albedo (near mesh + impostor bake) |
| `thalos::grass_displace` | `grass_blade_world_pos` |

## Bevy 0.19

Full notes: `docs/development/bevy.md`. Two ordering rules — both cost
runtime regressions:

- **Any post pass that calls `post_process_write()`** must sit in the exact
  chain slot its old render-graph node held (set membership *and* a relative
  `.after()`). A mis-slotted flip makes the presented buffer alternate →
  global flicker.
- **Retained binned render phases:** custom queue after Bevy's
  (`.after(RenderSystems::QueueMeshes).before(RenderSystems::PhaseSort)`),
  or a per-frame material mutation dequeues it and it never draws.

## Learned User Preferences

- Primary machine is a Mac; FPS and battery/power draw are first-class. This
  checkout also runs on Windows — the handle-inheritance trap still applies.
- 3D render scale must leave UI at OS HiDPI; never shrink the HUD with the world.
- Quality/Laptop profiles must not change window mode; default borderless fullscreen.
- Laptop quality keeps clouds and woody foliage; levers are 0.50× 3D scale and a 30 Hz cap. Capture stays on Showcase.

## Learned Workspace Facts

- Render scale is 3D-only: draw the main 3D target at a fraction and upscale; UI on a full-resolution overlay camera at OS HiDPI. `scale_factor_override` is the wrong lever. Overlay camera must clear to transparent and alpha-blend (Metal leaves an uncleared 2D target magenta). Atmosphere/sky must sample that scaled 3D depth, or Laptop 0.50× paints a wall on the geometric horizon.
- Quality presets live in `thalos_preferences`. First macOS run defaults to Laptop; `THALOS_QUALITY` / `just … quality=` pins one session without writing back; capture ignores the pin.
- `thalos_clouds` owns the shared compute marcher and weather cubes; `thalos_body_render` keeps the planetary BodySky composite. Kòrsou consumes the same marcher through a local Earth-shell adapter — do not fork a second cloud system.
- Kòrsou waterline follows mapped OSM + Sentinel-2 polylines (`coast_polylines.bin`); do not reconstruct the shore from the SDF or invent beach/cliff profiles from the 30 m DEM.
