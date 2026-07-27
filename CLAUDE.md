# CLAUDE.md

Operating manual for agents working in this repository. It carries only what an
agent needs **in context every session**: current direction, how to verify work,
and the invariants that are expensive to rediscover. Everything else lives in
`docs/` — **`docs/README.md` is the canonical documentation map**, and
`docs/architecture.md` holds the crate/module anatomy this file used to inline.

## Thalos

A planetary exploration / orbital-mechanics sandbox in Rust (edition 2024, Bevy
0.19, glam 0.32), in **pre-alpha**. Architecture and tooling are still being
shaped — you are encouraged to tear down infrastructure you find lacking and
replace it with something better. You don't need permission; you do need to
leave a trail: say what you replaced and why, and update this file plus the
relevant `docs/` spec in the same change. No silent rewrites.

## Current focus

**Keystone sprint (primary; 2026-07-23, ADR-20260723T142945Z)** — *make Thalos
look good*, through two paired efforts designed for each other:

- **Neural terrain** — terrain-diffusion-style hierarchical diffusion, fine-tuned
  offline behind the terrain-package boundary (ADR-20260723T143155Z);
  Thalos/earth-like first, airless second.
- **A renderer on Bevy's standard path** — terrain and every opaque surface as
  ordinary `Mesh` + `StandardMaterial`/`ExtendedMaterial` under Bevy
  lighting/shadows. One lighting universe is reached by moving terrain onto
  Bevy's path (where crafts already live), *not* by porting crafts onto the
  custom spine. Volumetrics/sky — BodySky atmosphere, clouds, analytic ocean,
  celestial sky — remain custom composites; that is the explicit carve-out.

The vehicle is a standalone clean-room probe repo (gates M0–M5) whose M5
extraction brings results back here. Strategy: `docs/roadmap/neural_terrain_renderer.md`
(`ntr §N`). Consequences in this repo: **the extracted tile renderer
(`thalos_body_render::tiles`) is now the default ground**, and **`thalos_udlod`
and the terrain WGSL stack are legacy / end-of-life** (defect-driven fixes only,
see *Ground renderer* below); the spine-port graphics items
(F4r/F5r/F7/F8/F9/W12r/TM1) are **frozen**; the composites (clouds, atmosphere,
ocean, plumes, celestial sky, capture harness) continue; MIRA-1 pauses after its
L2 gate.

**Background sprints** — architecture & code quality
(`docs/roadmap/architecture_cleanup.md`, `clean §N`) and the surviving
graphics-fidelity composites (`docs/roadmap/graphics_fidelity.md`, `gfx §N`,
whose **one-world principle** still holds: every surface obeys the same light,
shadows, occlusion, and air).

## Steering & memory

**Default: do the work.** The records below exist to keep one coherent roadmap
and to stop us re-learning expensive lessons — not to gate changes. Writing is
the exception; when you're unsure whether something earns a record, it doesn't.
(ADR-20260724T223339Z; conventions in `docs/README.md`.)

- **`docs/backlog.md`** — the queue, and the answer to "what's next?". Statuses
  `next` / `wip` / **`verify`** (landed compile-clean, awaiting runtime or
  screenshot verification) / `blocked` / `done` / `later`. It tracks work that
  **outlives the current session**: multi-step items, anything handed to the user,
  anything deliberately deferred. Something you find *and finish* in the same
  change needs no row — the commit is the record. The backlog is the **only**
  status authority; plan docs hold rationale and sequencing, never a parallel
  checkbox to keep in sync.
- **`steer` skill** — "what's next?" → propose the top item, then stop for a go;
  "add X / fix Y" → just do it (a row only if it won't finish now); vision talk →
  update the plan doc, then decompose into rows.
- **`docs/adr/`** — one record per decision that is **expensive to reverse** and
  would otherwise be re-litigated or re-explored: architecture seams, rejected
  approaches a future agent will be tempted to retry, constraints that look
  arbitrary from the code. Ordinary judgment calls belong in the commit message.
  Before reopening a settled area, `rg '<topic>' docs/adr` — don't read the
  directory. Immutable once accepted; supersede, don't rewrite.
- **`docs/incidents/`** — bugs whose diagnosis was **non-obvious**: the symptom,
  the mechanism, and the tell that identifies a recurrence. Short is correct.
  Written in the same change as the fix.
- **Identifiers are chronological, never sequential**:
  `<KIND>-<YYYYMMDDTHHMMSSZ>-<kebab-slug>` from `date -u '+%Y%m%dT%H%M%SZ'`.
  Never allocate "the next number" (ADR-20260722T170714Z).

## Standing quality bar

- **One canonical path per operation.** Spawning/placing/teleporting a craft,
  opening/closing a game mode, placing a structure — each gets one shared core
  every entry point routes through. A new entry point *parameterizes* the core;
  it does not fork it.
- **N by default.** No new system assumes "the one craft / runway / base". A kept
  single-instance assumption is behind an accessor and recorded in the plan doc.
- **Finish in-flight unifications first** — `GameContext` (`docs/gameplay/ui_flow.md`)
  and `CraftRegime` (`docs/simulation/regimes.md`) exist to kill state sprawl;
  completing them outranks new features.
- **Delete dead code on contact** — the superseded path you touch goes in the
  same change, not left "for reference".

## Bug fixing

Diagnose before patching:

1. **Reason to a hypothesis set** from the symptom — don't jump to the first
   plausible fix.
2. **Rule candidates out with targeted, falsifiable tests** that distinguish
   between hypotheses rather than confirming a guess.
3. **Fix the cause, not the symptom** — structurally, removing the class of bug.

A change that makes the symptom disappear without an explanation of *why* is not
a fix. Search `docs/incidents/` for a prior (`rg '<symptom>' docs/incidents/`)
before re-deriving a diagnosis. If the diagnosis was non-obvious — a wrong-looking
first hypothesis, a mechanism nobody would guess from the symptom — write the
post-mortem in the same change; a typo-grade fix needs none.

## Commands

```bash
just game [mode]        # USER-RUN ONLY (see below). modes: menu (default) orbit
                        #   polar eva landing final cruise runway runway-approach
                        #   shipyard hub mira mira-eva
just check [package]    # fast type-check (default thalos_game)
just build              # cargo build --workspace
just clippy             # cargo clippy --workspace
just test               # cargo test -p thalos_physics_canonical
cargo test -p thalos_physics_canonical -- test_name

just screenshot <name>  # headless still → artifacts/visual/latest/<name>.png.
                        #   <name> = a scripted preset or a saved viewpoint slug
                        #   ('latest' = newest). An unknown name prints the list.
just capture <name>...  # batch several scenes through one host invocation
just compare <preset> <axis>   # N-way matrix → artifacts/visual/runs/comparisons/
just screenshot-cold / just compare-cold   # clean-process isolated evidence
just capture-status / just capture-stop    # persistent capture host
just build-reset        # the ONE supported full artifact reset

just map                # whole-planet biome map + per-biome stats → target/
just preview            # headless procedural-object gallery → artifacts/visual/latest/
just ui-preview         # headless UI kitchen sink → artifacts/visual/latest/
just bake Mira          # rebuild assets/terrain_packages/Mira.bin offline
just validate-bake Mira # validate package schema/index/checksums/payload
just texgen             # rebuild versioned vegetation atlases offline
just trace              # profile-tracy build (needs Tracy GUI v0.11.x running)
just release <kind>     # bump version, commit, tag, push
```

`preview-window` / `ui-preview-window` are interactive variants — user-run.

The `justfile` is not a catalog of every useful command: add a recipe only for
stable human-facing entry points or multi-step orchestration agents get wrong in
bare shell. One-liners belong in this file or a `docs/` page as described shell.

## Verification: what you may and may not run

**Do not launch the game yourself.** Building and type-checking are encouraged;
*running* `just game` is the user's job — they have the display, the input
devices, and the judgment for "does it feel right". There is no remote-inspection
channel; don't try to drive or observe a live session programmatically. Ask the
user to run a mode and send a screenshot when you need interactive behaviour, a
specific in-flight moment, or temporal feel.

**Headless capture is yours to run, and you are expected to use it.**
`just screenshot` / `just capture` / `just compare` / `just preview` /
`just ui-preview` / `just map` are agent-runnable, render on the real GPU
off-screen, and write PNGs you **Read directly**. Framing knobs without a
recompile: `THALOS_SCREENSHOT_{AZIMUTH,ELEVATION,DISTANCE,SIZE,OUT,WARMUP,HUD}`.
`artifacts/visual/latest/` is the curated one-image-per-preset surface;
numbered experiments and alternate framings go to `artifacts/visual/runs/` via
`THALOS_SCREENSHOT_OUT`.

**A terrain / biome / scatter / visual change that compiles but hasn't been
screenshotted is `verify`, not `done`.** If no preset frames what you changed,
add a deterministic `ScreenshotPreset` — do not approximate it with manual camera
placement.

**A dirty worktree is not your problem — keep implementing.** Several agents may
be editing the workspace at once, so a build or capture can fail on errors in
code you never touched. Don't chase them, don't "fix" another agent's half-landed
change, and don't stall waiting for a green build. Confirm the breakage is
outside your own edits, finish your implementation, and hand verification to the
user — they run it once the worktree is clean and come back with feedback. Say
plainly in your report that verification was blocked and by what.

**Confirm you see what the user sees — and agree on scope — before you fix it.**
A visual complaint in words ("the terrain looks washed out", "the trees pop")
under-determines both the defect and how much should change; agents routinely fix
a different artifact than the one that was reported, or rewrite a subsystem when a
constant was meant. So, for any user-reported visual issue: capture the scene
first, then say back in plain language — no jargon, no code — **what you see in
the image**, **what you believe the problem is**, and **what you intend to touch
versus leave alone**. If the capture doesn't show the symptom, say that rather
than inventing a mechanism that explains it. Then stop for a yes. This is the one
place where asking beats doing: a wrong read costs a whole edit-capture cycle,
and the user has the screenshot open. Once confirmed, run the loop below without
further check-ins.

**The comparison loop** (full detail: `docs/development/visual_testing.md`):

1. State the visible symptom or goal and the plausible causes *before* editing.
2. Reproduce with an existing preset (or add one).
3. Choose **one typed axis** that separates the candidates. Never smuggle several
   setting changes into one variant.
4. Run the matrix, e.g. `just compare spaceport-aerial ssao`.
5. Inspect in order: process stderr → `manifest.json` → `contact_sheet.png` →
   full-resolution captures → wipes/diffs. Pixel metrics are evidence, not a
   verdict.
6. **Reject the whole comparison** if any variant logged a shader/pipeline error
   or is missing a render layer — known gap **BL-20**: Bevy can log a fatal
   pipeline validation error while the process still exits zero, so the existence
   of PNGs is not proof of a valid run.
7. Pin the root cause before patching, then rerun the exact preset/axis and keep
   the matched before/after evidence (the canonical comparison dir is overwritten
   on rerun — copy it first when revision-to-revision proof is needed).

Do **not** build a live multi-camera / split-screen comparison renderer, and do
not call two ad-hoc manual screenshots an A/B (ADR-20260721T192218Z): a split
viewport changes LOD, SSAO, shadows, and antialiasing under test. `just compare`
is not a performance benchmark — use profiling for that.

**Getting runtime numbers out: write them to a file.** Have the game log the
data and read the file afterwards. JSONL is the house style, under
`artifacts/diagnostics/` (never beside images). **The console is for humans:**
keep `info!` to short lifecycle/status messages and `warn!`/`error!` actionable.
Periodic gauges, numeric state dumps, calibration signals, and investigation
traces use structured fields on
`info!(target: "thalos::diagnostic::<subsystem>", event = "…", …)`;
the runtime routes them to `artifacts/diagnostics/runtime.jsonl` and omits them
from stdout/stderr. Do not invent another console diagnostic target. Existing
specialized recorders may keep their own JSONL when they have a distinct schema.
For performance, ask the user to run
`cargo run --release -p thalos_game --features profile-chrome`, then analyze the
artifact with `python3 scripts/analyze_trace.py trace-<date>.json`
(`--around-name <span> --window-ms <ms>` scopes it to windows around an event).

**Player→agent handoff:** in any 3-D view the user can press **F9** to save the
current view in one keypress (F9, Enter takes the suggested name; F9, type,
Enter overrides it) or **F8** to manage saved viewpoints in
`assets/viewpoints.json`; replay one headlessly with `just screenshot <slug>`
(`latest` = newest entry). ADR-20260724T211627Z.

## Build & iteration

`rust-toolchain.toml` pins Rust 1.97.0 with `clippy` + `rustfmt`, on the default
LLVM backend. Full policy and machine-specific recipes:
`docs/development/build_speed.md`. Load-bearing rules:

- **One Cargo command at a time** against the workspace `target/`. Parallel
  agents need separate worktrees (and therefore separate target dirs); size them
  with `scripts/setup-build-env.{sh,ps1}`. A worktree created *outside* the
  checkout needs `--all-worktrees` provisioning or it builds with the stock
  linker and no job budget.
- **The screenshot loop has exactly two speeds, by design** (ADR-20260724T153619Z):
  WGSL edits **hot-reload** in the running capture host (~3 s to a fresh PNG —
  keep visual iteration WGSL-first); any **Rust/manifest edit restarts the host
  automatically** on the next `just screenshot` (~1.5–2.5 min warm). There is no
  in-process Rust reload. `just capture-stop` is optional hygiene.
- **Never hand-roll `cargo clean -p <subset>`.** The dev lane links Bevy
  dynamically, so `bevy_dylib` and everything linked against it are *one artifact
  set*; a partial clean yields `undefined symbol: anon.*.llvm.*`
  (INC-20260724T182642Z) — an artifact problem, never a missing dependency. The
  capture client self-heals once; if its retry fails, run `just build-reset`.
- **Keep every dev renderer lane on the one `dev-renderer` fingerprint.** Any
  other Bevy/wgpu feature mixture forces another full graph rebuild. wgpu's
  `counters` stays opt-in behind `thalos_game/gpu-counters`.
- **No unstable `-Zthreads`** (INC-0006: parallel-MIR ICE poisoned incremental
  objects). No compiler cache — sccache was removed (ADR-20260723T222214Z).
- Expect the first build after changing flags, profiles, or Bevy/wgpu features to
  take minutes. That is a one-time graph rebuild; don't churn those inputs.

## Codebase map

Pure-Rust libraries (no Bevy) — `thalos_world` (authored body/system truth),
`thalos_physics_canonical` (orbital mechanics, aero, surface-local frame),
`thalos_control` (fly-by-wire), `thalos_terrain` (`SurfaceQuery` + generation),
`thalos_celestial` (sky model), `thalos_texgen` (offline textures).

Bevy consumers — `thalos_runtime` (`crates/runtime/game`, the sole app
composition: gameplay, rendering integration, UI, scenarios, capture presets),
`thalos_game` (`apps/game`, thin launcher), `thalos_body_render` +
`thalos_body_shading` (celestial-body rendering; owns both the default
`tiles` ground renderer and the legacy `thalos_udlod` one it is replacing —
sole consumer of that crate), `thalos_physics_local` (Avian boundary),
`thalos_shipyard` (construction model), `thalos_input`, `thalos_ui`,
`thalos_capture_{protocol,runtime}`, vendored `big_space`.

Systems run in `SimStage` order **Physics → Sync → Camera**; input intent
collection runs in `PreUpdate` before them.

**Module-level anatomy — what each crate owns, its submodules, and the data flow
— is in `docs/architecture.md`.** Subsystem behaviour is owned by the specs in
`docs/gameplay/`, `docs/simulation/`, `docs/world/`, `docs/rendering/`.

## Invariants

**Crate boundaries**

- **The pure crates have no Bevy**, even transitively: `thalos_world`,
  `thalos_physics_canonical`, `thalos_terrain`, `thalos_celestial`. CI-guarded
  with a `cargo tree` check. (`bevy_erosion_filter` is allowed only via
  `default-features = false`, which pulls no Bevy engine crate.)
- Avian lives behind `thalos_physics_local`; never add it to
  `physics_canonical`. Don't derive `Reflect` there either — mirror canonical
  state into a Bevy resource at the bridge (`CraftStateMirror`).
- `thalos_body_render` is the **sole** consumer of `thalos_udlod`, so replacing
  the ground backend stays localized.

**Ground renderer: tiles by default, udlod is legacy**

- **`thalos_body_render::tiles` is the ground renderer** — terrain as ordinary
  `Mesh` + `StandardMaterial` on Bevy's standard path, driven per frame from
  `ViewAnchor` by `rendering::tile_terrain`. New terrain, shading, scatter, or
  LOD work goes here.
- **`thalos_udlod` + `body_render::ground`'s terrain half + the terrain WGSL
  stack (`body_terrain.wgsl`, udlod's shaders) are legacy**: defect-driven
  fixes only, deleted once the remaining `ntr §6` rows close. They still stream
  bodies the tile driver has not installed on (it takes one body per session
  today), which is the only reason the path is still wired.
- **`THALOS_TILE_RENDERER=0` is an A/B baseline, not a supported mode** — it
  forces the whole process back onto legacy udlod, which is what the `renderer`
  compare axis drives. The gate is a boot `OnceLock`, so it always needs a cold
  run / host restart. Anything else (including unset) is the tile path.
- The analytic composites that live in `body_render::ground` — `BodySky`,
  `BodyOcean`, the impostor handoff — are **not** legacy; they are the
  ADR-20260723T142945Z carve-out and outlive udlod.

**One authority per concern**

- **One propagator everywhere.** Live stepping and prediction both route through
  the same `ShipPropagator`, or "where the ship is" diverges from "where it will
  be". Body positions are always queried through `BodyTrajectoryProvider`.
- **One craft state, one authority.** Presentation reads snapshots or accessors,
  never parallel transform-owned state.
- **One solar-system state.** `SolarSystemState` is the frame-local source for
  body + environment state; render, map, impostor, terrain, and material systems
  are projections of it, not co-owners.
- **Single-writer resources.** The sole writer is named in the resource's doc
  comment: `SolarSystemState` ← `sync_solar_system_state`, `MapSnapshot` ←
  `update_map_snapshot`, `CraftStateMirror` ← `refresh_craft_state_mirror`,
  `CraftRegimeState` ← `regime::resolve_regime`, `AvianAuthority` ←
  `compute_avian_authority`, `ViewAnchor` ← `update_view_anchor`. Don't add a
  second writer; route through an accessor or reconsider the ownership.
- **One height authority.** `BodySurfaceRegistry` builds one
  `Arc<dyn SurfaceQuery>` per body, and every consumer — ground LOD render, the
  near-surface `HeightSourceRegistry` (colliders, camera floor, HUD altitude,
  EVA), propagator collision — reads that same selected surface. Thalos has **no
  sea-level layer**: sea level is the constant **0 m**.
- **One view anchor.** `ViewAnchor` is the per-frame answer to "where is the
  view?" for every view-dependent detail system (scatter, shadow-cascade centre).
  Never anchor detail to the player craft or a mode flag — new camera modes must
  get correct detail with zero per-mode plumbing.
- **One sim-pause boundary.** `sim_clock::SimClock` folds every pause source into
  a zero sim delta. Do not pause `Time<Virtual>`/`Res<Time>` as the game-wide sim
  pause; presentation and UI animation use `Time<Real>`.
- **One shader library per concern** (below).
- Real-space rendering lives under BigSpace; the active ship camera owns
  `FloatingOrigin`. Map systems read `MapSnapshot` and never touch real-space
  entities.
- Semantic player input comes from `thalos_input` intent resources; raw Bevy
  input only for cursor positions, picking, and UI internals.

**Traps**

- **Every spawn scenario starts paused** (warp 0×). `spawn::apply_initial_warp`
  is the single source of truth, gated by `spawn::AutoRun` (`THALOS_AUTO_RUN=1`
  resumes and skips the start screen); deferred placement flows must not reset
  warp themselves.
- **`EditorPart` is the only thing separating the build world from the flying
  craft** in the same ECS `World`. A system aggregating part components for the
  *flight* craft must filter `Without<EditorPart>`.
- **Tile-cache staleness is silent.** If you add an input to tile synthesis, fold
  it into the cache namespace, and bump `thalos_terrain::GENERATOR_VERSION` when
  generation output changes — otherwise a cached run renders old terrain.
  `THALOS_TILE_CACHE=0` disables the disk tier while iterating.
- **Per-body failures stay per-body.** A missing/stale terrain package degrades
  *that body* (`BodySurfaceRegistry::degraded`) instead of panicking; integrity is
  enforced at the request boundary, where capture refuses a shot whose *target*
  body is degraded (INC-20260724T182643Z). Apply the same shape to new
  body-scoped resources.
- **A detached process must inherit nothing from the caller's console/pipes.** On
  Windows, `CreateProcess` inherits every inheritable handle; launch via
  `Start-Process` with redirection in a shim (INC-20260724T185500Z).
- **No planet/terrain generation tests.** Generation is in an iterative phase;
  these tests slow the visual loop. Verify with `just map` + `just screenshot`
  instead. `crates/domain/terrain` and its `ProceduralSurface` generator are the
  source of truth — there is no separate Terrain Lab.

## Shared shader library rule

Every surface material (`body_terrain.wgsl`, `tree.wgsl`, `grass.wgsl`,
`rock.wgsl`, `ground_patch.wgsl`, `tree_impostor.wgsl`) **must** import from the
shared libraries, never re-implement lighting/palette locally. When a palette or
BRDF constant moves, it moves in one place.

| Import path | Key exports |
|---|---|
| `thalos::lighting` | `shade_surface`, `shade_foliage`, `compute_surface_sky`, `moonlight_radiance`, `object_aerial_recession`, `sun_daylight` |
| `thalos::shadow` | `ShadowCascadeBlock`, `sun_shadow_factor` |
| `thalos::landcover` | `vegetation_color`, `vegetation_understory_color` (near-field: canopy darkening reduced to a residual), `forest_coverage` (CPU mirror: `ground/landcover.rs`) |
| `thalos::foliage` | foliage albedo model (near mesh + impostor bake) |
| `thalos::grass_displace` | `grass_blade_world_pos` |

The `wgsl-bevy` skill (`.claude/skills/wgsl-bevy/SKILL.md`) collects WGSL/naga
pitfalls — reserved words, strict type rules, `naga_oil` import quirks. Treat it
as a living document: add any WGSL error worth remembering.

## Bevy 0.19

Full notes: `docs/development/bevy.md`. Two ordering rules are non-negotiable —
both cost us runtime regressions:

- **Any post pass that calls `post_process_write()` must sit in the exact chain
  slot its old render-graph node held** (set membership *and* a relative
  `.after()`). `ViewTarget`'s ping-pong parity index persists across frames, so a
  mis-slotted flip makes the presented buffer alternate → global flicker.
- **Retained binned render phases**: a custom queue system must run *after*
  Bevy's (`.after(RenderSystems::QueueMeshes).before(RenderSystems::PhaseSort)`),
  or a per-frame material mutation dequeues it after it adds itself and it never
  draws.
