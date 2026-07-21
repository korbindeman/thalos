# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Thalos

Thalos is in **pre-alpha development**. Architecture and tooling are
still being shaped — agents are encouraged to tear down infrastructure
they find lacking and replace it with something better as the project
matures. **Do this explicitly**: announce the change before you make
it, explain why the existing approach falls short, and update this
file (and the relevant spec under `docs/`) so the next agent inherits
the new shape, not the old one. No silent rewrites.

## Current focus: architecture & code quality

The active sprint is a **consolidation pass**: the feature push (surface bases,
space-center hub, launch flows, GPU grass, shadow unification) left behind
sloppily hacked seams that now generate a steady stream of bugs — parallel
mechanisms for the same job, copy-pasted flows that drift apart, and
single-craft / single-site assumptions baked into what should be N-ary systems.
The goal is to go over every feature area and give it a **proper, unified
system**: one canonical way to do each thing, DRY across similar features, and
natural support for any N instances. The plan lives in
`docs/architecture_cleanup.md`. Guiding rules for this sprint:

- **One canonical path per operation.** Spawning/placing/teleporting a craft,
  opening/closing a game mode, placing a structure on terrain — each gets a
  single shared core that every entry point routes through. No parallel
  near-copies; if a new entry point needs a variation, it parameterizes the
  core, it does not fork it.
- **N by default.** New and reworked systems must not assume "the one craft",
  "the one runway", "the one base". Where a hard single-instance assumption is
  kept for now (e.g. `Simulation`'s one canonical craft), it is kept knowingly,
  behind an accessor, and recorded in the plan doc — not implied by a bare
  resource.
- **Finish the in-flight unifications before adding parallel machinery.** The
  `GameContext` sub-state (docs/ui_flow.md) and `CraftRegime`
  (docs/regimes.md) migrations exist precisely to kill boolean/state sprawl —
  new modes and regime consumers go through them, and completing their
  remaining phases outranks new features.
- **Delete dead code on contact.** The removed bake pipeline's remains and any
  superseded path you touch get deleted in the same change, not left "for
  reference".

## Secondary focus: graphics fidelity

The active sprint is pushing toward MSFS/KSP2-tier visuals. The full plan is in
`docs/graphics_fidelity.md` (restructured 2026-06-30 around a full architecture
review). The doc is now organised around two ideas, not a flat task list:

- **The one-world principle** — Thalos is one physical world; *every* surface
  (terrain, terrain detail, vegetation, rocks, **crafts, gear, buildings, pads,
  tanks, runway**, water) must obey the same light, cast into and receive from the
  same shadows, occlude each other, and recede into the same air. A surface that
  opts out reads as a pasted-on cut-out. See `docs/graphics_fidelity.md` §2.3.
- **The unification foundation** — the central debt is **two lighting universes**:
  terrain/vegetation/water/rock/impostors shade through the shared
  `thalos::lighting` spine (`shade_surface`/`ThalosSurface`), but **crafts and
  structures still use Bevy stock PBR**, reconciled only by a CPU day/night scalar
  in `rendering/lighting.rs`. The foundation (doc §3, steps **F1–F9**) collapses
  the craft/structure path into projections of the spine: one terminator, one
  heliocentric flux, one exposure, one atmosphere-derived environment (sky-view
  LUT → SH ambient → prefiltered IBL), one screen-space AO field, one shadow rig
  that everything casts into.

Status of in-flight work (now tracked by substrate in the doc's §4):

- **Shadows** ☑ — **one shadow world landed 2026-07-02** (F6, compile-clean,
  awaiting screenshots): everything casts into + receives the `thalos::shadow`
  rig — the `StandardMaterial` universe (structures/runway/plain parts/EVA) via
  the new `ShadowedStandardMaterial`, hull via `ship_part.wgsl`; **stock Bevy
  CSM on the sun is disabled**; the analytic terrain craft-shadow proxy is
  deleted; stable CSM (receiver normal-offset + slope-scaled bias); a craft-local
  single-cascade mode keeps hull self-shadow in orbit; W12 v1 (CPU horizon march
  dims the sun on objects parked behind terrain). *Next:* screenshot tuning,
  per-fragment horizon term for spine materials, PCSS (W18), cloud shadows (W2).
  See `docs/shadow_unification_prompt.md` for the full status block.
- **Landcover + palette / aerial recession / moonlight** ◐ — in `thalos::lighting`
  / `thalos::landcover`; awaiting screenshots. Moonlight converges into F1 (the
  two moon models become one); aerial recession folds inside `shade_surface`.
- **Lighting-input unification (F1)** ✅ and **exposure (F2)** ☑ — F2 retired the
  Bevy `AutoExposure` histogram + its vacuum/surface preset blend, so
  `CameraExposure`'s input distance-gain is the sole brightness authority (plus a
  fixed `color_grading.exposure` baseline, `GLOBAL_EXPOSURE_STOPS` in
  `post_stack.rs`); compile-landed, awaiting a noon screenshot to calibrate.
  **sky-view-LUT→IBL (F3/F4), AO (F5)** ◐ and **shadow-rig unification (F6)** ☑
  (landed 2026-07-02, see the Shadows bullet) — the rest of this sprint's
  foundation.
- **Terrain tiling-material detail, hull/structure spine port, LOD chain** ☐ — next.

### Shared shader library rule

Every surface material (`body_terrain.wgsl`, `tree.wgsl`, `grass.wgsl`,
`rock.wgsl`, `ground_patch.wgsl`, `tree_impostor.wgsl`) **must** import from
the shared libraries, never re-implement lighting/palette locally:

| Import path | Key exports |
|---|---|
| `thalos::lighting` | `shade_surface`, `shade_foliage`, `compute_surface_sky`, `moonlight_radiance`, `object_aerial_recession`, `sun_daylight` |
| `thalos::shadow` | `ShadowCascadeBlock`, `sun_shadow_factor` |
| `thalos::landcover` | `vegetation_color`, `forest_coverage` (CPU mirror: `ground/landcover.rs`) |
| `thalos::foliage` | foliage albedo model (near mesh + impostor bake) |
| `thalos::grass_displace` | `grass_blade_world_pos` |

When a palette or BRDF constant moves, it moves in one place.

## Steering, decisions & incidents

The project self-steers through an execution/strategy split plus durable
git-committed institutional memory (ADR-0001; full doc map + cross-ref
convention in `docs/README.md`):

- **`docs/backlog.md`** — the operational queue: status-tracked, stably-ID'd
  items across the two sprints above. Statuses: `next` / `wip` / **`verify`**
  (landed compile-clean, awaiting runtime/screenshot verification — the
  thalos-specific state, since agents can't run the game) / `blocked` / `done` /
  `later`. Keep it honest: flip a row, the plan doc's checkbox, and the spec doc
  **in the same change**; work discovered mid-task becomes a row, **never a
  silent TODO**.
- **`steer` skill** (`.claude/skills/steer/SKILL.md`) — the harness.
  **"what's next?"** → propose + scope the top item (bundling the `verify`
  queue into one verification session counts as an item), then stop for a
  go-ahead; **"add X / fix Y"** → file it in the backlog, then do it; **vision
  talk** → capture to the plan docs (ADR if a fork resolves), then decompose
  into backlog rows. Invoke it (or follow its procedure) for any of those three.
- **`docs/adr/`** — the decision log: *why* things are the way they are,
  including rejected alternatives. **Read the index before making or reopening
  a non-trivial design choice; write one at decision time** (choosing among
  alternatives, cutting/deferring scope, reversing an approach) — the reasoning
  is otherwise lost to context compaction. Immutable once accepted; supersede,
  don't rewrite.
- **`docs/incidents/`** — post-mortems for fixed non-obvious bugs (`INC-NNNN`):
  evidence, hypothesis differential, root cause, prevention, recurrence tells.
  Written in the same change as the fix (see "Bug fixing" below).

## Commands

```bash
just game                 # cargo run -p thalos_game — boots to the start screen
                          #   (scenario picker / shipyard / settings; naming a
                          #    mode below skips it, as does THALOS_AUTO_RUN=1)
just game orbit           # ship in low equatorial Thalos orbit, no start screen
just game polar           # ship in low polar (i≈90°) parking orbit
just game eva             # spawn on foot (EVA) on the Thalos surface instead
just game landing         # powered-descent approach over dry Thalos land
just game final           # low final approach over a flat dry Thalos patch
just game runway          # aircraft parked on the Thalos surface runway
just game runway-approach # aircraft on short final lined up with that runway
just game cruise          # Meridian at ~15,000 ft, level cruise over dry land
just game shipyard        # open straight into the in-game ship editor
                          #   (also: the pause menu's SHIPYARD button
                          #    from any running mode)
just game hub             # straight into the space-center hub over the
                          #   spaceport (the PLAY path minus the start
                          #   screen: base built, no craft placed)
just game mira            # ship in low Mira orbit (offline package terrain)
just game mira-eva        # spawn on foot on Mira
just bake Mira            # rebuild assets/terrain_packages/Mira.bin offline
just validate-bake Mira   # validate package schema/index/checksums/payload
just screenshot mira-orbit   # headless cratered-horizon verification
just screenshot mira-surface # headless close regolith/Hapke verification
just screenshot mira-eva     # canonical EVA-site horizon/LOD verification
just compare earth-reference atmosphere # isolated A/B + contact sheet/diff/manifest
just compare spaceport-aerial ssao       # off/on/raw multi-test of the AO path
just terrain-lab          # static slippy-map terrain sketchpad at localhost:8787/tools/terrain-lab/
just map                  # whole-planet biome map export → target/world_map.png
                          #   (true macro palette, hillshaded, web-mercator) +
                          #   target/world_biomes.png (flat MacroBiome class map)
                          #   + per-biome area stats on stdout. Headless — agents
                          #   Read the PNGs to iterate on biomes/landcover.
                          #   Knobs: WORLD_PROJ=equirect, WORLD_MODE=hypso,
                          #   WORLD_W, WORLD_SEED, WORLD_ZOOM, WORLD_TRANSECT
just preview              # headless procedural-object gallery → PNGs in
                          #   tools/preview/out/ (trees/shrub, grass, rocks/
                          #   pebbles, landcover preview). No window — agents
                          #   can run it and inspect
                          #   the images. See "Running and inspecting" below.
just preview-window       # interactive variant of `just preview`: a window with
                          #   an orbit camera (drag/scroll, ←/→ cycle, S = shot).
                          #   User-run (opens a window).
just ui-preview           # headless UI-kit kitchen sink → tools/ui_preview/
                          #   kitchen_sink.png (every thalos_ui token/widget
                          #   over a test scene). Agents iterate on the UI by
                          #   reading the PNG. See docs/ui.md.
just ui-preview-window    # interactive kitchen sink (hover/press/typing feel;
                          #   S = screenshot). User-run (opens a window).
just build                # cargo build --workspace
just test                 # cargo test -p thalos_physics_canonical
just clippy               # cargo clippy --workspace
just trace                # cargo run --release -p thalos_game --features profile-tracy
just release patch        # bump version, commit, tag, and push (patch|minor|major|x.y.z)

# Run a single test
cargo test -p thalos_physics_canonical -- test_name
```

The `justfile` is not a catalog of every useful command: add a recipe only for
stable human-facing entry points (`just game`, `just preview`) or multi-step
orchestration agents get wrong in bare shell. One-liners an agent can run
directly belong in this file or the relevant `docs/` README as described shell,
not as recipes.

## Toolchain

`rust-toolchain.toml` pins nightly only. There is no checked-in Cargo backend
override, so Cargo uses its default LLVM backend on every platform. Do not add
`rustc-codegen-cranelift-preview` or `codegen-backend = "cranelift"` to
project config unless the project intentionally re-adopts Cranelift for all
platforms.

Compiler/linker performance tuning that is platform-specific or finnicky
belongs in local Cargo config, not committed workspace config. This includes
incremental overrides, debug-info reductions, custom linkers, and backend
experiments. Bevy dynamic linking is the exception: it is cross-platform and not
finnicky, so every renderer dev entry point (`just game`, `just screenshot`,
`just compare`, `just preview`, `just ui-preview`) enables `bevy/dynamic_linking` by committed
default, while `just build`/`just trace`/release stay static. Override
`game_command` in `.env.just` to opt out for game/screenshot locally. The normal
Windows iteration path stays on LLVM. Do not add nightly `-Zthreads`: the
2026-07-20 parallel-MIR ICE poisoned incremental objects and turned a proposed
speedup into a full recovery build (INC-0006). The workspace-local
`.cargo/config.toml` and `.env.just` are ignored by Git for this purpose. The full policy plus
Windows fast-incremental and macOS workaround examples live in
`docs/tooling.md`.

**Fast iteration invariants:**

- Run only **one Cargo command at a time** against the workspace `target/`.
  Concurrent game/check/screenshot invocations mostly wait on Cargo's target
  lock while competing for CPU and memory; they make an already large Bevy link
  appear even slower. Use `cargo check-game` during edits, then perform one
  linked `just game` or `just screenshot` when needed.
- Keep every normal dev renderer on the same Bevy/wgpu feature fingerprint.
  Adding a feature to only one entry point forces Cargo to build another full
  `bevy_dylib` plus its Windows PDB. In particular, wgpu's diagnostic
  `counters` feature is intentionally opt-in as `thalos_game/gpu-counters`; do
  not enable it in the default dependency graph.
- Expect the first build after changing compiler flags, profiles, or Bevy/wgpu
  features to take several minutes. That is a one-time graph rebuild, not the
  steady-state iteration time. Avoid changing these inputs casually, because
  doing so invalidates most of the expensive renderer cache.

## Planet generation iteration

Planet generation is in an iterative development phase. Do not add or run
planet/terrain generation tests for now, including per-body generation
tests. This applies anywhere a test compiles or validates generated planet
data, even outside `thalos_terrain`; these tests slow down the visual
iteration loop.

For faster speculative design, use `just terrain-lab` and open
`http://127.0.0.1:8787/tools/terrain-lab/`. Terrain Lab is a dev-only static
browser sketchpad with Google-Maps-style panning/zooming and lazily generated
LOD tiles. It is for process-map exploration before porting good ideas into
`crates/terrain`; the runtime `ProceduralSurface` generator is the source of
truth for game output.

## Terrain generation (procedural bodies + Mira offline-package MVP)

`BodySurfaceRegistry` constructs one canonical `Arc<dyn SurfaceQuery>` per body.
Most bodies use `ProceduralSurface`, a pure analytic function of `(direction,
lod)` in f64 body-local coordinates. Mira is the first offline-package body:
`just bake Mira` writes `assets/terrain_packages/Mira.bin`, and startup validates
its content key before loading `BakedSurface`. There is no runtime/startup bake
check: a missing or stale package fails with the explicit bake command. Every
terrain-height consumer reads the same selected surface:

- **Ground LOD render** — a `FlattenedSurface`-wrapped canonical surface feeds
  the UDLOD tile provider.
- **Collider / camera terrain floor / runway / descent site / HUD altitude / EVA**
  — the near-surface `HeightSourceRegistry` (a GPU-atlas height mirror over the
  same surface, with a CPU fallback), via
  `HeightSource::sample_height_m`.
- **Propagator collision** — `GameTerrainRegistry` mirrors the canonical surface.

For the current Thalos generator there is **no sea-level layer**: the continent mask puts the shoreline at height
0 (the reference radius), so "sea level" is the constant **0 m** wherever a datum
is needed (dry-land checks, the camera's terrain floor). Do not read sea level
from a baked surface — that path is gone.

Mira's compatibility baker currently wraps the retained deterministic airless
compiler into a schema-v1 package with an explicit node/blob manifest,
32→512 adaptive residual height pyramid, ancestor fallback, checksums, exact
artifact identity, validator, and `PackageSurface`; player devices
never run the producer. The production
direction is an offline hierarchical-diffusion bakery that emits adaptive,
versioned cube-sphere terrain packages; see `docs/mira_airless_mvp.md` and
ADR-0008. Player devices stream packages through a `PackageSurface` backing and
reconstruct only bounded close-range detail. The authored package and the
disposable runtime tile cache are separate layers. Package-backed and procedural
bodies must still converge on the one `SurfaceQuery` consumer path and one height
authority.

The old runtime baked pipeline
— `thalos_bake_dump` (the "dumps"), `thalos_body_editor`, and the game's startup
`bake_check` — has been **deleted**. The remaining baked-pipeline modules in
`thalos_terrain` (the `Feature`/`Ocean` compiler, `PlanetSurface`/
`StaticSurfaceData`, `cache`, stages/tectonics/fields) are retained only as the
temporary offline Mira producer/global package payload. Do not reconnect them
to a runtime startup bake or revive the dump/editor flow. MIRA-0's adaptive
package tracer is complete; MIRA-1/2 replace the producer with diffusion. New bakery
work follows ADR-0008 and the package spec behind the same `SurfaceQuery` seam.
MIRA learned models are Rust-native and authored once with pinned Burn 0.21 per
ADR-0010. `thalos_terrain_learned` is Bevy-independent shared model/sampler code;
training checkpoints carry raw/EMA weights plus path-remapped Adam state so
cross-process resume remains deterministic. `thalos_terrain_train` is the
offline producer tool. Candle may be selected as a
Burn backend or used as a diffusion reference, but must not become a second
model implementation. Keep both learned crates out of `thalos_game` until a
measured optional runtime feature needs them; gameplay remains package-first.

## Running and inspecting the game

**Do not launch the game yourself.** Building and type-checking are fine and
encouraged (`just build`, `cargo check -p thalos_game`, `just clippy`), but
*running* the game is the user's job — they have the display, the input
devices, and the judgment for what "looks right". When you need to see visual
or runtime behaviour, ask the user to run `just game [mode]` and **send a
screenshot** (or describe what they see). There is no remote-inspection
channel: don't try to drive or observe a live session programmatically.

**Procedural objects, though, you *can* see yourself.** `just preview`
(`crates/body_render/examples/object_preview.rs`) is a **headless** renderer:
it draws each procedural object (trees, shrub, grass, and pebbles/rocks)
to a PNG under `tools/preview/out/` and exits — no window. It renders on the
real GPU off-screen (verified working from an agent shell: NVIDIA/Vulkan). Each
object is staged as a small **diorama** so it reads like the in-game surface,
not a floating cutout: a sky-model-lit ground (`GroundPatchMaterial` — the same
`thalos::lighting` BRDF the in-game terrain uses), a carpet of the real grass
blades around plants, a self-managed **sun-shadow** cascade (a trimmed copy of
`rendering::sun_shadow`, so trees cast leaf-shaped shadows on the ground and
themselves), and the game's camera post stack (AgX tonemap + bloom + SMAA),
minus the sensor-sim grain / chromatic aberration that only muddy small asset
shots. So when iterating on a procedural asset's *geometry/material*, run `just
preview` and **Read the output PNGs directly** instead of round-tripping a
screenshot through the user. Extend the gallery by adding an entry to
`object_preview.rs`; a larger composed scene (a patch of grass, a mountain
ringed by trees) is the planned next phase.

**Whole *game* scenes, too, you can now capture headlessly** — `just
screenshot` (`crate::screenshot::HeadlessScreenshotPlugin`, activated by the
`THALOS_SCREENSHOT` env var). This boots the **real game binary** with no window
and no winit (a `ScheduleRunnerPlugin` frame loop, same as `just preview`),
builds the world for a named **preset** (`spaceport-aerial`, which boots the
`runway` scenario so the whole spaceport + settled terrain + parked aircraft come
up behind the loading screen; `hub`, which boots the `just game hub` route —
the space-center view exactly as PLAY presents it, craft left in orbit — the
regression probe for view-anchored surface detail; and `dry-belt` (aliases
`dry` / `desert` / `biome`), which boots a plain orbit scenario then searches
the daylight hemisphere for the **driest sunlit dry-land site** and low-obliquely
surveys the surface there — the verification probe for **terrain-per-biome** work
(landcover palette, the tree/scatter biome gate): trees/shrubs should read
sparse-to-absent on tan desert, versus the green `spaceport-aerial` shot in the
equatorial wet belt), poses the *actual*
`ShipCamera` at a scripted god-view over the
focus — reusing the real camera keeps the scene-depth / atmosphere / SSAO /
sun-shadow render graph coupled — hides the HUD, renders one frame off-screen to
its stable preset filename under `tools/screenshots/`, and exits. This folder is
the curated latest-view surface: one image per canonical preset, overwritten on
the next capture. Numbered experiments, crops, and alternate framings belong in
`tools/agent_scratch/screenshots/` via `THALOS_SCREENSHOT_OUT`; `just compare`
uses that scratch tree automatically. So an agent can now see the composed
in-game world (lighting-in-context, terrain, base layout, shadows), not just
isolated assets — **Read the PNG directly**. Frame it without recompiling via
`THALOS_SCREENSHOT_{AZIMUTH,ELEVATION,DISTANCE,SIZE,OUT,WARMUP}` (angles in
degrees around the pad, 90° elevation = top-down), and
`THALOS_SCREENSHOT_HUD=1` keeps the flight HUD visible (it is hidden by
default) — the loop for iterating on HUD chrome. Add a preset by extending
`ScreenshotPreset`. Still the user's call: interactive "does it feel right",
live behaviour, and any framing that needs a specific in-flight moment — the tool
captures a static, scripted vantage of a fresh spawn, not a play session.

### Visual diagnosis and comparison workflow

**Visual work is headless-screenshot- and controlled-comparison-driven.** Use
the smallest canonical tool that answers the question:

- `just screenshot <preset>` — one in-context beauty frame: reproduce a visual
  symptom, establish a baseline, or verify a change when no competing renderer
  configuration needs comparison.
- `just compare <preset> <axis>` — A/B or N-way attribution: compare render
  paths, feature toggles, diagnostic fields, or tuning alternatives while one
  declared factor changes and every other capture input stays fixed.
- `just preview` — isolated procedural asset geometry/material only. It may
  supplement, but never replaces, the in-context game capture.
- `just game <mode>` — user-run only, for motion, interaction, temporal feel, or
  a specific play moment a scripted fresh-spawn capture cannot represent.

Do **not** build a live multi-camera/split-screen comparison renderer. Do not
take two ad-hoc manual screenshots and call them an A/B. ADR-0011 makes isolated,
sequential, full-resolution headless captures the one comparison path; a split
viewport changes LOD, SSAO, shadows, antialiasing, and other inputs under test.

**Required loop for every graphical bug or visual iteration:**

1. Before editing, state the visible symptom or visual goal and the plausible
   causes/alternatives. A comparison exists to distinguish hypotheses, not just
   to produce attractive images.
2. Reproduce it with an existing preset. If no preset frames the relevant
   surface, distance, time of day, or camera regime, add a deterministic
   `ScreenshotPreset`; do not approximate it with manual camera placement.
3. Choose one typed axis that separates the candidates. Initial axes are
   `atmosphere` (`custom`/`bevy`) and `ssao` (`off`/`on`/`raw`). Add a new axis in
   `crates/game/examples/visual_compare.rs` only when all variants are values of
   the **same factor** and use capture-only overrides that never persist user
   settings. Never smuggle several setting changes into one variant.
4. Run the matrix, for example:

   ```bash
   just compare earth-reference atmosphere   # orbital limb / space view
   just compare runway-atmosphere atmosphere # surface sky + long-path haze
   just compare spaceport-aerial ssao         # off / applied / raw AO
   ```

   `THALOS_SCREENSHOT_{SIZE,AZIMUTH,ELEVATION,DISTANCE,WARMUP,HUD}` overrides
   are allowed, but set them once for the whole run. The runner owns the output
   path and axis override so they cannot drift between variants.
5. Inspect `tools/agent_scratch/screenshots/comparisons/<preset>/<axis>/` in this order:
   **(a)** process output/stderr, **(b)** `manifest.json` for revision, dirty
   state, dimensions, and invariant inputs, **(c)** `contact_sheet.png`,
   **(d)** the full-resolution numbered captures, then **(e)** wipes and amplified
   diffs. Pixel metrics are evidence, not a verdict: temporal/stochastic noise
   can change many pixels without a meaningful regression.
6. Reject the entire comparison if any variant logged a shader/render-pipeline
   error or is missing a render layer. **Known gap BL-20:** Bevy can currently
   log a fatal pipeline validation error while the headless process still exits
   zero, so the existence of PNGs alone is not proof of a valid run.
7. Use the result to eliminate candidates. For bugs, do not patch until the root
   cause is pinned; for tuning, record which single change produced the desired
   effect. After editing, rerun the exact preset/axis and keep the matched
   before/after evidence. The canonical comparison directory is overwritten on
   rerun, so copy it to a clearly labelled evidence directory before editing
   whenever revision-to-revision proof is required.

`just compare` is not a performance benchmark: independent startup/warm-up and
artifact readback deliberately favor isolation over timing. Use the FPS/trace
profiling workflow for performance attribution. New debug channels (normals,
depth, LOD/tile IDs, shadow factor/cascade, material IDs, atmosphere
transmittance, lighting lobes) extend the typed comparison runner; they do not
get a parallel camera or orchestration path. Full details: `docs/visual_testing.md`.

**Terrain iteration/design is verified by screenshot — always.** Any change to
terrain generation, landcover/biomes, or surface scatter (trees/grass/rocks)
must be checked with a headless capture before it is called done; these tools
are agent-runnable, so **run them and Read the PNG yourself** rather than
round-tripping through the user:

- `just map` — whole-planet macro palette + `MacroBiome` class map + per-biome
  area stats (the design-level "does the planet read right" check).
- `just screenshot <preset>` — the in-context ground view. Pick the preset that
  frames what you changed: `dry-belt` for desert/biome/scatter work,
  `spaceport-aerial` / `hub` for the base and the wet-belt look, or
  `mira-orbit` / `mira-surface` for package/Hapke work, or `mira-eva` for the
  canonical eye-level spawn and horizon/LOD coverage. If no preset frames it,
  add one (`ScreenshotPreset`) rather than skipping the check.
- `just preview` — isolated procedural assets (a tree/rock/grass mesh).

A terrain/biome/scatter change that compiles but hasn't been screenshotted is
`verify`, not `done` (see `docs/backlog.md`). Only framings that genuinely need
a live play session (feel, in-flight moments) fall back to asking the user.

**Getting data out of a running session: write it to a file.** When you need
runtime numbers rather than a picture, don't reach for live inspection — have
the game *log* the data and read the file afterwards. JSONL (one JSON object
per line) is the house style for machine-readable runtime data, and belongs
under `tools/diagnostics/`, never beside images. Bare filename overrides for
the game diagnostics resolve there; explicit paths remain available for a
specific reproduction bundle. Existing sinks:
- **`tracing` / `info!` logs** — the game's stdout. Add an
  `info!(target: "thalos::…", …)` where you need a signal, then ask the user
  to run the game and paste the relevant console output.
- **Chrome trace** — `--features profile-chrome` writes `trace-*.json` (see
  Profiling below).

When a new runtime signal is needed, add a small file/JSONL log for it rather
than a live query, and have the user reproduce. Keep the physics core
Bevy-free: do not derive `Reflect` in `thalos_physics_canonical`; if a
Bevy-side `Reflect` projection is needed for a HUD or debug overlay, mirror
canonical state into a Bevy resource at the bridge (`CraftStateMirror`).

**Every spawn situation starts paused (warp 0×).** All `just game [mode]`
scenarios — orbit, eva, landing, final, cruise, runway(-approach) — hold time
at 0× once the loading screen clears, so the sim is frozen until something
advances warp. Two ways to start time:
- Launch with `THALOS_AUTO_RUN=1` (also `true`/`yes`/`on`) — resumes to 1× the
  instant `Loading → Running` fires. It also **skips the start screen** a bare
  `just game` would otherwise boot to; alternatively name a mode explicitly
  (`just game orbit`).
- At runtime, advance warp with the `.`/`,` keys — they step warp up/down, and
  `,` at the bottom is pause.
The single source of truth is `spawn::apply_initial_warp` (on
`OnEnter(AppState::Running)` — which fires on every entry, including
start-screen scenario launches), gated by the `spawn::AutoRun` resource;
deferred placement flows (runway, descent) must **not** reset warp themselves.

## Profiling

Two backends, both gated on cargo features so default builds stay clean.

**Tracy (human-driven, interactive):** `just trace`. Requires Tracy
Profiler GUI v0.11.x running on localhost before launch. Version must
match the linked `tracy-client` (Bevy 0.19 still links tracy-client
0.18.x → Tracy GUI 0.11.x).

**Chrome tracing (artifact-based):** when the user asks to investigate
performance, ask them to run the profile build and reproduce the issue. It's
not wired into `just` because it's a workflow, not a one-shot command — have
the user run:

```bash
cargo run --release -p thalos_game --features profile-chrome
# play ~5–10 s, Ctrl-C → trace-<date>.json in cwd
```

then you analyze the artifact:

```bash
python3 scripts/analyze_trace.py trace-<date>.json
```

The script streams the JSON (handles huge files), aggregates by span
name, and prints a top-N table to identify hot spots. Custom
`info_span!` markers live in `Simulation::step`, `propagate_flight_plan`,
`compute_preview_flight_plan`, `advance_simulation`, `update_prediction`,
`sync_maneuver_plan`.

`analyze_trace.py` can also scope its aggregation to a window around a named
event with `--around-name <span> --window-ms <ms>` (a union of ±window/2
windows around each occurrence) instead of averaging the whole capture, so a
long session full of mostly-fine frames doesn't wash out the bad windows.

## Bug fixing

Diagnose before patching. The loop is:

1. **Reason to a hypothesis set.** From the symptom, lay out the plausible
   causes first — don't jump to the first plausible-looking fix.
2. **Rule candidates out by testing for them.** Each test should be
   targeted and falsifiable: it distinguishes between hypotheses rather
   than merely confirming a guess. Narrow the set down one candidate at
   a time, with the user, until the actual root cause is pinned and
   agreed.
3. **Fix the cause, not the symptom.** Once the cause is known, fix it
   properly — structurally where applicable (remove the whole class of
   bug), not a local band-aid that hides the symptom.

A change that makes the symptom disappear without an explanation of *why*
is not a fix. Don't ship a speculative fix before the cause is confirmed.

Before re-deriving a diagnosis, search `docs/incidents/` for a matching prior
(`rg '<symptom>' docs/incidents/`). When a non-obvious diagnosis lands, write
the post-mortem (`docs/incidents/`, per its README) **in the same change** as
the fix, and promote any standing lesson to a gotcha here or an invariant in
the relevant spec doc.

## WGSL skill

A project skill at `.claude/skills/wgsl-bevy/SKILL.md` collects
WGSL / naga (Bevy) shader pitfalls — reserved words that can't be used
as identifiers, the strict type rules, `naga_oil` import quirks, and so
on. Treat it as a **living document**: whenever you hit a WGSL error
worth remembering (a keyword you couldn't use as a variable name, a
non-obvious error message, a Bevy-specific gotcha), add the case to the
skill so the next agent doesn't rediscover it from scratch.

## Bevy 0.19

The workspace is on **Bevy 0.19** (migrated 2026-07-01). Version stack:
glam **0.32**, wgpu **29**, avian3d **0.7**, bevy_egui **0.40**,
bevy_enhanced_input **0.26**, gilrs 0.11. The full migration write-up +
every API change we hit lives in the `bevy-019-migration` auto-memory and the
official [0.18→0.19 migration guide](https://bevy.org/learn/migration-guides/0-18-to-0-19/);
below is only what stays load-bearing for anyone editing this codebase.

**Render graph is gone — passes are systems.** 0.19 replaced the node-based
`RenderGraph` with ECS schedules. Our custom passes (`scene_depth`,
`sun_shadow`, `film_grain`, the `volumetric_clouds` compute, udlod) are now
**systems in the `Core3d` schedule** (`bevy::core_pipeline::{Core3d,
Core3dSystems}`; sets `Prepass`/`MainPass`/`EarlyPostProcess`/`PostProcess`) or
the root `RenderGraph` schedule (`RenderGraphSystems` `Begin→Render→Submit`).
`RenderContext` + `ViewQuery<D,F>` are **SystemParams** now (`ViewQuery`
auto-skips non-matching views; there is no `render_device()` on `RenderContext`
— add `Res<RenderDevice>`). Order view passes with `.after(main_opaque_pass_3d)
.before(main_transparent_pass_3d)`.

**Two ordering rules are non-negotiable** (both cost us runtime regressions):
- **Any post pass that calls `post_process_write()` must sit in the exact chain
  slot its old node held** — set membership *and* a relative `.after()` are both
  load-bearing. 0.19's `ViewTarget` ping-pong parity index is a persistent
  `Arc<AtomicUsize>` reused across frames, so a mis-slotted flip makes the
  presented buffer alternate → global brightness flicker. `film_grain` must be
  `.in_set(Core3dSystems::PostProcess).after(…::cas)` — last inside PostProcess,
  before the after-PostProcess UI/upscaling consumers.
- **Retained binned render phases**: mutating a material every frame (e.g.
  udlod's per-frame lighting write to `BodyTerrainMaterial`) flags it dirty, and
  Bevy's `queue_material_meshes` runs `phase.remove(main_entity)` for dirty
  entities. A custom queue system must run **after** Bevy's:
  `queue_terrain::<M>.after(RenderSystems::QueueMeshes).before(RenderSystems::PhaseSort)`,
  or it gets dequeued after it adds itself and never draws.

**Resources are components now.** `#[derive(Resource)]` also implements
`Component`. Broad `EntityRef` / `Query<Entity>`-style queries can conflict with
resource access — our `PartQuery` (fuel.rs / staging.rs) filters
`Without<bevy::ecs::resource::IsResource>` to avoid the B0001 panic; keep that on
any broad part query. Also: 0.19 validates `Res<T>` at **fetch time** and panics
if absent, so a `RenderStartup` system reading a resource another `RenderStartup`
system creates must `.after()` it (udlod pins
`init_terrain_render_pipeline::<M>.after(bevy::pbr::init_mesh_pipeline_view_layouts)`).

**Text moved cosmic-text → Parley.** `TextFont.font` is a `FontSource` (not
`Handle<Font>`; `.into()` a handle or name a family), `font_size` is
`FontSize::Px(f32)` (bare `.into()` on an f32 literal mis-infers — write
`FontSize::Px(N)`). The shared `HudTheme.font` is a `FontSource`.

**Notable renames/moves** if you touch these areas: `bevy_scene` →
`bevy_world_serialization` (`Scene`→`WorldAsset`, `SceneRoot`→`WorldAssetRoot`;
we don't use runtime scenes), atmosphere moved `bevy_pbr`→`bevy_light`, `Hdr`
→`bevy::camera::Hdr`, light `shadows_enabled`→`shadow_maps_enabled`,
`ShaderStorageBuffer`→`ShaderBuffer`, `insert_non_send_resource`→`insert_non_send`.
wgpu 29: pipeline `push_constant_ranges`→`immediate_size: u32`,
`DepthStencilState.depth_write_enabled/depth_compare` are `Option<_>`.

**0.19 features we deliberately do NOT use** (we have custom replacements):
Bevy's Skybox, the new BSN / Next-Gen Scenes, rectangular area lights,
`EditableText`. Bevy's built-in atmosphere is now the exception:
`AtmosphereMode::Raymarched` is the canonical rocky-body sky (ADR-0010),
projected through the active `ViewAnchor`; do not add a second production
atmosphere path. **Worth evaluating for the graphics sprint** (new in
0.19, not yet adopted): **contact shadows** (screen-space, kills close-geometry
peter-panning — complements our `thalos::shadow` rig), **physically-based SSR**,
**parallax-corrected cubemaps** (relevant to the F3/F4 IBL work), and the
vignette/lens-distortion post FX. See `docs/graphics_fidelity.md` before pulling
any of these in — they must obey the one-world / spine rules, not bypass them.

## Architecture

> **Crate-distillation refactor (see `docs/architecture.md`).**
> Landed: (1) one authored source of truth for the world — `thalos_world`
> (`BodyDefinition` + `OrbitalElements` + `StateVector` + the folded-in
> atmosphere data schemas + RON parsing, extracted from `physics_canonical`,
> which is now a true leaf depending on `world`); (2) a single celestial-body
> render crate `thalos_body_render` (the former `planet_lighting`+
> `planet_rendering`+`terrain_render` merged into `shading`/`impostor`/`ground`
> modules behind one `BodyRenderPlugin`), which is also the sole consumer of the
> vendored `thalos_udlod` (re-exported as `thalos_body_render::udlod`). Phases
> 1–2 complete (compile-verified; not yet runtime-verified by a `just game`
> launch). **Phase 3** — rename `planet_editor` → `body_editor` — is **done**.
> Remaining: migrate "planet" → celestial-body terminology in the render layer.
> Consult `docs/architecture.md` for phase status.

Thalos is a planetary exploration / orbital mechanics sandbox in Rust
(edition 2024, Bevy 0.19, glam 0.32). Workspace crates:

- **`thalos_world`** — *(Phase 1, new)* authored source of truth for the system
  and its bodies: `BodyDefinition`, `OrbitalElements`, `StateVector`, the RON
  loader (`parsing`), and the body subsystem-config aggregate. Pure Rust, no
  Bevy. Consumed by physics, terrain gen, and rendering.
- **`thalos_physics_canonical`** — pure Rust orbital-mechanics algorithms +
  runtime simulation state; depends on `thalos_world`. (Name contrasts with
  `physics_local`/Avian, not a claim of being the foundation.) Also hosts the
  native atmospheric-aero force model (`aero`): a whole-body lift/drag +
  stability/damping/control evaluator the game drives force-only in the local
  bubble — see `docs/aerodynamics.md`. Also hosts `surface_local`: the
  body-fixed Y-up tangent-frame math (anchor + ENU basis, inertial↔SLF
  conversions composed on `body_fixed`, exact gravity/centrifugal/Coriolis,
  re-anchor) the ship local-physics bubble integrates in — see
  `docs/surface_local.md`.
- **`thalos_control`** — pure-Rust fly-by-wire control layer. The single
  command vocabulary every ship-control source speaks (`AttitudeDemand` /
  `ControlDemand` tagged with a `DemandSource` priority), the priority
  `arbitrate`, the one `AttitudeController` (full-quaternion `Hold` PD +
  nose-`PointNose` PD, replacing the old per-frame deadbeat SAS damper),
  and the effector `allocate` (one torque → reaction wheels + aero control
  surfaces, so they stop fighting). **SAS is regime-dispatched**: on a
  winged craft flying in atmosphere (a `flight::FlightState` is supplied)
  the same SAS hold becomes a plane fly-by-wire law — pitch-attitude +
  bank-angle hold (heading free, so stick-free turns coordinate) with
  pitch auto-trim, sideslip damping, and an AoA envelope that
  stall-protects every pitch command including the pilot's stick —
  while spaceships / vacuum / SAS-off keep the quaternion path unchanged.
  Depends one-way on `physics_canonical`;
  no Bevy. The game-side glue is `thalos_game::control_bus`. See
  `docs/control.md`.
- **`thalos_input`** — Bevy enhanced-input contexts, RON binding loader, and per-binary input intent resources
- **`thalos_ui`** — the **game UI kit** (Bevy): design tokens (palette /
  spacing / type scale, `UiTheme`), the frosted-glass panel surface
  (`GlassMaterial` + the `UiBackdropSource` scene-copy pass), and the widget
  library (buttons/menu rows, sliders, checkboxes, cycle pickers, text
  fields, scroll columns, toasts, headings/dividers). **Every screen composes
  this kit — no per-screen colours, fonts, or interaction styling**; the
  flight HUD's `HudTheme` is a projection of the same tokens. Iterate with
  the kitchen-sink testbed (`just ui-preview` → PNG). Fonts: Inter
  (interface, OFL) + Fira Code (numeric/mono; Δ-strings stay mono by
  convention). See `docs/ui.md`.
- **`thalos_game`** — Bevy consumer of physics + terrain outputs
- **`thalos_terrain`** — procedural terrain generation pipeline (no Bevy dependency)
- **`thalos_celestial`** — procedural sky model: stars, galaxies, nebulae as physical flux sources (no Bevy dependency)
- **`thalos_texgen`** — procedural texture generation (no Bevy): CPU-rasterizes
  `TextureData` (sRGBA8), today the **foliage atlas** (small multi-toned leaf
  clusters + conifer needles + full-colour painterly bark) the tree meshes
  sample, plus a companion **foliage material atlas** (`foliage_material_atlas`,
  linear `Rgba8Unorm`: bark tangent-space normal in RGB + roughness in A). Bark
  albedo, normal, and roughness are all derived from one shared `bark_height`
  field, so cracks/ridges line up across channels; the height field uses
  **gradient (Perlin) noise**, not value noise, so the derived normal shows no
  lattice "weave" (see the `wgsl-bevy` skill note). Shared by the runtime
  (`body_render` wraps each in a GPU `Image` via `build_foliage_atlas` /
  `build_foliage_material_atlas`), the object preview, and an offline bake
  (`cargo run -p thalos_texgen --example bake` → `tools/texgen/out/*.png`, to
  inspect or prebake for the game). The atlas layout + `leaf_code` packing live
  here as the source of truth. Rocks / other procedural textures will live here
  too.

  *(The former `thalos_atmosphere` data crate — gas-giant cloud decks, hazes, rings, terrestrial scattering schemas — is folded into `thalos_world::atmosphere`; authored body data has one home.)*
- **`thalos_physics_local`** — Bevy/Avian f64 local-physics boundary for M5; aggregate craft hydration, terrain collider patches, contact/collapse helpers. **Ships integrate in the surface-local frame (SLF)** — a body-fixed tangent frame anchored under the craft, Y-up, small (meters–km) coordinates near the anchor, re-anchored at ~1.5 km drift; the frame math is `thalos_physics_canonical::surface_local` and the design/implementation notes are in `docs/surface_local.md`. The Avian rigid body persists across every regime; what *role* Avian plays each frame is a three-way `AvianRole`: `Paused` under warp / `BodyFixed` (canonical owns everything), `AttitudeOnly` while coasting in vacuum at 1× (Kepler owns translation, Avian still integrates rotation + contact for player input and SAS), `Full` when there's a non-gravity force to integrate (throttle active, terrain collider attached, or inside the atmosphere shell). Since the A3 port the role is **classified by the `CraftRegime` resolver** (`thalos_physics_canonical::regime`) and merely projected onto `AvianAuthority` by `compute_avian_authority` (`crates/game/src/local_physics.rs`), which keeps the `previous_role` edge the handoff snap reads. Coasting flight in vacuum stays under Kepler / `OnRails` so AP/PE do not drift. The role classifier (`compute_avian_authority`) lives in `crates/game/src/local_physics.rs`; the resulting **canonical authority transitions are owned by the regime executor** (`crate::regime::apply_regime_authority`, applying the unit-tested `thalos_physics_canonical::regime::expected_authority` — it subsumed the former `manage_authority`, the landed throttle release, and the timed settle collapse; see `docs/regimes.md` Phase A3). **Ground colliders are solid and static in the SLF**: terrain is a parry **heightfield** (not a one-sided trimesh — the trimesh's one-step penetration recovery flung landing craft off their gear), the runway is a solid cuboid slab (`crates/game/src/runway.rs`). A **wheeled craft's hull is filtered out of solver contact with the ground** via collision layers (`GROUND_LAYER`/`CRAFT_LAYER`); its raycast spring-damper landing gear is the sole ground interface and its force/torque is inertia-relative clamped. Gearless craft (landers) keep all-vs-all layers and rest on the heightfield directly. Fast descents are kept from tunneling by `SweptCcd` + the analytic `terrain_floor_backstop`, and a too-hard contact destroys the craft via the whole-craft impact model (`detect_terrain_impact` → `Simulation::mark_destroyed`, gated on `ShipParameters::impact_tolerance_m_s`; the contact signal is `weight_on_wheels` for wheeled craft, hull contact for gearless). **EVA is a deliberately separate kinematic path** — it is *not* an SLF citizen: it has no collider and computes its canonical state directly in the body-fixed frame (`player_controller::step_eva_controller`), so it gains nothing from the SLF's contact-solver stability; do not "unify" it into the SLF without on-foot walk-testing (see `docs/surface_local.md` §10). On destruction the game force-pauses and shows an in-place scenario-respawn picker (`crates/game/src/scenario_menu.rs`) offering the four start scenarios (ship orbit / landing / final approach / EVA); see `docs/surface.md`.
- **`thalos_body_render`** — *(Phase 2, new)* unified celestial-body rendering, one appearance model + regime-scaled projections. Four modules behind one `BodyRenderPlugin`: `shading` (shared `SceneLighting`/`AtmosphereBlock`/Hapke `shade_hapke_surface` + the `thalos::lighting`/`thalos::atmosphere` WGSL libraries), `impostor` (distant billboard materials for planets, gas giants, rings, solid bodies), `ground` (the `thalos_udlod`-backed terrain LOD: `ThalosTerrainPlugin`, `PipelineTileProvider`, `BodyTerrainMaterial`/`BodySkyMaterial`/`BodyWaterMaterial`, rendered-height patch utilities), and `clouds` (the absorbed MIT `bevy-volumetric-clouds` fork: spherical body-fixed volume march + cloud-local history targets). Merged from the former `planet_lighting`+`planet_rendering`+`terrain_render` and top-level `thalos_volumetric_clouds` mechanisms. A backend chooses geometry or cost, never its own lighting/atmosphere/weather authority. Terrestrial clouds obey ADR-0009: authored `CloudClimate` → one per-body `CloudWeatherField` → near/orbit/shadow projections.
- **`thalos_udlod`** — vendored UDLOD terrain renderer (lives at `crates/udlod/`). Forked from [`kurtkuehnert/bevy_terrain`](https://github.com/kurtkuehnert/bevy_terrain) by Kurt Kühnert (MIT OR Apache-2.0); attribution + license files travel with the source. Edit in-tree like any other workspace crate. The fork is **runtime-provider-first**: it renders sparse tile atlases fed by `TileProvider` implementations, not preprocessed Earth-style asset trees. The old GeoTIFF/preprocess path is gone. (Upstream's own successor is now a *different repo*, [`planetary_terrain_renderer`](https://github.com/kurtkuehnert/planetary_terrain_renderer) — the better diff target for fixes to the Taylor-series precision path.) A 2026-07 optimization pass tailored the fork to Thalos; **see `docs/terrain_lod_optimization.md`**, and note these load-bearing rules:
  - **Providers own mip generation.** `TileProvider::request_tile` must return the **full mip chain** (call `AttachmentData::generate_mipmaps` inside the task). The atlas does *not* regenerate mips — that kept per-tile mip filtering on the main thread and made cached payloads useless.
  - **Attachments may differ in resolution.** The GPU atlas sizes each attachment's texture array independently. Height keeps the full grid (it is the geometry, and the only attachment physics reads); albedo/roughness/material bake at half (`TierConfig::detail_texture_size`) — a >2× cut in the game's largest allocation.
  - **Tiles are cached, and the cache key is the contract** (`game::rendering::tile_cache`): memory (survives terrain despawn/respawn) over disk (survives the process) over synthesis. The namespace is a `NamespaceFn` resolved **per request**, not frozen at construction, because the flatten handle is read per tile *pixel* — a pad installed after spawn still changes what later tiles bake. **If you add an input to tile synthesis, fold it into the namespace, and bump `thalos_terrain::GENERATOR_VERSION` when generation output changes** — otherwise a cached run silently renders old terrain. `THALOS_TILE_CACHE=0` disables the disk tier while iterating on generation.
  - **CPU draw-tile selection is the sole tile-selection authority** (it enforces the 2:1 LOD balance across cube-face seams that the GPU's per-tile-independent predicate could not). The dead GPU tiling prepass has been **deleted** — do not reintroduce it. Refinement now also honours a screen-space-error hint (`TileProvider::subdivision_scale`, ≤ 1, so it can only *remove* detail on flat ground) and a hole-free behind-view streaming cull (`TerrainViewConfig::cull_behind_view`).
  - Tile **production** on the GPU remains the intended big win, but it is blocked on an architectural decision, not effort: porting the cascade to WGSL creates a *second height authority* that would drift from the CPU one the colliders and spawn-site search read. See the doc's "What did not land, and why". **`big_space` integration is unconditional** — the upstream `high_precision` Cargo feature has been removed, along with the runtime `DebugTerrain.high_precision` toggle and the `HIGH_PRECISION` shader define / pipeline flag. The Taylor-series relative-position path (`compute_relative_position` in `shaders/functions.wgsl`) is the only viable precision path at planet scale; gating it behind a feature only forced defensive `#[cfg]` plumbing in every consumer.
- **`big_space`** — *(vendored)* floating-origin / high-precision grid plugin at
  `crates/big_space/`. Fork of [`aevyrie/big_space`](https://github.com/aevyrie/big_space)
  0.12 (MIT OR Apache-2.0). Vendored during the Bevy 0.19 migration because it is
  the foundational precision substrate (real_space, the vendored udlod, body_render
  all build on it) yet upstream is a single-maintainer crate that lags each Bevy
  release — owning it keeps it off the version-bump critical path. Consumed via
  `[workspace.dependencies] big_space { path, features = ["i64"] }`. Port faithfully;
  keep diffable against upstream (the udlod playbook). The `i64` cell precision is
  the workspace-wide choice. udlod keeps its `big_space.rs` re-export shim.
- **`bevy_erosion_filter`** — *(vendored)* tiny GPU/CPU erosion-noise filter at
  `crates/bevy_erosion_filter/`. Fork of `bevy_erosion_filter` 0.1.2 (MIT). Vendored
  for the same reason as big_space (upstream targets Bevy 0.18 / glam 0.30). The
  `bevy` feature is optional so `thalos_terrain` uses the pure-glam `cpu` module with
  `default-features = false` and pulls no Bevy crate (the no-Bevy CI guard still
  holds); `thalos_body_render` uses the `bevy` feature for the shader-library plugin.
- **`thalos_shipyard`** — parametric ship **construction model** (ECS attach tree, RON blueprints): part components + catalog, resources, blueprint (de)serialization + spawn, attach nodes / surface mounts / KSP linked symmetry, parametric sizing + mass/capacity recompute, stats / staging, and the geometry mesh builders (cockpit / engine / fuselage / gear / wing) shared with the game's flight-craft rendering. It owns *what a craft is*; it does **not** own the interactive editor or any UI. The **editor application** lives with its sole consumer, the game, at `thalos_game::shipyard_editor` — a UI-agnostic `core` submodule (`ShipEditorCorePlugin`: `EditorState` command/state hub, placement, live mesh rebuilds, tank-resize handle, placement-preview ghost, shrouds, blueprint save/load against `ships/*.ron`) plus the native Bevy-UI front-end (scene + panels). There is no standalone editor binary (the old egui `just shipyard` tool was deleted). Every editor-owned entity carries the `EditorPart` marker (defined in `shipyard_editor::core`) and every core query filters on it; host systems that aggregate the same part components for the *flight* craft (fuel, staging, gear, ship visuals, colliders) must filter `Without<EditorPart>` — that marker is the only thing separating the build world from the flying craft in the same ECS `World`. Resource storage is whitelist-driven from the parts catalog: any part kind can declare `storage` entries for fixed (`units`) or volume-scaled (`units_per_m3`) capacity, and blueprints may only activate resources whitelisted by that part. Omitted blueprint resources mean "use catalog defaults"; explicit resource maps mean the user's selected active pools. Do not restore hard-coded per-resource tank fields such as `methane_l_per_m3` / `lox_l_per_m3`; add real resources (for example `Kerosene`) to `Resource` and catalog storage lists instead. Air intake is ambient capture, not stored oxidizer: engines declare `intake_requirement`, nacelles may provide `builtin_intake`, and separate `Intake` parts can feed future engine-core layouts. See `docs/construction.md`.
Core separation: `world`, `physics_canonical`, `control`, `terrain`,
`celestial`, and `texgen` are pure Rust libraries; `input`, `game`, `body_render`,
`physics_local`, and `shipyard` are Bevy consumers. Within
`body_render`, the `shading` module is the single source of truth for
`SceneLighting`, `AtmosphereBlock`, and the surface BRDF that both the
`impostor` and `ground` backends consume — no field-by-field mirror types.
Avian lives behind `thalos_physics_local`; do not add Avian to
`thalos_physics_canonical`.
Semantic input for the Bevy binaries flows through `thalos_input`
contexts and intent resources, with checked-in defaults at
`assets/input.ron`. HOTAS support also lives there: buttons can be
added to the existing action bindings as `GamepadButton(...)`, while
continuous pitch/yaw/roll/throttle axes are opt-in under
`game.hotas` and feed the same `GameInputIntent` fields as keyboard
flight controls. Those continuous axes are **not** read through Bevy's
gamepad layer: Bevy's `bevy_gilrs` converter drops every axis gilrs
labels `Unknown` (a bare flight stick's twist/throttle slider), so they
never reach `Gamepad` state. `thalos_input::joystick` instead runs its
own `gilrs::Gilrs` instance, reads every axis by raw platform `Code`,
and snapshots `code -> value` into `RawJoystickState`; HOTAS axes bind a
raw `u32` `code` (platform-specific — discover with `cargo run -p
thalos_input --example gamepad_axes`), not a `GamepadAxis`. Do not
revert HOTAS axes to `GamepadAxis` bindings; the twist literally cannot
be expressed that way.

### Physics crate (`crates/physics/`)

Two trait abstractions draw the boundaries:

- `BodyTrajectoryProvider` (`body_trajectory_provider.rs`) — answers
  "where is body `i` at epoch `t`?" Implemented by `PatchedConics`;
  could be swapped for a baked ephemeris.
- `ShipPropagator` (`ship_propagator.rs`) — propagates the ship across
  one segment of coast or burn. Implemented by `KeplerianPropagator`:
  analytical Kepler coast under a single SOI body + RK4 finite-burn.
  SOI transitions are detected per substep and refined by bisection
  (coast) or shortened RK4 (burn).

Key modules:

- `canonical` — world preset/config, `Epoch`, `CraftState`,
  `AuthorityMode`. `Simulation` owns exactly one canonical craft state
  and authority mode for the player craft; `WorldPreset::Classic` is
  the only wired preset.
- `types` — `StateVector`, `BodyDefinition`, `TrajectorySample` (each
  sample carries `anchor_body` + `ref_pos` so the renderer can draw
  without a per-sample ephemeris query).
- `orbital_math` — Cartesian↔Keplerian conversions, Kepler-equation
  solvers (elliptic + hyperbolic), `propagate_kepler`.
- `patched_conics` — `BodyTrajectoryProvider` impl. Bodies form a
  parent-child tree; queries walk the lineage and sum each ancestor's
  motion.
- `ship_propagator` — `ShipPropagator` trait + `KeplerianPropagator`
  impl. Segments terminate on the first of: target time, SOI exit,
  collision, SOI enter, burn end, or stable-orbit closure. Resolution
  order at boundaries: `exit > collision > enter`.
- `simulation` — `Simulation` struct: owns canonical craft state,
  authority bookkeeping, warp, `KeplerianPropagator` instance, and
  `ManeuverSequence`. `step()` is called each frame and consumes
  maneuver nodes as their start time arrives.
- `body_fixed` — pure inertial↔body-fixed frame helpers used by landed
  `BodyFixed` pose evaluation (the rotating-with-the-surface authority).
- `body_centered` — pure inertial↔body-centered (translates with body,
  inertial axes) frame helpers. The frame Avian's local rigid body lives
  in: gravity reduces to `−μr/r³` with no fictitious forces.
- `trajectory/` — Flight-plan prediction. `propagate_flight_plan` runs
  the same `ShipPropagator` across the maneuver sequence, producing
  `Leg`s of `(burn?, coast)` `NumericSegment`s. Includes event
  detection (SOI / apsis / impact), encounter aggregation,
  closest-approach scans.
  - `propagate_branch_stack` produces a `TrajectoryBranchStack` with
    one `BranchKind::Actual` branch (no-maneuver baseline) plus
    `BranchKind::Projected` branches that fork at each maneuver-node
    start; this is how the map view renders ghost/preview tracks for
    pending edits.
- `maneuver` — `ManeuverNode`: time, delta-v in local
  prograde/normal/radial frame, reference body. Plus frame-conversion
  helpers.
- `parsing` *(now in `thalos_world`; re-exported as
  `thalos_physics_canonical::parsing` during the migration)* — Loads
  `assets/solar_system.ron` plus per-body files at
  `assets/bodies/<lowercase_name>.ron` into `SolarSystemDefinition`.
  System-level file carries physical + orbital specs; per-body files
  carry `terrain: TerrainConfig` and `tectonics: TectonicConfig`. The
  authored `BodyDefinition`/`OrbitalElements`/`StateVector` types also
  live in `thalos_world` now (see `docs/architecture.md`); the player's
  debug spawn orbit is derived from `homeworld_id` via
  `debug_orbits::debug_parking_orbit_relative_state`, not stored on
  `SolarSystemDefinition`.

### Game crate (`crates/game/`)

- **Boot, loading, and the start screen** (see `docs/boot.md`).
  `loading.rs` owns `AppState` (`Loading` → `MainMenu` | `Running`) and the
  **`LoadingTracker`** — a declarative step registry (`begin` registers the
  step set for a load; producers update their step by id; the reveal fires
  when all registered steps complete, including deferred craft placement).
  A bare `just game` boots to the **start screen** (`main_menu.rs`:
  scenario picker / SHIPYARD / SETTINGS / QUIT) with the **world deferred**
  (`loading::WorldState::Absent`): no bodies / ship / sky spawn, no terrain
  streams, the boot loading pass registers zero steps (near-instant menu),
  and the winit loop is throttled to reactive mode while the menu is up.
  The world-spawn systems hang off `OnEnter(WorldState::Live)` instead of
  `Startup`; naming a scenario (`just game runway`) or setting
  `THALOS_AUTO_RUN` skips the menu and inserts `Live` (same chain, first
  frame, boot unchanged). The menu's first start after a deferred boot *is*
  a boot: it arms the boot placement flags, registers the boot step set,
  flips `Live`, and re-enters `Loading` (no craft swap needed —
  `spawn_player_ship` builds the chosen scenario's blueprint directly).
  With the world already live (menu re-entered from flight), scenario
  starts reuse the respawn / relaunch machinery in place — the runway pair
  re-arms its deferred placement + settle gate and re-enters `Loading`.
  `SpawnSituation` is therefore **mutable at runtime**; deferred placements
  are explicitly armed (`DescentPlacement`, `RunwayPlacement`), never keyed
  off the situation with a `Local<bool>`.
- **Spawn situation is a flag: ship in orbit, EVA on the
  surface, a landing approach over land, a final approach over
  flat land, or one of two surface-runway scenarios.** `main.rs` reads
  `just game [mode]` (passed as a CLI arg — default `menu`, the start
  screen; falls back to the `THALOS_SPAWN` env var for a direct
  `cargo run`) into a
  `spawn::SpawnSituation` resource (`ShipOrbit` | `PolarOrbit` | `Eva` |
  `Landing` | `FinalApproach` | `Runway` | `RunwayApproach` | `Cruise`).
  The canonical `CraftState` is the player either way — KSP-style: one
  craft, Ship or EVA, distinguished by `VesselKind`. The ship blueprint is
  chosen per scenario by `SpawnSituation::ship_blueprint_path`
  (`apollo.ron` by default, `meridian.ron` for the aircraft scenarios).
  **`orbit`**:
  `VesselKind::Ship` in a low Thalos parking orbit
  (`system.ship.initial_state`), nose along prograde;
  `ship_view::spawn_player_ship` loads the blueprint and pushes the real
  ship params. **`eva`**: the player on foot, `VesselKind::Eva` with
  `ShipParameters::eva()` (90 kg, no thrust, no torque), placed ~12 km
  above the Thalos sub-stellar point;
  `local_physics::spawn_player_avian_body` branches on `VesselKind` to
  spawn a 1.8 m capsule (rotation-locked, walking friction) carrying both
  `LocalCraftBody` and `PlayerControllerBody`, and `spawn_player_ship`
  early-returns (no rocket). Re-boarding a ship is not implemented yet;
  the `toggle_player_controller` input is unwired. **`land`** (aliases
  `landing` / `descent`): `VesselKind::Ship` on a vacuum powered-descent
  approach. Because placing it *over land* at a true above-ground
  altitude needs terrain heights (unknown until bakes load), `main.rs`
  seeds the parking orbit as a placeholder behind the loading screen and
  `spawn::refine_descent_spawn` installs the real state on the first
  `AppState::Running` frame: it searches the daylight hemisphere for a
  dry-land, ice-free site (terrain height > `sea_level_m`, low latitude),
  then drops the ship ~25 km AGL over it, descending, nose retrograde,
  coasting `OnRails` until it crosses the 20 km local-physics handoff for
  the powered touchdown. **`final`** (aliases `final-approach` /
  `final_approach` / `approach`) uses the same deferred terrain-aware
  path but scores daylight dry sites by local height relief, then starts
  the ship ~1.5 km AGL, low and slow, over the first sufficiently flat
  patch (or the flattest dry fallback). Thalos now has both a *visual*
  atmosphere + volumetric clouds (`terrestrial_atmosphere` in
  `assets/bodies/thalos.ron`; see `docs/atmosphere.md`) and a *physics*
  atmosphere (per-body density below the Kármán line; see
  `docs/aerodynamics.md`), so descents now experience aerodynamic **drag** —
  a descending/reentering ship decelerates toward a terminal velocity rather
  than free-falling lunar-style. **`runway`** (alias
  `rwy`) and **`runway-approach`** (aliases `rwy-approach` /
  `approach-runway`) put the `meridian.ron` aircraft on a fixed runway on
  the Thalos surface, owned by `crate::runway`. Like the descent modes
  these are deferred and terrain-aware: `runway::finish_runway_spawn`
  runs once per arming of `RunwayPlacement` (boot, or a start-screen
  runway launch) and places the runway at a **fixed body-fixed site**
  (constant lat/lon + heading, `runway::fixed_runway_site` from the
  `RUNWAY_SITE_*` constants, overridable at runtime with
  `THALOS_RUNWAY_SITE="lat,lon[,heading]"`). This replaced the former
  flat/dry/sunlit grid scan, whose night-side fallback could spawn the
  craft in the dark; the pad is flattened under the footprint regardless,
  so the coordinates only need to be dry land sunlit at the spawn epoch
  (the author's call — a below-sea-level sample is warned, not corrected).
  The scenario also seats the sim clock at a **morning boot epoch**
  (`runway::RUNWAY_MORNING_EPOCH_S`, via `Simulation::set_sim_time`) so the
  fixed site is lit by a low rising sun instead of the epoch-0 noon sun the
  sub-stellar point gives it — time-of-day is a planet-rotation lever, the
  authored site stays put. Re-derive the epoch with the `morning_probe`
  example if the site or Thalos's rotation changes. After placing, it tears
  down any live Avian
  bubble so the rebuild seeds from the placed state (a bubble seeded
  from the pre-placement placeholder orbit would leave the rendered
  craft coasting the wrong trajectory). **The terrain itself is
  flattened into a pad and the runway
  sits flush on it** (replacing the former raised-slab + skirt/runoff
  platform): a single fixed elevation `E = mean(natural terrain over the
  basin)` — levelling to the *mean*, not the max, so the wide basin balances
  cut against fill instead of becoming an all-fill plateau towering over its
  surroundings — is chosen, then a `thalos_terrain::TerrainFlatten` pad is
  installed through the body's shared
  `rendering::ground_terrain::TerrainFlattenRegistry` handle. The terrain
  tile provider (`PipelineTileProvider`, wrapped in
  `thalos_terrain::FlattenedSurface`) reads that handle as it bakes, so the
  *rendered* ground — and, via the GPU-atlas height mirror, the collider and
  CPU height queries — level out to `E` across the basin and smoothstep-blend
  back to natural terrain over a wide (~500 m) ramp.
  The handle is read per tile-pixel, so setting the region affects tiles
  baked afterward — no UDLOD tile-reload needed, because the pad is set
  before the aircraft/camera jump to the site, so the tiles that stream in
  there bake flattened from the start. The handle lives in a per-body
  registry so it survives terrain despawn/respawn churn. On top of the
  levelled ground the runway is just a paved strip + markings + posts (lifted
  a few cm so the paving reads on the grass). To stay rock-steady at high
  warp the runway is positioned like the player ship: a **root** big_space
  grid child re-placed in f64 every frame (`update_runway_transform`),
  *not* a fixed-cell child of the rotating body grid — whose multi-Mm cell
  offset rotated by an f32 quaternion jitters as the body spins fast (only
  the small child vertex offsets ride the f32 `Transform.rotation`). A
  **flat kinematic collider at `E`** (a trimesh posed each frame by
  `sync_runway_collider_pose`, exactly like the terrain collider patch via
  `terrain_patch_pose`) still backs the landing surface so it stays exactly
  flat regardless of tile-streaming timing.
  `runway` parks the aircraft at rest on the threshold (`BodyFixed`
  authority, the launch-clamp pattern), lifted by its own measured ground
  clearance — `craft_ground_clearance` walks the part-visual meshes
  (`Mesh::compute_aabb`, since they carry `NoFrustumCulling` and so have no
  Bevy `Aabb`), finds the lowest point in the craft body frame, and offsets
  the spawn so *any* craft rests on the surface rather than a hardcoded
  constant; `runway-approach` starts it on short final (`OnRails` → bubble
  handoff onto the flat collider). **The parked `runway` spawn happens *behind*
  the loading screen.** `finish_runway_spawn` runs during `AppState::Loading`
  (not the first `Running` frame), so the aircraft is parked + the flatten pad
  installed + the camera at the surface while the screen is up; `surface_settle`
  (`SurfaceSettlePlugin`) then holds the screen until the terrain under the view
  has streamed flush (resident LOD at `tile_tree.view_position()` plateaus past a
  target), so the first visible frame is already flush + settled instead of the
  ground heaving up to the strip. Do **not** pause the sim during this window —
  `readback_local_craft` stops following the craft when the integrator is idle,
  so a pause strands the camera in the placeholder orbit and the tiles never
  stream at the site. Only the *parked* runway is gated; the airborne approach
  flies its descent during loading, so gating it would fly it past the threshold.
  Cold tile streaming to a fresh surface site is slow (~15 s for Thalos) because
  each tile is an expensive CPU field bake; `body_render`'s `compute_tile_pixels`
  is rayon-parallelised on a bounded pool and `udlod`'s `TileAtlas::update`
  admits tile loads **coarse-first, then nearest-view-first within each LOD** —
  so a cold view fills the low-LOD pyramid (complete coarse coverage, cheaply)
  before refining the nearest tiles, instead of building outward from the single
  closest tile and leaving the rest of the view an empty hole. That progressive
  coarse→fine fill is what makes the wait tolerable rather than minutes — see the
  cold-streaming memory note.
- **EVA is a full craft with a real character controller and a
  grounded/airborne split.** The `EvaMode` resource
  (`player_controller.rs`, `Grounded` | `Airborne`, defaults `Grounded`)
  is the coarse surface↔orbit switch. **`Grounded`**: `step_eva_controller`
  runs a kinematic character controller in the body's **body-fixed
  (rotating) frame** — it tracks a body-fixed position + surface-relative
  velocity and runs a grounded/airborne state machine with surface gravity
  (`g = μ/r²`): camera-relative walking (Shift = sprint), jumping (Space),
  walking off ledges into a ballistic fall, and landing. Working in the
  body-fixed frame (not body-centred inertial) is the key fix: surface
  velocity is the player's walking speed (m/s), not the inertial
  co-rotation drag `ω×r` (hundreds of m/s → km/s). That keeps warp from
  exploding the integrator, and the terrain height query never
  sea-level-teleports (a `None`/missing-tile sample holds altitude instead
  of snapping to the reference radius). `snap_avian_from_canonical`,
  `apply_local_forces`, and the `readback_local_craft` translation all
  short-circuit for grounded EVA so the controller owns the capsule pose
  outright; canonical authority is pinned to `LocalRigidBody` by the
  regime executor's walking pin (`regime::expected_authority`, not
  `OnRails`) since grounded EVA co-rotates with the surface and
  `Simulation::step` must only advance sim-time.
  `sync_avian_time` still pauses Avian's clock under warp (the controller
  writes `Position` directly each frame; no integrator is needed and the
  km/s-scaled integration that broke it is gone).
  **Rest detection + warp gating (KSP-style):** the controller publishes
  `grounded` / `at_rest` / `surface_speed_m_s` on `PlayerControllerState`.
  `enforce_warp_altitude_limits` lets on-foot warp climb above 1× only once
  the player is *at rest* (standing, stopped); while walking / jumping /
  falling it clamps to 1×, and movement input drops warp back to 1×
  immediately. At rest it caps at `SURFACE_WARP_MAX_SPEED` = 100× (above
  that the UDLOD tile streamer can't follow the body rotating under the
  camera — a UDLOD limitation, not a sim one). The on-foot HUD pill
  (`hud/eva_panel.rs`) surfaces `MOVING` / `FALLING` / `STANDING — warp
  ready`. **`Airborne`**: Kepler owns translation, the snap drives the
  capsule from canonical (exactly like a ship coasting in vacuum), and
  `step_eva_controller` stands down. `apply_local_forces` short-circuits
  for EVA in both modes — EVA has no thrust yet (coast-only; a jetpack is
  the natural follow-up). The teleports mirror the ship's: body-tree
  cmd-click sends EVA to a low orbit (→ `Airborne`), and a row's `drop`
  button + map cursor (or F9) plants it on a surface point (→ `Grounded`,
  in place via `local_physics::place_eva_on_surface` — the bubble is never
  torn down, unlike ships). `EvaMode` flips only on these explicit
  teleports; suborbital ballistic flight (jump, walk off a cliff) stays
  *within* `Grounded`, switching the controller's internal grounded/airborne
  state, not `EvaMode`.
- Semantic player input is read from `thalos_input::game::GameInputIntent`.
  Keep raw Bevy input only for cursor positions, picking spatial data, and UI
  internals. See `docs/input.md`.
- `solar_system_state` — canonical game-level per-frame source for evaluated
  solar-system state. `SimulationState` owns the long-lived simulation,
  ephemeris, and authored system definition; `SolarSystemState` is refreshed
  once per frame from it and owns the `BodyState` vector plus per-body
  environment state (`DynamicSurfaceState`, cloud-band dynamics/phases, and
  later wind, storms, tides, dune movement). Map, real-space, impostor,
  terrain, halo/sky, and material systems are projections of this resource;
  they may cache derived data but must not independently own body/environment
  state.
- `bridge` — Core adapter. Calls `Simulation::step()` each frame,
  recomputes trajectory prediction *synchronously* on the main thread
  when the cached plan is dirty/stale, syncs maneuver edits, handles
  warp controls. (Single early-terminating `propagate_flight_plan`
  pass keeps the typical rebuild well under a frame; running in-line
  means an edit on frame N produces the fresh trajectory on frame N.)
- `control_bus` — game-side glue for the `thalos_control` fly-by-wire
  layer. Each frame `realize_control` collects every attitude command
  source as a tagged `ControlDemand` (pilot stick, the `T`-key SAS hold,
  the directional nav modes via `navigation::nav_attitude_demand`, the
  scheduled-burn autopilot via `Autopilot::attitude_demand`), arbitrates
  by priority, runs the one `AttitudeController`, and allocates the
  resulting torque to *both* effectors — reaction wheels
  (`ControlInput::torque_command` → `apply_local_forces`) and aero control
  surfaces (`RealizedControl::aero` → the aero force system). Attitude is
  no longer set anywhere else; the old `bridge::handle_attitude_controls`
  + `navigation::compute_attitude_control` + raw-stick aero paths are
  gone, and the per-frame deadbeat SAS damper with them. `realize_control`
  also arms the plane flight assist: when SAS is engaged on a winged craft
  flying in atmosphere it builds the body-frame `FlightState` (local up via
  `surface_local::radial_up`, air-relative velocity, the config's
  `stall_alpha`) that switches the controller's SAS hold to the fly-by-wire
  pitch/bank law with auto-trim + stall protection, and publishes
  `RealizedControl::assist` for the HUD (the SAS button reads FBW /
  warn-tints while protection clamps). **SAS defaults on** (`SasState`,
  surviving destruction/respawn); the HUD SAS button toggles that same
  switch as the `T` key. Throttle still
  rides its own setpoint path (`ThrottleState::commanded`, autopilot
  override, `ControlLocks`) pending a later fold-in. See `docs/control.md`.
- `map_view` — snapshot/projection boundary for map rendering. Copies
  `CraftState`, body states, and `FlightPlan` into `MapSnapshot`; map
  systems consume the snapshot and never mutate canonical simulation
  state.
- `hud/` — Bevy-UI HUD panels (`HudPlugin`), one file per panel, sharing
  the `theme::HudTheme` style. Includes the **MFD slot** (`hud/mfd/`): a
  contextual, customizable ship-view widget slot that auto-selects the
  widget relevant to the current flight context (orbital `Trajectory`
  plot, airliner `NavDisplay`, with `Docking`/`Interplanetary` stubs)
  with a manual pin/hide override. Each widget exposes `build` /
  `relevance(&FlightContext)` / optional `update`; `select_active_widget`
  is the single owner of widget + slot visibility, and `HudPanel` lives on
  the slot container only. This replaced the old single hardcoded
  "TRAJECTORY" panel, which surfaced an orbital plot during atmospheric
  flight. See `docs/hud_widgets.md`.
- `rendering/` — Every system that turns simulation state into
  rendered geometry. Submodules:
  - `types` — rendering-shared resources (`CameraExposure`,
    `PlanetshineTints`) and components (`CelestialBody`, `PlayerShip`,
    `PlanetMaterials`, `SolidPlanetMaterials`, etc.). It re-exports
    `SimulationState` / `SolarSystemState` for compatibility, but their
    definitions live in `solar_system_state`.
  - `real_space` — BigSpace root and per-body real-space grids. The
    ship camera carries `FloatingOrigin`; UI and map entities stay
    outside the BigSpace hierarchy.
  - `spawn` — startup system that creates map-side body entities plus
    real-space body grids. Procedural bodies get a `PlanetMaterial`
    impostor billboard; non-procedural bodies get a
    `SolidPlanetMaterial` solid-color billboard.
  - `generation` — polls the in-flight `PlanetSurface` async tasks,
    bakes the result into a `PlanetMaterial`, and handles
    reference-cloud loading.
  - `lighting` — `CameraExposure`, `SceneLighting` population, planet
    / solid-planet material light updates, sun-light direction.
  - `transforms` — map snapshot projection, body/ship transform sync,
    planet orientation (tidal lock + spin).
  - `materials` — per-frame parameter updates for gas-giant, ring,
    and cloud-band animation.
  - `trails` — orbit-line periodic recompute + gizmo draw with focus
    / sibling fade.
  - `body_lod` — screen-space LOD: icon ↔ impostor crossfade,
    moon-merge fade, double-click-to-focus picking, homeworld focus.
  - `scene_depth` — `SceneDepthImage` resource + copy pass. Copies the main
    view depth into a sample-able `Depth32Float` image for custom effects that
    still consume it. The canonical rocky-body atmosphere uses Bevy's own
    depth-aware raymarch; the retained `BodySkyMaterial` is debug/composite
    migration code, not a second production atmosphere.
  - `ground_terrain` — UDLOD terrain spawn for procedural bodies +
    impostor↔terrain LOD swap (`sync_terrain_impostor_swap`) at
    `4 × radius`. It still spawns the legacy `BodySky` fullscreen quad for
    debug comparison and retained composites; normal rendering force-hides it
    in favour of the Bevy raymarched atmosphere.
  - `view_anchor` — the **one per-frame answer to "where is the view?"**
    for every view-dependent detail system. `ViewAnchor` (sole writer
    `update_view_anchor`, top of `SimStage::Sync` before
    `sync_solar_system_state`) resolves the `ShipCamera`'s big_space pose
    against the nearest terrain-backed body, **body-fixed at a coherent
    epoch** (previous-frame pose paired with previous-frame body states, so
    re-projection with the current frame's `BodyState` is exact under
    co-rotation × warp — see the module doc). Surface scatter (trees /
    grass / GPU grass / rocks) and the sun-shadow cascade centre all read
    it; **nothing view-dependent may anchor to the player craft or a mode
    flag** — new camera modes (god views, freecam, screenshot rigs) get
    correct detail with zero per-mode plumbing. Replaced
    `scatter_view_center` + `ShadowFocusOverride`.
- `ghost_bodies` — Renders ghost planet positions during time warp
  preview.
- `sky_render` — Renders procedural sky from `thalos_celestial`
  catalog (stars, galaxies as GPU meshes).
- `maneuver/` — Maneuver node placement/editing UI. Delta-v handles
  in local reference frame.
- `flight_plan_view/` — Renders the trajectory branch stack as
  on-screen tracks.
- `camera` — KSP-style orbit camera.
- `shipyard_editor/` — the **in-game ship editor**: the UI-agnostic editor
  logic in the `core` submodule (`ShipEditorCorePlugin`, moved here from the
  former `thalos_shipyard::editor`) plus its native Bevy-UI front-end (`scene`
  + `ui`), over the `thalos_shipyard` construction model. A **separate scene**,
  not an `AppState`: `ShipyardEditor::open` is a `SimClock` pause source, and
  the three `SimStage` sets (Physics/Sync/Camera) + the HUD update systems are
  gated on `shipyard_editor::editor_closed` (configured in `main.rs` /
  `hud/mod.rs`), so **no game logic runs while the editor is open** — the
  flight world is suspended, not just hidden. While open, the scene cameras
  deactivate and a dedicated editor camera renders the build world on
  `coords::EDITOR_LAYER` (layer 5) — build entities (marked `EditorPart`,
  layer-stamped each frame) never bleed into flight/map views and survive
  close/reopen. All gameplay input contexts deactivate and the
  `ShipyardContext` (orbit/click) activates (`input.rs` gate); Escape-close is
  owned by `pause_menu::handle_escape_input`'s priority chain. UI is native
  Bevy UI in `HudTheme` style (`shipyard_editor/ui/`):
  parts palette, parametric slider inspector, staging readout, top bar with
  ship-name text field + mirror/snap/layout toggles, a **▶ Launch** button,
  status bar. Entry: the pause menu's SHIPYARD button, or `just game
  shipyard`. When adding game systems that
  aggregate shipyard part components, filter `Without<EditorPart>` (see
  fuel/staging/local_physics).
- `relaunch` — fly the editor's current design without a process restart
  (the editor's **Launch**). Carries the live build's collected
  `ShipBlueprint` (no file round-trip) to a two-phase relaunch: despawn the
  old `PlayerShip` + bubble, reset the sim, seat the craft into a scenario
  (cruise for aircraft, orbit otherwise — reusing the destruction-respawn
  helpers), then rebuild the craft via the shared
  `ship_view::build_player_ship` core. The new craft carries no `EditorPart`,
  so staging/bubble rebuild through the existing systems. Runway relaunch is
  deferred (one-shot terrain-aware startup placement).
- `base_editor/` — the **in-world surface base editor** (Cities:Skylines-style):
  pick a building site on the real terrain, flatten the land into a level pad,
  then click-and-place / edit placeholder buildings on it. Unlike the shipyard
  editor it is an **in-world overlay**, not a separate scene — the planet stays
  visible, the sim pauses (`BaseEditor::open` is a `sim_clock` pause source —
  like a warp-0 pause, so only the **Camera** `SimStage` set gates on
  `base_editor::base_editor_closed`; Physics/Sync keep running so the ground
  stays streamed/lit, just frozen), and the existing `ShipCamera` is
  repositioned to a god-view (`camera.rs`, WASD-pan + orbit/zoom) rather than
  swapped. Reuses the `StructureSite`/`StructureRegistry`/`apply_structure_flatten`
  layer (`structures.rs`, grown with `BaseSite`/`Building`/`Launchpad`/`Tank`
  kinds + `remove`/`update`) and the runway's per-frame f64 anchoring. Entry: the pause menu's
  SURFACE BASE button; Esc closes. The one new mechanism is **live runtime
  terrain invalidation** — UDLOD has no per-tile re-bake path, so a flatten
  installed after tiles stream is applied by `TerrainRebuildRequest`
  (`rendering::terrain_residency`) despawning + respawning the body terrain
  (reusing the persistent `TerrainFlattenRegistry` handle). Tab toggles placing
  a building vs a **launchpad** (a craft can be placed on it with **L**, reusing
  the runway's parked-placement helpers — runway spawn is horizontal-on-gear, a
  launchpad spawn is **vertical** nose-up), and a **typed auto-connection
  network** (`base_editor::connections`: `ConnectionKind` = taxiway / apron /
  road / crawlerway, each its own width+material; line networks are MST or
  explicit-edge strips, aprons are filled rects) regenerates as structures
  change. Runways are now **parametric + multi-instance**
  (`StructureKind::Runway { half_length_m, half_width_m }`, rendered + collided
  from the registry by `runway.rs`, with **ICAO-font designator numbers 01–36
  painted from the true compass heading**; every runway lies on the
  one shared basin tangent plane via `RunwayFrame::center_offset`, so an offset
  strip stays flush with the flattened ground). The
  **default base is a spaceport on a flat basin**: `runway::build_spaceport`
  registers a wide `BaseSite` basin, registers **two runways in a V** (the 5 km
  primary + an angled crosswind secondary diverging 30° from near the primary's
  threshold, on the side opposite the launch complex; the divergence gives each
  strip its own designator numbers, so no L/R suffix pair) draping on it,
  then `base_editor::spawn_default_base` authors the rest coplanar as **one
  core campus** on the launch-complex side (nothing between the runways): two
  launchpads with clearing, per-pad flame diverters + tank/fuel farms (`Tank`
  cylinders), a VAB-scale building, a row of hangars standing on a **large
  apron auto-derived from the hangar row**, blockhouses/ops, and the typed
  networks — a straight full-length core parallel taxiway plus three **curved
  link taxiways** (`connections::spawn_authored_path`, waypoint polylines with
  circular corner fillets) that cross the primary strip **split at the runway
  edges** (stub stops at one side, curve resumes at the other — no paving under
  the strip) and sweep tangentially onto the secondary's parallel taxiway,
  straight connectors + threshold **holding pads**, a curved landside road, and
  VAB→pad crawlerways — so the surface scenarios present a whole base. Spawn
  points are **intrinsic** to runways/launchpads, and the **create-craft→fly
  flow** is wired: the shipyard/VAB **LAUNCH** rebuilds the craft into an orbit
  hold then drops into an in-world **launch-point picker** — a
  `BaseEditorMode::SelectLaunch` god-view (`base_editor/launch_select.rs`) where
  clicking a runway lands the craft horizontal-on-gear and a launchpad lands it
  vertical-nose-up, via the shared placement cores (`runway::place_on_runway` /
  `place::place_on_launchpad`). The spaceport is built lazily on the first launch
  (`runway::build_spaceport`, extracted from `finish_runway_spawn`) behind a brief
  loading pass, and persists thereafter. Every **launchpad now carries its own
  kinematic collider** (`spawn_structure_entity`) — required because a
  `RunwaySite` makes `local_physics::terrain_patch` skip the generic ground patch
  on that body, so a pad-launched craft would otherwise fall through. The base's
  flattened ground reads as a **grass lawn**
  (thick short `GrassProfile::lawn`) with its paved/built footprints (runway,
  launchpads, buildings, tanks) **cleared** — the building-terrain scatter layer:
  `body_render::ground::scatter`'s `ScatterRegion`/`ScatterTreatment`/`classify_scatter`,
  derived from the `StructureRegistry` by `rendering::grass` and honoured by the
  grass tile builder (the seam future base trees/props extend). See
  `docs/base_building.md` *Ground scatter*.
- `space_center/` — the **KSP-style Space Center hub**: the god-view overview you
  land in when you press **PLAY** on the start screen. Like the base editor it is
  an **in-world modal mode** (a `sim_clock` pause source, `space_center_closed`
  gates only the **Camera** `SimStage`), not an `AppState`. It is the *hub* the
  facility editors hang off: a `space_center::ui` panel offers **EDIT BASE** (→
  base editor on the existing basin) and **VAB** (→ shipyard editor), plus a
  **building-selection framework** (`space_center::select`) — left-click raycasts
  the pad sphere for the nearest selectable structure, outlines it, and entering
  an already-selected **facility** building opens it. Facilities are tagged on the
  `StructureSite` via `structures::Facility` (only `Vab` wired today; the seam for
  runway/pad launch, tracking station, admin). Entry: start-screen PLAY (a
  **clean start** — builds the spaceport *base only*, `runway::build_spaceport`
  with **no craft parked**, behind the loading screen via
  `HubSpaceportBuild`/`finish_hub_spaceport`, then opens the hub on reveal), or
  the in-flight pause menu's **SPACE CENTER** button. The only way to fly is to
  launch a craft yourself from the **VAB** (→ `base_editor::launch_select`'s
  pick-a-runway/pad flow) — nothing is loaded on the pad by default. **EXIT /
  Escape** from a PLAY-opened hub returns to the **main menu**
  (`SpaceCenter::return_to_menu`, since nothing is flying yet); from a
  pause-menu-opened hub it returns to that flight. `ReturnToSpaceCenter` makes a
  facility entered *from* the hub return to the hub on close (a VAB **Launch**,
  which queues a `RelaunchRequest`, instead drops to flight). The god-view camera
  is the **shared** `god_view` module (`GodViewOrbit` + `drive_god_view`),
  extracted from the base editor so both modes share the exact 3/4 orbit/zoom/pan
  feel and one shadow-focus override. Start screen (`main_menu`) is now the usual
  **PLAY / SETTINGS / QUIT** with the direct-scenario shortcuts tucked in a
  collapsible **QUICK START / DEV** submenu.
- `settings` — **unified settings persistence**. All user preferences live in
  one `settings.ron` (`AppSettings { window, graphics, units }` — a section per
  domain). Storage location switches on build profile (Bevy 0.19's
  `bevy::platform::dirs::preferences_dir`): **debug** keeps it project-local at
  the gitignored `user/settings.ron` (easy to inspect/reset during dev),
  **release** uses the OS app-data dir `<preferences_dir>/thalos/settings.ron`.
  `settings::load()` runs in `main()` before the app (the window section shapes
  the initial `Window`) and **migrates the legacy per-domain files**
  (`user/{settings,graphics,units}.ron`) on first run — a sectioned vs flat-RON
  heuristic disambiguates the new unified file from the old window-only
  `settings.ron` at the same path. Each domain still gets its **own Bevy
  resource** (`WindowSettings`/`GraphicsSettings`/`UnitsSettings`, inserted in
  `main()`) so every consumer is unchanged; `AppSettingsPlugin`'s one autosave
  watches all three and rewrites the file when any value changes (value-compared,
  so an open settings tab doesn't churn it). The per-domain modules below own
  only their resource + runtime behaviour, not the file IO.
- `window_settings` — window/display preferences (mode, windowed resolution,
  vsync, fullscreen monitor, user UI scale; the `window` section). Edited live
  from the settings menu's Window tab (`settings_menu`).
  `THALOS_WINDOW_MODE` / `THALOS_WINDOW_SIZE` / `THALOS_VSYNC` are *session
  overrides* (`overrides_from_env`: they win for the run, grey out their UI
  control, never persist), and `THALOS_SCALE` stays a pure env diagnostic.
  `apply_window_settings` pushes settings onto the primary `Window`
  (value-compared) and writes back windowed drag-resizes (the unified autosave
  persists them); `apply_ui_scale` (which absorbed the former
  `compensate_fractional_ui_scale`) multiplies the user UI scale into the
  fractional-HiDPI crisp-text snap and drives `UiScale`. Caveat: runtime mode
  switches recreate the swapchain, which the known flaky Windows driver path
  (generic surface-acquire panic in Bevy's `prepare_windows`) can turn into a
  crash — pre-existing platform issue, newly reachable at runtime.
- `graphics_settings` — graphics/rendering preferences (the `graphics` section),
  edited live from the settings menu's Graphics tab (`settings_menu`). Knobs:
  the **volumetric-cloud toggle** (`GraphicsSettings::clouds`), read by
  `rendering::clouds::drive_clouds` — when off it parks the cloud raymarch via
  the no-cloud-body path (`ActiveCloudBody = None` → blank fallback bound onto
  `BodySkyMaterial`), so the sky renders clear at near-zero GPU cost; the grass
  toggle; and the MSAA level. Add new render knobs here as fields + a control in
  `show_graphics_tab`.

Systems run in `SimStage` order: `Physics → Sync → Camera`
(configured in `main.rs`), ensuring deterministic state flow each
frame. Enhanced input intent collection runs in `PreUpdate` before these
sets. Simulation pause is an explicit clock boundary, not a global Bevy
clock pause: `crates/game/src/sim_clock.rs` owns `SimClock`, whose sole
writer folds Escape pause, destruction scenario picker, freecam, the
shipyard editor, and warp pause into a zero sim delta. Canonical stepping, local physics ownership,
resource burn, and grounded EVA motion consume `SimClock`; presentation
and UI animation use `Time<Real>` directly. Do not reintroduce pausing
`Time<Virtual>`/plain `Res<Time>` as the game-wide sim pause switch.

### Terrain gen crate (`crates/terrain/`)

Cubemap-based procedural surface generation. No Bevy dependency.

- `TerrainConfig` — top-level enum in `terrain_config.rs`: `None`,
  `Feature(FeatureTerrainConfig)` for archetype-driven bodies (Mira,
  Vaelen, Thalos, Pelagos…), `Ocean(OceanTerrainConfig)` for
  flat-water placeholders. Current P2A.5 migration state: Thalos is
  authored as an ocean-bearing `OceanicTerrestrial` prototype; its ground
  LOD uses `RuntimeTerrainDetail::OceanicContinental` so the same signed
  continent/seabed evaluator defines bake height, runtime mesh height, and
  collider height instead of the legacy P0 HMF detail cascade. Runtime
  ground height is LOD-invariant in this slice to avoid contour-like
  parent/child tile handoff steps. Water is not a terrain material:
  underwater terrain keeps seabed albedo/material/roughness, and ocean
  color/reflection/optical absorption come from the separate water renderers.
  The fuller tectonic/hydrology Thalos route is parked for a later P2 slice.
- `compile_terrain_config(...)` — single entry point. Builds the
  optional `TectonicSystem` from `TectonicConfig`, compiles the
  static surface (via `feature_compiler` or `compile_ocean`), and
  compiles dynamic layers. Returns a `PlanetSurface`.
- `PlanetSurface` (in `static_surface.rs`) — the top-level product:
  - `static_surface: StaticSurfaceData` — immutable, cacheable
    cubemaps + `IcoBuckets` spatial index + optional `sea_level_m`.
  - `dynamic_layers: DynamicSurfaceLayers` — terrain-owned
    time-varying overlays (seasonal ice, unconsolidated aeolian
    bedforms).
  - `tectonics: Option<TectonicSystem>` — plate graph regenerated on
    each load (cheap, kept outside the static-surface cache).
- `BodyBuilder` — mutable build-time state. Stages mutate this
  (height/albedo accumulators, craters, volcanoes, channels,
  materials, detail noise params).
- `Stage` trait — `name()`, `apply(&mut BodyBuilder)`. Each stage is a
  pure transform; the feature compiler invokes them directly and sets
  `stage_seed` before each `apply()` so there's no ambient state.
- `tectonics/` — plate graph (`Plate`, `PlateKind`, `Boundary`,
  `BoundaryKind`), spherical mesh, plate-derived fields. Currently
  consumed by `ColdDesertFormerlyWet` (Vaelen) and by the parked fuller
  `AgingOceanicHomeworld` Thalos route; the active P2A Thalos prototype
  omits tectonics.
- `surface_field` / `aging_oceanic_field` / `cold_desert_field` /
  `vaelen_field` — continuous archetype surface fields sampled into
  the cubemap accumulators.
- `sample()` / `sample_static_surface()` / `SurfaceSample` — sampling
  contracts for reading finished surface data.

### Celestial crate (`crates/celestial/`)

Procedural sky model. Pure Rust, no Bevy. Works in physical
quantities (flux, temperature, SED) — never pre-baked RGB.

- `Universe` — collection of `Source` objects (stars, galaxies, nebulae)
- `Spectrum` — spectral energy distributions: `Blackbody`,
  `PowerLaw`, `Tabulated`; `Passband` filtering
- `generate/` — procedural star field and galaxy placement
- `render/` — cubemap baking and telescope PSF

### Shipyard crate (`crates/shipyard/`)

Parametric ship **construction model** (Bevy, no UI). Owns *what a craft is* —
parts, resources, blueprints, attach tree, sizing/stats/staging, geometry mesh
builders — not the editor (that's `thalos_game::shipyard_editor`, see the Game
crate section). The full next-gen construction model (planes/ships/stations from
one Module primitive) is specced in `docs/construction.md`; **Slices 1–2
(parametric wing, wing-pylon jet nacelle, and scalar intake flow) plus
visually-actuating control surfaces have landed** — see that doc's §0 for the
status boundary.

- `AttachNode` / `Ship` — ECS tree structure for ship assembly
- `Part` trait — `CommandPod`, `Engine`, `AirIntake`, `FuelTank`,
  `Decoupler`, `Adapter`, `Wing`
- **Two placement capabilities.** `Attachment` is end-node stacking (the
  rocket path: mate `top`/`bottom`, diameter propagates). `SurfaceMount`
  is surface/footprint placement: sit a part on a host's skin at
  `(station, angle)`, **opting out of diameter propagation** (wings,
  wing-pylon nacelles; later gear). It is a *parallel* component, not a
  fork of `Attachment` — graph traversals that need every part union both.
- **KSP linked symmetry** — `SymmetryGroup { id, role }`. Placing a
  footprint part under the editor's Mirror (2×) mode stamps a **real
  counterpart entity** (separate left/right parts) linked in a group;
  `sync_symmetry_groups` keeps the mirror in lockstep with its primary
  (params copied with handed fields like `Wing.incidence` negated, mount
  reflected across X = 0), and edits/deletes act on the whole group. Nested
  symmetry stamps a part placed on a mirrored host (a nacelle on a wing)
  onto both hosts. Downstream the mirror is a plain part: meshes are
  single-panel, stats/staging count each entity once (no ×2), and the
  blueprint persists a `symmetry_group` id only so the editor re-links on
  load (the game ignores it).
- `Wing` — tapered/swept/dihedral lifting surface (span, root/tip chord,
  sweep, dihedral, t/c, incidence). `wing_mesh::build_wing_mesh` builds it
  in the host-local frame, shared by the editor and the game's `ship_view`.
  **Control surfaces are `Wing` parameters** (`Wing::control_surfaces:
  Vec<ControlSurface>` — ailerons/elevator/rudder/flaps/spoilers, each a
  trailing-edge spanwise window + chord fraction + deflection limit), not
  separate parts:
  `build_wing_mesh` notches them out of the loft and `build_control_surface_mesh`
  meshes each as a separate hinged sub-mesh about a consistently-oriented
  hinge axis (`control_surface_geometry` is the shared seam a future
  per-surface force model reads). The game animates the attitude surfaces from
  the fly-by-wire command (`RealizedControl::command`, *not* the allocated
  aero fraction): elevators on pitch (symmetric), ailerons on roll
  (differential by mount side), rudder on yaw. `Flap`/`Spoiler` windows
  deflect instead from `thalos_game::flight_config::FlightConfig` — the
  three-detent flap lever (F extend / R retract) and the B brakes latch
  (wheel brakes + spoilers, KSP-style) — and their authored window geometry
  derives the craft's flap/spoiler ΔCL/ΔCD in `build_ship_aero_config`
  (`docs/aerodynamics.md` *Flight configuration*). **Per-surface control
  authority**: the per-axis control coefficients likewise derive from the
  authored aileron/elevator/rudder windows (deflection lift × real moment arm
  about the CoM — `derive_control_coefficients`), so surface sizing and
  placement show up in handling; the *moment structure* stays the whole-body
  stable model — forces are never an emergent per-surface strip sum, which
  pumps energy (`docs/aerodynamics.md` *Per-surface control authority*).
  Gear will be its own footprint part. No flight model yet (M6).
- `Engine` / `AirIntake` — rocket bells remain node-stacked; jet nacelles can
  surface-mount to wings as `WingPylon` mounts with generated pylons. Ambient
  flow is modeled separately from resources: engines declare required intake
  kind/area, nacelles may carry built-in capture, and standalone intakes
  provide capture for future separated inlet/core designs.
- `ShipBlueprint` — RON serialization. Node mates serialize as
  `connections`; surface mounts as a separate `surface_mounts` list
  (`#[serde(default)]`, so pre-wing saves load unchanged).
- `sizing` — parametric node sizing (adapters/tanks scale from parent);
  surface mounts are intentionally excluded.
- `stats` — aggregate mass / CoM / MOI / Δv, plus geometry-derived
  **wing area + mean aerodynamic chord** ("will it fly" feedback; no sim).
- `staging` — pure stage derivation from decoupler topology
  (`derive_stages`) + per-stage Δv/fuel accounting
  (`compute_stage_summaries`). Single home for the staging model: the
  game's live ECS staging systems (`crates/game/src/staging.rs`) and the
  editor's staging preview (`ShipBlueprint::stage_summaries`) both call it,
  so stage boundaries never diverge. Like `stats`, no Bevy beyond glam math.

### Body render crate (`crates/body_render/`)

Unified Bevy rendering for celestial bodies — one appearance model with
regime-scaled projections. No world generation logic. Added via a single
`BodyRenderPlugin` (which composes four module sub-plugins). Four modules:

**`shading`** — the single source of truth every body-surface material
reads from. No materials of its own.
- `PlanetLightingPlugin` — registers the `thalos::lighting` and
  `thalos::atmosphere` shader libraries. The `impostor` and `ground`
  sub-plugins also add it defensively (no-op if already added) so each
  works standalone.
- `SceneLighting` / `StarLight` / `MAX_STARS` / `MAX_ECLIPSE_OCCLUDERS`,
  `AtmosphereBlock` / `CLOUD_BAND_COUNT` — scene + per-body uniforms.
- `shaders/lighting.wgsl` — WGSL mirror of `SceneLighting` plus
  `eclipse_factor`, `planetshine_sample`, `hapke_brdf`, and
  `shade_hapke_surface`. The impostor always shades through
  `shade_hapke_surface`; the ground LOD picks a path per body via
  `TerrainShadingStyle` (`body_terrain.wgsl`, carried in
  `BodyTerrainExtras.inspection.y`, derived from the terrain archetype at
  spawn): airless `Regolith` bodies (Mira) route through the same
  `shade_hapke_surface` so they reconverge with the impostor across the LOD
  swap, while `Vegetated` bodies (Thalos) use the ground-only rough-dielectric
  BRDF + ecological albedo bands.
- `shaders/atmosphere.wgsl` — WGSL mirror of `AtmosphereBlock`,
  `integrate_atmosphere`, `composite_clouds`, `apply_limb_darkening`.

**`impostor`** — distant billboard materials.
- `PlanetRenderingPlugin` — registers `PlanetMaterial`,
  `GasGiantMaterial`, `RingMaterial`, `SolidPlanetMaterial`.
- `PlanetMaterial` (cubemap height/albedo, `assets/shaders/planet_impostor.wgsl`),
  `SolidPlanetMaterial` (`solid_planet.wgsl`), `GasGiantMaterial`
  (`gas_giant.wgsl`), `RingMaterial` (`ring.wgsl`). These four shaders
  are AssetServer-loaded from the workspace `assets/shaders/` dir.
- `bake_from_body_data()` — consumes `terrain::PlanetSurface` →
  `PlanetTextures` for upload into a `PlanetMaterial`.

**`ground`** — the `thalos_udlod`-backed terrain LOD (former
`terrain_render`).
- `ThalosTerrainPlugin` — adds `thalos_udlod::TerrainPlugin` +
  `BodyTerrainMaterial`/`BodySkyMaterial`/`BodyWaterMaterial`/`GrassMaterial`,
  embedding their `src/ground/*.wgsl` via `embedded://thalos_body_render/ground/…`.
- `PipelineTileProvider`, `rendered_height_*`, the `HeightSource` family,
  and rendered-height patch utilities used by M5 colliders.
- `vegetation` — near-camera grass-blade decoration layer for vegetated
  bodies: cube-sphere grass-tile lattice + batched blade-mesh builder
  (placement reuses the tile baker's grass-mask gate against the body's
  `HeightSource`) + `GrassMaterial`. Driven per-frame by
  `thalos_game::rendering::grass` (runway-style f64 body-fixed anchoring,
  revision-based rebuilds). See `docs/terrain.md` *Vegetation decoration
  layer*.
- `ground_patch` — `GroundPatchMaterial` / `GroundPatchMaterialPlugin`: a flat,
  sky-model-lit ground plane (the shared `thalos::lighting` dielectric BRDF, not
  the UDLOD stack) that **receives** the same cascaded sun-shadows trees cast —
  the diorama ground for the object preview, and the seed for the planned larger
  composed scenes. Deliberately simple: a flat-ground analogue of
  `body_terrain.wgsl` for tooling, sharing `TreeMaterial`'s cascade binding
  layout so one shadow rig feeds both.

**`clouds`** — spherical, body-fixed volumetric cloud render mechanism.
- Owns the absorbed `bevy-volumetric-clouds` compute pipeline, generated
  Perlin-Worley/Worley textures, cloud colour/distance targets, and cloud-local
  temporal history; upstream MIT attribution lives beside the module.
- Consumes a cube `CloudWeatherMap` uploaded from the active body's canonical
  `CloudWeatherField`. It does not create weather or choose a body.
- The game-side `rendering::clouds` driver selects the nearest authored cloudy
  body and projects `CloudClimate`/environment state into view uniforms. Near
  composition stays in `BodySkyMaterial`; the first orbit projection is in
  `SolidPlanetMaterial`. See ADR-0009 and `docs/clouds.md`.

`body_render` is the **sole consumer** of the vendored `thalos_udlod`,
re-exported as `thalos_body_render::udlod` (`{prelude, math, big_space}`); no
other crate depends on the fork directly. Replacing the ground backend stays
localized to the `ground` module + that re-export.

### Data flow

```
assets/solar_system.ron + assets/bodies/<body>.ron
  → [parsing] SolarSystemDefinition (physical + orbit + TerrainConfig + TectonicConfig)
  → [PatchedConics] body positions at any epoch t
  → [Simulation::step] per frame → canonical CraftState + authority, consumes ManeuverNodes
  → [solar_system_state::sync_solar_system_state] canonical per-frame BodyStates
       + per-body environment state (`CloudWeatherField`, dynamic surface;
         later weather evolution/wind/tides)
  → [propagate_flight_plan / propagate_branch_stack] synchronous prediction
       → FlightPlan / TrajectoryBranchStack (Actual + Projected branches)
  → [map_view] MapSnapshot → map rendering, maneuver UI, collision warnings
  → [rendering::real_space] BigSpace grids → ship-view body / camera transforms

[terrain::compile_terrain_config]
  → PlanetSurface { static_surface, dynamic_layers, tectonics }
  → [body_render::bake_from_body_data] PlanetTextures
  → PlanetMaterial uploaded to GPU; impostor billboard renders
```

### Design invariants

- **One propagator everywhere.** Live stepping and prediction route
  through the same `ShipPropagator` (today, `KeplerianPropagator`).
  Never split them or numerical divergence appears between "where
  ship is" and "where it will be."
- **One craft state, one authority.** Each craft has one `CraftState`
  and one `AuthorityMode`; presentation code reads snapshots or
  accessors, not parallel transform-owned state.
- **One solar-system state between projections.** `SolarSystemState` is the
  frame-local source for evaluated `BodyState`s and mutable per-body
  environment state. Render entities, map snapshots, impostor materials,
  terrain grids, sky/halo passes, and future weather/tide/wind systems are
  projections or mutators of this resource, not separate owners of equivalent
  state.
- **`BodyTrajectoryProvider` is the abstraction boundary.** Body
  positions are always queried through this trait. `PatchedConics`
  is the current impl; a precomputed ephemeris could replace it
  without touching simulation or rendering.
- **Physics crate has no Bevy.** All physics logic must remain in
  `thalos_physics_canonical`. `thalos_game` is only presentation and input.
  This extends to the other pure-Rust libraries (`thalos_world`,
  `thalos_terrain`, `thalos_celestial`): none may pull in Bevy, even
  transitively. The boundary is CI-guarded — `.github/workflows/ci.yml`
  runs a `cargo tree` check on every push/PR and fails if a real Bevy
  crate enters any of those four trees. (`bevy_erosion_filter` is allowed:
  terrain uses its pure-glam `cpu` module with `default-features = false`,
  so the crate is present by name but pulls no Bevy engine crate — keep
  that flag.)
- **Single-writer resources.** Frame-local projection/snapshot resources
  have exactly one writing system, named in the resource's doc comment as
  **Sole writer:**. Today: `SolarSystemState` ← `sync_solar_system_state`,
  `MapSnapshot` ← `update_map_snapshot`, `CraftStateMirror` ←
  `refresh_craft_state_mirror`, the per-craft `CraftRegimeState` component ←
  `regime::resolve_regime` (the regime decision record every per-frame
  ownership/warp/prediction consumer reads — see `docs/regimes.md`), and
  `AvianAuthority` ← `compute_avian_authority` (a derived projection of
  that record). Every other system reads. Don't add a second writer; if a
  resource needs to be mutated from elsewhere, route through an accessor on
  the sole writer or reconsider the ownership.
- **Map view is decoupled.** Map systems read `MapSnapshot`, projected
  body states, and trajectories. They do not share or mutate
  real-space rendering entities.
- **Real-space rendering lives under BigSpace.** One BigSpace root
  uses 1 km cells in the system frame; per-body grids are positioned
  with `Grid::translation_to_grid`, and the active ship camera owns
  `FloatingOrigin`.
- **`TrajectorySample` carries its own metadata.** `anchor_body` +
  `ref_pos` travel with each sample so the renderer can pin to its
  parent body without a per-sample ephemeris query.
- **Terrain gen stages are pure transforms.** Each `Stage` reads /
  writes `BodyBuilder` only. The feature compiler is the only caller;
  it sets `stage_seed` before each `apply()`. No ambient state.

### Assets

- `assets/solar_system.ron` — system-level definition: bodies'
  physical + orbital specs. RON format with `#![enable(implicit_some)]`.
- `assets/bodies/<lowercase_name>.ron` — per-body terrain + tectonics
  configuration. Loaded alongside the system file by `parsing`.
- `assets/parts.ron` — shipyard parts catalog.
- `assets/shaders/planet_impostor.wgsl` — procedural-body impostor
  (3-layer: baked cubemap low-freq, SSBO mid-freq features,
  shader-synthesized high-freq detail).
- `assets/shaders/solid_planet.wgsl` — solid-color billboard impostor.
- `assets/shaders/gas_giant.wgsl` — gas giant cloud / haze / rim
  rendering.
- `assets/shaders/ring.wgsl` — planetary ring system shader.

### Documentation (`docs/`)

Each major system has a unified spec doc. **The map + cross-ref convention
(`clean §N` = architecture_cleanup.md, `gfx §N` = graphics_fidelity.md,
`ADR-NNNN`, `INC-NNNN`) live in `docs/README.md`.** Steering docs —
`backlog.md` (the queue), `adr/` (decision log), `incidents/` (post-mortems) —
are covered in "Steering, decisions & incidents" above.

- `tooling.md` — toolchain policy and local-only compiler tuning recipes
  for Windows fast incremental builds, macOS incremental workarounds, and
  backend/linker experiments.
- `simulation.md` — simulation architecture (orbital mechanics,
  authority, time warp, map decoupling, big_space, Avian local
  bubble, navball velocity reference frames). Target design, plus a
  background note on layered astrodynamics methods.
- `regimes.md` — the per-craft `CraftRegime` resolver — one sole-writer
  classification (craft capabilities × environment) replacing the
  scattered `AvianRole`/`manage_authority`/warp-gate/`VesselKind`
  predicates; walking reframed as a locomotion mode (jetpack-EVA
  becomes a normal craft later); backend seam policy (Avian stays
  behind a swappable executor layer; parry-direct re-evaluated at
  Phase C). **Phase A2 (shadow mode) landed 2026-06-12**: pure
  classifier in `thalos_physics_canonical::regime`, sole-writer
  resolver + Reflect-registered drift checker in `crates/game/src/regime.rs`
  (`RegimeDriftDiagnostics`); legacy machinery still drives everything.
  Orbit + EVA scenarios verified drift-free; remaining scenario matrix
  + the A3 consumer ports are next.
- `boot.md` — boot pipeline: the `AppState` graph
  (`Loading` → `MainMenu` | `Running`), the declarative `LoadingTracker`
  step registry (how to add a loading step), the start screen and its
  in-place scenario starts, and the runtime-scenario invariants
  (explicitly-armed deferred placements, per-ship engine lighting,
  `relaunch_idle`).
- `ui_flow.md` — the screen/mode flow and its **in-flight unification**: the
  `GameContext` sub-state (`SpaceCenter | Vab | BaseEditor | Flight`, nested under
  `AppState::Running`) that replaces the cross-referencing `.open` booleans as the
  single in-game mode authority (one camera/HUD/pause/Escape owner, a return-stack,
  `ViewMode` demoted to Flight-only). Migration is staged shadow→flip→invert like
  `regimes.md`; **Phase 1 (shadow) landed 2026-07-04** in `game_context.rs` —
  `GameContext` is derived from the booleans and not yet consumed.
- `surface.md` — surface gameplay in two parts: **on foot (EVA)**
  (ground physics, body-fixed pose, the `HeightSource` interface,
  surface map view) and **landing & impact destruction** (landed-ship
  descent, terrain collision via `SweptCcd` anti-tunneling, and
  whole-craft impact destruction, `ShipParameters::impact_tolerance_m_s`
  → `Simulation::is_destroyed`). Merged from the former
  `surface_gameplay.md` + `landing.md`.
- `surface_local.md` — **implemented for ships (2026-06)**: the
  surface-local tangent frame (SLF) — a body-fixed Y-up frame anchored at
  a surface point, re-anchored on drift — as the ship near-surface physics
  regime, replacing the body-centered-inertial *ship* contact bubble with
  a **solid** ground collider (terrain heightfield / runway cuboid) and the
  gear-as-sole-ground-contact model. Also generalizes the runway into
  data-driven terrain-anchored **structures** (`crates/game/src/structures.rs`:
  `StructureSite`/`StructureRegistry`, the `apply_structure_flatten` path).
  §10 records what shipped vs the design: **EVA stayed on its body-centered
  kinematic seam** (intentional — it has no collider and gains nothing from
  the SLF), the runway keeps a solid cuboid collider, the physics step is
  still fixed-timestep, and the backstop demotion / async heightfield
  rebuild are deferred follow-ups. (Runtime structure placement is **now
  built** — see `base_building.md`.)
- `base_building.md` — the **in-world surface base editor** (`base_editor/`):
  pick a site, live-flatten the land, click-and-place / edit buildings. The
  in-world-overlay design, the god-view camera, the runtime terrain-invalidation
  MVP (despawn/respawn the body terrain), the multi-flatten `FlattenRegion`
  change, and the ordered follow-ups (launchpad, auto-connections, slider
  inspector, disk persistence, scoped-AABB invalidation).
- `construction.md` — next-gen shipyard / construction model design:
  one Module primitive (end-node / footprint+morph / end-cap / host /
  connector / reservation), stationed-loft fuselages and wings, a
  separate internal/loadout layer (compartments, role-fills, cargo
  doors), generalising the rocket-only shipyard to planes, ships, and
  stations. Target: M6. Design-only, no code yet.
- `input.md` — enhanced-input context model, binding file rules, and
  per-binary intent resources.
- `hud_widgets.md` — the **MFD slot**: the contextual/customizable ship-view
  HUD widget framework (`hud/mfd/`). The `WidgetKind`/`FlightContext`/
  relevance model, the container-only-`HudPanel` + selector-owns-visibility +
  one-pass-one-visible invariants, how to add a widget, and the shared
  `local_enu_basis` + navigation-display projection recipe.
- `control.md` — the **fly-by-wire control layer** (`thalos_control` +
  `game::control_bus`): the demand vocabulary + source priority, the one
  attitude controller (quaternion `Hold` PD / nose `PointNose` PD,
  replacing the deadbeat SAS damper), the effector allocator (reaction
  wheels + aero surfaces from one command), and the warp/EVA/RCS/throttle
  extension points.
- `terrain_macro.md` — **large-scale terrain plan**: the scale-ownership rule
  (everything ≥ ~250 m wavelength comes from the f64 `ProceduralSurface` bake;
  the 4 km-wrapped f32 shader noise carries only fine detail), the Phase 1
  macro-landcover implementation (moisture in `SurfaceSample` → albedo-alpha
  attachment → shader/grass consumers), and the Phase 2 (climate → biomes) /
  Phase 3 (plate margins, island arcs) designs.
- `terrain.md` — the **consumer-side terrain contract**: the tile primitive
  (the black-box boundary), ground-LOD rendering, surface shadows, colliders,
  and dynamic features. Terrain *generation* is treated as a black box behind
  the tile contract; its previous design is archived (see below) and a new
  generator is being built against the contract.
- `terrain_lod_optimization.md` — the **udlod fork's Thalos-specific optimization
  pass**: what the fork dropped vs upstream (only the preprocessing path — but
  with it, all persistence), the tile **cache** design (memory over disk over
  synthesis; the per-request `NamespaceFn` that makes stale tiles unreachable
  rather than wrong), provider-owned mip generation, per-attachment resolution,
  screen-space-error refinement, the behind-view streaming cull, the dead-prepass
  deletion — and why **GPU tile production** is blocked on a height-authority
  decision rather than on effort.
- `vegetation.md` — the **planet-scale vegetation plan**: grass/ground-cover to
  the horizon, shrubs, and trees as one unified consumer-side layer over the
  tile contract. Representation cascades that end in the terrain albedo,
  deterministic hashed placement, the constant-coverage rule, the tile-LOD
  clipmap, octahedral tree impostors, instancing paths (entity auto-batch →
  instanced material → GPU-driven indirect; not meshlets/Nanite for foliage),
  and the phased roadmap. Generalizes the shipped grass near-ring (see
  `terrain.md` *Vegetation decoration layer*).
- `atmosphere.md` — gas giants, rocky-atmosphere single-scattering
  raymarch (unified per-body fullscreen pass with scene-depth
  coupling for aerial perspective), Kármán-line authoring, ocean
  rendering, IBL/reflection probe. (Atmosphere *rendering*; the
  *physics* density/drag model is in `aerodynamics.md`.)
- `aerodynamics.md` — atmospheric flight forces via the native
  `thalos_physics_canonical::aero` whole-body model (drag + bluff-body
  weathervane stability; lift/control for planes): the per-body density model,
  force-only bubble coupling, the CoM/airspeed integration invariants, the
  in-atmosphere `Full`-role trigger + warp clamp, the transonic wave-drag wall
  (Korn-derived drag-divergence Mach from authored wing sweep/thickness) +
  air-breathing jet thrust lapse, the shallow flight-configuration layer
  (three-detent flap lever + brakes-driven spoilers, authored as wing
  `Flap`/`Spoiler` control-surface windows), and per-surface control
  authority (per-axis control coefficients derived from the authored
  aileron/elevator/rudder windows and their CoM moment arms). (Replaced the
  former vendored LGPL `avian_fdm` crate.)
- `celestial.md` — celestial sphere design: source model, spectrum,
  generation, rendering pipeline.
- `tooling.md` — Rust toolchain policy and local developer tooling notes.
- `lore/solar_system.md` — per-body reference with scale philosophy
  (hybrid 1:1/1:2/1:3 scale rationale) and formation scenario.
- `lore/civilization.md` — civilization, narrative progression
  phases, resource economy.

Archived (superseded terrain-*generation* design — reference only, see
`docs/archive/README.md`):

- `archive/terrain-generation-legacy.md` — the old feature-compiler chapter +
  v2 backlog extracted from `terrain.md`.
- `archive/planet-generation-pipeline-spec.md`,
  `archive/planet-generation-pipeline-migration.md`,
  `archive/planet-generation-method.md`,
  `archive/terrain-generation-cascade.md` — old field-DAG pipeline design,
  migration sequencing, authoring method, and semantic cascade model.
- `archive/gen/` — research surveys + aesthetic targets + per-body process
  notes (`terrestrial_pipeline_research.md`, `planet_aesthetics.md`,
  `dunes.md`, `vaelen_processes.md`). `planet_aesthetics.md` still captures
  visual targets the new generator should aim at.
