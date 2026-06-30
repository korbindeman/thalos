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

## Current focus: graphics fidelity

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

- **Shadows** ✅ — cascaded sun-shadow via `thalos::shadow`; terrain/trees/grass/
  rocks receive, trees+rocks cast. *Next:* craft/structures as casters+receivers
  (the core "everything interacts" fix), stable CSM, terrain horizon self-shadow.
- **Landcover + palette / aerial recession / moonlight** ◐ — in `thalos::lighting`
  / `thalos::landcover`; awaiting screenshots. Moonlight converges into F1 (the
  two moon models become one); aerial recession folds inside `shade_surface`.
- **Lighting-input unification (F1), exposure (F2), sky-view-LUT→IBL (F3/F4), AO
  (F5), shadow-rig unification (F6)** ☐ — this sprint's foundation.
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
| `thalos::grass_displace` | `grass_blade_world_pos`, `grass_tuft_alpha` (shared with depth-prepass) |

When a palette or BRDF constant moves, it moves in one place.

## Commands

```bash
just game                 # cargo run -p thalos_game — boots to the start screen
                          #   (scenario picker / shipyard / settings; naming a
                          #    mode below skips it, as does THALOS_AUTO_RUN=1)
just game orbit           # ship in low Thalos orbit, no start screen
just game eva             # spawn on foot (EVA) on the Thalos surface instead
just game landing         # powered-descent approach over dry Thalos land
just game final           # low final approach over a flat dry Thalos patch
just game runway          # aircraft parked on the Thalos surface runway
just game runway-approach # aircraft on short final lined up with that runway
just game cruise          # Meridian at ~15,000 ft, level cruise over dry land
just game shipyard        # open straight into the in-game ship editor
                          #   (also: the pause menu's SHIPYARD button
                          #    from any running mode)
just terrain-lab          # static slippy-map terrain sketchpad at localhost:8787/tools/terrain-lab/
just preview              # headless procedural-object gallery → PNGs in
                          #   tools/preview/out/ (trees/shrub, grass, rocks/
                          #   pebbles, landcover preview). No window — agents
                          #   can run it and inspect
                          #   the images. See "Running and inspecting" below.
just preview-window       # interactive variant of `just preview`: a window with
                          #   an orbit camera (drag/scroll, ←/→ cycle, S = shot).
                          #   User-run (opens a window).
just shipyard             # standalone egui ship editor (secondary front-end
                          #   over the same thalos_shipyard::editor core)
just build                # cargo build --workspace
just test                 # cargo test -p thalos_physics_canonical
just clippy               # cargo clippy --workspace
just trace                # cargo run --release -p thalos_game --features profile-tracy
just release patch        # bump version, commit, tag, and push (patch|minor|major|x.y.z)

# Run a single test
cargo test -p thalos_physics_canonical -- test_name
```

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
finnicky, so `just game` enables `bevy/dynamic_linking` by committed default (in
the `justfile`'s `game_command`), scoped to the dev run so it never reaches
`just build`/`just trace`/release. Override `game_command` in `.env.just` to opt
out locally. The normal Windows iteration path stays on LLVM. The workspace-local
`.cargo/config.toml` and `.env.just` are ignored by Git for this purpose. The full policy plus
Windows fast-incremental and macOS workaround examples live in
`docs/tooling.md`.

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

## Terrain generation (runtime — no bake)

Terrain is **generated at runtime**, not baked. Each procedural body builds one
`thalos_terrain::ProceduralSurface` (an impl of the `SurfaceQuery` seam) in
`rendering::ground_terrain` as `ProceduralSurface::new(radius, body.id)` — a pure
analytic function of `(direction, lod)` in f64 body-local coordinates, with no
disk artifact. There is **no `just bake`, no `target/bakes/`, no cache, no
startup bake check**. Every terrain-height consumer reads the *same* surface:

- **Ground LOD render** — a `FlattenedSurface`-wrapped `ProceduralSurface` feeds
  the UDLOD tile provider.
- **Collider / camera terrain floor / runway / descent site / HUD altitude / EVA**
  — the near-surface `HeightSourceRegistry` (a GPU-atlas height mirror over the
  same `ProceduralSurface`, with a CPU fallback), via
  `HeightSource::sample_height_m`.
- **Propagator collision** — `GameTerrainRegistry` mirrors the `ProceduralSurface`.

There is **no sea-level layer**: the continent mask puts the shoreline at height
0 (the reference radius), so "sea level" is the constant **0 m** wherever a datum
is needed (dry-land checks, the camera's terrain floor). Do not read sea level
from a baked surface — that path is gone.

`ProceduralSurface` is the **Slice 0** generator (competent but simple; height
tuning and procedural materials/shading are later slices). The old baked pipeline
— `thalos_bake_dump` (the "dumps"), `thalos_body_editor`, and the game's startup
`bake_check` — has been **deleted**. The remaining baked-pipeline modules in
`thalos_terrain` (the `Feature`/`Ocean` compiler, `PlanetSurface`/
`StaticSurfaceData`, `cache`, stages/tectonics/fields) and the
`body_render` impostor-bake path are now **dead code slated for removal**; the
distant-body view is an interim solid-color impostor (Slice 6 replaces it). Do
not reintroduce a bake/disk-artifact path — extend `ProceduralSurface` (or add a
new `SurfaceQuery` impl) instead.

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
screenshot through the user. (Whole-scene composition, terrain,
lighting-in-context, and "does it feel right" still need a real `just game`
run — that stays the user's call.) Extend the gallery by adding an entry to
`object_preview.rs`; a larger composed scene (a patch of grass, a mountain
ringed by trees) is the planned next phase.

**Getting data out of a running session: write it to a file.** When you need
runtime numbers rather than a picture, don't reach for live inspection — have
the game *log* the data and read the file afterwards. JSONL (one JSON object
per line) is the house style for machine-readable runtime data. Existing
sinks:
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
match the linked `tracy-client` (Bevy 0.18 → tracy-client 0.18.x).

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

## WGSL skill

A project skill at `.claude/skills/wgsl-bevy/SKILL.md` collects
WGSL / naga (Bevy) shader pitfalls — reserved words that can't be used
as identifiers, the strict type rules, `naga_oil` import quirks, and so
on. Treat it as a **living document**: whenever you hit a WGSL error
worth remembering (a keyword you couldn't use as a variable name, a
non-obvious error message, a Bevy-specific gotcha), add the case to the
skill so the next agent doesn't rediscover it from scratch.

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
(edition 2024, Bevy 0.18, glam 0.30). Workspace crates:

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
- **`thalos_body_render`** — *(Phase 2, new)* unified celestial-body rendering, one appearance model + two backends. Three modules behind one `BodyRenderPlugin`: `shading` (shared `SceneLighting`/`AtmosphereBlock`/Hapke `shade_hapke_surface` + the `thalos::lighting`/`thalos::atmosphere` WGSL libraries), `impostor` (distant billboard materials for planets, gas giants, rings, solid bodies), `ground` (the `thalos_udlod`-backed terrain LOD: `ThalosTerrainPlugin`, `PipelineTileProvider`, `BodyTerrainMaterial`/`BodySkyMaterial`/`BodyWaterMaterial`, rendered-height patch utilities). Merged from the former `planet_lighting`+`planet_rendering`+`terrain_render`. A backend chooses geometry, never its own lighting/atmosphere/cloud math.
- **`thalos_udlod`** — vendored UDLOD terrain renderer (lives at `crates/udlod/`). Forked from [`kurtkuehnert/bevy_terrain`](https://github.com/kurtkuehnert/bevy_terrain) by Kurt Kühnert (MIT OR Apache-2.0); attribution + license files travel with the source. Edit in-tree like any other workspace crate. The original fork at `~/dev/bevy_terrain` is kept around only as a reference point for diffing against upstream; daily edits happen here. The fork is now **runtime-provider-first**: it renders sparse tile atlases fed by `TileProvider` implementations, not preprocessed Earth-style asset trees. The old GeoTIFF/preprocess/`DiskTileProvider` path has been removed; if persistent reuse is needed, build it as a Thalos cache provider/wrapper keyed by body config + tile coordinate, not as `assets/<terrain>/data/*.bin`. CPU draw-tile selection is the current correctness path because it enforces 2:1 LOD balance across cube-face seams; tile *production* is the intended GPU extension point (job queue writes directly into atlas slots, later including diffusion). **`big_space` integration is unconditional** — the upstream `high_precision` Cargo feature has been removed, along with the runtime `DebugTerrain.high_precision` toggle and the `HIGH_PRECISION` shader define / pipeline flag. The Taylor-series relative-position path (`compute_relative_position` in `shaders/functions.wgsl`) is the only viable precision path at planet scale; gating it behind a feature only forced defensive `#[cfg]` plumbing in every consumer.
- **`thalos_shipyard`** — parametric ship editor (ECS attach tree, RON blueprints). The interactive editor is split **core/front-end**: `thalos_shipyard::editor` (`ShipEditorCorePlugin`) owns all UI-agnostic editing behaviour — `EditorState` command/state hub, attach-node + surface-mount placement, KSP linked symmetry, live mesh rebuilds, tank-resize handle, placement-preview ghost, shrouds, blueprint save/load against `ships/*.ron` — and two front-ends drive it: the **in-game Bevy-UI editor** (`thalos_game::shipyard_editor`, the primary) and the standalone egui binary (`just shipyard`, secondary). Every editor-owned entity carries the `EditorPart` marker and every core query filters on it; host systems that aggregate the same part components for the *flight* craft (fuel, staging, gear, ship visuals, colliders) must filter `Without<EditorPart>` — that marker is the only thing separating the build world from the flying craft in the same ECS `World`. Resource storage is whitelist-driven from the parts catalog: any part kind can declare `storage` entries for fixed (`units`) or volume-scaled (`units_per_m3`) capacity, and blueprints may only activate resources whitelisted by that part. Omitted blueprint resources mean "use catalog defaults"; explicit resource maps mean the user's selected active pools. Do not restore hard-coded per-resource tank fields such as `methane_l_per_m3` / `lox_l_per_m3`; add real resources (for example `Kerosene`) to `Resource` and catalog storage lists instead. Air intake is ambient capture, not stored oxidizer: engines declare `intake_requirement`, nacelles may provide `builtin_intake`, and separate `Intake` parts can feed future engine-core layouts. See `docs/construction.md`.
- **`thalos_volumetric_clouds`** — vendored fork of `bevy-volumetric-clouds`
  (MIT, evroon) at `crates/volumetric_clouds/`. HZD-style raymarched near-cloud
  layer (Perlin-Worley atlas + 3-D Worley detail, dual-lobe HG; compute →
  texture), reworked around Thalos's spherical / `big_space` / dual-camera
  engine: the raymarch runs in the **body-fixed frame** of the active cloud
  body (true ray-sphere shells from the camera's planet-centred position;
  wrap-first triplanar noise sampling, f32-safe at planet-radius coordinates),
  so clouds are planet-fixed — glued to the ground, co-rotating, horizon-
  correct at any altitude. Large-scale coverage is a planet-fixed equirect
  weather map (`CloudCoverageMap`) generated from per-body `CloudWeatherState`
  in `SolarSystemState` (latitude bands + seeded noise; version-gated
  re-upload) — the future weather system's write target. The game
  (`rendering/clouds.rs`) drives it via `drive_clouds` (`ActiveCloudBody` =
  nearest terrestrial-atmosphere body, sole writer); the cloud texture plus a
  per-pixel nearest-hit distance composite *inside* the `body_sky` atmosphere
  pass (bound as `BodySkyMaterial::cloud_layer` / `cloud_distance`), not as a
  separate quad (which sorts unreliably against the fullscreen sky under
  big_space). See `docs/atmosphere.md` *Cloud rendering*.

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
  scenario picker / SHIPYARD / SETTINGS / QUIT) over the placeholder
  parking-orbit world; naming a scenario (`just game runway`) or setting
  `THALOS_AUTO_RUN` skips it. Menu scenario starts reuse the respawn /
  relaunch machinery in place — the runway pair re-arms its deferred
  placement + settle gate and re-enters `Loading`. `SpawnSituation` is
  therefore **mutable at runtime**; deferred placements are explicitly
  armed (`DescentPlacement`, `RunwayPlacement`), never keyed off the
  situation with a `Local<bool>`.
- **Spawn situation is a flag: ship in orbit, EVA on the
  surface, a landing approach over land, a final approach over
  flat land, or one of two surface-runway scenarios.** `main.rs` reads
  `just game [mode]` (passed as a CLI arg — default `menu`, the start
  screen; falls back to the `THALOS_SPAWN` env var for a direct
  `cargo run`) into a
  `spawn::SpawnSituation` resource (`ShipOrbit` | `Eva` | `Landing` |
  `FinalApproach` | `Runway` | `RunwayApproach`).
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
  - `scene_depth` — `SceneDepthImage` resource + `CopySceneDepthNode`
    render-graph node. Copies the main pass's `ViewDepthTexture`
    into a sample-able `Depth32Float` Image between `MainOpaquePass`
    and `MainTransparentPass` so the unified atmosphere fullscreen
    pass (`BodySkyMaterial` / `sky_dome.wgsl`) can read terrain /
    impostor / hull depth and clip its raymarch. Filters via the
    extracted `ShipCamera` marker — the map camera is not touched.
  - `ground_terrain` — UDLOD terrain spawn for procedural bodies +
    impostor↔terrain LOD swap (`sync_terrain_impostor_swap`) at
    `4 × radius`. Also spawns the always-on `BodySky` fullscreen
    quad per body (rebranded from the deprecated "sky dome" — now
    handles halo, sky, and aerial perspective in one pass).
- `ghost_bodies` — Renders ghost planet positions during time warp
  preview.
- `sky_render` — Renders procedural sky from `thalos_celestial`
  catalog (stars, galaxies as GPU meshes).
- `maneuver/` — Maneuver node placement/editing UI. Delta-v handles
  in local reference frame.
- `flight_plan_view/` — Renders the trajectory branch stack as
  on-screen tracks.
- `camera` — KSP-style orbit camera.
- `shipyard_editor/` — the **in-game ship editor** (primary front-end over
  `thalos_shipyard::editor`'s `ShipEditorCorePlugin`). A **separate scene**,
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
  launchpad spawn is **vertical** nose-up), and a tarmac **auto-connection** mesh
  (an MST over the site's structures) regenerates as structures change. The
  **default base is a small spaceport on a flat basin**: `runway::finish_runway_spawn`
  registers a wide `BaseSite` basin — *offset toward the launch-complex side* of
  the runway, sized to hold the whole layout — and the runway drapes on it (no
  longer its own pad), then `base_editor::spawn_default_base` authors the complex
  coplanar on it: two large launchpads with clearing, per-pad flame diverters +
  tank farms (`Tank` cylinders), a VAB-scale building and hangars on the far edge,
  blockhouses/ops near the strip, and a tarmac MST linking the road structures —
  so the surface scenarios present a whole base. Spawn
  points are **intrinsic** to runways/launchpads (the shipyard create-craft→fly
  flow is the next step). The base's flattened ground reads as a **grass lawn**
  (thick short `GrassProfile::lawn`) with its paved/built footprints (runway,
  launchpads, buildings, tanks) **cleared** — the building-terrain scatter layer:
  `body_render::ground::scatter`'s `ScatterRegion`/`ScatterTreatment`/`classify_scatter`,
  derived from the `StructureRegistry` by `rendering::grass` and honoured by the
  grass tile builder (the seam future base trees/props extend). See
  `docs/base_building.md` *Ground scatter*.
- `window_settings` — persisted window/display preferences (mode, windowed
  resolution, vsync, fullscreen monitor, user UI scale), stored as RON at the
  gitignored `user/settings.ron` and edited live from the settings menu's
  Window tab (`settings_menu`). Loaded in `main()` before the app so the
  initial window honours it; `THALOS_WINDOW_MODE` / `THALOS_WINDOW_SIZE` /
  `THALOS_VSYNC` are *session overrides* (they win for the run, grey out
  their UI control, and never leak into the file), and `THALOS_SCALE` stays a
  pure env diagnostic. `apply_window_settings` pushes settings onto the
  primary `Window` (value-compared) and writes back windowed drag-resizes so
  they persist; `apply_ui_scale` (which absorbed the former
  `compensate_fractional_ui_scale`) multiplies the user UI scale into the
  fractional-HiDPI crisp-text snap and drives `UiScale`. Caveat: runtime mode
  switches recreate the swapchain, which the
  known flaky Windows driver path (generic surface-acquire panic in Bevy's
  `prepare_windows`) can turn into a crash — pre-existing platform issue,
  newly reachable at runtime.
- `graphics_settings` — persisted graphics/rendering preferences, stored as RON
  at the gitignored `user/graphics.ron` and edited live from the settings menu's
  Graphics tab (`settings_menu`). Loaded when `GraphicsSettingsPlugin` builds;
  `save_graphics_settings` value-compares against the last write so an open tab
  doesn't churn the file. Today the only knob is the **volumetric-cloud toggle**
  (`GraphicsSettings::clouds`), read by `rendering::clouds::drive_clouds` — when
  off it parks the cloud raymarch via the existing no-cloud-body path (`ActiveCloudBody = None`
  → blank fallback bound onto `BodySkyMaterial`), so the sky renders clear at
  near-zero GPU cost. Add new render knobs here as fields + a control in
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

Parametric ship editor. Bevy + egui. The full next-gen construction model
(planes/ships/stations from one Module primitive) is specced in
`docs/construction.md`; **Slices 1–2 (parametric wing, wing-pylon jet nacelle,
and scalar intake flow) plus visually-actuating control surfaces have landed**
— see that doc's §0 for the status boundary.

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

Unified Bevy rendering for celestial bodies — one appearance model, two
backends. No generation logic. Added via a single `BodyRenderPlugin`
(which composes the three module sub-plugins). Three modules:

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
       + per-body environment state (dynamic surface, clouds; later wind/tides)
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

Each major system has a unified spec doc.

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
- `terrain.md` — the **consumer-side terrain contract**: the tile primitive
  (the black-box boundary), ground-LOD rendering, surface shadows, colliders,
  and dynamic features. Terrain *generation* is treated as a black box behind
  the tile contract; its previous design is archived (see below) and a new
  generator is being built against the contract.
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
