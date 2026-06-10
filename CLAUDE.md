# Thalos

Thalos is in **pre-alpha development**. Architecture and tooling are
still being shaped — agents are encouraged to tear down infrastructure
they find lacking and replace it with something better as the project
matures. **Do this explicitly**: announce the change before you make
it, explain why the existing approach falls short, and update this
file (and the relevant spec under `docs/`) so the next agent inherits
the new shape, not the old one. No silent rewrites.

## Commands

```bash
just game                 # cargo run -p thalos_game — ship in low Thalos orbit (default)
just game eva             # spawn on foot (EVA) on the Thalos surface instead
just game landing         # powered-descent approach over dry Thalos land
just game final           # low final approach over a flat dry Thalos patch
just game runway          # aircraft parked on the Thalos surface runway
just game runway-approach # aircraft on short final lined up with that runway
just edit <body>          # cargo run -p thalos_body_editor -- <body>
just terrain-lab          # static slippy-map terrain sketchpad at localhost:8787/tools/terrain-lab/
just shipyard             # cargo run -p thalos_shipyard --bin ship_editor
just build                # cargo build --workspace
just test                 # cargo test -p thalos_physics_canonical
just clippy               # cargo clippy --workspace
just trace                # cargo run --release -p thalos_game --features profile-tracy
just bake Thalos          # full-res local bake → target/bakes/Thalos.bin
                          #                    + stage-bakes/Thalos/full/*.png
just bake Thalos --preview # fast 512² PNG previews → stage-bakes/Thalos/preview/
                          #                          (no local game bake)
just bake all             # bake every body with a terrain block
                          #   (skips bodies whose local bake hash is
                          #    already current; add --force to rebake)
just clear-terrain-cache  # wipe target/terrain_cache/ (editor only — game and
                          # `just bake` no longer use this directory)
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
iteration loop. Use `just bake <body> --preview` (see below) for visual
feedback against the real compiler.

For faster speculative design, use `just terrain-lab` and open
`http://127.0.0.1:8787/tools/terrain-lab/`. Terrain Lab is a dev-only static
browser sketchpad with Google-Maps-style panning/zooming and lazily generated
LOD tiles. It is for process-map exploration before porting good ideas into
`crates/terrain`; the Rust terrain compiler and bake/query pipeline remain the
source of truth for game output.

Terrain fields should be process-first, not naked-noise-first. Smooth fBM,
ridged noise, and domain warps may drive masks, placement, breakup, and
small local texture, but broad visible height/albedo/bathymetry must come
from named terrain processes/features (coast shelves, seamounts, fractures,
mountain patches, basins, etc.) with explicit spatial windows. Do not write
global macro fBM/ridged fields directly into visible terrain; they produce
unrealistic smoky/streaky contours across the planet.

## Bakes: production vs preview

The game **loads pre-baked terrain only** — it never compiles. Bakes live
at `target/bakes/<body>.bin` and are produced locally by either
`just bake` (headless `bake_dump`) or the planet editor's Full button.
Bakes are developer-local build artifacts: they are ignored by Git, are not
tracked with Git LFS, and are not the distribution path for release assets.
Missing or stale bakes are auto-repaired at startup: `crates/game/src/bake_check.rs`
runs a `peek_key`-only pre-flight against every procedural body and, on
any mismatch, shells out to `cargo run --quiet --release -p thalos_bake_dump -- all`
(inherits stdio so the indicatif progress bars render normally) before
launching Bevy. The game still panics if a bake is somehow invalid
*after* auto-repair, which would indicate a bake_dump bug worth
surfacing. Bodies without authored terrain (`TerrainConfig::None` or no
`terrain` field) fall through to a solid-color impostor tinted with
`body.color`, so release builds ship with un-authored bodies still
rendering.

### `just bake` — two modes

**Default (full-res):**

- Runs the compiler at the body's full authored / radius-derived resolution.
- Writes the local bake `target/bakes/<body>.bin` (this is what `just game` loads).
- Writes full-resolution equirect PNGs to `stage-bakes/<body>/full/`, plus the
  ground-scale patch tile columns to `stage-bakes/<body>/full/patch/<biome>/`.
- Slow — Thalos (3186 km, 4096² cubemap) takes several minutes.

**Preview (`just bake <body> --preview`):**

- 512² cubemap; PNG dumps only, **no local game bake**.
- Writes equirects to `stage-bakes/<body>/preview/`.
- Emits the equirect set **plus the ground-scale shaded-relief patch set** as
  per-biome tile columns: `stage-bakes/<body>/preview/patch/<biome>/<span>.png`,
  where each biome dir (`hill`, `plain`) holds the LOD cascade
  (`context-120km.png`, `close-12km.png`, `micro-3km.png`, `fine-300m.png`,
  `ultra-60m.png`). One preview run shows both orbital coloration and on-foot
  relief. These patches *are* the tiles that exist on the planet — the same
  thing the planet editor's tile view shows.
- Fast — primary visual-feedback loop for iteration on the compiler.
- Doesn't touch `target/bakes/`, so iterating doesn't invalidate the loaded
  local bake in `just game`.

### PNG outputs (per run, overwrites)

- `albedo-equirect.png` — baked albedo cubemap in a 2:1 lat/lon projection;
  on ocean worlds this is still raw land/seabed albedo, not water-composited.
- `orbit-color-equirect.png` — ocean worlds only: preview composite of the
  separate water layer over the raw terrain substrate, for judging from-orbit
  coloration without violating the "water is not terrain material" invariant.
- `height-equirect.png` — grayscale height normalized to the body's
  encoded ± range (range reported in `info.txt`).
- `roughness-equirect.png` — grayscale roughness (R8Unorm).
- `normal-equirect.png` — object-space normal map (RGBA8).
- `info.txt` — range, resolution, route (Feature/Ocean/None), feature counts.
- `patch/<biome>/{context-120km…ultra-60m}.png` — ground-scale shaded-relief
  patch tile columns of the runtime walkable height (`surface_height_m`),
  emitted in both full and `--preview` runs. Each biome dir is one tile site:
  `hill` = highest-relief, `plain` = flattest. Within a biome the five PNGs zoom
  from a 120 km context down to a 60 m on-foot view. These read ground character
  the equirects are far too coarse to show, and map directly to the planet
  editor's tile view. The patch grid is built in **f64** (mirroring the game
  tile path `pixel_direction`): an f32 grid would quantise sample positions to
  the ~0.25 m f32 lattice at planet scale and render a grid/checkerboard moiré
  the field doesn't contain.
- (Tectonic and debug overlays when `--debug` is passed or the body has tectonics.)

### Workflow

- **Iterating on the compiler:** `just bake <body> --preview`, then `Read`
  the equirect PNG. No game launch needed, no display needed, fast loop.
- **Producing a local game bake:** `just bake <body>` (slow). Verifies the
  full pipeline and produces the artifact your local game will load.
- **Hash invariant:** the cache key hashes the body config + a FNV walk of
  `crates/terrain/src/**`. Any source edit there moves the key, so
  yesterday's local bake is detected as stale. Re-bake before `just game`.
- **Up-to-date skip:** in production (non-`--preview`) mode, `just bake`
  reads the stored key from `target/bakes/<body>.bin` via
  `cache::peek_key` and skips recompile + PNG dump when the key already
  matches. `just bake all` becomes a no-op when nothing's changed. Pass
  `--force` to bypass and rebake unconditionally.
- **Loading from the bake** is read-only: `crates/terrain/src/cache.rs`
  provides `load(path, key) -> Result<_, LoadError>` with explicit
  `Missing` / `HashMismatch` / `Decode` variants, plus `peek_key(path)`
  for fast staleness checks that avoid decompressing the full payload.
  `crates/game/src/bake_check.rs` uses `peek_key` as a startup
  pre-flight and auto-bakes any mismatch via `thalos_bake_dump`
  before Bevy boots; the spawn-path `load` panics remain as
  defense-in-depth assertions that should be unreachable in practice.
- **Erosion shader source:** `thalos_bake_dump` depends on
  `bevy_erosion_filter` 0.1.2 from crates.io and imports the WGSL source
  via the crate's public `EROSION_WGSL` constant (works with
  `default-features = false`, no Bevy pull-in). Do not restore the old
  sibling-checkout `../../../../bevy_erosion_filter/...` include or the
  prior `build.rs` registry-source lookup. See `docs/tooling.md`.

## Agent-driven inspection (bevy_brp)

`just game` always exposes Bevy's Remote Protocol on `localhost:15702`
via `BrpExtrasPlugin`. A project-local `.mcp.json` wires
`bevy_brp_mcp` (install once with `cargo install bevy_brp_mcp`) as an
MCP server, so an agent can query/mutate live ECS state, watch
components, screenshot, send key/mouse input, and read FPS without
restarting the game.

Only `Reflect`-registered types are visible to BRP. Keep the set
small and grow on demand — see `docs/tooling.md` for the registration
policy and the canonical→Bevy mirror pattern used at the bridge
(`CraftStateMirror`). Do not derive `Reflect` in `thalos_physics_canonical`;
mirror into a Bevy-side resource at the bridge instead.

## Profiling

Two backends, both gated on cargo features so default builds stay clean.

**Tracy (human-driven, interactive):** `just trace`. Requires Tracy
Profiler GUI v0.11.x running on localhost before launch. Version must
match the linked `tracy-client` (Bevy 0.18 → tracy-client 0.18.x).

**Chrome tracing (Claude-driven, autonomous):** run this when the user
asks to investigate performance. Not wired into `just` because it's a
workflow, not a one-shot command:

```bash
cargo run --release -p thalos_game --features profile-chrome
# play ~5–10 s, Ctrl-C → trace-<date>.json in cwd
python3 scripts/analyze_trace.py trace-<date>.json
```

The script streams the JSON (handles huge files), aggregates by span
name, and prints a top-N table to identify hot spots. Custom
`info_span!` markers live in `Simulation::step`, `propagate_flight_plan`,
`compute_preview_flight_plan`, `advance_simulation`, `update_prediction`,
`sync_maneuver_plan`.

**Slow-frame log + windowed trace analysis.** `PerfLogPlugin`
(`crates/game/src/perf_log.rs`) detects frames above
`SlowFrameThresholdMs` (default 25 ms; override at startup with
`THALOS_SLOW_FRAME_MS=…` or live via BRP `world_mutate_resources` on
`thalos_game::perf_log::SlowFrameThresholdMs`), pushes a record into the
`SlowFrameLog` resource (BRP-readable; ring of 64), and emits a
`slow_frame` `info_span!` so the chrome trace contains a marker at the
exact `ts`. An agent can:

1. `world_get_resources` `thalos_game::perf_log::SlowFrameLog` while the
   game runs to confirm spikes happened and grab their frame indices.
2. After Ctrl-C, scope `analyze_trace.py` to just those windows:

```bash
python3 scripts/analyze_trace.py trace-<date>.json \
    --around-name slow_frame --window-ms 200
```

This builds a union of ±100 ms windows around each `slow_frame` event
and ranks spans within that union, instead of averaging the whole
capture. A long session full of mostly-fine frames is still useful —
the bad frames don't get washed out.

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
  `physics_local`/Avian, not a claim of being the foundation.)
- **`thalos_input`** — Bevy enhanced-input contexts, RON binding loader, and per-binary input intent resources
- **`thalos_game`** — Bevy consumer of physics + terrain outputs
- **`thalos_terrain`** — procedural terrain generation pipeline (no Bevy dependency)
- **`thalos_celestial`** — procedural sky model: stars, galaxies, nebulae as physical flux sources (no Bevy dependency)

  *(The former `thalos_atmosphere` data crate — gas-giant cloud decks, hazes, rings, terrestrial scattering schemas — is folded into `thalos_world::atmosphere`; authored body data has one home.)*
- **`thalos_physics_local`** — Bevy/Avian f64 local-physics boundary for M5; aggregate craft hydration, terrain collider patches, contact/collapse helpers. The Avian rigid body persists across every regime; what *role* Avian plays each frame is a three-way `AvianRole` decided in `crates/game/src/local_physics.rs`: `Paused` under warp / `BodyFixed` (canonical owns everything), `AttitudeOnly` while coasting in vacuum at 1× (Kepler owns translation, Avian still integrates rotation + contact for player input and SAS), `Full` when there's a non-gravity force to integrate (throttle active or terrain collider attached). Coasting flight in vacuum stays under Kepler / `OnRails` so AP/PE do not drift. The classifier (`compute_avian_authority`) and the resulting authority transitions (`manage_authority`) live next to each other in `crates/game/src/local_physics.rs`. Fast descents are kept from tunneling through the terrain trimesh by `SweptCcd` on the craft body, and a too-hard contact destroys the craft via the whole-craft impact model (`detect_terrain_impact` → `Simulation::mark_destroyed`, gated on `ShipParameters::impact_tolerance_m_s`). On destruction the game force-pauses and shows an in-place scenario-respawn picker (`crates/game/src/scenario_menu.rs`) offering the four start scenarios (ship orbit / landing / final approach / EVA); see `docs/surface.md`.
- **`thalos_body_render`** — *(Phase 2, new)* unified celestial-body rendering, one appearance model + two backends. Three modules behind one `BodyRenderPlugin`: `shading` (shared `SceneLighting`/`AtmosphereBlock`/Hapke `shade_hapke_surface` + the `thalos::lighting`/`thalos::atmosphere` WGSL libraries), `impostor` (distant billboard materials for planets, gas giants, rings, solid bodies), `ground` (the `thalos_udlod`-backed terrain LOD: `ThalosTerrainPlugin`, `PipelineTileProvider`, `BodyTerrainMaterial`/`BodySkyMaterial`/`BodyWaterMaterial`, rendered-height patch utilities). Merged from the former `planet_lighting`+`planet_rendering`+`terrain_render`. A backend chooses geometry, never its own lighting/atmosphere/cloud math.
- **`thalos_body_editor`** — interactive celestial-body editor tool
- **`thalos_udlod`** — vendored UDLOD terrain renderer (lives at `crates/udlod/`). Forked from [`kurtkuehnert/bevy_terrain`](https://github.com/kurtkuehnert/bevy_terrain) by Kurt Kühnert (MIT OR Apache-2.0); attribution + license files travel with the source. Edit in-tree like any other workspace crate. The original fork at `~/dev/bevy_terrain` is kept around only as a reference point for diffing against upstream; daily edits happen here. The fork is now **runtime-provider-first**: it renders sparse tile atlases fed by `TileProvider` implementations, not preprocessed Earth-style asset trees. The old GeoTIFF/preprocess/`DiskTileProvider` path has been removed; if persistent reuse is needed, build it as a Thalos cache provider/wrapper keyed by body config + tile coordinate, not as `assets/<terrain>/data/*.bin`. CPU draw-tile selection is the current correctness path because it enforces 2:1 LOD balance across cube-face seams; tile *production* is the intended GPU extension point (job queue writes directly into atlas slots, later including diffusion). **`big_space` integration is unconditional** — the upstream `high_precision` Cargo feature has been removed, along with the runtime `DebugTerrain.high_precision` toggle and the `HIGH_PRECISION` shader define / pipeline flag. The Taylor-series relative-position path (`compute_relative_position` in `shaders/functions.wgsl`) is the only viable precision path at planet scale; gating it behind a feature only forced defensive `#[cfg]` plumbing in every consumer.
- **`thalos_shipyard`** — parametric ship editor (ECS attach tree, RON blueprints). Resource storage is whitelist-driven from the parts catalog: any part kind can declare `storage` entries for fixed (`units`) or volume-scaled (`units_per_m3`) capacity, and blueprints may only activate resources whitelisted by that part. Omitted blueprint resources mean "use catalog defaults"; explicit resource maps mean the user's selected active pools. Do not restore hard-coded per-resource tank fields such as `methane_l_per_m3` / `lox_l_per_m3`; add real resources (for example `Kerosene`) to `Resource` and catalog storage lists instead. Air intake is ambient capture, not stored oxidizer: engines declare `intake_requirement`, nacelles may provide `builtin_intake`, and separate `Intake` parts can feed future engine-core layouts. See `docs/construction.md`.
- **`thalos_bake_dump`** — headless terrain-bake CLI used by `just bake`
- **`avian_fdm`** — vendored zone-based 6-DoF flight-dynamics model (lives at
  `crates/avian_fdm/`). Forked from [`viccuad/avian_fdm`](https://github.com/viccuad/avian_fdm)
  for atmospheric aerodynamics (drag now; lift + planes later). **LGPL-3.0-or-later**
  (its GPL J-3 Cub preset crate is *not* depended on) — the sole copyleft entry on
  an otherwise permissive stack. **Resolved by full-source distribution**: Thalos
  is now fully source-available (code PolyForm Noncommercial, assets CC BY — see
  `LICENSING.md`), so LGPL's relink requirement is satisfied for every build,
  including the paid one. No relicense or replacement is needed. **Keep it
  isolated and never add the GPL J-3 Cub preset crate (`avian_fdm_j3cub_jsbsim`)
  or any other GPL/AGPL (non-LGPL) dependency** — GPL is viral across the whole
  combined work and would void the noncommercial model; CI guards against this
  (`.github/workflows/ci.yml`). Used **force-only** in the local bubble (Thalos
  owns mass/inertia/gravity); only Bevy-side crates (`game`/`physics_local`) may
  depend on it. See `docs/aerodynamics.md` for the model and environment
  adaptations.
- **`thalos_volumetric_clouds`** — vendored fork of `bevy-volumetric-clouds`
  (MIT, evroon) at `crates/volumetric_clouds/`. HZD-style raymarched near-cloud
  layer (Perlin-Worley atlas + 3-D Worley detail, dual-lobe HG; compute →
  texture), adapted to Thalos's spherical / `big_space` / dual-camera engine.
  The game (`rendering/clouds.rs`) drives it in a planet-local tangent frame;
  the cloud texture composites *inside* the `body_sky` atmosphere pass (bound as
  `BodySkyMaterial::cloud_layer`), not as a separate quad (which sorts
  unreliably against the fullscreen sky under big_space). See
  `docs/atmosphere.md` *Cloud rendering*.

Core separation: `world`, `physics_canonical`, `terrain`, and
`celestial` are pure Rust libraries; `input`, `game`, `body_render`,
`physics_local`, `body_editor`, and `shipyard` are Bevy consumers. Within
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
flight controls.

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

- **Spawn situation is a flag: ship in orbit (default), EVA on the
  surface, a landing approach over land, a final approach over
  flat land, or one of two surface-runway scenarios.** `main.rs` reads
  `just game [mode]` (passed as a CLI arg — default `orbit`; falls back
  to the `THALOS_SPAWN` env var for a direct `cargo run`) into a
  `spawn::SpawnSituation` resource (`ShipOrbit` | `Eva` | `Landing` |
  `FinalApproach` | `Runway` | `RunwayApproach`).
  The canonical `CraftState` is the player either way — KSP-style: one
  craft, Ship or EVA, distinguished by `VesselKind`. The ship blueprint is
  chosen per scenario by `SpawnSituation::ship_blueprint_path`
  (`apollo.ron` by default, `skyhawk.ron` for the runway scenarios).
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
  `approach-runway`) put the `skyhawk.ron` aircraft on a fixed runway on
  the Thalos surface, owned by `crate::runway`. Like the descent modes
  these are deferred and terrain-aware: `runway::finish_runway_spawn`
  runs once on the first `AppState::Running` frame and picks a flat dry
  low-latitude site by a deterministic body-fixed search (epoch-
  independent). **The terrain itself is flattened into a pad and the runway
  sits flush on it** (replacing the former raised-slab + skirt/runoff
  platform): a single fixed elevation `E = max(natural terrain over the pad)
  + margin` is chosen, then a `thalos_terrain::TerrainFlatten` pad is
  installed through the body's shared
  `rendering::ground_terrain::TerrainFlattenRegistry` handle. The terrain
  tile provider (`PipelineTileProvider`, wrapped in
  `thalos_terrain::FlattenedSurface`) reads that handle as it bakes, so the
  *rendered* ground — and, via the GPU-atlas height mirror, the collider and
  CPU height queries — level out to `E` across the pad (runway + ~50 m
  shoulder) and smoothstep-blend back to natural terrain over a ~150 m ramp.
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
  admits tile loads nearest-view-first, which is what makes that wait tolerable
  rather than minutes — see the cold-streaming memory note.
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
  outright; canonical authority is pinned to `LocalRigidBody` by
  `manage_authority` (not `OnRails`) since grounded EVA co-rotates with the
  surface and `Simulation::step` must only advance sim-time.
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
- `map_view` — snapshot/projection boundary for map rendering. Copies
  `CraftState`, body states, and `FlightPlan` into `MapSnapshot`; map
  systems consume the snapshot and never mutate canonical simulation
  state.
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

Systems run in `SimStage` order: `Physics → Sync → Camera`
(configured in `main.rs`), ensuring deterministic state flow each
frame. Enhanced input intent collection runs in `PreUpdate` before these
sets. Simulation pause is an explicit clock boundary, not a global Bevy
clock pause: `crates/game/src/sim_clock.rs` owns `SimClock`, whose sole
writer folds Escape pause, destruction scenario picker, freecam, and warp
pause into a zero sim delta. Canonical stepping, local physics ownership,
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
and scalar intake flow) have landed** — see that doc's §0 for the status
boundary.

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
  in the host-local frame, shared by the editor and the game's
  `ship_view`. Control surfaces are planned as *wing parameters* (not
  separate parts); gear will be its own footprint part. No flight model
  yet (M6).
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
  `BodyTerrainMaterial`/`BodySkyMaterial`/`BodyWaterMaterial`, embedding
  their `src/ground/*.wgsl` via `embedded://thalos_body_render/ground/…`.
- `PipelineTileProvider`, `rendered_height_*`, the `HeightSource` family,
  and rendered-height patch utilities used by M5 colliders.

`body_render` is the **sole consumer** of the vendored `thalos_udlod`,
re-exported as `thalos_body_render::udlod` (`{prelude, math, big_space}`); no
other crate depends on the fork directly. Replacing the ground backend stays
localized to the `ground` module + that re-export.

### Body editor (`crates/body_editor/`)

Standalone Bevy binary for interactive celestial-body preview. Loads
`solar_system.ron` + per-body files, selects a body, runs
`compile_terrain_config`, renders with `PlanetMaterial` via billboard
mesh. Uses `bevy_egui` for UI. Live rebake, sketch tool, tectonic
overlay.

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
  `refresh_craft_state_mirror`, `AvianAuthority` ← `compute_avian_authority`.
  Every other system reads. Don't add a second writer; if a resource needs
  to be mutated from elsewhere, route through an accessor on the sole writer
  or reconsider the ownership.
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
- `surface.md` — surface gameplay in two parts: **on foot (EVA)**
  (ground physics, body-fixed pose, the `HeightSource` interface,
  surface map view) and **landing & impact destruction** (landed-ship
  descent, terrain collision via `SweptCcd` anti-tunneling, and
  whole-craft impact destruction, `ShipParameters::impact_tolerance_m_s`
  → `Simulation::is_destroyed`). Merged from the former
  `surface_gameplay.md` + `landing.md`.
- `construction.md` — next-gen shipyard / construction model design:
  one Module primitive (end-node / footprint+morph / end-cap / host /
  connector / reservation), stationed-loft fuselages and wings, a
  separate internal/loadout layer (compartments, role-fills, cargo
  doors), generalising the rocket-only shipyard to planes, ships, and
  stations. Target: M6. Design-only, no code yet.
- `input.md` — enhanced-input context model, binding file rules, and
  per-binary intent resources.
- `terrain.md` — the **consumer-side terrain contract**: the tile primitive
  (the black-box boundary), ground-LOD rendering, surface shadows, colliders,
  and dynamic features. Terrain *generation* is treated as a black box behind
  the tile contract; its previous design is archived (see below) and a new
  generator is being built against the contract.
- `atmosphere.md` — gas giants, rocky-atmosphere single-scattering
  raymarch (unified per-body fullscreen pass with scene-depth
  coupling for aerial perspective), Kármán-line authoring, ocean
  rendering, IBL/reflection probe. (Atmosphere *rendering*; the
  *physics* density/drag model is in `aerodynamics.md`.)
- `aerodynamics.md` — atmospheric flight forces (drag now; lift + planes
  later) via the vendored `avian_fdm`: the per-body density model, the
  body-centered/rotating-airmass adaptations, force-only bubble use, the
  in-atmosphere `Full`-role trigger + warp clamp, and the LGPL story (satisfied
  by full-source distribution; see `LICENSING.md`).
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
