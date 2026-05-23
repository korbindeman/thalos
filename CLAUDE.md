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
just edit <body>          # cargo run -p thalos_planet_editor -- <body>
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
feedback instead.

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
- Writes full-resolution equirect PNGs to `stage-bakes/<body>/full/`.
- Slow — Thalos (3186 km, 4096² cubemap) takes several minutes.

**Preview (`just bake <body> --preview`):**

- 512² cubemap; PNG dumps only, **no local game bake**.
- Writes to `stage-bakes/<body>/preview/`.
- Fast — primary visual-feedback loop for iteration on the compiler.
- Doesn't touch `target/bakes/`, so iterating doesn't invalidate the loaded
  local bake in `just game`.

### PNG outputs (per run, overwrites)

- `albedo-equirect.png` — baked albedo cubemap in a 2:1 lat/lon projection.
- `height-equirect.png` — grayscale height normalized to the body's
  encoded ± range (range reported in `info.txt`).
- `roughness-equirect.png` — grayscale roughness (R8Unorm).
- `normal-equirect.png` — object-space normal map (RGBA8).
- `info.txt` — range, resolution, route (Feature/Ocean/None), feature counts.
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

## Architecture

Thalos is a planetary exploration / orbital mechanics sandbox in Rust
(edition 2024, Bevy 0.18, glam 0.30). Workspace crates:

- **`thalos_physics_canonical`** — pure Rust library, zero Bevy dependency, fully testable in isolation
- **`thalos_input`** — Bevy enhanced-input contexts, RON binding loader, and per-binary input intent resources
- **`thalos_game`** — Bevy consumer of physics + terrain outputs
- **`thalos_terrain`** — procedural terrain generation pipeline (no Bevy dependency)
- **`thalos_atmosphere`** — gas giant atmosphere definitions (cloud decks, hazes, rings; no Bevy dependency)
- **`thalos_celestial`** — procedural sky model: stars, galaxies, nebulae as physical flux sources (no Bevy dependency)
- **`thalos_physics_local`** — Bevy/Avian f64 local-physics boundary for M5; aggregate craft hydration, terrain collider patches, contact/collapse helpers. The Avian rigid body persists across every regime; what *role* Avian plays each frame is a three-way `AvianRole` decided in `crates/game/src/local_physics.rs`: `Paused` under warp / `BodyFixed` (canonical owns everything), `AttitudeOnly` while coasting in vacuum at 1× (Kepler owns translation, Avian still integrates rotation + contact for player input and SAS), `Full` when there's a non-gravity force to integrate (throttle active or terrain collider attached). Coasting flight in vacuum stays under Kepler / `OnRails` so AP/PE do not drift. The classifier (`compute_avian_authority`) and the resulting authority transitions (`manage_authority`) live next to each other in `crates/game/src/local_physics.rs`. Fast descents are kept from tunneling through the terrain trimesh by `SweptCcd` on the craft body, and a too-hard contact destroys the craft via the whole-craft impact model (`detect_terrain_impact` → `Simulation::mark_destroyed`, gated on `ShipParameters::impact_tolerance_m_s`). On destruction the game force-pauses and shows an in-place scenario-respawn picker (`crates/game/src/scenario_menu.rs`) offering the four start scenarios (ship orbit / landing / final approach / EVA); see `docs/landing.md`.
- **`thalos_planet_lighting`** — shared planet lighting types (`SceneLighting`, `StarLight`, `AtmosphereBlock`, `CLOUD_BAND_COUNT`) + WGSL libraries (`thalos::lighting`, `thalos::atmosphere`) + the Hapke surface shading helper (`shade_hapke_surface`). Both `thalos_planet_rendering` and `thalos_terrain_render` depend on it.
- **`thalos_planet_rendering`** — Bevy materials for planets, gas giants, rings, solid bodies
- **`thalos_planet_editor`** — interactive planet editor tool
- **`thalos_udlod`** — vendored UDLOD terrain renderer (lives at `crates/udlod/`). Forked from [`kurtkuehnert/bevy_terrain`](https://github.com/kurtkuehnert/bevy_terrain) by Kurt Kühnert (MIT OR Apache-2.0); attribution + license files travel with the source. Edit in-tree like any other workspace crate. The original fork at `~/dev/bevy_terrain` is kept around only as a reference point for diffing against upstream; daily edits happen here. The fork is now **runtime-provider-first**: it renders sparse tile atlases fed by `TileProvider` implementations, not preprocessed Earth-style asset trees. The old GeoTIFF/preprocess/`DiskTileProvider` path has been removed; if persistent reuse is needed, build it as a Thalos cache provider/wrapper keyed by body config + tile coordinate, not as `assets/<terrain>/data/*.bin`. CPU draw-tile selection is the current correctness path because it enforces 2:1 LOD balance across cube-face seams; tile *production* is the intended GPU extension point (job queue writes directly into atlas slots, later including diffusion). **`big_space` integration is unconditional** — the upstream `high_precision` Cargo feature has been removed, along with the runtime `DebugTerrain.high_precision` toggle and the `HIGH_PRECISION` shader define / pipeline flag. The Taylor-series relative-position path (`compute_relative_position` in `shaders/functions.wgsl`) is the only viable precision path at planet scale; gating it behind a feature only forced defensive `#[cfg]` plumbing in every consumer.
- **`thalos_terrain_render`** — Bevy integration of the in-tree `thalos_udlod` UDLOD renderer; ships `ThalosTerrainPlugin`, `PipelineTileProvider`, and rendered-height terrain patch utilities used by M5 colliders
- **`thalos_shipyard`** — parametric ship editor (ECS attach tree, RON blueprints)
- **`thalos_bake_dump`** — headless terrain-bake CLI used by `just bake`

Core separation: `physics`, `terrain`, `atmosphere`, and `celestial`
are pure Rust libraries; `input`, `game`, `planet_lighting`,
`planet_rendering`, `terrain_render`, `physics_local`, `planet_editor`, and
`shipyard` are Bevy consumers. `planet_lighting` sits below
`planet_rendering` and `terrain` so both render paths share one source
of truth for `SceneLighting`, `AtmosphereBlock`, and the surface BRDF;
no field-by-field mirror types between crates. Avian lives behind
`thalos_physics_local`; do not add Avian to `thalos_physics_canonical`.
Semantic input for the Bevy binaries flows through `thalos_input`
contexts and intent resources, with checked-in defaults at
`assets/input.ron`.

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
- `parsing` — Loads `assets/solar_system.ron` plus per-body files at
  `assets/bodies/<lowercase_name>.ron` into `SolarSystemDefinition`.
  System-level file carries physical + orbital specs; per-body files
  carry `terrain: TerrainConfig` and `tectonics: TectonicConfig`.

### Game crate (`crates/game/`)

- **Spawn situation is a flag: ship in orbit (default), EVA on the
  surface, a landing approach over land, or a final approach over
  flat land.** `main.rs` reads
  `just game [mode]` (passed as a CLI arg — default `orbit`; falls back
  to the `THALOS_SPAWN` env var for a direct `cargo run`) into a
  `spawn::SpawnSituation` resource (`ShipOrbit` | `Eva` | `Landing` |
  `FinalApproach`).
  The canonical `CraftState` is the player either way — KSP-style: one
  craft, Ship or EVA, distinguished by `VesselKind`. **`orbit`**:
  `VesselKind::Ship` in a low Thalos parking orbit
  (`system.ship.initial_state`), nose along prograde;
  `ship_view::spawn_player_ship` loads `apollo.ron` and pushes the real
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
  patch (or the flattest dry fallback). Thalos has no atmosphere yet, so
  both descents are lunar-style — no aerobraking.
- **EVA is a full craft, with a grounded/airborne split.** The
  `EvaMode` resource (`player_controller.rs`, `Grounded` | `Airborne`,
  defaults `Grounded`) picks which regime owns the capsule.
  **`Grounded`**: `walk_eva_on_terrain` glues the capsule to the
  rendered surface and `snap_avian_from_canonical` + the
  `readback_local_craft` translation short-circuit so the controller
  owns state. **`Airborne`**: Kepler owns translation, the snap drives
  the capsule from canonical (exactly like a ship coasting in vacuum),
  and `walk_eva_on_terrain` stands down. `apply_local_forces`
  short-circuits for EVA in both modes — EVA has no thrust (coast-only;
  a jetpack is the natural follow-up). The teleports mirror the ship's:
  body-tree cmd-click sends EVA to a low orbit (→ `Airborne`), and a
  row's `drop` button + map cursor (or F9) plants it on a surface point
  (→ `Grounded`, in place via `local_physics::place_eva_on_surface` —
  the bubble is never torn down, unlike ships). Mode flips only on these
  explicit teleports; there is no automatic surface↔orbit transition yet.
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
  flat-water placeholders.
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
  consumed by `AgingOceanicHomeworld` (Thalos) and
  `ColdDesertFormerlyWet` (Vaelen).
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

Parametric ship editor. Bevy + egui.

- `AttachNode` / `Ship` — ECS tree structure for ship assembly
- `Part` trait — `CommandPod`, `Engine`, `FuelTank`, `Decoupler`,
  `Adapter`
- `ShipBlueprint` — RON serialization format for ship designs
- `sizing` — parametric node sizing (adapters/tanks scale from parent)
- `staging` — pure stage derivation from decoupler topology
  (`derive_stages`) + per-stage Δv/fuel accounting
  (`compute_stage_summaries`). Single home for the staging model: the
  game's live ECS staging systems (`crates/game/src/staging.rs`) and the
  editor's staging preview (`ShipBlueprint::stage_summaries`) both call it,
  so stage boundaries never diverge. Like `stats`, no Bevy beyond glam math.

### Planet lighting crate (`crates/planet_lighting/`)

Tiny shared crate that owns the data structures and WGSL libraries
every planet-surface material reads from. No generation logic, no
materials.

- `PlanetLightingPlugin` — registers the `thalos::lighting` and
  `thalos::atmosphere` shader libraries. Both `PlanetRenderingPlugin`
  and `ThalosTerrainPlugin` add it defensively (no-op if already
  added), so apps that pull only one of those still get the shared
  libraries.
- `SceneLighting` / `StarLight` / `MAX_STARS` / `MAX_ECLIPSE_OCCLUDERS`
  — scene-level lighting (primary star, eclipse occluders, ambient,
  planetshine parent).
- `AtmosphereBlock` / `CLOUD_BAND_COUNT` — per-body atmosphere uniform
  (Rayleigh + Mie + cloud bands + Minnaert limb).
- `shaders/lighting.wgsl` — WGSL mirror of `SceneLighting` plus the
  `eclipse_factor`, `planetshine_sample`, `hapke_brdf`, and
  `shade_hapke_surface` helpers. Both the impostor and the thalos_udlod
  ground LOD route through the same `shade_hapke_surface` function so
  shading matches across the LOD swap.
- `shaders/atmosphere.wgsl` — WGSL mirror of `AtmosphereBlock`,
  `integrate_atmosphere`, `composite_clouds`, `apply_limb_darkening`,
  and friends.

`thalos_planet_rendering` re-exports `SceneLighting`, `AtmosphereBlock`,
etc. so existing call sites in `thalos_game` and `thalos_planet_editor`
keep resolving without churn.

### Planet rendering crate (`crates/planet_rendering/`)

Thin Bevy rendering layer. No generation logic.

- `PlanetRenderingPlugin` — registers `PlanetMaterial`,
  `GasGiantMaterial`, `RingMaterial`, `SolidPlanetMaterial`.
- `PlanetMaterial` — procedural-body impostor with cubemap
  height/albedo textures. Uses `assets/shaders/planet_impostor.wgsl`.
- `SolidPlanetMaterial` — solid-color billboard impostor for
  non-procedural bodies. Uses `assets/shaders/solid_planet.wgsl`.
- `GasGiantMaterial` — gas giant cloud/haze rendering. Uses
  `assets/shaders/gas_giant.wgsl`.
- `RingMaterial` — ring system rendering. Uses
  `assets/shaders/ring.wgsl`.
- `bake_from_body_data()` — consumes `terrain::PlanetSurface` →
  `PlanetTextures` for upload into a `PlanetMaterial`.

### Planet editor (`crates/planet_editor/`)

Standalone Bevy binary for interactive planet preview. Loads
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
  → [planet_rendering::bake_from_body_data] PlanetTextures
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
  This extends to the other pure-Rust libraries (`thalos_terrain`,
  `thalos_atmosphere`, `thalos_celestial`): none may pull in Bevy, even
  transitively. The boundary is CI-guarded — `.github/workflows/ci.yml`
  runs a `cargo tree` check on every push/PR and fails if a real Bevy
  crate enters any of the four trees. (`bevy_erosion_filter` is allowed:
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

`ROADMAP.md` is the entry point: engineering milestones, current
status, dependency graph. Each major system has a unified spec doc.

- `ROADMAP.md` — top-level roadmap. Phase 1 (M1-M4) = architectural +
  rendering; Phase 2 (M5-M6) = gameplay; M7-M8 deferred.
- `tooling.md` — toolchain policy and local-only compiler tuning recipes
  for Windows fast incremental builds, macOS incremental workarounds, and
  backend/linker experiments.
- `simulation.md` — simulation architecture (orbital mechanics,
  authority, time warp, map decoupling, big_space, Avian local
  bubble). Target design.
- `surface_gameplay.md` — on-foot / EVA surface gameplay: ground
  physics, body-fixed pose, the `HeightSource` interface, surface map
  view. Defers landed-*ship* mechanics to `landing.md`.
- `landing.md` — landed-ship mechanics: descent, terrain collision
  (`SweptCcd` anti-tunneling), and whole-craft impact destruction
  (`ShipParameters::impact_tolerance_m_s` →
  `Simulation::is_destroyed`).
- `input.md` — enhanced-input context model, binding file rules, and
  per-binary intent resources.
- `terrain.md` — terrain generation (feature compiler) + ground LOD
  rendering. Includes v2 backlog from terrestrial-pipeline research.
- `atmosphere.md` — gas giants, rocky-atmosphere single-scattering
  raymarch (unified per-body fullscreen pass with scene-depth
  coupling for aerial perspective), Kármán-line authoring, ocean
  rendering, IBL/reflection probe.
- `celestial.md` — celestial sphere design: source model, spectrum,
  generation, rendering pipeline.
- `tooling.md` — Rust toolchain policy and local developer tooling notes.
- `lore/solar_system.md` — per-body reference with scale philosophy
  (hybrid 1:1/1:2/1:3 scale rationale) and formation scenario.
- `lore/civilization.md` — civilization, narrative progression
  phases, resource economy.

Standalone references (not consolidated):

- `gen/terrestrial_pipeline_research.md` — academic + industry survey
  for the terrestrial pipeline.
- `gen/planet_aesthetics.md` — visual target field guide.
- `gen/dunes.md` — dune-field generator algorithm.
- `gen/vaelen_processes.md` — per-body process notes for Vaelen.
