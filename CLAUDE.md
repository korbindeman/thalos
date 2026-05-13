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
just game                 # cargo run -p thalos_game
just edit <body>          # cargo run -p thalos_planet_editor -- <body>
just shipyard             # cargo run -p thalos_shipyard --bin ship_editor
just build                # cargo build --workspace
just test                 # cargo test -p thalos_physics
just clippy               # cargo clippy --workspace
just trace                # cargo run --release -p thalos_game --features profile-tracy
just bake Thalos          # headless terrain bake → PNGs in stage-bakes/Thalos/
just bake thalos          # body name is case-insensitive
just bake all             # bake every body with a terrain block
just bake Thalos --full   # use the body's authored/derived resolution
just clear-terrain-cache  # wipe target/terrain_cache/ after stage code changes

# Run a single test
cargo test -p thalos_physics -- test_name
```

## Toolchain

`rust-toolchain.toml` pins nightly only. There is no checked-in Cargo backend
override, so Cargo uses its default LLVM backend on every platform. Do not add
`rustc-codegen-cranelift-preview` or `codegen-backend = "cranelift"` to
project config unless the project intentionally re-adopts Cranelift for all
platforms.

macOS developers who want Cranelift for local iteration can install and
select it in their personal Cargo config or via one-off `cargo --config`
flags. Keep that opt-in local so Windows and Linux continue to use LLVM.

## Planet generation iteration

Planet generation is in an iterative development phase. Do not add or run
planet/terrain generation tests for now, including per-body generation
tests. This applies anywhere a test compiles or validates generated planet
data, even outside `thalos_terrain_gen`; these tests slow down the visual
iteration loop. Use the headless terrain bake workflow below for feedback
instead.

## Headless terrain bake (`bake_dump`)

`just bake <body>` runs a body's terrain compiler headlessly (no Bevy, no
window) and writes cubemap layers as PNGs to `stage-bakes/<body>/`. Body
name matching is case-insensitive; pass `all` to bake every body with a
terrain block. This is Claude Code's primary visual-feedback loop for
terrain work — read the PNGs directly as images to inspect output without
launching the editor.

**Outputs** (overwrites each run):
- `albedo-equirect.png` — baked albedo cubemap in a 2:1 lat/lon projection.
- `height-equirect.png` — grayscale height normalized to the body's
  encoded ± range (range reported in `info.txt`).
- `roughness-equirect.png` — grayscale roughness (R8Unorm).
- `normal-equirect.png` — object-space normal map (RGBA8).
- `info.txt` — range, resolution, route (Feature/Ocean/None), feature counts.

**Workflow:** after touching the compiler, run `just bake <body>`, then
`Read` the equirect PNG to check the result. Faster than the editor and
doesn't need a display.

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

- **`thalos_physics`** — pure Rust library, zero Bevy dependency, fully testable in isolation
- **`thalos_input`** — Bevy enhanced-input contexts, RON binding loader, and per-binary input intent resources
- **`thalos_game`** — Bevy consumer of physics + terrain outputs
- **`thalos_terrain_gen`** — procedural terrain generation pipeline (no Bevy dependency)
- **`thalos_atmosphere_gen`** — gas giant atmosphere definitions (cloud decks, hazes, rings; no Bevy dependency)
- **`thalos_celestial`** — procedural sky model: stars, galaxies, nebulae as physical flux sources (no Bevy dependency)
- **`thalos_planet_rendering`** — Bevy materials for planets, gas giants, rings, solid bodies
- **`thalos_planet_editor`** — interactive planet editor tool
- **`thalos_terrain`** — Bevy integration of the forked `bevy_terrain` UDLOD renderer; ships `ThalosTerrainPlugin` and a synthetic tile provider (M3 stage 1)
- **`thalos_shipyard`** — parametric ship editor (ECS attach tree, RON blueprints)
- **`thalos_bake_dump`** — headless terrain-bake CLI used by `just bake`

Core separation: `physics`, `terrain_gen`, `atmosphere_gen`, and `celestial`
are pure Rust libraries; `input`, `game`, `planet_rendering`, `terrain`,
`planet_editor`, and `shipyard` are Bevy consumers.
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

- Semantic player input is read from `thalos_input::game::GameInputIntent`.
  Keep raw Bevy input only for cursor positions, picking spatial data, and UI
  internals. See `docs/input.md`.
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
  - `types` — shared resources (`SimulationState`, `FrameBodyStates`,
    `CameraExposure`) and components (`CelestialBody`, `PlayerShip`,
    `PlanetMaterials`, `SolidPlanetMaterials`, etc.).
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
sets.

### Terrain gen crate (`crates/terrain_gen/`)

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
- `bake_from_body_data()` — consumes `terrain_gen::PlanetSurface` →
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
  → [propagate_flight_plan / propagate_branch_stack] synchronous prediction
       → FlightPlan / TrajectoryBranchStack (Actual + Projected branches)
  → [map_view] MapSnapshot → map rendering, maneuver UI, collision warnings
  → [rendering::real_space] BigSpace grids → ship-view body / camera transforms

[terrain_gen::compile_terrain_config]
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
- **`BodyTrajectoryProvider` is the abstraction boundary.** Body
  positions are always queried through this trait. `PatchedConics`
  is the current impl; a precomputed ephemeris could replace it
  without touching simulation or rendering.
- **Physics crate has no Bevy.** All physics logic must remain in
  `thalos_physics`. `thalos_game` is only presentation and input.
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
- `simulation.md` — simulation architecture (orbital mechanics,
  authority, time warp, map decoupling, big_space, Avian local
  bubble). Target design.
- `input.md` — enhanced-input context model, binding file rules, and
  per-binary intent resources.
- `terrain.md` — terrain generation (feature compiler) + ground LOD
  rendering. Includes v2 backlog from terrestrial-pipeline research.
- `atmosphere.md` — gas giants, rocky-atmosphere Bruneton scattering,
  ocean rendering, IBL/reflection probe.
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
