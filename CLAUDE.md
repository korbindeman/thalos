# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
just game                 # cargo run -p thalos_game --features dev
just editor               # cargo run -p thalos_planet_editor --features dev
just shipyard             # cargo run -p thalos_shipyard --bin ship_editor
just build                # cargo build --workspace
just test                 # cargo test -p thalos_physics -p thalos_terrain_gen
just clippy               # cargo clippy --workspace
just trace                # cargo run --release -p thalos_game --features profile-tracy
just bake Thalos          # headless terrain bake → PNGs in target/stage-bakes/Thalos/
just bake thalos          # body name is case-insensitive
just bake all             # bake every body with a generator block
just bake Thalos stage=1  # run only the first N stages of the body's pipeline

# Run a single test
cargo test -p thalos_physics -- test_name
```

## Headless terrain bake (`bake_dump`)

`just bake <body> [stage=N]` runs a body's terrain pipeline headlessly
(no Bevy, no window) and writes cubemap layers as PNGs to
`target/stage-bakes/<body>/`. Body name matching is case-insensitive;
pass `all` to bake every body with a generator block. This is Claude's
primary visual-feedback loop for terrain pipeline work — Claude can
`Read` the PNGs directly as images to inspect stage output without
anyone launching the editor.

**Outputs** (overwrites each run):
- `albedo-equirect.png` / `albedo-cross.png` — baked albedo cubemap in a
  2:1 lat/lon projection and a 4:3 cube-cross layout.
- `height-equirect.png` / `height-cross.png` — grayscale height
  normalized to the body's encoded ± range (range reported in
  `info.txt`).
- `material-equirect.png` / `material-cross.png` — per-material-ID
  hashed colors.
- `info.txt` — range, resolution, stage list, province/feature counts.

Use the equirect for a globe-at-a-glance read. Use the cross when
hunting face-seam defects — misaligned features across face edges jump
out in that layout.

**Partial runs:** `stage=N` truncates the body's `pipeline:` list to the
first N entries, so you can inspect each stage in isolation without
editing the RON. Iterate a stage by re-running the bake after each code
change.

**Workflow:** after touching a stage, run `just bake <body> stage=<n>`,
then `Read` the equirect PNG to check the result. Use this proactively
— it's faster than the editor and doesn't need a display. The tool is
the authoritative way to verify terrain pipeline output short of
rendering in the editor.

## Profiling

Two backends, both gated on cargo features so default builds stay clean.

**Tracy (human-driven, interactive):** `just trace`. Requires Tracy
Profiler GUI v0.11.x running on localhost before launch. Version must
match the linked `tracy-client` (Bevy 0.18 → tracy-client 0.18.x).

**Chrome tracing (Claude-driven, autonomous):** Claude runs this when
the user asks to investigate performance. Not wired into `just` because
it's a workflow, not a one-shot command:

```bash
cargo run --release -p thalos_game --features profile-chrome
# play ~5–10 s, Ctrl-C → trace-<date>.json in cwd
python3 scripts/analyze_trace.py trace-<date>.json
```

The script streams the JSON (handles huge files), aggregates by span
name, and prints a top-N table Claude reads to identify hot spots.

Custom `info_span!` markers live in `Simulation::step`,
`propagate_flight_plan`, `compute_preview_flight_plan`,
`advance_simulation`, `update_prediction`, `sync_maneuver_plan`.

## Architecture

Thalos is an orbital mechanics sandbox game in Rust (edition 2024, Bevy 0.18, glam 0.30). Workspace crates:

- **`thalos_physics`** — pure Rust library, zero Bevy dependency, fully testable in isolation
- **`thalos_game`** — thin Bevy consumer of physics outputs
- **`thalos_terrain_gen`** — procedural terrain generation pipeline framework (no Bevy dependency)
- **`thalos_atmosphere_gen`** — gas giant atmosphere definition (cloud decks, hazes, rings; no Bevy dependency)
- **`thalos_celestial`** — procedural sky model: stars, galaxies, nebulae as physical flux sources (no Bevy dependency)
- **`thalos_planet_rendering`** — Bevy rendering for planets (impostor), gas giants, and rings
- **`thalos_planet_editor`** — interactive planet editor tool
- **`thalos_shipyard`** — parametric ship editor (ECS attach tree, RON blueprints)

Core separation: physics, terrain_gen, atmosphere_gen, and celestial are pure Rust libraries; game, planet_rendering, planet_editor, and shipyard are Bevy consumers. `crates/planet_gen/` exists but is NOT in the workspace — older spec-faithful approach, kept for reference.

### Physics crate (`crates/physics/`)

Two trait abstractions draw the boundaries:

- `BodyStateProvider` (`body_state_provider.rs`) — answers "where is body `i` at time `t`?" Implemented by `PatchedConics`; could be swapped for a baked ephemeris.
- `ShipPropagator` (`ship_propagator.rs`) — propagates the ship across one segment of coast or burn. Implemented by `KeplerianPropagator`: analytical Kepler coast under a single SOI body + RK4 finite-burn. SOI transitions are detected per substep and refined by bisection (coast) or shortened RK4 (burn).

Key modules:

- `types` — `StateVector`, `BodyDefinition`, `TrajectorySample` (each sample carries `anchor_body` + `ref_pos` so the renderer can draw without a per-sample ephemeris query).
- `orbital_math` — Cartesian↔Keplerian conversions, Kepler-equation solvers (elliptic + hyperbolic), `propagate_kepler`. Pure math, well-tested.
- `patched_conics` — `BodyStateProvider` impl using analytical Keplerian orbits. Bodies form a parent-child tree; queries walk the lineage and sum each ancestor's motion.
- `ship_propagator` — `ShipPropagator` trait + `KeplerianPropagator` impl. Coast and burn segments terminate on the first of: target time, SOI exit, collision, SOI enter, burn end, or stable-orbit closure. Resolution order at boundaries: `exit > collision > enter`.
- `simulation` — Central `Simulation` struct: owns ship state, attitude, warp, `KeplerianPropagator` instance, `ManeuverSequence`. `step()` is called each frame and consumes maneuver nodes as their start time arrives.
- `trajectory` — Flight-plan prediction. `propagate_flight_plan` runs the same `ShipPropagator` across the maneuver sequence, producing `Leg`s of `(burn?, coast)` `NumericSegment`s. Includes event detection (SOI / apsis / impact), encounter aggregation, closest-approach scans.
- `maneuver` — `ManeuverNode`: time, delta-v in local prograde/normal/radial frame, reference body. Plus the frame-conversion helpers.
- `parsing` — Loads `assets/solar_system.ron` into `SolarSystemDefinition`.

### Game crate (`crates/game/`)

- `bridge` — The core adapter. Calls `Simulation::step()` each frame, manages async prediction worker thread, syncs maneuver edits, handles warp controls.
- `rendering` / `trajectory_rendering` — Reads body positions from `PatchedConics` and trajectory samples from prediction. Cone width is derived from perturbation ratio directly.
- `ghost_bodies` — Renders ghost planet positions during time warp preview.
- `sky_render` — Renders procedural sky from `thalos_celestial` catalog (stars, galaxies as GPU meshes).
- `maneuver/` — Maneuver node placement/editing UI. Delta-v handles in local reference frame.
- `camera` — KSP-style orbit camera.

Systems run in `SimStage` order: Physics → Sync → Camera, ensuring deterministic state flow each frame.

### Terrain gen crate (`crates/terrain_gen/`)

Cubemap-based procedural surface generation. No Bevy dependency.

- `BodyBuilder` — mutable build-time state. Stages mutate this (height/albedo accumulators, craters, volcanoes, channels, materials, detail noise params).
- `BodyData` — immutable GPU-facing output baked from `BodyBuilder`. Holds `Cubemap<u16>` height, `Cubemap<[u8; 4]>` albedo, `IcoBuckets` spatial index.
- `Stage` trait — `name()`, `dependencies()`, `apply(&mut BodyBuilder)`. Each stage is a pure transform.
- `Pipeline` — validates dependency ordering at construction; runs stages in order with deterministic per-stage seeding via `sub_seed()`.
- `sample()` / `SurfaceSample` — single sampling contract for reading finished surface data.

### Celestial crate (`crates/celestial/`)

Procedural sky model. Pure Rust, no Bevy. Works in physical quantities (flux, temperature, SED) — never pre-baked RGB.

- `Universe` — collection of `Source` objects (stars, galaxies, nebulae)
- `Spectrum` — spectral energy distributions: `Blackbody`, `PowerLaw`, `Tabulated`; `Passband` filtering
- `generate/` — procedural star field and galaxy placement
- `render/` — cubemap baking and telescope PSF

### Shipyard crate (`crates/shipyard/`)

Parametric ship editor. Bevy + egui.

- `AttachNode` / `Ship` — ECS tree structure for ship assembly
- `Part` trait — `CommandPod`, `Engine`, `FuelTank`, `Decoupler`, `Adapter`
- `ShipBlueprint` — RON serialization format for ship designs
- `sizing` — parametric node sizing (adapters/tanks scale from parent)

### Planet rendering crate (`crates/planet_rendering/`)

Thin Bevy rendering layer. No generation logic.

- `PlanetRenderingPlugin` — registers `PlanetMaterial`, `GasGiantMaterial`, `RingMaterial`.
- `PlanetMaterial` — cubemap height/albedo textures. Uses `assets/shaders/planet_impostor.wgsl`.
- `GasGiantMaterial` — gas giant cloud/haze rendering. Uses `assets/shaders/gas_giant.wgsl`.
- `RingMaterial` — ring system rendering. Uses `assets/shaders/ring.wgsl`.
- `bake_from_body_data()` — consumes `terrain_gen::BodyData` → `PlanetTextures`.

### Planet editor (`crates/planet_editor/`)

Standalone Bevy binary for interactive planet preview. Loads `solar_system.ron`, selects a body, runs terrain pipeline, renders with `PlanetMaterial` via billboard mesh. Uses `bevy_egui` for UI.

### Data flow

```
assets/solar_system.ron
  → [parsing] SolarSystemDefinition
  → [PatchedConics] body positions at any time t
  → [Simulation::step] per frame → live ship state, consumes ManeuverNodes
  → [propagate_flight_plan] background → FlightPlan (legs of burn?+coast NumericSegments)
  → [bridge] → rendering, maneuver UI, collision warnings
```

### Design invariants

- **One propagator everywhere.** Live stepping and prediction route through the same `ShipPropagator` (today, `KeplerianPropagator`). Never split them or numerical divergence appears between "where ship is" and "where it will be."
- **`BodyStateProvider` is the abstraction boundary.** Body positions are always queried through this trait. `PatchedConics` is the current impl; a precomputed ephemeris could replace it without touching simulation or rendering.
- **Physics crate has no Bevy.** All physics logic must remain in `thalos_physics`. `thalos_game` is only presentation and input.
- **`TrajectorySample` carries its own metadata.** `anchor_body` + `ref_pos` travel with each sample so the renderer can pin to its parent body without a per-sample ephemeris query.
- **Terrain gen stages are pure transforms.** Each `Stage` reads/writes `BodyBuilder` only. Pipeline handles seeding and ordering. No ambient state.

### Assets

- `assets/solar_system.ron` — full solar system definition (RON format with `#![enable(implicit_some)]`), consumed by game and editor.
- `assets/shaders/planet_impostor.wgsl` — impostor shader (3-layer: baked cubemap low-freq, SSBO mid-freq features, shader-synthesized high-freq detail).
- `assets/shaders/gas_giant.wgsl` — gas giant cloud/haze/rim rendering.
- `assets/shaders/ring.wgsl` — planetary ring system shader.
- `assets/shaders/stars.wgsl`, `galaxy.wgsl` — celestial sphere rendering from procedural catalog.
- `assets/shaders/lighting.wgsl` — shared lighting library (loaded via `load_shader_library!`).

### Documentation (`docs/`)

- `solar_system.md` — per-body reference with scale philosophy (hybrid 1:1/1:2/1:3 scale rationale) and formation scenario.
- `surface_generator_design.md` — architecture design doc for the 3-layer terrain representation, BodyData/BodyBuilder contract, sampling contract.
- `celestial.md` — celestial sphere design: source model, spectrum, generation, rendering pipeline.
- `new_physics.md` — future N-body ephemeris proposal (currently using patched conics).
- `gen/mira_processes.md` — stage-by-stage pipeline recipe for Mira.
