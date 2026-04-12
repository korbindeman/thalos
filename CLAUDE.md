# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
just run      # cargo run -p thalos_game
just editor   # cargo run -p thalos_planet_editor
just build    # cargo build --workspace
just test     # cargo test -p thalos_physics -p thalos_terrain_gen
just clippy   # cargo clippy --workspace

# Run a single test
cargo test -p thalos_physics -- test_name
```

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
- **`thalos_planet_rendering`** — Bevy rendering for generated planets (cubemap impostor)
- **`thalos_planet_editor`** — interactive planet editor tool

Core separation: physics and terrain_gen are pure Rust libraries; game, planet_rendering, and planet_editor are Bevy consumers. `crates/planet_gen/` exists but is NOT in the workspace — older spec-faithful approach, kept for reference.

### Physics crate (`crates/physics/`)

Key modules and their roles:

- `types` — `StateVector`, `BodyDefinition`, `TrajectorySample` (each sample carries dominant body ID, perturbation ratio, step size — used directly by the renderer)
- `patched_conics` — `BodyStateProvider` implementation using analytical Keplerian orbits. Bodies form a parent-child tree; evaluated in topological order. This is the authoritative source of body positions.
- `integrator` — Hybrid: symplectic leapfrog (fixed timestep, energy-conserving) for low-perturbation regions; adaptive RK45 (Dormand-Prince) when perturbation ratio exceeds threshold. Live stepping and trajectory prediction share the same `IntegratorConfig` — this is intentional and critical.
- `simulation` — Central `Simulation` struct: owns ship state, integrator, `ForceRegistry` (gravity + thrust), `ManeuverSequence`. `step()` is called each frame.
- `trajectory` — Background trajectory prediction engine. Runs on a worker thread, progressively refines (coarse → fine budget). Cancels and restarts on maneuver edits.
- `maneuver` — `ManeuverNode`: time, delta-v in prograde/radial/normal frame, optional reference body.
- `forces` — Force registry. MVP: gravity (sum over all bodies), thrust. Extensible by adding force functions.
- `parsing` — Loads `assets/solar_system.kdl` into `BodyDefinition[]`.

### Game crate (`crates/game/`)

- `bridge` — The core adapter. Calls `Simulation::step()` each frame, manages async prediction worker thread, syncs maneuver edits, handles warp controls.
- `rendering` / `trajectory_rendering` — Reads body positions from `PatchedConics` and trajectory samples from prediction. Cone width is derived from perturbation ratio directly.
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

### Planet rendering crate (`crates/planet_rendering/`)

Thin Bevy rendering layer. No generation logic.

- `PlanetRenderingPlugin` — registers `PlanetMaterial` asset type.
- `PlanetMaterial` — Bevy `Asset + AsBindGroup`. Binds cubemap height/albedo textures and uniforms. Uses `assets/shaders/planet_impostor.wgsl`.
- `bake_from_body_data()` — consumes `terrain_gen::BodyData` → `PlanetTextures` (albedo + height image handles).

### Planet editor (`crates/planet_editor/`)

Standalone Bevy binary for interactive planet preview. Loads `solar_system.kdl`, selects a body, runs terrain pipeline, renders with `PlanetMaterial` via billboard mesh. Uses `bevy_egui` for UI.

### Data flow

```
assets/solar_system.kdl
  → [parsing] BodyDefinition[]
  → [PatchedConics] body positions at any time t
  → [Simulation::step] per frame → TrajectorySample (ship)
  → [trajectory worker] background → TrajectoryPrediction (legs between maneuver nodes)
  → [bridge] → rendering, maneuver UI, collision warnings
```

### Design invariants

- **Same integrator config everywhere.** Live stepping and prediction use the same `IntegratorConfig`. Never split them or numerical divergence appears between "where ship is" and "where it will be."
- **`BodyStateProvider` is the abstraction boundary.** Body positions are always queried through this trait. `PatchedConics` is the current impl; a precomputed ephemeris could replace it without touching simulation or rendering.
- **Physics crate has no Bevy.** All physics logic must remain in `thalos_physics`. `thalos_game` is only presentation and input.
- **`TrajectorySample` carries its own metadata.** Dominant body, perturbation ratio, and step size travel with each sample so the renderer needs no heuristics to draw uncertainty cones.
- **Terrain gen stages are pure transforms.** Each `Stage` reads/writes `BodyBuilder` only. Pipeline handles seeding and ordering. No ambient state.

### Assets

- `assets/solar_system.kdl` — full solar system definition (KDL format), consumed by game and editor.
- `assets/simple_solar_system.kdl` — simpler variant for testing.
- `assets/shaders/planet_impostor.wgsl` — impostor shader (3-layer: baked cubemap low-freq, SSBO mid-freq features, shader-synthesized high-freq detail).

### Documentation (`docs/`)

- `solar_system.md` — per-body reference with scale philosophy (hybrid 1:1/1:2/1:3 scale rationale) and formation scenario.
- `surface_generator_design.md` — architecture design doc for the 3-layer terrain representation, BodyData/BodyBuilder contract, sampling contract.
- `mira_processes.md` — stage-by-stage pipeline recipe for Mira (Sphere → Differentiate → Megabasin → Cratering → MareFlood → Regolith → SpaceWeather).
