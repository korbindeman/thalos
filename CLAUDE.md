# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Thalos

N-body orbital mechanics sandbox built with Rust and Bevy. Ships fly through a full gravitational simulation (no patched conics). The solar system is defined in `assets/solar_system.kdl`, precomputed into an ephemeris at startup, and the ship is propagated through all bodies' gravity fields in real time.

## Build & Run

```bash
cargo build                          # build everything
cargo run -p thalos_game             # run the game (must be from workspace root — loads assets/solar_system.kdl)
cargo test -p thalos_physics         # run physics tests
cargo test -p thalos_physics -- test_name  # run a single test
cargo clippy --workspace             # lint
```

Bevy uses `dynamic_linking` feature in dev for faster incremental builds. Release builds use thin LTO with single codegen unit (`[profile.release]` in root Cargo.toml).

## Architecture

Two-crate workspace with a strict dependency boundary:

**`crates/physics` (`thalos_physics`)** — Pure Rust, no Bevy. All simulation math lives here.
- `types` — Core data types (`StateVector`, `BodyDefinition`, `SolarSystemDefinition`, `TrajectorySample`), KDL parser (`load_solar_system`), orbital element conversions. `BodyId` is `usize` indexing into the bodies vec.
- `ephemeris` — N-body precomputation via RK4 (1-hour fixed step). Adaptive per-body sampling driven by positional curvature threshold. Hermite interpolation lookups in O(log n). Star is pinned at origin.
- `integrator` — Hybrid ship propagator: Velocity-Verlet (symplectic) for stable coasting, Dormand-Prince RK45 for perturbed regimes. Switches on perturbation ratio with hysteresis.
- `forces` — `ForceRegistry` trait-object pattern. `GravityForce` and `ThrustForce` implementations. New forces plug in without touching the integrator.
- `trajectory` — Trajectory prediction: chains maneuver nodes, detects stable orbits, computes cone width from integrator metadata.
- `maneuver` — Maneuver node data model and sequencing.

**`crates/game` (`thalos_game`)** — Bevy application. Depends on `thalos_physics`.
- `main` — Loads KDL, builds ephemeris, resolves ship initial state relative to homeworld, launches Bevy app.
- `rendering` — Body spawning, orbit lines (precomputed once from ephemeris), ship marker, LOD crossfade (sphere → billboard).
- `bridge` — Physics↔ECS bridge: advances sim clock with sub-stepping at high warp, periodic trajectory prediction, warp controls (`+`/`-`/`\`).
- `camera` — KSP-style orbit camera with spherical coordinates. Distance in metres (f64) for full range from low orbit to system scale.
- `trajectory_rendering` — Draws predicted trajectory and cone from `PredictedTrajectory` resource.

## Coordinate System

Heliocentric inertial frame: star at origin, ecliptic = XZ plane, Y up. All physics in metres (f64). Rendering converts to render units via `RENDER_SCALE = 1e-6` (1 render unit = 1000 km) to stay within f32 precision.

## Key Design Decisions

- **All units are SI** (metres, seconds, kg). `G` constant in `types.rs`.
- **`glam::DVec3`** for all physics vectors (f64 precision). Bevy's `Vec3` (f32) only at the rendering boundary.
- **Body states are immutable** — the ship never affects celestial bodies. Ephemeris is a fixed lookup table.
- **No sphere-of-influence boundaries** — dominant body transitions are smooth gradients of gravitational influence stored per trajectory sample.
- **Solar system defined in KDL** (`assets/solar_system.kdl`) — body masses, radii, orbital elements, colors, parent relationships.
