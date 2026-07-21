# ADR-20260721T192218Z-persistent-visual-iteration: Reuse one hot-patched renderer for visual iteration

- **Status:** Accepted
- **Date:** 2026-07-21
- **Supersedes:** ADR-20260721T032344Z-isolated-headless-visual-comparisons for the default development loop; retains its isolated lane for final evidence

## Context

The deterministic screenshot and comparison tools made graphics work observable,
but each image still paid the full game startup, world construction, renderer
warm-up, and executable link. A three-way comparison paid that cost three times.
That makes agent feedback slow enough to suppress useful experiments.

Bevy 0.19 integrates Dioxus/Subsecond hot patching for ECS systems. Its asset
watcher also reloads file-backed shaders, and `embedded_watcher` extends this to
WGSL registered through `embedded_asset!`. These mechanisms can preserve the
already-built world and GPU renderer across source edits and capture variants.

The earlier ADR correctly rejected multiple live cameras and correctly identified
cross-variant state as a threat to authoritative comparisons. It treated process
isolation as the only acceptable workflow, however, so the safest path was also
the only iteration path.

## Decision

`just screenshot` and `just compare` use one persistent, headless
`thalos_game` process managed by `tools/visual_capture.py`.

- The Dioxus CLI launches the game with `thalos_game/dev-iteration`, which enables
  Bevy hotpatching plus file-backed and embedded asset watching. It intentionally
  does not combine this binary with `bevy/dynamic_linking`; the hotpatch binary is
  one stable development fingerprint and release/build/game paths remain unchanged.
- A versioned file protocol under `tools/diagnostics/` submits captures and reports
  readiness/completion. `just screenshot` and every live-compatible comparison
  variant use this same process and off-screen target.
- A request reapplies capture-only SSAO, terrain inspection, ocean, and cloud
  settings. Cloud temporal history is invalidated before each request. The first
  capture uses the preset's full warm-up; later captures default to a 60-frame
  settle period.
- Changing preset or viewport restarts the managed process because those values
  select the boot world and allocate viewport-sized render resources. Type/layout,
  plugin/schedule, resource initialization, and other structural Rust edits also
  require `just capture-stop` followed by the next capture. Ordinary Bevy system
  body edits hot-patch in place.
- WGSL changes under `assets/` or embedded from crates reload in place. The client
  waits for the corresponding Rust/shader reload notification before requesting
  an image, then allows render pipelines to settle.
- `just screenshot-cold` and `just compare-cold` retain clean-process, full-warm-up
  verification. `terrain-culling` automatically uses this path because its value
  is consumed during render-pipeline specialization and cannot be changed live.
- The one-camera rule remains. Persistence does not introduce another production
  view, split viewport, or comparison renderer.

## Consequences

The normal agent loop pays compilation, linking, startup, terrain population, and
GPU initialization once. A screenshot after a compatible Rust or shader edit pays
only the patch/reload, settle frames, and readback; an N-way comparison reuses the
same world for all N captures.

Persistent comparisons are rapid diagnostic evidence, not proof of isolation.
Caches, streamed terrain, and other global state can survive despite the explicit
temporal resets. A result used for final regression acceptance is rerun with the
cold command. The manifest records which lane produced it.

The Dioxus CLI becomes a one-time developer prerequisite. The controller owns only
the exact launcher/PID recorded in `tools/diagnostics/`, and exposes status/stop
commands so it does not leave an unmanageable background renderer.
