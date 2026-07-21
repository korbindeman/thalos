# Capture architecture

**Status:** target accepted 2026-07-21; the current persistent screenshot lane
is the working foundation, and CAP-1–CAP-4 migrate it to the target below.

## 1. Goal

Most graphics development should be possible without opening the interactive
game. A capture request must boot or reuse the real world, use the real render
camera and render graph, produce deterministic evidence, report invalid render
pipelines as failures, and support stills, comparisons, and video through one
contract.

The two operating lanes are both permanent:

- **Persistent:** hot-patched world/GPU reuse for the shortest agent feedback loop.
- **Cold:** clean process, full warm-up, isolated state for acceptance evidence.

## 2. Process and package shape

```text
thalos_game ───────────────┐
                           ├── thalos_runtime ── domain/simulation/render/interface
thalos_capture_host ───────┘
          │
          └── thalos_capture_runtime ── thalos_capture_protocol
                                                ▲
                                                │
                                      thalos_capture CLI
```

`thalos_runtime` is the sole plugin-graph/composition authority. Interactive
and headless launchers are thin. The capture host does not have a second renderer;
it selects a platform shell with no primary window and adds the capture runtime.

### 2.1 Runtime modules

The current `crates/game/src` moves first into a library and is organized by
responsibility, not one crate per feature:

```text
crates/runtime/game/src/
  lib.rs
  app_builder.rs
  boot/              loading, settings, menus, scenario entry
  gameplay/          craft systems, fuel, aero, control, EVA, maneuver
  facilities/        structures, runway, base editor, space center
  simulation/        canonical/local bridge, regimes, clock
  presentation/      camera, render drivers, HUD, map, photo mode
  construction/      craft view and in-game construction editor
```

Crate extraction is reserved for a real dependency boundary or a second
consumer. Ordinary feature size is handled with modules.

## 3. Typed capture contract

`thalos_capture_protocol` is Serde-only and defines a versioned request/result:

```text
CaptureSpec
  scene             canonical boot scenario, body, seed and time
  camera            preset, saved body-fixed pose, or camera track
  output            still, frame sequence, or video
  viewport          render and output extents
  timeline          simulation time, frame rate, duration, preroll
  warmup            scene and temporal convergence policy
  render_overrides  typed SSAO/cloud/terrain/ocean/debug variants
  diagnostics       requested logs, timings and debug channels
```

Environment variables remain only as a temporary compatibility adapter. New
capture features extend `CaptureSpec`; they do not add another environment-key
protocol.

## 4. Capture runtime

`thalos_capture_runtime` owns one explicit state machine:

```text
Idle → LoadingScene → Warming → Capturing → DrainingReadbacks → Complete → Idle
```

It owns the off-screen target, real-camera retargeting, deterministic site and
camera resolution, overlay policy, temporal cut/reset epoch, diagnostic
overrides, readback, and result validation. A request is not successful merely
because a PNG exists: shader compilation, pipeline validation, missing render
layers, write failures, and incomplete frame sequences fail the result.

The interactive launcher also installs the shared capture runtime for F2 and F8;
those are interactive clients of the same capture/perspective types.

## 5. Still, comparison, and video

The Rust `thalos_capture` CLI is the single public orchestrator:

```text
thalos capture shot spaceport-aerial
thalos capture compare spaceport-aerial ssao
thalos capture record cloud-sunset --seconds 8 --fps 60
thalos capture verify
thalos capture status
thalos capture stop
```

`just screenshot`, `just compare`, and their cold variants remain stable aliases.
The CLI owns host lifecycle, typed matrices, provenance, artifact assembly, and
hot-reload readiness. The persistent host reuses compatible scene state; the
cold lane starts a fresh host.

Video uses exact frame time `n / fps`, not wall time. The runtime renders into a
small readback ring while background workers write a lossless frame sequence.
FFmpeg muxes the finished sequence into MP4/WebM after rendering. Source frames
may be retained for evidence or discarded after a verified encode. Camera tracks
use body-fixed/inertial coordinates and the same cut rules as interactive views.

## 6. Artifact contract

Source tools and generated evidence are separate:

```text
artifacts/
  visual/latest/<preset>.png
  visual/runs/<capture-id>/
    manifest.json
    process.log
    diagnostics.jsonl
    still.png | frames/*.png
  video/<capture-id>/
    manifest.json
    video.mp4
    frames/                 optional
  diagnostics/
```

The latest tree is convenient and overwritten. A run bundle is immutable and
self-describing: revision/dirty state, lane, request, dimensions, frame clock,
renderer fingerprint, pipeline validity, output hashes, and metrics.

## 7. Migration

### CAP-1 — Shared game runtime and thin interactive launcher

Move the current game modules into `thalos_runtime`; expose one `AppBuilder` and
make `thalos_game` a thin binary. Preserve current behavior and cold capture as a
temporary internal mode.

### CAP-2 — Capture protocol, runtime, and headless host

Extract preset/configuration, server, camera, readback, diagnostics, and F2/F8
logic from `screenshot.rs`. Add the thin `thalos_capture_host` application and
prove it renders every canonical preset through the shared runtime.

### CAP-3 — Rust CLI, comparisons, and artifacts

Replace `visual_capture.py` and `visual_compare.rs`, retain `just` aliases,
introduce typed request/result manifests, move generated evidence to
`artifacts/`, and make pipeline/render errors fatal to a request.

### CAP-4 — Deterministic frame sequences and video

Add fixed capture time, camera tracks, readback pipelining, background frame
writes, resumable/validated sequences, and external FFmpeg muxing. Verify a
short motion/temporal probe in both persistent and cold lanes.

## 8. Exit criteria

- Interactive and capture apps demonstrably use the same runtime/plugin graph.
- Every current preset and typed comparison is available through the Rust CLI.
- Persistent capture still hot-patches Rust systems and reloads WGSL.
- Cold results match the prior canonical framing before old entry points retire.
- Invalid render pipelines fail requests.
- A deterministic video rerun produces the same frame count, timestamps,
  framing, and per-frame hashes before lossy muxing.
