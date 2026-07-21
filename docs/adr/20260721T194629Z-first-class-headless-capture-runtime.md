# ADR-20260721T194629Z-first-class-headless-capture-runtime: Headless capture is a first-class application over one shared runtime

- **Status:** Accepted
- **Date:** 2026-07-21

## Context

Headless screenshots are already the primary verification and graphics-iteration
loop. The persistent hot-patched lane in
ADR-20260721T192218Z-persistent-visual-iteration removes most steady-state
startup and link latency, but its implementation remains spread across the
`thalos_game` binary, a 2.9 kLOC `screenshot.rs`, a Cargo example, a Python
process manager, environment-variable configuration, and generated files under
the tool-source tree. Adding deterministic video to that shape would create a
second capture path or further enlarge the conditional game mode.

Interactive play, still capture, comparisons, and video must render the same
world through the same camera and plugin graph. Agents need a fast persistent
lane, while final evidence still needs a clean-process lane.

## Decision

Extract the current game composition into reusable `thalos_runtime`. Keep two
thin applications: `thalos_game` for interactive play and
`thalos_capture_host` for off-screen rendering. Both build the same runtime;
only their platform shell differs.

Make capture a subsystem with three reusable boundaries:

- `thalos_capture_protocol`: Serde-only, versioned request/result types.
- `thalos_capture_runtime`: Bevy capture state machine, real-camera retargeting,
  deterministic framing/timeline, temporal resets, diagnostics, GPU readback,
  and interactive F2/F8 support.
- `thalos_capture`: a lightweight Rust CLI that manages persistent/cold hosts,
  stills, typed comparisons, artifact assembly, and video jobs. It replaces
  `tools/visual_capture.py` and the `visual_compare` Cargo example.

Still images and video use one `CaptureSpec` and one frame producer. Video is a
deterministic fixed-time frame sequence with pipelined readback; external FFmpeg
muxes that sequence into delivery formats. Encoding never controls simulation
or render time.

The persistent hot-patched lane remains the default iteration path. The cold
host remains the acceptance path. Both return the same versioned result manifest
and must fail a request on shader/render-pipeline validation errors rather than
accepting a partially rendered PNG.

## Alternatives

- **Keep headless capture as environment-controlled branches in `main.rs`** —
  rejected because the composition root and capture lifecycle remain inseparable.
- **Build a separate simplified renderer** — rejected because it would diverge
  from gameplay lighting, LOD, temporal history, and view-dependent systems.
- **Retain Python plus the Cargo example as the public orchestration layer** —
  rejected because the protocol and comparison schema cannot be shared as Rust
  types and video would add another orchestration surface.
- **Encode video inside the render loop** — rejected because codec backpressure
  would make capture timing nondeterministic.

## Consequences

The current game package becomes a reusable runtime plus thin launchers before
the wider directory move. Capture gains more packages, but each seam has a clear
compile boundary: the CLI and protocol remain lightweight, while only the host
and capture runtime link Bevy. F2/F8, agent stills, comparisons, and video share
one implementation. Existing `just` commands remain compatibility aliases while
the typed CLI becomes the canonical interface.
