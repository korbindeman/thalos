# ADR-20260730T212556Z-cinematics-one-document-many-samplings: One sequence document, sampled at many granularities; replay is recorded state, not replayed input

- **Status:** Accepted
- **Date:** 2026-07-30

## Context

Thalos needs headless video (CI and development), an in-game camera
director/replay mode for hand-directing shots, and scriptable/autonomous craft
control that an agent can use to drive both. The user's framing is a **cinematic
trailer**: directed shots with cuts and moving cameras, renderable at the
current state of the tree, yielding both a full render and high-resolution
stills, with CI as a later consumer.

[CAP-4](../development/capture.md) already scoped the rendering half
(fixed frame time, camera tracks, readback ring, FFmpeg mux) and
ADR-20260721T194629Z already decided video is a mode of the one frame producer.
Neither settled the questions that decide the *shape* of the feature: what the
authored artifact is, how a hand-flown take is reproduced, and how a camera pose
stays meaningful across 10⁹ m of BigSpace.

Two failure shapes were visible from the start. The first is a **split between
"screenshot" and "video"** — two entry points, two configuration surfaces, two
notions of framing that agree at first and drift, which is the exact failure
mode ADR-20260730T005746Z documents for duplicated derivations. The second is
promising **determinism** the engine cannot currently deliver: `SimClock`
publishes a wall-clock delta and Avian's `Time<Physics>` follows
`Time<Virtual>`, so *nothing* in the simulation is reproducible frame-to-frame
today.

## Decision

**One sequence document; output kind and resolution are request parameters.**
The document (in `thalos_capture_protocol`, beside `Viewpoint`) describes a
scene, actors, and an ordered shot list. Sampling it at one time yields a still,
at its authored keys yields a keyframe set, and at `n / fps` yields a render.
There is no separate "video config". This mirrors the existing rule that render
extent may size GPU resources but is never viewpoint identity.

**A driven simulation clock.** `SimClock` gains `Wall` / `Driven { dt }`, and
its existing sole-writer system advances `Time<Physics>` by the same dt in
driven mode. This is the one place it can go: `SimClock` is already the single
fold-point for every pause source.

**Replay is recorded state, not replayed input.** A hand-flown take is sampled
as a pose track and played back on rails. Live re-simulation exists separately,
as *scripted* actors driven through `thalos_navigation` guidance into
`ControlDemand`.

**A camera is three tracks, each with its own anchor**, interpolated in the
anchor's own frame in `f64`: position (`BodyFixed` / `BodyInertial` /
`ActorLocal`), aim (`Keyframed` / `LookAt`), optics. **An anchor change is legal
only at a cut.**

**A rendered frame is a settled still, not a real-time frame.** Every frame
passes the same readiness gates a `just screenshot` still does.

## Alternatives

- **A separate video pipeline alongside stills** — rejected. It is the shape
  that produces two framing authorities which drift, and it would make
  "screenshot this keyframe at 4K" a feature request instead of a sampling
  parameter. The user's own framing (stills *and* renders from the same
  direction) is what makes the single-document rule load-bearing.
- **Input replay with re-simulation** (record stick + seed, re-derive the
  trajectory) — rejected, and this is the one a future agent will be tempted to
  retry: the files are tiny, the take stays "live", and it re-simulates against
  new physics. It requires bit-level determinism across floating-point,
  wall-clock, *and* async tile-load ordering, none of which hold today. Its
  failure mode is a craft that silently diverges mid-take, which is worse than
  not having the feature. Revisit only if per-frame hashes ever demonstrate
  actual determinism.
- **Keep the wall clock and render as fast as the machine allows** — rejected:
  frame *n* must be at *n*/fps, and a wall clock also makes today's still
  captures settle differently on a busy machine than an idle one.
- **A single body-fixed camera track** (extending `Viewpoint` directly) —
  rejected: body-fixed carries ~460 m/s of surface rotation at the equator and
  is unusable for orbital or chase framing, and interpolating a single track
  across a frame change is how a camera ends up kilometres off course.
- **Encode inside the render loop** — already rejected by
  ADR-20260721T194629Z (codec backpressure); restated here because a "record"
  verb invites it.
- **Design for CI now** — rejected as premature at the user's direction. CI
  becomes a flag over the same tool; what it needs that we owe anyway is the
  per-frame hash and the run-bundle receipt.
- **Build the in-game timeline editor first** — rejected with the user
  (agent-first). An editor built against an unsettled schema pays for every
  schema change twice.

## Consequences

- **Determinism is measurable, not asserted.** The per-frame hash ships with the
  frame writer, because it is the instrument that finds the remaining
  non-determinism (tile-stream order, temporal cloud history, unseeded RNG). No
  reproducibility claim is made before it reports.
- **Render cost scales with cut count, not clip length.** A continuous camera
  path keeps its tiles resident; every cut pays a full settle. Directing
  decisions therefore have a cost model, and the keyframe-still mode — not the
  full render — is the agent's iteration loop.
- **Offline gets a quality tier interactive cannot afford** (supersampling,
  raised tile budget, longer settle) as an extension of the existing typed
  `CaptureGraphicsOverrides` patch, not a second settings path.
- **Scripted control has a prerequisite**: the throttle setpoint must join the
  control bus, which `control_bus.rs` already documents as its next step.
- **Adding a route kind or an autopilot mode does not touch the director**, and
  adding a shot type does not touch navigation — the ADR-20260730T005746Z seam
  holds in both directions.
- `assets/viewpoints.json` remains the still-camera registry; a viewpoint
  promotes into a one-key shot rather than being replaced by one.
