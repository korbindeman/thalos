# Cinematics: director, replay, and scripted control

Reference key `cine §N`. Status lives in [backlog.jsonl](../backlog.jsonl)
(`CINE-*`) — this doc holds strategy and rationale only.

## 1. What this is

**One authored document that describes a shot list, and one renderer that can
sample it at any granularity.** The same file is:

- a **still** at an arbitrary time — `--at 4.5`
- a **keyframe set** — one high-resolution image per authored key, which is the
  agent's iteration loop
- a **full render** — every frame at `n / fps`, muxed to MP4/WebM
- later, a **regression probe** in CI — the same render with per-frame hashes
  compared against a baseline

Output kind and output resolution are **request parameters, never document
identity** — exactly the rule [capture.md §3](../development/capture.md) already
applies to render extent vs. viewpoint identity. Directing a trailer and
screenshotting a keyframe are the same act at different sampling rates, and
nothing in the pipeline may fork between them.

The document is authored three ways, all producing the same schema: by hand (an
agent writes JSON), in-game (fly it, then set framing on a timeline), and by
promoting an existing `assets/viewpoints.json` entry into a one-key shot.

## 2. What already exists — do not re-invent

This is largely the *widening* of [CAP-4](../development/capture.md) from
"render a camera track" to "author, replay, and script", plus the control seam.

| Existing | What it already gives us |
|---|---|
| [capture.md §5](../development/capture.md) + `CAP-4` | Fixed frame time `n/fps`, readback ring, background lossless frame writer, external FFmpeg mux, run-bundle contract, per-frame-hash exit criterion |
| ADR-20260721T194629Z | Video is a **mode of the one frame producer**, not a second orchestrator. Encoding stays out of the render loop |
| `thalos_capture_protocol::Viewpoint` | A portable, versioned, no-ECS camera key: body-frame `f64` pose + quaternion + `CameraOptics` + `sim_time_s` + spawn scenario. A shot key is nearly this struct already |
| `CameraOptics` (ADR-20260729T020148Z) | One optics authority — focal length is a first-class animatable channel, not an ad-hoc FOV |
| `thalos_navigation` (ADR-20260730T005746Z) | Pure, stateless guidance; `RouteFrame`, waypoints, Dubins, VNAV. The ADR names *"a new route kind extends `thalos_navigation` in one place"* and *"the same guidance drives a flight director, an autopilot, or nothing at all"* — scripted craft control is the anticipated extension, not a new system |
| `control_bus.rs` `DemandSource` | Priority-arbitrated attitude bus with a documented extension point: *"a new source is a new `DemandSource`"* |
| `SimClock` | The single-writer, single fold-point for every sim pause source — the one place a driven clock has to be installed |
| `TerrainReadiness` + the `terrain` receipt block | Per-frame proof that the ground rendered at the detail the shot authored |

## 3. The foundation: a driven clock (`cine §3`)

`sync_sim_clock` publishes `delta_s = wall-clock delta`, and Avian's
`Time<Physics>` follows `Time<Virtual>`, also wall-clock. **Frame *n* cannot be
rendered at *n*/60 s while the frame itself takes 300 ms.** Every other item
here is downstream of fixing this.

`SimClock` gains a mode — `Wall` for interactive, `Driven { dt }` for offline
render — and the same sole-writer system advances `Time<Physics>` by that dt in
driven mode. No new authority, no second clock: this is cashing in the
single-writer invariant that resource already carries.

Two things fall out immediately, independent of any video work:

- **Existing stills stop being machine-load-dependent.** Warmup frames are
  currently wall-clock, so a busy machine settles a preset differently than an
  idle one.
- **Per-frame hashing becomes meaningful**, which is what makes determinism
  measurable rather than asserted.

**Determinism is not promised until measured.** The driven clock removes the
dominant source; async tile-stream ordering, temporal cloud history, and any
unseeded RNG remain. The per-frame hash lands *with* the frame writer, not
after it, because it is the instrument that finds the rest.

## 4. The document (`cine §4`)

Lives in `thalos_capture_protocol` beside `Viewpoint` — that crate is already
"portable control-plane types, no ECS", already the shared schema home for the
in-game manager, agent edits, CLI, and headless runtime. Versioned the same way.

Shape:

- `scene` — spawn scenario, body, sim epoch, graphics/quality overrides, seed.
  This is what `Viewpoint` already carries, hoisted to the sequence.
- `actors` — see §6.
- `shots[]` — ordered, each with a duration and **three independent tracks**.

### 4.1 Three tracks, each with its own anchor

A pose is meaningless without a frame, and this world spans 10⁹ m. `Viewpoint`
is body-fixed, which is right for a landscape still, wrong for a chase cam
(craft at 200 m/s), and wrong for orbit (body-fixed carries ~460 m/s of surface
rotation at the equator). So the camera is not one track:

| Track | Anchors |
|---|---|
| **position** | `BodyFixed{body}` · `BodyInertial{body}` · `ActorLocal{actor, offset}` |
| **aim** | `Keyframed` rotation · `LookAt{actor \| point, damping}` |
| **optics** | focal length over time (`CameraOptics`, one authority) |

Interpolation happens **in the anchor's own frame, in `f64`** — Catmull-Rom for
position, squad for rotation, per-key ease. **An anchor change is only legal at
a cut.** That single rule removes an entire class of "the camera flew to the
sun" bugs, and it is why cuts are part of the schema rather than an editing
convenience.

## 5. The director (`cine §5`)

One pure evaluator: `t → (camera pose, optics)`. Used *identically* by the
in-game preview/scrub and by the headless renderer — the interactive timeline
must not be a look-alike, the same way `just nd-preview` renders the real ND
pipeline rather than a mock. The evaluator is Bevy-free; the runtime plugin that
drives a camera from it is not.

## 6. Actors (`cine §6`)

Two kinds, deliberately different mechanisms:

- **`Recorded { track }`** — you flew it; the pose was sampled; playback is on
  rails. **This is the authoring medium for hand-directed shots.**
- **`Scripted { program }`** — waypoints / throttle / attitude program, driven
  through `thalos_navigation` guidance into `ControlDemand`, re-simulated live.
  This is what CI and agent-authored sequences use.

**Rejected: input-replay-plus-resimulation** (record stick inputs + seed,
re-derive the trajectory). It is the tempting shape because the files are tiny
and the result stays "live", but it requires real bit-level determinism across
floating-point, wall-clock, and tile-load ordering — a research project whose
failure mode is a craft that silently diverges halfway through a take. State
playback is correct by construction and is what film-style camera work actually
needs. See ADR-20260730T212556Z.

Scripted control has one hard prerequisite: **the throttle setpoint is not on
the control bus yet.** `control_bus.rs` documents this as the next step; a
script that flies an aircraft cannot work without it. Folding it in is
independently valuable and is sequenced first.

## 7. Output (`cine §7`)

The renderer's job is *not* "record the game at 60 fps". It is:

> **render N independently settled stills that happen to be 1/60 s apart.**

That reframing is what makes it robust — every frame reuses the existing
per-still readiness machinery (`TerrainReadiness`, the residency brake gate,
the fatal-pipeline validation of BL-20) rather than inventing a real-time path
that would silently record LOD pop as if it were a real artifact.

Cost follows directly: the **first frame of each shot** pays a full settle;
frames along a continuous camera path are nearly free because the tiles are
already resident; **every cut pays again.** Render cost therefore scales with
cut count far more than with clip length — and it is precisely why the
keyframe-still mode is the agent's loop and the full render is not.

**Offline quality tier.** Because this is not real-time, an offline render can
afford what an interactive frame cannot: supersample-and-downsample, a raised
tile budget, longer settle, more warmup for temporal effects.
`CaptureGraphicsOverrides` is already a typed patch over deterministic capture
defaults — the quality tier extends that, it does not become a second settings
path. High resolution is first-class, not an afterthought: a trailer frame and a
4K keyframe still are the same request with a different extent.

**FFmpeg is assumed present on the machine** and invoked after rendering
completes (never inside the render loop — ADR-20260721T194629Z rejected that on
codec-backpressure grounds). A missing binary is a clear, early, actionable
failure with the frame sequence still on disk, not a lost render.

**CI is a future consumer, not a current design driver.** Nothing CI-specific
gets built now. What we do owe it is the two things it would need anyway: the
per-frame hash and the run-bundle receipt. CI then becomes a flag over the same
tool.

## 8. Sequencing

Agent-first, decided with the user 2026-07-30: the document format is proven by
agent use before an editor is built against it.

| Gate | Content | Why here |
|---|---|---|
| **A** | Driven `SimClock` | Blocks everything; independently fixes still reproducibility |
| **B** | Document + evaluator + `--at` / `--keyframes` stills | No UI, no encoder, no video. Agents author and iterate immediately; the schema settles under real use before anything expensive is built on it. Highest value per token in the plan |
| **C** | Frame sequence + per-frame hash + FFmpeg mux | Literally CAP-4's remaining scope once A+B exist |
| **D** | In-game recording + timeline UI | After B, so the editor targets a settled format. Native Bevy UI (`thalos_ui`), not egui |
| **E** | Throttle on the control bus → scripted actors + waypoint route kind | Different subsystem; runs in parallel with C and D |

C, D, and E are mutually independent once A and B land.

## 9. Decision log

- **2026-07-30** — Scoped with the user. Four forks settled: **agent-first**
  ordering; **state playback** over input replay; **FFmpeg assumed installed**;
  and CI deferred, with the utility required to *adapt* to it rather than be
  shaped by it — hence "output kind is a request parameter" as the load-bearing
  design rule. Recorded in ADR-20260730T212556Z.
