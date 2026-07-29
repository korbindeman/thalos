# Capture architecture

**Status:** migration active 2026-07-24. CAP-1 has landed compile-clean; the
interactive and headless apps are thin shells over `thalos_runtime`. The typed
protocol, dedicated capture host, and single Rust controller now own stills,
multi-scene batches, comparisons, lifecycle recovery, and evidence validation;
the coupled capture systems still move out of the runtime incrementally.

## 1. Goal

Most graphics development should be possible without opening the interactive
game. A capture request must boot or reuse the real world, use the real render
camera and render graph, produce deterministic evidence, report invalid render
pipelines as failures, and support stills, comparisons, and video through one
contract.

The two operating lanes are both permanent:

- **Persistent:** stable world/GPU reuse plus WGSL hot reload for the shortest
  agent feedback loop. Rust/manifest edits rebuild and restart automatically.
- **Cold:** clean process, full warm-up, isolated state for acceptance evidence.

Reliability is a contract, not a convenience. Valid use must either produce a
decodable image from a healthy renderer or return one actionable failure. The
controller rejects unknown scene names, stale/dead heartbeats, missing or corrupt
outputs, and shader/pipeline/device failures even if Bevy happened to write a
PNG. An ordinarily dead or wedged persistent host receives one automatic clean
restart and request retry. GPU out-of-memory, device loss, submission timeout,
or runaway host memory are terminal: retrying those makes the workstation less
healthy, so they trip the shared resource circuit breaker instead.

### 1.1 Usefulness today

The capture backend is already a strong visual-iteration instrument; its main
deficit is discoverability and composition at the CLI boundary.

| Workflow | State today | Important limit |
|----------|-------------|-----------------|
| One deterministic in-context still | Strong | First boot still pays world/terrain warm-up |
| Exact player→agent framing | Strong | A viewpoint restores camera, lens, body, scene, and time—not arbitrary transient craft state |
| Same framing at another time | Strong | Canonical seconds are currently the typed unit |
| Several scenes in one invocation | Strong | One shared override set; heterogeneous per-shot settings need a capture plan |
| A/B and N-way comparison | Strong | Axes are registered in Rust rather than discoverable/extensible from the CLI |
| Split, wipe, diff, and contact-sheet evidence | Strong | These are assembled offline from independent full-frame renders; there is deliberately no live split viewport |
| Debug/inspection views | Partial | SSAO, contact/cloud shadow, terrain lighting/normal, renderer, and cloud internals exist; depth, motion, LOD/tile IDs, material IDs, cascades, and atmosphere channels do not yet form one registry |
| Parallel-agent attribution | Strong | Persistent shots are serialized on one GPU host; source editing remains parallel and every result says which edits it contains |
| Human-in-the-loop handoff | Strong | F8/F9 handles viewpoints; transient gameplay moments still need a deterministic preset or an interactive screenshot |
| Self-discovering CLI | Partial | `list viewpoints` exposes text/JSON metadata and cached-capture state; `--gallery` builds a zero-GPU visual index. Scenes, axes, debug views, `describe`, and full subcommand help remain |
| Video / camera tracks | Planned | CAP-4 |

The correct description is therefore: **a capable renderer test harness with a
thin, under-designed command surface**. Agents can already run serious visual
diagnosis autonomously when they know the names; the next work makes that
knowledge discoverable rather than tribal.

### 1.2 Operator contract

An agent unfamiliar with the project must be able to:

1. discover scenes, saved viewpoints, comparison axes, debug channels, common
   options, boot cost, and output paths without reading Rust;
2. capture one shot, several compatible shots, or a heterogeneous typed plan;
3. hold one camera/time/configuration fixed while varying exactly one factor;
4. request an inspection channel directly, without learning its backing
   environment variable;
5. receive full-resolution originals plus offline split/wipe, contact-sheet,
   diff, metrics, manifest, logs, and provenance as applicable;
6. tell whether its own Rust, shader, configuration, texture, or terrain-package
   edits are present;
7. use the persistent lane for iteration and promote the final matched result to
   a cold acceptance run;
8. accept a viewpoint from a human through F8/F9, while keeping the rest of the
   loop agent-runnable.

The eventual command vocabulary is small and task-oriented:

```text
thalos_capture list viewpoints [--gallery] [--json] [--out DIR]
thalos_capture list scenes|axes|debug [--json]
thalos_capture describe <name> [--json]
thalos_capture shot <scene>... [--fidelity <tier>] [typed framing/output options]
thalos_capture run <capture-plan>
thalos_capture compare <scene> --axis <axis> [--cold]
thalos_capture inspect <scene> --view <debug-channel>
thalos_capture status [--json]
thalos_capture stop
thalos_capture reset
```

`--set KEY=VALUE` remains an expert escape hatch, never the primary documented
interface. Existing `just screenshot`, `just capture`, and `just compare`
recipes remain stable short aliases.

`list viewpoints --gallery` is the composition-discovery lane. It does not
acquire the capture lock or start the renderer: it downsamples matching
canonical PNGs already under `artifacts/visual/latest/` into
`artifacts/visual/catalog/viewpoints/`, writes one 320×180 thumbnail per catalog
entry plus `contact_sheet.png`, and records the source PNG/receipt and
current/stale/unattributed/missing state in `index.json`. Missing captures stay
visible as placeholders. These images answer “which camera is relevant?”;
they are never verification evidence, and an agent still reads the chosen
full-resolution PNG and receipt after capture.

Per-shot graphics preferences use the typed `--graphics` option rather than
mutating `user/settings.ron`:

```text
just screenshot spaceport-aerial --graphics clouds=off,grass=on
just capture spaceport-aerial dry-belt --graphics grass=off
```

The profile is a partial patch over deterministic capture defaults, so omitted
settings do not inherit either the player's preferences or the previous request
served by a persistent host. The current fields are `clouds` and `grass`;
additional graphics preferences extend the same typed protocol object.

One host means one world, one render device, and one active real camera.
Compatible requests may describe many camera poses, but the scheduler applies
and renders them **sequentially**. Extra camera entities rendered concurrently
would duplicate viewport-sized passes and GPU pressure without making the batch
finish safely; parallel agents queue their requests and share the resident host.

“Split view” means an **offline evidence composition** of independently rendered
matched frames. A live multi-camera split is not an alternative implementation:
it changes viewport-dependent LOD, SSAO, shadows, antialiasing, and temporal
history, invalidating the comparison
(ADR-20260721T192218Z-persistent-visual-iteration).

### 1.3 Workstation resource contract

Automation defaults are intentionally safer than the interactive renderer:

- viewpoint schema v2 stores typed lens/sensor state and no output pixels;
  implicit replay currently fits that sensor aspect inside 1920×1080 until
  named fidelity profiles land, while `--size` may choose another matching
  pixel extent but may not silently change the saved sensor aspect;
- headless capture defaults the machine-wide tile-mesh allowance to 2 GiB
  (`THALOS_TILE_BUDGET_MB` remains an expert override);
- the controller terminates a host above 8 GiB resident memory by default
  (`THALOS_CAPTURE_RSS_LIMIT_MB=0` disables only for deliberate diagnosis);
- OOM, device loss, GPU submission timeout, and RSS runaway never auto-retry.
  Resource pressure publishes a five-minute shared cooldown; device loss blocks
  capture for the remainder of the current OS boot and clears automatically
  after reboot. Already-waiting agents therefore cannot stampede the same
  failed device. WGPU may describe an OOM-induced teardown as `DeviceLost`; the
  controller classifies that by its causal `Out of memory` signature and keeps
  it on the bounded cooldown. Only an explicit lost-device diagnosis receives
  the reboot-long policy. `status` reports the active boundary.

The circuit breaker is a safety boundary, not an error-recovery claim. Windows
reporting “GPU is lost” requires a reboot; overriding the quarantine does not
repair the driver.

The quarantine record lives beside the machine-wide lock in the user temp
directory, not inside a worktree. Guard, expiry, record, and clear operations
therefore happen while the same machine mutex is owned; two checkouts cannot
independently clear or bypass one GPU's fault state.

Every lifecycle transition is also appended to
`artifacts/diagnostics/visual_capture_resource_events.jsonl`: fault recording,
blocked attempts, bounded expiry, reboot clearing, diagnostic override, and
successful clearing. Entries carry wall-clock and system-uptime timestamps,
fault age, policy, client PID/command, workspace, and the original triggering
log tail on the `recorded` event so the cooldown can be shortened later from
measured recovery evidence rather than guesswork. The active machine-wide
record retains the same log tail; `just capture-status` prints both paths.

### 1.4 Camera state is not capture fidelity

A viewpoint is a bookmark for **which rays the camera sees**, not how many
pixels are used to sample them. Its durable state belongs to the camera system:

- body-fixed pose and orientation;
- projection/lens state, including focal length or equivalent field of view;
- sensor gate/filmback dimensions and aspect;
- an optional sensor crop/window;
- later camera-owned focus, aperture, exposure, and related photographic state;
- scene/body and canonical time metadata needed to replay that camera.

The output pixel extent does **not** belong to the viewpoint. In physical-camera
terms, reducing the sensor gate while keeping pose and focal length fixed
introduces crop factor and narrows the field of view: that is a framing change
and therefore camera state. Increasing or reducing the pixel array over the
same gate changes sampling fidelity without changing the shot. A crop/window
must be applied before rendering—not cut out of a wider PNG afterwards—so
terrain LOD, shadows, screen-space effects, and temporal history see the camera
that actually made the image.

The output grid normally inherits the sensor window's aspect. A caller asking
for a different output aspect must also choose an explicit sensor crop/fit
policy; capture must not silently alter the saved camera to fill the pixels.

Capture chooses a named fidelity profile independently:

| Tier | Intended use |
|------|--------------|
| `draft` | Fast composition checks and binary debug channels where fine image detail is irrelevant |
| `standard` | Default autonomous visual iteration; deliberately sharp enough for ordinary full-frame diagnosis |
| `high` | Texture, foliage, aliasing, distant-detail, or other judgments for which `standard` is inconclusive |
| `reference` | Cold, highest-stable-quality acceptance evidence retained for final before/after proof |

A tier is a typed bundle, not an alias for one width: it selects a pixel budget
derived from the camera sensor aspect plus renderer-internal resolution scales,
AA/reconstruction quality, shadow/cloud quality, and convergence policy.
Explicit pixel extent remains an expert/output override. Every receipt and
comparison manifest records both the requested tier and its effective settings.

Agents begin at `standard`, use `draft` only when the question cannot depend on
fine detail, promote to `high` whenever the result looks soft or the diagnosis
is resolution-limited, and promote the matched final result to `reference`.
Changing tier must leave normalized sensor rays—and therefore composition—
unchanged. The previous blanket 1080p replay was too blurry to serve as this
default; it remains only as the short-lived safety shim above.

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

The current `crates/runtime/game/src` moves first into a library and is organized by
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
  camera            preset, saved physical-camera state, or camera track
  fidelity          draft, standard, high, or reference
  output            still, frame sequence, or video; optional pixel override
  timeline          simulation time, frame rate, duration, preroll
  warmup            scene and temporal convergence policy
  render_overrides  typed SSAO/cloud/terrain/ocean/debug variants
  graphics          typed partial graphics preferences (clouds, grass, ...)
  diagnostics       requested logs, timings and debug channels
```

Environment variables remain only as a temporary compatibility adapter. New
capture features extend `CaptureSpec`; they do not add another environment-key
protocol.

`CaptureRequest::graphics` is the first concrete slice of that typed settings
surface. The runtime resolves it from `GraphicsSettings::default()` for every
request and the receipt records the effective values. Direct cold launches may
temporarily use `THALOS_SCREENSHOT_GRAPHICS=clouds=off,grass=on`; the persistent
controller translates `--graphics` directly into the typed request.

Protocol v3 also publishes the boot-compatible preset set and exact source
provenance. A boot context is the tuple `(target body, spawn scenario, hub mode,
effective render extent, startup overrides)`. The extent is chosen by capture
fidelity and may size persistent GPU resources; it is not viewpoint identity.
Camera framing and live diagnostics may change
in-process, while a different tuple triggers a managed restart. This lets
runway/cloud/massif views or
orbit/ocean/cloud views amortize one expensive world boot without pretending
incompatible scenes share state.
See ADR-20260724T162943Z-capture-reuse-by-boot-context.

## 4. Capture runtime

`thalos_capture_runtime` owns one explicit state machine:

```text
Idle → LoadingScene → Warming → Capturing → DrainingReadbacks → Complete → Idle
```

It owns the off-screen target, real-camera retargeting, deterministic site and
camera-state application, output resolution, overlay policy, temporal
cut/reset epoch, diagnostic
overrides, readback, and result validation. A request is not successful merely
because a PNG exists: shader compilation, pipeline validation, missing render
layers, write failures, and incomplete frame sequences fail the result.

The persistent host is resident, not continuously rendering. With no active
request it parks the real capture camera (so the render world extracts no view)
and polls the control plane at 10 Hz. A request reactivates the camera before
that frame's extraction, preserving sub-100-ms wake latency and the already
booted ECS/GPU state without paying an idle 60-Hz render loop.

The controller serializes every shot, batch, comparison, and reset with one
machine-wide client lock. On Windows the authority is a kernel named mutex,
which is shared by every worktree and released automatically when its owner
dies; the JSON file under the machine temp directory is diagnostics only. A
concurrent client waits and reports the owning PID/command; it never starts a
second renderer or overwrites the shared request/response files.
`THALOS_CAPTURE_CLIENT_WAIT_SECS` bounds that wait (default 1800 s).

Serialization does not freeze the checkout: agents may keep editing while a
different capture is building or rendering. The controller fingerprints the
actual Rust/Cargo build inputs by content (not mtime), plus every runtime asset
(shaders, authored configuration, textures, terrain packages, and catalogs). A
resident host is reused only when its published build fingerprint matches the
request. If the tree changes while a host is being prepared, the controller
does not chase the moving aggregate fingerprint: the invocation snapshot is a
causal source floor, and a successful build may contain that state or edits
made later while it was in flight. Exact workspace equality is provenance
metadata, not a retry condition.

Each PNG is paired with `<image>.capture.json`
(`thalos.capture-receipt.v1`). The receipt records the request/preset, renderer
PID, Git revision and dirty state, full capture-input fingerprint, the host's
launch/build fingerprint, the post-readback workspace fingerprint, and
`source_floor_guaranteed`, `workspace_relation`, and the compatibility boolean
`workspace_matches`. Console success prints the short floor fingerprint as an
immediate inclusion check. `workspace_matches: false` means the checkout
advanced; it does not invalidate the capture or prove whether a particular
later edit was consumed. Comparison manifests reference every variant receipt
and carry a comparison-wide source-before/source-after match.

The interactive launcher also installs F2 plus the F8 egui viewpoint manager.
The portable capture protocol owns the versioned `assets/viewpoints.json` types,
so the manager, agent edits, CLI, and headless runtime share one schema. Exact
saved poses and agent-scripted views are both catalog entries. Procedural
drivers remain runtime behavior for searches and diagnostic setup, but they are
capabilities selected by catalog data rather than a second public view list.
An exact saved pose also carries its canonical `sim_time_s`; headless replay
applies that epoch after scene construction and before simulation/sync. A
per-request caller time supersedes the metadata without mutating the catalog,
so one persistent world can render the same camera at several epochs.

## 5. Still, comparison, and video

The Rust `thalos_capture` CLI is the single public orchestrator:

```text
thalos capture shot spaceport-aerial
thalos capture shot spaceport-aerial --graphics clouds=off,grass=on
thalos capture shot spaceport-aerial runway-atmosphere cloud-runway
thalos capture compare spaceport-aerial ssao
thalos capture record cloud-sunset --seconds 8 --fps 60
thalos capture verify
thalos capture status
thalos capture stop
```

`just screenshot`, `just capture`, `just compare`, and their cold variants
remain stable aliases. `shot` accepts multiple scene names; compatible scenes
reuse the current host and incompatible scenes restart automatically. The CLI
uses the host's advertised compatibility set to pull same-context scenes
together, minimizing boots even if the caller supplied an interleaved list. It
owns host lifecycle, typed matrices, provenance, artifact assembly, reload
readiness, output decoding, fatal-render-error promotion, and one-shot host
recovery. Comparison assembly is a module of this binary, not a second
controller executable.

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

The Python controller and separate comparison executable are gone. Retain
`just` aliases, finish typed request/result manifests, and complete immutable
per-run bundles. Multi-scene batching, boot-context reuse, generated-evidence
migration, output validation, automatic host recovery, and fatal
shader/pipeline/device-error promotion have landed compile-clean.

### CAP-4 — Deterministic frame sequences and video

Add fixed capture time, camera tracks, readback pipelining, background frame
writes, resumable/validated sequences, and external FFmpeg muxing. Verify a
short motion/temporal probe in both persistent and cold lanes.

## 8. Exit criteria

- Interactive and capture apps demonstrably use the same runtime/plugin graph.
- Every current preset and typed comparison is available through the Rust CLI.
- Persistent capture reloads WGSL in-process and automatically rebuilds/restarts
  after Rust or manifest edits.
- Cold results match the prior canonical framing before old entry points retire.
- Invalid render pipelines fail requests.
- A deterministic video rerun produces the same frame count, timestamps,
  framing, and per-frame hashes before lossy muxing.
