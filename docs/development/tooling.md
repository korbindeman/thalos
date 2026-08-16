# Tooling

Thalos keeps committed Cargo and Rust toolchain configuration deliberately
plain. `rust-toolchain.toml` pins Rust 1.97.0 (the current stable release) and
the `clippy`/`rustfmt` components; `Cargo.toml` sets only shared profile choices
that are expected to be sane across Windows, macOS, and Linux.

## Local compiler tuning

Platform-specific compiler and linker speedups belong in local Cargo config,
not in committed project config. Use either a personal Cargo config under your
home directory or the workspace-local `.cargo/config.toml`; the workspace file
is ignored by Git for this purpose. The `just game` command can also be
customized locally with `.env.just`, which is ignored by Git.

Portable incremental and debug-info policy lives in the root `Cargo.toml`.
Local config contains only:

- the supported platform linker;
- a per-Cargo-process job budget sized for expected concurrent agents.

There is no compiler cache: sccache was removed 2026-07-23
(ADR-20260723T222214Z-abandon-sccache). See build_speed.md §5.

Do not add a compiler-backend override. The pinned stable toolchain uses LLVM;
any future deviation is a cross-platform architecture decision, not local tuning.

### Renderer iteration: one stable dynamic-link lane

`just game` and the explicit cold screenshot lane use the `justfile`'s shared
`game_command`, which defaults to
`cargo run -p thalos_game --features dev-renderer`. `just preview` and
`just ui-preview` enable the same Bevy feature for their own packages; the UI
preview also requests the game's `wayland`/`jpeg` Bevy feature set so Cargo can
reuse the same `bevy_dylib` artifact instead of compiling a second variant.
Dynamic linking is a cross-platform interactive/cold-iteration speedup — Bevy links once into
a shared `bevy_dylib` and subsequent rebuilds relink only our crates — and it is
not platform-specific, so it belongs in committed config rather than each
developer's `.env.just`.

Cargo also supplies a dynamic-library search path when it runs a built renderer.
Any cold-path tool that launches a dynamic dev renderer executable directly must reproduce that
contract by prepending the Cargo profile directory, its `deps` directory, and
`rustc --print target-libdir` to `PATH` (Windows),
`DYLD_FALLBACK_LIBRARY_PATH` (macOS), or `LD_LIBRARY_PATH` (other Unix).
The integrated `thalos_capture compare` path owns this for `just compare-cold`;
omitting the rustc directory can
still miss dynamically linked `std` even after `bevy_dylib` is found. See
INC-0008.

It is scoped to dev renderer commands only. `just build`, `just trace`
(release), tests, and bakes do not enable it, so it never reaches a shipped/release build
(where a missing `bevy_dylib` would crash the binary). To opt out locally —
e.g. on a platform where the feature misbehaves — override the whole command in
`.env.just`:

```dotenv
THALOS_GAME_COMMAND="cargo run -p thalos_game"
```

The high-frequency graphics loop uses the same fingerprint: `just screenshot`
starts (or reuses) one persistent `thalos_capture_host` renderer, spawned by
the capture client as a detached `cargo run … --features dev-renderer`
(ADR-20260724T153619Z-retire-hotpatch-single-stable-capture-lane; no external
tooling required). Bevy watches normal asset shaders and WGSL registered via
`embedded_asset!`, so both shader forms reload in the same process (~3 s to a
fresh PNG). `just compare` sends its variants to that exact process.

HUD captures use `THALOS_SCREENSHOT_HUD=1`. To verify the floating widget
workspace deterministically, `THALOS_SCREENSHOT_WIDGETS` accepts a
comma-separated catalogue selection (`traj,nd,dock,xfer`); for example,
`THALOS_SCREENSHOT_HUD=1 THALOS_SCREENSHOT_WIDGETS=traj,nd just screenshot
interstage`. The headless layout is fixed and never reads or rewrites the
player's saved workspace.

Useful lifecycle commands:

```text
just capture-status
just capture-stop
just screenshot-cold spaceport-aerial
just compare-cold spaceport-aerial ssao
```

The server reuses presets with the same body, spawn scenario, hub mode,
viewport, and startup-only override fingerprint. It restarts automatically when
that boot context changes **and**
whenever any workspace `.rs`/`.toml` **that can link into the host** differs by
content from the running host's build fingerprint — the next capture stops it,
rebuilds (dynamic relink), and boots it again (~1.5–2.5 min warm).
`examples/`, `tests/`, and `benches/` trees are excluded from that fingerprint
(2026-07-29): they never link into the host binary, and they were a steady
source of spurious ~2 min restarts while agents iterated on exporters and
benches (`restart_stale_source` dominated a 73 % boot rate). A stop that the
host does not honour within 5 s now escalates to a confirmed forced kill;
a stop that cannot be confirmed fails the shot instead of booting a second
renderer beside a live one (INC-20260729T081809Z). There is no in-process Rust
reload: dx/subsecond hot-patching was retired after an applied patch
reproducibly crashed the app (INC-20260724T044418Z). `just capture-stop`
remains as manual hygiene.

`just capture <preset>...` batches several scenes through one controller
invocation, amortizing the world/GPU boot wherever the boot context matches.
`just compare` is a subcommand of that same controller, not a separate
executable. The controller validates scene names and decoded outputs, rejects
fatal shader/pipeline/device logs, and retries one dead or wedged host once.

### Game window / renderer launch toggles

`just game` normally starts borderless fullscreen. The renderer backend
defaults to **Vulkan on Windows** (set in `wgpu_settings_from_env`,
`crates/runtime/game/src/lib.rs`) — wgpu's own default prefers DX12 there, and DX12
is this dev machine's documented unstable path (swapchain-acquire panics,
silent device death, and a 2026-07-19 full DeviceLost wedge requiring a
reboot). Other platforms keep the wgpu default (Metal on macOS). If a
swapchain/device-loss style panic still appears, take the display mode and
backend choice out of the equation without changing source:

```dotenv
THALOS_WINDOW_MODE=windowed      # windowed | borderless | fullscreen
THALOS_WINDOW_SIZE=1600x900      # optional, used for windowed mode
THALOS_WGPU_BACKEND=vulkan       # auto | dx12 | vulkan | metal | gl
THALOS_SCALE=2                   # optional, pin the UI scale factor
THALOS_QUALITY=laptop            # optional, pin Showcase or Laptop for one session
```

### Developer quality profile (`THALOS_QUALITY`)

`THALOS_QUALITY=laptop|showcase` stamps the named bundle for one process and
does not write `preferences.ron` / `settings.ron`. On macOS, `just game` and
`just korsou` default that pin to laptop (graphics knobs only; the window
mode is unchanged) so an existing checkout does not stay on Showcase
layers.
`quality=showcase` forces the canonical look. Capture ignores the pin. Full
contract: `docs/development/quality_profiles.md`.

```bash
just game orbit
just game orbit quality=showcase
```

### HiDPI UI scale factor (`THALOS_SCALE`)

The UI rasterises at `window scale × UiScale`: the OS HiDPI scale carries the
display's sizing and `WindowSettings::ui_scale` (Settings → Window → UI scale)
layers the user's preference on top. `THALOS_SCALE=<float>` pins the window
scale factor instead, for isolating scale-dependent rendering (e.g.
`THALOS_SCALE=1` for native-pixel UI on a HiDPI laptop, smaller but sharp).

**Historic snap, now removed.** Through Bevy 0.18 the game snapped the
*effective* scale to the nearest integer, because cosmic-text rasterised glyphs
at inconsistent sizes on fractional scale factors (text looked non-uniform,
"not monospace"). That cost real estate — a 150 % display got 2.0, inflating the
whole UI by a third — which is why the HUD used to swallow a 4K screen. Bevy
0.19 replaced cosmic-text with parley and the bug is gone, verified by rendering
the kitchen sink at a fractional scale:

```bash
THALOS_UI_SCALE=1.5 cargo run -p thalos_ui --features bevy/dynamic_linking,bevy/jpeg --example kitchen_sink
```

`THALOS_UI_SCALE` is the kitchen sink's own knob (the game reads the settings
file); use it to re-test glyph quality at fractional scales before reaching for
any compensation scheme. If one is ever needed again, snap the **UI** scale, not
the window scale-factor override — `bevy_winit` treats a scale-factor change as
logical-size-preserving and physically resizes the window.

`THALOS_WGPU_BACKEND` is a Thalos-facing alias for the same class of wgpu
backend selection that `WGPU_BACKEND` provides, but it is scoped to our game
startup helper in `crates/runtime/game/src/lib.rs` and is easy to keep in `.env.just`.
It overrides the Vulkan-on-Windows default above: `auto` restores wgpu's own
selection (DX12 on Windows), `dx12` forces DX12 for A/B comparison.

### Uncapping the framerate for profiling (`THALOS_VSYNC`)

Frame time is the only meaningful performance signal, and vsync floors it at
the monitor's refresh budget — so a change that shaves real GPU/CPU time below
that floor shows no movement. Set `THALOS_VSYNC=off` (also accepts
`0`/`false`/`no`) to launch with `PresentMode::AutoNoVsync` and read the true,
uncapped frame time while still allowing wgpu to fall back to a supported
non-vsync present mode; anything else keeps the vsync default. Read by
`thalos_preferences::overrides_from_env` as a session override: it wins over
the persisted `user/preferences.ron` vsync preference and
greys out the VSync control in the settings menu, without being written into
the file. (Vsync can also be toggled live from the settings menu's Window
tab, which *does* persist.)

### Render-cost attribution matrix

`just perf-bisect [preset]` is the agent-runnable render-cost discriminator. It
boots the real screenshot world and offscreen render graph once, removes the
capture host's 60 Hz pacing and PNG readback, waits for terrain coverage plus
stable mesh/tile counts, then measures the four cells of foliage × custom
shadows in that same warmed process. Each cell gets a frame-history flush and
240-frame measurement window. Results and the computed main/interaction
effects go to `artifacts/diagnostics/reports/headless-matrix.json`.
The recipe explicitly enables the diagnostic target even when the calling
shell has a restrictive `RUST_LOG`; the report command fails if any cell is
missing rather than publishing a partial matrix.

`just perf-shadow-bisect [preset]` uses the same warmed scene and controls but
holds foliage resident while stepping the live shadow-camera budget 4→0. Its
`headless-shadow-cascades.json` report gives the marginal frame cost of every
cascade. Use that ladder before changing cascade coverage or resolution: it
separates a costly individual view from fixed rig overhead, and it fails closed
when any rung is missing.

The default preset is `forest-stand`; output is fixed at 1600×900 with clouds,
grass, and MSAA off so the matrix changes only woody vegetation and the custom
shadow rig. `THALOS_PERF_FOLIAGE` remains a session-only control that never
rewrites the persisted foliage preference. `THALOS_SHADOW_CASCADES=0..4`
remains a cold-run control for ordinary game/capture sessions; the benchmark
alone changes its live ceiling between warmed cells. The matrix excludes
swapchain presentation, input, and the interactive window, so use it to rank
render subsystems. Use the matching interactive scene only for the final
player-visible validation if the offscreen result identifies a lever.

Every `frame_gauge` records the effective foliage/cloud/grass/MSAA state,
shadow-cascade budget, VSync state, and physical window size. `just perf-report`
copies that identity into `summary.json`; a comparison is invalid when those
fields differ outside the intended axis. Each offscreen cell additionally emits
`headless_benchmark_{config,ready,start,end}` records with frame percentiles,
wall duration, scene counts, stage samples, resolution, and an explicit GPU
timing-availability bit.

For A/B attribution, have the user change one variable at a time and report
the frame time from the F3 debug view or capture a chrome trace
(`--features profile-chrome`, see below). Useful no-rebuild toggles: sim pause
(`Escape`) subtracts simulation cost; the settings menu's graphics toggles
(e.g. volumetric clouds) subtract a renderer subsystem; map view / freecam
change what the scene draws. Frame time unchanged with the heavy 3D path off ⇒
CPU-bound. This is how the surface frame cost was traced to the Avian terrain
collider (see `docs/simulation/surface.md`). The game has no remote-inspection channel —
you analyze the artifacts (trace JSON, slow-frame JSONL, console logs), the
user runs the game.

### Runtime logging contract

The terminal is the human/operator log. An `info!` line should describe a
concise lifecycle or user-visible state change; `warn!` and `error!` should say
what went wrong and, when possible, what to do. Do not print periodic gauges,
large debug values, calibration samples, per-tile counters, or probe output
there.

Machine-oriented runtime events use tracing's structured fields and a target
under `thalos::diagnostic::*`:

```rust
info!(
    target: "thalos::diagnostic::tile_terrain",
    event = "residency_gauge",
    resident = resident_count,
    resident_mib,
    "tile residency gauge"
);
```

`thalos_runtime` installs one JSONL tracing layer for the whole process, so this
also works in renderer crates. It writes
`artifacts/diagnostics/runtime.jsonl`; every line carries the schema, process
session, wall-clock timestamp, level, target, and the event's typed fields. The
normal console formatter deliberately omits informational events on this target;
warnings and errors remain visible. Set
`THALOS_RUNTIME_DIAGNOSTICS` to a bare filename (resolved under
`artifacts/diagnostics/`) or an explicit path to override the sink. `RUST_LOG`
still controls which events exist.

### Performance telemetry (perf lane)

`thalos::diagnostic::perf` is the always-on performance lane inside the shared
runtime stream. `thalos_diagnostics_ui::FrameSamples` is the one wall-clock
CPU/GPU ring consumed by the common F3 graph and the game recorder;
`crates/runtime/game/src/perf/::PerfSamples` adds only game stage and
memory/count gauges.

- **F3 debug view** — `thalos_diagnostics_ui` owns the single F3 reader,
  requested-open state, common frame/device/process/scene sections, and frame
  graph in every interactive application. Typed application extensions add
  their own facts. Thalos observes the shared state to toggle physics hitboxes
  and aero gizmos together; Kòrsou adds planar streaming and UTM/AGL position.
  Graphs render as `UiMaterial` quads (`assets/shaders/perf_graph.wgsl`, so
  they hot-reload). To screenshot it headlessly, `THALOS_DEBUG_VIEW=1` starts
  it visible and `THALOS_SCREENSHOT_HUD=1` is also required — the default
  capture path enters photo mode, which hides the view:

  ```bash
  THALOS_DEBUG_VIEW=1 THALOS_SCREENSHOT_HUD=1 just screenshot forest-stand
  ```

  `THALOS_DEBUG_VIEW` is read at host boot **and** per capture request
  (`perf::overlay::apply_debug_view_override`). The per-request path is the
  load-bearing one: the host is machine-wide and shared, so a boot-time-only
  flag silently returned a normal-looking PNG with the view off whenever
  another agent had already started the host (BL-20260730T184038Z). Prefer
  `just screenshot` over a cold `cargo run -p thalos_capture_host` — the client
  queues behind a peer's shot, while a cold host fails outright on the lease.

  Beyond the perf block it answers the *machine* and *place* questions a
  Minecraft-style F3 is for, so a screenshot of it is a self-contained bug
  report. It is laid out as captioned sections with one fps headline, not as a
  flat list — an earlier revision put the fps you watch constantly and the
  session id you read once a month in the same size and colour, which is
  unreadable at a glance:

  - **headline** — fps, with shared frame/GPU distribution and the top GPU
    passes beneath it in small dim type; Thalos adds stage/warp timing under
    **SIMULATION**;
  - **DEVICE** — the adapter actually in use (name, backend, driver — not
    always the one the machine advertises) and window size;
  - **MEMORY** — whole-card VRAM plus process RSS in the core; Thalos extends
    it with the attributed VRAM bar, terrain residency budget, CPU asset bytes,
    and the two-minute terrain/slab graph;
  - **SCENE** — common object counts; Thalos adds renderer instances, resident
    tiles, and the tile driver's `split_scale` (< 1 = the residency budget is
    actively coarsening the ground);
  - **POSITION** — body, latitude/longitude, altitude/AGL/ground, view speed,
    and the landcover moisture + canopy the generator reports at that exact
    direction — the same fields the ground shader and scatter placer read, so a
    disagreement between the image and this readout is itself the bug.

  Each source degrades to a stated gap, never a zero.
- **`frame_gauge`** every ~2 s: fps, cpu ms mean/p50/p95/max, GPU ms plus an
  explicit availability bit, SimStage
  wall times, entities, mesh/image counts, tile-resident MiB, mesh-slab MiB,
  effective render configuration, plus the CPU side: `rss_mib` (whole-process working set) with
  `mesh_cpu_mib` / `image_cpu_mib` to attribute it. Added 2026-07-29 because a
  capture host was killed at 8.1 GiB RSS while every GPU-side gauge summed to
  ~2 GiB (INC-20260729T081809Z) — `just diag`'s `memory_growth` prefers
  `rss_mib` when a session carries it. Under 1 MB per played hour.
- **`spike`** dumps: the collector keeps ~8.5 s of per-frame samples; a frame
  over max(3×median, 50 ms) dumps a 3 s window (2 s pre + 1 s post) of exact
  per-frame CPU/GPU values as comma-joined string fields. ≥5 s cooldown.
- **`frame_block`** (opt-in, `THALOS_PERF_RECORD=1`): every frame recorded,
  emitted in 1 s blocks (~15–30 MB/hour) — for runs where the offline report
  should carry the complete timeline.

The shadow rig emits `thalos::diagnostic::shadow` / `stability_gauge` once per
second. `origin_frame_error_m` compares the cascade rig's render origin with the
current camera cell origin; `footprint_scale`, `cascade0_texel_m`, and
`active_cascades` provide the scale denominator. Any origin error above 1 cm is
a frame-coherence failure, reported by `just diag` as `shadow_frame_desync`.

The ground-contact pair on `thalos::diagnostic::local_physics` (both 1 Hz
sim-time throttled, added with INC-20260729T073116Z): **`gear_contact`**
(`wheels`, `wheels_loaded`, `ray_misses`, `max_compression_frac`,
`normal_sum_n`) records how hard the wheels are actually carrying while any
wheel bears load — a loaded wheel with `ray_misses > 0` was rescued by the
analytic fallback; silent during airborne gear-down flight. Its counterpart
**`backstop_intervention`** (`penetration_m`, `excess_m`, `gear_down`,
`weight_on_wheels`, `destroyed`) records every frame-window where the terrain
floor backstop carried the hull. The load-bearing combination —
`gear_down = 1, weight_on_wheels = 0`, not destroyed, sustained ≥ 2 events — is
the buried-suspension-ray signature (belly slide, no brakes), reported by
`just diag` as `gear_carried_by_backstop`.

The runway-destination autopilot emits lifecycle records on
`thalos::diagnostic::approach_ap`: `land_engaged`, `land_phase`,
`land_disengaged`, and `land_completed`, plus the 1 Hz `appr_frame` gauge
(`dtg_m`, lateral/vertical errors, actual/target speed, selected throttle,
steering and braking, weight-on-wheels, confirmed-touchdown contact dwell, and
post-touchdown airborne dwell). A bounce-triggered `land_go_around` also records
the airborne dwell that crossed the recovery gate. `just diag` treats an autonomous
disengagement as `land_autonomous_disengage` and any completion above
0.75 m/s as `land_completed_while_rolling`; deliberate `pilot_override`
disengagements stay quiet.

The target-orbit autopilot emits `orbit_plan_result`,
`orbit_autoflight_transition`, one-second `orbit_autoflight_guidance` samples,
and terminal `orbit_autoflight_complete` / `orbit_autoflight_abort` records on
`thalos::diagnostic::orbit_autoflight`. Guidance samples carry phase, altitude,
predicted apoapsis, dynamic pressure, selected throttle, and TWR. `just diag`
repeats the executor's achieved-element completion tolerances as
`orbit_false_completion`, and reports three or more same-session guidance
samples above 42 kPa as `orbit_sustained_max_q_overshoot`.

On Windows, `THALOS_GPU_HEALTH=1` makes `thalos_diagnostics` load NVIDIA's NVML
dynamically and emit `thalos::diagnostic::gpu_health` / `sample` once per
second. This is an investigation mode, not an ordinary-play default. It reports
the whole card rather than one process: used/total VRAM, temperature, draw/limit
power, GPU/memory utilization, graphics clock, performance state, and throttle
reasons. Systems without NVML emit one `availability=false` record and carry on.
The first failed memory query emits `sample_error` and stops the sampler so a
lost adapter has one precise timestamp without hammering the failed driver.
This lane exists to separate per-process allocation growth from whole-card
pressure, thermal/power instability, and a driver/PCIe disappearance
(INC-20260729T092010Z). `just diag` treats NVML thermal throttle bits `0x20`
(software) and `0x40` (hardware protection) as their own signal. NVIDIA defines
software thermal slowdown as either the GPU or memory exceeding its maximum
operating temperature, which still detects a hidden memory hotspot when the
card exposes only its core-temperature sensor.

Independently of that gate, `thalos_diagnostics::gpu_memory()` answers
**whole-card VRAM used/total** for any in-process reader (the F3 debug view and
the loading screen both show it). It shares the same NVML handle, but its own
poller samples only memory, twice a second, on a background thread — so the
caller never blocks and never queries the driver itself, and the reading is
available without turning the investigation lane on. `None` means no reading:
non-Windows, no NVIDIA driver, before the first sample, or a card that stopped
answering. Readers must show that as a stated gap, never as a zero.

Whole-card is the point. Thalos is routinely run two instances at a time, and
per-process accounting cannot see the peer that is eating the headroom — the
mechanism behind INC-20260725T012104Z. Both readouts therefore print the live
instance count (`tiles::vram_share::live_instances`) beside the capacity, since
it is also the divisor of `tiles::residency_budget_bytes`.

**`just perf-report [session|latest|--list]`** (tools/perfreport) renders one
session of that lane to `artifacts/diagnostics/reports/<session>/report.html`
(self-contained charts: frame timeline with spike markers, tile-vs-slab memory
curves, counts, stage breakdown, per-spike sparklines) plus a `summary.json`
for agents. Note the honesty split in `summary.json`: `worst_window_*` fields
are window-level aggregates; exact frame percentiles (`full_rate.*`) appear
only when a full-rate recording exists. Missing GPU timing is `null` with
`gpu_timing_available: false`, never a misleading zero. Deep profiling remains Tracy /
`profile-chrome` — the perf lane tells you when to reach for them.

**Storage**: `runtime_diagnostics::jsonl_layer` calls
`artifact_paths::rotate_and_prune_diagnostics()` at every process start — any
top-level diagnostics JSONL over 64 MB rotates to `<stem>.rot<unix_ms>.jsonl`,
then rotated files are pruned oldest-first until the directory is back under
its 256 MB budget. Active sinks are never deleted; `perf-report` reads rotated
`runtime.rot*.jsonl` siblings transparently.

Keep a specialized JSONL recorder when it has a real independent schema or
lifecycle—capture reports, stage-separation audits, grass churn, GPU-memory
sampling, shadow traces, and shipyard selection traces are examples. Reuse
`artifact_paths` so bare names stay under the diagnostics tree. A new
diagnostic should default to the shared runtime stream; it should not add a new
file merely to avoid choosing an event name.

### Event shape

An event is data an agent greps and a report tool groups by, so its shape is
part of the contract:

- `event = "<snake_case_noun>"` is the stable key. Once written, treat renaming
  it as a breaking change to every reader.
- Fields are flat scalars with the unit in the name (`_ms`, `_mib`, `_m`,
  `_hz`, `_frac`, `_count`). No pre-formatted strings a reader has to re-parse;
  the trailing message string is a human label, never the payload.
- Emit the denominator with the numerator — `resident_mib` beside `budget_mib`,
  a count beside the cap that would bite, a rate beside its window. A number
  with no denominator can be consistent with every hypothesis at once, which is
  how the tile-residency OOM stayed undiagnosable (INC-20260725T012104Z).
- Anything per-process is joined by the line's own `pid`/`session`. Concurrent
  game instances are normal here; aggregate per session, never across the file.
- Always-on gauges are periodic (≥ 1 s) and allocation-free on the hot path.
  Full-rate or per-item traces are `THALOS_*` opt-in and must not need a
  recompile.

### Reading the lane: `just diag`

Writing events is half a diagnostics system; the other half is a reader that
says what deserves attention. `just diag [hours]` (`tools/diag`) loads the
window — runtime and tool lanes, rotated files included, foreign schemas
skipped — and reports **only what crossed a threshold**:

```bash
just diag           # last 24 h
just diag 168       # the week, for judging a trend
just diag 24 --json # machine-readable, for a routine or another tool
```

A healthy window prints its header and `nothing crossed a threshold`. An empty
window says *nothing ran* rather than implying health — the distinction matters,
because "no records" is equally consistent with "quiet day" and "the lane
stopped writing".

Findings carry a stable `id`, a headline with its **denominator**, at most four
evidence lines, and where to look next. Current checks: `error_events`,
`warn_events`, `capture_failures`, `capture_retries`, `capture_boot_rate`,
`capture_latency`, `capture_lock_contention`, `frame_spikes`, `slow_frames`,
`memory_growth`, `tile_budget_brake`, `shadow_frame_desync`, `silent_sessions`, `lane_noise`,
`land_autonomous_disengage`, `land_completed_while_rolling`,
`orbit_false_completion`, `orbit_sustained_max_q_overshoot`,
`gpu_adapter_lost`, `gpu_memory_pressure`, `gpu_thermal_pressure`,
`gpu_thermal_throttle`,
`empty_window`.

Thresholds live in `tools/diag/src/finding.rs`, one named constant each with the
reason it sits where it does. Two properties are tested and must stay true:
**a healthy window produces no findings** (otherwise real findings drown), and
**every check fires on the defect it exists for** (otherwise it is decoration
that passes forever).

**The daily pass** is the `diag-triage` skill (`.claude/skills/diag-triage/`):
run the reader, interpret each finding against known mechanisms, dedupe against
`docs/backlog.md` and `docs/incidents/`, and end each one in a row, an incident,
or an explicit *no action, because …*. Its second job is signal-to-noise
maintenance — see below.

**Keeping the lane worth reading.** The lane is shared, so its volume is a
shared cost, and `lane_noise` flags any single event that dominates it. The
policy, in order of preference: sample it, demote it to a `THALOS_*` opt-in
mode, or delete it. Investigation traces are the usual offender — they are
instruments for one question and rarely get retired when it is answered. The
vegetation drive trace was exactly this (60 % of every record in the lane, read
by nothing) and is now behind `THALOS_VEG_DIAG=1`. The reciprocal rule applies
to the reader: a check that fires daily and is dismissed daily is a wrong
threshold or a dead check.

**Adding a diagnostic** therefore means answering three questions: what does it
let a reader conclude, what value makes it actionable, and which check reads it?
If nothing reads it, add the check or don't add the event.

### Developer-tool instrumentation

The headless capture lane is the throughput floor for agent work, so its
latency and reliability are measured, not recalled. Every developer-tool run
(`just screenshot` / `capture` / `compare` / `preview` / `map` / `bake` /
`texgen`) should leave a machine-readable record of what it did, how long each
phase took, and how it ended: outcome, per-phase timings (lock wait, host reuse
vs. restart, rebuild and its reason, settle, render, encode), source
fingerprint, retries, and the failure reason on the error path. An exit code
plus stderr prose does not qualify — Bevy can log a fatal pipeline validation
error and still exit 0 (BL-20), which is why silent-success detection belongs in
recorded invariants rather than in the reader's eyes.

The contract, sink, session id, path resolution, and rotation live in
**`thalos_diagnostics`** (`crates/foundation/diagnostics`, no Bevy), so a tool
records on the same terms as the game. A tool calls
`thalos_diagnostics::install_tool_lane()` once at startup and wraps each unit of
work in a [`ToolRun`]:

```rust
let mut run = ToolRun::start("capture", format!("shot {preset}"));
run.field("host_action", "restart_stale_source");
run.phase("host_start", started.elapsed());   // repeats accumulate
run.count("retry");
run.ok();                                     // or .fail(reason)
```

`ToolRun` emits `tool_run_start` when the work begins — the line that survives a
hard kill — and `tool_run` when it ends, flattening phases and counters into
`phase_<name>_ms` / `<name>_count` beside `outcome`, `total_ms`, and any typed
fields. Tool records go to **`artifacts/diagnostics/tools.jsonl`**
(`THALOS_TOOL_DIAGNOSTICS` overrides), separate from `runtime.jsonl` because the
lifecycle is per-invocation and the question is different: *is the capture lane
fast and stable this week?* should not mean paging past a session of frame
gauges.

**The capture client is instrumented.** One record per shot — including each
scene of a `just capture` batch and each variant of a `just compare` matrix,
since both route through the same `capture()`:

| Field | Meaning |
|---|---|
| `host_action` | `reuse` · `start` · `restart_stale_source` · `restart_incompatible_scene` · `restart_startup_override` — why this shot did or did not pay for a boot. The same classifier drives the branch, so the record cannot drift from what happened. |
| `phase_source_snapshot_ms` | content-hashing the source + asset trees (several times per shot; accumulated) |
| `phase_host_start_ms` | stop → `cargo run` rebuild → host ready, on success *and* failure |
| `phase_shader_reload_ms` | waiting for a WGSL edit to land in the resident host |
| `phase_render_ms` | request written → response observed |
| `phase_validate_ms` | output decode, render-log scan, workspace re-snapshot, receipt |
| `host_start_count`, `retry_count`, `rebuild_recovery_count` | boots, unhealthy-host retries, stale-artifact rebuild recoveries |
| `outcome`, `error` | `ok` · `error` (first line of the failure) · `abandoned` (process died mid-run) |
| `launcher_exit_kind` | present when the host died before it was ready: `renderer busy` · `workspace build failure` · `capture host panic` · `capture host aborted` · `toolchain corruption` · `silent exit` · `unclassified launcher exit`. `renderer busy` is the intentional game/capture GPU lease boundary, not a crash or quarantine trigger. A stable grouping key, free of run-specific text. |
| `launcher_exit_detail` | the line that identifies *that* failure — the crate that failed to compile plus its `error[E….]` code, or the panic location plus its payload. |

`error` carries the first line only (`ToolRun::fail`'s contract: the classifier
belongs in the lane, the full text on stderr), so anything the lane must group by
has to *be* that first line. `capture launcher exited` alone was a bucket rather
than a classification — a triage window could see that 13 of 58 shots died and
nothing about why — so the launcher log is now classified the same way
`resource_fault_kind` classifies GPU faults, and `just diag` reports the cause
with one example payload beneath it instead of a repeated opaque string.
An `unclassified launcher exit` is a gap in that classifier and should be closed
by adding the signature, not by ignoring the count.

Machine-lock contention is its own `capture_lock` event (`wait_ms`, `queued`,
`owner_pid`) rather than a phase of whichever shot happened to be first —
otherwise one agent's queueing reads as another agent's slow capture.

Still open, and worth knowing when reading a slow record: `phase_render_ms` is
the client's outside view of the host. The host-internal split (scene settle vs.
warmup frames vs. readback/encode) is not separated yet, and the offline tools
(`bake`, `texgen`, `map`, `preview`) have no run record.

### `just nd-preview` — navigation-display preview

Renders the ND (`hud/mfd/widgets/nav_display.rs` + `assets/shaders/nav_display.wgsl`)
in eight approach situations and writes
`artifacts/visual/latest/nav_preview.png`, then exits. Agent-runnable, one
process, no game boot.

Each panel is a **real** `thalos_navigation::plan_approach` result, tessellated
by the game's own `route::plan_display`, projected by the game's own
`build_nav_scene`, and drawn by the real shader — so the image is evidence
about planner geometry, projection, scale, and symbology. It is **not**
evidence about ECS wiring: resource plumbing, widget auto-selection, click
handling, and the PFD deviation scales are not exercised and still need an
in-game check. Situations covered: straight-in, offset intercept,
overflown-and-turning-back, short final, reciprocal end, crosswind strip,
60 km out, and idle (nothing armed).

The preview exists because the ND's hard cases are the slow ones to fly to;
it turned up three real defects on its first two runs (a threshold bar that
read as a box, sub-pixel strip widths, and a final-approach highlight that
never drew). Source: `crates/runtime/game/examples/nav_preview.rs`. Spec:
`docs/gameplay/navigation.md`.

### `just loading-preview` — loading-screen preview

Renders the loading screen (`loading.rs`) to
`artifacts/visual/latest/loading_preview.png` and exits. Agent-runnable, one
process, no game boot.

It exists because that screen is **unreachable by every capture preset**: it
despawns the instant the last load step completes, which is strictly before the
capture host takes its shot, and holding a real load open long enough to shoot
would mean screenshotting a race. Until this preview it was the one surface in
the game changed blind.

`loading_preview::LoadingScreenPreviewPlugin` runs the real
`spawn_loading_screen` and the real `update_loading_progress_ui` /
`update_loading_diagnostics` against a seeded `LoadingTracker` and seeded
`PerfSamples`, so the image is evidence about layout, column alignment, and
number formatting. It is **not** evidence about the load: no step is driven by
a real producer, so step ordering, weights, and the `Loading → next` transition
still need an in-game check. The GPU and VRAM rows read the host machine live,
so they differ between runs by design.

Source: `crates/runtime/game/examples/loading_preview.rs`.

### Generated artifact layout

Generated evidence has three deliberately separate homes:

- `artifacts/visual/latest/` contains only the latest canonical whole-scene views.
  Each `just screenshot <preset>` overwrites its stable preset filename; do not
  keep numbered experiments or crops beside it.
- `artifacts/visual/catalog/` contains disposable offline discovery products.
  The viewpoint gallery downsamples canonical latest images here and records
  cache/provenance state in its index; thumbnails are navigation aids, not
  verification evidence.
- `artifacts/visual/runs/` is disposable visual working space. Put ad-hoc
  `THALOS_SCREENSHOT_OUT` captures there; `just compare` writes its complete
  matrices under `artifacts/visual/runs/comparisons/` automatically.
- `assets/viewpoints.json` is versioned developer-authored source data shared by
  the F8 manager, agents, and the headless capture CLI. It does not live under
  generated artifacts.
- `artifacts/diagnostics/` contains machine-readable runtime output.
  `runtime.jsonl` is the shared structured event stream; specialized memory,
  grass, shadow, capture, staging, and shipyard traces also live there. A bare
  filename supplied through `THALOS_RUNTIME_DIAGNOSTICS`,
  `THALOS_GRASS_LOG`, `THALOS_SHADOW_LOG`, or
  `THALOS_SHIPYARD_SELECT_LOG` is resolved there. An explicit relative path
  with a parent, or an absolute path, is still honored.

All generated output trees above are ignored by Git. The temporary procedural-object and
UI preview examples also write beneath `artifacts/visual/latest/`; future
in-context runtime presets will replace those examples without creating another
output root.

### Local storage hygiene

`target/` and `user/tilecache/` are disposable caches. Preserve the small
`user/*.ron` settings files, but clear the tile cache whenever its multi-gigabyte
footprint is no longer buying useful terrain-iteration latency. A full
`cargo clean` is the canonical reset for compiler output; do not park screenshots,
logs, or bakes directly under `target/`, because they disappear with that reset.

Remove obsolete worktrees with `git worktree remove`, followed by
`git worktree prune`, rather than deleting their directories manually. Prefer
worktrees outside the main checkout so their private `target/` trees are visible
as separate storage costs. Before removing a worktree, verify both its dirty
state and commits absent from `main`; unique research branches and their authored
data are not caches.

### Runtime content checkout

Runtime-ready learned terrain is versioned beside its provenance sidecars under
`assets/terrain_packages/`, with large `thalos_site_detail_*.f32` payloads stored
through Git LFS. A clone made with Git LFS installed materializes them during
checkout; `just terrain-assets` is the idempotent deferred-checkout/repair path.
It installs repository-local LFS filters, pulls only the learned terrain
payloads, and runs `git lfs fsck --objects`. The LFS object id is the payload's
SHA-256 and must match `sha256_le_f32` in its JSON sidecar. This bootstrap action
has no diagnostics lane: Git LFS's content-addressed object store and nonzero
exit status are the machine-readable integrity contract, and it runs before any
Thalos process exists. Training data, checkpoints, and optimizer state remain
outside the runtime checkout.

### Fast build setup — see docs/development/build_speed.md

The full cross-platform build-acceleration guide — fast linkers per platform
(rust-lld on Windows, mold on Linux/WSL), incremental + trimmed debug,
Windows Defender exclusions, headless-Vulkan requirements, and the
**per-environment agent build workflow** (single machine vs. parallel worktrees
vs. a Linux cloud box) — is [build_speed.md](build_speed.md). There is no
compiler cache (sccache removed, build_speed.md §5). Run
`scripts/setup-build-env.sh` (Linux/macOS/WSL) or `scripts/setup-build-env.ps1`
(Windows) to install the tools and write the local, gitignored config.
For WSL/Linux parallel boxes, create all worktrees first, run the shell setup
with `--agents <N> --all-worktrees`, and require
`bash scripts/check-build-env.sh --parallel` to pass before
the agent starts compiling or capturing.

Invariants that still bind here (detailed in build_speed.md):

- **One Cargo command at a time** against the workspace `target/` — concurrent
  `game`/`screenshot`/`check` invocations serialize on the target lock and
  contend for CPU + several GiB of compiler memory. Use `just check` while
  editing, then one linked `just game`/`just screenshot`. (Genuinely parallel
  agents get their own worktree + target dir instead — see
  build_speed.md §7.2.)
- **No unstable `-Zthreads`** — the 2026-07-20 parallel-MIR ICE (INC-0006) left
  unlinkable incremental objects; a speculative speedup became a full recovery
  build.
- **No compiler-backend experiments** — LLVM is the stable pinned toolchain's
  backend on every platform. The former Windows Cranelift attempt also failed
  with cross-crate undefined statics (reverted 2026-07-04).
- **One fingerprint per intentional renderer lane** — interactive/cold tools use
  the shared dynamic Bevy fingerprint, while the persistent capture server alone
  uses the static `dev-iteration` fingerprint. Do not create additional mixtures.
  The `mem_diag` GPU counters stay
  opt-in as `thalos_game/gpu-counters`; enabling them globally forces a second
  `bevy_dylib`. Use a temporary
  `THALOS_GAME_COMMAND="cargo run -p thalos_game --features bevy/dynamic_linking,gpu-counters"`
  with `THALOS_MEM_DIAG=1` only for a focused leak probe.
