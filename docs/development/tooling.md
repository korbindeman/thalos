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
whenever any workspace `.rs`/`.toml` is newer than the running host — the next
capture stops it, rebuilds (dynamic relink), and boots it again
(~1.5–2.5 min warm). There is no in-process Rust reload: dx/subsecond
hot-patching was retired after an applied patch reproducibly crashed the app
(INC-20260724T044418Z). `just capture-stop` remains as manual hygiene.

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
`overrides_from_env` in `crates/runtime/game/src/window_settings.rs` as a session
override: it wins over the persisted `user/settings.ron` vsync preference and
greys out the VSync control in the settings menu, without being written into
the file. (Vsync can also be toggled live from the settings menu's Window
tab, which *does* persist.)

For A/B attribution, have the user change one variable at a time and report
the on-screen frame time (the FPS overlay) or capture a chrome trace
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

Keep a specialized JSONL recorder when it has a real independent schema or
lifecycle—capture reports, stage-separation audits, grass churn, GPU-memory
sampling, shadow traces, and shipyard selection traces are examples. Reuse
`artifact_paths` so bare names stay under the diagnostics tree. A new
diagnostic should default to the shared runtime stream; it should not add a new
file merely to avoid choosing an event name.

### Generated artifact layout

Generated evidence has three deliberately separate homes:

- `artifacts/visual/latest/` contains only the latest canonical whole-scene views.
  Each `just screenshot <preset>` overwrites its stable preset filename; do not
  keep numbered experiments or crops beside it.
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

All three output trees are ignored by Git. The temporary procedural-object and
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
