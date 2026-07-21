# Tooling

Thalos keeps committed Cargo and Rust toolchain configuration deliberately
plain. `rust-toolchain.toml` pins nightly, and `Cargo.toml` sets only shared
profile choices that are expected to be sane across Windows, macOS, and Linux.

## Local compiler tuning

Platform-specific compiler and linker speedups belong in local Cargo config,
not in committed project config. Use either a personal Cargo config under your
home directory or the workspace-local `.cargo/config.toml`; the workspace file
is ignored by Git for this purpose. The `just game` command can also be
customized locally with `.env.just`, which is ignored by Git.

This includes:

- `CARGO_INCREMENTAL` overrides for platform-specific incremental behavior.
- Debug-info reductions for local iteration.
- Local linker or backend experiments.
- `rustc-codegen-cranelift-preview` / `codegen-backend = "cranelift"`.

Do not commit a Cargo backend override unless the project intentionally adopts
that backend for all supported platforms. The default checked-in backend is the
Rust toolchain's normal LLVM backend.

### Bevy dynamic linking (committed default for every dev renderer)

`just game` and `just screenshot` use the `justfile`'s shared `game_command`,
which defaults to
`cargo run -p thalos_game --features bevy/dynamic_linking`. `just preview` and
`just ui-preview` enable the same Bevy feature for their own packages; the UI
preview also requests the game's `wayland`/`jpeg` Bevy feature set so Cargo can
reuse the same `bevy_dylib` artifact instead of compiling a second variant.
Dynamic linking is a cross-platform dev-iteration speedup — Bevy links once into
a shared `bevy_dylib` and subsequent rebuilds relink only our crates — and it is
not platform-specific, so it belongs in committed config rather than each
developer's `.env.just`.

Cargo also supplies a dynamic-library search path when it runs a built renderer.
Any tool that launches a dev renderer executable directly must reproduce that
contract by prepending the Cargo profile directory, its `deps` directory, and
`rustc --print target-libdir` to `PATH` (Windows),
`DYLD_FALLBACK_LIBRARY_PATH` (macOS), or `LD_LIBRARY_PATH` (other Unix).
`visual_compare` owns this for `just compare`; omitting the rustc directory can
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

### Game window / renderer launch toggles

`just game` normally starts borderless fullscreen. The renderer backend
defaults to **Vulkan on Windows** (set in `wgpu_settings_from_env`,
`crates/game/src/main.rs`) — wgpu's own default prefers DX12 there, and DX12
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

Bevy 0.18 has a text-rendering bug at **fractional** window scale factors: on a
150 % display (scale 1.5), glyphs rasterise at inconsistent sizes and the UI
text looks broken (non-uniform, "not monospace"). Integer scale factors render
cleanly. So `main.rs`'s `snap_window_scale_to_integer` snaps the OS scale to the
nearest integer (≥ 1) at startup — 1.5 → 2, 1.25 → 1 — which makes text crisp at
the cost of the UI being slightly larger or smaller than the OS-intended size.
Set `THALOS_SCALE=<float>` to pin a specific factor instead (e.g. `THALOS_SCALE=1`
for native-pixel UI on a HiDPI laptop, smaller but sharp). Remove the snap once
the upstream Bevy fractional-scale text bug is fixed.

`THALOS_WGPU_BACKEND` is a Thalos-facing alias for the same class of wgpu
backend selection that `WGPU_BACKEND` provides, but it is scoped to our game
startup helper in `crates/game/src/main.rs` and is easy to keep in `.env.just`.
It overrides the Vulkan-on-Windows default above: `auto` restores wgpu's own
selection (DX12 on Windows), `dx12` forces DX12 for A/B comparison.

### Uncapping the framerate for profiling (`THALOS_VSYNC`)

Frame time is the only meaningful performance signal, and vsync floors it at
the monitor's refresh budget — so a change that shaves real GPU/CPU time below
that floor shows no movement. Set `THALOS_VSYNC=off` (also accepts
`0`/`false`/`no`) to launch with `PresentMode::AutoNoVsync` and read the true,
uncapped frame time while still allowing wgpu to fall back to a supported
non-vsync present mode; anything else keeps the vsync default. Read by
`overrides_from_env` in `crates/game/src/window_settings.rs` as a session
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
collider (see `docs/surface.md`). The game has no remote-inspection channel —
you analyze the artifacts (trace JSON, slow-frame JSONL, console logs), the
user runs the game.

### Generated artifact layout

Generated evidence has three deliberately separate homes:

- `tools/screenshots/` contains only the latest canonical whole-scene views.
  Each `just screenshot <preset>` overwrites its stable preset filename; do not
  keep numbered experiments or crops beside it.
- `tools/agent_scratch/` is disposable visual working space. Put ad-hoc
  `THALOS_SCREENSHOT_OUT` captures there; `just compare` writes its complete
  matrices under `tools/agent_scratch/screenshots/comparisons/` automatically.
- `tools/diagnostics/` contains machine-readable runtime output. F8 writes the
  latest player-to-agent camera handoff to `latest_perspective.json`; memory and
  grass diagnostics also default there, and a bare filename supplied through
  `THALOS_GRASS_LOG`, `THALOS_SHADOW_LOG`, or
  `THALOS_SHIPYARD_SELECT_LOG` is resolved there. An explicit relative path
  with a parent, or an absolute path, is still honored.

All three output trees are ignored by Git (apart from the scratchpad README).
The procedural-object and UI preview galleries keep their existing dedicated
locations because they are stable multi-view galleries, not whole-scene
iteration history.

### Windows fast incremental loop

A good Windows-local starting point is:

```toml
[env]
CARGO_INCREMENTAL = "1"

[profile.dev]
incremental = true
debug = "line-tables-only"

[profile.dev.package."*"]
debug = "line-tables-only"

[target.x86_64-pc-windows-msvc]
linker = "rust-lld.exe"

[alias]
check-game = "check -p thalos_game"
```

Use `just game` as the single app path on every platform. Bevy's
`dynamic_linking` is already the committed default for `just game` (see above),
so `.env.just` only needs to carry Windows-specific bits if any — the Cargo
config above keeps the compiler backend on LLVM and uses LLD for faster
MSVC-target linking. Release commands stay on the checked-in defaults because
neither `Cargo.toml` nor the release-path `just` recipes (`just build`,
`just trace`) enable `bevy/dynamic_linking`.

Use `cargo check-game` for fast type checking when no app launch is needed.
Avoid adding a second local run alias unless the default Windows path changes
deliberately.

Do **not** add nightly `-Zthreads` to this config. The 2026-07-20 Windows
nightly produced a parallel-MIR ICE under `-Zthreads=8`, then left incremental
codegen objects with missing LLVM symbols. That turns a speculative compile
speedup into a full non-incremental recovery build. Cargo's normal crate-level
parallelism plus reliable incremental reuse is the faster loop in practice.

Run one Cargo command at a time against the workspace `target/` directory.
Concurrent game/screenshot/check invocations serialize on Cargo locks, compete
for CPU and several GiB of compiler memory, and make failures harder to
attribute. Prefer `cargo check-game` while editing, then one `just game` or
`just screenshot` when a linked artifact is actually needed.

The temporary wgpu driver counters used by `mem_diag` are intentionally absent
from the default graph: enabling them globally changes wgpu's feature hash and
forces a second complete `bevy_dylib`. For a focused leak investigation, opt in
explicitly with a temporary
`THALOS_GAME_COMMAND="cargo run -p thalos_game --features bevy/dynamic_linking,gpu-counters"`
override together with `THALOS_MEM_DIAG=1`; normal iteration must leave the
feature off.

If `rust-lld.exe` is not on `PATH`, either install/update the Rust LLVM tools
for the active toolchain or use the absolute path to the toolchain copy in
local config. On Windows that copy usually lives under:

```text
%USERPROFILE%\.rustup\toolchains\<toolchain>\lib\rustlib\x86_64-pc-windows-msvc\bin\rust-lld.exe
```

### macOS incremental workaround

If a macOS toolchain hits stale `.llvm.<hash>` anonymous symbol references
between incremental codegen objects, disable incremental compilation locally
instead of changing the workspace profile:

```toml
[profile.dev]
incremental = false
```

macOS developers who want Cranelift for local iteration can also configure it
locally or pass one-off `cargo --config` flags. Keep that opt-in local so
Windows and Linux continue to use LLVM unless the project makes a deliberate
cross-platform backend decision.
