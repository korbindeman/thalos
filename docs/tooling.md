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

### Bevy dynamic linking (committed default for `just game`)

`just game` defaults to `cargo run -p thalos_game --features bevy/dynamic_linking`
on every platform (the default lives in the `justfile`'s `game_command`).
Dynamic linking is a cross-platform dev-iteration speedup — Bevy links once into
a shared `bevy_dylib` and subsequent rebuilds relink only our crates — and it is
not platform-specific, so it belongs in committed config rather than each
developer's `.env.just`.

It is scoped to the dev run command only. `just build`, `just trace` (release),
and the bakes do not enable it, so it never reaches a shipped/release build
(where a missing `bevy_dylib` would crash the binary). To opt out locally —
e.g. on a platform where the feature misbehaves — override the whole command in
`.env.just`:

```dotenv
THALOS_GAME_COMMAND="cargo run -p thalos_game"
```

### Game window / renderer launch toggles

`just game` normally starts borderless fullscreen with Bevy/wgpu's default
backend selection. If Windows reports a swapchain/device-loss style panic while
acquiring the next frame, the first local workaround is to take the display mode
and backend choice out of the equation without changing source:

```dotenv
THALOS_WINDOW_MODE=windowed      # windowed | borderless | fullscreen
THALOS_WINDOW_SIZE=1600x900      # optional, used for windowed mode
THALOS_WGPU_BACKEND=vulkan       # auto | dx12 | vulkan | metal | gl
```

`THALOS_WGPU_BACKEND` is a Thalos-facing alias for the same class of wgpu
backend selection that `WGPU_BACKEND` provides, but it is scoped to our game
startup helper in `crates/game/src/main.rs` and is easy to keep in `.env.just`.
Use `vulkan` on Windows when DX12 presents become unstable; remove the override
again after driver/Bevy/wgpu updates.

### Uncapping the framerate for profiling (`THALOS_VSYNC`)

Frame time is the only meaningful performance signal, and vsync floors it at
the monitor's refresh budget — so a change that shaves real GPU/CPU time below
that floor shows no movement. Set `THALOS_VSYNC=off` (also accepts
`0`/`false`/`no`) to launch with `PresentMode::AutoNoVsync` and read the true,
uncapped frame time while still allowing wgpu to fall back to a supported
non-vsync present mode; anything else keeps the vsync default. Read by
`present_mode_from_env` in `crates/game/src/main.rs`.

Pair it with the BRP MCP server (`bevy_brp_mcp`) for live, no-rebuild A/B
profiling of a running session: `brp_extras_get_diagnostics` reports
`frame_time_ms`; toggling the ship camera's `Camera.is_active` isolates
GPU-vs-CPU bound (frame time unchanged with the 3D render off ⇒ CPU-bound);
removing post-process components (`Bloom`, `Smaa`, …) or sending `Escape`
(sim pause) attributes cost to specific subsystems. This is how the surface
frame cost was traced to the Avian terrain collider (see `docs/surface.md`).

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
