# ADR-20260724T153619Z — Retire Rust hot-patching; one stable dynamic capture lane

**Status:** accepted
**Date:** 2026-07-24
**Supersedes:** the dx/subsecond hot-patch mechanism of
ADR-20260721T192218Z-persistent-visual-iteration (the *persistent capture
server* it establishes stays; only the reload mechanism changes) and the
hot-patch-scope rationale of
ADR-20260724T022732Z-render-crate-split-for-hot-iteration (the crate split
itself stands — it still bounds structural-rebuild scope).

## Context

The 2026-07-24 lane repair (INC-20260724T030400Z linker,
INC-20260724T040523Z watch scope) got dx-driven Rust hot-patching working for
the first time — and immediately exposed that it cannot be trusted:
a successfully applied patch stack-overflows the app the next time the patched
function runs (INC-20260724T044418Z; 100 % reproducible, opt-level-independent,
upstream subsecond immaturity on workspace crates). The failure mode is the
worst kind for agent-driven iteration: the host dies *after* reporting
success, subsequent captures hang or time out, and an agent burns multiple
turns diagnosing a dead server — dirtying the worktree along the way. Keeping
the lane also carried standing costs: a locally patched dioxus-cli that any
`cargo binstall` silently clobbers, a direct-dependency watch-scope hack in
`tools/capture_host/Cargo.toml`, a second full renderer fingerprint
(static `desktop-dev`, ~8 GB of target artifacts), and a 228 MB statically
linked host exe that alone costs seconds of process-load per boot.

Meanwhile the measured value split was lopsided: WGSL edits — the dominant
visual-iteration edit class — hot-reload through Bevy's `embedded_watcher`,
which has nothing to do with dx; and Rust edits were already forced onto the
stop → rebuild → reboot path by the crash.

## Decision

Remove Rust hot-patching from the project. The persistent capture host runs as
a plain detached `cargo run -p thalos_capture_host --features dev-renderer`
process — the **same dynamic-linking fingerprint as `just game` and the cold
capture lane** — with `bevy/embedded_watcher` folded into `dev-renderer`.

- **One renderer fingerprint, everywhere.** The `dev-iteration` feature chain
  (`bevy/hotpatching`) is deleted from `thalos_runtime`, `thalos_game`,
  `thalos_capture_runtime`, and `thalos_capture_host`. The static
  `desktop-dev` graph and dx session caches are dead artifacts.
- **WGSL stays hot** (~3 s save → fresh PNG): `embedded_watcher` reloads both
  `assets/shaders/*` and crate-embedded WGSL in the running process. Adding it
  to `dev-renderer` also gives the interactive game live shader reload.
- **Rust edits are always a restart — automatically.** The capture client
  compares the newest workspace `.rs` mtime against the running host's launch
  time; when the host is stale it stops it, rebuilds (dynamic relink), and
  relaunches before shooting. `just capture-stop` becomes optional hygiene,
  not a required ritual, and there is no state in which a Rust edit produces a
  crashed or silently-stale server.
- **dx is no longer a dependency of anything.** The patched dioxus-cli, the
  `scripts/patches/` file, and the direct-dep watch block in
  `tools/capture_host/Cargo.toml` are removed. The `lld-link.exe` shim stays —
  it is correct for plain rustc use and guards against any future tool that
  drives the linker raw.

## Rejected alternatives

- **Keep hot-patching behind a flag until upstream fixes the crash.** Rejected:
  an opt-in crash is still a crash an agent will eventually trigger; the
  maintenance surface (patched CLI, watch hack, second fingerprint) is paid
  even when unused. If dx/subsecond matures, re-evaluate under a new ADR — the
  INC records the re-test.
- **Static (non-dylib) persistent host without dx.** Rejected: with hot-patch
  gone there is nothing the static fingerprint buys; dynamic linking relinks
  Thalos code in seconds, shares the existing dev graph, and loads a far
  smaller exe.
- **Fix subsecond ourselves.** The jump-table/detour work is deep upstream
  surgery on a moving target, for a mechanism the loop has proven it can live
  without.

## Consequences

- The crash class is structurally gone: there is no code path that patches a
  running process.
- Iteration loop: WGSL ≈ 3 s (unchanged); Rust edit → auto-restart →
  PNG ≈ rebuild + relink + boot (measured after landing; the static lane's
  fat-link and 228 MB exe-load costs drop out, replaced by a dylib relink and
  a small exe).
- `target/x86_64-pc-windows-msvc/desktop-dev` and `target/dx` (~several GB)
  are reclaimable disk.
- The one-fingerprint rule in CLAUDE.md simplifies to: *every* dev renderer
  lane — interactive, cold, persistent — shares the dynamic fingerprint.
