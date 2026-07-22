# INC-0008: Direct dynamic game launch missed Cargo's library search path

- **Status:** Fixed
- **Date:** 2026-07-21 (observed) / 2026-07-21 (fixed)
- **Severity:** crash
- **Surface:** `just compare <preset> <axis>` before the first headless capture

## Summary

The first one-Cargo implementation of `just compare` built the dynamically linked
game and comparison orchestrator, then launched the orchestrator executable
directly. The orchestrator's child `thalos_game.exe` exited before `main` with
Windows status `0xc0000135`: direct launch did not inherit the dynamic-library
search path Cargo normally installs for `cargo run`. The runner now prepends the
game profile directory, its `deps` directory, and rustc's target library
directory to the platform loader path for every child capture.

## Symptoms

- `cargo run ... --example visual_compare` worked.
- `target/debug/examples/visual_compare.exe spaceport-aerial ssao` started the
  runner, but variant 1 immediately failed before the Thalos startup banner.
- The runner reported `variant 'off' exited with exit code: 0xc0000135` and wrote
  no PNG.

## Evidence

The A/B run launched through Cargo completed and wrote all artifacts. The same
orchestrator executable launched directly reached its own log line, then its game
child failed before any game log:

```text
visual comparison: preset=spaceport-aerial axis=ssao (3 isolated captures)
[1/3] THALOS_SSAO=off
visual comparison failed: variant 'off' exited with exit code: 0xc0000135
```

`target/debug/thalos_game.exe`, the Bevy dynamic libraries under
`target/debug` / `target/debug/deps`, and `std-*.dll` under the output of
`rustc --print target-libdir` existed. Adding only the profile + `deps` paths
still reproduced `0xc0000135`; adding the rustc target libdir made the same game
binary reach its startup banner and finish a headless probe. The distinguishing
input was Cargo's complete launch environment, not compilation or the selected
SSAO variant.

## Hypotheses considered

- **The SSAO-off configuration crashed game startup** — ruled out because the
  process failed before the Thalos banner and `THALOS_SSAO=off` is an established
  screenshot diagnostic.
- **The game binary was absent or stale** — ruled out by the runner's preflight
  file check and the freshly completed build.
- **The comparison executable itself could not load** — ruled out because it
  printed the comparison/variant lines before the child failed.
- **The child lacked Cargo's dynamic-library loader paths** — confirmed by the
  Windows missing-DLL status and a controlled path A/B: profile + `deps` still
  failed, while profile + `deps` + rustc target-libdir started successfully.

## Root cause

Bevy's committed dev path uses `bevy/dynamic_linking`. Cargo augments the child
process's platform loader environment when it executes a binary. Replacing
`cargo run` with a direct executable launch removed that implicit environment;
the visual runner then spawned the dynamic game without `target/debug/deps` on
`PATH` (or the Unix/macOS equivalent). The local toolchain also uses dynamic
Rust runtime linkage, so `std-*.dll` in `rustc --print target-libdir` was the
decisive missing dependency after the Bevy/profile paths were restored.

## Fix

Once per comparison, `visual_compare` resolves `rustc --print target-libdir`.
Before spawning each game variant it prepends that directory, the game profile
directory, and `<profile>/deps` to `PATH` on Windows,
`DYLD_FALLBACK_LIBRARY_PATH` on macOS, or `LD_LIBRARY_PATH` elsewhere. Existing
user paths are preserved. This reproduces the required part of Cargo's launch
contract while retaining the one-Cargo-build comparison recipe.

## Prevention & recurrence signals

- Any committed workflow that directly launches a dev renderer built with
  `bevy/dynamic_linking` must install the profile, profile/deps, **and rustc
  target-libdir** loader paths or keep using `cargo run`; see `docs/development/tooling.md`
  and `docs/development/visual_testing.md`.
- A direct-built renderer that exits before its first log with `0xc0000135`
  (Windows) or a missing `.so` / `.dylib` loader error should be checked for this
  path contract before debugging game startup.
