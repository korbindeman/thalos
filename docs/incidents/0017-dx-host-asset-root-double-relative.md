# INC-0017: dx-launched capture host resolved assets outside the workspace

- **Status:** Fixed
- **Date:** 2026-07-22 (observed) / 2026-07-22 (fixed)
- **Severity:** startup failure (headless capture unusable)
- **Surface:** `just screenshot <preset>` — the persistent `dx serve` capture
  lane on Linux/WSL2 (any platform where the host binary is launched by dx
  rather than `cargo run`)

## Summary

The persistent capture host booted, then failed to load every runtime shader
and asset with `Path not found: <workspace>/../../assets/...`. The runtime's
`AssetPlugin.file_path` is the relative hop `"../../assets"`, written for
`cargo run`, where Bevy resolves it against `CARGO_MANIFEST_DIR` (`apps/game`
or `tools/capture_host`, both two levels below the workspace root). The
dx-launched executable has no manifest dir at runtime, so Bevy fell through to
`BEVY_ASSET_ROOT` — which the capture client sets to the workspace root — and
prepended it verbatim: `<root>/../../assets`, two levels *outside* the
workspace. The runtime now collapses `file_path` to `"assets"` whenever
`BEVY_ASSET_ROOT` is present.

## Symptoms

- Every embedded-material shader that is loaded from `assets/shaders/*`
  errored: `Path not found: /home/korbi/thalos/../../assets/shaders/ssao.wgsl`
  (and dozens more), plus
  `Skip creating file watcher because path ".../../../assets" does not exist`.
- The world booted and simulated normally (RON configs are read through other
  paths), so the failure surfaced as missing render layers, not a crash.

## Root cause

Two path authorities composed: a manifest-relative `file_path` and an absolute
`BEVY_ASSET_ROOT`. Bevy joins them (`root.join(file_path)`), so the relative
hop is correct under exactly one launcher and wrong under the other. Any
launcher that strips `CARGO_MANIFEST_DIR` (dx serve, a raw binary run) got the
broken join.

## Fix

`crates/runtime/game/src/lib.rs` builds `AssetPlugin.file_path` conditionally:
`"assets"` when `BEVY_ASSET_ROOT` is set (the capture client always sets it to
the workspace root), `"../../assets"` otherwise (unchanged `cargo run`
behaviour on every platform).

## Prevention & recurrence signals

- The tell is `Path not found` errors whose path contains `/../../assets`
  while the game log otherwise looks healthy.
- If another launcher appears (CI runner, packaged build), decide its asset
  root explicitly via `BEVY_ASSET_ROOT` rather than adding a third relative
  hop.
