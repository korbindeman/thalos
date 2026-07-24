# INC-20260724T040523Z — Rust hot-patch never fired for game crates: dx watches direct deps only

**Status:** fixed (2026-07-24)
**Affected:** the persistent `just screenshot` / `just compare` lane. Every
Rust edit outside `tools/capture_host` / `crates/capture/*` silently produced
**no hot patch**; the capture client waited 180 s for a code reload that never
came and errored ("timed out waiting for hot reload"). WGSL edits were
unaffected (Bevy's `embedded_watcher` does those, not dx).

## Symptom

After any Rust function-body edit in `thalos_runtime`, `thalos_body_render`,
`thalos_body_shading`, etc.:

```
$ just screenshot spaceport-aerial
waiting for renderer hot reload
… (180 s) …
timed out waiting for hot reload
```

The dx log shows **no file-change event at all** for the edited path — not
even an "Ignoring file change" line.

## Root cause

dx 0.7.9's watcher (`dioxus-cli` `serve/runner.rs::watch_filesystem`) watches
recursively only:

- the target package's crate dir (`tools/capture_host`), and
- its **direct** local dependencies (`local_dependencies()` uses
  `krates.get_deps(crate_package)` — one level, not transitive).

`thalos_capture_host` had exactly one direct dep (`thalos_capture_runtime`),
so the watch set was `tools/capture_host` + `crates/capture/*` — three hops
short of the actual game code. dx also maps changed files onto crates *from
those watch events* to decide what a patch build recompiles, so an unwatched
edit can never ride along with a later patch either (verified: a pending
`thalos_body_shading` edit was not included in a patch triggered from a
watched crate; the patch rlib list contained only the watched crate).

Compounded by INC-20260724T030400Z (the same lane failed at *link* even when a
rebuild did run), the practical effect was that the advertised hot Rust loop
had likely never worked end-to-end on this machine — "hot" Rust edits always
degraded to timeout + full restart.

## Fix

Declare the workspace crates of the capture graph as **direct** dependencies
of `thalos_capture_host` (`tools/capture_host/Cargo.toml`). They are already
transitive deps, so the unit graph and build cost are unchanged; the only
effect is that dx's watcher covers their source dirs. A loud comment in that
Cargo.toml says to add new workspace crates there.

Verified after the fix: a fn-body edit in `thalos_body_shading` hot-patches in
seconds and the client returns with a fresh PNG (numbers in
`docs/development/build_speed.md` §9).

## Prevention / recurrence tells

- **Tell:** `just screenshot` prints "waiting for renderer hot reload" then
  times out after a Rust edit, and the dx log has **no event** for the file →
  the edited crate is not in dx's watch set. Check it is listed as a direct
  dep in `tools/capture_host/Cargo.toml`.
- When adding a workspace crate that `thalos_runtime` (or anything under it)
  depends on, add it to `tools/capture_host/Cargo.toml` in the same change.
- On a dx upgrade, re-check `local_dependencies` in dioxus-cli's
  `serve/runner.rs`: if dx ever walks transitive local deps itself, the
  direct-dep list can be dropped.

## Cross-references

- INC-20260724T030400Z-dx-linker-rust-lld-generic-driver — the sibling failure
  discovered in the same investigation.
- ADR-20260724T022732Z-render-crate-split-for-hot-iteration — the leaf-crate
  split whose value this bug was masking (per-crate patch compile is what the
  split shrinks).
- `docs/development/build_speed.md` §3.1, §9.
