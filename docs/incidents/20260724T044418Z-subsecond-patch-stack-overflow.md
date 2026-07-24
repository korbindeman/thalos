# INC-20260724T044418Z — Applied Rust hot patch stack-overflows the app (subsecond, Windows)

**Status:** diagnosed; **mitigated by workflow** (Rust edits use the
capture-stop → rebuild loop), root cause is upstream (dx/subsecond).

## Symptom

With the hot lane otherwise fully repaired (INC-20260724T030400Z linker,
INC-20260724T040523Z watch scope), a Rust fn-body edit hot-patches cleanly —
dx builds and applies the patch DLL in ~5–7 s, the host logs
"Rust hot patch applied" — and then the app dies within ~0.2 s:

```
thread 'Compute Task Pool (0)' (…) has overflowed its stack
Application [windows] exited with error: exit code: 0xc00000fd
```

100 % reproducible for a patched function that executes again after the patch
(`StarLight::default` in `thalos_body_shading`, exercised per frame via
`SceneLighting`). A patched function that never re-executes
(`CaptureApp::run`, called once at startup) "survives" — which is why the
first smoke test looked green.

## Differential

- **opt-level artifact** — ruled out: pinning the `desktop-dev` profile's
  workspace code to `opt-level = 0` (subsecond's recommended shape) reproduces
  the identical overflow (several task-pool threads at once).
- **Our dx replay patch** — ruled out: the crash follows a *successful* patch
  application; the replay fix only decides which crates get recompiled, and
  the same crash shape predates any deep-crate involvement (the overflow is in
  the running app, not the build).
- **Upstream immaturity of workspace hotpatching** — consistent with the
  ecosystem: dioxus#4160 tracks sub-crate patching as an open feature, and the
  Bevy hotpatching experiments README states outright that reliable patching
  assumes "your app is not allowed to have a lib.rs or a workspace setup".
  The stack-overflow shape (patched fn re-entered through its own detour /
  jump table on multiple task-pool threads) points at subsecond's Windows
  jump-table handling for deep-workspace crates.

## Impact / the honest loop

Rust hot-patching is **not usable** in this workspace until upstream matures.
The working iteration loop as of 2026-07-24:

| Edit class | Loop | Measured |
|---|---|---|
| WGSL (any shared lib or material shader) | save → `just screenshot` (Bevy `embedded_watcher` reload — dx not involved) | **2.6 s** to a fresh PNG |
| Rust (any) | `just capture-stop` → `just screenshot` (rebuild + fat link + boot; disk tile cache makes re-stream fast) | **~1.5–2.5 min** |
| Cold lane (fresh worktree / linker change) | first `just screenshot` | **~7 min** |

If a Rust edit is saved while the host runs, dx auto-patches, the app crashes,
and the *next* `just screenshot` boots a fresh server with the edit compiled
in — self-healing, similar total cost, but the triggering client may hang to
its timeout; prefer the explicit capture-stop flow.

## Prevention / recurrence tells

- **Tell:** "Rust hot patch applied" immediately followed by
  "has overflowed its stack" / exit `0xc00000fd` in
  `artifacts/diagnostics/visual_capture_server.log`.
- Re-test on every dx/subsecond upgrade with: host up → trivial fn-body edit
  in `thalos_body_shading` → does the app survive 30 s? If it survives, flip
  the workflow guidance in build_speed.md §3.1 back to hot Rust patching and
  retire this incident's workaround.
- If upstream stays stuck, the considered alternative is a persistent
  **dev-renderer** capture host (dynamic linking + `embedded_watcher`, no dx):
  WGSL reload identical, Rust edits become a dynamic relink + restart —
  tracked as a note on BL-20260724T041500Z.

## Cross-references

- INC-20260724T030400Z, INC-20260724T040523Z — the same-day lane repairs that
  exposed this (the crash was unreachable while the lane was link-dead).
- https://github.com/DioxusLabs/dioxus/issues/4160 — sub-crate hotpatching.
- `docs/development/build_speed.md` §3.1, §9.
