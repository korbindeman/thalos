# INC-20260724T030400Z — Hot screenshot lane broken: dx-driven link × bare rust-lld

**Status:** fixed (2026-07-24)
**Affected:** the persistent `just screenshot` / `just compare` lane (dx
hot-patch capture host). Every boot/rebuild of `thalos_capture_host` failed at
link; the lane was dead on this machine for at least several sessions.

## Symptom

`just screenshot <preset>` hangs ~30 min, then fails. The capture server log
ends with:

```
ERROR linking with `C:\Users\korbi\.cargo\bin\dx.exe` failed: exit code: 1
ERROR Build failed: cargo build finished with errors for target: thalos_capture_host
```

No linker diagnostic appears in the log itself. The real stderr is captured in
`target/dx/visual_capture/windows/link_err.txt`:

```
lld is a generic driver.
Invoke ld.lld (Unix), ld64.lld (macOS), lld-link (Windows), wasm-ld (WebAssembly) instead
```

## Hypothesis differential

- **GPU device loss** (the prior session's capture blocker) — ruled out: the
  failure is at *link time*, before any GPU work; the same signature repeats
  across several dx sessions in the accumulated server log.
- **Seam A crate-split breakage** — ruled out: `cargo check -p thalos_game`
  passes, and the failure predates the split (older attempts in the same log,
  from before the extraction, fail identically).
- **sccache corruption** — plausible contributor in *older* attempts (the log
  shows `sccache.exe dx.exe rustc …` wrapping, i.e. the removed cache was still
  active via stale user env vars — see ADR-20260723T222214Z), but a de-sccached
  rebuild still failed identically → not the cause of *this* failure.
- **Linker flavor mismatch in the dx lane** — confirmed by `link_err.txt`.

## Root cause

The machine-local `.cargo/config.toml` (written by `setup-build-env.ps1`) set

```toml
[target.x86_64-pc-windows-msvc]
linker = ".../rust-lld.exe"
```

`rustc` invokes a configured linker through its linker-flavor machinery, so
bare `rust-lld.exe` works for every normal cargo lane. But **dx drives the
final link itself** for hot-patching (it interposes as the linker to capture
object lists, then re-invokes the configured linker with the raw MSVC-style
argument list). lld run as `rust-lld` with no `-flavor` argument refuses
generic invocation → exit 1 → "linking with dx.exe failed", with the actual
diagnostic hidden in dx's `link_err.txt`, not the build log.

So the "fast linker" provisioning silently broke the one lane the project
optimizes hardest — and because the diagnostic is stashed in a dx-internal
file, it read as a mysterious dx/build failure, not a config error.

## Fix

lld dispatches its driver flavor on **argv[0]**. Provision a plain copy of
`rust-lld.exe` named `lld-link.exe` (`~/.cargo/shims/lld-link.exe`) and point
the config at it:

```toml
[target.x86_64-pc-windows-msvc]
linker = "C:/Users/korbi/.cargo/shims/lld-link.exe"
```

- `rustc` infers the MSVC-lld flavor from the `lld-link` stem → normal lanes
  unchanged.
- dx's raw invocation now hits the link.exe-compatible driver → hot lane links.

`scripts/setup-build-env.ps1` now creates/refreshes the shim and writes the
shim path (re-copying when the toolchain's `rust-lld.exe` is newer, so
toolchain bumps propagate on re-provisioning).

## Prevention / recurrence tells

- **Tell:** `just screenshot` fails with "linking with … dx.exe failed" and no
  visible linker error → read `target/dx/<session>/windows/link_err.txt`
  first.
- Any future change to the configured Windows linker must be verified against
  **both** lanes: a normal `cargo build -p thalos_capture` *and* a
  `just screenshot` host boot — the dx lane bypasses rustc's linker-flavor
  handling.
- Changing the configured linker path invalidates Cargo fingerprints
  workspace-wide; expect (and budget) one full rebuild per lane after
  re-provisioning.

## Cross-references

- ADR-20260723T222214Z-abandon-sccache — the parallel machine-state cleanup
  (stale sccache user env vars were still wrapping every rustc here until
  2026-07-24).
- ADR-20260724T022732Z-render-crate-split-for-hot-iteration — the crate-split
  work this investigation was verifying.
- `docs/development/build_speed.md` §6.1 — Windows linker provisioning.
