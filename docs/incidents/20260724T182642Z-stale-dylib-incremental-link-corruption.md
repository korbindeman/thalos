# INC-20260724T182642Z — Partial `cargo clean` of `bevy_dylib` poisons downstream incremental CGUs (`anon.*.llvm.*` link errors)

**Status:** fixed (the lane self-heals; `just build-reset` replaces hand-rolled
partial cleans).

## Symptom

`just capture massif-aerial massif-ridge massif-valley` failed to build the
capture host, with dozens of link errors of the form:

```
lld-link: error: undefined symbol: anon.36e01bade9299cfa31a13d2639e9c6ed.156.llvm.17986031716569828249
>>> referenced by …\bevy_ecs-0.19.0\src\query\fetch.rs:3323
>>>               libthalos_runtime-61e2dc29f6b42fb2.rlib(…rcgu.o):
                    (<bevy_ecs::query::fetch::Has<thalos_shipyard::part::Shroudable>
                      as bevy_ecs::query::world_query::WorldQuery>::init_state)
lld-link: error: too many errors emitted, stopping now
error: could not compile `thalos_capture_host` (bin "thalos_capture_host") due to 1 previous error
```

The failure appeared **immediately after** a manual cleanup of "stale
bevy_dylib/host artifacts" (`Removed 119 files, 2.2GiB total`) — i.e. the
cleanup intended to fix a flaky lane is what broke it. It is transient: later
in the same session an otherwise-identical build linked fine
(`Finished dev profile … in 1m 12s`) once the caches turned over.

## Evidence

- Every undefined symbol is an **LLVM-internalized local** (`anon.<cgu-hash>.N`
  or a real symbol carrying a `.llvm.<hash>` suffix) — not a Thalos or Bevy
  public symbol, so it is not a missing crate/feature/`extern`.
- Every reference comes from a **workspace rlib** (`libthalos_runtime-*.rlib`)
  at a **cross-crate generic instantiation** of `bevy_ecs` code
  (`Has<thalos_shipyard::part::*>::init_state`).
- The link line includes `bevy_dylib-adec0ffb8d4c0179.dll.lib`: the dev lane
  links Bevy dynamically (`--features dev-renderer`).
- `profile.dev` sets `incremental = true` (`Cargo.toml`).
- No project tooling runs `cargo clean` (`rg 'cargo clean' tools/ justfile
  scripts/` is empty), so the removal was operator-initiated and selective.

## Hypotheses considered

- **Missing feature / dependency drift** — ruled out: the symbols are compiler
  internals, and no source change accompanied the break.
- **The `lld-link.exe` shim (INC-20260724T030400Z)** — ruled out: the shim is a
  plain rust-lld copy, other links through it succeeded before and after, and
  the errors are undefined-symbol resolution, not driver-flavor aborts.
- **Concurrent Cargo processes sharing `target/`** — not the cause here: Cargo
  serializes on the target lock. It remains a *slowness* rule, not a
  corruption one.
- **Partial clean + incremental + dylib (accepted)** — the incremental cache of
  crates that were *not* cleaned still names internalized symbols from the
  **previous** `bevy_dylib` link. Rebuilding those crates from cache reproduces
  references that no object in the new link provides. Consistent with every
  observation: internal-only symbols, workspace-rlib referrers, cross-crate
  generic instantiations shared with the dylib, onset exactly at the selective
  clean, and self-clearing once the caches turn over.

## Root cause

`cargo clean -p bevy_dylib` (or any subset of the crates linked against it)
leaves `target/debug/incremental` holding codegen units that reference the old
dylib's internalized symbols. The dylib and the crates that link to it are
**one artifact set**; cleaning part of it is not a conservative repair, it is
the corruption.

## Fix

1. **The lane self-heals** (`tools/capture/src/main.rs`): `start_server` wraps
   `start_server_once`, and when the build log carries the corruption signature
   (`toolchain_corruption` — `undefined symbol: anon.` / an undefined symbol
   with `.llvm.`, invalid metadata, corrupt incremental) it drops
   `target/debug/incremental` and rebuilds **once**, rather than handing an
   agent a raw log tail. Ordinary compile errors are deliberately excluded and
   unit-tested (`ordinary_build_errors_are_not_treated_as_corruption`) so real
   errors still reach the agent verbatim.
2. **`just build-reset`** (`thalos_capture reset`) is the one supported full
   reset: stop the host, drop the incremental cache, and `cargo clean` the
   dynamic-linking crate set **as a unit**.

## Prevention

- **Never hand-roll `cargo clean -p <subset>` on this workspace.** Use
  `just build-reset`. Recorded in CLAUDE.md (Fast iteration invariants) and
  `docs/development/build_speed.md`.
- A build failure that names `anon.*` / `.llvm.*` symbols is an *artifact*
  problem, never a code problem — do not read it as a missing dependency.

## Recurrence tells

- Undefined symbols containing `anon.` or `.llvm.` in a `lld-link` failure.
- Referrers are workspace rlibs at `bevy_*` generic instantiations.
- The build succeeds again after an unrelated full rebuild — the tell that the
  tree, not the source, was inconsistent.
