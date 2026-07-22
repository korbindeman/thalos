# INC-0006: Experimental parallel rustc ICE poisoned incremental objects and broke the next link

- **Status:** Fixed
- **Date:** 2026-07-20 (observed and fixed)
- **Severity:** perf
- **Surface:** Windows dev build / headless screenshot link

## Summary

The local Windows Cargo config enabled nightly rustc's experimental
`-Zthreads=8`. During a normal game build, rustc panicked in the parallel MIR
query pool. Cargo retained incomplete incremental objects; the retry reached
`rust-lld` but failed with many missing anonymous LLVM symbols. A one-off
non-incremental build recovered the binary, and the durable fix removed
`-Zthreads` from local config. Dev renderer recipes now also share Bevy dynamic
linking so a screenshot iteration does not statically relink the whole engine.

## Symptoms

- `cargo run -p thalos_game` panicked inside `rustc_mir_transform`.
- The immediate retry compiled but `rust-lld` reported undefined
  `anon.<hash>.llvm.<hash>` and `drop_in_place` symbols from game codegen units.
- Recovery with `CARGO_INCREMENTAL=0` took several minutes because it discarded
  the incremental advantage and performed a full static game link.

## Evidence

The compiler-generated ICE report recorded:

```text
rustc 1.96.0-nightly (fb27476aa 2026-03-28)
thread 'rustc' panicked at rustc_mir_transform/src/lib.rs:547:25:
stealing value which is locked: ()
#0 mir_drops_elaborated_and_const_checked
#1 analysis (thalos_world)
```

The local `.cargo/config.toml` supplied `rustflags = ["-Zthreads=8"]`; the
backtrace ended in `rustc_thread_pool::WorkerThread::wait_or_steal_until_cold`.
The next link's missing symbols came from incremental `.rcgu.o` files. Rebuilding
without incremental codegen linked and ran successfully, ruling out an actual
unresolved symbol in Thalos source.

## Hypotheses considered

- **A Thalos type/link error:** ruled out because the focused `cargo check`
  passed and the same source linked after non-incremental recovery.
- **A standalone `rust-lld` defect:** ruled out as the initiating failure; the
  first failure occurred in MIR before linking, and fresh objects linked.
- **Concurrent Cargo commands:** they explained build-directory lock waits and
  resource contention, but Cargo serialized target writes; they did not explain
  the compiler's locked parallel query value.
- **Experimental within-crate parallelism:** matched the panic mechanism and
  stack, and was the only unstable compiler option in the local config.

## Root cause

The pinned nightly compiler raced in its experimental parallel frontend under
`-Zthreads=8`. The ICE interrupted incremental code generation after Cargo had
materialized part of the crate's cache. Reusing that cache gave the linker an
internally inconsistent set of codegen units.

## Fix

- Removed `-Zthreads=8` from the ignored Windows-local Cargo config. Cargo still
  parallelizes independent crates, while within-crate compilation uses rustc's
  supported default.
- Kept incremental compilation and line-table-only debug info.
- Routed game, screenshot, object-preview, and UI-preview dev recipes through
  `bevy/dynamic_linking`, so subsequent links relink Thalos rather than Bevy.
- Made `just screenshot` use the shared `game_command` instead of a parallel
  static-link command.
- Moved wgpu's diagnostic `counters` feature behind the explicit
  `thalos_game/gpu-counters` feature. Normal game and preview builds therefore
  share one wgpu/Bevy feature fingerprint and one `bevy_dylib` artifact.

## Prevention & recurrence signals

- Do not add nightly `-Zthreads` to the recommended local config; see
  [tooling.md](../development/tooling.md#windows-fast-incremental-loop) and `CLAUDE.md`.
- Run only one Cargo command at a time against the workspace `target/`.
- The recurrence signature is `stealing value which is locked` followed by
  anonymous `.llvm` undefined symbols. Stop retrying the poisoned cache: remove
  the experimental flag and perform one non-incremental recovery build.
