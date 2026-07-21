# ADR-20260721T212438Z: Portable build policy, local acceleration

**Status:** accepted
**Date:** 2026-07-21

## Context

Thalos needs a fast edit/check/headless-capture loop and must also support
several agents compiling independently. The existing ignored
`.cargo/config.toml` mixed portable profile settings, a user-specific linker
path, and a Cargo alias. Fresh worktrees did not reliably inherit the portable
parts, while simultaneous processes sharing one `target/` serialized on Cargo's
lock and competed for CPU and memory.

Incremental compilation and sccache serve different inputs. Incremental is best
for frequently edited workspace/path crates. Sccache can cache non-incremental
registry dependencies by default, and can cache workspace crates when a cold or
parallel build explicitly disables incremental compilation.

## Decision

- Commit portable development profile settings in the root `Cargo.toml`:
  incremental workspace compilation, line-table debug information, opt-level 1
  for Thalos crates, and opt-level 3 for dependencies.
- Keep `.cargo/config.toml` ignored and machine-local. Setup scripts generate
  only the fast linker, an always-on sccache wrapper, and a bounded Cargo job
  count sized for the expected number of concurrent agents.
- Expose stable repository commands through `just`, not local Cargo aliases.
- Parallel agents use separate worktrees and therefore separate `target/`
  directories. They share one sccache store, with every current worktree root
  configured as a normalization base. No two active Cargo processes may write
  to the same target directory.
- The normal edit loop keeps incremental compilation enabled. Cold/parallel
  population builds set `CARGO_INCREMENTAL=0`, allowing sccache to reuse
  workspace crates across worktrees.
- Keep LLVM, stable Rust, and supported fast linkers. Do not reintroduce
  experimental compiler backends or `-Zthreads`.

## Consequences

- Every checkout receives the correctness-neutral profile policy without
  inheriting another machine's paths or resource assumptions.
- Registry dependencies are cacheable during ordinary iteration; repeated
  worktrees can also cache Thalos crates in the explicit parallel regime.
- Each agent pays for its own final link and incremental state, but avoids target
  locks and most repeated dependency compilation.
- Parallelism consumes additional disk because target directories cannot safely
  be shared. Obsolete worktrees and their targets must be pruned deliberately.
