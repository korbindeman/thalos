# ADR-20260723T222214Z: Abandon sccache

**Status:** accepted
**Date:** 2026-07-23
**Supersedes:** the always-on-sccache portion of
ADR-20260721T212438Z-portable-build-policy-local-acceleration (the rest of that
ADR — portable `Cargo.toml` profiles, machine-local linker/job-budget config,
`just` over Cargo aliases, isolated worktree targets — still stands).

## Context

sccache was adopted as the shared compiler cache that turns a cold Bevy
dep-graph build into a cache hit, primarily for the cold/parallel regime (fresh
worktrees, a cloud box, N agents). It has instead been a recurring source of
cost with a thin payoff on the machine it actually runs on most:

- **It produced broken/corrupt builds.** The proximate trigger for this
  decision: build errors attributable to the sccache wrapper (bad cached
  invocations / inconsistent output), not to the source under edit. A compiler
  cache that makes a build *wrong* is worse than a slow build — it costs
  debugging time chasing phantom failures.
- **Its activation and normalization were chronically silent-fragile.** INC-0019
  documents two independent silent failures (directory-scoped activation left
  worktrees uncached; `SCCACHE_BASEDIRS` is a provisioning-time snapshot that
  decays on every `git worktree add`), plus a latent platform-separator footgun
  (`:` vs `;`) and a hard version floor (≥ 0.14.0). The fix for INC-0019 was an
  entire apparatus — a machine-global `RUSTC_WRAPPER`, longest-first root
  sorting, canonical path comparison in the checker, `-SyncOnly`/`-AllWorktrees`,
  daemon-restart guards. That apparatus is pure sccache-tax.
- **Its benefit on the dominant loop is marginal.** On a solo native-Windows
  iterate loop, `profile.dev` sets `incremental = true`, so every *workspace*
  crate is non-cacheable by construction (sccache's own accounting in INC-0019:
  250 invocations skipped for `incremental`, 374 for `crate-type`). sccache only
  caches registry dependencies on a cold/clean build there. The one regime where
  it genuinely paid — a shared cache across parallel worktrees — is not the
  regime this project runs in day to day.

Weighing a build-*correctness* hazard plus a standing maintenance surface
against a marginal, regime-specific speedup, the cache is not worth keeping.

## Decision

Remove sccache from Thalos entirely.

- **No compiler cache.** The build acceleration stack is now: a fast linker
  (rust-lld on Windows, mold on Linux/WSL, default `ld` on macOS), Bevy
  `dynamic_linking` on the dev renderer lanes, committed incremental + trimmed
  debug-info profiles, a bounded per-process Cargo job budget, Windows Defender
  exclusions, and the persistent Subsecond hotpatch capture host. None of these
  depend on a compiler cache.
- **The setup scripts stop touching sccache.** `setup-build-env.ps1` /
  `setup-build-env.sh` no longer install sccache, set `rustc-wrapper` /
  `RUSTC_WRAPPER`, or manage `SCCACHE_*` / basedirs. `setup-build-env.ps1`
  additionally **clears any stale sccache user environment variables**
  (`RUSTC_WRAPPER`, `SCCACHE_DIR`, `SCCACHE_CACHE_SIZE`, `SCCACHE_BASEDIRS`) it
  finds, so re-running it fully de-sccaches an already-provisioned Windows box —
  this is the step that actually stops the corrupt builds.
- **Delete the sccache-only helpers.** `scripts/sccache-on.ps1` and
  `scripts/sccache-on.sh` (the "cold/parallel regime" toggles whose only jobs
  were `CARGO_INCREMENTAL=0` + basedirs) are removed. `check-build-env.sh` drops
  its sccache probe and worktree-normalization checks.
- **Incremental stays on everywhere.** `CARGO_INCREMENTAL=0` existed only to make
  workspace crates sccache-cacheable across worktrees; with no cache it just
  disables a real speedup, so it is no longer advised for any regime.
- **Parallel agents no longer share a compiler cache.** Each worktree keeps its
  own `target/`, fast linker, and job budget, but the first-worktree-populates,
  every-other-worktree-hits trick is gone: a fresh worktree recompiles the dep
  graph from cold. This is the accepted cost.

## Consequences

- **Builds are correct again** — no cache layer that can serve a wrong object.
- **A large fragile surface is deleted** — no basedirs snapshot to resync after
  `git worktree add`, no global `RUSTC_WRAPPER`, no version floor, no
  platform-separator footgun, no "healthy `--show-stats` that lies" trap. The
  standing INC-0019 rule ("resync `SCCACHE_BASEDIRS` after every worktree add")
  is retired with the tool.
- **Cold/fresh-worktree/CI builds are slower** — they recompile Bevy + wgpu +
  deps from cold each time (minutes) instead of hitting a shared cache. The
  iterate loop (fast linker + dynamic linking + incremental) is unaffected; that
  is where the day-to-day time goes.
- **If cross-machine/cold throughput ever becomes the bottleneck again**, revisit
  with a *remote* cache backend (S3/GCS/Redis) evaluated for correctness first,
  or a different tool — not a return to the local-daemon + basedirs design this
  ADR removes. Any such return needs its own ADR and a correctness bar.

## Alternatives considered

- **Diagnose and fix the specific corruption, keep sccache.** Rejected: even a
  fixed cache carries the INC-0019 maintenance surface for a benefit that is
  marginal in this project's actual regime, and a cache that has already produced
  wrong builds has a low trust floor.
- **Make sccache opt-in (drop the always-on global default, keep it for the
  parallel/cloud regime).** A reasonable middle path, but it keeps the whole
  apparatus alive for a regime we rarely run, and "opt-in" caches drift back to
  silently-misactivated. Cleaner to remove now and re-add deliberately (remote
  backend) if the parallel regime becomes primary.
