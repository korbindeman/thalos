# ADR-20260802T232314Z: Share release dependency artifacts through the default-branch cache

**Status:** accepted
**Date:** 2026-08-02
**Related:** ADR-20260723T222214Z-abandon-sccache

## Context

The first successful distribution run spent 37 minutes 19 seconds compiling
Windows and 21 minutes 41 seconds compiling macOS. Checkout, packaging, and
upload together took well under a minute per platform; compilation was the
bottleneck.

The workflow cached only Cargo registry downloads: roughly 72 MB on Windows and
56 MB on macOS. The expensive `target/<triple>/release` dependency graph was
discarded after every run. Adding that directory to the existing tag workflow
would not solve the next release: GitHub Actions caches are scoped to the
current branch or tag plus the default branch, so `v0.1.1` cannot restore a
cache written by `v0.1.0`.

Thalos also cannot use sccache as the shortcut. It was removed after producing
corrupt builds and carrying a large silent-activation surface
(ADR-20260723T222214Z). Reintroducing a rustc wrapper for CI would reopen the
correctness decision that ADR closed.

## Decision

Split the tag event from the expensive release build:

1. `.github/workflows/dispatch-game-release.yml` runs for `v*` tags and uses its
   narrowly scoped `actions: write` token to dispatch **Build game** on `main`.
   It passes both the full tag ref and the tag's commit SHA.
2. `.github/workflows/build-game.yml` checks out the supplied tag, resolves
   `HEAD`, and rejects the run unless it exactly matches the supplied full SHA.
   Every build and packaging job checks out that verified revision, and package
   provenance records it instead of the dispatch run's `github.sha`.
3. Windows and macOS use `Swatinem/rust-cache` after the pinned toolchain is
   installed. Its key includes the rustc identity and compiler environment; an
   explicit suffix separates target, Cargo features, and default-feature mode.
   The action hashes manifests, lockfiles, toolchain files, and Cargo config.
4. Cache only Cargo downloads and dependency build artifacts. Workspace crates,
   executables, and incremental artifacts are pruned before save. A cache miss
   remains a normal locked Cargo build.
5. Save only from `refs/heads/main`. Manual builds on another ref may restore
   the default-branch cache but cannot create a competing cache lineage.

This is an artifact cache managed and validated by Cargo, not a compiler cache:
there is no rustc wrapper, daemon, or object-level lookup. The sccache ban
remains intact.

## Consequences

- The first default-branch-scoped release build is cold. Later releases can
  reuse the unchanged Bevy/wgpu dependency graph and rebuild only invalidated
  dependencies plus Thalos workspace crates.
- A tag produces a small **Dispatch game release** run and a separate
  **Build game** run. This indirection is required for a cache scope reusable by
  future tags; collapsing the build back into the tag run silently defeats it.
- GitHub may evict an inactive cache or enforce the repository storage limit.
  Cache availability is an optimization, never a correctness dependency.
- The cache action reports its exact-hit result in each job summary. Its own
  detailed key and cleanup logs remain the diagnostic surface for partial hits
  and unexpected rebuilds.
- `actions: write` exists only on the tag dispatcher. Build jobs remain
  `contents: read`, and only the final publishing job receives
  `contents: write`.

## Alternatives considered

- **Cache the raw target directory in each tag run.** Rejected: caches remain
  isolated per tag, so this only speeds reruns of the same release; caching the
  whole directory also retains large workspace and incremental artifacts.
- **Warm a release cache on every push to main.** Rejected: it pays the full
  release build on ordinary development pushes and moves rather than removes
  the cost.
- **Upload target as a cross-run workflow artifact.** Rejected: locating the
  previous run and managing large retained archives duplicates a cache service
  badly.
- **Restore sccache with GitHub or remote storage.** Rejected by
  ADR-20260723T222214Z unless new correctness evidence justifies superseding it.
