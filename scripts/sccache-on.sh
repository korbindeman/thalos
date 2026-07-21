#!/usr/bin/env bash
# Turn sccache ON for the CURRENT shell (Cold/parallel regime): cache the whole
# dependency graph across worktrees/branches/clean builds. Disables incremental
# (sccache cannot cache incremental crates) — do NOT use this for the everyday
# single-machine edit loop, where incremental is the win. See docs/build_speed.md §5.
#
# Usage:  source scripts/sccache-on.sh      (must be sourced, not executed)

if ! command -v sccache >/dev/null 2>&1; then
  echo "sccache not installed — run scripts/setup-build-env.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

export RUSTC_WRAPPER=sccache
export CARGO_INCREMENTAL=0
export SCCACHE_DIR="${SCCACHE_DIR:-$HOME/.cache/sccache}"
export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-50G}"

echo "sccache ON  (RUSTC_WRAPPER=sccache, CARGO_INCREMENTAL=0)"
echo "  SCCACHE_DIR=$SCCACHE_DIR  SCCACHE_CACHE_SIZE=$SCCACHE_CACHE_SIZE"
echo "  stats: sccache --show-stats   |   off: unset RUSTC_WRAPPER CARGO_INCREMENTAL"
