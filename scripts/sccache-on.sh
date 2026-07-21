#!/usr/bin/env bash
# Enter the cold/parallel cache regime in the CURRENT shell. The generated Cargo
# config already uses sccache for ordinary non-incremental dependencies; this
# additionally disables incremental so workspace crates can be shared across
# worktrees. See docs/build_speed.md §5.
#
# Usage:  source scripts/sccache-on.sh      (must be sourced, not executed)

if ! command -v sccache >/dev/null 2>&1; then
  echo "sccache not installed — run scripts/setup-build-env.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

export CARGO_INCREMENTAL=0
export SCCACHE_DIR="${SCCACHE_DIR:-$HOME/.cache/sccache}"
export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-50G}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
worktree_roots=()
while IFS= read -r root; do worktree_roots+=("$root"); done \
  < <(git -C "$repo_root" worktree list --porcelain | sed -n 's/^worktree //p')
(( ${#worktree_roots[@]} == 0 )) && worktree_roots=("$repo_root")
export SCCACHE_BASEDIRS="$(IFS=:; echo "${worktree_roots[*]}")"

echo "parallel cache mode ON  (CARGO_INCREMENTAL=0)"
echo "  SCCACHE_DIR=$SCCACHE_DIR  SCCACHE_CACHE_SIZE=$SCCACHE_CACHE_SIZE"
echo "  SCCACHE_BASEDIRS=$SCCACHE_BASEDIRS"
echo "  stats: sccache --show-stats   |   return to iterate mode: unset CARGO_INCREMENTAL"
