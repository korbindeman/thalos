#!/usr/bin/env bash
# Enter the cold/parallel cache regime in the CURRENT shell. The generated Cargo
# config already uses sccache; this additionally disables incremental output so
# workspace crates can be shared across worktrees and compatible Linux boxes.
#
# Usage: source scripts/sccache-on.sh

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cargo_config="$repo_root/.cargo/config.toml"

sccache_bin=""
if [[ -f "$cargo_config" ]]; then
  sccache_bin="$(sed -n 's/^[[:space:]]*rustc-wrapper[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$cargo_config" | head -1)"
fi
[[ -n "$sccache_bin" ]] || sccache_bin="$(command -v sccache 2>/dev/null || true)"
if [[ -z "$sccache_bin" || ! -x "$sccache_bin" ]]; then
  echo "sccache not found — run scripts/setup-build-env.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

export CARGO_INCREMENTAL=0
export SCCACHE_DIR="${SCCACHE_DIR:-$HOME/.cache/sccache}"
export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-50G}"
worktree_roots=()
while IFS= read -r root; do worktree_roots+=("$root"); done \
  < <(git -C "$repo_root" worktree list --porcelain | sed -n 's/^worktree //p')
(( ${#worktree_roots[@]} == 0 )) && worktree_roots=("$repo_root")
export SCCACHE_BASEDIRS="$(IFS=:; echo "${worktree_roots[*]}")"
export THALOS_SCCACHE_BIN="$sccache_bin"

# Hosted agent runners may reap daemons started by a provisioning process. Make
# activation self-healing without disturbing a healthy shared server.
if ! "$THALOS_SCCACHE_BIN" --show-stats >/dev/null 2>&1; then
  "$THALOS_SCCACHE_BIN" --start-server >/dev/null
fi

echo "parallel cache mode ON  (CARGO_INCREMENTAL=0)"
echo "  SCCACHE_DIR=$SCCACHE_DIR  SCCACHE_CACHE_SIZE=$SCCACHE_CACHE_SIZE"
echo "  SCCACHE_BASEDIRS=$SCCACHE_BASEDIRS"
echo "  stats: \"$THALOS_SCCACHE_BIN\" --show-stats"
echo "  return to iterate mode: unset CARGO_INCREMENTAL"
