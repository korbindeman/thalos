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

# sccache splits SCCACHE_BASEDIRS with the PLATFORM separator. Under git bash the
# roots are Windows paths ("C:/Users/..."), so joining with ':' shreds every
# drive letter into its own bogus root and normalization silently does nothing.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) path_sep=';' ;;
  *) path_sep=':' ;;
esac

# Longest-first: worktrees nest inside the repo root (.claude/worktrees/*), and a
# shorter parent matching first would rewrite paths differently than the main
# checkout does -- the two could then never share a cache entry.
worktree_roots=()
while IFS= read -r root; do worktree_roots+=("$root"); done \
  < <(git -C "$repo_root" worktree list --porcelain | sed -n 's/^worktree //p' \
      | awk '{ print length, $0 }' | sort -rn | cut -d' ' -f2-)
(( ${#worktree_roots[@]} == 0 )) && worktree_roots=("$repo_root")
export SCCACHE_BASEDIRS="$(IFS="$path_sep"; echo "${worktree_roots[*]}")"

# Activation must survive leaving this directory: a repo-local
# build.rustc-wrapper is invisible to worktrees outside the checkout.
export RUSTC_WRAPPER="$sccache_bin"
export THALOS_SCCACHE_BIN="$sccache_bin"

# Hosted agent runners may reap daemons started by a provisioning process. Make
# activation self-healing without disturbing a healthy shared server.
if ! "$THALOS_SCCACHE_BIN" --show-stats >/dev/null 2>&1; then
  "$THALOS_SCCACHE_BIN" --start-server >/dev/null
fi

# SCCACHE_BASEDIRS is read by the SERVER at startup, so exporting it into this
# shell does nothing for an already-running daemon. A worktree missing from the
# live server's set hashes absolute paths and can never hit another worktree's
# cache -- the failure is invisible, so say it out loud.
live_basedirs="$("$THALOS_SCCACHE_BIN" --show-stats 2>/dev/null | sed -n 's/^Base directories[[:space:]]*//p')"
if [[ -n "$live_basedirs" ]] \
  && ! grep -qiF "$(printf '%s' "$repo_root" | tr 'A-Z' 'a-z')" <<<"$(printf '%s' "$live_basedirs" | tr 'A-Z' 'a-z')"; then
  echo "!! sccache server does NOT normalize this worktree: $repo_root" >&2
  echo "   Cross-worktree cache hits are impossible until the server is restarted." >&2
  if pgrep -x cargo >/dev/null 2>&1 || pgrep -x rustc >/dev/null 2>&1; then
    echo "   Cargo/rustc active - restart it once builds finish:" >&2
  else
    "$THALOS_SCCACHE_BIN" --stop-server >/dev/null 2>&1 || true
    "$THALOS_SCCACHE_BIN" --start-server >/dev/null
    echo "   Restarted the server with the current worktree set."
  fi
fi

echo "parallel cache mode ON  (CARGO_INCREMENTAL=0)"
echo "  RUSTC_WRAPPER=$RUSTC_WRAPPER"
echo "  SCCACHE_DIR=$SCCACHE_DIR  SCCACHE_CACHE_SIZE=$SCCACHE_CACHE_SIZE"
echo "  SCCACHE_BASEDIRS=$SCCACHE_BASEDIRS"
echo "  stats: \"$THALOS_SCCACHE_BIN\" --show-stats"
echo "  return to iterate mode: unset CARGO_INCREMENTAL RUSTC_WRAPPER"
