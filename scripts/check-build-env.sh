#!/usr/bin/env bash
# Workspace-non-mutating readiness check for Linux, WSL2, and macOS agent
# environments. Verifies the toolchain, the local Cargo config (fast linker +
# job budget), headless Vulkan, and — on Linux — that clang + mold can actually
# link. There is no compiler cache to probe (sccache removed,
# ADR-20260723T222214Z). Use --parallel to also check every worktree's config.
set -u

require_parallel=0
case "${1:-}" in
  "") ;;
  --parallel) require_parallel=1 ;;
  -h|--help)
    echo "Usage: bash scripts/check-build-env.sh [--parallel]"
    exit 0
    ;;
  *) echo "Unknown option '$1'" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="$repo_root/.cargo/config.toml"
failures=0
warnings=0

ok() { printf 'ok    %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*" >&2; warnings=$((warnings + 1)); }
fail() { printf 'FAIL  %s\n' "$*" >&2; failures=$((failures + 1)); }
have() { command -v "$1" >/dev/null 2>&1; }

os="$(uname -s)"
is_wsl=0
if [[ "$os" == "Linux" ]] && grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then is_wsl=1; fi

if [[ $is_wsl == 1 && "$repo_root" == /mnt/* ]]; then
  fail "WSL checkout is under /mnt; clone it into the Linux filesystem (for example ~/thalos)."
else
  ok "checkout filesystem: $repo_root"
fi

for command_name in cargo rustc just; do
  if have "$command_name"; then ok "$command_name: $(command -v "$command_name")"; else fail "$command_name is missing"; fi
done

rust_host=""
if have rustc; then
  rust_host="$(rustc -vV | sed -n 's/^host: //p')"
  [[ -n "$rust_host" ]] && ok "Rust host: $rust_host" || fail "could not determine Rust host triple"
fi

if [[ -f "$config" ]]; then
  ok "local Cargo config: $config"
  grep -Eq '^[[:space:]]*jobs[[:space:]]*=' "$config" \
    && ok "Cargo job budget configured: $(sed -n 's/^[[:space:]]*jobs[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/p' "$config" | head -1)" \
    || fail "Cargo job budget missing from $config"
  if [[ "$os" == "Linux" && -n "$rust_host" ]]; then
    grep -Fq "[target.$rust_host]" "$config" \
      && ok "host-specific linker section configured" \
      || fail "missing [target.$rust_host] in $config"
  fi
else
  fail "$config is missing; run scripts/setup-build-env.sh (or .ps1 on Windows)"
fi

if [[ "$os" == "Linux" ]]; then
  have clang && ok "clang: $(clang --version | head -1)" || fail "clang is missing"
  have mold && ok "mold: $(mold --version | head -1)" || fail "mold is missing"
  if have vulkaninfo && vulkaninfo --summary >/dev/null 2>&1; then
    ok "headless Vulkan is available"
  else
    fail "headless Vulkan is unavailable; capture agents need a working GPU ICD or lavapipe"
  fi
fi

# Prove the configured linker actually links, not just that the binaries exist:
# a broken clang/mold combination is a common silent misconfiguration.
if [[ "$os" == "Linux" ]] && have rustc && have clang && have mold; then
  tmp_base="${TMPDIR:-/tmp}"
  tmp_base="${tmp_base%/}"
  probe_root="$(mktemp -d "$tmp_base/thalos-build-env.XXXXXX")"
  cleanup_probe() {
    if [[ -n "${probe_root:-}" && "$probe_root" == "$tmp_base"/thalos-build-env.* ]]; then
      rm -rf -- "$probe_root"
    fi
  }
  trap cleanup_probe EXIT

  link_source="$probe_root/link_probe.rs"
  link_output="$probe_root/link_probe"
  printf '%s\n' 'fn main() {}' > "$link_source"
  if rustc "$link_source" -C "linker=$(command -v clang)" \
    -C link-arg=-fuse-ld=mold -o "$link_output" >/dev/null 2>&1; then
    ok "clang + mold can link the Rust host target"
  else
    fail "clang + mold cannot link the Rust host target"
  fi

  cleanup_probe
  trap - EXIT
fi

if [[ $require_parallel -eq 1 ]]; then
  # Cargo discovers config by walking UP from the cwd, so a worktree nested
  # inside the checkout inherits the repo-root config for free. Only worktrees
  # outside it (C:/tmp/..., ~/.codex/worktrees/...) need their own -- those are
  # the ones that would otherwise build with no linker and no job budget.
  while IFS= read -r worktree_root; do
    [[ -n "$worktree_root" ]] || continue
    if [[ -f "$worktree_root/.cargo/config.toml" ]]; then
      ok "worktree config: $worktree_root"
    elif [[ "$worktree_root" == "$repo_root"/* ]]; then
      ok "worktree inherits repo-root config: $worktree_root"
    else
      fail "worktree outside the checkout has no .cargo/config.toml: $worktree_root"
      echo "      no fast linker and no job budget there; rerun setup with --all-worktrees/-AllWorktrees" >&2
    fi
  done < <(git -C "$repo_root" worktree list --porcelain | sed -n 's/^worktree //p')
fi

if [[ $failures -ne 0 ]]; then
  echo "NOT READY: $failures failure(s), $warnings warning(s)" >&2
  exit 1
fi
echo "READY: $warnings warning(s)"
