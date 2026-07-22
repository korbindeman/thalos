#!/usr/bin/env bash
# Workspace-non-mutating readiness check for Linux, WSL2, and macOS agent
# environments. It performs a tiny compile/cache-hit probe in the system temp
# directory. Use --parallel after sourcing scripts/sccache-on.sh.
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
version_at_least() {
  local actual="${1%%[-+]*}" required="$2"
  local a_major=0 a_minor=0 a_patch=0 r_major=0 r_minor=0 r_patch=0
  IFS=. read -r a_major a_minor a_patch <<<"$actual"
  IFS=. read -r r_major r_minor r_patch <<<"$required"
  (( 10#${a_major:-0} > 10#${r_major:-0} )) && return 0
  (( 10#${a_major:-0} < 10#${r_major:-0} )) && return 1
  (( 10#${a_minor:-0} > 10#${r_minor:-0} )) && return 0
  (( 10#${a_minor:-0} < 10#${r_minor:-0} )) && return 1
  (( 10#${a_patch:-0} >= 10#${r_patch:-0} ))
}

os="$(uname -s)"
is_wsl=0
if [[ "$os" == "Linux" ]] && grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then is_wsl=1; fi

if [[ $is_wsl == 1 && "$repo_root" == /mnt/* ]]; then
  fail "WSL checkout is under /mnt; clone it into the Linux filesystem (for example ~/thalos)."
else
  ok "checkout filesystem: $repo_root"
fi

for command_name in cargo rustc just dx; do
  if have "$command_name"; then ok "$command_name: $(command -v "$command_name")"; else fail "$command_name is missing"; fi
done
if have dx; then
  dx_version="$(dx --version 2>/dev/null | awk '{print $2}')"
  [[ "$dx_version" == "0.7.9" ]] \
    && ok "dx version matches Subsecond: $dx_version" \
    || fail "dx 0.7.9 required; found ${dx_version:-unknown}"
fi

rust_host=""
if have rustc; then
  rust_host="$(rustc -vV | sed -n 's/^host: //p')"
  [[ -n "$rust_host" ]] && ok "Rust host: $rust_host" || fail "could not determine Rust host triple"
fi

sccache_bin=""
if [[ -f "$config" ]]; then
  ok "local Cargo config: $config"
  grep -Eq '^[[:space:]]*rustc-wrapper[[:space:]]*=' "$config" \
    && ok "Cargo rustc-wrapper configured" || fail "Cargo rustc-wrapper missing from $config"
  grep -Eq '^[[:space:]]*jobs[[:space:]]*=' "$config" \
    && ok "Cargo job budget configured" || fail "Cargo job budget missing from $config"
  sccache_bin="$(sed -n 's/^[[:space:]]*rustc-wrapper[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$config" | head -1)"
  if [[ "$os" == "Linux" && -n "$rust_host" ]]; then
    grep -Fq "[target.$rust_host]" "$config" \
      && ok "host-specific linker section configured" \
      || fail "missing [target.$rust_host] in $config"
  fi
else
  fail "$config is missing; run scripts/setup-build-env.sh"
fi
[[ -n "$sccache_bin" ]] || sccache_bin="$(command -v sccache 2>/dev/null || true)"
if [[ -n "$sccache_bin" && -x "$sccache_bin" ]]; then
  ok "sccache: $sccache_bin"
else
  fail "configured sccache binary is missing or not executable: ${sccache_bin:-none}"
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

if [[ -n "$sccache_bin" && -x "$sccache_bin" ]]; then
  sccache_version="$($sccache_bin --version | awk '{print $2}')"
  version_at_least "$sccache_version" "0.14.0" \
    && ok "sccache version: $sccache_version" \
    || fail "sccache >= 0.14.0 required; found $sccache_version"
  stats="$("$sccache_bin" --show-stats 2>&1)"
  status=$?
  if [[ $status -eq 0 ]]; then
    ok "sccache server responds"
    if grep -qi 'Base director' <<<"$stats"; then
      ok "sccache base-directory normalization is active"
      grep -Fq "$repo_root" <<<"$stats" \
        && ok "sccache normalization includes this worktree" \
        || fail "sccache base directories do not include $repo_root; rerun setup after creating worktrees"
    else
      fail "sccache did not report base directories; verify SCCACHE_BASEDIRS/version"
    fi
  else
    fail "sccache stats failed: $stats"
  fi
fi

# A responsive daemon is not enough: prove that a real Rust compilation can be
# cached and replayed. This catches stale sockets, old/incompatible wrappers,
# permissions, and broken remote-cache credentials before an agent takes work.
if [[ -n "$sccache_bin" && -x "$sccache_bin" ]] && have rustc; then
  tmp_base="${TMPDIR:-/tmp}"
  tmp_base="${tmp_base%/}"
  probe_root="$(mktemp -d "$tmp_base/thalos-build-env.XXXXXX")"
  cleanup_probe() {
    if [[ -n "${probe_root:-}" && "$probe_root" == "$tmp_base"/thalos-build-env.* ]]; then
      rm -rf -- "$probe_root"
    fi
  }
  trap cleanup_probe EXIT

  probe_source="$probe_root/cache_probe.rs"
  probe_out_dir="$probe_root/out"
  mkdir -p "$probe_out_dir"
  printf '%s\n' 'pub fn thalos_cache_probe() -> u32 { 42 }' > "$probe_source"
  before_stats="$("$sccache_bin" --show-stats 2>&1)"
  before_hits="$(awk '$1 == "Cache" && $2 == "hits" && NF == 3 { print $3; exit }' <<<"$before_stats")"
  before_hits="${before_hits:-0}"

  cache_probe=("$sccache_bin" "$(command -v rustc)" --crate-name thalos_cache_probe \
    --edition=2024 --crate-type lib --emit=dep-info,metadata,link -C opt-level=1 \
    -C metadata=thalos_build_env -C extra-filename=-thalos_build_env \
    --out-dir "$probe_out_dir" "$probe_source")
  if "${cache_probe[@]}" >/dev/null 2>&1; then
    find "$probe_out_dir" -mindepth 1 -maxdepth 1 -type f -delete
    if "${cache_probe[@]}" >/dev/null 2>&1; then
      after_stats="$("$sccache_bin" --show-stats 2>&1)"
      after_hits="$(awk '$1 == "Cache" && $2 == "hits" && NF == 3 { print $3; exit }' <<<"$after_stats")"
      after_hits="${after_hits:-0}"
      (( after_hits > before_hits )) \
        && ok "real Rust compile produced an sccache hit" \
        || fail "Rust probe compiled but did not produce an sccache hit"
    else
      fail "second Rust compile through sccache failed"
    fi
  else
    fail "Rust compile through sccache failed"
  fi

  if [[ "$os" == "Linux" ]] && have clang && have mold; then
    link_source="$probe_root/link_probe.rs"
    link_output="$probe_root/link_probe"
    printf '%s\n' 'fn main() {}' > "$link_source"
    if rustc "$link_source" -C "linker=$(command -v clang)" \
      -C link-arg=-fuse-ld=mold -o "$link_output" >/dev/null 2>&1; then
      ok "clang + mold can link the Rust host target"
    else
      fail "clang + mold cannot link the Rust host target"
    fi
  fi

  cleanup_probe
  trap - EXIT
fi

if [[ $require_parallel -eq 1 ]]; then
  [[ "${CARGO_INCREMENTAL:-}" == "0" ]] \
    && ok "parallel mode: CARGO_INCREMENTAL=0" \
    || fail "parallel mode requires: source scripts/sccache-on.sh"
  [[ -n "${SCCACHE_BASEDIRS:-}" ]] \
    && ok "parallel mode: SCCACHE_BASEDIRS is set" \
    || fail "parallel mode is missing SCCACHE_BASEDIRS"
  while IFS= read -r worktree_root; do
    [[ -f "$worktree_root/.cargo/config.toml" ]] \
      && ok "worktree config: $worktree_root" \
      || fail "worktree lacks .cargo/config.toml: $worktree_root"
  done < <(git -C "$repo_root" worktree list --porcelain | sed -n 's/^worktree //p')
fi

if [[ $failures -ne 0 ]]; then
  echo "NOT READY: $failures failure(s), $warnings warning(s)" >&2
  exit 1
fi
echo "READY: $warnings warning(s)"
