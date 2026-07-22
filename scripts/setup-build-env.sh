#!/usr/bin/env bash
# Provision a fast Thalos build environment on Linux, WSL2, or macOS.
#
# Linux/WSL receive clang + mold, sccache, the agent-facing CLI tools, and a
# host-triple-specific Cargo config. Machine-local config stays gitignored.
# Provision the complete worktree set first, then use --all-worktrees once so
# every agent receives the same job budget and cache normalization roots.
#
# Usage:
#   scripts/setup-build-env.sh
#   scripts/setup-build-env.sh --agents 4 --all-worktrees
#   scripts/setup-build-env.sh --force
set -euo pipefail

FORCE=0
ALL_WORKTREES=0
AGENT_SLOTS="${THALOS_AGENT_SLOTS:-2}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --all-worktrees) ALL_WORKTREES=1; shift ;;
    --agents)
      [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || { echo "--agents requires a positive integer" >&2; exit 2; }
      AGENT_SLOTS="$2"; shift 2 ;;
    -h|--help)
      printf '%s\n' \
        'Usage: scripts/setup-build-env.sh [--agents N] [--all-worktrees] [--force]' \
        '  --agents N         divide logical CPUs across N simultaneous Cargo processes' \
        '  --all-worktrees    write matching local config into every current worktree' \
        '  --force            back up and replace existing local Cargo config'
      exit 0
      ;;
    *) echo "Unknown option '$1'" >&2; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# sccache splits SCCACHE_BASEDIRS with the platform separator. This script only
# supports Linux/macOS (see the OS guard below), but keep the selection explicit
# so relaxing that guard cannot silently produce drive-letter-shredded roots.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) path_sep=';' ;;
  *) path_sep=':' ;;
esac

# Longest-first: worktrees nest inside the repo root, and a shorter parent
# matching first would normalize their paths differently than the main checkout,
# so the two could never share a cache entry.
worktree_roots=()
while IFS= read -r root; do worktree_roots+=("$root"); done \
  < <(git -C "$repo_root" worktree list --porcelain | sed -n 's/^worktree //p' \
      | awk '{ print length, $0 }' | sort -rn | cut -d' ' -f2-)
(( ${#worktree_roots[@]} == 0 )) && worktree_roots=("$repo_root")
scc_basedirs="$(IFS="$path_sep"; echo "${worktree_roots[*]}")"

# --- platform detection ------------------------------------------------------
os="$(uname -s)"
is_wsl=0
if [[ "$os" == "Linux" ]] && grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
  is_wsl=1
fi
case "$os" in
  Linux)  platform=$([[ $is_wsl == 1 ]] && echo "wsl" || echo "linux") ;;
  Darwin) platform="macos" ;;
  *) echo "Unsupported OS '$os' — use scripts/setup-build-env.ps1 on Windows." >&2; exit 1 ;;
esac
echo "==> Platform: $platform"

if [[ $is_wsl == 1 && "$repo_root" == /mnt/* ]]; then
  echo "!! WSL checkout is under /mnt; builds will cross the Windows filesystem boundary." >&2
  echo "   Clone into the Linux filesystem (for example ~/thalos) before parallel work." >&2
fi

logical_cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
cargo_jobs=$((logical_cpus / AGENT_SLOTS))
(( cargo_jobs < 1 )) && cargo_jobs=1

have() { command -v "$1" >/dev/null 2>&1; }

# --- package-manager helpers -------------------------------------------------
as_root() {
  if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    "$@"
  elif have sudo; then
    sudo "$@"
  else
    echo "Need root privileges for: $* (run as root or install sudo)." >&2
    return 1
  fi
}

package_manager=""
apt_updated=0
if [[ "$platform" == "macos" ]]; then
  have brew && package_manager="brew"
elif have apt-get; then
  package_manager="apt"
elif have dnf; then
  package_manager="dnf"
elif have pacman; then
  package_manager="pacman"
elif have zypper; then
  package_manager="zypper"
fi

pkg_install() {
  case "$package_manager" in
    brew) brew install "$@" ;;
    apt)
      if [[ $apt_updated -eq 0 ]]; then as_root apt-get update; apt_updated=1; fi
      as_root apt-get install -y "$@"
      ;;
    dnf) as_root dnf install -y "$@" ;;
    pacman) as_root pacman -S --needed --noconfirm "$@" ;;
    zypper) as_root zypper --non-interactive install "$@" ;;
    *) echo "No supported package manager found; install $* manually." >&2; return 1 ;;
  esac
}

ensure_system_command() {
  local command_name="$1" package_name="$2"
  if ! have "$command_name"; then
    echo "==> Installing $package_name"
    pkg_install "$package_name"
  fi
  have "$command_name" || { echo "$command_name is still unavailable after installing $package_name." >&2; exit 1; }
}

install_cargo_binary() {
  local binary="$1" crate="$2"
  have "$binary" && return
  echo "==> Installing $crate"
  if have cargo-binstall; then
    cargo binstall -y "$crate"
  else
    echo "   (cargo-binstall unavailable; compiling $crate once)"
    cargo install "$crate" --locked
  fi
  have "$binary" || { echo "$binary is still unavailable after installing $crate." >&2; exit 1; }
}

resolve_installed_binary() {
  local binary="$1"
  local cargo_root="${CARGO_INSTALL_ROOT:-${CARGO_HOME:-$HOME/.cargo}}"
  local cargo_candidate="$cargo_root/bin/$binary"
  if [[ -x "$cargo_candidate" ]]; then
    printf '%s' "$cargo_candidate"
  else
    command -v "$binary" 2>/dev/null || true
  fi
}

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

# --- Rust + agent-facing commands -------------------------------------------
if ! have cargo || ! have rustc; then
  echo "!! cargo/rustc not found. Install rustup first: https://rustup.rs" >&2
  exit 1
fi
rust_host="$(rustc -vV | sed -n 's/^host: //p')"
[[ -n "$rust_host" ]] || { echo "Could not determine the rustc host triple." >&2; exit 1; }
echo "==> rustc: $(rustc --version) ($rust_host)"

install_cargo_binary just just

# --- fast linker -------------------------------------------------------------
if [[ "$platform" == "linux" || "$platform" == "wsl" ]]; then
  ensure_system_command clang clang
  ensure_system_command mold mold
  clang_bin="$(command -v clang)"
else
  clang_bin=""
  echo "==> macOS: using the default Apple linker."
fi

# --- headless Vulkan ---------------------------------------------------------
if [[ "$platform" == "linux" || "$platform" == "wsl" ]]; then
  if ! have vulkaninfo; then
    echo "==> Installing Vulkan diagnostics"
    pkg_install vulkan-tools || true
  fi

  if ! have vulkaninfo || ! vulkaninfo --summary >/dev/null 2>&1; then
    echo "==> No working Vulkan ICD detected; attempting a Mesa software/WSL fallback"
    case "$package_manager" in
      apt|dnf) pkg_install mesa-vulkan-drivers || true ;;
      pacman) pkg_install vulkan-swrast || true ;;
      *) echo "   Install the GPU vendor Vulkan driver or a lavapipe package manually." >&2 ;;
    esac
  fi
fi

# --- sccache + persistent capture controller --------------------------------
if ! have sccache; then
  if [[ "$platform" == "macos" && "$package_manager" == "brew" ]]; then
    pkg_install sccache
  elif have cargo-binstall; then
    cargo binstall -y sccache
  elif [[ "$package_manager" == "apt" ]] && apt-cache show sccache >/dev/null 2>&1; then
    pkg_install sccache
  else
    echo "==> Installing sccache from crates.io (one-time source build)"
    cargo install sccache --locked
  fi
fi
sccache_bin="$(resolve_installed_binary sccache)"
[[ -x "$sccache_bin" ]] || { echo "sccache installation failed." >&2; exit 1; }
sccache_version="$($sccache_bin --version | awk '{print $2}')"
if ! version_at_least "$sccache_version" "0.14.0"; then
  echo "==> sccache $sccache_version is too old for SCCACHE_BASEDIRS; upgrading to >= 0.14.0"
  if have cargo-binstall; then
    cargo binstall -y --force sccache
  else
    cargo install --locked --force sccache
  fi
  sccache_bin="$(resolve_installed_binary sccache)"
  sccache_version="$($sccache_bin --version | awk '{print $2}')"
fi
version_at_least "$sccache_version" "0.14.0" \
  || { echo "sccache >= 0.14.0 is required; found $sccache_version." >&2; exit 1; }
echo "==> sccache: $($sccache_bin --version)"

# `just screenshot` uses dx/Subsecond. Keep the CLI on the same release as the
# locked Subsecond runtime; a future CLI is not assumed wire-compatible.
dx_version="0.7.9"
installed_dx_version=""
have dx && installed_dx_version="$(dx --version 2>/dev/null | awk '{print $2}')"
if [[ "$installed_dx_version" != "$dx_version" ]]; then
  echo "==> Installing dioxus-cli $dx_version (found: ${installed_dx_version:-none})"
  if have cargo-binstall; then
    cargo binstall -y --force "dioxus-cli@$dx_version"
  else
    cargo install dioxus-cli --version "$dx_version" --locked --force
  fi
fi
have dx || { echo "dx installation failed." >&2; exit 1; }

sccache_dir="${SCCACHE_DIR:-$HOME/.cache/sccache}"
sccache_size="${SCCACHE_CACHE_SIZE:-50G}"

toml_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '%s' "$value"
}

write_config() {
  local target_root="$1"
  local cargo_dir="$target_root/.cargo"
  local cfg="$cargo_dir/config.toml"
  mkdir -p "$cargo_dir"

  if [[ -f "$cfg" && $FORCE -eq 0 ]] \
    && ! grep -Fq '# Generated by scripts/setup-build-env.sh' "$cfg"; then
    echo "==> $cfg is custom — leaving it (pass --force to back up and replace)."
    return
  fi
  if [[ -f "$cfg" && $FORCE -eq 1 ]]; then
    cp "$cfg" "$cfg.bak.$(date +%s 2>/dev/null || echo prev)"
    echo "   Backed up $cfg"
  fi

  {
    echo "# Generated by scripts/setup-build-env.sh for platform: $platform"
    echo "# Local, gitignored, per-machine. See docs/build_speed.md."
    echo
    echo "[env]"
    echo "SCCACHE_DIR = \"$(toml_escape "$sccache_dir")\""
    echo "SCCACHE_CACHE_SIZE = \"$(toml_escape "$sccache_size")\""
    echo "SCCACHE_BASEDIRS = \"$(toml_escape "$scc_basedirs")\""
    echo
    echo "[build]"
    echo "jobs = $cargo_jobs # $logical_cpus logical CPUs / $AGENT_SLOTS expected concurrent agents"
    echo "rustc-wrapper = \"$(toml_escape "$sccache_bin")\""
    echo
    if [[ "$platform" == "linux" || "$platform" == "wsl" ]]; then
      echo "[target.$rust_host]"
      echo "linker = \"$(toml_escape "$clang_bin")\""
      echo "rustflags = [\"-C\", \"link-arg=-fuse-ld=mold\"]"
      echo
    fi
  } > "$cfg"
  echo "==> Wrote $cfg"
}

if [[ $ALL_WORKTREES -eq 1 ]]; then
  for root in "${worktree_roots[@]}"; do write_config "$root"; done
else
  write_config "$repo_root"
fi

# SCCACHE_BASEDIRS is server configuration. Setup is the safe provisioning
# boundary at which to refresh it; refuse to interrupt active compilers.
if pgrep -x cargo >/dev/null 2>&1 || pgrep -x rustc >/dev/null 2>&1; then
  echo "!! Cargo/rustc is active; not restarting sccache. Restart it after builds finish." >&2
else
  env SCCACHE_DIR="$sccache_dir" SCCACHE_CACHE_SIZE="$sccache_size" \
    SCCACHE_BASEDIRS="$scc_basedirs" "$sccache_bin" --stop-server >/dev/null 2>&1 || true
  env SCCACHE_DIR="$sccache_dir" SCCACHE_CACHE_SIZE="$sccache_size" \
    SCCACHE_BASEDIRS="$scc_basedirs" "$sccache_bin" --start-server >/dev/null
fi

# --- verification summary ----------------------------------------------------
echo
echo "==> Provisioned $platform build environment"
echo "   host:     $rust_host"
echo "   jobs:     $cargo_jobs per Cargo process ($AGENT_SLOTS agent slots)"
[[ -n "$clang_bin" ]] && echo "   linker:   $clang_bin + mold"
echo "   sccache:  $sccache_bin ($sccache_dir, $sccache_size)"
if [[ "$platform" == "linux" || "$platform" == "wsl" ]]; then
  if have vulkaninfo && vulkaninfo --summary >/dev/null 2>&1; then
    echo "   Vulkan:   available"
  else
    echo "   Vulkan:   NOT READY — install/repair a Vulkan ICD before headless capture" >&2
  fi
fi
echo
echo "Next:"
echo "  source scripts/sccache-on.sh"
echo "  bash scripts/check-build-env.sh --parallel"
echo "  just check"
echo "  just screenshot hub"
