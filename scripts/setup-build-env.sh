#!/usr/bin/env bash
# Provision a fast Thalos build environment on Linux, WSL2, or macOS.
#
# Linux/WSL receive clang + mold, the agent-facing CLI tool (just), and a
# host-triple-specific Cargo config (fast linker + bounded job budget). Machine-
# local config stays gitignored. There is no compiler cache: sccache was removed
# (ADR-20260723T222214Z-abandon-sccache). Use --all-worktrees to write the same
# config into every worktree created outside the checkout.
#
# Usage:
#   scripts/setup-build-env.sh
#   scripts/setup-build-env.sh --agents 4 --all-worktrees
#   scripts/setup-build-env.sh --force
set -euo pipefail

FORCE=0
ALL_WORKTREES=0
AGENT_SLOTS="${THALOS_AGENT_SLOTS:-1}"
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

# Worktrees inside the checkout inherit the repo-root config via Cargo's upward
# discovery; only worktrees created outside the tree need their own copy, which
# --all-worktrees writes.
worktree_roots=()
while IFS= read -r root; do worktree_roots+=("$root"); done \
  < <(git -C "$repo_root" worktree list --porcelain | sed -n 's/^worktree //p')
(( ${#worktree_roots[@]} == 0 )) && worktree_roots=("$repo_root")

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

# --- Rust + agent-facing commands -------------------------------------------
if ! have cargo || ! have rustc; then
  echo "!! cargo/rustc not found. Install rustup first: https://rustup.rs" >&2
  exit 1
fi
rust_host="$(rustc -vV | sed -n 's/^host: //p')"
[[ -n "$rust_host" ]] || { echo "Could not determine the rustc host triple." >&2; exit 1; }
echo "==> rustc: $(rustc --version) ($rust_host)"

if ! have just; then
  echo "==> Installing just"
  if have cargo-binstall; then cargo binstall -y just; else cargo install just --locked; fi
  have just || { echo "just installation failed." >&2; exit 1; }
fi

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

# The persistent capture host needs no extra tooling: it is a plain detached
# `cargo run` on the same dynamic `dev-renderer` fingerprint as everything else
# (ADR-20260724T153619Z-retire-hotpatch-single-stable-capture-lane). The former
# dioxus-cli/Subsecond install lived here; nothing replaces it.

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
    echo "# Local, gitignored, per-machine. Fast linker + bounded job budget; no"
    echo "# compiler cache (sccache removed, ADR-20260723T222214Z)."
    echo "# See docs/development/build_speed.md."
    echo
    echo "[build]"
    echo "jobs = $cargo_jobs # $logical_cpus logical CPUs / $AGENT_SLOTS expected concurrent agents"
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

# --- verification summary ----------------------------------------------------
echo
echo "==> Provisioned $platform build environment"
echo "   host:     $rust_host"
echo "   jobs:     $cargo_jobs per Cargo process ($AGENT_SLOTS agent slots)"
[[ -n "$clang_bin" ]] && echo "   linker:   $clang_bin + mold"
if [[ "$platform" == "linux" || "$platform" == "wsl" ]]; then
  if have vulkaninfo && vulkaninfo --summary >/dev/null 2>&1; then
    echo "   Vulkan:   available"
  else
    echo "   Vulkan:   NOT READY — install/repair a Vulkan ICD before headless capture" >&2
  fi
fi
echo
echo "Next:"
echo "  bash scripts/check-build-env.sh --parallel"
echo "  just check"
echo "  just screenshot hub"
