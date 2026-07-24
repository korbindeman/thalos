<#
    Set up a high-speed Thalos build environment on native Windows.
    Verifies rust-lld, writes the local rust-lld + job-budget Cargo config, and
    (when elevated) offers Windows Defender exclusions for the build dirs.

    There is no compiler cache: sccache was removed (build corruption + a large
    silent-misactivation surface for a marginal solo-loop benefit) --
    ADR-20260723T222214Z-abandon-sccache. This script therefore also CLEARS any
    stale sccache user environment variables left by an earlier provisioning, so
    re-running it fully de-sccaches an already-set-up box. That is the step that
    actually stops the corrupt builds.

    Full rationale + the agent workflow: docs/development/build_speed.md

    Usage (from the repo root, PowerShell):
        scripts\setup-build-env.ps1
        scripts\setup-build-env.ps1 -Force          # overwrite existing .cargo\config.toml (backs it up)
        scripts\setup-build-env.ps1 -AgentSlots 4   # divide CPUs across 4 concurrent Cargo processes
        scripts\setup-build-env.ps1 -AllWorktrees   # write matching config into every worktree
#>
param(
    [switch]$Force,
    [switch]$AllWorktrees,
    [ValidateRange(1, 64)]
    [int]$AgentSlots = 1
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$logicalCpus = [Environment]::ProcessorCount
$cargoJobs = [Math]::Max(1, [Math]::Floor($logicalCpus / $AgentSlots))

# Worktrees inside the checkout inherit the repo-root .cargo\config.toml via
# Cargo's upward config discovery; worktrees created OUTSIDE the tree
# (C:\tmp, .codex\worktrees) find no config and need their own -- that is what
# -AllWorktrees writes. (There is no cache normalization to manage anymore.)
$worktreeRoots = @(
    git -C $repoRoot worktree list --porcelain |
        Where-Object { $_ -like 'worktree *' } |
        ForEach-Object { ($_.Substring('worktree '.Length)) -replace '\\', '/' }
)
if ($worktreeRoots.Count -eq 0) { $worktreeRoots = @(($repoRoot -replace '\\', '/')) }

function Have($name) { $null -ne (Get-Command $name -ErrorAction SilentlyContinue) }

Write-Host "==> Platform: windows (native, msvc)"
if (-not (Have 'cargo')) { throw "cargo not found. Install rustup first: https://rustup.rs" }

# --- retire any stale sccache activation --------------------------------------
# sccache is gone (ADR-20260723T222214Z). A previously-provisioned box still has
# RUSTC_WRAPPER + SCCACHE_* as USER env vars, which would keep routing every
# rustc through the (removed, possibly corrupting) cache. Clear them here so the
# migration is just "re-run this script".
$staleSccacheVars = @('RUSTC_WRAPPER', 'SCCACHE_DIR', 'SCCACHE_CACHE_SIZE', 'SCCACHE_BASEDIRS')
$cleared = @()
foreach ($name in $staleSccacheVars) {
    $val = [Environment]::GetEnvironmentVariable($name, 'User')
    if ($null -ne $val) {
        # Only touch RUSTC_WRAPPER if it actually points at sccache; leave a
        # user-configured non-sccache wrapper alone.
        if ($name -eq 'RUSTC_WRAPPER' -and $val -notmatch '(?i)sccache') { continue }
        [Environment]::SetEnvironmentVariable($name, $null, 'User')
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        $cleared += $name
    }
}
if ($cleared.Count -gt 0) {
    Write-Host "==> Cleared stale sccache user env var(s): $($cleared -join ', ')"
    Write-Host "    (open a NEW terminal before your next build for this to take effect)"
}

Write-Host "==> rustc: $(rustc --version)"

# --- rust-lld (fast linker), provisioned as an lld-link.exe shim -------------
# The configured linker is a copy of rust-lld.exe NAMED lld-link.exe: lld
# dispatches its driver flavor on argv[0], so this name selects the MSVC
# driver with no extra flags. rustc accepts either spelling, but any tool
# that drives the configured linker directly with raw MSVC args dies on bare
# rust-lld.exe with "lld is a generic driver" (INC-20260724T030400Z bit the
# since-retired dx lane exactly that way) — the shim is the safe spelling.
$tc = (rustc -vV | Select-String 'host:').ToString().Split(':')[1].Trim()
$sysroot = (rustc --print sysroot).Trim()
$lld = Join-Path $sysroot "lib\rustlib\$tc\bin\rust-lld.exe"
$shim = Join-Path $env:USERPROFILE '.cargo\shims\lld-link.exe'
if (Test-Path $lld) {
    Write-Host "==> rust-lld: $lld"
    New-Item -ItemType Directory -Force -Path (Split-Path $shim) | Out-Null
    $stale = -not (Test-Path $shim) -or
        ((Get-Item $lld).LastWriteTimeUtc -gt (Get-Item $shim).LastWriteTimeUtc)
    if ($stale) { Copy-Item $lld $shim -Force }
    Write-Host "==> lld-link shim: $shim"
} else {
    Write-Warning "rust-lld.exe not found at $lld"
    Write-Host   "   Install the LLVM tools:  rustup component add llvm-tools"
}

# --- write local .cargo\config.toml ------------------------------------------
# Forward slashes: Cargo fingerprints the linker path, so a stable spelling
# avoids a spurious full rebuild whenever this is re-provisioned.
$lldToml = ($shim -replace '\\', '/')

function Write-CargoConfig($targetRoot) {
    $cfg = Join-Path $targetRoot '.cargo\config.toml'
    New-Item -ItemType Directory -Force -Path (Split-Path $cfg) | Out-Null
    if ((Test-Path $cfg) -and -not $Force) {
        $existing = Get-Content $cfg -Raw -ErrorAction SilentlyContinue
        if ($existing -notmatch 'Generated by scripts\\setup-build-env\.ps1') {
            Write-Host "==> $cfg is custom - leaving it (use -Force to back up and replace)."
            return
        }
    }
    if (Test-Path $cfg) { Copy-Item $cfg "$cfg.bak"; Write-Host "   (backed up existing config to $cfg.bak)" }
@"
# Generated by scripts\setup-build-env.ps1 (Windows). Local, gitignored.
# rust-lld (fast linker) + a bounded Cargo job budget. No compiler cache:
# sccache was removed (ADR-20260723T222214Z). See docs/development/build_speed.md.

[build]
jobs = $cargoJobs # $logicalCpus logical CPUs / $AgentSlots expected concurrent Cargo process(es)

[target.x86_64-pc-windows-msvc]
linker = "$lldToml"
"@ | Set-Content -Encoding UTF8 $cfg
    Write-Host "==> Wrote $cfg"
}

if ($AllWorktrees) {
    foreach ($root in $worktreeRoots) { Write-CargoConfig $root }
} else {
    Write-CargoConfig $repoRoot
    $outside = @($worktreeRoots | Where-Object {
        $_ -ne ($repoRoot -replace '\\', '/') -and -not $_.StartsWith(($repoRoot -replace '\\', '/'), 'OrdinalIgnoreCase')
    })
    if ($outside.Count -gt 0) {
        Write-Warning "$($outside.Count) worktree(s) live outside this checkout and cannot inherit its Cargo config:"
        $outside | ForEach-Object { Write-Host "      $_" }
        Write-Host   "   They build with the stock linker and no job budget. Rerun with -AllWorktrees to provision them."
    }
}

# --- Windows Defender exclusions (elevated only) -----------------------------
$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
            ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($elevated) {
    Write-Host "==> Adding Defender exclusions for build dirs (big Windows win)"
    foreach ($p in @((Join-Path $repoRoot 'target'), (Join-Path $env:USERPROFILE '.cargo'),
                     (Join-Path $env:USERPROFILE '.rustup'))) {
        try { Add-MpPreference -ExclusionPath $p; Write-Host "   excluded $p" } catch { Write-Warning "   $p : $_" }
    }
} else {
    Write-Host "==> Not elevated: skipping Defender exclusions."
    Write-Host "   For a 20-40% Windows build win, run an elevated PowerShell and see docs/development/build_speed.md 3.8."
}

Write-Host ""
Write-Host "IMPORTANT: environment changes only reach NEW shells."
Write-Host "           Close and reopen your terminal before the next build."
Write-Host ""
Write-Host "Next:  bash scripts/check-build-env.sh ; just check"
