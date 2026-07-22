<#
    Set up a high-speed Thalos build environment on native Windows.
    Verifies rust-lld, installs sccache, sets the sccache environment, and
    (when elevated) offers Windows Defender exclusions for the build dirs.

    sccache activation on Windows is the machine-global RUSTC_WRAPPER user
    environment variable, NOT a per-directory .cargo\config.toml. A worktree can
    be created anywhere (C:\tmp, .codex\worktrees, .claude\worktrees) and Cargo's
    config discovery would never find a repo-local wrapper there, so the cache
    silently did nothing. The generated config carries only what genuinely
    differs per checkout: the job budget and the linker.

    SCCACHE_BASEDIRS is read by the sccache SERVER at startup and is what lets a
    compilation in one worktree hit an object cached from another. It is a
    snapshot of `git worktree list`, so it goes stale the moment an agent adds a
    worktree -- rerun with -SyncOnly after `git worktree add`.

    Full rationale + the agent workflow: docs/build_speed.md

    Usage (from the repo root, PowerShell):
        scripts\setup-build-env.ps1
        scripts\setup-build-env.ps1 -Force          # overwrite existing .cargo\config.toml (backs it up)
        scripts\setup-build-env.ps1 -AgentSlots 4   # divide CPUs across 4 concurrent Cargo processes
        scripts\setup-build-env.ps1 -AllWorktrees   # write matching config into every worktree
        scripts\setup-build-env.ps1 -SyncOnly       # only refresh SCCACHE_BASEDIRS + restart sccache
#>
param(
    [switch]$Force,
    [switch]$AllWorktrees,
    [switch]$SyncOnly,
    [ValidateRange(1, 64)]
    [int]$AgentSlots = 1
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$logicalCpus = [Environment]::ProcessorCount
$cargoJobs = [Math]::Max(1, [Math]::Floor($logicalCpus / $AgentSlots))

# Normalization roots must be listed longest-first: worktrees nest inside the
# repo root (.claude\worktrees\*), and a shorter parent that matched first would
# rewrite `.claude/worktrees/x/crates/...` where the main checkout produces
# `crates/...` -- different hashes, so the two could never share a cache entry.
$worktreeRoots = @(
    git -C $repoRoot worktree list --porcelain |
        Where-Object { $_ -like 'worktree *' } |
        ForEach-Object { ($_.Substring('worktree '.Length)) -replace '\\', '/' } |
        Sort-Object -Property Length -Descending
)
if ($worktreeRoots.Count -eq 0) { $worktreeRoots = @(($repoRoot -replace '\\', '/')) }
$sccBaseDirs = [string]::Join(';', $worktreeRoots)

function Have($name) { $null -ne (Get-Command $name -ErrorAction SilentlyContinue) }

function Restart-SccacheServer($sccacheExe, $sccDir, $sccBaseDirs) {
    # SCCACHE_BASEDIRS only takes effect at server startup, so a resync means a
    # restart. Refuse to yank the daemon out from under a live compile.
    $busy = @('cargo', 'rustc', 'dx') |
        ForEach-Object { Get-Process -Name $_ -ErrorAction SilentlyContinue } |
        Select-Object -First 1
    if ($busy) {
        Write-Warning "Cargo/rustc/dx is active - NOT restarting sccache."
        Write-Host   "   SCCACHE_BASEDIRS stays stale until you rerun: scripts\setup-build-env.ps1 -SyncOnly"
        return $false
    }
    $env:SCCACHE_DIR = $sccDir
    $env:SCCACHE_CACHE_SIZE = '50G'
    $env:SCCACHE_BASEDIRS = $sccBaseDirs
    & $sccacheExe --stop-server 2>&1 | Out-Null
    & $sccacheExe --start-server | Out-Null
    Write-Host "==> sccache server restarted with $($worktreeRoots.Count) normalization root(s)"
    return $true
}

Write-Host "==> Platform: windows (native, msvc)"
if (-not (Have 'cargo')) { throw "cargo not found. Install rustup first: https://rustup.rs" }

# --- refresh PATH so a freshly installed sccache is discoverable -------------
$env:Path = @(
    [Environment]::GetEnvironmentVariable('Path', 'Machine'),
    [Environment]::GetEnvironmentVariable('Path', 'User')
) -join ';'

$sccDir = Join-Path $env:LOCALAPPDATA 'Mozilla\sccache'

# --- -SyncOnly: the cheap post-`git worktree add` path ------------------------
if ($SyncOnly) {
    if (-not (Have 'sccache')) { throw 'sccache not found - run this script without -SyncOnly first.' }
    $sccacheExe = (Get-Command 'sccache').Source
    [Environment]::SetEnvironmentVariable('SCCACHE_BASEDIRS', $sccBaseDirs, 'User')
    Write-Host "==> SCCACHE_BASEDIRS ($($worktreeRoots.Count) roots, longest-first):"
    $worktreeRoots | ForEach-Object { Write-Host "      $_" }
    [void](Restart-SccacheServer $sccacheExe $sccDir $sccBaseDirs)
    Write-Host ""
    Write-Host "Verify:  sccache --show-stats   (Base directories should list every worktree)"
    return
}

Write-Host "==> rustc: $(rustc --version)"

# --- rust-lld (fast linker) --------------------------------------------------
$tc = (rustc -vV | Select-String 'host:').ToString().Split(':')[1].Trim()
$sysroot = (rustc --print sysroot).Trim()
$lld = Join-Path $sysroot "lib\rustlib\$tc\bin\rust-lld.exe"
if (Test-Path $lld) {
    Write-Host "==> rust-lld: $lld"
} else {
    Write-Warning "rust-lld.exe not found at $lld"
    Write-Host   "   Install the LLVM tools:  rustup component add llvm-tools"
}

# --- sccache -----------------------------------------------------------------
if (Have 'sccache') {
    Write-Host "==> sccache present: $(sccache --version)"
} else {
    Write-Host "==> Installing sccache"
    if     (Have 'scoop')  { scoop install sccache }
    elseif (Have 'winget') { winget install --id Mozilla.sccache --accept-source-agreements --accept-package-agreements }
    elseif (Have 'cargo-binstall') { cargo binstall -y sccache }
    else   { Write-Host "   (building from source - slow, one time)"; cargo install sccache --locked }
}
$env:Path = @(
    [Environment]::GetEnvironmentVariable('Path', 'Machine'),
    [Environment]::GetEnvironmentVariable('Path', 'User')
) -join ';'
if (-not (Have 'sccache')) {
    throw 'sccache was installed but is not discoverable after refreshing PATH'
}
$sccacheExe = (Get-Command 'sccache').Source
Write-Host "==> sccache executable: $sccacheExe"

# SCCACHE_BASEDIRS needs sccache >= 0.14.0. Verify rather than trust.
$sccacheVersion = [version](((sccache --version) -split '\s+')[1] -replace '[-+].*$', '')
if ($sccacheVersion -lt [version]'0.14.0') {
    throw "sccache >= 0.14.0 is required for SCCACHE_BASEDIRS; found $sccacheVersion"
}

# --- sccache env (machine-global activation) ---------------------------------
# RUSTC_WRAPPER here rather than build.rustc-wrapper in a repo-local config: the
# config is gitignored and per-directory, so every worktree outside this tree
# silently built with no cache and no rust-lld.
# Forward slashes: Cargo fingerprints the wrapper path, so a stable spelling
# avoids a spurious full rebuild whenever this is re-provisioned.
$sccacheExeToml = ($sccacheExe -replace '\\', '/')
[Environment]::SetEnvironmentVariable('RUSTC_WRAPPER', $sccacheExeToml, 'User')
[Environment]::SetEnvironmentVariable('SCCACHE_DIR', $sccDir, 'User')
[Environment]::SetEnvironmentVariable('SCCACHE_CACHE_SIZE', '50G', 'User')
[Environment]::SetEnvironmentVariable('SCCACHE_BASEDIRS', $sccBaseDirs, 'User')
Write-Host "==> RUSTC_WRAPPER=$sccacheExeToml (user env - applies to every worktree)"
Write-Host "==> SCCACHE_DIR=$sccDir  SCCACHE_CACHE_SIZE=50G (user env)"
Write-Host "==> SCCACHE_BASEDIRS ($($worktreeRoots.Count) roots, longest-first):"
$worktreeRoots | ForEach-Object { Write-Host "      $_" }

# --- write local .cargo\config.toml ------------------------------------------
$lldToml = ($lld -replace '\\', '/')

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
# sccache is activated machine-globally via the RUSTC_WRAPPER user environment
# variable, so it reaches worktrees this file can never be discovered from.
# See docs/build_speed.md.

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
        Write-Host   "   They get sccache (global RUSTC_WRAPPER) but NOT rust-lld or the job budget."
        Write-Host   "   Rerun with -AllWorktrees to provision them."
    }
}

[void](Restart-SccacheServer $sccacheExe $sccDir $sccBaseDirs)

# --- Windows Defender exclusions (elevated only) -----------------------------
$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
            ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($elevated) {
    Write-Host "==> Adding Defender exclusions for build dirs (big Windows win)"
    foreach ($p in @((Join-Path $repoRoot 'target'), (Join-Path $env:USERPROFILE '.cargo'),
                     (Join-Path $env:USERPROFILE '.rustup'), $sccDir)) {
        try { Add-MpPreference -ExclusionPath $p; Write-Host "   excluded $p" } catch { Write-Warning "   $p : $_" }
    }
} else {
    Write-Host "==> Not elevated: skipping Defender exclusions."
    Write-Host "   For a 20-40% Windows build win, run an elevated PowerShell and see docs/build_speed.md 3.8."
}

Write-Host ""
Write-Host "IMPORTANT: user environment variables only reach NEW shells."
Write-Host "           Close and reopen your terminal before the next build."
Write-Host ""
Write-Host "Next:  bash scripts/check-build-env.sh ; just check"
Write-Host "After every `git worktree add`:  scripts\setup-build-env.ps1 -SyncOnly"
