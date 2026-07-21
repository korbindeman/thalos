# Turn sccache ON for the CURRENT PowerShell (Cold/parallel regime): cache the
# whole dependency graph across worktrees/branches/clean builds. Disables
# incremental (sccache cannot cache incremental crates) — do NOT use this for the
# everyday single-machine edit loop, where incremental is the win.
# See docs/build_speed.md section 5.
#
# Usage:  . scripts\sccache-on.ps1      (dot-source, not run)

if ($null -eq (Get-Command sccache -ErrorAction SilentlyContinue)) {
    Write-Error "sccache not installed — run scripts\setup-build-env.ps1 first."
    return
}

$env:RUSTC_WRAPPER    = 'sccache'
$env:CARGO_INCREMENTAL = '0'
if (-not $env:SCCACHE_DIR)        { $env:SCCACHE_DIR = Join-Path $env:LOCALAPPDATA 'Mozilla\sccache' }
if (-not $env:SCCACHE_CACHE_SIZE) { $env:SCCACHE_CACHE_SIZE = '50G' }

Write-Host "sccache ON  (RUSTC_WRAPPER=sccache, CARGO_INCREMENTAL=0)"
Write-Host "  SCCACHE_DIR=$($env:SCCACHE_DIR)  SCCACHE_CACHE_SIZE=$($env:SCCACHE_CACHE_SIZE)"
Write-Host "  stats: sccache --show-stats   |   off: Remove-Item Env:RUSTC_WRAPPER,Env:CARGO_INCREMENTAL"
