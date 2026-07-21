# Enter the cold/parallel cache regime in the CURRENT PowerShell. The generated
# Cargo config already uses sccache for ordinary non-incremental dependencies;
# this additionally disables incremental so workspace crates can be shared
# across worktrees.
# See docs/build_speed.md section 5.
#
# Usage:  . scripts\sccache-on.ps1      (dot-source, not run)

$repoRoot = Split-Path -Parent $PSScriptRoot
$cargoConfig = Join-Path $repoRoot '.cargo\config.toml'
if (-not (Test-Path $cargoConfig) -or
    -not (Select-String -LiteralPath $cargoConfig -Pattern '^rustc-wrapper\s*=' -Quiet)) {
    Write-Error "generated Cargo config with sccache is missing — run scripts\setup-build-env.ps1 first."
    return
}

$env:CARGO_INCREMENTAL = '0'
if (-not $env:SCCACHE_DIR)        { $env:SCCACHE_DIR = Join-Path $env:LOCALAPPDATA 'Mozilla\sccache' }
if (-not $env:SCCACHE_CACHE_SIZE) { $env:SCCACHE_CACHE_SIZE = '50G' }
$worktreeRoots = @(
    git -C $repoRoot worktree list --porcelain |
        Where-Object { $_ -like 'worktree *' } |
        ForEach-Object { $_.Substring('worktree '.Length) }
)
if ($worktreeRoots.Count -eq 0) { $worktreeRoots = @($repoRoot) }
$env:SCCACHE_BASEDIRS = [string]::Join(';', $worktreeRoots)

Write-Host "parallel cache mode ON  (CARGO_INCREMENTAL=0)"
Write-Host "  SCCACHE_DIR=$($env:SCCACHE_DIR)  SCCACHE_CACHE_SIZE=$($env:SCCACHE_CACHE_SIZE)"
Write-Host "  SCCACHE_BASEDIRS=$($env:SCCACHE_BASEDIRS)"
Write-Host "  stats: sccache --show-stats   |   return to iterate mode: Remove-Item Env:CARGO_INCREMENTAL"
