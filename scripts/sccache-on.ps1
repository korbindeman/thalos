# Enter the cold/parallel cache regime in the CURRENT PowerShell. sccache is
# already active machine-globally via RUSTC_WRAPPER; this additionally disables
# incremental so workspace crates become cacheable and shareable across
# worktrees (Cargo's dev profile sets incremental = true, and sccache cannot
# cache an incrementally compiled crate at all).
# See docs/build_speed.md section 5.
#
# Usage:  . scripts\sccache-on.ps1      (dot-source, not run)

$repoRoot = Split-Path -Parent $PSScriptRoot
$cargoConfig = Join-Path $repoRoot '.cargo\config.toml'

# Activation is EITHER the global RUSTC_WRAPPER (normal Windows setup, reaches
# worktrees anywhere on disk) or a repo-local build.rustc-wrapper (legacy).
# Requiring the config form is what made this fail in worktrees.
$sccacheExe = $env:RUSTC_WRAPPER
if (-not $sccacheExe) {
    $sccacheExe = [Environment]::GetEnvironmentVariable('RUSTC_WRAPPER', 'User')
}
if (-not $sccacheExe -and (Test-Path $cargoConfig)) {
    $wrapperLine = Select-String -LiteralPath $cargoConfig -Pattern '^\s*rustc-wrapper\s*=\s*"([^"]+)"' |
        Select-Object -First 1
    if ($wrapperLine -and $wrapperLine.Matches.Count -gt 0) {
        $sccacheExe = $wrapperLine.Matches[0].Groups[1].Value
    }
}
if (-not $sccacheExe) { $sccacheExe = (Get-Command sccache -ErrorAction SilentlyContinue).Source }
if (-not $sccacheExe -or -not (Test-Path -LiteralPath $sccacheExe)) {
    Write-Error 'no sccache activation found - run scripts\setup-build-env.ps1 first.'
    return
}
$env:RUSTC_WRAPPER = $sccacheExe

$env:CARGO_INCREMENTAL = '0'
if (-not $env:SCCACHE_DIR)        { $env:SCCACHE_DIR = Join-Path $env:LOCALAPPDATA 'Mozilla\sccache' }
if (-not $env:SCCACHE_CACHE_SIZE) { $env:SCCACHE_CACHE_SIZE = '50G' }

# Longest-first so a nested worktree matches before the repo root it lives under.
$worktreeRoots = @(
    git -C $repoRoot worktree list --porcelain |
        Where-Object { $_ -like 'worktree *' } |
        ForEach-Object { ($_.Substring('worktree '.Length)) -replace '\\', '/' } |
        Sort-Object -Property Length -Descending
)
if ($worktreeRoots.Count -eq 0) { $worktreeRoots = @(($repoRoot -replace '\\', '/')) }
$env:SCCACHE_BASEDIRS = [string]::Join(';', $worktreeRoots)

& $sccacheExe --show-stats *> $null
if ($LASTEXITCODE -ne 0) {
    & $sccacheExe --start-server *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Error 'sccache server did not start.'
        return
    }
}

# SCCACHE_BASEDIRS is read by the SERVER at startup, so setting it in this shell
# does nothing to a daemon that is already running. A worktree missing from the
# live set hashes absolute paths and can never hit another worktree's cache --
# an entirely invisible failure, so check and say so.
$stats = & $sccacheExe --show-stats 2>$null
$liveLine = ($stats | Select-String -Pattern '^Base directories\s*(.*)$' | Select-Object -First 1)
$live = if ($liveLine) { $liveLine.Matches[0].Groups[1].Value.ToLowerInvariant() } else { '' }
$missing = @($worktreeRoots | Where-Object { -not $live.Contains($_.ToLowerInvariant().TrimEnd('/')) })
if ($missing.Count -gt 0) {
    Write-Warning "sccache does NOT normalize $($missing.Count) worktree(s):"
    $missing | ForEach-Object { Write-Host "      $_" }
    $busy = @('cargo', 'rustc', 'dx') |
        ForEach-Object { Get-Process -Name $_ -ErrorAction SilentlyContinue } |
        Select-Object -First 1
    if ($busy) {
        Write-Host "   Cargo/rustc/dx active - once idle run: scripts\setup-build-env.ps1 -SyncOnly"
    } else {
        & $sccacheExe --stop-server *> $null
        & $sccacheExe --start-server *> $null
        Write-Host "   Restarted sccache with the current worktree set."
    }
}

Write-Host "parallel cache mode ON  (CARGO_INCREMENTAL=0)"
Write-Host "  RUSTC_WRAPPER=$sccacheExe"
Write-Host "  SCCACHE_DIR=$($env:SCCACHE_DIR)  SCCACHE_CACHE_SIZE=$($env:SCCACHE_CACHE_SIZE)"
Write-Host "  SCCACHE_BASEDIRS=$($env:SCCACHE_BASEDIRS)"
Write-Host "  stats: & '$sccacheExe' --show-stats"
Write-Host "  return to iterate mode: Remove-Item Env:CARGO_INCREMENTAL"
