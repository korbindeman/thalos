[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,

    [Parameter(Mandatory = $true)]
    [string]$Artifact,

    [string]$OutputDirectory = "dist",
    [string]$RequestedFeatures = "",
    [bool]$UseDefaultFeatures = $false,
    [string]$Target = "x86_64-pc-windows-msvc",
    [string]$Revision = "unknown"
)

$ErrorActionPreference = "Stop"

if ([IO.Path]::GetFileName($Artifact) -ne $Artifact) {
    throw "Artifact must be one directory name, got '$Artifact'"
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$executablePath = (Resolve-Path -LiteralPath $Executable).Path
$outputRoot = if ([IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory
} else {
    Join-Path $repositoryRoot $OutputDirectory
}
$packageRoot = Join-Path $outputRoot $Artifact
$zipPath = Join-Path $outputRoot "$Artifact.zip"
$checksumPath = "$zipPath.sha256"

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
foreach ($path in @($packageRoot, $zipPath, $checksumPath)) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}
New-Item -ItemType Directory -Path $packageRoot | Out-Null

$buildInfoLines = @(& $executablePath --build-info)
if ($LASTEXITCODE -ne 0) {
    throw "thalos_game --build-info failed with exit code $LASTEXITCODE"
}
$neuralAvailable = $buildInfoLines -contains "neural_terrain_available=true"

Copy-Item -LiteralPath $executablePath -Destination (Join-Path $packageRoot "thalos_game.exe")
Copy-Item -LiteralPath (Join-Path $repositoryRoot "assets") -Destination (Join-Path $packageRoot "assets") -Recurse
Copy-Item -LiteralPath (Join-Path $repositoryRoot "ships") -Destination (Join-Path $packageRoot "ships") -Recurse
foreach ($license in @("LICENSE", "LICENSE-ASSETS", "LICENSING.md")) {
    Copy-Item -LiteralPath (Join-Path $repositoryRoot $license) -Destination $packageRoot
}
Copy-Item -LiteralPath (Join-Path $repositoryRoot "distribution/README-WINDOWS.txt") -Destination (Join-Path $packageRoot "README.txt")

if (-not $neuralAvailable) {
    $neuralContent = Join-Path $packageRoot "assets/terrain_packages/thalos_diffusion"
    if (Test-Path -LiteralPath $neuralContent) {
        Remove-Item -LiteralPath $neuralContent -Recurse -Force
    }
}

$featureLabel = if ([string]::IsNullOrWhiteSpace($RequestedFeatures)) { "<none>" } else { $RequestedFeatures }
$buildInfo = @(
    "Thalos pre-alpha build"
    "revision=$Revision"
    "target=$Target"
    "requested_features=$featureLabel"
    "use_default_features=$($UseDefaultFeatures.ToString().ToLowerInvariant())"
    $buildInfoLines
) -join [Environment]::NewLine
Set-Content -LiteralPath (Join-Path $packageRoot "BUILD_INFO.txt") -Value $buildInfo -Encoding utf8

# Prove the archive itself, not just the staging directory. The process runs
# from a separate empty directory, so a checkout-relative content path cannot
# accidentally pass this gate.
Compress-Archive -LiteralPath $packageRoot -DestinationPath $zipPath -CompressionLevel Optimal
$temporaryBase = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }
$verificationRoot = Join-Path $temporaryBase "thalos-package-$PID"
$foreignWorkingDirectory = Join-Path $temporaryBase "thalos-package-cwd-$PID"
try {
    New-Item -ItemType Directory -Path $verificationRoot | Out-Null
    New-Item -ItemType Directory -Path $foreignWorkingDirectory | Out-Null
    Expand-Archive -LiteralPath $zipPath -DestinationPath $verificationRoot
    $packagedRoot = Join-Path $verificationRoot $Artifact
    $packagedExecutable = Join-Path $packagedRoot "thalos_game.exe"
    # The release binary retains a compile-time developer fallback. Make the
    # extracted archive authoritative so a missing packaged file cannot be
    # satisfied accidentally by the runner's checkout.
    $env:THALOS_CONTENT_ROOT = $packagedRoot
    Push-Location $foreignWorkingDirectory
    try {
        & $packagedExecutable --verify-install
        if ($LASTEXITCODE -ne 0) {
            throw "packaged install verification failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
} finally {
    Remove-Item Env:THALOS_CONTENT_ROOT -ErrorAction SilentlyContinue
    foreach ($path in @($verificationRoot, $foreignWorkingDirectory)) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Recurse -Force
        }
    }
}

$checksum = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
$checksumLine = "$checksum  $([IO.Path]::GetFileName($zipPath))"
Set-Content -LiteralPath $checksumPath -Value $checksumLine -Encoding ascii

Write-Host "Packaged $zipPath"
Write-Host "SHA-256 $checksum"
