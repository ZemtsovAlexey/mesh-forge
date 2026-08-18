param(
    [string]$InstallDir = "",
    [string]$ListenHost = "0.0.0.0",
    [int]$Port = 8188
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$stateDir = Join-Path $root ".runtime"
$pidFile = Join-Path $stateDir "comfyui.pid"
. (Join-Path $PSScriptRoot "comfyui-common.ps1")

# Wildcard bind addresses are not reliable HTTP targets; probe loopback instead.
$probeHost = if ($ListenHost -in @("0.0.0.0", "::", "*")) { "127.0.0.1" } else { $ListenHost }
$probeUrl = "http://${probeHost}:${Port}"
$listenUrl = "http://${ListenHost}:${Port}"

function Test-ComfyHealth {
    try {
        $resp = Invoke-WebRequest -Uri "${probeUrl}/system_stats" -UseBasicParsing -TimeoutSec 3
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

if (Test-ComfyHealth) {
    Write-Host "ComfyUI already running at $listenUrl (probe $probeUrl)" -ForegroundColor Green
    exit 0
}

$layout = Find-ComfyLayout -ProjectRoot $root -InstallDir $InstallDir
if (-not (Test-ComfyLayoutReady $layout)) {
    throw "ComfyUI is not installed / not detectable. Run .\scripts\setup-comfyui.ps1 or install ComfyUI Desktop."
}

New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
if (Test-Path $pidFile) {
    $oldPid = Get-Content $pidFile -ErrorAction SilentlyContinue
    if ($oldPid -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
        Write-Host "ComfyUI process $oldPid already tracked; waiting for health..." -ForegroundColor Yellow
    } else {
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Starting ComfyUI ($($layout.Kind)) from $($layout.ComfyRoot) ..." -ForegroundColor Cyan
if ($layout.Kind -eq "desktop") {
    Write-Host "Base directory: $($layout.BaseDirectory)" -ForegroundColor Cyan
}

$outLog = Join-Path $stateDir "comfyui.out.log"
$errLog = Join-Path $stateDir "comfyui.err.log"

# disable-smart-memory: keep models off GPU after /free so LM Studio can load.
# SAM3 nodes can false-positive as "pytest mode" without this env toggle.
$prevSam3ForceInit = $env:SAM3_FORCE_INIT
$env:SAM3_FORCE_INIT = "1"
$args = @(
    "-s", ".\main.py",
    "--listen", $ListenHost,
    "--port", "$Port",
    "--disable-auto-launch",
    "--disable-smart-memory"
)
if ($layout.Kind -eq "portable") {
    $args += "--windows-standalone-build"
}
if ($layout.Kind -eq "desktop") {
    if ($layout.BaseDirectory) {
        $args += @("--base-directory", $layout.BaseDirectory)
    }
    if ($layout.ExtraModelPathsConfig -and (Test-Path $layout.ExtraModelPathsConfig)) {
        $args += @("--extra-model-paths-config", $layout.ExtraModelPathsConfig)
    }
    if ($layout.FrontEndRoot -and (Test-Path $layout.FrontEndRoot)) {
        $args += @("--front-end-root", $layout.FrontEndRoot)
    }
}

try {
    $proc = Start-Process `
        -FilePath $layout.Python `
        -ArgumentList $args `
        -WorkingDirectory $layout.ComfyRoot `
        -RedirectStandardOutput $outLog `
        -RedirectStandardError $errLog `
        -PassThru `
        -WindowStyle Hidden
} finally {
    if ($null -eq $prevSam3ForceInit) {
        Remove-Item Env:SAM3_FORCE_INIT -ErrorAction SilentlyContinue
    } else {
        $env:SAM3_FORCE_INIT = $prevSam3ForceInit
    }
}

Set-Content -Path $pidFile -Value $proc.Id -Encoding ascii
Write-Host "ComfyUI pid=$($proc.Id), logs=$outLog / $errLog"

$deadline = (Get-Date).AddMinutes(3)
while ((Get-Date) -lt $deadline) {
    if (Test-ComfyHealth) {
        Write-Host "ComfyUI is ready at $listenUrl (local $probeUrl)" -ForegroundColor Green
        exit 0
    }
    if ($proc.HasExited) {
        throw "ComfyUI exited early. Check logs: $outLog / $errLog"
    }
    Start-Sleep -Seconds 2
}

throw "Timed out waiting for ComfyUI on $listenUrl. Check logs: $outLog / $errLog"
