param(
    [string]$InstallDir = "",
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8188
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$stateDir = Join-Path $root ".runtime"
$pidFile = Join-Path $stateDir "comfyui.pid"
. (Join-Path $PSScriptRoot "comfyui-common.ps1")

function Test-ComfyHealth {
    try {
        $resp = Invoke-WebRequest -Uri "http://${ListenHost}:${Port}/system_stats" -UseBasicParsing -TimeoutSec 3
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

if (Test-ComfyHealth) {
    Write-Host "ComfyUI already running at http://${ListenHost}:${Port}" -ForegroundColor Green
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

$args = @("-s", ".\main.py", "--listen", $ListenHost, "--port", "$Port", "--disable-auto-launch")
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

$proc = Start-Process `
    -FilePath $layout.Python `
    -ArgumentList $args `
    -WorkingDirectory $layout.ComfyRoot `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -PassThru `
    -WindowStyle Hidden

Set-Content -Path $pidFile -Value $proc.Id -Encoding ascii
Write-Host "ComfyUI pid=$($proc.Id), logs=$outLog / $errLog"

$deadline = (Get-Date).AddMinutes(3)
while ((Get-Date) -lt $deadline) {
    if (Test-ComfyHealth) {
        Write-Host "ComfyUI is ready at http://${ListenHost}:${Port}" -ForegroundColor Green
        exit 0
    }
    if ($proc.HasExited) {
        throw "ComfyUI exited early. Check logs: $outLog / $errLog"
    }
    Start-Sleep -Seconds 2
}

throw "Timed out waiting for ComfyUI on http://${ListenHost}:${Port}. Check logs: $outLog / $errLog"
