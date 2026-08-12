param(
    [switch]$SkipComfyUI,
    [switch]$SkipCheckpoints,
    [switch]$StartComfyUI
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root "venv"
$python = Join-Path $venv "Scripts\python.exe"
$pip = Join-Path $venv "Scripts\pip.exe"

Write-Host "== MeshForge setup ==" -ForegroundColor Cyan

if (-not (Test-Path $venv)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $venv
}

Write-Host "Installing Python requirements..." -ForegroundColor Yellow
& $python -m pip install --upgrade pip
& $pip install -r (Join-Path $root "requirements.txt")

$configExample = Join-Path $root "config.yaml.example"
$configPath = Join-Path $root "config.yaml"
if (-not (Test-Path $configPath) -and (Test-Path $configExample)) {
    Copy-Item $configExample $configPath
    Write-Host "Created config.yaml from example. Review paths and model settings." -ForegroundColor Yellow
}

if (-not $SkipComfyUI) {
    Write-Host "Setting up ComfyUI..." -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot "setup-comfyui.ps1") -SkipCheckpoints:$SkipCheckpoints
}

Write-Host ""
Write-Host "Manual services to verify:" -ForegroundColor Cyan
Write-Host "  1. LM Studio local server at http://127.0.0.1:1234/v1"
Write-Host "  2. ComfyUI API at http://127.0.0.1:8188"
Write-Host "  3. config.yaml paths for Blender / projects / comfyui.install_dir"
Write-Host ""
Write-Host "Useful scripts:" -ForegroundColor Cyan
Write-Host "  .\scripts\start-comfyui.ps1"
Write-Host "  .\scripts\stop-comfyui.ps1"
Write-Host "  .\scripts\run.ps1"
Write-Host ""

if ($StartComfyUI) {
    & (Join-Path $PSScriptRoot "start-comfyui.ps1")
}

Write-Host "Next step: .\scripts\start-comfyui.ps1 ; .\scripts\run.ps1" -ForegroundColor Green
