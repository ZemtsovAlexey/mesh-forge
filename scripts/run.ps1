param(
    [string]$HostBind = "",
    [int]$Port = 0,
    [switch]$WithComfyUI
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root "venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

if ($WithComfyUI) {
    & (Join-Path $PSScriptRoot "start-comfyui.ps1")
}

if ($HostBind) {
    $env:MESHFORGE_HOST = $HostBind
}
if ($Port -gt 0) {
    $env:MESHFORGE_PORT = "$Port"
}

Set-Location $root
& $python ".\server.py"
