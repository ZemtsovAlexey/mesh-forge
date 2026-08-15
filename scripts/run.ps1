param(
    [string]$HostBind = "",
    [int]$Port = 0,
    [switch]$WithComfyUI
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

function Ensure-Uv {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) { return $uv.Source }
    $candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "uv\bin\uv.exe")
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) {
            $env:Path = "$(Split-Path $path -Parent);$env:Path"
            return $path
        }
    }
    throw "uv not found. Run .\scripts\setup.ps1 first."
}

if ($WithComfyUI) {
    & (Join-Path $PSScriptRoot "start-comfyui.ps1")
}

if ($HostBind) {
    $env:MESHFORGE_HOST = $HostBind
}
if ($Port -gt 0) {
    $env:MESHFORGE_PORT = "$Port"
}

$uvExe = Ensure-Uv
Set-Location $root

# Chat UI is rebuilt by the API when web/src is newer than web/dist.
& $uvExe run python ".\server.py"
