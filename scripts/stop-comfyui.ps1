param(
    [int]$Port = 8188
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$stateDir = Join-Path $root ".runtime"
$pidFile = Join-Path $stateDir "comfyui.pid"

function Stop-PidSafe([int]$ProcessId) {
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $proc) { return $false }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    return $true
}

$stopped = $false
if (Test-Path $pidFile) {
    $tracked = [int]((Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1))
    if ($tracked -gt 0) {
        if (Stop-PidSafe -ProcessId $tracked) {
            Write-Host "Stopped tracked ComfyUI pid=$tracked" -ForegroundColor Yellow
            $stopped = $true
        }
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

$conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $conns) {
    if (Stop-PidSafe -ProcessId $conn.OwningProcess) {
        Write-Host "Stopped listener pid=$($conn.OwningProcess) on port $Port" -ForegroundColor Yellow
        $stopped = $true
    }
}

if (-not $stopped) {
    Write-Host "No ComfyUI process found." -ForegroundColor Cyan
} else {
    Write-Host "ComfyUI stop requested." -ForegroundColor Green
}
