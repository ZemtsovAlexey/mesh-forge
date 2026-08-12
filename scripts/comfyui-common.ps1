# Shared ComfyUI helpers for MeshForge scripts.

$script:DefaultComfyPortableRoot = "C:\AI\ComfyUI_windows_portable"
$script:DefaultComfyRoot = Join-Path $script:DefaultComfyPortableRoot "ComfyUI"

function Get-DefaultComfyRoot {
    return $script:DefaultComfyRoot
}

function Read-ComfyInstallDir {
    param(
        [string]$ProjectRoot,
        [string]$Fallback = $script:DefaultComfyRoot
    )
    $cfg = Join-Path $ProjectRoot "config.yaml"
    if (-not (Test-Path $cfg)) { return $Fallback }
    $match = Select-String -Path $cfg -Pattern '^\s*install_dir:\s*(.+)$' | Select-Object -First 1
    if (-not $match) { return $Fallback }
    $value = $match.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
    if (-not $value) { return $Fallback }
    return $value
}

function Resolve-ComfyLayout {
    param([string]$InstallDir)

    $comfyRoot = $InstallDir
    if (-not (Test-Path (Join-Path $comfyRoot "main.py"))) {
        $nested = Join-Path $InstallDir "ComfyUI"
        if (Test-Path (Join-Path $nested "main.py")) {
            $comfyRoot = $nested
        }
    }

    $portablePython = Join-Path (Split-Path $comfyRoot -Parent) "python_embeded\python.exe"
    $venvPython = Join-Path $comfyRoot "venv\Scripts\python.exe"

    $python = $null
    $kind = "unknown"
    if (Test-Path $portablePython) {
        $python = $portablePython
        $kind = "portable"
    } elseif (Test-Path $venvPython) {
        $python = $venvPython
        $kind = "venv"
    }

    return [pscustomobject]@{
        ComfyRoot = $comfyRoot
        Python    = $python
        Kind      = $kind
        MainPy    = Join-Path $comfyRoot "main.py"
        CkptDir   = Join-Path $comfyRoot "models\checkpoints"
        UserDir   = Join-Path $comfyRoot "user\default"
    }
}

function Ensure-7Zip {
    $candidates = @(
        "C:\Program Files\7-Zip\7z.exe",
        "C:\Program Files (x86)\7-Zip\7z.exe",
        (Join-Path $env:LOCALAPPDATA "MeshForge\tools\7zr.exe")
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) { return $path }
    }

    $toolDir = Join-Path $env:LOCALAPPDATA "MeshForge\tools"
    New-Item -ItemType Directory -Force -Path $toolDir | Out-Null
    $sevenZr = Join-Path $toolDir "7zr.exe"
    Write-Host "Downloading 7zr.exe..." -ForegroundColor Yellow
    curl.exe -L --fail --output $sevenZr "https://www.7-zip.org/a/7zr.exe"
    if (-not (Test-Path $sevenZr)) {
        throw "Failed to download 7zr.exe. Install 7-Zip or place 7z.exe on PATH."
    }
    return $sevenZr
}

function Get-PortableRootFromComfyRoot {
    param([string]$ComfyRoot)
    return (Split-Path $ComfyRoot -Parent)
}

function Get-GpuKind {
    param(
        [ValidateSet("auto", "nvidia", "amd", "intel")]
        [string]$Gpu = "auto"
    )

    if ($Gpu -ne "auto") {
        return $Gpu
    }

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        try {
            & $nvidiaSmi.Source | Out-Null
            if ($LASTEXITCODE -eq 0) {
                return "nvidia"
            }
        } catch {
        }
    }

    $names = @()
    try {
        $names = Get-CimInstance Win32_VideoController -ErrorAction Stop | ForEach-Object { $_.Name }
    } catch {
    }
    $joined = ($names -join " ").ToLowerInvariant()
    if ($joined -match "nvidia") { return "nvidia" }
    if ($joined -match "amd|radeon") { return "amd" }
    if ($joined -match "intel") { return "intel" }
    throw "Unable to detect GPU vendor automatically. Re-run with -Gpu nvidia or -Gpu amd."
}

function Get-ComfyPortableDownloadInfo {
    param(
        [ValidateSet("auto", "nvidia", "amd", "intel")]
        [string]$Gpu = "auto"
    )

    $resolved = Get-GpuKind -Gpu $Gpu
    switch ($resolved) {
        "nvidia" {
            return [pscustomobject]@{
                Gpu = "nvidia"
                ArchiveName = "ComfyUI_windows_portable_nvidia.7z"
                Url = "https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z"
            }
        }
        "amd" {
            return [pscustomobject]@{
                Gpu = "amd"
                ArchiveName = "ComfyUI_windows_portable_amd.7z"
                Url = "https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_amd.7z"
            }
        }
        "intel" {
            return [pscustomobject]@{
                Gpu = "intel"
                ArchiveName = "ComfyUI_windows_portable_intel.7z"
                Url = "https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_intel.7z"
            }
        }
    }

    throw "Unsupported GPU kind: $resolved"
}

function Get-GpuMemoryHintGb {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        try {
            $totalMb = (& $nvidiaSmi.Source --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1).Trim()
            if ($totalMb) {
                return [Math]::Max(1, [int][Math]::Round(([double]$totalMb) / 1024.0))
            }
        } catch {
        }
    }
    return 0
}
