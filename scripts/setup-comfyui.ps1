param(
    [string]$InstallDir = "",
    [switch]$SkipCheckpoints,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

function Read-ComfyInstallDir {
    param([string]$Fallback)
    $cfg = Join-Path $root "config.yaml"
    if (-not (Test-Path $cfg)) { return $Fallback }
    $match = Select-String -Path $cfg -Pattern '^\s*install_dir:\s*(.+)$' | Select-Object -First 1
    if (-not $match) { return $Fallback }
    return $match.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
}

$installDir = if ($InstallDir) { $InstallDir } else { Read-ComfyInstallDir -Fallback "C:\AI\ComfyUI" }
$venvPython = Join-Path $installDir "venv\Scripts\python.exe"
$venvPip = Join-Path $installDir "venv\Scripts\pip.exe"
$mainPy = Join-Path $installDir "main.py"
$ckptDir = Join-Path $installDir "models\checkpoints"

Write-Host "== ComfyUI setup ($installDir) ==" -ForegroundColor Cyan

if ($ForceReinstall -and (Test-Path $installDir)) {
    throw "Refusing ForceReinstall without manual cleanup. Delete $installDir first if you really want a clean install."
}

if (-not (Test-Path $mainPy)) {
    Write-Host "Cloning ComfyUI..." -ForegroundColor Yellow
    $parent = Split-Path -Parent $installDir
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
    git clone https://github.com/comfyanonymous/ComfyUI.git $installDir
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating ComfyUI venv..." -ForegroundColor Yellow
    python -m venv (Join-Path $installDir "venv")
}

Write-Host "Installing ComfyUI Python deps..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip
# CUDA 13 build used on this machine previously; fall back to requirements if needed.
try {
    & $venvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
} catch {
    Write-Warning "CUDA 13 torch install failed, continuing with ComfyUI requirements only."
}
& $venvPip install -r (Join-Path $installDir "requirements.txt")

New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null

if (-not $SkipCheckpoints) {
    Write-Host "Ensuring ComfyUI checkpoints..." -ForegroundColor Yellow
    $checkpoints = @(
        @{
            Name = "sd_xl_turbo_1.0_fp16.safetensors"
            Url  = "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors"
        },
        @{
            Name = "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"
            Url  = "https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"
        }
    )

    foreach ($item in $checkpoints) {
        $dest = Join-Path $ckptDir $item.Name
        if (Test-Path $dest) {
            Write-Host "  OK $($item.Name)" -ForegroundColor Green
            continue
        }
        $part = "$dest.part"
        Write-Host "  Downloading $($item.Name)..." -ForegroundColor Yellow
        curl.exe -L --output $part $item.Url
        if (-not $?) {
            throw "Failed to download $($item.Name)"
        }
        Move-Item -Force $part $dest
        Write-Host "  Saved $($item.Name)" -ForegroundColor Green
    }
}

Write-Host "ComfyUI setup complete." -ForegroundColor Green
Write-Host "Start with: .\scripts\start-comfyui.ps1"
