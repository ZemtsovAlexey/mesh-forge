param(
    [string]$InstallDir = "",
    [string]$PortableRoot = "",
    [string]$DownloadUrl = "",
    [ValidateSet("auto", "nvidia", "amd", "intel")]
    [string]$Gpu = "auto",
    [switch]$SkipCheckpoints,
    [switch]$QualityModels,
    [switch]$ForcePortable,
    [switch]$ForceReinstall,
    [switch]$KeepArchive
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "comfyui-common.ps1")

$layout = Find-ComfyLayout -ProjectRoot $root -InstallDir $InstallDir
$legacyDir = "C:\AI\ComfyUI"

if ($ForcePortable) {
    $layout = $null
}

if ((Test-ComfyLayoutReady $layout) -and $layout.Kind -eq "desktop" -and -not $ForcePortable) {
    Write-Host "== ComfyUI Desktop detected ==" -ForegroundColor Cyan
    Write-Host "App:            $($layout.AppDir)"
    Write-Host "Code:           $($layout.ComfyRoot)"
    Write-Host "User data:      $($layout.BaseDirectory)"
    Write-Host "Checkpoints:    $($layout.CkptDir)  [$($layout.CkptDirSource)]"
    Write-Host "Python:         $($layout.Python)"
} elseif ((Test-ComfyLayoutReady $layout) -and $layout.Kind -in @("portable", "venv") -and -not $ForceReinstall) {
    Write-Host "== ComfyUI $($layout.Kind) detected ==" -ForegroundColor Cyan
    Write-Host "ComfyUI dir:    $($layout.ComfyRoot)"
    Write-Host "Checkpoints:    $($layout.CkptDir)  [$($layout.CkptDirSource)]"
} else {
    # Install portable only when nothing usable is present.
    $download = if ($DownloadUrl) {
        [pscustomobject]@{
            Gpu = $Gpu
            ArchiveName = (Split-Path $DownloadUrl -Leaf)
            Url = $DownloadUrl
        }
    } else {
        Get-ComfyPortableDownloadInfo -Gpu $Gpu
    }

    if (-not $PortableRoot) {
        $configured = if ($InstallDir) { $InstallDir } else { Read-ComfyInstallDir -ProjectRoot $root -Fallback "" }
        if ($configured -and $configured -notmatch 'Documents[\\/]+ComfyUI') {
            $PortableRoot = Get-PortableRootFromComfyRoot -ComfyRoot $configured
            # If configured path is .../ComfyUI, parent is portable root.
            if ((Split-Path $configured -Leaf) -ieq "ComfyUI") {
                $PortableRoot = Split-Path $configured -Parent
            }
        }
        if (-not $PortableRoot) {
            $PortableRoot = Get-DefaultComfyPortableRoot
        }
    }
    $comfyRoot = Join-Path $PortableRoot "ComfyUI"
    $archivePath = Join-Path $env:TEMP $download.ArchiveName

    Write-Host "== ComfyUI portable setup ==" -ForegroundColor Cyan
    Write-Host "GPU build:      $($download.Gpu)"
    Write-Host "Portable root:  $PortableRoot"
    Write-Host "ComfyUI dir:    $comfyRoot"

    if ($ForceReinstall -and (Test-Path $PortableRoot)) {
        throw "Refusing ForceReinstall without manual cleanup. Delete $PortableRoot first if you really want a clean install."
    }

    $existing = Resolve-ComfyLayout -InstallDir $comfyRoot
    $needsInstall = -not (Test-ComfyLayoutReady $existing)

    if ($needsInstall) {
        if (-not (Test-Path $archivePath)) {
            Write-Host "Downloading portable ComfyUI (large, ~2GB)..." -ForegroundColor Yellow
            curl.exe -L --fail --retry 3 --output $archivePath $download.Url
            if (-not $?) { throw "Failed to download $($download.Url)" }
        } else {
            Write-Host "Using existing archive: $archivePath" -ForegroundColor Cyan
        }

        $sevenZip = Ensure-7Zip
        $extractParent = Split-Path $PortableRoot -Parent
        if (-not (Test-Path $extractParent)) {
            New-Item -ItemType Directory -Force -Path $extractParent | Out-Null
        }

        $stage = Join-Path $env:TEMP ("ComfyUI_portable_extract_" + [guid]::NewGuid().ToString("n"))
        New-Item -ItemType Directory -Force -Path $stage | Out-Null
        Write-Host "Extracting with $sevenZip ..." -ForegroundColor Yellow
        & $sevenZip x $archivePath "-o$stage" -y
        if ($LASTEXITCODE -ne 0) { throw "7-Zip extract failed with code $LASTEXITCODE" }

        $extracted = Get-ChildItem $stage -Directory | Select-Object -First 1
        if (-not $extracted) { throw "Extracted archive did not contain a top-level folder." }

        if (Test-Path $PortableRoot) {
            throw "Target already exists: $PortableRoot. Delete it or pass -PortableRoot."
        }
        Move-Item -LiteralPath $extracted.FullName -Destination $PortableRoot
        Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue

        if (-not $KeepArchive) {
            Remove-Item -Force $archivePath -ErrorAction SilentlyContinue
        }

        $layout = Resolve-ComfyLayout -InstallDir $comfyRoot
        if (-not (Test-ComfyLayoutReady $layout)) {
            throw "Portable install looks incomplete at $PortableRoot"
        }
        Write-Host "Portable ComfyUI ready ($($layout.Kind))." -ForegroundColor Green
    } else {
        $layout = $existing
        Write-Host "Portable ComfyUI already present." -ForegroundColor Green
    }
}

if (-not (Test-ComfyLayoutReady $layout)) {
    throw "No usable ComfyUI install found. Install ComfyUI Desktop, or re-run with -ForcePortable."
}

# Re-query live folder list if ComfyUI is already up (portable install path may skip Find-ComfyLayout enrichment).
Update-ComfyLayoutCheckpointDir -Layout $layout -ProjectRoot $root | Out-Null
Write-Host "Checkpoint dir: $($layout.CkptDir)  [$($layout.CkptDirSource)]" -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path $layout.CkptDir | Out-Null
$workflowsDir = Join-Path $layout.UserDir "workflows"
New-Item -ItemType Directory -Force -Path $workflowsDir | Out-Null

# Migrate checkpoints / workflows from legacy git install if present.
if ((Test-Path $legacyDir) -and ($legacyDir -ne $layout.ComfyRoot) -and ($legacyDir -ne $layout.BaseDirectory)) {
    $legacyCkpt = Join-Path $legacyDir "models\checkpoints"
    if (Test-Path $legacyCkpt) {
        Get-ChildItem $legacyCkpt -File -ErrorAction SilentlyContinue | ForEach-Object {
            $dest = Join-Path $layout.CkptDir $_.Name
            if (-not (Test-Path $dest)) {
                Write-Host "Migrating checkpoint $($_.Name)..." -ForegroundColor Yellow
                Copy-Item $_.FullName $dest
            }
        }
    }
    $legacyWf = Join-Path $legacyDir "user\default\workflows"
    if (Test-Path $legacyWf) {
        Get-ChildItem $legacyWf -Filter "*.json" -ErrorAction SilentlyContinue | ForEach-Object {
            $dest = Join-Path $workflowsDir $_.Name
            if (-not (Test-Path $dest)) {
                Copy-Item $_.FullName $dest
            }
        }
    }
}

# Sync MeshForge API workflows into ComfyUI UI browser.
$projectWf = Join-Path $root "mesh_forge\workflows"
if (Test-Path $projectWf) {
    Get-ChildItem $projectWf -Filter "*.json" | ForEach-Object {
        Copy-Item -Force $_.FullName (Join-Path $workflowsDir $_.Name)
    }
}

if (-not $SkipCheckpoints) {
    Write-Host "Ensuring ComfyUI checkpoints in $($layout.CkptDir) ..." -ForegroundColor Yellow
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
    if ($QualityModels) {
        $checkpoints += @(
            @{
                Name = "sd_xl_base_1.0_0.9vae.safetensors"
                Url  = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors"
            },
            @{
                Name = "hunyuan3d-dit-v2-mv_fp16.safetensors"
                Url  = "https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/hunyuan3d-dit-v2-mv_fp16.safetensors"
            }
        )
    }

    foreach ($item in $checkpoints) {
        $dest = Join-Path $layout.CkptDir $item.Name
        if (Test-Path $dest) {
            Write-Host "  OK $($item.Name)" -ForegroundColor Green
            continue
        }
        $part = "$dest.part"
        Write-Host "  Downloading $($item.Name)..." -ForegroundColor Yellow
        curl.exe -L --fail --retry 3 --output $part $item.Url
        if (-not $?) { throw "Failed to download $($item.Name)" }
        Move-Item -Force $part $dest
        Write-Host "  Saved $($item.Name)" -ForegroundColor Green
    }
}

# Persist resolved install_dir into config.yaml when present.
$configPath = Join-Path $root "config.yaml"
if (Test-Path $configPath) {
    $installPosix = ($layout.InstallDir -replace '\\', '/')
    $raw = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8
    if ($raw -match '(?m)^(\s*install_dir:\s*).*$') {
        $raw = [regex]::Replace($raw, '(?m)^(\s*install_dir:\s*).*$', "`${1}$installPosix", 1)
        Set-Content -LiteralPath $configPath -Value $raw -Encoding UTF8 -NoNewline
        Write-Host "Updated config.yaml comfyui.install_dir -> $installPosix" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "comfyui.install_dir:" -ForegroundColor Cyan
Write-Host "  $($layout.InstallDir -replace '\\','/')"
Write-Host "Kind: $($layout.Kind)"
Write-Host "Start with: .\scripts\start-comfyui.ps1"
Write-Host "ComfyUI setup complete." -ForegroundColor Green
