param(
    [string]$InstallDir = "",
    [string]$PortableRoot = "",
    [string]$DownloadUrl = "",
    [ValidateSet("auto", "nvidia", "amd", "intel")]
    [string]$Gpu = "auto",
    [switch]$SkipCheckpoints,
    [switch]$WithSegmentation,
    [ValidateSet("groundingdino", "owlvit")]
    [string]$SegmentationDetector = "groundingdino",
    [switch]$SkipSegmentationNodes,
    [switch]$SkipSegmentationSmoke,
    [switch]$QualityModels,
    [switch]$ForcePortable,
    [switch]$ForceReinstall,
    [switch]$KeepArchive
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "comfyui-common.ps1")

function Ensure-Git {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) {
        throw "git is required to install Comfy custom nodes. Install Git and retry."
    }
    return $git.Source
}

function Ensure-CustomNode {
    param(
        [string]$RepoUrl,
        [string]$NodeName,
        $Layout
    )

    $gitExe = Ensure-Git
    $customNodes = Join-Path $Layout.ComfyRoot "custom_nodes"
    New-Item -ItemType Directory -Force -Path $customNodes | Out-Null
    $nodeDir = Join-Path $customNodes $NodeName
    if (Test-Path $nodeDir) {
        Write-Host "Updating custom node $NodeName..." -ForegroundColor Yellow
        & $gitExe -C $nodeDir pull --ff-only
    } else {
        Write-Host "Installing custom node $NodeName..." -ForegroundColor Yellow
        & $gitExe clone $RepoUrl $nodeDir
    }
    $req = Join-Path $nodeDir "requirements.txt"
    if (Test-Path $req) {
        Write-Host "  pip install -r requirements.txt ($NodeName)" -ForegroundColor Yellow
        & $Layout.Python -m pip install -r $req
    }
    $installPy = Join-Path $nodeDir "install.py"
    if (Test-Path $installPy) {
        Write-Host "  running install.py ($NodeName)" -ForegroundColor Yellow
        & $Layout.Python $installPy
    }
    return $nodeDir
}

function Test-Sam3Available {
    param($Layout)
    $builtin = Join-Path $Layout.ComfyRoot "comfy_extras\nodes_sam3.py"
    if (Test-Path $builtin) { return $builtin }
    foreach ($name in @("ComfyUI-SAM3", "ComfyUI-segment-anything-3")) {
        $candidate = Join-Path $Layout.ComfyRoot "custom_nodes\$name"
        if (Test-Path $candidate) { return $candidate }
    }
    return ""
}

function Test-GroundingAvailable {
    param($Layout)
    foreach ($name in @("ComfyUI-Grounding", "owl-vit-comfyui")) {
        $candidate = Join-Path $Layout.ComfyRoot "custom_nodes\$name"
        if (Test-Path $candidate) { return $candidate }
    }
    return ""
}

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
$liveCkptDir = Resolve-LiveComfyCheckpointDir -Layout $layout -ProjectRoot $root
if ($liveCkptDir) {
    $layout.CkptDir = $liveCkptDir
    $layout.CkptDirSource = "live"
}
Write-Host "Checkpoint dir: $($layout.CkptDir)  [$($layout.CkptDirSource)]" -ForegroundColor Cyan

$comfyUrl = Read-ComfyBaseUrl -ProjectRoot $root
$needSegmentation = [bool]($WithSegmentation -or (Read-SegmentationEnabled -ProjectRoot $root))
$canWriteCheckpoints = [bool]$liveCkptDir
if ($canWriteCheckpoints) {
    New-Item -ItemType Directory -Force -Path $layout.CkptDir | Out-Null
} elseif (Test-ComfyUrlIsLocal -BaseUrl $comfyUrl) {
    New-Item -ItemType Directory -Force -Path $layout.CkptDir | Out-Null
    $canWriteCheckpoints = $true
} else {
    Write-Warning "Not creating $($layout.CkptDir) here: live ComfyUI is $comfyUrl and this PC cannot write its checkpoint folder."
}
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

if ($WithSegmentation -or $needSegmentation) {
    $liveHasNativeSam3 = $false
    try {
        $info = Invoke-RestMethod -Uri ($comfyUrl.TrimEnd("/") + "/object_info/SAM3_Detect") -TimeoutSec 8
        $liveHasNativeSam3 = [bool]$info.SAM3_Detect
    } catch {
    }
    if ($canWriteCheckpoints -or (Test-ComfyUrlIsLocal -BaseUrl $comfyUrl)) {
        Write-Host "Ensuring Comfy segmentation nodes..." -ForegroundColor Yellow
        if (-not $SkipSegmentationNodes) {
            $groundingRepo = "https://github.com/PozzettiAndrea/ComfyUI-Grounding.git"
            Ensure-CustomNode -RepoUrl $groundingRepo -NodeName "ComfyUI-Grounding" -Layout $layout | Out-Null
            if ((-not $liveHasNativeSam3) -and (-not (Test-Sam3Available -Layout $layout))) {
                $sam3Repo = "https://github.com/PozzettiAndrea/ComfyUI-SAM3.git"
                Ensure-CustomNode -RepoUrl $sam3Repo -NodeName "ComfyUI-SAM3" -Layout $layout | Out-Null
            }
        }
        if (-not $SkipSegmentationSmoke) {
            $grounding = Test-GroundingAvailable -Layout $layout
            if ($grounding) {
                Write-Host "  detector nodes: OK ($grounding)" -ForegroundColor Green
            } else {
                Write-Warning "Detector node was not found after setup. Expected ComfyUI-Grounding or owl-vit-comfyui."
            }
            if ($liveHasNativeSam3) {
                Write-Host "  SAM3 nodes: OK (native SAM3_Detect on $comfyUrl)" -ForegroundColor Green
            } else {
                $sam3 = Test-Sam3Available -Layout $layout
                if ($sam3) {
                    Write-Host "  SAM3 nodes: OK ($sam3)" -ForegroundColor Green
                } else {
                    Write-Warning "SAM3 node was not found after setup."
                }
            }
            Write-Host "  detector backend: $SegmentationDetector" -ForegroundColor Cyan
        }
    } else {
        Write-Host "ComfyUI is remote ($comfyUrl); skipping local custom_nodes install." -ForegroundColor Cyan
        if ($liveHasNativeSam3) {
            Write-Host "  SAM3 nodes: OK (native SAM3_Detect)" -ForegroundColor Green
        }
    }
}

if ((-not $SkipCheckpoints) -and $canWriteCheckpoints) {
    Write-Host "Ensuring ComfyUI checkpoints in $($layout.CkptDir) ..." -ForegroundColor Yellow
    $checkpoints = @(
        @{
            Name = "sd_xl_turbo_1.0_fp16.safetensors"
            Url  = "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors"
        },
        @{
            Name = "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"
            Url  = "https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"
        },
        @{
            Name = "stable_zero123.ckpt"
            Url  = "https://huggingface.co/stabilityai/stable-zero123/resolve/main/stable_zero123.ckpt"
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
        Ensure-DownloadedFile -Url $item.Url -Dest (Join-Path $layout.CkptDir $item.Name) | Out-Null
    }
} elseif (-not $SkipCheckpoints) {
    Write-Warning "Skipping Hunyuan/SDXL checkpoint downloads: ComfyUI checkpoints are not writable from this PC."
}

if ($needSegmentation) {
    Ensure-Sam3Checkpoint -Layout $layout -ProjectRoot $root -Required
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
if ($WithSegmentation) {
    Write-Host "Segmentation: detector=$SegmentationDetector via Comfy custom nodes" -ForegroundColor Cyan
    Write-Host "If ComfyUI was already running, restart it to load newly installed nodes." -ForegroundColor Cyan
}
Write-Host "Start with: .\scripts\start-comfyui.ps1"
Write-Host "ComfyUI setup complete." -ForegroundColor Green
