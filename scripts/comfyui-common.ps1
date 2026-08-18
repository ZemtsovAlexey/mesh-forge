# Shared ComfyUI helpers for MeshForge scripts.
# Supports: ComfyUI Desktop, Windows portable, and git/venv installs.

$script:DefaultComfyPortableRoot = Join-Path $env:LOCALAPPDATA "MeshForge\ComfyUI_windows_portable"
$script:DefaultComfyRoot = Join-Path $script:DefaultComfyPortableRoot "ComfyUI"

function Get-DefaultComfyRoot {
    return $script:DefaultComfyRoot
}

function Get-DefaultComfyPortableRoot {
    return $script:DefaultComfyPortableRoot
}

function Read-ComfyInstallDir {
    param(
        [string]$ProjectRoot,
        [string]$Fallback = ""
    )
    $cfg = Join-Path $ProjectRoot "config.yaml"
    if (-not (Test-Path $cfg)) { return $Fallback }
    $match = Select-String -Path $cfg -Pattern '^\s*install_dir:\s*(.+)$' | Select-Object -First 1
    if (-not $match) { return $Fallback }
    $value = $match.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
    if (-not $value) { return $Fallback }
    return $value
}

function Get-ComfyDesktopAppDir {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\ComfyUI"),
        (Join-Path $env:LOCALAPPDATA "Programs\comfyui")
    ) | Select-Object -Unique

    foreach ($dir in $candidates) {
        $mainPy = Join-Path $dir "resources\ComfyUI\main.py"
        $exe = @(
            (Join-Path $dir "ComfyUI.exe"),
            (Join-Path $dir "Comfy Desktop.exe")
        ) | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ((Test-Path $mainPy) -and $exe) {
            return $dir
        }
    }
    return $null
}

function Get-ComfyDesktopBasePath {
    param([string]$AppDir = "")

    $cfg = Join-Path $env:APPDATA "ComfyUI\config.json"
    if (Test-Path $cfg) {
        try {
            $json = Get-Content -LiteralPath $cfg -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($json.basePath -and (Test-Path ([string]$json.basePath))) {
                return [string]$json.basePath
            }
        } catch {
        }
    }

    $fallback = Join-Path $env:USERPROFILE "Documents\ComfyUI"
    if (Test-Path $fallback) {
        return $fallback
    }

    if ($AppDir) {
        $nested = Join-Path $AppDir "resources\ComfyUI"
        if (Test-Path (Join-Path $nested "main.py")) {
            return $nested
        }
    }
    return $null
}

function New-ComfyLayoutObject {
    param(
        [string]$Kind,
        [string]$ComfyRoot,
        [string]$Python,
        [string]$BaseDirectory = "",
        [string]$AppDir = "",
        [string]$DesktopExe = "",
        [string]$ExtraModelPathsConfig = "",
        [string]$FrontEndRoot = "",
        [string]$InstallDir = ""
    )

    if (-not $BaseDirectory) { $BaseDirectory = $ComfyRoot }
    if (-not $InstallDir) { $InstallDir = $BaseDirectory }

    return [pscustomobject]@{
        Kind                   = $Kind
        ComfyRoot              = $ComfyRoot
        Python                 = $Python
        MainPy                 = Join-Path $ComfyRoot "main.py"
        BaseDirectory          = $BaseDirectory
        CkptDir                = Join-Path $BaseDirectory "models\checkpoints"
        CkptDirSource          = "layout"
        UserDir                = Join-Path $BaseDirectory "user\default"
        AppDir                 = $AppDir
        DesktopExe             = $DesktopExe
        ExtraModelPathsConfig  = $ExtraModelPathsConfig
        FrontEndRoot           = $FrontEndRoot
        InstallDir             = $InstallDir
    }
}

function Read-ComfyBaseUrl {
    param(
        [string]$ProjectRoot,
        [string]$Fallback = "http://127.0.0.1:8188"
    )
    $cfg = Join-Path $ProjectRoot "config.yaml"
    if (-not (Test-Path $cfg)) { return $Fallback }

    # Prefer the value under the comfyui: section (llm also has base_url).
    $lines = Get-Content -LiteralPath $cfg -Encoding UTF8
    $inComfy = $false
    foreach ($line in $lines) {
        if ($line -match '^\s*comfyui\s*:') {
            $inComfy = $true
            continue
        }
        if ($inComfy -and $line -match '^[^\s#]') {
            $inComfy = $false
        }
        if ($inComfy -and $line -match '^\s*base_url\s*:\s*(.+)$') {
            $value = $Matches[1].Trim().Trim('"').Trim("'")
            if ($value) { return $value.TrimEnd("/") }
        }
    }
    return $Fallback
}

function Get-ComfyApiCheckpointFolders {
    param(
        [string]$BaseUrl = "http://127.0.0.1:8188",
        [int]$TimeoutSec = 3
    )

    $base = $BaseUrl.TrimEnd("/")
    foreach ($path in @("/experiment/models", "/api/experiment/models")) {
        try {
            $resp = Invoke-WebRequest -Uri ($base + $path) -UseBasicParsing -TimeoutSec $TimeoutSec
            if ($resp.StatusCode -ne 200) { continue }
            $json = $resp.Content | ConvertFrom-Json
            foreach ($entry in $json) {
                if ($entry.name -ne "checkpoints") { continue }
                $folders = @($entry.folders | Where-Object { $_ -and "$_".Trim() })
                if ($folders.Count -gt 0) {
                    return $folders
                }
            }
        } catch {
        }
    }
    return @()
}

function Update-ComfyLayoutCheckpointDir {
    <#
    Prefer the live ComfyUI folder list (/experiment/models).
    Fallback stays at layout default ({base}/models/checkpoints).
    #>
    param(
        $Layout,
        [string]$ProjectRoot = "",
        [string]$BaseUrl = ""
    )
    if (-not $Layout) { return $null }

    if (-not $BaseUrl) {
        if ($ProjectRoot) {
            $BaseUrl = Read-ComfyBaseUrl -ProjectRoot $ProjectRoot
        } else {
            $BaseUrl = "http://127.0.0.1:8188"
        }
    }

    $folders = @(Get-ComfyApiCheckpointFolders -BaseUrl $BaseUrl)
    if ($folders.Count -gt 0) {
        $preferred = $null
        foreach ($folder in $folders) {
            if (Test-Path $folder) {
                $preferred = $folder
                break
            }
        }
        if (-not $preferred) { $preferred = [string]$folders[0] }
        $Layout.CkptDir = $preferred
        $Layout.CkptDirSource = "api:$BaseUrl"
    } else {
        $Layout.CkptDirSource = "layout"
    }
    return $Layout
}

function Resolve-ComfyDesktopLayout {
    param(
        [string]$AppDir = "",
        [string]$HintPath = ""
    )

    if (-not $AppDir) {
        $AppDir = Get-ComfyDesktopAppDir
    }
    if (-not $AppDir) { return $null }

    $comfyRoot = Join-Path $AppDir "resources\ComfyUI"
    if (-not (Test-Path (Join-Path $comfyRoot "main.py"))) { return $null }

    $basePath = Get-ComfyDesktopBasePath -AppDir $AppDir
    if ($HintPath -and (Test-Path $HintPath)) {
        # Prefer explicit install_dir / hint when it looks like user data or app root.
        $hintMain = Join-Path $HintPath "main.py"
        $hintModels = Join-Path $HintPath "models\checkpoints"
        if ((Test-Path $hintModels) -or (Test-Path (Join-Path $HintPath ".venv\Scripts\python.exe"))) {
            $basePath = $HintPath
        } elseif (Test-Path $hintMain) {
            # Hint points at code root; keep Documents basePath if present.
            if (-not $basePath) { $basePath = $HintPath }
        }
    }
    if (-not $basePath) {
        $basePath = Join-Path $env:USERPROFILE "Documents\ComfyUI"
    }

    $pythonCandidates = @(
        (Join-Path $basePath ".venv\Scripts\python.exe"),
        (Join-Path $AppDir "resources\bootstrap-python\python.exe")
    )
    $python = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $python) { return $null }

    $desktopExe = @(
        (Join-Path $AppDir "ComfyUI.exe"),
        (Join-Path $AppDir "Comfy Desktop.exe")
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1

    $extra = Join-Path $env:APPDATA "ComfyUI\extra_models_config.yaml"
    if (-not (Test-Path $extra)) { $extra = "" }

    $frontEndRoot = Join-Path $AppDir "resources\desktop-ui"
    if (-not (Test-Path (Join-Path $frontEndRoot "index.html"))) {
        $frontEndRoot = Join-Path $AppDir "resources\UI"
        if (-not (Test-Path (Join-Path $frontEndRoot "index.html"))) {
            $frontEndRoot = ""
        }
    }

    return New-ComfyLayoutObject `
        -Kind "desktop" `
        -ComfyRoot $comfyRoot `
        -Python $python `
        -BaseDirectory $basePath `
        -AppDir $AppDir `
        -DesktopExe $desktopExe `
        -ExtraModelPathsConfig $extra `
        -FrontEndRoot $frontEndRoot `
        -InstallDir $basePath
}

function Resolve-ComfyLayout {
    param([string]$InstallDir = "")

    $hint = if ($InstallDir) { $InstallDir.Trim() } else { "" }

    # Explicit Desktop app / resources path.
    if ($hint) {
        $appFromHint = $null
        if (Test-Path (Join-Path $hint "resources\ComfyUI\main.py")) {
            $appFromHint = $hint
        } elseif ((Split-Path $hint -Leaf) -ieq "ComfyUI") {
            $parent = Split-Path $hint -Parent
            $grand = if ($parent) { Split-Path $parent -Parent } else { $null }
            if ($grand -and (Test-Path (Join-Path $grand "resources\ComfyUI\main.py"))) {
                $appFromHint = $grand
            } elseif ($parent -and ((Test-Path (Join-Path $parent "ComfyUI.exe")) -or (Test-Path (Join-Path $parent "Comfy Desktop.exe")))) {
                # resources\ComfyUI
                $appFromHint = $parent
            }
        }
        if ($appFromHint) {
            $desktop = Resolve-ComfyDesktopLayout -AppDir $appFromHint -HintPath $hint
            if ($desktop) { return $desktop }
        }

        # Desktop user data path (Documents\ComfyUI) while Desktop app is installed.
        $desktopApp = Get-ComfyDesktopAppDir
        if ($desktopApp) {
            $base = Get-ComfyDesktopBasePath -AppDir $desktopApp
            $hintFull = [System.IO.Path]::GetFullPath($hint)
            $baseFull = if ($base) { [System.IO.Path]::GetFullPath($base) } else { "" }
            if ($baseFull -and ($hintFull -ieq $baseFull)) {
                $desktop = Resolve-ComfyDesktopLayout -AppDir $desktopApp -HintPath $hint
                if ($desktop) { return $desktop }
            }
        }
    }

    $comfyRoot = $hint
    if ($comfyRoot -and -not (Test-Path (Join-Path $comfyRoot "main.py"))) {
        $nested = Join-Path $comfyRoot "ComfyUI"
        if (Test-Path (Join-Path $nested "main.py")) {
            $comfyRoot = $nested
        }
    }

    if ($comfyRoot -and (Test-Path (Join-Path $comfyRoot "main.py"))) {
        $portablePython = Join-Path (Split-Path $comfyRoot -Parent) "python_embeded\python.exe"
        $venvPython = Join-Path $comfyRoot "venv\Scripts\python.exe"
        if (Test-Path $portablePython) {
            return New-ComfyLayoutObject -Kind "portable" -ComfyRoot $comfyRoot -Python $portablePython -InstallDir $comfyRoot
        }
        if (Test-Path $venvPython) {
            return New-ComfyLayoutObject -Kind "venv" -ComfyRoot $comfyRoot -Python $venvPython -InstallDir $comfyRoot
        }
    }

    # No usable path from hint — try Desktop discovery.
    $desktop = Resolve-ComfyDesktopLayout
    if ($desktop) { return $desktop }

    if ($comfyRoot) {
        return New-ComfyLayoutObject -Kind "unknown" -ComfyRoot $comfyRoot -Python "" -InstallDir $comfyRoot
    }

    return New-ComfyLayoutObject -Kind "unknown" -ComfyRoot "" -Python "" -InstallDir ""
}

function Find-ComfyLayout {
    param(
        [string]$ProjectRoot,
        [string]$InstallDir = "",
        [switch]$SkipApiPaths
    )

    $found = $null
    $configured = if ($InstallDir) { $InstallDir } else { Read-ComfyInstallDir -ProjectRoot $ProjectRoot -Fallback "" }
    if ($configured) {
        $layout = Resolve-ComfyLayout -InstallDir $configured
        if ($layout.Kind -ne "unknown" -and (Test-Path $layout.MainPy) -and $layout.Python) {
            $found = $layout
        }
    }

    if (-not $found) {
        $desktop = Resolve-ComfyDesktopLayout
        if ($desktop) { $found = $desktop }
    }

    if (-not $found) {
        foreach ($candidate in @(
                (Get-DefaultComfyRoot),
                "C:\AI\ComfyUI_windows_portable\ComfyUI",
                "C:\AI\ComfyUI",
                (Join-Path $env:USERPROFILE "ComfyUI")
            )) {
            if (-not $candidate -or -not (Test-Path $candidate)) { continue }
            $layout = Resolve-ComfyLayout -InstallDir $candidate
            if ($layout.Kind -in @("portable", "venv") -and (Test-Path $layout.MainPy) -and $layout.Python) {
                $found = $layout
                break
            }
        }
    }

    if ($found -and -not $SkipApiPaths) {
        Update-ComfyLayoutCheckpointDir -Layout $found -ProjectRoot $ProjectRoot | Out-Null
    }
    return $found
}

function Test-ComfyLayoutReady {
    param($Layout)
    if (-not $Layout) { return $false }
    if (-not $Layout.Python) { return $false }
    if (-not (Test-Path $Layout.MainPy)) { return $false }
    return $Layout.Kind -in @("desktop", "portable", "venv")
}

function Get-PortableRootFromComfyRoot {
    param([string]$ComfyRoot)
    return (Split-Path $ComfyRoot -Parent)
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

function Get-LocalIPv4Addresses {
    $addrs = @()
    try {
        $addrs = @(
            Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
                Where-Object { $_.IPAddress -and $_.IPAddress -notlike "127.*" } |
                ForEach-Object { $_.IPAddress }
        )
    } catch {
    }
    if ($addrs.Count -eq 0) {
        $output = ipconfig 2>$null
        foreach ($line in @($output)) {
            if ($line -match "IPv4.*:\s*([0-9.]+)") {
                $ip = $Matches[1]
                if ($ip -notlike "127.*") { $addrs += $ip }
            }
        }
    }
    return @($addrs | Select-Object -Unique)
}

function Test-ComfyUrlIsLocal {
    param([string]$BaseUrl)
    try {
        $uri = [Uri]$BaseUrl
    } catch {
        return $false
    }
$hostName = ([string]$uri.Host).Trim().ToLowerInvariant()
    if ($hostName -in @("localhost", "127.0.0.1", "::1", "0.0.0.0")) {
        return $true
    }
    return (Get-LocalIPv4Addresses) -contains $uri.Host
}

function Read-YamlSectionFlag {
    param(
        [string]$ProjectRoot,
        [string]$Section,
        [string]$Key
    )
    $cfg = Join-Path $ProjectRoot "config.yaml"
    if (-not (Test-Path $cfg)) { return $null }
    $inSection = $false
    foreach ($line in (Get-Content -LiteralPath $cfg -Encoding UTF8)) {
        if ($line -match "^${Section}\s*:") {
            $inSection = $true
            continue
        }
        if ($inSection -and $line -match "^\S") {
            break
        }
        if ($inSection -and $line -match "^\s*${Key}\s*:\s*(.+)$") {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return $null
}

function Read-SegmentationEnabled {
    param([string]$ProjectRoot)
    $raw = Read-YamlSectionFlag -ProjectRoot $ProjectRoot -Section "segmentation" -Key "enabled"
    return [bool]($raw -and ($raw.ToLowerInvariant() -in @("true", "yes", "on", "1")))
}

function Read-SegmenterModel {
    param(
        [string]$ProjectRoot,
        [string]$Fallback = "sam3.1_multiplex_fp16.safetensors"
    )
    $raw = Read-YamlSectionFlag -ProjectRoot $ProjectRoot -Section "segmentation" -Key "segmenter_model"
    if ($raw) { return $raw }
    return $Fallback
}

function Get-ComfyApiCheckpointNames {
    param(
        [string]$BaseUrl = "http://127.0.0.1:8188",
        [int]$TimeoutSec = 5
    )
    $probe = ($BaseUrl.TrimEnd("/") + "/models/checkpoints")
    try {
        $json = Invoke-RestMethod -Uri $probe -TimeoutSec $TimeoutSec
        return @($json | ForEach-Object { "$_" } | Where-Object { $_ })
    } catch {
        return @()
    }
}

function Get-Sam3CheckpointSpec {
    param(
        [string]$ProjectRoot = "",
        [string]$Name = ""
    )
    if (-not $Name) {
        $Name = if ($ProjectRoot) {
            Read-SegmenterModel -ProjectRoot $ProjectRoot
        } else {
            "sam3.1_multiplex_fp16.safetensors"
        }
    }
    return [pscustomobject]@{
        Name     = $Name
        Url      = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"
        MinBytes = 500MB
    }
}

function Resolve-LiveComfyCheckpointDir {
    param(
        $Layout,
        [string]$ProjectRoot = "",
        [string]$BaseUrl = ""
    )
    if (-not $BaseUrl) {
        $BaseUrl = if ($ProjectRoot) { Read-ComfyBaseUrl -ProjectRoot $ProjectRoot } else { "http://127.0.0.1:8188" }
    }
    $folders = @(Get-ComfyApiCheckpointFolders -BaseUrl $BaseUrl)
    $liveNames = @(Get-ComfyApiCheckpointNames -BaseUrl $BaseUrl)
    foreach ($folder in $folders) {
        if (-not (Test-Path -LiteralPath $folder)) { continue }
        $localNames = @(
            Get-ChildItem -LiteralPath $folder -File -ErrorAction SilentlyContinue |
                ForEach-Object { $_.Name }
        )
        if ($liveNames.Count -gt 0) {
            $overlap = @($localNames | Where-Object { $liveNames -contains $_ })
            if ($overlap.Count -gt 0) {
                return $folder
            }
            continue
        }
        if (($localNames.Count -gt 0) -and (Test-ComfyUrlIsLocal -BaseUrl $BaseUrl)) {
            return $folder
        }
    }
    if (Test-ComfyUrlIsLocal -BaseUrl $BaseUrl) {
        if ($Layout -and $Layout.CkptDir) {
            return [string]$Layout.CkptDir
        }
        if ($folders.Count -gt 0 -and (Test-Path -LiteralPath $folders[0])) {
            return [string]$folders[0]
        }
    }
    return $null
}

function Ensure-DownloadedFile {
    param(
        [string]$Url,
        [string]$Dest,
        [long]$MinBytes = 10485760
    )
    $destItem = Get-Item -LiteralPath $Dest -ErrorAction SilentlyContinue
    if ($destItem -and $destItem.Length -ge $MinBytes) {
        Write-Host "  OK $(Split-Path $Dest -Leaf)" -ForegroundColor Green
        return $Dest
    }
    $dir = Split-Path $Dest -Parent
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $part = "$Dest.part"
    Write-Host "  Downloading $(Split-Path $Dest -Leaf)..." -ForegroundColor Yellow
    curl.exe -L --fail --retry 3 --output $part $Url
    if (-not $?) { throw "Failed to download $Url" }
    $got = (Get-Item -LiteralPath $part).Length
    if ($got -lt $MinBytes) {
        Remove-Item -Force $part -ErrorAction SilentlyContinue
        throw "Downloaded $got bytes for $(Split-Path $Dest -Leaf); expected at least $MinBytes. The URL likely returned an error page."
    }
    Move-Item -Force $part $Dest
    Write-Host "  Saved $Dest" -ForegroundColor Green
    return $Dest
}

function Ensure-Sam3Checkpoint {
    param(
        $Layout,
        [string]$ProjectRoot,
        [switch]$Required
    )
    $baseUrl = Read-ComfyBaseUrl -ProjectRoot $ProjectRoot
    $spec = Get-Sam3CheckpointSpec -ProjectRoot $ProjectRoot
    $liveNames = @(Get-ComfyApiCheckpointNames -BaseUrl $baseUrl)
    if ($liveNames | Where-Object { $_ -match '(?i)sam3' }) {
        Write-Host "SAM3 checkpoint: OK on $baseUrl ($(($liveNames | Where-Object { $_ -match '(?i)sam3' }) -join ', '))" -ForegroundColor Green
        return
    }
    $dir = Resolve-LiveComfyCheckpointDir -Layout $Layout -ProjectRoot $ProjectRoot -BaseUrl $baseUrl
    $folders = @(Get-ComfyApiCheckpointFolders -BaseUrl $baseUrl)
    $hint = if ($folders.Count -gt 0) { [string]$folders[0] } else { "ComfyUI\models\checkpoints" }
    if (-not $dir) {
        $curlLine = "curl.exe -L --fail --retry 3 --output `"$hint\$($spec.Name)`" `"$($spec.Url)`""
        $msg = @"
SAM3 checkpoint $($spec.Name) is missing from ComfyUI at $baseUrl.
This machine cannot write the live checkpoint folder ($hint).
Run .\scripts\setup-segmentation.ps1 on the Comfy host, or paste:
  $curlLine
Then restart ComfyUI so CheckpointLoaderSimple sees the new file.
"@
        if ($Required) { throw $msg.Trim() }
        Write-Warning $msg.Trim()
        return
    }
    Write-Host "Ensuring SAM3 checkpoint in $dir ..." -ForegroundColor Yellow
    Ensure-DownloadedFile -Url $spec.Url -Dest (Join-Path $dir $spec.Name) -MinBytes ([int64]$spec.MinBytes)
    Write-Host "Restart ComfyUI if it was already running so it picks up $($spec.Name)." -ForegroundColor Cyan
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
