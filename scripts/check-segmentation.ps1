param(
    [switch]$Deep
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "comfyui-common.ps1")

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

function Test-ComfyHealth {
    param([string]$BaseUrl)
    $probe = ($BaseUrl.TrimEnd("/") + "/system_stats")
    try {
        $resp = Invoke-WebRequest -Uri $probe -UseBasicParsing -TimeoutSec 3
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

$uvExe = Ensure-Uv
Set-Location $root

$tmp = Join-Path $env:TEMP ("meshforge_segmentation_check_" + [guid]::NewGuid().ToString("n") + ".py")
@"
from mesh_forge.config import load_config, segmentation_segmenter_base_url

cfg = load_config()
print(f"enabled={cfg.segmentation.enabled}")
print(f"provider={cfg.segmentation.provider}")
print(f"detector={cfg.segmentation.detector}")
print(f"detector_backend={cfg.segmentation.detector_backend}")
print(f"detector_model={cfg.segmentation.detector_model}")
print(f"detector_node_repo={cfg.segmentation.detector_node_repo}")
print(f"segmenter={cfg.segmentation.segmenter}")
print(f"segmenter_backend={cfg.segmentation.segmenter_backend}")
print(f"segmenter_base_url={segmentation_segmenter_base_url(cfg)}")
print(f"max_views={cfg.segmentation.max_views}")
print(f"render_size={cfg.segmentation.render_size}")
print(f"debug_emit_every_step={cfg.segmentation.debug_emit_every_step}")
print(f"free_gpu_between_steps={cfg.segmentation.free_gpu_between_steps}")
"@ | Set-Content -Path $tmp -Encoding utf8

try {
    Write-Host "== Segmentation config ==" -ForegroundColor Cyan
    & $uvExe run python $tmp
} finally {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
}

$baseUrl = Read-ComfyBaseUrl -ProjectRoot $root
if (Test-ComfyHealth -BaseUrl $baseUrl) {
    Write-Host "ComfyUI API: OK ($baseUrl)" -ForegroundColor Green
} else {
    Write-Warning "ComfyUI API did not respond at $baseUrl"
}

$layout = Find-ComfyLayout -ProjectRoot $root
if (Test-ComfyLayoutReady $layout) {
    $groundingNode = Join-Path $layout.ComfyRoot "custom_nodes\ComfyUI-Grounding"
    if (Test-Path $groundingNode) {
        Write-Host "Grounding node: OK ($groundingNode)" -ForegroundColor Green
    } else {
        Write-Warning "Grounding node not found at $groundingNode"
    }
    $sam3Node = Join-Path $layout.ComfyRoot "comfy_extras\nodes_sam3.py"
    if (Test-Path $sam3Node) {
        Write-Host "SAM3 node: OK ($sam3Node)" -ForegroundColor Green
    } else {
        $sam3Custom = Join-Path $layout.ComfyRoot "custom_nodes\ComfyUI-SAM3"
        if (Test-Path $sam3Custom) {
            Write-Host "SAM3 node: OK ($sam3Custom)" -ForegroundColor Green
        } else {
            Write-Warning "SAM3 node not found at $sam3Node or $sam3Custom"
        }
    }
} else {
    Write-Warning "ComfyUI layout was not detected locally."
}

if ($Deep) {
    if (Test-ComfyHealth -BaseUrl $baseUrl) {
        Write-Host "Deep check: ComfyUI API is alive; restart may still be required after node install." -ForegroundColor Cyan
    } else {
        Write-Warning "Deep check skipped because ComfyUI API is offline."
    }
}
