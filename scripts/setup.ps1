param(
    [switch]$SkipComfyUI,
    [switch]$SkipCheckpoints,
    [switch]$StartComfyUI,
    [switch]$WithSegmentation,
    [ValidateSet("groundingdino", "owlvit")]
    [string]$SegmentationDetector = "groundingdino",
    [string]$SegmentationModel = "",
    [switch]$SkipSegmentationNodes,
    [switch]$SkipSegmentationSmoke,
    [ValidateSet("auto", "nvidia", "amd", "intel")]
    [string]$Gpu = "auto"
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
    Write-Host "Installing uv..." -ForegroundColor Yellow
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$(Join-Path $env:USERPROFILE '.local\bin');$env:Path"
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        throw "uv is not available after install. Restart the shell and retry."
    }
    return $uv.Source
}

function Update-LocalConfig {
    param(
        [string]$UvExe,
        [string]$ProjectRoot,
        [string]$ComfyRoot,
        [int]$GpuVramGb,
        [bool]$SegmentationEnabled,
        [string]$SegmentationDetector,
        [string]$SegmentationModel
    )

    $tmp = Join-Path $env:TEMP ("meshforge_config_update_" + [guid]::NewGuid().ToString("n") + ".py")
    @"
from pathlib import Path
import os
import yaml

root = Path(os.environ["MF_ROOT"])
cfg_path = root / "config.yaml"
data = {}
if cfg_path.is_file():
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

paths = dict(data.get("paths") or {})
paths.pop("blender", None)
if not paths.get("projects"):
    paths["projects"] = os.environ["MF_PROJECTS"]
if not paths.get("openscad"):
    default_scad = Path(r"C:\Program Files\OpenSCAD\openscad.exe")
    if default_scad.is_file():
        paths["openscad"] = default_scad.as_posix()
data["paths"] = paths

gpu = dict(data.get("gpu") or {})
gpu.setdefault("sequential_models", True)
gpu_hint = int(os.environ.get("MF_GPU_VRAM_GB", "0") or "0")
if int(gpu.get("vram_gb") or 0) <= 0 and gpu_hint > 0:
    gpu["vram_gb"] = gpu_hint
elif "vram_gb" not in gpu:
    gpu["vram_gb"] = 0
data["gpu"] = gpu

comfy = dict(data.get("comfyui") or {})
comfy["enabled"] = comfy.get("enabled", True)
comfy["base_url"] = str(comfy.get("base_url") or "http://127.0.0.1:8188").rstrip("/")
comfy["view_consistency"] = "zero123"
if not comfy.get("install_dir") and os.environ.get("MF_COMFY_ROOT"):
    comfy["install_dir"] = Path(os.environ["MF_COMFY_ROOT"]).as_posix()
data["comfyui"] = comfy

seg = dict(data.get("segmentation") or {})
seg["enabled"] = bool(seg.get("enabled", False))
if os.environ.get("MF_SEGMENTATION_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
    seg["enabled"] = True
detector = str(os.environ.get("MF_SEGMENTATION_DETECTOR") or "groundingdino").strip().lower()
seg["provider"] = str(seg.get("provider") or "comfyui")
seg["detector"] = detector
seg["detector_backend"] = str(seg.get("detector_backend") or ("owlvit" if detector == "owlvit" else "grounding"))
if not seg.get("detector_model"):
    default_model = "google/owlv2-base-patch16-ensemble" if detector == "owlvit" else "IDEA-Research/grounding-dino-tiny"
    seg["detector_model"] = str(os.environ.get("MF_SEGMENTATION_MODEL") or default_model)
seg["detector_node_repo"] = str(seg.get("detector_node_repo") or "https://github.com/PozzettiAndrea/ComfyUI-Grounding.git")
seg["detector_device"] = str(seg.get("detector_device") or "cuda")
seg["detector_dtype"] = str(seg.get("detector_dtype") or "float16")
seg["segmenter"] = str(seg.get("segmenter") or "sam3")
seg["segmenter_backend"] = str(seg.get("segmenter_backend") or "sam3")
seg["segmenter_model"] = str(seg.get("segmenter_model") or "sam3.1_multiplex_fp16.safetensors")
if not seg.get("segmenter_base_url"):
    seg["segmenter_base_url"] = comfy["base_url"]
seg["workflow_detect"] = str(seg.get("workflow_detect") or "")
seg["workflow_segment"] = str(seg.get("workflow_segment") or "")
seg["render_size"] = int(seg.get("render_size") or 768)
seg["max_views"] = int(seg.get("max_views") or 4)
seg["sequential"] = bool(seg.get("sequential", True))
seg["min_box_score"] = float(seg.get("min_box_score") or 0.12)
seg["mask_threshold"] = float(seg.get("mask_threshold") or 0.5)
seg["projection_min_views"] = int(seg.get("projection_min_views") or 2)
seg["debug_emit_every_step"] = bool(seg.get("debug_emit_every_step", True))
seg["free_gpu_between_steps"] = bool(seg.get("free_gpu_between_steps", True))
data["segmentation"] = seg

cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False), encoding="utf-8")
print(cfg_path)
"@ | Set-Content -Path $tmp -Encoding utf8

    $env:MF_ROOT = $ProjectRoot
    $env:MF_PROJECTS = (Join-Path $ProjectRoot "projects")
    $env:MF_COMFY_ROOT = $ComfyRoot
    $env:MF_GPU_VRAM_GB = "$GpuVramGb"
    $env:MF_SEGMENTATION_ENABLED = if ($SegmentationEnabled) { "1" } else { "0" }
    $env:MF_SEGMENTATION_DETECTOR = $SegmentationDetector
    $env:MF_SEGMENTATION_MODEL = $SegmentationModel
    try {
        & $UvExe run python $tmp
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
        Remove-Item Env:MF_ROOT, Env:MF_PROJECTS, Env:MF_COMFY_ROOT, Env:MF_GPU_VRAM_GB, Env:MF_SEGMENTATION_ENABLED, Env:MF_SEGMENTATION_DETECTOR, Env:MF_SEGMENTATION_MODEL -ErrorAction SilentlyContinue
    }
}

Write-Host "== MeshForge setup (uv) ==" -ForegroundColor Cyan
$uvExe = Ensure-Uv
Write-Host "Using uv: $uvExe"

Set-Location $root
Write-Host "Syncing Python environment..." -ForegroundColor Yellow
& $uvExe sync

$configExample = Join-Path $root "config.yaml.example"
$configPath = Join-Path $root "config.yaml"
if (-not (Test-Path $configPath) -and (Test-Path $configExample)) {
    Copy-Item $configExample $configPath
    Write-Host "Created config.yaml from example." -ForegroundColor Yellow
}

if (-not $SkipComfyUI) {
    Write-Host "Setting up ComfyUI..." -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot "setup-comfyui.ps1") `
        -Gpu $Gpu `
        -SkipCheckpoints:$SkipCheckpoints `
        -WithSegmentation:$WithSegmentation `
        -SegmentationDetector $SegmentationDetector `
        -SkipSegmentationNodes:$SkipSegmentationNodes `
        -SkipSegmentationSmoke:$SkipSegmentationSmoke
}

$layout = Find-ComfyLayout -ProjectRoot $root
$resolvedComfy = if (Test-ComfyLayoutReady $layout) { $layout.InstallDir } else { "" }

$gpuVram = Get-GpuMemoryHintGb
Write-Host "Refreshing local config..." -ForegroundColor Yellow
Update-LocalConfig `
    -UvExe $uvExe `
    -ProjectRoot $root `
    -ComfyRoot $resolvedComfy `
    -GpuVramGb $gpuVram `
    -SegmentationEnabled:$WithSegmentation `
    -SegmentationDetector $SegmentationDetector `
    -SegmentationModel $SegmentationModel

Write-Host ""
Write-Host "Manual services to verify:" -ForegroundColor Cyan
Write-Host "  1. LM Studio local server at http://127.0.0.1:1234/v1"
Write-Host "  2. ComfyUI API at http://127.0.0.1:8188"
Write-Host "  3. config.yaml values for projects / comfyui.install_dir / gpu.vram_gb"
Write-Host ""
Write-Host "Useful scripts:" -ForegroundColor Cyan
Write-Host "  .\scripts\start-comfyui.ps1"
Write-Host "  .\scripts\stop-comfyui.ps1"
Write-Host "  .\scripts\setup-segmentation.ps1"
Write-Host "  .\scripts\check-segmentation.ps1"
Write-Host "  .\scripts\run.ps1"
Write-Host ""

if ($StartComfyUI) {
    & (Join-Path $PSScriptRoot "start-comfyui.ps1")
}

Write-Host "Next step: .\scripts\start-comfyui.ps1 ; .\scripts\run.ps1" -ForegroundColor Green
