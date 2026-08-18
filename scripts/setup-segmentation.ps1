param(
    [string]$InstallDir = "",
    [ValidateSet("groundingdino", "owlvit")]
    [string]$SegmentationDetector = "groundingdino",
    [switch]$SkipSegmentationNodes,
    [switch]$SkipSegmentationSmoke
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "setup-segmentation.ps1 now delegates to setup-comfyui.ps1 -WithSegmentation" -ForegroundColor Cyan
Write-Host "It will download sam3.1_multiplex_fp16.safetensors into the live ComfyUI checkpoints folder." -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "setup-comfyui.ps1") `
    -InstallDir $InstallDir `
    -WithSegmentation `
    -SegmentationDetector $SegmentationDetector `
    -SkipSegmentationNodes:$SkipSegmentationNodes `
    -SkipSegmentationSmoke:$SkipSegmentationSmoke
