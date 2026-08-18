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
& (Join-Path $PSScriptRoot "setup-comfyui.ps1") `
    -InstallDir $InstallDir `
    -WithSegmentation `
    -SegmentationDetector $SegmentationDetector `
    -SkipSegmentationNodes:$SkipSegmentationNodes `
    -SkipSegmentationSmoke:$SkipSegmentationSmoke
