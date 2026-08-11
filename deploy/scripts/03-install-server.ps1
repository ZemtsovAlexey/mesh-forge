# Run on compute server (DESKTOP-HOME)
# Installs: Git, Python, Blender, OpenSCAD, Docker, MeshForge venv, TripoSR Docker image
$ErrorActionPreference = "Continue"
$AIRoot = "C:\AI"
$MeshForgeRoot = Join-Path $AIRoot "mesh-forge"
$TripoRoot = Join-Path $AIRoot "TripoSR"
$LogFile = Join-Path $AIRoot "install-meshforge.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Log "=== MeshForge server install on $env:COMPUTERNAME ==="

function Install-WingetPkg($id, $name) {
    $p = winget list --id $id -e 2>$null
    if ($LASTEXITCODE -eq 0 -and $p -match $id) { Log "$name already installed"; return }
    Log "Installing $name..."
    winget install --id $id -e --accept-package-agreements --accept-source-agreements --silent
}

Install-WingetPkg "Git.Git" "Git"
Install-WingetPkg "Python.Python.3.11" "Python 3.11"
Install-WingetPkg "BlenderFoundation.Blender" "Blender"
Install-WingetPkg "OpenSCAD.OpenSCAD" "OpenSCAD"
Install-WingetPkg "Docker.DockerDesktop" "Docker Desktop"

$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path","User")

New-Item -ItemType Directory -Force -Path $AIRoot, $MeshForgeRoot, (Join-Path $MeshForgeRoot "projects") | Out-Null

$venv = Join-Path $MeshForgeRoot "venv"
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python311\python.exe" }
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) { Log "Creating venv..."; & $py -m venv $venv }

$pip = Join-Path $venv "Scripts\pip.exe"
$python = Join-Path $venv "Scripts\python.exe"
Log "Upgrading pip..."; & $pip install --upgrade pip wheel setuptools

$reqPath = Join-Path $PSScriptRoot "..\..\requirements-server.txt"
if (-not (Test-Path $reqPath)) { $reqPath = Join-Path $AIRoot "mesh-forge-remote\requirements-server.txt" }
if (Test-Path $reqPath) { Log "pip install -r requirements-server.txt"; & $pip install -r $reqPath }
else { & $pip install fastapi uvicorn trimesh numpy pillow pyyaml httpx openai rembg pymeshlab open3d einops onnxruntime }

if (-not (Test-Path (Join-Path $TripoRoot ".git"))) {
    Log "Cloning TripoSR (reference only; inference runs in Docker)..."
    git clone https://github.com/VAST-AI-Research/TripoSR.git $TripoRoot
}

$buildScript = Join-Path $MeshForgeRoot "docker\triposr\build.ps1"
if ((Get-Command docker -ErrorAction SilentlyContinue) -and (Test-Path $buildScript)) {
    Log "Building TripoSR Docker image (first run may take 10-20 min)..."
    & $buildScript
    if ($LASTEXITCODE -ne 0) { Log "WARN: Docker image build failed - run docker\triposr\build.ps1 manually" }
} else {
    Log "WARN: Docker not available yet - build meshforge/triposr after Docker Desktop starts"
}

$blender = Get-ChildItem "C:\Program Files\Blender Foundation" -Recurse -Filter "blender.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
$openscad = Get-ChildItem "C:\Program Files\OpenSCAD","C:\Program Files (x86)\OpenSCAD" -Filter "openscad.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
$b = if ($blender) { $blender.FullName.Replace('\','/') } else { "" }
$o = if ($openscad) { $openscad.FullName.Replace('\','/') } else { "" }
$t = $TripoRoot.Replace('\','/')
$p = (Join-Path $MeshForgeRoot "projects").Replace('\','/')
$h = (Join-Path $MeshForgeRoot ".cache\huggingface").Replace('\','/')

@"
llm:
  provider: lmstudio
  base_url: http://127.0.0.1:1234/v1
  api_key: lm-studio
  planner_model: qwen2.5-7b-instruct
  vision_model: qwen2.5-vl-7b-instruct
paths:
  blender: "$b"
  openscad: "$o"
  triposr: "$t"
  projects: "$p"
server:
  host: 0.0.0.0
  port: 7860
gpu:
  vram_gb: 8
  sequential_models: true
docker:
  enabled: true
  triposr_image: meshforge/triposr:latest
  hf_cache: "$h"
"@ | Set-Content (Join-Path $MeshForgeRoot "config.yaml") -Encoding UTF8

try { nvidia-smi 2>&1 | ForEach-Object { Log $_ } } catch { Log "nvidia-smi not found" }
& $python -c "import torch; print('CUDA', torch.cuda.is_available())" 2>&1 | ForEach-Object { Log $_ }
Log "Install complete. Log: $LogFile"
