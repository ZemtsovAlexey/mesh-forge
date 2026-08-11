#Requires -Version 5.1
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Image = if ($env:MESHFORGE_TRIPOSR_IMAGE) { $env:MESHFORGE_TRIPOSR_IMAGE } else { "meshforge/triposr:latest" }

function Test-DockerReady {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker daemon is not running. Start Docker Desktop and wait until it is ready."
    }
}

function Repair-DockerCredentials {
    $cfgPath = Join-Path $env:USERPROFILE ".docker\config.json"
    if (-not (Test-Path $cfgPath)) { return }
    $raw = Get-Content $cfgPath -Raw
    if ($raw -notmatch '"credsStore"') { return }
    Write-Warning "Broken docker credsStore detected; removing it for public image pulls."
    $bak = "$cfgPath.bak"
    if (-not (Test-Path $bak)) { Copy-Item $cfgPath $bak -Force }
    $cfg = $raw | ConvertFrom-Json
    $cfg.PSObject.Properties.Remove("credsStore") | Out-Null
    $out = ($cfg | ConvertTo-Json -Depth 5 -Compress:$false)
    [System.IO.File]::WriteAllText($cfgPath, $out, [System.Text.UTF8Encoding]::new($false))
}

Test-DockerReady
Repair-DockerCredentials

Write-Host "Building $Image from $Root ..."
docker build -f (Join-Path $PSScriptRoot "Dockerfile") -t $Image $Root
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done: $Image"
docker images $Image