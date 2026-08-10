# Run on client (Z13) - uploads scripts and runs install on remote server
param([string]$ConfigPath)
$ErrorActionPreference = "Stop"
$deployRoot = Split-Path $PSScriptRoot -Parent
$projectRoot = Split-Path $deployRoot -Parent
. (Join-Path $deployRoot "lib\Remote.ps1")

$cfg = Get-DeployConfig -ConfigPath $ConfigPath
$ru = $cfg.remote.user
$rh = $cfg.remote.host
$port = $cfg.remote.sshPort
$remoteDir = $cfg.paths.remoteInstallDir -replace '\\','/'

Write-Host "=== 04-deploy-remote ===" -ForegroundColor Cyan
Write-Host "Target: ${ru}@${rh}"

if (-not (Test-RemoteSsh -RemoteUser $ru -RemoteHost $rh -SshPort $port)) {
    Write-Host "SSH failed. Run:" -ForegroundColor Red
    Write-Host "  1) On server: deploy\scripts\01-bootstrap-ssh-server.ps1 (Admin)"
    Write-Host "  2) On client: deploy\scripts\02-copy-ssh-key.ps1"
    exit 1
}
Write-Host "SSH OK" -ForegroundColor Green

Send-RemoteFile -LocalPath (Join-Path $PSScriptRoot "03-install-server.ps1") `
    -RemotePath "$remoteDir/03-install-server.ps1" -RemoteUser $ru -RemoteHost $rh -SshPort $port
Send-RemoteFile -LocalPath (Join-Path $projectRoot "requirements-server.txt") `
    -RemotePath "$remoteDir/requirements-server.txt" -RemoteUser $ru -RemoteHost $rh -SshPort $port

Write-Host "Starting remote install (15-30 min)..."
$run = @"
`$log = 'C:\AI\install-stdout.log'
Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','C:\AI\mesh-forge-remote\03-install-server.ps1' -RedirectStandardOutput `$log -RedirectStandardError 'C:\AI\install-stderr.log' -WindowStyle Hidden
Start-Sleep 3
Get-Content C:\AI\install-meshforge.log -Tail 5 -ErrorAction SilentlyContinue
"@
Invoke-RemotePowerShell -RemoteUser $ru -RemoteHost $rh -SshPort $port -Script $run
Write-Host "Install started in background. Monitor:" -ForegroundColor Yellow
Write-Host "  ssh ${ru}@${rh} `"powershell -Command Get-Content C:\AI\install-meshforge.log -Tail 20 -Wait`""
