# Run on client - verify deployment
param([string]$ConfigPath)
$deployRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $deployRoot "lib\Remote.ps1")
$cfg = Get-DeployConfig -ConfigPath $ConfigPath

Write-Host "=== 05-verify-deployment ===" -ForegroundColor Cyan
$checks = @()

if (Test-RemoteSsh -RemoteUser $cfg.remote.user -RemoteHost $cfg.remote.host -SshPort $cfg.remote.sshPort) {
    $checks += "[OK] SSH"
} else { $checks += "[FAIL] SSH" }

$script = @"
`$ok = @()
if (Test-Path C:\AI\mesh-forge\config.yaml) { `$ok += 'config' }
if (Test-Path C:\AI\mesh-forge\venv\Scripts\python.exe) { `$ok += 'venv' }
if (Test-Path C:\AI\TripoSR\run.py) { `$ok += 'triposr' }
try { `$r = & C:\AI\mesh-forge\venv\Scripts\python.exe -c 'import torch; print(torch.cuda.is_available())'; `$ok += "cuda:`$r" } catch { `$ok += 'cuda:fail' }
`$ok -join ','
"@
$remote = Invoke-RemotePowerShell -RemoteUser $cfg.remote.user -RemoteHost $cfg.remote.host -SshPort $cfg.remote.sshPort -Script $script
$checks += "[REMOTE] $remote"

try {
    $r = Invoke-WebRequest -Uri "http://$($cfg.remote.host):$($cfg.ports.lmstudio)/v1/models" -TimeoutSec 5 -UseBasicParsing
    $checks += "[OK] LM Studio API"
} catch { $checks += "[WARN] LM Studio API not reachable (install models + start server)" }

$checks | ForEach-Object { Write-Host $_ }
