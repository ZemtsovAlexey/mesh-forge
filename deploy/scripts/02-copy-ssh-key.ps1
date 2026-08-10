# Run on client (Z13) - copies SSH key to server (password prompt once)
param([string]$ConfigPath)
$lib = Join-Path (Split-Path $PSScriptRoot -Parent) "lib\Remote.ps1"
. $lib
$cfg = Get-DeployConfig -ConfigPath $ConfigPath
$pub = "$env:USERPROFILE\.ssh\id_ed25519.pub"
if (-not (Test-Path $pub)) { ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_ed25519" -N '""' }
$key = (Get-Content $pub -Raw).Trim()
Write-Host "Copying key to $($cfg.remote.user)@$($cfg.remote.host) ..."
ssh -p $cfg.remote.sshPort "$($cfg.remote.user)@$($cfg.remote.host)" @"
powershell -NoProfile -Command "`$f='$env:USERPROFILE\.ssh\authorized_keys'; New-Item -Force -ItemType Directory '$env:USERPROFILE\.ssh' | Out-Null; if (-not (Select-String -Path `$f -Pattern 'Alexey' -Quiet -ErrorAction SilentlyContinue)) { Add-Content `$f '$key' }"
"@
if ($LASTEXITCODE -eq 0) { Write-Host "OK" -ForegroundColor Green } else { Write-Host "Failed" -ForegroundColor Red; exit 1 }
