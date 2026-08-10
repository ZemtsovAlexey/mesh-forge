# Full redeploy: upload scripts + reinstall server stack
param(
    [string]$ConfigPath,
    [switch]$SkipBootstrap
)
$deployRoot = Split-Path $PSScriptRoot -Parent
Write-Host "=== 06-redeploy (full) ===" -ForegroundColor Cyan
if (-not $SkipBootstrap) {
    Write-Host "Note: SSH bootstrap is manual on server (01-bootstrap-ssh-server.ps1)"
}
& (Join-Path $PSScriptRoot "04-deploy-remote.ps1") -ConfigPath $ConfigPath
Start-Sleep 5
& (Join-Path $PSScriptRoot "05-verify-deployment.ps1") -ConfigPath $ConfigPath
Write-Host ""
Write-Host "Redeploy initiated. Wait for install log to show 'Install complete'." -ForegroundColor Green
