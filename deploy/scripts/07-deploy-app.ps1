param([string]$ConfigPath)
$ErrorActionPreference = "Stop"
$deployRoot = Split-Path $PSScriptRoot -Parent
$projectRoot = Split-Path $deployRoot -Parent
. (Join-Path $deployRoot "lib\Remote.ps1")
$cfg = Get-DeployConfig -ConfigPath $ConfigPath
$ru = $cfg.remote.user
$rh = $cfg.remote.host
$port = $cfg.remote.sshPort
$appRoot = ($cfg.paths.remoteAppRoot -replace '\\','/')

Write-Host "=== 07-deploy-app ===" -ForegroundColor Cyan
if (-not (Test-RemoteSsh -RemoteUser $ru -RemoteHost $rh -SshPort $port)) { Write-Host "SSH failed"; exit 1 }

$skip = @('.git', 'venv', 'projects', '__pycache__')
$files = Get-ChildItem $projectRoot -Recurse -File | Where-Object {
    $rel = $_.FullName.Substring($projectRoot.Length + 1)
    $skipRel = $false
    foreach ($s in $skip) { if ($rel -like "$s*" -or $rel -like "*\$s\*") { $skipRel = $true } }
    if ($skipRel) { return $false }
    if ($rel -like "*__pycache__*" -or $rel -like "*.pyc") { return $false }
    if ($rel -eq 'deploy\deploy.config.json') { return $false }
    return $true
}

Write-Host "Uploading $($files.Count) files..."
foreach ($f in $files) {
    $rel = $f.FullName.Substring($projectRoot.Length + 1).Replace('\','/')
    Send-RemoteFile -LocalPath $f.FullName -RemotePath "$appRoot/$rel" -RemoteUser $ru -RemoteHost $rh -SshPort $port
    Write-Host "  $rel"
}

$script = "& 'C:\AI\mesh-forge\venv\Scripts\pip.exe' install -r 'C:\AI\mesh-forge\requirements.txt'"
Invoke-RemotePowerShell -RemoteUser $ru -RemoteHost $rh -SshPort $port -Script $script | Out-Null
Write-Host "Done. UI: http://$($cfg.remote.host):7860"
