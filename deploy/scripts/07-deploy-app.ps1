param(
    [string]$ConfigPath,
    [ValidateSet('auto', 'git', 'archive')]
    [string]$Method = 'auto',
    [switch]$Push,
    [switch]$NoRestart
)
$ErrorActionPreference = "Stop"
$deployRoot = Split-Path $PSScriptRoot -Parent
$projectRoot = Split-Path $deployRoot -Parent
. (Join-Path $deployRoot "lib\Remote.ps1")
$cfg = Get-DeployConfig -ConfigPath $ConfigPath
$ru = $cfg.remote.user
$rh = $cfg.remote.host
$port = $cfg.remote.sshPort
$appRoot = ($cfg.paths.remoteAppRoot -replace '\\', '/')
$gitBranch = if ($cfg.git.branch) { $cfg.git.branch } else { 'main' }
$gitUrl = if ($cfg.git.remoteUrl) { $cfg.git.remoteUrl } else { 'https://github.com/ZemtsovAlexey/mesh-forge.git' }

Write-Host "=== 07-deploy-app ===" -ForegroundColor Cyan
if (-not (Test-RemoteSsh -RemoteUser $ru -RemoteHost $rh -SshPort $port)) {
    Write-Host "SSH failed"
    exit 1
}

$deployed = $false
$sw = [Diagnostics.Stopwatch]::StartNew()

if ($Method -in @('auto', 'git')) {
    Write-Host "Trying git pull on server..." -ForegroundColor Cyan
    try {
        $deployed = Sync-RemoteAppGit -RemoteAppRoot $appRoot -RemoteUser $ru -RemoteHost $rh -SshPort $port `
            -Branch $gitBranch -RemoteUrl $gitUrl -PushLocalFirst:$Push -LocalProjectRoot $projectRoot
        if ($deployed) {
            Write-Host "Deployed via git in $($sw.Elapsed.TotalSeconds.ToString('0.0'))s" -ForegroundColor Green
        }
    }
    catch {
        if ($Method -eq 'git') { throw }
        Write-Host "git sync skipped: $_" -ForegroundColor Yellow
    }
}

if (-not $deployed) {
    if ($Method -eq 'git') {
        Write-Host "git deploy failed and -Method git was set" -ForegroundColor Red
        exit 1
    }
    $files = Get-DeployableAppFiles -ProjectRoot $projectRoot
    Write-Host "Deploying $($files.Count) files via archive (1 SSH)..." -ForegroundColor Cyan
    Send-RemoteArchive -Files $files -ProjectRoot $projectRoot -RemoteAppRoot $appRoot `
        -RemoteUser $ru -RemoteHost $rh -SshPort $port
    Write-Host "Deployed via archive in $($sw.Elapsed.TotalSeconds.ToString('0.0'))s" -ForegroundColor Green
}

$pip = "& '$($appRoot -replace '/','\')\venv\Scripts\pip.exe' install -q -r '$($appRoot -replace '/','\')\requirements.txt'"
Invoke-RemotePowerShell -RemoteUser $ru -RemoteHost $rh -SshPort $port -Script $pip | Out-Null

if (-not $NoRestart) {
    Write-Host "Restarting UI..." -ForegroundColor Cyan
    Restart-RemoteApp -RemoteAppRoot $appRoot -RemoteUser $ru -RemoteHost $rh -SshPort $port
}

Write-Host "Done. UI: http://$($cfg.remote.host):$($cfg.ports.meshforge)" -ForegroundColor Green
