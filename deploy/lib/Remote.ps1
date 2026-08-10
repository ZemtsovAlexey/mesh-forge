function Get-DeployConfig {
    param([string]$ConfigPath)
    if (-not $ConfigPath) {
        $ConfigPath = Join-Path (Split-Path $PSScriptRoot -Parent) "deploy.config.json"
    }
    if (-not (Test-Path $ConfigPath)) {
        throw "Config not found: $ConfigPath. Copy deploy.config.example.json to deploy.config.json"
    }
    return Get-Content $ConfigPath -Raw | ConvertFrom-Json
}

function Invoke-RemotePowerShell {
    param(
        [Parameter(Mandatory)][string]$RemoteUser,
        [Parameter(Mandatory)][string]$RemoteHost,
        [Parameter(Mandatory)][string]$Script,
        [int]$SshPort = 22
    )
    $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Script))
    $target = "${RemoteUser}@${RemoteHost}"
    $result = ssh -p $SshPort -o BatchMode=yes -o ConnectTimeout=20 $target "powershell.exe -NoProfile -EncodedCommand $enc" 2>&1
    return $result
}

function Send-RemoteFile {
    param(
        [Parameter(Mandatory)][string]$LocalPath,
        [Parameter(Mandatory)][string]$RemotePath,
        [Parameter(Mandatory)][string]$RemoteUser,
        [Parameter(Mandatory)][string]$RemoteHost,
        [int]$SshPort = 22
    )
    $bytes = [IO.File]::ReadAllBytes($LocalPath)
    $b64 = [Convert]::ToBase64String($bytes)
    $remoteDir = Split-Path $RemotePath -Parent
    $fileName = Split-Path $RemotePath -Leaf
    $winPath = $RemotePath -replace '/', '\'
    $winDir = $remoteDir -replace '/', '\'
    $script = @"
New-Item -Force -ItemType Directory '$winDir' | Out-Null
[IO.File]::WriteAllBytes('$winPath', [Convert]::FromBase64String('$b64'))
Write-Host "Uploaded: $winPath ($($bytes.Length) bytes)"
"@
    Invoke-RemotePowerShell -RemoteUser $RemoteUser -RemoteHost $RemoteHost -SshPort $SshPort -Script $script | Out-Null
}

function Test-RemoteSsh {
    param([string]$RemoteUser, [string]$RemoteHost, [int]$SshPort = 22)
    ssh -p $SshPort -o BatchMode=yes -o ConnectTimeout=10 "${RemoteUser}@${RemoteHost}" "hostname" 2>$null
    return $LASTEXITCODE -eq 0
}
