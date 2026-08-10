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

function Get-RemoteTarget {
    param(
        [string]$RemoteUser,
        [string]$RemoteHost,
        [int]$SshPort = 22
    )
    return @{
        SshTarget = "${RemoteUser}@${RemoteHost}"
        SshArgs   = @('-p', "$SshPort", '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30')
    }
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
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $result = ssh -p $SshPort -o BatchMode=yes -o ConnectTimeout=30 $target "powershell.exe -NoProfile -EncodedCommand $enc" 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Remote command failed (exit $LASTEXITCODE): $($result | Out-String)"
        }
        return $result
    }
    finally {
        $ErrorActionPreference = $prev
    }
}

function Invoke-RemoteScriptStdin {
    param(
        [Parameter(Mandatory)][string]$Script,
        [Parameter(Mandatory)][string]$RemoteUser,
        [Parameter(Mandatory)][string]$RemoteHost,
        [int]$SshPort = 22
    )
    $rt = Get-RemoteTarget -RemoteUser $RemoteUser -RemoteHost $RemoteHost -SshPort $SshPort
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $Script | & ssh @($rt.SshArgs) $rt.SshTarget "powershell.exe -NoProfile -Command -" 2>&1 | ForEach-Object { Write-Verbose $_; $_ }
        if ($LASTEXITCODE -ne 0) {
            throw "Remote script failed (exit $LASTEXITCODE)"
        }
    }
    finally {
        $ErrorActionPreference = $prev
    }
}

function New-RemoteB64WriteScript {
    param(
        [Parameter(Mandatory)][byte[]]$Bytes,
        [Parameter(Mandatory)][string]$RemotePath,
        [int]$ChunkSize = 5000
    )
    $b64 = [Convert]::ToBase64String($Bytes)
    $winPath = $RemotePath -replace '/', '\'
    $winDir = (Split-Path $winPath -Parent) -replace '/', '\'
    $lines = New-Object System.Collections.Generic.List[string]
    if ($winDir) {
        [void]$lines.Add("New-Item -Force -ItemType Directory '$winDir' | Out-Null")
    }
    [void]$lines.Add("`$b = ''")
    for ($i = 0; $i -lt $b64.Length; $i += $ChunkSize) {
        $chunk = $b64.Substring($i, [Math]::Min($ChunkSize, $b64.Length - $i))
        [void]$lines.Add("`$b += '$chunk'")
    }
    [void]$lines.Add("[IO.File]::WriteAllBytes('$winPath', [Convert]::FromBase64String(`$b))")
    [void]$lines.Add("Write-Host ('Wrote $winPath (' + (Get-Item '$winPath').Length + ' bytes)')")
    return ($lines -join "`n")
}

function Send-RemoteFile {
    param(
        [Parameter(Mandatory)][string]$LocalPath,
        [Parameter(Mandatory)][string]$RemotePath,
        [Parameter(Mandatory)][string]$RemoteUser,
        [Parameter(Mandatory)][string]$RemoteHost,
        [int]$SshPort = 22
    )
    if (-not (Test-Path $LocalPath)) {
        throw "Local file not found: $LocalPath"
    }
    $bytes = [IO.File]::ReadAllBytes($LocalPath)
    $script = New-RemoteB64WriteScript -Bytes $bytes -RemotePath $RemotePath
    Invoke-RemoteScriptStdin -Script $script -RemoteUser $RemoteUser -RemoteHost $RemoteHost -SshPort $SshPort
}

function Send-RemoteArchive {
    param(
        [Parameter(Mandatory)][System.IO.FileInfo[]]$Files,
        [Parameter(Mandatory)][string]$ProjectRoot,
        [Parameter(Mandatory)][string]$RemoteAppRoot,
        [Parameter(Mandatory)][string]$RemoteUser,
        [Parameter(Mandatory)][string]$RemoteHost,
        [int]$SshPort = 22
    )
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $stage = Join-Path $env:TEMP ("meshforge-deploy-" + [guid]::NewGuid().ToString('n'))
    $zipPath = Join-Path $env:TEMP ("meshforge-deploy-" + [guid]::NewGuid().ToString('n') + '.zip')
    $remoteZip = ($RemoteAppRoot.TrimEnd('/') + '/.deploy.zip') -replace '\\', '/'
    try {
        New-Item -ItemType Directory -Path $stage -Force | Out-Null
        foreach ($f in $Files) {
            $rel = $f.FullName.Substring($ProjectRoot.Length + 1)
            $dest = Join-Path $stage $rel
            $destDir = Split-Path $dest -Parent
            if ($destDir -and -not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            if ($f.Extension -match '^\.(py|yaml|yml|txt|md|json|ps1|scad|mlx|css|js|html)$') {
                $text = [IO.File]::ReadAllText($f.FullName)
                [IO.File]::WriteAllText($dest, $text, [Text.UTF8Encoding]::new($false))
            } else {
                Copy-Item -LiteralPath $f.FullName -Destination $dest -Force
            }
        }
        if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
        [System.IO.Compression.ZipFile]::CreateFromDirectory($stage, $zipPath)

        $zipBytes = [IO.File]::ReadAllBytes($zipPath)
        Write-Host "Archive: $([math]::Round($zipBytes.Length / 1KB, 1)) KB, $($Files.Count) files (1 SSH session)"

        $writeZip = New-RemoteB64WriteScript -Bytes $zipBytes -RemotePath $remoteZip
        $winApp = $RemoteAppRoot -replace '/', '\'
        $winZip = $remoteZip -replace '/', '\'
        $extract = @"

New-Item -Force -ItemType Directory '$winApp' | Out-Null
Expand-Archive -Path '$winZip' -DestinationPath '$winApp' -Force
Remove-Item '$winZip' -Force -ErrorAction SilentlyContinue
Get-ChildItem '$winApp' -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host 'Extracted to $winApp'
"@
        Invoke-RemoteScriptStdin -Script ($writeZip + "`n" + $extract) `
            -RemoteUser $RemoteUser -RemoteHost $RemoteHost -SshPort $SshPort
    }
    finally {
        if (Test-Path $stage) { Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue }
        if (Test-Path $zipPath) { Remove-Item $zipPath -Force -ErrorAction SilentlyContinue }
    }
}

function Initialize-RemoteAppGit {
    param(
        [Parameter(Mandatory)][string]$RemoteAppRoot,
        [Parameter(Mandatory)][string]$RemoteUser,
        [Parameter(Mandatory)][string]$RemoteHost,
        [string]$Branch = "main",
        [string]$RemoteUrl = "https://github.com/ZemtsovAlexey/mesh-forge.git",
        [int]$SshPort = 22
    )
    $winApp = $RemoteAppRoot -replace '/', '\'
    $script = @"
`$app = '$winApp'
`$branch = '$Branch'
`$url = '$RemoteUrl'
if (-not (Test-Path `$app)) { New-Item -Force -ItemType Directory `$app | Out-Null }
Set-Location `$app
if (Test-Path .git) {
    Write-Host 'Git already initialized'
    git remote -v 2>&1
    exit 0
}
git --version 2>&1
git init 2>&1
git remote add origin `$url 2>&1
git fetch origin `$branch 2>&1
git checkout -B `$branch 2>&1
git reset --hard "origin/`$branch" 2>&1
if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }
Write-Host ('Synced to origin/' + `$branch)
git log -1 --oneline 2>&1
"@
    Invoke-RemoteScriptStdin -Script $script -RemoteUser $RemoteUser -RemoteHost $RemoteHost -SshPort $SshPort
}

function Sync-RemoteAppGit {
    param(
        [Parameter(Mandatory)][string]$RemoteAppRoot,
        [Parameter(Mandatory)][string]$RemoteUser,
        [Parameter(Mandatory)][string]$RemoteHost,
        [string]$Branch = "main",
        [string]$RemoteUrl = "https://github.com/ZemtsovAlexey/mesh-forge.git",
        [int]$SshPort = 22,
        [switch]$PushLocalFirst,
        [string]$LocalProjectRoot
    )
    if ($PushLocalFirst -and $LocalProjectRoot) {
        Push-Location $LocalProjectRoot
        try {
            $branch = (git rev-parse --abbrev-ref HEAD 2>$null)
            if ($branch) {
                Write-Host "Pushing local branch '$branch'..."
                git push -u origin $branch 2>&1 | ForEach-Object { Write-Host $_ }
            }
        }
        finally {
            Pop-Location
        }
    }

    $winApp = $RemoteAppRoot -replace '/', '\'
    $script = @"
`$app = '$winApp'
`$branch = '$Branch'
if (-not (Test-Path `$app)) { New-Item -Force -ItemType Directory `$app | Out-Null }
Set-Location `$app
if (-not (Test-Path .git)) {
    Write-Host 'NO_GIT'
    exit 2
}
git fetch origin 2>&1
git checkout `$branch 2>&1
git pull --ff-only origin `$branch 2>&1
if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }
Write-Host 'git pull OK'
"@
    $out = Invoke-RemotePowerShell -RemoteUser $RemoteUser -RemoteHost $RemoteHost -SshPort $SshPort -Script $script
    $text = ($out | Out-String).Trim()
    if ($text -match 'NO_GIT') {
        return $false
    }
    if ($text -match 'fatal:|CONFLICT') {
        throw "git sync failed: $text"
    }
    Write-Host $text
    return $true
}

function Restart-RemoteApp {
    param(
        [Parameter(Mandatory)][string]$RemoteAppRoot,
        [Parameter(Mandatory)][string]$RemoteUser,
        [Parameter(Mandatory)][string]$RemoteHost,
        [int]$SshPort = 22
    )
    $winApp = $RemoteAppRoot -replace '/', '\'
    $script = @"
`$app = '$winApp'
`$py = Join-Path `$app 'venv\Scripts\python.exe'
`$entry = Join-Path `$app 'app.py'

`$listener = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if (`$listener) {
    Write-Host ('Stopping listener pid ' + `$listener.OwningProcess)
    Stop-Process -Id `$listener.OwningProcess -Force -ErrorAction SilentlyContinue
}
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { `$_.CommandLine -like '*mesh-forge*' } |
    ForEach-Object {
        Write-Host ('Stopping pid ' + `$_.ProcessId)
        Stop-Process -Id `$_.ProcessId -Force -ErrorAction SilentlyContinue
    }
Start-Sleep 3
Get-ChildItem `$app -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

if (-not (Test-Path `$py)) { Write-Host 'venv python not found'; exit 1 }
Start-Process -FilePath `$py -ArgumentList `$entry -WorkingDirectory `$app -WindowStyle Hidden | Out-Null
Start-Sleep 20

`$listener = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not `$listener) {
    Write-Host 'UI not listening'
    Get-Content (Join-Path `$app 'ui-stderr.log') -Tail 12 -ErrorAction SilentlyContinue
    exit 1
}
Write-Host ('UI listening on 7860 (pid ' + `$listener.OwningProcess + ')')
"@
    Invoke-RemoteScriptStdin -Script $script -RemoteUser $RemoteUser -RemoteHost $RemoteHost -SshPort $SshPort
}

function Test-RemoteSsh {
    param([string]$RemoteUser, [string]$RemoteHost, [int]$SshPort = 22)
    ssh -p $SshPort -o BatchMode=yes -o ConnectTimeout=10 "${RemoteUser}@${RemoteHost}" "hostname" 2>$null
    return $LASTEXITCODE -eq 0
}

function Get-DeployableAppFiles {
    param([Parameter(Mandatory)][string]$ProjectRoot)
    $skip = @('.git', 'venv', 'projects', '__pycache__')
    return @(Get-ChildItem $ProjectRoot -Recurse -File | Where-Object {
        $rel = $_.FullName.Substring($ProjectRoot.Length + 1)
        foreach ($s in $skip) {
            if ($rel -like "$s*" -or $rel -like "*\$s\*") { return $false }
        }
        if ($rel -like "*__pycache__*" -or $rel -like "*.pyc") { return $false }
        if ($rel -eq 'deploy\deploy.config.json') { return $false }
        return $true
    })
}
