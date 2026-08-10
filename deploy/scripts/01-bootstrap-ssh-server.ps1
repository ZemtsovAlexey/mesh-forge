# Run ONCE on compute server (DESKTOP-HOME) as Administrator
# Enables SSH, firewall, and authorizes client SSH key
param(
    [string]$ClientPublicKey = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPQJ/NroSSJxMpAge0HSwY059lH31bB1cYZYzVJzgZAe zemtsovalexey@Alexey"
)
$ErrorActionPreference = "Stop"
Write-Host "=== 01-bootstrap-ssh-server ===" -ForegroundColor Cyan
Write-Host "Computer: $env:COMPUTERNAME  User: $env:USERNAME"

$cap = Get-WindowsCapability -Online | Where-Object { $_.Name -like "OpenSSH.Server*" }
if ($cap.State -ne "Installed") { Add-WindowsCapability -Online -Name $cap.Name }
Set-Service sshd -StartupType Automatic
Start-Service sshd

foreach ($rule in @(
    @{Name="OpenSSH-Server-In-TCP"; Port=22; Display="OpenSSH"},
    @{Name="MeshForge-API-In-TCP"; Port=7860; Display="MeshForge"},
    @{Name="LMStudio-API-In-TCP"; Port=1234; Display="LM Studio"}
)) {
    if (-not (Get-NetFirewallRule -Name $rule.Name -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -Name $rule.Name -DisplayName $rule.Display -Enabled True `
            -Direction Inbound -Protocol TCP -Action Allow -LocalPort $rule.Port | Out-Null
        Write-Host "Firewall: port $($rule.Port)"
    }
}

$userFile = "$env:USERPROFILE\.ssh\authorized_keys"
$adminFile = "C:\ProgramData\ssh\administrators_authorized_keys"
New-Item -Force -ItemType Directory "$env:USERPROFILE\.ssh" | Out-Null
[System.IO.File]::WriteAllText($userFile, $ClientPublicKey.Trim() + "`n")
[System.IO.File]::WriteAllText($adminFile, $ClientPublicKey.Trim() + "`n")
icacls $userFile /inheritance:r /grant "${env:USERNAME}:(F)" /grant "SYSTEM:(F)" | Out-Null
icacls $adminFile /inheritance:r /grant "*S-1-5-32-544:(F)" /grant "SYSTEM:F" | Out-Null

$config = "C:\ProgramData\ssh\sshd_config"
$c = Get-Content $config -Raw
$c = $c -replace "(?m)^#?\s*PubkeyAuthentication.*", "PubkeyAuthentication yes"
$c = $c -replace "(?m)^#?\s*PasswordAuthentication.*", "PasswordAuthentication yes"
Set-Content $config $c -Encoding UTF8
Restart-Service sshd

Write-Host "Done. SSH user for client: $env:USERNAME" -ForegroundColor Green
