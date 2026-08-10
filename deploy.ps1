param([ValidateSet("bootstrap","copy-key","deploy","verify","redeploy","deploy-app","init-git")][string]$Action = "deploy")
$scriptDir = Join-Path $PSScriptRoot "deploy\scripts"
switch ($Action) {
    "bootstrap" { Write-Host "Run ON SERVER as Admin: $scriptDir\01-bootstrap-ssh-server.ps1"; exit 0 }
    "copy-key"  { & "$scriptDir\02-copy-ssh-key.ps1"; break }
    "deploy"    { & "$scriptDir\04-deploy-remote.ps1"; break }
    "verify"    { & "$scriptDir\05-verify-deployment.ps1"; break }
    "redeploy"  { & "$scriptDir\06-redeploy.ps1"; break }
    "deploy-app"{ & "$scriptDir\07-deploy-app.ps1"; break }
    "init-git"  { & "$scriptDir\08-init-git-server.ps1"; break }
}
