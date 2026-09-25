# Run by the "hermes-trading reflect" scheduled task every 30 minutes.
# Pulls the Railway worker's state, lets Hermes reflect once enough new trades have
# closed (reflection_every in goal.yaml), and pushes the one-variable change back.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root "reflect.log"

# Scheduled tasks start with the PATH from the registry; make sure the user entries are there.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8  # decode python's UTF-8 output correctly

Set-Location $root
"==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====" | Add-Content $log -Encoding utf8
& uv run python -m hermes_trading.remote reflect --hermes 2>&1 | ForEach-Object { "$_" } | Add-Content $log -Encoding utf8
"exit $LASTEXITCODE" | Add-Content $log -Encoding utf8
