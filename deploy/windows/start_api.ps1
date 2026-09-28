# Starts the RenderFlow dashboard API on this Windows dev box, hidden, at logon.
# Registered as a scheduled task by install_autostart.ps1 — don't run by hand
# unless the task isn't installed. Safe to re-run: exits if the API already
# answers on its port, so a second logon/trigger never spawns a duplicate.
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $repo ".venv\Scripts\python.exe"
$log = Join-Path $repo "logs_api.log"

# The repo lives inside OneDrive, which can mount a little after logon — wait
# (up to ~2 minutes) for the venv to actually be reachable before starting.
$tries = 0
while (-not (Test-Path $python) -and $tries -lt 24) { Start-Sleep -Seconds 5; $tries++ }
if (-not (Test-Path $python)) { exit 1 }

# Already running? (e.g. started by hand in a terminal) — leave it alone.
try {
    $probe = Invoke-WebRequest -Uri "http://127.0.0.1:8321/api/auth/dev-login" -UseBasicParsing -TimeoutSec 3
    if ($probe.StatusCode -eq 200) { exit 0 }
} catch {}

Set-Location $repo
$env:PYTHONIOENCODING = "utf-8"
# Foreground on purpose: the scheduled task stays "Running" for as long as the
# API is alive, so Task Scheduler's history shows whether it's up.
& $python -m renderflow.api *>> $log
