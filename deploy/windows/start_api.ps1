# Starts the RenderFlow dashboard API on this Windows dev box, hidden, at logon,
# then opens the dashboard in Chrome once the API answers.
# Registered as a scheduled task by install_autostart.ps1 — don't run by hand
# unless the task isn't installed. Safe to re-run: if the API already answers
# on its port it just opens the dashboard tab and exits, so a second
# logon/trigger never spawns a duplicate server.
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $repo ".venv\Scripts\python.exe"
$log = Join-Path $repo "logs_api.log"
$url = "http://127.0.0.1:8321"

function Test-ApiUp {
    try {
        return (Invoke-WebRequest -Uri "$url/api/auth/dev-login" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200
    } catch { return $false }
}

function Open-Dashboard {
    # Chrome by name (resolved via the App Paths registry key on a standard
    # install); fall back to whatever the default browser is.
    try { Start-Process "chrome.exe" -ArgumentList $url } catch { Start-Process $url }
}

# The repo lives inside OneDrive, which can mount a little after logon — wait
# (up to ~2 minutes) for the venv to actually be reachable before starting.
$tries = 0
while (-not (Test-Path $python) -and $tries -lt 24) { Start-Sleep -Seconds 5; $tries++ }
if (-not (Test-Path $python)) { exit 1 }

# Already running (e.g. started by hand in a terminal)? Just open the tab.
if (Test-ApiUp) { Open-Dashboard; exit 0 }

# Launch the API hidden via cmd so stdout+stderr both append to one log file
# (Start-Process can't redirect both streams to the same file itself).
$env:PYTHONIOENCODING = "utf-8"
$proc = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c `"`"$python`" -m renderflow.api >> `"$log`" 2>&1`"" `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru

# Wait (up to ~90s) for the API to answer, then open the dashboard.
$tries = 0
while (-not (Test-ApiUp) -and $tries -lt 30 -and -not $proc.HasExited) { Start-Sleep -Seconds 3; $tries++ }
if (Test-ApiUp) { Open-Dashboard }

# Stay alive for as long as the API runs, so Task Scheduler shows the task
# as "Running" while the dashboard is actually up.
Wait-Process -Id $proc.Id
exit $proc.ExitCode
