# One-time: register (or re-register) the "RenderFlow API" scheduled task so the
# dashboard starts hidden every time this user logs on to Windows.
# Run:   powershell -ExecutionPolicy Bypass -File deploy\windows\install_autostart.ps1
# Undo:  Unregister-ScheduledTask -TaskName "RenderFlow API" -Confirm:$false
# Uses the ScheduledTasks cmdlets, not `schtasks /Create /SC ONLOGON` — the
# latter returns "Access is denied" from a non-elevated shell, while the
# cmdlets register a current-user logon task without admin rights.
$script = Join-Path $PSScriptRoot "start_api.ps1"
$user = "$env:USERDOMAIN\$env:USERNAME"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
# No 72-hour kill (the API runs for as long as the session does), and keep it
# going on battery — this is a laptop that publishes at 6pm from wherever it is.
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "RenderFlow API" -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force | Format-List TaskName, State
