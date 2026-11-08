# WakeTorch installer (Windows)
#   .\install.ps1             install: creates .venv, installs dependencies and the `waketorch` command
#   .\install.ps1 uninstall   removes the command and .venv (your projects are NOT touched)
param([string]$Action = "install")
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
$Root = (Get-Location).Path
$BinDir = Join-Path $env:LOCALAPPDATA "WakeTorch\bin"
$Shim = Join-Path $BinDir "waketorch.cmd"

function Say($message) { Write-Host "> " -ForegroundColor DarkYellow -NoNewline; Write-Host $message }
function Fail($message) { Write-Host "X $message" -ForegroundColor Red; exit 1 }

if ($Action -eq "uninstall") {
    if (Test-Path $BinDir) { Remove-Item -Recurse -Force $BinDir }
    $parts = ([Environment]::GetEnvironmentVariable("Path", "User") -split ";") | Where-Object { $_ -and $_ -ne $BinDir }
    [Environment]::SetEnvironmentVariable("Path", ($parts -join ";"), "User")
    if (Test-Path (Join-Path $Root ".venv")) { Remove-Item -Recurse -Force (Join-Path $Root ".venv") }
    Say "Removed the waketorch command and .venv. Your projects were not touched."
    exit 0
}

# 1) Python 3.10 or newer
$Python = $null
$candidates = @(@("py", "-3.13"), @("py", "-3.12"), @("py", "-3.11"), @("py", "-3.10"), @("python"))
foreach ($c in $candidates) {
    $exe = $c[0]
    $extra = @($c | Select-Object -Skip 1)
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    & $exe @extra -c "import sys; sys.exit(sys.version_info < (3, 10))" 2>$null
    if ($LASTEXITCODE -eq 0) { $Python = $c; break }
}
if (-not $Python) { Fail "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH')." }
$pyExe = $Python[0]
$pyExtra = @($Python | Select-Object -Skip 1)
Say ("Using " + (& $pyExe @pyExtra --version))

# 2) Virtual environment + dependencies (from pyproject.toml)
Say "Creating the environment (.venv)"
& $pyExe @pyExtra -m venv .venv
if ($LASTEXITCODE -ne 0) { Fail "Could not create the environment." }
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
Say "Installing dependencies (PyTorch is large, this can take a few minutes)"
& $venvPython -m pip install --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) { Fail "Could not update pip." }
& $venvPython -m pip install --quiet -e .
if ($LASTEXITCODE -ne 0) { Fail "Could not install the dependencies." }

# 3) `waketorch` command (a small launcher in a folder that is added to your PATH)
New-Item -ItemType Directory -Force $BinDir | Out-Null
$exePath = Join-Path $Root ".venv\Scripts\waketorch.exe"
Set-Content -Path $Shim -Encoding Oem -Value "@echo off`r`nset PYTHONUTF8=1`r`n`"$exePath`" %*"
Say "Command created: $Shim"

# 4) Make sure the launcher folder is in your PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$needsRestart = $false
if ((($userPath -split ";") -notcontains $BinDir)) {
    [Environment]::SetEnvironmentVariable("Path", (($userPath, $BinDir | Where-Object { $_ }) -join ";"), "User")
    $needsRestart = $true
}

Write-Host ""
Say "Done. Run it with:  waketorch"
if ($needsRestart) { Say "Open a new terminal first so the PATH change takes effect." }
Say "Tip: use Windows Terminal for the best look."
exit 0