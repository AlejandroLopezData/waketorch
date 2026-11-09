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
function Done($message) { Write-Host "OK " -ForegroundColor Green -NoNewline; Write-Host $message }
function Fail($message) {
    Write-Progress -Activity "Installing WakeTorch" -Completed
    Write-Host "X $message" -ForegroundColor Red
    exit 1
}

# Runs a command showing a progress bar and the package being installed.
# The bar moves from $Base% to $Base+$Span% while pip collects packages.
function Invoke-Step([string]$Label, [int]$Base, [int]$Span, [string]$File, [string[]]$ArgList, [string]$Hint = "") {
    $out = Join-Path $env:TEMP "waketorch-install.out.log"
    $err = Join-Path $env:TEMP "waketorch-install.err.log"
    Remove-Item $out, $err -ErrorAction SilentlyContinue
    $quoted = @($ArgList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } })
    $proc = Start-Process -FilePath $File -ArgumentList $quoted -WorkingDirectory $Root -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $proc.Handle   # keeps the exit code available after the process ends
    while (-not $proc.HasExited) {
        $pkg = ""
        $seen = 0
        if (Test-Path $out) {
            $collecting = @(Get-Content $out -ErrorAction SilentlyContinue | Where-Object { $_ -like "Collecting *" })
            $seen = [Math]::Min($collecting.Count, 30)
            if ($collecting.Count -gt 0) {
                $pkg = (($collecting[-1] -split " ")[1]) -replace '[<>=!~;(\[].*$', ''
            }
        }
        $pct = $Base + [int]($Span * $seen / 30)
        Write-Progress -Activity "Installing WakeTorch" -Status "$Label  $pkg" -PercentComplete $pct
        Start-Sleep -Milliseconds 200
    }
    $proc.WaitForExit()
    if ($proc.ExitCode -ne 0) {
        Get-Content $out, $err -ErrorAction SilentlyContinue | Select-Object -Last 15 | ForEach-Object { Write-Host $_ }
        Fail "$Label failed. $Hint"
    }
    Write-Progress -Activity "Installing WakeTorch" -Status $Label -PercentComplete ($Base + $Span)
    Done $Label
}

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
Write-Host ""
Say ("WakeTorch installer (" + (& $pyExe @pyExtra --version) + ")")
Write-Host ""

# 2) Environment + dependencies (from pyproject.toml)
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
Invoke-Step "Creating the environment" 5 10 $pyExe (@($pyExtra) + @("-m", "venv", ".venv"))
Invoke-Step "Updating pip" 15 5 $venvPython @("-m", "pip", "install", "--disable-pip-version-check", "--progress-bar", "off", "--upgrade", "pip")
Say "PyTorch is large: the dependencies can take a few minutes."
Invoke-Step "Installing dependencies" 20 75 $venvPython @("-m", "pip", "install", "--disable-pip-version-check", "--progress-bar", "off", "-e", ".")

# 3) `waketorch` command (a small launcher in a folder that is added to your PATH)
New-Item -ItemType Directory -Force $BinDir | Out-Null
$exePath = Join-Path $Root ".venv\Scripts\waketorch.exe"
Set-Content -Path $Shim -Encoding Oem -Value "@echo off`r`nset PYTHONUTF8=1`r`n`"$exePath`" %*"

# 4) Make sure the launcher folder is in your PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$needsRestart = $false
if ((($userPath -split ";") -notcontains $BinDir)) {
    [Environment]::SetEnvironmentVariable("Path", (($userPath, $BinDir | Where-Object { $_ }) -join ";"), "User")
    $needsRestart = $true
}
Write-Progress -Activity "Installing WakeTorch" -Completed
Done "Creating the waketorch command"

Write-Host ""
Say "Done. Run it with:  waketorch"
if ($needsRestart) { Say "Open a new terminal first so the PATH change takes effect." }
Say "Tip: use Windows Terminal for the best look."
exit 0