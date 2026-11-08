@echo off
rem Double-click friendly launcher for install.ps1 (bypasses the script execution policy for this run only).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if "%~1"=="" pause