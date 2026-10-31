@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
    echo Installation failed. Read the error above and README.md.
) else (
    echo Installation complete. Open start-ocr.cmd to begin.
)
pause
