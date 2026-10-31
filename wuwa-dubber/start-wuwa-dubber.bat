@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
".venv\Scripts\python.exe" --version >nul 2>&1
if errorlevel 1 (
    echo The local Python environment is missing or no longer valid.
    echo Run PowerShell: .\scripts\bootstrap.ps1
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m wuwa_dubber
if errorlevel 1 pause
endlocal
