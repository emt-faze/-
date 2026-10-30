@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
".venv\Scripts\python.exe" "scripts\launch_app.py"
if errorlevel 1 pause
endlocal
