@echo off
chcp 65001 >nul
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0local-voice-model\scripts\stop-server.ps1"
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
