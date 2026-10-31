@echo off
chcp 65001 >nul
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" -WithModel
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
