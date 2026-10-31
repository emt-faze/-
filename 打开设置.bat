@echo off
chcp 65001 >nul
setlocal
call "%~dp0wuwa-dubber\start-wuwa-dubber.bat"
exit /b %ERRORLEVEL%
