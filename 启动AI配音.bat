@echo off
chcp 65001 >nul
setlocal
if not exist "%~dp0wuwa-dubber\.venv\Scripts\python.exe" (
    echo 请先双击“安装基础环境.bat”或“安装角色配音模型.bat”。
    pause
    exit /b 1
)
call "%~dp0wuwa-dubber\start-ai-dubbing.bat"
exit /b %ERRORLEVEL%
