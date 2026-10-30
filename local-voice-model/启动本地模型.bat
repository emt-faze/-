@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo 正在检查本地配音模型，服务会在后台运行。
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-server.ps1"
if errorlevel 1 (
    echo.
    echo 请根据上方提示查看日志后再尝试。
)
echo.
pause
endlocal
