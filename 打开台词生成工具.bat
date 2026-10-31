@echo off
chcp 65001 >nul
setlocal
if not exist "%~dp0nte-dialogue-ocr\.venv\Scripts\pythonw.exe" (
  echo 请先运行“安装台词生成工具.bat”。
  pause
  exit /b 1
)
call "%~dp0nte-dialogue-ocr\start-dialogue-generator.cmd"
exit /b %ERRORLEVEL%
