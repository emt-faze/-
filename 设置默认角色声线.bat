@echo off
chcp 65001 >nul
setlocal
if not exist "%~dp0nte-dialogue-ocr\.venv\Scripts\python.exe" (
  echo 请先运行“安装台词生成工具.bat”。
  pause
  exit /b 1
)
"%~dp0nte-dialogue-ocr\.venv\Scripts\python.exe" -X utf8 "%~dp0scripts\choose_default_voice.py"
exit /b %ERRORLEVEL%
