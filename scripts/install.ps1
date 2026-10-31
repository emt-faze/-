param(
    [switch]$WithModel,
    [switch]$ModelScope,
    [string]$PythonPath = ""
)

$ErrorActionPreference = 'Stop'
$PackageRoot = Split-Path -Parent $PSScriptRoot
$AppRoot = Join-Path $PackageRoot 'wuwa-dubber'
$ModelRoot = Join-Path $PackageRoot 'local-voice-model'
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'

try {
    $BootstrapArguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        (Join-Path $AppRoot 'scripts\bootstrap.ps1'))
    if ($PythonPath) { $BootstrapArguments += @('-PythonPath', $PythonPath) }
    & powershell.exe @BootstrapArguments
    if ($LASTEXITCODE -ne 0) { throw 'Application dependency installation failed.' }
    $AppPython = Join-Path $AppRoot '.venv\Scripts\python.exe'
    $Configure = Join-Path $PSScriptRoot 'configure_install.py'
    if (-not $WithModel) {
        & $AppPython $Configure
        if ($LASTEXITCODE -ne 0) { throw 'Could not initialize application settings.' }
        Write-Host '基础环境已就绪。请双击“启动AI配音.bat”，默认使用系统中文朗读。'
        exit 0
    }

    & $AppPython -c "import struct,sys; raise SystemExit(not ((3,11) <= sys.version_info[:2] <= (3,12) and struct.calcsize('P') == 8))"
    if ($LASTEXITCODE -ne 0) { throw 'The model requires 64-bit Python 3.11 or 3.12; Python 3.12 is recommended.' }
    $ModelEnvironment = Join-Path $ModelRoot '.venv'
    $ModelPython = Join-Path $ModelEnvironment 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $ModelPython -PathType Leaf)) {
        if (Test-Path -LiteralPath $ModelEnvironment) {
            throw 'An incomplete model environment already exists. Preserve it and install in a fresh extracted folder.'
        }
        & $AppPython -m venv $ModelEnvironment
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the model environment.' }
    }
    & $ModelPython -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r (Join-Path $ModelRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Model dependency installation failed. Check the connection and retry this installer.' }
    & $ModelPython -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU unavailable; check the NVIDIA driver'; print('GPU:', torch.cuda.get_device_name(0)); x=torch.ones(1,device='cuda'); print('CUDA check:', x.item())"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA validation failed. The system speech mode remains available.' }
    $DownloadArguments = @((Join-Path $ModelRoot 'download_models.py'))
    if ($ModelScope) { $DownloadArguments += '--modelscope-fallback' }
    & $ModelPython @DownloadArguments
    if ($LASTEXITCODE -ne 0) { throw 'Model download incomplete. Retry to resume and verify the download.' }
    $FFmpegExecutable = & $ModelPython -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
    if ($LASTEXITCODE -ne 0 -or -not $FFmpegExecutable) { throw 'FFmpeg dependency verification failed.' }
    & $AppPython $Configure --with-model --ffmpeg ([string]$FFmpegExecutable)
    if ($LASTEXITCODE -ne 0) { throw 'Could not connect the application to the installed model.' }
    Write-Host '角色配音环境已就绪。请启动配音程序，在“声线与音频”中导入参考录音。'
} catch {
    Write-Host ("安装未完成：" + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
