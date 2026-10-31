param([string]$PythonPath = "", [switch]$SkipModels)
$ErrorActionPreference = 'Stop'
$PackageRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'
try {
    if (-not $PythonPath) {
        $Launcher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($Launcher) {
            $PythonPath = & $Launcher.Source -3.12 -c 'import sys; print(sys.executable)'
            if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 (64-bit) first.' }
        } else {
            $Launcher = Get-Command python.exe -ErrorAction SilentlyContinue
            if (-not $Launcher) { throw 'Python 3.12 (64-bit) is required.' }
            $PythonPath = $Launcher.Source
        }
    }
    & $PythonPath -c 'import sys; assert sys.version_info[:2] == (3,12) and sys.maxsize > 2**32'
    if ($LASTEXITCODE -ne 0) { throw 'The dialogue generator requires 64-bit Python 3.12.' }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'install.ps1') -PythonPath $PythonPath
    if ($LASTEXITCODE -ne 0) { throw 'Shared voice environment installation failed.' }
    $SetupArguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        (Join-Path $PackageRoot 'nte-dialogue-ocr\setup.ps1'), '-PythonPath', $PythonPath)
    if ($SkipModels) { $SetupArguments += '-SkipModels' }
    & powershell.exe @SetupArguments
    if ($LASTEXITCODE -ne 0) { throw 'OCR environment installation failed.' }
    & (Join-Path $PackageRoot 'nte-dialogue-ocr\.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'configure_generator.py')
    if ($LASTEXITCODE -ne 0) { throw 'Generator configuration failed.' }
    Write-Host 'Ready. Open the dialogue generator from the root folder. Character cloning additionally requires the voice-model installer.'
} catch {
    Write-Host ('Installation did not finish: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
