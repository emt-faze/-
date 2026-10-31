param([switch]$SkipModels, [string]$PythonPath = "")
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ocrRoot = $PSScriptRoot
$ocrPython = Join-Path $ocrRoot '.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'

function Assert-Python312([string]$File, [string[]]$Prefix) {
    & $File @Prefix -c 'import sys; assert sys.version_info[:2]==(3,12) and sys.maxsize>2**32'
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 (64-bit), then run setup.cmd again.' }
}

if (-not (Test-Path -LiteralPath $ocrPython)) {
    $ocrLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($PythonPath) {
        $ocrBasePython = $PythonPath
        $ocrPrefix = @()
    } elseif ($ocrLauncher) {
        $ocrBasePython = $ocrLauncher.Source
        $ocrPrefix = @('-3.12')
    } else {
        $ocrLauncher = Get-Command python.exe -ErrorAction SilentlyContinue
        if (-not $ocrLauncher) { throw 'Python 3.12 (64-bit) is required: https://www.python.org/downloads/windows/' }
        $ocrBasePython = $ocrLauncher.Source
        $ocrPrefix = @()
    }
    Assert-Python312 $ocrBasePython $ocrPrefix
    & $ocrBasePython @ocrPrefix -m venv (Join-Path $ocrRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the local Python environment.' }
}
Assert-Python312 $ocrPython @()
& $ocrPython -m pip install -r (Join-Path $ocrRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; check the network and run setup.cmd again.' }
& $ocrPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed dependencies are inconsistent.' }

$ocrConfig = Join-Path $ocrRoot 'config.json'
if (-not (Test-Path -LiteralPath $ocrConfig)) {
    Copy-Item -LiteralPath (Join-Path $ocrRoot 'config.example.json') -Destination $ocrConfig
}
if (-not $SkipModels) {
    & $ocrPython -X utf8 (Join-Path $ocrRoot 'scripts\download_models.py')
    if ($LASTEXITCODE -ne 0) { throw 'Model preparation failed. Run setup.cmd again to retry.' }
}
Write-Host 'OCR is ready. Open start-ocr.cmd, or use the root dialogue-generator launcher after its installer finishes.'
