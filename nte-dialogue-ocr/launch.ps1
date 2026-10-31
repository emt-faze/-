param([switch]$Admin, [switch]$AutoConnect, [switch]$Dub, [switch]$Generate, [string]$Project)
$ErrorActionPreference = 'Stop'
$ocrRoot = $PSScriptRoot
$ocrPython = Join-Path $ocrRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $ocrPython)) {
    $runtimeFile = Join-Path $ocrRoot 'runtime.json'
    if (Test-Path -LiteralPath $runtimeFile) {
        $runtime = Get-Content -LiteralPath $runtimeFile -Raw -Encoding UTF8 | ConvertFrom-Json
        $ocrPython = [string]$runtime.pythonw_path
    }
}
if (-not (Test-Path -LiteralPath $ocrPython)) {
    $ocrPython = Join-Path (Split-Path -Parent $ocrRoot) 'wuwa-dubber\.venv\Scripts\pythonw.exe'
}
if (-not (Test-Path -LiteralPath $ocrPython)) {
    throw 'Python OCR runtime is missing. See the local usage instructions.'
}
$ocrScript = Join-Path $ocrRoot 'app.py'
if ($Generate) { $ocrScript = Join-Path $ocrRoot 'dialogue_generator_ui.py' }
$ocrArguments = @(('"' + $ocrScript + '"'))
if ($Generate) {
    if ($Project) { $ocrArguments += @('--project', ('"' + $Project + '"')) }
} else {
    if ($AutoConnect) { $ocrArguments += '--autoconnect' }
    if ($Dub) { $ocrArguments += '--dub' }
}
if ($Admin) {
    Start-Process -FilePath $ocrPython -ArgumentList $ocrArguments -WorkingDirectory $ocrRoot -WindowStyle Hidden -Verb RunAs
} else {
    Start-Process -FilePath $ocrPython -ArgumentList $ocrArguments -WorkingDirectory $ocrRoot -WindowStyle Hidden
}
