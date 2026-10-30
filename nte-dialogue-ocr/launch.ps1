param([switch]$Admin, [switch]$AutoConnect)
$ErrorActionPreference = 'Stop'
$ocrRoot = $PSScriptRoot
$ocrPython = Join-Path $ocrRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $ocrPython)) {
    throw 'Run setup.cmd first to create this tool''s own Python environment.'
}
$ocrScript = Join-Path $ocrRoot 'app.py'
$ocrArguments = @(('"' + $ocrScript + '"'))
if ($AutoConnect) { $ocrArguments += '--autoconnect' }
if ($Admin) {
    Start-Process -FilePath $ocrPython -ArgumentList $ocrArguments -WorkingDirectory $ocrRoot -WindowStyle Hidden -Verb RunAs
} else {
    Start-Process -FilePath $ocrPython -ArgumentList $ocrArguments -WorkingDirectory $ocrRoot -WindowStyle Hidden
}
