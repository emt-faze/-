param(
    [switch]$CoreOnly,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $ProjectRoot
$EnvironmentRoot = Join-Path $ProjectRoot ".venv"
$EnvironmentPython = Join-Path $EnvironmentRoot "Scripts\python.exe"
$env:PYTHONUTF8 = "1"

$EnvironmentWorks = $false
if (Test-Path -LiteralPath $EnvironmentPython) {
    & $EnvironmentPython -c "import sys; raise SystemExit(sys.version_info < (3, 11))" 2>$null
    $EnvironmentWorks = ($LASTEXITCODE -eq 0)
}

if (-not $EnvironmentWorks) {
    $Candidates = @()
    if ($PythonPath) { $Candidates += $PythonPath }
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        foreach ($PythonVersion in @('-3.12', '-3.11')) {
            $DetectedPython = & $PyLauncher.Source $PythonVersion -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $DetectedPython) { $Candidates += [string]$DetectedPython }
        }
    }
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) { $Candidates += $PythonCommand.Source }
    $ChosenPython = $null
    foreach ($Candidate in $Candidates) {
        if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { continue }
        & $Candidate -c "import sys; raise SystemExit(sys.version_info < (3, 11))" 2>$null
        if ($LASTEXITCODE -eq 0) { $ChosenPython = $Candidate; break }
    }
    if (-not $ChosenPython) {
        throw "Python 3.11 or newer was not found. Install Python, then run this script with -PythonPath 'C:\path\to\python.exe'."
    }
    if (Test-Path -LiteralPath $EnvironmentRoot) {
        # Preserve an invalid environment instead of deleting its packages.
        $ResolvedEnvironment = [IO.Path]::GetFullPath($EnvironmentRoot)
        $ExpectedEnvironment = [IO.Path]::GetFullPath((Join-Path $ProjectRoot '.venv'))
        if ($ResolvedEnvironment -ne $ExpectedEnvironment) { throw "Unexpected environment path." }
        $BackupEnvironment = Join-Path $ProjectRoot ('.venv-backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
        if ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($BackupEnvironment)) -ne [IO.Path]::GetFullPath($ProjectRoot)) {
            throw "Unexpected environment backup path."
        }
        Move-Item -LiteralPath $ResolvedEnvironment -Destination $BackupEnvironment
        Write-Host "Preserved the previous environment at $BackupEnvironment"
    }
    & $ChosenPython -m venv $EnvironmentRoot
    if ($LASTEXITCODE -ne 0) { throw "Could not create the local Python environment." }
}

# Reusing an already working installation does not download anything.
$DependencyCheck = "import PySide6, cv2, rapidocr_onnxruntime, windows_capture, sounddevice, numpy, pynput, win32com.client"
if ($CoreOnly) { $DependencyCheck = "import json, pathlib" }
& $EnvironmentPython -c $DependencyCheck 2>$null
if ($LASTEXITCODE -ne 0) {
    & $EnvironmentPython -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        & $EnvironmentPython -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) { throw "Could not prepare pip in the local environment." }
    }
    Write-Host "Installing missing application dependencies (internet access may be required)..."
    $InstallTarget = ".[all]"
    if ($CoreOnly) { $InstallTarget = "." }
    & $EnvironmentPython -m pip install $InstallTarget
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
    & $EnvironmentPython -c $DependencyCheck
    if ($LASTEXITCODE -ne 0) { throw "The dependency check failed after installation." }
}

Write-Host "Environment ready. Double-click start-wuwa-dubber.bat."
