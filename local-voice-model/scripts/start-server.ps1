param(
    [ValidateRange(0, 65535)][int]$Port = 0,
    [string]$AppConfig = ""
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'server-common.ps1')
$Paths = Get-VoicePaths -ProjectRoot (Split-Path -Parent $PSScriptRoot) -AppConfig $AppConfig -Port $Port
$Port = $Paths.Port
$StartupLock = $null
$StdoutLog = ''
$StderrLog = ''

try {
    New-Item -ItemType Directory -Path $Paths.Logs -Force | Out-Null
    # Prevent two simultaneous double-clicks from loading the model twice.
    try {
        $StartupLock = [IO.File]::Open((Join-Path $Paths.Logs 'qwen-start.lock'), [IO.FileMode]::OpenOrCreate,
            [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    } catch {
        throw '另一个启动窗口正在检查或加载模型，请等待那个窗口的结果。'
    }
    $Token = Get-VoiceToken -ConfigPath $Paths.Config
    $ExistingHealth = Get-VoiceHealth -Port $Port -Token $Token
    $Existing = @(Get-OwnedVoiceProcesses -Paths $Paths)
    if ($Existing.Count -gt 1) {
        throw '检测到多个同路径的模型进程，请查看日志后处理；本次不会再启动。'
    }
    if ($ExistingHealth -and $Existing.Count -eq 1) {
        if (-not (Test-Path -LiteralPath $Paths.State)) {
            Save-VoiceState -Paths $Paths -ProcessInfo $Existing[0]
        }
        Write-Host '本地模型已经就绪，可以直接回到配音程序使用。'
        exit 0
    }
    if ($ExistingHealth -and $Existing.Count -eq 0) {
        throw '该端口已有语音服务，但进程路径不属于本目录；本次不会启动或结束任何其他服务。'
    }
    if ($Existing.Count -eq 1) {
        $ServiceInfo = $Existing[0]
        Write-Host '检测到本目录的模型进程，正在等待它完成加载，不会重复启动。'
    } else {
        $Listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
        if ($Listeners.Count -gt 0) {
            throw "端口 $Port 已被占用，且当前令牌无法确认服务就绪。请检查已有服务和令牌；不会结束占用进程。"
        }
        foreach ($RequiredFile in @($Paths.Python, $Paths.Server,
            (Join-Path $Paths.Model 'model.safetensors'),
            (Join-Path $Paths.Model 'speech_tokenizer\model.safetensors'))) {
            if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
                throw "本地环境或模型文件缺失：$RequiredFile"
            }
        }
        $CacheRoot = Join-Path $Paths.Root 'downloads'
        $env:HF_HOME = Join-Path $CacheRoot 'hf-cache'
        $env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
        $env:TORCH_HOME = Join-Path $CacheRoot 'torch-cache'
        $env:NUMBA_CACHE_DIR = Join-Path $CacheRoot 'numba-cache'
        $env:HF_HUB_OFFLINE = '1'
        $env:TRANSFORMERS_OFFLINE = '1'
        $env:HF_HUB_DISABLE_TELEMETRY = '1'
        $env:TOKENIZERS_PARALLELISM = 'false'
        $env:PYTHONUTF8 = '1'
        $Stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        $StdoutLog = Join-Path $Paths.Logs ("qwen-$Stamp.stdout.log")
        $StderrLog = Join-Path $Paths.Logs ("qwen-$Stamp.stderr.log")
        $Arguments = @('-u', ('"' + $Paths.Server + '"'), '--port', [string]$Port,
            '--app-config', ('"' + $Paths.Config + '"'))
        $Started = Start-Process -FilePath $Paths.Python -ArgumentList $Arguments -WorkingDirectory $Paths.Root `
            -WindowStyle Hidden -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -PassThru
        $ServiceInfo = Get-VoiceProcessById -ProcessId $Started.Id
        if (-not (Test-OwnedVoiceProcess -ProcessInfo $ServiceInfo -Paths $Paths)) {
            throw '无法确认新进程身份；请查看启动日志，本次不会结束身份不明的进程。'
        }
        Save-VoiceState -Paths $Paths -ProcessInfo $ServiceInfo -StdoutLog $StdoutLog -StderrLog $StderrLog
        Write-Host '正在后台加载本地模型，最多等待 120 秒。请保留此提示窗口。'
    }

    $ServicePid = [int]$ServiceInfo.ProcessId
    $ServiceCreated = $ServiceInfo.CreationDate
    $Deadline = [Diagnostics.Stopwatch]::StartNew()
    while ($Deadline.Elapsed.TotalSeconds -lt 120) {
        $Current = Get-VoiceProcessById -ProcessId $ServicePid
        if (-not (Test-OwnedVoiceProcess -ProcessInfo $Current -Paths $Paths) -or $Current.CreationDate -ne $ServiceCreated) {
            throw '模型进程已退出或发生变化。请查看日志中的具体错误。'
        }
        if (Get-VoiceHealth -Port $Port -Token $Token) {
            Write-Host '本地模型已就绪。请在配音程序中选择 Qwen3-TTS，然后试听。'
            Write-Host "服务地址：http://127.0.0.1:$Port"
            exit 0
        }
        Start-Sleep -Milliseconds 800
    }
    throw '等待超过 120 秒，尚未确认就绪。进程可能仍在加载，请查看日志；需要释放显存时使用“停止本地模型”。'
} catch {
    Write-Host ("启动未完成：" + $_.Exception.Message) -ForegroundColor Red
    Write-Host ("日志目录：" + $Paths.Logs)
    if ($StderrLog) { Write-Host ("本次错误日志：" + $StderrLog) }
    exit 1
} finally {
    if ($null -ne $StartupLock) { $StartupLock.Dispose() }
}
