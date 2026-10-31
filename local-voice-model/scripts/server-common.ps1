Set-StrictMode -Version Latest

function Get-VoicePaths {
    param([string]$ProjectRoot, [string]$AppConfig = "", [int]$Port = 0)
    $ResolvedRoot = [IO.Path]::GetFullPath($ProjectRoot)
    if ([string]::IsNullOrWhiteSpace($AppConfig)) {
        $AppConfig = Join-Path (Split-Path -Parent $ResolvedRoot) 'wuwa-dubber\data\config.json'
    }
    if ($Port -eq 0) {
        $Port = 9882
        if (Test-Path -LiteralPath $AppConfig -PathType Leaf) {
            $Settings = Get-Content -LiteralPath $AppConfig -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($Settings.PSObject.Properties['tts_endpoint']) {
                $ServiceUri = [Uri]([string]$Settings.tts_endpoint)
                if ($ServiceUri.Scheme -ne 'http' -or $ServiceUri.Host -notin @('127.0.0.1', 'localhost') -or
                    $ServiceUri.AbsolutePath -ne '/' -or $ServiceUri.UserInfo -or $ServiceUri.Query -or $ServiceUri.Fragment) {
                    throw '本地模型启动地址必须为本机 HTTP 地址，且不能包含额外路径。'
                }
                $Port = $ServiceUri.Port
            }
        }
    }
    [pscustomobject]@{
        Root = $ResolvedRoot
        Python = Join-Path $ResolvedRoot '.venv\Scripts\python.exe'
        Server = Join-Path $ResolvedRoot 'qwen_server.py'
        Model = Join-Path $ResolvedRoot 'models\Qwen3-TTS-12Hz-0.6B-Base'
        Logs = Join-Path $ResolvedRoot 'logs'
        State = Join-Path $ResolvedRoot 'logs\qwen-server-state.json'
        Config = [IO.Path]::GetFullPath($AppConfig)
        Port = $Port
    }
}

function Test-OwnedVoiceProcess {
    param($ProcessInfo, $Paths)
    if ($null -eq $ProcessInfo -or [string]::IsNullOrWhiteSpace($ProcessInfo.CommandLine)) {
        return $false
    }
    if (-not [string]::Equals($ProcessInfo.ExecutablePath, $Paths.Python, [StringComparison]::OrdinalIgnoreCase)) {
        return $false
    }
    # The Python interpreter and script must be the first two command arguments.
    # A script name embedded in -c text, an option value or a different directory is rejected.
    $InterpreterPattern = [regex]::Escape($Paths.Python)
    $ServerPattern = [regex]::Escape($Paths.Server)
    $CommandPattern = '^(?:"' + $InterpreterPattern + '"|' + $InterpreterPattern + ')\s+(?:-u\s+)?(?:"' + $ServerPattern + '"|' + $ServerPattern + ')(?=\s|$)'
    $PortPattern = '(?:^|\s)--port(?:\s+|=)' + $Paths.Port + '(?=\s|$)'
    return [regex]::IsMatch($ProcessInfo.CommandLine, $CommandPattern, [Text.RegularExpressions.RegexOptions]::IgnoreCase) -and
        [regex]::IsMatch($ProcessInfo.CommandLine, $PortPattern)
}

function Get-OwnedVoiceProcesses {
    param($Paths)
    $Candidates = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction Stop
    @($Candidates | Where-Object { Test-OwnedVoiceProcess -ProcessInfo $_ -Paths $Paths })
}

function Get-VoiceProcessById {
    param([int]$ProcessId)
    Get-CimInstance Win32_Process -Filter ("ProcessId = " + $ProcessId) -ErrorAction Stop
}

function Get-VoiceToken {
    param([string]$ConfigPath)
    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
        throw "找不到配音程序配置：$ConfigPath"
    }
    $ConfigValue = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $ConfigValue.PSObject.Properties['tts_token'] -or [string]::IsNullOrWhiteSpace([string]$ConfigValue.tts_token)) {
        throw '请先在配音程序中设置并保存本地令牌。'
    }
    return [string]$ConfigValue.tts_token
}

function Get-VoiceHealth {
    param([int]$Port, [string]$Token)
    $Response = $null
    $Reader = $null
    try {
        $Request = [Net.HttpWebRequest]::Create("http://127.0.0.1:$Port/v1/health")
        $Request.Method = 'GET'
        $Request.Proxy = $null
        $Request.AllowAutoRedirect = $false
        $Request.Timeout = 2000
        $Request.ReadWriteTimeout = 2000
        $Request.Headers['Authorization'] = 'Bearer ' + $Token
        $Response = $Request.GetResponse()
        $Reader = New-Object IO.StreamReader($Response.GetResponseStream(), [Text.Encoding]::UTF8)
        $Result = $Reader.ReadToEnd() | ConvertFrom-Json
        if ($Result.status -eq 'ok' -and $Result.backend -eq 'qwen3-tts' -and
            $Result.model_id -eq 'Qwen/Qwen3-TTS-12Hz-0.6B-Base') {
            return $Result
        }
    } catch {
        # Do not log authentication headers or the user's token.
        return $null
    } finally {
        if ($null -ne $Reader) { $Reader.Dispose() }
        if ($null -ne $Response) { $Response.Dispose() }
    }
    return $null
}

function Save-VoiceState {
    param($Paths, $ProcessInfo, [string]$StdoutLog = '', [string]$StderrLog = '')
    $StateValue = [ordered]@{
        pid = [int]$ProcessInfo.ProcessId
        creation_time_utc = $ProcessInfo.CreationDate.ToUniversalTime().ToString('o')
        server_path = $Paths.Server
        python_path = $Paths.Python
        port = $Paths.Port
        stdout_log = $StdoutLog
        stderr_log = $StderrLog
    }
    $StateValue | ConvertTo-Json | Set-Content -LiteralPath $Paths.State -Encoding UTF8
}
