param([ValidateRange(1, 65535)][int]$Port = 9882)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'server-common.ps1')
$Paths = Get-VoicePaths -ProjectRoot (Split-Path -Parent $PSScriptRoot) -Port $Port

try {
    $Candidate = $null
    if (Test-Path -LiteralPath $Paths.State -PathType Leaf) {
        try {
            $Saved = Get-Content -LiteralPath $Paths.State -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($Saved.server_path -eq $Paths.Server -and $Saved.python_path -eq $Paths.Python -and $Saved.port -eq $Port) {
                $RecordedProcess = Get-VoiceProcessById -ProcessId ([int]$Saved.pid)
                if ((Test-OwnedVoiceProcess -ProcessInfo $RecordedProcess -Paths $Paths) -and
                    $RecordedProcess.CreationDate.ToUniversalTime().ToString('o') -eq $Saved.creation_time_utc) {
                    $Candidate = $RecordedProcess
                }
            }
        } catch {
            Write-Host '进程记录已过期或无效，将仅查找本目录的模型进程。'
        }
    }
    if ($null -eq $Candidate) {
        $Matches = @(Get-OwnedVoiceProcesses -Paths $Paths)
        if ($Matches.Count -gt 1) {
            throw '发现多个同路径进程且记录无法确定目标。为避免误停，本次没有结束任何进程。'
        }
        if ($Matches.Count -eq 0) {
            Write-Host '本目录的模型服务当前没有运行。其他 Python 或端口占用进程均未改动。'
            exit 0
        }
        $Candidate = $Matches[0]
    }

    # Recheck immediately before stopping, including creation time to catch PID reuse.
    $Verified = Get-VoiceProcessById -ProcessId ([int]$Candidate.ProcessId)
    if (-not (Test-OwnedVoiceProcess -ProcessInfo $Verified -Paths $Paths) -or
        $Verified.CreationDate -ne $Candidate.CreationDate) {
        throw '进程身份已改变，本次没有结束任何进程。'
    }
    Stop-Process -Id ([int]$Verified.ProcessId) -Force -ErrorAction Stop
    if (Test-Path -LiteralPath $Paths.State -PathType Leaf) {
        Remove-Item -LiteralPath $Paths.State -Force
    }
    Write-Host '本地模型服务已停止。需要配音时，再双击“启动本地模型”。'
} catch {
    Write-Host ("未能停止：" + $_.Exception.Message) -ForegroundColor Red
    Write-Host ("日志目录：" + $Paths.Logs)
    exit 1
}
