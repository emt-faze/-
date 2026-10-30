# 本地 Qwen3-TTS 服务

供配音应用使用的本机服务，默认监听 `127.0.0.1:9882`，从旁边的 `wuwa-dubber/data/config.json` 读取本地令牌。
使用发布包根目录的“安装角色配音模型.bat”安装，详见 [模型安装](../docs/模型安装.md)。

- 模型：`Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- 固定版本：`5d83992436eae1d760afd27aff78a71d676296fc`
- 模型目录：`models/Qwen3-TTS-12Hz-0.6B-Base`，初次安装时下载
- 服务：`qwen_server.py`
- 启停：`scripts/start-server.ps1`、`scripts/stop-server.ps1`
- 无模型协议测试：`python test_server_protocol.py`

启停核对 Python 路径、脚本路径、端口和进程创建时间，只处理本目录模型进程。
关闭配音窗口后模型可能继续运行，不用时请运行停止入口。

v0.5.0 在端口绑定后后台预热两句短文本，输出丢弃，不写入应用音频缓存。有效声线缺失时跳过预热；导入声线后正常请求仍可使用。用户请求优先处理，预热失败只记录警告。

健康接口增加 `warm`、`warming`、`warmup` 和 `prompt_cache`。参考编码缓存默认 16 项，可用 `QWEN_PROMPT_CACHE_SIZE` 设置 1–64，键包括录音内容与准确参考文字，服务重启后重建。

原 `/v1/synthesize` 默认仍返回 WAV；诊断请求增加 `response_format: "json"` 后返回 `audio_wav_base64`、`timing_ms` 和 `metrics`。生成日志包含各阶段毫秒值，实际推理参数保持 BF16、SDPA 和原生成长度上限。详见 [开发说明](../docs/开发说明.md) 和 [参数实测](../docs/延迟优化.md)。
