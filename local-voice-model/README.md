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
