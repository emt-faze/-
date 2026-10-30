# 配音应用 v0.5.0

此目录包含字幕捕获、OCR、声线管理、整句缓存、首段短切、可调预取和播放界面。
请先阅读发布包根目录的 [完整说明](../README.md)，使用根目录的中文安装和启动入口。

- 源码：`src/wuwa_dubber`
- 本地数据：`data`（运行后创建，不提交到仓库）
- 核心测试：`python scripts/run_tests.py`
- 基础环境安装：`powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1`

`start-wuwa-dubber.bat` 直接打开界面；`start-ai-dubbing.bat` 按保存的模式启动模型后打开界面。

角色模式默认首段 10 字、后续 36 字、提前准备 2 段；首段设为 0 可恢复原自然分段。字幕确认默认 2 帧，已有配置明确保存的值继续保留。新功能、实测限制和升级方法见 [使用指南](../docs/使用指南.md) 与 [延迟优化记录](../docs/延迟优化.md)。
