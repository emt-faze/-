"""Start the selected local model when needed, then open the desktop UI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from wuwa_dubber.tts.url_policy import managed_qwen_port
    data_root = Path(os.environ.get("WUWA_DUBBER_DATA", ROOT / "data")).expanduser().resolve()
    config_file = data_root / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8-sig")) if config_file.is_file() else {}
    qwen = config.get("tts_backend") == "qwen3-tts" and not config.get("use_fake_tts")
    endpoint = str(config.get("tts_endpoint", "http://127.0.0.1:9882")).rstrip("/")
    port = managed_qwen_port(endpoint)
    managed_endpoint = port is not None
    if qwen and not managed_endpoint:
        print("当前使用自定义语音服务地址，请确保对应服务已经启动。")
    if qwen and managed_endpoint:
        runtime_file = data_root / "model_runtime.json"
        if not runtime_file.is_file():
            print("缺少本地模型路径配置，请查看使用说明。")
            return 1
        runtime = json.loads(runtime_file.read_text(encoding="utf-8-sig"))
        model_root = Path(runtime["model_root"]).expanduser()
        if not model_root.is_absolute():
            model_root = ROOT / model_root
        starter = model_root.resolve() / "scripts/start-server.ps1"
        if not starter.is_file():
            print(f"找不到本地模型启动脚本：{starter}")
            return 1
        code = subprocess.call(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                str(starter), "-AppConfig", str(config_file), "-Port", str(port)])
        if code:
            print("模型未就绪，请检查模型日志。也可以用原启动方式打开设置，切换系统中文朗读。")
            return code
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONUTF8="1")
    return subprocess.call([sys.executable, "-m", "wuwa_dubber"], cwd=ROOT, env=env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"启动失败：{exc}")
        raise SystemExit(1)
