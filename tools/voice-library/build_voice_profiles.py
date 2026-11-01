from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wuwa_dubber.config import AppPaths, default_data_root, load_config
from wuwa_dubber.voice_library import BatchVoiceProfileBuilder, FasterWhisperTranscriber
from wuwa_dubber.voice_profiles import VoiceProfileStore


def main() -> int:
    parser = argparse.ArgumentParser(description="从角色文件夹批量建立 GPT-SoVITS 声线档案")
    parser.add_argument("--root", type=Path, default=ROOT, help="包含一级角色目录的项目目录")
    parser.add_argument("--data-dir", type=Path, help="声线档案保存目录")
    parser.add_argument("--model", default="medium", help="faster-whisper 模型名或本地模型路径")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--force", action="store_true", help="覆盖已有声线档案")
    args = parser.parse_args()

    paths = AppPaths(args.data_dir.resolve() if args.data_dir else default_data_root())
    config = load_config(paths)
    store = VoiceProfileStore(paths)
    transcriber = FasterWhisperTranscriber(
        model_name=args.model,
        device=args.device,
        download_root=paths.cache / "whisper",
    )
    builder = BatchVoiceProfileBuilder(
        store=store,
        transcriber=transcriber,
        ffmpeg_path=config.ffmpeg_path,
        ffprobe_path=config.ffprobe_path,
        model_name=args.model,
    )
    summary = builder.build(
        args.root.resolve(),
        skip_existing=not args.force,
        progress=lambda completed, total, message: print(
            f"[{completed}/{total}] {message}",
            flush=True,
        ),
    )
    print(
        f"完成：成功 {summary.imported}，跳过 {summary.skipped}，"
        f"失败 {summary.failed}，取消 {summary.cancelled_count}"
    )
    for result in summary.results:
        if result.status == "failed":
            print(f"失败 {result.speaker}: {result.message}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
