# 声线批量构建模块（voice-library）

从早期项目 `F:\AI配音` 中抽出的**声线自动构建**能力，供 ASGSV 主项目参考或直接移植。

本目录是同一次验证的**原始可运行代码**，不是重新实现。

---

## 一、它解决什么问题

游戏原语音素材是**一堆零散的 FLAC**，无法直接用作 TTS 参考音频：

- 有的只有 0.3 秒（半个词）
- 有的 26 秒（含大段环境音）
- 没有人知道哪一段说的是什么

本模块做三件事：

```
扫描角色目录 → 按「时长 + 来源」排序候选 → Whisper 转写参考文本 → 生成声线档案
```

---

## 二、实测验证结果（本机真实数据）

### 原始素材质量：不合格

以「长离」的 `角色包_人声` 目录为例（81 个 FLAC，ffprobe 实测前 30 个）：

| 指标 | 值 |
|---|---|
| 最小时长 | 0.33 秒 |
| 最大时长 | 26.43 秒 |
| **中位时长** | **2.0 秒** |
| 落在可用范围（3–30 秒） | **11 / 30（37%）** |
| 落在理想区间（6–12 秒） | **3 / 30（10%）** |

**结论：原始素材大部分不能直接用，必须经过筛选。**

### 处理后的成品：全部合格

54 个角色的 `reference.wav`（全部实测）：

| 指标 | 值 |
|---|---|
| 最小 | **7.37 秒** |
| 最大 | **11.37 秒** |
| 中位 | **8.93 秒** |
| 落在 3–30 秒 | **54 / 54（100%）** |
| 落在 6–12 秒 | **54 / 54（100%）** |

**从 10% 合格率提升到 100%** —— 这是选片算法实际工作的证据。

### 转写文本质量

54 条 `prompt_text`，无空文本，平均 34 字，有完整语义。

**但存在错误**，抽查发现的例子：

| 角色 | 转写结果 | 问题 |
|---|---|---|
| 秋水 | 「今后，**误会**一直在你身边」 | 应为「我**会**一直在你身边」 |
| 秧秧·玄翎 | 「会架起一道连接**万阴**的桥梁」 | 疑为「万**音**」或专名 |

**结论：转写需人工复核，不能全自动信任。**

---

## 三、文件说明

```
wuwa_dubber/
├─ voice_library.py        核心：扫描、排序、转写、批量构建
├─ voice_profiles.py       声线档案存取（json + 参考音频）
├─ geometry.py             控件坐标 → 画面坐标（DPI 缩放换算）
├─ models.py               Rect / CaptureProfile / VoiceProfile 数据类
├─ config.py               配置与路径
├─ text_processing.py      文本归一化
├─ tool_discovery.py       ffmpeg / ffprobe 自动查找
build_voice_profiles.py    命令行入口
test_voice_library.py      测试（pytest 风格）
test_config_and_profiles.py
```

### 关键实现细节

**选片排序**（`rank_audio_candidates`）——四维排序，不是按体积：

```python
key = (
    duration_score,                    # 6–12s=0, 5–15s=1, 其他=2
    source_priority,                   # 「角色包_人声」优先
    abs(duration - 9.0),               # 越接近 9 秒越好
    path.name,                         # 稳定的最终排序
)
```

**Whisper 参数**（都是显式指定的，不是默认值）：

```python
language="zh", beam_size=5, best_of=5, temperature=0.0,
condition_on_previous_text=False,      # 避免串到上一段
vad_filter=True,                       # 过滤静音
vad_parameters={"min_silence_duration_ms": 300},
initial_prompt="以下是简体中文剧情对白。",   # 引导输出简体+标点
```

**批量构建**（`BatchVoiceProfileBuilder.build`）：
- 扫描角色目录（跳过项目/缓存等干扰目录）
- 已有档案默认跳过（`skip_existing=True`）
- 支持进度回调与取消（`threading.Event`）
- 取消后为未处理角色补 `cancelled` 状态

---

## 四、怎么跑

```bash
pip install faster-whisper        # 仅在批量转写时需要
pip install pytest                # 仅测试时需要

cd tools/voice-library
python -m pytest test_voice_library.py -q
```

命令行批量构建：

```bash
PYTHONPATH=. python build_voice_profiles.py --root <角色目录> --model medium
```

`--model` 可选 `small` / `medium` / `large-v3`。默认 `medium` + CPU，避免占用 TTS 显存。

---

## 五、移植到 ASGSV 的注意事项

### ① 声线档案格式兼容

`profile.json` 结构：

```json
{
  "speaker": "丹瑾",
  "aliases": [],
  "reference_audio": "丹瑾\\reference.wav",
  "prompt_text": "星星酥虽然很漂亮，但也太不容易保存了……",
  "reference_language": "zh",
  "speed": 1.0,
  "enabled": true
}
```

ASGSV 现有 `wuwa-dubber/data/profiles/` 使用**相同结构**，可直接复用。

### ② TTS 后端不同

本模块产出的参考音频是给 **GPT-SoVITS** 用的。
ASGSV 主项目用 **Qwen3-TTS**，两者对参考音频的要求可能不同：

| | GPT-SoVITS | Qwen3-TTS |
|---|---|---|
| 理想时长 | 5–15 秒 | 3–30 秒（建议 6–12） |
| 采样率 | 32 kHz | 未验证 |
| 需要转写文本 | 是（prompt_text） | 是 |

**时长要求接近，但采样率和预处理需要实机验证。**

### ③ 依赖 faster-whisper

只在**批量建声线**时需要，实时配音链路不依赖它。
若不想引入，可保留已有的 54 个声线档案，跳过批量重做。

---

## 六、已知问题

| 问题 | 说明 |
|---|---|
| 转写有错字 | 实测存在「误会/我会」这类错误，必须人工复核 |
| 依赖本地素材 | 需要事先准备好角色目录和 FLAC 素材 |
| 只在 CPU 测过 | 未验证 GPU 加速下的行为与显存占用 |
| 理想时长 9 秒是经验值 | 未针对 Qwen3-TTS 调优 |

---

## 七、来源

代码来源：`F:\AI配音\src\wuwa_dubber\`（早期版「鸣潮 AI 实时配音」项目）
本目录为原样拷贝，仅补充了包结构（`__init__.py`）与缓存清理。

原项目说明见其 `README.md`，其 TTS 后端为 GPT-SoVITS，与本仓库主项目不同。
