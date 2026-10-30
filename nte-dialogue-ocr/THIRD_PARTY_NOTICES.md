# 第三方来源与许可

本次上传保留第三方资产的原有许可，不将其归为本项目原创内容。

## ok-nte 图像模板

- 来源：[BnanZ0/ok-nte](https://github.com/BnanZ0/ok-nte)。
- 固定版本：`893525d5c62eb86fb596956b85d3f136e0209918`。
- 位置：`assets/game-entry/`；每个文件的原始地址见 `source.json`。
- 许可：原项目 GPL-3.0 文本保留在 `assets/game-entry/LICENSE`。
- 用途：识别标题设置图标、游戏内小地图、角色状态及对话界面特征。
- 图像内容未修改。运行时不导入或执行完整上游自动化项目。

## RapidOCR 与 PaddleOCR 模型

- [RapidOCR 3.10.0](https://github.com/RapidAI/RapidOCR) 提供运行接口及模型注册表。
- 模型来自 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)，权重版权属于 Baidu 和/或对应的 PaddleOCR 权利人。
- 本工具使用 RapidOCR 注册表中的 PP-OCRv6 small / medium 检测与识别 ONNX 文件，未改变模型内容。
- 四个实际文件的发布地址、字节数和 SHA-256 保存在 `assets/ocr-v6/models.json`，运行前会核验。
- Apache-2.0 许可全文保留在 `assets/ocr-v6/LICENSE.Apache-2.0`。
- RapidOCR 随包提供的模型说明原文保留为 `assets/ocr-v6/MODEL_LICENSES.RapidOCR.md`。该说明的表格主要列出默认 small 模型，不替代 medium 的单独来源记录。

medium 模型另核对官方模型卡（2026-10-30，均标注 Apache-2.0）：

| 模型 | 官方模型卡 | 核查版本 |
| --- | --- | --- |
| PP-OCRv6 medium detection | [PaddlePaddle/PP-OCRv6_medium_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx) | `61323801669c338b7891481ec7bac61ce31b576a` |
| PP-OCRv6 medium recognition | [PaddlePaddle/PP-OCRv6_medium_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx) | `50c7eacafc52fa7bcf4194e8cd08e46f8558504b` |

官方模型卡说明上游许可；实际随包文件的身份以 RapidOCR 注册表对应的 SHA-256 为准，不声称与另一发布渠道的文件逐字节相同。

## 运行依赖

其他依赖包括 ONNX Runtime、OpenCV、PySide6、windows-capture、MSS、NumPy、pywin32 和 psutil。安装版本记录在 `requirements.txt`，各依赖适用各自许可证。发布压缩包不包含这些依赖的已安装虚拟环境，也不包含 Windows 字体或游戏客户端。
