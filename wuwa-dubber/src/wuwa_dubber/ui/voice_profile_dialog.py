from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class VoiceProfileDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("导入声线")
        self.resize(560, 430)
        self.speaker = QLineEdit()
        self.aliases = QLineEdit()
        self.aliases.setPlaceholderText("多个别名用逗号分隔")
        self.audio = QLineEdit()
        browse = QPushButton("浏览")
        browse.clicked.connect(self._browse_audio)
        audio_row = QWidget()
        audio_layout = QHBoxLayout(audio_row)
        audio_layout.setContentsMargins(0, 0, 0, 0)
        audio_layout.addWidget(self.audio, 1)
        audio_layout.addWidget(browse)
        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("输入参考音频中实际说出的文字")
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.5, 2.0)
        self.speed.setSingleStep(0.05)
        self.speed.setValue(1.0)

        form = QFormLayout()
        form.addRow("角色名", self.speaker)
        form.addRow("别名", self.aliases)
        form.addRow("参考音频", audio_row)
        form.addRow("参考台词", self.prompt)
        form.addRow("语速", self.speed)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, list[str], Path, str, float]:
        aliases = [item.strip() for item in self.aliases.text().replace("，", ",").split(",") if item.strip()]
        return (
            self.speaker.text().strip(),
            aliases,
            Path(self.audio.text().strip()),
            self.prompt.toPlainText().strip(),
            self.speed.value(),
        )

    def _browse_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择参考音频",
            "",
            "Audio (*.wav *.mp3 *.flac *.m4a *.ogg);;All Files (*.*)",
        )
        if path:
            self.audio.setText(path)

    def _validate(self) -> None:
        speaker, _, audio, prompt, _ = self.values()
        if not speaker or not prompt or not audio.is_file():
            return
        self.accept()
