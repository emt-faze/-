"""Local transcription UI: show original pixels before revealing OCR results."""
import json
import sys
from pathlib import Path

from accuracy_eval import DEFAULT_MANIFEST, ROOT, evaluate, prepare
from dialogue_store import atomic_text


def main():
    from PySide6 import QtCore, QtGui, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    font=app.font()
    font.setFamilies(["Microsoft YaHei UI", "Microsoft YaHei", "SimSun"])
    font.setPointSize(10)
    app.setFont(font)
    dataset = prepare(ROOT/"records", DEFAULT_MANIFEST)

    class Review(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("异环 · 真实台词逐字核对")
            self.resize(1000, 760)
            self.position = 0
            layout = QtWidgets.QVBoxLayout(self)
            self.status = QtWidgets.QLabel()
            self.status.setWordWrap(True)
            layout.addWidget(self.status)
            note = QtWidgets.QLabel("先看原图填写正确文字，再查看识别结果。50 条中 20 条用于调整，30 条保留验收；不要把验收集错字用于调参后仍称为首次验收。")
            note.setWordWrap(True)
            layout.addWidget(note)
            self.image = QtWidgets.QLabel("尚无真实对话样本，请先校准并运行识别。")
            self.image.setMinimumHeight(140)
            self.image.setAlignment(QtCore.Qt.AlignCenter)
            self.image.setStyleSheet("background:#17202a;color:white;padding:12px")
            layout.addWidget(self.image)
            self.name_image = QtWidgets.QLabel()
            layout.addWidget(self.name_image)
            layout.addWidget(QtWidgets.QLabel("逐字核对的正确正文："))
            self.truth = QtWidgets.QPlainTextEdit()
            self.truth.setPlaceholderText("从上方原图录入，保留数字、否定词、专有名词和标点。")
            layout.addWidget(self.truth)
            self.speaker = QtWidgets.QLineEdit()
            self.speaker.setPlaceholderText("原图中的角色名；画面没有人名时留空")
            layout.addWidget(self.speaker)
            self.complete = QtWidgets.QCheckBox("已看原图，确认正文没有截断或遮挡")
            layout.addWidget(self.complete)
            self.tags = {}
            row = QtWidgets.QHBoxLayout()
            for key, text in (("long","长句"),("short","短句"),("multiline","多行"),("numbers","数字"),
                              ("negation","否定词"),("names","人名"),("light_background","亮背景"),("dark_background","暗背景")):
                check = QtWidgets.QCheckBox(text)
                self.tags[key] = check
                row.addWidget(check)
            layout.addLayout(row)
            self.prediction = QtWidgets.QPlainTextEdit()
            self.prediction.setReadOnly(True)
            self.prediction.setMaximumHeight(130)
            self.prediction.setPlaceholderText("识别结果暂时隐藏，防止照抄结果影响核对。")
            layout.addWidget(self.prediction)
            row = QtWidgets.QHBoxLayout()
            for label, function in (("上一条",self.previous),("显示识别结果",self.reveal),
                                    ("保存核对并下一条",self.save_next),("刷新样本",self.refresh),
                                    ("生成验收报告",self.report)):
                button=QtWidgets.QPushButton(label)
                button.clicked.connect(function)
                row.addWidget(button)
            layout.addLayout(row)
            self.load()

        def load(self):
            samples=dataset["samples"]
            self.status.setText(f"已收集 {len(samples)}/50 条 · 已核对 {sum(bool(s.get('truth_checked')) for s in samples)} 条")
            if not samples:
                return
            self.position=min(self.position,len(samples)-1)
            item=samples[self.position]
            split={"tune":"调整集","acceptance":"保留验收集",None:"收集中"}[item.get("split")]
            source={"assistant_visual_transcription":"助手看图转录，未获用户确认", "user_visual_transcription":"用户逐字核对"}.get(item.get('label_source'),'尚未转录')
            self.status.setText(self.status.text()+f" · 当前 {self.position+1} · {split} · {source}")
            pm=QtGui.QPixmap(item["body_image"])
            self.image.setPixmap(pm.scaled(950,230,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
            self.name_image.clear()
            if item.get("speaker_image"):
                self.name_image.setPixmap(QtGui.QPixmap(item["speaker_image"]).scaled(500,65,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
            self.truth.setPlainText(item.get("truth_body") or "")
            self.speaker.setText(item.get("truth_speaker") or "")
            self.complete.setChecked(item.get("source_complete_confirmed",False))
            if item.get("source_clipped"):
                self.status.setText(self.status.text()+" · 原图疑似截断，请先检查")
            for key,check in self.tags.items():
                check.setChecked(key in item.get("tags",[]))
            self.prediction.clear()

        def previous(self):
            self.position=max(0,self.position-1)
            self.load()

        def reveal(self):
            if dataset["samples"]:
                item=dataset["samples"][self.position]
                self.prediction.setPlainText("首次："+item["original_text"]+"\n复核："+item["review_text"])

        def save_next(self):
            if not dataset["samples"]:
                return
            body=self.truth.toPlainText().strip()
            if not body:
                self.status.setText("请先根据原图填写正确正文。")
                return
            item=dataset["samples"][self.position]
            item.update(truth_body=body,truth_speaker=self.speaker.text().strip(),truth_checked=True,
                        source_complete_confirmed=self.complete.isChecked(),
                        label_source="user_visual_transcription",tags=[key for key,check in self.tags.items() if check.isChecked()])
            atomic_text(DEFAULT_MANIFEST,json.dumps(dataset,ensure_ascii=False,indent=2))
            self.position=min(self.position+1,len(dataset["samples"])-1)
            self.load()

        def refresh(self):
            nonlocal dataset
            dataset=prepare(ROOT/"records",DEFAULT_MANIFEST)
            self.load()

        def report(self):
            result=evaluate(DEFAULT_MANIFEST)
            if result["status"]=="not_measured":
                self.status.setText("尚不能计算验收成绩："+"；".join(result["issues"][:3]))
            else:
                score=result["displayed"]
                self.status.setText(f"正文字符错误率 {score['cer']:.2%}；整句正确率 {score['exact_sentence_rate']:.2%}。报告已保存，角色名与错例另列。")

    widget=Review()
    widget.show()
    return app.exec()


if __name__=="__main__":
    raise SystemExit(main())
