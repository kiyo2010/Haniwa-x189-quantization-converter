# SPDX-License-Identifier: Apache-2.0
"""Desktop front end for the existing X189 conversion command."""
import argparse
import codecs
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
from tempfile import TemporaryDirectory

try:
    from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
        QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
        QSizePolicy, QToolButton, QVBoxLayout, QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    raise SystemExit("GUI requires PySide6-Essentials: python -m pip install -r requirements-gui.txt") from None


RECIPE = "X189-Generic-v1"
TENSOR_PROGRESS = re.compile(r"^\[(\d+)/(\d+)\] (?:E8|IQ4_NL) ", re.M)
STYLES = """
QWidget#window { background: #f2f5fa; color: #15243a; }
QLabel { color: #15243a; }
QLabel#title { font-size: 29px; font-weight: 700; }
QLabel#subtitle { color: #54647c; font-size: 14px; }
QFrame#inputs, QFrame#run { background: #ffffff; border: 1px solid #d7dfeb; border-radius: 12px; }
QGroupBox { background: #ffffff; border: 1px solid #d7dfeb; border-radius: 9px;
            margin-top: 10px; padding-top: 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #263951; }
QLineEdit, QComboBox { background: #ffffff; color: #182a41; border: 1px solid #bcc9da;
                      border-radius: 6px; min-height: 32px; padding: 1px 9px; }
QLineEdit:focus, QComboBox:focus { border-color: #2962ba; }
QLineEdit:disabled, QComboBox:disabled { background: #f4f6fa; color: #5c6b81; }
QPushButton { background: #f1f5fb; border: 1px solid #c5d2e2; border-radius: 6px;
              color: #1c385d; min-height: 32px; padding: 1px 13px; }
QPushButton:hover { background: #e6eef9; }
QPushButton:focus, QToolButton:focus { border: 1px solid #2962ba; }
QPushButton#start { color: #ffffff; background: #2459a8; border-color: #2459a8;
                   font-weight: 700; min-width: 112px; }
QPushButton#start:hover { background: #194784; }
QPushButton:disabled, QPushButton#start:disabled { background: #edf1f7; color: #66768d;
                                                 border-color: #d6deea; }
QProgressBar { border: 0; background: #e5ecf7; border-radius: 4px; min-height: 8px; max-height: 8px; }
QProgressBar::chunk { background: #316cc4; border-radius: 4px; }
QToolButton { border: 1px solid transparent; color: #274e83; padding: 2px 3px; }
QPlainTextEdit { color: #263951; background: #f4f7fc; border: 1px solid #dae3ef; border-radius: 6px; }
"""


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("window")
        self.setWindowTitle("X189 BF16 → GGUF")
        self.resize(800, 580)
        self.setMinimumWidth(640)
        self.settings = QSettings("X189", "Converter")
        self.work = None
        self.destination = None
        self.cancelled = False
        self.process = QProcess(self)
        parameters = QProcess.UnixProcessParameters()
        parameters.flags = QProcess.UnixProcessFlag.CreateNewSession
        self.process.setUnixProcessParameters(parameters)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.started.connect(lambda: self.stop.setEnabled(True))
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)
        title = QLabel("X189 BF16 → GGUF")
        title.setObjectName("title")
        subtitle = QLabel("Hugging FaceのBF16モデルをX189で量子化し、標準GGUFに保存します。")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.inputs = QFrame()
        self.inputs.setObjectName("inputs")
        paths = QGridLayout(self.inputs)
        paths.setContentsMargins(16, 16, 16, 16)
        self.checkpoint = self.path_row(paths, 0, "BF16モデル", "モデルのフォルダ", self.pick_checkpoint)
        self.calibration = self.path_row(paths, 1, "校正テキスト", "UTF-8のテキストファイル", self.pick_calibration)
        self.output = self.path_row(paths, 2, "GGUFの保存先", "新しい .gguf ファイル", self.pick_output)
        layout.addWidget(self.inputs)

        self.runtime = QGroupBox("llama.cpp 設定")
        tools = QGridLayout(self.runtime)
        tools.setContentsMargins(16, 11, 16, 11)
        self.source = self.path_row(tools, 0, "ソース", "変換スクリプトを含むフォルダ",
                                    lambda: self.pick_folder(self.source, "llama.cppのソース"))
        self.binaries = self.path_row(tools, 1, "CPUバイナリ", "llama-imatrix を含むフォルダ",
                                      lambda: self.pick_folder(self.binaries, "llama.cppのバイナリ"))
        self.device = QComboBox()
        self.device.addItem("CPU", "cpu")
        self.device.addItem("GPU（CUDA）", "cuda:0")
        tools.addWidget(QLabel("変換デバイス", buddy=self.device), 2, 0)
        tools.addWidget(self.device, 2, 1)
        layout.addWidget(self.runtime)

        conversion = QFrame()
        conversion.setObjectName("run")
        run = QVBoxLayout(conversion)
        run.setContentsMargins(16, 14, 16, 12)
        self.status = QLabel("モデル、校正テキスト、保存先を選択してください。")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setValue(0)
        self.progress.setAccessibleName("変換の進捗")
        run.addWidget(self.status)
        run.addWidget(self.progress)

        buttons = QHBoxLayout()
        self.open_folder = QPushButton("保存先を開く", enabled=False)
        self.open_folder.clicked.connect(self.reveal_destination)
        self.stop = QPushButton("停止", enabled=False)
        self.stop.clicked.connect(self.cancel)
        self.start = QPushButton("変換を開始")
        self.start.setObjectName("start")
        self.start.clicked.connect(self.convert)
        self.start.setAutoDefault(False)
        buttons.addWidget(self.open_folder)
        buttons.addStretch(1)
        buttons.addWidget(self.stop)
        buttons.addWidget(self.start)
        run.addLayout(buttons)

        self.log_toggle = QToolButton(text="ログを表示", checkable=True)
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setAccessibleName("変換ログ")
        self.log.setMaximumBlockCount(400)
        self.log.setMaximumHeight(148)
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.log.hide()
        self.log_toggle.toggled.connect(self.show_log)
        run.addWidget(self.log_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        run.addWidget(self.log)
        layout.addWidget(conversion)
        layout.addStretch(1)

        binary = shutil.which("llama-imatrix")
        self.source.setText(self.settings.value("source", os.environ.get("LLAMA_SRC", ""), str))
        self.binaries.setText(self.settings.value("binaries",
            os.environ.get("LLAMA_CPU", str(Path(binary).resolve().parent) if binary else ""), str))
        self.device.setCurrentIndex(max(0, self.device.findData(self.settings.value("device", "cpu", str))))

    def path_row(self, layout, row, name, placeholder, pick):
        field = QLineEdit(placeholderText=placeholder)
        button = QPushButton("選択…")
        button.setAccessibleName(f"{name}を選択")
        button.clicked.connect(pick)
        layout.addWidget(QLabel(name, buddy=field), row, 0)
        layout.addWidget(field, row, 1)
        layout.addWidget(button, row, 2)
        layout.setColumnStretch(1, 1)
        return field

    def pick_folder(self, field, title):
        chosen = QFileDialog.getExistingDirectory(self, title, field.text() or str(Path.home()))
        if chosen:
            field.setText(chosen)
        return chosen

    def pick_checkpoint(self):
        chosen = self.pick_folder(self.checkpoint, "BF16モデルのフォルダ")
        if chosen and not self.output.text():
            model = Path(chosen)
            self.output.setText(str(model.parent / f"{model.name}-x189.gguf"))

    def pick_calibration(self):
        chosen, _ = QFileDialog.getOpenFileName(self, "校正テキスト", self.calibration.text() or str(Path.home()),
                                               "テキスト (*.txt);;すべてのファイル (*)")
        if chosen:
            self.calibration.setText(chosen)

    def pick_output(self):
        chosen, _ = QFileDialog.getSaveFileName(self, "GGUFの保存先", self.output.text() or str(Path.home()),
                                               "GGUF (*.gguf)")
        if chosen:
            self.output.setText(chosen if Path(chosen).suffix else chosen + ".gguf")

    @staticmethod
    def require_path(field, valid, message):
        value = field.text()
        path = Path(value).expanduser()
        if not value.strip() or not valid(path):
            field.setFocus()
            raise ValueError(message)
        return path

    def convert(self):
        if self.work is not None:
            return
        path = self.require_path
        try:
            model = path(self.checkpoint, lambda p: (p / "config.json").is_file(),
                         "config.json を含むBF16モデルのフォルダを選択してください。").resolve()
            calibration = path(self.calibration, Path.is_file, "校正テキストを選択してください。").resolve()
            destination = path(self.output, lambda p: p.suffix.lower() == ".gguf" and p.parent.is_dir(),
                               "既存のフォルダに新しい .gguf ファイル名を指定してください。").absolute()
            if os.path.lexists(destination) or destination.resolve().is_relative_to(model):
                self.output.setFocus()
                raise ValueError("モデルのフォルダの外に、未使用の .gguf ファイル名を指定してください。")
            source = path(self.source, lambda p: (p / "convert_hf_to_gguf.py").is_file() and
                          (p / "gguf-py" / "gguf").is_dir(), "llama.cppのソースフォルダを選択してください。").resolve()
            binaries = path(self.binaries, lambda p: (p / "libggml-base.so").is_file() and
                            all((p / name).is_file() and os.access(p / name, os.X_OK)
                                for name in ("llama-imatrix", "llama-tokenize")),
                            "llama-imatrix、llama-tokenize、libggml-base.so のあるフォルダを選択してください。").resolve()
            self.work = TemporaryDirectory(dir=destination.parent, prefix=".x189-gui-")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
            return

        device = self.device.currentData()
        for name, value in (("source", str(source)), ("binaries", str(binaries)), ("device", device)):
            self.settings.setValue(name, value)
        self.destination = destination
        self.staged_output = Path(self.work.name) / destination.name
        self.cancelled = False
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.last_line = ""
        self.log.clear()
        self.log_toggle.setChecked(False)
        self.inputs.setEnabled(False)
        self.runtime.setEnabled(False)
        self.start.setEnabled(False)
        self.open_folder.setEnabled(False)
        self.progress.setRange(0, 0)
        self.status.setText("準備と校正を実行中…")

        environment = QProcessEnvironment.systemEnvironment()
        if device == "cuda:0":
            visible = environment.value("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
            environment.insert("CUDA_VISIBLE_DEVICES", visible)
        self.process.setProcessEnvironment(environment)
        self.process.start(sys.executable, ["-B", "-u", "-m", "x189_clean", "convert",
                           "--checkpoint", str(model), "--calibration-text", str(calibration),
                           "--output", str(self.staged_output), "--work", self.work.name,
                           "--llama-source", str(source), "--llama-imatrix", str(binaries / "llama-imatrix"),
                           "--ggml-lib", str(binaries / "libggml-base.so"), "--device", device, "--threads", "2"])

    def show_log(self, visible):
        self.log_toggle.setText("ログを隠す" if visible else "ログを表示")
        self.log.setVisible(visible)

    def read_output(self, *, final=False):
        raw = bytes(self.process.readAllStandardOutput())
        content = self.decoder.decode(raw, final=final)
        if not content:
            return
        bar = self.log.verticalScrollBar()
        follow = bar.value() >= bar.maximum()
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(content[-65536:])
        if follow:
            bar.setValue(bar.maximum())
        lines = self.last_line + content
        self.last_line = lines.rsplit("\n", 1)[-1][-512:]
        if self.cancelled:
            return
        matches = TENSOR_PROGRESS.findall(lines)
        if matches:
            done, total = map(int, matches[-1])
            if done < total:
                self.progress.setRange(0, total)
                self.progress.setValue(done)
                self.status.setText(f"X189で量子化中（{done} / {total} テンソル）")
            else:
                self.progress.setRange(0, 0)
                self.status.setText("GGUFを書き出し、読み戻して検証中…")

    def cancel(self):
        pid = self.process.processId()
        if pid:
            self.cancelled = True
            self.stop.setEnabled(False)
            self.status.setText("変換を停止中…")
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.log.appendPlainText(self.process.errorString())
            self.finished(-1, QProcess.ExitStatus.CrashExit)

    def finished(self, code, exit_status):
        self.read_output(final=True)
        success = False
        if self.cancelled:
            message = "変換を停止しました。"
        elif code != 0 or exit_status != QProcess.ExitStatus.NormalExit:
            message = "変換に失敗しました。ログを確認してください。"
            self.log_toggle.setChecked(True)
        else:
            try:
                report_path = self.staged_output.with_name(self.staged_output.name + ".x189.json")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                size = self.staged_output.stat().st_size
                if not isinstance(report, dict) or report.get("recipe") != RECIPE or report.get("bytes") != size:
                    raise ValueError("X189の変換結果を確認できませんでした。")
                os.link(self.staged_output, self.destination)
                message = f"変換完了  ·  {self.destination.name}（{size:,} B）"
                success = True
            except (OSError, ValueError) as exc:
                self.log.appendPlainText(str(exc))
                self.log_toggle.setChecked(True)
                message = "GGUFを保存できませんでした。ログを確認してください。"
        try:
            self.work.cleanup()
        except OSError as exc:
            self.log.appendPlainText(str(exc))
            self.log_toggle.setChecked(True)
            message += " 作業ファイルを削除できませんでした。"
        self.work = None
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if success else 0)
        self.status.setText(message)
        self.inputs.setEnabled(True)
        self.runtime.setEnabled(True)
        self.start.setEnabled(True)
        self.stop.setEnabled(False)
        self.open_folder.setEnabled(success)

    def reveal_destination(self):
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.destination.parent))):
            self.status.setText(f"GGUFの保存先  ·  {self.destination}")

    def closeEvent(self, event):
        if self.work is not None:
            self.status.setText("変換中です。閉じるには変換を停止してください。")
            event.ignore()
        else:
            event.accept()


def main(argv):
    parser = argparse.ArgumentParser(prog="python -m x189_clean gui", description="X189 desktop converter")
    parser.parse_args(argv)
    app = QApplication([sys.argv[0]])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLES)
    window = Window()
    window.show()
    return app.exec()
