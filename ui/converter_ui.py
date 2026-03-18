import os
import sys
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QComboBox, QTextEdit, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor
from tools.export_ncnn import export_to_ncnn

class ExportThread(QThread):
    """
    模型转换工作线程 (Prevent UI freezing)
    """
    finished_signal = pyqtSignal(bool, str)
    log_signal = pyqtSignal(str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        def log_callback(msg):
            self.log_signal.emit(msg)
            
        success = export_to_ncnn(self.model_path, callback=log_callback)
        self.finished_signal.emit(success, self.model_path)

class ModelConverterDialog(QDialog):
    """
    模型转换对话框 (Model Conversion Dialog)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO to NCNN Converter (模型转换器)")
        self.setMinimumSize(500, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 1. 说明 (Instruction)
        desc = QLabel("<b>Why NCNN?</b><br>"
                      "NCNN models are highly optimized for mobile/embedded devices like Raspberry Pi.<br>"
                      "Converting .pt to NCNN can boost inference speed by 2x-4x.")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # 2. 模型选择 (Model Select)
        file_layout = QHBoxLayout()
        self.txt_path = QLabel("Select a .pt model to convert...")
        self.txt_path.setStyleSheet("background: #f0f0f0; padding: 5px; border: 1px solid #ccc;")
        btn_browse = QPushButton("Browse (浏览)")
        btn_browse.clicked.connect(self.browse_model)
        file_layout.addWidget(self.txt_path, 4)
        file_layout.addWidget(btn_browse, 1)
        layout.addLayout(file_layout)

        # 3. 日志区域 (Log Area)
        layout.addWidget(QLabel("Conversion Progress (转换日志):"))
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background: #000; color: #0f0; font-family: 'Courier New';")
        layout.addWidget(self.log_area)

        # 4. 按钮控制 (Buttons)
        btn_layout = QHBoxLayout()
        self.btn_convert = QPushButton("Start Conversion (开始转换)")
        self.btn_convert.setFixedHeight(40)
        self.btn_convert.setStyleSheet("background: #4CAF50; color: white; font-weight: bold;")
        self.btn_convert.clicked.connect(self.start_conversion)
        
        btn_close = QPushButton("Close (关闭)")
        btn_close.setFixedHeight(40)
        btn_close.clicked.connect(self.close)
        
        btn_layout.addWidget(self.btn_convert)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.model_path = ""

    def browse_model(self):
        # 默认定位到 models 目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models")
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", models_dir, "PyTorch Models (*.pt)"
        )
        if path:
            self.model_path = path
            self.txt_path.setText(os.path.basename(path))

    def start_conversion(self):
        if not self.model_path:
            QMessageBox.warning(self, "Error", "Please select a model first!")
            return

        self.btn_convert.setEnabled(False)
        self.log_area.clear()
        self.log_area.append(f"Starting conversion for: {self.model_path}")

        # 启动工作线程
        self.thread = ExportThread(self.model_path)
        self.thread.log_signal.connect(self.update_log)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.start()

    def update_log(self, msg):
        self.log_area.append(msg)
        # 自动滚动到底部 (Auto-scroll to bottom - PyQt6 syntax)
        self.log_area.moveCursor(QTextCursor.MoveOperation.End)

    def on_finished(self, success, path):
        self.btn_convert.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success", f"Model converted successfully!\nNew model folder is in the same directory as {os.path.basename(path)}")
        else:
            QMessageBox.critical(self, "Error", "Conversion failed. Check log for details.")
